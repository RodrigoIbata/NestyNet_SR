# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Unit-torus and Buckingham-pi helpers for the GS layer.

Physical dimensions define commuting scale symmetries.  If variable ``x_i`` has
base-dimension vector D[:, i], the base-unit generator is
``sum_i D[a, i] x_i d/dx_i``.  Dimensionless Buckingham-pi coordinates are
monomials with exponent vector q satisfying ``D q = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from math import gcd
from typing import Any, Iterable, Mapping, Sequence

from nestynet_sr.sr_core.bridges import ConstNode, MulNode, PowNode, Var, ast_to_human_readable

Dim = tuple[Fraction, ...]


@dataclass(frozen=True)
class UnitTorusGeneratorSpec:
    basis_index: int
    basis_name: str
    x_weights: tuple[Fraction, ...]
    y_weight: Fraction
    invariant_condition: str = "D*q=0"
    family: str = "unit_torus"


@dataclass(frozen=True)
class PiInvariantSpec:
    exponents: tuple[Fraction, ...]
    support: tuple[int, ...]
    l1: Fraction
    ast_human: str | None = None
    confidence: float = 1.0
    family: str = "buckingham_pi"


def _as_fraction(value: Any, *, max_den: int = 256) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        return Fraction.from_float(float(value)).limit_denominator(max_den)
    if isinstance(value, str):
        return Fraction(value.strip())
    return Fraction.from_float(float(value)).limit_denominator(max_den)


def as_dim(values: Sequence[Any]) -> Dim:
    return tuple(_as_fraction(v) for v in values)


def _lcm(a: int, b: int) -> int:
    if a == 0:
        return abs(b)
    if b == 0:
        return abs(a)
    return abs(a * b) // gcd(abs(a), abs(b))


def _lcm_many(values: Iterable[int]) -> int:
    out = 1
    for v in values:
        out = _lcm(out, int(v))
    return max(1, out)


def _gcd_many(values: Iterable[int]) -> int:
    out = 0
    for v in values:
        out = gcd(out, abs(int(v)))
    return max(1, out)


def normalize_exponents(exponents: Sequence[Any]) -> tuple[Fraction, ...]:
    raw = tuple(_as_fraction(v) for v in exponents)
    if not raw:
        return ()
    den = _lcm_many(v.denominator for v in raw)
    ints = [int(v * den) for v in raw]
    common = _gcd_many(ints)
    ints = [i // common for i in ints]
    first = next((i for i in ints if i != 0), 0)
    if first < 0:
        ints = [-i for i in ints]
    return tuple(Fraction(i, den) for i in ints)


def projective_exponent_key(exponents: Sequence[Any]) -> tuple[int, ...]:
    """Return a primitive integer ray key for exponent vectors.

    Buckingham-pi vectors that differ by a nonzero scalar describe powers of
    the same dimensionless group. This key is for deduplication only; callers
    can still keep the original bounded exponent vector for display.
    """

    raw = tuple(_as_fraction(v) for v in exponents)
    if not raw:
        return ()
    den = _lcm_many(v.denominator for v in raw)
    ints = [int(v * den) for v in raw]
    common = _gcd_many(ints)
    ints = [i // common for i in ints]
    first = next((i for i in ints if i != 0), 0)
    if first < 0:
        ints = [-i for i in ints]
    return tuple(int(i) for i in ints)


def exponent_l1(exponents: Sequence[Any]) -> Fraction:
    return sum((abs(_as_fraction(v)) for v in exponents), Fraction(0))


def dimension_matrix_from_dims(dims: Sequence[Sequence[Any]]) -> tuple[Dim, ...]:
    cols = tuple(as_dim(d) for d in dims)
    if not cols:
        return ()
    rank = len(cols[0])
    for d in cols:
        if len(d) != rank:
            raise ValueError("all dimension vectors must have the same rank")
    return tuple(tuple(col[a] for col in cols) for a in range(rank))


def dimensions_from_units_spec(units_spec: Any, *, include_y: bool = False) -> tuple[tuple[Dim, ...], Dim, tuple[str, ...]]:
    if units_spec is None:
        raise ValueError("units_spec is required for unit-torus analysis")
    x_dims = tuple(as_dim(d) for d in getattr(units_spec, "x_dims"))
    y_raw = getattr(units_spec, "y_phi_dim", None)
    if y_raw is None:
        y_raw = getattr(units_spec, "y_dim")
    y_dim = as_dim(y_raw)
    base = tuple(str(b) for b in getattr(getattr(units_spec, "unit_system", None), "base", tuple(range(len(y_dim)))))
    if include_y:
        x_dims = x_dims + (y_dim,)
    return x_dims, y_dim, base


def unit_torus_generators_from_dims(
    x_dims: Sequence[Sequence[Any]],
    y_dim: Sequence[Any],
    *,
    basis: Sequence[str] | None = None,
) -> list[UnitTorusGeneratorSpec]:
    cols = tuple(as_dim(d) for d in x_dims)
    y = as_dim(y_dim)
    rank = len(y)
    if any(len(d) != rank for d in cols):
        raise ValueError("x_dims and y_dim must share the same rank")
    names = tuple(str(b) for b in (basis or tuple(f"d{i}" for i in range(rank))))
    return [
        UnitTorusGeneratorSpec(
            basis_index=a,
            basis_name=names[a] if a < len(names) else f"d{a}",
            x_weights=tuple(d[a] for d in cols),
            y_weight=y[a],
        )
        for a in range(rank)
    ]


def unit_torus_generators_from_units_spec(units_spec: Any) -> list[UnitTorusGeneratorSpec]:
    x_dims, y_dim, base = dimensions_from_units_spec(units_spec)
    return unit_torus_generators_from_dims(x_dims, y_dim, basis=base)


def _dot_dim(matrix_rows: Sequence[Sequence[Fraction]], exponents: Sequence[Fraction]) -> Dim:
    return tuple(sum(row[i] * exponents[i] for i in range(len(exponents))) for row in matrix_rows)


def _bounded_exponent_vectors(
    n: int,
    *,
    max_exponent: int,
    max_l1: int,
    max_support: int,
    rational_denom: int = 1,
    normalize: bool = True,
):
    n = int(n)
    max_support = max(1, min(int(max_support), n))
    max_exponent = max(1, int(max_exponent))
    max_l1_f = Fraction(max(1, int(max_l1)), 1)
    denoms = tuple(range(1, max(1, int(rational_denom)) + 1))
    for support_size in range(1, max_support + 1):
        for support in combinations(range(n), support_size):
            for denom in denoms:
                values = [v for v in range(-max_exponent, max_exponent + 1) if v != 0]
                for ints in product(values, repeat=support_size):
                    q = [Fraction(0) for _ in range(n)]
                    for idx, val in zip(support, ints):
                        q[idx] = Fraction(int(val), int(denom))
                    q_out = normalize_exponents(q) if bool(normalize) else tuple(q)
                    if exponent_l1(q_out) > max_l1_f:
                        continue
                    yield q_out


def enumerate_nullspace_exponents(
    x_dims: Sequence[Sequence[Any]],
    *,
    max_exponent: int = 3,
    max_l1: int = 6,
    max_proposals: int = 24,
    max_basis: int = 8,
    rational_denom: int = 1,
) -> list[tuple[Fraction, ...]]:
    cols = tuple(as_dim(d) for d in x_dims)
    if not cols:
        return []
    matrix = dimension_matrix_from_dims(cols)
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[Fraction, ...]] = []
    for q in _bounded_exponent_vectors(
        len(cols),
        max_exponent=max_exponent,
        max_l1=max_l1,
        max_support=max_basis,
        rational_denom=rational_denom,
    ):
        if all(v == 0 for v in q):
            continue
        if _dot_dim(matrix, q) != tuple(Fraction(0) for _ in range(len(matrix))):
            continue
        key = projective_exponent_key(q)
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    out.sort(key=lambda q: (exponent_l1(q), sum(1 for v in q if v != 0), tuple(float(v) for v in q)))
    return out[: max(0, int(max_proposals))]


def enumerate_prefactor_exponents(
    x_dims: Sequence[Sequence[Any]],
    y_dim: Sequence[Any],
    *,
    max_exponent: int = 3,
    max_l1: int = 6,
    max_proposals: int = 12,
    max_basis: int = 8,
    rational_denom: int = 1,
) -> list[tuple[Fraction, ...]]:
    cols = tuple(as_dim(d) for d in x_dims)
    target = as_dim(y_dim)
    if not cols:
        return []
    matrix = dimension_matrix_from_dims(cols)
    seen: set[tuple[Fraction, ...]] = set()
    out: list[tuple[Fraction, ...]] = []
    for q in _bounded_exponent_vectors(
        len(cols),
        max_exponent=max_exponent,
        max_l1=max_l1,
        max_support=max_basis,
        rational_denom=rational_denom,
        normalize=False,
    ):
        if q in seen:
            continue
        if _dot_dim(matrix, q) != target:
            continue
        seen.add(q)
        out.append(q)
    out.sort(key=lambda q: (exponent_l1(q), sum(1 for v in q if v != 0), tuple(float(v) for v in q)))
    return out[: max(0, int(max_proposals))]


def build_monomial_ast(exponents: Sequence[Any], *, variable_nodes: Sequence[Any] | None = None):
    q = tuple(_as_fraction(v) for v in exponents)
    nodes = tuple(variable_nodes) if variable_nodes is not None else tuple(Var(i) for i in range(len(q)))
    if len(nodes) != len(q):
        raise ValueError("variable_nodes length must match exponent vector length")
    out = None
    for exp, node in zip(q, nodes):
        if exp == 0:
            continue
        factor = node if exp == 1 else PowNode(node, float(exp))
        out = factor if out is None else MulNode(out, factor)
    return out if out is not None else ConstNode(1.0)


def pi_invariants_from_units_spec(units_spec: Any, **kwargs: Any) -> list[PiInvariantSpec]:
    x_dims, _y_dim, _base = dimensions_from_units_spec(units_spec)
    specs: list[PiInvariantSpec] = []
    for q in enumerate_nullspace_exponents(x_dims, **kwargs):
        ast = build_monomial_ast(q)
        try:
            human = ast_to_human_readable(ast)
        except Exception:
            human = repr(ast)
        specs.append(
            PiInvariantSpec(
                exponents=q,
                support=tuple(i for i, v in enumerate(q) if v != 0),
                l1=exponent_l1(q),
                ast_human=human,
                confidence=1.0 / (1.0 + 0.05 * float(exponent_l1(q))),
            )
        )
    return specs


def _rank_fraction_matrix(rows: Sequence[Sequence[Fraction]]) -> int:
    mat = [list(row) for row in rows if any(v != 0 for v in row)]
    if not mat:
        return 0
    n_rows = len(mat)
    n_cols = len(mat[0])
    rank = 0
    col = 0
    while rank < n_rows and col < n_cols:
        pivot = None
        for r in range(rank, n_rows):
            if mat[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            col += 1
            continue
        mat[rank], mat[pivot] = mat[pivot], mat[rank]
        pv = mat[rank][col]
        mat[rank] = [v / pv for v in mat[rank]]
        for r in range(n_rows):
            if r == rank or mat[r][col] == 0:
                continue
            factor = mat[r][col]
            mat[r] = [a - factor * b for a, b in zip(mat[r], mat[rank])]
        rank += 1
        col += 1
    return rank


def dim_in_rational_span(target: Sequence[Any], basis_dims: Sequence[Sequence[Any]]) -> bool:
    target_d = as_dim(target)
    cols = tuple(as_dim(d) for d in basis_dims)
    if all(v == 0 for v in target_d):
        return True
    if not cols:
        return False
    rank = len(target_d)
    if any(len(d) != rank for d in cols):
        return False
    base_rows = [tuple(col[a] for col in cols) for a in range(rank)]
    aug_rows = [tuple(list(row) + [target_d[a]]) for a, row in enumerate(base_rows)]
    return _rank_fraction_matrix(base_rows) == _rank_fraction_matrix(aug_rows)


def constant_dims_from_units_spec(units_spec: Any, *, include_free: bool = True, include_fixed: bool = True) -> tuple[Dim, ...]:
    dims: list[Dim] = []
    if include_free:
        dims.extend(as_dim(d) for d in (getattr(units_spec, "free_const_dims", {}) or {}).values())
    if include_fixed:
        dims.extend(as_dim(d) for d in (getattr(units_spec, "fixed_const_dims", {}) or {}).values())
    return tuple(dims)


def unit_torus_report(units_spec: Any, *, include_pi: bool = False, **pi_kwargs: Any) -> Mapping[str, Any]:
    if units_spec is None:
        return {"status": "skipped", "reason": "units_spec_missing"}
    x_dims, y_dim, base = dimensions_from_units_spec(units_spec)
    payload: dict[str, Any] = {
        "status": "available",
        "basis": base,
        "x_dims": x_dims,
        "y_dim": y_dim,
        "generators": unit_torus_generators_from_dims(x_dims, y_dim, basis=base),
    }
    if include_pi:
        payload["pi_invariants"] = pi_invariants_from_units_spec(units_spec, **pi_kwargs)
    return payload
