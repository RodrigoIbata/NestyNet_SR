# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Bounded generalized-symmetry proposal support for factorized search.

The helpers in this module are deliberately generic. They expose invariant-like
and covariant atoms to FSS without encoding benchmark identities or target
formulae.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..expr_ast import (
    BINARY_OPS,
    collect_paths,
    dim_round,
    dims_eq,
    get_at,
    is_valid_node,
    node_depth,
    node_dims,
    node_size,
    node_str,
    replace_at,
    simplify,
)


@dataclass(frozen=True)
class GSFSSAtom:
    """One GS-derived tuple-AST atom exposed to FSS."""

    expr: Any
    dim: tuple[float, ...] | None
    name: str
    family: str
    source: str = "gs_fss"


@dataclass
class GSFSSContext:
    """Runtime GS/FSS bridge state."""

    enabled: bool = False
    active: bool = False
    feature_names: tuple[str, ...] = ()
    var_dims: tuple[tuple[float, ...], ...] | None = None
    y_dims: tuple[float, ...] | None = None
    dimless_dim: tuple[float, ...] | None = None
    atoms: tuple[GSFSSAtom, ...] = ()
    atom_strings: frozenset[str] = frozenset()
    seed_strings: frozenset[str] = frozenset()
    aux_generator_enabled: bool = False
    score_enabled: bool = False
    max_aux_atoms: int = 0
    max_seed_blocks: int = 0
    source_fraction: float = 0.0
    stats: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    @property
    def seed_atoms(self) -> tuple[GSFSSAtom, ...]:
        if self.y_dims is None:
            return ()
        return tuple(a for a in self.atoms if _dims_eq(a.dim, self.y_dims))


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _bool_attr(obj: Any, name: str, default: bool = False) -> bool:
    value = _get_attr(obj, name, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _get_any_attr(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        value = _get_attr(obj, name, None)
        if value is not None:
            return value
    return default


def _bool_any_attr(obj: Any, names: Sequence[str], default: bool = False) -> bool:
    return _bool_attr({"value": _get_any_attr(obj, names, default)}, "value", default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _dims_eq(a: Sequence[float] | None, b: Sequence[float] | None) -> bool:
    if a is None or b is None:
        return False
    return dims_eq(tuple(a), tuple(b))


def _normalise_dims(dims: Sequence[Sequence[float]] | None) -> tuple[tuple[float, ...], ...] | None:
    if dims is None:
        return None
    out = []
    for d in dims:
        try:
            out.append(dim_round(tuple(float(v) for v in d)))
        except Exception:
            return None
    return tuple(out)


def _normalise_dim(d: Sequence[float] | None) -> tuple[float, ...] | None:
    if d is None:
        return None
    try:
        return dim_round(tuple(float(v) for v in d))
    except Exception:
        return None


def _dimless_from_dims(
    var_dims: Sequence[Sequence[float]] | None,
    y_dims: Sequence[float] | None,
) -> tuple[float, ...] | None:
    if var_dims:
        return (0.0,) * len(tuple(var_dims[0]))
    if y_dims is not None:
        return (0.0,) * len(tuple(y_dims))
    return None


def _classify_feature(name: str) -> str:
    n = str(name or "").strip().lower().replace("-", "_")
    if n in {"du", "d_u", "u_x", "u_t", "ux", "ut", "ydot", "dy", "v"}:
        return "du"
    if n.startswith("du") or n.startswith("d1u") or n.startswith("u_"):
        return "du"
    if n in {"u", "y", "state"}:
        return "u"
    if n in {"x", "t", "r", "time"}:
        return "x"
    if n.startswith("x") and (len(n) == 1 or n[1].isdigit() or n[1] == "_"):
        return "x"
    return "const"


def _feature_names(feature_names: Sequence[str] | None, nvars: int) -> tuple[str, ...]:
    names = [str(v) for v in (feature_names or ())]
    if len(names) < int(nvars):
        names.extend(f"x{i}" for i in range(len(names), int(nvars)))
    return tuple(names[: int(nvars)])


def _node_dim(expr: Any, var_dims: tuple[tuple[float, ...], ...] | None) -> tuple[float, ...] | None:
    if var_dims is None:
        return None
    try:
        d = node_dims(expr, var_dims)
    except Exception:
        return None
    return None if d is None else tuple(d)


def _dedupe_add(
    atoms: list[GSFSSAtom],
    seen: set[str],
    expr: Any,
    *,
    var_dims: tuple[tuple[float, ...], ...] | None,
    family: str,
    source: str,
    name: str | None = None,
    max_depth: int | None = None,
) -> None:
    try:
        expr = simplify(expr)
    except Exception:
        return
    if not is_valid_node(expr):
        return
    if max_depth is not None:
        try:
            if node_depth(expr) > int(max_depth):
                return
        except Exception:
            return
    dim = _node_dim(expr, var_dims)
    if var_dims is not None and dim is None:
        return
    key = node_str(expr)
    if key in seen:
        return
    seen.add(key)
    atoms.append(GSFSSAtom(expr=expr, dim=dim, family=str(family), source=str(source), name=str(name or key)))


def _limited_pairs(n: int, *, max_pairs: int = 64) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for i in range(int(n)):
        for j in range(i + 1, int(n)):
            out.append((i, j))
            if len(out) >= int(max_pairs):
                return out
    return out


def build_gs_fss_context(
    *,
    feature_names: Sequence[str] | None = None,
    nvars: int | None = None,
    var_dims: Sequence[Sequence[float]] | None = None,
    y_dims: Sequence[float] | None = None,
    cfg: Any = None,
    order: int | None = None,
    max_aux_atoms: int | None = None,
    max_seed_blocks: int | None = None,
    source_fraction: float | None = None,
    enabled: bool | None = None,
    max_depth: int | None = None,
) -> GSFSSContext:
    """Build a bounded GS/FSS atom context from feature metadata."""

    aux_enabled = _bool_any_attr(cfg, ("expr_gs_fss_aux_generator", "gs_fss_aux_generator"), False)
    score_enabled = _bool_any_attr(cfg, ("expr_gs_fss_score", "gs_fss_score"), False)
    enabled_b = bool(aux_enabled or score_enabled) if enabled is None else bool(enabled)
    max_aux = _safe_int(
        max_aux_atoms
        if max_aux_atoms is not None
        else _get_any_attr(cfg, ("expr_gs_fss_max_aux_atoms", "gs_fss_max_aux_atoms"), 0),
        0,
    )
    max_seed = _safe_int(
        max_seed_blocks
        if max_seed_blocks is not None
        else _get_any_attr(cfg, ("expr_gs_fss_max_seed_blocks", "gs_fss_max_seed_blocks"), 0),
        0,
    )
    src_frac = max(
        0.0,
        min(
            1.0,
            _safe_float(
                source_fraction
                if source_fraction is not None
                else _get_any_attr(cfg, ("expr_gs_fss_max_source_fraction", "gs_fss_max_source_fraction"), 0.0),
                0.0,
            ),
        ),
    )
    if nvars is None:
        nvars = len(tuple(var_dims)) if var_dims is not None else len(tuple(feature_names or ()))
    nvars = max(0, int(nvars or 0))
    names = _feature_names(feature_names, nvars)
    var_dims_n = _normalise_dims(var_dims)
    y_dims_n = _normalise_dim(y_dims)
    dimless = _dimless_from_dims(var_dims_n, y_dims_n)
    effective_max_aux = max_aux if aux_enabled else 0
    effective_max_seed = max_seed if aux_enabled else 0
    requested_cap = max(effective_max_aux, effective_max_seed)
    inactive_base = dict(
        enabled=enabled_b,
        active=False,
        feature_names=names,
        var_dims=var_dims_n,
        y_dims=y_dims_n,
        dimless_dim=dimless,
        aux_generator_enabled=bool(aux_enabled),
        score_enabled=bool(score_enabled),
        max_aux_atoms=effective_max_aux,
        max_seed_blocks=effective_max_seed,
        source_fraction=src_frac,
        stats={
            "score_enabled": bool(score_enabled),
            "aux_generator_enabled": bool(aux_enabled),
            "score_path_wired": False,
        },
    )
    if score_enabled and not aux_enabled:
        return GSFSSContext(**inactive_base, reason="score_only_not_wired")
    if requested_cap <= 0:
        return GSFSSContext(**inactive_base, reason="no_aux_atom_budget")
    if not enabled_b or nvars <= 0:
        return GSFSSContext(**inactive_base, reason="disabled_or_empty_features")

    cap = max(1, int(requested_cap))
    atoms: list[GSFSSAtom] = []
    seen: set[str] = set()
    kinds = [_classify_feature(name) for name in names]
    x_idxs = [i for i, k in enumerate(kinds) if k == "x"]
    u_idxs = [i for i, k in enumerate(kinds) if k == "u"]
    du_idxs = [i for i, k in enumerate(kinds) if k == "du"]
    lorentz = bool(
        _bool_any_attr(
            cfg,
            ("gs_lorentz_boosts", "expr_gs_fss_lorentz_boosts", "gs_fss_lorentz_boosts"),
            False,
        )
    )

    def add(expr: Any, family: str, source: str, name: str | None = None) -> None:
        if len(atoms) >= cap:
            return
        _dedupe_add(atoms, seen, expr, var_dims=var_dims_n, family=family, source=source, name=name, max_depth=max_depth)

    if var_dims_n is not None:
        dim_pairs = [(i, j) for i, j in _limited_pairs(nvars, max_pairs=max(64, 4 * cap)) if _dims_eq(var_dims_n[i], var_dims_n[j])]
    else:
        dim_pairs = _limited_pairs(nvars, max_pairs=max(64, 4 * cap))

    for i, j in dim_pairs:
        if len(atoms) >= cap:
            break
        vi = ("var", i)
        vj = ("var", j)
        add(("div", vi, vj), "same_dim_ratio", "unit_torus", f"{names[i]}/{names[j]}")
        add(("div", vj, vi), "same_dim_ratio", "unit_torus", f"{names[j]}/{names[i]}")
        add(("add", ("sqr", vi), ("sqr", vj)), "quadratic_carrier", "rotation", f"{names[i]}^2+{names[j]}^2")
        if lorentz:
            add(("sub", ("sqr", vi), ("sqr", vj)), "quadratic_carrier", "lorentz", f"{names[i]}^2-{names[j]}^2")

    for xi in x_idxs:
        if len(atoms) >= cap:
            break
        x = ("var", xi)
        x2 = ("sqr", x)
        add(("div", ("const", 1.0), x), "radial_singular", "de_radial", f"1/{names[xi]}")
        for ui in u_idxs:
            u = ("var", ui)
            add(("div", u, x), "radial_ratio", "de_radial", f"{names[ui]}/{names[xi]}")
            add(("div", u, x2), "radial_ratio", "de_radial", f"{names[ui]}/{names[xi]}^2")
            add(("mul", x, u), "scaling_covariant", "weighted_scaling", f"{names[xi]}*{names[ui]}")
        for di in du_idxs:
            du = ("var", di)
            add(("div", du, x), "radial_velocity", "de_radial", f"{names[di]}/{names[xi]}")
            add(("div", du, x2), "radial_velocity", "de_radial", f"{names[di]}/{names[xi]}^2")
            add(("mul", x, du), "velocity_covariant", "weighted_scaling", f"{names[xi]}*{names[di]}")

    for ui in u_idxs:
        if len(atoms) >= cap:
            break
        u = ("var", ui)
        for di in du_idxs:
            du = ("var", di)
            add(("mul", u, du), "velocity_coupling", "contact_template", f"{names[ui]}*{names[di]}")
            add(("div", du, u), "velocity_ratio", "contact_template", f"{names[di]}/{names[ui]}")

    stats = {
        "requested": int(requested_cap),
        "built": int(len(atoms)),
        "order": None if order is None else int(order),
        "feature_kinds": {k: int(kinds.count(k)) for k in sorted(set(kinds))},
        "score_enabled": bool(score_enabled),
        "aux_generator_enabled": bool(aux_enabled),
        "score_path_wired": False,
    }
    atom_strings = frozenset(node_str(a.expr) for a in atoms)
    seed_strings = frozenset(node_str(a.expr) for a in atoms if _dims_eq(a.dim, y_dims_n))
    return GSFSSContext(
        enabled=enabled_b,
        active=bool(atoms),
        feature_names=names,
        var_dims=var_dims_n,
        y_dims=y_dims_n,
        dimless_dim=dimless,
        atoms=tuple(atoms),
        atom_strings=atom_strings,
        seed_strings=seed_strings,
        aux_generator_enabled=bool(aux_enabled),
        score_enabled=bool(score_enabled),
        max_aux_atoms=effective_max_aux,
        max_seed_blocks=effective_max_seed,
        source_fraction=src_frac,
        stats=stats,
        reason="" if atoms else "empty_atom_set",
    )


def coerce_gs_fss_context(value: Any) -> GSFSSContext | None:
    if value is None:
        return None
    if isinstance(value, GSFSSContext):
        return value
    if isinstance(value, Mapping):
        return build_gs_fss_context(**dict(value))
    return None


def _canonical_key(expr: Any, *, ir_cfg: Any, ir_stats: Any, signature_context: Any) -> tuple[Any, ...]:
    try:
        from nestynet_sr.sr_expr_ir.config import expr_ir_active
        from nestynet_sr.sr_expr_ir.tuple_bridge import canonical_key_tuple_ast

        if expr_ir_active(ir_cfg):
            return canonical_key_tuple_ast(expr, ir_cfg, stats=ir_stats, signature_context=signature_context)
    except Exception:
        pass
    return ("legacy", node_str(expr))


def _maybe_ir_canonicalize(expr: Any, *, ir_cfg: Any, ir_stats: Any, signature_context: Any) -> Any:
    try:
        from nestynet_sr.sr_expr_ir.config import expr_ir_active
        from nestynet_sr.sr_expr_ir.tuple_bridge import maybe_canonicalize_tuple_ast

        if expr_ir_active(ir_cfg):
            return maybe_canonicalize_tuple_ast(expr, ir_cfg, stats=ir_stats, signature_context=signature_context)
    except Exception:
        return expr
    return expr


def extend_pool_with_gs_atoms(
    pool_nodes: Sequence[Any],
    context: GSFSSContext | None,
    *,
    ir_cfg: Any = None,
    ir_stats: Any = None,
    signature_context: Any = None,
    max_depth: int | None = None,
) -> list[Any]:
    """Return ``pool_nodes`` plus bounded GS atoms, deduped with the active IR."""

    ctx = coerce_gs_fss_context(context)
    out = list(pool_nodes)
    if (
        ctx is None
        or not ctx.active
        or not bool(getattr(ctx, "aux_generator_enabled", False))
        or int(ctx.max_aux_atoms) <= 0
    ):
        return out
    seen = {_canonical_key(n, ir_cfg=ir_cfg, ir_stats=ir_stats, signature_context=signature_context) for n in out}
    added = 0
    for atom in ctx.atoms:
        if added >= int(ctx.max_aux_atoms):
            break
        expr = atom.expr
        if max_depth is not None:
            try:
                if node_depth(expr) > int(max_depth):
                    continue
            except Exception:
                continue
        expr = _maybe_ir_canonicalize(expr, ir_cfg=ir_cfg, ir_stats=ir_stats, signature_context=signature_context)
        if not is_valid_node(expr):
            continue
        key = _canonical_key(expr, ir_cfg=ir_cfg, ir_stats=ir_stats, signature_context=signature_context)
        if key in seen:
            continue
        seen.add(key)
        out.append(expr)
        added += 1
    ctx.stats["pool_atoms_added"] = int(ctx.stats.get("pool_atoms_added", 0)) + int(added)
    return out


def _contains_atom_hits(expr: Any, atom_strings: frozenset[str]) -> int:
    if not atom_strings or not is_valid_node(expr):
        return 0
    hits = 0
    try:
        for path in collect_paths(expr):
            try:
                if node_str(get_at(expr, path)) in atom_strings:
                    hits += 1
            except Exception:
                continue
    except Exception:
        return 0
    return hits


def gs_fss_score_bonus(expr: Any, context: GSFSSContext | None) -> float:
    """Return a bounded 0..1 weak GS support score for an expression."""

    ctx = coerce_gs_fss_context(context)
    if ctx is None or not ctx.active:
        return 0.0
    hits = _contains_atom_hits(expr, ctx.atom_strings)
    if hits <= 0:
        return 0.0
    try:
        size = max(1, int(node_size(expr)))
    except Exception:
        size = 1
    bonus = min(1.0, 0.25 + 0.75 * float(hits) / float(size))
    ctx.stats["score_bonus_hits"] = int(ctx.stats.get("score_bonus_hits", 0)) + int(hits)
    return float(bonus)


def _candidate_ops_for_dims(
    parent: Any,
    atom: GSFSSAtom,
    *,
    var_dims: tuple[tuple[float, ...], ...] | None,
    y_dims: tuple[float, ...] | None,
    dimless_dim: tuple[float, ...] | None,
) -> list[Any]:
    if var_dims is None:
        a = atom.expr
        return [("add", parent, a), ("sub", parent, a), ("mul", parent, a), ("div", parent, a), a]
    parent_dim = _node_dim(parent, var_dims)
    atom_dim = atom.dim
    if parent_dim is None or atom_dim is None:
        return []
    out: list[Any] = []
    a = atom.expr
    if _dims_eq(parent_dim, atom_dim):
        out.append(("add", parent, a))
        out.append(("sub", parent, a))
    if _dims_eq(atom_dim, dimless_dim):
        out.append(("mul", parent, a))
        out.append(("div", parent, a))
    if _dims_eq(parent_dim, dimless_dim) and _dims_eq(atom_dim, y_dims):
        out.append(("mul", parent, a))
    if _dims_eq(atom_dim, y_dims):
        out.append(a)
    return out


def _valid_output(
    expr: Any,
    *,
    max_depth: int,
    var_dims: tuple[tuple[float, ...], ...] | None,
    y_dims: tuple[float, ...] | None,
) -> bool:
    if not is_valid_node(expr):
        return False
    try:
        if node_depth(expr) > int(max_depth):
            return False
    except Exception:
        return False
    if var_dims is not None and y_dims is not None:
        return _dims_eq(_node_dim(expr, var_dims), y_dims)
    return True


def apply_gs_fss_proposal(
    parent: Any | None,
    context: GSFSSContext | None,
    rng: Any,
    *,
    max_depth: int,
    nvars: int,
    var_dims: Sequence[Sequence[float]] | None = None,
    y_dims: Sequence[float] | None = None,
) -> Any | None:
    """Sample one bounded GS-derived proposal expression."""

    del nvars
    ctx = coerce_gs_fss_context(context)
    if ctx is None or not ctx.active or not ctx.atoms:
        return None
    var_dims_n = ctx.var_dims if ctx.var_dims is not None else _normalise_dims(var_dims)
    y_dims_n = ctx.y_dims if ctx.y_dims is not None else _normalise_dim(y_dims)
    dimless = ctx.dimless_dim if ctx.dimless_dim is not None else _dimless_from_dims(var_dims_n, y_dims_n)

    atoms = list(ctx.atoms)
    if y_dims_n is not None:
        target_atoms = [a for a in atoms if _dims_eq(a.dim, y_dims_n)]
        if target_atoms and (parent is None or rng.random() < 0.35):
            atoms = target_atoms + atoms

    for _ in range(24):
        atom = rng.choice(atoms)
        if parent is None or not is_valid_node(parent):
            expr = atom.expr
        else:
            if rng.random() < 0.20:
                try:
                    paths = [p for p in collect_paths(parent) if get_at(parent, p)[0] not in BINARY_OPS]
                    expr = replace_at(parent, rng.choice(paths or [()]), atom.expr)
                except Exception:
                    expr = atom.expr
            else:
                options = _candidate_ops_for_dims(parent, atom, var_dims=var_dims_n, y_dims=y_dims_n, dimless_dim=dimless)
                if not options:
                    continue
                expr = rng.choice(options)
        try:
            expr = simplify(expr)
        except Exception:
            continue
        if _valid_output(expr, max_depth=max_depth, var_dims=var_dims_n, y_dims=y_dims_n):
            ctx.stats["proposal_count"] = int(ctx.stats.get("proposal_count", 0)) + 1
            return expr
    ctx.stats["proposal_reject_count"] = int(ctx.stats.get("proposal_reject_count", 0)) + 1
    return None


def gs_fss_report(context: GSFSSContext | None) -> dict[str, Any]:
    ctx = coerce_gs_fss_context(context)
    if ctx is None:
        return {"enabled": False, "active": False}
    families: dict[str, int] = {}
    sources: dict[str, int] = {}
    for atom in ctx.atoms:
        families[atom.family] = int(families.get(atom.family, 0)) + 1
        sources[atom.source] = int(sources.get(atom.source, 0)) + 1
    return {
        "enabled": bool(ctx.enabled),
        "active": bool(ctx.active),
        "reason": str(ctx.reason),
        "atom_count": int(len(ctx.atoms)),
        "seed_atom_count": int(len(ctx.seed_atoms)),
        "max_aux_atoms": int(ctx.max_aux_atoms),
        "max_seed_blocks": int(ctx.max_seed_blocks),
        "source_fraction": float(ctx.source_fraction),
        "families": families,
        "sources": sources,
        "feature_names": list(ctx.feature_names),
        "atom_samples": [{"expr": node_str(a.expr), "family": str(a.family), "source": str(a.source)} for a in list(ctx.atoms)[:8]],
        "stats": dict(ctx.stats),
    }


__all__ = [
    "GSFSSAtom",
    "GSFSSContext",
    "apply_gs_fss_proposal",
    "build_gs_fss_context",
    "coerce_gs_fss_context",
    "extend_pool_with_gs_atoms",
    "gs_fss_report",
    "gs_fss_score_bonus",
]
