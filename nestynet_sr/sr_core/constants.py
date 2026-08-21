# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Shared scalar-constant utilities.

This module centralizes scalar constant handling for SR candidate construction:
- deterministic lookup of declared fixed constants from ``UnitsSpec``
- deterministic lookup of declared free constants by required dimension/scope
- creation of scalar-constant variants (trainable scale + fixed alternatives)
- construction of scalar atom nodes from variant specs
- unit-aware scalar kind selection (``scale`` vs declared ``free_const``)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from .bridges import AtomNode, FixedConst, FreeConst, Scale
from .units import is_dimless, normalize_free_const_scope


def safe_const_token(s: str) -> str:
    """Return a filesystem/tag-safe token for constant names."""
    out = []
    for ch in str(s):
        out.append(ch if (ch.isalnum() or ch in ("_", "-", ".")) else "_")
    return "".join(out) or "c"


def matching_fixed_const_specs(
    units_spec: Any,
    required_dim=None,
) -> List[Tuple[str, float]]:
    """Return declared fixed constants matching ``required_dim`` in sorted order."""
    if units_spec is None:
        return []

    fixed_dims = dict(getattr(units_spec, "fixed_const_dims", {}) or {})
    fixed_vals = dict(getattr(units_spec, "fixed_const_values", {}) or {})
    if not fixed_dims or not fixed_vals:
        return []

    target_dim = required_dim
    if target_dim is None:
        try:
            target_dim = units_spec.unit_system.dimless()
        except Exception:
            target_dim = None

    out: List[Tuple[str, float]] = []
    for name in sorted(fixed_dims.keys()):
        nm = str(name)
        if nm not in fixed_vals:
            continue
        if target_dim is not None and tuple(fixed_dims[nm]) != tuple(target_dim):
            continue
        try:
            val = float(fixed_vals[nm])
        except Exception:
            continue
        if not math.isfinite(val):
            continue
        out.append((nm, val))
    return out


def matching_free_const_specs(
    units_spec: Any,
    required_dim,
    *,
    prefer_scope: str = "experiment",
) -> List[Tuple[str, str]]:
    """Return declared free constants matching ``required_dim``.

    Output is sorted deterministically with preferred scope first.
    """
    if units_spec is None or required_dim is None:
        return []

    free_dims = dict(getattr(units_spec, "free_const_dims", {}) or {})
    if not free_dims:
        return []

    free_scope = dict(getattr(units_spec, "free_const_scope", {}) or {})
    pref = normalize_free_const_scope(prefer_scope, default="experiment")

    out: List[Tuple[str, str]] = []
    for name in sorted(free_dims.keys()):
        nm = str(name)
        if tuple(free_dims[nm]) != tuple(required_dim):
            continue
        scope = normalize_free_const_scope(free_scope.get(nm, "experiment"), default="experiment")
        out.append((nm, scope))

    out.sort(key=lambda kv: (0 if kv[1] == pref else 1, kv[0]))
    return out


def pick_declared_free_const(
    units_spec: Any,
    required_dim,
    *,
    prefer_scope: str = "experiment",
) -> Optional[Tuple[str, str]]:
    """Pick one declared free constant matching required dimension and scope preference."""
    matches = matching_free_const_specs(units_spec, required_dim, prefer_scope=prefer_scope)
    return matches[0] if matches else None


def scalar_constant_variants(
    units_spec: Any,
    *,
    base_tag: str,
    scale_init: float,
    required_dim=None,
    max_fixed: int = 4,
    include_fixed: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Build scalar constant alternatives: trainable Scale + FixedConst variants."""
    tag = str(base_tag)
    out: List[Dict[str, Any]] = [
        {
            "mode": "scale",
            "name": tag,
            "tag": tag,
            "value": float(scale_init),
            "label_suffix": "",
        }
    ]
    mode = str(getattr(units_spec, "fixed_const_mode", "strict")).strip().lower()
    if include_fixed is None:
        include_fixed = (mode == "strict")
    if not bool(include_fixed):
        return out

    for name, val in matching_fixed_const_specs(units_spec, required_dim=required_dim)[: max(0, int(max_fixed))]:
        out.append(
            {
                "mode": "fixed",
                "name": str(name),
                "tag": f"{tag}__fx_{safe_const_token(name)}",
                "value": float(val),
                "label_suffix": f"[fixed:{name}]",
            }
        )
    return out


def build_scalar_atom_from_variant(variant: Dict[str, Any]) -> AtomNode:
    """Instantiate a scalar atom from a variant specification."""
    mode = str(variant.get("mode", "scale")).lower()
    tag_raw = variant.get("tag", None)
    tag = (None if tag_raw is None else str(tag_raw))
    if mode == "fixed":
        default_name = tag if tag is not None else "c"
        return FixedConst(
            str(variant.get("name", default_name)),
            value=float(variant.get("value", 1.0)),
            tag=tag,
        )
    if mode == "free_const":
        scope = normalize_free_const_scope(variant.get("scope", "experiment"), default="experiment")
        default_name = tag if tag is not None else "c"
        name = str(variant.get("name", default_name))
        return FreeConst(
            name,
            tag=str(variant.get("tag", name)) if variant.get("tag", None) is not None else None,
            init=float(variant.get("value", 1.0)),
            scope=scope,
        )
    default_name = tag if tag is not None else "s"
    return Scale(
        name=str(variant.get("name", default_name)),
        tag=tag,
        init=float(variant.get("value", 1.0)),
    )


def unit_aware_scalar_choice(
    required_dim,
    units_spec: Any,
    *,
    prefer_scope: str = "experiment",
) -> Optional[Dict[str, str]]:
    """Choose scalar kind for a required dimension.

    Returns one of:
      - ``{"kind": "scale"}`` for dimensionless scalars
      - ``{"kind": "free_const", "name": ..., "scope": ...}`` for unitful scalars
      - ``None`` when no declared unitful free constant can satisfy ``required_dim``
    """
    if required_dim is None:
        return {"kind": "scale"}
    if is_dimless(required_dim):
        return {"kind": "scale"}

    pick = pick_declared_free_const(units_spec, required_dim, prefer_scope=prefer_scope)
    if pick is None:
        return None
    name, scope = pick
    return {"kind": "free_const", "name": str(name), "scope": str(scope)}


def make_unit_aware_scalar_atom(
    required_dim,
    units_spec: Any,
    *,
    base_tag: str,
    init: float = 1.0,
    prefer_scope: str = "experiment",
    strict: bool = False,
) -> AtomNode:
    """Create a scalar atom consistent with ``required_dim``.

    - Dimensionless -> ``Scale``
    - Unitful      -> declared ``FreeConst`` (by dim/scope)

    If no matching declared free constant exists for a unitful requirement:
    - ``strict=True``  -> raise ``ValueError``
    - ``strict=False`` -> fallback to ``Scale`` (useful in exploratory mode)
    """
    choice = unit_aware_scalar_choice(required_dim, units_spec, prefer_scope=prefer_scope)
    if choice is None:
        if strict:
            raise ValueError("No declared free constant matches required dimension.")
        return Scale(name=str(base_tag), tag=str(base_tag), init=float(init))

    if choice["kind"] == "free_const":
        name = str(choice["name"])
        scope = normalize_free_const_scope(choice.get("scope", prefer_scope), default="experiment")
        return FreeConst(name, tag=name, init=float(init), scope=scope)

    return Scale(name=str(base_tag), tag=str(base_tag), init=float(init))


__all__ = [
    "safe_const_token",
    "matching_fixed_const_specs",
    "matching_free_const_specs",
    "pick_declared_free_const",
    "scalar_constant_variants",
    "build_scalar_atom_from_variant",
    "unit_aware_scalar_choice",
    "make_unit_aware_scalar_atom",
]
