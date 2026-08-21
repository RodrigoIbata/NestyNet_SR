# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Stage-A replace-shadowed policy: suppress legacy proposals GS subsumes.

Under ``--gs-policy replace-shadowed`` a *promoted* GS reduction may suppress
the legacy detector proposals it provably shadows.  Matching is deliberately
conservative — exact variable-support equality plus a content proof (projective
exponent key for monomials, covector direction for linear compounds), never a
subset match.  Legacy variants carrying extra hypotheses (``extra_override``,
prefactor peels, retained-axis wrappers, subset compounds) encode different
structural claims than pure invariance ``f = g(z)`` and are never suppressed.
The worst failure mode is therefore under-suppression, which is the status quo.

Under the default ``augment`` policy this module is never invoked.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .reporting import record_policy_event
from .unit_torus import projective_exponent_key

# Meta keys that mark a legacy monomial proposal as a different hypothesis than
# pure invariance in the proposed coordinate (prefactor peel, retained axes,
# subset retries).  Their presence always blocks suppression.
_EXTRA_HYPOTHESIS_META_KEYS = (
    "prefactor_exponents",
    "retained_axis_wrapper",
    "compound_subset",
)

_LINEAR_GS_COORDINATE_KINDS = {"linear_projection", "translation_invariant_linear"}
_LINEAR_COS_TOL = 1.0e-8


def suppress_shadowed_stagea_proposals(
    legacy_proposals: Sequence[tuple],
    gs_proposals: Sequence[tuple],
    *,
    cols: Sequence[int],
    cfg: Any,
) -> tuple[list[tuple], list[dict[str, Any]]]:
    """Return (filtered legacy proposals, suppression events).

    Never mutates its inputs and never touches GS entries; with no promoted GS
    proposal present it is a strict pass-through.
    """

    legacy_list = list(legacy_proposals)
    cols_t = tuple(int(c) for c in cols)
    monomial_keys: dict[tuple, set[tuple]] = {}
    monomial_meta: dict[tuple, dict[str, Any]] = {}
    linear_covectors: dict[tuple, list[tuple[np.ndarray, dict[str, Any]]]] = {}
    radial_meta: dict[tuple, dict[str, Any]] = {}
    quadratic_sigs: dict[tuple, tuple[tuple, dict[str, Any]]] = {}
    dp_metas: list[dict[str, Any]] = []
    for prop in gs_proposals:
        meta = _promoted_gs_meta(prop)
        if meta is None:
            continue
        support = _support_from_mask(prop[0], cols_t)
        if not support:
            continue
        exps_key = meta.get("gs_monomial_exponents_key")
        if exps_key:
            key = tuple(int(v) for v in exps_key)
            monomial_keys.setdefault(support, set()).add(key)
            monomial_meta.setdefault(support, meta)
        covector = meta.get("gs_linear_covector")
        if covector is not None and str(meta.get("gs_coordinate_kind", "")) in _LINEAR_GS_COORDINATE_KINDS:
            vec = np.asarray([float(v) for v in covector], dtype=float)
            if vec.size == len(cols_t) and float(np.linalg.norm(vec)) > 0.0:
                linear_covectors.setdefault(support, []).append((vec, meta))
        if meta.get("gs_radial_support") is not None and str(meta.get("form", "")) == "r2":
            radial_meta.setdefault(support, meta)
        signature = meta.get("gs_quadratic_signature")
        if signature is not None:
            quadratic_sigs.setdefault(support, (tuple(int(v) for v in signature), meta))
        if str(meta.get("gs_kind", "")) == "pairwise_composed_difference_product":
            dp_metas.append(meta)
    if not monomial_keys and not linear_covectors and not radial_meta and not quadratic_sigs and not dp_metas:
        return legacy_list, []

    filtered: list[tuple] = []
    events: list[dict[str, Any]] = []
    policy = _canonical_policy(cfg)
    for prop in legacy_list:
        shadow = _shadowing_match(
            prop, cols_t, monomial_keys, monomial_meta, linear_covectors, radial_meta, quadratic_sigs, dp_metas
        )
        if shadow is None:
            filtered.append(prop)
            continue
        gs_meta, legacy_kind, legacy_pattern = shadow
        event = {
            "cols": list(cols_t),
            "legacy_kind": str(legacy_kind),
            "legacy_pattern": [int(v) for v in legacy_pattern],
            "legacy_confidence": _confidence_of(prop),
            "gs_chart": str(gs_meta.get("gs_chart", "identity")),
            "gs_coordinate_kind": str(gs_meta.get("gs_coordinate_kind", "")),
            "gs_exponents_key": list(gs_meta.get("gs_monomial_exponents_key") or ()),
            "gs_confidence": float(gs_meta.get("gs_confidence", 0.0)),
            "z_human": str(gs_meta.get("z_human", "")),
        }
        events.append(event)
        try:
            record_policy_event(
                policy=policy,
                action="stagea_replace_shadowed_suppression",
                details=event,
            )
        except Exception:
            pass
    return filtered, events


def _shadowing_match(
    prop: tuple,
    cols_t: tuple,
    monomial_keys: dict[tuple, set[tuple]],
    monomial_meta: dict[tuple, dict[str, Any]],
    linear_covectors: dict[tuple, list[tuple[np.ndarray, dict[str, Any]]]],
    radial_meta: dict[tuple, dict[str, Any]] | None = None,
    quadratic_sigs: dict[tuple, tuple[tuple, dict[str, Any]]] | None = None,
    dp_metas: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str, tuple] | None:
    """Return (gs_meta, legacy_kind, legacy_pattern) when ``prop`` is shadowed."""

    try:
        if not isinstance(prop, tuple) or len(prop) < 3:
            return None
        meta = prop[4] if len(prop) >= 5 and isinstance(prop[4], dict) else {}
        # Meta-less 3/4-tuples are monomial proposals by construction (they are
        # normalized to kind="monomial" immediately after the merge point).
        kind = str(meta.get("kind", "monomial"))
        if str(meta.get("source", "")) == "generalized_symmetry":
            return None
        pattern = prop[0]
        if not isinstance(pattern, (tuple, list)) or len(pattern) != len(cols_t):
            return None
        pattern_t = tuple(pattern)
        if not all(isinstance(v, (int, float, np.integer, np.floating)) for v in pattern_t):
            return None
        extra_override = prop[3] if len(prop) >= 4 else None
        if kind == "monomial":
            if extra_override:
                return None
            if any(meta.get(k) for k in _EXTRA_HYPOTHESIS_META_KEYS):
                return None
            support = tuple(c for c, v in zip(cols_t, pattern_t) if float(v) != 0.0)
            keys = monomial_keys.get(support)
            if not keys:
                return None
            legacy_key = tuple(int(v) for v in projective_exponent_key(pattern_t))
            if legacy_key in keys:
                return monomial_meta[support], kind, pattern_t
            return None
        if kind == "power_difference":
            # z = x_i**n - x_j**n. n=1 is a linear ray; n=2 a boost quadratic.
            if extra_override or meta.get("extra_override"):
                return None
            power = int(meta.get("power", 0) or 0)
            indices = tuple(int(v) for v in (meta.get("indices") or ()))
            if len(indices) != 2:
                return None
            if power == 1:
                support = tuple(c for c, v in zip(cols_t, pattern_t) if float(v) != 0.0)
                matched = _linear_direction_match(pattern_t, support, linear_covectors)
                if matched is not None:
                    return matched, kind, pattern_t
                return None
            if power == 2:
                expected = tuple(
                    1 if c == indices[0] else (-1 if c == indices[1] else 0) for c in cols_t
                )
                support = tuple(sorted(indices))
                entry = (quadratic_sigs or {}).get(support)
                if entry is not None:
                    signature, gs_meta = entry
                    if signature == expected or signature == tuple(-v for v in expected):
                        return gs_meta, kind, pattern_t
                return None
            return None
        if kind == "power_pair_sumdiff":
            # z = x_i**n (+/-) x_j**n, optionally with reciprocal factors.
            if extra_override or meta.get("extra_override"):
                return None
            if bool(meta.get("left_inverse", False)) or bool(meta.get("right_inverse", False)):
                # Reciprocal variants are non-affine in every chart: never ours.
                return None
            power = int(meta.get("power", 0) or 0)
            support = tuple(c for c, v in zip(cols_t, pattern_t) if float(v) != 0.0)
            if power == 1:
                matched = _linear_direction_match(pattern_t, support, linear_covectors)
                if matched is not None:
                    return matched, kind, pattern_t
                return None
            if power == 2:
                expected = tuple(int(round(float(v))) for v in pattern_t)
                entry = (quadratic_sigs or {}).get(support)
                if entry is not None:
                    signature, gs_meta = entry
                    if signature == expected or signature == tuple(-v for v in expected):
                        return gs_meta, kind, pattern_t
                return None
            return None
        if kind == "power_diffprod":
            # z = (x_i**n - x_j**n) * x_k**p. The meta's prefactor_exponents
            # describes the in-z factor x_k**p, not an extra peel hypothesis,
            # so only extra_override blocks suppression here.
            if extra_override or meta.get("extra_override"):
                return None
            power = int(meta.get("power", 0) or 0)
            outer = int(meta.get("outer_power", 0) or 0)
            indices = tuple(int(v) for v in (meta.get("indices") or ()))
            if power != 1 or outer not in (-1, 1) or len(indices) != 3:
                return None
            i_idx, j_idx, k_idx = indices
            virtual_support = tuple(sorted((i_idx, j_idx)))
            expected_coeffs = (1, -1) if virtual_support == (i_idx, j_idx) else (-1, 1)
            for gs_meta in dp_metas or ():
                if tuple(gs_meta.get("gs_dp_virtual_support") or ()) != virtual_support:
                    continue
                coeffs = tuple(int(v) for v in (gs_meta.get("gs_dp_virtual_coeffs") or ()))
                if coeffs != expected_coeffs and coeffs != tuple(-v for v in expected_coeffs):
                    continue
                axis_exps = tuple(
                    (int(a), int(e)) for a, e in (gs_meta.get("gs_dp_axis_exponents") or ())
                )
                if axis_exps != ((k_idx, outer),):
                    continue
                return gs_meta, kind, pattern_t
            return None
        if kind == "radial":
            if extra_override:
                return None
            if str(meta.get("form", "")) != "r2":
                # sqrt-only or exotic radial variants are different hypotheses.
                return None
            support = tuple(c for c, v in zip(cols_t, pattern_t) if float(v) != 0.0)
            gs_meta = (radial_meta or {}).get(support)
            if gs_meta is not None:
                return gs_meta, kind, pattern_t
            return None
        if kind == "linear":
            if extra_override:
                return None
            support = tuple(c for c, v in zip(cols_t, pattern_t) if float(v) != 0.0)
            matched = _linear_direction_match(pattern_t, support, linear_covectors)
            if matched is not None:
                return matched, kind, pattern_t
            return None
    except Exception:
        return None
    return None


def _linear_direction_match(
    pattern_t: tuple,
    support: tuple,
    linear_covectors: dict[tuple, list[tuple[np.ndarray, dict[str, Any]]]],
) -> dict[str, Any] | None:
    """Promoted GS linear covector matching a legacy coefficient pattern."""

    candidates = linear_covectors.get(support)
    if not candidates:
        return None
    coeffs = np.asarray([float(v) for v in pattern_t], dtype=float)
    norm = float(np.linalg.norm(coeffs))
    if norm <= 0.0:
        return None
    unit = coeffs / norm
    for vec, gs_meta in candidates:
        vec_unit = vec / float(np.linalg.norm(vec))
        if abs(float(np.dot(unit, vec_unit))) >= 1.0 - _LINEAR_COS_TOL:
            return gs_meta
    return None


def _promoted_gs_meta(prop: Any) -> dict[str, Any] | None:
    try:
        if not isinstance(prop, tuple) or len(prop) < 5 or not isinstance(prop[4], dict):
            return None
        meta = prop[4]
        if str(meta.get("kind", "")) != "gs_promoted_reduction":
            return None
        if str(meta.get("gs_promotion_state", "")) != "promoted":
            return None
        return meta
    except Exception:
        return None


def _support_from_mask(pattern: Any, cols_t: tuple) -> tuple:
    try:
        mask = tuple(pattern)
        if len(mask) != len(cols_t):
            return ()
        return tuple(int(c) for c, v in zip(cols_t, mask) if int(v) != 0)
    except Exception:
        return ()


def _confidence_of(prop: tuple) -> float:
    try:
        return float(prop[2])
    except Exception:
        return 0.0


def _canonical_policy(cfg: Any) -> str:
    try:
        return str(cfg.canonical_policy())
    except Exception:
        return str(getattr(cfg, "policy", "replace-shadowed") or "replace-shadowed")
