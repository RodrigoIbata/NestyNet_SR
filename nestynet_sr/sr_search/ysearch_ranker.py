# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
"""Ranking helpers for virtual y-transform probes.

These utilities intentionally avoid hard exclusion based on heuristic probe
signals. All valid transforms remain eligible; metrics only affect ordering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class VirtualProbeHint:
    idx: int
    name: str
    domain_ok_frac: float
    candidate_flag: bool
    sep_has_split: bool
    sep_proposals: int
    # Soft structure signals from virtual probing.
    trig_strength: float = 0.0
    # Proxy fit in transformed label space (lower is better).
    virtual_mse: float = float("inf")
    # Certificate from a full-compound z probe: φ(y) ≈ a*z + b.
    outer_affine_confirmed: bool = False
    outer_affine_rms_rel: float = float("inf")
    outer_affine_domain_ok_frac: float = 0.0
    # A direct certificate on >=2 raw axes, transported through an exact
    # homogeneous output map such as square, sqrt, or reciprocal.
    joint_homogeneity_verified: bool = False
    joint_homogeneity_indices: tuple[int, ...] = ()
    joint_homogeneity_degree: float = float("nan")
    joint_homogeneity_rel_std: float = float("inf")
    joint_homogeneity_n_points: int = 0


def derive_joint_homogeneity_certificate(base_specs, homogeneity_power):
    """Transport the best direct joint certificate through ``phi(y)=y**p``."""
    try:
        power = float(homogeneity_power)
    except Exception:
        return None
    if not math.isfinite(power) or power == 0.0:
        return None

    candidates = []
    for spec in base_specs or []:
        try:
            indices = tuple(int(i) for i in getattr(spec, "indices", ()) or ())
            degree = float(getattr(spec, "oracle_k", None))
            rel_std = float(getattr(spec, "oracle_rel_std", None))
            n_points = int(getattr(spec, "n_points", 0))
            if (
                len(indices) < 2
                or not bool(getattr(spec, "oracle_verified", False))
                or not math.isfinite(degree)
                or not math.isfinite(rel_std)
                or n_points <= 0
            ):
                continue
            candidates.append((rel_std, len(indices), indices, -n_points, degree, n_points))
        except Exception:
            continue
    if not candidates:
        return None
    rel_std, _arity, indices, _neg_n, degree, n_points = min(candidates)
    return {
        "verified": True,
        "indices": indices,
        "degree": float(power * degree),
        "rel_std": float(rel_std),
        "n_points": int(n_points),
        "transform_power": float(power),
    }


def _to_int_or_zero(v) -> int:
    try:
        return max(0, int(v))
    except Exception:
        return 0


def _to_float_or_zero(v) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _to_float_or_inf(v) -> float:
    try:
        out = float(v)
        return out if out >= 0.0 else float("inf")
    except Exception:
        return float("inf")


def _finite_pos(v) -> bool:
    try:
        f = float(v)
        return math.isfinite(f) and f > 0.0
    except Exception:
        return False


def _median(vals: List[float]) -> float:
    if not vals:
        return float("nan")
    vv = sorted(float(v) for v in vals)
    return vv[len(vv) // 2]


def _vmse_band(v: float, ref: float, *, decade_step: float = 1.5) -> int:
    """Coarse 'badness' band for vmse relative to a reference.

    band=0 means 'close' to ref (within ~10**decade_step).
    Larger bands indicate progressively worse transformed-space stability.
    """
    if not _finite_pos(v) or not _finite_pos(ref):
        return 999
    try:
        decades = math.log10(float(v) / float(ref))
    except Exception:
        return 999
    if decades <= 0.0:
        return 0
    step = float(decade_step) if float(decade_step) > 0.0 else 1.0
    return int(max(0.0, decades) // step)


def rank_virtual_hints(hints: Iterable[VirtualProbeHint]) -> List[VirtualProbeHint]:
    """Deterministically rank virtual probe hints.

    Order (best first):
    1) exact/full-compound outer-affine certificates
    2) transforms with detected split opportunities
    3) more split proposals
    4) lower virtual-MSE "band" (guardrail against trig false positives)
    5) stronger trig signal from virtual probing
    6) lower transformed-space proxy MSE
    7) transforms already flagged by candidate-sep quickscan (soft preference)
    8) higher domain coverage
    9) deterministic tie-break by name, then index
    """
    ranked = list(hints)
    vmse_ref = _median(
        [
            float(getattr(h, "virtual_mse", float("inf")))
            for h in ranked
            if _finite_pos(getattr(h, "virtual_mse", None))
        ]
    )
    ranked.sort(
        key=lambda h: (
            0 if bool(getattr(h, "outer_affine_confirmed", False)) else 1,
            _to_float_or_inf(getattr(h, "outer_affine_rms_rel", float("inf"))),
            0 if bool(h.sep_has_split) else 1,
            -_to_int_or_zero(h.sep_proposals),
            _vmse_band(_to_float_or_inf(getattr(h, "virtual_mse", float("inf"))), vmse_ref),
            -_to_float_or_zero(getattr(h, "trig_strength", 0.0)),
            _to_float_or_inf(getattr(h, "virtual_mse", float("inf"))),
            0 if bool(h.candidate_flag) else 1,
            -max(_to_float_or_zero(h.domain_ok_frac), _to_float_or_zero(getattr(h, "outer_affine_domain_ok_frac", 0.0))),
            str(h.name),
            _to_int_or_zero(h.idx),
        )
    )
    return ranked


def select_virtual_portfolio(
    hints: Iterable[VirtualProbeHint],
    expand_k: int,
    *,
    margin_decades: float = 0.0,
    max_k: int | None = None,
) -> tuple[List[int], dict[int, str]]:
    ranked = rank_virtual_hints(hints)
    k = max(1, _to_int_or_zero(expand_k))
    selected = list(ranked[:k])
    reasons = {int(h.idx): "top_k" for h in selected}

    # Portfolio mode: when the kth and later candidates are effectively tied
    # by proxy loss, carry the nearby alternatives forward instead of letting
    # a soft ranker prune them.  Hard certificates are retained regardless.
    try:
        margin = max(0.0, float(margin_decades))
    except Exception:
        margin = 0.0
    cap = len(ranked) if max_k is None else max(k, _to_int_or_zero(max_k))
    kth_vmse = (
        _to_float_or_inf(getattr(ranked[min(k, len(ranked)) - 1], "virtual_mse", float("inf")))
        if ranked
        else float("inf")
    )
    vmse_limit = kth_vmse * (10.0 ** margin) if _finite_pos(kth_vmse) else float("inf")

    selected_ids = {int(h.idx) for h in selected}
    for h in ranked[k:]:
        idx = int(h.idx)
        if idx in selected_ids:
            continue
        certified = bool(getattr(h, "outer_affine_confirmed", False))
        tied = bool(margin > 0.0 and _to_float_or_inf(getattr(h, "virtual_mse", float("inf"))) <= vmse_limit)
        if certified or (tied and len(selected) < cap):
            selected.append(h)
            selected_ids.add(idx)
            reasons[idx] = (
                "outer_affine_certificate" if certified else "proxy_loss_margin"
            )

    # One structural insurance slot: if ordinary ranking/tie expansion did
    # not retain a joint-homogeneity-certified transform, carry the first
    # ranked omitted certificate.  This schedules one full fit; it does not
    # change any downstream acceptance or certification rule.
    have_joint = any(
        bool(getattr(h, "joint_homogeneity_verified", False)) for h in selected
    )
    if not have_joint:
        for h in ranked:
            idx = int(h.idx)
            if idx in selected_ids:
                continue
            if bool(getattr(h, "joint_homogeneity_verified", False)):
                selected.append(h)
                selected_ids.add(idx)
                reasons[idx] = "joint_homogeneity_reserve"
                break

    return [int(h.idx) for h in selected], reasons


def select_virtual_indices(
    hints: Iterable[VirtualProbeHint],
    expand_k: int,
    *,
    margin_decades: float = 0.0,
    max_k: int | None = None,
) -> List[int]:
    selected, _reasons = select_virtual_portfolio(
        hints,
        expand_k,
        margin_decades=margin_decades,
        max_k=max_k,
    )
    return selected
