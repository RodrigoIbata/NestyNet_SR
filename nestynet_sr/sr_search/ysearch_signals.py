# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
"""Structural progress policy for y-search branch confirmation."""

from __future__ import annotations

import math
from typing import Iterable


def _to_float(v, default: float = 0.0) -> float:
    try:
        f = float(v)
        if math.isfinite(f):
            return f
    except Exception:
        pass
    return float(default)


def normalize_signal_dict(signals) -> dict[str, float]:
    """Convert a possibly mixed signal mapping to finite float values."""
    out: dict[str, float] = {}
    if not isinstance(signals, dict):
        return out
    for k, v in signals.items():
        key = str(k)
        try:
            f = float(v)
            if math.isfinite(f):
                out[key] = f
        except Exception:
            continue
    return out


def structural_progress(
    parent_signals,
    child_signals,
    *,
    trig_affine_thr: float = 0.90,
    sep_min_thr: float = 0.70,
    sep_delta_thr: float = 0.20,
    split_score_thr: float = 0.85,
    split_margin_thr: float = 0.10,
) -> tuple[bool, list[str]]:
    """Return whether a child y-branch has *confirmed* structural progress.

    The important distinction is deliberately encoded here: cheap Stage-A
    signals can rank or seed branches, but only certificate-like events may
    confirm a branch.  In particular, a pure NN[z(x)] compression is reported
    as ``full_compound_compressed`` and is not, by itself, a simplification.
    """
    p = normalize_signal_dict(parent_signals)
    c = normalize_signal_dict(child_signals)
    confirmed: list[str] = []
    provisional: list[str] = []

    p_sep = _to_float(p.get("sep_score", 0.0))
    c_sep = _to_float(c.get("sep_score", 0.0))
    if c_sep >= float(sep_min_thr) and (c_sep - p_sep) >= float(sep_delta_thr):
        provisional.append("sep_score_up")

    p_split = _to_float(p.get("best_split_score", 0.0))
    c_split = _to_float(c.get("best_split_score", 0.0))
    if c_split >= float(split_score_thr) and (c_split - p_split) >= float(split_margin_thr):
        provisional.append("split_score_up")

    c_split_success = _to_float(c.get("split_success", 0.0))
    if c_split_success >= 0.5:
        confirmed.append("split_success")

    p_outer = _to_float(p.get("outer_affine_confirmed", 0.0))
    c_outer = _to_float(c.get("outer_affine_confirmed", 0.0))
    if c_outer >= 0.5 and p_outer < 0.5:
        confirmed.append("outer_affine_confirmed")

    p_stageb = _to_float(p.get("stageB_confirmed", 0.0))
    c_stageb = _to_float(c.get("stageB_confirmed", 0.0))
    if c_stageb >= 0.5 and p_stageb < 0.5:
        confirmed.append("stageB_confirmed")

    p_analytic = _to_float(p.get("analytic_rewrite_confirmed", 0.0))
    c_analytic = _to_float(c.get("analytic_rewrite_confirmed", 0.0))
    if c_analytic >= 0.5 and p_analytic < 0.5:
        confirmed.append("analytic_rewrite_confirmed")

    p_trig = _to_float(p.get("trig_affine_conf", 0.0))
    c_trig = _to_float(c.get("trig_affine_conf", 0.0))
    if c_trig >= float(trig_affine_thr) and p_trig < float(trig_affine_thr):
        provisional.append("trig_affine_up")

    p_simple_ok = _to_float(p.get("simplicity_hint_ok", 0.0))
    c_simple_ok = _to_float(c.get("simplicity_hint_ok", 0.0))
    if c_simple_ok >= 0.5 and p_simple_ok < 0.5:
        provisional.append("simplicity_hint_ok")

    p_logq = _to_float(p.get("logquad_ok", 0.0))
    c_logq = _to_float(c.get("logquad_ok", 0.0))
    if c_logq >= 0.5 and p_logq < 0.5:
        provisional.append("logquad_ok")

    p_sqq = _to_float(p.get("squarequad_ok", 0.0))
    c_sqq = _to_float(c.get("squarequad_ok", 0.0))
    if c_sqq >= 0.5 and p_sqq < 0.5:
        provisional.append("squarequad_ok")

    p_full = max(
        _to_float(p.get("full_compound_compressed", 0.0)),
        _to_float(p.get("full_compound_solved", 0.0)),
    )
    c_full = max(
        _to_float(c.get("full_compound_compressed", 0.0)),
        _to_float(c.get("full_compound_solved", 0.0)),
    )
    if c_full >= 0.5 and p_full < 0.5:
        provisional.append("full_compound_compressed")

    reasons = confirmed + [f"provisional:{r}" for r in provisional]
    return bool(confirmed), reasons


def format_progress_reasons(reasons: Iterable[str]) -> str:
    vals = [str(r) for r in reasons if str(r)]
    return ",".join(vals)
