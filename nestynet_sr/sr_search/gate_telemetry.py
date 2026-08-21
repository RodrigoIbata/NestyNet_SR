# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Central collector for Stage-B gate decisions (margin telemetry).

The benchmark score is a step function of continuous pipeline internals: any
legitimate surrogate change can reshuffle candidates sitting near an absolute
accept/reject threshold. Recording the signed margin of every gate decision
turns that fragility into a measurement: one benchmark run yields a flip-risk
watchlist and data-driven hardening queue, and future regressions can be
triaged by diffing margins instead of hunting.

Usage at a gate site (purely additive, never changes the decision):

    from .gate_telemetry import record_gate
    record_gate("symexp_denom_1d", "rel_rms", rms_rel, rel_rms_threshold,
                accepted=bool(rms_rel <= rel_rms_threshold),
                context={"variant": "std"})
    if rms_rel <= rel_rms_threshold:
        ...

Records accumulate in a module-level list and are drained into the report
JSON by ``write_json_report`` (each suite problem runs in its own subprocess,
so records cannot leak across problems).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

__all__ = ["record_gate", "snapshot", "drain", "reset", "summarize"]

_RECORDS: List[Dict[str, Any]] = []


def _jsonable(v: Any) -> Any:
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return str(v)


def record_gate(
    rule: str,
    gate: str,
    value: Any,
    threshold: Any,
    accepted: bool,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Record one threshold decision. Never raises; never changes behavior."""
    try:
        try:
            value_f: Optional[float] = float(value)
        except (TypeError, ValueError):
            value_f = None
        try:
            thr_f: Optional[float] = float(threshold)
        except (TypeError, ValueError):
            thr_f = None

        margin_ratio: Optional[float] = None
        if (
            value_f is not None
            and thr_f is not None
            and math.isfinite(value_f)
            and math.isfinite(thr_f)
            and thr_f > 0.0
        ):
            margin_ratio = value_f / thr_f

        rec: Dict[str, Any] = {
            "rule": str(rule),
            "gate": str(gate),
            "value": _jsonable(value_f),
            "threshold": _jsonable(thr_f),
            "accepted": bool(accepted),
            "margin_ratio": _jsonable(margin_ratio),
        }
        if context:
            rec["context"] = {str(k): _jsonable(v) for k, v in context.items()}
        _RECORDS.append(rec)
    except Exception:
        # Telemetry must never break the pipeline.
        pass


def snapshot() -> List[Dict[str, Any]]:
    """Return a copy of the accumulated records without clearing them."""
    return list(_RECORDS)


def drain() -> List[Dict[str, Any]]:
    """Return the accumulated records and clear the collector."""
    out = list(_RECORDS)
    _RECORDS.clear()
    return out


def reset() -> None:
    _RECORDS.clear()


def summarize(records: List[Dict[str, Any]], band: float = 1.5) -> Dict[str, Any]:
    """Compact summary: which decisions sit within ``band`` of flipping.

    A record is flip-risk when 1/band <= margin_ratio <= band: the statistic
    is within a factor ``band`` of its threshold, so a small upstream change
    (e.g. a surrogate nudge) can flip the decision.
    """
    flip_risk = []
    for r in records:
        m = r.get("margin_ratio")
        if m is None or not isinstance(m, (int, float)) or m <= 0:
            continue
        if (1.0 / band) <= m <= band:
            flip_risk.append(
                {
                    "rule": r.get("rule"),
                    "gate": r.get("gate"),
                    "value": r.get("value"),
                    "threshold": r.get("threshold"),
                    "margin_ratio": m,
                    "accepted": r.get("accepted"),
                }
            )
    flip_risk.sort(key=lambda r: abs(math.log(r["margin_ratio"])))
    return {
        "n_records": len(records),
        "flip_risk_band": band,
        "n_flip_risk": len(flip_risk),
        "flip_risk": flip_risk,
    }
