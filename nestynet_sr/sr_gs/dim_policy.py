# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Dimensional-decision policy adapter for GS unit-torus checks."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping


@dataclass(frozen=True)
class DimensionalDecision:
    candidate: str
    policy: str
    validator: str
    baseline_accept: bool | None
    gs_accept: bool | None
    final_accept: bool
    reason: str = ""
    both_rule: str = "rref-dominates"
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["metadata"] = dict(self.metadata or {})
        return out


def canonical_dim_policy(policy: Any) -> str:
    p = str(policy or "audit").strip().lower().replace("_", "-")
    aliases = {
        "rref": "baseline",
        "baseline-only": "baseline",
        "report": "audit",
        "gs": "gs-only",
        "gsonly": "gs-only",
        "replace": "replace-rref",
    }
    p = aliases.get(p, p)
    if p not in {"baseline", "audit", "augment", "both", "replace-rref", "gs-only"}:
        p = "audit"
    return p


def canonical_both_rule(rule: Any) -> str:
    r = str(rule or "rref-dominates").strip().lower().replace("_", "-")
    aliases = {"baseline-dominates": "rref-dominates", "baseline-wins": "rref-dominates", "gs-wins": "gs-dominates"}
    r = aliases.get(r, r)
    if r not in {"rref-dominates", "require-both", "either", "gs-dominates"}:
        r = "rref-dominates"
    return r


def canonical_dim_validator(validator: Any) -> str:
    v = str(validator or "nullspace").strip().lower().replace("_", "-")
    if v not in {"local", "nullspace", "linear"}:
        v = "nullspace"
    return v


def _bool_or_default(value: bool | None, default: bool) -> bool:
    return bool(default) if value is None else bool(value)


def combine_dimensional_decision(
    *,
    candidate: str,
    baseline_accept: bool | None,
    gs_accept: bool | None,
    policy: Any = "audit",
    both_rule: Any = "rref-dominates",
    validator: Any = "nullspace",
    reason: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> DimensionalDecision:
    p = canonical_dim_policy(policy)
    r = canonical_both_rule(both_rule)
    v = canonical_dim_validator(validator)
    baseline = _bool_or_default(baseline_accept, True)
    gs = _bool_or_default(gs_accept, baseline)

    if p == "baseline":
        final = baseline
    elif p == "audit":
        final = baseline
    elif p == "augment":
        final = baseline
    elif p == "replace-rref":
        final = gs if gs_accept is not None else baseline
    elif p == "gs-only":
        final = gs if gs_accept is not None else False
    elif p == "both":
        if r == "require-both":
            final = baseline and gs
        elif r == "either":
            final = baseline or gs
        elif r == "gs-dominates":
            final = gs
        else:
            final = baseline
    else:
        final = baseline

    return DimensionalDecision(
        candidate=str(candidate),
        policy=p,
        validator=v,
        baseline_accept=baseline_accept,
        gs_accept=gs_accept,
        final_accept=bool(final),
        reason=str(reason or ""),
        both_rule=r,
        metadata=dict(metadata or {}),
    )


def should_record_decision(decision: DimensionalDecision, *, report_disagreements: bool = True) -> bool:
    if bool(report_disagreements) and decision.baseline_accept is not None and decision.gs_accept is not None:
        if bool(decision.baseline_accept) != bool(decision.gs_accept):
            return True
    return decision.policy not in {"baseline"} or decision.reason != ""
