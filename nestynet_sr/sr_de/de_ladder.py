# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Library-level DE proposal ladder helpers.

This is the migration layer between the current CLI-driven DE implementation
and the planned proposal-slate/committee architecture.  The first version
keeps legacy selection semantics while making the slate and committee decision
reusable outside ``run_de.py``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from .de_committee import run_de_committee_audit, selected_summary_from_decision
from .proposals import build_proposal_slate


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if Mapping is not None and isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


@dataclass
class DELadderPolicy:
    coe_mode: str = "off"
    source: str = "run_de"
    reservoir_scouts_requested: int = 0
    run_compile_domain: bool = True
    domain_samples: list[dict[str, Any]] | None = None
    committee_config: dict[str, Any] | None = None

    @property
    def mode(self) -> str:
        return str(self.coe_mode or "off").strip().lower()

    def to_committee_config(self) -> dict[str, Any]:
        cfg = dict(self.committee_config or {})
        cfg.setdefault("mode", self.mode)
        cfg.setdefault("source", str(self.source))
        cfg.setdefault("reservoir_scouts_requested", int(self.reservoir_scouts_requested))
        return cfg


@dataclass
class LegacyDEResultPayloads:
    first_line: dict[str, Any] | None = None
    factorized: dict[str, Any] | None = None
    factorized_search: dict[str, Any] | None = None
    selected: dict[str, Any] | None = None
    selected_engine: str | None = None


@dataclass
class DELadderReport:
    proposal_slate: list[dict[str, Any]]
    committee_decision: dict[str, Any] | None
    selected_payload: dict[str, Any] | None
    selected_engine: str
    internal_selected_engine: str
    internal_selected_payload: dict[str, Any] | None
    committee_adjudicated: bool
    committee_adjudication_fallback: bool

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def payload_from_committee_selection(
    proposal_slate: list[dict[str, Any]],
    committee_decision: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    summary = selected_summary_from_decision(committee_decision)
    if not isinstance(summary, dict):
        return None
    selected_id = str(summary.get("proposal_id", "") or "")
    for proposal in list(proposal_slate or []):
        if not isinstance(proposal, dict):
            continue
        if str(proposal.get("proposal_id", "") or "") != selected_id:
            continue
        payload = proposal.get("rhs_payload", None)
        if isinstance(payload, dict):
            out = dict(payload)
            if not out.get("engine"):
                out["engine"] = proposal.get("engine", None)
            if out.get("order") is None:
                out["order"] = proposal.get("order", None)
            if out.get("x_axis") is None:
                out["x_axis"] = proposal.get("x_axis", None)
            if not out.get("canonical_equation"):
                out["canonical_equation"] = proposal.get("canonical_equation", None)
            return out
    return None


def append_committee_warning(committee_decision: dict[str, Any] | None, warning: str) -> None:
    if not isinstance(committee_decision, dict):
        return
    warnings = committee_decision.get("warnings", None)
    if not isinstance(warnings, list):
        warnings = []
    warnings.append(str(warning))
    committee_decision["warnings"] = warnings


def run_legacy_de_ladder(
    payloads: LegacyDEResultPayloads,
    *,
    policy: DELadderPolicy | None = None,
) -> DELadderReport:
    """Build a proposal slate and optional committee decision from legacy payloads."""

    policy = policy or DELadderPolicy()
    selected_payload = dict(payloads.selected) if isinstance(payloads.selected, dict) else payloads.selected
    selected_engine = str(payloads.selected_engine or (selected_payload or {}).get("engine", "stlsq"))
    internal_selected_engine = str(selected_engine)
    internal_selected_payload = dict(selected_payload) if isinstance(selected_payload, dict) else selected_payload
    proposal_slate = build_proposal_slate(
        first_line=payloads.first_line,
        factorized=payloads.factorized,
        factorized_search=payloads.factorized_search,
        selected=selected_payload,
        selected_engine=selected_engine,
    )

    committee_decision = None
    committee_adjudicated = False
    committee_adjudication_fallback = False
    if policy.mode in {"audit", "adjudicate", "reservoir"}:
        committee_decision = run_de_committee_audit(
            proposal_slate,
            selected_engine=internal_selected_engine,
            config=policy.to_committee_config(),
            run_compile_domain=bool(policy.run_compile_domain),
            domain_samples=policy.domain_samples,
        ).to_dict()
        # In run_de.py, reservoir mode records a support-aware audit.  The
        # benchmark runner owns reservoir scouts and rollout adjudication.
        if policy.mode == "adjudicate":
            adjudicated_payload = payload_from_committee_selection(proposal_slate, committee_decision)
            adjudicated_summary = selected_summary_from_decision(committee_decision)
            if isinstance(adjudicated_payload, dict) and isinstance(adjudicated_summary, dict):
                selected_payload = adjudicated_payload
                selected_engine = str(
                    adjudicated_summary.get(
                        "engine",
                        selected_payload.get("engine", selected_engine),
                    )
                    or selected_engine
                )
                committee_adjudicated = True
            else:
                committee_adjudication_fallback = True
                append_committee_warning(
                    committee_decision,
                    "committee adjudication found no materializable selected proposal; kept legacy internal selection",
                )

    return DELadderReport(
        proposal_slate=proposal_slate,
        committee_decision=committee_decision,
        selected_payload=selected_payload,
        selected_engine=selected_engine,
        internal_selected_engine=internal_selected_engine,
        internal_selected_payload=internal_selected_payload,
        committee_adjudicated=committee_adjudicated,
        committee_adjudication_fallback=committee_adjudication_fallback,
    )


__all__ = [
    "DELadderPolicy",
    "DELadderReport",
    "LegacyDEResultPayloads",
    "append_committee_warning",
    "payload_from_committee_selection",
    "run_legacy_de_ladder",
]
