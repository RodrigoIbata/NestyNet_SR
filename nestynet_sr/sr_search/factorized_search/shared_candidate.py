# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


SHARED_CANDIDATE_SCHEMA_VERSION = 1

SHARED_CANDIDATE_EVIDENCE_LEVELS: tuple[str, ...] = (
    "preview_only",
    "preview_support",
    "exact_known",
)

SHARED_CANDIDATE_MASK_FIELD_NAMES: tuple[str, ...] = (
    "candidate_has_path",
    "candidate_has_path_source",
    "candidate_has_target_mode",
    "candidate_has_provenance",
    "candidate_route_valid_repair",
    "candidate_route_valid_build",
    "candidate_evidence_preview_only",
    "candidate_evidence_preview_support",
    "candidate_evidence_exact_known",
)


def _normalize_action_name(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _normalize_route_source(value: Any, *, action_name: str = "") -> str:
    route = str(value or "").strip().lower().replace("-", "_")
    if route in {"repair", "build"}:
        return route
    if action_name in {"inv_steer", "repair_option"}:
        return "repair"
    return "build"


def _normalize_mode_name(value: Any) -> str:
    mode = str(value or "").strip().lower().replace("-", "_")
    if mode in ("id",):
        return "identity"
    if mode in ("fitbest", "fitted", "legacy"):
        return "full"
    return mode


def _normalize_path_source(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    if token.startswith("critic"):
        return "critic"
    if token.startswith("inverse"):
        return "inverse"
    if token == "random":
        return "random"
    return token


def _normalize_evidence_level(value: Any) -> str:
    level = str(value or "").strip().lower().replace("-", "_")
    if level in SHARED_CANDIDATE_EVIDENCE_LEVELS:
        return level
    return ""


def _bool01(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _derive_evidence_level(row: Mapping[str, Any]) -> str:
    explicit = _normalize_evidence_level(row.get("evidence_level", ""))
    if explicit:
        return explicit
    if bool(row.get("exact_child_score_observed", False)):
        return "exact_known"
    prov_rows = row.get("provenance_rows", None)
    prov_count = row.get("provenance_count", None)
    try:
        prov_count_value = int(prov_count) if prov_count is not None else 0
    except Exception:
        prov_count_value = 0
    if prov_count_value > 1:
        return "preview_support"
    if isinstance(prov_rows, (list, tuple)) and len(prov_rows) > 1:
        return "preview_support"
    if bool(row.get("provenance_grouped", False)):
        return "preview_support"
    return "preview_only"


@dataclass(frozen=True)
class SharedCandidateRecord:
    route_source: str
    action: str
    child_key: str
    child_expr: str
    path: tuple[int, ...] = ()
    path_source: str = ""
    target_mode: str = ""
    evidence_level: str = "preview_only"
    exact_child_score_observed: bool = False
    proposal_family: str = ""
    generation_source: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SHARED_CANDIDATE_SCHEMA_VERSION

    @property
    def masks(self) -> dict[str, float]:
        evidence = _normalize_evidence_level(self.evidence_level) or "preview_only"
        has_provenance = False
        prov_count = self.payload.get("provenance_count", None)
        try:
            has_provenance = int(prov_count) > 1
        except Exception:
            has_provenance = False
        if not has_provenance:
            prov_rows = self.payload.get("provenance_rows", None)
            has_provenance = isinstance(prov_rows, (list, tuple)) and len(prov_rows) > 0
        return {
            "candidate_has_path": _bool01(bool(self.path)),
            "candidate_has_path_source": _bool01(bool(self.path_source)),
            "candidate_has_target_mode": _bool01(bool(self.target_mode)),
            "candidate_has_provenance": _bool01(has_provenance),
            "candidate_route_valid_repair": _bool01(self.route_source == "repair"),
            "candidate_route_valid_build": _bool01(self.route_source == "build"),
            "candidate_evidence_preview_only": _bool01(evidence == "preview_only"),
            "candidate_evidence_preview_support": _bool01(evidence == "preview_support"),
            "candidate_evidence_exact_known": _bool01(evidence == "exact_known"),
        }

    def to_row(self) -> dict[str, Any]:
        row = dict(self.payload)
        row.update({
            "shared_candidate_schema_version": int(self.schema_version),
            "route_source": str(self.route_source),
            "action": str(self.action),
            "child_key": str(self.child_key),
            "child_expr": str(self.child_expr),
            "path": [int(v) for v in self.path],
            "path_source": str(self.path_source),
            "target_mode": str(self.target_mode),
            "evidence_level": str(self.evidence_level),
            "exact_child_score_observed": bool(self.exact_child_score_observed),
            "proposal_family": str(self.proposal_family),
            "generation_source": str(self.generation_source),
        })
        row.update(self.masks)
        return row


def coerce_shared_candidate_record(
    row: Mapping[str, Any] | SharedCandidateRecord | None,
    *,
    route_source: str = "",
) -> SharedCandidateRecord:
    if isinstance(row, SharedCandidateRecord):
        return row
    row_map = row if isinstance(row, Mapping) else {}
    action_name = _normalize_action_name(row_map.get("action", ""))
    route_name = _normalize_route_source(route_source or row_map.get("route_source", ""), action_name=action_name)
    path_like = row_map.get("path", ())
    try:
        path = tuple(int(v) for v in (path_like or ()))
    except Exception:
        path = ()
    target_mode = _normalize_mode_name(row_map.get("target_mode", ""))
    path_source = _normalize_path_source(row_map.get("path_source", ""))
    child_key = str(row_map.get("child_key", "") or row_map.get("child_expr", "") or "").strip()
    child_expr = str(row_map.get("child_expr", "") or child_key).strip()
    tuple_provenance = str(row_map.get("tuple_provenance", "") or "").strip().lower().replace("-", "_")
    proposal_family = str(row_map.get("proposal_family", "") or tuple_provenance or action_name or route_name)
    generation_source = str(
        row_map.get("generation_source", "") or tuple_provenance or path_source or action_name or route_name
    )
    evidence_level = _derive_evidence_level(row_map)
    return SharedCandidateRecord(
        route_source=route_name,
        action=action_name,
        child_key=child_key,
        child_expr=child_expr,
        path=path,
        path_source=path_source,
        target_mode=target_mode if route_name == "repair" else "",
        evidence_level=evidence_level,
        exact_child_score_observed=bool(row_map.get("exact_child_score_observed", False)),
        proposal_family=proposal_family,
        generation_source=generation_source,
        payload=dict(row_map),
        schema_version=int(row_map.get("shared_candidate_schema_version", SHARED_CANDIDATE_SCHEMA_VERSION) or SHARED_CANDIDATE_SCHEMA_VERSION),
    )


def shared_candidate_row_dict(
    row: Mapping[str, Any] | SharedCandidateRecord | None,
    *,
    route_source: str = "",
) -> dict[str, Any]:
    return coerce_shared_candidate_record(row, route_source=route_source).to_row()
