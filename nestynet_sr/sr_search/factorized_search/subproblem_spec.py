# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


SUBPROBLEM_SPEC_SCHEMA_NAME = "factorized_search.subproblem_spec"
SUBPROBLEM_SPEC_SCHEMA_VERSION = 1
FAMILY_EVIDENCE_HARD_CONSTRAINTS_SCHEMA_NAME = "factorized_search.family_evidence.hard_constraints"
FAMILY_EVIDENCE_HARD_CONSTRAINTS_SCHEMA_VERSION = 1

_FAMILY_REGIME_FIELD_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dataset_ids", ("dataset_ids",)),
    ("regime_ids", ("regime_ids", "constant_lift_regime_ids")),
    ("dataset_metadata", ("dataset_metadata", "constant_lift_dataset_metadata")),
    ("local_constants_by_experiment", ("local_constants_by_experiment",)),
    ("parameter_stability", ("parameter_stability",)),
    ("constant_name", ("constant_name", "constant_lift_constant_name")),
    ("task_source", ("task_source",)),
    ("feature_source", ("feature_source", "constant_lift_feature_source")),
    ("mean_cv", ("mean_cv",)),
    ("sample_count", ("sample_count",)),
    ("trigger_mean_cv", ("trigger_mean_cv",)),
)


def _clone_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clone_value(item) for key, item in dict(value).items()}
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    return value


def _tuple_ints(values: Any) -> tuple[int, ...]:
    out: list[int] = []
    for value in list(values or ()):
        try:
            out.append(int(value))
        except Exception:
            continue
    return tuple(out)


def _tuple_any(values: Any) -> tuple[Any, ...]:
    return tuple(_clone_value(item) for item in list(values or ()))


def _safe_int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _normalize_bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _normalize_optional_str(value: Any) -> str:
    return "" if value is None else str(value)


def _normalize_active_vars(values: Any) -> tuple[int, ...]:
    return _tuple_ints(values)


def _is_nonempty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        return bool(dict(value))
    if isinstance(value, (list, tuple, set)):
        return bool(list(value))
    if isinstance(value, str):
        return bool(str(value).strip())
    return True


def infer_primary_family_name(family_scores: Mapping[str, float] | None) -> str:
    keys = [str(key) for key in dict(family_scores or {}).keys() if str(key)]
    if len(keys) == 1:
        return str(keys[0])
    return ""


def extract_family_regime_metadata(*sources: Any) -> dict[str, Any]:
    regime: dict[str, Any] = {}
    for canonical_name, aliases in _FAMILY_REGIME_FIELD_ALIASES:
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            for alias in aliases:
                if alias not in source:
                    continue
                value = source.get(alias, None)
                if not _is_nonempty_value(value):
                    continue
                regime[canonical_name] = _clone_value(value)
                break
            if canonical_name in regime:
                break
    return regime


def canonicalize_family_hard_constraints(
    family_name: str | None,
    hard_constraints: Mapping[str, Any] | None = None,
    *,
    status: str | None = None,
    should_run: bool | None = None,
    advisory_only: bool | None = None,
    target_dim: Any = None,
    active_vars: Any = None,
    wrappers_left: int | None = None,
    recursion_level: int | None = None,
    direction: str | None = None,
    target_mode: str | None = None,
    target_mapping_kind: str | None = None,
    dimensionless_target_required: bool | None = None,
    target_dim_ok: bool | None = None,
    domain_masks: Mapping[str, Any] | None = None,
    regime_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw = dict(_clone_value(dict(hard_constraints or {})))
    family_token = str(
        raw.get("family_name", family_name or raw.get("family", ""))
        or family_name
        or raw.get("family", "")
        or ""
    )

    status_value = str(raw.get("status", status or "") or status or "")
    should_run_value = _normalize_bool_or_none(raw.get("should_run", should_run))
    advisory_only_value = _normalize_bool_or_none(raw.get("advisory_only", advisory_only))

    target_dim_value = _clone_value(raw.get("target_dim", target_dim))
    active_vars_value = _normalize_active_vars(raw.get("active_vars", active_vars))
    wrappers_left_value = _safe_int_or_none(raw.get("wrappers_left", wrappers_left))
    recursion_level_value = _safe_int_or_none(raw.get("recursion_level", recursion_level))
    direction_value = _normalize_optional_str(raw.get("direction", direction))
    target_mode_value = _normalize_optional_str(raw.get("target_mode", target_mode))
    target_mapping_kind_value = _normalize_optional_str(
        raw.get("target_mapping_kind", target_mapping_kind)
    )

    dim_required_value = _normalize_bool_or_none(
        raw.get("dimensionless_target_required", dimensionless_target_required)
    )
    target_dim_ok_value = _normalize_bool_or_none(raw.get("target_dim_ok", target_dim_ok))

    domain_masks_value = dict(
        _clone_value(dict(raw.get("domain_masks", {}) or {}))
    )
    domain_masks_value.update(
        dict(_clone_value(dict(domain_masks or {})))
    )
    domain_hazard_value = _normalize_bool_or_none(raw.get("domain_hazard", None))
    hazard_severe_value = _normalize_bool_or_none(
        raw.get(
            "hazard_severe",
            domain_masks_value.get("hazard_severe", None),
        )
    )
    singularity_margin_value = _clone_value(
        raw.get("singularity_margin_proxy", raw.get("domain_singularity_margin_proxy", None))
    )
    domain_status_value = str(
        raw.get(
            "domain_hazard_status",
            status_value if family_token == "domain_hazard" else "",
        )
        or ""
    )
    if domain_hazard_value is None and (
        family_token == "domain_hazard"
        or hazard_severe_value is not None
        or singularity_margin_value is not None
        or domain_status_value
        or domain_masks_value
    ):
        domain_hazard_value = True

    regime_value = extract_family_regime_metadata(
        raw.get("regime", None),
        raw.get("regime_metadata", None),
        regime_metadata,
    )

    context_payload = {
        "target_dim": _clone_value(target_dim_value),
        "active_vars": tuple(int(v) for v in active_vars_value),
        "wrappers_left": wrappers_left_value,
        "recursion_level": recursion_level_value,
        "direction": str(direction_value),
        "target_mode": str(target_mode_value),
        "target_mapping_kind": str(target_mapping_kind_value),
    }
    dimensionless_payload = {
        "required": dim_required_value,
        "target_dim_ok": target_dim_ok_value,
    }
    domain_payload = {
        "hazard_present": domain_hazard_value,
        "hazard_severe": hazard_severe_value,
        "status": str(domain_status_value),
        "singularity_margin_proxy": _clone_value(singularity_margin_value),
        "masks": _clone_value(domain_masks_value),
    }

    raw["schema_name"] = str(FAMILY_EVIDENCE_HARD_CONSTRAINTS_SCHEMA_NAME)
    raw["schema_version"] = int(FAMILY_EVIDENCE_HARD_CONSTRAINTS_SCHEMA_VERSION)
    raw["family_name"] = str(family_token)
    raw["status"] = str(status_value)
    raw["should_run"] = False if should_run_value is None else bool(should_run_value)
    raw["advisory_only"] = True if advisory_only_value is None else bool(advisory_only_value)
    raw["context"] = context_payload
    raw["dimensionless"] = dimensionless_payload
    raw["domain"] = domain_payload
    raw["regime"] = _clone_value(regime_value)

    if target_dim_value is not None and "target_dim" not in raw:
        raw["target_dim"] = _clone_value(target_dim_value)
    if active_vars_value and "active_vars" not in raw:
        raw["active_vars"] = tuple(int(v) for v in active_vars_value)
    if wrappers_left_value is not None and "wrappers_left" not in raw:
        raw["wrappers_left"] = int(wrappers_left_value)
    if recursion_level_value is not None and "recursion_level" not in raw:
        raw["recursion_level"] = int(recursion_level_value)
    if direction_value and "direction" not in raw:
        raw["direction"] = str(direction_value)
    if target_mode_value and "target_mode" not in raw:
        raw["target_mode"] = str(target_mode_value)
    if target_mapping_kind_value and "target_mapping_kind" not in raw:
        raw["target_mapping_kind"] = str(target_mapping_kind_value)
    if dim_required_value is not None and "dimensionless_target_required" not in raw:
        raw["dimensionless_target_required"] = bool(dim_required_value)
    if target_dim_ok_value is not None and "target_dim_ok" not in raw:
        raw["target_dim_ok"] = bool(target_dim_ok_value)
    if domain_masks_value and "domain_masks" not in raw:
        raw["domain_masks"] = _clone_value(domain_masks_value)
    if regime_value and "regime_metadata" not in raw:
        raw["regime_metadata"] = _clone_value(regime_value)

    return raw


@dataclass(frozen=True)
class WitnessBundle:
    x_fit: Any
    t_fit: Any
    x_probe: Any
    t_probe: Any
    grad_fit: Any = None
    grad_probe: Any = None
    d2_fit: Any = None
    d2_probe: Any = None
    masks: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubproblemSpec:
    problem_id: str
    problem_kind: str
    parent_expr: Any
    path: tuple[int, ...]
    direction: str
    target_mode: str
    target_mapping_kind: str
    target_dim: Any
    continuation_frames: tuple[Any, ...] = ()
    wrappers_left: int = 0
    recursion_level: int = 0
    active_vars: tuple[int, ...] = ()
    witness: WitnessBundle | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FamilyEvidence:
    family_scores: Mapping[str, float]
    hard_constraints: Mapping[str, Any]
    seed_nodes: tuple[Any, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SolverProposal:
    expr_ast: Any
    mapping: Mapping[str, Any]
    source: str
    family: str
    preview_loss: float
    global_probe_mse: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def normalize_family_evidence(evidence: FamilyEvidence | None) -> FamilyEvidence | None:
    if evidence is None:
        return None
    family_scores = {
        str(key): float(value)
        for key, value in dict(evidence.family_scores or {}).items()
    }
    return FamilyEvidence(
        family_scores=family_scores,
        hard_constraints=canonicalize_family_hard_constraints(
            infer_primary_family_name(family_scores),
            evidence.hard_constraints,
        ),
        seed_nodes=_tuple_any(evidence.seed_nodes),
        metadata=dict(_clone_value(dict(evidence.metadata or {}))),
    )


def serialize_witness_bundle(bundle: WitnessBundle | None) -> dict[str, Any] | None:
    if bundle is None:
        return None
    return {
        "x_fit": _clone_value(bundle.x_fit),
        "t_fit": _clone_value(bundle.t_fit),
        "x_probe": _clone_value(bundle.x_probe),
        "t_probe": _clone_value(bundle.t_probe),
        "grad_fit": _clone_value(bundle.grad_fit),
        "grad_probe": _clone_value(bundle.grad_probe),
        "d2_fit": _clone_value(bundle.d2_fit),
        "d2_probe": _clone_value(bundle.d2_probe),
        "masks": _clone_value(dict(bundle.masks or {})),
        "diagnostics": _clone_value(dict(bundle.diagnostics or {})),
    }


def deserialize_witness_bundle(payload: Mapping[str, Any] | None) -> WitnessBundle | None:
    if not isinstance(payload, Mapping):
        return None
    required = ("x_fit", "t_fit", "x_probe", "t_probe")
    if any(key not in payload for key in required):
        return None
    return WitnessBundle(
        x_fit=_clone_value(payload.get("x_fit", None)),
        t_fit=_clone_value(payload.get("t_fit", None)),
        x_probe=_clone_value(payload.get("x_probe", None)),
        t_probe=_clone_value(payload.get("t_probe", None)),
        grad_fit=_clone_value(payload.get("grad_fit", None)),
        grad_probe=_clone_value(payload.get("grad_probe", None)),
        d2_fit=_clone_value(payload.get("d2_fit", None)),
        d2_probe=_clone_value(payload.get("d2_probe", None)),
        masks=dict(_clone_value(payload.get("masks", {}) or {})),
        diagnostics=dict(_clone_value(payload.get("diagnostics", {}) or {})),
    )


def serialize_subproblem_spec(spec: SubproblemSpec | None) -> dict[str, Any] | None:
    if spec is None:
        return None
    return {
        "problem_id": str(spec.problem_id),
        "problem_kind": str(spec.problem_kind),
        "parent_expr": _clone_value(spec.parent_expr),
        "path": [int(v) for v in tuple(spec.path or ())],
        "direction": str(spec.direction),
        "target_mode": str(spec.target_mode),
        "target_mapping_kind": str(spec.target_mapping_kind),
        "target_dim": _clone_value(spec.target_dim),
        "continuation_frames": [_clone_value(item) for item in tuple(spec.continuation_frames or ())],
        "wrappers_left": int(spec.wrappers_left),
        "recursion_level": int(spec.recursion_level),
        "active_vars": [int(v) for v in tuple(spec.active_vars or ())],
        "witness": serialize_witness_bundle(spec.witness),
        "metadata": _clone_value(dict(spec.metadata or {})),
    }


def deserialize_subproblem_spec(payload: Mapping[str, Any] | None) -> SubproblemSpec | None:
    if not isinstance(payload, Mapping):
        return None
    raw = payload
    if isinstance(raw.get("subproblem_spec", None), Mapping):
        raw = raw.get("subproblem_spec", None)
    if not isinstance(raw, Mapping):
        return None
    if "problem_kind" not in raw:
        return None
    return SubproblemSpec(
        problem_id=str(raw.get("problem_id", "") or ""),
        problem_kind=str(raw.get("problem_kind", "") or ""),
        parent_expr=_clone_value(raw.get("parent_expr", None)),
        path=_tuple_ints(raw.get("path", ())),
        direction=str(raw.get("direction", "") or ""),
        target_mode=str(raw.get("target_mode", "") or ""),
        target_mapping_kind=str(raw.get("target_mapping_kind", "") or ""),
        target_dim=_clone_value(raw.get("target_dim", None)),
        continuation_frames=_tuple_any(raw.get("continuation_frames", ())),
        wrappers_left=int(raw.get("wrappers_left", 0) or 0),
        recursion_level=int(raw.get("recursion_level", 0) or 0),
        active_vars=_tuple_ints(raw.get("active_vars", ())),
        witness=deserialize_witness_bundle(raw.get("witness", None)),
        metadata=dict(_clone_value(raw.get("metadata", {}) or {})),
    )


def wrap_subproblem_spec_payload(
    spec: SubproblemSpec | None,
    *,
    extra_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(_clone_value(dict(extra_payload or {})))
    serialized = serialize_subproblem_spec(spec)
    if serialized is None:
        return payload
    payload["schema_name"] = str(SUBPROBLEM_SPEC_SCHEMA_NAME)
    payload["schema_version"] = int(SUBPROBLEM_SPEC_SCHEMA_VERSION)
    payload["subproblem_spec"] = serialized
    return payload


def serialize_family_evidence(evidence: FamilyEvidence | None) -> dict[str, Any] | None:
    normalized = normalize_family_evidence(evidence)
    if normalized is None:
        return None
    return {
        "family_scores": {
            str(key): float(value) for key, value in dict(normalized.family_scores or {}).items()
        },
        "hard_constraints": _clone_value(dict(normalized.hard_constraints or {})),
        "seed_nodes": [_clone_value(item) for item in tuple(normalized.seed_nodes or ())],
        "metadata": _clone_value(dict(normalized.metadata or {})),
    }


def deserialize_family_evidence(payload: Mapping[str, Any] | None) -> FamilyEvidence | None:
    if not isinstance(payload, Mapping):
        return None
    family_scores = {
        str(key): float(value)
        for key, value in dict(payload.get("family_scores", {}) or {}).items()
    }
    return FamilyEvidence(
        family_scores=family_scores,
        hard_constraints=canonicalize_family_hard_constraints(
            infer_primary_family_name(family_scores),
            dict(_clone_value(payload.get("hard_constraints", {}) or {})),
        ),
        seed_nodes=_tuple_any(payload.get("seed_nodes", ())),
        metadata=dict(_clone_value(payload.get("metadata", {}) or {})),
    )


def serialize_solver_proposal(proposal: SolverProposal | None) -> dict[str, Any] | None:
    if proposal is None:
        return None
    return {
        "expr_ast": _clone_value(proposal.expr_ast),
        "mapping": _clone_value(dict(proposal.mapping or {})),
        "source": str(proposal.source),
        "family": str(proposal.family),
        "preview_loss": float(proposal.preview_loss),
        "global_probe_mse": None if proposal.global_probe_mse is None else float(proposal.global_probe_mse),
        "metadata": _clone_value(dict(proposal.metadata or {})),
    }


def deserialize_solver_proposal(payload: Mapping[str, Any] | None) -> SolverProposal | None:
    if not isinstance(payload, Mapping):
        return None
    if "preview_loss" not in payload:
        return None
    return SolverProposal(
        expr_ast=_clone_value(payload.get("expr_ast", None)),
        mapping=dict(_clone_value(payload.get("mapping", {}) or {})),
        source=str(payload.get("source", "") or ""),
        family=str(payload.get("family", "") or ""),
        preview_loss=float(payload.get("preview_loss", 0.0) or 0.0),
        global_probe_mse=None
        if payload.get("global_probe_mse", None) is None
        else float(payload.get("global_probe_mse", 0.0) or 0.0),
        metadata=dict(_clone_value(payload.get("metadata", {}) or {})),
    )


__all__ = [
    "FamilyEvidence",
    "FAMILY_EVIDENCE_HARD_CONSTRAINTS_SCHEMA_NAME",
    "FAMILY_EVIDENCE_HARD_CONSTRAINTS_SCHEMA_VERSION",
    "SolverProposal",
    "SubproblemSpec",
    "SUBPROBLEM_SPEC_SCHEMA_NAME",
    "SUBPROBLEM_SPEC_SCHEMA_VERSION",
    "WitnessBundle",
    "canonicalize_family_hard_constraints",
    "deserialize_family_evidence",
    "deserialize_solver_proposal",
    "deserialize_subproblem_spec",
    "deserialize_witness_bundle",
    "extract_family_regime_metadata",
    "infer_primary_family_name",
    "normalize_family_evidence",
    "serialize_family_evidence",
    "serialize_solver_proposal",
    "serialize_subproblem_spec",
    "serialize_witness_bundle",
    "wrap_subproblem_spec_payload",
]
