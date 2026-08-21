# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from nestynet_sr.sr_search.factorized_search.research_profiles import (
    apply_research_profile_overrides,
    resolve_discovery_research_profile,
)
from nestynet_sr.sr_core import ast_to_human_readable
from nestynet_sr.sr_core.bridges import (
    AbsNode,
    AddNode,
    ArgNode,
    AtomNode,
    ConjNode,
    ConstNode,
    CosNode,
    ExpNode,
    ImagNode,
    LogNode,
    MulNode,
    PowNode,
    RealNode,
    SinNode,
    collect_all_atoms,
)
from nestynet_sr.sr_core.units import UnitsSpec

from .active_design import (
    ExperimentCandidate,
    resolve_surface_disagreement_mode,
    select_next_experiment,
)
from .committee import CommitteeMember, build_committee_state
from .constant_lift import (
    apply_constant_lift_proposals,
    discover_constant_lifts,
    parameter_samples_from_local_constants,
)
from .closed_loop import run_closed_loop_iteration
from .experiment_opt import optimize_continuous_experiment_candidates
from .physics_tests import score_physics_consistency
from .witness import capture_runtime_witness


@dataclass
class RuntimeDiscoveryCandidate:
    member: CommitteeMember
    model: Any = None
    y_inverse: Any = None


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _jsonable(value: Any) -> Any:
    if isinstance(value, AtomNode):
        return {
            "__bridge_node__": "AtomNode",
            "kind": str(getattr(value, "kind", "") or ""),
            "var_idxs": [int(v) for v in tuple(getattr(value, "var_idxs", ()) or ())],
            "kwargs": _jsonable(dict(getattr(value, "kwargs", {}) or {})),
            "tag": getattr(value, "tag", None),
            "scope": str(getattr(value, "scope", "experiment") or "experiment"),
            "inputs": None
            if getattr(value, "inputs", None) is None
            else [_jsonable(item) for item in tuple(getattr(value, "inputs", ()) or ())],
        }
    if isinstance(value, AddNode):
        return {"__bridge_node__": "AddNode", "left": _jsonable(value.left), "right": _jsonable(value.right)}
    if isinstance(value, MulNode):
        return {"__bridge_node__": "MulNode", "left": _jsonable(value.left), "right": _jsonable(value.right)}
    if isinstance(value, PowNode):
        return {"__bridge_node__": "PowNode", "base": _jsonable(value.base), "exponent": _jsonable(value.exponent)}
    if isinstance(value, LogNode):
        return {"__bridge_node__": "LogNode", "arg": _jsonable(value.arg)}
    if isinstance(value, ExpNode):
        return {"__bridge_node__": "ExpNode", "arg": _jsonable(value.arg)}
    if isinstance(value, SinNode):
        return {"__bridge_node__": "SinNode", "arg": _jsonable(value.arg)}
    if isinstance(value, CosNode):
        return {"__bridge_node__": "CosNode", "arg": _jsonable(value.arg)}
    if isinstance(value, ConjNode):
        return {"__bridge_node__": "ConjNode", "arg": _jsonable(value.arg)}
    if isinstance(value, RealNode):
        return {"__bridge_node__": "RealNode", "arg": _jsonable(value.arg)}
    if isinstance(value, ImagNode):
        return {"__bridge_node__": "ImagNode", "arg": _jsonable(value.arg)}
    if isinstance(value, AbsNode):
        return {"__bridge_node__": "AbsNode", "arg": _jsonable(value.arg)}
    if isinstance(value, ArgNode):
        return {"__bridge_node__": "ArgNode", "arg": _jsonable(value.arg)}
    if isinstance(value, ConstNode):
        const_value = value.value
        if isinstance(const_value, complex):
            return {
                "__bridge_node__": "ConstNode",
                "value": {"__complex__": [float(const_value.real), float(const_value.imag)]},
            }
        return {"__bridge_node__": "ConstNode", "value": _jsonable(const_value)}
    if torch.is_tensor(value):
        if value.ndim == 0:
            return float(value.item())
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return None if not math.isfinite(value) else float(value)
    return str(value)


def _discovery_research_activation_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(payload.get("config", {}) or {})
    selection = payload.get("experiment_selection", None)
    selected = selection.get("selected", None) if isinstance(selection, Mapping) else None
    optimization = dict(selection.get("optimization", {}) or {}) if isinstance(selection, Mapping) else {}
    constant_lift = dict(payload.get("constant_lift_summary", {}) or {})
    witness_candidate_count = 0
    derivative_prediction_members: set[str] = set()
    diagnostic_prediction_members: set[str] = set()
    for candidate in list(payload.get("experiment_candidates_full", []) or []):
        if not isinstance(candidate, Mapping):
            continue
        derivative_predictions = dict(candidate.get("derivative_predictions", {}) or {})
        diagnostic_predictions = dict(candidate.get("diagnostic_predictions", {}) or {})
        derivative_prediction_members.update(str(k) for k in derivative_predictions.keys())
        diagnostic_prediction_members.update(str(k) for k in diagnostic_predictions.keys())
        if derivative_predictions or diagnostic_predictions:
            witness_candidate_count += 1
    applied_count = int(
        constant_lift.get(
            "surviving_applied_member_count",
            constant_lift.get("applied_member_count", 0),
        )
        or 0
    )
    proposal_count = int(constant_lift.get("proposal_count", 0) or 0)
    return {
        "research_profile": str(config.get("research_profile", "legacy") or "legacy"),
        "selected_experiment_id": None if not isinstance(selected, Mapping) else str(selected.get("experiment_id", "") or ""),
        "witness_mode_selected": bool(
            isinstance(selection, Mapping)
            and str(selection.get("disagreement_mode", "witness") or "witness") == "witness"
        ),
        "witness_capture_active": bool(witness_candidate_count > 0),
        "derivative_prediction_member_count": int(len(derivative_prediction_members)),
        "diagnostic_prediction_member_count": int(len(diagnostic_prediction_members)),
        "witness_candidate_count": int(witness_candidate_count),
        "experiment_optimization_used": bool(int(optimization.get("optimized_candidate_count", 0) or 0) > 0),
        "constant_lift_proposal_count": int(proposal_count),
        "constant_lift_applied_count": int(applied_count),
        "theory_benchmark_enabled": bool(config.get("theory_benchmark_enable", False)),
        "new_stack_active": bool(
            str(config.get("research_profile", "legacy") or "legacy") != "legacy"
            or witness_candidate_count > 0
            or int(applied_count) > 0
            or bool(config.get("experiment_optimize_enable", False))
        ),
    }


def _display_expr(structure: Any, *, x_transform_map: Mapping[str, Any] | None = None) -> str:
    if structure is None:
        return ""
    if isinstance(structure, str):
        return str(structure)
    try:
        return str(ast_to_human_readable(structure, x_transform_map))
    except Exception:
        return str(structure)


def _simplicity_score(*, n_params: Any = None, ast_cost: Any = None, expr: Any = None) -> float:
    for value in (ast_cost, n_params):
        score = _safe_float(value, float("nan"))
        if math.isfinite(score) and score > 0.0:
            return float(1.0 / score)
    expr_len = max(1, len(str(expr or "")))
    return float(1.0 / expr_len)


def _units_spec_from_payload(
    units_payload: Mapping[str, Any] | None,
    *,
    y_transform_name: str,
) -> UnitsSpec | None:
    if not isinstance(units_payload, Mapping):
        return None
    try:
        return UnitsSpec(
            unit_system=units_payload["unit_system"],
            x_dims=tuple(units_payload["x_dims"]),
            y_dim=units_payload["y_dim"],
            y_transform_name=str(y_transform_name or "identity"),
            free_const_dims=dict(units_payload.get("free_const_dims", {}) or {}),
            free_const_scope=dict(units_payload.get("free_const_scope", {}) or {}),
            fixed_const_dims=dict(units_payload.get("fixed_const_dims", {}) or {}),
            fixed_const_values=dict(units_payload.get("fixed_const_values", {}) or {}),
            fixed_const_mode=str(units_payload.get("fixed_const_mode", "strict") or "strict"),
        )
    except Exception:
        return None


def _serialize_units_payload(
    units_payload: Mapping[str, Any] | None,
    *,
    y_transform_name: str,
) -> dict[str, Any] | None:
    if not isinstance(units_payload, Mapping):
        return None
    try:
        return {
            "unit_system": units_payload["unit_system"],
            "x_dims": list(units_payload["x_dims"]),
            "y_dim": units_payload["y_dim"],
            "y_transform_name": str(y_transform_name or "identity"),
            "free_const_dims": dict(units_payload.get("free_const_dims", {}) or {}),
            "free_const_scope": dict(units_payload.get("free_const_scope", {}) or {}),
            "fixed_const_dims": dict(units_payload.get("fixed_const_dims", {}) or {}),
            "fixed_const_values": dict(units_payload.get("fixed_const_values", {}) or {}),
            "fixed_const_mode": str(units_payload.get("fixed_const_mode", "strict") or "strict"),
        }
    except Exception:
        return None


def _flatten_param_values(prefix: str, values: Any) -> dict[str, float]:
    if values is None:
        return {}
    try:
        tensor = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    except Exception:
        scalar = _safe_float(values, float("nan"))
        if math.isfinite(scalar):
            return {str(prefix): float(scalar)}
        return {}
    if tensor.numel() == 0:
        return {}
    flat = tensor.detach().cpu().reshape(-1).tolist()
    if len(flat) == 1:
        scalar = _safe_float(flat[0], float("nan"))
        return {} if not math.isfinite(scalar) else {str(prefix): float(scalar)}
    out: dict[str, float] = {}
    for idx, item in enumerate(flat):
        scalar = _safe_float(item, float("nan"))
        if math.isfinite(scalar):
            out[f"{str(prefix)}__{int(idx)}"] = float(scalar)
    return out


def _restore_structure(value: Any) -> Any:
    if isinstance(value, dict):
        bridge_kind = str(value.get("__bridge_node__", "") or "")
        if bridge_kind == "AtomNode":
            inputs = value.get("inputs", None)
            restored_inputs = None if inputs is None else tuple(_restore_structure(item) for item in list(inputs or []))
            return AtomNode(
                kind=str(value.get("kind", "") or ""),
                var_idxs=tuple(int(v) for v in list(value.get("var_idxs", []) or [])),
                kwargs=dict(_restore_structure(value.get("kwargs", {})) or {}),
                tag=value.get("tag", None),
                inputs=restored_inputs,
                scope=str(value.get("scope", "experiment") or "experiment"),
            )
        if bridge_kind == "AddNode":
            return AddNode(_restore_structure(value.get("left", None)), _restore_structure(value.get("right", None)))
        if bridge_kind == "MulNode":
            return MulNode(_restore_structure(value.get("left", None)), _restore_structure(value.get("right", None)))
        if bridge_kind == "PowNode":
            return PowNode(_restore_structure(value.get("base", None)), float(value.get("exponent", 1.0)))
        if bridge_kind == "LogNode":
            return LogNode(_restore_structure(value.get("arg", None)))
        if bridge_kind == "ExpNode":
            return ExpNode(_restore_structure(value.get("arg", None)))
        if bridge_kind == "SinNode":
            return SinNode(_restore_structure(value.get("arg", None)))
        if bridge_kind == "CosNode":
            return CosNode(_restore_structure(value.get("arg", None)))
        if bridge_kind == "ConjNode":
            return ConjNode(_restore_structure(value.get("arg", None)))
        if bridge_kind == "RealNode":
            return RealNode(_restore_structure(value.get("arg", None)))
        if bridge_kind == "ImagNode":
            return ImagNode(_restore_structure(value.get("arg", None)))
        if bridge_kind == "AbsNode":
            return AbsNode(_restore_structure(value.get("arg", None)))
        if bridge_kind == "ArgNode":
            return ArgNode(_restore_structure(value.get("arg", None)))
        if bridge_kind == "ConstNode":
            const_value = value.get("value", None)
            if isinstance(const_value, dict) and "__complex__" in const_value:
                pair = list(const_value.get("__complex__", []) or [])
                real = float(pair[0]) if len(pair) >= 1 else 0.0
                imag = float(pair[1]) if len(pair) >= 2 else 0.0
                return ConstNode(complex(real, imag))
            return ConstNode(const_value)
    if isinstance(value, list):
        return tuple(_restore_structure(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _restore_structure(item) for key, item in value.items()}
    return value


def _extract_model_leaf_constants(symbolic_structure: Any, model: Any) -> dict[str, float]:
    if symbolic_structure is None or model is None:
        return {}
    leaves = getattr(model, "leaf", None)
    if leaves is None:
        return {}
    try:
        atoms = list(collect_all_atoms(symbolic_structure) or [])
    except Exception:
        return {}
    out: dict[str, float] = {}
    for idx, atom in enumerate(atoms):
        if idx >= len(leaves):
            break
        try:
            params = [param.detach().reshape(-1).cpu() for param in leaves[idx].parameters()]
        except Exception:
            continue
        if not params:
            continue
        vec = torch.cat(params)
        tag = str(getattr(atom, "tag", "") or f"leaf{int(idx)}")
        out.update(_flatten_param_values(tag, vec))
    return out


def _dataset_ids_from_context(
    *,
    filepaths: Sequence[str] | None,
    stageA_data: Mapping[str, Any] | None,
    stageB_data: Mapping[str, Any] | None,
    stageB_state: Any = None,
) -> list[str]:
    dataset_ids = list(
        (stageB_data or {}).get("dataset_ids", None)
        or (stageA_data or {}).get("dataset_ids", None)
        or getattr(stageB_state, "dataset_ids", None)
        or []
    )
    if dataset_ids:
        return [str(item) for item in dataset_ids]
    return [
        pathlib.Path(str(path)).stem
        for path in list(filepaths or [])
        if str(path).strip()
    ]


def _dataset_metadata_from_context(
    *,
    stageA_data: Mapping[str, Any] | None,
    stageB_data: Mapping[str, Any] | None,
) -> Mapping[str, Any] | Sequence[Mapping[str, Any]] | None:
    for payload in (stageB_data, stageA_data):
        if not isinstance(payload, Mapping):
            continue
        metadata = payload.get("dataset_metadata", None)
        if isinstance(metadata, Mapping):
            return dict(metadata)
        if isinstance(metadata, Sequence) and not isinstance(metadata, (str, bytes)):
            return [dict(row) for row in list(metadata) if isinstance(row, Mapping)]
    return None


def _class_sr_shared_constants(class_sr_result: Any) -> dict[str, float]:
    params = dict(getattr(class_sr_result, "class_params", {}) or {})
    out: dict[str, float] = {}
    for tag, values in params.items():
        out.update(_flatten_param_values(str(tag), values))
    return out


def _class_sr_local_constants(class_sr_result: Any, dataset_ids: Sequence[str]) -> dict[str, dict[str, float]]:
    params_by_dataset = list(getattr(class_sr_result, "experiment_params", []) or [])
    ids = [str(item) for item in list(dataset_ids or [])]
    out: dict[str, dict[str, float]] = {}
    for idx, payload in enumerate(params_by_dataset):
        ds_name = ids[idx] if idx < len(ids) else f"dataset_{int(idx)}"
        ds_out: dict[str, float] = {}
        for tag, values in dict(payload or {}).items():
            ds_out.update(_flatten_param_values(str(tag), values))
        if ds_out:
            out[str(ds_name)] = ds_out
    return out


def _local_constants_from_stageb_models(
    symbolic_structure: Any,
    *,
    stageB_state: Any,
    dataset_ids: Sequence[str],
) -> dict[str, dict[str, float]]:
    models = list(getattr(stageB_state, "models", None) or [])
    if not models:
        return {}
    ids = [str(item) for item in list(dataset_ids or [])]
    out: dict[str, dict[str, float]] = {}
    for idx, model in enumerate(models):
        ds_name = ids[idx] if idx < len(ids) else f"dataset_{int(idx)}"
        ds_constants = _extract_model_leaf_constants(symbolic_structure, model)
        if ds_constants:
            out[str(ds_name)] = ds_constants
    return out


def _select_constant_payload(
    *,
    source: str,
    symbolic_structure: Any,
    model: Any = None,
    stageB_state: Any = None,
    class_sr_result: Any = None,
    dataset_ids: Sequence[str] = (),
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, float]], str]:
    source_name = str(source or "")
    fitted_constants: dict[str, float] = {}
    shared_constants: dict[str, float] = {}
    local_constants_by_experiment: dict[str, dict[str, float]] = {}
    constant_source = ""
    if source_name.startswith("stageB"):
        if class_sr_result is not None:
            shared_constants = _class_sr_shared_constants(class_sr_result)
            local_constants_by_experiment = _class_sr_local_constants(class_sr_result, dataset_ids)
            constant_source = "class_sr"
        elif stageB_state is not None:
            local_constants_by_experiment = _local_constants_from_stageb_models(
                symbolic_structure,
                stageB_state=stageB_state,
                dataset_ids=dataset_ids,
            )
            constant_source = "stageb_models" if local_constants_by_experiment else ""
        if local_constants_by_experiment:
            first_dataset = sorted(local_constants_by_experiment.keys())[0]
            fitted_constants.update(dict(local_constants_by_experiment.get(first_dataset, {}) or {}))
        if shared_constants:
            fitted_constants = {
                **dict(shared_constants),
                **dict(fitted_constants),
            }
        if not fitted_constants and model is not None:
            fitted_constants = _extract_model_leaf_constants(symbolic_structure, model)
            if fitted_constants and not constant_source:
                constant_source = "stageb_model"
    else:
        fitted_constants = _extract_model_leaf_constants(symbolic_structure, model)
        if fitted_constants:
            constant_source = "stage_model"
    return (
        dict(fitted_constants),
        dict(shared_constants),
        {str(k): dict(v) for k, v in dict(local_constants_by_experiment).items()},
        str(constant_source),
    )


def serialize_committee_member(member: CommitteeMember) -> dict[str, Any]:
    return {
        "member_id": str(member.member_id),
        "symbolic_structure": _jsonable(member.symbolic_structure),
        "fitted_constants": _jsonable(dict(member.fitted_constants or {})),
        "shared_constants": _jsonable(dict(member.shared_constants or {})),
        "local_constants_by_experiment": _jsonable(dict(member.local_constants_by_experiment or {})),
        "train_error": _jsonable(member.train_error),
        "validation_error": _jsonable(member.validation_error),
        "regime_holdout_error": _jsonable(member.regime_holdout_error),
        "simplicity_score": _jsonable(member.simplicity_score),
        "physics_consistency_score": _jsonable(member.physics_consistency_score),
        "committee_weight": _jsonable(member.committee_weight),
        "canonical_key": str(member.canonical_key),
        "display_expr": str(member.display_expr),
        "metadata": _jsonable(dict(member.metadata or {})),
    }


def serialize_experiment_candidate(candidate: ExperimentCandidate) -> dict[str, Any]:
    return {
        "experiment_id": str(candidate.experiment_id),
        "conditions": _jsonable(candidate.conditions),
        "observable_predictions": _jsonable(dict(candidate.observable_predictions or {})),
        "derivative_predictions": _jsonable(dict(candidate.derivative_predictions or {})),
        "diagnostic_predictions": _jsonable(dict(candidate.diagnostic_predictions or {})),
        "cost": float(candidate.cost),
        "noise_risk": float(candidate.noise_risk),
        "feasibility_penalty": float(candidate.feasibility_penalty),
        "metadata": _jsonable(dict(candidate.metadata or {})),
    }


def deserialize_committee_members(rows: Sequence[Mapping[str, Any]]) -> list[CommitteeMember]:
    out: list[CommitteeMember] = []
    for idx, row in enumerate(list(rows or [])):
        if not isinstance(row, Mapping):
            continue
        out.append(
            CommitteeMember(
                member_id=str(row.get("member_id", "") or f"member_{int(idx)}"),
                symbolic_structure=_restore_structure(
                    row.get("symbolic_structure", row.get("expr", row.get("law", None)))
                ),
                fitted_constants=dict(row.get("fitted_constants", {}) or {}),
                shared_constants=dict(row.get("shared_constants", {}) or {}),
                local_constants_by_experiment=dict(row.get("local_constants_by_experiment", {}) or {}),
                train_error=_safe_float(row.get("train_error", float("nan"))),
                validation_error=_safe_float(row.get("validation_error", float("nan"))),
                regime_holdout_error=row.get("regime_holdout_error", None),
                simplicity_score=_safe_float(row.get("simplicity_score", 1.0), 1.0),
                physics_consistency_score=_safe_float(row.get("physics_consistency_score", 1.0), 1.0),
                committee_weight=_safe_float(row.get("committee_weight", 0.0), 0.0),
                canonical_key=str(row.get("canonical_key", "") or ""),
                display_expr=str(row.get("display_expr", "") or ""),
                metadata=dict(row.get("metadata", {}) or {}),
            )
        )
    return out


def deserialize_experiment_candidates(rows: Sequence[Mapping[str, Any]]) -> list[ExperimentCandidate]:
    out: list[ExperimentCandidate] = []
    for idx, row in enumerate(list(rows or [])):
        if not isinstance(row, Mapping):
            continue
        out.append(
            ExperimentCandidate(
                experiment_id=str(row.get("experiment_id", "") or f"experiment_{int(idx)}"),
                conditions=row.get("conditions", None),
                observable_predictions=dict(row.get("observable_predictions", {}) or {}),
                derivative_predictions=dict(row.get("derivative_predictions", {}) or {}),
                diagnostic_predictions=dict(row.get("diagnostic_predictions", {}) or {}),
                cost=_safe_float(row.get("cost", 0.0), 0.0),
                noise_risk=_safe_float(row.get("noise_risk", 0.0), 0.0),
                feasibility_penalty=_safe_float(row.get("feasibility_penalty", 0.0), 0.0),
                metadata=dict(row.get("metadata", {}) or {}),
            )
        )
    return out


def _build_stage_candidate(
    *,
    member_id: str,
    source: str,
    symbolic_structure: Any,
    validation_error: Any,
    train_error: Any = None,
    n_params: Any = None,
    ast_cost: Any = None,
    x_transform_map: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    model: Any = None,
    y_inverse: Any = None,
    fitted_constants: Mapping[str, float] | None = None,
    shared_constants: Mapping[str, float] | None = None,
    local_constants_by_experiment: Mapping[str, Mapping[str, float]] | None = None,
) -> RuntimeDiscoveryCandidate | None:
    if symbolic_structure is None:
        return None
    display_expr = _display_expr(symbolic_structure, x_transform_map=x_transform_map)
    val = _safe_float(validation_error, float("nan"))
    train = _safe_float(train_error, val)
    member = CommitteeMember(
        member_id=str(member_id),
        symbolic_structure=symbolic_structure,
        fitted_constants=dict(fitted_constants or {}),
        shared_constants=dict(shared_constants or {}),
        local_constants_by_experiment=dict(local_constants_by_experiment or {}),
        train_error=float(train),
        validation_error=float(val),
        simplicity_score=_simplicity_score(n_params=n_params, ast_cost=ast_cost, expr=display_expr),
        physics_consistency_score=1.0,
        display_expr=str(display_expr),
        metadata={
            "source": str(source),
            "display_expr": str(display_expr),
            "n_params": None if n_params is None else int(n_params),
            "ast_cost": None if ast_cost is None else float(ast_cost),
            **dict(metadata or {}),
        },
    )
    return RuntimeDiscoveryCandidate(member=member, model=model, y_inverse=y_inverse)


def _collect_runtime_candidates(
    *,
    filepaths: Sequence[str] | None,
    stageA_data: Mapping[str, Any] | None,
    stageB_data: Mapping[str, Any] | None,
    final_model: Any = None,
    final_y_op_inv: Any = None,
    stageB_state: Any = None,
    class_sr_result: Any = None,
    committee_topk: int = 8,
) -> list[RuntimeDiscoveryCandidate]:
    candidates: list[RuntimeDiscoveryCandidate] = []
    dataset_ids = _dataset_ids_from_context(
        filepaths=filepaths,
        stageA_data=stageA_data,
        stageB_data=stageB_data,
        stageB_state=stageB_state,
    )
    dataset_metadata = _dataset_metadata_from_context(
        stageA_data=stageA_data,
        stageB_data=stageB_data,
    )
    if isinstance(stageA_data, Mapping):
        stagea_fit, stagea_shared, stagea_local, stagea_constant_source = _select_constant_payload(
            source="stageA_final",
            symbolic_structure=stageA_data.get("ast", None),
            model=final_model,
            stageB_state=stageB_state,
            class_sr_result=None,
            dataset_ids=dataset_ids,
        )
        stagea_candidate = _build_stage_candidate(
            member_id="stageA_final",
            source="stageA_final",
            symbolic_structure=stageA_data.get("ast", None),
            validation_error=stageA_data.get("val_loss", stageA_data.get("nn_val_loss", float("nan"))),
            train_error=stageA_data.get("val_loss", stageA_data.get("nn_val_loss", float("nan"))),
            n_params=stageA_data.get("nn_n_params", None),
            x_transform_map=stageA_data.get("x_transform_map", None),
            metadata={
                "y_transform": str(stageA_data.get("y_op_name", "identity") or "identity"),
                "dataset_ids": list(dataset_ids),
                "dataset_metadata": dataset_metadata,
                "constant_source": str(stagea_constant_source),
            },
            model=final_model,
            y_inverse=final_y_op_inv,
            fitted_constants=stagea_fit,
            shared_constants=stagea_shared,
            local_constants_by_experiment=stagea_local,
        )
        if stagea_candidate is not None:
            candidates.append(stagea_candidate)
    if isinstance(stageB_data, Mapping):
        stageb_fit, stageb_shared, stageb_local, stageb_constant_source = _select_constant_payload(
            source="stageB_final",
            symbolic_structure=stageB_data.get("ast", None),
            model=getattr(stageB_state, "model", None),
            stageB_state=stageB_state,
            class_sr_result=class_sr_result,
            dataset_ids=dataset_ids,
        )
        stageb_x_transform = getattr(stageB_state, "x_transform_map", None)
        if stageb_x_transform is None and isinstance(stageA_data, Mapping):
            stageb_x_transform = stageA_data.get("x_transform_map", None)
        stageb_candidate = _build_stage_candidate(
            member_id="stageB_final",
            source="stageB_final",
            symbolic_structure=stageB_data.get("ast", None),
            validation_error=stageB_data.get("val_loss", float("nan")),
            train_error=stageB_data.get("val_loss", float("nan")),
            n_params=stageB_data.get("params", None),
            x_transform_map=stageb_x_transform,
            metadata={
                "y_expr_raw_str": stageB_data.get("y_expr_raw_str", None),
                "phi_expr_raw_str": stageB_data.get("phi_expr_raw_str", None),
                "dataset_ids": list(dataset_ids),
                "dataset_metadata": dataset_metadata,
                "constant_source": str(stageb_constant_source),
            },
            model=getattr(stageB_state, "model", None),
            y_inverse=final_y_op_inv,
            fitted_constants=stageb_fit,
            shared_constants=stageb_shared,
            local_constants_by_experiment=stageb_local,
        )
        if stageb_candidate is not None:
            candidates.append(stageb_candidate)
        final_exprs = [
            ("stageB_yspace", stageB_data.get("y_expr_raw_str", None) or stageB_data.get("y_expr_str", None)),
            ("stageB_phispace", stageB_data.get("phi_expr_raw_str", None) or stageB_data.get("phi_expr_str", None)),
        ]
        seen_exprs = {
            str(item.member.display_expr)
            for item in candidates
            if item is not None
        }
        for source, expr in final_exprs:
            expr_text = str(expr or "").strip()
            if not expr_text or expr_text in seen_exprs:
                continue
            string_candidate = _build_stage_candidate(
                member_id=str(source),
                source=str(source),
                symbolic_structure=expr_text,
                validation_error=stageB_data.get("val_loss", float("nan")),
                train_error=stageB_data.get("val_loss", float("nan")),
                n_params=stageB_data.get("params", None),
                metadata={
                    "stageB_string_variant": True,
                    "dataset_ids": list(dataset_ids),
                    "dataset_metadata": dataset_metadata,
                    "constant_source": str(stageb_constant_source),
                },
                fitted_constants=stageb_fit,
                shared_constants=stageb_shared,
                local_constants_by_experiment=stageb_local,
            )
            if string_candidate is not None:
                candidates.append(string_candidate)
                seen_exprs.add(expr_text)
        simp_entries = list(stageB_data.get("simplification_path", []) or [])
        for offset, entry in enumerate(reversed(simp_entries[-max(0, int(committee_topk)) :])):
            if not isinstance(entry, Mapping):
                continue
            expr = str(entry.get("expression", "") or "").strip()
            if not expr or expr in seen_exprs:
                continue
            step = int(entry.get("step", len(simp_entries) - offset) or (len(simp_entries) - offset))
            step_candidate = _build_stage_candidate(
                member_id=f"stageB_step_{int(step)}",
                source="stageB_simplification_path",
                symbolic_structure=expr,
                validation_error=entry.get("mse_eff", entry.get("mse_raw", entry.get("val_loss", float("nan")))),
                train_error=entry.get("mse_raw", entry.get("val_loss", float("nan"))),
                n_params=entry.get("n_params", None),
                ast_cost=entry.get("ast_cost", None),
                metadata={
                    "step": int(step),
                    "action": str(entry.get("action", "") or ""),
                    "detail": str(entry.get("detail", "") or ""),
                    "dataset_ids": list(dataset_ids),
                    "dataset_metadata": dataset_metadata,
                    "constant_source": str(stageb_constant_source),
                },
                fitted_constants=stageb_fit,
                shared_constants=stageb_shared,
                local_constants_by_experiment=stageb_local,
            )
            if step_candidate is not None:
                candidates.append(step_candidate)
                seen_exprs.add(expr)
    return candidates


def _predict_runtime_candidate(candidate: RuntimeDiscoveryCandidate, x: torch.Tensor) -> torch.Tensor | None:
    if candidate.model is None:
        return None
    try:
        with torch.no_grad():
            y_hat = candidate.model(x)
            if not torch.is_tensor(y_hat):
                y_hat = torch.as_tensor(y_hat, dtype=x.dtype, device=x.device)
            if y_hat.ndim == 1:
                y_hat = y_hat.reshape(-1, 1)
            elif y_hat.ndim > 2:
                y_hat = y_hat.reshape(y_hat.shape[0], -1)
            if callable(candidate.y_inverse):
                y_hat = candidate.y_inverse(y_hat)
            if y_hat.ndim == 2 and y_hat.shape[1] == 1:
                y_hat = y_hat[:, 0]
            else:
                y_hat = y_hat.reshape(y_hat.shape[0], -1).mean(dim=1)
            if not torch.isfinite(y_hat).all():
                return None
            return y_hat
    except Exception:
        return None


def _runtime_forward_value(candidate: RuntimeDiscoveryCandidate, x: torch.Tensor) -> torch.Tensor | None:
    if candidate.model is not None:
        try:
            y_hat = candidate.model(x)
            if not torch.is_tensor(y_hat):
                y_hat = torch.as_tensor(y_hat, dtype=x.dtype, device=x.device)
            else:
                y_hat = y_hat.to(dtype=x.dtype, device=x.device)
            if y_hat.ndim == 1:
                y_hat = y_hat.reshape(-1, 1)
            elif y_hat.ndim > 2:
                y_hat = y_hat.reshape(y_hat.shape[0], -1)
            if callable(candidate.y_inverse):
                y_hat = candidate.y_inverse(y_hat)
            if y_hat.ndim == 1:
                y_hat = y_hat.reshape(-1, 1)
            if y_hat.ndim != 2 or int(y_hat.shape[0]) != int(x.shape[0]):
                return None
            y_hat = y_hat[:, :1]
            if not torch.isfinite(y_hat).all():
                return None
            return y_hat
        except Exception:
            return None
    pred = _predict_runtime_candidate(candidate, x)
    if pred is None:
        return None
    if not torch.is_tensor(pred):
        try:
            pred = torch.as_tensor(pred, dtype=x.dtype, device=x.device)
        except Exception:
            return None
    else:
        pred = pred.to(dtype=x.dtype, device=x.device)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    elif pred.ndim > 2:
        pred = pred.reshape(pred.shape[0], -1)
    if pred.ndim != 2 or int(pred.shape[0]) != int(x.shape[0]):
        return None
    return pred[:, :1]


def _row_points_tensor(points: Sequence[Sequence[Any]], *, nvars: int, dtype: torch.dtype) -> torch.Tensor:
    rows = [[float(v) for v in list(row)] for row in list(points or [])]
    if not rows:
        raise ValueError("points experiment requires at least one row")
    if any(len(row) != int(nvars) for row in rows):
        raise ValueError(f"points experiment width must be {int(nvars)}")
    return torch.tensor(rows, dtype=dtype)


def _sample_box_tensor(
    entry: Mapping[str, Any],
    *,
    nvars: int,
    dtype: torch.dtype,
    default_bounds: Sequence[Sequence[float]] | None = None,
) -> torch.Tensor:
    n_points = max(1, int(entry.get("n_points", 32) or 32))
    seed = int(entry.get("seed", 0) or 0)
    g = torch.Generator(device="cpu").manual_seed(seed)
    bounds = entry.get("bounds", None)
    if isinstance(bounds, Mapping):
        lo_vals: list[float] = []
        hi_vals: list[float] = []
        for idx in range(int(nvars)):
            raw = bounds.get(str(idx), bounds.get(f"x{int(idx)}", None))
            if raw is None:
                if default_bounds is not None and idx < len(default_bounds):
                    raw = default_bounds[idx]
                else:
                    raw = (0.0, 1.0)
            lo_vals.append(float(raw[0]))
            hi_vals.append(float(raw[1]))
    else:
        lo = list(entry.get("lo", []) or [])
        hi = list(entry.get("hi", []) or [])
        if len(lo) == int(nvars) and len(hi) == int(nvars):
            lo_vals = [float(v) for v in lo]
            hi_vals = [float(v) for v in hi]
        elif default_bounds is not None and len(default_bounds) == int(nvars):
            lo_vals = [float(item[0]) for item in default_bounds]
            hi_vals = [float(item[1]) for item in default_bounds]
        else:
            lo_vals = [0.0] * int(nvars)
            hi_vals = [1.0] * int(nvars)
    lo_t = torch.tensor(lo_vals, dtype=dtype).reshape(1, int(nvars))
    hi_t = torch.tensor(hi_vals, dtype=dtype).reshape(1, int(nvars))
    u = torch.rand((n_points, int(nvars)), generator=g, dtype=dtype)
    return lo_t + (hi_t - lo_t) * u


def _entry_bounds_list(
    entry: Mapping[str, Any],
    *,
    nvars: int,
    default_bounds: Sequence[Sequence[float]] | None = None,
    points_tensor: torch.Tensor | None = None,
) -> list[list[float]]:
    bounds = entry.get("bounds", None)
    if isinstance(bounds, Mapping):
        out: list[list[float]] = []
        for idx in range(int(nvars)):
            raw = bounds.get(str(idx), bounds.get(f"x{int(idx)}", None))
            if raw is None:
                if default_bounds is not None and idx < len(default_bounds):
                    raw = default_bounds[idx]
                elif points_tensor is not None:
                    lo = float(points_tensor[:, idx].min().item())
                    hi = float(points_tensor[:, idx].max().item())
                    raw = (lo, hi if hi > lo else lo + 1.0)
                else:
                    raw = (0.0, 1.0)
            out.append([float(raw[0]), float(raw[1])])
        return out
    lo = list(entry.get("lo", []) or [])
    hi = list(entry.get("hi", []) or [])
    if len(lo) == int(nvars) and len(hi) == int(nvars):
        return [[float(lo[idx]), float(hi[idx])] for idx in range(int(nvars))]
    if default_bounds is not None and len(default_bounds) == int(nvars):
        return [[float(item[0]), float(item[1])] for item in default_bounds]
    if points_tensor is not None:
        out = []
        for idx in range(int(nvars)):
            lo_v = float(points_tensor[:, idx].min().item())
            hi_v = float(points_tensor[:, idx].max().item())
            if not hi_v > lo_v:
                hi_v = lo_v + 1.0
            out.append([lo_v, hi_v])
        return out
    return [[0.0, 1.0] for _ in range(int(nvars))]


def _load_manifest(path: str | pathlib.Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else None


def build_sr_experiment_candidates(
    *,
    committee: Sequence[CommitteeMember],
    runtime_candidates: Mapping[str, RuntimeDiscoveryCandidate],
    experiment_manifest: Mapping[str, Any] | None,
    nvars: int,
    dtype: torch.dtype,
    default_bounds: Sequence[Sequence[float]] | None = None,
    witness_capture_enable: bool = False,
    witness_hessian_diag_enable: bool = False,
    diagnostic_set: str = "basic",
) -> list[ExperimentCandidate]:
    if not isinstance(experiment_manifest, Mapping):
        return []
    witness_enabled = bool(witness_capture_enable)
    candidates: list[ExperimentCandidate] = []
    entries = list(experiment_manifest.get("experiments", experiment_manifest.get("candidates", [])) or [])
    for idx, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        kind = str(entry.get("type", "box") or "box").strip().lower()
        if kind == "points":
            x = _row_points_tensor(entry.get("points", []), nvars=int(nvars), dtype=dtype)
        elif kind == "box":
            x = _sample_box_tensor(
                entry,
                nvars=int(nvars),
                dtype=dtype,
                default_bounds=default_bounds,
            )
        else:
            raise ValueError(f"unsupported SR experiment type {kind!r}")
        observable_predictions: dict[str, Any] = {}
        derivative_predictions: dict[str, Any] = {}
        diagnostic_predictions: dict[str, Any] = {}
        for member in committee:
            runtime = runtime_candidates.get(str(member.member_id), None)
            if runtime is None:
                continue
            pred = _predict_runtime_candidate(runtime, x)
            if witness_enabled:
                witness = capture_runtime_witness(
                    runtime,
                    x,
                    predict_value_fn=_predict_runtime_candidate,
                    capture_gradients=True,
                    capture_hessian_diag=bool(witness_hessian_diag_enable),
                    diagnostic_set=str(diagnostic_set or "basic"),
                )
                pred = witness.get("observable", pred)
            else:
                witness = {}
            if pred is None:
                continue
            observable_predictions[str(member.member_id)] = _jsonable(pred)
            deriv = witness.get("derivative", None) if witness_enabled else None
            if deriv is not None:
                derivative_predictions[str(member.member_id)] = _jsonable(deriv)
            diag = dict(witness.get("diagnostic", {}) or {}) if witness_enabled else {}
            if diag:
                diagnostic_predictions[str(member.member_id)] = _jsonable(diag)
        candidates.append(
            ExperimentCandidate(
                experiment_id=str(entry.get("experiment_id", "") or entry.get("id", "") or f"experiment_{int(idx)}"),
                conditions={
                    "type": str(kind),
                    "n_points": int(x.shape[0]),
                    "shape": [int(v) for v in x.shape],
                },
                observable_predictions=observable_predictions,
                derivative_predictions=derivative_predictions,
                diagnostic_predictions=diagnostic_predictions,
                cost=float(entry.get("cost", 0.0) or 0.0),
                noise_risk=float(entry.get("noise_risk", 0.0) or 0.0),
                feasibility_penalty=float(entry.get("feasibility_penalty", 0.0) or 0.0),
                metadata={
                    "points_preview": _jsonable(x[: min(4, int(x.shape[0]))]),
                    "continuous_optimizer": {
                        "enabled": True,
                        "source_type": str(kind),
                        "points": _jsonable(x),
                        "bounds": _entry_bounds_list(
                            entry,
                            nvars=int(nvars),
                            default_bounds=default_bounds,
                            points_tensor=x,
                        ),
                    },
                    "witness_capture": {
                        "enabled": bool(witness_enabled),
                        "hessian_diag_enabled": bool(witness_enabled and witness_hessian_diag_enable),
                        "diagnostic_set": str(diagnostic_set or "basic"),
                    },
                },
            )
        )
    return candidates


def run_sr_discovery_integration(
    *,
    filepath: str,
    filepaths: Sequence[str] | None,
    report_path: str,
    stageA_data: Mapping[str, Any] | None,
    stageB_data: Mapping[str, Any] | None,
    final_model: Any = None,
    final_y_op_inv: Any = None,
    final_y_op_name: str = "identity",
    stageB_state: Any = None,
    class_sr_result: Any = None,
    units_payload: Mapping[str, Any] | None = None,
    committee_topk: int = 8,
    max_members: int | None = None,
    experiment_manifest_path: str | pathlib.Path | None = None,
    beta: float = 0.0,
    gamma: float = 0.0,
    disagreement_mode: str | None = None,
    lambda_cost: float = 1.0,
    lambda_noise: float = 1.0,
    lambda_feasibility: float = 1.0,
    nvars: int = 0,
    dtype: torch.dtype = torch.float64,
    discovery_constant_lift_enable: bool = False,
    discovery_constant_lift_min_regimes: int = 3,
    discovery_constant_lift_trigger_mean_cv: float = 0.5,
    discovery_constant_lift_apply_enable: bool = False,
    discovery_constant_lift_apply_topk: int = 1,
    discovery_constant_lift_min_rel_gain: float = 1.01,
    witness_capture_enable: bool = False,
    witness_hessian_diag_enable: bool = False,
    diagnostic_set: str = "basic",
    experiment_optimize_enable: bool = False,
    experiment_opt_steps: int = 32,
    experiment_opt_lr: float = 0.05,
    experiment_project_mode: str = "nearest_box",
    theory_benchmark_enable: bool = False,
    research_profile: str | None = None,
) -> dict[str, Any]:
    profile_requested = research_profile is not None and str(research_profile).strip() != ""
    if profile_requested:
        resolved_profile, profile_overrides = resolve_discovery_research_profile(research_profile)
    else:
        resolved_profile, profile_overrides = "default", {}
    profile_values = apply_research_profile_overrides(
        {
            "beta": float(beta),
            "gamma": float(gamma),
            "disagreement_mode": disagreement_mode,
            "discovery_constant_lift_enable": bool(discovery_constant_lift_enable),
            "discovery_constant_lift_apply_enable": bool(discovery_constant_lift_apply_enable),
            "discovery_constant_lift_apply_topk": int(max(0, int(discovery_constant_lift_apply_topk))),
            "witness_capture_enable": bool(witness_capture_enable),
            "witness_hessian_diag_enable": bool(witness_hessian_diag_enable),
            "diagnostic_set": str(diagnostic_set or "basic"),
            "experiment_optimize_enable": bool(experiment_optimize_enable),
            "theory_benchmark_enable": bool(theory_benchmark_enable),
        },
        overrides=profile_overrides,
    )
    beta = float(profile_values["beta"])
    gamma = float(profile_values["gamma"])
    disagreement_mode = resolve_surface_disagreement_mode(
        profile_values.get("disagreement_mode", None),
        default_mode="witness",
    )
    discovery_constant_lift_enable = bool(profile_values["discovery_constant_lift_enable"])
    discovery_constant_lift_apply_enable = bool(profile_values["discovery_constant_lift_apply_enable"])
    discovery_constant_lift_apply_topk = int(max(0, int(profile_values["discovery_constant_lift_apply_topk"])))
    witness_capture_enable = bool(profile_values["witness_capture_enable"])
    witness_hessian_diag_enable = bool(profile_values["witness_hessian_diag_enable"])
    diagnostic_set = str(profile_values["diagnostic_set"] or "basic")
    experiment_optimize_enable = bool(profile_values["experiment_optimize_enable"])
    theory_benchmark_enable = bool(profile_values["theory_benchmark_enable"])

    dataset_ids = _dataset_ids_from_context(
        filepaths=filepaths,
        stageA_data=stageA_data,
        stageB_data=stageB_data,
        stageB_state=stageB_state,
    )
    runtime_candidates = _collect_runtime_candidates(
        filepaths=filepaths,
        stageA_data=stageA_data,
        stageB_data=stageB_data,
        final_model=final_model,
        final_y_op_inv=final_y_op_inv,
        stageB_state=stageB_state,
        class_sr_result=class_sr_result,
        committee_topk=max(1, int(committee_topk)),
    )
    dataset_metadata = _dataset_metadata_from_context(
        stageA_data=stageA_data,
        stageB_data=stageB_data,
    )
    units_spec = _units_spec_from_payload(
        units_payload,
        y_transform_name=str(final_y_op_name or "identity"),
    )
    physics_reports: dict[str, Any] = {}
    enriched: list[CommitteeMember] = []
    for runtime in runtime_candidates:
        member = runtime.member
        report = score_physics_consistency(
            {
                "symbolic_structure": member.symbolic_structure,
                "train_error": member.train_error,
                "validation_error": member.validation_error,
                "metadata": dict(member.metadata or {}),
            },
            units_spec=units_spec,
            parameter_samples=parameter_samples_from_local_constants(
                member.local_constants_by_experiment,
                regime_ids=dataset_ids,
            ),
        )
        physics_reports[str(member.member_id)] = report
        enriched.append(
            CommitteeMember(
                member_id=member.member_id,
                symbolic_structure=member.symbolic_structure,
                fitted_constants=member.fitted_constants,
                shared_constants=member.shared_constants,
                local_constants_by_experiment=member.local_constants_by_experiment,
                train_error=member.train_error,
                validation_error=member.validation_error,
                regime_holdout_error=member.regime_holdout_error,
                simplicity_score=member.simplicity_score,
                physics_consistency_score=float(report.get("overall_score", 1.0) or 1.0),
                committee_weight=member.committee_weight,
                canonical_key=member.canonical_key,
                display_expr=member.display_expr,
                metadata=member.metadata,
            )
        )
    committee = build_committee_state(
        enriched,
        max_members=max_members,
        deduplicate=True,
    )
    constant_lift_summary = None
    if bool(discovery_constant_lift_enable):
        constant_lift_summary = discover_constant_lifts(
            list(committee.members),
            dataset_ids=dataset_ids,
            dataset_metadata=dataset_metadata,
            min_regimes=max(2, int(discovery_constant_lift_min_regimes)),
            trigger_mean_cv=float(discovery_constant_lift_trigger_mean_cv),
            dtype=dtype,
        )
        if bool(discovery_constant_lift_apply_enable):
            applied = apply_constant_lift_proposals(
                list(committee.members),
                constant_lift_summary,
                apply_topk=max(0, int(discovery_constant_lift_apply_topk)),
                min_rel_gain=float(discovery_constant_lift_min_rel_gain),
            )
            constant_lift_summary = dict(applied.get("summary", {}) or {})
            applied_members: list[CommitteeMember] = []
            for member in list(applied.get("applied_members", []) or []):
                if not isinstance(member, CommitteeMember):
                    continue
                report = score_physics_consistency(
                    {
                        "symbolic_structure": member.symbolic_structure,
                        "train_error": member.train_error,
                        "validation_error": member.validation_error,
                        "metadata": dict(member.metadata or {}),
                    },
                    units_spec=units_spec,
                    parameter_samples=parameter_samples_from_local_constants(
                        member.local_constants_by_experiment,
                        regime_ids=dataset_ids,
                    ),
                )
                physics_reports[str(member.member_id)] = report
                applied_members.append(
                    CommitteeMember(
                        member_id=member.member_id,
                        symbolic_structure=member.symbolic_structure,
                        fitted_constants=member.fitted_constants,
                        shared_constants=member.shared_constants,
                        local_constants_by_experiment=member.local_constants_by_experiment,
                        train_error=member.train_error,
                        validation_error=member.validation_error,
                        regime_holdout_error=member.regime_holdout_error,
                        simplicity_score=member.simplicity_score,
                        physics_consistency_score=float(report.get("overall_score", 1.0) or 1.0),
                        committee_weight=member.committee_weight,
                        canonical_key=member.canonical_key,
                        display_expr=member.display_expr,
                        metadata=member.metadata,
                    )
                )
            if applied_members:
                committee = build_committee_state(
                    list(committee.members) + applied_members,
                    max_members=max_members,
                    deduplicate=True,
                )
            surviving_applied_ids = [
                str(member.member_id)
                for member in committee.members
                if bool(dict(member.metadata or {}).get("constant_lift_applied", False))
            ]
            constant_lift_summary["surviving_applied_member_count"] = int(len(surviving_applied_ids))
            constant_lift_summary["surviving_applied_member_ids"] = list(surviving_applied_ids)
    runtime_by_member_id = {
        str(runtime.member.member_id): runtime
        for runtime in runtime_candidates
    }
    manifest_payload = _load_manifest(experiment_manifest_path)
    experiment_candidates = build_sr_experiment_candidates(
        committee=list(committee.members),
        runtime_candidates=runtime_by_member_id,
        experiment_manifest=manifest_payload,
        nvars=max(0, int(nvars)),
        dtype=dtype,
        witness_capture_enable=bool(witness_capture_enable),
        witness_hessian_diag_enable=bool(witness_hessian_diag_enable),
        diagnostic_set=str(diagnostic_set or "basic"),
    )
    selection_payload = None
    if experiment_candidates and committee.members:
        experiment_optimizer = None
        optimization_result_holder: dict[str, Any] = {}
        if bool(experiment_optimize_enable):
            forward_fns_by_member_id = {
                str(member_id): (lambda xx, runtime=runtime: _runtime_forward_value(runtime, xx))
                for member_id, runtime in runtime_by_member_id.items()
                if runtime is not None
            }
            def _experiment_optimizer(current_state, current_candidates, **kwargs):
                result = optimize_continuous_experiment_candidates(
                    current_state,
                    current_candidates,
                    forward_fns_by_member_id=forward_fns_by_member_id,
                    beta=float(kwargs.get("beta", beta)),
                    gamma=float(kwargs.get("gamma", gamma)),
                    disagreement_mode=resolve_surface_disagreement_mode(
                        kwargs.get("disagreement_mode", disagreement_mode),
                        default_mode=disagreement_mode,
                    ),
                    lambda_cost=float(kwargs.get("lambda_cost", lambda_cost)),
                    lambda_noise=float(kwargs.get("lambda_noise", lambda_noise)),
                    lambda_feasibility=float(kwargs.get("lambda_feasibility", lambda_feasibility)),
                    opt_steps=int(max(1, int(experiment_opt_steps))),
                    opt_lr=float(experiment_opt_lr),
                    project_mode=str(experiment_project_mode or "nearest_box"),
                    include_gradients=bool(witness_capture_enable or float(beta) > 0.0 or str(disagreement_mode) == "witness"),
                    include_diagnostics=bool(float(gamma) > 0.0),
                )
                optimization_result_holder["result"] = result
                return result
            experiment_optimizer = _experiment_optimizer
        selection_payload = select_next_experiment(
            committee,
            experiment_candidates,
            beta=float(beta),
            gamma=float(gamma),
            disagreement_mode=disagreement_mode,
            lambda_cost=float(lambda_cost),
            lambda_noise=float(lambda_noise),
            lambda_feasibility=float(lambda_feasibility),
            optimize_continuous=bool(experiment_optimize_enable),
            experiment_optimizer=experiment_optimizer,
        )
        optimized_candidates = dict(optimization_result_holder.get("result", {}) or {}).get("candidates", None)
        if isinstance(optimized_candidates, Sequence):
            experiment_candidates = list(optimized_candidates)
    class_sr_summary = None
    if class_sr_result is not None:
        class_sr_summary = {
            "class_tags": [str(tag) for tag in list(getattr(class_sr_result, "class_tags", []) or [])],
            "experiment_tags": [str(tag) for tag in list(getattr(class_sr_result, "experiment_tags", []) or [])],
            "class_params": _jsonable({
                str(tag): values
                for tag, values in dict(getattr(class_sr_result, "class_params", {}) or {}).items()
            }),
            "experiment_params": _jsonable([
                {
                    str(tag): values
                    for tag, values in dict(payload or {}).items()
                }
                for payload in list(getattr(class_sr_result, "experiment_params", []) or [])
            ]),
            "val_losses": _jsonable(list(getattr(class_sr_result, "val_losses", []) or [])),
            "val_loss_agg": _jsonable(getattr(class_sr_result, "val_loss_agg", None)),
            "val_loss_agg_mode": str(getattr(class_sr_result, "val_loss_agg_mode", "") or ""),
        }
    theory_benchmark = None
    if bool(theory_benchmark_enable):
        best_member = None if not committee.members else min(
            committee.members,
            key=lambda member: (
                _safe_float(member.validation_error),
                -_safe_float(member.physics_consistency_score, 0.0),
                str(member.member_id),
            ),
        )
        selected = selection_payload.get("selected", None) if isinstance(selection_payload, Mapping) else None
        ranking = list(selection_payload.get("ranking", []) or []) if isinstance(selection_payload, Mapping) else []
        selection_margin = None
        if len(ranking) >= 2:
            selection_margin = float(ranking[0]["score"]) - float(ranking[1]["score"])
        regime_score = None
        if best_member is not None:
            regime_score = _safe_float(
                dict(physics_reports.get(str(best_member.member_id), {}) or {})
                .get("checks", {})
                .get("regime_generalization", {})
                .get("score", None)
            )
        proposal_count = int(dict(constant_lift_summary or {}).get("proposal_count", 0) or 0)
        applied_count = int(dict(constant_lift_summary or {}).get("surviving_applied_member_count", 0) or 0)
        constant_lift_success = bool(applied_count > 0) if bool(discovery_constant_lift_apply_enable) else bool(proposal_count > 0)
        theory_benchmark = {
            "enabled": True,
            "best_member_id": None if best_member is None else str(best_member.member_id),
            "best_member_validation_error": None if best_member is None else _safe_float(best_member.validation_error),
            "best_member_physics_score": None if best_member is None else _safe_float(best_member.physics_consistency_score),
            "next_experiment_quality": None if not isinstance(selected, Mapping) else _safe_float(selected.get("score", None)),
            "selection_margin": None if selection_margin is None else float(selection_margin),
            "ood_survival_score": None if regime_score is None or not math.isfinite(float(regime_score)) else float(regime_score),
            "constant_lift_success": bool(constant_lift_success),
            "constant_lift_proposal_count": int(proposal_count),
            "constant_lift_applied_count": int(applied_count),
        }
    payload = {
        "mode": "sr_discovery_integration",
        "dataset": str(filepath),
        "datasets": [str(path) for path in list(filepaths or [filepath])],
        "dataset_ids": [str(item) for item in list(dataset_ids)],
        "report_path": str(report_path),
        "committee_members": [serialize_committee_member(member) for member in committee.members],
        "committee_summary": {
            "member_count": int(len(committee.members)),
            "canonical_member_count": int(committee.canonical_member_count),
            "discarded_member_ids": list(committee.discarded_member_ids),
            "members": [
                {
                    "member_id": member.member_id,
                    "display_expr": member.display_expr,
                    "validation_error": member.validation_error,
                    "committee_weight": member.committee_weight,
                    "physics_consistency_score": member.physics_consistency_score,
                    "canonical_key": member.canonical_key,
                    "source": str(dict(member.metadata or {}).get("source", "")),
                    "shared_constant_count": int(len(dict(member.shared_constants or {}))),
                    "local_constant_dataset_count": int(len(dict(member.local_constants_by_experiment or {}))),
                }
                for member in committee.members
            ],
        },
        "physics_summary": physics_reports,
        "constant_lift_summary": constant_lift_summary,
        "theory_benchmark": theory_benchmark,
        "experiment_selection": selection_payload,
        "experiment_candidates_full": [serialize_experiment_candidate(candidate) for candidate in experiment_candidates],
        "experiment_candidates": [
            {
                "experiment_id": candidate.experiment_id,
                "conditions": _jsonable(candidate.conditions),
                "cost": float(candidate.cost),
                "noise_risk": float(candidate.noise_risk),
                "feasibility_penalty": float(candidate.feasibility_penalty),
                "observable_prediction_members": sorted(str(k) for k in candidate.observable_predictions.keys()),
                "derivative_prediction_members": sorted(str(k) for k in candidate.derivative_predictions.keys()),
                "diagnostic_prediction_members": sorted(str(k) for k in candidate.diagnostic_predictions.keys()),
            }
            for candidate in experiment_candidates
        ],
        "units_payload": _serialize_units_payload(
            units_payload,
            y_transform_name=str(final_y_op_name or "identity"),
        ),
        "class_sr_summary": class_sr_summary,
        "config": {
            "research_profile": str(resolved_profile),
            "committee_topk": int(max(1, int(committee_topk))),
            "max_members": None if max_members is None else int(max_members),
            "experiment_manifest_path": None if experiment_manifest_path is None else str(experiment_manifest_path),
            "beta": float(beta),
            "gamma": float(gamma),
            "disagreement_mode": str(disagreement_mode),
            "lambda_cost": float(lambda_cost),
            "lambda_noise": float(lambda_noise),
            "lambda_feasibility": float(lambda_feasibility),
            "nvars": int(max(0, int(nvars))),
            "y_transform_name": str(final_y_op_name or "identity"),
            "discovery_constant_lift_enable": bool(discovery_constant_lift_enable),
            "discovery_constant_lift_min_regimes": int(max(2, int(discovery_constant_lift_min_regimes))),
            "discovery_constant_lift_trigger_mean_cv": float(discovery_constant_lift_trigger_mean_cv),
            "discovery_constant_lift_apply_enable": bool(discovery_constant_lift_apply_enable),
            "discovery_constant_lift_apply_topk": int(max(0, int(discovery_constant_lift_apply_topk))),
            "discovery_constant_lift_min_rel_gain": float(discovery_constant_lift_min_rel_gain),
            "witness_capture_enable": bool(witness_capture_enable),
            "witness_hessian_diag_enable": bool(witness_hessian_diag_enable),
            "diagnostic_set": str(diagnostic_set or "basic"),
            "experiment_optimize_enable": bool(experiment_optimize_enable),
            "experiment_opt_steps": int(max(1, int(experiment_opt_steps))),
            "experiment_opt_lr": float(experiment_opt_lr),
            "experiment_project_mode": str(experiment_project_mode or "nearest_box"),
            "theory_benchmark_enable": bool(theory_benchmark_enable),
        },
        "runtime_summary": {
            "candidate_count": int(len(runtime_candidates)),
            "evaluable_member_count": int(sum(1 for item in runtime_candidates if item.model is not None)),
            "witness_capture_enable": bool(witness_capture_enable),
        },
    }
    payload["research_activation"] = _discovery_research_activation_from_payload(payload)
    return payload


def discovery_summary_from_payload(
    payload: Mapping[str, Any],
    *,
    results_path: str | pathlib.Path,
) -> dict[str, Any]:
    committee_summary = dict(payload.get("committee_summary", {}) or {})
    selection = payload.get("experiment_selection", None)
    selected = selection.get("selected", None) if isinstance(selection, Mapping) else None
    constant_lift = dict(payload.get("constant_lift_summary", {}) or {})
    research_activation = _discovery_research_activation_from_payload(payload)
    return {
        "enabled": True,
        "results_path": str(results_path),
        "committee_member_count": int(committee_summary.get("member_count", 0) or 0),
        "canonical_member_count": int(committee_summary.get("canonical_member_count", 0) or 0),
        "constant_lift_proposal_count": int(constant_lift.get("proposal_count", 0) or 0),
        "constant_lift_applied_count": int(constant_lift.get("surviving_applied_member_count", constant_lift.get("applied_member_count", 0)) or 0),
        "selected_experiment": _jsonable(selected),
        "closed_loop_ready": bool(payload.get("committee_members")) and bool(payload.get("experiment_candidates_full")),
        "theory_benchmark_enabled": bool(dict(payload.get("config", {}) or {}).get("theory_benchmark_enable", False)),
        "research_profile": str(research_activation.get("research_profile", "legacy") or "legacy"),
        "research_activation": research_activation,
    }


def run_closed_loop_from_discovery_payload(
    payload: Mapping[str, Any],
    *,
    committee_max_members: int | None = None,
    weight_temperature: float = 1.0,
    beta: float | None = None,
    gamma: float | None = None,
    disagreement_mode: str | None = None,
    lambda_cost: float | None = None,
    lambda_noise: float | None = None,
    lambda_feasibility: float | None = None,
) -> dict[str, Any]:
    candidate_laws = deserialize_committee_members(payload.get("committee_members", []) or [])
    experiment_candidates = deserialize_experiment_candidates(
        payload.get("experiment_candidates_full", []) or []
    )
    if not candidate_laws:
        raise ValueError("discovery payload does not contain committee_members")
    if not experiment_candidates:
        raise ValueError("discovery payload does not contain experiment_candidates_full")
    config = dict(payload.get("config", {}) or {})
    y_transform_name = str(config.get("y_transform_name", "identity") or "identity")
    units_spec = _units_spec_from_payload(
        payload.get("units_payload", None),
        y_transform_name=y_transform_name,
    )
    resolved_mode = resolve_surface_disagreement_mode(
        config.get("disagreement_mode", None) if disagreement_mode is None else disagreement_mode,
        default_mode="witness",
    )
    result = run_closed_loop_iteration(
        candidate_laws,
        experiment_candidates,
        units_spec=units_spec,
        committee_max_members=committee_max_members,
        weight_temperature=float(weight_temperature),
        beta=float(config.get("beta", 0.0) if beta is None else beta),
        gamma=float(config.get("gamma", 0.0) if gamma is None else gamma),
        disagreement_mode=resolved_mode,
        lambda_cost=float(config.get("lambda_cost", 1.0) if lambda_cost is None else lambda_cost),
        lambda_noise=float(config.get("lambda_noise", 1.0) if lambda_noise is None else lambda_noise),
        lambda_feasibility=float(
            config.get("lambda_feasibility", 1.0)
            if lambda_feasibility is None
            else lambda_feasibility
        ),
    )
    return {
        "mode": "sr_closed_loop_replay",
        "dataset": str(payload.get("dataset", "") or ""),
        "datasets": [str(item) for item in list(payload.get("datasets", []) or [])],
        "report_path": str(payload.get("report_path", "") or ""),
        "committee_summary": {
            "member_count": int(len(result.committee_state.members)),
            "canonical_member_count": int(result.committee_state.canonical_member_count),
            "discarded_member_ids": list(result.committee_state.discarded_member_ids),
        },
        "selected_experiment": _jsonable(result.selected_experiment),
        "ranked_experiments": _jsonable(list(result.ranked_experiments)),
        "physics_reports": _jsonable(dict(result.physics_reports or {})),
        "config": {
            "committee_max_members": None if committee_max_members is None else int(committee_max_members),
            "weight_temperature": float(weight_temperature),
            "beta": float(config.get("beta", 0.0) if beta is None else beta),
            "gamma": float(config.get("gamma", 0.0) if gamma is None else gamma),
            "disagreement_mode": str(resolved_mode),
            "lambda_cost": float(config.get("lambda_cost", 1.0) if lambda_cost is None else lambda_cost),
            "lambda_noise": float(config.get("lambda_noise", 1.0) if lambda_noise is None else lambda_noise),
            "lambda_feasibility": float(
                config.get("lambda_feasibility", 1.0)
                if lambda_feasibility is None
                else lambda_feasibility
            ),
        },
    }


__all__ = [
    "RuntimeDiscoveryCandidate",
    "build_sr_experiment_candidates",
    "deserialize_committee_members",
    "deserialize_experiment_candidates",
    "discovery_summary_from_payload",
    "run_sr_discovery_integration",
    "run_closed_loop_from_discovery_payload",
    "serialize_committee_member",
    "serialize_experiment_candidate",
]
