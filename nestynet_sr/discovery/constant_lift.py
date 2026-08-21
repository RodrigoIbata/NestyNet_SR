# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

import torch

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
    ast_to_human_readable,
)
from nestynet_sr.sr_search.factorized_search.constant_lift_solver import (
    _ast_from_jsonable as _lift_ast_from_jsonable,
    _mapping_to_ast as _lift_mapping_to_tuple_ast,
    solve_constant_lift_task,
)
from nestynet_sr.sr_search.factorized_search.expr_ast import (
    is_valid_node as is_valid_tuple_ast,
    simplify as simplify_tuple_ast,
)

from .committee import CommitteeMember, canonicalize_candidate_law
from .physics_tests import check_parameter_stability


_TOKEN_RE_TEMPLATE = r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])"


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _jsonable(value: Any) -> Any:
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
    scalar = _safe_float(value)
    return None if scalar is None else float(scalar)


def parameter_samples_from_local_constants(
    local_constants_by_experiment: Mapping[str, Mapping[str, Any]] | None,
    *,
    regime_ids: Sequence[str] | None = None,
) -> list[dict[str, float]]:
    rows = dict(local_constants_by_experiment or {})
    ordered_ids = [
        str(item)
        for item in (
            list(regime_ids or [])
            if regime_ids is not None
            else sorted(rows.keys())
        )
        if str(item) in rows
    ]
    out: list[dict[str, float]] = []
    for regime_id in ordered_ids:
        sample = dict(rows.get(str(regime_id), {}) or {})
        numeric: dict[str, float] = {}
        for key, value in sample.items():
            scalar = _safe_float(value)
            if scalar is not None:
                numeric[str(key)] = float(scalar)
        if numeric:
            out.append(numeric)
    return out


def _substitution_preview(expr: str, *, constant_name: str, lift_expr: str) -> str:
    expr_text = str(expr or "").strip()
    if not expr_text:
        return f"{str(constant_name)} -> ({str(lift_expr)})"
    pattern = re.compile(_TOKEN_RE_TEMPLATE % re.escape(str(constant_name)))
    if pattern.search(expr_text):
        return pattern.sub(f"({str(lift_expr)})", expr_text)
    return f"{expr_text} [{str(constant_name)} -> ({str(lift_expr)})]"


def _drop_lifted_constant_rows(
    local_constants_by_experiment: Mapping[str, Mapping[str, Any]] | None,
    *,
    constant_name: str,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for regime_id, payload in dict(local_constants_by_experiment or {}).items():
        kept: dict[str, float] = {}
        for key, value in dict(payload or {}).items():
            if str(key) == str(constant_name):
                continue
            scalar = _safe_float(value)
            if scalar is not None:
                kept[str(key)] = float(scalar)
        if kept:
            out[str(regime_id)] = kept
    return out


def _proposal_sort_key(payload: Mapping[str, Any]) -> tuple[float, float, str, str]:
    proposal = dict(payload.get("proposal", {}) or {})
    improvement = _safe_float(proposal.get("improvement_ratio", None))
    mean_cv = _safe_float(proposal.get("mean_cv", None))
    return (
        -float(improvement) if improvement is not None else float("inf"),
        -float(mean_cv) if mean_cv is not None else float("inf"),
        str(payload.get("member_id", "") or ""),
        str(payload.get("constant_name", "") or ""),
    )


def _applied_simplicity_score(parent: CommitteeMember, lifted_display_expr: str) -> float:
    parent_score = _safe_float(parent.simplicity_score)
    if parent_score is None or not math.isfinite(float(parent_score)) or float(parent_score) <= 0.0:
        parent_score = 1.0
    parent_len = max(1, len(str(parent.display_expr or parent.symbolic_structure or "")))
    lifted_len = max(1, len(str(lifted_display_expr or "")))
    return float(parent_score) * float(parent_len) / float(lifted_len)


def _structured_parent_expr(parent: CommitteeMember) -> Any:
    structure = parent.symbolic_structure
    if structure is not None and not isinstance(structure, str):
        return structure
    metadata = dict(parent.metadata or {})
    for key in ("structured_symbolic_structure", "symbolic_structure_ast", "ast"):
        value = metadata.get(key, None)
        if value is not None and not isinstance(value, str):
            return value
    return None


def _proposal_lift_tuple_ast(proposal: Mapping[str, Any]) -> Any:
    raw_expr = _lift_ast_from_jsonable(proposal.get("expr_ast", None))
    if not is_valid_tuple_ast(raw_expr):
        return None
    mapped = _lift_mapping_to_tuple_ast(raw_expr, dict(proposal.get("mapping", {}) or {}))
    if is_valid_tuple_ast(mapped):
        try:
            return simplify_tuple_ast(mapped)
        except Exception:
            return mapped
    try:
        return simplify_tuple_ast(raw_expr)
    except Exception:
        return raw_expr


def _feature_atom_name(feature_names: Sequence[Any], idx: int) -> str:
    items = [str(item) for item in list(feature_names or [])]
    if 0 <= int(idx) < len(items):
        name = str(items[int(idx)] or "").strip()
        if name:
            return name
    return f"meta_feature_{int(idx)}"


def _tuple_meta_ast_to_bridge(node: Any, *, feature_names: Sequence[Any]) -> Any:
    op = str(node[0])
    if op == "var":
        idx = int(node[1])
        feature_name = _feature_atom_name(feature_names, idx)
        return AtomNode(
            kind=str(feature_name),
            var_idxs=(int(idx),),
            kwargs={"name": str(feature_name), "source": "constant_lift_feature"},
            tag=str(feature_name),
        )
    if op == "const":
        return ConstNode(float(node[1]))
    if op == "sin":
        return SinNode(_tuple_meta_ast_to_bridge(node[1], feature_names=feature_names))
    if op == "cos":
        return CosNode(_tuple_meta_ast_to_bridge(node[1], feature_names=feature_names))
    if op == "exp":
        return ExpNode(_tuple_meta_ast_to_bridge(node[1], feature_names=feature_names))
    if op == "log":
        return LogNode(_tuple_meta_ast_to_bridge(node[1], feature_names=feature_names))
    if op == "sqrt":
        return PowNode(_tuple_meta_ast_to_bridge(node[1], feature_names=feature_names), 0.5)
    if op == "sqr":
        return PowNode(_tuple_meta_ast_to_bridge(node[1], feature_names=feature_names), 2.0)
    if op == "neg":
        return MulNode(ConstNode(-1.0), _tuple_meta_ast_to_bridge(node[1], feature_names=feature_names))
    if op == "add":
        return AddNode(
            _tuple_meta_ast_to_bridge(node[1], feature_names=feature_names),
            _tuple_meta_ast_to_bridge(node[2], feature_names=feature_names),
        )
    if op == "sub":
        return AddNode(
            _tuple_meta_ast_to_bridge(node[1], feature_names=feature_names),
            MulNode(ConstNode(-1.0), _tuple_meta_ast_to_bridge(node[2], feature_names=feature_names)),
        )
    if op == "mul":
        return MulNode(
            _tuple_meta_ast_to_bridge(node[1], feature_names=feature_names),
            _tuple_meta_ast_to_bridge(node[2], feature_names=feature_names),
        )
    if op == "div":
        return MulNode(
            _tuple_meta_ast_to_bridge(node[1], feature_names=feature_names),
            PowNode(_tuple_meta_ast_to_bridge(node[2], feature_names=feature_names), -1.0),
        )
    raise ValueError(f"unsupported tuple AST op {op!r}")


def _named_constant_from_bridge_leaf(node: Any) -> str | None:
    if not isinstance(node, AtomNode):
        return None
    kind = str(getattr(node, "kind", "") or "").lower()
    if kind not in ("free_const", "freeconst", "free_constant", "fixed_const", "fixedconst", "fixed_constant", "scale"):
        return None
    kwargs = dict(getattr(node, "kwargs", {}) or {})
    name = kwargs.get("name", None)
    if name is None:
        name = getattr(node, "tag", None)
    if name is None:
        return None
    return str(name)


def _clone_bridge_atom(node: AtomNode, *, inputs: Sequence[Any] | None = None) -> AtomNode:
    return AtomNode(
        kind=str(getattr(node, "kind", "") or ""),
        var_idxs=tuple(int(i) for i in tuple(getattr(node, "var_idxs", ()) or ())),
        kwargs=dict(getattr(node, "kwargs", {}) or {}),
        tag=getattr(node, "tag", None),
        inputs=None if inputs is None and getattr(node, "inputs", None) is None else tuple(inputs if inputs is not None else tuple(getattr(node, "inputs", ()) or ())),
        scope=str(getattr(node, "scope", "experiment") or "experiment"),
    )


def _substitute_named_constant_in_bridge(node: Any, *, constant_name: str, replacement: Any) -> tuple[Any, int]:
    if isinstance(node, AtomNode):
        leaf_name = _named_constant_from_bridge_leaf(node)
        if leaf_name is not None and str(leaf_name) == str(constant_name):
            return replacement, 1
        inputs = getattr(node, "inputs", None)
        if inputs is None:
            return _clone_bridge_atom(node), 0
        new_inputs: list[Any] = []
        replace_count = 0
        for child in tuple(inputs or ()):
            new_child, child_count = _substitute_named_constant_in_bridge(
                child,
                constant_name=constant_name,
                replacement=replacement,
            )
            new_inputs.append(new_child)
            replace_count += int(child_count)
        return _clone_bridge_atom(node, inputs=new_inputs), int(replace_count)
    if isinstance(node, AddNode):
        left, left_count = _substitute_named_constant_in_bridge(node.left, constant_name=constant_name, replacement=replacement)
        right, right_count = _substitute_named_constant_in_bridge(node.right, constant_name=constant_name, replacement=replacement)
        return AddNode(left, right), int(left_count + right_count)
    if isinstance(node, MulNode):
        left, left_count = _substitute_named_constant_in_bridge(node.left, constant_name=constant_name, replacement=replacement)
        right, right_count = _substitute_named_constant_in_bridge(node.right, constant_name=constant_name, replacement=replacement)
        return MulNode(left, right), int(left_count + right_count)
    if isinstance(node, PowNode):
        base, replace_count = _substitute_named_constant_in_bridge(node.base, constant_name=constant_name, replacement=replacement)
        return PowNode(base, float(node.exponent)), int(replace_count)
    if isinstance(node, LogNode):
        arg, replace_count = _substitute_named_constant_in_bridge(node.arg, constant_name=constant_name, replacement=replacement)
        return LogNode(arg), int(replace_count)
    if isinstance(node, ExpNode):
        arg, replace_count = _substitute_named_constant_in_bridge(node.arg, constant_name=constant_name, replacement=replacement)
        return ExpNode(arg), int(replace_count)
    if isinstance(node, SinNode):
        arg, replace_count = _substitute_named_constant_in_bridge(node.arg, constant_name=constant_name, replacement=replacement)
        return SinNode(arg), int(replace_count)
    if isinstance(node, CosNode):
        arg, replace_count = _substitute_named_constant_in_bridge(node.arg, constant_name=constant_name, replacement=replacement)
        return CosNode(arg), int(replace_count)
    if isinstance(node, ConjNode):
        arg, replace_count = _substitute_named_constant_in_bridge(node.arg, constant_name=constant_name, replacement=replacement)
        return ConjNode(arg), int(replace_count)
    if isinstance(node, RealNode):
        arg, replace_count = _substitute_named_constant_in_bridge(node.arg, constant_name=constant_name, replacement=replacement)
        return RealNode(arg), int(replace_count)
    if isinstance(node, ImagNode):
        arg, replace_count = _substitute_named_constant_in_bridge(node.arg, constant_name=constant_name, replacement=replacement)
        return ImagNode(arg), int(replace_count)
    if isinstance(node, AbsNode):
        arg, replace_count = _substitute_named_constant_in_bridge(node.arg, constant_name=constant_name, replacement=replacement)
        return AbsNode(arg), int(replace_count)
    if isinstance(node, ArgNode):
        arg, replace_count = _substitute_named_constant_in_bridge(node.arg, constant_name=constant_name, replacement=replacement)
        return ArgNode(arg), int(replace_count)
    if isinstance(node, ConstNode):
        return ConstNode(node.value), 0
    return node, 0


def _ast_reinserted_structure(
    parent: CommitteeMember,
    *,
    constant_name: str,
    proposal: Mapping[str, Any],
) -> tuple[Any | None, str]:
    parent_structure = _structured_parent_expr(parent)
    if parent_structure is None:
        return None, "display_fallback"
    lift_tuple_ast = _proposal_lift_tuple_ast(proposal)
    if lift_tuple_ast is None:
        return None, "display_fallback"
    try:
        replacement = _tuple_meta_ast_to_bridge(
            lift_tuple_ast,
            feature_names=list(proposal.get("feature_names", []) or []),
        )
    except Exception:
        return None, "display_fallback"
    try:
        replaced, replace_count = _substitute_named_constant_in_bridge(
            parent_structure,
            constant_name=str(constant_name),
            replacement=replacement,
        )
    except Exception:
        return None, "display_fallback"
    if int(replace_count) <= 0:
        return None, "display_fallback"
    return replaced, "ast"


def _proposal_lift_display_expr(proposal: Mapping[str, Any]) -> str:
    tuple_ast = _proposal_lift_tuple_ast(proposal)
    if tuple_ast is None:
        return str(proposal.get("expr", "") or "")
    try:
        bridge = _tuple_meta_ast_to_bridge(
            tuple_ast,
            feature_names=list(proposal.get("feature_names", []) or []),
        )
        rendered = str(ast_to_human_readable(bridge))
    except Exception:
        rendered = ""
    return rendered or str(proposal.get("expr", "") or "")


def _build_applied_member(
    parent: CommitteeMember,
    *,
    proposal: Mapping[str, Any],
    applied_index: int,
) -> CommitteeMember | None:
    constant_name = str(proposal.get("constant_name", "") or "")
    if not constant_name:
        return None
    lifted_display_expr = str(
        proposal.get("lifted_display_expr", "")
        or _substitution_preview(
            str(parent.display_expr or parent.symbolic_structure or ""),
            constant_name=constant_name,
            lift_expr=str(
                dict(proposal.get("substitution_preview", {}) or {}).get(
                    "lift_expr",
                    _proposal_lift_display_expr(proposal),
                )
                or ""
            ),
        )
    )
    if not lifted_display_expr:
        return None
    fitted_constants = {
        str(key): float(value)
        for key, value in dict(parent.fitted_constants or {}).items()
        if str(key) != constant_name and _safe_float(value) is not None
    }
    local_constants = _drop_lifted_constant_rows(
        parent.local_constants_by_experiment,
        constant_name=constant_name,
    )
    structured_symbolic_structure, structure_mode = _ast_reinserted_structure(
        parent,
        constant_name=constant_name,
        proposal=proposal,
    )
    symbolic_structure = structured_symbolic_structure if structured_symbolic_structure is not None else str(lifted_display_expr)
    canonical = canonicalize_candidate_law(symbolic_structure)
    metadata = dict(parent.metadata or {})
    metadata.update(
        {
            "source": "constant_lift_apply",
            "constant_lift_applied": True,
            "constant_lift_parent_member_id": str(parent.member_id),
            "constant_lift_constant_name": str(constant_name),
            "constant_lift_lift_expr": str(
                dict(proposal.get("substitution_preview", {}) or {}).get(
                    "lift_expr",
                    _proposal_lift_display_expr(proposal),
                )
                or ""
            ),
            "constant_lift_improvement_ratio": _safe_float(proposal.get("improvement_ratio", None)),
            "constant_lift_feature_names": [
                str(item)
                for item in list(proposal.get("feature_names", []) or [])
            ],
            "constant_lift_feature_source": str(proposal.get("feature_source", "") or ""),
            "constant_lift_solver": str(proposal.get("solver", "") or ""),
            "constant_lift_mapping": _jsonable(dict(proposal.get("mapping", {}) or {})),
            "constant_lift_expr_ast": _jsonable(proposal.get("expr_ast", None)),
            "constant_lift_symbolic_structure_mode": str(structure_mode),
            "constant_lift_regime_ids": [
                str(item)
                for item in list(proposal.get("regime_ids", []) or [])
            ],
        }
    )
    return CommitteeMember(
        member_id=f"{str(parent.member_id)}:lift:{str(constant_name)}:{int(applied_index)}",
        symbolic_structure=symbolic_structure,
        fitted_constants=fitted_constants,
        shared_constants=dict(parent.shared_constants or {}),
        local_constants_by_experiment=local_constants,
        train_error=float(parent.train_error),
        validation_error=float(parent.validation_error),
        regime_holdout_error=parent.regime_holdout_error,
        simplicity_score=_applied_simplicity_score(parent, lifted_display_expr),
        physics_consistency_score=float(parent.physics_consistency_score),
        committee_weight=0.0,
        canonical_key=str(canonical["canonical_key"]),
        display_expr=str(lifted_display_expr),
        metadata=metadata,
    )


def apply_constant_lift_proposals(
    committee: Sequence[CommitteeMember],
    summary: Mapping[str, Any] | None,
    *,
    apply_topk: int = 1,
    min_rel_gain: float = 1.01,
) -> dict[str, Any]:
    summary_rows = [dict(row) for row in list(dict(summary or {}).get("members", []) or []) if isinstance(row, Mapping)]
    by_member_id = {str(member.member_id): member for member in list(committee or [])}
    eligible: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        member_id = str(row.get("member_id", "") or "")
        proposals_out: list[dict[str, Any]] = []
        for proposal in [dict(item) for item in list(row.get("proposals", []) or []) if isinstance(item, Mapping)]:
            normalized = dict(proposal)
            normalized["applied"] = False
            normalized["applied_member_id"] = None
            improvement = _safe_float(normalized.get("improvement_ratio", None))
            if (
                member_id in by_member_id
                and improvement is not None
                and float(improvement) >= float(min_rel_gain)
            ):
                eligible.append(
                    {
                        "member_id": str(member_id),
                        "constant_name": str(normalized.get("constant_name", "") or ""),
                        "proposal": normalized,
                    }
                )
            proposals_out.append(normalized)
        normalized_rows.append({**row, "proposals": proposals_out})
    eligible.sort(key=_proposal_sort_key)
    applied_members: list[CommitteeMember] = []
    applied_rows: list[dict[str, Any]] = []
    for applied_index, item in enumerate(eligible[: max(0, int(apply_topk))]):
        parent = by_member_id.get(str(item["member_id"]), None)
        proposal = dict(item["proposal"])
        if parent is None:
            continue
        applied_member = _build_applied_member(
            parent,
            proposal=proposal,
            applied_index=int(applied_index),
        )
        if applied_member is None:
            continue
        applied_members.append(applied_member)
        applied_rows.append(
            {
                "member_id": str(applied_member.member_id),
                "parent_member_id": str(parent.member_id),
                "constant_name": str(proposal.get("constant_name", "") or ""),
                "display_expr": str(applied_member.display_expr),
                "improvement_ratio": _safe_float(proposal.get("improvement_ratio", None)),
                "feature_source": str(proposal.get("feature_source", "") or ""),
            }
        )
        for row in normalized_rows:
            if str(row.get("member_id", "") or "") != str(parent.member_id):
                continue
            for row_proposal in row["proposals"]:
                if (
                    str(row_proposal.get("constant_name", "") or "") == str(proposal.get("constant_name", "") or "")
                    and str(row_proposal.get("lifted_display_expr", "") or "") == str(proposal.get("lifted_display_expr", "") or "")
                ):
                    row_proposal["applied"] = True
                    row_proposal["applied_member_id"] = str(applied_member.member_id)
                    break
            break
    merged_summary = dict(summary or {})
    merged_summary.update(
        {
            "enabled": bool(dict(summary or {}).get("enabled", False)),
            "apply_enabled": True,
            "apply_topk": int(max(0, int(apply_topk))),
            "min_rel_gain": float(min_rel_gain),
            "members": normalized_rows,
            "applied_member_count": int(len(applied_members)),
            "applied_member_ids": [str(member.member_id) for member in applied_members],
            "applied_members": applied_rows,
        }
    )
    return {
        "summary": merged_summary,
        "applied_members": applied_members,
    }


def discover_constant_lifts(
    committee: Sequence[CommitteeMember],
    *,
    dataset_ids: Sequence[str] | None = None,
    dataset_metadata: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    min_regimes: int = 3,
    trigger_mean_cv: float = 0.5,
    min_improvement_ratio: float = 1.01,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    member_rows: list[dict[str, Any]] = []
    proposal_count = 0
    triggered_member_count = 0
    for member in list(committee or []):
        local_by_experiment = dict(member.local_constants_by_experiment or {})
        regime_order = [
            str(item)
            for item in (
                list(dataset_ids or [])
                if dataset_ids
                else sorted(local_by_experiment.keys())
            )
            if str(item) in local_by_experiment
        ]
        parameter_samples = parameter_samples_from_local_constants(
            local_by_experiment,
            regime_ids=regime_order,
        )
        stability = check_parameter_stability(
            parameter_samples,
            max_mean_cv=float(trigger_mean_cv),
        )
        per_param_cvs = dict(stability.details.get("parameter_cvs", {}) or {})
        per_param_counts = dict(stability.details.get("parameter_sample_counts", {}) or {})
        proposals: list[dict[str, Any]] = []
        if len(regime_order) >= int(min_regimes):
            for constant_name in sorted(per_param_cvs.keys()):
                mean_cv = _safe_float(per_param_cvs.get(constant_name, None))
                sample_count = int(per_param_counts.get(constant_name, 0) or 0)
                if mean_cv is None or float(mean_cv) <= float(trigger_mean_cv):
                    continue
                if sample_count < int(min_regimes):
                    continue
                values_by_regime = {
                    str(regime_id): dict(local_by_experiment.get(str(regime_id), {}) or {}).get(constant_name, None)
                    for regime_id in regime_order
                    if constant_name in dict(local_by_experiment.get(str(regime_id), {}) or {})
                }
                lift = solve_constant_lift_task(
                    regime_ids=[str(regime_id) for regime_id in regime_order if str(regime_id) in values_by_regime],
                    values_by_regime=values_by_regime,
                    dataset_metadata=dataset_metadata,
                    dtype=dtype,
                )
                if not isinstance(lift, Mapping):
                    continue
                improvement_ratio = _safe_float(lift.get("improvement_ratio", None))
                if improvement_ratio is None or float(improvement_ratio) < float(min_improvement_ratio):
                    continue
                lift_expr = _proposal_lift_display_expr(lift)
                lifted_display_expr = _substitution_preview(
                    str(member.display_expr or member.symbolic_structure or ""),
                    constant_name=str(constant_name),
                    lift_expr=lift_expr,
                )
                proposal = {
                    "constant_name": str(constant_name),
                    "mean_cv": float(mean_cv),
                    "sample_count": int(sample_count),
                    "substitution_preview": {
                        "constant_name": str(constant_name),
                        "lift_expr": str(lift_expr),
                    },
                    "lifted_display_expr": str(lifted_display_expr),
                    **dict(_jsonable(dict(lift))),
                }
                proposals.append(proposal)
        if proposals:
            triggered_member_count += 1
            proposal_count += int(len(proposals))
        member_rows.append(
            {
                "member_id": str(member.member_id),
                "triggered": bool(proposals),
                "parameter_stability": _jsonable(
                    {
                        "passed": stability.passed,
                        "score": stability.score,
                        "details": dict(stability.details or {}),
                    }
                ),
                "proposals": proposals,
            }
        )
    return {
        "enabled": True,
        "min_regimes": int(min_regimes),
        "trigger_mean_cv": float(trigger_mean_cv),
        "min_improvement_ratio": float(min_improvement_ratio),
        "triggered_member_count": int(triggered_member_count),
        "proposal_count": int(proposal_count),
        "members": member_rows,
    }


__all__ = [
    "apply_constant_lift_proposals",
    "discover_constant_lifts",
    "parameter_samples_from_local_constants",
]
