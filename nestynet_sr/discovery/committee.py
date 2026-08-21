# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from nestynet_sr.sr_core.bridges import (
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
    AbsNode,
    ast_to_human_readable,
    get_input_exprs,
)
from nestynet_sr.sr_search.factorized_search.expr_ast import is_valid_node as is_valid_tuple_ast
from nestynet_sr.sr_search.factorized_search.expr_ast import node_str as tuple_node_str
from nestynet_sr.sr_search.gauge_fix_canonical import gauge_fix_multiplicative


_NUMERIC_TOKEN_RE = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class CommitteeMember:
    member_id: str
    symbolic_structure: Any
    fitted_constants: Mapping[str, float] = field(default_factory=dict)
    shared_constants: Mapping[str, float] = field(default_factory=dict)
    local_constants_by_experiment: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    train_error: float = float("nan")
    validation_error: float = float("nan")
    regime_holdout_error: float | None = None
    simplicity_score: float = 1.0
    physics_consistency_score: float = 1.0
    committee_weight: float = 0.0
    canonical_key: str = ""
    display_expr: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommitteeState:
    members: tuple[CommitteeMember, ...]
    canonical_member_count: int
    discarded_member_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _candidate_error(member: CommitteeMember) -> float:
    for value in (
        member.validation_error,
        member.regime_holdout_error,
        member.train_error,
    ):
        score = _safe_float(value)
        if math.isfinite(score):
            return float(score)
    return float("inf")


def _const_placeholder() -> tuple[str, str]:
    return ("const", "C")


def _serialize_canonical(node: Any) -> str:
    if isinstance(node, (tuple, list)):
        return "(" + ",".join(_serialize_canonical(part) for part in node) + ")"
    return repr(node)


def _canonicalize_string_expr(expr: str, *, strip_numeric_constants: bool) -> str:
    raw = re.sub(r"\s+", "", str(expr))
    if strip_numeric_constants:
        raw = _NUMERIC_TOKEN_RE.sub("C", raw)
    return raw


def _canonicalize_tuple_ast(node: Any, *, strip_numeric_constants: bool) -> Any:
    if not isinstance(node, (tuple, list)) or not node:
        return repr(node)
    op = str(node[0])
    if op == "var":
        return ("var", int(node[1]))
    if op == "hparam":
        return ("hparam", "H")
    if op == "const":
        return _const_placeholder()
    if op in ("sin", "cos", "exp", "log", "sqrt", "sqr", "neg"):
        return (op, _canonicalize_tuple_ast(node[1], strip_numeric_constants=strip_numeric_constants))
    if op == "sub":
        return (
            "sub",
            _canonicalize_tuple_ast(node[1], strip_numeric_constants=strip_numeric_constants),
            _canonicalize_tuple_ast(node[2], strip_numeric_constants=strip_numeric_constants),
        )
    if op in ("add", "mul", "div"):
        left = _canonicalize_tuple_ast(node[1], strip_numeric_constants=strip_numeric_constants)
        right = _canonicalize_tuple_ast(node[2], strip_numeric_constants=strip_numeric_constants)
        if op == "mul":
            factors = [left, right]
            if strip_numeric_constants:
                factors = [factor for factor in factors if factor != _const_placeholder()]
            if not factors:
                return _const_placeholder()
            if len(factors) == 1:
                return factors[0]
            ordered = sorted(factors, key=_serialize_canonical)
            return ("mul", ordered[0], ordered[1])
        if op == "add":
            ordered = sorted((left, right), key=_serialize_canonical)
            return ("add", ordered[0], ordered[1])
        return ("div", left, right)
    return _canonicalize_string_expr(repr(node), strip_numeric_constants=strip_numeric_constants)


def _canonicalize_bridge_node(node: Any, *, strip_numeric_constants: bool, apply_multiplicative_gauge_fix: bool) -> Any:
    current = gauge_fix_multiplicative(node) if apply_multiplicative_gauge_fix else node
    if isinstance(current, ConstNode):
        return _const_placeholder()
    if isinstance(current, AtomNode):
        kind = str(getattr(current, "kind", "")).lower()
        if kind in ("var", "x", "input") and len(tuple(getattr(current, "var_idxs", ()) or ())) == 1:
            return ("var", int(current.var_idxs[0]))
        if kind in ("free_const", "freeconst", "free_constant", "fixed_const", "scale"):
            return _const_placeholder()
        inputs = tuple(get_input_exprs(current))
        if inputs:
            return (
                "atom",
                kind,
                tuple(
                    _canonicalize_bridge_node(
                        inp,
                        strip_numeric_constants=strip_numeric_constants,
                        apply_multiplicative_gauge_fix=False,
                    )
                    for inp in inputs
                ),
            )
        raw_vars = tuple(int(idx) for idx in tuple(getattr(current, "var_idxs", ()) or ()))
        return ("atom", kind, raw_vars)
    if isinstance(current, AddNode):
        left = _canonicalize_bridge_node(
            current.left,
            strip_numeric_constants=strip_numeric_constants,
            apply_multiplicative_gauge_fix=False,
        )
        right = _canonicalize_bridge_node(
            current.right,
            strip_numeric_constants=strip_numeric_constants,
            apply_multiplicative_gauge_fix=False,
        )
        ordered = sorted((left, right), key=_serialize_canonical)
        return ("add", ordered[0], ordered[1])
    if isinstance(current, MulNode):
        left = _canonicalize_bridge_node(
            current.left,
            strip_numeric_constants=strip_numeric_constants,
            apply_multiplicative_gauge_fix=False,
        )
        right = _canonicalize_bridge_node(
            current.right,
            strip_numeric_constants=strip_numeric_constants,
            apply_multiplicative_gauge_fix=False,
        )
        factors = [left, right]
        if strip_numeric_constants:
            factors = [factor for factor in factors if factor != _const_placeholder()]
        if not factors:
            return _const_placeholder()
        if len(factors) == 1:
            return factors[0]
        ordered = sorted(factors, key=_serialize_canonical)
        return ("mul", ordered[0], ordered[1])
    if isinstance(current, PowNode):
        return (
            "pow",
            _canonicalize_bridge_node(
                current.base,
                strip_numeric_constants=strip_numeric_constants,
                apply_multiplicative_gauge_fix=False,
            ),
            float(current.exponent),
        )
    if isinstance(current, LogNode):
        return ("log", _canonicalize_bridge_node(current.arg, strip_numeric_constants=strip_numeric_constants, apply_multiplicative_gauge_fix=False))
    if isinstance(current, ExpNode):
        return ("exp", _canonicalize_bridge_node(current.arg, strip_numeric_constants=strip_numeric_constants, apply_multiplicative_gauge_fix=False))
    if isinstance(current, SinNode):
        return ("sin", _canonicalize_bridge_node(current.arg, strip_numeric_constants=strip_numeric_constants, apply_multiplicative_gauge_fix=False))
    if isinstance(current, CosNode):
        return ("cos", _canonicalize_bridge_node(current.arg, strip_numeric_constants=strip_numeric_constants, apply_multiplicative_gauge_fix=False))
    if isinstance(current, ConjNode):
        return ("conj", _canonicalize_bridge_node(current.arg, strip_numeric_constants=strip_numeric_constants, apply_multiplicative_gauge_fix=False))
    if isinstance(current, RealNode):
        return ("real", _canonicalize_bridge_node(current.arg, strip_numeric_constants=strip_numeric_constants, apply_multiplicative_gauge_fix=False))
    if isinstance(current, ImagNode):
        return ("imag", _canonicalize_bridge_node(current.arg, strip_numeric_constants=strip_numeric_constants, apply_multiplicative_gauge_fix=False))
    if isinstance(current, AbsNode):
        return ("abs", _canonicalize_bridge_node(current.arg, strip_numeric_constants=strip_numeric_constants, apply_multiplicative_gauge_fix=False))
    if isinstance(current, ArgNode):
        return ("arg", _canonicalize_bridge_node(current.arg, strip_numeric_constants=strip_numeric_constants, apply_multiplicative_gauge_fix=False))
    return _canonicalize_string_expr(ast_to_human_readable(current), strip_numeric_constants=strip_numeric_constants)


def canonicalize_candidate_law(
    expr: Any,
    *,
    strip_numeric_constants: bool = True,
    apply_multiplicative_gauge_fix: bool = True,
) -> dict[str, str]:
    if isinstance(expr, str):
        canonical_obj = _canonicalize_string_expr(expr, strip_numeric_constants=strip_numeric_constants)
        display = str(expr)
    elif is_valid_tuple_ast(expr):
        canonical_obj = _canonicalize_tuple_ast(expr, strip_numeric_constants=strip_numeric_constants)
        display = tuple_node_str(expr)
    elif isinstance(expr, (AtomNode, AddNode, MulNode, PowNode, LogNode, ExpNode, SinNode, CosNode, ConjNode, RealNode, ImagNode, AbsNode, ArgNode, ConstNode)):
        canonical_obj = _canonicalize_bridge_node(
            expr,
            strip_numeric_constants=strip_numeric_constants,
            apply_multiplicative_gauge_fix=apply_multiplicative_gauge_fix,
        )
        display = ast_to_human_readable(expr)
    else:
        raw = str(expr)
        canonical_obj = _canonicalize_string_expr(raw, strip_numeric_constants=strip_numeric_constants)
        display = raw
    return {
        "canonical_key": _serialize_canonical(canonical_obj),
        "display_expr": str(display),
    }


def _coerce_member(candidate: CommitteeMember | Mapping[str, Any], index: int) -> CommitteeMember:
    if isinstance(candidate, CommitteeMember):
        member = candidate
    else:
        row = dict(candidate)
        member = CommitteeMember(
            member_id=str(row.get("member_id", "") or f"member_{int(index)}"),
            symbolic_structure=row.get("symbolic_structure", row.get("expr", row.get("law", None))),
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
    if not member.canonical_key or not member.display_expr:
        info = canonicalize_candidate_law(member.symbolic_structure)
        member = replace(
            member,
            canonical_key=str(member.canonical_key or info["canonical_key"]),
            display_expr=str(member.display_expr or info["display_expr"]),
        )
    return member


def _quality_weight(member: CommitteeMember, *, temperature: float) -> float:
    error = _candidate_error(member)
    if not math.isfinite(error):
        quality = 0.0
    else:
        quality = math.exp(-max(0.0, float(error)) / max(1.0e-9, float(temperature)))
    simplicity = _safe_float(member.simplicity_score, 1.0)
    if not math.isfinite(simplicity) or simplicity <= 0.0:
        simplicity = 1.0
    physics = _safe_float(member.physics_consistency_score, 1.0)
    if not math.isfinite(physics) or physics <= 0.0:
        physics = 1.0e-6
    return float(quality) * float(simplicity) * float(physics)


def build_committee_state(
    candidates: Sequence[CommitteeMember | Mapping[str, Any]],
    *,
    max_members: int | None = None,
    deduplicate: bool = True,
    weight_temperature: float = 1.0,
) -> CommitteeState:
    members = [_coerce_member(candidate, idx) for idx, candidate in enumerate(list(candidates or []))]
    discarded: list[str] = []
    if deduplicate:
        by_key: dict[str, CommitteeMember] = {}
        for member in members:
            incumbent = by_key.get(member.canonical_key, None)
            if incumbent is None:
                by_key[member.canonical_key] = member
                continue
            keep_current = (
                _candidate_error(member),
                -_safe_float(member.physics_consistency_score, 0.0),
                str(member.member_id),
            ) < (
                _candidate_error(incumbent),
                -_safe_float(incumbent.physics_consistency_score, 0.0),
                str(incumbent.member_id),
            )
            if keep_current:
                discarded.append(str(incumbent.member_id))
                by_key[member.canonical_key] = member
            else:
                discarded.append(str(member.member_id))
        members = list(by_key.values())
    members.sort(
        key=lambda member: (
            _candidate_error(member),
            -_safe_float(member.physics_consistency_score, 0.0),
            str(member.member_id),
        )
    )
    if max_members is not None:
        limit = max(0, int(max_members))
        if len(members) > limit:
            discarded.extend(str(member.member_id) for member in members[limit:])
            members = members[:limit]
    raw_weights = [_quality_weight(member, temperature=float(weight_temperature)) for member in members]
    weight_sum = float(sum(raw_weights))
    if weight_sum <= 0.0 and members:
        raw_weights = [1.0 for _ in members]
        weight_sum = float(len(members))
    normalized = [
        replace(member, committee_weight=float(raw_weight) / max(1.0e-12, weight_sum))
        for member, raw_weight in zip(members, raw_weights)
    ]
    return CommitteeState(
        members=tuple(normalized),
        canonical_member_count=int(len(normalized)),
        discarded_member_ids=tuple(str(member_id) for member_id in discarded),
        metadata={
            "deduplicated": bool(deduplicate),
            "weight_temperature": float(weight_temperature),
        },
    )


__all__ = [
    "CommitteeMember",
    "CommitteeState",
    "build_committee_state",
    "canonicalize_candidate_law",
]
