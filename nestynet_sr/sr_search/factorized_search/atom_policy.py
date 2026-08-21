# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Evidence-bearing policy layer for emergent FSS atoms.

The low-level proposal enumerator consumes ``SeedBlock`` objects.  This module
keeps a higher-level view of the same atoms: what roles they can play, which
families they should gently steer, and which small atom relations are worth
protecting during enumeration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .expr_ast import dims_eq, is_valid_node, node_depth, node_dims, node_size, node_str, simplify
from .proposal_families.common import dim0, dim_add
from .proposal_families.seed_blocks import SeedBlock, make_seed_block


def _jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonish(v) for k, v in dict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_jsonish(v) for v in value]
    if isinstance(value, set):
        return sorted(str(v) for v in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _valid_node(node: Any) -> tuple | None:
    if isinstance(node, tuple) and is_valid_node(node):
        return node
    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _root_op(node: Any) -> str:
    if isinstance(node, tuple) and node:
        return str(node[0])
    return ""


def _active_vars(node: Any) -> tuple[int, ...]:
    seen: set[int] = set()

    def visit(cur: Any) -> None:
        if not isinstance(cur, tuple) or not cur:
            return
        if str(cur[0]) == "var":
            try:
                seen.add(int(cur[1]))
            except Exception:
                pass
            return
        for child in cur[1:]:
            visit(child)

    visit(node)
    return tuple(sorted(seen))


def _node_dim(node: tuple, var_dims: Sequence[Sequence[float]] | None) -> Any:
    if var_dims is None:
        return None
    try:
        return node_dims(node, var_dims)
    except Exception:
        return None


def _dims_match(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    try:
        return bool(dims_eq(left, right))
    except Exception:
        return False


def _dim_confidence(dim: Any) -> float:
    return 1.0 if dim is not None else 0.35


def _domain_tags_for_node(node: tuple) -> tuple[str, ...]:
    op = _root_op(node)
    if op == "const":
        value = _safe_float(node[1], 0.0) if len(node) > 1 else 0.0
        if value > 0.0:
            return ("nonnegative_output", "positive_output")
        if value == 0.0:
            return ("nonnegative_output",)
        return ()
    if op == "exp":
        return ("nonnegative_output", "positive_output")
    if op in {"sqrt", "sqr"}:
        return ("nonnegative_output",)
    return ()


def _source_rows(evidence: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(v) for v in tuple(evidence.get("sources", ()) or ()) if str(v))


def _parent_rows(evidence: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(v) for v in tuple(evidence.get("parent_exprs", ()) or ()) if str(v))


def _utility_from_evidence(evidence: Mapping[str, Any]) -> tuple[float, float, float, float]:
    fit_gain = _safe_float(evidence.get("best_fit_gain_rel", 0.0), 0.0)
    probe_gain = _safe_float(evidence.get("best_probe_gain_rel", 0.0), 0.0)
    marginal = max(0.0, probe_gain, 0.5 * fit_gain)
    parent_probe = _safe_float(evidence.get("best_parent_probe", math.inf), math.inf)
    parent_signal = 0.0 if not math.isfinite(parent_probe) else 1.0 / (1.0 + max(0.0, parent_probe))
    conditional = max(marginal, 0.35 * parent_signal)
    if fit_gain > 0.0 and probe_gain > 0.0:
        stability = min(fit_gain, probe_gain) / max(fit_gain, probe_gain, 1.0e-300)
    elif probe_gain > 0.0:
        stability = 0.5
    else:
        stability = 0.0
    return float(marginal), float(marginal), float(conditional), float(stability)


def _artifact_penalty(*, roles: Sequence[str], families: Sequence[str], evidence: Mapping[str, Any]) -> float:
    role_text = " ".join(str(v).lower() for v in tuple(roles or ()))
    family_text = " ".join(str(v).lower() for v in tuple(families or ()))
    penalty = 0.0
    if bool(evidence.get("rational_derived", False)):
        penalty += 0.25
    if "rational" in family_text and "denominator" in role_text:
        penalty += 0.9
    if "denominator" in role_text and "numerator" not in role_text and "expr" not in role_text:
        penalty += 0.6
    if bool(evidence.get("common_denominator_stripped", False)):
        penalty = max(0.0, penalty - 0.25)
    if _safe_float(evidence.get("best_probe_gain_rel", 0.0), 0.0) <= 0.0 and "denominator" in role_text:
        penalty += 0.4
    return float(penalty)


@dataclass(frozen=True)
class AtomRecord:
    key: str
    node: tuple
    dim: Any = None
    dim_confidence: float = 0.0
    domain_tags: tuple[str, ...] = ()
    active_vars: tuple[int, ...] = ()
    root_op: str = ""
    size: int = 1
    depth: int = 1
    role_scores: Mapping[str, float] = field(default_factory=dict)
    family_scores: Mapping[str, float] = field(default_factory=dict)
    slot_scores: Mapping[str, float] = field(default_factory=dict)
    marginal_target_gain: float = 0.0
    marginal_residual_gain: float = 0.0
    conditional_gain: float = 0.0
    source_support: float = 0.0
    source_diversity: float = 0.0
    stability: float = 0.0
    artifact_penalty: float = 0.0
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def policy_utility(self) -> float:
        utility = (
            0.6 * float(self.marginal_residual_gain)
            + 0.8 * float(self.conditional_gain)
            + 0.4 * float(self.source_diversity)
            + 0.3 * float(self.stability)
            - float(self.artifact_penalty)
            - 0.02 * max(1, int(self.size))
        )
        return max(0.0, float(utility))

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": str(self.key),
            "expr": str(node_str(self.node)),
            "dim": _jsonish(self.dim),
            "dim_confidence": float(self.dim_confidence),
            "domain_tags": [str(v) for v in tuple(self.domain_tags or ())],
            "active_vars": [int(v) for v in tuple(self.active_vars or ())],
            "root_op": str(self.root_op),
            "size": int(self.size),
            "depth": int(self.depth),
            "role_scores": {str(k): float(v) for k, v in dict(self.role_scores or {}).items()},
            "family_scores": {str(k): float(v) for k, v in dict(self.family_scores or {}).items()},
            "slot_scores": {str(k): float(v) for k, v in dict(self.slot_scores or {}).items()},
            "marginal_target_gain": float(self.marginal_target_gain),
            "marginal_residual_gain": float(self.marginal_residual_gain),
            "conditional_gain": float(self.conditional_gain),
            "source_support": float(self.source_support),
            "source_diversity": float(self.source_diversity),
            "stability": float(self.stability),
            "artifact_penalty": float(self.artifact_penalty),
            "policy_utility": float(self.policy_utility()),
            "evidence": _jsonish(self.evidence),
        }


@dataclass(frozen=True)
class AtomRelation:
    left_key: str
    right_key: str
    relation_kind: str
    node: tuple
    dim: Any = None
    dim_confidence: float = 0.0
    role_scores: Mapping[str, float] = field(default_factory=dict)
    family_scores: Mapping[str, float] = field(default_factory=dict)
    slot_scores: Mapping[str, float] = field(default_factory=dict)
    conditional_gain: float = 0.0
    source_support: float = 0.0
    stability: float = 0.0
    artifact_penalty: float = 0.0
    compatible_families: Mapping[str, float] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def policy_utility(self) -> float:
        try:
            size = max(1, int(node_size(self.node)))
        except Exception:
            size = 8
        utility = (
            0.9 * float(self.conditional_gain)
            + 0.45 * float(self.source_support)
            + 0.35 * float(self.stability)
            - float(self.artifact_penalty)
            - 0.015 * size
        )
        return max(0.0, float(utility))

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_key": str(self.left_key),
            "right_key": str(self.right_key),
            "relation_kind": str(self.relation_kind),
            "expr": str(node_str(self.node)),
            "dim": _jsonish(self.dim),
            "dim_confidence": float(self.dim_confidence),
            "role_scores": {str(k): float(v) for k, v in dict(self.role_scores or {}).items()},
            "family_scores": {str(k): float(v) for k, v in dict(self.family_scores or {}).items()},
            "slot_scores": {str(k): float(v) for k, v in dict(self.slot_scores or {}).items()},
            "conditional_gain": float(self.conditional_gain),
            "source_support": float(self.source_support),
            "stability": float(self.stability),
            "artifact_penalty": float(self.artifact_penalty),
            "compatible_families": {
                str(k): float(v) for k, v in dict(self.compatible_families or {}).items()
            },
            "policy_utility": float(self.policy_utility()),
            "evidence": _jsonish(self.evidence),
        }


@dataclass(frozen=True)
class AtomLibrary:
    records: tuple[AtomRecord, ...] = ()
    relations: tuple[AtomRelation, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [record.to_dict() for record in tuple(self.records or ())],
            "relations": [relation.to_dict() for relation in tuple(self.relations or ())],
            "metadata": _jsonish(self.metadata),
        }

    def is_empty(self) -> bool:
        return not tuple(self.records or ()) and not tuple(self.relations or ())

    def record_by_key(self) -> dict[str, AtomRecord]:
        return {str(record.key): record for record in tuple(self.records or ())}


def coerce_atom_library(value: Any) -> AtomLibrary | None:
    if isinstance(value, AtomLibrary):
        return value
    return None


def _record_sort_key(record: AtomRecord) -> tuple[float, float, int, str]:
    return (-float(record.policy_utility()), float(record.artifact_penalty), int(record.size), str(record.key))


def _relation_sort_key(relation: AtomRelation) -> tuple[float, float, int, str]:
    try:
        size = int(node_size(relation.node))
    except Exception:
        size = 99
    return (-float(relation.policy_utility()), float(relation.artifact_penalty), int(size), str(node_str(relation.node)))


def _role_family_slot_scores(
    *,
    node: tuple,
    dim: Any,
    kind: str,
    y_dims: Any,
    dim0_value: Any,
    roles: Sequence[str],
    families: Sequence[str],
    evidence: Mapping[str, Any],
    derived_role: str | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    root = _root_op(node)
    kind_token = str(kind or "").strip()
    role_text = " ".join(str(v).lower() for v in tuple(roles or ()))
    role_scores: dict[str, float] = {}
    family_scores: dict[str, float] = {}
    slot_scores: dict[str, float] = {}
    target_like = _dims_match(dim, y_dims)
    dimless = _dims_match(dim, dim0_value)

    if target_like or kind_token in {"target_term", "carrier"}:
        role_scores["affine_term"] = 1.0
        slot_scores["affine.terms"] = 1.2
        slot_scores["affine:latent.terms"] = 1.2
        family_scores["affine"] = 1.0
    if kind_token == "target_term":
        role_scores["target_term"] = 1.0
        slot_scores["target_term"] = 1.0
    if kind_token == "carrier" or (target_like and root in {"sqrt", "mul", "sqr"}):
        role_scores["envelope"] = 0.95
        role_scores["prefactor"] = 0.85
        slot_scores["periodic.envelope"] = 1.15
        slot_scores["periodic.anchor"] = 1.0
        slot_scores["power.anchor"] = 0.6
        family_scores["periodic"] = max(family_scores.get("periodic", 0.0), 0.85)
        family_scores["power"] = max(family_scores.get("power", 0.0), 0.35)
    if dimless:
        role_scores["dimensionless_feature"] = 0.8
        if root in {"sin", "cos"}:
            role_scores["completed_modulator"] = 1.2
            slot_scores["affine.modulator"] = 0.8
            slot_scores["periodic.completed_modulator"] = 0.9
            family_scores["affine"] = max(family_scores.get("affine", 0.0), 0.45)
            family_scores["periodic"] = max(family_scores.get("periodic", 0.0), 0.75)
        else:
            role_scores["carrier_argument"] = 1.0
            slot_scores["periodic.carrier"] = 1.2
            slot_scores["periodic:sin_base.carrier"] = 1.0
            slot_scores["periodic:cos_base.carrier"] = 1.0
            slot_scores["periodic:sin_mul.carrier"] = 1.0
            slot_scores["periodic:cos_mul.carrier"] = 1.0
            family_scores["periodic"] = max(family_scores.get("periodic", 0.0), 1.0)
    if "numerator" in role_text:
        role_scores["numerator"] = 0.75
        slot_scores["rational.numerator"] = 0.7
        family_scores["rational"] = max(family_scores.get("rational", 0.0), 0.35)
    if "denominator" in role_text:
        role_scores["denominator"] = 0.35
        slot_scores["rational.denominator"] = 0.2
        family_scores["rational"] = max(family_scores.get("rational", 0.0), 0.12)
    if bool(evidence.get("common_denominator_stripped", False)):
        role_scores["affine_term"] = max(role_scores.get("affine_term", 0.0), 1.05)
        family_scores["affine"] = max(family_scores.get("affine", 0.0), 1.05)
    if derived_role == "carrier_argument":
        role_scores["carrier_argument"] = 1.15
        slot_scores["periodic.carrier"] = 1.3
        family_scores["periodic"] = max(family_scores.get("periodic", 0.0), 1.1)
    return role_scores, family_scores, slot_scores


def _record_from_atom(
    atom: Any,
    *,
    var_dims: Sequence[Sequence[float]] | None,
    y_dims: Any,
    dim0_value: Any,
    derived_role: str | None = None,
    derived_evidence: Mapping[str, Any] | None = None,
) -> AtomRecord | None:
    node = _valid_node(getattr(atom, "node", None))
    if node is None:
        return None
    try:
        node = simplify(node)
    except Exception:
        return None
    if not isinstance(node, tuple) or not is_valid_node(node):
        return None
    dim = getattr(atom, "dim", None)
    if dim is None:
        dim = _node_dim(node, var_dims)
    evidence = dict(getattr(atom, "evidence", {}) or {})
    if isinstance(derived_evidence, Mapping):
        evidence.update(dict(derived_evidence))
    roles = tuple(str(v) for v in tuple(getattr(atom, "roles", ()) or ()))
    families = tuple(str(v) for v in tuple(getattr(atom, "families", ()) or ()))
    kind = str(getattr(atom, "kind", "") or "")
    root = _root_op(node)
    role_scores, family_scores, slot_scores = _role_family_slot_scores(
        node=node,
        dim=dim,
        kind=kind,
        y_dims=y_dims,
        dim0_value=dim0_value,
        roles=roles,
        families=families,
        evidence=evidence,
        derived_role=derived_role,
    )
    marginal_target, marginal_residual, conditional, stability = _utility_from_evidence(evidence)
    sources = _source_rows(evidence)
    parents = _parent_rows(evidence)
    source_support = math.log1p(max(0, len(set(sources))))
    source_diversity = math.log1p(max(0, len(set(families)))) + 0.4 * math.log1p(max(0, len(set(parents))))
    penalty = _artifact_penalty(roles=roles, families=families, evidence=evidence)
    if derived_role:
        penalty = max(0.0, penalty - 0.15)
    try:
        size = int(node_size(node))
    except Exception:
        size = 99
    try:
        depth = int(node_depth(node))
    except Exception:
        depth = 99
    return AtomRecord(
        key=str(node_str(node)),
        node=node,
        dim=dim,
        dim_confidence=_dim_confidence(dim),
        domain_tags=_domain_tags_for_node(node),
        active_vars=_active_vars(node),
        root_op=root,
        size=size,
        depth=depth,
        role_scores=role_scores,
        family_scores=family_scores,
        slot_scores=slot_scores,
        marginal_target_gain=marginal_target,
        marginal_residual_gain=marginal_residual,
        conditional_gain=conditional,
        source_support=source_support,
        source_diversity=source_diversity,
        stability=stability,
        artifact_penalty=penalty,
        evidence=evidence,
    )


def _merge_record(left: AtomRecord, right: AtomRecord) -> AtomRecord:
    role_scores = dict(left.role_scores or {})
    for key, value in dict(right.role_scores or {}).items():
        role_scores[str(key)] = max(float(role_scores.get(str(key), 0.0)), float(value))
    family_scores = dict(left.family_scores or {})
    for key, value in dict(right.family_scores or {}).items():
        family_scores[str(key)] = max(float(family_scores.get(str(key), 0.0)), float(value))
    slot_scores = dict(left.slot_scores or {})
    for key, value in dict(right.slot_scores or {}).items():
        slot_scores[str(key)] = max(float(slot_scores.get(str(key), 0.0)), float(value))
    evidence = dict(left.evidence or {})
    evidence.update(dict(right.evidence or {}))
    return AtomRecord(
        key=left.key,
        node=left.node,
        dim=left.dim if left.dim is not None else right.dim,
        dim_confidence=max(float(left.dim_confidence), float(right.dim_confidence)),
        domain_tags=tuple(sorted(set(left.domain_tags).union(str(v) for v in right.domain_tags))),
        active_vars=tuple(sorted(set(left.active_vars).union(int(v) for v in right.active_vars))),
        root_op=left.root_op or right.root_op,
        size=min(int(left.size), int(right.size)),
        depth=min(int(left.depth), int(right.depth)),
        role_scores=role_scores,
        family_scores=family_scores,
        slot_scores=slot_scores,
        marginal_target_gain=max(float(left.marginal_target_gain), float(right.marginal_target_gain)),
        marginal_residual_gain=max(float(left.marginal_residual_gain), float(right.marginal_residual_gain)),
        conditional_gain=max(float(left.conditional_gain), float(right.conditional_gain)),
        source_support=max(float(left.source_support), float(right.source_support)),
        source_diversity=max(float(left.source_diversity), float(right.source_diversity)),
        stability=max(float(left.stability), float(right.stability)),
        artifact_penalty=min(float(left.artifact_penalty), float(right.artifact_penalty)),
        evidence=evidence,
    )


def _derived_wrapper_argument_records(
    atom: Any,
    *,
    var_dims: Sequence[Sequence[float]] | None,
    y_dims: Any,
    dim0_value: Any,
) -> list[AtomRecord]:
    node = _valid_node(getattr(atom, "node", None))
    if node is None or _root_op(node) not in {"sin", "cos"} or len(node) < 2:
        return []
    arg = _valid_node(node[1])
    if arg is None:
        return []
    class _Derived:
        pass

    derived = _Derived()
    derived.node = arg
    derived.dim = _node_dim(arg, var_dims)
    derived.kind = "dimensionless_feature"
    derived.roles = ("derived:wrapper_argument",)
    derived.families = ("periodic",)
    derived.evidence = dict(getattr(atom, "evidence", {}) or {})
    derived.evidence.update(
        {
            "derived_from": str(node_str(node)),
            "wrapper_op": _root_op(node),
            "source_role": "carrier_argument",
        }
    )
    record = _record_from_atom(
        derived,
        var_dims=var_dims,
        y_dims=y_dims,
        dim0_value=dim0_value,
        derived_role="carrier_argument",
        derived_evidence={"derived_wrapper_argument": True},
    )
    return [record] if isinstance(record, AtomRecord) else []


def _compatible_product_dim(left: AtomRecord, right: AtomRecord, *, target_dim: Any) -> tuple[Any, float] | None:
    dim = dim_add(left.dim, right.dim)
    if target_dim is not None:
        if dim is None:
            return (None, 0.35)
        if not _dims_match(dim, target_dim):
            return None
    return (dim, min(float(left.dim_confidence), float(right.dim_confidence)))


def _records_can_product(left: AtomRecord, right: AtomRecord, *, target_dim: Any, dim0_value: Any) -> bool:
    left_roles = dict(left.role_scores or {})
    right_roles = dict(right.role_scores or {})
    left_target = "affine_term" in left_roles or "envelope" in left_roles
    right_target = "affine_term" in right_roles or "envelope" in right_roles
    left_mod = "completed_modulator" in left_roles
    right_mod = "completed_modulator" in right_roles
    if not ((left_target and right_mod) or (right_target and left_mod)):
        return False
    return _compatible_product_dim(left, right, target_dim=target_dim) is not None


def _build_relations(
    records: Sequence[AtomRecord],
    *,
    target_dim: Any,
    dim0_value: Any,
    max_relations: int,
) -> tuple[AtomRelation, ...]:
    out: list[AtomRelation] = []
    seen: set[str] = set()
    rows = sorted(tuple(records or ()), key=_record_sort_key)
    for i, left in enumerate(rows):
        for right in rows[i + 1 :]:
            if len(out) >= max(0, int(max_relations)) * 3:
                break
            if not _records_can_product(left, right, target_dim=target_dim, dim0_value=dim0_value):
                continue
            dim_info = _compatible_product_dim(left, right, target_dim=target_dim)
            if dim_info is None:
                continue
            dim, dim_conf = dim_info
            try:
                node = simplify(("mul", left.node, right.node))
            except Exception:
                continue
            if not isinstance(node, tuple) or not is_valid_node(node):
                continue
            key = str(node_str(node))
            if key in seen:
                continue
            seen.add(key)
            cond = max(
                float(left.conditional_gain),
                float(right.conditional_gain),
                0.5 * (left.policy_utility() + right.policy_utility()),
            )
            left_roles = dict(left.role_scores or {})
            right_roles = dict(right.role_scores or {})
            envelope_modulator = (
                ("completed_modulator" in left_roles and ("envelope" in right_roles or "prefactor" in right_roles))
                or ("completed_modulator" in right_roles and ("envelope" in left_roles or "prefactor" in left_roles))
            )
            if envelope_modulator:
                cond += 0.30
            source_support = max(float(left.source_support), float(right.source_support))
            stability = max(float(left.stability), float(right.stability))
            penalty = 0.5 * (float(left.artifact_penalty) + float(right.artifact_penalty))
            if not envelope_modulator:
                penalty += 0.10
            out.append(
                AtomRelation(
                    left_key=str(left.key),
                    right_key=str(right.key),
                    relation_kind="product",
                    node=node,
                    dim=dim,
                    dim_confidence=float(dim_conf),
                    role_scores={"affine_term": 1.25, "prefactor_modulated": 1.0},
                    family_scores={"affine": 1.35, "periodic": 0.55},
                    slot_scores={"affine.terms": 1.6, "affine:latent.terms": 1.6},
                    conditional_gain=float(cond),
                    source_support=float(source_support),
                    stability=float(stability),
                    artifact_penalty=float(penalty),
                    compatible_families={"affine": 1.0, "periodic": 0.35},
                    evidence={
                        "left": left.to_dict(),
                        "right": right.to_dict(),
                    },
                )
            )
    return tuple(sorted(out, key=_relation_sort_key)[: max(0, int(max_relations))])


def build_atom_library(
    atoms: Sequence[Any] | None,
    *,
    var_dims: Sequence[Sequence[float]] | None = None,
    y_dims: Sequence[float] | None = None,
    max_records: int = 16,
    max_relations: int = 16,
    stats: dict[str, Any] | None = None,
) -> AtomLibrary:
    dim0_value = dim0(var_dims)
    target_dim = tuple(y_dims) if isinstance(y_dims, (list, tuple)) else y_dims
    by_key: dict[str, AtomRecord] = {}
    for atom in tuple(atoms or ()):
        record = _record_from_atom(atom, var_dims=var_dims, y_dims=target_dim, dim0_value=dim0_value)
        if isinstance(record, AtomRecord):
            by_key[record.key] = _merge_record(by_key[record.key], record) if record.key in by_key else record
        for derived in _derived_wrapper_argument_records(atom, var_dims=var_dims, y_dims=target_dim, dim0_value=dim0_value):
            by_key[derived.key] = _merge_record(by_key[derived.key], derived) if derived.key in by_key else derived
    records = tuple(sorted(by_key.values(), key=_record_sort_key)[: max(0, int(max_records))])
    relations = _build_relations(records, target_dim=target_dim, dim0_value=dim0_value, max_relations=max_relations)
    library = AtomLibrary(
        records=records,
        relations=relations,
        metadata={
            "record_count": int(len(records)),
            "relation_count": int(len(relations)),
        },
    )
    if isinstance(stats, dict):
        stats["atom_policy_library_records"] = int(len(records))
        stats["atom_policy_library_relations"] = int(len(relations))
        stats["atom_policy_library"] = library.to_dict()
    return library


def atom_policy_family_scores(
    library: AtomLibrary | None,
    families: Sequence[str] | None,
) -> tuple[dict[str, float], dict[str, Any]]:
    family_tokens = [str(f or "").strip().lower() for f in tuple(families or ()) if str(f or "").strip()]
    scores = {family: 0.0 for family in family_tokens}
    decomposition: dict[str, Any] = {
        family: {"atom_policy": 0.0, "top_atom_reasons": []}
        for family in family_tokens
    }
    lib = coerce_atom_library(library)
    if lib is None or lib.is_empty():
        return scores, decomposition
    for record in tuple(lib.records or ()):
        utility = record.policy_utility()
        if utility <= 0.0:
            continue
        for family in family_tokens:
            compat = _safe_float(dict(record.family_scores or {}).get(family, 0.0), 0.0)
            if compat <= 0.0:
                continue
            contribution = min(2.0, compat * utility)
            if contribution <= 0.0:
                continue
            scores[family] += float(contribution)
            reasons = decomposition[family]["top_atom_reasons"]
            if len(reasons) < 5:
                reasons.append(
                    {
                        "expr": str(node_str(record.node)),
                        "kind": "atom",
                        "contribution": float(contribution),
                        "roles": sorted(str(k) for k, v in dict(record.role_scores or {}).items() if float(v) > 0.0),
                    }
                )
    for relation in tuple(lib.relations or ()):
        utility = relation.policy_utility()
        if utility <= 0.0:
            continue
        for family in family_tokens:
            compat = _safe_float(dict(relation.family_scores or {}).get(family, 0.0), 0.0)
            if compat <= 0.0:
                continue
            contribution = min(2.0, compat * utility)
            if contribution <= 0.0:
                continue
            scores[family] += float(contribution)
            reasons = decomposition[family]["top_atom_reasons"]
            if len(reasons) < 5:
                reasons.append(
                    {
                        "expr": str(node_str(relation.node)),
                        "kind": str(relation.relation_kind),
                        "contribution": float(contribution),
                    }
                )
    for family, value in scores.items():
        decomposition[family]["atom_policy"] = float(value)
    return scores, decomposition


def seed_blocks_from_atom_relations(
    library: AtomLibrary | None,
    *,
    var_dims: Sequence[Sequence[float]] | None = None,
    required_dim: Any = None,
    limit: int = 8,
) -> tuple[SeedBlock, ...]:
    lib = coerce_atom_library(library)
    if lib is None:
        return ()
    out: list[SeedBlock] = []
    seen: set[str] = set()
    for relation in sorted(tuple(lib.relations or ()), key=_relation_sort_key):
        if len(out) >= max(0, int(limit)):
            break
        if required_dim is not None and relation.dim is not None and not _dims_match(relation.dim, required_dim):
            continue
        node = _valid_node(relation.node)
        if node is None:
            continue
        key = str(node_str(node))
        if key in seen:
            continue
        seen.add(key)
        dim = relation.dim
        if dim is None and var_dims is not None:
            dim = _node_dim(node, var_dims)
        out.append(
            make_seed_block(
                node,
                dim=dim,
                source=f"aux:policy:{str(relation.relation_kind)}",
                builder=str(relation.relation_kind or "product"),
                metadata={
                    "origin": "aux:policy",
                    "atom_policy_relation": relation.to_dict(),
                    "policy_slot_scores": dict(relation.slot_scores or {}),
                    "policy_family_scores": dict(relation.family_scores or {}),
                    "policy_role_scores": dict(relation.role_scores or {}),
                    "policy_score": float(relation.policy_utility()),
                },
            )
        )
    return tuple(out)


def enrich_seed_block_from_library(block: SeedBlock, library: AtomLibrary | None) -> SeedBlock:
    lib = coerce_atom_library(library)
    if not isinstance(block, SeedBlock) or lib is None:
        return block
    key = str(node_str(block.node))
    record = lib.record_by_key().get(key)
    if not isinstance(record, AtomRecord):
        return block
    metadata = dict(block.metadata or {})
    metadata.setdefault("atom_policy_record", record.to_dict())
    metadata.setdefault("policy_slot_scores", dict(record.slot_scores or {}))
    metadata.setdefault("policy_family_scores", dict(record.family_scores or {}))
    metadata.setdefault("policy_role_scores", dict(record.role_scores or {}))
    metadata.setdefault("policy_score", float(record.policy_utility()))
    return SeedBlock(
        node=block.node,
        dim=block.dim,
        source=block.source,
        builder=block.builder,
        active_vars=tuple(block.active_vars or ()),
        domain_tags=tuple(block.domain_tags or ()),
        metadata=metadata,
    )


def build_aux_policy_plan(
    *,
    families: Sequence[str] | None,
    library: AtomLibrary | None,
    max_scaffolds: int,
    anchors_per_family: int,
) -> list[dict[str, Any]]:
    total = max(0, int(max_scaffolds))
    if total <= 0:
        return []
    family_tokens = [str(f or "").strip().lower() for f in tuple(families or ()) if str(f or "").strip()]
    lib = coerce_atom_library(library)
    if lib is not None and tuple(lib.relations or ()):
        scores, _decomp = atom_policy_family_scores(lib, set(family_tokens).union({"affine"}))
        return [
            {
                "family": "affine",
                "max_scaffolds": int(total),
                "anchors_per_family": max(1, int(anchors_per_family)),
                "priority_score": float(max(0.25, scores.get("affine", 0.0))),
                "reason": "atom_policy_affine_relation_sweep",
            }
        ]
    plan_families: list[tuple[str, float, str]] = []
    scores, _decomp = atom_policy_family_scores(lib, set(family_tokens).union({"affine"}))
    if scores.get("affine", 0.0) > 0.0 or (lib is not None and tuple(lib.relations or ())):
        plan_families.append(("affine", max(0.25, float(scores.get("affine", 0.0))), "atom_policy_affine"))
    for family in family_tokens:
        if family == "affine":
            continue
        score = float(scores.get(family, 0.0))
        if score <= 0.0:
            continue
        if family == "rational":
            score *= 0.35
        plan_families.append((family, score, "atom_policy_family"))
    if not plan_families:
        plan_families.append(("affine", 0.25, "aux_fallback_affine"))
    plan_families.sort(key=lambda item: (-float(item[1]), str(item[0])))
    selected = plan_families[: max(1, min(len(plan_families), total))]
    budgets = {family: 1 for family, _score, _reason in selected}
    remaining = max(0, total - len(selected))
    weights = {family: max(0.1, float(score)) for family, score, _reason in selected}
    while remaining > 0 and selected:
        family = max(selected, key=lambda item: (weights.get(item[0], 0.0) / max(1, budgets[item[0]]), item[0]))[0]
        budgets[family] += 1
        remaining -= 1
    return [
        {
            "family": family,
            "max_scaffolds": int(budgets.get(family, 1)),
            "anchors_per_family": max(1, int(anchors_per_family)),
            "priority_score": float(score),
            "reason": reason,
        }
        for family, score, reason in selected
    ]


__all__ = [
    "AtomLibrary",
    "AtomRecord",
    "AtomRelation",
    "atom_policy_family_scores",
    "build_atom_library",
    "build_aux_policy_plan",
    "coerce_atom_library",
    "enrich_seed_block_from_library",
    "seed_blocks_from_atom_relations",
]
