# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import itertools
import math
import os
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Collection, Mapping, Sequence

import torch

from nestynet_sr.sr_core.carrier_units import (
    CarrierUnitResult,
    context_from_metadata,
    is_certified_inner_coordinate,
    precheck_carrier_units,
    validate_outer_map_units,
)

from ..atom_policy import build_atom_library
from ..basis_scoring import build_scaffold_candidate_score_cfg, record_anchor_head_compare, score_bound_closure
from ..basis_head import fit_basis_state_head, score_basis_state_conditional_gain
from ..basis_state import (
    BasisState,
    FeatureBlock,
    ProposalCandidate,
    ProposalContext,
    admit_basis_state_to_beam,
    basis_state_covers_feature_block,
    basis_state_from_additive_transition,
    basis_state_retarget,
    prepare_basis_state_candidate,
)
from ..closures import BoundClosure, ClosureDesign, bound_closure_identity_key, make_direct_affine_closure
from ..emergent_atoms import (
    EmergentAtom,
    harvest_emergent_atoms,
    merge_emergent_atom_registry,
    seed_blocks_from_emergent_atoms,
)
from ..emergent_basis import propose_emergent_basis_rows
from ..expr_ast import dims_eq, eval_node, is_valid_node, node_dims, node_size, node_str, simplify
from ..proposal_families.closure_eval import make_direct_preview_row
from ..proposal_families.steering import allocate_family_budgets


@dataclass
class ProposalScoringState:
    n_evaluated: int
    best_raw_mse: float
    best_raw_mse_struct: float
    best_mse: float


def _record_carrier_unit_result(
    stats: dict[str, Any],
    result: CarrierUnitResult,
    *,
    context,
) -> dict[str, Any]:
    """Record one bounded, report-ready carrier unit decision."""

    diagnostic = str(result.diagnostic)
    stats[diagnostic] = int(stats.get(diagnostic, 0) or 0) + 1
    metadata = result.to_metadata(context=context)
    events = stats.setdefault("unit_handoff_events", [])
    if isinstance(events, list) and len(events) < 32:
        events.append(dict(metadata))
    return metadata


@dataclass(frozen=True)
class _SpanAtom:
    node: tuple
    dim: Any
    kind: str
    score: float
    evidence: Mapping[str, Any]
    source_count: int = 1
    roles: tuple[str, ...] = ()
    families: tuple[str, ...] = ()


@dataclass(frozen=True)
class PairEntry:
    sort_key: Any
    idx: int
    row: Mapping[str, Any]
    proposal_key: str
    family: str
    spec_key: str
    interaction_key: str
    support_key: str
    anchor_key: str
    source_pool: str
    not_exact: bool
    child_size: int

    def to_record(self) -> dict[str, Any]:
        return {
            "sort_key": self.sort_key,
            "idx": int(self.idx),
            "row": self.row,
            "proposal_key": str(self.proposal_key),
            "family": str(self.family),
            "spec_key": str(self.spec_key),
            "interaction_key": str(self.interaction_key),
            "support_key": str(self.support_key),
            "anchor_key": str(self.anchor_key),
            "source_pool": str(self.source_pool),
            "not_exact": bool(self.not_exact),
            "child_size": max(0, int(self.child_size)),
        }


@dataclass(frozen=True)
class PairProfile:
    name: str
    stats_prefix: str
    pair_key_prefix: str
    proposal_family: str
    enabled: bool
    topk: int
    max_pairs: int
    debug_pool_stage: str
    debug_attempt_stage: str
    require_interaction_support: bool
    use_pool_selection: bool
    source_pool_mode: str
    template_kinds: tuple[str, ...]
    route_prepare_a: str
    route_prepare_b: str
    route_score: str
    route_refit: str


@dataclass
class RoundCommitObject:
    kind: str
    profile: str
    proposal_key: str
    basis_state: BasisState | None
    eff_mse: float
    accepted: bool
    accept_reason: str
    complexity: float
    source_pool: str = ""
    priority_rank: int = 1_000_000_000
    candidate_meta: Mapping[str, Any] | None = None
    accept_meta: Mapping[str, Any] | None = None
    pair_template: str = ""
    relation_tags: tuple[str, ...] = ()
    source_members: tuple[str, ...] = ()
    gain_vs_best_singleton: float = math.nan
    best_singleton_key: str = ""
    best_singleton_eff_mse: float = math.inf
    best_singleton_source: str = ""
    pair_entries: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None


_OUTER_SCAFFOLD_FASTTRACK_MSE = 1.0e-8
_ROUND_COMMIT_EFF_MSE_REL_SLACK = 2.0e-2
_ROUND_COMMIT_EFF_MSE_ABS_SLACK = 1.0e-6
_PAIR_RESCUE_WEAK_GAIN_REL_SLACK = 2.0e-2
_PAIR_RESCUE_WEAK_GAIN_ABS_SLACK = 1.0e-6
_PAIR_RESCUE_CLOSE_COMMIT_REL_SLACK = 2.0e-2
_PAIR_RESCUE_CLOSE_COMMIT_ABS_SLACK = 1.0e-6


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(str(name))
    if raw is None:
        return bool(default)
    token = str(raw).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(str(name))
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _finite_float(value: Any, default: float = math.inf) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _atom_dim(atom: EmergentAtom, var_dims) -> Any:
    dim = getattr(atom, "dim", None)
    if dim is not None:
        return dim
    node = getattr(atom, "node", None)
    if var_dims is None or not (isinstance(node, tuple) and is_valid_node(node)):
        return None
    try:
        return node_dims(node, var_dims)
    except Exception:
        return None


def _dim_is_dimensionless(dim: Any) -> bool:
    if dim is None:
        return False
    try:
        return all(abs(float(v)) <= 1.0e-12 for v in tuple(dim))
    except Exception:
        return False


def _atom_rank(atom: EmergentAtom) -> tuple[int, float, int, int, str]:
    kind = str(getattr(atom, "kind", "") or "").strip().lower()
    roles = {str(v).strip().lower() for v in tuple(getattr(atom, "roles", ()) or ())}
    if kind == "target_term" or "expr" in roles or "target_term" in roles:
        kind_rank = 0
    elif kind == "carrier":
        kind_rank = 1
    elif kind == "dimensionless_feature":
        kind_rank = 2
    else:
        kind_rank = 3
    node = getattr(atom, "node", None)
    try:
        size = int(node_size(node))
    except Exception:
        size = 99
    return (
        int(kind_rank),
        -_finite_float(getattr(atom, "score", 0.0), 0.0),
        int(size),
        -int(getattr(atom, "source_count", 0) or 0),
        str(node_str(node)) if isinstance(node, tuple) else "",
    )


def _append_unique_node(
    rows: list[tuple[tuple, str, float]],
    seen: set[str],
    node: Any,
    *,
    source: str,
    rank_score: float,
) -> None:
    if not (isinstance(node, tuple) and is_valid_node(node)):
        return
    simp = simplify(node)
    if not (isinstance(simp, tuple) and is_valid_node(simp)):
        return
    key = str(node_str(simp))
    if key in seen:
        return
    seen.add(key)
    rows.append((simp, str(source), float(rank_score)))


def _propose_atomized_linear_span_rows(
    *,
    atoms: Collection[EmergentAtom],
    atom_origin_by_key: Mapping[str, str] | None = None,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    var_dims,
    y_dims,
    max_depth: int,
    max_rows: int,
    stats: dict[str, Any] | None = None,
    debug_limit: int = 0,
) -> list[dict[str, Any]]:
    """Fit tiny linear heads over unit-compatible emergent atom products."""

    max_rows_i = max(0, int(max_rows))
    if max_rows_i <= 0 or y_dims is None:
        return []
    atom_list: list[Any] = [
        atom
        for atom in tuple(atoms or ())
        if isinstance(atom, (EmergentAtom, _SpanAtom))
        and isinstance(getattr(atom, "node", None), tuple)
        and is_valid_node(getattr(atom, "node", None))
    ]
    if isinstance(stats, dict):
        stats["atomized_linear_span_calls"] = int(stats.get("atomized_linear_span_calls", 0) or 0) + 1
    origin_by_key = {str(k): str(v) for k, v in dict(atom_origin_by_key or {}).items()}

    structural_seed_atoms = 0
    structural_seed_target_atoms = 0
    structural_seed_dimless_atoms = 0
    structural_seed_budget = max(0, _env_int("NESTY_ATOMIZED_STRUCTURAL_ATOM_BUDGET", 64))
    structural_seed_mode = str(os.environ.get("NESTY_ATOMIZED_STRUCTURAL_SEED_MODE", "topup") or "topup").strip().lower()
    pre_seed_target_atoms = 0
    pre_seed_dimless_atoms = 0
    if var_dims is not None:
        for atom in tuple(atom_list):
            dim = _atom_dim(atom, var_dims)
            if dim is not None and dims_eq(dim, y_dims):
                pre_seed_target_atoms += 1
            elif _dim_is_dimensionless(dim):
                pre_seed_dimless_atoms += 1
    structural_seed_enable = bool(
        var_dims is not None
        and structural_seed_budget > 0
        and structural_seed_mode not in {"0", "false", "no", "off", "none", "disabled"}
        and (
            structural_seed_mode in {"topup", "top-up", "fallback_topup", "fallback-topup", "always"}
            or pre_seed_target_atoms <= 0
            or pre_seed_dimless_atoms <= 0
        )
    )
    if var_dims is not None and structural_seed_budget > 0 and structural_seed_enable:
        seen_structural = {str(node_str(getattr(atom, "node", None))) for atom in atom_list}
        dimless_unary_bases: list[tuple] = []

        def _record_dimless_unary_base(node: Any) -> None:
            if not (isinstance(node, tuple) and is_valid_node(node)):
                return
            simp = simplify(node)
            try:
                dim = node_dims(simp, var_dims)
            except Exception:
                return
            if _dim_is_dimensionless(dim):
                key = str(node_str(simp))
                if key not in {str(node_str(base)) for base in dimless_unary_bases}:
                    dimless_unary_bases.append(simp)

        def _add_structural_atom(
            node: Any,
            *,
            source: str,
            target_score: float,
            dimless_score: float,
            roles: tuple[str, ...],
        ) -> bool:
            nonlocal structural_seed_atoms, structural_seed_target_atoms, structural_seed_dimless_atoms
            if structural_seed_atoms >= structural_seed_budget:
                return False
            if not (isinstance(node, tuple) and is_valid_node(node)):
                return False
            simp = simplify(node)
            if not (isinstance(simp, tuple) and is_valid_node(simp)):
                return False
            try:
                dim = node_dims(simp, var_dims)
            except Exception:
                return False
            is_target = dims_eq(dim, y_dims)
            is_dimless = _dim_is_dimensionless(dim)
            if not is_target and not is_dimless:
                return False
            key = str(node_str(simp))
            if key in seen_structural:
                return False
            seen_structural.add(key)
            if is_target:
                kind = "target_term"
                atom_roles = tuple(dict.fromkeys(tuple(roles) + ("target_term",)))
                score = float(target_score)
                structural_seed_target_atoms += 1
            else:
                kind = "dimensionless_feature"
                atom_roles = tuple(dict.fromkeys(tuple(roles) + ("dimensionless_feature",)))
                score = float(dimless_score)
                structural_seed_dimless_atoms += 1
                _record_dimless_unary_base(simp)
            atom_list.append(
                _SpanAtom(
                    node=simp,
                    dim=dim,
                    kind=kind,
                    score=score,
                    evidence={"sources": ["atomized_structural_seed", str(source)]},
                    source_count=1,
                    roles=atom_roles,
                    families=("atomized_linear_span",),
                )
            )
            structural_seed_atoms += 1
            return True

        var_count = len(tuple(var_dims or ()))
        for var_idx in range(var_count):
            node = ("var", int(var_idx))
            _add_structural_atom(
                node,
                source="raw_variable",
                target_score=2.0,
                dimless_score=1.5,
                roles=("raw_variable",),
            )
            _record_dimless_unary_base(node)

        dimless_unary_ops_raw = os.environ.get("NESTY_ATOMIZED_STRUCTURAL_DIMLESS_UNARY_OPS", "sin,cos")
        dimless_unary_ops = tuple(
            op
            for op in (part.strip() for part in str(dimless_unary_ops_raw).split(","))
            if op in {"sin", "cos", "exp", "log", "sqrt", "sqr"}
        )

        for base in tuple(dimless_unary_bases):
            for op in dimless_unary_ops:
                _add_structural_atom(
                    (op, base),
                    source=f"dimless_unary:{op}",
                    target_score=1.4,
                    dimless_score=1.35,
                    roles=("structural_dimless_unary",),
                )

        pair_dimless_bases: list[tuple] = []

        def _record_pair_dimless_base(node: Any) -> None:
            if not (isinstance(node, tuple) and is_valid_node(node)):
                return
            simp = simplify(node)
            try:
                dim = node_dims(simp, var_dims)
            except Exception:
                return
            if not _dim_is_dimensionless(dim):
                return
            key = str(node_str(simp))
            if key not in {str(node_str(base)) for base in tuple(dimless_unary_bases) + tuple(pair_dimless_bases)}:
                pair_dimless_bases.append(simp)

        for left_idx, right_idx in itertools.combinations(range(var_count), 2):
            left = ("var", int(left_idx))
            right = ("var", int(right_idx))
            try:
                same_dim = dims_eq(var_dims[left_idx], var_dims[right_idx])
            except Exception:
                same_dim = False
            if same_dim:
                for op, score in (("add", 1.9), ("sub", 1.7)):
                    node = (op, left, right)
                    _add_structural_atom(
                        node,
                        source=f"same_dim_{op}",
                        target_score=score,
                        dimless_score=1.25,
                        roles=("same_dim_composite",),
                    )
                    _record_pair_dimless_base(node)
            for node, source, target_score, dimless_score in (
                (("sqrt", ("mul", left, right)), "half_power_product", 1.95, 1.15),
                (("mul", left, right), "binary_product", 1.55, 1.2),
                (("div", left, right), "binary_ratio", 1.45, 1.2),
                (("div", right, left), "binary_ratio", 1.45, 1.2),
            ):
                _add_structural_atom(
                    node,
                    source=source,
                    target_score=target_score,
                    dimless_score=dimless_score,
                    roles=("binary_composite",),
                )
                _record_pair_dimless_base(node)

        for base in tuple(pair_dimless_bases)[:16]:
            for op in dimless_unary_ops:
                _add_structural_atom(
                    (op, base),
                    source=f"dimless_pair_unary:{op}",
                    target_score=1.2,
                    dimless_score=1.05,
                    roles=("structural_dimless_unary", "binary_composite"),
                )
    if isinstance(stats, dict):
        stats["atomized_linear_span_structural_seed_mode"] = str(structural_seed_mode)
        stats["atomized_linear_span_structural_seed_enabled"] = bool(structural_seed_enable)
        stats["atomized_linear_span_pre_seed_target_atoms"] = int(pre_seed_target_atoms)
        stats["atomized_linear_span_pre_seed_dimless_atoms"] = int(pre_seed_dimless_atoms)
        stats["atomized_linear_span_structural_seed_atoms"] = int(structural_seed_atoms)
        stats["atomized_linear_span_structural_seed_target_atoms"] = int(structural_seed_target_atoms)
        stats["atomized_linear_span_structural_seed_dimless_atoms"] = int(structural_seed_dimless_atoms)
        stats["atomized_linear_span_structural_seed_budget"] = int(structural_seed_budget)

    if not atom_list:
        return []

    def atom_provenance(atom: EmergentAtom) -> dict[str, Any]:
        node = getattr(atom, "node", None)
        key = str(node_str(node)) if isinstance(node, tuple) else str(node)
        evidence = dict(getattr(atom, "evidence", {}) or {})
        return {
            "expr": key,
            "origin": str(origin_by_key.get(key, "unknown")),
            "kind": str(getattr(atom, "kind", "") or ""),
            "source_count": int(getattr(atom, "source_count", 0) or 0),
            "rational_derived": bool(evidence.get("rational_derived", False)),
            "common_denominator_stripped": bool(evidence.get("common_denominator_stripped", False)),
            "sources": [str(v) for v in tuple(evidence.get("sources", ()) or ())],
            "roles": [str(v) for v in tuple(getattr(atom, "roles", ()) or ())],
            "families": [str(v) for v in tuple(getattr(atom, "families", ()) or ())],
        }

    target_atoms: list[EmergentAtom] = []
    dimensionless_atoms: list[EmergentAtom] = []
    seen_atom_keys: set[str] = set()
    for atom in sorted(atom_list, key=_atom_rank):
        node = getattr(atom, "node", None)
        key = str(node_str(node))
        if key in seen_atom_keys:
            continue
        seen_atom_keys.add(key)
        dim = _atom_dim(atom, var_dims)
        if dim is not None and dims_eq(dim, y_dims):
            target_atoms.append(atom)
        elif _dim_is_dimensionless(dim):
            dimensionless_atoms.append(atom)

    base_dimensionless_atoms = tuple(dimensionless_atoms)
    derived_dimless_keys = {
        str(node_str(getattr(atom, "node", None)))
        for atom in tuple(dimensionless_atoms)
        if isinstance(getattr(atom, "node", None), tuple)
    }

    def append_derived_dimensionless(
        node: Any,
        *,
        score: float,
        source: str,
        source_atoms: Sequence[Any],
        roles: tuple[str, ...],
    ) -> None:
        nonlocal dimensionless_atoms
        if not (isinstance(node, tuple) and is_valid_node(node)):
            return
        simp = simplify(node)
        if not (isinstance(simp, tuple) and is_valid_node(simp)):
            return
        try:
            dim = node_dims(simp, var_dims) if var_dims is not None else tuple(0.0 for _ in tuple(y_dims or ()))
        except Exception:
            return
        if not _dim_is_dimensionless(dim):
            return
        key = str(node_str(simp))
        if key in derived_dimless_keys:
            return
        derived_dimless_keys.add(key)
        dimensionless_atoms.append(
            _SpanAtom(
                node=simp,
                dim=dim,
                kind="dimensionless_feature",
                score=float(score),
                evidence={
                    "sources": [str(source)],
                    "atomized_derived_dimensionless": True,
                    "source_atom_exprs": [
                        str(node_str(getattr(atom, "node", None)))
                        for atom in tuple(source_atoms or ())
                        if isinstance(getattr(atom, "node", None), tuple)
                    ],
                },
                source_count=max(1, len(tuple(source_atoms or ()))),
                roles=roles,
                families=("atomized_linear_span",),
            )
        )

    dimless_closure_base = sorted(base_dimensionless_atoms, key=_atom_rank)[:8]
    for atom in dimless_closure_base:
        node = getattr(atom, "node", None)
        if not (isinstance(node, tuple) and is_valid_node(node)):
            continue
        append_derived_dimensionless(
            ("sqr", simplify(node)),
            score=0.9 * _finite_float(getattr(atom, "score", 0.0), 0.0),
            source="atomized_dimless_square",
            source_atoms=(atom,),
            roles=("derived_dimensionless", "square"),
        )
    dimless_closure_pool = sorted(base_dimensionless_atoms, key=_atom_rank)[:12]
    for left, right in itertools.combinations(dimless_closure_pool, 2):
        left_node = getattr(left, "node", None)
        right_node = getattr(right, "node", None)
        if not (isinstance(left_node, tuple) and isinstance(right_node, tuple)):
            continue
        append_derived_dimensionless(
            ("mul", simplify(left_node), simplify(right_node)),
            score=_finite_float(getattr(left, "score", 0.0), 0.0)
            + _finite_float(getattr(right, "score", 0.0), 0.0),
            source="atomized_dimless_product",
            source_atoms=(left, right),
            roles=("derived_dimensionless", "product"),
        )

    if isinstance(stats, dict):
        stats["atomized_linear_span_source_atoms"] = int(len(atom_list))
        stats["atomized_linear_span_target_atoms"] = int(len(target_atoms))
        stats["atomized_linear_span_dimensionless_atoms"] = int(len(dimensionless_atoms))
        stats["atomized_linear_span_derived_dimensionless_atoms"] = int(
            max(0, len(dimensionless_atoms) - len(base_dimensionless_atoms))
        )
    if not target_atoms:
        return []

    target_cap = min(16, max(4, max_rows_i))
    dimless_cap = min(16, max(4, max_rows_i // 2))
    target_atoms = target_atoms[:target_cap]
    base_dimless_keys = {
        str(node_str(getattr(atom, "node", None)))
        for atom in tuple(base_dimensionless_atoms)
        if isinstance(getattr(atom, "node", None), tuple)
    }
    base_dimless_kept = sorted(base_dimensionless_atoms, key=_atom_rank)[:dimless_cap]
    base_kept_keys = {
        str(node_str(getattr(atom, "node", None)))
        for atom in tuple(base_dimless_kept)
        if isinstance(getattr(atom, "node", None), tuple)
    }
    derived_dimless_atoms = [
        atom
        for atom in tuple(dimensionless_atoms)
        if isinstance(getattr(atom, "node", None), tuple)
        and str(node_str(getattr(atom, "node", None))) not in base_dimless_keys
        and str(node_str(getattr(atom, "node", None))) not in base_kept_keys
    ]
    remaining_dimless_cap = max(0, int(dimless_cap) - len(base_dimless_kept))
    dimensionless_atoms = [
        *base_dimless_kept,
        *sorted(derived_dimless_atoms, key=_atom_rank)[:remaining_dimless_cap],
    ]
    if isinstance(stats, dict) and debug_limit > 0:
        stats["debug_atomized_linear_span_atoms"] = {
            "target": [str(node_str(getattr(atom, "node", None))) for atom in tuple(target_atoms)],
            "dimensionless": [
                str(node_str(getattr(atom, "node", None))) for atom in tuple(dimensionless_atoms)
            ],
        }

    term_rows: list[tuple[tuple, str, float]] = []
    seen_terms: set[str] = set()
    term_meta_by_key: dict[str, dict[str, Any]] = {}
    target_term_rows: list[tuple[tuple, str, float]] = []
    product_term_rows: list[tuple[tuple, str, float]] = []

    for atom in target_atoms:
        node = simplify(atom.node)
        score = _finite_float(getattr(atom, "score", 0.0), 0.0)
        kind = str(getattr(atom, "kind", "") or "").strip().lower()
        roles = {str(v).strip().lower() for v in tuple(getattr(atom, "roles", ()) or ())}
        term_rank_bias = 0.0 if kind == "target_term" or "target_term" in roles or "expr" in roles else 10.0
        before = len(term_rows)
        _append_unique_node(
            term_rows,
            seen_terms,
            node,
            source=f"atom:{node_str(node)}",
            rank_score=term_rank_bias - score,
        )
        if len(term_rows) > before:
            target_term_rows.append(term_rows[-1])
            term_key = str(node_str(term_rows[-1][0]))
            term_meta_by_key[term_key] = {
                "term_expr": term_key,
                "source": str(term_rows[-1][1]),
                "atoms": [atom_provenance(atom)],
            }

    for dimless_atom in dimensionless_atoms:
        dimless_node = simplify(dimless_atom.node)
        dimless_score = _finite_float(getattr(dimless_atom, "score", 0.0), 0.0)
        for target_atom in target_atoms:
            target_node = simplify(target_atom.node)
            product_node = simplify(("mul", dimless_node, target_node))
            try:
                product_dim = node_dims(product_node, var_dims) if var_dims is not None else y_dims
            except Exception:
                continue
            if product_dim is None or not dims_eq(product_dim, y_dims):
                continue
            target_score = _finite_float(getattr(target_atom, "score", 0.0), 0.0)
            dimless_evidence = dict(getattr(dimless_atom, "evidence", {}) or {})
            dimless_roles = {str(v) for v in tuple(getattr(dimless_atom, "roles", ()) or ())}
            derived_dimless_penalty = 2.0 if (
                bool(dimless_evidence.get("atomized_derived_dimensionless", False))
                or "derived_dimensionless" in dimless_roles
            ) else 0.0
            before = len(term_rows)
            _append_unique_node(
                term_rows,
                seen_terms,
                product_node,
                source=f"product:{node_str(dimless_node)}*{node_str(target_node)}",
                rank_score=float(derived_dimless_penalty) - (dimless_score + target_score),
            )
            if len(term_rows) > before:
                product_term_rows.append(term_rows[-1])
                term_key = str(node_str(term_rows[-1][0]))
                term_meta_by_key[term_key] = {
                    "term_expr": term_key,
                    "source": str(term_rows[-1][1]),
                    "atoms": [atom_provenance(dimless_atom), atom_provenance(target_atom)],
                }

    term_rows.sort(key=lambda item: (float(item[2]), int(node_size(item[0])), str(node_str(item[0]))))
    target_term_rows.sort(key=lambda item: (float(item[2]), int(node_size(item[0])), str(node_str(item[0]))))
    product_term_rows.sort(key=lambda item: (float(item[2]), int(node_size(item[0])), str(node_str(item[0]))))
    if isinstance(stats, dict):
        stats["atomized_linear_span_terms"] = int(len(term_rows))
        stats["atomized_linear_span_product_terms"] = int(len(product_term_rows))
    if not term_rows:
        return []

    combo_rows: list[tuple[tuple[tuple, ...], str]] = []
    seen_combos: set[str] = set()

    def add_combo(nodes: Collection[tuple], source: str) -> None:
        valid_nodes = tuple(node for node in tuple(nodes or ()) if isinstance(node, tuple) and is_valid_node(node))
        if not valid_nodes:
            return
        key = "|".join(str(node_str(node)) for node in valid_nodes)
        if key in seen_combos:
            return
        seen_combos.add(key)
        combo_rows.append((valid_nodes, str(source)))

    for row in target_term_rows[:4]:
        add_combo((row[0],), "target_singleton")
    for row in product_term_rows[:8]:
        add_combo((row[0],), "product_singleton")
    coverage_before = len(combo_rows)
    coverage_product_cap = min(len(product_term_rows), max(128, max_rows_i * 4))
    for target_row in target_term_rows[:8]:
        for product_row in product_term_rows[:coverage_product_cap]:
            add_combo((target_row[0], product_row[0]), "target_plus_product_coverage")
    product_pair_cap = min(len(product_term_rows), max(96, max_rows_i * 3))
    for left_row, right_row in itertools.combinations(product_term_rows[:product_pair_cap], 2):
        add_combo((left_row[0], right_row[0]), "product_pair_coverage")
    coverage_added = len(combo_rows) - coverage_before
    for target_row in target_term_rows[:4]:
        for product_row in product_term_rows[:16]:
            add_combo((target_row[0], product_row[0]), "target_plus_product")
    for left, right in itertools.combinations([row[0] for row in term_rows[:16]], 2):
        add_combo((left, right), "term_pair")
    for combo in itertools.combinations([row[0] for row in term_rows[:10]], 3):
        add_combo(combo, "term_triple")

    if isinstance(stats, dict):
        stats["atomized_linear_span_candidates"] = int(
            stats.get("atomized_linear_span_candidates", 0) or 0
        ) + int(len(combo_rows))
        stats["atomized_linear_span_coverage_candidates"] = int(
            stats.get("atomized_linear_span_coverage_candidates", 0) or 0
        ) + int(coverage_added)

    rows: list[dict[str, Any]] = []
    seen_preview_keys: set[str] = set()
    best_probe = math.inf
    scored = 0
    include_intercept = bool(_dim_is_dimensionless(y_dims)) and _env_bool(
        "NESTY_ATOMIZED_LINEAR_SPAN_ALLOW_DIMLESS_BIAS",
        True,
    )
    if isinstance(stats, dict):
        stats["atomized_linear_span_include_intercept"] = bool(include_intercept)
    combo_score_limit = min(max(2048, max_rows_i * 32), 4096)
    if isinstance(stats, dict):
        stats["atomized_linear_span_score_limit"] = int(combo_score_limit)
    for term_nodes, source in combo_rows:
        if len(rows) >= combo_score_limit:
            break
        fit_cols: list[torch.Tensor] = []
        probe_cols: list[torch.Tensor] = []
        valid = True
        for node in term_nodes:
            try:
                fit_val = eval_node(node, x_fit)
                probe_val = eval_node(node, x_probe)
            except Exception:
                valid = False
                break
            if (not torch.is_tensor(fit_val)) or (not torch.is_tensor(probe_val)):
                valid = False
                break
            if (not torch.isfinite(fit_val).all()) or (not torch.isfinite(probe_val).all()):
                valid = False
                break
            fit_cols.append(fit_val.reshape(int(x_fit.shape[0]), -1)[:, 0])
            probe_cols.append(probe_val.reshape(int(x_probe.shape[0]), -1)[:, 0])
        if not valid or not fit_cols:
            continue
        materializer_payload: dict[str, Any] = {"terms": list(term_nodes)}
        if include_intercept:
            fit_cols.append(torch.ones(int(x_fit.shape[0]), dtype=x_fit.dtype, device=x_fit.device))
            probe_cols.append(torch.ones(int(x_probe.shape[0]), dtype=x_probe.dtype, device=x_probe.device))
            materializer_payload["bias_index"] = int(len(term_nodes))
        bound_closure = make_direct_affine_closure(
            scaffold_id="atomized_linear_span",
            term_nodes=term_nodes,
        )
        design = ClosureDesign(
            fit_matrix=torch.stack(fit_cols, dim=1),
            probe_matrix=torch.stack(probe_cols, dim=1),
            materializer="linear_combo_scaled",
            materializer_payload=materializer_payload,
            metadata={"source": "atomized_linear_span", "combo_source": str(source)},
        )
        scored_ret = score_bound_closure(bound_closure, design=design, y_fit=y_fit, y_probe=y_probe)
        if not isinstance(scored_ret, Mapping):
            continue
        scored += 1
        coeffs = [float(v) for v in list(scored_ret.get("coeffs", []) or ())]
        proposal_key = "atomized_linear_span::" + "|".join(str(node_str(node)) for node in term_nodes)
        row = make_direct_preview_row(
            bound_closure=bound_closure,
            child_expr=scored_ret["expr"],
            fit_mse=float(scored_ret.get("fit_mse", math.inf)),
            probe_mse=float(scored_ret.get("probe_mse", math.inf)),
            max_depth=int(max_depth) + max(0, int(len(term_nodes))),
            var_dims=var_dims,
            y_dims=y_dims,
            candidate_subtree_node=scored_ret["expr"] if isinstance(scored_ret.get("expr"), tuple) else None,
            parent_sub_size=0,
            parent_sub_depth=0,
            parent_size=0,
            parent_depth=0,
            generation_source="atomized_linear_span",
            tuple_provenance="atomized_linear_span",
            proposal_family="atomized_linear_span",
            local_mapping_kind="atomized_linear_span",
            direct_metadata={
                "proposal_key": proposal_key,
                "source": "atomized_linear_span",
                "combo_source": str(source),
                "term_nodes": list(term_nodes),
                "term_exprs": [str(node_str(node)) for node in term_nodes],
                "term_atom_provenance": [
                    term_meta_by_key.get(str(node_str(node)), {"term_expr": str(node_str(node)), "atoms": []})
                    for node in term_nodes
                ],
                "include_intercept": bool(include_intercept),
            },
            seen_child_keys=seen_preview_keys,
            proposal_key=proposal_key,
            local_mapping_coeffs=coeffs,
            local_mapping_nparams=int(len(term_nodes) + (1 if include_intercept else 0)),
        )
        if row is None:
            continue
        row["proposal_lane"] = "atomized"
        row["operator_id"] = "atomized:linear_span"
        row["scaffold_id"] = "atomized:linear_span"
        row["scaffold_family"] = "affine"
        row["atomized_linear_span"] = True
        row["atomized_linear_span_atom_provenance"] = list(
            dict(row.get("direct_metadata", {}) or {}).get("term_atom_provenance", []) or []
        )
        rows.append(row)
        best_probe = min(best_probe, float(row.get("local_probe_mse", math.inf) or math.inf))

    rows.sort(
        key=lambda row: (
            float(row.get("local_probe_mse", math.inf) or math.inf),
            float(row.get("local_fit_mse", math.inf) or math.inf),
            int(row.get("candidate_child_size", 0) or 0),
            str(row.get("proposal_key", "") or ""),
        )
    )
    rows = rows[:max_rows_i]
    if isinstance(stats, dict):
        stats["atomized_linear_span_scored"] = int(stats.get("atomized_linear_span_scored", 0) or 0) + int(scored)
        stats["atomized_linear_span_rows"] = int(stats.get("atomized_linear_span_rows", 0) or 0) + int(len(rows))
        if math.isfinite(best_probe):
            prev = _finite_float(stats.get("atomized_linear_span_best_probe", math.inf), math.inf)
            stats["atomized_linear_span_best_probe"] = float(min(prev, best_probe))
        if debug_limit > 0:
            debug_rows = stats.get("debug_atomized_linear_span_rows", None)
            if not isinstance(debug_rows, list):
                debug_rows = []
                stats["debug_atomized_linear_span_rows"] = debug_rows
            for row in rows[: max(0, int(debug_limit))]:
                debug_rows.append(
                    {
                        "expr": str(node_str(row.get("expr"))),
                        "local_fit_mse": float(row.get("local_fit_mse", math.inf)),
                        "local_probe_mse": float(row.get("local_probe_mse", math.inf)),
                        "term_exprs": list(dict(row.get("direct_metadata", {}) or {}).get("term_exprs", []) or []),
                        "term_atom_provenance": list(
                            dict(row.get("direct_metadata", {}) or {}).get("term_atom_provenance", []) or []
                        ),
                        "proposal_key": str(row.get("proposal_key", "") or ""),
                    }
                )
    return rows


def _candidate_float(row: Mapping[str, Any] | None, key: str, default: float = math.inf) -> float:
    if not isinstance(row, Mapping):
        return float(default)
    try:
        value = float(row.get(str(key), default) or default)
    except Exception:
        value = float(default)
    return value


def _candidate_lane(row: Mapping[str, Any] | None) -> str:
    if not isinstance(row, Mapping):
        return ""
    return str(row.get("proposal_lane", "") or "").strip().lower()


def _candidate_spec_key(row: Mapping[str, Any] | None) -> str:
    if not isinstance(row, Mapping):
        return ""
    family = str(row.get("scaffold_family", "") or row.get("proposal_family", "") or "")
    operator_id = str(row.get("operator_id", "") or row.get("operator_spec_key", "") or "")
    scaffold_id = str(row.get("scaffold_id", "") or "")
    if family or operator_id:
        return f"{family}::{operator_id or scaffold_id}"
    return scaffold_id


def _candidate_family(row: Mapping[str, Any] | None) -> str:
    if not isinstance(row, Mapping):
        return ""
    return str(row.get("scaffold_family", "") or row.get("proposal_family", "") or "").strip()


def _candidate_variant_token(row: Mapping[str, Any] | None) -> str:
    if not isinstance(row, Mapping):
        return ""
    if _candidate_family(row) != "power":
        return ""
    direct_metadata = row.get("direct_metadata", None)
    if not isinstance(direct_metadata, Mapping):
        return ""
    return str(direct_metadata.get("power_variant", "") or "").strip().lower()


def _candidate_variant_group_key(row: Mapping[str, Any] | None) -> str:
    variant_token = _candidate_variant_token(row)
    if not variant_token:
        return ""
    if not isinstance(row, Mapping):
        return ""
    direct_metadata = row.get("direct_metadata", None)
    if not isinstance(direct_metadata, Mapping):
        return ""

    def _node_key(node: Any) -> str:
        if isinstance(node, tuple) and is_valid_node(node):
            return str(node_str(node))
        return str(node)

    spec_key = _candidate_spec_key(row)
    anchor_node = direct_metadata.get("anchor_node", None)
    hole_node = direct_metadata.get("hole_node", None)
    return f"{spec_key}::{_node_key(anchor_node)}::{_node_key(hole_node)}"


def _candidate_variant_nparams(row: Mapping[str, Any] | None) -> int:
    if not isinstance(row, Mapping):
        return math.inf
    try:
        value = int(row.get("local_mapping_nparams", math.inf) or math.inf)
    except Exception:
        value = math.inf
    return max(0, int(value)) if math.isfinite(value) else math.inf


def _candidate_feature_metadata(row: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        return {}
    feature_block = row.get("feature_block_obj", None)
    metadata = dict(getattr(feature_block, "metadata", {}) or {})
    if metadata:
        return metadata
    feature_block_dict = row.get("feature_block_dict", None)
    if isinstance(feature_block_dict, Mapping):
        feature_meta = feature_block_dict.get("metadata", None)
        if isinstance(feature_meta, Mapping):
            return dict(feature_meta)
    return {}


def _candidate_interaction_key(row: Mapping[str, Any] | None) -> str:
    metadata = _candidate_feature_metadata(row)
    token = str(metadata.get("interaction_key", "") or "").strip()
    if token:
        return token
    if not isinstance(row, Mapping):
        return ""
    direct_metadata = row.get("direct_metadata", None)
    if not isinstance(direct_metadata, Mapping):
        return ""

    def _node_key(node: Any) -> str:
        if isinstance(node, tuple) and is_valid_node(node):
            return str(node_str(node))
        return ""

    family = _candidate_family(row)
    if family == "periodic":
        for key in ("feature_node", "hole_node"):
            token_local = _node_key(direct_metadata.get(key, None))
            if token_local:
                return token_local
    for key in ("power_inner_node", "hole_node", "feature_node"):
        token_local = _node_key(direct_metadata.get(key, None))
        if token_local:
            return token_local
    return ""


def _candidate_anchor_diversity_key(row: Mapping[str, Any] | None) -> str:
    metadata = _candidate_feature_metadata(row)
    token = str(metadata.get("anchor_diversity_key", "") or "").strip()
    if token:
        return token
    if not isinstance(row, Mapping):
        return ""
    direct_metadata = row.get("direct_metadata", None)
    if not isinstance(direct_metadata, Mapping):
        return ""
    for key in ("anchor_node", "envelope_node", "anchor_lift_node"):
        node = direct_metadata.get(key, None)
        if isinstance(node, tuple) and is_valid_node(node):
            return str(node_str(node))
    return ""


def _candidate_support_key(row: Mapping[str, Any] | None) -> str:
    metadata = _candidate_feature_metadata(row)
    token = str(metadata.get("support_id", "") or "").strip()
    if token:
        return token
    if not isinstance(row, Mapping):
        return ""
    feature_block = row.get("feature_block_obj", None)
    if isinstance(feature_block, FeatureBlock):
        feature_meta = dict(getattr(feature_block, "metadata", {}) or {})
        token = str(feature_meta.get("support_id", "") or "").strip()
        if token:
            return token
        token = str(getattr(feature_block, "block_id", "") or "").strip()
        if token:
            return token
    bound_closure = row.get("bound_closure_obj", None)
    if isinstance(bound_closure, BoundClosure):
        token = str(bound_closure_identity_key(bound_closure) or "").strip()
        if token:
            return token
    return str(row.get("proposal_key", "") or row.get("child_key", "") or "").strip()


def _pair_entry_record_sort_key(entry: Mapping[str, Any]) -> tuple[Any, bool, int, int, str]:
    return (
        entry.get("sort_key", None),
        bool(entry.get("not_exact", True)),
        max(0, int(entry.get("child_size", 0) or 0)),
        int(entry.get("idx", 0) or 0),
        str(entry.get("proposal_key", "") or ""),
    )


def _build_pair_entry_record(
    *,
    base_basis_state: BasisState | None,
    row: Mapping[str, Any] | None,
    sort_key: Any,
    idx: int,
    proposal_key: str,
    source_pool: str,
    require_interaction_support: bool,
    allow_covered_feature: bool = False,
    allow_preview_materialization: bool = False,
) -> dict[str, Any] | None:
    if not isinstance(row, Mapping):
        return None
    row_local: Mapping[str, Any] = row
    candidate_basis_state = row_local.get("basis_state_obj", None)
    if not isinstance(candidate_basis_state, BasisState):
        proposal_candidate = row_local.get("proposal_candidate_obj", None)
        if isinstance(proposal_candidate, ProposalCandidate) and isinstance(proposal_candidate.basis_state, BasisState):
            candidate_basis_state = proposal_candidate.basis_state
        elif bool(allow_preview_materialization):
            feature_block = row_local.get("feature_block_obj", None)
            expr = row_local.get("expr", None)
            if isinstance(feature_block, FeatureBlock):
                diagnostics = {
                    "route": "closure_search_pair_entry_preview",
                    "scaffold_id": str(row_local.get("scaffold_id", "") or proposal_key),
                    "family": str(row_local.get("scaffold_family", "") or row_local.get("proposal_family", "") or ""),
                }
                provenance = (
                    f"closure_search_pair_entry_preview:{str(row_local.get('scaffold_id', '') or proposal_key)}",
                )
                fit_bundle = {}
                local_mapping_kind = str(row_local.get("local_mapping_kind", "") or "")
                if local_mapping_kind:
                    fit_bundle["mapping_kind"] = local_mapping_kind
                local_mapping_coeffs = row_local.get("local_mapping_coeffs", None)
                if isinstance(local_mapping_coeffs, (list, tuple)):
                    fit_bundle["mapping_coeffs"] = [float(v) for v in local_mapping_coeffs]
                candidate_basis_state = BasisState(
                    blocks=(feature_block,),
                    fit_bundle=fit_bundle,
                    fit_loss=_candidate_float(row_local, "local_fit_mse"),
                    probe_loss=_candidate_float(row_local, "local_probe_mse"),
                    complexity=float(feature_block.complexity()),
                    diagnostics=diagnostics,
                    provenance=provenance,
                    compiled_expr=expr if isinstance(expr, tuple) and is_valid_node(expr) else None,
                )
        if not isinstance(candidate_basis_state, BasisState):
            return None
        row_local = dict(row_local)
        row_local["basis_state_obj"] = candidate_basis_state
    feature_block = row_local.get("feature_block_obj", None)
    if (not bool(allow_covered_feature)) and basis_state_covers_feature_block(base_basis_state, feature_block):
        return None
    interaction_key = _candidate_interaction_key(row_local)
    support_key = _candidate_support_key(row_local)
    if require_interaction_support and (not interaction_key or not support_key):
        return None
    try:
        child_size = int(row_local.get("candidate_child_size", 0) or 0)
    except Exception:
        child_size = 0
    entry = PairEntry(
        sort_key=sort_key,
        idx=int(idx),
        row=row_local,
        proposal_key=str(proposal_key),
        family=str(row_local.get("scaffold_family", "") or row_local.get("proposal_family", "") or ""),
        spec_key=str(_candidate_spec_key(row_local) or ""),
        interaction_key=str(interaction_key),
        support_key=str(support_key),
        anchor_key=str(_candidate_anchor_diversity_key(row_local)),
        source_pool=str(source_pool or ""),
        not_exact=bool(not _candidate_is_exact_bound(row_local)),
        child_size=max(0, int(child_size)),
    )
    return entry.to_record()


def _pair_relation_tags(entry_a: Mapping[str, Any], entry_b: Mapping[str, Any]) -> list[str]:
    tags: list[str] = []
    interaction_a = str(entry_a.get("interaction_key", "") or "")
    interaction_b = str(entry_b.get("interaction_key", "") or "")
    support_a = str(entry_a.get("support_key", "") or "")
    support_b = str(entry_b.get("support_key", "") or "")
    family_a = str(entry_a.get("family", "") or "")
    family_b = str(entry_b.get("family", "") or "")
    spec_a = str(entry_a.get("spec_key", "") or "")
    spec_b = str(entry_b.get("spec_key", "") or "")
    anchor_a = str(entry_a.get("anchor_key", "") or "")
    anchor_b = str(entry_b.get("anchor_key", "") or "")
    if interaction_a and interaction_a == interaction_b:
        tags.append("same_interaction")
    if support_a and support_b:
        tags.append("same_support" if support_a == support_b else "different_support")
    if family_a and family_b:
        tags.append("same_family" if family_a == family_b else "cross_family")
    if spec_a and spec_b:
        tags.append("same_spec" if spec_a == spec_b else "distinct_spec")
    if anchor_a and anchor_b:
        tags.append("same_anchor" if anchor_a == anchor_b else "different_anchor")
    return tags


def _best_round_singleton(
    round_singleton_commits: list[RoundCommitObject] | tuple[RoundCommitObject, ...] | None,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_score = math.inf
    for commit in tuple(round_singleton_commits or ()):
        if not isinstance(commit, RoundCommitObject) or str(commit.kind) != "singleton" or not bool(commit.accepted):
            continue
        try:
            score = float(commit.eff_mse)
        except Exception:
            score = math.inf
        if not math.isfinite(score):
            continue
        if score < best_score:
            best_score = score
            best = {
                "proposal_key": str(commit.proposal_key),
                "eff_mse": float(score),
                "source_pool": str(commit.source_pool or ""),
            }
    return best


def _mse_slack(*, base_value: float, rel_slack: float, abs_slack: float) -> float:
    base_mag = 0.0
    try:
        if math.isfinite(float(base_value)):
            base_mag = max(0.0, abs(float(base_value)))
    except Exception:
        base_mag = 0.0
    return max(float(abs_slack), float(rel_slack) * base_mag)


def _pair_template_matches(
    entry_a: Mapping[str, Any],
    entry_b: Mapping[str, Any],
    template_kind: str,
) -> bool:
    interaction_a = str(entry_a.get("interaction_key", "") or "")
    interaction_b = str(entry_b.get("interaction_key", "") or "")
    support_a = str(entry_a.get("support_key", "") or "")
    support_b = str(entry_b.get("support_key", "") or "")
    family_a = str(entry_a.get("family", "") or "")
    family_b = str(entry_b.get("family", "") or "")
    spec_a = str(entry_a.get("spec_key", "") or "")
    spec_b = str(entry_b.get("spec_key", "") or "")
    if template_kind == "same_interaction_diff_support":
        return bool(interaction_a and interaction_a == interaction_b and support_a and support_b and support_a != support_b)
    if template_kind == "cross_family_complement":
        return bool(family_a and family_b and family_a != family_b)
    if template_kind == "distinct_spec_sibling":
        return bool(spec_a and spec_b and spec_a != spec_b)
    if template_kind == "generic_complement":
        return True
    return False


def _pair_member_id(key_a: str, key_b: str) -> tuple[str, str]:
    pair_keys = (str(key_a), str(key_b))
    return tuple(sorted(pair_keys))


def _build_pair_candidate_records(
    pair_entries: list[Mapping[str, Any]],
    *,
    template_kinds: tuple[str, ...],
    blocked_pair_ids: Collection[tuple[str, str]] | None = None,
    require_exact_scored_member: bool = False,
) -> list[dict[str, Any]]:
    pair_candidates: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    blocked_ids = {tuple(item) for item in tuple(blocked_pair_ids or ())}
    for template_rank, template_kind in enumerate(tuple(template_kinds or ())):
        for entry_a, entry_b in itertools.combinations(pair_entries, 2):
            if not _pair_template_matches(entry_a, entry_b, str(template_kind)):
                continue
            key_a = str(entry_a.get("proposal_key", "") or "")
            key_b = str(entry_b.get("proposal_key", "") or "")
            pair_id = _pair_member_id(key_a, key_b)
            if pair_id in blocked_ids:
                continue
            if bool(require_exact_scored_member):
                source_a = str(entry_a.get("source_pool", "") or "")
                source_b = str(entry_b.get("source_pool", "") or "")
                if "exact_scored_singleton" not in (source_a, source_b):
                    continue
            if pair_id in seen_pairs:
                continue
            seen_pairs.add(pair_id)
            family_a = str(entry_a.get("family", "") or "")
            family_b = str(entry_b.get("family", "") or "")
            spec_a = str(entry_a.get("spec_key", "") or "")
            spec_b = str(entry_b.get("spec_key", "") or "")
            anchor_a = str(entry_a.get("anchor_key", "") or "")
            anchor_b = str(entry_b.get("anchor_key", "") or "")
            pair_candidates.append(
                {
                    "template_kind": str(template_kind),
                    "template_rank": int(template_rank),
                    "anchor_rank": 0 if anchor_a and anchor_b and anchor_a != anchor_b else 1,
                    "cross_family_rank": 0 if family_a and family_b and family_a != family_b else 1,
                    "distinct_spec_rank": 0 if spec_a and spec_b and spec_a != spec_b else 1,
                    "not_exact_rank": int(entry_a.get("not_exact", True)) + int(entry_b.get("not_exact", True)),
                    "idx_rank": int(entry_a.get("idx", 0)) + int(entry_b.get("idx", 0)),
                    "child_size_rank": int(entry_a.get("child_size", 0)) + int(entry_b.get("child_size", 0)),
                    "pair_member_id": pair_id,
                    "key_a": key_a,
                    "key_b": key_b,
                    "entry_a": entry_a,
                    "entry_b": entry_b,
                }
            )
    pair_candidates.sort(
        key=lambda item: (
            int(item.get("template_rank", 0)),
            int(item.get("anchor_rank", 0)),
            int(item.get("cross_family_rank", 0)),
            int(item.get("distinct_spec_rank", 0)),
            int(item.get("not_exact_rank", 0)),
            int(item.get("idx_rank", 0)),
            int(item.get("child_size_rank", 0)),
            str(item.get("key_a", "") or ""),
            str(item.get("key_b", "") or ""),
        )
    )
    return pair_candidates


def _count_candidate_families(
    rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] | None,
    *,
    native_only: bool = False,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in tuple(rows or ()):
        if not isinstance(row, Mapping):
            continue
        if native_only and not isinstance(row.get("basis_state_obj", None), BasisState):
            continue
        family = _candidate_family(row) or "<unknown>"
        counts[family] = int(counts.get(family, 0) or 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (str(kv[0]), int(kv[1]))))


def _candidate_is_fasttrack(row: Mapping[str, Any] | None, *, threshold: float = _OUTER_SCAFFOLD_FASTTRACK_MSE) -> bool:
    if not isinstance(row, Mapping):
        return False
    if bool(row.get("preview_fasttrack", False)):
        return True
    probe_mse = _candidate_float(row, "local_probe_mse")
    fit_mse = _candidate_float(row, "local_fit_mse")
    return (math.isfinite(probe_mse) and probe_mse <= float(threshold)) or (
        math.isfinite(fit_mse) and fit_mse <= float(threshold)
    )


def _candidate_execution_mode(row: Mapping[str, Any] | None) -> str:
    if not isinstance(row, Mapping):
        return ""
    for key in ("execution_mode", "proposal_execution_mode"):
        value = str(row.get(str(key), "") or "").strip().lower()
        if value:
            return value
    direct_metadata = row.get("direct_metadata", None)
    if isinstance(direct_metadata, Mapping):
        return str(direct_metadata.get("execution_mode", "") or "").strip().lower()
    return ""


def _candidate_is_exact_bound(row: Mapping[str, Any] | None) -> bool:
    return "exact_bound" in _candidate_execution_mode(row)


def _basis_state_signature(state: BasisState | None) -> tuple[Any, ...] | None:
    if not isinstance(state, BasisState):
        return None
    block_rows: list[tuple[Any, ...]] = []
    for block in tuple(getattr(state, "blocks", ()) or ()):
        atoms = tuple(
            str(node_str(node))
            for node in tuple(getattr(block, "atoms", ()) or ())
            if isinstance(node, tuple) and is_valid_node(node)
        )
        bundle = tuple(
            (
                str(role),
                str(node_str(node)),
            )
            for role, node in zip(
                tuple(getattr(block, "latent_bundle_roles", ()) or ()),
                tuple(getattr(block, "latent_bundle_nodes", ()) or ()),
            )
            if isinstance(node, tuple) and is_valid_node(node)
        )
        block_rows.append((str(getattr(block, "family", "")), atoms, bundle))
    compiled_key = (
        str(node_str(state.compiled_expr))
        if isinstance(state.compiled_expr, tuple) and is_valid_node(state.compiled_expr)
        else ""
    )
    probe_loss = float(getattr(state, "probe_loss", math.inf))
    fit_loss = float(getattr(state, "fit_loss", math.inf))
    return (
        tuple(block_rows),
        compiled_key,
        round(probe_loss, 18) if math.isfinite(probe_loss) else math.inf,
        round(fit_loss, 18) if math.isfinite(fit_loss) else math.inf,
    )


def _basis_beam_signature(beam: tuple[BasisState, ...] | list[BasisState] | None) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        sig
        for sig in (
            _basis_state_signature(row)
            for row in tuple(beam or ())
            if isinstance(row, BasisState)
        )
        if sig is not None
    )


def _basis_state_identity_key(state: BasisState | None) -> tuple[Any, ...] | None:
    sig = _basis_state_signature(state)
    if sig is None:
        return None
    return sig[:2]


def _basis_state_rank_key(state: BasisState | None) -> tuple[float, float, float, str]:
    if not isinstance(state, BasisState):
        return (math.inf, math.inf, math.inf, "")
    probe_loss = float(getattr(state, "probe_loss", math.inf))
    fit_loss = float(getattr(state, "fit_loss", math.inf))
    complexity = float(getattr(state, "complexity", math.inf))
    compiled_key = (
        str(node_str(state.compiled_expr))
        if isinstance(state.compiled_expr, tuple) and is_valid_node(state.compiled_expr)
        else ""
    )
    return (probe_loss, fit_loss, complexity, compiled_key)


def _basis_state_primary_family(state: BasisState | None) -> str:
    if not isinstance(state, BasisState):
        return ""
    for block in tuple(getattr(state, "blocks", ()) or ()):
        family = str(getattr(block, "family", "") or "").strip()
        if family:
            return family
    return ""


def _admit_seed_basis_state_to_beam(
    beam: tuple[BasisState, ...] | list[BasisState] | None,
    state: BasisState | None,
    *,
    beam_width: int,
    family_cap: int,
) -> tuple[BasisState, ...]:
    rows = [row for row in tuple(beam or ()) if isinstance(row, BasisState)]
    if isinstance(state, BasisState):
        rows.append(state)
    if not rows or int(beam_width) <= 0:
        return ()
    best_by_identity: dict[tuple[Any, ...], BasisState] = {}
    for row in rows:
        identity = _basis_state_identity_key(row)
        if identity is None:
            continue
        prev = best_by_identity.get(identity, None)
        if prev is None or _basis_state_rank_key(row) < _basis_state_rank_key(prev):
            best_by_identity[identity] = row
    ranked = sorted(best_by_identity.values(), key=_basis_state_rank_key)
    kept: list[BasisState] = []
    family_cap_i = max(0, int(family_cap))
    family_counts: dict[str, int] = {}
    for row in ranked:
        family = _basis_state_primary_family(row)
        if family_cap_i > 0 and family and int(family_counts.get(family, 0) or 0) >= family_cap_i:
            continue
        kept.append(row)
        if family:
            family_counts[family] = int(family_counts.get(family, 0) or 0) + 1
        if len(kept) >= int(beam_width):
            break
    return tuple(kept)


def _admit_basis_state_to_beam_preserving_unexpanded(
    beam: tuple[BasisState, ...] | list[BasisState] | None,
    state: BasisState | None,
    *,
    beam_width: int,
    expanded_state_ids: set[tuple[Any, ...]],
) -> tuple[BasisState, ...]:
    rows = [row for row in tuple(beam or ()) if isinstance(row, BasisState)]
    beam_width_i = max(0, int(beam_width))
    if beam_width_i <= 0:
        return ()
    protected: list[BasisState] = []
    mutable_rows: list[BasisState] = []
    seen_protected_ids: set[tuple[Any, ...]] = set()
    for row in rows:
        state_id = _basis_state_identity_key(row)
        if state_id is not None and state_id not in set(expanded_state_ids or set()):
            if state_id in seen_protected_ids:
                continue
            seen_protected_ids.add(state_id)
            protected.append(row)
            continue
        mutable_rows.append(row)
    protected.sort(key=_basis_state_rank_key)
    reserve_limit = int(beam_width_i - 1) if isinstance(state, BasisState) else int(beam_width_i)
    reserve_limit = max(0, reserve_limit)
    protected = protected[:reserve_limit]
    remaining_width = max(0, int(beam_width_i) - int(len(protected)))
    admitted_tail = admit_basis_state_to_beam(
        mutable_rows,
        state,
        beam_width=max(remaining_width, 0),
    )
    merged_rows = [*protected, *tuple(admitted_tail or ())]
    best_by_identity: dict[tuple[Any, ...], BasisState] = {}
    for row in merged_rows:
        state_id = _basis_state_identity_key(row)
        if state_id is None:
            continue
        prev = best_by_identity.get(state_id, None)
        if prev is None or _basis_state_rank_key(row) < _basis_state_rank_key(prev):
            best_by_identity[state_id] = row
    kept: list[BasisState] = []
    for row in protected:
        state_id = _basis_state_identity_key(row)
        if state_id is None:
            continue
        best = best_by_identity.pop(state_id, None)
        if isinstance(best, BasisState):
            kept.append(best)
    ranked_tail = sorted(best_by_identity.values(), key=_basis_state_rank_key)
    kept.extend(ranked_tail)
    return tuple(kept[:beam_width_i])


def _remaining_wall_seconds(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    try:
        return max(0.0, float(deadline) - float(time.perf_counter()))
    except Exception:
        return 0.0


def _as_col_tensor(value: Any) -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor):
        return None
    out = value.detach()
    if out.ndim == 1:
        out = out.unsqueeze(-1)
    elif out.ndim >= 2:
        out = out.reshape(out.shape[0], -1)
    else:
        return None
    if out.ndim != 2 or out.shape[0] <= 0:
        return None
    if out.shape[1] > 1:
        out = out[:, :1]
    if not out.is_floating_point():
        out = out.to(dtype=torch.float64)
    return out


def _safe_eval_node_col(node: Any, x: torch.Tensor | None) -> torch.Tensor | None:
    if not isinstance(x, torch.Tensor):
        return None
    if not isinstance(node, tuple) or not is_valid_node(node):
        return None
    try:
        out = eval_node(node, x)
    except Exception:
        return None
    out_col = _as_col_tensor(out)
    if out_col is None or out_col.shape[0] != x.shape[0]:
        return None
    return out_col


def _finite_mask(*cols: torch.Tensor | None) -> torch.Tensor | None:
    valid_cols = [col for col in cols if isinstance(col, torch.Tensor)]
    if not valid_cols:
        return None
    mask = torch.ones((valid_cols[0].shape[0],), dtype=torch.bool, device=valid_cols[0].device)
    for col in valid_cols:
        if col.ndim != 2 or col.shape[1] != 1 or col.shape[0] != mask.shape[0]:
            return None
        mask &= torch.isfinite(col.squeeze(-1))
    return mask


def _rms(col: torch.Tensor | None) -> float | None:
    if not isinstance(col, torch.Tensor):
        return None
    mask = _finite_mask(col)
    if mask is None or int(mask.sum().item()) <= 0:
        return None
    vals = col.squeeze(-1)[mask]
    if vals.numel() <= 0:
        return None
    try:
        return float(torch.sqrt(torch.mean(vals.square())).item())
    except Exception:
        return None


def _fit_probe_gain(
    cols_fit: list[torch.Tensor],
    cols_probe: list[torch.Tensor],
    target_fit: torch.Tensor | None,
    target_probe: torch.Tensor | None,
) -> float:
    target_fit_col = _as_col_tensor(target_fit)
    target_probe_col = _as_col_tensor(target_probe)
    if target_fit_col is None or target_probe_col is None:
        return 0.0
    if not cols_fit or len(cols_fit) != len(cols_probe):
        return 0.0
    mask_fit = _finite_mask(target_fit_col, *cols_fit)
    mask_probe = _finite_mask(target_probe_col, *cols_probe)
    if mask_fit is None or mask_probe is None:
        return 0.0
    rank = int(len(cols_fit))
    if int(mask_fit.sum().item()) <= rank or int(mask_probe.sum().item()) <= 0:
        return 0.0
    try:
        a_fit = torch.cat([col[mask_fit] for col in cols_fit], dim=1).to(dtype=torch.float64)
        b_fit = target_fit_col[mask_fit].to(dtype=torch.float64)
        a_probe = torch.cat([col[mask_probe] for col in cols_probe], dim=1).to(dtype=torch.float64)
        b_probe = target_probe_col[mask_probe].to(dtype=torch.float64)
    except Exception:
        return 0.0
    if a_fit.shape[0] <= a_fit.shape[1] or a_probe.shape[0] <= 0:
        return 0.0
    baseline = float(torch.mean(b_probe.square()).item())
    if not math.isfinite(baseline) or baseline <= 1.0e-20:
        return 0.0
    try:
        beta = torch.linalg.lstsq(a_fit, b_fit).solution
    except Exception:
        try:
            beta = torch.linalg.pinv(a_fit) @ b_fit
        except Exception:
            return 0.0
    try:
        pred_probe = a_probe @ beta
        mse = float(torch.mean((pred_probe - b_probe).square()).item())
    except Exception:
        return 0.0
    if not math.isfinite(mse):
        return 0.0
    gain = 1.0 - (float(mse) / float(baseline))
    return float(max(0.0, min(1.0, gain)))


def _candidate_seed_nodes(
    *,
    nvars: int,
    boost_pool_nodes,
    basis_state: BasisState | None,
    aux_seed_blocks=(),
    limit: int = 12,
) -> list[tuple]:
    out: list[tuple] = []
    seen: set[str] = set()

    def _add(node: Any) -> None:
        if len(out) >= int(max(1, limit)):
            return
        if not isinstance(node, tuple) or not is_valid_node(node):
            return
        if str(node[0]) == "const":
            return
        key = str(node_str(node))
        if key in seen:
            return
        seen.add(key)
        out.append(node)

    if isinstance(basis_state, BasisState):
        for block in tuple(getattr(basis_state, "blocks", ()) or ()):
            for atom in tuple(getattr(block, "atoms", ()) or ()):
                _add(atom)
    for block in tuple(aux_seed_blocks or ()):
        node = getattr(block, "node", None)
        if isinstance(node, tuple):
            _add(node)
    for idx in range(max(0, int(nvars))):
        _add(("var", int(idx)))
    pool_nodes_sorted = sorted(
        [
            node
            for node in list(boost_pool_nodes or ())
            if isinstance(node, tuple) and is_valid_node(node)
        ],
        key=lambda node: (int(node_size(node)), str(node_str(node))),
    )
    for node in pool_nodes_sorted:
        _add(node)
    return out


def _best_periodic_probe(
    *,
    seed_nodes: list[tuple],
    x_fit: torch.Tensor | None,
    x_probe: torch.Tensor | None,
    residual_fit: torch.Tensor | None,
    residual_probe: torch.Tensor | None,
) -> float:
    best = 0.0
    one_fit = torch.ones_like(residual_fit) if isinstance(residual_fit, torch.Tensor) else None
    one_probe = torch.ones_like(residual_probe) if isinstance(residual_probe, torch.Tensor) else None
    for node in seed_nodes:
        phi_fit = _safe_eval_node_col(node, x_fit)
        phi_probe = _safe_eval_node_col(node, x_probe)
        if phi_fit is None or phi_probe is None:
            continue
        score = _fit_probe_gain(
            [torch.sin(phi_fit), torch.cos(phi_fit), one_fit],
            [torch.sin(phi_probe), torch.cos(phi_probe), one_probe],
            residual_fit,
            residual_probe,
        )
        if score > best:
            best = score
    return float(best)


def _best_exp_probe(
    *,
    seed_nodes: list[tuple],
    x_fit: torch.Tensor | None,
    x_probe: torch.Tensor | None,
    residual_fit: torch.Tensor | None,
    residual_probe: torch.Tensor | None,
) -> float:
    best = 0.0
    one_fit = torch.ones_like(residual_fit) if isinstance(residual_fit, torch.Tensor) else None
    one_probe = torch.ones_like(residual_probe) if isinstance(residual_probe, torch.Tensor) else None
    for node in seed_nodes:
        phi_fit = _safe_eval_node_col(node, x_fit)
        phi_probe = _safe_eval_node_col(node, x_probe)
        if phi_fit is None or phi_probe is None:
            continue
        score = _fit_probe_gain(
            [torch.exp(torch.clamp(phi_fit, min=-8.0, max=8.0)), phi_fit, one_fit],
            [torch.exp(torch.clamp(phi_probe, min=-8.0, max=8.0)), phi_probe, one_probe],
            residual_fit,
            residual_probe,
        )
        if score > best:
            best = score
    return float(best)


def _best_log_probe(
    *,
    seed_nodes: list[tuple],
    x_fit: torch.Tensor | None,
    x_probe: torch.Tensor | None,
    residual_fit: torch.Tensor | None,
    residual_probe: torch.Tensor | None,
) -> float:
    best = 0.0
    one_fit = torch.ones_like(residual_fit) if isinstance(residual_fit, torch.Tensor) else None
    one_probe = torch.ones_like(residual_probe) if isinstance(residual_probe, torch.Tensor) else None
    eps = 1.0e-9
    for node in seed_nodes:
        phi_fit = _safe_eval_node_col(node, x_fit)
        phi_probe = _safe_eval_node_col(node, x_probe)
        if phi_fit is None or phi_probe is None:
            continue
        if not bool(torch.all(phi_fit > eps).item()) or not bool(torch.all(phi_probe > eps).item()):
            continue
        score = _fit_probe_gain(
            [torch.log(torch.clamp_min(phi_fit, eps)), phi_fit, one_fit],
            [torch.log(torch.clamp_min(phi_probe, eps)), phi_probe, one_probe],
            residual_fit,
            residual_probe,
        )
        if score > best:
            best = score
    return float(best)


def _best_rational_probe(
    *,
    seed_nodes: list[tuple],
    x_fit: torch.Tensor | None,
    x_probe: torch.Tensor | None,
    residual_fit: torch.Tensor | None,
    residual_probe: torch.Tensor | None,
) -> float:
    best = 0.0
    one_fit = torch.ones_like(residual_fit) if isinstance(residual_fit, torch.Tensor) else None
    one_probe = torch.ones_like(residual_probe) if isinstance(residual_probe, torch.Tensor) else None
    eps = 1.0e-8
    for node in seed_nodes:
        phi_fit = _safe_eval_node_col(node, x_fit)
        phi_probe = _safe_eval_node_col(node, x_probe)
        if phi_fit is None or phi_probe is None:
            continue
        den_fit = 1.0 + phi_fit
        den_probe = 1.0 + phi_probe
        if bool(torch.any(torch.abs(den_fit) <= eps).item()) or bool(torch.any(torch.abs(den_probe) <= eps).item()):
            continue
        recip_fit = 1.0 / den_fit
        recip_probe = 1.0 / den_probe
        score = _fit_probe_gain(
            [recip_fit, phi_fit, one_fit],
            [recip_probe, phi_probe, one_probe],
            residual_fit,
            residual_probe,
        )
        if score > best:
            best = score
    return float(best)


def _best_power_probe(
    *,
    seed_nodes: list[tuple],
    x_fit: torch.Tensor | None,
    x_probe: torch.Tensor | None,
    residual_fit: torch.Tensor | None,
    residual_probe: torch.Tensor | None,
) -> float:
    best = 0.0
    one_fit = torch.ones_like(residual_fit) if isinstance(residual_fit, torch.Tensor) else None
    one_probe = torch.ones_like(residual_probe) if isinstance(residual_probe, torch.Tensor) else None
    eps = 1.0e-9
    for node in seed_nodes:
        phi_fit = _safe_eval_node_col(node, x_fit)
        phi_probe = _safe_eval_node_col(node, x_probe)
        if phi_fit is None or phi_probe is None:
            continue
        candidates: list[tuple[torch.Tensor, torch.Tensor]] = [
            (phi_fit.square(), phi_probe.square()),
        ]
        if bool(torch.all(phi_fit > eps).item()) and bool(torch.all(phi_probe > eps).item()):
            sqrt_fit = torch.sqrt(torch.clamp_min(phi_fit, eps))
            sqrt_probe = torch.sqrt(torch.clamp_min(phi_probe, eps))
            candidates.extend(
                [
                    (sqrt_fit, sqrt_probe),
                    (1.0 / sqrt_fit, 1.0 / sqrt_probe),
                    (1.0 / torch.clamp_min(phi_fit, eps), 1.0 / torch.clamp_min(phi_probe, eps)),
                ]
            )
        for cand_fit, cand_probe in candidates:
            score = _fit_probe_gain(
                [cand_fit, one_fit],
                [cand_probe, one_probe],
                residual_fit,
                residual_probe,
            )
            if score > best:
                best = score
    return float(best)


def _best_quadratic_probe(
    *,
    nvars: int,
    x_fit: torch.Tensor | None,
    x_probe: torch.Tensor | None,
    residual_fit: torch.Tensor | None,
    residual_probe: torch.Tensor | None,
) -> float:
    if not isinstance(x_fit, torch.Tensor) or not isinstance(x_probe, torch.Tensor):
        return 0.0
    if x_fit.ndim != 2 or x_probe.ndim != 2:
        return 0.0
    nvars_i = min(int(nvars), int(x_fit.shape[1]), int(x_probe.shape[1]))
    if nvars_i <= 0:
        return 0.0
    one_fit = torch.ones_like(residual_fit) if isinstance(residual_fit, torch.Tensor) else None
    one_probe = torch.ones_like(residual_probe) if isinstance(residual_probe, torch.Tensor) else None
    best = 0.0
    max_subset = min(3, nvars_i)
    for width in range(2, max_subset + 1):
        for subset in itertools.combinations(range(nvars_i), width):
            radial_fit = torch.sqrt(torch.clamp_min(torch.sum(x_fit[:, list(subset)].square(), dim=1, keepdim=True), 0.0))
            radial_probe = torch.sqrt(torch.clamp_min(torch.sum(x_probe[:, list(subset)].square(), dim=1, keepdim=True), 0.0))
            score = _fit_probe_gain(
                [radial_fit, one_fit],
                [radial_probe, one_probe],
                residual_fit,
                residual_probe,
            )
            if score > best:
                best = score
            for pref_idx in range(nvars_i):
                if pref_idx in subset:
                    continue
                pref_fit = x_fit[:, pref_idx : pref_idx + 1]
                pref_probe = x_probe[:, pref_idx : pref_idx + 1]
                score = _fit_probe_gain(
                    [pref_fit * radial_fit, radial_fit, pref_fit, one_fit],
                    [pref_probe * radial_probe, radial_probe, pref_probe, one_probe],
                    residual_fit,
                    residual_probe,
                )
                if score > best:
                    best = score
    return float(best)


def _basis_state_with_residuals(
    state: BasisState | None,
    *,
    residual_fit: torch.Tensor | None,
    residual_probe: torch.Tensor | None,
    residual_witness: Mapping[str, Any] | None,
) -> BasisState | None:
    if not isinstance(state, BasisState):
        return None
    return BasisState(
        blocks=tuple(state.blocks),
        fit_bundle=dict(state.fit_bundle or {}),
        fit_loss=float(state.fit_loss),
        probe_loss=float(state.probe_loss),
        complexity=float(state.complexity),
        residual_fit=residual_fit,
        residual_probe=residual_probe,
        residual_witness=dict(residual_witness or {}) if isinstance(residual_witness, Mapping) else residual_witness,
        diagnostics=dict(state.diagnostics or {}),
        provenance=tuple(state.provenance or ()),
        compiled_expr=state.compiled_expr,
    )


def _build_residual_guided_context(
    *,
    base_context: ProposalContext | None,
    basis_state: BasisState | None,
    basis_state_beam,
    families,
    total_budget: int,
    wall_time_remaining_s: float | None,
    nvars: int,
    x_fit: torch.Tensor | None,
    y_fit: torch.Tensor | None,
    x_probe: torch.Tensor | None,
    y_probe: torch.Tensor | None,
    boost_pool_nodes,
) -> ProposalContext:
    base = base_context if isinstance(base_context, ProposalContext) else ProposalContext()
    refit_basis_state = fit_basis_state_head(
        basis_state,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        route_name="closure_search_residual_refit",
    )
    active_basis_state = refit_basis_state if isinstance(refit_basis_state, BasisState) else basis_state
    y_fit_col = _as_col_tensor(y_fit)
    y_probe_col = _as_col_tensor(y_probe)
    residual_fit = (
        active_basis_state.residual_fit
        if isinstance(active_basis_state, BasisState) and isinstance(active_basis_state.residual_fit, torch.Tensor)
        else None
    )
    residual_probe = (
        active_basis_state.residual_probe
        if isinstance(active_basis_state, BasisState) and isinstance(active_basis_state.residual_probe, torch.Tensor)
        else None
    )
    if residual_fit is None or residual_probe is None:
        pred_fit = None
        pred_probe = None
        if isinstance(active_basis_state, BasisState) and isinstance(active_basis_state.compiled_expr, tuple):
            pred_fit = _safe_eval_node_col(active_basis_state.compiled_expr, x_fit)
            pred_probe = _safe_eval_node_col(active_basis_state.compiled_expr, x_probe)
        if pred_fit is None and isinstance(y_fit_col, torch.Tensor):
            pred_fit = torch.zeros_like(y_fit_col)
        if pred_probe is None and isinstance(y_probe_col, torch.Tensor):
            pred_probe = torch.zeros_like(y_probe_col)
        residual_fit = (
            (y_fit_col - pred_fit)
            if isinstance(y_fit_col, torch.Tensor) and isinstance(pred_fit, torch.Tensor)
            else None
        )
        residual_probe = (
            (y_probe_col - pred_probe)
            if isinstance(y_probe_col, torch.Tensor) and isinstance(pred_probe, torch.Tensor)
            else None
        )

    seed_nodes = _candidate_seed_nodes(
        nvars=int(nvars),
        boost_pool_nodes=boost_pool_nodes,
        basis_state=basis_state,
        aux_seed_blocks=tuple(getattr(base, "aux_seed_blocks", ()) or ()),
        limit=12,
    )
    family_tokens = [
        str(token or "").strip().lower()
        for token in list(families or ())
        if str(token or "").strip()
    ]
    family_probe_scores: dict[str, float] = {}
    for family in family_tokens:
        if family == "periodic":
            family_probe_scores[family] = _best_periodic_probe(
                seed_nodes=seed_nodes,
                x_fit=x_fit,
                x_probe=x_probe,
                residual_fit=residual_fit,
                residual_probe=residual_probe,
            )
        elif family == "exp":
            family_probe_scores[family] = _best_exp_probe(
                seed_nodes=seed_nodes,
                x_fit=x_fit,
                x_probe=x_probe,
                residual_fit=residual_fit,
                residual_probe=residual_probe,
            )
        elif family == "log":
            family_probe_scores[family] = _best_log_probe(
                seed_nodes=seed_nodes,
                x_fit=x_fit,
                x_probe=x_probe,
                residual_fit=residual_fit,
                residual_probe=residual_probe,
            )
        elif family == "rational":
            family_probe_scores[family] = _best_rational_probe(
                seed_nodes=seed_nodes,
                x_fit=x_fit,
                x_probe=x_probe,
                residual_fit=residual_fit,
                residual_probe=residual_probe,
            )
        elif family == "power":
            family_probe_scores[family] = _best_power_probe(
                seed_nodes=seed_nodes,
                x_fit=x_fit,
                x_probe=x_probe,
                residual_fit=residual_fit,
                residual_probe=residual_probe,
            )
        elif family == "quadratic":
            family_probe_scores[family] = _best_quadratic_probe(
                nvars=int(nvars),
                x_fit=x_fit,
                x_probe=x_probe,
                residual_fit=residual_fit,
                residual_probe=residual_probe,
            )

    derived_hints: dict[str, float] = {}
    for family, raw_score in family_probe_scores.items():
        score = float(raw_score)
        if not math.isfinite(score) or score <= 0.05:
            continue
        derived_hints[str(family)] = float(3.0 * score)
    merged_hints = {
        str(key): float(value)
        for key, value in dict(getattr(base, "family_hints", {}) or {}).items()
        if isinstance(value, (int, float))
    }
    for family, score in derived_hints.items():
        merged_hints[family] = float(merged_hints.get(family, 0.0)) + float(score)

    sorted_families = [
        family
        for family, score in sorted(
            family_probe_scores.items(),
            key=lambda item: (-float(item[1]), str(item[0])),
        )
        if float(score) > 0.05
    ]
    tags: list[str] = []
    for family in sorted_families[:3]:
        tags.append(str(family))
        if family == "periodic":
            tags.extend(["oscillatory", "harmonic"])
        elif family == "quadratic":
            tags.extend(["radial", "norm"])
        elif family == "power":
            tags.extend(["sqrt", "inverse_power"])
        elif family == "rational":
            tags.extend(["quotient", "ratio"])
    witness: dict[str, Any] = {
        "residual_fit_rms": _rms(residual_fit),
        "residual_probe_rms": _rms(residual_probe),
        "target_fit_rms": _rms(y_fit_col),
        "target_probe_rms": _rms(y_probe_col),
        "relative_probe_rms": None,
        "family_probe_scores": {
            str(key): float(value) for key, value in family_probe_scores.items() if math.isfinite(float(value))
        },
        "suggested_families": [str(token) for token in sorted_families[:3]],
        "tags": tags,
        "seed_exprs": [str(node_str(node)) for node in seed_nodes[:8]],
    }
    if witness["residual_probe_rms"] is not None and witness["target_probe_rms"] not in (None, 0.0):
        denom = float(witness["target_probe_rms"])
        if math.isfinite(denom) and abs(denom) > 1.0e-20:
            witness["relative_probe_rms"] = float(witness["residual_probe_rms"]) / denom
    if base.residual_witness is not None and not isinstance(base.residual_witness, Mapping):
        witness["upstream_residual_witness"] = str(base.residual_witness)
    elif isinstance(base.residual_witness, Mapping):
        witness["upstream_residual_witness"] = {
            str(key): value for key, value in dict(base.residual_witness).items() if key not in witness
        }

    diagnostics = dict(getattr(base, "diagnostics", {}) or {})
    diagnostics.update(
        {
            "route": "closure_search",
            "families": [str(token) for token in family_tokens],
            "basis_block_families": (
                [str(block.family) for block in tuple(getattr(active_basis_state, "blocks", ()) or ())]
                if isinstance(active_basis_state, BasisState)
                else []
            ),
            "basis_compiled_expr": (
                str(node_str(active_basis_state.compiled_expr))
                if isinstance(active_basis_state, BasisState) and isinstance(active_basis_state.compiled_expr, tuple)
                else ""
            ),
            "residual_fit_rms": witness.get("residual_fit_rms", None),
            "residual_probe_rms": witness.get("residual_probe_rms", None),
            "residual_family_probe_scores": witness.get("family_probe_scores", {}),
            "aux_seed_block_count": int(len(tuple(getattr(base, "aux_seed_blocks", ()) or ()))),
        }
    )

    updated_basis_state = _basis_state_with_residuals(
        active_basis_state,
        residual_fit=residual_fit,
        residual_probe=residual_probe,
        residual_witness=witness,
    )
    beam_out: list[BasisState] = []
    replaced = False
    for row in tuple(basis_state_beam or ()):
        if not isinstance(row, BasisState):
            continue
        if not replaced and isinstance(updated_basis_state, BasisState) and row is basis_state:
            beam_out.append(updated_basis_state)
            replaced = True
        else:
            beam_out.append(row)
    if isinstance(updated_basis_state, BasisState) and not beam_out:
        beam_out.append(updated_basis_state)
    elif isinstance(updated_basis_state, BasisState) and not replaced and isinstance(basis_state, BasisState):
        beam_out = [updated_basis_state, *beam_out]

    return ProposalContext(
        basis_state=updated_basis_state,
        basis_state_beam=tuple(beam_out),
        aux_seed_blocks=tuple(getattr(base, "aux_seed_blocks", ()) or ()),
        atom_library=getattr(base, "atom_library", None),
        residual_witness=witness,
        diagnostics=diagnostics,
        family_hints=merged_hints,
        total_budget=int(total_budget),
        wall_time_remaining_s=wall_time_remaining_s,
    )


def _basis_state_from_scored_result(
    *,
    score_ret: Mapping[str, Any] | None,
    candidate_meta: Mapping[str, Any] | None,
    current_basis_state: BasisState | None,
    route_name: str,
) -> BasisState | None:
    if not isinstance(score_ret, Mapping):
        return None
    scored_expr = score_ret.get("expr", None)
    raw_mse = float(score_ret.get("raw_mse", math.inf) or math.inf)
    fit_loss = float(score_ret.get("fit_loss", raw_mse) or raw_mse)
    probe_loss = float(score_ret.get("probe_loss", raw_mse) or raw_mse)
    promoted = score_ret.get("basis_state_obj", None)
    if isinstance(promoted, BasisState):
        return basis_state_retarget(
            promoted,
            fit_loss=float(fit_loss),
            probe_loss=float(probe_loss),
            compiled_expr=scored_expr if isinstance(scored_expr, tuple) else promoted.compiled_expr,
            route_name=str(route_name),
            provenance_tag=f"{str(route_name)}:scored",
        )
    if not isinstance(candidate_meta, Mapping):
        return None
    preview_state = candidate_meta.get("basis_state_obj", None)
    if not isinstance(preview_state, BasisState):
        return None
    return basis_state_retarget(
        preview_state,
        fit_loss=float(fit_loss),
        probe_loss=float(probe_loss),
        compiled_expr=scored_expr if isinstance(scored_expr, tuple) else None,
        route_name=str(route_name),
        provenance_tag=f"{str(route_name)}:scored",
    )


def _proposal_family_from_candidate_meta(candidate_meta: Mapping[str, Any] | None, route_name: str) -> str:
    if not isinstance(candidate_meta, Mapping):
        return str(route_name)
    proposal_candidate = candidate_meta.get("proposal_candidate_obj", None)
    if isinstance(proposal_candidate, ProposalCandidate) and str(getattr(proposal_candidate, "family", "") or "").strip():
        return str(proposal_candidate.family)
    for key in ("scaffold_family", "proposal_family", "family"):
        value = str(candidate_meta.get(key, "") or "").strip()
        if value:
            return value
    return str(route_name)


def _basis_state_acceptance_decision(
    *,
    current_basis_state: BasisState | None,
    candidate_basis_state: BasisState | None,
    candidate_meta: Mapping[str, Any] | None,
    family_hints: Mapping[str, float] | None,
    complexity_penalty: float,
    route_name: str,
    min_probe_gain_rel: float = 1.0e-9,
    min_probe_gain_abs: float = 1.0e-12,
) -> dict[str, Any]:
    if not isinstance(candidate_basis_state, BasisState):
        return {"accept": False, "reason": "missing_candidate_basis_state"}
    candidate_probe = float(getattr(candidate_basis_state, "probe_loss", math.inf))
    candidate_fit = float(getattr(candidate_basis_state, "fit_loss", math.inf))
    if not math.isfinite(candidate_probe) or not math.isfinite(candidate_fit):
        return {"accept": False, "reason": "nonfinite_candidate_loss"}

    family = _proposal_family_from_candidate_meta(candidate_meta, route_name=route_name)
    hints = {
        str(key): float(value)
        for key, value in dict(family_hints or {}).items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    hint_value = float(hints.get(str(family), 0.0))
    max_hint = max((float(v) for v in hints.values()), default=0.0)
    hint_ratio = 0.0 if max_hint <= 0.0 else max(0.0, min(1.0, float(hint_value) / float(max_hint)))

    if not isinstance(current_basis_state, BasisState):
        return {
            "accept": True,
            "reason": "no_current_basis_state",
            "family": str(family),
            "probe_gain": math.inf,
            "required_gain": 0.0,
            "hint_ratio": float(hint_ratio),
        }

    current_probe = float(getattr(current_basis_state, "probe_loss", math.inf))
    current_complexity = float(getattr(current_basis_state, "complexity", 0.0) or 0.0)
    candidate_complexity = float(getattr(candidate_basis_state, "complexity", 0.0) or 0.0)
    if not math.isfinite(current_probe):
        return {
            "accept": True,
            "reason": "current_probe_unset",
            "family": str(family),
            "probe_gain": math.inf,
            "required_gain": 0.0,
            "hint_ratio": float(hint_ratio),
        }

    probe_gain = float(current_probe - candidate_probe)
    complexity_cost = float(max(0.0, candidate_complexity - current_complexity)) * float(max(0.0, complexity_penalty))
    complexity_bonus = float(max(0.0, current_complexity - candidate_complexity)) * float(max(0.0, complexity_penalty))
    base_required_gain = max(float(min_probe_gain_abs), max(1.0, abs(float(current_probe))) * float(min_probe_gain_rel))
    # Stronger residual family evidence lowers the required gain, but only mildly.
    hint_scale = 1.0 - (0.5 * float(hint_ratio))
    required_gain = max(0.0, (float(base_required_gain) + float(complexity_cost)) * float(hint_scale))
    effective_gain = float(probe_gain + complexity_bonus)
    accept = bool(effective_gain + 1.0e-18 >= required_gain)
    return {
        "accept": bool(accept),
        "reason": "accepted" if accept else "insufficient_probe_gain",
        "family": str(family),
        "probe_gain": float(probe_gain),
        "effective_gain": float(effective_gain),
        "required_gain": float(required_gain),
        "current_probe": float(current_probe),
        "candidate_probe": float(candidate_probe),
        "current_complexity": float(current_complexity),
        "candidate_complexity": float(candidate_complexity),
        "hint_ratio": float(hint_ratio),
    }


def score_native_candidate_basis_state(
    *,
    candidate_meta: Mapping[str, Any] | None,
    stats: dict[str, Any],
    route_name: str,
    state: ProposalScoringState,
    x_fit,
    y_fit,
    x_probe,
    y_probe,
    complexity_penalty: float,
    node_str_fn: Callable[[Any], str],
    arch,
) -> Mapping[str, Any] | None:
    if not isinstance(candidate_meta, Mapping):
        return None
    preview_state = candidate_meta.get("basis_state_obj", None)
    if not isinstance(preview_state, BasisState):
        return None
    proposal_candidate = candidate_meta.get("proposal_candidate_obj", None)
    if not isinstance(proposal_candidate, ProposalCandidate) and not str(candidate_meta.get("proposal_key", "") or ""):
        return None
    preserve_direct_state = bool(candidate_meta.get("basis_state_direct_preserve", False))
    if preserve_direct_state:
        preserved_expr = candidate_meta.get("expr", None)
        if not (isinstance(preserved_expr, tuple) and is_valid_node(preserved_expr)):
            preserved_expr = preview_state.compiled_expr
        scored_state = basis_state_retarget(
            preview_state,
            fit_loss=_candidate_float(candidate_meta, "local_fit_mse", default=float(getattr(preview_state, "fit_loss", math.inf))),
            probe_loss=_candidate_float(candidate_meta, "local_probe_mse", default=float(getattr(preview_state, "probe_loss", math.inf))),
            compiled_expr=preserved_expr if isinstance(preserved_expr, tuple) else preview_state.compiled_expr,
            route_name=f"{str(route_name)}:native_basis_preserve",
            provenance_tag=f"{str(route_name)}:native_preserve",
        )
        stats["basis_state_native_preserved"] = int(stats.get("basis_state_native_preserved", 0)) + 1
    else:
        scored_state = fit_basis_state_head(
            preview_state,
            x_fit=x_fit,
            y_fit=y_fit,
            x_probe=x_probe,
            y_probe=y_probe,
            route_name=f"{str(route_name)}:native_basis_score",
        )
    if not isinstance(scored_state, BasisState):
        return None
    scored_expr = scored_state.compiled_expr
    if not isinstance(scored_expr, tuple) or not is_valid_node(scored_expr):
        return None

    fit_mse = float(getattr(scored_state, "fit_loss", math.inf))
    probe_mse = float(getattr(scored_state, "probe_loss", math.inf))
    raw_mse = float(probe_mse)
    mse_eff = float(raw_mse) + float(max(0.0, complexity_penalty)) * float(getattr(scored_state, "complexity", 0.0))

    z_col = _safe_eval_node_col(scored_expr, x_fit)
    if z_col is None:
        target_fit_col = _as_col_tensor(y_fit)
        nrows = int(target_fit_col.shape[0]) if isinstance(target_fit_col, torch.Tensor) else 1
        z_col = torch.zeros((nrows, 1), dtype=torch.float64)

    key = str(
        candidate_meta.get("proposal_key", "")
        or getattr(proposal_candidate, "identity_key", "")
        or node_str_fn(scored_expr)
    )
    basis_state_summary = scored_state.to_dict()
    block_families = [
        str(block.family)
        for block in tuple(getattr(scored_state, "blocks", ()) or ())
        if isinstance(block, FeatureBlock)
    ]
    fit_bundle = dict(getattr(scored_state, "fit_bundle", {}) or {})
    basis_head = fit_bundle.get("basis_head", None)
    basis_head_terms = []
    if isinstance(basis_head, Mapping):
        basis_head_terms = list(basis_head.get("term_exprs", []) or basis_head.get("block_exprs", []) or [])
    acceptance_basis = "basis_state_native"
    key_l = str(key).lower()
    if "atomized_linear_span" in key_l:
        acceptance_basis = "atomized_linear_span"
    elif preserve_direct_state:
        acceptance_basis = "basis_state_native_preserve"
    compiled_expr_str = str(node_str_fn(scored_expr))
    score_ladder = {
        "schema_version": 1,
        "carrier": {
            "expr": compiled_expr_str,
            "probe_mse_identity": float(raw_mse) if math.isfinite(float(raw_mse)) else None,
        },
        "mapped": {
            "available": False,
            "mapping_kind": "basis_state_native",
            "mapping_structural": True,
            "probe_mse": None,
        },
        "head_augmented": {
            "available": isinstance(basis_head, Mapping),
            "accepted": isinstance(basis_head, Mapping),
            "probe_mse": float(raw_mse) if math.isfinite(float(raw_mse)) else None,
            "fit_mse": float(fit_mse) if math.isfinite(float(fit_mse)) else None,
            "kind": str(basis_head.get("kind", "basis_linear_head")) if isinstance(basis_head, Mapping) else "",
            "term_count": int(len(basis_head_terms)),
            "block_count": int(len(block_families)),
            "block_families": block_families,
        },
        "compiled_structural": {
            "available": True,
            "accepted": True,
            "probe_mse": float(raw_mse) if math.isfinite(float(raw_mse)) else None,
            "fit_mse": float(fit_mse) if math.isfinite(float(fit_mse)) else None,
            "expr": compiled_expr_str,
            "route": str(route_name),
            "source": acceptance_basis,
        },
        "refined": {
            "enabled": False,
            "attempted": False,
            "accepted": False,
            "probe_mse": None,
            "expr": None,
        },
        "final_validation": {
            "available": False,
        },
    }
    mapping = {
        "kind": "basis_state_native",
        "_basis_state_native": True,
        "_basis_transition": None,
        "route": str(route_name),
        "_acceptance_basis": acceptance_basis,
        "_score_ladder": score_ladder,
        "_basis_state_summary": basis_state_summary,
    }

    state.n_evaluated += 1
    stats["evals_used"] = int(stats.get("evals_used", 0)) + 1
    stats["scored"] = int(stats.get("scored", 0)) + 1
    stats["basis_state_native_scores"] = int(stats.get("basis_state_native_scores", 0)) + 1

    if raw_mse < state.best_raw_mse:
        state.best_raw_mse = raw_mse
    if raw_mse < state.best_raw_mse_struct:
        state.best_raw_mse_struct = raw_mse

    best_mse_before_update = float(state.best_mse)
    is_new = arch.update(key, mse_eff, scored_expr, z_col, mapping, raw_mse=raw_mse)
    if bool(is_new):
        stats["new_residual_basins"] = int(stats.get("new_residual_basins", 0)) + 1
    if mse_eff < best_mse_before_update:
        stats["global_best_updates"] = int(stats.get("global_best_updates", 0)) + 1
    if mse_eff < state.best_mse:
        state.best_mse = mse_eff

    out_candidate = proposal_candidate
    if isinstance(proposal_candidate, ProposalCandidate):
        out_candidate = ProposalCandidate(
            family=str(proposal_candidate.family),
            rendered_expr=scored_expr,
            scaffold_id=str(proposal_candidate.scaffold_id),
            identity_key=str(proposal_candidate.identity_key),
            feature_block=proposal_candidate.feature_block,
            basis_state=scored_state,
            bound_closure=proposal_candidate.bound_closure,
            local_fit_loss=float(fit_mse),
            local_probe_loss=float(probe_mse),
            complexity=float(scored_state.complexity),
            metadata={
                **dict(proposal_candidate.metadata or {}),
                "route": str(route_name),
                "rendered_expr": str(node_str_fn(scored_expr)),
                "native_basis_score": True,
            },
        )

    return {
        "expr": scored_expr,
        "mapping": mapping,
        "raw_mse": float(raw_mse),
        "fit_loss": float(fit_mse),
        "probe_loss": float(probe_mse),
        "eff_mse": float(mse_eff),
        "is_new": bool(is_new),
        "basis_state_obj": scored_state,
        "basis_state_dict": scored_state.to_dict(),
        "proposal_candidate_obj": out_candidate,
        "proposal_candidate_dict": (
            out_candidate.to_dict() if isinstance(out_candidate, ProposalCandidate) else None
        ),
    }


def score_external_candidate_expr(
    expr,
    *,
    parent_raw_mse: float | None,
    stats: dict[str, Any],
    route_name: str,
    candidate_meta: Mapping[str, Any] | None,
    state: ProposalScoringState,
    dm: bool,
    var_dims,
    y_dims,
    refine_cfg: Mapping[str, Any],
    score_prescreen_stats: dict[str, Any],
    closure_search_anchor_head_compare_enable: bool,
    x_fit,
    y_fit,
    x_probe,
    y_probe,
    proj,
    fp_mode,
    q_scale,
    q_clip,
    poly_degree: int,
    refine_enable: bool,
    refine_state,
    early_stop_mse: float,
    complexity_penalty: float,
    score_expr_fn: Callable[..., Any],
    simplify_fn: Callable[[Any], Any],
    is_valid_node_fn: Callable[[Any], bool],
    node_str_fn: Callable[[Any], str],
    node_dims_fn: Callable[[Any, Any], Any],
    dims_eq_fn: Callable[[Any, Any], bool],
    node_size_fn: Callable[[Any], int],
    mapping_cost_fn: Callable[[Any], float],
    mapping_is_structural_fn: Callable[[Any], bool],
    arch,
):
    candidate_is_carrier = is_certified_inner_coordinate(candidate_meta)
    carrier_context = None

    if not is_valid_node_fn(expr):
        stats["skipped_invalid_expr"] = int(stats.get("skipped_invalid_expr", 0)) + 1
        return None

    expr = simplify_fn(expr)
    if not is_valid_node_fn(expr):
        stats["skipped_invalid_expr"] = int(stats.get("skipped_invalid_expr", 0)) + 1
        return None

    try:
        while isinstance(expr, tuple) and expr and expr[0] == "neg":
            expr = expr[1]
    except Exception:
        pass
    try:
        if (
            isinstance(expr, tuple)
            and len(expr) >= 3
            and str(expr[0]) == "sub"
            and node_str_fn(expr[1]) > node_str_fn(expr[2])
        ):
            expr = ("sub", expr[2], expr[1])
    except Exception:
        pass

    if dm:
        expr_dim = node_dims_fn(expr, var_dims)
        if candidate_is_carrier:
            carrier_context = context_from_metadata(
                candidate_meta,
                carrier_dim=expr_dim,
                target_dim=y_dims,
            )
            carrier_precheck = precheck_carrier_units(carrier_context)
            _record_carrier_unit_result(
                stats,
                carrier_precheck,
                context=carrier_context,
            )
            if carrier_precheck.decision.value != "DEFERRED_UNTIL_OUTER_MAP":
                stats["skipped_invalid_expr"] = int(stats.get("skipped_invalid_expr", 0)) + 1
                return None
        elif expr_dim is None:
            stats["skipped_invalid_expr"] = int(stats.get("skipped_invalid_expr", 0)) + 1
            return None
        elif y_dims is not None and not dims_eq_fn(expr_dim, y_dims):
            stats["skipped_invalid_expr"] = int(stats.get("skipped_invalid_expr", 0)) + 1
            return None

    global_best_raw_mse_for_scoring = None
    if math.isfinite(state.best_raw_mse_struct):
        global_best_raw_mse_for_scoring = float(state.best_raw_mse_struct)
    elif math.isfinite(state.best_raw_mse):
        global_best_raw_mse_for_scoring = float(state.best_raw_mse)

    score_cfg_base = dict(refine_cfg)
    score_cfg_base["score_prescreen_enable"] = False
    score_cfg_base["score_prescreen_force_full"] = True
    score_cfg_base["score_prescreen_action_name"] = str(route_name)
    score_cfg_base["score_prescreen_parent_mse"] = parent_raw_mse
    score_cfg_base["score_prescreen_global_best_mse"] = global_best_raw_mse_for_scoring
    score_cfg_base["score_prescreen_stats"] = score_prescreen_stats
    score_cfg, scaffold_score_ctx = build_scaffold_candidate_score_cfg(
        score_cfg_base,
        route_name=str(route_name),
        candidate_meta=candidate_meta,
        stats=stats,
    )
    use_anchor_head = bool(scaffold_score_ctx.get("use_anchor_head", False))

    def _run_score(local_score_cfg: dict[str, Any], *, force_refine_enable: bool = False):
        sc_local = score_expr_fn(
            expr,
            x_fit,
            y_fit,
            x_probe,
            y_probe,
            proj,
            fp_mode,
            q_scale,
            q_clip,
            poly_degree,
            refine_enable=bool(refine_enable or force_refine_enable),
            refine_cfg=local_score_cfg,
            refine_best_mse=(
                float(state.best_raw_mse_struct)
                if math.isfinite(state.best_raw_mse_struct)
                else float(max(state.best_raw_mse, float(early_stop_mse)))
            ),
            refine_state=refine_state,
            return_expr=True,
        )
        state.n_evaluated += 1
        stats["evals_used"] = int(stats.get("evals_used", 0)) + 1
        return sc_local

    compare_sc = None
    if use_anchor_head and bool(closure_search_anchor_head_compare_enable):
        compare_sc = _run_score(dict(score_cfg_base))
        stats["anchor_head_compare_attempts"] = int(stats.get("anchor_head_compare_attempts", 0)) + 1

    sc = _run_score(score_cfg)
    if sc is None:
        stats["skipped_score_none"] = int(stats.get("skipped_score_none", 0)) + 1
        return None

    if compare_sc is not None:
        try:
            base_mse = float(compare_sc[0])
        except Exception:
            base_mse = float("inf")
        try:
            head_mse = float(sc[0])
        except Exception:
            head_mse = float("inf")
        record_anchor_head_compare(
            stats,
            context=scaffold_score_ctx,
            expr=expr,
            base_mse=base_mse,
            head_mse=head_mse,
        )

    mse, key, z, mapping, scored_expr = sc
    stats["scored"] = int(stats.get("scored", 0)) + 1

    unit_handoff_metadata = None
    if dm and candidate_is_carrier:
        scored_carrier_dim = node_dims_fn(scored_expr, var_dims)
        carrier_context = context_from_metadata(
            candidate_meta,
            carrier_dim=scored_carrier_dim,
            target_dim=y_dims,
        )
        if scored_carrier_dim is None:
            carrier_postcheck = precheck_carrier_units(carrier_context)
        else:
            head_term_dims = None
            if isinstance(mapping, Mapping):
                head = mapping.get("_lin_head")
                if isinstance(head, Mapping):
                    terms = list(head.get("terms") or ())
                    head_term_dims = [
                        node_dims_fn(term, var_dims)
                        for term in terms
                    ]
            carrier_postcheck = validate_outer_map_units(
                carrier_context,
                mapping,
                linear_head_term_dims=head_term_dims,
            )
        unit_handoff_metadata = _record_carrier_unit_result(
            stats,
            carrier_postcheck,
            context=carrier_context,
        )
        if not carrier_postcheck.ok:
            return None
        if isinstance(mapping, Mapping):
            mapping = {
                **dict(mapping),
                "_unit_handoff": dict(unit_handoff_metadata),
            }

    if mse < state.best_raw_mse:
        state.best_raw_mse = mse
    if mapping_is_structural_fn(mapping) and mse < state.best_raw_mse_struct:
        state.best_raw_mse_struct = mse

    best_mse_before_update = float(state.best_mse)
    mse_eff = mse + complexity_penalty * (
        float(node_size_fn(expr)) + float(mapping_cost_fn(mapping))
    )
    is_new = arch.update(key, mse_eff, scored_expr, z, mapping, raw_mse=mse)
    if bool(is_new):
        stats["new_residual_basins"] = int(stats.get("new_residual_basins", 0)) + 1
    if mse_eff < best_mse_before_update:
        stats["global_best_updates"] = int(stats.get("global_best_updates", 0)) + 1
    if mse_eff < state.best_mse:
        state.best_mse = mse_eff

    basis_state_obj = None
    proposal_candidate_obj = None
    basis_transition = None
    if isinstance(mapping, dict):
        basis_transition = mapping.get("_basis_transition", None)
    if isinstance(basis_transition, Mapping):
        candidate_basis_state = None
        if isinstance(candidate_meta, Mapping):
            maybe_state = candidate_meta.get("basis_state_obj", None)
            if isinstance(maybe_state, BasisState):
                candidate_basis_state = maybe_state
        basis_state_obj = basis_state_from_additive_transition(
            basis_transition,
            base_state=candidate_basis_state,
            family=(
                str(dict(candidate_meta or {}).get("scaffold_family", "") or route_name)
                if isinstance(candidate_meta, Mapping)
                else str(route_name)
            ),
            route_name=str(route_name),
            fit_loss=float(mse),
            probe_loss=float(mse),
            compiled_expr=scored_expr,
        )
        if isinstance(basis_state_obj, BasisState):
            stats["basis_state_promotions"] = int(stats.get("basis_state_promotions", 0)) + 1
            if isinstance(candidate_meta, Mapping):
                base_candidate = candidate_meta.get("proposal_candidate_obj", None)
                proposal_candidate_obj = ProposalCandidate(
                    family=str(
                        dict(candidate_meta).get(
                            "scaffold_family",
                            getattr(base_candidate, "family", route_name) if base_candidate is not None else route_name,
                        )
                    ),
                    rendered_expr=scored_expr,
                    scaffold_id=str(
                        dict(candidate_meta).get(
                            "scaffold_id",
                            getattr(base_candidate, "scaffold_id", "") if base_candidate is not None else "",
                        )
                    ),
                    identity_key=str(
                        dict(candidate_meta).get(
                            "proposal_key",
                            getattr(base_candidate, "identity_key", "") if base_candidate is not None else "",
                        )
                    ),
                    feature_block=getattr(base_candidate, "feature_block", None),
                    basis_state=basis_state_obj,
                    bound_closure=getattr(base_candidate, "bound_closure", None),
                    local_fit_loss=float(mse),
                    local_probe_loss=float(mse),
                    complexity=float(basis_state_obj.complexity),
                    metadata={
                        "route": str(route_name),
                        "basis_transition_kind": str(basis_transition.get("kind", "")),
                        "rendered_expr": str(node_str_fn(scored_expr)) if isinstance(scored_expr, tuple) else "",
                    },
                )

    return {
        "expr": scored_expr,
        "mapping": mapping,
        "raw_mse": float(mse),
        "eff_mse": float(mse_eff),
        "is_new": bool(is_new),
        "basis_state_obj": basis_state_obj,
        "basis_state_dict": basis_state_obj.to_dict() if isinstance(basis_state_obj, BasisState) else None,
        "proposal_candidate_obj": proposal_candidate_obj,
        "proposal_candidate_dict": (
            proposal_candidate_obj.to_dict() if isinstance(proposal_candidate_obj, ProposalCandidate) else None
        ),
        "unit_handoff": unit_handoff_metadata,
    }


def merge_route_status_counts(stats: dict[str, Any], counts: Mapping[str, Any] | None) -> None:
    if not isinstance(counts, Mapping):
        return
    for key, value in dict(counts).items():
        try:
            inc = int(value)
        except Exception:
            inc = 0
        if inc <= 0:
            continue
        bucket = stats.get("status_counts", None)
        if not isinstance(bucket, dict):
            bucket = {}
            stats["status_counts"] = bucket
        bucket[str(key)] = int(bucket.get(str(key), 0) or 0) + int(inc)


def record_route_status(stats: dict[str, Any], status: object) -> None:
    status_key = str(status or "")
    if not status_key:
        return
    status_counts = stats.get("status_counts", None)
    if not isinstance(status_counts, dict):
        status_counts = {}
        stats["status_counts"] = status_counts
    status_counts[status_key] = int(status_counts.get(status_key, 0)) + 1


def run_closure_search_pass(
    *,
    closure_search_enable: bool,
    closure_search_stats: dict[str, Any],
    closure_search_families,
    closure_search_max_proposals: int,
    closure_search_anchors_per_family: int,
    closure_search_preview_topk: int,
    closure_search_exact_topk: int,
    closure_search_beam_width: int = 4,
    closure_search_seed_exact_topk: int = 6,
    closure_search_seed_beam_width: int = 4,
    closure_search_seed_scaffold_reserve: int = 8,
    closure_search_seed_family_cap: int = 2,
    closure_search_seed_exact_bound_bonus: float = 0.25,
    closure_search_pair_normal_enable: bool = False,
    closure_search_pair_normal_topk: int = 3,
    closure_search_pair_normal_max_pairs: int = 1,
    closure_search_pair_rescue_enable: bool = True,
    closure_search_pair_rescue_topk: int = 4,
    closure_search_pair_rescue_max_pairs: int = 6,
    closure_search_debug_topk: int = 0,
    closure_search_emergent_basis_enable: bool = False,
    closure_search_emergent_basis_max_source_rows: int = 32,
    closure_search_emergent_basis_score_topk: int = 8,
    closure_search_emergent_basis_max_per_round: int = 1,
    closure_search_emergent_basis_max_total: int = 4,
    closure_search_emergent_basis_min_probe_gain_rel: float = 5.0e-3,
    closure_search_emergent_aux_atoms_enable: bool = False,
    closure_search_emergent_aux_atoms_max_source_rows: int = 48,
    closure_search_emergent_aux_atoms_max_new_per_round: int = 5,
    closure_search_emergent_aux_atoms_max_total: int = 8,
    closure_search_emergent_aux_atoms_max_target: int = 4,
    closure_search_emergent_aux_atoms_max_dimensionless: int = 3,
    closure_search_emergent_aux_atoms_max_rational_derived: int = 2,
    closure_search_emergent_aux_atoms_max_seed_blocks: int = 8,
    closure_search_min_valid_frac: float,
    closure_search_min_confidence: float,
    closure_search_periodic_min_valid_scale: float,
    closure_search_periodic_min_confidence_scale: float,
    closure_search_transport_min_lin_rel: float,
    inverse_periodic_path_penalty: float,
    inverse_nonperiodic_muldiv_bonus: float,
    inverse_nonperiodic_explogsqrt_bonus: float,
    inverse_branch_beam_width: int,
    inverse_topk_terms: int,
    inverse_shortlist_mult: int,
    inverse_local_score_mode: str,
    inverse_micro_search_enable: bool,
    inverse_micro_search_max_depth: int,
    inverse_micro_search_beam_width: int,
    inverse_micro_search_topk: int,
    inverse_micro_search_seed_terms: int,
    inverse_target_mode: str,
    inverse_safe_eps: float,
    inverse_confidence_mode: str,
    inverse_confidence_target_gain: float,
    inverse_confidence_floor: float,
    inverse_full_mapping_penalty: float,
    inverse_exact_simple_target_bonus: float,
    inverse_additive_descend_penalty: float,
    inverse_nonadditive_leaf_penalty: float,
    inverse_exact_path_eta: float,
    inverse_branch_ambiguity_penalty: float,
    inverse_transport_min_effective_n: float,
    inverse_spec_regime_metadata,
    inverse_spec_local_score_mode: str,
    inverse_spec_enum_max_depth: int,
    inverse_spec_enum_max_trees: int,
    inverse_spec_max_subtree_depth,
    inverse_spec_complexity_penalty: float,
    inverse_spec_family_battery_enable: bool,
    inverse_spec_family_battery_mode: str,
    inverse_spec_recursive_enable: bool,
    inverse_spec_recursive_max_depth: int,
    inverse_spec_recursive_trigger_rel_mse: float,
    inverse_spec_recursive_seed_cap: int,
    inverse_spec_recursive_branch_topk: int,
    inverse_spec_recursive_child_topk: int,
    inverse_spec_witness_jets_enable: bool,
    inverse_spec_witness_d2_enable: bool,
    inverse_spec_witness_max_rows: int,
    inverse_spec_witness_loss_enable: bool,
    inverse_spec_witness_grad_weight: float,
    inverse_spec_witness_d2_weight: float,
    inverse_spec_witness_diag_weight: float,
    inverse_spec_witness_physics_weight: float,
    inverse_spec_active_var_screen_enable: bool,
    inverse_spec_active_var_grad_tol: float,
    inverse_spec_active_var_max_count: int,
    wall_time_deadline,
    wall_time_limit_s,
    max_depth: int,
    poly_degree: int,
    nvars: int,
    x_fit,
    y_fit,
    x_probe,
    y_probe,
    var_dims,
    y_dims,
    boost_pool_nodes,
    boost_pool_phi_fit,
    boost_pool_phi,
    boost_pool_dims,
    dm: bool,
    wall_time_exceeded_fn: Callable[[], bool],
    run_closure_search_pass_impl: Callable[..., Any],
    score_external_candidate_expr_fn: Callable[..., Any],
    score_native_candidate_basis_state_fn: Callable[..., Any] | None = None,
    node_str_fn: Callable[[Any], str],
    proposal_context: ProposalContext | None = None,
    family_allocator_fn: Callable[..., Any] | None = allocate_family_budgets,
) -> None:
    if not bool(closure_search_enable):
        return
    if wall_time_exceeded_fn():
        closure_search_stats["deadline_exceeded"] = True
        closure_search_stats["wall_time_budget_s"] = 0.0
        record_route_status(closure_search_stats, "deadline_exceeded")
        return

    family_tokens = [str(v) for v in list(closure_search_families or ()) if str(v or "").strip()]
    max_scaffolds_i = max(0, int(closure_search_max_proposals))
    anchors_per_family_i = max(0, int(closure_search_anchors_per_family))
    preview_topk_i = max(1, int(closure_search_preview_topk))
    exact_topk_i = max(0, int(closure_search_exact_topk))
    beam_width_i = max(1, int(closure_search_beam_width))
    seed_exact_topk_i = max(1, int(closure_search_seed_exact_topk))
    seed_beam_width_i = max(1, int(closure_search_seed_beam_width))
    seed_scaffold_reserve_i = max(0, int(closure_search_seed_scaffold_reserve))
    seed_family_cap_i = max(0, int(closure_search_seed_family_cap))
    seed_exact_bound_bonus_f = max(0.0, float(closure_search_seed_exact_bound_bonus))
    pair_normal_enable_b = bool(closure_search_pair_normal_enable)
    pair_normal_topk_i = max(0, int(closure_search_pair_normal_topk))
    pair_normal_max_pairs_i = max(0, int(closure_search_pair_normal_max_pairs))
    pair_rescue_enable_b = bool(closure_search_pair_rescue_enable)
    pair_rescue_topk_i = max(0, int(closure_search_pair_rescue_topk))
    pair_rescue_max_pairs_i = max(0, int(closure_search_pair_rescue_max_pairs))
    debug_topk_i = max(0, int(closure_search_debug_topk))
    debug_capture_limit_i = (max(1, int(debug_topk_i)) * 8) if debug_topk_i > 0 else 0
    emergent_basis_enable_b = bool(closure_search_emergent_basis_enable)
    emergent_basis_max_source_rows_i = max(0, int(closure_search_emergent_basis_max_source_rows))
    emergent_basis_score_topk_i = max(0, int(closure_search_emergent_basis_score_topk))
    emergent_basis_max_per_round_i = max(0, int(closure_search_emergent_basis_max_per_round))
    emergent_basis_max_total_i = max(0, int(closure_search_emergent_basis_max_total))
    emergent_basis_min_probe_gain_rel_f = max(0.0, float(closure_search_emergent_basis_min_probe_gain_rel))
    emergent_aux_atoms_enable_b = bool(closure_search_emergent_aux_atoms_enable)
    emergent_aux_atoms_max_source_rows_i = max(0, int(closure_search_emergent_aux_atoms_max_source_rows))
    emergent_aux_atoms_max_new_per_round_i = max(0, int(closure_search_emergent_aux_atoms_max_new_per_round))
    emergent_aux_atoms_max_total_i = max(0, int(closure_search_emergent_aux_atoms_max_total))
    emergent_aux_atoms_max_target_i = max(0, int(closure_search_emergent_aux_atoms_max_target))
    emergent_aux_atoms_max_dimensionless_i = max(0, int(closure_search_emergent_aux_atoms_max_dimensionless))
    emergent_aux_atoms_max_rational_derived_i = max(0, int(closure_search_emergent_aux_atoms_max_rational_derived))
    emergent_aux_atoms_max_seed_blocks_i = max(0, int(closure_search_emergent_aux_atoms_max_seed_blocks))
    atomized_linear_span_enable_b = bool(emergent_aux_atoms_enable_b) and _env_bool(
        "NESTY_ATOMIZED_LINEAR_SPAN_ENABLE",
        True,
    )
    atomized_linear_span_use_obs_pool_b = _env_bool("NESTY_ATOMIZED_LINEAR_SPAN_USE_OBS_POOL", False)
    atomized_linear_span_same_round_b = _env_bool("NESTY_ATOMIZED_LINEAR_SPAN_SAME_ROUND", False)
    atom_policy_use_obs_pool_b = _env_bool("NESTY_ATOM_POLICY_USE_OBS_POOL", False)
    atomized_linear_span_budget_i = max(0, _env_int("NESTY_ATOMIZED_LINEAR_SPAN_BUDGET", 48))
    default_atomized_exact_quota = max(0, min(2, int(exact_topk_i) - 1))
    atomized_linear_span_exact_quota_i = max(
        0,
        _env_int("NESTY_ATOMIZED_LINEAR_SPAN_EXACT_QUOTA", default_atomized_exact_quota),
    )
    emergent_aux_atoms_followup_budget_i = max(
        0,
        _env_int("NESTY_EMERGENT_AUX_ATOM_FOLLOWUP_BUDGET", 16),
    )
    if not family_tokens or max_scaffolds_i <= 0 or exact_topk_i <= 0:
        return
    scaffold_deadline = None
    if wall_time_deadline is not None:
        try:
            remaining_wall_s = max(0.0, float(wall_time_deadline) - float(time.perf_counter()))
        except Exception:
            remaining_wall_s = 0.0
        if remaining_wall_s <= 0.0:
            closure_search_stats["deadline_exceeded"] = True
            closure_search_stats["wall_time_budget_s"] = 0.0
            record_route_status(closure_search_stats, "deadline_exceeded")
            return
        scaffold_budget_fraction = 0.25
        scaffold_budget_s = min(
            float(remaining_wall_s),
            max(1.0, float(scaffold_budget_fraction) * float(wall_time_limit_s or remaining_wall_s)),
        )
        closure_search_stats["wall_time_budget_s"] = float(scaffold_budget_s)
        closure_search_stats["wall_time_budget_fraction"] = float(scaffold_budget_fraction)
        scaffold_deadline = float(time.perf_counter()) + float(scaffold_budget_s)
    else:
        closure_search_stats["wall_time_budget_s"] = None
        closure_search_stats["wall_time_budget_fraction"] = None
        remaining_wall_s = None

    proposal_context_local = proposal_context
    if not isinstance(proposal_context_local, ProposalContext):
        proposal_context_local = ProposalContext(
            basis_state=None,
            basis_state_beam=(),
            residual_witness=None,
            diagnostics={
                "route": "closure_search",
                "families": [str(v) for v in list(closure_search_families or ()) if str(v or "").strip()],
            },
            family_hints={},
            total_budget=int(max_scaffolds_i),
            wall_time_remaining_s=(
                None if remaining_wall_s is None else float(max(0.0, remaining_wall_s))
            ),
        )
    proposal_context_local = _build_residual_guided_context(
        base_context=proposal_context_local,
        basis_state=proposal_context_local.basis_state,
        basis_state_beam=proposal_context_local.basis_state_beam,
        families=family_tokens,
        total_budget=int(max_scaffolds_i),
        wall_time_remaining_s=(
            None if remaining_wall_s is None else float(max(0.0, remaining_wall_s))
        ),
        nvars=int(nvars),
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        boost_pool_nodes=boost_pool_nodes,
    )
    aux_atom_registry = tuple()
    aux_atom_observation_pool = tuple()
    closure_search_stats.setdefault("proposal_context", proposal_context_local.to_dict())
    closure_search_stats.setdefault("family_priority_scores", {})
    closure_search_stats.setdefault("family_budget_plan", [])
    closure_search_stats.setdefault("family_steering_applied", False)
    closure_search_stats.setdefault("basis_state_beam", [])
    closure_search_stats.setdefault("basis_state_beam_count", 0)
    closure_search_stats.setdefault("basis_state_beam_width", int(beam_width_i))
    closure_search_stats.setdefault("basis_state_skip_covered", 0)
    closure_search_stats.setdefault("closure_search_rounds", 0)
    closure_search_stats.setdefault("basis_state_round_updates", 0)
    closure_search_stats.setdefault("basis_state_round_commit_scored", 0)
    closure_search_stats.setdefault("basis_state_round_commit_accepted", 0)
    closure_search_stats.setdefault("basis_state_round_commit_selected", 0)
    closure_search_stats.setdefault("basis_state_round_commit_selected_singleton", 0)
    closure_search_stats.setdefault("basis_state_round_commit_selected_pair", 0)
    closure_search_stats.setdefault("basis_state_controller_mode", "iterative_basis_loop")
    closure_search_stats.setdefault("basis_state_controller_stop_reason", "")
    closure_search_stats.setdefault("basis_state_accept_attempts", 0)
    closure_search_stats.setdefault("basis_state_accept_accepted", 0)
    closure_search_stats.setdefault("basis_state_accept_rejected", 0)
    closure_search_stats.setdefault("basis_state_accept_status_counts", {})
    closure_search_stats.setdefault("basis_state_fasttrack_candidates", 0)
    closure_search_stats.setdefault("basis_state_fasttrack_scored", 0)
    closure_search_stats.setdefault("basis_state_core_lane_reservations", 0)
    closure_search_stats.setdefault("basis_state_seed_rounds", 0)
    closure_search_stats.setdefault("basis_state_seed_scored", 0)
    closure_search_stats.setdefault("basis_state_seed_mode_used", False)
    closure_search_stats.setdefault("basis_state_seed_exact_topk", int(seed_exact_topk_i))
    closure_search_stats.setdefault("basis_state_seed_beam_width", int(seed_beam_width_i))
    closure_search_stats.setdefault("basis_state_seed_scaffold_reserve", int(seed_scaffold_reserve_i))
    closure_search_stats.setdefault("basis_state_seed_family_cap", int(seed_family_cap_i))
    closure_search_stats.setdefault("basis_state_seed_exact_bound_bonus", float(seed_exact_bound_bonus_f))
    closure_search_stats.setdefault("basis_state_seed_family_reservations", 0)
    closure_search_stats.setdefault("basis_state_variant_reservations", 0)
    closure_search_stats.setdefault("basis_state_interaction_reservations", 0)
    closure_search_stats.setdefault("basis_state_interaction_anchor_reservations", 0)
    closure_search_stats.setdefault("basis_state_family_reservations", 0)
    closure_search_stats.setdefault("basis_state_spec_reservations", 0)
    closure_search_stats.setdefault("basis_state_pair_normal_enable", bool(pair_normal_enable_b))
    closure_search_stats.setdefault("basis_state_pair_normal_topk", int(pair_normal_topk_i))
    closure_search_stats.setdefault("basis_state_pair_normal_max_pairs", int(pair_normal_max_pairs_i))
    closure_search_stats.setdefault("basis_state_pair_normal_rounds", 0)
    closure_search_stats.setdefault("basis_state_pair_normal_candidates", 0)
    closure_search_stats.setdefault("basis_state_pair_normal_scored", 0)
    closure_search_stats.setdefault("basis_state_pair_normal_accepted", 0)
    closure_search_stats.setdefault("basis_state_pair_rescue_enable", bool(pair_rescue_enable_b))
    closure_search_stats.setdefault("basis_state_pair_rescue_topk", int(pair_rescue_topk_i))
    closure_search_stats.setdefault("basis_state_pair_rescue_max_pairs", int(pair_rescue_max_pairs_i))
    closure_search_stats.setdefault("basis_state_pair_rescue_rounds", 0)
    closure_search_stats.setdefault("basis_state_pair_rescue_candidates", 0)
    closure_search_stats.setdefault("basis_state_pair_rescue_scored", 0)
    closure_search_stats.setdefault("basis_state_pair_rescue_accepted", 0)
    closure_search_stats.setdefault("basis_state_pair_precommit_rounds", 0)
    closure_search_stats.setdefault("basis_state_pair_precommit_candidates", 0)
    closure_search_stats.setdefault("basis_state_pair_precommit_scored", 0)
    closure_search_stats.setdefault("basis_state_pair_precommit_accepted", 0)
    closure_search_stats.setdefault("accepted_pair_events", [])
    closure_search_stats.setdefault("selected_pair_events", [])
    closure_search_stats.setdefault("emergent_basis_enable", bool(emergent_basis_enable_b))
    closure_search_stats.setdefault("emergent_basis_calls", 0)
    closure_search_stats.setdefault("emergent_basis_source_rows", 0)
    closure_search_stats.setdefault("emergent_basis_sightings", 0)
    closure_search_stats.setdefault("emergent_basis_unique", 0)
    closure_search_stats.setdefault("emergent_basis_scored", 0)
    closure_search_stats.setdefault("emergent_basis_rows", 0)
    closure_search_stats.setdefault("emergent_basis_rounds_with_rows", 0)
    closure_search_stats.setdefault("emergent_basis_reject_counts", {})
    closure_search_stats.setdefault("emergent_aux_atoms_enable", bool(emergent_aux_atoms_enable_b))
    closure_search_stats.setdefault("emergent_aux_atom_calls", 0)
    closure_search_stats.setdefault("emergent_aux_atom_source_rows", 0)
    closure_search_stats.setdefault("emergent_aux_atom_sightings", 0)
    closure_search_stats.setdefault("emergent_aux_atom_unique", 0)
    closure_search_stats.setdefault("emergent_aux_atom_accepted", 0)
    closure_search_stats.setdefault("emergent_aux_atom_by_kind", {})
    closure_search_stats.setdefault("emergent_aux_atom_registry_size", 0)
    closure_search_stats.setdefault("emergent_aux_atom_registry_new", 0)
    closure_search_stats.setdefault("emergent_aux_atom_registry", [])
    closure_search_stats.setdefault("emergent_aux_atom_registry_by_bucket", {})
    closure_search_stats.setdefault("emergent_aux_atom_registry_not_retained", [])
    closure_search_stats.setdefault("emergent_aux_atom_observed_bucket_counts", {})
    closure_search_stats.setdefault("emergent_aux_atom_observed_top", [])
    closure_search_stats.setdefault("emergent_aux_atom_seen_not_retained", [])
    closure_search_stats.setdefault("emergent_aux_atom_observation_pool_size", 0)
    closure_search_stats.setdefault("emergent_aux_atom_observation_pool", [])
    closure_search_stats.setdefault("emergent_aux_atom_followup_reserved", 0)
    closure_search_stats.setdefault("emergent_aux_atom_followup_budget", 0)
    closure_search_stats.setdefault("emergent_aux_atom_followup_unlocked", False)
    closure_search_stats.setdefault("emergent_aux_atom_seed_blocks", 0)
    closure_search_stats.setdefault("emergent_aux_atom_seed_exprs", [])
    closure_search_stats.setdefault("emergent_aux_atom_rounds_with_new", 0)
    closure_search_stats.setdefault("emergent_aux_atom_reject_counts", {})
    closure_search_stats.setdefault("atom_policy_library_records", 0)
    closure_search_stats.setdefault("atom_policy_library_relations", 0)
    closure_search_stats.setdefault("atom_policy_source_atoms", 0)
    closure_search_stats.setdefault("atom_policy_family_scores", {})
    closure_search_stats.setdefault("atom_policy_library", {})
    closure_search_stats.setdefault("atomized_linear_span_enable", False)
    closure_search_stats.setdefault("atomized_linear_span_budget", 0)
    closure_search_stats.setdefault("atomized_linear_span_calls", 0)
    closure_search_stats.setdefault("atomized_linear_span_source_atoms", 0)
    closure_search_stats.setdefault("atomized_linear_span_structural_seed_mode", "topup")
    closure_search_stats.setdefault("atomized_linear_span_structural_seed_enabled", False)
    closure_search_stats.setdefault("atomized_linear_span_pre_seed_target_atoms", 0)
    closure_search_stats.setdefault("atomized_linear_span_pre_seed_dimless_atoms", 0)
    closure_search_stats.setdefault("atomized_linear_span_target_atoms", 0)
    closure_search_stats.setdefault("atomized_linear_span_dimensionless_atoms", 0)
    closure_search_stats.setdefault("atomized_linear_span_terms", 0)
    closure_search_stats.setdefault("atomized_linear_span_product_terms", 0)
    closure_search_stats.setdefault("atomized_linear_span_candidates", 0)
    closure_search_stats.setdefault("atomized_linear_span_coverage_candidates", 0)
    closure_search_stats.setdefault("atomized_linear_span_scored", 0)
    closure_search_stats.setdefault("atomized_linear_span_rows", 0)
    closure_search_stats.setdefault("atomized_linear_span_best_probe", math.inf)
    closure_search_stats["atomized_linear_span_enable"] = bool(atomized_linear_span_enable_b)
    closure_search_stats["atomized_linear_span_budget"] = int(atomized_linear_span_budget_i)
    closure_search_stats["atomized_linear_span_use_obs_pool"] = bool(atomized_linear_span_use_obs_pool_b)
    closure_search_stats["atomized_linear_span_same_round"] = bool(atomized_linear_span_same_round_b)
    closure_search_stats["atomized_linear_span_exact_quota"] = int(atomized_linear_span_exact_quota_i)
    closure_search_stats["emergent_aux_atom_followup_budget"] = int(emergent_aux_atoms_followup_budget_i)
    closure_search_stats.setdefault("atomized_linear_span_exact_scored", 0)
    closure_search_stats.setdefault("atomized_linear_span_exact_quota_skipped", 0)
    closure_search_stats["atom_policy_use_obs_pool"] = bool(atom_policy_use_obs_pool_b)
    closure_search_stats.setdefault("aux_scaffolds_enumerated", 0)
    closure_search_stats.setdefault("protected_aux_scaffolds_enumerated", 0)
    closure_search_stats.setdefault("basis_state_rank_proxy_mode", "conditional_gain")
    closure_search_stats.setdefault("basis_state_rank_proxy_candidates", 0)
    closure_search_stats.setdefault("basis_state_rank_proxy_scored", 0)
    closure_search_stats.setdefault("debug_topk", int(debug_topk_i))
    if debug_topk_i > 0:
        closure_search_stats.setdefault("debug_preview_rows", [])
        closure_search_stats.setdefault("debug_exact_rows", [])
        closure_search_stats.setdefault("debug_pair_pool", [])
        closure_search_stats.setdefault("debug_pair_attempts", [])
        closure_search_stats.setdefault("debug_round_summaries", [])
        closure_search_stats.setdefault("debug_emergent_aux_atoms", [])

    beam_cfg = {
        "min_valid_frac": float(closure_search_min_valid_frac),
        "min_confidence": float(closure_search_min_confidence),
        "periodic_min_valid_scale": float(closure_search_periodic_min_valid_scale),
        "periodic_min_confidence_scale": float(closure_search_periodic_min_confidence_scale),
        "periodic_path_penalty": float(inverse_periodic_path_penalty),
        "nonperiodic_muldiv_bonus": float(inverse_nonperiodic_muldiv_bonus),
        "nonperiodic_explogsqrt_bonus": float(inverse_nonperiodic_explogsqrt_bonus),
        "branch_beam_width": int(inverse_branch_beam_width),
        "topk_terms": int(inverse_topk_terms),
        "shortlist_mult": int(inverse_shortlist_mult),
        "local_mode": str(inverse_local_score_mode),
        "poly_degree": int(poly_degree),
        "max_depth": int(max_depth),
        "micro_search_enable": bool(inverse_micro_search_enable),
        "micro_search_max_depth": int(inverse_micro_search_max_depth),
        "micro_search_beam_width": int(inverse_micro_search_beam_width),
        "micro_search_topk": int(inverse_micro_search_topk),
        "micro_search_seed_terms": int(inverse_micro_search_seed_terms),
        "target_mode": str(inverse_target_mode),
        "safe_eps": float(inverse_safe_eps),
        "confidence_mode": str(inverse_confidence_mode),
        "confidence_target_gain": float(inverse_confidence_target_gain),
        "confidence_floor": float(inverse_confidence_floor),
        "full_mapping_penalty": float(inverse_full_mapping_penalty),
        "exact_simple_target_bonus": float(inverse_exact_simple_target_bonus),
        "additive_descend_penalty": float(inverse_additive_descend_penalty),
        "nonadditive_leaf_penalty": float(inverse_nonadditive_leaf_penalty),
        "exact_path_eta": float(inverse_exact_path_eta),
        "branch_ambiguity_penalty": float(inverse_branch_ambiguity_penalty),
        "transport_min_lin_rel": float(closure_search_transport_min_lin_rel),
        "transport_min_effective_n": float(inverse_transport_min_effective_n),
        "var_dims": var_dims,
        "dm": bool(dm),
        "debug_topk": int(debug_topk_i),
    }
    solver_kwargs = {
        "regime_metadata": inverse_spec_regime_metadata,
        "local_score_mode": str(inverse_spec_local_score_mode),
        "enum_max_depth": int(inverse_spec_enum_max_depth),
        "enum_max_trees": int(inverse_spec_enum_max_trees),
        "max_subtree_depth": inverse_spec_max_subtree_depth,
        "complexity_penalty": float(inverse_spec_complexity_penalty),
        "family_battery_enable": bool(inverse_spec_family_battery_enable),
        "family_battery_mode": str(inverse_spec_family_battery_mode or "outer"),
        "recursive_enable": bool(inverse_spec_recursive_enable),
        "recursive_max_depth": int(inverse_spec_recursive_max_depth),
        "recursive_trigger_rel_mse": float(inverse_spec_recursive_trigger_rel_mse),
        "recursive_seed_cap": int(inverse_spec_recursive_seed_cap),
        "recursive_branch_topk": int(inverse_spec_recursive_branch_topk),
        "recursive_child_topk": int(inverse_spec_recursive_child_topk),
        "safe_eps": float(inverse_safe_eps),
        "confidence_mode": str(inverse_confidence_mode),
        "confidence_target_gain": float(inverse_confidence_target_gain),
        "confidence_floor": float(inverse_confidence_floor),
        "branch_beam_width": int(inverse_branch_beam_width),
        "witness_jets_enable": bool(inverse_spec_witness_jets_enable),
        "witness_d2_enable": bool(inverse_spec_witness_d2_enable),
        "witness_max_rows": int(inverse_spec_witness_max_rows),
        "witness_loss_enable": bool(inverse_spec_witness_loss_enable),
        "witness_grad_weight": float(inverse_spec_witness_grad_weight),
        "witness_d2_weight": float(inverse_spec_witness_d2_weight),
        "witness_diag_weight": float(inverse_spec_witness_diag_weight),
        "witness_physics_weight": float(inverse_spec_witness_physics_weight),
        "active_var_screen_enable": bool(inverse_spec_active_var_screen_enable),
        "active_var_grad_tol": float(inverse_spec_active_var_grad_tol),
        "active_var_max_count": int(inverse_spec_active_var_max_count),
    }
    basis_beam: tuple[BasisState, ...] = ()
    if isinstance(proposal_context_local.basis_state, BasisState):
        basis_beam = admit_basis_state_to_beam(
            basis_beam,
            proposal_context_local.basis_state,
            beam_width=beam_width_i,
        )
    for row in tuple(getattr(proposal_context_local, "basis_state_beam", ()) or ()):
        if isinstance(row, BasisState):
            basis_beam = admit_basis_state_to_beam(
                basis_beam,
                row,
                beam_width=beam_width_i,
            )
    expanded_beam_state_ids: set[tuple[Any, ...]] = set()

    def _prune_expanded_beam_state_ids() -> None:
        nonlocal expanded_beam_state_ids
        live_ids = {
            state_id
            for state_id in (
                _basis_state_identity_key(state)
                for state in tuple(basis_beam or ())
                if isinstance(state, BasisState)
            )
            if state_id is not None
        }
        expanded_beam_state_ids.intersection_update(live_ids)

    def _select_round_basis_state() -> tuple[BasisState | None, bool]:
        _prune_expanded_beam_state_ids()
        for state in tuple(basis_beam or ()):
            if not isinstance(state, BasisState):
                continue
            state_id = _basis_state_identity_key(state)
            if state_id is None or state_id not in expanded_beam_state_ids:
                if state_id is not None:
                    expanded_beam_state_ids.add(state_id)
                return state, True
        if basis_beam:
            return basis_beam[0], False
        base_state = proposal_context_local.basis_state
        return base_state if isinstance(base_state, BasisState) else None, False

    def _has_unexpanded_beam_state() -> bool:
        _prune_expanded_beam_state_ids()
        for state in tuple(basis_beam or ()):
            if not isinstance(state, BasisState):
                continue
            state_id = _basis_state_identity_key(state)
            if state_id is None or state_id not in expanded_beam_state_ids:
                return True
        return False

    def _debug_append(key: str, row: Mapping[str, Any]) -> None:
        if debug_topk_i <= 0 or not isinstance(row, Mapping):
            return
        bucket = closure_search_stats.get(key, None)
        if not isinstance(bucket, list):
            bucket = []
            closure_search_stats[key] = bucket
        if len(bucket) >= int(debug_capture_limit_i):
            return
        bucket.append(dict(row))

    def _expr_debug_str(expr: Any) -> str:
        if isinstance(expr, tuple) and is_valid_node(expr):
            try:
                return str(node_str_fn(expr))
            except Exception:
                return str(expr)
        return str(expr)

    def _candidate_debug_row(
        row: Mapping[str, Any],
        *,
        round_idx: int,
        stage: str,
        proposal_key: str | None = None,
        sort_key: Any = None,
    ) -> dict[str, Any]:
        expr = row.get("expr", None) if isinstance(row, Mapping) else None
        out = {
            "round": int(round_idx),
            "stage": str(stage),
            "proposal_key": str(proposal_key or row.get("proposal_key", "") or row.get("child_key", "") or ""),
            "scaffold_id": str(row.get("scaffold_id", "") or ""),
            "family": str(row.get("scaffold_family", "") or row.get("proposal_family", "") or ""),
            "operator_id": str(row.get("operator_id", "") or ""),
            "spec_key": str(_candidate_spec_key(row)),
            "lane": str(_candidate_lane(row)),
            "exact_bound": bool(_candidate_is_exact_bound(row)),
            "fasttrack": bool(_candidate_is_fasttrack(row)),
            "child_size": int(row.get("candidate_child_size", 0) or 0),
            "local_probe_mse": float(_candidate_float(row, "local_probe_mse")),
            "local_fit_mse": float(_candidate_float(row, "local_fit_mse")),
            "expr": _expr_debug_str(expr),
            "interaction_key": str(_candidate_interaction_key(row)),
            "anchor_diversity_key": str(_candidate_anchor_diversity_key(row)),
        }
        if sort_key is not None:
            try:
                out["sort_key"] = [
                    float(sort_key[0]),
                    float(sort_key[1]),
                    float(sort_key[2]),
                    int(sort_key[3]),
                    int(sort_key[4]),
                    str(sort_key[5]),
                ]
            except Exception:
                out["sort_key"] = str(sort_key)
        return out

    def _aux_atom_observation_rank(atom: EmergentAtom) -> tuple[float, int, int, str]:
        try:
            score = float(atom.score)
        except Exception:
            score = 0.0
        kind_rank = {"target_term": 0, "carrier": 1, "dimensionless_feature": 2}.get(str(atom.kind), 3)
        try:
            size = int(node_size(atom.node))
        except Exception:
            size = 999
        try:
            key = str(node_str(atom.node))
        except Exception:
            key = str(atom.node)
        return (-score, int(kind_rank), int(size), key)

    def _merge_aux_atom_observation_pool(
        existing: Collection[EmergentAtom] | None,
        observed: Collection[EmergentAtom] | None,
        *,
        max_count: int,
    ) -> tuple[EmergentAtom, ...]:
        by_key: dict[str, EmergentAtom] = {}
        for atom in tuple(existing or ()) + tuple(observed or ()):
            if not isinstance(atom, EmergentAtom):
                continue
            try:
                key = str(node_str(atom.node))
            except Exception:
                continue
            current = by_key.get(key)
            if current is None or _aux_atom_observation_rank(atom) < _aux_atom_observation_rank(current):
                by_key[key] = atom
        return tuple(sorted(by_key.values(), key=_aux_atom_observation_rank)[: max(0, int(max_count))])

    def _atom_key(atom: EmergentAtom) -> str | None:
        if not isinstance(atom, EmergentAtom):
            return None
        try:
            return str(node_str(atom.node))
        except Exception:
            return None

    def _atom_origin_map(
        registry: Collection[EmergentAtom] | None,
        observed: Collection[EmergentAtom] | None,
        *,
        include_observed: bool,
    ) -> dict[str, str]:
        out: dict[str, str] = {}
        for atom in tuple(registry or ()):
            key = _atom_key(atom)
            if key:
                out[key] = "retained_registry"
        if include_observed:
            for atom in tuple(observed or ()):
                key = _atom_key(atom)
                if not key:
                    continue
                if key in out:
                    out[key] = "retained_registry+observed_pool"
                else:
                    out[key] = "observed_pool"
        return out

    def _atomized_source_atoms() -> tuple[EmergentAtom, ...]:
        observed = tuple(aux_atom_observation_pool or ()) if bool(atomized_linear_span_use_obs_pool_b) else ()
        return tuple(aux_atom_registry or ()) + observed

    def _candidate_sort_key(
        row: Mapping[str, Any],
        *,
        seed_mode: bool = False,
        current_basis_state: BasisState | None = None,
    ) -> tuple[float, float, float, int, int, str]:
        gain_rank = math.inf
        candidate_basis_state = row.get("basis_state_obj", None) if isinstance(row, Mapping) else None
        if isinstance(candidate_basis_state, BasisState):
            rank_ret = score_basis_state_conditional_gain(
                current_basis_state,
                candidate_basis_state,
                x_fit=x_fit,
                y_fit=y_fit,
            )
            if isinstance(rank_ret, Mapping):
                try:
                    gain_value = max(0.0, float(rank_ret.get("gain", 0.0) or 0.0))
                except Exception:
                    gain_value = 0.0
                if math.isfinite(gain_value):
                    gain_rank = -float(gain_value)
                    closure_search_stats["basis_state_rank_proxy_scored"] = int(
                        closure_search_stats.get("basis_state_rank_proxy_scored", 0) or 0
                    ) + 1
        probe_mse = _candidate_float(row, "local_probe_mse")
        fit_mse = _candidate_float(row, "local_fit_mse")
        if seed_mode and _candidate_is_exact_bound(row):
            scale = 1.0 / (1.0 + float(seed_exact_bound_bonus_f))
            probe_mse *= scale
            fit_mse *= scale
        try:
            child_size = int(row.get("candidate_child_size", 0) or 0)
        except Exception:
            child_size = 0
        lane_rank = 0 if _candidate_lane(row) == "core" else 1
        proposal_key = str(row.get("proposal_key", "") or row.get("child_key", "") or "")
        return (gain_rank, probe_mse, fit_mse, lane_rank, child_size, proposal_key)

    def _record_acceptance_meta(accept_meta: Mapping[str, Any]) -> None:
        closure_search_stats["basis_state_accept_attempts"] = int(
            closure_search_stats.get("basis_state_accept_attempts", 0) or 0
        ) + 1
        accept_counts = closure_search_stats.get("basis_state_accept_status_counts", None)
        if not isinstance(accept_counts, dict):
            accept_counts = {}
            closure_search_stats["basis_state_accept_status_counts"] = accept_counts
        reason_key = str(accept_meta.get("reason", "") or "unknown")
        accept_counts[reason_key] = int(accept_counts.get(reason_key, 0) or 0) + 1
        closure_search_stats["last_basis_state_acceptance"] = dict(accept_meta)
        if bool(accept_meta.get("accept", False)):
            closure_search_stats["basis_state_accept_accepted"] = int(
                closure_search_stats.get("basis_state_accept_accepted", 0) or 0
            ) + 1
        else:
            closure_search_stats["basis_state_accept_rejected"] = int(
                closure_search_stats.get("basis_state_accept_rejected", 0) or 0
            ) + 1

    def _record_round_commit(round_commit: RoundCommitObject) -> None:
        closure_search_stats["basis_state_round_commit_scored"] = int(
            closure_search_stats.get("basis_state_round_commit_scored", 0) or 0
        ) + 1
        if bool(round_commit.accepted):
            closure_search_stats["basis_state_round_commit_accepted"] = int(
                closure_search_stats.get("basis_state_round_commit_accepted", 0) or 0
            ) + 1

    def _preview_round_commit_beam(
        round_commit: RoundCommitObject | None,
        *,
        seed_row_mode: bool,
        round_beam_width: int,
    ) -> tuple[BasisState, ...] | None:
        if not isinstance(round_commit, RoundCommitObject):
            return None
        if not bool(round_commit.accepted) or not isinstance(round_commit.basis_state, BasisState):
            return None
        beam_sig_before = _basis_beam_signature(basis_beam)
        if seed_row_mode:
            updated_beam = _admit_seed_basis_state_to_beam(
                basis_beam,
                round_commit.basis_state,
                beam_width=round_beam_width,
                family_cap=seed_family_cap_i,
            )
        else:
            updated_beam = _admit_basis_state_to_beam_preserving_unexpanded(
                basis_beam,
                round_commit.basis_state,
                beam_width=round_beam_width,
                expanded_state_ids=expanded_beam_state_ids,
            )
        updated_beam_tuple = tuple(updated_beam or ())
        if _basis_beam_signature(updated_beam_tuple) == beam_sig_before:
            return None
        return updated_beam_tuple

    def _round_commit_priority_key(round_commit: RoundCommitObject) -> tuple[int, float, float, int, str]:
        return (
            int(round_commit.priority_rank),
            float(round_commit.eff_mse),
            float(round_commit.complexity),
            0 if str(round_commit.kind) == "singleton" else 1,
            str(round_commit.proposal_key),
        )

    def _pair_rescue_trigger_reasons(
        *,
        selected_round_commit: RoundCommitObject | None,
        early_round_commits: list[RoundCommitObject] | tuple[RoundCommitObject, ...],
        best_singleton: Mapping[str, Any] | None,
        base_basis_state: BasisState | None,
        seed_row_mode: bool,
    ) -> list[str]:
        if bool(seed_row_mode):
            if selected_round_commit is None:
                return ["no_early_commit"]
            return []
        if selected_round_commit is None:
            return ["no_early_commit"]
        if not isinstance(selected_round_commit, RoundCommitObject):
            return []
        if str(selected_round_commit.kind) != "singleton":
            return []

        reasons: list[str] = []
        selected_eff_mse = math.inf
        try:
            selected_eff_mse = float(selected_round_commit.eff_mse)
        except Exception:
            selected_eff_mse = math.inf

        base_probe = math.inf
        if isinstance(base_basis_state, BasisState):
            try:
                base_probe = float(getattr(base_basis_state, "probe_loss", math.inf))
            except Exception:
                base_probe = math.inf
        if math.isfinite(base_probe) and math.isfinite(selected_eff_mse):
            gain_vs_current = float(base_probe - selected_eff_mse)
            weak_gain_slack = _mse_slack(
                base_value=base_probe,
                rel_slack=float(_PAIR_RESCUE_WEAK_GAIN_REL_SLACK),
                abs_slack=float(_PAIR_RESCUE_WEAK_GAIN_ABS_SLACK),
            )
            if gain_vs_current <= float(weak_gain_slack):
                reasons.append("weak_singleton_gain")

        best_singleton_eff_mse = math.inf
        if isinstance(best_singleton, Mapping):
            try:
                best_singleton_eff_mse = float(best_singleton.get("eff_mse", math.inf))
            except Exception:
                best_singleton_eff_mse = math.inf
        if math.isfinite(best_singleton_eff_mse) and math.isfinite(selected_eff_mse):
            weaker_singleton_slack = _mse_slack(
                base_value=best_singleton_eff_mse,
                rel_slack=float(_PAIR_RESCUE_CLOSE_COMMIT_REL_SLACK),
                abs_slack=float(_PAIR_RESCUE_CLOSE_COMMIT_ABS_SLACK),
            )
            if selected_eff_mse > float(best_singleton_eff_mse) + float(weaker_singleton_slack):
                reasons.append("priority_selected_weaker_singleton")

        best_close_pair_eff_mse = math.inf
        for round_commit in tuple(early_round_commits or ()):
            if not isinstance(round_commit, RoundCommitObject):
                continue
            if str(round_commit.kind) != "pair" or not bool(round_commit.accepted):
                continue
            try:
                pair_eff_mse = float(round_commit.eff_mse)
            except Exception:
                pair_eff_mse = math.inf
            if not math.isfinite(pair_eff_mse):
                continue
            best_close_pair_eff_mse = min(best_close_pair_eff_mse, pair_eff_mse)
        if math.isfinite(best_close_pair_eff_mse) and math.isfinite(selected_eff_mse):
            close_pair_slack = _mse_slack(
                base_value=selected_eff_mse,
                rel_slack=float(_PAIR_RESCUE_CLOSE_COMMIT_REL_SLACK),
                abs_slack=float(_PAIR_RESCUE_CLOSE_COMMIT_ABS_SLACK),
            )
            if best_close_pair_eff_mse <= float(selected_eff_mse) + float(close_pair_slack):
                reasons.append("close_accepted_pair")

        return reasons

    def _select_round_commit(
        round_commits: list[RoundCommitObject] | tuple[RoundCommitObject, ...],
        *,
        seed_row_mode: bool,
        round_beam_width: int,
    ) -> tuple[RoundCommitObject | None, tuple[BasisState, ...] | None]:
        viable_commits: list[tuple[RoundCommitObject, tuple[BasisState, ...]]] = []
        for round_commit in tuple(round_commits or ()):
            if not isinstance(round_commit, RoundCommitObject) or not bool(round_commit.accepted):
                continue
            updated_beam = _preview_round_commit_beam(
                round_commit,
                seed_row_mode=seed_row_mode,
                round_beam_width=round_beam_width,
            )
            if updated_beam is None:
                continue
            viable_commits.append((round_commit, updated_beam))
        if not viable_commits:
            return None, None
        best_eff_mse = min(float(commit.eff_mse) for commit, _beam in viable_commits)
        eff_mse_slack = _mse_slack(
            base_value=float(best_eff_mse),
            rel_slack=float(_ROUND_COMMIT_EFF_MSE_REL_SLACK),
            abs_slack=float(_ROUND_COMMIT_EFF_MSE_ABS_SLACK),
        )
        shortlisted_commits = [
            (commit, updated_beam)
            for commit, updated_beam in viable_commits
            if float(commit.eff_mse) <= float(best_eff_mse) + float(eff_mse_slack)
        ]
        best_commit, best_updated_beam = min(
            shortlisted_commits,
            key=lambda item: _round_commit_priority_key(item[0]),
        )
        return best_commit, best_updated_beam

    def _pair_event_record(round_commit: RoundCommitObject) -> dict[str, Any] | None:
        if not isinstance(round_commit, RoundCommitObject):
            return None
        if str(round_commit.kind) != "pair" or not isinstance(round_commit.basis_state, BasisState):
            return None
        pair_entries = round_commit.pair_entries
        if not isinstance(pair_entries, tuple) or len(pair_entries) != 2:
            return None
        entry_a, entry_b = pair_entries
        return {
            "round": int(closure_search_stats.get("closure_search_rounds", 0) or 0),
            "profile": str(round_commit.profile),
            "pair_template": str(round_commit.pair_template),
            "proposal_key": str(round_commit.proposal_key),
            "pair_members": [
                str(entry_a.get("proposal_key", "") or ""),
                str(entry_b.get("proposal_key", "") or ""),
            ],
            "pair_member_families": [
                str(entry_a.get("family", "") or ""),
                str(entry_b.get("family", "") or ""),
            ],
            "pair_member_specs": [
                str(entry_a.get("spec_key", "") or ""),
                str(entry_b.get("spec_key", "") or ""),
            ],
            "pair_member_sources": list(round_commit.source_members or ()),
            "pair_member_supports": [
                str(entry_a.get("support_key", "") or ""),
                str(entry_b.get("support_key", "") or ""),
            ],
            "pair_member_anchors": [
                str(entry_a.get("anchor_key", "") or ""),
                str(entry_b.get("anchor_key", "") or ""),
            ],
            "pair_member_interactions": [
                str(entry_a.get("interaction_key", "") or ""),
                str(entry_b.get("interaction_key", "") or ""),
            ],
            "relation_tags": list(round_commit.relation_tags or ()),
            "pair_eff_mse": float(round_commit.eff_mse),
            "pair_gain_over_best_singleton": float(round_commit.gain_vs_best_singleton),
            "best_singleton_key": str(round_commit.best_singleton_key),
            "best_singleton_eff_mse": float(round_commit.best_singleton_eff_mse),
            "best_singleton_source": str(round_commit.best_singleton_source),
            "expr": _expr_debug_str(round_commit.basis_state.compiled_expr),
        }

    def _append_pair_event(stats_key: str, round_commit: RoundCommitObject) -> None:
        pair_event = _pair_event_record(round_commit)
        if not isinstance(pair_event, dict):
            return
        pair_events = closure_search_stats.get(stats_key, None)
        if not isinstance(pair_events, list):
            pair_events = []
            closure_search_stats[stats_key] = pair_events
        pair_events.append(pair_event)

    def _append_selected_pair_event(round_commit: RoundCommitObject) -> None:
        _append_pair_event("selected_pair_events", round_commit)

    def _append_accepted_pair_event(round_commit: RoundCommitObject) -> None:
        _append_pair_event("accepted_pair_events", round_commit)

    def _select_pair_pool_records(
        pair_pool_entries: list[dict[str, Any]],
        *,
        topk: int,
    ) -> list[dict[str, Any]]:
        pair_pool: list[dict[str, Any]] = []
        selected_keys: set[str] = set()
        family_counts: dict[str, int] = {}
        selected_specs: set[str] = set()

        def _select(*, unique_spec_only: bool, unique_family_only: bool) -> None:
            for entry in pair_pool_entries:
                if len(pair_pool) >= int(topk):
                    break
                proposal_key_local = str(entry.get("proposal_key", "") or "")
                if proposal_key_local in selected_keys:
                    continue
                spec_key_local = str(entry.get("spec_key", "") or "")
                if unique_spec_only and spec_key_local and spec_key_local in selected_specs:
                    continue
                family_name_local = str(entry.get("family", "") or "")
                family_count = int(family_counts.get(family_name_local, 0) or 0)
                if unique_family_only and family_name_local and family_count > 0:
                    continue
                if family_name_local and family_count >= 2:
                    continue
                pair_pool.append(entry)
                selected_keys.add(proposal_key_local)
                if spec_key_local:
                    selected_specs.add(spec_key_local)
                if family_name_local:
                    family_counts[family_name_local] = family_count + 1

        _select(unique_spec_only=False, unique_family_only=True)
        _select(unique_spec_only=True, unique_family_only=False)
        _select(unique_spec_only=False, unique_family_only=False)
        return pair_pool

    def _select_rescue_pair_pool_records(
        pair_pool_entries: list[dict[str, Any]],
        *,
        topk: int,
    ) -> list[dict[str, Any]]:
        target_companion_count = max(0, int(topk))
        exact_entries: list[dict[str, Any]] = []
        companion_entries: list[dict[str, Any]] = []
        seen_exact_keys: set[str] = set()
        for entry in pair_pool_entries:
            proposal_key_local = str(entry.get("proposal_key", "") or "")
            if str(entry.get("source_pool", "") or "") == "exact_scored_singleton":
                if proposal_key_local in seen_exact_keys:
                    continue
                exact_entries.append(entry)
                seen_exact_keys.add(proposal_key_local)
            else:
                companion_entries.append(entry)

        if not exact_entries:
            return _select_pair_pool_records(pair_pool_entries, topk=int(topk))

        pair_pool: list[dict[str, Any]] = list(exact_entries)
        selected_keys: set[str] = set(seen_exact_keys)

        def _append_selected(entries: list[dict[str, Any]]) -> None:
            for entry in entries:
                proposal_key_local = str(entry.get("proposal_key", "") or "")
                if proposal_key_local in selected_keys:
                    continue
                pair_pool.append(entry)
                selected_keys.add(proposal_key_local)

        _append_selected(
            _select_pair_pool_records(companion_entries, topk=int(target_companion_count))
        )

        if len(pair_pool) < len(exact_entries) + int(target_companion_count):
            leftovers = [
                entry
                for entry in pair_pool_entries
                if str(entry.get("proposal_key", "") or "") not in selected_keys
            ]
            _append_selected(
                _select_pair_pool_records(
                    leftovers,
                    topk=(len(exact_entries) + int(target_companion_count) - len(pair_pool)),
                )
            )
        return pair_pool

    def _run_pair_profile(
        *,
        profile: PairProfile,
        base_basis_state: BasisState | None,
        candidate_items: list[tuple[Any, Mapping[str, Any], str]],
        seed_row_mode: bool,
        round_beam_width: int,
        best_singleton: Mapping[str, Any] | None,
        round_singleton_metrics: Mapping[str, Mapping[str, Any]] | None = None,
        blocked_pair_ids: Collection[tuple[str, str]] | None = None,
        attempted_pair_ids_sink: set[tuple[str, str]] | None = None,
        allow_covered_feature_entries: bool = False,
        allow_preview_materialization: bool = False,
    ) -> list[RoundCommitObject]:
        if (not bool(profile.enabled)) or int(profile.topk) <= 1 or int(profile.max_pairs) <= 0:
            return []

        pair_entries: list[dict[str, Any]] = []
        for idx, (sort_key, row, proposal_key) in enumerate(candidate_items):
            if str(profile.source_pool_mode) == "singleton_metrics":
                source_pool = (
                    "exact_scored_singleton"
                    if str(proposal_key) in dict(round_singleton_metrics or {})
                    else "prioritized_only"
                )
            else:
                source_pool = str(
                    row.get("_round_source_pool", "exact_scored_singleton") if isinstance(row, Mapping) else "exact_scored_singleton"
                )
            entry = _build_pair_entry_record(
                base_basis_state=base_basis_state,
                row=row,
                sort_key=sort_key,
                idx=int(idx),
                proposal_key=str(proposal_key),
                source_pool=source_pool,
                require_interaction_support=bool(profile.require_interaction_support),
                allow_covered_feature=bool(allow_covered_feature_entries),
                allow_preview_materialization=bool(allow_preview_materialization),
            )
            if isinstance(entry, dict):
                pair_entries.append(entry)

        pair_entries.sort(key=_pair_entry_record_sort_key)
        candidate_count = int(len(pair_entries))
        pair_source_entries = pair_entries
        if bool(profile.use_pool_selection):
            if bool(allow_covered_feature_entries):
                pair_source_entries = _select_rescue_pair_pool_records(
                    pair_entries,
                    topk=int(profile.topk),
                )
            else:
                pair_source_entries = _select_pair_pool_records(
                    pair_entries,
                    topk=int(profile.topk),
                )
            candidate_count = int(len(pair_source_entries))
        closure_search_stats[f"{profile.stats_prefix}_candidates"] = int(
            closure_search_stats.get(f"{profile.stats_prefix}_candidates", 0) or 0
        ) + int(candidate_count)
        if len(pair_source_entries) < 2:
            return []

        closure_search_stats[f"{profile.stats_prefix}_rounds"] = int(
            closure_search_stats.get(f"{profile.stats_prefix}_rounds", 0) or 0
        ) + 1

        if debug_topk_i > 0:
            debug_round = int(closure_search_stats.get("closure_search_rounds", 0) or 0)
            for entry in pair_source_entries[: int(debug_topk_i)]:
                row_dbg = entry.get("row", None)
                if not isinstance(row_dbg, Mapping):
                    continue
                dbg = _candidate_debug_row(
                    row_dbg,
                    round_idx=debug_round,
                    stage=str(profile.debug_pool_stage),
                    proposal_key=str(entry.get("proposal_key", "") or ""),
                    sort_key=entry.get("sort_key", None),
                )
                dbg["interaction_key"] = str(entry.get("interaction_key", "") or dbg.get("interaction_key", ""))
                dbg["anchor_diversity_key"] = str(entry.get("anchor_key", "") or dbg.get("anchor_diversity_key", ""))
                _debug_append("debug_pair_pool", dbg)

        pair_candidates = _build_pair_candidate_records(
            pair_source_entries,
            template_kinds=tuple(profile.template_kinds or ()),
            blocked_pair_ids=blocked_pair_ids,
            require_exact_scored_member=bool(allow_covered_feature_entries),
        )
        if not pair_candidates:
            return []

        seen_pair_states: set[tuple[Any, ...]] = set()
        pair_attempts = 0
        pair_commit_objects: list[RoundCommitObject] = []
        best_singleton_key = str(best_singleton.get("proposal_key", "") if isinstance(best_singleton, Mapping) else "")
        best_singleton_eff_mse = float(
            best_singleton.get("eff_mse", math.inf) if isinstance(best_singleton, Mapping) else math.inf
        )
        best_singleton_source = str(best_singleton.get("source_pool", "") if isinstance(best_singleton, Mapping) else "")
        for pair_candidate in pair_candidates:
            if wall_time_exceeded_fn():
                break
            if pair_attempts >= int(profile.max_pairs):
                break
            key_a = str(pair_candidate.get("key_a", "") or "")
            key_b = str(pair_candidate.get("key_b", "") or "")
            pair_member_id = _pair_member_id(key_a, key_b)
            entry_a = pair_candidate.get("entry_a", None)
            entry_b = pair_candidate.get("entry_b", None)
            template_kind = str(pair_candidate.get("template_kind", "") or "")
            if not isinstance(entry_a, Mapping) or not isinstance(entry_b, Mapping):
                continue
            row_a = entry_a.get("row", None)
            row_b = entry_b.get("row", None)
            state_a = row_a.get("basis_state_obj", None) if isinstance(row_a, Mapping) else None
            state_b = row_b.get("basis_state_obj", None) if isinstance(row_b, Mapping) else None
            if not isinstance(state_a, BasisState) or not isinstance(state_b, BasisState):
                continue
            pair_preview = prepare_basis_state_candidate(
                base_basis_state,
                state_a,
                route_name=str(profile.route_prepare_a),
                compiled_expr=row_a.get("expr", None) if isinstance(row_a, Mapping) else None,
            )
            pair_preview = prepare_basis_state_candidate(
                pair_preview,
                state_b,
                route_name=str(profile.route_prepare_b),
                compiled_expr=row_b.get("expr", None) if isinstance(row_b, Mapping) else None,
            )
            if not isinstance(pair_preview, BasisState):
                continue
            pair_identity = _basis_state_identity_key(pair_preview)
            if pair_identity is not None and pair_identity in seen_pair_states:
                continue
            if pair_identity is not None:
                seen_pair_states.add(pair_identity)
            pair_attempts += 1
            if isinstance(attempted_pair_ids_sink, set):
                attempted_pair_ids_sink.add(pair_member_id)
            pair_key = f"{profile.pair_key_prefix}::{key_a}++{key_b}"
            pair_meta = {
                "basis_state_obj": pair_preview,
                "basis_state_dict": pair_preview.to_dict(),
                "proposal_key": str(pair_key),
                "child_key": str(pair_key),
                "scaffold_id": str(pair_key),
                "proposal_family": str(profile.proposal_family),
                "scaffold_family": str(profile.proposal_family),
                "candidate_child_size": max(
                    int(row_a.get("candidate_child_size", 0) or 0) if isinstance(row_a, Mapping) else 0,
                    int(row_b.get("candidate_child_size", 0) or 0) if isinstance(row_b, Mapping) else 0,
                ),
                f"{profile.pair_key_prefix}_members": [str(key_a), str(key_b)],
            }
            if str(profile.name) == "seed_precommit":
                pair_meta["pair_precommit_interaction_key"] = str(entry_a.get("interaction_key", "") or "")
            elif str(profile.name) == "stall_expanded":
                pair_meta["pair_rescue_member_families"] = [
                    str(row_a.get("scaffold_family", "") or row_a.get("proposal_family", "") or ""),
                    str(row_b.get("scaffold_family", "") or row_b.get("proposal_family", "") or ""),
                ]
            score_ret = None
            if callable(score_native_candidate_basis_state_fn):
                score_ret = score_native_candidate_basis_state_fn(
                    candidate_meta=pair_meta,
                    stats=closure_search_stats,
                    route_name=str(profile.route_score),
                )
            if score_ret is None:
                pair_expr = pair_preview.compiled_expr
                if not (isinstance(pair_expr, tuple) and is_valid_node(pair_expr)):
                    continue
                score_ret = score_external_candidate_expr_fn(
                    pair_expr,
                    parent_raw_mse=None,
                    stats=closure_search_stats,
                    route_name=str(profile.route_score),
                    candidate_meta=pair_meta,
                )
            closure_search_stats[f"{profile.stats_prefix}_scored"] = int(
                closure_search_stats.get(f"{profile.stats_prefix}_scored", 0) or 0
            ) + 1
            candidate_basis_state = _basis_state_from_scored_result(
                score_ret=score_ret,
                candidate_meta=pair_meta,
                current_basis_state=base_basis_state,
                route_name=str(profile.route_score),
            )
            if isinstance(candidate_basis_state, BasisState):
                refit_admitted_state = fit_basis_state_head(
                    candidate_basis_state,
                    x_fit=x_fit,
                    y_fit=y_fit,
                    x_probe=x_probe,
                    y_probe=y_probe,
                    route_name=str(profile.route_refit),
                )
                if isinstance(refit_admitted_state, BasisState):
                    candidate_basis_state = refit_admitted_state
            accept_meta: dict[str, Any] = {"accept": False, "reason": "missing_candidate_basis_state"}
            if isinstance(candidate_basis_state, BasisState):
                accept_meta = _basis_state_acceptance_decision(
                    current_basis_state=base_basis_state,
                    candidate_basis_state=candidate_basis_state,
                    candidate_meta=pair_meta,
                    family_hints=getattr(proposal_context_local, "family_hints", {}) if isinstance(proposal_context_local, ProposalContext) else {},
                    complexity_penalty=float(inverse_spec_complexity_penalty),
                    route_name=str(profile.route_score),
                )
                _record_acceptance_meta(accept_meta)
            accepted = bool(accept_meta.get("accept", False))
            if accepted:
                closure_search_stats[f"{profile.stats_prefix}_accepted"] = int(
                    closure_search_stats.get(f"{profile.stats_prefix}_accepted", 0) or 0
                ) + 1
            debug_pair_row = {
                "round": int(closure_search_stats.get("closure_search_rounds", 0) or 0),
                "stage": str(profile.debug_attempt_stage),
                "profile": str(profile.name),
                "pair_template": str(template_kind),
                "proposal_key": str(pair_key),
                "pair_members": [str(key_a), str(key_b)],
                "pair_member_families": [
                    str(entry_a.get("family", "") or ""),
                    str(entry_b.get("family", "") or ""),
                ],
                "pair_member_specs": [
                    str(entry_a.get("spec_key", "") or ""),
                    str(entry_b.get("spec_key", "") or ""),
                ],
                "pair_member_sources": [
                    str(entry_a.get("source_pool", "") or ""),
                    str(entry_b.get("source_pool", "") or ""),
                ],
                "pair_member_supports": [
                    str(entry_a.get("support_key", "") or ""),
                    str(entry_b.get("support_key", "") or ""),
                ],
                "pair_member_anchors": [
                    str(entry_a.get("anchor_key", "") or ""),
                    str(entry_b.get("anchor_key", "") or ""),
                ],
                "pair_member_interactions": [
                    str(entry_a.get("interaction_key", "") or ""),
                    str(entry_b.get("interaction_key", "") or ""),
                ],
                "interaction_key": str(entry_a.get("interaction_key", "") or ""),
                "relation_tags": list(_pair_relation_tags(entry_a, entry_b)),
                "best_singleton_key": str(best_singleton_key),
                "best_singleton_eff_mse": float(best_singleton_eff_mse),
                "best_singleton_source": str(best_singleton_source),
                "pair_gain_over_best_singleton": math.nan,
                "accepted": bool(accepted),
                "reason": str(accept_meta.get("reason", "") or "missing_candidate_basis_state"),
                "expr": _expr_debug_str(pair_preview.compiled_expr),
            }
            if isinstance(score_ret, Mapping):
                try:
                    debug_pair_row["eff_mse"] = float(score_ret.get("eff_mse", math.inf))
                except Exception:
                    debug_pair_row["eff_mse"] = math.inf
            if isinstance(candidate_basis_state, BasisState):
                debug_pair_row["expr"] = _expr_debug_str(candidate_basis_state.compiled_expr)
                debug_pair_row["eff_mse"] = float(getattr(candidate_basis_state, "probe_loss", math.inf))
                debug_pair_row["candidate_complexity"] = float(
                    getattr(candidate_basis_state, "complexity", 0.0) or 0.0
                )
            if math.isfinite(float(debug_pair_row.get("best_singleton_eff_mse", math.inf))) and math.isfinite(float(debug_pair_row.get("eff_mse", math.inf))):
                debug_pair_row["pair_gain_over_best_singleton"] = float(
                    float(debug_pair_row["best_singleton_eff_mse"]) - float(debug_pair_row["eff_mse"])
                )
            _debug_append("debug_pair_attempts", debug_pair_row)
            pair_commit = RoundCommitObject(
                kind="pair",
                profile=str(profile.name),
                proposal_key=str(pair_key),
                basis_state=candidate_basis_state if isinstance(candidate_basis_state, BasisState) else None,
                eff_mse=float(
                    getattr(candidate_basis_state, "probe_loss", debug_pair_row.get("eff_mse", math.inf))
                )
                if isinstance(candidate_basis_state, BasisState)
                else float(debug_pair_row.get("eff_mse", math.inf)),
                accepted=bool(accepted),
                accept_reason=str(accept_meta.get("reason", "") or "missing_candidate_basis_state"),
                complexity=float(getattr(candidate_basis_state, "complexity", 0.0) or 0.0)
                if isinstance(candidate_basis_state, BasisState)
                else 0.0,
                source_pool=str(profile.name),
                priority_rank=min(int(entry_a.get("idx", 0) or 0), int(entry_b.get("idx", 0) or 0)),
                candidate_meta=pair_meta,
                accept_meta=dict(accept_meta),
                pair_template=str(template_kind),
                relation_tags=tuple(_pair_relation_tags(entry_a, entry_b)),
                source_members=(
                    str(entry_a.get("source_pool", "") or ""),
                    str(entry_b.get("source_pool", "") or ""),
                ),
                gain_vs_best_singleton=float(debug_pair_row.get("pair_gain_over_best_singleton", math.nan)),
                best_singleton_key=str(best_singleton_key),
                best_singleton_eff_mse=float(best_singleton_eff_mse),
                best_singleton_source=str(best_singleton_source),
                pair_entries=(entry_a, entry_b),
            )
            _record_round_commit(pair_commit)
            if bool(accepted):
                _append_accepted_pair_event(pair_commit)
            pair_commit_objects.append(pair_commit)
        return pair_commit_objects

    def _run_pair_precommit(
        *,
        base_basis_state: BasisState | None,
        candidate_items: list[tuple[Any, Mapping[str, Any], str]],
        seed_row_mode: bool,
        round_beam_width: int,
        best_singleton: Mapping[str, Any] | None,
        attempted_pair_ids_sink: set[tuple[str, str]] | None = None,
    ) -> list[RoundCommitObject]:
        profile = PairProfile(
            name="seed_precommit",
            stats_prefix="basis_state_pair_precommit",
            pair_key_prefix="pair_precommit",
            proposal_family="pair_precommit",
            enabled=bool(pair_rescue_enable_b),
            topk=int(pair_rescue_topk_i),
            max_pairs=int(pair_rescue_max_pairs_i),
            debug_pool_stage="pair_precommit_pool",
            debug_attempt_stage="pair_precommit",
            require_interaction_support=True,
            use_pool_selection=False,
            source_pool_mode="row",
            template_kinds=("same_interaction_diff_support",),
            route_prepare_a="closure_search_pair_precommit_prepare_a",
            route_prepare_b="closure_search_pair_precommit_prepare_b",
            route_score="closure_search_pair_precommit",
            route_refit="closure_search_pair_precommit_refit",
        )
        return _run_pair_profile(
            profile=profile,
            base_basis_state=base_basis_state,
            candidate_items=candidate_items,
            seed_row_mode=seed_row_mode,
            round_beam_width=round_beam_width,
            best_singleton=best_singleton,
            attempted_pair_ids_sink=attempted_pair_ids_sink,
        )

    def _run_pair_normal(
        *,
        base_basis_state: BasisState | None,
        candidate_items: list[tuple[Any, Mapping[str, Any], str]],
        seed_row_mode: bool,
        round_beam_width: int,
        best_singleton: Mapping[str, Any] | None,
        attempted_pair_ids_sink: set[tuple[str, str]] | None = None,
    ) -> list[RoundCommitObject]:
        profile = PairProfile(
            name="normal",
            stats_prefix="basis_state_pair_normal",
            pair_key_prefix="pair_normal",
            proposal_family="pair_normal",
            enabled=bool(pair_normal_enable_b),
            topk=int(pair_normal_topk_i),
            max_pairs=int(pair_normal_max_pairs_i),
            debug_pool_stage="pair_normal_pool",
            debug_attempt_stage="pair_normal",
            require_interaction_support=False,
            use_pool_selection=True,
            source_pool_mode="row",
            template_kinds=(
                "same_interaction_diff_support",
                "cross_family_complement",
                "distinct_spec_sibling",
            ),
            route_prepare_a="closure_search_pair_normal_prepare_a",
            route_prepare_b="closure_search_pair_normal_prepare_b",
            route_score="closure_search_pair_normal",
            route_refit="closure_search_pair_normal_refit",
        )
        return _run_pair_profile(
            profile=profile,
            base_basis_state=base_basis_state,
            candidate_items=candidate_items,
            seed_row_mode=seed_row_mode,
            round_beam_width=round_beam_width,
            best_singleton=best_singleton,
            attempted_pair_ids_sink=attempted_pair_ids_sink,
        )

    def _run_pair_rescue(
        *,
        base_basis_state: BasisState | None,
        candidate_items: list[tuple[Any, Mapping[str, Any], str]],
        seed_row_mode: bool,
        round_beam_width: int,
        round_singleton_metrics: Mapping[str, Mapping[str, Any]] | None,
        best_singleton: Mapping[str, Any] | None,
        blocked_pair_ids: Collection[tuple[str, str]] | None = None,
    ) -> list[RoundCommitObject]:
        profile = PairProfile(
            name="stall_expanded",
            stats_prefix="basis_state_pair_rescue",
            pair_key_prefix="pair_rescue",
            proposal_family="pair_rescue",
            enabled=bool(pair_rescue_enable_b),
            topk=int(pair_rescue_topk_i),
            max_pairs=int(pair_rescue_max_pairs_i),
            debug_pool_stage="pair_pool",
            debug_attempt_stage="pair_rescue",
            require_interaction_support=False,
            use_pool_selection=True,
            source_pool_mode="singleton_metrics",
            template_kinds=(
                "cross_family_complement",
                "distinct_spec_sibling",
                "generic_complement",
            ),
            route_prepare_a="closure_search_pair_prepare_a",
            route_prepare_b="closure_search_pair_prepare_b",
            route_score="closure_search_pair_rescue",
            route_refit="closure_search_pair_rescue_refit",
        )
        return _run_pair_profile(
            profile=profile,
            base_basis_state=base_basis_state,
            candidate_items=candidate_items,
            seed_row_mode=seed_row_mode,
            round_beam_width=round_beam_width,
            best_singleton=best_singleton,
            round_singleton_metrics=round_singleton_metrics,
            blocked_pair_ids=blocked_pair_ids,
            allow_covered_feature_entries=True,
            allow_preview_materialization=True,
        )

    scaffolds_used = 0
    beam_width = int(beam_width_i)
    initial_seed_round = not basis_beam and not isinstance(proposal_context_local.basis_state, BasisState)
    max_rounds = max(1, min(max_scaffolds_i, exact_topk_i + (1 if initial_seed_round else 0)))

    def _aux_followup_unlocked() -> bool:
        return bool(
            emergent_aux_atoms_enable_b
            and int(emergent_aux_atoms_followup_budget_i) > 0
            and tuple(aux_atom_registry or ())
        )

    def _effective_scaffold_cap() -> int:
        extra = int(emergent_aux_atoms_followup_budget_i) if _aux_followup_unlocked() else 0
        return int(max_scaffolds_i) + int(extra)

    def _remaining_scaffold_budget() -> int:
        return max(0, int(_effective_scaffold_cap()) - int(scaffolds_used))

    for round_idx in range(max_rounds):
        if wall_time_exceeded_fn():
            break
        aux_followup_unlocked = _aux_followup_unlocked()
        if bool(aux_followup_unlocked):
            closure_search_stats["emergent_aux_atom_followup_unlocked"] = True
            closure_search_stats["emergent_aux_atom_followup_reserved"] = max(
                int(closure_search_stats.get("emergent_aux_atom_followup_reserved", 0) or 0),
                int(emergent_aux_atoms_followup_budget_i),
            )
        remaining_scaffolds = _remaining_scaffold_budget()
        if remaining_scaffolds <= 0:
            closure_search_stats["basis_state_controller_stop_reason"] = "scaffold_budget_exhausted"
            break
        remaining_wall_now = _remaining_wall_seconds(wall_time_deadline)
        current_basis_state, _fresh_beam_state = _select_round_basis_state()
        if current_basis_state is None:
            current_basis_state = proposal_context_local.basis_state
        if bool(emergent_aux_atoms_enable_b):
            library_atoms = tuple(aux_atom_registry or ()) + (
                tuple(aux_atom_observation_pool or ()) if bool(atom_policy_use_obs_pool_b) else ()
            )
            closure_search_stats["atom_policy_source_atoms"] = int(len(library_atoms))
            closure_search_stats["emergent_aux_atom_observation_pool_size"] = int(
                len(tuple(aux_atom_observation_pool or ()))
            )
            closure_search_stats["emergent_aux_atom_observation_pool"] = [
                atom.to_dict()
                for atom in tuple(aux_atom_observation_pool or ())[: max(0, int(debug_capture_limit_i))]
                if isinstance(atom, EmergentAtom)
            ]
            aux_seed_blocks = seed_blocks_from_emergent_atoms(
                aux_atom_registry,
                var_dims=var_dims,
                limit=int(emergent_aux_atoms_max_seed_blocks_i),
            )
            atom_library = build_atom_library(
                library_atoms,
                var_dims=var_dims,
                y_dims=y_dims,
                max_records=max(16, int(emergent_aux_atoms_max_seed_blocks_i) * 3),
                max_relations=max(12, int(emergent_aux_atoms_max_seed_blocks_i) * 3),
                stats=closure_search_stats,
            )
        else:
            aux_seed_blocks = ()
            atom_library = None
        closure_search_stats["emergent_aux_atom_seed_blocks"] = int(len(aux_seed_blocks))
        closure_search_stats["emergent_aux_atom_seed_exprs"] = [
            str(node_str(block.node))
            for block in tuple(aux_seed_blocks or ())
            if hasattr(block, "node") and isinstance(block.node, tuple) and is_valid_node(block.node)
        ]
        if isinstance(proposal_context_local, ProposalContext):
            proposal_context_local = replace(
                proposal_context_local,
                aux_seed_blocks=tuple(aux_seed_blocks),
                atom_library=atom_library,
            )
        pre_harvest_atomized_rows: list[dict[str, Any]] = []
        if (
            bool(emergent_aux_atoms_enable_b)
            and bool(atomized_linear_span_enable_b)
            and not bool(atomized_linear_span_same_round_b)
            and int(atomized_linear_span_budget_i) > 0
        ):
            atomized_atoms = _atomized_source_atoms()
            pre_harvest_atomized_rows = _propose_atomized_linear_span_rows(
                atoms=atomized_atoms,
                atom_origin_by_key=_atom_origin_map(
                    aux_atom_registry,
                    aux_atom_observation_pool,
                    include_observed=bool(atomized_linear_span_use_obs_pool_b),
                ),
                x_fit=x_fit,
                y_fit=y_fit,
                x_probe=x_probe,
                y_probe=y_probe,
                var_dims=var_dims,
                y_dims=y_dims,
                max_depth=int(max_depth),
                max_rows=int(atomized_linear_span_budget_i),
                stats=closure_search_stats,
                debug_limit=int(debug_capture_limit_i),
            )
        seed_mode = (
            (not isinstance(current_basis_state, BasisState))
            and (len(tuple(basis_beam or ())) == 0)
            and (not bool(tuple(aux_seed_blocks or ())))
        )
        round_scaffold_reserve = 0
        if bool(seed_mode) and int(round_idx) + 1 < int(max_rounds) and int(seed_scaffold_reserve_i) > 0:
            round_scaffold_reserve = min(
                int(seed_scaffold_reserve_i),
                max(0, int(remaining_scaffolds) - 1),
            )
        round_scaffold_budget = max(1, int(remaining_scaffolds) - int(round_scaffold_reserve))
        proposal_context_local = _build_residual_guided_context(
            base_context=proposal_context_local,
            basis_state=current_basis_state,
            basis_state_beam=basis_beam,
            families=family_tokens,
            total_budget=int(round_scaffold_budget),
            wall_time_remaining_s=remaining_wall_now,
            nvars=int(nvars),
            x_fit=x_fit,
            y_fit=y_fit,
            x_probe=x_probe,
            y_probe=y_probe,
            boost_pool_nodes=boost_pool_nodes,
        )
        closure_search_stats["closure_search_rounds"] = int(closure_search_stats.get("closure_search_rounds", 0)) + 1
        try:
            pass_ret = run_closure_search_pass_impl(
                families=family_tokens,
                nvars=int(nvars),
                max_scaffolds=int(round_scaffold_budget),
                anchors_per_family=anchors_per_family_i,
                max_depth=int(max_depth),
                poly_degree=int(poly_degree),
                x_fit=x_fit,
                y_fit=y_fit,
                x_probe=x_probe,
                y_probe=y_probe,
                var_dims=var_dims,
                y_dims=y_dims,
                pool_nodes=boost_pool_nodes,
                pool_phi_fit=boost_pool_phi_fit,
                pool_phi_probe=boost_pool_phi,
                pool_dims=boost_pool_dims,
                safe_eps=float(inverse_safe_eps),
                preview_topk=preview_topk_i,
                beam_cfg=beam_cfg,
                solver_kwargs=solver_kwargs,
                deadline_s=scaffold_deadline,
                proposal_context=proposal_context_local,
                family_allocator_fn=family_allocator_fn,
            )
        except Exception:
            record_route_status(closure_search_stats, "closure_search_exception")
            return

        preview_stats = dict((pass_ret or {}).get("stats", {}) or {})
        for key in (
            "families_considered",
            "scaffolds_enumerated",
            "scaffolds_considered",
            "preview_calls",
            "preview_candidates",
            "direct_calls",
            "direct_candidates",
            "direct_anchor_lift_attempts",
            "direct_anchor_lift_applied",
            "aux_scaffolds_enumerated",
            "protected_aux_scaffolds_enumerated",
        ):
            closure_search_stats[key] = int(closure_search_stats.get(key, 0) or 0) + int(preview_stats.get(key, 0) or 0)
        for key in (
            "proposal_context",
            "family_priority_scores",
            "family_budget_plan",
            "family_steering_applied",
            "family_priority_decomposition",
            "proposal_lane_budgets",
            "scaffolds_enumerated_by_lane",
            "aux_seed_blocks_count",
            "aux_seed_block_exprs",
            "family_budget_plan_by_lane",
            "proposal_context",
        ):
            if key in preview_stats:
                closure_search_stats[key] = preview_stats.get(key)
        closure_search_stats["deadline_exceeded"] = bool(
            preview_stats.get("deadline_exceeded", closure_search_stats.get("deadline_exceeded", False))
        )
        merge_route_status_counts(closure_search_stats, preview_stats.get("status_counts", {}))
        scaffolds_used += int(preview_stats.get("scaffolds_enumerated", 0) or 0)
        failure_examples = preview_stats.get("failure_examples", None)
        if isinstance(failure_examples, list):
            closure_search_stats["failure_examples"] = [dict(row) for row in failure_examples if isinstance(row, Mapping)]

        candidate_rows = [
            dict(row)
            for row in list((pass_ret or {}).get("candidate_rows", []) or [])
            if isinstance(row, Mapping)
        ]
        if pre_harvest_atomized_rows:
            candidate_rows.extend(pre_harvest_atomized_rows)
        if not candidate_rows:
            closure_search_stats["basis_state_controller_stop_reason"] = "no_candidates"
            break
        seed_row_mode = bool(seed_mode) and any(
            isinstance(row.get("basis_state_obj", None), BasisState)
            for row in candidate_rows
        )
        rank_basis_state = current_basis_state if isinstance(current_basis_state, BasisState) else (
            proposal_context_local.basis_state if isinstance(
                proposal_context_local.basis_state, BasisState
            ) else None
        )
        if bool(emergent_aux_atoms_enable_b):
            observed_atoms: list[EmergentAtom] = []
            new_atoms = harvest_emergent_atoms(
                candidate_rows=candidate_rows,
                x_fit=x_fit,
                y_fit=y_fit,
                x_probe=x_probe,
                y_probe=y_probe,
                var_dims=var_dims,
                y_dims=y_dims,
                stats=closure_search_stats,
                max_source_rows=int(emergent_aux_atoms_max_source_rows_i),
                max_new=int(emergent_aux_atoms_max_new_per_round_i),
                debug_limit=int(debug_capture_limit_i),
                observed_atom_sink=observed_atoms,
            )
            aux_atom_observation_pool = _merge_aux_atom_observation_pool(
                aux_atom_observation_pool,
                observed_atoms,
                max_count=max(24, min(96, int(emergent_aux_atoms_max_total_i) * 6)),
            )
            previous_keys = {str(node_str(atom.node)) for atom in tuple(aux_atom_registry or ())}
            aux_atom_registry = merge_emergent_atom_registry(
                aux_atom_registry,
                new_atoms,
                max_total=int(emergent_aux_atoms_max_total_i),
                max_target=int(emergent_aux_atoms_max_target_i),
                max_dimensionless=int(emergent_aux_atoms_max_dimensionless_i),
                max_rational_derived=int(emergent_aux_atoms_max_rational_derived_i),
                stats=closure_search_stats,
            )
            current_keys = {str(node_str(atom.node)) for atom in tuple(aux_atom_registry or ())}
            if current_keys.difference(previous_keys):
                closure_search_stats["emergent_aux_atom_rounds_with_new"] = int(
                    closure_search_stats.get("emergent_aux_atom_rounds_with_new", 0) or 0
                ) + 1
            if (
                bool(atomized_linear_span_enable_b)
                and bool(atomized_linear_span_same_round_b)
                and int(atomized_linear_span_budget_i) > 0
            ):
                atomized_atoms = _atomized_source_atoms()
                atomized_rows = _propose_atomized_linear_span_rows(
                    atoms=atomized_atoms,
                    atom_origin_by_key=_atom_origin_map(
                        aux_atom_registry,
                        aux_atom_observation_pool,
                        include_observed=bool(atomized_linear_span_use_obs_pool_b),
                    ),
                    x_fit=x_fit,
                    y_fit=y_fit,
                    x_probe=x_probe,
                    y_probe=y_probe,
                    var_dims=var_dims,
                    y_dims=y_dims,
                    max_depth=int(max_depth),
                    max_rows=int(atomized_linear_span_budget_i),
                    stats=closure_search_stats,
                    debug_limit=int(debug_capture_limit_i),
                )
                if atomized_rows:
                    candidate_rows.extend(atomized_rows)
        if bool(emergent_basis_enable_b):
            emergent_rows = propose_emergent_basis_rows(
                candidate_rows=candidate_rows,
                current_basis_state=rank_basis_state,
                x_fit=x_fit,
                y_fit=y_fit,
                x_probe=x_probe,
                y_probe=y_probe,
                var_dims=var_dims,
                y_dims=y_dims,
                stats=closure_search_stats,
                max_source_rows=int(emergent_basis_max_source_rows_i),
                score_topk=int(emergent_basis_score_topk_i),
                max_promoted_per_round=int(emergent_basis_max_per_round_i),
                max_promoted_total=int(emergent_basis_max_total_i),
                min_probe_gain_rel=float(emergent_basis_min_probe_gain_rel_f),
                debug_limit=int(debug_capture_limit_i),
            )
            if emergent_rows:
                candidate_rows.extend(emergent_rows)
                closure_search_stats["emergent_basis_rounds_with_rows"] = int(
                    closure_search_stats.get("emergent_basis_rounds_with_rows", 0) or 0
                ) + 1
                seed_row_mode = bool(seed_mode) and any(
                    isinstance(row.get("basis_state_obj", None), BasisState)
                    for row in candidate_rows
                )
        closure_search_stats["basis_state_rank_proxy_candidates"] = int(
            closure_search_stats.get("basis_state_rank_proxy_candidates", 0) or 0
        ) + int(
            sum(1 for row in candidate_rows if isinstance(row.get("basis_state_obj", None), BasisState))
        )
        round_exact_budget = int(seed_exact_topk_i if seed_row_mode else exact_topk_i)
        round_beam_width = int(seed_beam_width_i if seed_row_mode else beam_width)
        round_exact_scored = 0
        if seed_row_mode:
            closure_search_stats["basis_state_seed_mode_used"] = True
            closure_search_stats["basis_state_seed_rounds"] = int(
                closure_search_stats.get("basis_state_seed_rounds", 0) or 0
            ) + 1

        round_debug_summary = None
        expr_valid_rows: list[Mapping[str, Any]] = []
        best_by_proposal = {}
        for idx, row in enumerate(candidate_rows):
            expr = row.get("expr", None)
            if not isinstance(expr, tuple):
                continue
            expr_valid_rows.append(row)
            proposal_key = str(row.get("proposal_key", "") or row.get("child_key", "") or node_str_fn(expr))
            sort_key = _candidate_sort_key(
                row,
                seed_mode=seed_row_mode,
                current_basis_state=rank_basis_state,
            )
            prev = best_by_proposal.get(proposal_key, None)
            if prev is not None and not (sort_key < prev[0]):
                continue
            best_by_proposal[proposal_key] = (sort_key, dict(row), proposal_key)

        ranked_scaffold_rows = sorted(best_by_proposal.values(), key=lambda item: item[0])
        fasttrack_rows: list[tuple[Any, Mapping[str, Any], str]] = []
        seed_family_rows: list[tuple[Any, Mapping[str, Any], str]] = []
        reserved_variant_rows: list[tuple[Any, Mapping[str, Any], str]] = []
        reserved_interaction_anchor_rows: list[tuple[Any, Mapping[str, Any], str]] = []
        reserved_interaction_rows: list[tuple[Any, Mapping[str, Any], str]] = []
        reserved_core_rows: list[tuple[Any, Mapping[str, Any], str]] = []
        reserved_family_rows: list[tuple[Any, Mapping[str, Any], str]] = []
        reserved_spec_rows: list[tuple[Any, Mapping[str, Any], str]] = []
        late_reserved_interaction_anchor_rows: list[tuple[Any, Mapping[str, Any], str]] = []
        regular_rows: list[tuple[Any, Mapping[str, Any], str]] = []
        seen_reserved_specs: set[str] = set()
        seen_selected_proposals: set[str] = set()
        reserved_family_counts: dict[str, int] = {}
        reserved_variant_counts: dict[str, int] = {}
        selected_variant_tokens_by_group: dict[str, set[str]] = {}
        selected_interaction_keys: set[str] = set()
        selected_interaction_anchor_keys: dict[str, set[str]] = {}
        interaction_anchor_reservation_counts: dict[str, int] = {}

        def _mark_reserved_item(item: tuple[Any, Mapping[str, Any], str]) -> None:
            _item_sort_key, item_row, item_proposal_key = item
            seen_selected_proposals.add(str(item_proposal_key))
            family_name = _candidate_family(item_row)
            if family_name:
                reserved_family_counts[family_name] = int(
                    reserved_family_counts.get(family_name, 0) or 0
                ) + 1
            spec_name = _candidate_spec_key(item_row)
            if spec_name:
                seen_reserved_specs.add(spec_name)
            variant_group = _candidate_variant_group_key(item_row)
            variant_token = _candidate_variant_token(item_row)
            if variant_group and variant_token:
                selected_variant_tokens_by_group.setdefault(variant_group, set()).add(variant_token)
            interaction_key = _candidate_interaction_key(item_row)
            anchor_key = _candidate_anchor_diversity_key(item_row)
            if interaction_key:
                selected_interaction_keys.add(interaction_key)
                if anchor_key:
                    selected_interaction_anchor_keys.setdefault(interaction_key, set()).add(anchor_key)

        for item in ranked_scaffold_rows:
            _sort_key, row, proposal_key = item
            if _candidate_is_fasttrack(row):
                fasttrack_rows.append(item)
                _mark_reserved_item(item)
                continue
        if seed_row_mode:
            for item in ranked_scaffold_rows:
                _sort_key, row, proposal_key = item
                if str(proposal_key) in seen_selected_proposals:
                    continue
                if not isinstance(row.get("basis_state_obj", None), BasisState):
                    continue
                family_name = _candidate_family(row)
                if not family_name:
                    continue
                if int(reserved_family_counts.get(family_name, 0) or 0) > 0:
                    continue
                seed_family_rows.append(item)
                _mark_reserved_item(item)
        for item in ranked_scaffold_rows:
            _sort_key, row, proposal_key = item
            if str(proposal_key) in seen_selected_proposals:
                continue
            if not isinstance(row.get("basis_state_obj", None), BasisState):
                continue
            family_name = _candidate_family(row)
            if not family_name:
                continue
            if int(reserved_family_counts.get(family_name, 0) or 0) > 0:
                continue
            reserved_family_rows.append(item)
            _mark_reserved_item(item)
        variant_candidates_by_group: dict[str, list[tuple[Any, Mapping[str, Any], str]]] = {}
        for item in ranked_scaffold_rows:
            _sort_key, row, proposal_key = item
            if str(proposal_key) in seen_selected_proposals:
                continue
            if not isinstance(row.get("basis_state_obj", None), BasisState):
                continue
            variant_group = _candidate_variant_group_key(row)
            variant_token = _candidate_variant_token(row)
            if not variant_group or not variant_token:
                continue
            selected_variant_tokens = selected_variant_tokens_by_group.get(variant_group, set())
            if not selected_variant_tokens or variant_token in selected_variant_tokens:
                continue
            variant_candidates_by_group.setdefault(variant_group, []).append(item)
        for variant_group, items in variant_candidates_by_group.items():
            if int(reserved_variant_counts.get(variant_group, 0) or 0) >= 1:
                continue
            best_item = min(
                items,
                key=lambda item: (
                    _candidate_variant_nparams(item[1]),
                    item[0],
                    str(item[2]),
                ),
            )
            reserved_variant_rows.append(best_item)
            reserved_variant_counts[variant_group] = int(
                reserved_variant_counts.get(variant_group, 0) or 0
            ) + 1
            _mark_reserved_item(best_item)
        for item in ranked_scaffold_rows:
            _sort_key, row, proposal_key = item
            if str(proposal_key) in seen_selected_proposals:
                continue
            if not isinstance(row.get("basis_state_obj", None), BasisState):
                continue
            interaction_key = _candidate_interaction_key(row)
            anchor_key = _candidate_anchor_diversity_key(row)
            if not interaction_key or not anchor_key:
                continue
            if interaction_key not in selected_interaction_keys:
                continue
            existing_anchor_keys = selected_interaction_anchor_keys.get(interaction_key, set())
            if anchor_key in existing_anchor_keys:
                continue
            if int(interaction_anchor_reservation_counts.get(interaction_key, 0) or 0) >= 1:
                continue
            reserved_interaction_anchor_rows.append(item)
            interaction_anchor_reservation_counts[interaction_key] = int(
                interaction_anchor_reservation_counts.get(interaction_key, 0) or 0
            ) + 1
            _mark_reserved_item(item)
        for item in ranked_scaffold_rows:
            _sort_key, row, proposal_key = item
            if str(proposal_key) in seen_selected_proposals:
                continue
            if _candidate_lane(row) == "core":
                spec_key = _candidate_spec_key(row)
                if spec_key and spec_key not in seen_reserved_specs:
                    reserved_core_rows.append(item)
                    _mark_reserved_item(item)
                    continue
        for item in ranked_scaffold_rows:
            _sort_key, row, proposal_key = item
            if str(proposal_key) in seen_selected_proposals:
                continue
            if not isinstance(row.get("basis_state_obj", None), BasisState):
                continue
            spec_key = _candidate_spec_key(row)
            if not spec_key or spec_key in seen_reserved_specs:
                continue
            family_name = _candidate_family(row)
            if family_name and int(reserved_family_counts.get(family_name, 0) or 0) >= 2:
                continue
            reserved_spec_rows.append(item)
            _mark_reserved_item(item)
        for item in ranked_scaffold_rows:
            _sort_key, row, proposal_key = item
            if str(proposal_key) in seen_selected_proposals:
                continue
            if not isinstance(row.get("basis_state_obj", None), BasisState):
                continue
            interaction_key = _candidate_interaction_key(row)
            anchor_key = _candidate_anchor_diversity_key(row)
            if not interaction_key or not anchor_key:
                continue
            if interaction_key not in selected_interaction_keys:
                continue
            existing_anchor_keys = selected_interaction_anchor_keys.get(interaction_key, set())
            if anchor_key in existing_anchor_keys:
                continue
            if int(interaction_anchor_reservation_counts.get(interaction_key, 0) or 0) >= 1:
                continue
            late_reserved_interaction_anchor_rows.append(item)
            interaction_anchor_reservation_counts[interaction_key] = int(
                interaction_anchor_reservation_counts.get(interaction_key, 0) or 0
            ) + 1
            _mark_reserved_item(item)
        for item in ranked_scaffold_rows:
            _sort_key, row, proposal_key = item
            if str(proposal_key) in seen_selected_proposals:
                continue
            if not isinstance(row.get("basis_state_obj", None), BasisState):
                continue
            interaction_key = _candidate_interaction_key(row)
            if not interaction_key or interaction_key in selected_interaction_keys:
                continue
            reserved_interaction_rows.append(item)
            _mark_reserved_item(item)
        for item in ranked_scaffold_rows:
            _sort_key, _row, proposal_key = item
            if str(proposal_key) in seen_selected_proposals:
                continue
            regular_rows.append(item)
        closure_search_stats["basis_state_fasttrack_candidates"] = int(
            closure_search_stats.get("basis_state_fasttrack_candidates", 0)
        ) + int(len(fasttrack_rows))
        closure_search_stats["basis_state_core_lane_reservations"] = int(
            closure_search_stats.get("basis_state_core_lane_reservations", 0)
        ) + int(len(reserved_core_rows))
        closure_search_stats["basis_state_seed_family_reservations"] = int(
            closure_search_stats.get("basis_state_seed_family_reservations", 0) or 0
        ) + int(len(seed_family_rows))
        closure_search_stats["basis_state_variant_reservations"] = int(
            closure_search_stats.get("basis_state_variant_reservations", 0) or 0
        ) + int(len(reserved_variant_rows))
        closure_search_stats["basis_state_interaction_anchor_reservations"] = int(
            closure_search_stats.get("basis_state_interaction_anchor_reservations", 0) or 0
        ) + int(len(reserved_interaction_anchor_rows) + len(late_reserved_interaction_anchor_rows))
        closure_search_stats["basis_state_interaction_reservations"] = int(
            closure_search_stats.get("basis_state_interaction_reservations", 0) or 0
        ) + int(len(reserved_interaction_rows))
        closure_search_stats["basis_state_family_reservations"] = int(
            closure_search_stats.get("basis_state_family_reservations", 0) or 0
        ) + int(len(reserved_family_rows))
        closure_search_stats["basis_state_spec_reservations"] = int(
            closure_search_stats.get("basis_state_spec_reservations", 0) or 0
        ) + int(len(reserved_spec_rows))
        seed_family_strong_rows = list(seed_family_rows)
        seed_family_weak_rows: list[tuple[Any, Mapping[str, Any], str]] = []
        if seed_row_mode and reserved_interaction_anchor_rows:
            interaction_probe_cutoff = min(
                float(item[1].get("local_probe_mse", float("inf")) or float("inf"))
                for item in reserved_interaction_anchor_rows
                if isinstance(item[1], Mapping)
            )
            seed_family_strong_rows = []
            seed_family_weak_rows = []
            for item in seed_family_rows:
                row = item[1]
                try:
                    row_probe = float(row.get("local_probe_mse", float("inf")) or float("inf"))
                except Exception:
                    row_probe = float("inf")
                if row_probe <= float(interaction_probe_cutoff):
                    seed_family_strong_rows.append(item)
                else:
                    seed_family_weak_rows.append(item)
        prioritized_rows = [
            *fasttrack_rows,
            *seed_family_strong_rows,
            *reserved_family_rows,
            *reserved_variant_rows,
            *reserved_interaction_anchor_rows,
            *reserved_core_rows,
            *reserved_spec_rows,
            *seed_family_weak_rows,
            *late_reserved_interaction_anchor_rows,
            *reserved_interaction_rows,
            *regular_rows,
        ]
        if debug_topk_i > 0:
            dedup_rows = [item[1] for item in ranked_scaffold_rows]
            prioritized_debug_rows = [item[1] for item in prioritized_rows]
            round_debug_summary = {
                "round": int(closure_search_stats.get("closure_search_rounds", 0) or 0),
                "seed_mode": bool(seed_row_mode),
                "scaffold_budget": int(round_scaffold_budget),
                "scaffold_reserve": int(round_scaffold_reserve),
                "scaffold_cap": int(_effective_scaffold_cap()),
                "aux_followup_unlocked": bool(aux_followup_unlocked),
                "exact_budget": int(round_exact_budget),
                "aux_seed_blocks": int(len(tuple(getattr(proposal_context_local, "aux_seed_blocks", ()) or ()))),
                "aux_registry_size": int(len(tuple(aux_atom_registry or ()))),
                "aux_registry_exprs": [str(node_str(atom.node)) for atom in tuple(aux_atom_registry or ())],
                "proposal_lane_budgets": dict(preview_stats.get("proposal_lane_budgets", {}) or {}),
                "scaffolds_enumerated_by_lane": dict(preview_stats.get("scaffolds_enumerated_by_lane", {}) or {}),
                "aux_scaffolds_enumerated": int(preview_stats.get("aux_scaffolds_enumerated", 0) or 0),
                "protected_aux_scaffolds_enumerated": int(
                    preview_stats.get("protected_aux_scaffolds_enumerated", 0) or 0
                ),
                "aux_seed_block_exprs": list(preview_stats.get("aux_seed_block_exprs", []) or []),
                "atom_policy_library_records": int(preview_stats.get("atom_policy_library_records", closure_search_stats.get("atom_policy_library_records", 0)) or 0),
                "atom_policy_library_relations": int(preview_stats.get("atom_policy_library_relations", closure_search_stats.get("atom_policy_library_relations", 0)) or 0),
                "atomized_linear_span_rows": int(closure_search_stats.get("atomized_linear_span_rows", 0) or 0),
                "atomized_linear_span_best_probe": float(
                    closure_search_stats.get("atomized_linear_span_best_probe", math.inf)
                ),
                "family_priority_scores": dict(preview_stats.get("family_priority_scores", {}) or {}),
                "family_priority_decomposition": dict(preview_stats.get("family_priority_decomposition", {}) or {}),
                "family_budget_plan": list(preview_stats.get("family_budget_plan", []) or []),
                "family_budget_plan_by_lane": dict(preview_stats.get("family_budget_plan_by_lane", {}) or {}),
                "raw_family_counts": _count_candidate_families(candidate_rows),
                "raw_native_family_counts": _count_candidate_families(candidate_rows, native_only=True),
                "expr_family_counts": _count_candidate_families(expr_valid_rows),
                "expr_native_family_counts": _count_candidate_families(expr_valid_rows, native_only=True),
                "dedup_family_counts": _count_candidate_families(dedup_rows),
                "dedup_native_family_counts": _count_candidate_families(dedup_rows, native_only=True),
                "prioritized_family_counts": _count_candidate_families(prioritized_debug_rows),
                "prioritized_native_family_counts": _count_candidate_families(prioritized_debug_rows, native_only=True),
            }
            debug_rounds = closure_search_stats.get("debug_round_summaries", None)
            if not isinstance(debug_rounds, list):
                debug_rounds = []
                closure_search_stats["debug_round_summaries"] = debug_rounds
            debug_rounds.append(round_debug_summary)
        if debug_topk_i > 0:
            debug_round = int(closure_search_stats.get("closure_search_rounds", 0) or 0)
            for sort_key, row, proposal_key in prioritized_rows[: int(debug_topk_i)]:
                _debug_append(
                    "debug_preview_rows",
                    _candidate_debug_row(
                        row,
                        round_idx=debug_round,
                        stage="prioritized",
                        proposal_key=str(proposal_key),
                        sort_key=sort_key,
                    ),
                )
        seen_child_keys: set[str] = set()
        exact_scored_items: list[tuple[Any, Mapping[str, Any], str]] = []
        prepared_pair_items: list[tuple[Any, Mapping[str, Any], str]] = []
        round_singleton_metrics: dict[str, dict[str, Any]] = {}
        round_singleton_commits: list[RoundCommitObject] = []
        attempted_pair_ids: set[tuple[str, str]] = set()
        round_updated_state: BasisState | None = None
        round_base_basis_state = current_basis_state
        round_atomized_exact_scored = 0
        for _sort_key, row, proposal_key in prioritized_rows:
            if proposal_key in seen_child_keys:
                continue
            seen_child_keys.add(proposal_key)
            is_fasttrack = _candidate_is_fasttrack(row)
            is_atomized = bool(row.get("atomized_linear_span", False))
            if (
                is_atomized
                and not is_fasttrack
                and int(round_atomized_exact_scored) >= int(atomized_linear_span_exact_quota_i)
            ):
                closure_search_stats["atomized_linear_span_exact_quota_skipped"] = int(
                    closure_search_stats.get("atomized_linear_span_exact_quota_skipped", 0) or 0
                ) + 1
                continue
            if (not is_fasttrack) and (not is_atomized) and (round_exact_scored >= round_exact_budget):
                continue
            if wall_time_exceeded_fn():
                break
            expr = row.get("expr", None)
            if not isinstance(expr, tuple):
                continue
            current_basis_state = round_base_basis_state
            current_feature_block = row.get("feature_block_obj", None)
            if basis_state_covers_feature_block(current_basis_state, current_feature_block):
                closure_search_stats["basis_state_skip_covered"] = int(closure_search_stats.get("basis_state_skip_covered", 0)) + 1
                continue
            candidate_meta = dict(row)
            preview_state = candidate_meta.get("basis_state_obj", None)
            preserve_direct_state = bool(_candidate_is_fasttrack(row))
            candidate_meta["basis_state_direct_preserve"] = bool(preserve_direct_state)
            candidate_meta["basis_state_prepare_mode"] = "preview_only" if preserve_direct_state else "merged"
            if preserve_direct_state and isinstance(preview_state, BasisState):
                prepared_state = basis_state_retarget(
                    preview_state,
                    compiled_expr=expr if isinstance(expr, tuple) else preview_state.compiled_expr,
                    route_name="closure_search_preview_preserve",
                    provenance_tag="closure_search_preview_preserve",
                )
            else:
                prepared_state = prepare_basis_state_candidate(
                    current_basis_state,
                    preview_state if isinstance(preview_state, BasisState) else None,
                    route_name="closure_search_preview_prepare",
                    compiled_expr=expr if isinstance(expr, tuple) else None,
                )
            if isinstance(prepared_state, BasisState):
                candidate_meta["basis_state_obj"] = prepared_state
                candidate_meta["basis_state_dict"] = prepared_state.to_dict()
            score_ret = None
            if callable(score_native_candidate_basis_state_fn):
                score_ret = score_native_candidate_basis_state_fn(
                    candidate_meta=candidate_meta,
                    stats=closure_search_stats,
                    route_name="closure_search",
                )
            if score_ret is None:
                score_ret = score_external_candidate_expr_fn(
                    expr,
                    parent_raw_mse=None,
                    stats=closure_search_stats,
                    route_name="closure_search",
                    candidate_meta=candidate_meta,
                )
            if is_fasttrack:
                closure_search_stats["basis_state_fasttrack_scored"] = int(
                    closure_search_stats.get("basis_state_fasttrack_scored", 0)
                ) + 1
            else:
                if is_atomized:
                    round_atomized_exact_scored += 1
                    closure_search_stats["atomized_linear_span_exact_scored"] = int(
                        closure_search_stats.get("atomized_linear_span_exact_scored", 0) or 0
                    ) + 1
                else:
                    round_exact_scored += 1
                if seed_row_mode:
                    closure_search_stats["basis_state_seed_scored"] = int(
                        closure_search_stats.get("basis_state_seed_scored", 0) or 0
                    ) + 1
                pair_item_row = dict(candidate_meta)
                scored_preview_state = score_ret.get("basis_state_obj", None) if isinstance(score_ret, Mapping) else None
                if isinstance(scored_preview_state, BasisState):
                    pair_item_row["basis_state_obj"] = scored_preview_state
                    pair_item_row["basis_state_dict"] = scored_preview_state.to_dict()
                scored_candidate = score_ret.get("proposal_candidate_obj", None) if isinstance(score_ret, Mapping) else None
                if isinstance(scored_candidate, ProposalCandidate):
                    pair_item_row["proposal_candidate_obj"] = scored_candidate
                    pair_item_row["proposal_candidate_dict"] = scored_candidate.to_dict()
                eff_mse = math.inf
                if isinstance(score_ret, Mapping):
                    try:
                        eff_mse = float(score_ret.get("eff_mse", math.inf) or math.inf)
                    except Exception:
                        eff_mse = math.inf
                pair_item_row["_round_source_pool"] = "exact_scored_singleton"
                pair_item_row["_round_eff_mse"] = float(eff_mse)
                exact_scored_items.append((_sort_key, pair_item_row, str(proposal_key)))
                round_singleton_metrics[str(proposal_key)] = {
                    "proposal_key": str(proposal_key),
                    "eff_mse": float(eff_mse),
                    "source_pool": "exact_scored_singleton",
                }
            debug_exact_row = None
            if debug_topk_i > 0:
                debug_exact_row = _candidate_debug_row(
                    row,
                    round_idx=int(closure_search_stats.get("closure_search_rounds", 0) or 0),
                    stage="exact",
                    proposal_key=str(proposal_key),
                    sort_key=_sort_key,
                )
                debug_exact_row["route"] = "closure_search"
                debug_exact_row["accepted"] = False
                debug_exact_row["accept_reason"] = "missing_candidate_basis_state"
                if isinstance(score_ret, Mapping):
                    try:
                        debug_exact_row["eff_mse"] = float(score_ret.get("eff_mse", math.inf))
                    except Exception:
                        debug_exact_row["eff_mse"] = math.inf
                    try:
                        debug_exact_row["raw_mse"] = float(score_ret.get("raw_mse", math.inf))
                    except Exception:
                        debug_exact_row["raw_mse"] = math.inf
                    try:
                        debug_exact_row["fit_loss"] = float(score_ret.get("fit_loss", math.inf))
                    except Exception:
                        debug_exact_row["fit_loss"] = math.inf
                    try:
                        debug_exact_row["probe_loss"] = float(score_ret.get("probe_loss", math.inf))
                    except Exception:
                        debug_exact_row["probe_loss"] = math.inf
            candidate_basis_state = _basis_state_from_scored_result(
                score_ret=score_ret,
                candidate_meta=candidate_meta,
                current_basis_state=current_basis_state,
                route_name="closure_search_basis_loop",
            )
            accept_meta: dict[str, Any] = {"accept": False, "reason": "missing_candidate_basis_state"}
            if isinstance(candidate_basis_state, BasisState):
                refit_admitted_state = fit_basis_state_head(
                    candidate_basis_state,
                    x_fit=x_fit,
                    y_fit=y_fit,
                    x_probe=x_probe,
                    y_probe=y_probe,
                    route_name="closure_search_basis_loop_refit",
                )
                if isinstance(refit_admitted_state, BasisState):
                    candidate_basis_state = refit_admitted_state
                accept_meta = _basis_state_acceptance_decision(
                    current_basis_state=current_basis_state,
                    candidate_basis_state=candidate_basis_state,
                    candidate_meta=candidate_meta,
                    family_hints=getattr(proposal_context_local, "family_hints", {}) if isinstance(proposal_context_local, ProposalContext) else {},
                    complexity_penalty=float(inverse_spec_complexity_penalty),
                    route_name="closure_search",
                )
                _record_acceptance_meta(accept_meta)
            accepted_singleton = bool(accept_meta.get("accept", False))
            source_pool = "fasttrack_singleton" if is_fasttrack else "exact_scored_singleton"
            eff_mse = float(debug_exact_row.get("eff_mse", math.inf)) if isinstance(debug_exact_row, dict) else math.inf
            if isinstance(candidate_basis_state, BasisState):
                eff_mse = float(getattr(candidate_basis_state, "probe_loss", eff_mse))
            if isinstance(debug_exact_row, dict):
                debug_exact_row["accepted"] = bool(accepted_singleton)
                debug_exact_row["accept_reason"] = str(accept_meta.get("reason", "") or "missing_candidate_basis_state")
                if isinstance(candidate_basis_state, BasisState):
                    debug_exact_row["candidate_probe"] = float(getattr(candidate_basis_state, "probe_loss", math.inf))
                    debug_exact_row["candidate_complexity"] = float(
                        getattr(candidate_basis_state, "complexity", 0.0) or 0.0
                    )
                    debug_exact_row["expr"] = _expr_debug_str(candidate_basis_state.compiled_expr)
                _debug_append("debug_exact_rows", debug_exact_row)
            prepared_pair_row = dict(candidate_meta)
            if isinstance(candidate_basis_state, BasisState):
                prepared_pair_row["basis_state_obj"] = candidate_basis_state
                prepared_pair_row["basis_state_dict"] = candidate_basis_state.to_dict()
            scored_candidate = score_ret.get("proposal_candidate_obj", None) if isinstance(score_ret, Mapping) else None
            if isinstance(scored_candidate, ProposalCandidate):
                prepared_pair_row["proposal_candidate_obj"] = scored_candidate
                prepared_pair_row["proposal_candidate_dict"] = scored_candidate.to_dict()
            prepared_pair_items.append((_sort_key, prepared_pair_row, str(proposal_key)))
            if not is_fasttrack:
                singleton_metric = round_singleton_metrics.setdefault(
                    str(proposal_key),
                    {
                        "proposal_key": str(proposal_key),
                        "eff_mse": float(eff_mse),
                        "source_pool": "exact_scored_singleton",
                    },
                )
                singleton_metric["post_refit_probe"] = float(eff_mse)
                singleton_metric["accepted"] = bool(accepted_singleton)
            singleton_commit = RoundCommitObject(
                kind="singleton",
                profile="singleton",
                proposal_key=str(proposal_key),
                basis_state=candidate_basis_state if isinstance(candidate_basis_state, BasisState) else None,
                eff_mse=float(eff_mse),
                accepted=bool(accepted_singleton),
                accept_reason=str(accept_meta.get("reason", "") or "missing_candidate_basis_state"),
                complexity=float(getattr(candidate_basis_state, "complexity", 0.0) or 0.0)
                if isinstance(candidate_basis_state, BasisState)
                else 0.0,
                source_pool=str(source_pool),
                priority_rank=int(len(round_singleton_commits)),
                candidate_meta=candidate_meta,
                accept_meta=dict(accept_meta),
            )
            _record_round_commit(singleton_commit)
            round_singleton_commits.append(singleton_commit)

        best_singleton = _best_round_singleton(round_singleton_commits)
        early_round_commits = list(round_singleton_commits)
        if seed_row_mode and exact_scored_items:
            early_round_commits.extend(
                _run_pair_precommit(
                    base_basis_state=round_base_basis_state,
                    candidate_items=exact_scored_items,
                    seed_row_mode=bool(seed_row_mode),
                    round_beam_width=round_beam_width,
                    best_singleton=best_singleton,
                    attempted_pair_ids_sink=attempted_pair_ids,
                )
            )
        elif (not seed_row_mode) and exact_scored_items:
            early_round_commits.extend(
                _run_pair_normal(
                    base_basis_state=round_base_basis_state,
                    candidate_items=exact_scored_items,
                    seed_row_mode=seed_row_mode,
                    round_beam_width=round_beam_width,
                    best_singleton=best_singleton,
                    attempted_pair_ids_sink=attempted_pair_ids,
                )
            )

        selected_round_commit, selected_round_beam = _select_round_commit(
            early_round_commits,
            seed_row_mode=seed_row_mode,
            round_beam_width=round_beam_width,
        )
        rescue_trigger_reasons = _pair_rescue_trigger_reasons(
            selected_round_commit=selected_round_commit,
            early_round_commits=early_round_commits,
            best_singleton=best_singleton,
            base_basis_state=round_base_basis_state,
            seed_row_mode=bool(seed_row_mode),
        )
        if isinstance(round_debug_summary, dict):
            round_debug_summary["pair_rescue_trigger_reasons"] = list(rescue_trigger_reasons)
            round_debug_summary["early_selected_commit_kind"] = (
                str(selected_round_commit.kind) if isinstance(selected_round_commit, RoundCommitObject) else ""
            )
            round_debug_summary["early_selected_commit_profile"] = (
                str(selected_round_commit.profile) if isinstance(selected_round_commit, RoundCommitObject) else ""
            )
            round_debug_summary["early_selected_commit_key"] = (
                str(selected_round_commit.proposal_key) if isinstance(selected_round_commit, RoundCommitObject) else ""
            )
            if isinstance(selected_round_commit, RoundCommitObject):
                round_debug_summary["early_selected_commit_eff_mse"] = float(selected_round_commit.eff_mse)
        if rescue_trigger_reasons:
            prepared_pair_items_by_key = {
                str(item[2]): item
                for item in tuple(prepared_pair_items or ())
                if isinstance(item, tuple) and len(item) == 3
            }
            rescue_candidate_items: list[tuple[Any, Mapping[str, Any], str]] = []
            seen_rescue_keys: set[str] = set()
            for rescue_sort_key, rescue_row, rescue_proposal_key in prioritized_rows:
                rescue_key = str(rescue_proposal_key)
                if rescue_key in seen_rescue_keys:
                    continue
                rescue_candidate_items.append(
                    prepared_pair_items_by_key.get(
                        rescue_key,
                        (rescue_sort_key, rescue_row, rescue_key),
                    )
                )
                seen_rescue_keys.add(rescue_key)
            for rescue_key, rescue_item in prepared_pair_items_by_key.items():
                if rescue_key in seen_rescue_keys:
                    continue
                rescue_candidate_items.append(rescue_item)
                seen_rescue_keys.add(rescue_key)
            rescue_round_commits = _run_pair_rescue(
                base_basis_state=round_base_basis_state,
                candidate_items=rescue_candidate_items,
                seed_row_mode=seed_row_mode,
                round_beam_width=round_beam_width,
                round_singleton_metrics=round_singleton_metrics,
                best_singleton=best_singleton,
                blocked_pair_ids=attempted_pair_ids,
            )
            selected_round_commit, selected_round_beam = _select_round_commit(
                [*early_round_commits, *rescue_round_commits],
                seed_row_mode=seed_row_mode,
                round_beam_width=round_beam_width,
            )

        if isinstance(selected_round_commit, RoundCommitObject) and isinstance(selected_round_beam, tuple):
            basis_beam = selected_round_beam
            _prune_expanded_beam_state_ids()
            round_updated_state = (
                selected_round_commit.basis_state if isinstance(selected_round_commit.basis_state, BasisState) else None
            )
            closure_search_stats["basis_state_round_updates"] = int(
                closure_search_stats.get("basis_state_round_updates", 0) or 0
            ) + 1
            closure_search_stats["basis_state_round_commit_selected"] = int(
                closure_search_stats.get("basis_state_round_commit_selected", 0) or 0
            ) + 1
            selected_kind = "pair" if str(selected_round_commit.kind) == "pair" else "singleton"
            closure_search_stats[f"basis_state_round_commit_selected_{selected_kind}"] = int(
                closure_search_stats.get(f"basis_state_round_commit_selected_{selected_kind}", 0) or 0
            ) + 1
            if isinstance(round_debug_summary, dict):
                round_debug_summary["selected_commit_kind"] = str(selected_round_commit.kind)
                round_debug_summary["selected_commit_profile"] = str(selected_round_commit.profile)
                round_debug_summary["selected_commit_key"] = str(selected_round_commit.proposal_key)
                round_debug_summary["selected_commit_eff_mse"] = float(selected_round_commit.eff_mse)
                round_debug_summary["selected_commit_accept_reason"] = str(selected_round_commit.accept_reason)
            if str(selected_round_commit.kind) == "pair":
                _append_selected_pair_event(selected_round_commit)
            remaining_wall_now = _remaining_wall_seconds(wall_time_deadline)
            proposal_context_local = _build_residual_guided_context(
                base_context=proposal_context_local,
                basis_state=(
                    round_updated_state
                    if isinstance(round_updated_state, BasisState)
                    else (basis_beam[0] if basis_beam else None)
                ),
                basis_state_beam=basis_beam,
                families=family_tokens,
                total_budget=int(_remaining_scaffold_budget()),
                wall_time_remaining_s=remaining_wall_now,
                nvars=int(nvars),
                x_fit=x_fit,
                y_fit=y_fit,
                x_probe=x_probe,
                y_probe=y_probe,
                boost_pool_nodes=boost_pool_nodes,
            )
            continue

        if isinstance(round_debug_summary, dict):
            round_debug_summary.setdefault("selected_commit_kind", "")
            round_debug_summary.setdefault("selected_commit_profile", "")
            round_debug_summary.setdefault("selected_commit_key", "")

        if basis_beam and _has_unexpanded_beam_state():
            continue
        if wall_time_exceeded_fn():
            closure_search_stats["basis_state_controller_stop_reason"] = "deadline_exceeded"
        else:
            closure_search_stats["basis_state_controller_stop_reason"] = "no_basis_update"
        break
    if not str(closure_search_stats.get("basis_state_controller_stop_reason", "") or ""):
        if wall_time_exceeded_fn():
            closure_search_stats["basis_state_controller_stop_reason"] = "deadline_exceeded"
        elif scaffolds_used >= _effective_scaffold_cap():
            closure_search_stats["basis_state_controller_stop_reason"] = "scaffold_budget_exhausted"
        else:
            closure_search_stats["basis_state_controller_stop_reason"] = "round_limit"

    closure_search_stats["basis_state_beam_count"] = int(len(basis_beam))
    closure_search_stats["basis_state_beam"] = [state.to_dict() for state in basis_beam]
    closure_search_stats["proposal_context"] = proposal_context_local.to_dict()


def run_outer_scaffold_pass(**kwargs) -> None:
    """Backward-compatible wrapper for the old outer-scaffold entry point."""
    kwargs = dict(kwargs)
    rename_map = {
        "outer_scaffold_enable": "closure_search_enable",
        "outer_scaffold_stats": "closure_search_stats",
        "outer_scaffold_families": "closure_search_families",
        "outer_scaffold_max_scaffolds": "closure_search_max_proposals",
        "outer_scaffold_anchors_per_family": "closure_search_anchors_per_family",
        "outer_scaffold_preview_topk": "closure_search_preview_topk",
        "outer_scaffold_exact_topk": "closure_search_exact_topk",
        "outer_scaffold_min_valid_frac": "closure_search_min_valid_frac",
        "outer_scaffold_min_confidence": "closure_search_min_confidence",
        "outer_scaffold_periodic_min_valid_scale": "closure_search_periodic_min_valid_scale",
        "outer_scaffold_periodic_min_confidence_scale": "closure_search_periodic_min_confidence_scale",
        "outer_scaffold_transport_min_lin_rel": "closure_search_transport_min_lin_rel",
        "run_outer_scaffold_pass_impl": "run_closure_search_pass_impl",
    }
    for old_name, new_name in rename_map.items():
        if old_name in kwargs and new_name not in kwargs:
            kwargs[new_name] = kwargs.pop(old_name)
        else:
            kwargs.pop(old_name, None)
    kwargs.setdefault("closure_search_beam_width", 4)
    kwargs.setdefault("closure_search_seed_exact_topk", 6)
    kwargs.setdefault("closure_search_seed_beam_width", 4)
    kwargs.setdefault("closure_search_seed_scaffold_reserve", 8)
    kwargs.setdefault("closure_search_seed_family_cap", 2)
    kwargs.setdefault("closure_search_seed_exact_bound_bonus", 0.25)
    kwargs.setdefault("closure_search_pair_normal_enable", False)
    kwargs.setdefault("closure_search_pair_normal_topk", 3)
    kwargs.setdefault("closure_search_pair_normal_max_pairs", 1)
    kwargs.setdefault("closure_search_pair_rescue_enable", True)
    kwargs.setdefault("closure_search_pair_rescue_topk", 4)
    kwargs.setdefault("closure_search_pair_rescue_max_pairs", 6)
    kwargs.setdefault("closure_search_debug_topk", 0)
    return run_closure_search_pass(**kwargs)


__all__ = [
    "ProposalScoringState",
    "merge_route_status_counts",
    "record_route_status",
    "run_closure_search_pass",
    "run_outer_scaffold_pass",
    "score_native_candidate_basis_state",
    "score_external_candidate_expr",
]
