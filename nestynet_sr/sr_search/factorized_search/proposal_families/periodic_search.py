# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from ..basis_state import BasisState, ProposalContext
from ..basis_scoring import fit_direct_linear_design, score_bound_closure
from ..closures import make_bound_closure, make_direct_periodic_closure
from ..expr_ast import (
    dims_eq,
    eval_node,
    is_valid_node,
    node_depth,
    node_dims,
    node_size,
    node_str,
    replace_at,
    simplify,
)
from .binding_search import (
    collect_shortlisted_hole_candidates,
    dedup_seed_blocks,
    evaluate_seed_blocks,
    filter_seed_blocks_for_dim,
)
from .closure_builders import (
    build_harmonic_periodic_candidate,
    build_linear_combo_candidate,
    build_literal_periodic_candidate,
    build_materialized_candidate,
)
from .closure_runners import (
    CustomDirectSearchPlan,
    PreparedCandidatesSearchPlan,
    PreparedClosureCandidate,
    execute_direct_search_plan,
)
from .closure_eval import finalize_direct_preview_rows, make_direct_preview_row, scaffold_parent_stats
from .common import deadline_exceeded, dim0
from .seed_blocks import (
    SeedBlock,
    build_recursive_seed_pool,
    seed_anchor_blocks,
)
from .types import OperatorApplication


def _scaffold_form(spec: OperatorApplication) -> str:
    return str(dict(spec.metadata or {}).get("form", "") or "").strip().lower()


def _bound_periodic_metadata(spec: Any) -> dict[str, Any]:
    bound = getattr(spec, "bound_closure", None)
    if hasattr(bound, "metadata"):
        return dict(getattr(bound, "metadata", {}) or {})
    return {}


def _bound_periodic_bindings(spec: Any) -> dict[str, Any]:
    bound = getattr(spec, "bound_closure", None)
    if hasattr(bound, "bindings"):
        return dict(getattr(bound, "bindings", {}) or {})
    return {}


def _valid_bound_node(raw: Any) -> tuple | None:
    node = getattr(raw, "node", raw)
    if isinstance(node, tuple) and is_valid_node(node):
        return node
    return None


def _valid_bound_nodes(raw_values: Any) -> tuple[tuple, ...]:
    out: list[tuple] = []
    for raw in list(raw_values or ()):
        node = _valid_bound_node(raw)
        if node is not None:
            out.append(node)
    return tuple(out)


def _best_preview_rows_mse(rows: Sequence[Mapping[str, Any]] | None) -> float:
    best = float("inf")
    for row in list(rows or ()):
        if not isinstance(row, Mapping):
            continue
        for key in ("local_probe_mse", "probe_mse", "mse"):
            try:
                value = float(row.get(key, float("inf")))
            except Exception:
                value = float("inf")
            if math.isfinite(value):
                best = min(best, value)
                break
    return best


def _periodic_phase_seed_from_node(raw: Any) -> tuple | None:
    node = _valid_bound_node(raw)
    if node is None:
        return None
    op = str(node[0])
    if op in {"sin", "cos"} and len(node) == 2:
        inner = _valid_bound_node(node[1])
        if inner is not None:
            return simplify(inner)
    return simplify(node)


def _periodic_phase_dim_ok(node: tuple | None, *, var_dims, target_dim) -> bool:
    if node is None or not is_valid_node(node):
        return False
    if var_dims is None or target_dim is None:
        return True
    try:
        node_dim = node_dims(node, var_dims)
    except Exception:
        node_dim = None
    return node_dim is not None and dims_eq(node_dim, target_dim)


def _prepend_pinned_candidates(
    candidate_nodes: Sequence[tuple[str, tuple]],
    *,
    pinned_rows: Sequence[tuple[str, tuple]],
    shortlist_count: int,
) -> list[tuple[str, tuple]]:
    out: list[tuple[str, tuple]] = []
    seen: set[str] = set()

    def _append(row: tuple[str, tuple] | None) -> None:
        if row is None:
            return
        source, node = row
        if not isinstance(node, tuple) or not is_valid_node(node):
            return
        key = str(node_str(node))
        if key in seen:
            return
        seen.add(key)
        out.append((str(source), node))

    for row in list(pinned_rows or ()):
        _append(row)
    for row in list(candidate_nodes or ()):
        _append(row)
        if len(out) >= max(0, int(shortlist_count)):
            break
    return out[: max(0, int(shortlist_count))]


def _normalize_periodic_candidate_rows(
    rows: Sequence[tuple[str, tuple]],
    *,
    var_dims,
    target_dim,
) -> list[tuple[str, tuple]]:
    out: list[tuple[str, tuple]] = []
    seen: set[str] = set()
    for source, raw_node in list(rows or ()):
        seed = _periodic_phase_seed_from_node(raw_node)
        if not _periodic_phase_dim_ok(seed, var_dims=var_dims, target_dim=target_dim):
            continue
        key = str(node_str(seed))
        if key in seen:
            continue
        seen.add(key)
        out.append((str(source), seed))
    return out


def _append_periodic_seed_rows_from_state(
    out: list[tuple[str, tuple]],
    seen: set[str],
    *,
    state: BasisState | None,
    var_dims,
    target_dim,
) -> None:
    if not isinstance(state, BasisState):
        return
    role_priority = {"carrier": 0, "harmonic_feature": 1, "feature": 2}
    ranked_rows: list[tuple[tuple[Any, ...], tuple]] = []
    for block in tuple(state.blocks or ()):
        latent_roles = tuple(getattr(block, "latent_bundle_roles", ()) or ())
        latent_nodes = tuple(getattr(block, "latent_bundle_nodes", ()) or ())
        head_roles = tuple(getattr(block, "head_bundle_roles", ()) or ())
        head_nodes = tuple(getattr(block, "head_bundle_nodes", ()) or ())
        for role, raw_node in (*zip(latent_roles, latent_nodes), *zip(head_roles, head_nodes)):
            role_token = str(role)
            if role_token not in role_priority:
                continue
            seed = _periodic_phase_seed_from_node(raw_node)
            if not _periodic_phase_dim_ok(seed, var_dims=var_dims, target_dim=target_dim):
                continue
            key = str(node_str(seed))
            if key in seen:
                continue
            ranked_rows.append(
                (
                    (
                        int(role_priority[role_token]),
                        int(node_size(seed)),
                        int(node_depth(seed)),
                        key,
                    ),
                    seed,
                )
            )
    ranked_rows.sort(key=lambda item: item[0])
    for _, seed in ranked_rows:
        key = str(node_str(seed))
        if key in seen:
            continue
        seen.add(key)
        out.append(("basis_seed", seed))


def _collect_basis_periodic_seed_rows(
    proposal_context: ProposalContext | None,
    *,
    var_dims,
    target_dim,
    limit: int,
) -> list[tuple[str, tuple]]:
    if not isinstance(proposal_context, ProposalContext) or int(limit) <= 0:
        return []
    out: list[tuple[str, tuple]] = []
    seen: set[str] = set()
    _append_periodic_seed_rows_from_state(
        out,
        seen,
        state=getattr(proposal_context, "basis_state", None),
        var_dims=var_dims,
        target_dim=target_dim,
    )
    for state in tuple(getattr(proposal_context, "basis_state_beam", ()) or ()):
        _append_periodic_seed_rows_from_state(
            out,
            seen,
            state=state,
            var_dims=var_dims,
            target_dim=target_dim,
        )
        if len(out) >= int(limit):
            break
    return out[: int(limit)]

def direct_periodic_scaffold_kind(spec: OperatorApplication) -> str | None:
    form = _scaffold_form(spec)
    if form in {"sin_base", "sin_add", "sin_mul"}:
        return "sin"
    if form in {"cos_base", "cos_add", "cos_mul"}:
        return "cos"
    meta = {**dict(getattr(spec, "metadata", {}) or {}), **_bound_periodic_metadata(spec)}
    kind = str(meta.get("periodic_kind", "") or "").strip().lower()
    if kind in {"sin", "cos"}:
        return kind
    return None


def periodic_scaffold_mode(spec: OperatorApplication) -> str:
    form = _scaffold_form(spec)
    if form.endswith("_add"):
        return "add"
    if form.endswith("_mul"):
        return "mul"
    meta = {**dict(getattr(spec, "metadata", {}) or {}), **_bound_periodic_metadata(spec)}
    if str(meta.get("form", "") or "").strip().lower().endswith("_add"):
        return "add"
    if str(meta.get("form", "") or "").strip().lower().endswith("_mul"):
        return "mul"
    return "base"


def collect_var_ids(node: tuple | None) -> set[int]:
    out: set[int] = set()

    def _visit(cur: Any) -> None:
        if not isinstance(cur, tuple) or not cur:
            return
        if str(cur[0]) == "var":
            try:
                out.add(int(cur[1]))
            except Exception:
                pass
            return
        for child in cur[1:]:
            _visit(child)

    _visit(node)
    return out


def rank_periodic_companions(
    *,
    companion_rows: Sequence[dict[str, Any]],
    residual_fit: torch.Tensor,
    envelope_node: tuple | None,
    anchor_node: tuple | None,
    topk: int,
) -> list[dict[str, Any]]:
    target = residual_fit.reshape(-1)
    target_norm = float(torch.linalg.vector_norm(target).item()) if target.numel() else 0.0
    ranked: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    envelope_key = str(node_str(envelope_node)) if isinstance(envelope_node, tuple) and is_valid_node(envelope_node) else ""
    anchor_key = str(node_str(anchor_node)) if isinstance(anchor_node, tuple) and is_valid_node(anchor_node) else ""
    for row in list(companion_rows or ()):
        block = row.get("block", None)
        if not isinstance(block, SeedBlock):
            continue
        key = str(node_str(block.node))
        if key == envelope_key:
            continue
        values = row.get("fit", None)
        if not torch.is_tensor(values):
            continue
        denom = float(torch.linalg.vector_norm(values).item()) if values.numel() else 0.0
        if denom <= 1.0e-12 or target_norm <= 1.0e-12:
            score = 0.0
        else:
            score = abs(float(torch.dot(values.reshape(-1), target).item())) / max(1.0e-12, denom * target_norm)
        rank_key = (
            0 if key == anchor_key and anchor_key else 1,
            -float(score),
            int(node_size(block.node)),
            key,
        )
        ranked.append((rank_key, row))
    ranked.sort(key=lambda item: item[0])
    return [row for _, row in ranked[: max(0, int(topk))]]


def try_direct_anchor_multiplier_lift(
    *,
    periodic_kind: str,
    hole_node: tuple,
    anchor_node: tuple,
    anchor_fit: torch.Tensor,
    anchor_probe: torch.Tensor,
    trig_fit: torch.Tensor,
    trig_probe: torch.Tensor,
    x_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_fit: torch.Tensor,
    y_probe: torch.Tensor,
    nvars: int,
    max_depth: int,
    var_dims,
    y_dims,
    base_probe_mse: float,
    base_coeffs: Sequence[float],
) -> dict[str, Any] | None:
    try:
        base_anchor_coeff = float(list(base_coeffs or [0.0, 1.0])[1])
    except Exception:
        base_anchor_coeff = 1.0
    if not math.isfinite(base_anchor_coeff) or abs(base_anchor_coeff - 1.0) < 5.0e-2:
        return None

    used_vars = collect_var_ids(anchor_node) | collect_var_ids(hole_node)
    best: dict[str, Any] | None = None

    for var_idx in range(max(0, int(nvars))):
        if int(var_idx) in used_vars:
            continue
        mult_node = simplify(("mul", ("var", int(var_idx)), anchor_node))
        if not is_valid_node(mult_node):
            continue
        try:
            expr = simplify(("add", (str(periodic_kind), hole_node), mult_node))
        except Exception:
            continue
        if not is_valid_node(expr):
            continue
        if int(node_depth(expr)) > int(max_depth):
            continue
        if var_dims is not None:
            try:
                mult_dim = node_dims(mult_node, var_dims)
                expr_dim = node_dims(expr, var_dims)
            except Exception:
                mult_dim = None
                expr_dim = None
            if mult_dim is None or expr_dim is None:
                continue
            if y_dims is not None and (not dims_eq(mult_dim, y_dims) or not dims_eq(expr_dim, y_dims)):
                continue
        try:
            mult_fit = eval_node(mult_node, x_fit)
            mult_probe = eval_node(mult_node, x_probe)
        except Exception:
            continue
        if (not torch.is_tensor(mult_fit)) or (not torch.is_tensor(mult_probe)):
            continue
        if (not torch.isfinite(mult_fit).all()) or (not torch.isfinite(mult_probe).all()):
            continue
        fit_ret = fit_direct_linear_design(
            design_fit=torch.stack(
                [
                    trig_fit.squeeze(-1),
                    mult_fit.squeeze(-1),
                    torch.ones(int(x_fit.shape[0]), dtype=x_fit.dtype, device=x_fit.device),
                ],
                dim=1,
            ),
            y_fit=y_fit,
            design_probe=torch.stack(
                [
                    trig_probe.squeeze(-1),
                    mult_probe.squeeze(-1),
                    torch.ones(int(x_probe.shape[0]), dtype=x_probe.dtype, device=x_probe.device),
                ],
                dim=1,
            ),
            y_probe=y_probe,
        )
        if fit_ret is None:
            continue
        fit_mse, probe_mse, coeffs = fit_ret
        try:
            coeff_trig, coeff_mult, coeff_bias = [float(v) for v in coeffs[:3]]
        except Exception:
            continue
        if not math.isfinite(probe_mse):
            continue
        if abs(coeff_mult) < 2.5e-1:
            continue
        if probe_mse > float(base_probe_mse) + 1.0e-9:
            continue
        rank_key = (
            float(probe_mse),
            abs(abs(float(coeff_mult)) - 1.0),
            int(node_size(expr)),
            str(node_str(expr)),
        )
        row = {
            "expr": expr,
            "probe_mse": float(probe_mse),
            "fit_mse": float(fit_mse),
            "coeffs": [float(coeff_trig), float(coeff_mult), float(coeff_bias)],
            "mult_node": mult_node,
            "mult_var_idx": int(var_idx),
            "rank_key": rank_key,
        }
        if best is None or tuple(rank_key) < tuple(best.get("rank_key", ()) or (float("inf"),)):
            best = row

    return best


def build_exact_bound_periodic_search_plan(
    spec: OperatorApplication,
    *,
    max_depth: int,
    x_fit: torch.Tensor,
    x_probe: torch.Tensor,
    var_dims,
    y_dims,
) -> PreparedCandidatesSearchPlan | None:
    periodic_kind = direct_periodic_scaffold_kind(spec)
    if periodic_kind is None:
        return None
    mode = periodic_scaffold_mode(spec)
    bindings = _bound_periodic_bindings(spec)
    hole_node = _valid_bound_node(bindings.get("carrier"))
    if hole_node is None:
        return None
    bound_anchor = spec.anchor_node if isinstance(spec.anchor_node, tuple) and is_valid_node(spec.anchor_node) else None
    envelope_node = _valid_bound_node(bindings.get("envelope"))
    if envelope_node is None and mode == "mul":
        envelope_node = bound_anchor
    if envelope_node is None:
        envelope_node = ("const", 1.0)
    anchor_node = _valid_bound_node(bindings.get("companion"))
    if anchor_node is None and mode == "add":
        anchor_node = bound_anchor
    companion_nodes = list(_valid_bound_nodes(bindings.get("companions")))
    if mode == "add" and isinstance(anchor_node, tuple) and is_valid_node(anchor_node):
        anchor_key = str(node_str(anchor_node))
        if all(str(node_str(node)) != anchor_key for node in companion_nodes):
            companion_nodes.append(anchor_node)
    trig_node = (str(periodic_kind), hole_node)
    try:
        trig_fit = eval_node(trig_node, x_fit)
        trig_probe = eval_node(trig_node, x_probe)
        env_fit = eval_node(envelope_node, x_fit)
        env_probe = eval_node(envelope_node, x_probe)
    except Exception:
        return None
    tensors = (trig_fit, trig_probe, env_fit, env_probe)
    if any((not torch.is_tensor(value)) for value in tensors):
        return None
    if any((not torch.isfinite(value).all()) for value in tensors):
        return None
    env_trig_node = simplify(("mul", envelope_node, trig_node))
    design_fit_cols = [env_fit.squeeze(-1) * trig_fit.squeeze(-1)]
    design_probe_cols = [env_probe.squeeze(-1) * trig_probe.squeeze(-1)]
    companion_sources: list[str] = []
    for node in companion_nodes:
        try:
            comp_fit = eval_node(node, x_fit)
            comp_probe = eval_node(node, x_probe)
        except Exception:
            return None
        if (not torch.is_tensor(comp_fit)) or (not torch.is_tensor(comp_probe)):
            return None
        if (not torch.isfinite(comp_fit).all()) or (not torch.isfinite(comp_probe).all()):
            return None
        design_fit_cols.append(comp_fit.squeeze(-1))
        design_probe_cols.append(comp_probe.squeeze(-1))
        companion_sources.append("bound_companion")
    design_fit_cols.append(torch.ones(int(x_fit.shape[0]), dtype=x_fit.dtype, device=x_fit.device))
    design_probe_cols.append(torch.ones(int(x_probe.shape[0]), dtype=x_probe.dtype, device=x_probe.device))
    built = build_linear_combo_candidate(
        bound_closure=make_direct_periodic_closure(
            scaffold_id=str(spec.scaffold_id),
            periodic_kind=str(periodic_kind),
            hole_node=hole_node,
            feature_node=trig_node,
            anchor_node=anchor_node if mode == "add" else None,
            envelope_node=envelope_node,
            companion_nodes=tuple(companion_nodes),
            harmonic_feature_nodes=(trig_node,),
        ),
        fit_matrix=torch.stack(design_fit_cols, dim=1),
        probe_matrix=torch.stack(design_probe_cols, dim=1),
        terms=[env_trig_node, *list(companion_nodes or ())],
        bias_index=int(len(design_fit_cols) - 1),
        design_metadata={
            "periodic_kind": str(periodic_kind),
            "source": "bound_carrier",
            "mode": str(mode),
            "envelope_source": "bound_envelope",
        },
        generation_source="closure_search_direct_exact_bound_periodic",
        tuple_provenance="closure_search_direct_periodic",
        proposal_family="closure_search_direct_periodic",
        local_mapping_kind="direct_linear_head",
        local_mapping_nparams=int(len(design_fit_cols)),
        direct_metadata={
            "feature_kind": str(periodic_kind),
            "hole_node": hole_node,
            "feature_node": trig_node,
            "harmonic_feature_nodes": [trig_node],
            "envelope_node": envelope_node,
            "companion_nodes": list(companion_nodes or ()),
            "envelope_source": "bound_envelope",
            "companion_sources": companion_sources,
            "periodic_mode": str(mode),
        },
    )
    literal_bound = make_bound_closure(
        closure_id=f"periodic:{str(periodic_kind)}:literal_atom",
        family="periodic",
        head_solver="identity",
        slot_specs=tuple(built.bound_closure.spec.slot_specs),
        bindings=dict(built.bound_closure.bindings or {}),
        diagnostics=dict(built.bound_closure.diagnostics or {}),
        metadata=dict(built.bound_closure.metadata or {}),
    )
    literal_built = build_materialized_candidate(
        bound_closure=literal_bound,
        payload=None,
        materializer="literal",
        materializer_payload={"expr": env_trig_node},
        design_metadata={
            "periodic_kind": str(periodic_kind),
            "source": "bound_carrier_literal",
            "mode": str(mode),
            "envelope_source": "bound_envelope",
        },
        generation_source="closure_search_direct_exact_bound_periodic_literal",
        tuple_provenance="closure_search_direct_periodic",
        proposal_family="closure_search_direct_periodic",
        local_mapping_kind="direct_literal_atom",
        local_mapping_nparams=1,
        direct_metadata={
            "feature_kind": str(periodic_kind),
            "hole_node": hole_node,
            "feature_node": trig_node,
            "harmonic_feature_nodes": [trig_node],
            "envelope_node": envelope_node,
            "companion_nodes": [],
            "envelope_source": "bound_envelope",
            "companion_sources": [],
            "periodic_mode": str(mode),
            "literal_exact_bound_atom": True,
        },
        fit_matrix=torch.stack([design_fit_cols[0]], dim=1),
        probe_matrix=torch.stack([design_probe_cols[0]], dim=1),
    )
    return PreparedCandidatesSearchPlan(
        candidates=(
            PreparedClosureCandidate(
                built=built,
                candidate_subtree_node=hole_node,
            ),
            PreparedClosureCandidate(
                built=literal_built,
                candidate_subtree_node=hole_node,
            ),
        ),
        var_dims=var_dims,
        parent_stats=scaffold_parent_stats(spec),
        meta={
            "execution_mode": "exact_bound",
            "bound_slot_names": [
                "carrier",
                *([] if envelope_node == ("const", 1.0) else ["envelope"]),
                *([] if not companion_nodes else ["companions"]),
            ],
        },
    )


def _evaluate_exact_bound_periodic(
    spec: OperatorApplication,
    *,
    max_depth: int,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    var_dims,
    y_dims,
    preview_topk: int,
    deadline_s: float | None,
    collect_direct_hole_candidates_fn,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]] | None:
    plan = build_exact_bound_periodic_search_plan(
        spec,
        max_depth=int(max_depth),
        x_fit=x_fit,
        x_probe=x_probe,
        var_dims=var_dims,
        y_dims=y_dims,
    )
    if plan is None:
        return None
    rows, status, meta = execute_direct_search_plan(
        plan,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        max_depth=int(max_depth),
        y_dims=y_dims,
        preview_topk=int(preview_topk),
        deadline_s=deadline_s,
        collect_direct_hole_candidates_fn=collect_direct_hole_candidates_fn,
    )
    meta_out = dict(meta or {})
    meta_out.setdefault("execution_mode", "exact_bound")
    return rows, status, meta_out


def _search_periodic_rebindings(
    spec: OperatorApplication,
    *,
    nvars: int,
    max_depth: int,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    var_dims,
    y_dims,
    pool_nodes,
    pool_dims,
    preview_topk: int,
    solver_kwargs: Mapping[str, Any],
    proposal_context: ProposalContext | None = None,
    deadline_s: float | None = None,
    collect_direct_hole_candidates_fn,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    periodic_kind = direct_periodic_scaffold_kind(spec)
    if periodic_kind is None:
        return [], "direct_unsupported_form", {}
    mode = periodic_scaffold_mode(spec)
    anchor_node = spec.anchor_node if isinstance(spec.anchor_node, tuple) and is_valid_node(spec.anchor_node) else None
    if mode == "add" and anchor_node is None:
        return [], "direct_missing_anchor", {}

    target_dim = dim0(var_dims) if var_dims is not None else None
    periodic_enum_max_depth = max(
        1,
        int(
            solver_kwargs.get(
                "direct_periodic_enum_max_depth",
                min(int(max_depth), 3),
            )
            or min(int(max_depth), 3)
        ),
    )
    periodic_enum_max_trees = max(
        32,
        int(solver_kwargs.get("direct_periodic_enum_max_trees", 256) or 256),
    )
    candidate_nodes, meta = collect_shortlisted_hole_candidates(
        collect_direct_hole_candidates_fn=collect_direct_hole_candidates_fn,
        nvars=int(nvars),
        enum_max_depth=int(periodic_enum_max_depth),
        enum_max_trees=int(periodic_enum_max_trees),
        var_dims=var_dims,
        target_dim=target_dim,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        shortlist_k=max(4, int(solver_kwargs.get("direct_periodic_feature_topk", 6) or 6)),
        deadline_s=deadline_s,
    )
    candidate_nodes = _normalize_periodic_candidate_rows(
        candidate_nodes,
        var_dims=var_dims,
        target_dim=target_dim,
    )
    if not candidate_nodes:
        if bool(dict(meta).get("deadline_exceeded", False)):
            return [], "direct_deadline_exceeded", dict(meta)
        return [], "direct_no_hole_candidates", dict(meta)
    bindings = _bound_periodic_bindings(spec)
    pinned_carrier_row: tuple[str, tuple] | None = None
    bound_carrier = _valid_bound_node(bindings.get("carrier"))
    if isinstance(bound_carrier, tuple) and is_valid_node(bound_carrier):
        carrier_dim_ok = True
        if var_dims is not None and target_dim is not None:
            try:
                carrier_dim = node_dims(bound_carrier, var_dims)
            except Exception:
                carrier_dim = None
            carrier_dim_ok = carrier_dim is not None and dims_eq(carrier_dim, target_dim)
        if carrier_dim_ok:
            carrier_node = simplify(bound_carrier)
            carrier_key = str(node_str(carrier_node))
            candidate_nodes = [
                (str(source), node)
                for source, node in list(candidate_nodes or ())
                if str(node_str(node)) != carrier_key
            ]
            pinned_carrier_row = ("bound_carrier", carrier_node)
    pinned_anchor_row: tuple[str, tuple] | None = None
    if isinstance(anchor_node, tuple) and is_valid_node(anchor_node):
        anchor_dim_ok = True
        if var_dims is not None and target_dim is not None:
            try:
                anchor_dim = node_dims(anchor_node, var_dims)
            except Exception:
                anchor_dim = None
            anchor_dim_ok = anchor_dim is not None and dims_eq(anchor_dim, target_dim)
        if anchor_dim_ok:
            anchor_key = str(node_str(simplify(anchor_node)))
            filtered_candidates = [
                (str(source), node)
                for source, node in list(candidate_nodes or ())
                if str(node_str(node)) != anchor_key
            ]
            pinned_anchor_row = ("anchor", simplify(anchor_node))
            candidate_nodes = filtered_candidates
    shortlist_count = max(4, int(solver_kwargs.get("direct_periodic_feature_topk", 6) or 6))
    basis_seed_rows = _collect_basis_periodic_seed_rows(
        proposal_context,
        var_dims=var_dims,
        target_dim=target_dim,
        limit=max(0, int(solver_kwargs.get("direct_periodic_basis_seed_topk", 3) or 3)),
    )
    candidate_nodes = _prepend_pinned_candidates(
        candidate_nodes,
        pinned_rows=[
            row
            for row in (
                pinned_anchor_row,
                *list(basis_seed_rows or ()),
                pinned_carrier_row,
            )
            if row is not None
        ],
        shortlist_count=int(shortlist_count),
    )

    anchor_fit = None
    anchor_probe = None
    if anchor_node is not None:
        try:
            anchor_fit = eval_node(anchor_node, x_fit)
            anchor_probe = eval_node(anchor_node, x_probe)
        except Exception:
            if mode == "add":
                return [], "direct_anchor_eval_failed", dict(meta)
            anchor_fit = None
            anchor_probe = None
        if mode == "add" and (
            (not torch.is_tensor(anchor_fit))
            or (not torch.is_tensor(anchor_probe))
            or (not torch.isfinite(anchor_fit).all())
            or (not torch.isfinite(anchor_probe).all())
        ):
            return [], "direct_anchor_nonfinite", dict(meta)

    parent_stats = scaffold_parent_stats(spec)
    rows: list[dict[str, Any]] = []
    seen_child_keys: set[str] = set()
    raw_candidate_count = 0
    scored_candidate_count = 0
    harmonic_candidate_count_raw = 0
    harmonic_candidate_count_scored = 0
    anchor_lift_attempts = 0
    anchor_lift_applied = 0

    seed_cap = max(3, int(solver_kwargs.get("direct_periodic_seed_topk", 3) or 3))
    base_seed_blocks = seed_anchor_blocks(
        nvars=int(nvars),
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        var_dims=var_dims,
        max_count=seed_cap,
    )
    if anchor_node is not None:
        anchor_dim = None
        if var_dims is not None:
            try:
                anchor_dim = node_dims(anchor_node, var_dims)
            except Exception:
                anchor_dim = None
        base_seed_blocks = [
            SeedBlock(
                node=anchor_node,
                dim=anchor_dim,
                source="anchor",
                builder="identity",
            ),
            *list(base_seed_blocks or ()),
        ]
    base_seed_blocks = dedup_seed_blocks(base_seed_blocks)
    seed_pool = build_recursive_seed_pool(
        base_seed_blocks,
        rounds=max(1, int(solver_kwargs.get("direct_periodic_seed_rounds", 2) or 2)),
        include_product=True,
        include_monomial=True,
        include_quadratic=False,
        product_max_arity=max(2, int(solver_kwargs.get("direct_periodic_product_max_arity", 3) or 3)),
        product_limit=max(4, int(solver_kwargs.get("direct_periodic_product_limit", 6) or 6)),
        monomial_limit=max(4, int(solver_kwargs.get("direct_periodic_monomial_limit", 8) or 8)),
        max_builder_depth=max(
            1,
            int(solver_kwargs.get("direct_periodic_seed_builder_depth", 2) or 2),
        ),
        max_nonlinear_depth=max(
            1,
            int(solver_kwargs.get("direct_periodic_seed_nonlinear_depth", 1) or 1),
        ),
    )
    harmonic_output_dim = y_dims if y_dims is not None else None
    envelope_blocks = filter_seed_blocks_for_dim(
        seed_pool,
        target_dim=harmonic_output_dim,
        var_dims=var_dims,
        drop_const=False,
    )
    companion_blocks = filter_seed_blocks_for_dim(
        base_seed_blocks,
        target_dim=harmonic_output_dim,
        var_dims=var_dims,
        drop_const=True,
    )
    envelope_eval_rows = evaluate_seed_blocks(
        envelope_blocks,
        x_fit=x_fit,
        x_probe=x_probe,
        deadline_s=deadline_s,
    )
    companion_eval_rows = evaluate_seed_blocks(
        companion_blocks,
        x_fit=x_fit,
        x_probe=x_probe,
        deadline_s=deadline_s,
    )
    envelope_topk = max(1, int(solver_kwargs.get("direct_periodic_envelope_topk", 3) or 3))
    companion_topk = max(0, int(solver_kwargs.get("direct_periodic_companion_topk", 2) or 2))
    harmonic_skip_probe_mse = float(
        solver_kwargs.get("direct_periodic_harmonic_skip_mse", 1.0e-10) or 1.0e-10
    )

    for source, hole_node in candidate_nodes:
        if deadline_exceeded(deadline_s):
            break
        raw_candidate_count += 1
        trig_node = (str(periodic_kind), hole_node)
        cos_node = ("cos", hole_node)
        sin_node = ("sin", hole_node)
        try:
            trig_fit = eval_node(trig_node, x_fit)
            trig_probe = eval_node(trig_node, x_probe)
            cos_fit = eval_node(cos_node, x_fit)
            cos_probe = eval_node(cos_node, x_probe)
            sin_fit = eval_node(sin_node, x_fit)
            sin_probe = eval_node(sin_node, x_probe)
        except Exception:
            continue
        if (
            (not torch.is_tensor(trig_fit))
            or (not torch.is_tensor(trig_probe))
            or (not torch.is_tensor(cos_fit))
            or (not torch.is_tensor(cos_probe))
            or (not torch.is_tensor(sin_fit))
            or (not torch.is_tensor(sin_probe))
        ):
            continue
        if (
            (not torch.isfinite(trig_fit).all())
            or (not torch.isfinite(trig_probe).all())
            or (not torch.isfinite(cos_fit).all())
            or (not torch.isfinite(cos_probe).all())
            or (not torch.isfinite(sin_fit).all())
            or (not torch.isfinite(sin_probe).all())
        ):
            continue

        cos_fit_1d = cos_fit.squeeze(-1)
        cos_probe_1d = cos_probe.squeeze(-1)
        sin_fit_1d = sin_fit.squeeze(-1)
        sin_probe_1d = sin_probe.squeeze(-1)
        skip_harmonic = False
        if torch.is_tensor(anchor_fit) and torch.is_tensor(anchor_probe):
            cheap_fit = fit_direct_linear_design(
                design_fit=torch.stack(
                    [
                        trig_fit.squeeze(-1),
                        anchor_fit.squeeze(-1),
                        torch.ones(int(x_fit.shape[0]), dtype=x_fit.dtype, device=x_fit.device),
                    ],
                    dim=1,
                ),
                y_fit=y_fit,
                design_probe=torch.stack(
                    [
                        trig_probe.squeeze(-1),
                        anchor_probe.squeeze(-1),
                        torch.ones(int(x_probe.shape[0]), dtype=x_probe.dtype, device=x_probe.device),
                    ],
                    dim=1,
                ),
                y_probe=y_probe,
            )
            if cheap_fit is not None:
                _cheap_fit_mse, cheap_probe_mse, _cheap_coeffs = cheap_fit
                if float(cheap_probe_mse) <= float(harmonic_skip_probe_mse):
                    skip_harmonic = True

        if not skip_harmonic:
            harmonic_env_rows: list[dict[str, Any]] = []
            for env_row in envelope_eval_rows:
                if deadline_exceeded(deadline_s):
                    break
                block = env_row.get("block", None)
                env_fit = env_row.get("fit", None)
                env_probe = env_row.get("probe", None)
                if not isinstance(block, SeedBlock) or (not torch.is_tensor(env_fit)) or (not torch.is_tensor(env_probe)):
                    continue
                harmonic_candidate_count_raw += 1
                env_cos_fit = env_fit * cos_fit_1d
                env_sin_fit = env_fit * sin_fit_1d
                env_cos_probe = env_probe * cos_probe_1d
                env_sin_probe = env_probe * sin_probe_1d
                base_design_fit = torch.stack(
                    [
                        env_cos_fit,
                        env_sin_fit,
                        torch.ones(int(x_fit.shape[0]), dtype=x_fit.dtype, device=x_fit.device),
                    ],
                    dim=1,
                )
                base_design_probe = torch.stack(
                    [
                        env_cos_probe,
                        env_sin_probe,
                        torch.ones(int(x_probe.shape[0]), dtype=x_probe.dtype, device=x_probe.device),
                    ],
                    dim=1,
                )
                base_fit = fit_direct_linear_design(
                    design_fit=base_design_fit,
                    y_fit=y_fit,
                    design_probe=base_design_probe,
                    y_probe=y_probe,
                )
                if base_fit is None:
                    continue
                _base_fit_mse, base_probe_mse, base_coeffs = base_fit
                coeff_t = torch.tensor(base_coeffs, dtype=x_fit.dtype, device=x_fit.device)
                residual_fit = y_fit.squeeze(-1) - (base_design_fit @ coeff_t)
                selected_companion_rows = rank_periodic_companions(
                    companion_rows=companion_eval_rows,
                    residual_fit=residual_fit,
                    envelope_node=block.node,
                    anchor_node=anchor_node,
                    topk=companion_topk,
                )
                harmonic_env_rows.append(
                    {
                        "block": block,
                        "env_fit": env_fit,
                        "env_probe": env_probe,
                        "base_probe_mse": float(base_probe_mse),
                        "selected_companion_rows": selected_companion_rows,
                    }
                )
            harmonic_env_rows.sort(
                key=lambda row: (
                    float(row.get("base_probe_mse", float("inf"))),
                    int(node_size(row["block"].node)),
                    str(node_str(row["block"].node)),
                )
            )
            for env_row in harmonic_env_rows[:envelope_topk]:
                if deadline_exceeded(deadline_s):
                    break
                block = env_row["block"]
                env_fit = env_row["env_fit"]
                env_probe = env_row["env_probe"]
                env_cos_fit = env_fit * cos_fit_1d
                env_sin_fit = env_fit * sin_fit_1d
                env_cos_probe = env_probe * cos_probe_1d
                env_sin_probe = env_probe * sin_probe_1d
                companion_nodes: list[tuple] = []
                design_fit_cols = [env_cos_fit, env_sin_fit]
                design_probe_cols = [env_cos_probe, env_sin_probe]
                companion_sources: list[str] = []
                for companion_row in list(env_row.get("selected_companion_rows", []) or ()):
                    companion_block = companion_row.get("block", None)
                    comp_fit = companion_row.get("fit", None)
                    comp_probe = companion_row.get("probe", None)
                    if not isinstance(companion_block, SeedBlock) or (not torch.is_tensor(comp_fit)) or (not torch.is_tensor(comp_probe)):
                        continue
                    companion_nodes.append(companion_block.node)
                    companion_sources.append(str(companion_block.source))
                    design_fit_cols.append(comp_fit)
                    design_probe_cols.append(comp_probe)
                design_fit_cols.append(torch.ones(int(x_fit.shape[0]), dtype=x_fit.dtype, device=x_fit.device))
                design_probe_cols.append(torch.ones(int(x_probe.shape[0]), dtype=x_probe.dtype, device=x_probe.device))
                built = build_harmonic_periodic_candidate(
                    scaffold_id=str(spec.scaffold_id),
                    periodic_kind=str(periodic_kind),
                    hole_node=hole_node,
                    trig_node=trig_node,
                    cos_node=cos_node,
                    sin_node=sin_node,
                    anchor_node=anchor_node,
                    envelope_node=block.node,
                    companion_nodes=tuple(companion_nodes),
                    fit_cols=design_fit_cols,
                    probe_cols=design_probe_cols,
                    source=str(source),
                    mode=str(mode),
                    envelope_source=str(block.source),
                    companion_sources=companion_sources,
                )
                scored = score_bound_closure(
                    built.bound_closure,
                    design=built.design,
                    y_fit=y_fit,
                    y_probe=y_probe,
                )
                if scored is None:
                    continue
                coeffs = [float(v) for v in list(scored["coeffs"] or ())]
                row = make_direct_preview_row(
                    bound_closure=built.bound_closure,
                    child_expr=scored["expr"],
                    fit_mse=float(scored["fit_mse"]),
                    probe_mse=float(scored["probe_mse"]),
                    max_depth=int(max_depth),
                    var_dims=var_dims,
                    y_dims=y_dims,
                    candidate_subtree_node=hole_node,
                    parent_sub_size=parent_stats["parent_sub_size"],
                    parent_sub_depth=parent_stats["parent_sub_depth"],
                    parent_size=parent_stats["parent_size"],
                    parent_depth=parent_stats["parent_depth"],
                    generation_source=built.generation_source,
                    tuple_provenance=built.tuple_provenance,
                    proposal_family=built.proposal_family,
                    local_mapping_kind=built.local_mapping_kind,
                    direct_metadata=built.direct_metadata,
                    seen_child_keys=seen_child_keys,
                    local_mapping_coeffs=coeffs,
                    local_mapping_nparams=built.local_mapping_nparams or int(len(coeffs)),
                )
                if row is None:
                    continue
                scored_candidate_count += 1
                harmonic_candidate_count_scored += 1
                rows.append(row)

        if mode == "add" and anchor_node is not None and torch.is_tensor(anchor_fit) and torch.is_tensor(anchor_probe):
            try:
                child_expr = simplify(replace_at(spec.parent_node, spec.hole_path, hole_node))
            except Exception:
                child_expr = None
            if isinstance(child_expr, tuple) and is_valid_node(child_expr):
                if int(node_depth(child_expr)) <= int(max_depth):
                    child_dim = None
                    if var_dims is not None:
                        try:
                            child_dim = node_dims(child_expr, var_dims)
                        except Exception:
                            child_dim = None
                    if var_dims is None or (
                        child_dim is not None and (y_dims is None or dims_eq(child_dim, y_dims))
                    ):
                        built = build_literal_periodic_candidate(
                            scaffold_id=str(spec.scaffold_id),
                            periodic_kind=str(periodic_kind),
                            hole_node=hole_node,
                            trig_node=trig_node,
                            anchor_node=anchor_node,
                            child_expr=child_expr,
                            fit_matrix=torch.stack(
                                [
                                    trig_fit.squeeze(-1),
                                    anchor_fit.squeeze(-1),
                                    torch.ones(int(x_fit.shape[0]), dtype=x_fit.dtype, device=x_fit.device),
                                ],
                                dim=1,
                            ),
                            probe_matrix=torch.stack(
                                [
                                    trig_probe.squeeze(-1),
                                    anchor_probe.squeeze(-1),
                                    torch.ones(int(x_probe.shape[0]), dtype=x_probe.dtype, device=x_probe.device),
                                ],
                                dim=1,
                            ),
                            source=str(source),
                        )
                        scored = score_bound_closure(
                            built.bound_closure,
                            design=built.design,
                            y_fit=y_fit,
                            y_probe=y_probe,
                        )
                        if scored is not None:
                            local_fit_mse = float(scored["fit_mse"])
                            local_probe_mse = float(scored["probe_mse"])
                            coeffs = list(scored["coeffs"])
                            child_expr = scored["expr"]

                            generation_source = built.generation_source
                            local_mapping_kind = built.local_mapping_kind
                            local_mapping_nparams = built.local_mapping_nparams or 3
                            direct_meta: dict[str, Any] = dict(built.direct_metadata or {})
                            lift_ret = try_direct_anchor_multiplier_lift(
                                periodic_kind=str(periodic_kind),
                                hole_node=hole_node,
                                anchor_node=anchor_node,
                                anchor_fit=anchor_fit,
                                anchor_probe=anchor_probe,
                                trig_fit=trig_fit,
                                trig_probe=trig_probe,
                                x_fit=x_fit,
                                x_probe=x_probe,
                                y_fit=y_fit,
                                y_probe=y_probe,
                                nvars=int(nvars),
                                max_depth=int(max_depth),
                                var_dims=var_dims,
                                y_dims=y_dims,
                                base_probe_mse=float(local_probe_mse),
                                base_coeffs=coeffs,
                            )
                            if abs(float(coeffs[1] if len(coeffs) > 1 else 1.0) - 1.0) >= 5.0e-2:
                                anchor_lift_attempts += 1
                            if isinstance(lift_ret, Mapping):
                                child_expr = lift_ret.get("expr", child_expr)
                                if isinstance(child_expr, tuple) and is_valid_node(child_expr):
                                    local_fit_mse = float(lift_ret.get("fit_mse", local_fit_mse))
                                    local_probe_mse = float(lift_ret.get("probe_mse", local_probe_mse))
                                    coeffs = list(lift_ret.get("coeffs", coeffs))
                                    generation_source = "closure_search_direct_anchor_lift"
                                    local_mapping_kind = "direct_anchor_lift"
                                    local_mapping_nparams = 3
                                    direct_meta.update(
                                        {
                                            "anchor_lift_var_idx": int(lift_ret.get("mult_var_idx", -1)),
                                            "anchor_lift_node": lift_ret.get("mult_node", None),
                                        }
                                    )
                                    anchor_lift_applied += 1

                            row = make_direct_preview_row(
                                bound_closure=built.bound_closure,
                                child_expr=child_expr,
                                fit_mse=float(local_fit_mse),
                                probe_mse=float(local_probe_mse),
                                max_depth=int(max_depth),
                                var_dims=var_dims,
                                y_dims=y_dims,
                                candidate_subtree_node=hole_node,
                                parent_sub_size=parent_stats["parent_sub_size"],
                                parent_sub_depth=parent_stats["parent_sub_depth"],
                                parent_size=parent_stats["parent_size"],
                                parent_depth=parent_stats["parent_depth"],
                                generation_source=str(generation_source),
                                tuple_provenance="closure_search_direct_periodic",
                                proposal_family="closure_search_direct_periodic",
                                local_mapping_kind=str(local_mapping_kind),
                                direct_metadata=direct_meta,
                                seen_child_keys=seen_child_keys,
                                local_mapping_coeffs=coeffs,
                                local_mapping_nparams=int(local_mapping_nparams),
                            )
                            if row is not None:
                                scored_candidate_count += 1
                                rows.append(row)

    return finalize_direct_preview_rows(
        rows,
        preview_topk=int(preview_topk),
        raw_candidate_count=int(raw_candidate_count),
        scored_candidate_count=int(scored_candidate_count),
        deadline_s=deadline_s,
        meta={
            **dict(meta),
            "basis_seed_candidates": int(len(basis_seed_rows)),
            "harmonic_candidate_count_raw": int(harmonic_candidate_count_raw),
            "harmonic_candidate_count_scored": int(harmonic_candidate_count_scored),
            "harmonic_envelope_count": int(len(envelope_eval_rows)),
            "harmonic_companion_count": int(len(companion_eval_rows)),
            "anchor_lift_attempts": int(anchor_lift_attempts),
            "anchor_lift_applied": int(anchor_lift_applied),
        },
    )


def _run_periodic_search_plan(
    spec: OperatorApplication,
    *,
    nvars: int,
    max_depth: int,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    var_dims,
    y_dims,
    pool_nodes,
    pool_dims,
    preview_topk: int,
    solver_kwargs: Mapping[str, Any],
    proposal_context: ProposalContext | None = None,
    deadline_s: float | None = None,
    collect_direct_hole_candidates_fn,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    allow_slot_rebinding = bool(dict(solver_kwargs or {}).get("allow_slot_rebinding", False))
    exact_result = _evaluate_exact_bound_periodic(
        spec,
        max_depth=int(max_depth),
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        var_dims=var_dims,
        y_dims=y_dims,
        preview_topk=int(preview_topk),
        deadline_s=deadline_s,
        collect_direct_hole_candidates_fn=collect_direct_hole_candidates_fn,
    )
    if exact_result is not None:
        exact_rows, _exact_status, _exact_meta = exact_result
        keep_threshold = float(dict(solver_kwargs or {}).get("direct_exact_bound_keep_mse", 1.0e-10))
        if exact_rows and not allow_slot_rebinding and _best_preview_rows_mse(exact_rows) <= float(keep_threshold):
            return exact_result

    rows, status, meta = _search_periodic_rebindings(
        spec,
        nvars=int(nvars),
        max_depth=int(max_depth),
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        var_dims=var_dims,
        y_dims=y_dims,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        preview_topk=int(preview_topk),
        solver_kwargs=solver_kwargs,
        proposal_context=proposal_context,
        deadline_s=deadline_s,
        collect_direct_hole_candidates_fn=collect_direct_hole_candidates_fn,
    )
    meta_out = dict(meta or {})
    meta_out.setdefault("execution_mode", "slot_search")
    if rows:
        return rows, status, meta_out
    if exact_result is not None:
        # Slot search scored nothing, so fall back to the exact-bound rows. Label them
        # as exact and record why the rebinding search came up empty: the caller has
        # already computed these same rows, and reporting them as slot output would
        # both double-count them and hide the reason no rebinding was proposed.
        exact_rows, exact_status, exact_meta = exact_result
        fallback_meta = dict(exact_meta or {})
        fallback_meta["execution_mode"] = "exact_bound"
        fallback_meta["slot_status"] = str(status)
        fallback_meta["slot_candidate_count_raw"] = int(meta_out.get("candidate_count_raw", 0) or 0)
        fallback_meta["slot_candidate_count_scored"] = int(meta_out.get("candidate_count_scored", 0) or 0)
        return exact_rows, exact_status, fallback_meta
    return rows, status, meta_out


def build_periodic_search_plan(
    spec: OperatorApplication,
    *,
    nvars: int,
    max_depth: int,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    var_dims,
    y_dims,
    pool_nodes,
    pool_dims,
    preview_topk: int,
    solver_kwargs: Mapping[str, Any],
    proposal_context: ProposalContext | None = None,
    deadline_s: float | None = None,
    collect_direct_hole_candidates_fn,
) -> CustomDirectSearchPlan:
    return CustomDirectSearchPlan(
        run_fn=_run_periodic_search_plan,
        kwargs={
            "spec": spec,
            "nvars": int(nvars),
            "max_depth": int(max_depth),
            "x_fit": x_fit,
            "y_fit": y_fit,
            "x_probe": x_probe,
            "y_probe": y_probe,
            "var_dims": var_dims,
            "y_dims": y_dims,
            "pool_nodes": pool_nodes,
            "pool_dims": pool_dims,
            "preview_topk": int(preview_topk),
            "solver_kwargs": solver_kwargs,
            "proposal_context": proposal_context,
            "deadline_s": deadline_s,
            "collect_direct_hole_candidates_fn": collect_direct_hole_candidates_fn,
        },
    )


def solve_direct_periodic_add_preview_rows(
    spec: OperatorApplication,
    *,
    nvars: int,
    max_depth: int,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    var_dims,
    y_dims,
    pool_nodes,
    pool_dims,
    preview_topk: int,
    solver_kwargs: Mapping[str, Any],
    proposal_context: ProposalContext | None = None,
    deadline_s: float | None = None,
    collect_direct_hole_candidates_fn,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    plan = build_periodic_search_plan(
        spec,
        nvars=int(nvars),
        max_depth=int(max_depth),
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        var_dims=var_dims,
        y_dims=y_dims,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        preview_topk=int(preview_topk),
        solver_kwargs=solver_kwargs,
        proposal_context=proposal_context,
        deadline_s=deadline_s,
        collect_direct_hole_candidates_fn=collect_direct_hole_candidates_fn,
    )
    return execute_direct_search_plan(
        plan,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        max_depth=int(max_depth),
        y_dims=y_dims,
        preview_topk=int(preview_topk),
        deadline_s=deadline_s,
        collect_direct_hole_candidates_fn=collect_direct_hole_candidates_fn,
    )


__all__ = [
    "build_periodic_search_plan",
    "collect_var_ids",
    "direct_periodic_scaffold_kind",
    "periodic_scaffold_mode",
    "rank_periodic_companions",
    "solve_direct_periodic_add_preview_rows",
    "try_direct_anchor_multiplier_lift",
]
