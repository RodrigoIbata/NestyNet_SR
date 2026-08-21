# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Recursive local SR solver for first-class subproblem specs."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import replace
from typing import Any, Mapping, Sequence

import torch

from .bridge import run_explorer
from .expr_ast import eval_node, node_size
from .expr_mapping import eval_mapping
from .inverse_core import _normalize_inverse_local_score_mode, _weighted_mse_cols
from .inverse_spec_solver import (
    _ScoredLocalCandidate,
    _apply_continuation_frames,
    _candidate_to_preview_row,
    _deserialize_local_problem,
    _proposal_satisfies_hard_constraints,
    _problem_witness_provenance,
    _subproblem_spec_to_local_problem,
)
from .local_teacher_loss import score_local_teacher_prediction_loss
from .subproblem_active_vars import (
    normalize_active_vars,
    remap_local_node_vars,
    subset_var_dims,
    subset_var_tensor,
)
from .subproblem_spec import deserialize_subproblem_spec


def _ensure_col(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"expected torch.Tensor, got {type(x).__name__}")
    if x.ndim == 1:
        return x.unsqueeze(-1)
    return x


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _local_problem_from_payload(
    spec_payload: Mapping[str, Any] | None,
) -> tuple[Any, list[dict[str, Any]], Any, str, Any]:
    payload = dict(spec_payload or {})
    subproblem_spec = deserialize_subproblem_spec(payload)
    if subproblem_spec is not None:
        problem, continuation_frames, hole_sub = _subproblem_spec_to_local_problem(subproblem_spec)
        if problem is not None:
            problem_id = str(subproblem_spec.problem_id or "")
            return problem, continuation_frames, hole_sub, problem_id, subproblem_spec
    problem = _deserialize_local_problem(payload.get("problem", {}))
    continuation_frames = [
        dict(frame)
        for frame in list(payload.get("continuation_frames", []) or [])
        if isinstance(frame, Mapping)
    ]
    hole_sub = payload.get("hole_sub", None)
    trace = tuple(str(v) for v in ((payload.get("problem", {}) or {}).get("trace", ()) or ()))
    digest = hashlib.sha1()
    for token in trace:
        digest.update(str(token).encode("utf-8", errors="ignore"))
        digest.update(b"\x1f")
    return problem, continuation_frames, hole_sub, digest.hexdigest()[:16], None


def _local_search_seed(problem_id: str, *, slate_id: str) -> int:
    token = str(problem_id or "") or str(slate_id or "")
    digest = hashlib.sha1(token.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16)


def _mapped_local_mse(
    node,
    *,
    mapping: Mapping[str, Any],
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor | None,
) -> float | None:
    try:
        pred = eval_node(node, x)
        pred = eval_mapping(_ensure_col(pred), mapping)
    except Exception:
        return None
    pred_col = _ensure_col(pred)
    mse = _weighted_mse_cols(_ensure_col(y), pred_col, w)
    return _finite_float(mse)


def _preview_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    probe_mse = _finite_float(row.get("local_probe_mse", None))
    fit_mse = _finite_float(row.get("local_fit_mse", None))
    expr = row.get("expr", None)
    try:
        size = int(node_size(expr)) if isinstance(expr, tuple) else 10**9
    except Exception:
        size = 10**9
    return (
        float("inf") if probe_mse is None else float(probe_mse),
        float("inf") if fit_mse is None else float(fit_mse),
        size,
    )


def _score_mapped_local_candidate(
    node,
    *,
    mapping: Mapping[str, Any],
    problem,
    var_dims,
    nvars: int,
    poly_degree: int,
    generation_kind: str | None,
    witness_loss_enable: bool,
    witness_grad_weight: float,
    witness_d2_weight: float,
    witness_diag_weight: float,
    witness_physics_weight: float,
) -> dict[str, float | None] | None:
    if not _proposal_satisfies_hard_constraints(
        node,
        problem=problem,
        var_dims=var_dims,
        nvars=int(nvars),
        generation_kind=generation_kind,
    ):
        return None
    if bool(witness_loss_enable):
        teacher_loss = score_local_teacher_prediction_loss(
            lambda xx: eval_mapping(_ensure_col(eval_node(node, xx)), mapping),
            x_fit=problem.xf,
            x_probe=problem.xp,
            target_fit=problem.tf,
            target_probe=problem.tp,
            w_fit=problem.wf,
            w_probe=problem.wp,
            target_grad_fit=problem.grad_fit,
            target_grad_probe=problem.grad_probe,
            target_d2_fit=problem.d2_fit,
            target_d2_probe=problem.d2_probe,
            poly_degree=int(poly_degree),
            mode="strict",
            grad_weight=float(witness_grad_weight),
            d2_weight=float(witness_d2_weight),
            diag_weight=float(witness_diag_weight),
            physics_weight=float(witness_physics_weight),
            target_diagnostics=problem.diagnostics,
        )
        if teacher_loss is not None:
            return {
                "local_fit_mse": float(teacher_loss.fit_total),
                "local_probe_mse": float(teacher_loss.probe_total),
                "value_fit_mse": float(teacher_loss.value_fit_loss),
                "value_probe_mse": float(teacher_loss.value_probe_loss),
                "witness_grad_loss": None if teacher_loss.grad_probe_loss is None else float(teacher_loss.grad_probe_loss),
                "witness_d2_loss": None if teacher_loss.d2_probe_loss is None else float(teacher_loss.d2_probe_loss),
                "witness_diag_loss": None if teacher_loss.diag_probe_loss is None else float(teacher_loss.diag_probe_loss),
                "witness_physics_loss": (
                    None if teacher_loss.physics_probe_loss is None else float(teacher_loss.physics_probe_loss)
                ),
                "witness_energy_total": float(teacher_loss.probe_total),
                "witness_fit_jet_source": str(teacher_loss.fit_jet_source or ""),
                "witness_probe_jet_source": str(teacher_loss.probe_jet_source or ""),
                "witness_fit_jet_requested_source": str(teacher_loss.fit_jet_requested_source or ""),
                "witness_probe_jet_requested_source": str(teacher_loss.probe_jet_requested_source or ""),
                "witness_fit_jet_fallback_used": bool(teacher_loss.fit_jet_fallback_used),
                "witness_probe_jet_fallback_used": bool(teacher_loss.probe_jet_fallback_used),
                "witness_numeric_jet_fallback_used": bool(
                    teacher_loss.fit_jet_fallback_used or teacher_loss.probe_jet_fallback_used
                ),
                "witness_exact_jet_used": bool(teacher_loss.exact_jet_used),
                "calibration_gap": max(0.0, float(teacher_loss.probe_total) - float(teacher_loss.value_probe_loss)),
            }
    provenance = _problem_witness_provenance(problem)
    local_fit_mse = _mapped_local_mse(
        node,
        mapping=mapping,
        x=problem.xf,
        y=problem.tf,
        w=problem.wf,
    )
    local_probe_mse = _mapped_local_mse(
        node,
        mapping=mapping,
        x=problem.xp,
        y=problem.tp,
        w=problem.wp,
    )
    if local_fit_mse is None or local_probe_mse is None:
        return None
    return {
        "local_fit_mse": float(local_fit_mse),
        "local_probe_mse": float(local_probe_mse),
        "value_fit_mse": float(local_fit_mse),
        "value_probe_mse": float(local_probe_mse),
        "witness_grad_loss": None,
        "witness_d2_loss": None,
        "witness_diag_loss": None,
        "witness_physics_loss": None,
        "witness_energy_total": float(local_probe_mse) if bool(witness_loss_enable) else float(local_probe_mse),
        "witness_fit_jet_source": str(provenance["witness_fit_jet_source"]),
        "witness_probe_jet_source": str(provenance["witness_probe_jet_source"]),
        "witness_fit_jet_requested_source": str(provenance["witness_fit_jet_requested_source"]),
        "witness_probe_jet_requested_source": str(provenance["witness_probe_jet_requested_source"]),
        "witness_fit_jet_fallback_used": bool(provenance["witness_fit_jet_fallback_used"]),
        "witness_probe_jet_fallback_used": bool(provenance["witness_probe_jet_fallback_used"]),
        "witness_numeric_jet_fallback_used": bool(provenance["witness_numeric_jet_fallback_used"]),
        "witness_exact_jet_used": bool(provenance["witness_exact_jet_used"]),
        "calibration_gap": 0.0,
    }


@torch.no_grad()
def solve_local_recursive_sr_preview_rows(
    *,
    parent_node,
    spec_payload: Mapping[str, Any],
    path: Sequence[int],
    target_mode: str,
    target_mapping_kind: str,
    beam_rank: int,
    slate_id: str,
    path_gain: float,
    max_depth: int,
    nvars: int,
    poly_degree: int,
    var_dims=None,
    local_score_mode: str = "affine",
    preview_topk: int = 4,
    exact_budget: int = 2,
    max_subtree_depth: int | None = None,
    witness_loss_enable: bool = False,
    witness_grad_weight: float = 0.0,
    witness_d2_weight: float = 0.0,
    witness_diag_weight: float = 0.0,
    witness_physics_weight: float = 0.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    mode_name = _normalize_inverse_local_score_mode(local_score_mode, default="affine")
    hole_path = tuple(int(v) for v in (path or ()))
    problem, continuation_frames, hole_sub, problem_id, subproblem_spec = _local_problem_from_payload(spec_payload)
    solver_meta: dict[str, Any] = {
        "proposal_family": "local_recursive_sr",
        "generation_source": "local_recursive_sr",
        "path": [int(v) for v in hole_path],
        "target_mode": str(target_mode or ""),
        "target_mapping_kind": str(target_mapping_kind or ""),
        "local_score_mode": str(mode_name),
        "preview_count": 0,
        "candidate_count_scored": 0,
        "child_spec_states": [],
        "child_spec_state_count": 0,
        "wall_seconds": 0.0,
        "status": "started",
    }
    if problem is None:
        solver_meta["status"] = "missing_spec_payload"
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}

    if int(problem.xf.shape[0]) < 4 or int(problem.xp.shape[0]) < 4:
        solver_meta["status"] = "insufficient_points"
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}

    base_nvars = int(problem.xf.shape[1]) if getattr(problem.xf, "ndim", 0) >= 2 else int(nvars)
    active_vars = ()
    if subproblem_spec is not None:
        active_vars = normalize_active_vars(tuple(subproblem_spec.active_vars or ()), nvars=int(base_nvars))
    use_active_var_subset = bool(active_vars) and len(active_vars) < int(base_nvars)
    search_x_fit = problem.xf
    search_x_probe = problem.xp
    search_var_dims = var_dims
    local_nvars = int(base_nvars)
    if bool(use_active_var_subset):
        search_x_fit = subset_var_tensor(problem.xf, active_vars)
        search_x_probe = subset_var_tensor(problem.xp, active_vars)
        search_var_dims = subset_var_dims(var_dims, active_vars=active_vars)
        local_nvars = int(search_x_fit.shape[1]) if getattr(search_x_fit, "ndim", 0) >= 2 else len(active_vars)
    local_max_depth = int(max_subtree_depth if max_subtree_depth is not None else max_depth)
    route_preview_topk = max(1, int(preview_topk))
    route_exact_budget = max(1, int(exact_budget))
    search_seed = _local_search_seed(problem_id, slate_id=str(slate_id))
    local_n_iter = max(512, min(4000, 512 * max(1, int(local_nvars)) * min(4, route_preview_topk)))

    raw_results = list(
        run_explorer(
            nvars=int(local_nvars),
            n_iter=int(local_n_iter),
            max_depth=int(max(1, local_max_depth)),
            poly_degree=int(poly_degree),
            seed=int(search_seed),
            var_dims=search_var_dims,
            y_dims=problem.target_dim,
            return_topk=int(route_preview_topk),
            dtype=problem.xf.dtype,
            x_fit_data=search_x_fit,
            y_fit_data=problem.tf,
            x_probe_data=search_x_probe,
            y_probe_data=problem.tp,
            simplify_skeletons=False,
            print_every=0,
            verbose=False,
        ) or []
    )
    solver_meta["bridge_candidate_count"] = int(len(raw_results))
    solver_meta["search_seed"] = int(search_seed)
    solver_meta["search_n_iter"] = int(local_n_iter)
    solver_meta["search_max_depth"] = int(max(1, local_max_depth))
    solver_meta["requested_preview_topk"] = int(route_preview_topk)
    solver_meta["exact_budget"] = int(route_exact_budget)
    solver_meta["active_vars"] = [int(v) for v in tuple(active_vars or ())]
    solver_meta["active_var_subsetting_used"] = bool(use_active_var_subset)
    solver_meta["search_nvars"] = int(local_nvars)

    preview_rows: list[dict[str, Any]] = []
    for result_rank, result in enumerate(raw_results):
        if not isinstance(result, Mapping):
            continue
        node = result.get("toy_ast", None)
        mapping = dict(result.get("mapping", {}) or {})
        if not isinstance(node, tuple) or not node or not mapping:
            continue
        try:
            global_node = remap_local_node_vars(node, active_vars=active_vars) if bool(use_active_var_subset) else node
        except Exception:
            continue
        score = _score_mapped_local_candidate(
            global_node,
            mapping=mapping,
            problem=problem,
            var_dims=var_dims,
            nvars=int(nvars),
            poly_degree=int(poly_degree),
            generation_kind="recursive_local_sr",
            witness_loss_enable=bool(witness_loss_enable),
            witness_grad_weight=float(witness_grad_weight),
            witness_d2_weight=float(witness_d2_weight),
            witness_diag_weight=float(witness_diag_weight),
            witness_physics_weight=float(witness_physics_weight),
        )
        if score is None:
            continue
        cand = _ScoredLocalCandidate(
            node=global_node,
            local_probe_mse=float(score["local_probe_mse"]),
            local_fit_mse=float(score["local_fit_mse"]),
            source="recursive_local_sr",
            generation_kind="recursive_local_sr",
            recursion_depth=int(problem.recursion_level),
            confidence=float(problem.confidence),
            valid_frac=float(problem.valid_frac),
            trace=tuple(problem.trace or ()),
            family="recursive_local_sr",
            payload={
                "mapping": mapping,
                "mse_raw": _finite_float(result.get("mse_raw", None)),
                "mse_eff": _finite_float(result.get("mse_eff", None)),
                "result_rank": int(result_rank),
            },
            surrogate_probe_mse=_finite_float(result.get("mse_eff", score["local_probe_mse"])),
            surrogate_fit_mse=float(score["value_fit_mse"]),
            value_probe_mse=float(score["value_probe_mse"]),
            value_fit_mse=float(score["value_fit_mse"]),
            witness_value_loss=float(score["value_probe_mse"]),
            witness_grad_loss=None if score["witness_grad_loss"] is None else float(score["witness_grad_loss"]),
            witness_d2_loss=None if score["witness_d2_loss"] is None else float(score["witness_d2_loss"]),
            witness_diag_loss=None if score["witness_diag_loss"] is None else float(score["witness_diag_loss"]),
            witness_physics_loss=(
                None if score["witness_physics_loss"] is None else float(score["witness_physics_loss"])
            ),
            witness_energy_total=None if score["witness_energy_total"] is None else float(score["witness_energy_total"]),
            witness_fit_jet_source=str(score.get("witness_fit_jet_source", "") or ""),
            witness_probe_jet_source=str(score.get("witness_probe_jet_source", "") or ""),
            witness_fit_jet_requested_source=str(score.get("witness_fit_jet_requested_source", "") or ""),
            witness_probe_jet_requested_source=str(score.get("witness_probe_jet_requested_source", "") or ""),
            witness_fit_jet_fallback_used=bool(score.get("witness_fit_jet_fallback_used", False)),
            witness_probe_jet_fallback_used=bool(score.get("witness_probe_jet_fallback_used", False)),
            witness_numeric_jet_fallback_used=bool(score.get("witness_numeric_jet_fallback_used", False)),
            witness_exact_jet_used=bool(score.get("witness_exact_jet_used", False)),
            calibration_gap=float(score["calibration_gap"]),
        )
        try:
            wrapped_node = _apply_continuation_frames(cand.node, continuation_frames)
        except Exception:
            continue
        wrapped_cand = replace(
            cand,
            node=wrapped_node,
            generation_kind=f"followup:{str(cand.generation_kind)}",
        )
        row = _candidate_to_preview_row(
            wrapped_cand,
            parent_node=parent_node,
            beam_state={
                "sub": hole_sub,
                "target_mode": str(target_mode or ""),
                "target_mapping_kind": str(target_mapping_kind or ""),
                "path_gain": float(path_gain),
                "poly_degree": int(poly_degree),
            },
            beam_rank=int(beam_rank),
            slate_id=str(slate_id),
            path=hole_path,
            xf=problem.xf,
            tf=problem.tf,
            xp=problem.xp,
            tp=problem.tp,
            max_depth=int(max_depth),
            var_dims=var_dims,
            local_score_mode=str(mode_name),
        )
        if row is None:
            continue
        row["proposal_family"] = "local_recursive_sr"
        row["generation_source"] = "local_recursive_sr"
        row["tuple_provenance"] = "local_recursive_sr"
        row["local_recursive_sr_result_rank"] = int(result_rank)
        row["local_recursive_sr_mse_eff"] = _finite_float(result.get("mse_eff", None))
        row["local_recursive_sr_mse_raw"] = _finite_float(result.get("mse_raw", None))
        preview_rows.append(row)

    preview_rows.sort(key=_preview_sort_key)
    preview_rows = preview_rows[: min(route_preview_topk, route_exact_budget)]
    for local_rank, row in enumerate(preview_rows):
        row["local_rank"] = int(local_rank)
        row["local_candidate_count"] = int(len(preview_rows))

    solver_meta["candidate_count_scored"] = int(len(preview_rows))
    solver_meta["preview_count"] = int(len(preview_rows))
    solver_meta["status"] = "ok" if preview_rows else "no_recursive_sr_candidates"
    solver_meta["wall_seconds"] = float(time.perf_counter() - started)
    return {
        "rows": preview_rows,
        "solver_meta": solver_meta,
    }


__all__ = ["solve_local_recursive_sr_preview_rows"]
