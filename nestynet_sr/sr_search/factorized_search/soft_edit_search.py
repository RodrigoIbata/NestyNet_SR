# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Soft local edit search over a tiny tangent-edit neighborhood."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Mapping, Sequence

import torch

from .inverse_core import _normalize_inverse_local_score_mode
from .inverse_spec_solver import (
    _apply_continuation_frames,
    _candidate_to_preview_row,
    _dedup_scored_candidates,
    _score_node_against_problem,
)
from .tangent_edit import (
    _alignment_score,
    _build_solver_context,
    _candidate_value_grad,
    _ensure_col,
    _enumerate_tangent_edit_nodes,
    _finite_float,
    _local_problem_from_payload,
    _normalize_grad_tensor,
    _select_diverse_candidates,
    _subset_rows,
)


def _rank_initial_edit_candidates(
    *,
    base_node,
    problem,
    subproblem_spec,
    nvars: int,
    pool_nodes,
    var_dims,
    x_rank: torch.Tensor,
    t_rank: torch.Tensor,
    target_grad_rank: torch.Tensor | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_dim = problem.target_dim
    active_vars = ()
    if subproblem_spec is not None:
        active_vars = tuple(int(v) for v in tuple(subproblem_spec.active_vars or ()))
        target_dim = subproblem_spec.target_dim

    need_grad = bool(target_grad_rank is not None)
    base_value_fit, base_grad_fit = _candidate_value_grad({"node": base_node}, x_rank, capture_gradients=need_grad)
    edit_candidates = _enumerate_tangent_edit_nodes(
        base_node,
        target_dim=target_dim,
        nvars=int(nvars),
        active_vars=active_vars,
        wrappers_left=int(problem.wrappers_left),
        pool_nodes=pool_nodes,
        var_dims=var_dims,
        x_rank=x_rank,
        t_rank=t_rank,
        target_grad_rank=target_grad_rank,
        base_value_fit=base_value_fit,
        base_grad_fit=base_grad_fit,
    )

    if base_value_fit is None:
        return [], {"status": "invalid_base_prediction"}
    if need_grad and base_grad_fit is None:
        target_grad_rank = None
        need_grad = False

    residual_fit = _ensure_col(t_rank) - _ensure_col(base_value_fit)
    grad_residual_fit = None
    if target_grad_rank is not None and base_grad_fit is not None:
        grad_residual_fit = target_grad_rank - base_grad_fit

    ranked: list[dict[str, Any]] = []
    for candidate in edit_candidates:
        cand_value_fit, cand_grad_fit = _candidate_value_grad(candidate, x_rank, capture_gradients=need_grad)
        if cand_value_fit is None:
            continue
        delta_fit = _ensure_col(cand_value_fit) - _ensure_col(base_value_fit)
        prediction_score = _alignment_score(residual_fit, delta_fit)
        gradient_score = None
        if grad_residual_fit is not None and cand_grad_fit is not None and base_grad_fit is not None:
            gradient_score = _alignment_score(grad_residual_fit, cand_grad_fit - base_grad_fit)
        total_score = float(prediction_score + (0.5 * float(gradient_score) if gradient_score is not None else 0.0))
        ranked.append(
            {
                **candidate,
                "value_fit": cand_value_fit.detach(),
                "grad_fit": None if cand_grad_fit is None else cand_grad_fit.detach(),
                "prediction_score": float(prediction_score),
                "gradient_score": None if gradient_score is None else float(gradient_score),
                "tangent_score": float(total_score),
            }
        )

    ranked.sort(
        key=lambda item: (
            -float(item.get("tangent_score", 0.0)),
            -float(item.get("gradient_score", 0.0) or 0.0),
            -float(item.get("prediction_score", 0.0)),
            float(item.get("fit_mse", float("inf"))),
        )
    )
    return ranked, {
        "status": "ok",
        "base_value_fit": base_value_fit.detach(),
        "base_grad_fit": None if base_grad_fit is None else base_grad_fit.detach(),
        "target_grad_rank": target_grad_rank,
    }


def _soft_loss(
    alpha_raw: torch.Tensor,
    *,
    base_value_fit: torch.Tensor,
    cand_value_fit: torch.Tensor,
    target_value_fit: torch.Tensor,
    base_grad_fit: torch.Tensor | None,
    cand_grad_fit: torch.Tensor | None,
    target_grad_fit: torch.Tensor | None,
    l1_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = torch.nn.functional.softplus(alpha_raw)
    denom = 1.0 + alpha.sum()
    base_weight = 1.0 / denom
    cand_weights = alpha / denom

    soft_value = (base_weight * base_value_fit) + torch.sum(
        cand_weights.view(-1, 1, 1) * cand_value_fit,
        dim=0,
    )
    loss = torch.mean((soft_value - target_value_fit) ** 2)
    if target_grad_fit is not None and base_grad_fit is not None and cand_grad_fit is not None:
        soft_grad = (base_weight * base_grad_fit) + torch.sum(
            cand_weights.view(-1, 1, 1) * cand_grad_fit,
            dim=0,
        )
        loss = loss + (0.5 * torch.mean((soft_grad - target_grad_fit) ** 2))
    loss = loss + (float(l1_weight) * torch.mean(alpha))
    return loss, alpha.detach()


def _optimize_soft_gates(
    *,
    ranked_candidates: Sequence[Mapping[str, Any]],
    base_value_fit: torch.Tensor,
    target_value_fit: torch.Tensor,
    base_grad_fit: torch.Tensor | None,
    target_grad_fit: torch.Tensor | None,
    steps: int,
    l1_weight: float,
) -> tuple[torch.Tensor | None, float | None, int]:
    if not ranked_candidates:
        return None, None, 0
    with torch.enable_grad():
        cand_value_fit = torch.stack([_ensure_col(item["value_fit"]) for item in ranked_candidates], dim=0)
        cand_grad_fit = None
        if target_grad_fit is not None and base_grad_fit is not None:
            grad_rows = [item.get("grad_fit", None) for item in ranked_candidates]
            if all(torch.is_tensor(value) for value in grad_rows):
                cand_grad_fit = torch.stack([value for value in grad_rows], dim=0)

        init_logits = []
        for item in ranked_candidates:
            score = max(0.0, float(item.get("tangent_score", 0.0)))
            init_logits.append(max(-4.0, min(2.0, (2.5 * score) - 3.0)))
        raw = torch.tensor(
            init_logits,
            dtype=base_value_fit.dtype,
            device=base_value_fit.device,
            requires_grad=True,
        )
        opt = torch.optim.Adam([raw], lr=0.15)

        best_loss: float | None = None
        best_alpha: torch.Tensor | None = None
        patience = 8
        since_improve = 0
        executed = 0
        for step in range(max(1, int(steps))):
            executed = step + 1
            opt.zero_grad(set_to_none=True)
            loss, alpha = _soft_loss(
                raw,
                base_value_fit=base_value_fit,
                cand_value_fit=cand_value_fit,
                target_value_fit=target_value_fit,
                base_grad_fit=base_grad_fit,
                cand_grad_fit=cand_grad_fit,
                target_grad_fit=target_grad_fit,
                l1_weight=float(l1_weight),
            )
            loss.backward()
            opt.step()

            loss_value = _finite_float(loss.item())
            if loss_value is None:
                continue
            if best_loss is None or loss_value < (best_loss - 1.0e-9):
                best_loss = float(loss_value)
                best_alpha = alpha.detach().clone()
                since_improve = 0
            else:
                since_improve += 1
                if since_improve >= patience:
                    break
        return best_alpha, best_loss, executed


def solve_local_soft_edit_preview_rows(
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
    pool_nodes=None,
    pool_dims=None,
    local_score_mode: str = "affine",
    preview_topk: int = 8,
    max_subtree_depth: int | None = None,
    soft_edit_steps: int = 64,
    soft_edit_l1: float = 1.0e-3,
    witness_loss_enable: bool = False,
    witness_grad_weight: float = 0.0,
    witness_d2_weight: float = 0.0,
    witness_diag_weight: float = 0.0,
    witness_physics_weight: float = 0.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    route_preview_topk = max(1, int(preview_topk))
    hole_path = tuple(int(v) for v in tuple(path or ()))
    mode_name = _normalize_inverse_local_score_mode(local_score_mode)
    solver_meta: dict[str, Any] = {
        "status": "started",
        "requested_preview_topk": int(route_preview_topk),
        "soft_edit_steps_requested": int(max(1, int(soft_edit_steps))),
        "soft_edit_steps_executed": 0,
        "soft_edit_l1": float(soft_edit_l1),
        "soft_edit_fit_points_used": 0,
        "target_gradient_used": False,
        "candidate_count_generated": 0,
        "candidate_count_ranked": 0,
        "candidate_count_optimized": 0,
        "candidate_count_scored": 0,
        "preview_count": 0,
        "best_soft_loss": None,
    }

    problem, continuation_frames, hole_sub, subproblem_spec = _local_problem_from_payload(spec_payload)
    if problem is None:
        solver_meta["status"] = "invalid_local_problem"
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}

    base_node = hole_sub
    if not isinstance(base_node, tuple) or not base_node:
        base_node = None
    if base_node is None:
        solver_meta["status"] = "missing_hole_sub"
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}

    witness = getattr(subproblem_spec, "witness", None) if subproblem_spec is not None else None
    target_grad_fit = None
    if witness is not None:
        target_grad_fit = _normalize_grad_tensor(getattr(witness, "grad_fit", None), x=problem.xf)
    x_rank, t_rank, target_grad_rank = _subset_rows(problem.xf, problem.tf, grad=target_grad_fit, max_rows=32)
    solver_meta["soft_edit_fit_points_used"] = int(x_rank.shape[0])
    solver_meta["target_gradient_used"] = bool(target_grad_rank is not None)

    ranked_candidates, rank_meta = _rank_initial_edit_candidates(
        base_node=base_node,
        problem=problem,
        subproblem_spec=subproblem_spec,
        nvars=int(nvars),
        pool_nodes=pool_nodes,
        var_dims=var_dims,
        x_rank=x_rank,
        t_rank=t_rank,
        target_grad_rank=target_grad_rank,
    )
    if str(rank_meta.get("status", "")) != "ok":
        solver_meta["status"] = str(rank_meta.get("status", "ranking_failed") or "ranking_failed")
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}

    solver_meta["candidate_count_generated"] = int(len(ranked_candidates))
    solver_meta["candidate_count_ranked"] = int(len(ranked_candidates))
    if not ranked_candidates:
        solver_meta["status"] = "no_soft_edit_candidates"
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}

    menu_cap = min(len(ranked_candidates), max(12, 4 * route_preview_topk))
    optimized_candidates = _select_diverse_candidates(ranked_candidates, menu_cap)
    solver_meta["candidate_count_optimized"] = int(len(optimized_candidates))

    base_value_fit = _ensure_col(rank_meta["base_value_fit"])
    base_grad_fit = rank_meta.get("base_grad_fit", None)
    best_alpha, best_loss, executed = _optimize_soft_gates(
        ranked_candidates=optimized_candidates,
        base_value_fit=base_value_fit,
        target_value_fit=_ensure_col(t_rank),
        base_grad_fit=base_grad_fit,
        target_grad_fit=target_grad_rank,
        steps=int(max(1, int(soft_edit_steps))),
        l1_weight=float(soft_edit_l1),
    )
    solver_meta["soft_edit_steps_executed"] = int(executed)
    solver_meta["best_soft_loss"] = None if best_loss is None else float(best_loss)
    if best_alpha is None:
        solver_meta["status"] = "soft_optimization_failed"
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}

    denom = 1.0 + torch.sum(best_alpha)
    best_weights = best_alpha / denom
    snap_order = sorted(
        range(len(optimized_candidates)),
        key=lambda idx: (
            -float(best_weights[idx].item()),
            -float(optimized_candidates[idx].get("tangent_score", 0.0)),
            int(idx),
        ),
    )
    snap_cap = min(len(snap_order), max(route_preview_topk, 2 * route_preview_topk))
    ctx = _build_solver_context(
        parent_node=parent_node,
        hole_path=hole_path,
        hole_sub=base_node,
        max_depth=int(max_depth),
        max_subtree_depth=int(max_subtree_depth if max_subtree_depth is not None else max_depth),
        nvars=int(nvars),
        poly_degree=int(poly_degree),
        var_dims=var_dims,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        local_score_mode=str(mode_name),
        target_mode=str(target_mode or ""),
        target_mapping_kind=str(target_mapping_kind or ""),
        witness_loss_enable=bool(witness_loss_enable),
        witness_grad_weight=float(witness_grad_weight),
        witness_d2_weight=float(witness_d2_weight),
        witness_diag_weight=float(witness_diag_weight),
        witness_physics_weight=float(witness_physics_weight),
    )

    scored_rows = []
    candidate_meta_by_key: dict[str, dict[str, Any]] = {}
    for soft_rank, idx in enumerate(snap_order[:snap_cap]):
        item = optimized_candidates[idx]
        scored = _score_node_against_problem(
            item["node"],
            problem=problem,
            ctx=ctx,
            source="soft_edit_search",
            generation_kind=f"soft_edit:{str(item.get('edit_kind', 'edit'))}",
        )
        if scored is None:
            continue
        payload = dict(scored.payload or {})
        payload["soft_weight"] = float(best_weights[idx].item())
        payload["soft_loss"] = None if best_loss is None else float(best_loss)
        payload["tangent_score"] = float(item.get("tangent_score", 0.0))
        payload["prediction_score"] = float(item.get("prediction_score", 0.0))
        payload["gradient_score"] = (
            None if item.get("gradient_score", None) is None else float(item.get("gradient_score", 0.0))
        )
        payload["edit_kind"] = str(item.get("edit_kind", "") or "")
        payload["anchor"] = (
            None
            if item.get("anchor", None) is None
            else str(getattr(item["anchor"], "__class__", tuple).__name__ == "tuple" and item["anchor"] or item["anchor"])
        )
        scored = replace(
            scored,
            family="soft_edit_search",
            payload=payload,
        )
        scored_rows.append(scored)
        candidate_meta_by_key[str(scored.node)] = {
            "soft_rank": int(soft_rank),
            "soft_weight": float(best_weights[idx].item()),
            "soft_loss": None if best_loss is None else float(best_loss),
            "tangent_score": float(item.get("tangent_score", 0.0)),
            "prediction_score": float(item.get("prediction_score", 0.0)),
            "gradient_score": (
                None if item.get("gradient_score", None) is None else float(item.get("gradient_score", 0.0))
            ),
            "edit_kind": str(item.get("edit_kind", "") or ""),
            "anchor": None if item.get("anchor", None) is None else str(item.get("anchor")),
        }

    deduped = _dedup_scored_candidates(scored_rows, complexity_penalty=0.0)
    solver_meta["candidate_count_scored"] = int(len(deduped))

    preview_rows: list[dict[str, Any]] = []
    for cand in deduped[:route_preview_topk]:
        try:
            wrapped_node = _apply_continuation_frames(cand.node, continuation_frames)
        except Exception:
            continue
        wrapped_cand = replace(cand, node=wrapped_node, generation_kind=f"followup:{str(cand.generation_kind)}")
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
        meta = candidate_meta_by_key.get(str(cand.node), {})
        row["proposal_family"] = "soft_edit_search"
        row["generation_source"] = "soft_edit_search"
        row["tuple_provenance"] = "soft_edit_search"
        row["soft_edit_rank"] = int(meta.get("soft_rank", 0))
        row["soft_edit_weight"] = _finite_float(meta.get("soft_weight", None))
        row["soft_edit_loss"] = _finite_float(meta.get("soft_loss", None))
        row["soft_edit_tangent_score"] = _finite_float(meta.get("tangent_score", None))
        row["soft_edit_prediction_score"] = _finite_float(meta.get("prediction_score", None))
        row["soft_edit_gradient_score"] = _finite_float(meta.get("gradient_score", None))
        row["soft_edit_kind"] = str(meta.get("edit_kind", "") or "")
        row["soft_edit_anchor"] = meta.get("anchor", None)
        preview_rows.append(row)

    preview_rows.sort(
        key=lambda row: (
            float(row.get("local_probe_mse", float("inf"))),
            float(row.get("local_fit_mse", float("inf"))),
            int(row.get("soft_edit_rank", 0)),
        )
    )
    preview_rows = preview_rows[:route_preview_topk]
    for local_rank, row in enumerate(preview_rows):
        row["local_rank"] = int(local_rank)
        row["local_candidate_count"] = int(len(preview_rows))

    solver_meta["preview_count"] = int(len(preview_rows))
    solver_meta["status"] = "ok" if preview_rows else "no_soft_edit_candidates"
    solver_meta["wall_seconds"] = float(time.perf_counter() - started)
    return {"rows": preview_rows, "solver_meta": solver_meta}


__all__ = ["solve_local_soft_edit_preview_rows"]
