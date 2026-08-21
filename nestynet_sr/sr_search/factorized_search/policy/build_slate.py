# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Build-slate helper logic layered on top of the factorized symbolic search explorer."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ..shared_opportunity import shared_opportunity_row_dict


def normalize_controller_build_slate_actions(
    action_names: Sequence[Any] | None,
    *,
    default_actions: Sequence[str],
    action_id_by_name: Mapping[str, int],
    allowed_action_ids: Sequence[int],
) -> tuple[int, ...]:
    raw_names = tuple(action_names or default_actions)
    out: list[int] = []
    seen: set[int] = set()
    allowed = {int(action_id) for action_id in allowed_action_ids}
    for name in raw_names:
        action_id = action_id_by_name.get(str(name).strip(), None)
        if action_id is None:
            continue
        if int(action_id) not in allowed:
            continue
        if int(action_id) in seen:
            continue
        seen.add(int(action_id))
        out.append(int(action_id))
    return tuple(out)


def controller_selected_action_path(
    node,
    action,
    *,
    controller_policy_guidance: Mapping[str, Any] | None = None,
    macro_decision: Any = None,
    macro_state: Any = None,
    inverse_gate_diag: Any = None,
    action_candidate_paths_fn,
    coerce_guided_path_fn,
):
    candidates = action_candidate_paths_fn(node, action)
    if not candidates:
        return None, ""
    candidate_set = set(candidates)
    seen: set[tuple[int, ...]] = set()

    def _choose(paths_like, source: str):
        for path_like in paths_like:
            path = coerce_guided_path_fn(path_like)
            if path is None or path in seen:
                continue
            seen.add(path)
            if path in candidate_set:
                return path, source
        return None, ""

    macro_decision_paths = []
    decision_path = getattr(macro_decision, "selected_path", ())
    if decision_path:
        macro_decision_paths.append(decision_path)
    best_path = getattr(macro_decision, "learned_best_path", ())
    if best_path:
        macro_decision_paths.append(best_path)
    path, source = _choose(macro_decision_paths[:1], "macro_decision_path_tuple")
    if path is not None:
        return path, source
    path, source = _choose(macro_decision_paths[1:], "macro_decision_best_path")
    if path is not None:
        return path, source

    policy_paths = []
    if isinstance(controller_policy_guidance, Mapping):
        policy_paths.append(controller_policy_guidance.get("selected_path", None))
        policy_paths.extend(list(controller_policy_guidance.get("candidate_paths", []) or []))
    path, source = _choose(policy_paths[:1], "critic_path_head")
    if path is not None:
        return path, source
    path, source = _choose(policy_paths[1:], "critic_path_candidates")
    if path is not None:
        return path, source

    macro_paths = []
    best_path = getattr(macro_state, "best_path", ())
    if best_path:
        macro_paths.append(best_path)
    for row in getattr(macro_state, "path_summaries", ()) or ():
        path_like = getattr(row, "path", ())
        if path_like:
            macro_paths.append(path_like)
    path, source = _choose(macro_paths[:1], "macro_state_best_path")
    if path is not None:
        return path, source
    path, source = _choose(macro_paths[1:], "macro_state_path_summary")
    if path is not None:
        return path, source

    inverse_paths = []
    best_path = getattr(inverse_gate_diag, "best_path", ())
    if best_path:
        inverse_paths.append(best_path)
    inverse_paths.extend(list(getattr(inverse_gate_diag, "candidate_paths", ()) or ()))
    path, source = _choose(inverse_paths[:1], "inverse_gate_best_path")
    if path is not None:
        return path, source
    path, source = _choose(inverse_paths[1:], "inverse_gate_candidates")
    if path is not None:
        return path, source

    return None, ""


def collect_controller_build_slate(
    *,
    parent_key: Any,
    parent_rec: Any,
    n_evaluated: int,
    seed_search: int | None,
    active_actions: Sequence[int],
    action_names: Sequence[Any] | None,
    max_actions: int,
    controller_policy_guidance: Any,
    macro_decision: Any,
    macro_state: Any,
    inverse_gate_diag: Any,
    x_fit,
    y_fit,
    x_probe,
    y_probe,
    proj,
    fp_mode: str,
    q_scale: float,
    q_clip: float,
    poly_degree: int,
    refine_enable: bool,
    refine_cfg: dict[str, Any] | None,
    refine_state: dict[str, Any] | None,
    best_raw_mse_struct: float,
    best_raw_mse: float,
    early_stop_mse: float,
    complexity_penalty: float,
    boost_enable: bool,
    boost_pool_nodes: Sequence[Any],
    boost_pool_phi_fit,
    boost_pool_norms_fit,
    boost_pool_phi,
    boost_pool_norms,
    boost_pool_dims: Sequence[Any] | None,
    boost_selection_split: str,
    boost_ridge: float | None,
    boost_include_parent: bool,
    boost_from_scratch_prob: float,
    boost_prune_rel: float,
    boost_max_terms: int,
    boost_topk_try: int,
    boost_min_rel_improve: float,
    max_depth: int,
    nvars: int,
    var_dims: Sequence[Any] | None,
    y_dims: Any,
    reach: Any,
    preview_only: bool,
    normalize_actions_fn,
    controller_selected_action_path_fn,
    controller_build_slate_id_fn,
    derived_controller_build_rng_fn,
    apply_residual_action_fn,
    apply_boost_action_fn,
    apply_action_fn,
    simplify_fn,
    node_str_fn,
    node_size_fn,
    node_depth_fn,
    node_dims_fn,
    dims_eq_fn,
    score_expr_fn,
    shared_candidate_row_dict_fn,
    mapping_cost_fn,
    action_name_map: Mapping[int, str],
    path_select_action_ids: Sequence[int],
    residual_action_id: int,
    boost_action_id: int,
) -> dict[str, Any]:
    normalized_actions = [
        int(action_id)
        for action_id in normalize_actions_fn(action_names)
        if int(action_id) in set(int(v) for v in active_actions)
    ]
    normalized_actions = normalized_actions[: max(1, int(max_actions))]
    if not normalized_actions:
        return {
            "controller_build_slate_id": "",
            "controller_build_slate_count": 0,
            "controller_build_slate_exact_observed_count": 0,
            "controller_build_slate_preview_only": bool(preview_only),
            "controller_build_slate": [],
            "build_opportunity_slate_id": "",
            "build_opportunity_slate_count": 0,
            "build_opportunity_slate_preview_only": bool(preview_only),
            "build_opportunity_slate": [],
        }
    parent_expr = getattr(parent_rec, "best_expr", None)
    parent_eff = float(getattr(parent_rec, "best_mse", float("inf")))
    parent_size = float(node_size_fn(parent_expr)) if parent_expr is not None else 0.0
    parent_depth = float(node_depth_fn(parent_expr)) if parent_expr is not None else 0.0
    slate_id = controller_build_slate_id_fn(seed_search, n_evaluated, parent_key, parent_expr)
    opportunity_slate_id = f"buildopp_{str(slate_id)}"
    rows: list[dict[str, Any]] = []
    opportunity_rows: list[dict[str, Any]] = []
    exact_observed = 0
    dm = var_dims is not None
    base_seed = int(seed_search if seed_search is not None else n_evaluated)
    active_action_set = {int(v) for v in active_actions}
    path_select_action_set = {int(v) for v in path_select_action_ids}

    for ordinal, action_id in enumerate(normalized_actions):
        if int(action_id) not in active_action_set:
            continue
        action_name = str(action_name_map.get(int(action_id), f"action_{int(action_id)}"))
        action_path = None
        action_path_source = ""
        if int(action_id) in path_select_action_set:
            action_path, action_path_source = controller_selected_action_path_fn(
                parent_expr,
                action_id,
                controller_policy_guidance=controller_policy_guidance,
                macro_decision=macro_decision,
                macro_state=macro_state,
                inverse_gate_diag=inverse_gate_diag,
            )
        local_rng = derived_controller_build_rng_fn(base_seed, parent_key, parent_expr, int(action_id), int(ordinal))
        row_out: dict[str, Any] = {
            "slate_id": str(slate_id),
            "slate_rank": int(ordinal),
            "action_id": int(action_id),
            "action": str(action_name),
            "path": [] if action_path is None else [int(v) for v in action_path],
            "path_source": str(action_path_source or ("guided" if action_path is not None else "random")),
            "tuple_provenance": "build_slate",
            "exact_child_score_observed": False,
            "build_preview_only": bool(preview_only),
        }

        def _append_opportunity_row(*, candidate_observed: bool) -> None:
            opportunity_rows.append(shared_opportunity_row_dict({
                "route_source": "build",
                "opportunity_type": "build_action",
                "decision_id": str(opportunity_slate_id),
                "beam_id": f"{str(opportunity_slate_id)}:{int(ordinal)}",
                "parent_expr": "" if parent_expr is None else str(node_str_fn(parent_expr)),
                "action": str(action_name),
                "path": [] if action_path is None else [int(v) for v in action_path],
                "path_source": str(action_path_source or ("guided" if action_path is not None else "random")),
                "budget_exact_spent": 1 if bool(row_out.get("exact_child_score_observed", False)) else 0,
                "budget_remaining": 0,
                "budget_widen_spent": 0,
                "budget_micro_spent": 0,
                "current_best_child_expr": str(row_out.get("child_expr", "") or row_out.get("child_key", "") or ""),
                "current_best_child_eff_mse": row_out.get("child_eff_mse", None),
                "candidate_count_observed": 1 if bool(candidate_observed) else 0,
                "candidate_count_unique": 1 if bool(candidate_observed) else 0,
                "preview_candidate_count_total": 1 if str(row_out.get("status", "") or "") not in {"proposal_none", ""} else 0,
                "preview_candidate_count_unique_total": 1 if str(row_out.get("status", "") or "") not in {"proposal_none", ""} else 0,
                "exact_child_observed_count": 1 if bool(row_out.get("exact_child_score_observed", False)) else 0,
                "status": str(row_out.get("status", "") or ""),
                "tuple_provenance": str(row_out.get("tuple_provenance", "build_slate") or "build_slate"),
                "proposal_source": "build_slate",
                "candidate_child_size": row_out.get("candidate_child_size", None),
                "candidate_child_depth": row_out.get("candidate_child_depth", None),
                "candidate_child_size_delta": row_out.get("candidate_child_size_delta", None),
                "candidate_child_depth_delta": row_out.get("candidate_child_depth_delta", None),
                "candidate_root_op": str(row_out.get("candidate_root_op", "") or ""),
                "path_length": int(row_out.get("path_length", 0) or 0),
                "build_preview_only": bool(preview_only),
            }, route_source="build"))
        expr = None
        if int(action_id) == int(residual_action_id):
            expr = apply_residual_action_fn(
                parent_expr,
                getattr(parent_rec, "mapping", None),
                x_fit,
                y_fit,
                x_probe,
                y_probe,
                boost_pool_nodes,
                boost_pool_phi,
                boost_pool_norms,
                boost_pool_dims,
                local_rng,
                max_depth,
                nvars,
                poly_degree,
                var_dims=var_dims,
                topk=5,
                complexity_penalty=complexity_penalty,
            )
        elif int(action_id) == int(boost_action_id) and bool(boost_enable):
            expr = apply_boost_action_fn(
                parent_expr,
                x_fit,
                y_fit,
                x_probe,
                y_probe,
                boost_pool_nodes,
                boost_pool_phi_fit,
                boost_pool_norms_fit,
                boost_pool_phi,
                boost_pool_norms,
                boost_pool_dims,
                local_rng,
                max_depth,
                nvars,
                poly_degree,
                var_dims=var_dims,
                y_dims=y_dims,
                max_terms=boost_max_terms,
                topk_try=boost_topk_try,
                min_rel_improve=boost_min_rel_improve,
                selection_split=boost_selection_split,
                ridge=boost_ridge,
                include_parent=boost_include_parent,
                from_scratch_prob=boost_from_scratch_prob,
                prune_rel=boost_prune_rel,
                complexity_penalty=complexity_penalty,
            )
        elif int(action_id) in path_select_action_set:
            expr = apply_action_fn(
                parent_expr,
                action_id,
                local_rng,
                max_depth,
                nvars,
                var_dims=var_dims,
                reach=reach,
                path=action_path,
            )
        if expr is None:
            row_out["status"] = "proposal_none"
            _append_opportunity_row(candidate_observed=False)
            rows.append(shared_candidate_row_dict_fn({**row_out, "route_source": "build"}, route_source="build"))
            continue
        expr = simplify_fn(expr)
        while isinstance(expr, tuple) and expr and expr[0] == "neg":
            expr = expr[1]
        if isinstance(expr, tuple) and expr and expr[0] == "sub" and node_str_fn(expr[1]) > node_str_fn(expr[2]):
            expr = ("sub", expr[2], expr[1])
        child_size = float(node_size_fn(expr))
        child_depth = float(node_depth_fn(expr))
        child_root = str(expr[0]) if isinstance(expr, tuple) and expr else ("const" if isinstance(expr, (int, float)) else "")
        row_out.update({
            "candidate_child_size": float(child_size),
            "candidate_child_depth": float(child_depth),
            "candidate_child_size_delta": float(child_size - parent_size),
            "candidate_child_depth_delta": float(child_depth - parent_depth),
            "candidate_root_op": str(child_root),
            "path_length": int(len(action_path or ())),
        })
        if dm:
            expr_dim = node_dims_fn(expr, var_dims)
            if expr_dim is None:
                row_out["status"] = "dim_invalid"
                _append_opportunity_row(candidate_observed=False)
                rows.append(shared_candidate_row_dict_fn({**row_out, "route_source": "build"}, route_source="build"))
                continue
            if y_dims is not None and not dims_eq_fn(expr_dim, y_dims):
                row_out["status"] = "dim_mismatch"
                _append_opportunity_row(candidate_observed=False)
                rows.append(shared_candidate_row_dict_fn({**row_out, "route_source": "build"}, route_source="build"))
                continue
        if bool(preview_only):
            row_out["status"] = "preview_only"
            _append_opportunity_row(candidate_observed=False)
            rows.append(shared_candidate_row_dict_fn({**row_out, "route_source": "build"}, route_source="build"))
            continue
        sc = score_expr_fn(
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
            refine_enable=refine_enable,
            refine_cfg=refine_cfg,
            refine_best_mse=(
                float(best_raw_mse_struct)
                if math.isfinite(best_raw_mse_struct)
                else float(max(best_raw_mse, float(early_stop_mse)))
            ),
            refine_state=(dict(refine_state) if isinstance(refine_state, dict) else None),
            return_expr=True,
        )
        if sc is None:
            row_out["status"] = "score_none"
            _append_opportunity_row(candidate_observed=False)
            rows.append(shared_candidate_row_dict_fn({**row_out, "route_source": "build"}, route_source="build"))
            continue
        mse_raw, child_key, _z, _mapping, scored_expr = sc
        mse_eff = float(mse_raw) + float(complexity_penalty) * (
            float(node_size_fn(expr)) + float(mapping_cost_fn(_mapping))
        )
        row_out.update({
            "status": "scored",
            "exact_child_score_observed": True,
            "child_expr": str(node_str_fn(scored_expr)),
            "child_key": child_key,
            "child_raw_mse": float(mse_raw),
            "child_eff_mse": float(mse_eff),
            "accepted": bool(math.isfinite(parent_eff) and mse_eff < parent_eff),
        })
        exact_observed += 1
        _append_opportunity_row(candidate_observed=True)
        rows.append(shared_candidate_row_dict_fn({**row_out, "route_source": "build"}, route_source="build"))

    return {
        "controller_build_slate_id": str(slate_id),
        "controller_build_slate_count": int(len(rows)),
        "controller_build_slate_exact_observed_count": int(exact_observed),
        "controller_build_slate_preview_only": bool(preview_only),
        "controller_build_slate": rows,
        "build_opportunity_slate_id": str(opportunity_slate_id),
        "build_opportunity_slate_count": int(len(opportunity_rows)),
        "build_opportunity_slate_preview_only": bool(preview_only),
        "build_opportunity_slate": opportunity_rows,
    }


__all__ = [
    "collect_controller_build_slate",
    "controller_selected_action_path",
    "normalize_controller_build_slate_actions",
]
