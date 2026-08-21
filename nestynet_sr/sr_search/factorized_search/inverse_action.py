# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Inverse steering action orchestration."""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Callable, Mapping, Sequence

import torch

from nestynet_sr.sr_search.model_selection import mapping_cost

from .expr_ast import (
    collect_paths,
    eval_node,
    get_at,
    node_depth,
    node_dims,
    node_size,
    node_str,
    replace_at,
    simplify,
)
from .expr_mapping import eval_mapping, fit_best, mean_squared_error_same_shape
from .inverse_core import (
    _blend_inverse_backprop_target,
    _bool_col,
    _effective_sample_size,
    _ensure_col,
    _estimate_path_transport_scores,
    _inverse_target_mode_rows,
    _linearized_residual_gain,
    _mask_fraction,
    _masked_point_weight,
    _normalize_inverse_local_score_mode,
    _normalize_inverse_target_mode,
    _score_inverse_local_predictions,
    _slice_by_mask,
    _weighted_centered_mse,
    _weighted_inner_cols,
)
from .inverse_search import (
    _deterministic_row_subset,
    _inverse_branch_beam_factor,
    _inverse_collect_local_repair_candidates,
    _inverse_effective_branch_beam_width,
    _inverse_effective_thresholds,
    _inverse_family_gain_scale,
    _inverse_path_cut_factor,
    _inverse_path_profile,
    _inverse_pool_shortlist,
    _inverse_rank_local_repair_candidates,
    _inverse_static_path_score,
)
from .inverse_spec_solver import solve_inverse_spec_preview_rows
from .opportunity_critic import predict_opportunity_slate
from .shared_candidate import shared_candidate_row_dict
from .shared_opportunity import normalize_witness_energy_fields, shared_opportunity_row_dict


ScoreExprFn = Callable[..., Any]


def _preview_witness_fields(row: Mapping[str, Any] | None) -> dict[str, Any]:
    return normalize_witness_energy_fields(row if isinstance(row, Mapping) else {})


def _inverse_action_meta_template() -> dict[str, Any]:
    return {
        "status": "started",
        "selected_path": None,
        "selected_target_mode": None,
        "selected_path_gain": None,
        "selected_rel_gain": None,
        "selected_transport_rel": None,
        "selected_lin_rel": None,
        "selected_branch_factor": None,
        "selected_cut_factor": None,
        "selected_effective_n": None,
        "selected_path_gain_pre_cut": None,
        "local_candidate_count": 0,
        "estimated_child_raw_mse": None,
        "estimated_child_eff_mse": None,
        "estimated_parent_raw_mse": None,
        "estimated_parent_eff_mse": None,
        "estimated_one_hole_rel_improve_raw": None,
        "estimated_one_hole_rel_improve_eff": None,
        "inverse_path_mode_beam_count": 0,
        "inverse_path_mode_beam": [],
        "inverse_exact_score_budget": 0,
        "inverse_exact_support_floor_beams": 0,
        "inverse_exact_support_floor_selected": 0,
        "inverse_exact_global_allocated": 0,
        "inverse_exact_score_observed_count": 0,
        "inverse_repair_slate_id": "",
        "inverse_repair_slate_count": 0,
        "inverse_repair_slate": [],
        "repair_opportunity_slate_id": "",
        "repair_opportunity_slate_count": 0,
        "repair_opportunity_slate": [],
        "repair_opportunity_slate_final_count": 0,
        "repair_opportunity_slate_final": [],
        "inverse_exact_allocator_mode": "legacy",
        "inverse_opportunity_controller_requested": False,
        "inverse_opportunity_controller_used": False,
        "inverse_opportunity_controller_error": "",
        "inverse_exact_budget_trace_count": 0,
        "inverse_exact_budget_trace": [],
        "inverse_tuple_ranker_used": False,
        "inverse_tuple_ranker_best_child_key": "",
        "inverse_tuple_ranker_row_count": 0,
        "inverse_tuple_ranker_child_value_lambda": 0.0,
        "inverse_tuple_ranker_regret_weight": 1.0,
        "inverse_spec_enable": False,
        "inverse_spec_used": False,
        "inverse_spec_family_battery_enable": False,
        "inverse_spec_family_battery_mode": "outer",
        "inverse_spec_recursive_enable": False,
        "inverse_spec_witness_jets_enable": False,
        "inverse_spec_witness_d2_enable": False,
        "inverse_spec_witness_loss_enable": False,
        "inverse_spec_witness_grad_weight": 1.0,
        "inverse_spec_witness_d2_weight": 0.0,
        "inverse_spec_witness_diag_weight": 0.0,
        "inverse_spec_witness_physics_weight": 0.0,
        "inverse_spec_active_var_screen_enable": False,
        "inverse_spec_recursive_used": False,
        "inverse_spec_candidate_count": 0,
        "inverse_spec_beam_count": 0,
        "inverse_spec_solver_meta": [],
        "inverse_spec_enum_tree_count": 0,
        "inverse_spec_enum_depth_reached": 0,
        "inverse_spec_recursive_candidate_count": 0,
        "inverse_spec_recursive_expand_count": 0,
        "inverse_spec_recursive_depth_reached": 0,
        "observed_wall_seconds": None,
        "observed_exact_evals": 0,
        "observed_preview_evals": 0,
        "observed_micro_tokens": 0,
        "observed_widen_tokens": 0,
    }


def _finalize_inverse_action_cost_fields(action_meta: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(action_meta)
    started = out.pop("_started_perf_counter", None)
    wall_seconds = out.get("observed_wall_seconds", None)
    if wall_seconds is None and started is not None:
        try:
            wall_seconds = float(max(0.0, time.perf_counter() - float(started)))
        except Exception:
            wall_seconds = None
    preview_evals = (
        int(out.get("inverse_repair_slate_count", 0) or 0)
        + int(out.get("repair_opportunity_slate_count", 0) or 0)
        + int(out.get("inverse_spec_candidate_count", 0) or 0)
    )
    out["observed_wall_seconds"] = None if wall_seconds is None else float(wall_seconds)
    out["observed_exact_evals"] = int(out.get("observed_exact_evals", out.get("inverse_exact_score_observed_count", 0)) or 0)
    out["observed_preview_evals"] = int(out.get("observed_preview_evals", preview_evals) or 0)
    out["observed_micro_tokens"] = int(out.get("observed_micro_tokens", 0) or 0)
    out["observed_widen_tokens"] = int(out.get("observed_widen_tokens", 0) or 0)
    return out


def _inverse_action_return(
    expr,
    action_meta: Mapping[str, Any],
    *,
    return_meta: bool,
    **updates,
):
    if not return_meta:
        return expr
    out = _finalize_inverse_action_cost_fields(action_meta)
    out.update(updates)
    return expr, out


def _normalize_inverse_action_limits(
    *,
    max_paths: Any,
    topk_terms: Any,
    shortlist_mult: Any,
    local_score_mode: Any,
    transport_min_lin_rel: Any,
    transport_min_effective_n: Any,
) -> dict[str, Any]:
    try:
        max_paths_i = max(1, int(max_paths))
    except Exception:
        max_paths_i = 12
    try:
        topk_terms_i = max(1, int(topk_terms))
    except Exception:
        topk_terms_i = 6
    try:
        shortlist_mult_i = max(1, int(shortlist_mult))
    except Exception:
        shortlist_mult_i = 4
    try:
        min_lin_rel = max(0.0, float(transport_min_lin_rel))
    except Exception:
        min_lin_rel = 0.02
    try:
        min_effective_n = max(0.0, float(transport_min_effective_n))
    except Exception:
        min_effective_n = 8.0
    return {
        "max_paths": int(max_paths_i),
        "topk_terms": int(topk_terms_i),
        "shortlist_mult": int(shortlist_mult_i),
        "local_mode": _normalize_inverse_local_score_mode(local_score_mode, default="affine"),
        "transport_min_lin_rel": float(min_lin_rel),
        "transport_min_effective_n": float(min_effective_n),
    }


def _inverse_action_candidate_paths(
    parent_node,
    candidate_paths: Sequence[Sequence[int]] | None,
) -> list[tuple[int, ...]]:
    raw_paths = [tuple(int(v) for v in p) for p in collect_paths(parent_node) if p]
    if candidate_paths is None:
        return raw_paths
    raw_set = set(raw_paths)
    filtered: list[tuple[int, ...]] = []
    seen_paths: set[tuple[int, ...]] = set()
    for path in candidate_paths:
        try:
            pp = tuple(int(v) for v in path)
        except Exception:
            continue
        if pp in raw_set and pp not in seen_paths:
            filtered.append(pp)
            seen_paths.add(pp)
    return filtered or raw_paths


def _estimate_inverse_action_transport(
    parent_node,
    parent_mapping,
    x_fit,
    y_fit,
    x_probe,
    y_probe,
    raw_paths: Sequence[tuple[int, ...]],
    *,
    safe_eps: float,
) -> dict[str, Any]:
    transport_adj_fit: dict[tuple[int, ...], torch.Tensor] = {}
    transport_residual_fit: torch.Tensor | None = None
    transport_scores_probe: dict[tuple[int, ...], float] = {}
    transport_adj_probe: dict[tuple[int, ...], torch.Tensor] = {}
    transport_residual_probe: torch.Tensor | None = None
    try:
        _transport_scores_fit, transport_adj_fit, _transport_out_fit, transport_residual_fit = _estimate_path_transport_scores(
            parent_node,
            parent_mapping,
            x_fit,
            y_fit,
            raw_paths,
            safe_eps=float(safe_eps),
        )
    except Exception:
        transport_adj_fit = {}
        transport_residual_fit = None
    try:
        transport_scores_probe, transport_adj_probe, _transport_out_probe, transport_residual_probe = _estimate_path_transport_scores(
            parent_node,
            parent_mapping,
            x_probe,
            y_probe,
            raw_paths,
            safe_eps=float(safe_eps),
        )
    except Exception:
        transport_scores_probe = {}
        transport_adj_probe = {}
        transport_residual_probe = None
    if transport_scores_probe:
        tvals = [float(v) for v in transport_scores_probe.values() if math.isfinite(float(v))]
        transport_max = max(tvals) if tvals else 0.0
    else:
        transport_max = 0.0
    transport_den = max(1.0e-18, float(transport_max))
    ranked_paths = []
    for path in raw_paths:
        try:
            sub = get_at(parent_node, path)
        except Exception:
            continue
        static_score, nonadditive = _inverse_static_path_score(parent_node, path, parent_mapping)
        transport_rel = max(0.0, float(transport_scores_probe.get(tuple(path), 0.0)) / transport_den)
        ranked_paths.append(
            (
                float(static_score),
                float(transport_rel),
                int(nonadditive),
                len(path),
                -node_size(sub),
                tuple(path),
            )
        )
    ranked_paths.sort(reverse=True)
    return {
        "ranked_paths": ranked_paths,
        "path_transport_rel": {tuple(row[-1]): float(row[1]) for row in ranked_paths},
        "transport_adj_fit": transport_adj_fit,
        "transport_residual_fit": transport_residual_fit,
        "transport_adj_probe": transport_adj_probe,
        "transport_residual_probe": transport_residual_probe,
    }


def _select_inverse_action_paths(
    ranked_paths: Sequence[tuple[float, float, int, int, int, tuple[int, ...]]],
    rng,
    *,
    max_paths: int,
    candidate_paths_given: bool,
) -> list[tuple[int, ...]]:
    all_paths = [row[-1] for row in ranked_paths]
    if len(all_paths) <= int(max_paths):
        return all_paths
    if candidate_paths_given:
        return all_paths[:max_paths]
    keep_head = max(1, min(len(all_paths), int(max_paths) // 2))
    head = list(all_paths[:keep_head])
    tail = list(all_paths[keep_head:])
    rng.shuffle(tail)
    return head + tail[: max(0, int(max_paths) - len(head))]


def _masked_inverse_action_transport(
    path: Sequence[int],
    *,
    mfit: torch.Tensor,
    mprobe: torch.Tensor,
    transport_ctx: Mapping[str, Any],
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    g_fit_mask = None
    r_fit_mask = None
    g_probe_mask = None
    r_probe_mask = None
    g_fit_full = (transport_ctx.get("transport_adj_fit", {}) or {}).get(tuple(path), None)
    g_probe_full = (transport_ctx.get("transport_adj_probe", {}) or {}).get(tuple(path), None)
    transport_residual_fit = transport_ctx.get("transport_residual_fit", None)
    transport_residual_probe = transport_ctx.get("transport_residual_probe", None)
    if (g_fit_full is not None) and (transport_residual_fit is not None):
        try:
            g_fit_mask = _ensure_col(g_fit_full)[mfit]
            r_fit_mask = _ensure_col(transport_residual_fit)[mfit]
        except Exception:
            g_fit_mask = None
            r_fit_mask = None
    if (g_probe_full is not None) and (transport_residual_probe is not None):
        try:
            g_probe_mask = _ensure_col(g_probe_full)[mprobe]
            r_probe_mask = _ensure_col(transport_residual_probe)[mprobe]
        except Exception:
            g_probe_mask = None
            r_probe_mask = None
    return g_fit_mask, r_fit_mask, g_probe_mask, r_probe_mask


def _blend_inverse_action_targets(
    *,
    tf: torch.Tensor,
    tp: torch.Tensor,
    wf: torch.Tensor,
    wp: torch.Tensor,
    cur_pf: torch.Tensor,
    cur_pp: torch.Tensor,
    profile: Mapping[str, Any],
    mode_name: str,
    conf: float,
    valid_frac: float,
    g_fit_mask: torch.Tensor | None,
    r_fit_mask: torch.Tensor | None,
    g_probe_mask: torch.Tensor | None,
    r_probe_mask: torch.Tensor | None,
    exact_path_eta: float,
    safe_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tf_eff, tp_eff = tf, tp
    wf_eff, wp_eff = wf, wp
    if (g_probe_mask is None) or (r_probe_mask is None):
        return tf_eff, tp_eff, wf_eff, wp_eff
    eta = min(0.95, max(0.15, float(conf)))
    if bool(profile.get("has_ambiguous_inverse", False)):
        eta *= 0.70
    if float(valid_frac) < 0.50:
        eta *= 0.85
    if bool(profile.get("exact_monotone", False)) and mode_name in ("identity", "affine"):
        try:
            eta = max(float(exact_path_eta), eta)
        except Exception:
            eta = max(0.98, eta)
    eta = min(0.999, max(0.10, eta))
    tp_eff, wp_eff = _blend_inverse_backprop_target(
        tp,
        cur_pp,
        r_probe_mask,
        g_probe_mask,
        wp,
        eta=float(eta),
        safe_eps=float(safe_eps),
    )
    if (g_fit_mask is not None) and (r_fit_mask is not None):
        tf_eff, wf_eff = _blend_inverse_backprop_target(
            tf,
            cur_pf,
            r_fit_mask,
            g_fit_mask,
            wf,
            eta=float(eta),
            safe_eps=float(safe_eps),
        )
    return tf_eff, tp_eff, wf_eff, wp_eff


def _best_linearized_path_gain(
    path_rows: Sequence[tuple[float, float, tuple]],
    *,
    xp: torch.Tensor,
    cur_pp: torch.Tensor,
    r_probe_mask: torch.Tensor | None,
    g_probe_mask: torch.Tensor | None,
    wp_eff: torch.Tensor,
    safe_eps: float,
) -> tuple[float, float]:
    if (g_probe_mask is None) or (r_probe_mask is None):
        return 0.0, -float("inf")
    res_energy = _weighted_inner_cols(r_probe_mask, r_probe_mask, wp_eff)
    if res_energy is None or res_energy <= 1.0e-12:
        return 0.0, -float("inf")
    lin_best_gain = -float("inf")
    for cand_row in path_rows[: max(1, min(len(path_rows), 6))]:
        cand_sub = cand_row[2]
        try:
            cand_pp = eval_node(cand_sub, xp)
        except Exception:
            continue
        if not torch.isfinite(cand_pp).all():
            continue
        delta_u = cand_pp - cur_pp
        lg = _linearized_residual_gain(
            r_probe_mask,
            g_probe_mask,
            delta_u,
            w=wp_eff,
            safe_eps=float(safe_eps),
        )
        if math.isfinite(lg) and lg > lin_best_gain:
            lin_best_gain = float(lg)
    if not math.isfinite(lin_best_gain):
        return 0.0, -float("inf")
    lin_rel = max(0.0, float(lin_best_gain) / max(float(res_energy), 1.0e-12))
    return float(lin_rel), float(lin_best_gain)


def _inverse_action_branch_state(
    *,
    parent_node,
    path: Sequence[int],
    sub,
    target_dim,
    inv_fit,
    inv_probe,
    profile: Mapping[str, Any],
    static_score: float,
    transport_rel: float,
    transport_factor: float,
    cut_factor: float,
    mode_name: str,
    mode_factor: float,
    x_fit: torch.Tensor,
    x_probe: torch.Tensor,
    pool_nodes,
    pool_phi_fit: torch.Tensor,
    pool_phi_probe: torch.Tensor,
    pool_dims,
    transport_ctx: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, Any] | None:
    valid_frac = min(_mask_fraction(inv_fit.valid_mask), _mask_fraction(inv_probe.valid_mask))
    conf = min(float(inv_fit.confidence), float(inv_probe.confidence))
    if valid_frac < float(cfg["min_valid_eff"]) or conf < float(cfg["min_conf_eff"]):
        return None

    mfit = _bool_col(inv_fit.valid_mask).squeeze(-1)
    mprobe = _bool_col(inv_probe.valid_mask).squeeze(-1)
    if int(mfit.sum().item()) < 4 or int(mprobe.sum().item()) < 4:
        return None

    xf, tf = _slice_by_mask(x_fit, inv_fit.target, inv_fit.valid_mask)
    xp, tp = _slice_by_mask(x_probe, inv_probe.target, inv_probe.valid_mask)
    if int(xf.shape[0]) < 4 or int(xp.shape[0]) < 4:
        return None
    wf = _masked_point_weight(
        inv_fit.point_weight,
        inv_fit.valid_mask,
        dtype=tf.dtype,
        device=tf.device,
    )
    wp = _masked_point_weight(
        inv_probe.point_weight,
        inv_probe.valid_mask,
        dtype=tp.dtype,
        device=tp.device,
    )

    try:
        cur_pf = eval_node(sub, xf)
        cur_pp = eval_node(sub, xp)
    except Exception:
        return None

    g_fit_mask, r_fit_mask, g_probe_mask, r_probe_mask = _masked_inverse_action_transport(
        path,
        mfit=mfit,
        mprobe=mprobe,
        transport_ctx=transport_ctx,
    )
    tf_eff, tp_eff, wf_eff, wp_eff = _blend_inverse_action_targets(
        tf=tf,
        tp=tp,
        wf=wf,
        wp=wp,
        cur_pf=cur_pf,
        cur_pp=cur_pp,
        profile=profile,
        mode_name=mode_name,
        conf=float(conf),
        valid_frac=float(valid_frac),
        g_fit_mask=g_fit_mask,
        r_fit_mask=r_fit_mask,
        g_probe_mask=g_probe_mask,
        r_probe_mask=r_probe_mask,
        exact_path_eta=float(cfg["exact_path_eta"]),
        safe_eps=float(cfg["safe_eps"]),
    )
    eff_n_fit = _effective_sample_size(wf_eff, tf_eff)
    eff_n_probe = _effective_sample_size(wp_eff, tp_eff)
    eff_n = min(float(eff_n_fit), float(eff_n_probe))
    if eff_n < float(cfg["transport_min_effective_n"]):
        return None

    cur_stats = _score_inverse_local_predictions(
        cur_pf,
        cur_pp,
        tf_eff,
        tp_eff,
        w_fit=wf_eff,
        w_probe=wp_eff,
        poly_degree=cfg["poly_degree"],
        mode=str(cfg["local_mode"]),
    )
    if cur_stats is None:
        cur_probe_mse = _weighted_centered_mse(tp_eff, wp_eff)
    else:
        cur_probe_mse = float(cur_stats[1])

    idxs = _inverse_pool_shortlist(
        pool_phi_fit,
        inv_fit.target,
        inv_fit.valid_mask,
        pool_dims=pool_dims if bool(cfg["dm"]) else None,
        target_dim=target_dim,
        shortlist_k=max(int(cfg["topk_terms"]), int(cfg["topk_terms"]) * int(cfg["shortlist_mult"])),
    )
    if not idxs:
        return None

    path_cands = _inverse_collect_local_repair_candidates(
        parent_node=parent_node,
        path=path,
        sub=sub,
        target_dim=target_dim,
        xf=xf,
        tf=tf_eff,
        xp=xp,
        tp=tp_eff,
        wf=wf_eff,
        wp=wp_eff,
        mfit=mfit,
        mprobe=mprobe,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        pool_phi_fit=pool_phi_fit,
        pool_phi_probe=pool_phi_probe,
        idxs=idxs,
        poly_degree=cfg["poly_degree"],
        local_mode=str(cfg["local_mode"]),
        topk_terms=max(2, int(cfg["topk_terms"])),
        shortlist_mult=max(1, int(cfg["shortlist_mult"])),
        safe_eps=float(cfg["safe_eps"]),
        var_dims=cfg["var_dims"] if bool(cfg["dm"]) else None,
        max_depth=int(cfg["max_depth"]),
        micro_search_enable=bool(cfg["micro_search_enable"]),
        micro_search_max_depth=max(2, min(int(cfg["micro_search_max_depth"]), 3)),
        micro_search_beam_width=max(8, min(int(cfg["micro_search_beam_width"]), 24)),
        micro_search_topk=max(4, min(int(cfg["micro_search_topk"]), 12)),
        micro_search_seed_terms=max(4, min(int(cfg["micro_search_seed_terms"]), 8)),
    )
    if not path_cands:
        return None
    path_rows = _inverse_rank_local_repair_candidates(
        path_cands,
        xf=xf,
        tf=tf_eff,
        xp=xp,
        tp=tp_eff,
        wf=wf_eff,
        wp=wp_eff,
        poly_degree=cfg["poly_degree"],
        local_mode=str(cfg["local_mode"]),
    )
    if not path_rows:
        return None
    best_here = float(path_rows[0][0])

    lin_rel, lin_best_gain = _best_linearized_path_gain(
        path_rows,
        xp=xp,
        cur_pp=cur_pp,
        r_probe_mask=r_probe_mask,
        g_probe_mask=g_probe_mask,
        wp_eff=wp_eff,
        safe_eps=float(cfg["safe_eps"]),
    )
    lin_rel_floor = float(cfg["transport_min_lin_rel"])
    if bool(profile.get("exact_monotone", False)) and mode_name in ("identity", "affine"):
        try:
            lin_rel_floor = min(
                lin_rel_floor,
                max(0.0, float(cfg["exact_transport_min_lin_rel"])),
            )
        except Exception:
            lin_rel_floor = 0.0
    if float(lin_rel) < float(lin_rel_floor):
        return None

    gain = (cur_probe_mse - best_here) * max(0.0, conf) * max(0.0, valid_frac)
    gain *= (1.0 + 0.15 * max(0.0, float(static_score)))
    gain *= float(transport_factor)
    gain *= float(mode_factor)
    if math.isfinite(lin_best_gain) and lin_best_gain < 0.0:
        gain *= 0.70
    gain *= (1.0 + 0.10 * min(2.0, float(lin_rel)))
    return {
        "sub": sub,
        "target_dim": target_dim,
        "xf": xf,
        "tf": tf_eff,
        "xp": xp,
        "tp": tp_eff,
        "wf": wf_eff,
        "wp": wp_eff,
        "cur_pp": cur_pp,
        "mfit": mfit,
        "mprobe": mprobe,
        "pool_idx": list(idxs),
        "gain_raw": float(gain),
        "best_alt_mse": float(best_here),
        "best_alt_probe_mse": float(best_here),
        "valid_frac": float(valid_frac),
        "confidence": float(conf),
        "cur_probe_mse": float(cur_probe_mse),
        "rel_gain": float(max(0.0, cur_probe_mse - best_here) / max(cur_probe_mse, 1.0e-12)),
        "transport_rel": float(transport_rel),
        "transport_factor": float(transport_factor),
        "lin_rel": float(lin_rel),
        "lin_rel_floor": float(lin_rel_floor),
        "lin_gain": float(lin_best_gain) if math.isfinite(lin_best_gain) else 0.0,
        "effective_n": float(eff_n),
        "target_mode": mode_name,
        "target_mode_factor": float(mode_factor),
        "profile_exact_monotone": bool(profile.get("exact_monotone", False)),
        "path_cut_factor": float(cut_factor),
    }


def _inverse_action_branch_state_debug(
    *,
    parent_node,
    path: Sequence[int],
    sub,
    target_dim,
    inv_fit,
    inv_probe,
    profile: Mapping[str, Any],
    static_score: float,
    transport_rel: float,
    transport_factor: float,
    cut_factor: float,
    mode_name: str,
    mode_factor: float,
    x_fit: torch.Tensor,
    x_probe: torch.Tensor,
    pool_nodes,
    pool_phi_fit: torch.Tensor,
    pool_phi_probe: torch.Tensor,
    pool_dims,
    transport_ctx: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    valid_frac = min(_mask_fraction(inv_fit.valid_mask), _mask_fraction(inv_probe.valid_mask))
    conf = min(float(inv_fit.confidence), float(inv_probe.confidence))
    diag: dict[str, Any] = {
        "valid_frac": float(valid_frac),
        "confidence": float(conf),
        "target_mode": str(mode_name),
        "transport_rel": float(transport_rel),
        "cut_factor": float(cut_factor),
    }
    if valid_frac < float(cfg["min_valid_eff"]) or conf < float(cfg["min_conf_eff"]):
        diag["min_valid_eff"] = float(cfg["min_valid_eff"])
        diag["min_conf_eff"] = float(cfg["min_conf_eff"])
        return None, "low_valid_conf", diag

    mfit = _bool_col(inv_fit.valid_mask).squeeze(-1)
    mprobe = _bool_col(inv_probe.valid_mask).squeeze(-1)
    fit_points = int(mfit.sum().item())
    probe_points = int(mprobe.sum().item())
    diag["fit_points"] = int(fit_points)
    diag["probe_points"] = int(probe_points)
    if fit_points < 4 or probe_points < 4:
        return None, "too_few_mask_points", diag

    xf, tf = _slice_by_mask(x_fit, inv_fit.target, inv_fit.valid_mask)
    xp, tp = _slice_by_mask(x_probe, inv_probe.target, inv_probe.valid_mask)
    diag["fit_rows"] = int(xf.shape[0])
    diag["probe_rows"] = int(xp.shape[0])
    if int(xf.shape[0]) < 4 or int(xp.shape[0]) < 4:
        return None, "slice_too_small", diag
    wf = _masked_point_weight(
        inv_fit.point_weight,
        inv_fit.valid_mask,
        dtype=tf.dtype,
        device=tf.device,
    )
    wp = _masked_point_weight(
        inv_probe.point_weight,
        inv_probe.valid_mask,
        dtype=tp.dtype,
        device=tp.device,
    )

    try:
        cur_pf = eval_node(sub, xf)
        cur_pp = eval_node(sub, xp)
    except Exception:
        return None, "sub_eval_failed", diag

    g_fit_mask, r_fit_mask, g_probe_mask, r_probe_mask = _masked_inverse_action_transport(
        path,
        mfit=mfit,
        mprobe=mprobe,
        transport_ctx=transport_ctx,
    )
    tf_eff, tp_eff, wf_eff, wp_eff = _blend_inverse_action_targets(
        tf=tf,
        tp=tp,
        wf=wf,
        wp=wp,
        cur_pf=cur_pf,
        cur_pp=cur_pp,
        profile=profile,
        mode_name=mode_name,
        conf=float(conf),
        valid_frac=float(valid_frac),
        g_fit_mask=g_fit_mask,
        r_fit_mask=r_fit_mask,
        g_probe_mask=g_probe_mask,
        r_probe_mask=r_probe_mask,
        exact_path_eta=float(cfg["exact_path_eta"]),
        safe_eps=float(cfg["safe_eps"]),
    )
    eff_n_fit = _effective_sample_size(wf_eff, tf_eff)
    eff_n_probe = _effective_sample_size(wp_eff, tp_eff)
    eff_n = min(float(eff_n_fit), float(eff_n_probe))
    diag["effective_n"] = float(eff_n)
    diag["transport_min_effective_n"] = float(cfg["transport_min_effective_n"])
    if eff_n < float(cfg["transport_min_effective_n"]):
        return None, "low_effective_n", diag

    cur_stats = _score_inverse_local_predictions(
        cur_pf,
        cur_pp,
        tf_eff,
        tp_eff,
        w_fit=wf_eff,
        w_probe=wp_eff,
        poly_degree=cfg["poly_degree"],
        mode=str(cfg["local_mode"]),
    )
    if cur_stats is None:
        cur_probe_mse = _weighted_centered_mse(tp_eff, wp_eff)
    else:
        cur_probe_mse = float(cur_stats[1])
    diag["cur_probe_mse"] = float(cur_probe_mse)

    idxs = _inverse_pool_shortlist(
        pool_phi_fit,
        inv_fit.target,
        inv_fit.valid_mask,
        pool_dims=pool_dims if bool(cfg["dm"]) else None,
        target_dim=target_dim,
        shortlist_k=max(int(cfg["topk_terms"]), int(cfg["topk_terms"]) * int(cfg["shortlist_mult"])),
    )
    diag["pool_shortlist_count"] = int(len(list(idxs or ())))
    if not idxs:
        return None, "empty_pool_shortlist", diag

    path_cands = _inverse_collect_local_repair_candidates(
        parent_node=parent_node,
        path=path,
        sub=sub,
        target_dim=target_dim,
        xf=xf,
        tf=tf_eff,
        xp=xp,
        tp=tp_eff,
        wf=wf_eff,
        wp=wp_eff,
        mfit=mfit,
        mprobe=mprobe,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        pool_phi_fit=pool_phi_fit,
        pool_phi_probe=pool_phi_probe,
        idxs=idxs,
        poly_degree=cfg["poly_degree"],
        local_mode=str(cfg["local_mode"]),
        topk_terms=max(2, int(cfg["topk_terms"])),
        shortlist_mult=max(1, int(cfg["shortlist_mult"])),
        safe_eps=float(cfg["safe_eps"]),
        var_dims=cfg["var_dims"] if bool(cfg["dm"]) else None,
        max_depth=int(cfg["max_depth"]),
        micro_search_enable=bool(cfg["micro_search_enable"]),
        micro_search_max_depth=max(2, min(int(cfg["micro_search_max_depth"]), 3)),
        micro_search_beam_width=max(8, min(int(cfg["micro_search_beam_width"]), 24)),
        micro_search_topk=max(4, min(int(cfg["micro_search_topk"]), 12)),
        micro_search_seed_terms=max(4, min(int(cfg["micro_search_seed_terms"]), 8)),
    )
    diag["local_candidate_count"] = int(len(list(path_cands or ())))
    if not path_cands:
        return None, "no_local_candidates", diag
    path_rows = _inverse_rank_local_repair_candidates(
        path_cands,
        xf=xf,
        tf=tf_eff,
        xp=xp,
        tp=tp_eff,
        wf=wf_eff,
        wp=wp_eff,
        poly_degree=cfg["poly_degree"],
        local_mode=str(cfg["local_mode"]),
    )
    diag["ranked_candidate_count"] = int(len(list(path_rows or ())))
    if not path_rows:
        return None, "no_ranked_candidates", diag
    best_here = float(path_rows[0][0])
    diag["best_alt_mse"] = float(best_here)

    lin_rel, lin_best_gain = _best_linearized_path_gain(
        path_rows,
        xp=xp,
        cur_pp=cur_pp,
        r_probe_mask=r_probe_mask,
        g_probe_mask=g_probe_mask,
        wp_eff=wp_eff,
        safe_eps=float(cfg["safe_eps"]),
    )
    lin_rel_floor = float(cfg["transport_min_lin_rel"])
    if bool(profile.get("exact_monotone", False)) and mode_name in ("identity", "affine"):
        try:
            lin_rel_floor = min(
                lin_rel_floor,
                max(0.0, float(cfg["exact_transport_min_lin_rel"])),
            )
        except Exception:
            lin_rel_floor = 0.0
    diag["lin_rel"] = float(lin_rel)
    diag["lin_rel_floor"] = float(lin_rel_floor)
    diag["lin_best_gain"] = float(lin_best_gain) if math.isfinite(lin_best_gain) else None
    if float(lin_rel) < float(lin_rel_floor):
        return None, "low_linearized_gain", diag

    gain = (cur_probe_mse - best_here) * max(0.0, conf) * max(0.0, valid_frac)
    gain *= (1.0 + 0.15 * max(0.0, float(static_score)))
    gain *= float(transport_factor)
    gain *= float(mode_factor)
    if math.isfinite(lin_best_gain) and lin_best_gain < 0.0:
        gain *= 0.70
    gain *= (1.0 + 0.10 * min(2.0, float(lin_rel)))
    diag["gain_raw"] = float(gain)
    if (not math.isfinite(gain)) or float(gain) <= 0.0:
        return None, "nonpositive_gain", diag
    return {
        "sub": sub,
        "target_dim": target_dim,
        "xf": xf,
        "tf": tf_eff,
        "xp": xp,
        "tp": tp_eff,
        "wf": wf_eff,
        "wp": wp_eff,
        "cur_pp": cur_pp,
        "mfit": mfit,
        "mprobe": mprobe,
        "pool_idx": list(idxs),
        "gain_raw": float(gain),
        "best_alt_mse": float(best_here),
        "best_alt_probe_mse": float(best_here),
        "valid_frac": float(valid_frac),
        "confidence": float(conf),
        "cur_probe_mse": float(cur_probe_mse),
        "rel_gain": float(max(0.0, cur_probe_mse - best_here) / max(cur_probe_mse, 1.0e-12)),
        "transport_rel": float(transport_rel),
        "transport_factor": float(transport_factor),
        "lin_rel": float(lin_rel),
        "lin_rel_floor": float(lin_rel_floor),
        "lin_gain": float(lin_best_gain) if math.isfinite(lin_best_gain) else 0.0,
        "effective_n": float(eff_n),
        "target_mode": mode_name,
        "target_mode_factor": float(mode_factor),
        "profile_exact_monotone": bool(profile.get("exact_monotone", False)),
        "path_cut_factor": float(cut_factor),
    }, "ok", diag


def _serialize_inverse_action_path_state(state: Mapping[str, Any] | None) -> dict[str, Any]:
    state = state if isinstance(state, Mapping) else {}
    path_like = state.get("path", ())
    try:
        path = [int(v) for v in (path_like or ())]
    except Exception:
        path = []
    target_mode = str(state.get("target_mode", "") or "")
    target_mapping_kind = str(state.get("target_mapping_kind", "") or "")
    valid_frac = float(state.get("valid_frac", 0.0) or 0.0)
    confidence = float(state.get("confidence", 0.0) or 0.0)
    best_alt_probe_mse = float(state.get("best_alt_probe_mse", state.get("best_alt_mse", float("inf"))) or float("inf"))
    weighted_rel_gain_raw = float(
        state.get(
            "weighted_rel_gain_raw",
            state.get("path_gain_raw", state.get("gain_raw", 0.0)),
        ) or 0.0
    )
    weighted_rel_gain_pre_cut = float(state.get("weighted_rel_gain_pre_cut", state.get("path_gain_pre_cut", 0.0)) or 0.0)
    weighted_rel_gain = float(state.get("weighted_rel_gain", state.get("path_gain", 0.0)) or 0.0)
    cut_factor = float(state.get("cut_factor", state.get("path_cut_factor", 1.0)) or 1.0)
    mode_rows_raw = list(state.get("mode_rows", []) or [])
    if not mode_rows_raw and target_mode:
        mode_rows_raw = [{
            "target_mode": target_mode,
            "target_mapping_kind": target_mapping_kind,
            "weighted_rel_gain": weighted_rel_gain,
            "weighted_rel_gain_raw": weighted_rel_gain_raw,
            "rel_gain": float(state.get("rel_gain", 0.0) or 0.0),
            "best_alt_probe_mse": best_alt_probe_mse,
            "cur_probe_mse": float(state.get("cur_probe_mse", float("inf")) or float("inf")),
            "confidence": confidence,
            "valid_frac": valid_frac,
        }]
    return {
        "path": path,
        "branch_id": str(state.get("branch_id", "") or ""),
        "static_score": float(state.get("static_score", 0.0) or 0.0),
        "transport_rel": float(state.get("transport_rel", 0.0) or 0.0),
        "transport_factor": float(state.get("transport_factor", 1.0) or 1.0),
        "nonadditive": int(state.get("nonadditive", 0) or 0),
        "valid_frac": valid_frac,
        "confidence": confidence,
        "cur_probe_mse": float(state.get("cur_probe_mse", float("inf")) or float("inf")),
        "best_alt_probe_mse": best_alt_probe_mse,
        "rel_gain": float(state.get("rel_gain", 0.0) or 0.0),
        "weighted_rel_gain_raw": weighted_rel_gain_raw,
        "target_mode": target_mode,
        "target_mode_factor": float(state.get("target_mode_factor", 1.0) or 1.0),
        "target_mapping_kind": target_mapping_kind,
        "weighted_rel_gain": weighted_rel_gain,
        "branch_factor": float(state.get("branch_factor", 1.0) or 1.0),
        "branch_support": float(state.get("branch_support", 1.0) or 1.0),
        "branch_positive_count": int(state.get("branch_positive_count", 1) or 1),
        "family_scale": float(state.get("family_scale", 1.0) or 1.0),
        "cut_factor": cut_factor,
        "weighted_rel_gain_pre_cut": weighted_rel_gain_pre_cut,
        "mode_rows": mode_rows_raw,
        # Compatibility aliases for existing logs/readers.
        "path_gain": weighted_rel_gain,
        "path_gain_pre_cut": weighted_rel_gain_pre_cut,
        "path_cut_factor": cut_factor,
        "effective_n": float(state.get("effective_n", 0.0) or 0.0),
        "best_alt_mse": float(state.get("best_alt_mse", float("inf")) or float("inf")),
        "min_valid_frac_eff": float(state.get("min_valid_frac_eff", 0.0) or 0.0),
        "min_confidence_eff": float(state.get("min_confidence_eff", 0.0) or 0.0),
        "profile_has_periodic": bool(state.get("profile_has_periodic", False)),
        "profile_has_muldiv": bool(state.get("profile_has_muldiv", False)),
        "profile_has_explogsqrt": bool(state.get("profile_has_explogsqrt", False)),
        "profile_exact_monotone": bool(state.get("profile_exact_monotone", False)),
    }


def _inverse_action_path_mode_beam_states(
    *,
    parent_node,
    parent_mapping,
    x_fit,
    y_fit,
    x_probe,
    y_probe,
    pool_nodes,
    pool_phi_fit,
    pool_phi_probe,
    pool_dims,
    all_paths: Sequence[tuple[int, ...]],
    path_target_modes: Mapping[tuple[int, ...], str] | None,
    transport_ctx: Mapping[str, Any],
    cfg: Mapping[str, Any],
    beam_width: int,
) -> list[dict[str, Any]]:
    out_states: list[dict[str, Any]] = []
    path_transport_rel = transport_ctx.get("path_transport_rel", {}) or {}

    for path in all_paths:
        try:
            sub = get_at(parent_node, path)
        except Exception:
            continue
        profile = _inverse_path_profile(parent_node, path, parent_mapping)
        min_valid_eff, min_conf_eff = _inverse_effective_thresholds(
            float(cfg["min_valid_frac"]),
            float(cfg["min_confidence"]),
            profile=profile,
            periodic_min_valid_scale=float(cfg["periodic_min_valid_scale"]),
            periodic_min_confidence_scale=float(cfg["periodic_min_confidence_scale"]),
        )
        family_scale = _inverse_family_gain_scale(
            profile,
            periodic_path_penalty=float(cfg["periodic_path_penalty"]),
            nonperiodic_muldiv_bonus=float(cfg["nonperiodic_muldiv_bonus"]),
            nonperiodic_explogsqrt_bonus=float(cfg["nonperiodic_explogsqrt_bonus"]),
        )
        transport_rel = float(path_transport_rel.get(tuple(path), 0.0))
        transport_factor = 1.0 + 0.35 * max(0.0, transport_rel)
        path_beam_width = _inverse_effective_branch_beam_width(profile, int(cfg["branch_beam_width"]))
        cut_factor = _inverse_path_cut_factor(
            parent_node,
            path,
            profile,
            additive_descend_penalty=float(cfg["additive_descend_penalty"]),
            nonadditive_leaf_penalty=float(cfg["nonadditive_leaf_penalty"]),
        )
        path_mode = str(cfg["target_mode"])
        if isinstance(path_target_modes, Mapping):
            path_mode = _normalize_inverse_target_mode(
                path_target_modes.get(tuple(path), None),
                default=str(cfg["target_mode"]),
            )
        target_mode_rows = _inverse_target_mode_rows(
            parent_node,
            parent_mapping,
            path,
            x_fit,
            y_fit,
            x_probe,
            y_probe,
            profile=profile,
            safe_eps=float(cfg["safe_eps"]),
            confidence_mode=str(cfg["confidence_mode"]),
            confidence_target_gain=float(cfg["confidence_target_gain"]),
            confidence_floor=float(cfg["confidence_floor"]),
            branch_beam_width=int(path_beam_width),
            target_mode=str(path_mode),
            full_mapping_penalty=float(cfg["full_mapping_penalty"]),
            exact_simple_target_bonus=float(cfg["exact_simple_target_bonus"]),
        )
        if not target_mode_rows:
            continue

        static_score, nonadditive = _inverse_static_path_score(parent_node, path, parent_mapping)
        target_dim = node_dims(sub, cfg["var_dims"]) if bool(cfg["dm"]) else None
        path_cfg = dict(cfg)
        path_cfg["min_valid_eff"] = float(min_valid_eff)
        path_cfg["min_conf_eff"] = float(min_conf_eff)
        mode_best_states = []

        for mode_row in target_mode_rows:
            mode_name = str(mode_row.get("mode", "full"))
            mode_factor = float(mode_row.get("mode_factor", 1.0))
            inv_fit_list = list(mode_row.get("fit_list", []) or [])
            inv_probe_list = list(mode_row.get("probe_list", []) or [])
            probe_by_branch = {str(t.branch_id): t for t in inv_probe_list}
            branch_state_rows = []

            for inv_fit in inv_fit_list:
                inv_probe = probe_by_branch.get(str(inv_fit.branch_id), None)
                if inv_probe is None and inv_probe_list:
                    inv_probe = inv_probe_list[0]
                if inv_probe is None:
                    continue
                state = _inverse_action_branch_state(
                    parent_node=parent_node,
                    path=path,
                    sub=sub,
                    target_dim=target_dim,
                    inv_fit=inv_fit,
                    inv_probe=inv_probe,
                    profile=profile,
                    static_score=float(static_score),
                    transport_rel=float(transport_rel),
                    transport_factor=float(transport_factor),
                    cut_factor=float(cut_factor),
                    mode_name=mode_name,
                    mode_factor=float(mode_factor),
                    x_fit=x_fit,
                    x_probe=x_probe,
                    pool_nodes=pool_nodes,
                    pool_phi_fit=pool_phi_fit,
                    pool_phi_probe=pool_phi_probe,
                    pool_dims=pool_dims,
                    transport_ctx=transport_ctx,
                    cfg=path_cfg,
                )
                if state is not None:
                    state["branch_id"] = str(getattr(inv_fit, "branch_id", "") or "")
                    state["target_mapping_kind"] = str(getattr(inv_fit, "mapping_kind", "") or "")
                    state["static_score"] = float(static_score)
                    state["nonadditive"] = int(nonadditive)
                    branch_state_rows.append(state)

            if not branch_state_rows:
                continue
            branch_state_rows.sort(
                key=lambda row: (
                    float(row.get("gain_raw", -float("inf"))),
                    -float(row.get("best_alt_mse", float("inf"))),
                ),
                reverse=True,
            )
            path_best_state = dict(branch_state_rows[0])
            path_best_gain_raw = float(path_best_state.get("gain_raw", -float("inf")))
            if bool(profile.get("has_ambiguous_inverse", False)):
                branch_rows_for_factor = [
                    {"weighted_rel_gain_raw": float(r.get("gain_raw", 0.0))}
                    for r in branch_state_rows
                ]
                branch_factor, branch_support, branch_positive = _inverse_branch_beam_factor(
                    branch_rows_for_factor,
                    ambiguity_penalty=float(cfg["branch_ambiguity_penalty"]),
                )
            else:
                branch_factor, branch_support, branch_positive = 1.0, 1.0, 1
            path_best_gain_pre_cut = path_best_gain_raw * float(family_scale) * float(branch_factor)
            path_best_gain = path_best_gain_pre_cut * float(cut_factor)
            path_best_state["path_gain"] = float(path_best_gain)
            path_best_state["path_gain_pre_cut"] = float(path_best_gain_pre_cut)
            path_best_state["path_gain_raw"] = float(path_best_gain_raw)
            path_best_state["branch_factor"] = float(branch_factor)
            path_best_state["branch_support"] = float(branch_support)
            path_best_state["branch_positive_count"] = int(branch_positive)
            path_best_state["family_scale"] = float(family_scale)
            path_best_state["min_valid_frac_eff"] = float(min_valid_eff)
            path_best_state["min_confidence_eff"] = float(min_conf_eff)
            path_best_state["profile_has_periodic"] = bool(profile.get("has_periodic", False))
            path_best_state["profile_has_muldiv"] = bool(profile.get("has_muldiv", False))
            path_best_state["profile_has_explogsqrt"] = bool(profile.get("has_explogsqrt", False))
            path_best_state["transport_rel"] = float(transport_rel)
            path_best_state["transport_factor"] = float(transport_factor)
            if math.isfinite(path_best_gain) and path_best_gain > 0.0:
                mode_best_states.append(path_best_state)

        if not mode_best_states:
            continue
        mode_rows = [
            {
                "target_mode": str(mr.get("target_mode", "")),
                "target_mapping_kind": str(mr.get("target_mapping_kind", "")),
                "weighted_rel_gain": float(mr.get("path_gain", mr.get("weighted_rel_gain", 0.0)) or 0.0),
                "weighted_rel_gain_raw": float(mr.get("path_gain_raw", mr.get("weighted_rel_gain_raw", mr.get("gain_raw", 0.0))) or 0.0),
                "rel_gain": float(mr.get("rel_gain", 0.0) or 0.0),
                "best_alt_probe_mse": float(mr.get("best_alt_probe_mse", mr.get("best_alt_mse", float("inf"))) or float("inf")),
                "cur_probe_mse": float(mr.get("cur_probe_mse", float("inf")) or float("inf")),
                "confidence": float(mr.get("confidence", 0.0) or 0.0),
                "valid_frac": float(mr.get("valid_frac", 0.0) or 0.0),
            }
            for mr in sorted(mode_best_states, key=lambda row: str(row.get("target_mode", "")))
        ]
        for row in mode_best_states:
            row_out = dict(row)
            row_out["path"] = tuple(path)
            row_out["mode_rows"] = mode_rows
            out_states.append(row_out)

    out_states.sort(
        key=lambda row: (
            float(row.get("path_gain", -float("inf"))),
            -float(row.get("best_alt_mse", float("inf"))),
            -len(tuple(row.get("path", ()) or ())),
        ),
        reverse=True,
    )
    beam_width_i = max(1, int(beam_width))
    return [dict(row) for row in out_states[:beam_width_i]]


def _update_inverse_action_meta_for_path(
    action_meta: dict[str, Any],
    best_path: Sequence[int],
    best_state: Mapping[str, Any],
) -> None:
    action_meta.update(
        {
            "selected_path": [int(v) for v in best_path],
            "selected_target_mode": str(best_state.get("target_mode", "")),
            "selected_path_gain": float(best_state.get("path_gain", 0.0)),
            "selected_path_gain_pre_cut": float(best_state.get("path_gain_pre_cut", 0.0)),
            "selected_rel_gain": float(best_state.get("rel_gain", 0.0)),
            "selected_transport_rel": float(best_state.get("transport_rel", 0.0)),
            "selected_lin_rel": float(best_state.get("lin_rel", 0.0)),
            "selected_branch_factor": float(best_state.get("branch_factor", 1.0)),
            "selected_cut_factor": float(best_state.get("path_cut_factor", 1.0)),
            "selected_effective_n": float(best_state.get("effective_n", 0.0)),
            "status": "selected_path",
        }
    )


def _mapping_param_count(mapping: Mapping[str, Any] | None) -> int:
    if not isinstance(mapping, Mapping):
        return 0
    count = 0

    def _visit(value: Any) -> None:
        nonlocal count
        if isinstance(value, Mapping):
            for key, inner in value.items():
                if str(key) == "kind":
                    continue
                _visit(inner)
            return
        if isinstance(value, (list, tuple)):
            for inner in value:
                _visit(inner)
            return
        try:
            fv = float(value)
        except Exception:
            return
        if math.isfinite(fv):
            count += 1

    _visit(dict(mapping))
    return int(count)


def _inverse_local_mapping_preview(
    cand_sub,
    *,
    xf: torch.Tensor,
    tf: torch.Tensor,
    xp: torch.Tensor,
    tp: torch.Tensor,
    poly_degree: int,
    local_mode: str,
) -> dict[str, Any]:
    mode_name = _normalize_inverse_local_score_mode(local_mode, default="affine")
    if mode_name in ("strict", "direct"):
        return {
            "local_mapping_kind": "identity",
            "local_mapping_nparams": 0,
        }
    if mode_name in ("affine", "lin", "linear"):
        return {
            "local_mapping_kind": "affine",
            "local_mapping_nparams": 2,
        }
    try:
        pred_fit = eval_node(cand_sub, xf)
        pred_probe = eval_node(cand_sub, xp)
    except Exception:
        return {
            "local_mapping_kind": "",
            "local_mapping_nparams": 0,
        }
    if (not torch.isfinite(pred_fit).all()) or (not torch.isfinite(pred_probe).all()):
        return {
            "local_mapping_kind": "",
            "local_mapping_nparams": 0,
        }
    fb = fit_best(pred_fit, tf, int(poly_degree))
    if fb is None:
        return {
            "local_mapping_kind": "",
            "local_mapping_nparams": 0,
        }
    _, mapping = fb
    try:
        eval_mapping(pred_probe, mapping)
    except Exception:
        pass
    mapping_kind = str((mapping or {}).get("kind", "") or "")
    return {
        "local_mapping_kind": mapping_kind,
        "local_mapping_nparams": int(_mapping_param_count(mapping)),
    }


def _serialize_inverse_action_slate_row(
    scored: Mapping[str, Any] | None,
) -> dict[str, Any]:
    scored = scored if isinstance(scored, Mapping) else {}
    path_like = scored.get("path", ())
    try:
        path = [int(v) for v in (path_like or ())]
    except Exception:
        path = []
    return shared_candidate_row_dict({
        "slate_id": str(scored.get("slate_id", "") or ""),
        "path": path,
        "target_mode": str(scored.get("target_mode", "") or ""),
        "route": str(scored.get("route", "repair") or "repair"),
        "route_source": "repair",
        "action": str(scored.get("action", "inv_steer") or "inv_steer"),
        "tuple_provenance": str(scored.get("tuple_provenance", "beam_local_repair") or "beam_local_repair"),
        "beam_rank": int(scored.get("beam_rank", 0) or 0),
        "local_rank": int(scored.get("local_rank", 0) or 0),
        "pre_dedup_rank": int(scored.get("pre_dedup_rank", 0) or 0),
        "post_dedup_rank": int(scored.get("post_dedup_rank", 0) or 0),
        "dedup_kept": bool(scored.get("dedup_kept", False)),
        "exact_child_score_observed": bool(scored.get("exact_child_score_observed", False)),
        "proposal_family": str(scored.get("proposal_family", "") or ""),
        "generation_source": str(scored.get("generation_source", "") or ""),
        "inverse_spec_generation_kind": str(scored.get("inverse_spec_generation_kind", "") or ""),
        "inverse_spec_recursion_depth": int(scored.get("inverse_spec_recursion_depth", 0) or 0),
        "inverse_spec_solver_confidence": float(scored.get("inverse_spec_solver_confidence", 0.0) or 0.0),
        "inverse_spec_solver_valid_frac": float(scored.get("inverse_spec_solver_valid_frac", 0.0) or 0.0),
        "inverse_spec_trace": [str(v) for v in list(scored.get("inverse_spec_trace", []) or [])],
        "path_gain": float(scored.get("path_gain", 0.0) or 0.0),
        "target_mapping_kind": str(scored.get("target_mapping_kind", "") or ""),
        "local_mapping_kind": str(scored.get("local_mapping_kind", "") or ""),
        "local_mapping_nparams": int(scored.get("local_mapping_nparams", 0) or 0),
        "local_probe_mse": None if scored.get("local_probe_mse", None) is None else float(scored.get("local_probe_mse")),
        "local_fit_mse": None if scored.get("local_fit_mse", None) is None else float(scored.get("local_fit_mse")),
        "local_fit_probe_gap": None if scored.get("local_fit_probe_gap", None) is None else float(scored.get("local_fit_probe_gap")),
        "candidate_subtree_size": int(scored.get("candidate_subtree_size", 0) or 0),
        "candidate_subtree_depth": int(scored.get("candidate_subtree_depth", 0) or 0),
        "candidate_subtree_size_delta": int(scored.get("candidate_subtree_size_delta", 0) or 0),
        "candidate_subtree_depth_delta": int(scored.get("candidate_subtree_depth_delta", 0) or 0),
        "candidate_child_size": int(scored.get("candidate_child_size", 0) or 0),
        "candidate_child_depth": int(scored.get("candidate_child_depth", 0) or 0),
        "candidate_child_size_delta": int(scored.get("candidate_child_size_delta", 0) or 0),
        "candidate_child_depth_delta": int(scored.get("candidate_child_depth_delta", 0) or 0),
        "candidate_root_op": str(scored.get("candidate_root_op", "") or ""),
        "local_candidate_count": int(scored.get("local_candidate_count", 0) or 0),
        "provenance_grouped": bool(scored.get("provenance_grouped", False)),
        "provenance_count": int(scored.get("provenance_count", 1) or 1),
        "distinct_path_count": int(scored.get("distinct_path_count", 1) or 1),
        "distinct_mode_count": int(scored.get("distinct_mode_count", 1) or 1),
        "distinct_local_mapping_count": int(scored.get("distinct_local_mapping_count", 1) or 1),
        "best_local_probe_mse": None if scored.get("best_local_probe_mse", None) is None else float(scored.get("best_local_probe_mse")),
        "mean_local_probe_mse": None if scored.get("mean_local_probe_mse", None) is None else float(scored.get("mean_local_probe_mse")),
        "worst_local_probe_mse": None if scored.get("worst_local_probe_mse", None) is None else float(scored.get("worst_local_probe_mse")),
        "best_local_fit_mse": None if scored.get("best_local_fit_mse", None) is None else float(scored.get("best_local_fit_mse")),
        "mean_local_fit_mse": None if scored.get("mean_local_fit_mse", None) is None else float(scored.get("mean_local_fit_mse")),
        "worst_local_fit_mse": None if scored.get("worst_local_fit_mse", None) is None else float(scored.get("worst_local_fit_mse")),
        "best_second_probe_gap": None if scored.get("best_second_probe_gap", None) is None else float(scored.get("best_second_probe_gap")),
        "mean_fit_probe_gap": None if scored.get("mean_fit_probe_gap", None) is None else float(scored.get("mean_fit_probe_gap")),
        "provenance_rows": list(scored.get("provenance_rows", []) or []),
        "tuple_utility_estimate": None if scored.get("tuple_utility_estimate", None) is None else float(scored.get("tuple_utility_estimate")),
        "tuple_value_estimate": None if scored.get("tuple_value_estimate", None) is None else float(scored.get("tuple_value_estimate")),
        "tuple_regret_estimate": None if scored.get("tuple_regret_estimate", None) is None else float(scored.get("tuple_regret_estimate")),
        "tuple_combined_estimate": None if scored.get("tuple_combined_estimate", None) is None else float(scored.get("tuple_combined_estimate")),
        "tuple_allocation_estimate": None if scored.get("tuple_allocation_estimate", None) is None else float(scored.get("tuple_allocation_estimate")),
        "child_key": str(scored.get("child_key", "") or ""),
        "child_raw_mse": None if scored.get("raw_mse", None) is None else float(scored.get("raw_mse")),
        "child_eff_mse": None if scored.get("eff_mse", None) is None else float(scored.get("eff_mse")),
        "child_expr": str(node_str(scored.get("expr", None))),
    }, route_source="repair")


def _inverse_preview_row_rep_key(
    row: Mapping[str, Any],
) -> tuple[Any, ...]:
    return (
        float(row.get("local_probe_mse", float("inf"))),
        float(row.get("local_fit_mse", float("inf"))),
        int(row.get("candidate_subtree_size", 1 << 30) or (1 << 30)),
        int(row.get("beam_rank", 1 << 30) or (1 << 30)),
        int(row.get("local_rank", 1 << 30) or (1 << 30)),
        str(row.get("tuple_provenance", row.get("proposal_family", "")) or ""),
        str(row.get("generation_source", "") or ""),
    )


def _sort_inverse_action_candidate_rows_by_preview(
    beam_rows: list[dict[str, Any]],
) -> None:
    beam_rows.sort(
        key=lambda row: (
            0 if bool(row.get("dedup_kept", False)) else 1,
            float(row.get("local_probe_mse", float("inf"))),
            float(row.get("local_fit_mse", float("inf"))),
            int(row.get("candidate_subtree_size", 1 << 30) or (1 << 30)),
            int(row.get("beam_rank", 1 << 30) or (1 << 30)),
            int(row.get("local_rank", 1 << 30) or (1 << 30)),
            str(row.get("tuple_provenance", row.get("proposal_family", "")) or ""),
            str(row.get("generation_source", "") or ""),
        )
    )


def _group_inverse_action_preview_rows(
    preview_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for idx, row in enumerate(preview_rows):
        if not isinstance(row, dict):
            continue
        row["dedup_kept"] = False
        row["post_dedup_rank"] = 0
        child_key = str(row.get("child_key", "") or row.get("child_expr", "") or f"__row_{idx}")
        groups.setdefault(child_key, []).append(row)

    unique_rows: list[dict[str, Any]] = []
    duplicate_rows_by_key: dict[str, list[dict[str, Any]]] = {}
    for child_key, group_rows in groups.items():
        duplicate_rows_by_key[str(child_key)] = list(group_rows)
        if not group_rows:
            continue
        group_rows.sort(key=_inverse_preview_row_rep_key)
        rep = group_rows[0]

        def _provenance_row(row: Mapping[str, Any]) -> dict[str, Any]:
            payload = {k: v for k, v in dict(row).items() if k not in {"beam_state", "expr"}}
            payload["child_expr"] = str(payload.get("child_expr", payload.get("child_key", "")) or payload.get("child_key", ""))
            return shared_candidate_row_dict(payload, route_source="repair")

        probe_vals = [
            float(row.get("local_probe_mse"))
            for row in group_rows
            if row.get("local_probe_mse", None) is not None
        ]
        fit_vals = [
            float(row.get("local_fit_mse"))
            for row in group_rows
            if row.get("local_fit_mse", None) is not None
        ]
        gap_vals = [
            max(0.0, float(probe) - float(fit))
            for probe, fit in zip(probe_vals, fit_vals)
        ]
        sorted_probe = sorted(probe_vals)
        best_probe = float(sorted_probe[0]) if sorted_probe else float("inf")
        second_probe = float(sorted_probe[1]) if len(sorted_probe) > 1 else best_probe
        best_fit = float(min(fit_vals)) if fit_vals else float("inf")
        path_keys = {
            tuple(int(v) for v in (row.get("path", []) or ()))
            for row in group_rows
        }
        mode_keys = {
            str(row.get("target_mode", "") or "")
            for row in group_rows
            if str(row.get("target_mode", "") or "")
        }
        local_mapping_keys = {
            str(row.get("local_mapping_kind", "") or "")
            for row in group_rows
            if str(row.get("local_mapping_kind", "") or "")
        }
        rep.update({
            "child_key": str(child_key),
            "provenance_grouped": bool(len(group_rows) > 1),
            "provenance_count": int(len(group_rows)),
            "distinct_path_count": int(len(path_keys) or 1),
            "distinct_mode_count": int(len(mode_keys) or 1),
            "distinct_local_mapping_count": int(len(local_mapping_keys) or 1),
            "best_local_probe_mse": float(best_probe),
            "mean_local_probe_mse": float(sum(probe_vals) / len(probe_vals)) if probe_vals else float("inf"),
            "worst_local_probe_mse": float(max(probe_vals)) if probe_vals else float("inf"),
            "best_local_fit_mse": float(best_fit),
            "mean_local_fit_mse": float(sum(fit_vals) / len(fit_vals)) if fit_vals else float("inf"),
            "worst_local_fit_mse": float(max(fit_vals)) if fit_vals else float("inf"),
            "best_second_probe_gap": float(max(0.0, second_probe - best_probe)) if probe_vals else 0.0,
            "mean_fit_probe_gap": float(sum(gap_vals) / len(gap_vals)) if gap_vals else 0.0,
            "provenance_rows": [_provenance_row(row) for row in group_rows],
        })
        rep["dedup_kept"] = True
        unique_rows.append(rep)

    unique_rows.sort(key=_inverse_preview_row_rep_key)
    for rank, row in enumerate(unique_rows):
        row["post_dedup_rank"] = int(rank)
    return unique_rows, duplicate_rows_by_key


def _serialize_inverse_action_opportunity_row(
    *,
    parent_node,
    decision_id: str,
    beam_rank: int,
    beam_state: Mapping[str, Any] | None,
    beam_rows: Sequence[Mapping[str, Any]],
    budget_remaining: int,
    local_limit: int,
) -> dict[str, Any]:
    beam_state = beam_state if isinstance(beam_state, Mapping) else {}
    dedup_rows = [row for row in beam_rows if isinstance(row, Mapping) and bool(row.get("dedup_kept", False))]
    sort_rows = dedup_rows if dedup_rows else [row for row in beam_rows if isinstance(row, Mapping)]
    sort_rows = sorted(
        sort_rows,
        key=lambda row: (
            float(row.get("tuple_allocation_estimate", row.get("tuple_combined_estimate", row.get("tuple_utility_estimate", float("-inf"))))),
            -float(row.get("local_probe_mse", float("inf"))),
            -float(row.get("local_fit_mse", float("inf"))),
            -int(row.get("local_rank", 0) or 0),
        ),
        reverse=True,
    )
    best_preview = sort_rows[0] if sort_rows else {}
    path_like = beam_state.get("path", ())
    try:
        path = [int(v) for v in (path_like or ())]
    except Exception:
        path = []
    witness_fields = _preview_witness_fields(best_preview)
    return shared_opportunity_row_dict({
        "route_source": "repair",
        "opportunity_type": "repair_beam",
        "decision_id": str(decision_id),
        "beam_id": f"{str(decision_id)}:{int(beam_rank)}",
        "parent_expr": str(node_str(parent_node)),
        "action": "inv_steer",
        "path": path,
        "path_source": "inverse_beam",
        "target_mode": str(beam_state.get("target_mode", "") or ""),
        "budget_exact_spent": 0,
        "budget_remaining": int(max(0, budget_remaining)),
        "budget_widen_spent": 0,
        "budget_micro_spent": 0,
        "current_best_child_expr": str(best_preview.get("child_key", "") or best_preview.get("child_expr", "") or ""),
        "current_best_child_eff_mse": None,
        "candidate_count_observed": int(len(list(beam_rows))),
        "candidate_count_unique": int(len(dedup_rows)),
        "exact_child_observed_count": 0,
        "beam_rank": int(beam_rank),
        "path_gain": float(beam_state.get("path_gain", 0.0) or 0.0),
        "path_gain_pre_cut": float(beam_state.get("path_gain_pre_cut", 0.0) or 0.0),
        "rel_gain": float(beam_state.get("rel_gain", 0.0) or 0.0),
        "transport_rel": float(beam_state.get("transport_rel", 0.0) or 0.0),
        "lin_rel": float(beam_state.get("lin_rel", 0.0) or 0.0),
        "valid_frac": float(beam_state.get("valid_frac", 0.0) or 0.0),
        "confidence": float(beam_state.get("confidence", 0.0) or 0.0),
        "effective_n": float(beam_state.get("effective_n", 0.0) or 0.0),
        "branch_factor": float(beam_state.get("branch_factor", 0.0) or 0.0),
        "cut_factor": float(beam_state.get("path_cut_factor", beam_state.get("cut_factor", 0.0)) or 0.0),
        "branch_support": float(beam_state.get("branch_support", 0.0) or 0.0),
        "family_scale": float(beam_state.get("family_scale", 0.0) or 0.0),
        "target_mapping_kind": str(beam_state.get("target_mapping_kind", "") or ""),
        "local_limit": int(max(0, local_limit)),
        "best_preview_child_expr": str(best_preview.get("child_key", "") or best_preview.get("child_expr", "") or ""),
        "best_preview_probe_mse": (
            None if best_preview.get("local_probe_mse", None) is None else float(best_preview.get("local_probe_mse"))
        ),
        "best_preview_fit_mse": (
            None if best_preview.get("local_fit_mse", None) is None else float(best_preview.get("local_fit_mse"))
        ),
        "best_tuple_utility_estimate": (
            None if best_preview.get("tuple_utility_estimate", None) is None else float(best_preview.get("tuple_utility_estimate"))
        ),
        "best_tuple_allocation_estimate": (
            None if best_preview.get("tuple_allocation_estimate", None) is None else float(best_preview.get("tuple_allocation_estimate"))
        ),
        "mode_rows": list(beam_state.get("mode_rows", []) or []),
        **witness_fields,
    }, route_source="repair")


def _reorder_inverse_action_candidates_with_tuple_critic(
    *,
    repair_tuple_bundle: Mapping[str, Any] | None,
    repair_tuple_controller_row: Any,
    beam_states: Sequence[Mapping[str, Any]],
    preview_rows: Sequence[dict[str, Any]],
    candidate_rows_by_beam: Mapping[int, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    if not isinstance(repair_tuple_bundle, Mapping) or not bool(repair_tuple_bundle.get("repair_tuple_ranker_trained", False)):
        return None
    if not preview_rows:
        return None
    try:
        from .repair_critic import predict_repair_tuple_slate
    except Exception:
        return None
    try:
        tuple_guidance = predict_repair_tuple_slate(
            dict(repair_tuple_bundle),
            repair_tuple_controller_row,
            path_rows=beam_states,
            preview_rows=preview_rows,
            repair_action_names=("inv_steer",),
        )
    except Exception as exc:
        return {"trained": False, "error": str(exc)}
    if not bool((tuple_guidance or {}).get("trained", False)):
        return tuple_guidance if isinstance(tuple_guidance, dict) else None

    score_by_child_key: dict[str, tuple[float, float, float, float, float]] = {}
    for row in list(tuple_guidance.get("rows", []) or []):
        if not isinstance(row, Mapping):
            continue
        child_key = str(row.get("child_key", "") or "")
        if not child_key:
            continue
        score_by_child_key[child_key] = (
            float(row.get("utility_estimate", 0.0) or 0.0),
            float(row.get("value_estimate", 0.0) or 0.0),
            float(row.get("regret_estimate", 0.0) or 0.0),
            float(row.get("combined_estimate", row.get("utility_estimate", 0.0)) or 0.0),
            float(row.get("allocation_estimate", row.get("combined_estimate", row.get("utility_estimate", 0.0))) or 0.0),
        )
    if not score_by_child_key:
        return tuple_guidance

    for beam_rows in candidate_rows_by_beam.values():
        for row in beam_rows:
            utility_estimate, value_estimate, regret_estimate, combined_estimate, allocation_estimate = score_by_child_key.get(
                str(row.get("child_key", "") or ""),
                (float("-inf"), float("-inf"), float("inf"), float("-inf"), float("-inf")),
            )
            if math.isfinite(utility_estimate):
                row["tuple_utility_estimate"] = float(utility_estimate)
            if math.isfinite(value_estimate):
                row["tuple_value_estimate"] = float(value_estimate)
            if math.isfinite(regret_estimate):
                row["tuple_regret_estimate"] = float(regret_estimate)
            if math.isfinite(combined_estimate):
                row["tuple_combined_estimate"] = float(combined_estimate)
            if math.isfinite(allocation_estimate):
                row["tuple_allocation_estimate"] = float(allocation_estimate)
        beam_rows.sort(
            key=lambda row: (
                1 if bool(row.get("dedup_kept", False)) else 0,
                float(row.get("tuple_allocation_estimate", row.get("tuple_combined_estimate", row.get("tuple_utility_estimate", float("-inf"))))),
                float(row.get("tuple_utility_estimate", float("-inf"))),
                -float(row.get("tuple_regret_estimate", float("inf"))),
                -float(row.get("local_probe_mse", float("inf"))),
                -float(row.get("local_fit_mse", float("inf"))),
                -int(row.get("local_rank", 0) or 0),
            ),
            reverse=True,
        )
    return tuple_guidance


def _select_inverse_exact_budget_rows(
    *,
    candidate_rows_by_beam: Mapping[int, list[dict[str, Any]]],
    global_exact_score_budget: int,
    support_floor_beams: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    budget = max(0, int(global_exact_score_budget))
    if budget <= 0:
        return [], {
            "support_floor_beams": 0,
            "support_floor_selected": 0,
            "global_allocated": 0,
        }

    beam_order = [beam_rank for beam_rank, beam_rows in sorted(candidate_rows_by_beam.items()) if beam_rows]
    selected_unique_rows: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    support_floor = max(0, min(int(support_floor_beams), budget, len(beam_order)))

    for beam_rank in beam_order[:support_floor]:
        for row in candidate_rows_by_beam.get(int(beam_rank), []):
            if not bool(row.get("dedup_kept", False)):
                continue
            child_key = str(row.get("child_key", "") or "")
            if child_key in selected_keys:
                continue
            selected_unique_rows.append(row)
            selected_keys.add(child_key)
            break
        if len(selected_unique_rows) >= budget:
            break

    remaining_rows: list[dict[str, Any]] = []
    for beam_rows in candidate_rows_by_beam.values():
        for row in beam_rows:
            if not bool(row.get("dedup_kept", False)):
                continue
            child_key = str(row.get("child_key", "") or "")
            if child_key in selected_keys:
                continue
            remaining_rows.append(row)
    remaining_rows.sort(
        key=lambda row: (
            float(row.get("tuple_allocation_estimate", row.get("tuple_combined_estimate", row.get("tuple_utility_estimate", float("-inf"))))),
            float(row.get("tuple_utility_estimate", float("-inf"))),
            -float(row.get("tuple_regret_estimate", float("inf"))),
            -float(row.get("local_probe_mse", float("inf"))),
            -float(row.get("local_fit_mse", float("inf"))),
            -int(row.get("beam_rank", 0) or 0),
            -int(row.get("local_rank", 0) or 0),
        ),
        reverse=True,
    )
    for row in remaining_rows:
        if len(selected_unique_rows) >= budget:
            break
        child_key = str(row.get("child_key", "") or "")
        if child_key in selected_keys:
            continue
        selected_unique_rows.append(row)
        selected_keys.add(child_key)

    return selected_unique_rows, {
        "support_floor_beams": int(support_floor),
        "support_floor_selected": int(min(support_floor, len(selected_unique_rows))),
        "global_allocated": int(max(0, len(selected_unique_rows) - min(support_floor, len(selected_unique_rows)))),
    }


def _next_inverse_executor_row(
    beam_rows: Sequence[Mapping[str, Any]],
    *,
    selected_keys: set[str],
) -> dict[str, Any] | None:
    for row in beam_rows:
        if not isinstance(row, Mapping):
            continue
        if not bool(row.get("dedup_kept", False)):
            continue
        child_key = str(row.get("child_key", "") or "")
        if not child_key or child_key in selected_keys:
            continue
        return row if isinstance(row, dict) else dict(row)
    return None


def _remaining_inverse_executor_rows(
    beam_rows: Sequence[Mapping[str, Any]],
    *,
    selected_keys: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in beam_rows:
        if not isinstance(row, Mapping):
            continue
        if not bool(row.get("dedup_kept", False)):
            continue
        child_key = str(row.get("child_key", "") or "")
        if not child_key or child_key in selected_keys:
            continue
        out.append(row if isinstance(row, dict) else dict(row))
    return out


def _best_inverse_observed_row_for_beam(
    beam_rows: Sequence[Mapping[str, Any]],
    *,
    observed_by_child_key: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    best_row = None
    best_key = None
    for row in beam_rows:
        if not isinstance(row, Mapping):
            continue
        child_key = str(row.get("child_key", "") or "")
        observed = observed_by_child_key.get(child_key, None)
        if observed is None:
            continue
        eff_mse = observed.get("eff_mse", None)
        raw_mse = observed.get("raw_mse", None)
        if eff_mse is None:
            continue
        key = (
            float(eff_mse),
            float("inf") if raw_mse is None else float(raw_mse),
            int(row.get("local_rank", 0) or 0),
            str(child_key),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_row = {
                "child_key": str(child_key),
                "child_expr": str(row.get("child_expr", "") or child_key),
                "child_eff_mse": float(eff_mse),
                **_preview_witness_fields(row),
            }
    return best_row


def _build_inverse_allocator_opportunity_rows(
    *,
    parent_node,
    decision_id: str,
    beam_states: Sequence[Mapping[str, Any]],
    candidate_rows_by_beam: Mapping[int, Sequence[Mapping[str, Any]]],
    local_limit: int,
    selected_keys: set[str],
    selected_counts_by_beam: Mapping[int, int],
    observed_by_child_key: Mapping[str, Mapping[str, Any]],
    parent_eff_mse: float | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for beam_rank, beam_state in enumerate(beam_states):
        beam_rows = list(candidate_rows_by_beam.get(int(beam_rank), []) or [])
        remaining_rows = _remaining_inverse_executor_rows(beam_rows, selected_keys=selected_keys)
        base_row = _serialize_inverse_action_opportunity_row(
            parent_node=parent_node,
            decision_id=str(decision_id),
            beam_rank=int(beam_rank),
            beam_state=beam_state,
            beam_rows=beam_rows,
            budget_remaining=int(len(remaining_rows)),
            local_limit=int(local_limit),
        )
        observed_count = int(selected_counts_by_beam.get(int(beam_rank), 0) or 0)
        preview_count = int(base_row.get("candidate_count_observed", 0) or 0)
        preview_unique = int(base_row.get("candidate_count_unique", 0) or 0)
        best_observed = _best_inverse_observed_row_for_beam(beam_rows, observed_by_child_key=observed_by_child_key)
        current_best_eff = None if best_observed is None else float(best_observed.get("child_eff_mse", float("inf")))
        current_route_eff = parent_eff_mse
        if current_best_eff is not None and math.isfinite(current_best_eff):
            if current_route_eff is None or not math.isfinite(current_route_eff):
                current_route_eff = float(current_best_eff)
            else:
                current_route_eff = float(min(float(current_route_eff), float(current_best_eff)))
        base_row.update({
            "parent_depth": int(node_depth(parent_node)),
            "parent_eff_mse": None if parent_eff_mse is None or not math.isfinite(parent_eff_mse) else float(parent_eff_mse),
            "budget_exact_spent": int(observed_count),
            "budget_remaining": int(len(remaining_rows)),
            "candidate_count_observed": int(observed_count),
            "candidate_count_unique": int(observed_count),
            "preview_candidate_count_total": int(preview_count),
            "preview_candidate_count_unique_total": int(preview_unique),
            "shadow_total_exact_available": int(preview_unique),
            "shadow_total_preview_available": int(preview_count),
            "shadow_executor_reveals_observed": int(observed_count),
            "current_best_child_expr": (
                str(best_observed.get("child_expr", "") or best_observed.get("child_key", ""))
                if best_observed is not None
                else str(base_row.get("best_preview_child_expr", base_row.get("current_best_child_expr", "")) or "")
            ),
            "current_best_child_eff_mse": None if best_observed is None else float(best_observed.get("child_eff_mse", float("inf"))),
            "current_best_route_eff_mse": (
                None if current_route_eff is None or not math.isfinite(current_route_eff) else float(current_route_eff)
            ),
            "evidence_level": "exact_known" if observed_count > 0 else ("preview_support" if preview_unique > 1 else "preview_only"),
        })
        if best_observed is not None:
            base_row.update(_preview_witness_fields(best_observed))
        out.append(base_row)
    return out


def _allocate_inverse_exact_budget_with_opportunity_controller(
    *,
    opportunity_bundle: Mapping[str, Any],
    parent_node,
    decision_id: str,
    beam_states: Sequence[Mapping[str, Any]],
    candidate_rows_by_beam: Mapping[int, list[dict[str, Any]]],
    global_exact_score_budget: int,
    local_limit: int,
    parent_eff_mse: float | None,
    score_candidate_fn: ScoreExprFn,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    budget = max(0, int(global_exact_score_budget))
    selected_unique_rows: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    selected_counts_by_beam: dict[int, int] = {}
    observed_by_child_key: dict[str, dict[str, Any]] = {}
    trace: list[dict[str, Any]] = []
    if budget <= 0:
        final_rows = _build_inverse_allocator_opportunity_rows(
            parent_node=parent_node,
            decision_id=decision_id,
            beam_states=beam_states,
            candidate_rows_by_beam=candidate_rows_by_beam,
            local_limit=local_limit,
            selected_keys=selected_keys,
            selected_counts_by_beam=selected_counts_by_beam,
            observed_by_child_key=observed_by_child_key,
            parent_eff_mse=parent_eff_mse,
        )
        return [], {
            "allocator_mode": "opportunity_controller",
            "support_floor_beams": 0,
            "support_floor_selected": 0,
            "global_allocated": 0,
            "trace": trace,
        }, observed_by_child_key, final_rows

    for token_index in range(int(budget)):
        live_rows = _build_inverse_allocator_opportunity_rows(
            parent_node=parent_node,
            decision_id=decision_id,
            beam_states=beam_states,
            candidate_rows_by_beam=candidate_rows_by_beam,
            local_limit=local_limit,
            selected_keys=selected_keys,
            selected_counts_by_beam=selected_counts_by_beam,
            observed_by_child_key=observed_by_child_key,
            parent_eff_mse=parent_eff_mse,
        )
        eligible_rows = [row for row in live_rows if int(row.get("budget_remaining", 0) or 0) > 0]
        if not eligible_rows:
            break
        pred = predict_opportunity_slate(dict(opportunity_bundle), eligible_rows)
        if not bool((pred or {}).get("trained", False)):
            raise RuntimeError("opportunity controller prediction unavailable")
        pred_rows = [dict(row) for row in list(pred.get("rows", []) or []) if isinstance(row, Mapping)]
        if not pred_rows:
            raise RuntimeError("opportunity controller returned no opportunity rows")
        chosen_pred_row = pred_rows[0]
        chosen_beam_rank = None
        for beam_rank, candidate_row in enumerate(live_rows):
            if str(candidate_row.get("opportunity_id", "")) == str(chosen_pred_row.get("opportunity_id", "")):
                chosen_beam_rank = int(beam_rank)
                break
        if chosen_beam_rank is None:
            chosen_beam_rank = int(live_rows.index(eligible_rows[0]))
        next_row = _next_inverse_executor_row(
            candidate_rows_by_beam.get(int(chosen_beam_rank), []),
            selected_keys=selected_keys,
        )
        if next_row is None:
            continue
        child_key = str(next_row.get("child_key", "") or "")
        if child_key in selected_keys:
            continue
        selected_keys.add(child_key)
        selected_unique_rows.append(next_row)
        selected_counts_by_beam[int(chosen_beam_rank)] = int(selected_counts_by_beam.get(int(chosen_beam_rank), 0) + 1)
        scored = score_candidate_fn(next_row)
        if scored is not None:
            next_row["raw_mse"] = float(scored["raw_mse"])
            next_row["eff_mse"] = float(scored["eff_mse"])
            next_row["mapping"] = scored.get("mapping", None)
            next_row["exact_child_score_observed"] = True
            observed_by_child_key[str(child_key)] = {
                "raw_mse": float(scored["raw_mse"]),
                "eff_mse": float(scored["eff_mse"]),
                "mapping": scored.get("mapping", None),
            }
        trace.append({
            "allocator_mode": "opportunity_controller",
            "token_index": int(token_index),
            "beam_rank": int(chosen_beam_rank),
            "opportunity_id": str(chosen_pred_row.get("opportunity_id", "")),
            "beam_id": str(chosen_pred_row.get("beam_id", "")),
            "path": list(chosen_pred_row.get("path", []) or []),
            "target_mode": str(chosen_pred_row.get("target_mode", "") or ""),
            "selected_child_key": str(child_key),
            "selected_child_expr": str(next_row.get("child_expr", "") or child_key),
            "selected_child_eff_mse": None if next_row.get("eff_mse", None) is None else float(next_row.get("eff_mse")),
            "expected_gain_next_under_executor": float(chosen_pred_row.get("expected_gain_next_under_executor", 0.0) or 0.0),
            "cost_estimate": float(chosen_pred_row.get("cost_estimate", 0.0) or 0.0),
            "fragility_prob": float(chosen_pred_row.get("fragility_prob", 0.0) or 0.0),
            "route_flip_prob": float(chosen_pred_row.get("route_flip_prob", 0.0) or 0.0),
            "new_residual_basin_prob": float(chosen_pred_row.get("new_residual_basin_prob", 0.0) or 0.0),
            "acquisition_estimate": float(chosen_pred_row.get("acquisition_estimate", 0.0) or 0.0),
            "budget_remaining_before": int(chosen_pred_row.get("budget_remaining", 0) or 0),
            "budget_exact_spent_before": int(chosen_pred_row.get("budget_exact_spent", 0) or 0),
        })

    final_rows = _build_inverse_allocator_opportunity_rows(
        parent_node=parent_node,
        decision_id=decision_id,
        beam_states=beam_states,
        candidate_rows_by_beam=candidate_rows_by_beam,
        local_limit=local_limit,
        selected_keys=selected_keys,
        selected_counts_by_beam=selected_counts_by_beam,
        observed_by_child_key=observed_by_child_key,
        parent_eff_mse=parent_eff_mse,
    )
    return selected_unique_rows, {
        "allocator_mode": "opportunity_controller",
        "support_floor_beams": 0,
        "support_floor_selected": 0,
        "global_allocated": int(len(selected_unique_rows)),
        "trace": trace,
    }, observed_by_child_key, final_rows


def _inverse_action_slate_id(
    parent_node,
    beam_states: Sequence[Mapping[str, Any]],
) -> str:
    parent_expr = str(node_str(parent_node))
    beam_bits: list[str] = []
    for state in beam_states:
        path = tuple(int(v) for v in (state.get("path", ()) or ()))
        mode = str(state.get("target_mode", "") or "")
        beam_bits.append(f"{path}:{mode}")
    payload = f"{parent_expr}|{'|'.join(beam_bits)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _build_inverse_action_child_expr(
    cand_sub,
    *,
    parent_node,
    best_path: Sequence[int],
    max_depth: int,
    var_dims=None,
):
    try:
        cand = simplify(replace_at(parent_node, best_path, cand_sub))
    except Exception:
        return None
    cand = _canonicalize_inverse_action_candidate(cand)
    if node_depth(cand) > int(max_depth):
        return None
    if var_dims is not None:
        try:
            d = node_dims(cand, var_dims)
        except Exception:
            return None
        if d is None:
            return None
    return cand


def _score_inverse_action_expr(
    cand,
    *,
    x_fit,
    y_fit,
    x_probe,
    y_probe,
    poly_degree: int,
    complexity_penalty: float = 0.0,
    proj=None,
    fp_mode: str = "bits",
    q_scale: float = 2.0,
    q_clip: float = 8.0,
    score_expr_cfg: dict[str, Any] | None = None,
    score_expr_fn: ScoreExprFn | None = None,
) -> dict[str, Any] | None:
    mapping = None
    mse = float("inf")
    if proj is not None and score_expr_fn is not None:
        try:
            sc = score_expr_fn(
                cand,
                x_fit,
                y_fit,
                x_probe,
                y_probe,
                proj,
                fp_mode,
                q_scale,
                q_clip,
                poly_degree,
                refine_enable=False,
                refine_cfg=score_expr_cfg if isinstance(score_expr_cfg, dict) else {},
                return_expr=False,
            )
        except Exception:
            sc = None
        if sc is not None:
            try:
                mse = float(sc[0])
                mapping = sc[3]
            except Exception:
                mapping = None
                mse = float("inf")
    if mapping is None:
        try:
            pf = eval_node(cand, x_fit)
            pp = eval_node(cand, x_probe)
        except Exception:
            return None
        if (not torch.isfinite(pf).all()) or (not torch.isfinite(pp).all()):
            return None
        fb = fit_best(pf, y_fit, poly_degree)
        if fb is None:
            return None
        _, mapping = fb
        try:
            yh = eval_mapping(pp, mapping)
        except Exception:
            return None
        if not torch.isfinite(yh).all():
            return None
        mse = mean_squared_error_same_shape(y_probe, yh)
    if mapping is None or not math.isfinite(float(mse)):
        return None
    map_cost = float(mapping_cost(mapping))
    mse_eff = float(mse + float(complexity_penalty) * float(node_size(cand) + map_cost))
    return {
        "expr": cand,
        "raw_mse": float(mse),
        "eff_mse": float(mse_eff),
        "mapping": mapping,
    }


def _transport_aligned_local_rows(
    local_rows: Sequence[tuple[float, float, tuple]],
    *,
    best_path: Sequence[int],
    best_state: Mapping[str, Any],
    transport_ctx: Mapping[str, Any],
    safe_eps: float,
    exact_transport_min_lin_rel: float,
) -> list[tuple[float, float, tuple]]:
    g_probe_best = (transport_ctx.get("transport_adj_probe", {}) or {}).get(tuple(best_path), None)
    r_probe_best = transport_ctx.get("transport_residual_probe", None)
    cur_pp_best = best_state.get("cur_pp", None)
    mprobe = best_state.get("mprobe", None)
    xp = best_state.get("xp", None)
    wp = best_state.get("wp", None)
    if (
        g_probe_best is None
        or r_probe_best is None
        or cur_pp_best is None
        or mprobe is None
        or xp is None
        or wp is None
    ):
        return list(local_rows)
    try:
        g_probe_mask = _ensure_col(g_probe_best)[mprobe]
        r_probe_mask = _ensure_col(r_probe_best)[mprobe]
    except Exception:
        return list(local_rows)

    require_positive_lin = True
    if bool(best_state.get("profile_exact_monotone", False)) and str(best_state.get("target_mode", "")) in ("identity", "affine"):
        try:
            require_positive_lin = float(exact_transport_min_lin_rel) > 0.0
        except Exception:
            require_positive_lin = False
    filtered_rows = []
    for row in local_rows:
        cand_sub = row[2]
        try:
            cand_pp = eval_node(cand_sub, xp)
        except Exception:
            continue
        if not torch.isfinite(cand_pp).all():
            continue
        lg = _linearized_residual_gain(
            r_probe_mask,
            g_probe_mask,
            cand_pp - cur_pp_best,
            w=wp,
            safe_eps=float(safe_eps),
        )
        if not math.isfinite(lg):
            continue
        if (not require_positive_lin) or (lg > 0.0):
            filtered_rows.append(row)
    return filtered_rows or list(local_rows)


def _canonicalize_inverse_action_candidate(cand):
    while isinstance(cand, tuple) and cand and cand[0] == "neg":
        cand = cand[1]
    if isinstance(cand, tuple) and cand and cand[0] == "sub" and node_str(cand[1]) > node_str(cand[2]):
        cand = ("sub", cand[2], cand[1])
    return cand


def _score_inverse_action_parent(
    *,
    parent_node,
    x_fit,
    y_fit,
    x_probe,
    y_probe,
    proj,
    fp_mode: str,
    q_scale: float,
    q_clip: float,
    poly_degree: int,
    complexity_penalty: float,
    score_expr_cfg: dict[str, Any] | None,
    score_expr_fn: ScoreExprFn | None,
) -> dict[str, float | None] | None:
    if proj is None or score_expr_fn is None:
        return None
    try:
        sc_parent = score_expr_fn(
            parent_node,
            x_fit,
            y_fit,
            x_probe,
            y_probe,
            proj,
            fp_mode,
            q_scale,
            q_clip,
            poly_degree,
            refine_enable=False,
            refine_cfg=score_expr_cfg if isinstance(score_expr_cfg, dict) else {},
            return_expr=False,
        )
    except Exception:
        sc_parent = None
    if sc_parent is None:
        return None
    try:
        parent_raw = float(sc_parent[0])
        parent_mapping = sc_parent[3]
        parent_eff = float(
            parent_raw
            + float(complexity_penalty)
            * float(node_size(parent_node) + mapping_cost(parent_mapping))
        )
    except Exception:
        parent_raw = None
        parent_eff = None
    return {
        "raw_mse": parent_raw,
        "eff_mse": parent_eff,
    }


@torch.no_grad()
def run_inverse_steering_action(
    parent_node,
    parent_mapping,
    x_fit,
    y_fit,
    x_probe,
    y_probe,
    pool_nodes,
    pool_phi_fit,
    pool_phi_probe,
    pool_dims,
    rng,
    max_depth,
    nvars,
    poly_degree,
    *,
    var_dims=None,
    max_paths=12,
    topk_terms=6,
    shortlist_mult=4,
    min_valid_frac=0.25,
    min_confidence=0.10,
    safe_eps=1.0e-12,
    confidence_mode="conditioning",
    confidence_target_gain=4.0,
    confidence_floor=0.05,
    branch_beam_width=1,
    micro_search_enable=False,
    micro_search_max_depth=3,
    micro_search_beam_width=24,
    micro_search_topk=16,
    micro_search_seed_terms=8,
    local_score_mode="affine",
    inverse_spec_enable: bool = False,
    inverse_spec_enum_max_depth: int = 4,
    inverse_spec_enum_max_trees: int = 5000,
    inverse_spec_preview_topk: int = 16,
    inverse_spec_local_score_mode: str = "affine",
    inverse_spec_include_legacy_seed: bool = True,
    inverse_spec_complexity_penalty: float = 0.0,
    inverse_spec_family_battery_enable: bool = False,
    inverse_spec_family_battery_mode: str = "outer",
    inverse_spec_recursive_enable: bool = True,
    inverse_spec_recursive_max_depth: int = 2,
    inverse_spec_recursive_trigger_rel_mse: float = 0.25,
    inverse_spec_recursive_seed_cap: int = 6,
    inverse_spec_recursive_branch_topk: int = 4,
    inverse_spec_recursive_child_topk: int = 2,
    inverse_spec_witness_jets_enable: bool = False,
    inverse_spec_witness_d2_enable: bool = False,
    inverse_spec_witness_max_rows: int = 64,
    inverse_spec_witness_loss_enable: bool = False,
    inverse_spec_witness_grad_weight: float = 1.0,
    inverse_spec_witness_d2_weight: float = 0.0,
    inverse_spec_witness_diag_weight: float = 0.0,
    inverse_spec_witness_physics_weight: float = 0.0,
    inverse_spec_active_var_screen_enable: bool = False,
    inverse_spec_active_var_grad_tol: float = 1.0e-3,
    inverse_spec_active_var_max_count: int = 4,
    inverse_spec_max_subtree_depth: int | None = None,
    inverse_spec_fit_cap: int = 96,
    inverse_spec_probe_cap: int = 192,
    inverse_spec_exact_budget: int = 4,
    target_mode="robust",
    full_mapping_penalty=0.75,
    exact_simple_target_bonus=0.10,
    additive_descend_penalty=0.15,
    nonadditive_leaf_penalty=0.20,
    exact_path_eta=0.98,
    exact_transport_min_lin_rel=0.0,
    periodic_min_valid_scale=1.25,
    periodic_min_confidence_scale=1.35,
    periodic_path_penalty=0.65,
    nonperiodic_muldiv_bonus=0.10,
    nonperiodic_explogsqrt_bonus=0.05,
    branch_ambiguity_penalty=0.50,
    transport_min_lin_rel=0.02,
    transport_min_effective_n=8.0,
    complexity_penalty=0.0,
    candidate_paths=None,
    path_target_modes=None,
    proj=None,
    fp_mode="bits",
    q_scale=2.0,
    q_clip=8.0,
    score_expr_cfg=None,
    return_meta=False,
    score_expr_fn: ScoreExprFn | None = None,
    repair_tuple_bundle: Mapping[str, Any] | None = None,
    repair_tuple_controller_row: Any = None,
    repair_opportunity_controller_enable: bool = False,
    repair_opportunity_bundle: Mapping[str, Any] | None = None,
    inverse_spec_regime_metadata: Mapping[str, Any] | None = None,
):
    """Run the inverse-steering proposal action and optionally return diagnostics."""
    dm = var_dims is not None

    # Proposal actions only need a ranking signal, not exact scoring.
    # When inverse-spec is enabled, use larger sample caps because periodic
    # signals and latent constants need more points for reliable fits.
    if bool(inverse_spec_enable):
        fit_cap = min(int(x_fit.shape[0]), max(32, int(inverse_spec_fit_cap)))
        probe_cap = min(int(x_probe.shape[0]), max(64, int(inverse_spec_probe_cap)))
    else:
        fit_cap = min(int(x_fit.shape[0]), 32)
        probe_cap = min(int(x_probe.shape[0]), 64)
    x_fit, y_fit, pool_phi_fit = _deterministic_row_subset(fit_cap, x_fit, y_fit, pool_phi_fit)
    x_probe, y_probe, pool_phi_probe = _deterministic_row_subset(probe_cap, x_probe, y_probe, pool_phi_probe)

    action_meta = _inverse_action_meta_template()
    action_meta["_started_perf_counter"] = float(time.perf_counter())
    action_meta["inverse_opportunity_controller_requested"] = bool(repair_opportunity_controller_enable)
    action_meta["inverse_spec_enable"] = bool(inverse_spec_enable)
    action_meta["inverse_spec_family_battery_enable"] = bool(inverse_spec_family_battery_enable)
    action_meta["inverse_spec_family_battery_mode"] = str(inverse_spec_family_battery_mode or "outer")
    action_meta["inverse_spec_recursive_enable"] = bool(inverse_spec_recursive_enable)
    action_meta["inverse_spec_witness_jets_enable"] = bool(inverse_spec_witness_jets_enable)
    action_meta["inverse_spec_witness_d2_enable"] = bool(inverse_spec_witness_d2_enable)
    action_meta["inverse_spec_witness_loss_enable"] = bool(inverse_spec_witness_loss_enable)
    action_meta["inverse_spec_witness_grad_weight"] = float(inverse_spec_witness_grad_weight)
    action_meta["inverse_spec_witness_d2_weight"] = float(inverse_spec_witness_d2_weight)
    action_meta["inverse_spec_witness_diag_weight"] = float(inverse_spec_witness_diag_weight)
    action_meta["inverse_spec_witness_physics_weight"] = float(inverse_spec_witness_physics_weight)
    action_meta["inverse_spec_active_var_screen_enable"] = bool(inverse_spec_active_var_screen_enable)
    raw_paths = _inverse_action_candidate_paths(parent_node, candidate_paths)
    if not raw_paths:
        return _inverse_action_return(None, action_meta, return_meta=return_meta, status="no_paths")

    limits = _normalize_inverse_action_limits(
        max_paths=max_paths,
        topk_terms=topk_terms,
        shortlist_mult=shortlist_mult,
        local_score_mode=local_score_mode,
        transport_min_lin_rel=transport_min_lin_rel,
        transport_min_effective_n=transport_min_effective_n,
    )
    transport_ctx = _estimate_inverse_action_transport(
        parent_node,
        parent_mapping,
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        raw_paths,
        safe_eps=float(safe_eps),
    )
    ranked_paths = list(transport_ctx.get("ranked_paths", []) or [])
    if not ranked_paths:
        return _inverse_action_return(None, action_meta, return_meta=return_meta, status="no_ranked_paths")

    all_paths = _select_inverse_action_paths(
        ranked_paths,
        rng,
        max_paths=int(limits["max_paths"]),
        candidate_paths_given=candidate_paths is not None,
    )
    cfg = {
        "max_paths": int(limits["max_paths"]),
        "dm": bool(dm),
        "var_dims": var_dims,
        "max_depth": int(max_depth),
        "poly_degree": int(poly_degree),
        "topk_terms": int(limits["topk_terms"]),
        "shortlist_mult": int(limits["shortlist_mult"]),
        "local_mode": str(limits["local_mode"]),
        "min_valid_frac": float(min_valid_frac),
        "min_confidence": float(min_confidence),
        "safe_eps": float(safe_eps),
        "confidence_mode": str(confidence_mode),
        "confidence_target_gain": float(confidence_target_gain),
        "confidence_floor": float(confidence_floor),
        "branch_beam_width": int(branch_beam_width),
        "micro_search_enable": bool(micro_search_enable),
        "micro_search_max_depth": int(micro_search_max_depth),
        "micro_search_beam_width": int(micro_search_beam_width),
        "micro_search_topk": int(micro_search_topk),
        "micro_search_seed_terms": int(micro_search_seed_terms),
        "target_mode": str(target_mode),
        "full_mapping_penalty": float(full_mapping_penalty),
        "exact_simple_target_bonus": float(exact_simple_target_bonus),
        "additive_descend_penalty": float(additive_descend_penalty),
        "nonadditive_leaf_penalty": float(nonadditive_leaf_penalty),
        "exact_path_eta": float(exact_path_eta),
        "exact_transport_min_lin_rel": float(exact_transport_min_lin_rel),
        "periodic_min_valid_scale": float(periodic_min_valid_scale),
        "periodic_min_confidence_scale": float(periodic_min_confidence_scale),
        "periodic_path_penalty": float(periodic_path_penalty),
        "nonperiodic_muldiv_bonus": float(nonperiodic_muldiv_bonus),
        "nonperiodic_explogsqrt_bonus": float(nonperiodic_explogsqrt_bonus),
        "branch_ambiguity_penalty": float(branch_ambiguity_penalty),
        "transport_min_lin_rel": float(limits["transport_min_lin_rel"]),
        "transport_min_effective_n": float(limits["transport_min_effective_n"]),
    }
    beam_width = max(1, min(int(cfg["max_paths"]), 4))
    beam_states = _inverse_action_path_mode_beam_states(
        parent_node=parent_node,
        parent_mapping=parent_mapping,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        pool_nodes=pool_nodes,
        pool_phi_fit=pool_phi_fit,
        pool_phi_probe=pool_phi_probe,
        pool_dims=pool_dims,
        all_paths=all_paths,
        path_target_modes=path_target_modes if isinstance(path_target_modes, Mapping) else None,
        transport_ctx=transport_ctx,
        cfg=cfg,
        beam_width=int(beam_width),
    )
    if not beam_states:
        return _inverse_action_return(None, action_meta, return_meta=return_meta, status="no_best_path")
    action_meta["inverse_path_mode_beam_count"] = int(len(beam_states))
    action_meta["inverse_path_mode_beam"] = [
        _serialize_inverse_action_path_state(state)
        for state in beam_states
    ]
    slate_id = _inverse_action_slate_id(parent_node, beam_states)
    action_meta["inverse_repair_slate_id"] = str(slate_id)
    action_meta["repair_opportunity_slate_id"] = f"repairopp_{str(slate_id)}"

    best_result = None
    selected_state = None
    selected_local_count = 0
    scored_slate: list[dict[str, Any]] = []
    any_repair_candidates = False
    any_ranked_repairs = False
    local_limit = max(2, min(4, int(limits["topk_terms"])))
    spec_exact_slots = max(0, int(inverse_spec_exact_budget)) if bool(inverse_spec_enable) else 0
    global_exact_score_budget = int(local_limit) + int(spec_exact_slots)
    action_meta["inverse_exact_score_budget"] = int(global_exact_score_budget)
    action_meta["inverse_spec_exact_budget"] = int(spec_exact_slots)
    support_floor_beams = max(1, min(len(beam_states), (int(global_exact_score_budget) + 1) // 2)) if int(global_exact_score_budget) > 0 else 0
    action_meta["inverse_exact_support_floor_beams"] = int(support_floor_beams)
    candidate_rows_by_beam: dict[int, list[dict[str, Any]]] = {}
    all_candidate_rows: list[dict[str, Any]] = []
    inverse_spec_solver_meta_rows: list[dict[str, Any]] = []
    inverse_spec_candidate_count = 0
    inverse_spec_beam_count = 0
    inverse_spec_enum_tree_count = 0
    inverse_spec_enum_depth_reached = 0
    inverse_spec_used = False
    inverse_spec_recursive_used = False
    inverse_spec_recursive_candidate_count = 0
    inverse_spec_recursive_expand_count = 0
    inverse_spec_recursive_depth_reached = 0
    inverse_spec_local_mode = _normalize_inverse_local_score_mode(inverse_spec_local_score_mode, default="affine")
    for beam_rank, beam_state in enumerate(beam_states):
        beam_path = tuple(int(v) for v in (beam_state.get("path", ()) or ()))
        candidate_rows_by_beam.setdefault(int(beam_rank), [])
        cand_subtrees = _inverse_collect_local_repair_candidates(
            parent_node=parent_node,
            path=beam_path,
            sub=beam_state["sub"],
            target_dim=beam_state["target_dim"],
            xf=beam_state["xf"],
            tf=beam_state["tf"],
            xp=beam_state["xp"],
            tp=beam_state["tp"],
            wf=beam_state["wf"],
            wp=beam_state["wp"],
            mfit=beam_state["mfit"],
            mprobe=beam_state["mprobe"],
            pool_nodes=pool_nodes,
            pool_dims=pool_dims,
            pool_phi_fit=pool_phi_fit,
            pool_phi_probe=pool_phi_probe,
            idxs=beam_state["pool_idx"],
            poly_degree=int(poly_degree),
            local_mode=str(limits["local_mode"]),
            topk_terms=max(2, int(limits["topk_terms"])),
            shortlist_mult=max(1, int(limits["shortlist_mult"])),
            safe_eps=float(safe_eps),
            var_dims=var_dims if dm else None,
            max_depth=int(max_depth),
            micro_search_enable=bool(micro_search_enable),
            micro_search_max_depth=int(micro_search_max_depth),
            micro_search_beam_width=int(micro_search_beam_width),
            micro_search_topk=int(micro_search_topk),
            micro_search_seed_terms=int(micro_search_seed_terms),
        )
        legacy_seed_nodes = list(cand_subtrees or [])
        if cand_subtrees:
            any_repair_candidates = True
        local_rows = []
        if cand_subtrees:
            local_rows = _inverse_rank_local_repair_candidates(
                cand_subtrees,
                xf=beam_state["xf"],
                tf=beam_state["tf"],
                xp=beam_state["xp"],
                tp=beam_state["tp"],
                wf=beam_state["wf"],
                wp=beam_state["wp"],
                poly_degree=int(poly_degree),
                local_mode=str(limits["local_mode"]),
            )
            if local_rows:
                any_ranked_repairs = True
                local_rows.sort(key=lambda row: (row[0], row[1], node_size(row[2]), node_str(row[2])))
                local_rows = local_rows[: int(local_limit)]
                local_rows = _transport_aligned_local_rows(
                    local_rows,
                    best_path=beam_path,
                    best_state=beam_state,
                    transport_ctx=transport_ctx,
                    safe_eps=float(safe_eps),
                    exact_transport_min_lin_rel=float(exact_transport_min_lin_rel),
                )
        parent_sub = beam_state.get("sub", None)
        parent_sub_size = int(node_size(parent_sub)) if parent_sub is not None else 0
        parent_sub_depth = int(node_depth(parent_sub)) if parent_sub is not None else 0
        parent_size = int(node_size(parent_node))
        parent_depth = int(node_depth(parent_node))
        for local_rank, (local_probe_mse, local_fit_mse, cand_sub) in enumerate(local_rows):
            child_expr = _build_inverse_action_child_expr(
                cand_sub,
                parent_node=parent_node,
                best_path=beam_path,
                max_depth=int(max_depth),
                var_dims=var_dims if dm else None,
            )
            if child_expr is None:
                continue
            child_key = str(node_str(child_expr))
            cand_root_op = ""
            if isinstance(cand_sub, tuple) and cand_sub:
                cand_root_op = str(cand_sub[0])
            elif isinstance(cand_sub, str):
                cand_root_op = "var"
            else:
                cand_root_op = "const"
            cand_sub_size = int(node_size(cand_sub))
            cand_sub_depth = int(node_depth(cand_sub))
            child_size = int(node_size(child_expr))
            child_depth = int(node_depth(child_expr))
            local_mapping_preview = _inverse_local_mapping_preview(
                cand_sub,
                xf=beam_state["xf"],
                tf=beam_state["tf"],
                xp=beam_state["xp"],
                tp=beam_state["tp"],
                poly_degree=int(poly_degree),
                local_mode=str(limits["local_mode"]),
            )
            row = {
                "slate_id": str(slate_id),
                "expr": child_expr,
                "child_key": child_key,
                "path": beam_path,
                "target_mode": str(beam_state.get("target_mode", "") or ""),
                "target_mapping_kind": str(beam_state.get("target_mapping_kind", "") or ""),
                "beam_rank": int(beam_rank),
                "local_rank": int(local_rank),
                "path_gain": float(beam_state.get("path_gain", 0.0) or 0.0),
                "route": "repair",
                "action": "inv_steer",
                "tuple_provenance": "beam_local_repair",
                "beam_state": beam_state,
                "local_candidate_count": int(len(local_rows)),
                "local_probe_mse": float(local_probe_mse),
                "local_fit_mse": float(local_fit_mse),
                "local_fit_probe_gap": float(max(0.0, float(local_probe_mse) - float(local_fit_mse))),
                "local_mapping_kind": str(local_mapping_preview.get("local_mapping_kind", "") or ""),
                "local_mapping_nparams": int(local_mapping_preview.get("local_mapping_nparams", 0) or 0),
                "candidate_subtree_size": int(cand_sub_size),
                "candidate_subtree_depth": int(cand_sub_depth),
                "candidate_subtree_size_delta": int(cand_sub_size - parent_sub_size),
                "candidate_subtree_depth_delta": int(cand_sub_depth - parent_sub_depth),
                "candidate_child_size": int(child_size),
                "candidate_child_depth": int(child_depth),
                "candidate_child_size_delta": int(child_size - parent_size),
                "candidate_child_depth_delta": int(child_depth - parent_depth),
                "candidate_root_op": str(cand_root_op),
                "exact_child_score_observed": False,
                "dedup_kept": False,
                "pre_dedup_rank": 0,
                "post_dedup_rank": 0,
                "raw_mse": None,
                "eff_mse": None,
            }
            candidate_rows_by_beam[int(beam_rank)].append(row)
            all_candidate_rows.append(row)

        if bool(inverse_spec_enable):
            spec_result = solve_inverse_spec_preview_rows(
                parent_node=parent_node,
                beam_state=beam_state,
                regime_metadata=inverse_spec_regime_metadata,
                beam_rank=int(beam_rank),
                slate_id=str(slate_id),
                max_depth=int(max_depth),
                nvars=int(nvars),
                poly_degree=int(poly_degree),
                var_dims=var_dims if dm else None,
                pool_nodes=pool_nodes,
                pool_dims=pool_dims,
                include_legacy_seed_nodes=(legacy_seed_nodes if bool(inverse_spec_include_legacy_seed) else None),
                local_score_mode=str(inverse_spec_local_mode),
                enum_max_depth=int(inverse_spec_enum_max_depth),
                enum_max_trees=int(inverse_spec_enum_max_trees),
                preview_topk=int(inverse_spec_preview_topk),
                complexity_penalty=float(inverse_spec_complexity_penalty),
                family_battery_enable=bool(inverse_spec_family_battery_enable),
                family_battery_mode=str(inverse_spec_family_battery_mode or "outer"),
                recursive_enable=bool(inverse_spec_recursive_enable),
                recursive_max_depth=int(inverse_spec_recursive_max_depth),
                recursive_trigger_rel_mse=float(inverse_spec_recursive_trigger_rel_mse),
                recursive_seed_cap=int(inverse_spec_recursive_seed_cap),
                recursive_branch_topk=int(inverse_spec_recursive_branch_topk),
                recursive_child_topk=int(inverse_spec_recursive_child_topk),
                witness_jets_enable=bool(inverse_spec_witness_jets_enable),
                witness_d2_enable=bool(inverse_spec_witness_d2_enable),
                witness_max_rows=int(max(4, int(inverse_spec_witness_max_rows))),
                witness_loss_enable=bool(inverse_spec_witness_loss_enable),
                witness_grad_weight=float(inverse_spec_witness_grad_weight),
                witness_d2_weight=float(inverse_spec_witness_d2_weight),
                witness_diag_weight=float(inverse_spec_witness_diag_weight),
                witness_physics_weight=float(inverse_spec_witness_physics_weight),
                active_var_screen_enable=bool(inverse_spec_active_var_screen_enable),
                active_var_grad_tol=float(inverse_spec_active_var_grad_tol),
                active_var_max_count=max(1, int(inverse_spec_active_var_max_count)),
                max_subtree_depth=(int(inverse_spec_max_subtree_depth) if inverse_spec_max_subtree_depth is not None else None),
                safe_eps=float(safe_eps),
                confidence_mode=str(confidence_mode),
                confidence_target_gain=float(confidence_target_gain),
                confidence_floor=float(confidence_floor),
                branch_beam_width=int(branch_beam_width),
            )
            spec_rows = [row for row in list(spec_result.get("rows", []) or []) if isinstance(row, dict)]
            solver_meta = dict(spec_result.get("solver_meta", {}) or {})
            solver_meta["beam_rank"] = int(beam_rank)
            inverse_spec_solver_meta_rows.append(solver_meta)
            inverse_spec_candidate_count += int(len(spec_rows))
            inverse_spec_enum_tree_count += int(solver_meta.get("enum_tree_count", 0) or 0)
            inverse_spec_enum_depth_reached = max(
                int(inverse_spec_enum_depth_reached),
                int(solver_meta.get("enum_depth_reached", 0) or 0),
            )
            inverse_spec_recursive_used = bool(inverse_spec_recursive_used or solver_meta.get("recursive_used", False))
            inverse_spec_recursive_candidate_count += int(solver_meta.get("recursive_candidate_count", 0) or 0)
            inverse_spec_recursive_expand_count += int(solver_meta.get("recursive_expand_count", 0) or 0)
            inverse_spec_recursive_depth_reached = max(
                int(inverse_spec_recursive_depth_reached),
                int(solver_meta.get("recursive_depth_reached", 0) or 0),
            )
            if spec_rows:
                inverse_spec_used = True
                inverse_spec_beam_count += 1
                any_repair_candidates = True
                any_ranked_repairs = True
                candidate_rows_by_beam[int(beam_rank)].extend(spec_rows)
                all_candidate_rows.extend(spec_rows)

    for idx, row in enumerate(all_candidate_rows):
        row["pre_dedup_rank"] = int(idx)

    action_meta["inverse_spec_used"] = bool(inverse_spec_used)
    action_meta["inverse_spec_recursive_used"] = bool(inverse_spec_recursive_used)
    action_meta["inverse_spec_candidate_count"] = int(inverse_spec_candidate_count)
    action_meta["inverse_spec_beam_count"] = int(inverse_spec_beam_count)
    action_meta["inverse_spec_solver_meta"] = [dict(row) for row in inverse_spec_solver_meta_rows]
    action_meta["inverse_spec_enum_tree_count"] = int(inverse_spec_enum_tree_count)
    action_meta["inverse_spec_enum_depth_reached"] = int(inverse_spec_enum_depth_reached)
    action_meta["inverse_spec_recursive_candidate_count"] = int(inverse_spec_recursive_candidate_count)
    action_meta["inverse_spec_recursive_expand_count"] = int(inverse_spec_recursive_expand_count)
    action_meta["inverse_spec_recursive_depth_reached"] = int(inverse_spec_recursive_depth_reached)

    if not all_candidate_rows:
        action_meta["repair_opportunity_slate_count"] = int(len(beam_states))
        action_meta["repair_opportunity_slate"] = [
            _serialize_inverse_action_opportunity_row(
                parent_node=parent_node,
                decision_id=str(action_meta.get("repair_opportunity_slate_id", "")),
                beam_rank=int(beam_rank),
                beam_state=beam_state,
                beam_rows=candidate_rows_by_beam.get(int(beam_rank), []),
                budget_remaining=int(global_exact_score_budget),
                local_limit=int(local_limit),
            )
            for beam_rank, beam_state in enumerate(beam_states)
        ]
        if not any_repair_candidates:
            return _inverse_action_return(None, action_meta, return_meta=return_meta, status="no_repair_candidates")
        if not any_ranked_repairs:
            return _inverse_action_return(None, action_meta, return_meta=return_meta, status="no_ranked_repairs")
        return _inverse_action_return(None, action_meta, return_meta=return_meta, status="no_global_child")

    _unique_rows, duplicate_rows_by_key = _group_inverse_action_preview_rows(all_candidate_rows)
    for beam_rows in candidate_rows_by_beam.values():
        _sort_inverse_action_candidate_rows_by_preview(beam_rows)

    tuple_guidance = _reorder_inverse_action_candidates_with_tuple_critic(
        repair_tuple_bundle=repair_tuple_bundle,
        repair_tuple_controller_row=repair_tuple_controller_row,
        beam_states=beam_states,
        preview_rows=all_candidate_rows,
        candidate_rows_by_beam=candidate_rows_by_beam,
    )
    if isinstance(tuple_guidance, Mapping) and bool(tuple_guidance.get("trained", False)):
        action_meta["inverse_tuple_ranker_used"] = True
        action_meta["inverse_tuple_ranker_row_count"] = int(len(list(tuple_guidance.get("rows", []) or [])))
        action_meta["inverse_tuple_ranker_best_child_key"] = str(tuple_guidance.get("best_child_key", "") or "")
        action_meta["inverse_tuple_ranker_child_value_lambda"] = float(tuple_guidance.get("child_value_lambda", 0.0) or 0.0)
        action_meta["inverse_tuple_ranker_regret_weight"] = float(tuple_guidance.get("regret_weight", 1.0) or 1.0)
    elif isinstance(tuple_guidance, Mapping) and str(tuple_guidance.get("error", "") or "").strip():
        action_meta["inverse_tuple_ranker_error"] = str(tuple_guidance.get("error", "") or "")
    else:
        for beam_rows in candidate_rows_by_beam.values():
            _sort_inverse_action_candidate_rows_by_preview(beam_rows)

    action_meta["repair_opportunity_slate_count"] = int(len(beam_states))
    action_meta["repair_opportunity_slate"] = [
        _serialize_inverse_action_opportunity_row(
            parent_node=parent_node,
            decision_id=str(action_meta.get("repair_opportunity_slate_id", "")),
            beam_rank=int(beam_rank),
            beam_state=beam_state,
            beam_rows=candidate_rows_by_beam.get(int(beam_rank), []),
            budget_remaining=int(global_exact_score_budget),
            local_limit=int(local_limit),
        )
        for beam_rank, beam_state in enumerate(beam_states)
    ]

    parent_score = _score_inverse_action_parent(
        parent_node=parent_node,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        proj=proj,
        fp_mode=str(fp_mode),
        q_scale=float(q_scale),
        q_clip=float(q_clip),
        poly_degree=int(poly_degree),
        complexity_penalty=float(complexity_penalty),
        score_expr_cfg=score_expr_cfg if isinstance(score_expr_cfg, dict) else {},
        score_expr_fn=score_expr_fn,
    )
    parent_eff_for_controller = None
    if isinstance(parent_score, Mapping):
        parent_eff_value = parent_score.get("eff_mse", None)
        if parent_eff_value is not None:
            try:
                parent_eff_for_controller = float(parent_eff_value)
            except Exception:
                parent_eff_for_controller = None

    def _score_candidate_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
        return _score_inverse_action_expr(
            row["expr"],
            x_fit=x_fit,
            y_fit=y_fit,
            x_probe=x_probe,
            y_probe=y_probe,
            poly_degree=int(poly_degree),
            complexity_penalty=float(complexity_penalty),
            proj=proj,
            fp_mode=str(fp_mode),
            q_scale=float(q_scale),
            q_clip=float(q_clip),
            score_expr_cfg=score_expr_cfg if isinstance(score_expr_cfg, dict) else {},
            score_expr_fn=score_expr_fn,
        )

    selected_unique_rows: list[dict[str, Any]]
    allocation_meta: dict[str, Any]
    observed_by_child_key: dict[str, dict[str, Any]]
    final_opportunity_rows: list[dict[str, Any]]
    use_opportunity_controller = bool(repair_opportunity_controller_enable) and isinstance(repair_opportunity_bundle, Mapping)
    if use_opportunity_controller and bool(repair_opportunity_bundle.get("opportunity_controller_trained", False)):
        try:
            selected_unique_rows, allocation_meta, observed_by_child_key, final_opportunity_rows = (
                _allocate_inverse_exact_budget_with_opportunity_controller(
                    opportunity_bundle=repair_opportunity_bundle,
                    parent_node=parent_node,
                    decision_id=str(action_meta.get("repair_opportunity_slate_id", "")),
                    beam_states=beam_states,
                    candidate_rows_by_beam=candidate_rows_by_beam,
                    global_exact_score_budget=int(global_exact_score_budget),
                    local_limit=int(local_limit),
                    parent_eff_mse=parent_eff_for_controller,
                    score_candidate_fn=_score_candidate_row,
                )
            )
            action_meta["inverse_exact_allocator_mode"] = str(allocation_meta.get("allocator_mode", "opportunity_controller") or "opportunity_controller")
            action_meta["inverse_opportunity_controller_used"] = True
        except Exception as exc:
            action_meta["inverse_opportunity_controller_error"] = str(exc)
            use_opportunity_controller = False
    if not use_opportunity_controller or not bool(action_meta.get("inverse_opportunity_controller_used", False)):
        action_meta["inverse_exact_allocator_mode"] = "legacy"
        selected_unique_rows, allocation_meta = _select_inverse_exact_budget_rows(
            candidate_rows_by_beam=candidate_rows_by_beam,
            global_exact_score_budget=int(global_exact_score_budget),
            support_floor_beams=int(support_floor_beams),
        )
        observed_by_child_key = {}
        for row in selected_unique_rows:
            scored = _score_candidate_row(row)
            if scored is None:
                continue
            row["raw_mse"] = float(scored["raw_mse"])
            row["eff_mse"] = float(scored["eff_mse"])
            row["mapping"] = scored.get("mapping", None)
            row["exact_child_score_observed"] = True
            observed_by_child_key[str(row.get("child_key", "") or "")] = {
                "raw_mse": float(scored["raw_mse"]),
                "eff_mse": float(scored["eff_mse"]),
                "mapping": scored.get("mapping", None),
            }
        selected_counts_by_beam: dict[int, int] = {}
        selected_keys = {str(row.get("child_key", "") or "") for row in selected_unique_rows}
        for row in selected_unique_rows:
            beam_rank = int(row.get("beam_rank", 0) or 0)
            selected_counts_by_beam[beam_rank] = int(selected_counts_by_beam.get(beam_rank, 0) + 1)
        final_opportunity_rows = _build_inverse_allocator_opportunity_rows(
            parent_node=parent_node,
            decision_id=str(action_meta.get("repair_opportunity_slate_id", "")),
            beam_states=beam_states,
            candidate_rows_by_beam=candidate_rows_by_beam,
            local_limit=int(local_limit),
            selected_keys=selected_keys,
            selected_counts_by_beam=selected_counts_by_beam,
            observed_by_child_key=observed_by_child_key,
            parent_eff_mse=parent_eff_for_controller,
        )
        allocation_meta = {
            **dict(allocation_meta),
            "allocator_mode": "legacy",
            "trace": [
                {
                    "allocator_mode": "legacy",
                    "token_index": int(idx),
                    "beam_rank": int(row.get("beam_rank", 0) or 0),
                    "path": [int(v) for v in (row.get("path", ()) or ())],
                    "target_mode": str(row.get("target_mode", "") or ""),
                    "selected_child_key": str(row.get("child_key", "") or ""),
                    "selected_child_expr": str(row.get("child_expr", "") or row.get("child_key", "") or ""),
                    "selected_child_eff_mse": None if row.get("eff_mse", None) is None else float(row.get("eff_mse")),
                }
                for idx, row in enumerate(selected_unique_rows)
            ],
        }

    for child_key, dup_rows in duplicate_rows_by_key.items():
        observed = observed_by_child_key.get(str(child_key), None)
        if observed is None:
            continue
        for row in dup_rows:
            row["raw_mse"] = float(observed["raw_mse"])
            row["eff_mse"] = float(observed["eff_mse"])
            row["mapping"] = observed.get("mapping", None)
            row["exact_child_score_observed"] = True

    action_meta["inverse_exact_support_floor_selected"] = int(allocation_meta.get("support_floor_selected", 0))
    action_meta["inverse_exact_global_allocated"] = int(allocation_meta.get("global_allocated", 0))
    action_meta["inverse_exact_budget_trace"] = [dict(item) for item in list(allocation_meta.get("trace", []) or []) if isinstance(item, Mapping)]
    action_meta["inverse_exact_budget_trace_count"] = int(len(list(action_meta.get("inverse_exact_budget_trace", []) or [])))
    action_meta["repair_opportunity_slate_final"] = [dict(row) for row in list(final_opportunity_rows or [])]
    action_meta["repair_opportunity_slate_final_count"] = int(len(list(action_meta.get("repair_opportunity_slate_final", []) or [])))
    action_meta["inverse_exact_score_observed_count"] = int(len(observed_by_child_key))
    scored_slate = list(all_candidate_rows)

    observed_rows = [row for row in selected_unique_rows if bool(row.get("exact_child_score_observed", False))]
    if observed_rows:
        best_result = min(
            observed_rows,
            key=lambda row: (
                float(row.get("eff_mse", float("inf"))),
                float(row.get("raw_mse", float("inf"))),
                int(row.get("beam_rank", 0)),
                int(row.get("local_rank", 0)),
            ),
        )
        selected_state = best_result.get("beam_state", None)
        selected_local_count = int(best_result.get("local_candidate_count", 0) or 0)

    if best_result is None:
        if not any_repair_candidates:
            return _inverse_action_return(None, action_meta, return_meta=return_meta, status="no_repair_candidates")
        if not any_ranked_repairs:
            return _inverse_action_return(None, action_meta, return_meta=return_meta, status="no_ranked_repairs")
        return _inverse_action_return(None, action_meta, return_meta=return_meta, status="no_global_child")

    scored_slate.sort(
        key=lambda row: (
            float("inf") if row.get("eff_mse", None) is None else float(row.get("eff_mse")),
            float("inf") if row.get("raw_mse", None) is None else float(row.get("raw_mse")),
            int(not bool(row.get("exact_child_score_observed", False))),
            int(not bool(row.get("dedup_kept", False))),
            int(row.get("beam_rank", 0)),
            int(row.get("local_rank", 0)),
        )
    )
    action_meta["inverse_repair_slate_count"] = int(len(scored_slate))
    action_meta["inverse_repair_slate"] = [
        _serialize_inverse_action_slate_row(row)
        for row in scored_slate[: max(8, min(24, len(scored_slate)))]
    ]
    best_path = tuple(int(v) for v in (best_result.get("path", ()) or ()))
    best_state = dict(selected_state) if isinstance(selected_state, Mapping) else {}
    action_meta["local_candidate_count"] = int(selected_local_count)
    _update_inverse_action_meta_for_path(action_meta, best_path, best_state)
    action_meta["estimated_child_raw_mse"] = float(best_result["raw_mse"])
    action_meta["estimated_child_eff_mse"] = float(best_result["eff_mse"])
    action_meta["status"] = "ok"

    if parent_score is not None:
        parent_raw = parent_score.get("raw_mse", None)
        parent_eff = parent_score.get("eff_mse", None)
        action_meta["estimated_parent_raw_mse"] = parent_raw
        action_meta["estimated_parent_eff_mse"] = parent_eff
        if parent_raw is not None and math.isfinite(parent_raw):
            action_meta["estimated_one_hole_rel_improve_raw"] = float(
                max(0.0, parent_raw - float(action_meta["estimated_child_raw_mse"]))
                / max(parent_raw, 1.0e-30)
            )
        if parent_eff is not None and math.isfinite(parent_eff):
            action_meta["estimated_one_hole_rel_improve_eff"] = float(
                max(0.0, parent_eff - float(action_meta["estimated_child_eff_mse"]))
                / max(parent_eff, 1.0e-30)
            )

    return _inverse_action_return(
        best_result["expr"],
        action_meta,
        return_meta=return_meta,
    )
