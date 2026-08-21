# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from typing import Any, Mapping

import torch

from ..expr_ast import node_dims
from ..expr_mapping import fit_best
from ..inverse_action import (
    _estimate_inverse_action_transport,
    _inverse_action_branch_state_debug,
    _inverse_action_path_mode_beam_states,
)
from ..inverse_core import _inverse_target_mode_rows, _normalize_inverse_target_mode
from ..inverse_search import (
    _inverse_branch_beam_factor,
    _inverse_effective_branch_beam_width,
    _inverse_effective_thresholds,
    _inverse_family_gain_scale,
    _inverse_path_cut_factor,
    _inverse_path_profile,
    _inverse_static_path_score,
)
from ..expr_ast import eval_node
from .compat import OuterScaffoldSpec


def fit_scaffold_mapping(
    parent_node,
    *,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    poly_degree: int,
    var_dims,
) -> tuple[Mapping[str, Any] | None, str]:
    if var_dims is not None:
        try:
            if node_dims(parent_node, var_dims) is None:
                return None, "invalid_parent_dims"
        except Exception:
            return None, "invalid_parent_dims"
    try:
        pred_fit = eval_node(parent_node, x_fit)
        pred_probe = eval_node(parent_node, x_probe)
    except Exception:
        return None, "parent_eval_failed"
    if (not torch.is_tensor(pred_fit)) or (not torch.is_tensor(pred_probe)):
        return None, "parent_eval_failed"
    if (not torch.isfinite(pred_fit).all()) or (not torch.isfinite(pred_probe).all()):
        return None, "parent_nonfinite"
    fit_ret = fit_best(pred_fit, y_fit, int(poly_degree))
    if fit_ret is None or len(fit_ret) != 2:
        return None, "parent_mapping_failed"
    _mse, mapping = fit_ret
    if not isinstance(mapping, Mapping):
        return None, "parent_mapping_failed"
    return dict(mapping), "ok"


def build_scaffold_beam_state(
    spec: OuterScaffoldSpec,
    *,
    parent_mapping: Mapping[str, Any],
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    pool_nodes,
    pool_phi_fit: torch.Tensor,
    pool_phi_probe: torch.Tensor,
    pool_dims,
    safe_eps: float,
    beam_cfg: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, str, dict[str, Any]]:
    path = tuple(int(v) for v in tuple(spec.hole_path or ()))
    try:
        transport_ctx = _estimate_inverse_action_transport(
            spec.parent_node,
            dict(parent_mapping or {}),
            x_fit,
            y_fit,
            x_probe,
            y_probe,
            [path],
            safe_eps=float(safe_eps),
        )
    except Exception:
        return None, "transport_failed", {}
    try:
        beam_states = _inverse_action_path_mode_beam_states(
            parent_node=spec.parent_node,
            parent_mapping=dict(parent_mapping or {}),
            x_fit=x_fit,
            y_fit=y_fit,
            x_probe=x_probe,
            y_probe=y_probe,
            pool_nodes=pool_nodes,
            pool_phi_fit=pool_phi_fit,
            pool_phi_probe=pool_phi_probe,
            pool_dims=pool_dims,
            all_paths=[path],
            path_target_modes={path: str(spec.target_mode)} if spec.target_mode else None,
            transport_ctx=transport_ctx,
            cfg=dict(beam_cfg or {}),
            beam_width=1,
        )
    except Exception:
        return None, "beam_state_failed", {}
    if not beam_states:
        return None, "no_beam_state", debug_scaffold_beam_state_failure(
            spec,
            parent_mapping=dict(parent_mapping or {}),
            x_fit=x_fit,
            y_fit=y_fit,
            x_probe=x_probe,
            y_probe=y_probe,
            pool_nodes=pool_nodes,
            pool_phi_fit=pool_phi_fit,
            pool_phi_probe=pool_phi_probe,
            pool_dims=pool_dims,
            transport_ctx=transport_ctx,
            beam_cfg=beam_cfg,
        )
    return dict(beam_states[0]), "ok", {}


def debug_scaffold_beam_state_failure(
    spec: OuterScaffoldSpec,
    *,
    parent_mapping: Mapping[str, Any],
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    pool_nodes,
    pool_phi_fit: torch.Tensor,
    pool_phi_probe: torch.Tensor,
    pool_dims,
    transport_ctx: Mapping[str, Any],
    beam_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    path = tuple(int(v) for v in tuple(spec.hole_path or ()))
    diag: dict[str, Any] = {
        "path": [int(v) for v in path],
        "scaffold_id": str(spec.scaffold_id),
        "family": str(spec.family),
        "target_mode": str(spec.target_mode),
    }
    try:
        sub = spec.parent_node
        for idx in path:
            sub = sub[int(idx)]
    except Exception:
        diag["status"] = "path_missing"
        return diag

    profile = _inverse_path_profile(spec.parent_node, path, parent_mapping)
    min_valid_eff, min_conf_eff = _inverse_effective_thresholds(
        float(beam_cfg["min_valid_frac"]),
        float(beam_cfg["min_confidence"]),
        profile=profile,
        periodic_min_valid_scale=float(beam_cfg["periodic_min_valid_scale"]),
        periodic_min_confidence_scale=float(beam_cfg["periodic_min_confidence_scale"]),
    )
    family_scale = _inverse_family_gain_scale(
        profile,
        periodic_path_penalty=float(beam_cfg["periodic_path_penalty"]),
        nonperiodic_muldiv_bonus=float(beam_cfg["nonperiodic_muldiv_bonus"]),
        nonperiodic_explogsqrt_bonus=float(beam_cfg["nonperiodic_explogsqrt_bonus"]),
    )
    path_transport_rel = dict(transport_ctx.get("path_transport_rel", {}) or {})
    transport_rel = float(path_transport_rel.get(path, 0.0))
    transport_factor = 1.0 + 0.35 * max(0.0, transport_rel)
    path_beam_width = _inverse_effective_branch_beam_width(profile, int(beam_cfg["branch_beam_width"]))
    cut_factor = _inverse_path_cut_factor(
        spec.parent_node,
        path,
        profile,
        additive_descend_penalty=float(beam_cfg["additive_descend_penalty"]),
        nonadditive_leaf_penalty=float(beam_cfg["nonadditive_leaf_penalty"]),
    )
    path_mode = _normalize_inverse_target_mode(
        spec.target_mode,
        default=str(beam_cfg["target_mode"]),
    )
    target_mode_rows = _inverse_target_mode_rows(
        spec.parent_node,
        parent_mapping,
        path,
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        profile=profile,
        safe_eps=float(beam_cfg["safe_eps"]),
        confidence_mode=str(beam_cfg["confidence_mode"]),
        confidence_target_gain=float(beam_cfg["confidence_target_gain"]),
        confidence_floor=float(beam_cfg["confidence_floor"]),
        branch_beam_width=int(path_beam_width),
        target_mode=str(path_mode),
        full_mapping_penalty=float(beam_cfg["full_mapping_penalty"]),
        exact_simple_target_bonus=float(beam_cfg["exact_simple_target_bonus"]),
    )

    diag.update(
        {
            "profile": {
                "has_periodic": bool(profile.get("has_periodic", False)),
                "has_muldiv": bool(profile.get("has_muldiv", False)),
                "has_explogsqrt": bool(profile.get("has_explogsqrt", False)),
                "has_ambiguous_inverse": bool(profile.get("has_ambiguous_inverse", False)),
                "exact_monotone": bool(profile.get("exact_monotone", False)),
            },
            "min_valid_eff": float(min_valid_eff),
            "min_conf_eff": float(min_conf_eff),
            "family_scale": float(family_scale),
            "transport_rel": float(transport_rel),
            "transport_factor": float(transport_factor),
            "cut_factor": float(cut_factor),
            "target_mode_rows_count": int(len(list(target_mode_rows or ()))),
        }
    )
    if not target_mode_rows:
        diag["status"] = "target_mode_rows_empty"
        return diag

    static_score, _nonadditive = _inverse_static_path_score(spec.parent_node, path, parent_mapping)
    target_dim = node_dims(sub, beam_cfg["var_dims"]) if bool(beam_cfg.get("dm", False)) else None
    path_cfg = dict(beam_cfg or {})
    path_cfg["min_valid_eff"] = float(min_valid_eff)
    path_cfg["min_conf_eff"] = float(min_conf_eff)
    mode_debug: list[dict[str, Any]] = []
    overall_reason_counts: dict[str, int] = {}
    positive_state_count = 0

    for mode_row in list(target_mode_rows or ()):
        mode_name = str(mode_row.get("mode", "full"))
        mode_factor = float(mode_row.get("mode_factor", 1.0))
        inv_fit_list = list(mode_row.get("fit_list", []) or [])
        inv_probe_list = list(mode_row.get("probe_list", []) or [])
        probe_by_branch = {str(t.branch_id): t for t in inv_probe_list}
        reason_counts: dict[str, int] = {}
        kept_states: list[dict[str, Any]] = []

        for inv_fit in inv_fit_list:
            inv_probe = probe_by_branch.get(str(inv_fit.branch_id), None)
            if inv_probe is None and inv_probe_list:
                inv_probe = inv_probe_list[0]
            if inv_probe is None:
                reason = "missing_probe_branch"
                reason_counts[reason] = int(reason_counts.get(reason, 0) or 0) + 1
                overall_reason_counts[reason] = int(overall_reason_counts.get(reason, 0) or 0) + 1
                continue
            state, reason, _branch_diag = _inverse_action_branch_state_debug(
                parent_node=spec.parent_node,
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
            reason_counts[str(reason)] = int(reason_counts.get(str(reason), 0) or 0) + 1
            overall_reason_counts[str(reason)] = int(overall_reason_counts.get(str(reason), 0) or 0) + 1
            if state is not None:
                kept_states.append(dict(state))

        mode_entry: dict[str, Any] = {
            "mode": str(mode_name),
            "mode_factor": float(mode_factor),
            "fit_branches": int(len(inv_fit_list)),
            "probe_branches": int(len(inv_probe_list)),
            "kept_branches": int(len(kept_states)),
            "reason_counts": dict(sorted(reason_counts.items())),
        }
        if kept_states:
            kept_states.sort(
                key=lambda row: (
                    float(row.get("gain_raw", -float("inf"))),
                    -float(row.get("best_alt_mse", float("inf"))),
                ),
                reverse=True,
            )
            best_state = dict(kept_states[0])
            path_best_gain_raw = float(best_state.get("gain_raw", -float("inf")))
            if bool(profile.get("has_ambiguous_inverse", False)):
                branch_rows_for_factor = [
                    {"weighted_rel_gain_raw": float(r.get("gain_raw", 0.0))}
                    for r in kept_states
                ]
                branch_factor, branch_support, branch_positive = _inverse_branch_beam_factor(
                    branch_rows_for_factor,
                    ambiguity_penalty=float(beam_cfg["branch_ambiguity_penalty"]),
                )
            else:
                branch_factor, branch_support, branch_positive = 1.0, 1.0, 1
            path_gain_pre_cut = path_best_gain_raw * float(family_scale) * float(branch_factor)
            path_gain = path_gain_pre_cut * float(cut_factor)
            mode_entry["best_gain_raw"] = float(path_best_gain_raw)
            mode_entry["best_path_gain"] = float(path_gain)
            mode_entry["branch_factor"] = float(branch_factor)
            mode_entry["branch_support"] = float(branch_support)
            mode_entry["branch_positive_count"] = int(branch_positive)
            mode_entry["best_effective_n"] = float(best_state.get("effective_n", 0.0) or 0.0)
            mode_entry["best_lin_rel"] = float(best_state.get("lin_rel", 0.0) or 0.0)
            mode_entry["best_alt_mse"] = float(best_state.get("best_alt_mse", float("inf")) or float("inf"))
            if path_gain > 0.0 and float(best_state.get("gain_raw", 0.0) or 0.0) > 0.0:
                positive_state_count += 1
        mode_debug.append(mode_entry)

    diag["mode_debug"] = mode_debug
    diag["reason_counts"] = dict(sorted(overall_reason_counts.items()))
    diag["status"] = "no_positive_path_gain" if positive_state_count > 0 else "no_branch_state"
    return diag


__all__ = [
    "build_scaffold_beam_state",
    "debug_scaffold_beam_state_failure",
    "fit_scaffold_mapping",
]
