# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from .basis_compile import canonicalize_basis_expr, compile_basis_linear_combo
from .basis_state import (
    BasisState,
    FeatureBlock,
    closure_keep_feature_blocks,
    drop_feature_block_with_dependents,
    ensure_feature_block_head_bundle,
    feature_block_id,
    topologically_order_feature_blocks,
)
from .expr_ast import eval_node, is_valid_node, node_str


def _valid_node(node: Any) -> tuple | None:
    if isinstance(node, tuple) and is_valid_node(node):
        return node
    return None


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
    valid = _valid_node(node)
    if valid is None:
        return None
    try:
        out = eval_node(valid, x)
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


def basis_block_primary_expr(block: FeatureBlock | None) -> tuple | None:
    if not isinstance(block, FeatureBlock):
        return None
    metadata = dict(getattr(block, "metadata", {}) or {})
    for key in ("block_expr_obj", "block_expr", "expr_obj", "expr"):
        node = _valid_node(metadata.get(key, None))
        if node is not None:
            return node
    atoms = tuple(getattr(block, "atoms", ()) or ())
    if len(atoms) == 1:
        return _valid_node(atoms[0])
    return _valid_node(atoms[0]) if atoms else None


def _bundle_entries(
    block: FeatureBlock | None,
    *,
    roles_attr: str = "latent_bundle_roles",
    nodes_attr: str = "latent_bundle_nodes",
) -> list[tuple[str, tuple]]:
    if not isinstance(block, FeatureBlock):
        return []
    out: list[tuple[str, tuple]] = []
    for role, node in zip(
        tuple(getattr(block, str(roles_attr), ()) or ()),
        tuple(getattr(block, str(nodes_attr), ()) or ()),
    ):
        valid = _valid_node(node)
        if valid is None:
            continue
        out.append((str(role), valid))
    return out


def _dedup_nodes(nodes: Sequence[tuple]) -> tuple[tuple, ...]:
    out: list[tuple] = []
    seen: set[str] = set()
    for node in list(nodes or ()):
        valid = _valid_node(node)
        if valid is None:
            continue
        key = str(node_str(valid))
        if key in seen:
            continue
        seen.add(key)
        out.append(valid)
    return tuple(out)


def _bundle_nodes_for_role(block: FeatureBlock | None, role: str) -> tuple[tuple, ...]:
    token = str(role or "")
    return tuple(node for entry_role, node in _bundle_entries(block) if entry_role == token)


def basis_block_head_terms(block: FeatureBlock | None) -> tuple[tuple[str, tuple], ...]:
    normalized = ensure_feature_block_head_bundle(block)
    if not isinstance(normalized, FeatureBlock):
        return ()
    explicit = _bundle_entries(
        normalized,
        roles_attr="head_bundle_roles",
        nodes_attr="head_bundle_nodes",
    )
    return tuple(explicit)


def basis_block_head_exprs(block: FeatureBlock | None) -> tuple[tuple, ...]:
    return _dedup_nodes([expr for _role, expr in basis_block_head_terms(block)])


def basis_state_block_exprs(state: BasisState | None) -> tuple[tuple, ...]:
    if not isinstance(state, BasisState):
        return ()
    out: list[tuple] = []
    seen: set[str] = set()
    for block in tuple(getattr(state, "blocks", ()) or ()):
        for expr in basis_block_head_exprs(block):
            if expr is None:
                continue
            key = str(node_str(expr))
            if key in seen:
                continue
            seen.add(key)
            out.append(expr)
    return tuple(out)


def _block_term_rows(blocks: Sequence[FeatureBlock] | None) -> tuple[tuple[int, str, tuple], ...]:
    out: list[tuple[int, str, tuple]] = []
    seen: set[tuple[int, str, str]] = set()
    for block_idx, block in enumerate(tuple(blocks or ())):
        if not isinstance(block, FeatureBlock):
            continue
        for role, expr in basis_block_head_terms(block):
            valid = _valid_node(expr)
            if valid is None:
                continue
            key = (int(block_idx), str(role), str(node_str(valid)))
            if key in seen:
                continue
            seen.add(key)
            out.append((int(block_idx), str(role), valid))
    return tuple(out)


def _fit_constant_head(
    *,
    y_fit_col: torch.Tensor,
    y_probe_col: torch.Tensor,
) -> tuple[float, float, float, torch.Tensor, torch.Tensor] | None:
    one_fit = torch.ones_like(y_fit_col)
    one_probe = torch.ones_like(y_probe_col)
    mask_fit = _finite_mask(y_fit_col, one_fit)
    mask_probe = _finite_mask(y_probe_col, one_probe)
    if mask_fit is None or mask_probe is None:
        return None
    if int(mask_fit.sum().item()) <= 0 or int(mask_probe.sum().item()) <= 0:
        return None
    a_fit = one_fit[mask_fit].to(dtype=torch.float64)
    b_fit = y_fit_col[mask_fit].to(dtype=torch.float64)
    try:
        beta = torch.linalg.lstsq(a_fit, b_fit).solution
    except Exception:
        try:
            beta = torch.linalg.pinv(a_fit) @ b_fit
        except Exception:
            return None
    intercept = float(beta[0, 0].item())
    pred_fit = one_fit.to(dtype=torch.float64) * float(intercept)
    pred_probe = one_probe.to(dtype=torch.float64) * float(intercept)
    residual_fit = y_fit_col.to(dtype=torch.float64) - pred_fit
    residual_probe = y_probe_col.to(dtype=torch.float64) - pred_probe
    fit_loss = float(torch.mean(residual_fit.square()).item())
    probe_loss = float(torch.mean(residual_probe.square()).item())
    return fit_loss, probe_loss, intercept, residual_fit, residual_probe


def _fit_blocks_linear_head(
    blocks: Sequence[FeatureBlock] | None,
    *,
    x_fit: torch.Tensor | None,
    y_fit_col: torch.Tensor,
    x_probe: torch.Tensor | None,
    y_probe_col: torch.Tensor,
    ridge_value: float,
) -> dict[str, Any] | None:
    current_blocks = tuple(
        normalized
        for normalized in (
            ensure_feature_block_head_bundle(block)
            for block in tuple(blocks or ())
        )
        if isinstance(normalized, FeatureBlock)
    )
    block_rows = _block_term_rows(current_blocks)
    block_exprs = tuple(expr for _, _, expr in block_rows)
    block_term_counts = [0 for _ in current_blocks]
    block_term_roles = [[] for _ in current_blocks]
    for block_idx, role, _expr in block_rows:
        if 0 <= int(block_idx) < len(block_term_counts):
            block_term_counts[int(block_idx)] += 1
            block_term_roles[int(block_idx)].append(str(role))

    if not block_exprs:
        const_fit = _fit_constant_head(y_fit_col=y_fit_col, y_probe_col=y_probe_col)
        if const_fit is None:
            return None
        fit_loss, probe_loss, intercept, residual_fit, residual_probe = const_fit
        return {
            "blocks": current_blocks,
            "block_rows": (),
            "block_exprs": (),
            "block_term_counts": block_term_counts,
            "block_term_roles": block_term_roles,
            "coeffs": [],
            "intercept": float(intercept),
            "fit_loss": float(fit_loss),
            "probe_loss": float(probe_loss),
            "residual_fit": residual_fit,
            "residual_probe": residual_probe,
            "block_coeff_scores": [0.0 for _ in current_blocks],
        }

    fit_cols = [_safe_eval_node_col(node, x_fit) for node in block_exprs]
    probe_cols = [_safe_eval_node_col(node, x_probe) for node in block_exprs]
    if any(col is None for col in fit_cols) or any(col is None for col in probe_cols):
        return None
    fit_cols = [col for col in fit_cols if isinstance(col, torch.Tensor)]
    probe_cols = [col for col in probe_cols if isinstance(col, torch.Tensor)]
    if not fit_cols or len(fit_cols) != len(block_exprs):
        return None

    one_fit = torch.ones_like(y_fit_col)
    one_probe = torch.ones_like(y_probe_col)
    mask_fit = _finite_mask(y_fit_col, one_fit, *fit_cols)
    mask_probe = _finite_mask(y_probe_col, one_probe, *probe_cols)
    if mask_fit is None or mask_probe is None:
        return None
    if int(mask_fit.sum().item()) <= len(fit_cols) or int(mask_probe.sum().item()) <= 0:
        return None

    a_fit = torch.cat([col[mask_fit] for col in fit_cols] + [one_fit[mask_fit]], dim=1).to(dtype=torch.float64)
    b_fit = y_fit_col[mask_fit].to(dtype=torch.float64)
    if a_fit.shape[0] <= a_fit.shape[1]:
        return None

    try:
        if ridge_value > 0.0:
            eye = torch.eye(a_fit.shape[1], dtype=a_fit.dtype, device=a_fit.device)
            beta = torch.linalg.solve(
                a_fit.T @ a_fit + ridge_value * eye,
                a_fit.T @ b_fit,
            )
        else:
            beta = torch.linalg.lstsq(a_fit, b_fit).solution
    except Exception:
        try:
            beta = torch.linalg.pinv(a_fit) @ b_fit
        except Exception:
            return None

    pred_fit = torch.cat(fit_cols + [one_fit], dim=1).to(dtype=torch.float64) @ beta
    pred_probe = torch.cat(probe_cols + [one_probe], dim=1).to(dtype=torch.float64) @ beta
    residual_fit = y_fit_col.to(dtype=torch.float64) - pred_fit
    residual_probe = y_probe_col.to(dtype=torch.float64) - pred_probe
    fit_loss = float(torch.mean(residual_fit.square()).item())
    probe_loss = float(torch.mean(residual_probe.square()).item())

    coeffs = [float(beta[idx, 0].item()) for idx in range(len(block_exprs))]
    intercept = float(beta[len(block_exprs), 0].item())
    block_coeff_scores = [0.0 for _ in current_blocks]
    for term_idx, (block_idx, _role, _expr) in enumerate(block_rows):
        if 0 <= int(block_idx) < len(block_coeff_scores):
            block_coeff_scores[int(block_idx)] = max(
                float(block_coeff_scores[int(block_idx)]),
                abs(float(coeffs[term_idx])),
            )

    return {
        "blocks": current_blocks,
        "block_rows": block_rows,
        "block_exprs": block_exprs,
        "block_term_counts": block_term_counts,
        "block_term_roles": block_term_roles,
        "coeffs": coeffs,
        "intercept": float(intercept),
        "fit_loss": float(fit_loss),
        "probe_loss": float(probe_loss),
        "residual_fit": residual_fit,
        "residual_probe": residual_probe,
        "block_coeff_scores": block_coeff_scores,
    }


def _least_squares_solve(
    design: torch.Tensor,
    rhs: torch.Tensor,
    *,
    ridge_value: float = 0.0,
) -> torch.Tensor | None:
    if not isinstance(design, torch.Tensor) or not isinstance(rhs, torch.Tensor):
        return None
    if design.ndim != 2 or rhs.ndim != 2 or design.shape[0] != rhs.shape[0]:
        return None
    try:
        if ridge_value > 0.0:
            eye = torch.eye(design.shape[1], dtype=design.dtype, device=design.device)
            return torch.linalg.solve(design.T @ design + ridge_value * eye, design.T @ rhs)
        return torch.linalg.lstsq(design, rhs).solution
    except Exception:
        try:
            return torch.linalg.pinv(design) @ rhs
        except Exception:
            return None


def score_basis_state_conditional_gain(
    current_state: BasisState | None,
    candidate_state: BasisState | None,
    *,
    x_fit: torch.Tensor | None,
    y_fit: torch.Tensor | None,
    ridge: float = 1.0e-8,
    orth_eps: float = 1.0e-10,
) -> Mapping[str, Any] | None:
    y_fit_col = _as_col_tensor(y_fit)
    if y_fit_col is None or not isinstance(x_fit, torch.Tensor):
        return None
    candidate_exprs = basis_state_block_exprs(candidate_state)
    if not candidate_exprs:
        return None
    active_exprs = basis_state_block_exprs(current_state)
    one_fit = torch.ones_like(y_fit_col)
    active_cols = [_safe_eval_node_col(node, x_fit) for node in active_exprs]
    candidate_cols = [_safe_eval_node_col(node, x_fit) for node in candidate_exprs]
    if any(col is None for col in active_cols) or any(col is None for col in candidate_cols):
        return None
    active_cols = [col for col in active_cols if isinstance(col, torch.Tensor)]
    candidate_cols = [col for col in candidate_cols if isinstance(col, torch.Tensor)]
    if not candidate_cols:
        return None
    mask = _finite_mask(y_fit_col, one_fit, *active_cols, *candidate_cols)
    if mask is None:
        return None
    if int(mask.sum().item()) <= (1 + len(active_cols) + len(candidate_cols)):
        return None

    target = y_fit_col[mask].to(dtype=torch.float64)
    intercept = one_fit[mask].to(dtype=torch.float64)
    active_fit_cols = [col[mask].to(dtype=torch.float64) for col in active_cols]
    candidate_fit_cols = [col[mask].to(dtype=torch.float64) for col in candidate_cols]
    active_design = torch.cat([*active_fit_cols, intercept], dim=1) if active_fit_cols else intercept
    candidate_design = torch.cat(candidate_fit_cols, dim=1)
    if active_design.ndim != 2 or candidate_design.ndim != 2:
        return None
    if active_design.shape[0] <= active_design.shape[1]:
        return None

    base_beta = _least_squares_solve(active_design, target, ridge_value=max(0.0, float(ridge)))
    if base_beta is None:
        return None
    residual = target - (active_design @ base_beta)

    proj_beta = _least_squares_solve(active_design, candidate_design, ridge_value=max(0.0, float(ridge)))
    if proj_beta is None:
        return None
    candidate_orth = candidate_design - (active_design @ proj_beta)
    if candidate_orth.shape[0] <= 0 or candidate_orth.shape[1] <= 0:
        return None

    gram = candidate_orth.T @ candidate_orth
    rhs = candidate_orth.T @ residual
    orth_scale = float(torch.max(torch.abs(candidate_orth)).item()) if candidate_orth.numel() > 0 else 0.0
    if not math.isfinite(orth_scale) or orth_scale <= float(orth_eps):
        gain = 0.0
        rank = 0
    else:
        eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        try:
            alpha = torch.linalg.solve(gram + float(orth_eps) * eye, rhs)
        except Exception:
            try:
                alpha = torch.linalg.pinv(gram + float(orth_eps) * eye) @ rhs
            except Exception:
                return None
        gain = float((rhs.T @ alpha).reshape(()).item())
        gain = max(0.0, float(gain))
        try:
            rank = int(torch.linalg.matrix_rank(candidate_orth).item())
        except Exception:
            rank = int(candidate_orth.shape[1])
    return {
        "gain": float(gain),
        "candidate_rank": int(rank),
        "candidate_term_count": int(candidate_design.shape[1]),
        "active_term_count": int(active_design.shape[1] - 1),
        "n_rows": int(target.shape[0]),
    }


def fit_basis_state_head(
    state: BasisState | None,
    *,
    x_fit: torch.Tensor | None,
    y_fit: torch.Tensor | None,
    x_probe: torch.Tensor | None,
    y_probe: torch.Tensor | None,
    ridge: float = 0.0,
    route_name: str = "basis_head_refit",
    backward_prune_enable: bool = True,
    backward_prune_rel: float = 1.0e-8,
    backward_prune_abs: float = 1.0e-10,
) -> BasisState | None:
    if not isinstance(state, BasisState):
        return None
    y_fit_col = _as_col_tensor(y_fit)
    y_probe_col = _as_col_tensor(y_probe)
    if y_fit_col is None or y_probe_col is None:
        return state
    ridge_value = max(0.0, float(ridge))
    prune_rel = max(0.0, float(backward_prune_rel))
    prune_abs = max(0.0, float(backward_prune_abs))
    current_blocks = tuple(
        normalized
        for normalized in (
            ensure_feature_block_head_bundle(block)
            for block in tuple(state.blocks)
        )
        if isinstance(normalized, FeatureBlock)
    )
    current_blocks = topologically_order_feature_blocks(current_blocks, drop_orphans=True)
    coeff_prune_passes = 0
    coeff_pruned_block_count = 0
    subset_prune_passes = 0
    subset_pruned_block_count = 0

    block_exprs: tuple[tuple, ...] = ()
    coeffs: list[float] = []
    intercept = 0.0
    fit_loss = float(state.fit_loss)
    probe_loss = float(state.probe_loss)
    residual_fit = state.residual_fit
    residual_probe = state.residual_probe
    block_coeff_scores: list[float] = []
    block_term_counts: list[int] = []

    while True:
        fit_result = _fit_blocks_linear_head(
            current_blocks,
            x_fit=x_fit,
            y_fit_col=y_fit_col,
            x_probe=x_probe,
            y_probe_col=y_probe_col,
            ridge_value=ridge_value,
        )
        if fit_result is None:
            return state
        current_blocks = tuple(fit_result["blocks"])
        block_rows = tuple(fit_result["block_rows"])
        block_exprs = tuple(fit_result["block_exprs"])
        block_term_counts = list(fit_result["block_term_counts"])
        block_term_roles = list(fit_result["block_term_roles"])
        coeffs = list(fit_result["coeffs"])
        intercept = float(fit_result["intercept"])
        fit_loss = float(fit_result["fit_loss"])
        probe_loss = float(fit_result["probe_loss"])
        residual_fit = fit_result["residual_fit"]
        residual_probe = fit_result["residual_probe"]
        block_coeff_scores = list(fit_result["block_coeff_scores"])

        if (not bool(backward_prune_enable)) or len(current_blocks) <= 0:
            break
        coeff_scale = max(
            1.0,
            abs(float(intercept)),
            max((abs(float(value)) for value in coeffs), default=0.0),
        )
        prune_threshold = max(float(prune_abs), coeff_scale * float(prune_rel))
        keep_indices = [
            idx
            for idx, score in enumerate(block_coeff_scores)
            if float(score) > float(prune_threshold)
        ]
        if len(keep_indices) == len(current_blocks):
            break
        keep_block_ids = [
            feature_block_id(current_blocks[idx])
            for idx in keep_indices
            if 0 <= int(idx) < len(current_blocks)
        ]
        kept_blocks = closure_keep_feature_blocks(current_blocks, keep_block_ids)
        if len(kept_blocks) == len(current_blocks):
            break
        coeff_pruned_block_count += int(len(current_blocks) - len(kept_blocks))
        coeff_prune_passes += 1
        current_blocks = topologically_order_feature_blocks(kept_blocks, drop_orphans=True)

    if bool(backward_prune_enable) and len(current_blocks) > 1:
        while True:
            baseline = _fit_blocks_linear_head(
                current_blocks,
                x_fit=x_fit,
                y_fit_col=y_fit_col,
                x_probe=x_probe,
                y_probe_col=y_probe_col,
                ridge_value=ridge_value,
            )
            if baseline is None:
                return state
            current_blocks = tuple(baseline["blocks"])
            block_rows = tuple(baseline["block_rows"])
            block_exprs = tuple(baseline["block_exprs"])
            block_term_counts = list(baseline["block_term_counts"])
            block_term_roles = list(baseline["block_term_roles"])
            coeffs = list(baseline["coeffs"])
            intercept = float(baseline["intercept"])
            fit_loss = float(baseline["fit_loss"])
            probe_loss = float(baseline["probe_loss"])
            residual_fit = baseline["residual_fit"]
            residual_probe = baseline["residual_probe"]
            block_coeff_scores = list(baseline["block_coeff_scores"])
            if len(current_blocks) <= 1:
                break
            fit_tol = max(float(prune_abs), max(1.0, abs(float(fit_loss))) * float(prune_rel))
            probe_tol = max(float(prune_abs), max(1.0, abs(float(probe_loss))) * float(prune_rel))
            best_idx: int | None = None
            best_result: dict[str, Any] | None = None
            best_sort: tuple[Any, ...] | None = None
            for drop_idx in range(len(current_blocks)):
                candidate_blocks = drop_feature_block_with_dependents(
                    current_blocks,
                    feature_block_id(current_blocks[int(drop_idx)]),
                )
                candidate = _fit_blocks_linear_head(
                    candidate_blocks,
                    x_fit=x_fit,
                    y_fit_col=y_fit_col,
                    x_probe=x_probe,
                    y_probe_col=y_probe_col,
                    ridge_value=ridge_value,
                )
                if candidate is None:
                    continue
                cand_fit = float(candidate["fit_loss"])
                cand_probe = float(candidate["probe_loss"])
                if cand_fit > float(fit_loss) + fit_tol:
                    continue
                if cand_probe > float(probe_loss) + probe_tol:
                    continue
                cand_complexity = sum(float(block.complexity()) for block in tuple(candidate["blocks"]))
                sort_key = (
                    cand_probe,
                    cand_fit,
                    cand_complexity,
                    int(drop_idx),
                )
                if best_sort is None or sort_key < best_sort:
                    best_idx = int(drop_idx)
                    best_result = candidate
                    best_sort = sort_key
            if best_idx is None or best_result is None:
                break
            subset_pruned_block_count += 1
            subset_prune_passes += 1
            current_blocks = topologically_order_feature_blocks(best_result["blocks"], drop_orphans=True)
            block_rows = tuple(best_result["block_rows"])
            block_exprs = tuple(best_result["block_exprs"])
            block_term_counts = list(best_result["block_term_counts"])
            block_term_roles = list(best_result["block_term_roles"])
            coeffs = list(best_result["coeffs"])
            intercept = float(best_result["intercept"])
            fit_loss = float(best_result["fit_loss"])
            probe_loss = float(best_result["probe_loss"])
            residual_fit = best_result["residual_fit"]
            residual_probe = best_result["residual_probe"]
            block_coeff_scores = list(best_result["block_coeff_scores"])

    compiled_expr = compile_basis_linear_combo(block_exprs, coeffs, intercept)
    compiled_expr = compiled_expr or canonicalize_basis_expr(("const", float(intercept)))
    total_complexity = sum(float(block.complexity()) for block in current_blocks)
    total_prune_passes = int(coeff_prune_passes + subset_prune_passes)
    total_pruned_block_count = int(coeff_pruned_block_count + subset_pruned_block_count)

    fit_bundle = dict(getattr(state, "fit_bundle", {}) or {})
    fit_bundle["basis_head"] = {
        "kind": "basis_linear_head",
        "coeffs": coeffs,
        "intercept": float(intercept),
        "ridge": float(ridge_value),
        "backward_prune_enable": bool(backward_prune_enable),
        "backward_prune_rel": float(prune_rel),
        "backward_prune_abs": float(prune_abs),
        "backward_prune_passes": int(total_prune_passes),
        "pruned_block_count": int(total_pruned_block_count),
        "coeff_prune_passes": int(coeff_prune_passes),
        "coeff_pruned_block_count": int(coeff_pruned_block_count),
        "subset_prune_passes": int(subset_prune_passes),
        "subset_pruned_block_count": int(subset_pruned_block_count),
        "kept_block_count": int(len(current_blocks)),
        "block_coeff_scores": [float(v) for v in block_coeff_scores],
        "block_term_counts": [int(v) for v in block_term_counts],
        "term_roles": [str(role) for _, role, _ in block_rows],
        "block_term_roles": [[str(role) for role in roles] for roles in block_term_roles],
        "term_exprs": [str(node_str(node)) for node in block_exprs],
        "block_exprs": [str(node_str(node)) for node in block_exprs],
    }
    diagnostics = dict(getattr(state, "diagnostics", {}) or {})
    diagnostics["route"] = str(route_name)
    diagnostics["basis_head_kind"] = "basis_linear_head"
    diagnostics["basis_head_pruned"] = bool(total_pruned_block_count > 0)
    diagnostics["basis_head_prune_passes"] = int(total_prune_passes)
    diagnostics["basis_head_subset_pruned"] = bool(subset_pruned_block_count > 0)
    provenance = tuple(getattr(state, "provenance", ()) or ())
    provenance = (*provenance, f"{str(route_name)}:basis_head")
    return BasisState(
        blocks=topologically_order_feature_blocks(tuple(current_blocks), drop_orphans=True),
        fit_bundle=fit_bundle,
        fit_loss=float(fit_loss),
        probe_loss=float(probe_loss),
        complexity=float(total_complexity),
        residual_fit=residual_fit,
        residual_probe=residual_probe,
        residual_witness=state.residual_witness,
        diagnostics=diagnostics,
        provenance=provenance,
        compiled_expr=compiled_expr,
    )


__all__ = [
    "basis_block_head_terms",
    "basis_block_head_exprs",
    "basis_block_primary_expr",
    "basis_state_block_exprs",
    "fit_basis_state_head",
    "score_basis_state_conditional_gain",
]
