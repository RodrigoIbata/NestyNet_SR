# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

"""Typed analytic-family candidate construction for factorized DE."""

import math
import time
from typing import Any, Sequence
import torch
from nestynet_sr.sr_core.ast_simplify import ast_node_count
from nestynet_sr.sr_core.bridges import Add, Cos, Exp, Log, Mul, Pow, Sin, Var
from nestynet_sr.sr_de.de_search import ridge_lstsq

from ._factorized_de_frontend import (
    DEFeatureGroup,
    _diag_inc,
)
from ._factorized_de_operator import (
    FactorizedDEBlock,
    _anchor_ast,
    _anchor_tensor,
    _candidate_identity_key,
    _canonical_equation,
    _consistency_evidence_tier,
    _eval_ast_on_features,
    _eval_univariate_ast_on_values,
    _finite_xy_rows,
    _fit_original_scale_family_basis,
    _lane_witness_stats,
    _masked_original_scale_probe_mse,
    _normalized_group_quality_weights,
    _pooled_same_coord_coeff_target,
    _pooled_target_mse_from_local_ast,
    _probe_mse_from_residuals,
    _quality_weighted_probe_mse_from_residuals,
    _residual_ratio_collapse_diagnostics,
    _safe_ratio_target,
    _scale_weighted_trimmed_lstsq,
    _simplify_de_ast,
    _sum_linear_terms_ast,
    _trimmed_mean_sq,
    _trimmed_probe_mse_from_residuals,
    _x_lane_shape_stats,
)

def _make_candidate_row(
    *,
    lane: str,
    family: str,
    base_mode: str,
    groups: Sequence[DEFeatureGroup],
    order: int,
    x_axis: int,
    base_ast,
    carrier_ast,
    coord_ast,
    coeff_ast,
    coeff_expr: str,
    mapping: dict[str, Any],
    size: int,
    ratio_probe_mse: float,
    fit_target_mse: float | None = None,
    probe_target_mse: float | None = None,
    resid_probe_parts: Sequence[torch.Tensor],
    rel_eps: float,
    coeff_local_ast=None,
) -> dict[str, Any] | None:
    if coeff_ast is None:
        return None
    block_ast = Mul(coeff_ast, carrier_ast)
    nonanchor_ast = block_ast if base_ast is None else Add(base_ast, block_ast)

    probe_residuals = []
    for group in groups:
        anchor_probe = _anchor_tensor(group.features, order=int(order))[1].reshape(-1)
        nonanchor_probe = _eval_ast_on_features(
            nonanchor_ast,
            features=group.features,
            split="probe",
            x_axis=int(x_axis),
        ).reshape(-1)
        rr = anchor_probe + nonanchor_probe
        if not torch.isfinite(rr).all():
            return None
        probe_residuals.append(rr)

    raw_probe_mse = _probe_mse_from_residuals(probe_residuals)
    lane_norm = str(lane or "")
    if lane_norm == "x_coeff_on_u":
        probe_mse = _masked_original_scale_probe_mse(
            groups=groups,
            resid_probe_parts=resid_probe_parts,
            carrier_ast=carrier_ast,
            coeff_ast=coeff_ast,
            x_axis=int(x_axis),
            rel_eps=float(rel_eps),
            robust=True,
        )
    elif lane_norm == "state_nonlinearity":
        probe_mse = _quality_weighted_probe_mse_from_residuals(groups, probe_residuals, robust=True)
    else:
        # Damping lanes carry singular coordinate features (e.g. du/x); a few
        # unresolvable boundary-layer rows must not dominate the headline score.
        probe_mse = _trimmed_probe_mse_from_residuals(probe_residuals)
    probe_rms = math.sqrt(probe_mse) if math.isfinite(probe_mse) else float("inf")
    raw_probe_rms = math.sqrt(raw_probe_mse) if math.isfinite(raw_probe_mse) else float("inf")
    witness_kind, consistency_score, consistency_pairs, consistency_total_pairs = _lane_witness_stats(
        lane=str(lane),
        groups=groups,
        resid_probe_parts=resid_probe_parts,
        carrier_ast=carrier_ast,
        coord_ast=coord_ast,
        x_axis=int(x_axis),
        rel_eps=float(rel_eps),
    )
    collapse_diag = _residual_ratio_collapse_diagnostics(
        groups=groups,
        resid_parts=resid_probe_parts,
        carrier_ast=carrier_ast,
        coord_ast=coord_ast,
        split="probe",
        x_axis=int(x_axis),
        rel_eps=float(rel_eps),
        min_rows=2,
    )
    shape_stats = (
        _x_lane_shape_stats(
            groups=groups,
            resid_probe_parts=resid_probe_parts,
            carrier_ast=carrier_ast,
            coord_ast=coord_ast,
            coeff_ast=coeff_ast,
            x_axis=int(x_axis),
            rel_eps=float(rel_eps),
        )
        if lane_norm == "x_coeff_on_u"
        else {
            "shape_score": 0.0,
            "sign_changes": 0.0,
            "curvature_ratio": 0.0,
            "tv_ratio": 1.0,
        }
    )
    nonanchor_ast_raw = nonanchor_ast
    residual_ast_raw = Add(_anchor_ast(int(order), x_axis=int(x_axis)), nonanchor_ast_raw)
    nonanchor_ast_simplified = _simplify_de_ast(nonanchor_ast_raw)
    residual_ast_simplified = _simplify_de_ast(residual_ast_raw)
    residual_ast = residual_ast_simplified
    nonanchor_ast = nonanchor_ast_simplified
    canonical_equation_raw = _canonical_equation(int(order), int(x_axis), nonanchor_ast_raw)
    canonical_equation_simplified = _canonical_equation(int(order), int(x_axis), nonanchor_ast_simplified)
    canonical_equation = canonical_equation_simplified
    block = FactorizedDEBlock(
        role=str(lane),
        carrier_ast=carrier_ast,
        coord_ast=coord_ast,
        coeff_ast=coeff_ast,
        block_ast=block_ast,
        diagnostics={
            "family": str(family),
            "base_mode": str(base_mode),
            "coeff_expr": str(coeff_expr),
            "ratio_probe_mse": float(ratio_probe_mse),
            "fit_target_mse": None if fit_target_mse is None else float(fit_target_mse),
            "probe_target_mse": None if probe_target_mse is None else float(probe_target_mse),
            "raw_probe_mse": float(raw_probe_mse),
            "raw_probe_rms": float(raw_probe_rms),
            "mapping": dict(mapping or {}),
            "consistency_score": float(consistency_score),
            "consistency_pairs": int(consistency_pairs),
            "consistency_total_pairs": int(consistency_total_pairs),
            "witness_kind": str(witness_kind),
            "shape_score": float(shape_stats["shape_score"]),
            "sign_changes": float(shape_stats["sign_changes"]),
            "curvature_ratio": float(shape_stats["curvature_ratio"]),
            "tv_ratio": float(shape_stats["tv_ratio"]),
            **collapse_diag,
            "evidence_tier": _consistency_evidence_tier(
                {
                    "consistency_score": float(consistency_score),
                    "consistency_pairs": int(consistency_pairs),
                    "consistency_total_pairs": int(consistency_total_pairs),
                }
            )[1],
        },
    )

    return {
        "order": int(order),
        "carrier_ast": carrier_ast,
        "coord_ast": coord_ast,
        "coeff_ast": coeff_ast,
        "block_ast": block_ast,
        "blocks": [block],
        "lane": str(lane),
        "family": str(family),
        "base_mode": str(base_mode),
        "nonanchor_ast": nonanchor_ast,
        "nonanchor_ast_raw": nonanchor_ast_raw,
        "nonanchor_ast_simplified": nonanchor_ast_simplified,
        "residual_ast": residual_ast,
        "residual_ast_raw": residual_ast_raw,
        "residual_ast_simplified": residual_ast_simplified,
        "canonical_equation": canonical_equation,
        "canonical_equation_raw": canonical_equation_raw,
        "canonical_equation_simplified": canonical_equation_simplified,
        "probe_mse": float(probe_mse),
        "probe_rms": float(probe_rms),
        "raw_probe_mse": float(raw_probe_mse),
        "raw_probe_rms": float(raw_probe_rms),
        "coeff_expr": str(coeff_expr),
        "mapping": dict(mapping or {}),
        "size": int(size),
        "symbolic_size_raw": int(ast_node_count(residual_ast_raw)),
        "symbolic_size_simplified": int(ast_node_count(residual_ast_simplified)),
        "ratio_probe_mse": float(ratio_probe_mse),
        "fit_target_mse": None if fit_target_mse is None else float(fit_target_mse),
        "probe_target_mse": None if probe_target_mse is None else float(probe_target_mse),
        "coeff_local_ast": coeff_local_ast,
        "consistency_score": float(consistency_score),
        "consistency_pairs": int(consistency_pairs),
        "consistency_total_pairs": int(consistency_total_pairs),
        "witness_kind": str(witness_kind),
        "shape_score": float(shape_stats["shape_score"]),
        "sign_changes": float(shape_stats["sign_changes"]),
        "curvature_ratio": float(shape_stats["curvature_ratio"]),
        "tv_ratio": float(shape_stats["tv_ratio"]),
        **collapse_diag,
        "evidence_tier": _consistency_evidence_tier(
            {
                "consistency_score": float(consistency_score),
                "consistency_pairs": int(consistency_pairs),
                "consistency_total_pairs": int(consistency_total_pairs),
            }
        )[1],
    }


def _finite_design_rows(Phi: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if int(Phi.ndim) != 2:
        raise ValueError("Phi must be a matrix")
    mask = torch.isfinite(y.reshape(-1)) & torch.isfinite(Phi).all(dim=1)
    if int(mask.sum()) <= 0:
        return Phi[:0], y[:0]
    return Phi[mask], y[mask]


def _family_basis_asts(coord_ast, family: str) -> list[Any | None]:
    family_norm = str(family).strip().lower()
    if family_norm == "poly2":
        return [None, coord_ast, Pow(coord_ast, 2.0)]
    if family_norm == "poly3":
        return [None, coord_ast, Pow(coord_ast, 2.0), Pow(coord_ast, 3.0)]
    if family_norm == "exp":
        return [None, Exp(coord_ast)]
    if family_norm == "sin":
        return [None, Sin(coord_ast)]
    if family_norm == "cos":
        return [None, Cos(coord_ast)]
    if family_norm == "reciprocal":
        return [None, Pow(coord_ast, -1.0)]
    if family_norm == "inv_square":
        return [None, Pow(coord_ast, -2.0)]
    if family_norm == "log":
        return [None, Log(coord_ast)]
    raise ValueError(f"Unsupported cheap factorized family: {family!r}")


def _family_design_from_coord_values(z: torch.Tensor, family: str) -> torch.Tensor:
    z1 = z.reshape(-1)
    family_norm = str(family).strip().lower()
    if family_norm == "poly2":
        cols = [torch.ones_like(z1), z1, torch.pow(z1, 2.0)]
    elif family_norm == "poly3":
        cols = [torch.ones_like(z1), z1, torch.pow(z1, 2.0), torch.pow(z1, 3.0)]
    elif family_norm == "exp":
        cols = [torch.ones_like(z1), torch.exp(z1)]
    elif family_norm == "sin":
        cols = [torch.ones_like(z1), torch.sin(z1)]
    elif family_norm == "cos":
        cols = [torch.ones_like(z1), torch.cos(z1)]
    elif family_norm == "reciprocal":
        cols = [torch.ones_like(z1), torch.pow(z1, -1.0)]
    elif family_norm == "inv_square":
        cols = [torch.ones_like(z1), torch.pow(z1, -2.0)]
    elif family_norm == "log":
        cols = [torch.ones_like(z1), torch.log(z1)]
    else:
        raise ValueError(f"Unsupported cheap factorized family: {family!r}")
    return torch.stack(cols, dim=1)


def _linear_family_support_subsets(n_basis: int) -> list[tuple[int, ...]]:
    n = int(n_basis)
    if n <= 1 or n > 8:
        return []
    full = tuple(range(n))
    supports: list[tuple[int, ...]] = []
    for mask in range(1, (1 << n)):
        support = tuple(i for i in range(n) if bool(mask & (1 << i)))
        if support == full:
            continue
        supports.append(support)

    def _key(support: tuple[int, ...]) -> tuple[int, int, tuple[int, ...]]:
        const_only = int(len(support) == 1 and support[0] == 0)
        return (len(support), const_only, support)

    return sorted(supports, key=_key)


def _lane_allows_linear_projection_variants(lane: str) -> bool:
    lane_norm = str(lane or "")
    return lane_norm == "x_coeff_on_u"


def _nearest_simple_coeff_snap(value: float) -> float | None:
    try:
        val = float(value)
    except Exception:
        return None
    if not math.isfinite(val):
        return None
    targets = (0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1.0 / 3.0, -1.0 / 3.0, 3.0, -3.0)
    best: tuple[float, float] | None = None
    for target in targets:
        tol = 0.05 if target == 0.0 else max(0.05, 0.05 * abs(float(target)))
        delta = abs(val - float(target))
        if delta > tol:
            continue
        score = delta / max(1.0, abs(float(target)))
        if best is None or score < best[0]:
            best = (score, float(target))
    if best is None:
        return None
    target = float(best[1])
    if abs(val - target) <= 1.0e-12 * max(1.0, abs(val), abs(target)):
        return None
    return target


def _snapped_coeff_vectors(coeffs: Sequence[float]) -> list[tuple[list[float], list[dict[str, float]]]]:
    base = [float(c) for c in list(coeffs)]
    snaps: list[tuple[int, float, float]] = []
    for idx, coeff in enumerate(base):
        target = _nearest_simple_coeff_snap(float(coeff))
        if target is not None:
            snaps.append((int(idx), float(coeff), float(target)))
    if not snaps:
        return []

    out: list[tuple[list[float], list[dict[str, float]]]] = []
    seen: set[tuple[float, ...]] = set()

    def _add(changes: Sequence[tuple[int, float, float]]) -> None:
        vec = list(base)
        report: list[dict[str, float]] = []
        for idx, old, new in list(changes):
            vec[int(idx)] = float(new)
            report.append({"index": int(idx), "from": float(old), "to": float(new)})
        key = tuple(round(float(v), 14) for v in vec)
        if key in seen:
            return
        seen.add(key)
        out.append((vec, report))

    _add(snaps)
    if len(snaps) > 1:
        for snap in snaps:
            _add([snap])
    return out[:4]


def _annotate_projection_row(
    row: dict[str, Any] | None,
    *,
    kind: str,
    support: Sequence[int],
    coeffs: Sequence[float],
    full_basis_size: int,
    snap_report: Sequence[dict[str, float]] = (),
) -> dict[str, Any] | None:
    if row is None:
        return None
    support_i = [int(i) for i in list(support)]
    coeffs_f = [float(c) for c in list(coeffs)]
    row["projection_kind"] = str(kind)
    row["projection_support"] = support_i
    row["projection_coeffs"] = coeffs_f
    row["projection_full_basis_size"] = int(full_basis_size)
    row["projection_signature"] = (
        ""
        if str(kind) == "full"
        else (
            f"{kind}:"
            f"{','.join(str(i) for i in support_i)}:"
            f"{','.join(f'{c:.12g}' for c in coeffs_f)}"
        )
    )
    if snap_report:
        report = [dict(item) for item in list(snap_report)]
        row["projection_snap_report"] = report
        row["projection_snap_cost"] = float(
            sum(abs(float(item["from"]) - float(item["to"])) / max(1.0, abs(float(item["to"]))) for item in report)
        )
    else:
        row["projection_snap_cost"] = 0.0
    return row


def _make_x_family_projected_row(
    *,
    lane: str,
    family: str,
    base_mode: str,
    groups: Sequence[DEFeatureGroup],
    order: int,
    x_axis: int,
    base_ast,
    resid_fit_parts: Sequence[torch.Tensor],
    resid_probe_parts: Sequence[torch.Tensor],
    carrier_ast,
    coord_ast,
    rel_eps: float,
    min_ratio_rows: int,
    basis_asts: Sequence[Any | None],
    basis_asts_local: Sequence[Any | None],
    support: Sequence[int],
    coeffs: Sequence[float],
    coeff_expr: str,
    projection_kind: str,
    snap_report: Sequence[dict[str, float]] = (),
) -> dict[str, Any] | None:
    support_i = [int(i) for i in list(support)]
    coeff_ast = _sum_linear_terms_ast([basis_asts[i] for i in support_i], list(coeffs))
    coeff_local_ast = _sum_linear_terms_ast([basis_asts_local[i] for i in support_i], list(coeffs))
    if coeff_ast is None or coeff_local_ast is None:
        return None
    ratio_probe_mse = _masked_original_scale_probe_mse(
        groups=groups,
        resid_probe_parts=resid_probe_parts,
        carrier_ast=carrier_ast,
        coeff_ast=coeff_ast,
        x_axis=int(x_axis),
        rel_eps=float(rel_eps),
        robust=True,
    )
    fit_target_mse = _pooled_target_mse_from_local_ast(
        groups=groups,
        resid_parts=resid_fit_parts,
        carrier_ast=carrier_ast,
        coord_ast=coord_ast,
        coeff_local_ast=coeff_local_ast,
        split="fit",
        x_axis=int(x_axis),
        rel_eps=float(rel_eps),
        min_rows=int(min_ratio_rows),
        robust=False,
    )
    probe_target_mse = _pooled_target_mse_from_local_ast(
        groups=groups,
        resid_parts=resid_probe_parts,
        carrier_ast=carrier_ast,
        coord_ast=coord_ast,
        coeff_local_ast=coeff_local_ast,
        split="probe",
        x_axis=int(x_axis),
        rel_eps=float(rel_eps),
        min_rows=int(min_ratio_rows),
        robust=True,
    )
    row = _make_candidate_row(
        lane=str(lane),
        family=str(family),
        base_mode=str(base_mode),
        groups=groups,
        order=int(order),
        x_axis=int(x_axis),
        base_ast=base_ast,
        carrier_ast=carrier_ast,
        coord_ast=coord_ast,
        coeff_ast=coeff_ast,
        coeff_local_ast=coeff_local_ast,
        coeff_expr=str(coeff_expr),
        mapping={
            "projection_kind": str(projection_kind),
            "projection_support": support_i,
            "projection_coeffs": [float(c) for c in list(coeffs)],
        },
        size=int(sum(abs(float(c)) > 1.0e-14 for c in list(coeffs))),
        ratio_probe_mse=float(ratio_probe_mse),
        fit_target_mse=float(fit_target_mse),
        probe_target_mse=float(probe_target_mse),
        resid_probe_parts=resid_probe_parts,
        rel_eps=float(rel_eps),
    )
    return _annotate_projection_row(
        row,
        kind=str(projection_kind),
        support=support_i,
        coeffs=coeffs,
        full_basis_size=len(list(basis_asts)),
        snap_report=snap_report,
    )


def _make_general_family_projected_row(
    *,
    lane: str,
    family: str,
    base_mode: str,
    groups: Sequence[DEFeatureGroup],
    order: int,
    x_axis: int,
    base_ast,
    resid_probe_parts: Sequence[torch.Tensor],
    carrier_ast,
    coord_ast,
    rel_eps: float,
    basis_asts: Sequence[Any | None],
    Phi_fit_cat: torch.Tensor,
    y_fit_cat: torch.Tensor,
    Phi_probe_cat: torch.Tensor,
    y_probe_cat: torch.Tensor,
    support: Sequence[int],
    coeffs: Sequence[float],
    coeff_expr: str,
    projection_kind: str,
    snap_report: Sequence[dict[str, float]] = (),
) -> dict[str, Any] | None:
    support_i = [int(i) for i in list(support)]
    coeff_ast = _sum_linear_terms_ast([basis_asts[i] for i in support_i], list(coeffs))
    if coeff_ast is None:
        return None
    coeffs_fit = torch.as_tensor(list(coeffs), dtype=Phi_fit_cat.dtype, device=Phi_fit_cat.device)
    coeffs_probe = torch.as_tensor(list(coeffs), dtype=Phi_probe_cat.dtype, device=Phi_probe_cat.device)
    fit_pred = Phi_fit_cat[:, support_i] @ coeffs_fit
    probe_pred = Phi_probe_cat[:, support_i] @ coeffs_probe
    fit_target_mse = _trimmed_mean_sq(fit_pred - y_fit_cat.reshape(fit_pred.shape))
    probe_target_mse = _trimmed_mean_sq(probe_pred - y_probe_cat.reshape(probe_pred.shape))
    row = _make_candidate_row(
        lane=str(lane),
        family=str(family),
        base_mode=str(base_mode),
        groups=groups,
        order=int(order),
        x_axis=int(x_axis),
        base_ast=base_ast,
        carrier_ast=carrier_ast,
        coord_ast=coord_ast,
        coeff_ast=coeff_ast,
        coeff_expr=str(coeff_expr),
        mapping={
            "projection_kind": str(projection_kind),
            "projection_support": support_i,
            "projection_coeffs": [float(c) for c in list(coeffs)],
        },
        size=int(sum(abs(float(c)) > 1.0e-14 for c in list(coeffs))),
        ratio_probe_mse=float(probe_target_mse),
        fit_target_mse=float(fit_target_mse),
        probe_target_mse=float(probe_target_mse),
        resid_probe_parts=resid_probe_parts,
        rel_eps=float(rel_eps),
    )
    return _annotate_projection_row(
        row,
        kind=str(projection_kind),
        support=support_i,
        coeffs=coeffs,
        full_basis_size=len(list(basis_asts)),
        snap_report=snap_report,
    )


def _distill_x_lane_explorer_candidate(
    *,
    explorer_candidate: dict[str, Any],
    groups: Sequence[DEFeatureGroup],
    order: int,
    x_axis: int,
    base_ast,
    base_mode: str,
    resid_fit_parts: Sequence[torch.Tensor],
    resid_probe_parts: Sequence[torch.Tensor],
    carrier_ast,
    coord_ast,
    rel_eps: float,
    min_ratio_rows: int,
    family_names: Sequence[str] = ("log", "reciprocal", "inv_square", "poly2"),
) -> list[dict[str, Any]]:
    coeff_ast_teacher = explorer_candidate.get("coeff_local_ast", None)
    if coeff_ast_teacher is None:
        coeff_ast_teacher = explorer_candidate.get("coeff_ast", None)
    if coeff_ast_teacher is None:
        return []

    pooled_fit = _pooled_same_coord_coeff_target(
        groups=groups,
        resid_parts=resid_fit_parts,
        carrier_ast=carrier_ast,
        coord_ast=coord_ast,
        split="fit",
        x_axis=int(x_axis),
        rel_eps=float(rel_eps),
        min_rows=int(min_ratio_rows),
    )
    pooled_probe = _pooled_same_coord_coeff_target(
        groups=groups,
        resid_parts=resid_probe_parts,
        carrier_ast=carrier_ast,
        coord_ast=coord_ast,
        split="probe",
        x_axis=int(x_axis),
        rel_eps=float(rel_eps),
        min_rows=int(min_ratio_rows),
    )
    if pooled_fit is None or pooled_probe is None:
        return []

    z_fit, _y_fit_unused, w_fit = pooled_fit
    z_probe, _y_probe_unused, w_probe = pooled_probe
    try:
        teacher_fit = _eval_univariate_ast_on_values(coeff_ast_teacher, z_fit.reshape(-1)).reshape(-1, 1)
        teacher_probe = _eval_univariate_ast_on_values(coeff_ast_teacher, z_probe.reshape(-1)).reshape(-1, 1)
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    for family in list(family_names):
        basis_asts = _family_basis_asts(coord_ast, family)
        basis_asts_local = _family_basis_asts(Var(0), family)
        try:
            Phi_fit = _family_design_from_coord_values(z_fit, family)
            Phi_probe = _family_design_from_coord_values(z_probe, family)
        except Exception:
            continue

        Phi_fit_valid, y_fit_valid = _finite_design_rows(Phi_fit, teacher_fit)
        Phi_probe_valid, y_probe_valid = _finite_design_rows(Phi_probe, teacher_probe)
        fit_mask = torch.isfinite(teacher_fit.reshape(-1)) & torch.isfinite(Phi_fit).all(dim=1)
        probe_mask = torch.isfinite(teacher_probe.reshape(-1)) & torch.isfinite(Phi_probe).all(dim=1)
        w_fit_valid = w_fit.reshape(-1, 1)[fit_mask]
        w_probe_valid = w_probe.reshape(-1, 1)[probe_mask]
        if int(Phi_fit_valid.shape[0]) < int(min_ratio_rows) or int(Phi_probe_valid.shape[0]) < int(min_ratio_rows):
            continue

        fit_scale = torch.sqrt(torch.clamp(w_fit_valid, min=1.0e-12))
        probe_scale = torch.sqrt(torch.clamp(w_probe_valid, min=1.0e-12))
        coeffs = ridge_lstsq(Phi_fit_valid * fit_scale, y_fit_valid * fit_scale, ridge=0.0).detach().cpu().reshape(-1)
        coeff_ast = _sum_linear_terms_ast(basis_asts, coeffs.tolist())
        coeff_local_ast = _sum_linear_terms_ast(basis_asts_local, coeffs.tolist())
        if coeff_ast is None:
            continue

        distill_probe_pred = (
            Phi_probe_valid @ coeffs.to(dtype=Phi_probe_valid.dtype, device=Phi_probe_valid.device)
        ).reshape(-1, 1)
        distill_probe_mse = float(
            torch.mean(((distill_probe_pred - y_probe_valid) * probe_scale).square()).detach().cpu().item()
            / max(float(torch.mean(probe_scale.square()).detach().cpu().item()), 1.0e-12)
        )
        ratio_probe_mse = _masked_original_scale_probe_mse(
            groups=groups,
            resid_probe_parts=resid_probe_parts,
            carrier_ast=carrier_ast,
            coeff_ast=coeff_ast,
            x_axis=int(x_axis),
            rel_eps=float(rel_eps),
            robust=True,
        )
        fit_target_mse = _pooled_target_mse_from_local_ast(
            groups=groups,
            resid_parts=resid_fit_parts,
            carrier_ast=carrier_ast,
            coord_ast=coord_ast,
            coeff_local_ast=coeff_local_ast,
            split="fit",
            x_axis=int(x_axis),
            rel_eps=float(rel_eps),
            min_rows=int(min_ratio_rows),
            robust=False,
        )
        probe_target_mse = _pooled_target_mse_from_local_ast(
            groups=groups,
            resid_parts=resid_probe_parts,
            carrier_ast=carrier_ast,
            coord_ast=coord_ast,
            coeff_local_ast=coeff_local_ast,
            split="probe",
            x_axis=int(x_axis),
            rel_eps=float(rel_eps),
            min_rows=int(min_ratio_rows),
            robust=True,
        )
        cand = _make_candidate_row(
            lane="x_coeff_on_u",
            family=str(family),
            base_mode=str(base_mode),
            groups=groups,
            order=int(order),
            x_axis=int(x_axis),
            base_ast=base_ast,
            carrier_ast=carrier_ast,
            coord_ast=coord_ast,
            coeff_ast=coeff_ast,
            coeff_local_ast=coeff_local_ast,
            coeff_expr=f"{family}[distilled]",
            mapping={"_distilled_from": "explorer", "_distill_probe_mse": float(distill_probe_mse)},
            size=int(sum(abs(float(c)) > 1.0e-14 for c in coeffs.tolist())),
            ratio_probe_mse=float(ratio_probe_mse),
            fit_target_mse=float(fit_target_mse),
            probe_target_mse=float(probe_target_mse),
            resid_probe_parts=resid_probe_parts,
            rel_eps=float(rel_eps),
        )
        if cand is not None:
            out.append(cand)
    return out


def _distill_state_lane_explorer_candidate(
    *,
    explorer_candidate: dict[str, Any],
    groups: Sequence[DEFeatureGroup],
    order: int,
    x_axis: int,
    base_ast,
    base_mode: str,
    resid_fit_parts: Sequence[torch.Tensor],
    resid_probe_parts: Sequence[torch.Tensor],
    carrier_ast,
    coord_asts: Sequence[Any],
    rel_eps: float,
    min_ratio_rows: int,
    family_names: Sequence[str] = ("poly2", "poly3", "sin", "cos", "exp", "reciprocal", "log"),
) -> list[dict[str, Any]]:
    teacher_ast = explorer_candidate.get("coeff_ast", None)
    if teacher_ast is None:
        return []

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    group_weights = _normalized_group_quality_weights(groups)

    for coord_ast in list(coord_asts):
        coord_key = repr(coord_ast)
        for family in list(family_names):
            basis_asts = _family_basis_asts(coord_ast, family)
            basis_asts_local = _family_basis_asts(Var(0), family)
            Phi_fit_parts: list[torch.Tensor] = []
            y_fit_parts: list[torch.Tensor] = []
            Phi_probe_parts: list[torch.Tensor] = []
            y_probe_parts: list[torch.Tensor] = []

            for group, resid_fit, resid_probe, group_weight in zip(groups, resid_fit_parts, resid_probe_parts, group_weights):
                phi_fit = _eval_ast_on_features(
                    carrier_ast,
                    features=group.features,
                    split="fit",
                    x_axis=int(x_axis),
                ).reshape(-1)
                phi_probe = _eval_ast_on_features(
                    carrier_ast,
                    features=group.features,
                    split="probe",
                    x_axis=int(x_axis),
                ).reshape(-1)
                _, mask_fit = _safe_ratio_target(resid_fit, phi_fit, rel_eps=float(rel_eps))
                _, mask_probe = _safe_ratio_target(resid_probe, phi_probe, rel_eps=float(rel_eps))
                if mask_fit is None or mask_probe is None:
                    continue

                z_fit = _eval_ast_on_features(
                    coord_ast,
                    features=group.features,
                    split="fit",
                    x_axis=int(x_axis),
                ).reshape(-1, 1)
                z_probe = _eval_ast_on_features(
                    coord_ast,
                    features=group.features,
                    split="probe",
                    x_axis=int(x_axis),
                ).reshape(-1, 1)
                teacher_fit = _eval_ast_on_features(
                    teacher_ast,
                    features=group.features,
                    split="fit",
                    x_axis=int(x_axis),
                ).reshape(-1, 1)
                teacher_probe = _eval_ast_on_features(
                    teacher_ast,
                    features=group.features,
                    split="probe",
                    x_axis=int(x_axis),
                ).reshape(-1, 1)

                z_fit_valid, teacher_fit_valid = _finite_xy_rows(z_fit[mask_fit], teacher_fit[mask_fit])
                z_probe_valid, teacher_probe_valid = _finite_xy_rows(z_probe[mask_probe], teacher_probe[mask_probe])
                if int(z_fit_valid.shape[0]) < 3 or int(z_probe_valid.shape[0]) < 3:
                    continue

                try:
                    Phi_fit = _family_design_from_coord_values(z_fit_valid, family)
                    Phi_probe = _family_design_from_coord_values(z_probe_valid, family)
                except Exception:
                    continue

                Phi_fit_valid, y_fit_valid = _finite_design_rows(Phi_fit, teacher_fit_valid)
                Phi_probe_valid, y_probe_valid = _finite_design_rows(Phi_probe, teacher_probe_valid)
                if int(Phi_fit_valid.shape[0]) <= 0 or int(Phi_probe_valid.shape[0]) <= 0:
                    continue

                scale = math.sqrt(max(float(group_weight), 1.0e-12))
                Phi_fit_parts.append(Phi_fit_valid * scale)
                y_fit_parts.append(y_fit_valid * scale)
                Phi_probe_parts.append(Phi_probe_valid * scale)
                y_probe_parts.append(y_probe_valid * scale)

            if not Phi_fit_parts or not Phi_probe_parts:
                continue

            Phi_fit_cat = torch.cat(Phi_fit_parts, dim=0)
            y_fit_cat = torch.cat(y_fit_parts, dim=0)
            Phi_probe_cat = torch.cat(Phi_probe_parts, dim=0)
            y_probe_cat = torch.cat(y_probe_parts, dim=0)
            if int(Phi_fit_cat.shape[0]) < int(min_ratio_rows) or int(Phi_probe_cat.shape[0]) < int(min_ratio_rows):
                continue

            coeffs = ridge_lstsq(Phi_fit_cat, y_fit_cat, ridge=0.0).detach().cpu().reshape(-1)
            coeff_ast = _sum_linear_terms_ast(basis_asts, coeffs.tolist())
            coeff_local_ast = _sum_linear_terms_ast(basis_asts_local, coeffs.tolist())
            if coeff_ast is None:
                continue

            distill_probe_pred = (
                Phi_probe_cat @ coeffs.to(dtype=Phi_probe_cat.dtype, device=Phi_probe_cat.device)
            ).reshape(-1, 1)
            distill_probe_mse = float(torch.mean((distill_probe_pred - y_probe_cat) ** 2).detach().cpu().item())
            fit_target_mse = _pooled_target_mse_from_local_ast(
                groups=groups,
                resid_parts=resid_fit_parts,
                carrier_ast=carrier_ast,
                coord_ast=coord_ast,
                coeff_local_ast=coeff_local_ast,
                split="fit",
                x_axis=int(x_axis),
                rel_eps=float(rel_eps),
                min_rows=int(min_ratio_rows),
                robust=False,
            )
            probe_target_mse = _pooled_target_mse_from_local_ast(
                groups=groups,
                resid_parts=resid_probe_parts,
                carrier_ast=carrier_ast,
                coord_ast=coord_ast,
                coeff_local_ast=coeff_local_ast,
                split="probe",
                x_axis=int(x_axis),
                rel_eps=float(rel_eps),
                min_rows=int(min_ratio_rows),
                robust=True,
            )
            cand = _make_candidate_row(
                lane="state_nonlinearity",
                family=str(family),
                base_mode=str(base_mode),
                groups=groups,
                order=int(order),
                x_axis=int(x_axis),
                base_ast=base_ast,
                carrier_ast=carrier_ast,
                coord_ast=coord_ast,
                coeff_ast=coeff_ast,
                coeff_local_ast=coeff_local_ast,
                coeff_expr=f"{family}[distilled]",
                mapping={"_distilled_from": "explorer", "_distill_probe_mse": float(distill_probe_mse)},
                size=int(sum(abs(float(c)) > 1.0e-14 for c in coeffs.tolist())),
                ratio_probe_mse=float(probe_target_mse),
                fit_target_mse=float(fit_target_mse),
                probe_target_mse=float(probe_target_mse),
                resid_probe_parts=resid_probe_parts,
                rel_eps=float(rel_eps),
            )
            if cand is None:
                continue
            key = (str(family), coord_key, repr(cand.get("coeff_ast", None)))
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)
    return out


def _fit_family_lane_candidate(
    *,
    lane: str,
    family: str,
    base_mode: str,
    groups: Sequence[DEFeatureGroup],
    order: int,
    x_axis: int,
    base_ast,
    resid_fit_parts: Sequence[torch.Tensor],
    resid_probe_parts: Sequence[torch.Tensor],
    carrier_ast,
    coord_ast,
    rel_eps: float,
    min_ratio_rows: int,
    return_variants: bool = False,
):
    basis_asts = _family_basis_asts(coord_ast, family)
    basis_asts_local = _family_basis_asts(Var(0), family)
    lane_norm = str(lane or "")

    if lane_norm == "x_coeff_on_u":
        pooled_fit = _pooled_same_coord_coeff_target(
            groups=groups,
            resid_parts=resid_fit_parts,
            carrier_ast=carrier_ast,
            coord_ast=coord_ast,
            split="fit",
            x_axis=int(x_axis),
            rel_eps=float(rel_eps),
            min_rows=int(min_ratio_rows),
        )
        pooled_probe = _pooled_same_coord_coeff_target(
            groups=groups,
            resid_parts=resid_probe_parts,
            carrier_ast=carrier_ast,
            coord_ast=coord_ast,
            split="probe",
            x_axis=int(x_axis),
            rel_eps=float(rel_eps),
            min_rows=int(min_ratio_rows),
        )
        if pooled_fit is None or pooled_probe is None:
            return None
        coeff_ast, coeff_local_ast, _probe_mse, coeffs = _fit_original_scale_family_basis(
            basis_asts=basis_asts,
            basis_asts_local=basis_asts_local,
            groups=groups,
            x_axis=int(x_axis),
            resid_fit_parts=resid_fit_parts,
            resid_probe_parts=resid_probe_parts,
            carrier_ast=carrier_ast,
            rel_eps=float(rel_eps),
            min_ratio_rows=int(min_ratio_rows),
        )
        if coeff_ast is None:
            return None

        ratio_probe_mse = _masked_original_scale_probe_mse(
            groups=groups,
            resid_probe_parts=resid_probe_parts,
            carrier_ast=carrier_ast,
            coeff_ast=coeff_ast,
            x_axis=int(x_axis),
            rel_eps=float(rel_eps),
            robust=True,
        )
        fit_target_mse = _pooled_target_mse_from_local_ast(
            groups=groups,
            resid_parts=resid_fit_parts,
            carrier_ast=carrier_ast,
            coord_ast=coord_ast,
            coeff_local_ast=coeff_local_ast,
            split="fit",
            x_axis=int(x_axis),
            rel_eps=float(rel_eps),
            min_rows=int(min_ratio_rows),
            robust=False,
        )
        probe_target_mse = _pooled_target_mse_from_local_ast(
            groups=groups,
            resid_parts=resid_probe_parts,
            carrier_ast=carrier_ast,
            coord_ast=coord_ast,
            coeff_local_ast=coeff_local_ast,
            split="probe",
            x_axis=int(x_axis),
            rel_eps=float(rel_eps),
            min_rows=int(min_ratio_rows),
            robust=True,
        )
        parent = _make_candidate_row(
            lane=str(lane),
            family=str(family),
            base_mode=str(base_mode),
            groups=groups,
            order=int(order),
            x_axis=int(x_axis),
            base_ast=base_ast,
            carrier_ast=carrier_ast,
            coord_ast=coord_ast,
            coeff_ast=coeff_ast,
            coeff_local_ast=coeff_local_ast,
            coeff_expr=str(family),
            mapping={},
            size=int(len(list(basis_asts))),
            ratio_probe_mse=float(ratio_probe_mse),
            fit_target_mse=float(fit_target_mse),
            probe_target_mse=float(probe_target_mse),
            resid_probe_parts=resid_probe_parts,
            rel_eps=float(rel_eps),
        )
        parent = _annotate_projection_row(
            parent,
            kind="full",
            support=range(len(list(basis_asts))),
            coeffs=coeffs,
            full_basis_size=len(list(basis_asts)),
        )
        if not bool(return_variants) or not _lane_allows_linear_projection_variants(str(lane)):
            return parent

        rows: list[dict[str, Any]] = [] if parent is None else [parent]
        seen = {_candidate_identity_key(parent)} if parent is not None else set()
        for support in _linear_family_support_subsets(len(list(basis_asts))):
            subset_basis = [basis_asts[i] for i in support]
            subset_basis_local = [basis_asts_local[i] for i in support]
            sub_ast, _sub_local_ast, _sub_probe_mse, sub_coeffs = _fit_original_scale_family_basis(
                basis_asts=subset_basis,
                basis_asts_local=subset_basis_local,
                groups=groups,
                x_axis=int(x_axis),
                resid_fit_parts=resid_fit_parts,
                resid_probe_parts=resid_probe_parts,
                carrier_ast=carrier_ast,
                rel_eps=float(rel_eps),
                min_ratio_rows=int(min_ratio_rows),
            )
            if sub_ast is None or not sub_coeffs:
                continue
            sub_row = _make_x_family_projected_row(
                lane=str(lane),
                family=str(family),
                base_mode=str(base_mode),
                groups=groups,
                order=int(order),
                x_axis=int(x_axis),
                base_ast=base_ast,
                resid_fit_parts=resid_fit_parts,
                resid_probe_parts=resid_probe_parts,
                carrier_ast=carrier_ast,
                coord_ast=coord_ast,
                rel_eps=float(rel_eps),
                min_ratio_rows=int(min_ratio_rows),
                basis_asts=basis_asts,
                basis_asts_local=basis_asts_local,
                support=support,
                coeffs=sub_coeffs,
                coeff_expr=f"{family}[support={','.join(str(i) for i in support)}]",
                projection_kind="support",
            )
            if sub_row is not None and _candidate_identity_key(sub_row) not in seen:
                seen.add(_candidate_identity_key(sub_row))
                rows.append(sub_row)
            for snap_coeffs, snap_report in _snapped_coeff_vectors(sub_coeffs):
                snap_row = _make_x_family_projected_row(
                    lane=str(lane),
                    family=str(family),
                    base_mode=str(base_mode),
                    groups=groups,
                    order=int(order),
                    x_axis=int(x_axis),
                    base_ast=base_ast,
                    resid_fit_parts=resid_fit_parts,
                    resid_probe_parts=resid_probe_parts,
                    carrier_ast=carrier_ast,
                    coord_ast=coord_ast,
                    rel_eps=float(rel_eps),
                    min_ratio_rows=int(min_ratio_rows),
                    basis_asts=basis_asts,
                    basis_asts_local=basis_asts_local,
                    support=support,
                    coeffs=snap_coeffs,
                    coeff_expr=f"{family}[snap={','.join(str(i) for i in support)}]",
                    projection_kind="snap",
                    snap_report=snap_report,
                )
                if snap_row is not None and _candidate_identity_key(snap_row) not in seen:
                    seen.add(_candidate_identity_key(snap_row))
                    rows.append(snap_row)
        return rows

    Phi_fit_parts: list[torch.Tensor] = []
    y_fit_parts: list[torch.Tensor] = []
    Phi_probe_parts: list[torch.Tensor] = []
    y_probe_parts: list[torch.Tensor] = []
    group_weights = _normalized_group_quality_weights(groups)
    for group, resid_fit, resid_probe, group_weight in zip(groups, resid_fit_parts, resid_probe_parts, group_weights):
        phi_fit = _eval_ast_on_features(carrier_ast, features=group.features, split="fit", x_axis=int(x_axis)).reshape(-1)
        phi_probe = _eval_ast_on_features(carrier_ast, features=group.features, split="probe", x_axis=int(x_axis)).reshape(-1)
        ratio_fit, mask_fit = _safe_ratio_target(resid_fit, phi_fit, rel_eps=float(rel_eps))
        ratio_probe, mask_probe = _safe_ratio_target(resid_probe, phi_probe, rel_eps=float(rel_eps))
        if ratio_fit is None or ratio_probe is None:
            continue

        cols_fit = []
        cols_probe = []
        for basis_ast in basis_asts:
            if basis_ast is None:
                cols_fit.append(torch.ones_like(resid_fit))
                cols_probe.append(torch.ones_like(resid_probe))
            else:
                cols_fit.append(
                    _eval_ast_on_features(basis_ast, features=group.features, split="fit", x_axis=int(x_axis)).reshape(-1)
                )
                cols_probe.append(
                    _eval_ast_on_features(basis_ast, features=group.features, split="probe", x_axis=int(x_axis)).reshape(-1)
                )
        Phi_fit_full = torch.stack(cols_fit, dim=1)
        Phi_probe_full = torch.stack(cols_probe, dim=1)

        if lane_norm == "x_coeff_on_u":
            Phi_fit = (phi_fit.unsqueeze(1) * Phi_fit_full)[mask_fit]
            Phi_probe = (phi_probe.unsqueeze(1) * Phi_probe_full)[mask_probe]
            y_fit = (-resid_fit[mask_fit]).reshape(-1, 1)
            y_probe = (-resid_probe[mask_probe]).reshape(-1, 1)
            Phi_fit_valid, y_fit_valid = _finite_design_rows(Phi_fit, y_fit)
            Phi_probe_valid, y_probe_valid = _finite_design_rows(Phi_probe, y_probe)
        else:
            Phi_fit = Phi_fit_full[mask_fit]
            Phi_probe = Phi_probe_full[mask_probe]
            Phi_fit_valid, y_fit_valid = _finite_design_rows(Phi_fit, ratio_fit)
            Phi_probe_valid, y_probe_valid = _finite_design_rows(Phi_probe, ratio_probe)
        if int(Phi_fit_valid.shape[0]) < int(min_ratio_rows) or int(Phi_probe_valid.shape[0]) < int(min_ratio_rows):
            continue

        if lane_norm == "state_nonlinearity":
            scale = math.sqrt(max(float(group_weight), 1.0e-12))
            Phi_fit_parts.append(Phi_fit_valid * scale)
            y_fit_parts.append(y_fit_valid * scale)
            Phi_probe_parts.append(Phi_probe_valid * scale)
            y_probe_parts.append(y_probe_valid * scale)
        else:
            Phi_fit_parts.append(Phi_fit_valid)
            y_fit_parts.append(y_fit_valid)
            Phi_probe_parts.append(Phi_probe_valid)
            y_probe_parts.append(y_probe_valid)

    if not Phi_fit_parts or not Phi_probe_parts:
        return None

    Phi_fit_cat = torch.cat(Phi_fit_parts, dim=0)
    y_fit_cat = torch.cat(y_fit_parts, dim=0)
    Phi_probe_cat = torch.cat(Phi_probe_parts, dim=0)
    y_probe_cat = torch.cat(y_probe_parts, dim=0)
    coeffs = _scale_weighted_trimmed_lstsq(Phi_fit_cat, y_fit_cat, ridge=0.0).detach().cpu().reshape(-1)
    coeff_ast = _sum_linear_terms_ast(basis_asts, coeffs.tolist())
    if coeff_ast is None:
        return None

    coeffs_fit = coeffs.to(dtype=Phi_fit_cat.dtype, device=Phi_fit_cat.device)
    coeffs_probe = coeffs.to(dtype=Phi_probe_cat.dtype, device=Phi_probe_cat.device)
    fit_pred = Phi_fit_cat @ coeffs_fit
    probe_pred = Phi_probe_cat @ coeffs_probe
    fit_target_mse = _trimmed_mean_sq(fit_pred - y_fit_cat.reshape(fit_pred.shape))
    probe_target_mse = _trimmed_mean_sq(probe_pred - y_probe_cat.reshape(probe_pred.shape))
    ratio_probe_mse = float(probe_target_mse)
    parent = _make_candidate_row(
        lane=str(lane),
        family=str(family),
        base_mode=str(base_mode),
        groups=groups,
        order=int(order),
        x_axis=int(x_axis),
        base_ast=base_ast,
        carrier_ast=carrier_ast,
        coord_ast=coord_ast,
        coeff_ast=coeff_ast,
        coeff_expr=str(family),
        mapping={},
        size=int(sum(abs(float(c)) > 1.0e-14 for c in coeffs.tolist())),
        ratio_probe_mse=float(ratio_probe_mse),
        fit_target_mse=float(fit_target_mse),
        probe_target_mse=float(probe_target_mse),
        resid_probe_parts=resid_probe_parts,
        rel_eps=float(rel_eps),
    )
    parent = _annotate_projection_row(
        parent,
        kind="full",
        support=range(len(list(basis_asts))),
        coeffs=coeffs.tolist(),
        full_basis_size=len(list(basis_asts)),
    )
    if not bool(return_variants) or not _lane_allows_linear_projection_variants(str(lane)):
        return parent

    rows: list[dict[str, Any]] = [] if parent is None else [parent]
    seen = {_candidate_identity_key(parent)} if parent is not None else set()
    for support in _linear_family_support_subsets(len(list(basis_asts))):
        support_i = [int(i) for i in support]
        try:
            sub_coeffs = (
                _scale_weighted_trimmed_lstsq(Phi_fit_cat[:, support_i], y_fit_cat, ridge=0.0)
                .detach()
                .cpu()
                .reshape(-1)
            )
        except Exception:
            continue
        sub_coeffs_list = [float(c) for c in sub_coeffs.tolist()]
        sub_row = _make_general_family_projected_row(
            lane=str(lane),
            family=str(family),
            base_mode=str(base_mode),
            groups=groups,
            order=int(order),
            x_axis=int(x_axis),
            base_ast=base_ast,
            resid_probe_parts=resid_probe_parts,
            carrier_ast=carrier_ast,
            coord_ast=coord_ast,
            rel_eps=float(rel_eps),
            basis_asts=basis_asts,
            Phi_fit_cat=Phi_fit_cat,
            y_fit_cat=y_fit_cat,
            Phi_probe_cat=Phi_probe_cat,
            y_probe_cat=y_probe_cat,
            support=support,
            coeffs=sub_coeffs_list,
            coeff_expr=f"{family}[support={','.join(str(i) for i in support)}]",
            projection_kind="support",
        )
        if sub_row is not None and _candidate_identity_key(sub_row) not in seen:
            seen.add(_candidate_identity_key(sub_row))
            rows.append(sub_row)
        for snap_coeffs, snap_report in _snapped_coeff_vectors(sub_coeffs_list):
            snap_row = _make_general_family_projected_row(
                lane=str(lane),
                family=str(family),
                base_mode=str(base_mode),
                groups=groups,
                order=int(order),
                x_axis=int(x_axis),
                base_ast=base_ast,
                resid_probe_parts=resid_probe_parts,
                carrier_ast=carrier_ast,
                coord_ast=coord_ast,
                rel_eps=float(rel_eps),
                basis_asts=basis_asts,
                Phi_fit_cat=Phi_fit_cat,
                y_fit_cat=y_fit_cat,
                Phi_probe_cat=Phi_probe_cat,
                y_probe_cat=y_probe_cat,
                support=support,
                coeffs=snap_coeffs,
                coeff_expr=f"{family}[snap={','.join(str(i) for i in support)}]",
                projection_kind="snap",
                snap_report=snap_report,
            )
            if snap_row is not None and _candidate_identity_key(snap_row) not in seen:
                seen.add(_candidate_identity_key(snap_row))
                rows.append(snap_row)
    return rows


def _build_family_lane_candidates(
    *,
    lane: str,
    base_mode: str,
    groups: Sequence[DEFeatureGroup],
    order: int,
    x_axis: int,
    base_ast,
    resid_fit_parts: Sequence[torch.Tensor],
    resid_probe_parts: Sequence[torch.Tensor],
    carrier_ast,
    coord_asts: Sequence[Any],
    family_names: Sequence[str],
    rel_eps: float,
    min_ratio_rows: int,
    diagnostics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    out: list[dict[str, Any]] = []
    attempts = 0
    for coord_ast in list(coord_asts):
        for family in list(family_names):
            attempts += 1
            cand = _fit_family_lane_candidate(
                lane=str(lane),
                family=str(family),
                base_mode=str(base_mode),
                groups=groups,
                order=int(order),
                x_axis=int(x_axis),
                base_ast=base_ast,
                resid_fit_parts=resid_fit_parts,
                resid_probe_parts=resid_probe_parts,
                carrier_ast=carrier_ast,
                coord_ast=coord_ast,
                rel_eps=float(rel_eps),
                min_ratio_rows=int(min_ratio_rows),
                return_variants=True,
            )
            if isinstance(cand, list):
                out.extend(row for row in cand if isinstance(row, dict))
            elif cand is not None:
                out.append(cand)
    if diagnostics is not None:
        elapsed = float(time.perf_counter() - started)
        _diag_inc(diagnostics, "family_fit_attempts", attempts)
        _diag_inc(diagnostics, "family_candidates", len(out))
        _diag_inc(diagnostics, "family_wall_seconds", elapsed)
        diagnostics.setdefault("family_lane_calls", []).append(
            {
                "lane": str(lane),
                "base_mode": str(base_mode),
                "order": int(order),
                "coord_count": int(len(list(coord_asts))),
                "family_count": int(len(list(family_names))),
                "attempts": int(attempts),
                "candidates": int(len(out)),
                "wall_seconds": elapsed,
            }
        )
    return out


_EXPLORER_REFINE_DIAG_KEYS = {
    "score_calls",
    "base_score_s",
    "fit_poly_calls",
    "fit_poly_degree1_calls",
    "fit_poly_affine_fast_calls",
    "fit_poly_constant_calls",
    "fit_poly_lstsq_calls",
    "fit_poly_stlsq_calls",
    "fit_poly_wall_seconds",
    "fit_poly_s",
    "fit_poly_affine_fast_wall_seconds",
    "fit_poly_lstsq_wall_seconds",
    "fit_poly_stlsq_wall_seconds",
    "run_explorer_wall_s",
    "search_wall_time_elapsed_s",
    "setup_wall_s",
    "pool_eval_wall_s",
    "brute_wall_s",
    "brute_scored",
    "mutation_wall_s",
    "prescore_calls",
    "prescore_promoted",
    "prescore_dropped",
    "full_score_calls",
    "negated_variant_scores",
    "negated_variant_skipped_affine_poly_only",
    "stall_checks",
    "stall_triggered",
    "soft_restarts",
    "plateau_stop_requested",
    "plateau_stop_eval",
    "plateau_stop_soft_restarts",
}

__factorized_de_definitions__ = (
    "_make_candidate_row",
    "_finite_design_rows",
    "_family_basis_asts",
    "_family_design_from_coord_values",
    "_linear_family_support_subsets",
    "_lane_allows_linear_projection_variants",
    "_nearest_simple_coeff_snap",
    "_snapped_coeff_vectors",
    "_annotate_projection_row",
    "_make_x_family_projected_row",
    "_make_general_family_projected_row",
    "_distill_x_lane_explorer_candidate",
    "_distill_state_lane_explorer_candidate",
    "_fit_family_lane_candidate",
    "_build_family_lane_candidates",
)

__factorized_de_constants__ = (
    "_EXPLORER_REFINE_DIAG_KEYS",
)

__factorized_de_late_bindings__ = (

)
