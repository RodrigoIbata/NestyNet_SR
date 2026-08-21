# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

"""Second-order typed lanes and the factorized coefficient rescue entry point."""

import copy
import math
import time
from functools import cmp_to_key
from typing import Any, Mapping, Sequence
import torch
from nestynet_sr.sr_core.ast_simplify import ast_node_count
from nestynet_sr.sr_core.bridges import Add, ConstNode, DU, Mul, U, Var
from nestynet_sr.sr_de.de_search import DESearchResult, DESearchResultMulti

from ._factorized_de_frontend import (
    DEFeatureGroup,
    _diag_inc,
    _feature_group_row_summary,
    _merge_diagnostics,
    _process_memory_report,
)
from ._factorized_de_operator import (
    FactorizedDEBlock,
    FactorizedDERescueConfig,
    FactorizedDEResult,
    _active_first_order_typed_lanes,
    _anchor_ast,
    _anchor_tensor,
    _base_variants,
    _best_probe_rms,
    _build_zero_base_x_lane_diagnostics,
    _candidate_identity_key,
    _canonical_equation,
    _carrier_pool,
    _choose_preferred_zero_lane,
    _compare_candidate_rows,
    _compose_nonanchor_ast,
    _consistency_evidence_tier,
    _coord_pool,
    _dedupe_ast_list,
    _diverse_candidate_shortlist,
    _eval_ast_on_features,
    _material_improvement,
    _multiprocessing_start_method_name,
    _probe_mse_from_residuals,
    _residual_parts_for_base,
    _row_domain_safe,
    _scale_weighted_trimmed_lstsq,
    _select_state_lane_candidates,
    _select_x_lane_candidates,
    _simplify_de_ast,
    _split_tensor,
    _sum_linear_terms_ast,
    _trimmed_mean_sq,
    _trimmed_probe_mse_from_residuals,
)
from ._factorized_de_explorer import (
    _build_explorer_lane_candidates,
    _build_typed_lane_candidates_with_gate,
    _carrier_role,
    _family_first_gate_decision,
    _schedule_explorer_coord_asts_by_collapse,
    _should_run_two_block_shared_coord,
    _typed_explorer_caps_for_order,
    _typed_explorer_caps_from_hp,
)
from ._factorized_de_lanes import (
    _build_family_lane_candidates,
    _family_basis_asts,
)
from ._factorized_de_search import (
    default_physics_rescue_hp,
)

def _second_order_state_coord_asts() -> list[Any]:
    u = U()
    neg_u = Mul(ConstNode(-1.0), U())
    return _dedupe_ast_list([u, Add(ConstNode(1.0), u), neg_u, Add(ConstNode(1.0), neg_u)])


def _second_order_x_coord_asts(x_axis: int, frequency_hints: Sequence[float] = ()) -> list[Any]:
    x = Var(int(x_axis))
    coords: list[Any] = [x, Add(ConstNode(1.0), x)]
    # Periodogram-hinted coordinates: the sin/cos families otherwise only see
    # canonical frequency-1 coords, which are uncorrelated with forcing or
    # parametric drives at any other frequency.
    for omega in frequency_hints:
        try:
            w = float(omega)
        except Exception:
            continue
        if math.isfinite(w) and w > 0.0 and abs(w - 1.0) > 1.0e-6:
            coords.append(Mul(ConstNode(w), Var(int(x_axis))))
    return _dedupe_ast_list(coords)


def _second_order_velocity_coord_asts(x_axis: int) -> list[Any]:
    du = DU(int(x_axis))
    neg_du = Mul(ConstNode(-1.0), du)
    return _dedupe_ast_list([du, Add(ConstNode(1.0), du), neg_du, Add(ConstNode(1.0), neg_du)])


def _typed_lane_frequency_hints(
    groups: Sequence[DEFeatureGroup],
    base_fit_parts: Sequence[torch.Tensor],
    *,
    order: int,
    x_axis: int,
    max_hints: int = 2,
) -> list[float]:
    """Periodogram hints for the x-dependent coefficient lanes.

    Scans the anchored residual y(x) (additive forcing, de301-class) and the
    ratio y/u (parametric drives) per trajectory for dominant
    frequencies, then polishes each candidate against the full linear model

        y ~ a*u + f + (b*cos + c*sin)(w*x)*u + (d*cos + e*sin)(w*x)

    over the pooled rows via Gauss-Newton steps on w. The ratio scan alone is
    biased by zero crossings of u (the resampled signal is gappy), and even a
    0.1% frequency error leaves visible phase drift over a many-period
    record; the model polish uses every row with no division or resampling.
    Hints whose trig terms explain no variance are dropped.
    """
    from nestynet_sr.sr_search.factorized_search.engine.search import (
        _periodogram_frequency_hints,
    )

    xs: list[torch.Tensor] = []
    us: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for group, base_fit in zip(groups, base_fit_parts):
        try:
            anchor_fit, _ = _anchor_tensor(group.features, order=int(order))
            y = anchor_fit.reshape(-1) + base_fit.reshape(-1)
            x = group.features.x_fit[:, int(x_axis)].reshape(-1)
            u = group.features.u_fit.reshape(-1)
        except Exception:
            continue
        if int(x.numel()) != int(y.numel()) or int(x.numel()) != int(u.numel()):
            continue
        xs.append(x)
        us.append(u)
        ys.append(y)

    if not xs:
        return []

    hints: list[float] = []
    for x_g, u_g, y_g in zip(xs, us, ys):
        x_col = x_g.reshape(-1, 1)
        u_floor = 0.1 * float(u_g.abs().median().clamp_min(1.0e-12))
        ratio = y_g / torch.where(u_g.abs() < u_floor, torch.full_like(u_g, float("nan")), u_g)
        for target in (y_g, ratio):
            finite = torch.isfinite(target)
            if int(finite.sum()) < 64:
                continue
            try:
                rows = _periodogram_frequency_hints(
                    x_col[finite],
                    target[finite],
                    max_hints=int(max_hints),
                )
            except Exception:
                continue
            hints.extend(w for _, w in rows)

    deduped: list[float] = []
    for w in hints:
        if all(abs(w - prev) > 0.05 * max(w, prev) for prev in deduped):
            deduped.append(float(w))
    deduped = deduped[: max(0, 2 * int(max_hints))]
    if not deduped:
        return []

    # Model-based Gauss-Newton polish over the pooled rows.
    x_all = torch.cat(xs)
    u_all = torch.cat(us)
    y_all = torch.cat(ys)
    finite_all = torch.isfinite(x_all) & torch.isfinite(u_all) & torch.isfinite(y_all)
    x_all, u_all, y_all = x_all[finite_all], u_all[finite_all], y_all[finite_all]
    if int(x_all.numel()) < 64:
        return deduped
    span = float(x_all.max() - x_all.min())
    ones = torch.ones_like(u_all)
    y_col = y_all.reshape(-1, 1)
    try:
        base_design = torch.stack([u_all, ones], dim=1)
        base_sol = torch.linalg.lstsq(base_design, y_col).solution
        base_mse = float(torch.mean((y_all - (base_design @ base_sol).reshape(-1)) ** 2))
    except Exception:
        return deduped

    polished: list[float] = []
    for omega in deduped:
        w = float(omega)
        mse_w = float("inf")
        # Gauss-Newton converges linearly here (frequency-phase coupling over
        # a long record); iterations are 6-column lstsq calls, so run plenty.
        for _ in range(30):
            cw = torch.cos(w * x_all)
            sw = torch.sin(w * x_all)
            design = torch.stack([u_all, ones, cw * u_all, sw * u_all, cw, sw], dim=1)
            try:
                sol = torch.linalg.lstsq(design, y_col).solution.reshape(-1)
            except Exception:
                break
            resid = y_all - design @ sol
            mse_w = float(torch.mean(resid * resid))
            grad = (
                float(sol[2]) * (-x_all * sw * u_all)
                + float(sol[3]) * (x_all * cw * u_all)
                + float(sol[4]) * (-x_all * sw)
                + float(sol[5]) * (x_all * cw)
            )
            denom = float((grad * grad).sum())
            if not math.isfinite(denom) or denom <= 0.0:
                break
            delta = float((resid * grad).sum()) / denom
            if not math.isfinite(delta) or (span > 0.0 and abs(delta) > 2.0 * math.pi / span):
                break
            w = w + delta
            if abs(delta) < 1.0e-9 * max(1.0, abs(w)):
                break
        if not (math.isfinite(w) and w > 0.0):
            continue
        # Keep only hints whose trig terms actually explain variance.
        if math.isfinite(mse_w) and mse_w < base_mse * (1.0 - 1.0e-3):
            polished.append(float(w))
    return polished


def _second_order_velocity_correction_diagnostic(
    *,
    groups: Sequence[DEFeatureGroup],
    order: int,
    x_axis: int,
    base_fit_parts: Sequence[torch.Tensor],
    base_probe_parts: Sequence[torch.Tensor],
) -> dict[str, Any] | None:
    """Cheap probe for the leading relativistic correction u * du^2."""
    if int(order) != 2 or not groups:
        return None
    if len(base_fit_parts) != len(groups) or len(base_probe_parts) != len(groups):
        return None

    fit_phi_base: list[torch.Tensor] = []
    fit_phi_corr: list[torch.Tensor] = []
    fit_y: list[torch.Tensor] = []
    probe_phi_base: list[torch.Tensor] = []
    probe_phi_corr: list[torch.Tensor] = []
    probe_y: list[torch.Tensor] = []

    for group, base_fit, base_probe in zip(groups, base_fit_parts, base_probe_parts):
        try:
            anchor_fit, anchor_probe = _anchor_tensor(group.features, order=int(order))
            u_fit = _split_tensor(group.features, "fit", "u").reshape(-1)
            du_fit = _split_tensor(group.features, "fit", "du").reshape(-1)
            u_probe = _split_tensor(group.features, "probe", "u").reshape(-1)
            du_probe = _split_tensor(group.features, "probe", "du").reshape(-1)
            y_fit = -(anchor_fit.reshape(-1) + base_fit.reshape(-1))
            y_probe = -(anchor_probe.reshape(-1) + base_probe.reshape(-1))
            phi_base_fit = torch.stack([u_fit, du_fit], dim=1)
            phi_base_probe = torch.stack([u_probe, du_probe], dim=1)
            phi_corr_fit = torch.stack([u_fit, u_fit * du_fit.square(), du_fit], dim=1)
            phi_corr_probe = torch.stack([u_probe, u_probe * du_probe.square(), du_probe], dim=1)
        except Exception:
            continue

        fit_mask = torch.isfinite(y_fit) & torch.isfinite(phi_corr_fit).all(dim=1)
        probe_mask = torch.isfinite(y_probe) & torch.isfinite(phi_corr_probe).all(dim=1)
        if int(fit_mask.sum()) < 8 or int(probe_mask.sum()) < 8:
            continue
        fit_phi_base.append(phi_base_fit[fit_mask])
        fit_phi_corr.append(phi_corr_fit[fit_mask])
        fit_y.append(y_fit[fit_mask])
        probe_phi_base.append(phi_base_probe[probe_mask])
        probe_phi_corr.append(phi_corr_probe[probe_mask])
        probe_y.append(y_probe[probe_mask])

    if not fit_y or not probe_y:
        return None

    Phi_base_fit = torch.cat(fit_phi_base, dim=0)
    Phi_corr_fit = torch.cat(fit_phi_corr, dim=0)
    y_fit_cat = torch.cat(fit_y, dim=0)
    Phi_base_probe = torch.cat(probe_phi_base, dim=0)
    Phi_corr_probe = torch.cat(probe_phi_corr, dim=0)
    y_probe_cat = torch.cat(probe_y, dim=0)
    if int(Phi_corr_fit.shape[0]) < int(Phi_corr_fit.shape[1]) or int(Phi_corr_probe.shape[0]) <= 0:
        return None

    try:
        coeff_base = _scale_weighted_trimmed_lstsq(Phi_base_fit, y_fit_cat, ridge=1.0e-12).reshape(-1)
        coeff_corr = _scale_weighted_trimmed_lstsq(Phi_corr_fit, y_fit_cat, ridge=1.0e-12).reshape(-1)
    except Exception:
        return None
    if int(coeff_base.numel()) != 2 or int(coeff_corr.numel()) != 3:
        return None
    if not bool(torch.isfinite(coeff_base).all().item()) or not bool(torch.isfinite(coeff_corr).all().item()):
        return None

    base_fit_resid = y_fit_cat - (Phi_base_fit @ coeff_base.reshape(-1, 1)).reshape(-1)
    corr_fit_resid = y_fit_cat - (Phi_corr_fit @ coeff_corr.reshape(-1, 1)).reshape(-1)
    base_probe_resid = y_probe_cat - (Phi_base_probe @ coeff_base.reshape(-1, 1)).reshape(-1)
    corr_probe_resid = y_probe_cat - (Phi_corr_probe @ coeff_corr.reshape(-1, 1)).reshape(-1)
    base_fit_mse = _trimmed_mean_sq(base_fit_resid)
    corr_fit_mse = _trimmed_mean_sq(corr_fit_resid)
    base_probe_mse = _trimmed_mean_sq(base_probe_resid)
    corr_probe_mse = _trimmed_mean_sq(corr_probe_resid)
    base_probe_rms = math.sqrt(base_probe_mse) if math.isfinite(base_probe_mse) else float("inf")
    corr_probe_rms = math.sqrt(corr_probe_mse) if math.isfinite(corr_probe_mse) else float("inf")
    probe_ratio = (
        float(corr_probe_rms / base_probe_rms)
        if math.isfinite(base_probe_rms) and base_probe_rms > 0.0 and math.isfinite(corr_probe_rms)
        else None
    )
    return {
        "model": "d2u_base_vs_u_du2",
        "base_terms": ["u", "du"],
        "corrected_terms": ["u", "u*du^2", "du"],
        "fit_rows": int(Phi_corr_fit.shape[0]),
        "probe_rows": int(Phi_corr_probe.shape[0]),
        "base_coeffs": [float(v) for v in coeff_base.detach().cpu().tolist()],
        "corrected_coeffs": [float(v) for v in coeff_corr.detach().cpu().tolist()],
        "u_du2_coeff": float(coeff_corr.detach().cpu().reshape(-1)[1].item()),
        "base_fit_rms": math.sqrt(base_fit_mse) if math.isfinite(base_fit_mse) else None,
        "corrected_fit_rms": math.sqrt(corr_fit_mse) if math.isfinite(corr_fit_mse) else None,
        "base_probe_rms": base_probe_rms if math.isfinite(base_probe_rms) else None,
        "corrected_probe_rms": corr_probe_rms if math.isfinite(corr_probe_rms) else None,
        "probe_rms_ratio": probe_ratio,
        "probe_rms_improvement": (
            float(base_probe_rms - corr_probe_rms)
            if math.isfinite(base_probe_rms) and math.isfinite(corr_probe_rms)
            else None
        ),
        "probe_improved": bool(probe_ratio is not None and probe_ratio < 0.98),
    }


def _second_order_typed_lane_specs(
    *, cfg, x_axis: int, frequency_hints: Sequence[float] = ()
) -> list[dict[str, Any]]:
    state_families = ("poly2", "poly3", "sin", "cos", "exp", "reciprocal", "log")
    x_families = ("reciprocal", "inv_square", "log", "poly2", "sin", "cos")
    velocity_families = ("poly2", "poly3")
    specs: list[dict[str, Any]] = []
    if bool(getattr(cfg, "include_const", True)) and bool(getattr(cfg, "include_u", True)):
        specs.append(
            {
                "lane": "second_order_state_nonlinearity",
                "carrier_ast": ConstNode(1.0),
                "coord_asts": _second_order_state_coord_asts(),
                "family_names": state_families,
            }
        )
    if bool(getattr(cfg, "include_x", True)) and bool(getattr(cfg, "include_u", True)):
        specs.append(
            {
                "lane": "second_order_x_coeff_on_u",
                "carrier_ast": U(),
                "coord_asts": _second_order_x_coord_asts(int(x_axis), frequency_hints),
                "family_names": x_families,
            }
        )
    if bool(getattr(cfg, "include_x", True)) and bool(getattr(cfg, "include_const", True)):
        # Additive forcing f(x): without this lane the periodogram hints can
        # only enter multiplied by u or du, so a pure drive F*cos(w*x) is
        # expressible only as a parametric alias (de301).
        specs.append(
            {
                "lane": "second_order_x_forcing",
                "carrier_ast": ConstNode(1.0),
                "coord_asts": _second_order_x_coord_asts(int(x_axis), frequency_hints),
                "family_names": x_families,
            }
        )
    if bool(getattr(cfg, "include_du", True)) and bool(getattr(cfg, "include_u", True)):
        specs.append(
            {
                "lane": "second_order_velocity_coeff_on_u",
                "carrier_ast": U(),
                "coord_asts": _second_order_velocity_coord_asts(int(x_axis)),
                "family_names": velocity_families,
            }
        )
    if bool(getattr(cfg, "include_du", True)) and bool(getattr(cfg, "include_u", True)):
        specs.append(
            {
                "lane": "second_order_state_damping_on_du",
                "carrier_ast": DU(int(x_axis)),
                "coord_asts": _second_order_state_coord_asts(),
                "family_names": state_families,
            }
        )
    if bool(getattr(cfg, "include_du", True)) and bool(getattr(cfg, "include_x", True)):
        specs.append(
            {
                "lane": "second_order_x_damping_on_du",
                "carrier_ast": DU(int(x_axis)),
                "coord_asts": _second_order_x_coord_asts(int(x_axis), frequency_hints),
                "family_names": x_families,
            }
        )
    return specs


def _trig_frequency_consts(ast, *, x_axis: int) -> list[ConstNode]:
    """ConstNodes c of every sin/cos(c * x_axis) subtree of ``ast``.

    bridges' Mul/Var/Sin/Cos are factory functions, so nodes are matched
    structurally: binary nodes by class name with left/right, atoms by their
    ``kind``/``inputs``/``var_idxs`` fields.
    """
    found: list[ConstNode] = []

    def _children(node):
        if type(node).__name__ in ("MulNode", "AddNode"):
            return (node.left, node.right)
        base = getattr(node, "base", None)
        if base is not None:
            return (base,)
        arg = getattr(node, "arg", None)
        if arg is not None:
            return (arg,)
        return tuple(getattr(node, "inputs", ()) or ())

    def _visit(node):
        if type(node).__name__ in ("SinNode", "CosNode"):
            inner = getattr(node, "arg", None)
            if type(inner).__name__ == "MulNode":
                a, b = inner.left, inner.right
                for const, var in ((a, b), (b, a)):
                    if (
                        isinstance(const, ConstNode)
                        and getattr(var, "kind", None) == "var"
                        and tuple(getattr(var, "var_idxs", ()) or ()) == (int(x_axis),)
                    ):
                        found.append(const)
        for ch in _children(node):
            if ch is not None:
                _visit(ch)

    _visit(ast)
    return found


def _trig_frequency_rescaled_copy(ast, *, x_axis: int, scale: float):
    """Deep copy of ``ast`` with every sin/cos(c*x_axis) constant scaled.

    Constants are deduplicated by object identity before mutation: a node
    aliased into several trig subtrees must be scaled exactly once.
    """
    dup = copy.deepcopy(ast)
    seen_ids: set[int] = set()
    for const in _trig_frequency_consts(dup, x_axis=int(x_axis)):
        if id(const) in seen_ids:
            continue
        seen_ids.add(id(const))
        const.value = float(const.value) * float(scale)
    return dup


def _refit_shared_block_combo(
    block_rows: Sequence[dict[str, Any]],
    *,
    base_mode: str,
    groups: Sequence[DEFeatureGroup],
    base_fit_parts: Sequence[torch.Tensor],
    base_probe_parts: Sequence[torch.Tensor],
    base_ast,
    order: int,
    x_axis: int,
    dtype: torch.dtype,
    lane: str = "two_block_shared_coord",
    role: str = "two_block_shared_coord",
):
    if len(block_rows) != 2:
        return None

    # Joint inner refit: a 2-weight rescale of the single-lane shapes cannot
    # reach the true law when each shape absorbed the other block's
    # contribution (Lane-Emden: the state cubic soaks up the 2/x damping and
    # vice versa). For blocks from known closed-form families, refit the union
    # of both blocks' basis columns jointly; unknown shapes keep the frozen
    # single-column behavior.
    row_column_asts: list[list[Any]] = []
    row_basis_asts: list[list[Any] | None] = []
    seen_cols: set[str] = set()
    for row in block_rows:
        carrier_ast = row.get("carrier_ast", None)
        coord_ast = row.get("coord_ast", None)
        basis_asts = None
        try:
            if carrier_ast is not None and coord_ast is not None:
                basis_asts = _family_basis_asts(coord_ast, str(row.get("family", "") or ""))
        except Exception:
            basis_asts = None
        if basis_asts:
            cols = [carrier_ast if b is None else Mul(b, carrier_ast) for b in basis_asts]
        else:
            cols = [row["block_ast"]]
        kept_cols: list[Any] = []
        kept_basis: list[Any] | None = [] if basis_asts else None
        for j, col in enumerate(cols):
            key = repr(col)
            if key in seen_cols:
                continue  # column shared with the other block; first block owns it
            seen_cols.add(key)
            kept_cols.append(col)
            if kept_basis is not None:
                kept_basis.append(basis_asts[j])
        if not kept_cols:
            return None
        row_column_asts.append(kept_cols)
        row_basis_asts.append(kept_basis)

    scaled_blocks: list[FactorizedDEBlock] = []

    def _assemble_and_solve(column_asts: list[list[Any]]):
        n_cols = sum(len(cols) for cols in column_asts)
        Phi_parts: list[torch.Tensor] = []
        y_parts: list[torch.Tensor] = []
        for group, base_fit in zip(groups, base_fit_parts):
            cols_fit = []
            for cols in column_asts:
                for col_ast in cols:
                    col = _eval_ast_on_features(
                        col_ast,
                        features=group.features,
                        split="fit",
                        x_axis=int(x_axis),
                    ).reshape(-1)
                    cols_fit.append(col)
            Phi_fit = torch.stack(cols_fit, dim=1)
            anchor_fit, _ = _anchor_tensor(group.features, order=int(order))
            y_fit = -(anchor_fit.reshape(-1) + base_fit.reshape(-1))
            mask_fit = torch.isfinite(y_fit) & torch.isfinite(Phi_fit).all(dim=1)
            if int(mask_fit.sum()) < 10:
                return None
            Phi_parts.append(Phi_fit[mask_fit])
            y_parts.append(y_fit[mask_fit])
        Phi_cat = torch.cat(Phi_parts, dim=0)
        y_cat = torch.cat(y_parts, dim=0)
        if int(Phi_cat.shape[1]) != n_cols or int(Phi_cat.shape[0]) < 10:
            return None
        try:
            sol = _scale_weighted_trimmed_lstsq(Phi_cat, y_cat, ridge=1.0e-12).detach().cpu().reshape(-1)
        except Exception:
            return None
        if int(sol.numel()) != n_cols or not bool(torch.isfinite(sol).all().item()):
            return None
        resid = y_cat - Phi_cat @ sol.to(Phi_cat)
        return sol, float((resid * resid).mean().item())

    solved = _assemble_and_solve(row_column_asts)
    if solved is None:
        return None
    coeffs, best_rss = solved

    # One-scalar VarPro frequency polish: a periodogram-hinted or
    # explorer-fitted trig frequency is only as good as its estimate, and
    # over a many-period record even a percent of frequency error dephases
    # into an order-of-magnitude residual (de301).  Refine w by
    # golden-section with the linear coefficients re-solved at each trial.
    # The frequency can live in TWO parameterizations: hint-coord family
    # rows carry it in coord_ast = w*x, while frozen explorer rows carry it
    # inside the block AST as sin/cos(w*x) subtrees; handle both.

    def _walk_trig_freq_consts(ast) -> list[ConstNode]:
        return _trig_frequency_consts(ast, x_axis=int(x_axis))

    def _rescaled_copy(ast, scale: float):
        return _trig_frequency_rescaled_copy(ast, x_axis=int(x_axis), scale=float(scale))

    def _trig_row_frequency(idx: int) -> float | None:
        row = block_rows[idx]
        coord = row.get("coord_ast", None)
        left = getattr(coord, "left", None)
        right = getattr(coord, "right", None)
        if (
            str(row.get("family", "") or "") in ("sin", "cos")
            and type(coord).__name__ == "MulNode"
            and isinstance(left, ConstNode)
            and getattr(right, "kind", None) == "var"
            and tuple(getattr(right, "var_idxs", ()) or ()) == (int(x_axis),)
        ):
            return float(left.value)
        if row_basis_asts[idx] is None:
            consts = _walk_trig_freq_consts(row.get("block_ast", None) or ConstNode(0.0))
            values = {round(float(c.value), 12) for c in consts}
            if len(values) == 1:
                return float(next(iter(values)))
        return None

    polished_rows = list(block_rows)
    for trig_idx in range(len(block_rows)):
        w0 = _trig_row_frequency(trig_idx)
        if w0 is None or not math.isfinite(w0) or w0 <= 0.0:
            continue
        start = sum(len(row_column_asts[j]) for j in range(trig_idx))
        trig_coeffs = coeffs[start:start + len(row_column_asts[trig_idx])]
        if float(trig_coeffs.abs().max().item()) < 1.0e-12:
            continue  # degenerate trig block; polishing noise is wasted solves

        def _columns_for_w(w: float) -> tuple[list[list[Any]], list[Any] | None] | None:
            """Trial column lists with the SAME first-owner dedup as the
            initial assembly; without it the trial design gains columns the
            baseline dropped (e.g. both blocks' shared constant column) and
            the RSS comparison is between different models."""
            row = block_rows[trig_idx]
            if row_basis_asts[trig_idx] is not None:
                coord_w = Mul(ConstNode(float(w)), Var(int(x_axis)))
                try:
                    basis_w = _family_basis_asts(coord_w, str(row.get("family", "") or ""))
                except Exception:
                    return None
                if not basis_w:
                    return None
                carrier_ast = row.get("carrier_ast", None)
                raw_cols = [carrier_ast if b is None else Mul(b, carrier_ast) for b in basis_w]
            else:
                basis_w = None
                raw_cols = [_rescaled_copy(row["block_ast"], float(w) / float(w0))]

            out: list[list[Any]] = []
            trial_seen: set[str] = set()
            kept_basis_w: list[Any] | None = [] if basis_w is not None else None
            for idx2 in range(len(block_rows)):
                if idx2 == trig_idx:
                    cand_cols, cand_basis = raw_cols, basis_w
                else:
                    cand_cols, cand_basis = row_column_asts[idx2], None
                kept: list[Any] = []
                for j, col in enumerate(cand_cols):
                    key = repr(col)
                    if key in trial_seen:
                        continue
                    trial_seen.add(key)
                    kept.append(col)
                    if idx2 == trig_idx and kept_basis_w is not None and cand_basis is not None:
                        kept_basis_w.append(cand_basis[j])
                if not kept:
                    return None
                out.append(kept)
            return out, kept_basis_w

        lo, hi = 0.9 * w0, 1.1 * w0
        invphi = (math.sqrt(5.0) - 1.0) / 2.0
        a, b = lo, hi
        c, d = b - invphi * (b - a), a + invphi * (b - a)
        evals: dict[float, tuple[Any, float]] = {}

        def _rss_at(w: float) -> float:
            trial = _columns_for_w(w)
            if trial is None:
                return float("inf")
            got = _assemble_and_solve(trial[0])
            if got is None:
                return float("inf")
            evals[w] = got
            return got[1]

        fc, fd = _rss_at(c), _rss_at(d)
        for _ in range(16):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - invphi * (b - a)
                fc = _rss_at(c)
            else:
                a, c, fc = c, d, fd
                d = a + invphi * (b - a)
                fd = _rss_at(d)
        w_best = c if fc < fd else d
        got = evals.get(w_best)
        if got is not None and got[1] < best_rss * (1.0 - 1.0e-9):
            trial_best = _columns_for_w(w_best)
            if trial_best is not None:
                cols_best, kept_basis_best = trial_best
                row_column_asts = cols_best
                scale = float(w_best) / float(w0)
                row = dict(block_rows[trig_idx])
                if row_basis_asts[trig_idx] is not None:
                    row_basis_asts = list(row_basis_asts)
                    row_basis_asts[trig_idx] = kept_basis_best
                    # Keep the emitted coord in sync with the polished basis:
                    # reports and identity keys read the row's coord_ast.
                    row["coord_ast"] = Mul(ConstNode(float(w_best)), Var(int(x_axis)))
                else:
                    # Frozen explorer row: the downstream assembly reads the
                    # row's own ASTs, so the adopted frequency must be
                    # propagated into them or it is silently discarded.
                    row["block_ast"] = _rescaled_copy(row["block_ast"], scale)
                    if row.get("coeff_ast", None) is not None:
                        row["coeff_ast"] = _rescaled_copy(row["coeff_ast"], scale)
                row["freq_polish"] = {
                    "w0": float(w0),
                    "w_best": float(w_best),
                    "rss0": float(best_rss),
                    "rss_best": float(got[1]),
                }
                polished_rows[trig_idx] = row
                coeffs, best_rss = got
        break

    block_rows = polished_rows

    block_asts = []
    coeff_exprs = []
    offset = 0
    for row, cols, basis in zip(list(block_rows), row_column_asts, row_basis_asts):
        row_coeffs = [float(c) for c in coeffs[offset:offset + len(cols)].tolist()]
        offset += len(cols)
        if basis is not None:
            scaled_coeff_ast = _sum_linear_terms_ast(basis, row_coeffs)
            if scaled_coeff_ast is None:
                continue
            top_level_weight = 1.0
            coeff_expr = f"jointfit[{row.get('family', '')}]"
        else:
            coeff = row_coeffs[0]
            if abs(coeff) < 1.0e-14:
                continue
            scaled_coeff_ast = row["coeff_ast"] if abs(coeff - 1.0) < 1.0e-14 else Mul(ConstNode(coeff), row["coeff_ast"])
            top_level_weight = coeff
            coeff_expr = f"{coeff:.6g}*{row.get('coeff_expr', '')}"
        scaled_block_ast = Mul(scaled_coeff_ast, row["carrier_ast"])
        block_asts.append(scaled_block_ast)
        coeff_exprs.append(coeff_expr)
        scaled_blocks.append(
            FactorizedDEBlock(
                role=str(role),
                carrier_ast=row["carrier_ast"],
                coord_ast=row["coord_ast"],
                coeff_ast=scaled_coeff_ast,
                block_ast=scaled_block_ast,
                diagnostics={
                    "top_level_weight": float(top_level_weight),
                    "base_mode": str(base_mode),
                    "coeff_expr": row.get("coeff_expr", ""),
                    "mapping": row.get("mapping", {}),
                },
            )
        )

    if len(block_asts) < 2:
        return None

    nonanchor_ast = _compose_nonanchor_ast(base_ast, block_asts)
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

    probe_mse = _trimmed_probe_mse_from_residuals(probe_residuals)
    probe_rms = math.sqrt(probe_mse) if math.isfinite(probe_mse) else float("inf")
    coord_ast = block_rows[0]["coord_ast"]
    carrier_ast = tuple(row["carrier_ast"] for row in block_rows)
    nonanchor_ast_raw = nonanchor_ast
    residual_ast_raw = Add(_anchor_ast(int(order), x_axis=int(x_axis)), nonanchor_ast_raw)
    nonanchor_ast_simplified = _simplify_de_ast(nonanchor_ast_raw)
    residual_ast_simplified = _simplify_de_ast(residual_ast_raw)
    residual_ast = residual_ast_simplified
    nonanchor_ast = nonanchor_ast_simplified
    canonical_equation_raw = _canonical_equation(int(order), int(x_axis), nonanchor_ast_raw)
    canonical_equation_simplified = _canonical_equation(int(order), int(x_axis), nonanchor_ast_simplified)
    canonical_equation = canonical_equation_simplified
    return {
        "order": int(order),
        "carrier_ast": carrier_ast,
        "coord_ast": coord_ast,
        "coeff_ast": None,
        "coeff_asts": [blk.coeff_ast for blk in scaled_blocks],
        "block_ast": None,
        "blocks": scaled_blocks,
        "lane": str(lane),
        "family": "shared_refit",
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
        "coeff_expr": " + ".join(coeff_exprs),
        "mapping": {},
        "size": sum(int(row.get("size", 0)) for row in block_rows),
        "symbolic_size_raw": int(ast_node_count(residual_ast_raw)),
        "symbolic_size_simplified": int(ast_node_count(residual_ast_simplified)),
        "ratio_probe_mse": max(float(row.get("ratio_probe_mse", float("inf"))) for row in block_rows),
        "consistency_score": float(
            sum(float(row.get("consistency_score", float("inf"))) for row in block_rows) / max(len(block_rows), 1)
        ),
        "consistency_pairs": max(int(row.get("consistency_pairs", 0)) for row in block_rows),
        "consistency_total_pairs": max(int(row.get("consistency_total_pairs", 0)) for row in block_rows),
        "evidence_tier": _consistency_evidence_tier(
            {
                "consistency_score": float(
                    sum(float(row.get("consistency_score", float("inf"))) for row in block_rows) / max(len(block_rows), 1)
                ),
                "consistency_pairs": max(int(row.get("consistency_pairs", 0)) for row in block_rows),
                "consistency_total_pairs": max(int(row.get("consistency_total_pairs", 0)) for row in block_rows),
            }
        )[1],
    }


def _build_two_block_shared_coord_candidates(
    single_rows: Sequence[dict[str, Any]],
    *,
    base_mode: str,
    groups: Sequence[DEFeatureGroup],
    base_fit_parts: Sequence[torch.Tensor],
    base_probe_parts: Sequence[torch.Tensor],
    base_ast,
    order: int,
    x_axis: int,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    if int(order) != 2:
        return []

    by_coord_role: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in list(single_rows):
        coord_key = repr(row.get("coord_ast", None))
        role = _carrier_role(row.get("carrier_ast", None), x_axis=int(x_axis))
        by_coord_role.setdefault((coord_key, role), []).append(row)

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    coord_keys = sorted({key for key, _ in by_coord_role.keys()})
    for coord_key in coord_keys:
        du_rows = by_coord_role.get((coord_key, "du"), [])
        u_rows = by_coord_role.get((coord_key, "u"), [])
        for du_row in du_rows:
            for u_row in u_rows:
                pair_key = (
                    coord_key,
                    repr(du_row.get("coeff_ast", None)),
                    repr(u_row.get("coeff_ast", None)),
                )
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                cand = _refit_shared_block_combo(
                    [du_row, u_row],
                    base_mode=str(base_mode),
                    groups=groups,
                    base_fit_parts=base_fit_parts,
                    base_probe_parts=base_probe_parts,
                    base_ast=base_ast,
                    order=int(order),
                    x_axis=int(x_axis),
                    dtype=dtype,
                )
                if cand is not None:
                    out.append(cand)
    return out


def _typed_two_block_pair_allowed(lhs: dict[str, Any], rhs: dict[str, Any]) -> bool:
    lanes = {str(lhs.get("lane", "") or ""), str(rhs.get("lane", "") or "")}
    if "two_block_typed_assembly" in lanes or "two_block_shared_coord" in lanes:
        return False
    if "second_order_state_nonlinearity" in lanes and "second_order_x_coeff_on_u" in lanes:
        return True
    # Driven oscillators: state restoring force plus additive forcing,
    # u'' + a*u + F*cos(w*x) = 0 (de301-class); the forcing lane alone
    # cannot carry the state term, so it must pair with a state block.
    if "second_order_state_nonlinearity" in lanes and "second_order_x_forcing" in lanes:
        return True
    if "second_order_state_nonlinearity" in lanes and "second_order_state_damping_on_du" in lanes:
        return True
    if "second_order_state_nonlinearity" in lanes and "second_order_x_damping_on_du" in lanes:
        return True
    # Bessel-type equations: x-dependent stiffness plus x-dependent damping,
    # u'' + g(x)*u' + h(x)*u = 0 (g=1/x, h=1-nu^2/x^2).
    if "second_order_x_coeff_on_u" in lanes and "second_order_x_damping_on_du" in lanes:
        return True
    return False


def _build_two_block_typed_candidates(
    single_rows: Sequence[dict[str, Any]],
    *,
    base_mode: str,
    groups: Sequence[DEFeatureGroup],
    base_fit_parts: Sequence[torch.Tensor],
    base_probe_parts: Sequence[torch.Tensor],
    base_ast,
    order: int,
    x_axis: int,
    dtype: torch.dtype,
    replace_rel_factor: float,
) -> list[dict[str, Any]]:
    if int(order) != 2:
        return []
    rows = [
        row
        for row in list(single_rows)
        if str(row.get("base_mode", "")) == str(base_mode)
        and str(row.get("family", "")) not in ("",)
        and row.get("block_ast", None) is not None
    ]
    if len(rows) < 2:
        return []

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for i, lhs in enumerate(rows):
        for rhs in rows[i + 1:]:
            if not _typed_two_block_pair_allowed(lhs, rhs):
                continue
            key = tuple(sorted((repr(lhs.get("block_ast", None)), repr(rhs.get("block_ast", None)))))
            if key in seen:
                continue
            seen.add(key)
            cand = _refit_shared_block_combo(
                [lhs, rhs],
                base_mode=str(base_mode),
                groups=groups,
                base_fit_parts=base_fit_parts,
                base_probe_parts=base_probe_parts,
                base_ast=base_ast,
                order=int(order),
                x_axis=int(x_axis),
                dtype=dtype,
                lane="two_block_typed_assembly",
                role="two_block_typed_assembly",
            )
            if cand is None:
                continue
            best_single = min(
                float(lhs.get("probe_rms", float("inf"))),
                float(rhs.get("probe_rms", float("inf"))),
            )
            combo_rms = float(cand.get("probe_rms", float("inf")))
            prune_rel = max(float(replace_rel_factor), 0.98)
            if math.isfinite(best_single) and not _material_improvement(
                combo_rms,
                best_single,
                replace_rel_factor=float(prune_rel),
            ):
                continue
            cand["assembly_lanes"] = [str(lhs.get("lane", "")), str(rhs.get("lane", ""))]
            cand["assembly_families"] = [str(lhs.get("family", "")), str(rhs.get("family", ""))]
            cand["assembly_pruned"] = False
            out.append(cand)
    return out


def run_factorized_coeff_rescue_from_feature_groups(
    groups: Sequence[DEFeatureGroup],
    *,
    cfg,
    rescue_cfg: FactorizedDERescueConfig,
    primary: DESearchResult | DESearchResultMulti | None = None,
    dtype: torch.dtype = torch.float64,
):
    if not groups:
        return None

    hp = copy.deepcopy(rescue_cfg.hp)
    if hp is None:
        hp = default_physics_rescue_hp(preset="fast")

    n_iter = max(2000, min(int(getattr(hp, "n_iter", 8000)), 8000))
    max_depth = min(int(getattr(hp, "max_depth", 4)), 4)
    shortlist_topk = max(1, int(getattr(rescue_cfg, "shortlist_topk", 8)))
    seed = int(getattr(hp, "seed", 0))
    explorer_topk = max(1, int(getattr(hp, "return_topk", 2) or 2))
    explorer_fit_cap, explorer_probe_cap = _typed_explorer_caps_from_hp(hp)
    typed_lane_workers = max(1, int(getattr(rescue_cfg, "typed_lane_workers", 1) or 1))
    typed_process_backend_requested = int(typed_lane_workers) > 1
    measurement_started = time.perf_counter()
    measurement_diag: dict[str, Any] = {
        "mode": "operator_factorized",
        "n_groups_total": int(len(groups)),
        "group_rows": _feature_group_row_summary(groups),
        "process_memory_start": _process_memory_report("factorized_de_typed_start"),
        "typed_lane_workers": int(typed_lane_workers),
        "typed_parallel_backend": "process_explorer_tasks" if typed_process_backend_requested else "serial",
        "typed_parallel_start_method": _multiprocessing_start_method_name()
        if typed_process_backend_requested
        else "none",
        "typed_parallel_workers": int(typed_lane_workers) if typed_process_backend_requested else 1,
        "typed_parallel_workers_requested": int(typed_lane_workers),
        "typed_explorer_fit_cap": None if explorer_fit_cap is None else int(explorer_fit_cap),
        "typed_explorer_probe_cap": None if explorer_probe_cap is None else int(explorer_probe_cap),
        "typed_explorer_cap_source": "explicit" if (explorer_fit_cap is not None or explorer_probe_cap is not None) else "per_order",
        "orders": [],
        "family_fit_attempts": 0,
        "family_candidates": 0,
        "family_gate_evaluations": 0,
        "family_gate_passes": 0,
        "explorer_skipped": 0,
        "scheduler_coord_candidates_considered": 0,
        "scheduler_coord_candidates_skipped": 0,
        "explorer_pairs_considered": 0,
        "explorer_pairs_with_targets": 0,
        "explorer_launches": 0,
        "explorer_rows": 0,
        "explorer_candidates": 0,
        "typed_explorer_launches": 0,
        "typed_explorer_candidates": 0,
        "typed_explorer_fit_rows_before": 0,
        "typed_explorer_fit_rows_after": 0,
        "typed_explorer_probe_rows_before": 0,
        "typed_explorer_probe_rows_after": 0,
        "typed_explorer_subsampled_launches": 0,
        "typed_tasks_planned": 0,
        "typed_tasks_submitted": 0,
        "typed_tasks_started": 0,
        "typed_tasks_finished": 0,
        "typed_tasks_failed": 0,
        "typed_tasks_rerun_serial": 0,
        "typed_tasks_inflight": 0,
        "typed_tasks_inflight_peak": 0,
        "typed_eval_budget_total": 0,
        "typed_eval_budget_finished": 0,
        "typed_best_probe_mse_so_far": None,
        "typed_best_probe_rms_so_far": None,
        "typed_task_wall_seconds_finished": 0.0,
        "typed_process_pool_launches": 0,
        "typed_process_pool_tasks": 0,
        "typed_process_pool_future_failures": 0,
        "typed_process_pool_failures": 0,
        "generic_explorer_launches": 0,
        "generic_explorer_candidates": 0,
        "two_block_attempts": 0,
        "two_block_typed_candidates": 0,
    }

    candidates: list[dict[str, Any]] = []
    zero_base_x_lane_diag_ctx: dict[str, Any] | None = None
    if primary is not None:
        orders = [int(getattr(primary, "order", -1))]
    else:
        orders = [int(o) for o in getattr(cfg, "order_candidates", (1, 2))]

    for order in orders:
        if int(order) not in (1, 2):
            continue
        order_explorer_fit_cap, order_explorer_probe_cap, order_explorer_cap_source = _typed_explorer_caps_for_order(
            int(order),
            explorer_fit_cap,
            explorer_probe_cap,
        )
        preferred_zero_lane: str | None = None
        for base_variant in _base_variants(
            primary,
            groups,
            order=int(order),
            x_axis=int(cfg.x_axis),
            dtype=dtype,
            base_modes=getattr(rescue_cfg, "base_modes", ("zero", "primary")),
        ):
            base_mode = str(base_variant.get("mode", "zero"))
            base_started = time.perf_counter()
            base_fit_parts = list(base_variant.get("fit_parts", []) or [])
            base_probe_parts = list(base_variant.get("probe_parts", []) or [])
            base_ast = base_variant.get("ast", None)
            resid_fit_parts, resid_probe_parts = _residual_parts_for_base(
                groups,
                order=int(order),
                base_fit_parts=base_fit_parts,
                base_probe_parts=base_probe_parts,
            )
            baseline_probe_mse = _probe_mse_from_residuals(resid_probe_parts)
            baseline_probe_rms = math.sqrt(baseline_probe_mse) if math.isfinite(baseline_probe_mse) else float("inf")
            base_diag: dict[str, Any] = {
                "order": int(order),
                "base_mode": str(base_mode),
                "baseline_probe_rms": float(baseline_probe_rms) if math.isfinite(baseline_probe_rms) else None,
                "candidates_before": int(len(candidates)),
                "typed_explorer_fit_cap": None
                if order_explorer_fit_cap is None
                else int(order_explorer_fit_cap),
                "typed_explorer_probe_cap": None
                if order_explorer_probe_cap is None
                else int(order_explorer_probe_cap),
                "typed_explorer_cap_source": str(order_explorer_cap_source),
            }
            if (
                int(order) == 2
                and bool(getattr(cfg, "include_u", True))
                and bool(getattr(cfg, "include_du", True))
            ):
                velocity_diag = _second_order_velocity_correction_diagnostic(
                    groups=groups,
                    order=int(order),
                    x_axis=int(cfg.x_axis),
                    base_fit_parts=base_fit_parts,
                    base_probe_parts=base_probe_parts,
                )
                if velocity_diag is not None:
                    base_diag["velocity_coeff_on_u_diagnostic"] = velocity_diag
                    measurement_diag.setdefault("velocity_coeff_on_u_diagnostics", []).append(
                        {
                            "order": int(order),
                            "base_mode": str(base_mode),
                            **velocity_diag,
                        }
                    )
            single_rows_for_variant: list[dict[str, Any]] = []
            typed_rows_for_variant: list[dict[str, Any]] = []
            typed_all_rows_for_variant: list[dict[str, Any]] = []

            if int(order) == 1:
                neg_u = Mul(ConstNode(-1.0), U())
                state_coord_asts = _dedupe_ast_list([U(), Add(ConstNode(1.0), U()), neg_u, Add(ConstNode(1.0), neg_u)])
                x_coeff_coord_asts = _dedupe_ast_list([Var(int(cfg.x_axis)), Add(ConstNode(1.0), Var(int(cfg.x_axis)))])
                state_rows: list[dict[str, Any]] = []
                x_coeff_rows: list[dict[str, Any]] = []
                allow_state_lane_raw = bool(getattr(cfg, "include_const", True)) and bool(getattr(cfg, "include_u", True))
                allow_x_coeff_lane_raw = bool(getattr(cfg, "include_x", True)) and bool(getattr(cfg, "include_u", True))
                allow_state_lane, allow_x_coeff_lane = _active_first_order_typed_lanes(
                    base_mode=base_mode,
                    preferred_zero_lane=preferred_zero_lane,
                    allow_state_lane=allow_state_lane_raw,
                    allow_x_coeff_lane=allow_x_coeff_lane_raw,
                )

                if allow_state_lane:
                    state_family_rows = _build_family_lane_candidates(
                        lane="state_nonlinearity",
                        base_mode=base_mode,
                        groups=groups,
                        order=int(order),
                        x_axis=int(cfg.x_axis),
                        base_ast=base_ast,
                        resid_fit_parts=resid_fit_parts,
                        resid_probe_parts=resid_probe_parts,
                        carrier_ast=ConstNode(1.0),
                        coord_asts=state_coord_asts,
                        family_names=("poly2", "poly3", "sin", "cos", "exp", "reciprocal", "log"),
                        rel_eps=float(rescue_cfg.ratio_rel_eps),
                        min_ratio_rows=int(rescue_cfg.min_ratio_rows),
                        diagnostics=measurement_diag,
                    )
                    state_skip_explorer, state_skip_reason, _state_gate_row = _family_first_gate_decision(
                        lane="state_nonlinearity",
                        base_mode=base_mode,
                        order=int(order),
                        family_rows=state_family_rows,
                        baseline_probe_rms=float(baseline_probe_rms),
                        replace_rel_factor=float(getattr(rescue_cfg, "replace_rel_factor", 0.98)),
                        trigger_val_rms=float(getattr(rescue_cfg, "trigger_val_rms", 1.0e-3)),
                        diagnostics=measurement_diag,
                    )
                    if state_skip_explorer:
                        state_explorer_rows = []
                        base_diag.setdefault("explorer_skips", []).append(
                            {
                                "lane": "state_nonlinearity",
                                "reason": str(state_skip_reason),
                            }
                        )
                    else:
                        state_explorer_coord_asts, _state_scheduler_reports = _schedule_explorer_coord_asts_by_collapse(
                            lane="state_nonlinearity",
                            base_mode=base_mode,
                            order=int(order),
                            groups=groups,
                            resid_probe_parts=resid_probe_parts,
                            carrier_ast=ConstNode(1.0),
                            coord_asts=state_coord_asts,
                            x_axis=int(cfg.x_axis),
                            rel_eps=float(rescue_cfg.ratio_rel_eps),
                            diagnostics=measurement_diag,
                        )
                        state_explorer_rows = _build_explorer_lane_candidates(
                            lane="state_nonlinearity",
                            base_mode=base_mode,
                            groups=groups,
                            order=int(order),
                            x_axis=int(cfg.x_axis),
                            base_ast=base_ast,
                            resid_fit_parts=resid_fit_parts,
                            resid_probe_parts=resid_probe_parts,
                            carrier_asts=[ConstNode(1.0)],
                            coord_asts=state_explorer_coord_asts,
                            rel_eps=float(rescue_cfg.ratio_rel_eps),
                            min_ratio_rows=int(rescue_cfg.min_ratio_rows),
                            n_iter=int(n_iter),
                            max_depth=int(max_depth),
                            explorer_topk=int(explorer_topk),
                            seed=int(seed),
                            dtype=dtype,
                            explorer_fit_cap=order_explorer_fit_cap,
                            explorer_probe_cap=order_explorer_probe_cap,
                            explorer_workers=int(typed_lane_workers),
                            diagnostics=measurement_diag,
                        )
                    state_choice, state_kept_rows = _select_state_lane_candidates(
                        state_family_rows + state_explorer_rows
                    )
                    if state_choice is not None:
                        state_rows = list(state_kept_rows)
                        if state_skip_explorer and _state_gate_row is not None:
                            gate_key = _candidate_identity_key(_state_gate_row)
                            state_rows = [_state_gate_row] + [
                                row for row in state_rows if _candidate_identity_key(row) != gate_key
                            ]
                        typed_rows_for_variant.extend(state_rows)

                if allow_x_coeff_lane:
                    x_family_rows = _build_family_lane_candidates(
                        lane="x_coeff_on_u",
                        base_mode=base_mode,
                        groups=groups,
                        order=int(order),
                        x_axis=int(cfg.x_axis),
                        base_ast=base_ast,
                        resid_fit_parts=resid_fit_parts,
                        resid_probe_parts=resid_probe_parts,
                        carrier_ast=U(),
                        coord_asts=x_coeff_coord_asts,
                        family_names=("reciprocal", "inv_square", "log", "poly2"),
                        rel_eps=float(rescue_cfg.ratio_rel_eps),
                        min_ratio_rows=int(rescue_cfg.min_ratio_rows),
                        diagnostics=measurement_diag,
                    )
                    x_skip_explorer, x_skip_reason, _x_gate_row = _family_first_gate_decision(
                        lane="x_coeff_on_u",
                        base_mode=base_mode,
                        order=int(order),
                        family_rows=x_family_rows,
                        baseline_probe_rms=float(baseline_probe_rms),
                        replace_rel_factor=float(getattr(rescue_cfg, "replace_rel_factor", 0.98)),
                        trigger_val_rms=float(getattr(rescue_cfg, "trigger_val_rms", 1.0e-3)),
                        diagnostics=measurement_diag,
                    )
                    if x_skip_explorer:
                        x_explorer_rows = []
                        base_diag.setdefault("explorer_skips", []).append(
                            {
                                "lane": "x_coeff_on_u",
                                "reason": str(x_skip_reason),
                            }
                        )
                    else:
                        x_explorer_coord_asts, _x_scheduler_reports = _schedule_explorer_coord_asts_by_collapse(
                            lane="x_coeff_on_u",
                            base_mode=base_mode,
                            order=int(order),
                            groups=groups,
                            resid_probe_parts=resid_probe_parts,
                            carrier_ast=U(),
                            coord_asts=x_coeff_coord_asts,
                            x_axis=int(cfg.x_axis),
                            rel_eps=float(rescue_cfg.ratio_rel_eps),
                            diagnostics=measurement_diag,
                        )
                        x_explorer_rows = _build_explorer_lane_candidates(
                            lane="x_coeff_on_u",
                            base_mode=base_mode,
                            groups=groups,
                            order=int(order),
                            x_axis=int(cfg.x_axis),
                            base_ast=base_ast,
                            resid_fit_parts=resid_fit_parts,
                            resid_probe_parts=resid_probe_parts,
                            carrier_asts=[U()],
                            coord_asts=x_explorer_coord_asts,
                            rel_eps=float(rescue_cfg.ratio_rel_eps),
                            min_ratio_rows=int(rescue_cfg.min_ratio_rows),
                            n_iter=int(n_iter),
                            max_depth=int(max_depth),
                            explorer_topk=int(explorer_topk),
                            seed=int(seed),
                            dtype=dtype,
                            explorer_fit_cap=order_explorer_fit_cap,
                            explorer_probe_cap=order_explorer_probe_cap,
                            explorer_workers=int(typed_lane_workers),
                            diagnostics=measurement_diag,
                        )
                    x_choice, x_kept_rows = _select_x_lane_candidates(x_family_rows + x_explorer_rows)
                    if x_choice is not None:
                        x_coeff_rows = list(x_kept_rows)
                        if x_skip_explorer and _x_gate_row is not None:
                            gate_key = _candidate_identity_key(_x_gate_row)
                            x_coeff_rows = [_x_gate_row] + [
                                row for row in x_coeff_rows if _candidate_identity_key(row) != gate_key
                            ]
                        typed_rows_for_variant.extend(x_coeff_rows)
                    if base_mode == "zero":
                        zero_base_x_lane_diag_ctx = {
                            "groups": groups,
                            "x_axis": int(cfg.x_axis),
                            "rel_eps": float(rescue_cfg.ratio_rel_eps),
                            "min_ratio_rows": int(rescue_cfg.min_ratio_rows),
                            "resid_fit_parts": list(resid_fit_parts),
                            "resid_probe_parts": list(resid_probe_parts),
                            "coord_asts": list(x_coeff_coord_asts),
                            "x_rows": list(x_family_rows + x_explorer_rows),
                        }

                if base_mode == "zero":
                    preferred_zero_lane = _choose_preferred_zero_lane(
                        state_rows=state_rows,
                        x_coeff_rows=x_coeff_rows,
                    )

                candidates.extend(typed_rows_for_variant)
                single_rows_for_variant.extend(typed_rows_for_variant)

                allow_generic = preferred_zero_lane is None
                best_typed_probe_rms = _best_probe_rms(typed_rows_for_variant)
                if allow_generic and not _material_improvement(
                    best_typed_probe_rms,
                    baseline_probe_rms,
                    replace_rel_factor=float(getattr(rescue_cfg, "replace_rel_factor", 0.98)),
                ):
                    generic_rows = _build_explorer_lane_candidates(
                        lane="generic_coeff_on_carrier",
                        base_mode=base_mode,
                        groups=groups,
                        order=int(order),
                        x_axis=int(cfg.x_axis),
                        base_ast=base_ast,
                        resid_fit_parts=resid_fit_parts,
                        resid_probe_parts=resid_probe_parts,
                        carrier_asts=_carrier_pool(cfg=cfg, order=int(order), x_axis=int(cfg.x_axis)),
                        coord_asts=_coord_pool(cfg=cfg, order=int(order), x_axis=int(cfg.x_axis)),
                        rel_eps=float(rescue_cfg.ratio_rel_eps),
                        min_ratio_rows=int(rescue_cfg.min_ratio_rows),
                        n_iter=int(n_iter),
                        max_depth=int(max_depth),
                        explorer_topk=int(explorer_topk),
                        seed=int(seed),
                        dtype=dtype,
                        explorer_fit_cap=order_explorer_fit_cap,
                        explorer_probe_cap=order_explorer_probe_cap,
                        explorer_workers=int(typed_lane_workers),
                        diagnostics=measurement_diag,
                    )
                    candidates.extend(generic_rows)
                    single_rows_for_variant.extend(generic_rows)
            else:
                typed_lane_reports: list[dict[str, Any]] = []
                frequency_hints = _typed_lane_frequency_hints(
                    groups,
                    base_fit_parts,
                    order=int(order),
                    x_axis=int(cfg.x_axis),
                )
                if frequency_hints:
                    measurement_diag["frequency_hints"] = [float(w) for w in frequency_hints]
                lane_specs = _second_order_typed_lane_specs(
                    cfg=cfg, x_axis=int(cfg.x_axis), frequency_hints=frequency_hints
                )

                def _run_second_order_lane_spec(lane_spec: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
                    lane_diag: dict[str, Any] = {}
                    lane_rows, lane_all_rows, lane_report = _build_typed_lane_candidates_with_gate(
                        lane=str(lane_spec["lane"]),
                        base_mode=base_mode,
                        groups=groups,
                        order=int(order),
                        x_axis=int(cfg.x_axis),
                        base_ast=base_ast,
                        resid_fit_parts=resid_fit_parts,
                        resid_probe_parts=resid_probe_parts,
                        carrier_ast=lane_spec["carrier_ast"],
                        coord_asts=list(lane_spec["coord_asts"]),
                        family_names=list(lane_spec["family_names"]),
                        baseline_probe_rms=float(baseline_probe_rms),
                        rescue_cfg=rescue_cfg,
                        n_iter=int(n_iter),
                        max_depth=int(max_depth),
                        explorer_topk=int(explorer_topk),
                        seed=int(seed),
                        dtype=dtype,
                        explorer_fit_cap=order_explorer_fit_cap,
                        explorer_probe_cap=order_explorer_probe_cap,
                        explorer_workers=int(typed_lane_workers),
                        diagnostics=lane_diag,
                    )
                    return lane_rows, lane_all_rows, lane_report, lane_diag

                base_diag["typed_lane_spec_execution"] = "serial"
                for lane_spec in lane_specs:
                    lane_rows, lane_all_rows, lane_report, lane_diag = _run_second_order_lane_spec(lane_spec)
                    _merge_diagnostics(measurement_diag, lane_diag)
                    typed_lane_reports.append(lane_report)
                    typed_rows_for_variant.extend(lane_rows)
                    typed_all_rows_for_variant.extend(lane_all_rows)

                candidates.extend(typed_rows_for_variant)
                single_rows_for_variant.extend(typed_rows_for_variant)
                base_diag["typed_lane_reports"] = typed_lane_reports

                best_typed_probe_rms = _best_probe_rms(typed_rows_for_variant)
                force_generic_for_two_block = (
                    str(getattr(rescue_cfg, "two_block_shared_coord_mode", "never") or "never").strip().lower()
                    == "always"
                )
                if force_generic_for_two_block or not _material_improvement(
                    best_typed_probe_rms,
                    baseline_probe_rms,
                    replace_rel_factor=float(getattr(rescue_cfg, "replace_rel_factor", 0.98)),
                ):
                    generic_rows = _build_explorer_lane_candidates(
                        lane="generic_coeff_on_carrier",
                        base_mode=base_mode,
                        groups=groups,
                        order=int(order),
                        x_axis=int(cfg.x_axis),
                        base_ast=base_ast,
                        resid_fit_parts=resid_fit_parts,
                        resid_probe_parts=resid_probe_parts,
                        carrier_asts=_carrier_pool(cfg=cfg, order=int(order), x_axis=int(cfg.x_axis)),
                        coord_asts=_coord_pool(cfg=cfg, order=int(order), x_axis=int(cfg.x_axis)),
                        rel_eps=float(rescue_cfg.ratio_rel_eps),
                        min_ratio_rows=int(rescue_cfg.min_ratio_rows),
                        n_iter=int(n_iter),
                        max_depth=int(max_depth),
                        explorer_topk=int(explorer_topk),
                        seed=int(seed),
                        dtype=dtype,
                        explorer_fit_cap=order_explorer_fit_cap,
                        explorer_probe_cap=order_explorer_probe_cap,
                        explorer_workers=int(typed_lane_workers),
                        diagnostics=measurement_diag,
                    )
                    candidates.extend(generic_rows)
                    single_rows_for_variant.extend(generic_rows)

            best_single_probe_rms = min(
                (float(row.get("probe_rms", float("inf"))) for row in single_rows_for_variant),
                default=float("inf"),
            )
            if (
                int(order) == 2
                and str(getattr(rescue_cfg, "two_block_shared_coord_mode", "never") or "never").strip().lower()
                != "never"
            ):
                typed_two_block_rows = _build_two_block_typed_candidates(
                    typed_all_rows_for_variant or typed_rows_for_variant,
                    base_mode=base_mode,
                    groups=groups,
                    base_fit_parts=base_fit_parts,
                    base_probe_parts=base_probe_parts,
                    base_ast=base_ast,
                    order=int(order),
                    x_axis=int(cfg.x_axis),
                    dtype=dtype,
                    replace_rel_factor=float(getattr(rescue_cfg, "replace_rel_factor", 0.98)),
                )
                if typed_two_block_rows:
                    candidates.extend(typed_two_block_rows)
                    _diag_inc(measurement_diag, "two_block_attempts", 1)
                    _diag_inc(measurement_diag, "two_block_typed_candidates", len(typed_two_block_rows))
            if _should_run_two_block_shared_coord(
                getattr(rescue_cfg, "two_block_shared_coord_mode", "never"),
                order=int(order),
                baseline_probe_rms=float(baseline_probe_rms),
                best_single_probe_rms=float(best_single_probe_rms),
                replace_rel_factor=float(getattr(rescue_cfg, "replace_rel_factor", 0.98)),
            ):
                candidates.extend(
                    _build_two_block_shared_coord_candidates(
                        single_rows_for_variant,
                        base_mode=base_mode,
                        groups=groups,
                        base_fit_parts=base_fit_parts,
                        base_probe_parts=base_probe_parts,
                        base_ast=base_ast,
                        order=int(order),
                        x_axis=int(cfg.x_axis),
                        dtype=dtype,
                    )
                )
                _diag_inc(measurement_diag, "two_block_attempts", 1)
            base_diag.update(
                {
                    "typed_rows": int(len(typed_rows_for_variant)),
                    "single_rows": int(len(single_rows_for_variant)),
                    "candidates_after": int(len(candidates)),
                    "candidates_added": int(len(candidates) - int(base_diag["candidates_before"])),
                    "wall_seconds": float(time.perf_counter() - base_started),
                }
            )
            measurement_diag.setdefault("orders", []).append(base_diag)

    if not candidates:
        return None

    before_domain_filter = int(len(candidates))
    candidates = [row for row in candidates if _row_domain_safe(row)]
    measurement_diag["domain_rejected_candidates"] = int(before_domain_filter - len(candidates))
    if not candidates:
        return None

    candidates.sort(key=cmp_to_key(_compare_candidate_rows))

    selection_candidates = [
        row for row in candidates if str(row.get("projection_kind", "") or "") not in {"support", "snap"}
    ]
    best = selection_candidates[0] if selection_candidates else candidates[0]
    shortlist = _diverse_candidate_shortlist(candidates, shortlist_topk)
    best_key = _candidate_identity_key(best)
    selected_shortlist_rank = next(
        (idx for idx, row in enumerate(shortlist) if _candidate_identity_key(row) == best_key),
        None,
    )
    if selected_shortlist_rank is None:
        shortlist = [best] + [row for row in shortlist if _candidate_identity_key(row) != best_key]
        shortlist = shortlist[:shortlist_topk]
        selected_shortlist_rank = 0
    measurement_diag["n_candidates"] = int(len(candidates))
    measurement_diag["shortlist_topk"] = int(shortlist_topk)
    measurement_diag["shortlist_size"] = int(len(shortlist))
    measurement_diag["selected_lane"] = str(best.get("lane", "single_block"))
    measurement_diag["selected_family"] = str(best.get("family", ""))
    measurement_diag["selected_base_mode"] = str(best.get("base_mode", "zero"))
    velocity_diags = list(measurement_diag.get("velocity_coeff_on_u_diagnostics", []) or [])
    if velocity_diags:
        measurement_diag["best_velocity_coeff_on_u_diagnostic"] = min(
            velocity_diags,
            key=lambda row: (
                float(row.get("corrected_probe_rms"))
                if row.get("corrected_probe_rms") is not None
                else float("inf"),
                float(row.get("probe_rms_ratio"))
                if row.get("probe_rms_ratio") is not None
                else float("inf"),
            ),
        )
    measurement_diag["wall_seconds"] = float(time.perf_counter() - measurement_started)
    measurement_diag["process_memory_end"] = _process_memory_report("factorized_de_typed_end")
    diagnostics = {
        "shortlist_rows": shortlist,
        "selected_shortlist_rank": int(selected_shortlist_rank),
        "lane": str(best.get("lane", "single_block")),
        "family": str(best.get("family", "")),
        "base_mode": str(best.get("base_mode", "zero")),
        "witness_kind": str(best.get("witness_kind", "")),
        "consistency_score": float(best["consistency_score"])
        if best.get("consistency_score", None) is not None and math.isfinite(float(best["consistency_score"]))
        else None,
        "consistency_pairs": int(best.get("consistency_pairs", 0)),
        "consistency_total_pairs": int(best.get("consistency_total_pairs", 0)),
        "evidence_tier": str(best.get("evidence_tier", "unverified")),
        "shape_score": float(best.get("shape_score", 0.0) or 0.0),
        "sign_changes": float(best.get("sign_changes", 0.0) or 0.0),
        "curvature_ratio": float(best.get("curvature_ratio", 0.0) or 0.0),
        "tv_ratio": float(best.get("tv_ratio", 1.0) or 1.0),
        "factorized_de_diagnostics": measurement_diag,
    }
    if zero_base_x_lane_diag_ctx is not None:
        x_lane_report = _build_zero_base_x_lane_diagnostics(
            groups=zero_base_x_lane_diag_ctx["groups"],
            x_axis=int(zero_base_x_lane_diag_ctx["x_axis"]),
            rel_eps=float(zero_base_x_lane_diag_ctx["rel_eps"]),
            min_ratio_rows=int(zero_base_x_lane_diag_ctx["min_ratio_rows"]),
            resid_fit_parts=zero_base_x_lane_diag_ctx["resid_fit_parts"],
            resid_probe_parts=zero_base_x_lane_diag_ctx["resid_probe_parts"],
            coord_asts=zero_base_x_lane_diag_ctx["coord_asts"],
            x_rows=zero_base_x_lane_diag_ctx["x_rows"],
            selected_row=best,
        )
        if x_lane_report is not None:
            diagnostics["zero_base_x_lane_diagnostics"] = x_lane_report
    return FactorizedDEResult(
        order=int(best["order"]),
        x_axis=int(cfg.x_axis),
        nonanchor_ast=best["nonanchor_ast"],
        residual_ast=best["residual_ast"],
        canonical_equation=str(best["canonical_equation"]),
        probe_mse=float(best["probe_mse"]),
        probe_rms=float(best["probe_rms"]),
        blocks=list(best.get("blocks", []) or []),
        diagnostics=diagnostics,
        residual_ast_raw=best.get("residual_ast_raw", best["residual_ast"]),
        residual_ast_simplified=best.get("residual_ast_simplified", best["residual_ast"]),
        canonical_equation_raw=str(best.get("canonical_equation_raw", best["canonical_equation"])),
        canonical_equation_simplified=str(
            best.get("canonical_equation_simplified", best["canonical_equation"])
        ),
    )

__factorized_de_definitions__ = (
    "_second_order_state_coord_asts",
    "_second_order_x_coord_asts",
    "_second_order_velocity_coord_asts",
    "_typed_lane_frequency_hints",
    "_second_order_velocity_correction_diagnostic",
    "_second_order_typed_lane_specs",
    "_refit_shared_block_combo",
    "_build_two_block_shared_coord_candidates",
    "_typed_two_block_pair_allowed",
    "_build_two_block_typed_candidates",
    "run_factorized_coeff_rescue_from_feature_groups",
)

__factorized_de_constants__ = (

)

__factorized_de_late_bindings__ = (

)
