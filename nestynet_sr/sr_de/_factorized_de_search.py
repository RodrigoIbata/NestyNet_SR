# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

"""Factorized-search reports, direct residuals, and implicit DE lanes."""

from typing import TYPE_CHECKING
import copy
import itertools
import math
import time
from dataclasses import replace
from typing import Any, Mapping, Sequence
import numpy as np
import torch
import nestynet_sr.sr_search.factorized_search.oracle_lab_de as oracle_de
from nestynet_sr.sr_core.bridges import Add, ConstNode, Mul
from nestynet_sr.sr_search.factorized_search.bridge import factorized_search_to_nestynet, embed_mapping_in_ast
from nestynet_sr.sr_search.factorized_search.config import FactorizedSearchConfig, apply_refine_profile, factorized_config_report
from nestynet_sr.sr_search.factorized_search.domain_projection import domain_projection_is_acceptable, eval_node_with_domain_projection, merge_domain_projection_diagnostics
from nestynet_sr.sr_search.factorized_search.engine.scoring import score_expr as _score_expr_diagnostic

from ._factorized_de_frontend import (
    DEFeatureGroup,
    FactorizedSearchDERescueConfig,
    FactorizedSearchDEResult,
    _DOMAIN_PROJECTION_DEFAULT_ABS_TOL,
    _DOMAIN_PROJECTION_DEFAULT_MAX_FRAC,
    _DOMAIN_PROJECTION_DEFAULT_POSITIVE_FLOOR,
    _DOMAIN_PROJECTION_DEFAULT_REL_TOL,
    _GENERATOR_WITNESS_MATERIALIZABLE_STATUSES,
    _anchor_for_order,
    _append_reason,
    _broad_row_structural_safety,
    _build_domain_eval_cloud,
    _candidate_eval_array,
    _coefficient_dim_jsonable,
    _coefficient_dim_mode,
    _de_effective_early_stop_mse,
    _domain_projection_cfg_from_candidate,
    _expr_and_mapping_from_candidate,
    _factorized_candidate_id,
    _factorized_candidate_key_payload,
    _feature_group_row_summary,
    _global_group_budgets,
    _group_to_validation_trajectory,
    _input_exprs_from_report,
    _jsonable_ast_to_tuple,
    _materialize_generator_witness_result,
    _nestynet_structural_reasons,
    _no_candidate_result_from_report,
    _node_dim_jsonable,
    _observed_derivative_scale,
    _ordered_constants_from_report,
    _partition_feature_groups,
    _report_global_candidate_rows,
    _row_mse_value,
    _row_rerank_key,
    _row_structurally_eligible,
    _safe_float,
    _safe_probe_rel_rms,
    _score_candidate_domain_fragility,
    _select_best_global,
    de_lab_spec_from_de_cfg,
    validate_order2_generator_witness,
)

if TYPE_CHECKING:
    from ._factorized_de_operator import (
        _compiled_de_ast_payload,
    )

def default_physics_rescue_hp(*, preset: str = "default") -> FactorizedSearchConfig:
    """Return factorized symbolic search hyperparameters suitable for DE rescue runs."""

    hp = oracle_de.default_oracle_de_hyperparams()
    hp.refine_enable = True
    hp.refine_joint_score_enable = False
    hp.refine_joint_enable = False
    hp.refine_joint_terms_enable = False
    hp.refine_fit_subset_mode = "stratified"
    # Match the oracle benchmark's stricter shared-RHS configuration rather
    # than the looser raw library defaults from default_oracle_de_hyperparams().
    hp.poly_degree = 1
    hp.brute_max_expressions = 50_000
    hp.complexity_penalty = 1.0e-4
    hp.mapping_complexity_penalty = 0.01
    hp.score_head_enable = False
    hp.de_score_head_policy = "proposal_only"
    hp.de_accept_hidden_score_head = False
    hp.de_score_head_untyped_enable = False
    hp.score_finite_mask_enable = True
    hp.score_finite_mask_min_fit_frac = 0.98
    hp.score_finite_mask_min_probe_frac = 0.98
    hp.score_finite_mask_min_dataset_frac = 0.95
    hp.score_finite_mask_min_points = 8
    hp.score_domain_projection_enable = True
    hp.score_domain_projection_abs_tol = _DOMAIN_PROJECTION_DEFAULT_ABS_TOL
    hp.score_domain_projection_rel_tol = _DOMAIN_PROJECTION_DEFAULT_REL_TOL
    hp.score_domain_projection_max_frac = _DOMAIN_PROJECTION_DEFAULT_MAX_FRAC
    hp.score_domain_projection_positive_floor = _DOMAIN_PROJECTION_DEFAULT_POSITIVE_FLOOR
    hp.score_mapping_family_mode = "poly_only"
    hp.brute_score_mapping_family_mode = "poly_only"
    hp.score_pade_structural_enable = True
    hp.de_sparse_combo_enable = False
    hp.de_sparse_combo_pool_topk = 8
    hp.de_sparse_combo_max_terms = 2
    hp.de_sparse_combo_beam = 16
    hp.de_sparse_combo_backward_prune = True
    hp.de_sparse_combo_ridge = 1.0e-8
    hp.de_sparse_combo_prune_rel = 1.0e-5
    hp.de_sparse_combo_mapping_mode = "affine_only"
    hp.de_sparse_combo_corr_eps = 1.0e-8
    hp.de_sparse_combo_rank_eps = 1.0e-10
    hp.de_sparse_combo_max_condition = 1.0e10
    hp.de_sparse_combo_cond_penalty = 0.0
    hp.de_sparse_combo_coeff_stability_penalty = 0.0
    hp.de_sparse_combo_coeff_spread_warn = 2.0
    hp.de_score_head_max_terms = 2

    preset_l = str(preset).strip().lower()
    if preset_l == "fast":
        hp.n_iter = 15_000
        hp.max_depth = 5
        hp.n_fit = 2_000
        hp.n_probe = 2_000
        hp.return_topk = 16
        hp.refine_fit_subset = 512
    elif preset_l == "paper":
        hp.n_iter = 60_000
        hp.max_depth = 6
        hp.n_fit = 6_000
        hp.n_probe = 6_000
        hp.return_topk = 16
        hp.refine_fit_subset = 1_024
    elif preset_l in {"compositional_fast", "sparse_combo_fast"}:
        hp.n_iter = 15_000
        hp.max_depth = 5
        hp.n_fit = 2_000
        hp.n_probe = 2_000
        hp.return_topk = 16
        hp.refine_fit_subset = 512
        hp.de_sparse_combo_enable = True
        hp.de_sparse_combo_pool_topk = 8
        hp.de_sparse_combo_max_terms = 3
        hp.de_sparse_combo_beam = 12
        hp.de_sparse_combo_backward_prune = True
        hp.de_sparse_combo_ridge = 1.0e-8
        hp.de_sparse_combo_prune_rel = 1.0e-5
        hp.de_score_head_max_terms = 2
    elif preset_l in {"compositional", "sparse_combo"}:
        hp.n_iter = 30_000
        hp.max_depth = 5
        hp.n_fit = 4_000
        hp.n_probe = 4_000
        hp.return_topk = 24
        hp.refine_fit_subset = 768
        hp.de_sparse_combo_enable = True
        hp.de_sparse_combo_pool_topk = 12
        hp.de_sparse_combo_max_terms = 4
        hp.de_sparse_combo_beam = 24
        hp.de_sparse_combo_backward_prune = True
        hp.de_sparse_combo_ridge = 1.0e-8
        hp.de_sparse_combo_prune_rel = 1.0e-5
        hp.de_score_head_max_terms = 3
    else:
        hp.n_iter = 30_000
        hp.max_depth = 5
        hp.n_fit = 4_000
        hp.n_probe = 4_000
        hp.return_topk = 16
        hp.refine_fit_subset = 768
    apply_refine_profile(hp, "rare_slate")
    return hp


def run_factorized_de_from_feature_groups(
    spec: oracle_de.DELabSpec,
    groups: Sequence[DEFeatureGroup],
    *,
    factorized_search_hp: FactorizedSearchConfig,
    seed: int,
    dtype: torch.dtype = torch.float64,
    enforce_dims: bool = True,
    verbose: bool = True,
    parallel_orders: bool = True,
    budget_scope: str = "per_group",
    coefficient_dim_mode: str = "strict_expression",
) -> dict[str, Any]:
    """Run factorized symbolic search DE search from already-built feature groups.

    Each group corresponds to one trajectory or dataset id and retains separate
    fit/probe identities for shared-physics scoring.
    """

    if not groups:
        raise ValueError("No DE feature groups were provided")

    hp = factorized_search_hp
    run_seed = int(seed)
    coeff_dim_mode = _coefficient_dim_mode(coefficient_dim_mode)
    budget_scope_l = str(budget_scope or "per_group").strip().lower()
    if budget_scope_l not in {"per_group", "global"}:
        raise ValueError(f"budget_scope must be 'per_group' or 'global', got {budget_scope!r}")
    started = time.perf_counter()
    validated_groups, fit_groups, probe_groups, probe_fallback_to_fit = _partition_feature_groups(
        groups,
        dtype=dtype,
    )
    validation_trajectories = [
        _group_to_validation_trajectory(group, spec=spec)
        for group in probe_groups
    ]
    fit_group_ids = [str(group.id) for group in fit_groups]
    probe_group_ids = [str(group.id) for group in probe_groups]
    measurement_diag: dict[str, Any] = {
        "mode": "broad_whole_rhs",
        "attempts": 1,
        "budget_scope": str(budget_scope_l),
        "n_groups_total": int(len(validated_groups)),
        "n_groups_fit": int(len(fit_groups)),
        "n_groups_probe": int(len(probe_groups)),
        "group_rows": _feature_group_row_summary(validated_groups),
        "orders": [],
    }

    def _order_fn(order: int, stop_event) -> dict[str, Any]:
        order_started = time.perf_counter()
        order_i = int(order)
        x_fit_parts: list[torch.Tensor] = []
        y_fit_parts: list[torch.Tensor] = []
        x_probe_parts: list[torch.Tensor] = []
        y_probe_parts: list[torch.Tensor] = []
        fit_meta: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        probe_meta: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        fit_row_counts: list[dict[str, Any]] = []
        probe_row_counts: list[dict[str, Any]] = []
        feat_names: list[str] | None = None
        fit_budgets = (
            _global_group_budgets(int(getattr(hp, "n_fit", 1)), len(fit_groups))
            if budget_scope_l == "global"
            else [None for _ in fit_groups]
        )
        probe_budgets = (
            _global_group_budgets(int(getattr(hp, "n_probe", 1)), len(probe_groups))
            if budget_scope_l == "global"
            else [None for _ in probe_groups]
        )

        for gi, group in enumerate(fit_groups):
            z_fit, y_fit, names = oracle_de._build_table_from_features(
                spec,
                group.features,
                order=order_i,
                split="fit",
            )
            if feat_names is None:
                feat_names = list(names)
            elif list(names) != feat_names:
                raise ValueError(
                    f"Feature-name mismatch across fit groups for order={order_i}: "
                    f"{feat_names} vs {list(names)}"
                )

            original_rows = int(z_fit.shape[0])
            take_n = fit_budgets[gi]
            if take_n is not None and int(take_n) < int(z_fit.shape[0]):
                z_fit, y_fit = oracle_de._subsample_table(
                    z_fit,
                    y_fit,
                    n_take=int(take_n),
                    seed=run_seed + order_i * 100_003 + gi * 31_337,
                )
            x_fit_parts.append(z_fit)
            y_fit_parts.append(y_fit)
            fit_meta.append((str(group.id), z_fit, y_fit))
            fit_row_counts.append(
                {
                    "id": str(group.id),
                    "original_rows": int(original_rows),
                    "rows": int(z_fit.shape[0]),
                    "target_rows": int(y_fit.shape[0]),
                    "budget": None if take_n is None else int(take_n),
                }
            )

        for gi, group in enumerate(probe_groups):
            z_probe, y_probe, names_probe = oracle_de._build_table_from_features(
                spec,
                group.features,
                order=order_i,
                split="probe",
            )
            if feat_names is None:
                feat_names = list(names_probe)
            else:
                if list(names_probe) != feat_names:
                    raise ValueError(
                        f"Feature-name mismatch across probe groups for order={order_i}: "
                        f"{feat_names} vs {list(names_probe)}"
                    )

            original_rows = int(z_probe.shape[0])
            take_n = probe_budgets[gi]
            if take_n is not None and int(take_n) < int(z_probe.shape[0]):
                z_probe, y_probe = oracle_de._subsample_table(
                    z_probe,
                    y_probe,
                    n_take=int(take_n),
                    seed=run_seed + order_i * 200_003 + gi * 41_321 + 1,
                )
            x_probe_parts.append(z_probe)
            y_probe_parts.append(y_probe)
            probe_meta.append((str(group.id), z_probe, y_probe))
            probe_row_counts.append(
                {
                    "id": str(group.id),
                    "original_rows": int(original_rows),
                    "rows": int(z_probe.shape[0]),
                    "target_rows": int(y_probe.shape[0]),
                    "budget": None if take_n is None else int(take_n),
                }
            )

        if feat_names is None:
            raise ValueError(f"No feature names available for order={order_i}")

        if not x_fit_parts or not y_fit_parts:
            raise ValueError(f"No fit feature rows available for order={order_i}")
        if not x_probe_parts or not y_probe_parts:
            raise ValueError(f"No probe feature rows available for order={order_i}")

        x_fit_cat = torch.cat(x_fit_parts, dim=0)
        y_fit_cat = torch.cat(y_fit_parts, dim=0)
        x_probe_cat = torch.cat(x_probe_parts, dim=0)
        y_probe_cat = torch.cat(y_probe_parts, dim=0)

        var_dims, y_dims = oracle_de._dims_for_order(spec, order_i)
        target_y_dims = y_dims
        if not bool(enforce_dims):
            var_dims = None
            y_dims = None
            target_y_dims = None
        elif coeff_dim_mode == "inferred_outer":
            y_dims = None

        observed_scale = _observed_derivative_scale(y_fit_parts, y_probe_parts)
        search_started = time.perf_counter()
        base_early_stop_mse = _safe_float(getattr(hp, "early_stop_mse", 0.0), default=0.0)
        effective_early_stop_mse = _de_effective_early_stop_mse(
            hp,
            observed_scale=observed_scale,
        )
        hp_search = hp
        if float(effective_early_stop_mse) != float(base_early_stop_mse):
            hp_search = copy.deepcopy(hp)
            hp_search.early_stop_mse = float(effective_early_stop_mse)
        order_diagnostics: dict[str, Any] = {}
        rows, n_seeds, n_seeds_ran, n_iter_each = oracle_de._run_order_search(
            spec=spec,
            order=order_i,
            x_fit=x_fit_cat,
            y_fit=y_fit_cat,
            x_probe=x_probe_cat,
            y_probe=y_probe_cat,
            hp=hp_search,
            dtype=dtype,
            run_seed=run_seed,
            var_dims=var_dims,
            y_dims=y_dims,
            verbose=verbose,
            fit_meta=fit_meta,
            probe_meta=probe_meta,
            traj_metric=str(spec.traj_metric),
            stop_event=stop_event,
            diagnostics_out=order_diagnostics,
        )
        search_wall_seconds = float(time.perf_counter() - search_started)
        for i, row in enumerate(rows):
            row["original_rank"] = int(i)
            row["score_rank"] = int(i)
            row["coefficient_dim_mode"] = str(coeff_dim_mode)
            row["target_scale"] = float(row.get("target_scale", observed_scale))
            mse_value = _row_mse_value(row)
            probe_rms = math.sqrt(max(0.0, mse_value)) if math.isfinite(mse_value) else float("inf")
            row["probe_rms"] = float(probe_rms)
            row["probe_rel_rms"] = _safe_probe_rel_rms(float(probe_rms), float(observed_scale))
            row["target_dim"] = None if target_y_dims is None else [float(v) for v in target_y_dims]
            expr_dim = _node_dim_jsonable(row.get("_expr_obj"), var_dims)
            row["expr_dim"] = expr_dim
            row["coefficient_dim"] = _coefficient_dim_jsonable(
                expr_dim=expr_dim,
                target_dim=target_y_dims,
                coefficient_dim_mode=coeff_dim_mode,
            )

        domain_started = time.perf_counter()
        actual_cloud, perturbed_cloud = _build_domain_eval_cloud(
            x_fit_parts,
            x_probe_parts,
            seed=run_seed + order_i * 1_000_003,
            dtype=dtype,
        )
        domain_projection_cfg = {
            "score_domain_projection_enable": bool(getattr(hp_search, "score_domain_projection_enable", False)),
            "score_domain_projection_abs_tol": float(getattr(hp_search, "score_domain_projection_abs_tol", 1.0e-8)),
            "score_domain_projection_rel_tol": float(getattr(hp_search, "score_domain_projection_rel_tol", 1.0e-8)),
            "score_domain_projection_max_frac": float(getattr(hp_search, "score_domain_projection_max_frac", 1.0)),
            "score_domain_projection_positive_floor": float(
                getattr(hp_search, "score_domain_projection_positive_floor", 1.0e-12)
            ),
        }
        for row in rows:
            row.update(
                _score_candidate_domain_fragility(
                    row.get("_expr_obj"),
                    row.get("_mapping_obj"),
                    actual_cloud=actual_cloud,
                    perturbed_cloud=perturbed_cloud,
                    observed_scale=observed_scale,
                    dtype=dtype,
                    domain_projection_cfg=domain_projection_cfg,
                )
            )
            row.update(
                _broad_row_structural_safety(
                    row,
                    expr_ast=row.get("_expr_obj"),
                    mapping=row.get("_mapping_obj"),
                    input_exprs=oracle_de._input_exprs_for_order(spec, order_i),
                    order=order_i,
                    x_axis=int(spec.x_axis),
                )
            )

        rows = sorted(rows, key=_row_rerank_key)
        domain_wall_seconds = float(time.perf_counter() - domain_started)
        integration_wall_seconds = 0.0
        val_topk = max(0, int(spec.validate_integrate_topk))
        if val_topk > 0:
            integration_started = time.perf_counter()
            for i, row in enumerate(rows):
                if i < val_topk:
                    mse_int = oracle_de._validate_candidate_by_integration(
                        spec=spec,
                        order=order_i,
                        expr_ast=row.get("_expr_obj"),
                        mapping=row.get("_mapping_obj"),
                        trajectories=validation_trajectories,
                        dtype=dtype,
                        domain_projection_cfg=domain_projection_cfg,
                    )
                    row["integrate_mse"] = float(mse_int)
                    row["integrate_ok"] = bool(math.isfinite(mse_int))
                else:
                    row["integrate_mse"] = None
                    row["integrate_ok"] = None
            integration_wall_seconds = float(time.perf_counter() - integration_started)

        for row in rows:
            row.pop("_expr_obj", None)
            row.pop("_mapping_obj", None)

        rows = sorted(rows, key=_row_rerank_key)
        for i, row in enumerate(rows):
            row["rerank_rank"] = int(i)

        best_row = rows[0] if rows else None
        order_measurement_diag = {
            "mode": "broad_whole_rhs_order",
            "order": int(order_i),
            "budget_scope": str(budget_scope_l),
            "fit_row_counts": fit_row_counts,
            "probe_row_counts": probe_row_counts,
            "n_points_fit": int(sum(int(part.shape[0]) for part in x_fit_parts)),
            "n_points_probe": int(sum(int(part.shape[0]) for part in x_probe_parts)),
            "n_points_total": int(
                sum(int(part.shape[0]) for part in x_fit_parts)
                + sum(int(part.shape[0]) for part in x_probe_parts)
            ),
            "n_results": int(len(rows)),
            "n_seeds": int(n_seeds),
            "n_seeds_ran": int(n_seeds_ran),
            "n_iter_each": int(n_iter_each),
            "validate_integrate_topk": int(val_topk),
            "target_scale": float(observed_scale),
            "base_early_stop_mse": float(base_early_stop_mse),
            "effective_early_stop_mse": float(effective_early_stop_mse),
            "max_depth": int(getattr(hp, "max_depth", -1)),
            "brute_depth": None if getattr(hp, "brute_depth", None) is None else int(hp.brute_depth),
            "search_wall_seconds": float(search_wall_seconds),
            "domain_wall_seconds": float(domain_wall_seconds),
            "integration_wall_seconds": float(integration_wall_seconds),
            "wall_seconds": float(time.perf_counter() - order_started),
            "search_diagnostics": oracle_de._to_jsonable(order_diagnostics),
        }
        return {
            "order": int(order_i),
            "nvars": int(x_fit_cat.shape[1]),
            "n_points_fit": int(sum(int(part.shape[0]) for part in x_fit_parts)),
            "n_points_probe": int(sum(int(part.shape[0]) for part in x_probe_parts)),
            "n_points_total": int(
                sum(int(part.shape[0]) for part in x_fit_parts)
                + sum(int(part.shape[0]) for part in x_probe_parts)
            ),
            "feature_names": list(feat_names or []),
            "target_name": oracle_de._target_name(spec, order_i),
            "split_mode": "prebuilt_feature_groups",
            "traj_metric": str(spec.traj_metric),
            "n_traj_total": int(len(validated_groups)),
            "n_traj_fit": int(len(fit_groups)),
            "n_traj_probe": int(len(probe_groups)),
            "fit_traj_ids": list(fit_group_ids),
            "probe_traj_ids": list(probe_group_ids),
            "probe_fallback_to_fit": bool(probe_fallback_to_fit),
            "var_dims": None if var_dims is None else [list(d) for d in var_dims],
            "target_y_dims": None if target_y_dims is None else list(target_y_dims),
            "y_dims": None if y_dims is None else list(y_dims),
            "coefficient_dim_mode": str(coeff_dim_mode),
            "n_seeds": int(n_seeds),
            "n_seeds_ran": int(n_seeds_ran),
            "n_iter_each": int(n_iter_each),
            "validate_integrate_topk": int(val_topk),
            "search_diagnostics": oracle_de._to_jsonable(order_diagnostics),
            "factorized_de_diagnostics": order_measurement_diag,
            "best_selection_mode": "integrate_rerank" if int(val_topk) > 0 else "score_rank",
            "best_original_rank": None if best_row is None else int(best_row.get("original_rank", -1)),
            "best_rerank_rank": None if best_row is None else int(best_row.get("rerank_rank", -1)),
            "results": rows,
            "best": best_row,
        }

    per_order = oracle_de._dispatch_orders_parallel(
        spec.order_candidates,
        _order_fn,
        parallel=parallel_orders,
        verbose=verbose,
    )
    best_global = _select_best_global(spec, per_order)
    elapsed = float(time.perf_counter() - started)
    order_measurements = [
        dict(row.get("factorized_de_diagnostics", {}) or {})
        for row in per_order
        if isinstance(row, dict)
    ]
    measurement_diag["orders"] = order_measurements
    measurement_diag["n_orders"] = int(len(order_measurements))
    measurement_diag["search_diagnostics_summary"] = oracle_de._to_jsonable(
        oracle_de._search_diagnostics_summary(order_measurements)
    )
    measurement_diag["effective_fit_rows"] = int(
        sum(int(row.get("n_points_fit", 0) or 0) for row in order_measurements)
    )
    measurement_diag["effective_probe_rows"] = int(
        sum(int(row.get("n_points_probe", 0) or 0) for row in order_measurements)
    )
    measurement_diag["effective_total_rows"] = int(
        sum(int(row.get("n_points_total", 0) or 0) for row in order_measurements)
    )
    measurement_diag["wall_seconds"] = float(elapsed)

    report = {
        "spec_id": spec.id,
        "csv_paths": list(spec.csv_paths),
        "order_candidates": [int(o) for o in spec.order_candidates],
        "x_axis": int(spec.x_axis),
        "include_x": bool(spec.include_x),
        "include_u": bool(spec.include_u),
        "include_du": bool(spec.include_du),
        "x_col": str(spec.x_col),
        "u_col": str(spec.u_col),
        "out_idx": int(spec.out_idx),
        "y_transform": str(spec.y_transform),
        "split_mode": "prebuilt_feature_groups",
        "budget_scope": str(budget_scope_l),
        "traj_metric": str(spec.traj_metric),
        "trajectories": [{"id": str(group.id), "csv": None} for group in validated_groups],
        "fit_trajectories": [{"id": str(group.id), "csv": None} for group in fit_groups],
        "probe_trajectories": [{"id": str(group.id), "csv": None} for group in probe_groups],
        "probe_fallback_to_fit": bool(probe_fallback_to_fit),
        "constants": {c.name: float(c.value) for c in spec.constants},
        "constants_ordered": [
            {"name": str(c.name), "value": float(c.value)}
            for c in spec.constants
        ],
        "deriv": {
            "method": str(spec.derivative.method),
            "s": float(spec.derivative.spline_s),
            "k": int(spec.derivative.spline_k),
            "du_col": spec.derivative.du_col,
            "d2u_col": spec.derivative.d2u_col,
        },
        "dims": None
        if spec.dims is None
        else {
            "basis": list(spec.dims.basis),
            "x": list(spec.dims.x_dim),
            "u": list(spec.dims.u_dim),
        },
        "extra": None if spec.extra is None else oracle_de._to_jsonable(spec.extra),
        "dtype": str(dtype),
        "enforce_dims": bool(enforce_dims),
        "coefficient_dim_mode": str(coeff_dim_mode),
        "seed": int(run_seed),
        "wall_seconds": elapsed,
        "factorized_de_diagnostics": oracle_de._to_jsonable(measurement_diag),
        "resolved_config": oracle_de._to_jsonable(factorized_config_report(hp)),
        "hp": {
            "n_iter": int(hp.n_iter),
            "max_depth": int(hp.max_depth),
            "poly_degree": int(hp.poly_degree),
            "return_topk": int(hp.return_topk),
            "n_fit": int(hp.n_fit),
            "n_probe": int(hp.n_probe),
            "n_seeds": int(hp.n_seeds),
            "split_iter_across_seeds": bool(hp.split_iter_across_seeds),
            "brute_depth": None if hp.brute_depth is None else int(hp.brute_depth),
            "early_stop_mse": float(hp.early_stop_mse),
            "brute_max_expressions": int(hp.brute_max_expressions),
            "refine_enable": bool(hp.refine_enable),
            "refine_profile": str(getattr(hp, "refine_profile", "default")),
            "refine_mode": str(getattr(hp, "refine_mode", "slate")),
            "refine_during_brute": bool(getattr(hp, "refine_during_brute", False)),
            "refine_during_mutation": bool(getattr(hp, "refine_during_mutation", False)),
            "refine_during_slate": bool(getattr(hp, "refine_during_slate", True)),
            "refine_slate_budget": int(getattr(hp, "refine_slate_budget", 32)),
            "refine_optimizer": str(getattr(hp, "refine_optimizer", "lbfgs")),
            "refine_lbfgs_escalate_improve_factor": float(
                getattr(hp, "refine_lbfgs_escalate_improve_factor", 2.0)
            ),
            "refine_lbfgs_steps": int(hp.refine_lbfgs_steps),
            "refine_num_restarts": int(hp.refine_num_restarts),
            "refine_max_variants": int(hp.refine_max_variants),
            "refine_max_params": int(hp.refine_max_params),
            "refine_linear_combo_enable": bool(hp.refine_linear_combo_enable),
            "refine_gate_best_factor": float(hp.refine_gate_best_factor),
            "refine_max_trials": int(hp.refine_max_trials),
            "complexity_penalty": float(getattr(hp, "complexity_penalty", 0.0)),
            "mapping_complexity_penalty": float(getattr(hp, "mapping_complexity_penalty", 0.0)),
            "score_head_enable": bool(getattr(hp, "score_head_enable", False)),
            "de_sparse_combo_enable": bool(getattr(hp, "de_sparse_combo_enable", False)),
            "de_sparse_combo_pool_topk": int(getattr(hp, "de_sparse_combo_pool_topk", 8)),
            "de_sparse_combo_max_terms": int(getattr(hp, "de_sparse_combo_max_terms", 2)),
            "de_sparse_combo_beam": int(getattr(hp, "de_sparse_combo_beam", 16)),
            "de_sparse_combo_backward_prune": bool(getattr(hp, "de_sparse_combo_backward_prune", True)),
            "de_score_head_policy": str(getattr(hp, "de_score_head_policy", "proposal_only")),
            "de_accept_hidden_score_head": bool(getattr(hp, "de_accept_hidden_score_head", False)),
            "de_score_head_untyped_enable": bool(getattr(hp, "de_score_head_untyped_enable", False)),
            "de_score_head_max_terms": int(getattr(hp, "de_score_head_max_terms", 2)),
        },
        "per_order": per_order,
        "best": best_global,
    }
    return report


def factorized_search_report_to_de_result(report: dict[str, Any]) -> FactorizedSearchDEResult:
    """Convert a factorized symbolic search DE report into a normalized DE-facing result."""

    best = report.get("best", None)
    if not isinstance(best, dict):
        return _no_candidate_result_from_report(
            report,
            reason="no_factorized_search_candidate",
        )
    feature_names_hint: list[str] | None = None
    report_for_diagnostics = report
    if not _row_structurally_eligible(best):
        replacement = None
        for _, row, feature_names in _report_global_candidate_rows(report):
            if _row_structurally_eligible(row):
                replacement = row
                feature_names_hint = list(feature_names or [])
                break
        if replacement is None:
            return _no_candidate_result_from_report(
                report,
                reason="no_structurally_eligible_factorized_search_candidate",
            )
        best = replacement
        report_for_diagnostics = dict(report)
        report_for_diagnostics["best"] = best
        report_for_diagnostics["best_replaced_due_to_structural_reject"] = True

    order = int(best.get("order", -1))
    if order not in (1, 2):
        raise ValueError(f"Unsupported factorized symbolic search DE order: {order}")

    mapping = best.get("mapping", None)
    if not isinstance(mapping, dict):
        return _no_candidate_result_from_report(
            report,
            order=order,
            reason="factorized symbolic search DE report best candidate is missing a mapping dict",
        )

    expr_ast = _jsonable_ast_to_tuple(best.get("expr_ast", None))
    if expr_ast is None:
        return _no_candidate_result_from_report(
            report,
            order=order,
            reason="factorized symbolic search DE report best candidate is missing expr_ast",
        )

    rhs_ast = None
    residual_ast = None
    try:
        inner_nn = factorized_search_to_nestynet(expr_ast)
        rhs_ast = embed_mapping_in_ast(
            inner_nn,
            mapping,
            _input_exprs_from_report(report, order=order),
            units_mode="raw",
        )
        if rhs_ast is not None:
            anchor = _anchor_for_order(order, x_axis=int(report.get("x_axis", 0)))
            residual_ast = Add(anchor, Mul(ConstNode(-1.0), rhs_ast))
    except Exception:
        rhs_ast = None
        residual_ast = None

    feature_names: list[str] = list(feature_names_hint or [])
    for row in report.get("per_order", []):
        if feature_names:
            break
        if int(row.get("order", -1)) == order:
            feature_names = list(row.get("feature_names", []) or [])
            break

    if rhs_ast is None or residual_ast is None:
        return _no_candidate_result_from_report(
            report,
            order=order,
            feature_names=feature_names,
            reason="factorized symbolic search DE report best candidate did not materialize rhs/residual AST",
        )
    compiled_structural_reasons: list[str] = []
    for node in (rhs_ast, residual_ast):
        for reason in _nestynet_structural_reasons(node):
            _append_reason(compiled_structural_reasons, reason)
    if compiled_structural_reasons:
        return _no_candidate_result_from_report(
            report,
            order=order,
            feature_names=feature_names,
            reason="structurally_invalid_materialized_factorized_search_candidate:"
            + ",".join(compiled_structural_reasons),
        )
    compiled_payload = _compiled_de_ast_payload(rhs_ast=rhs_ast, residual_ast=residual_ast)
    rhs_ast_raw = compiled_payload["rhs_ast_raw"]
    residual_ast_raw = compiled_payload["residual_ast_raw"]
    rhs_ast_simplified = compiled_payload["rhs_ast_simplified"]
    residual_ast_simplified = compiled_payload["residual_ast_simplified"]

    probe_mse = float(best.get("mse", float("inf")))
    probe_rms = math.sqrt(max(0.0, probe_mse)) if math.isfinite(probe_mse) else float("inf")
    canonical_equation_raw = str(
        best.get("residual_ast_raw", "")
        or (repr(residual_ast_raw) if residual_ast_raw is not None else best.get("expr", ""))
    )
    canonical_equation_simplified = str(
        best.get("residual_ast_simplified", "")
        or (repr(residual_ast_simplified) if residual_ast_simplified is not None else canonical_equation_raw)
    )
    canonical_equation = canonical_equation_simplified
    candidate_id = _factorized_candidate_id(best)

    target_scale = _safe_float(best.get("target_scale", None), default=float("nan"))
    probe_rel_rms = _safe_float(best.get("probe_rel_rms", None), default=float("inf"))
    domain_projection = best.get("domain_projection", None)
    best_mapping_payload = best.get("mapping", None)
    if not isinstance(domain_projection, Mapping) and isinstance(best_mapping_payload, Mapping):
        mapping_domain_projection = best_mapping_payload.get("_domain_projection", None)
        if isinstance(mapping_domain_projection, Mapping):
            domain_projection = mapping_domain_projection
    diagnostics = {
        "score": float(best.get("score", probe_mse)),
        "score_raw": float(best.get("score_raw", probe_mse)),
        "size": int(best.get("size", 0)),
        "symbolic_size_raw": best.get("symbolic_size_raw", compiled_payload["symbolic_size_raw"]),
        "symbolic_size_simplified": best.get(
            "symbolic_size_simplified",
            compiled_payload["symbolic_size_simplified"],
        ),
        "mapping_complexity": int(best.get("mapping_complexity", 0)),
        "canonical_equation_raw": canonical_equation_raw,
        "canonical_equation_simplified": canonical_equation_simplified,
        "candidate_id": candidate_id,
        "candidate_key": _factorized_candidate_key_payload(best),
        "target_scale": None if not math.isfinite(target_scale) else float(target_scale),
        "probe_rel_rms": None if not math.isfinite(probe_rel_rms) else float(probe_rel_rms),
        "candidate_source": best.get("candidate_source", None),
        "integrate_ok": None if best.get("integrate_ok", None) is None else bool(best.get("integrate_ok")),
        "integrate_mse": _safe_float(best.get("integrate_mse", None), default=float("inf")),
        "domain_ok": None if best.get("domain_ok", None) is None else bool(best.get("domain_ok")),
        "domain_fragility_penalty": float(best.get("domain_fragility_penalty", 0.0)),
        "domain_failure_reason": best.get("domain_failure_reason", None),
        "finite_mask": best.get("finite_mask", None),
        "domain_projection": domain_projection,
        "structural_ok": bool(best.get("structural_ok", True)),
        "structural_hard_reject": bool(best.get("structural_hard_reject", False)),
        "structural_reasons": list(best.get("structural_reasons", []) or []),
        "structural_gate_version": best.get("structural_gate_version", None),
        "include_x": bool(report.get("include_x", True)),
        "include_u": bool(report.get("include_u", True)),
        "include_du": bool(report.get("include_du", True)),
        "n_traj_total": int(len(report.get("trajectories", []) or [])),
        "fit_traj_ids": [
            str(t.get("id")) for t in (report.get("fit_trajectories", []) or []) if isinstance(t, dict)
        ],
        "probe_traj_ids": [
            str(t.get("id")) for t in (report.get("probe_trajectories", []) or []) if isinstance(t, dict)
        ],
        "factorized_de_diagnostics": report.get("factorized_de_diagnostics", {}),
        "report": report_for_diagnostics,
    }

    return FactorizedSearchDEResult(
        order=order,
        x_axis=int(report.get("x_axis", 0)),
        rhs_ast=rhs_ast_simplified,
        residual_ast=residual_ast_simplified,
        canonical_equation=canonical_equation,
        probe_mse=probe_mse,
        probe_rms=probe_rms,
        expr_ast=expr_ast,
        mapping=dict(mapping),
        mapping_kind=str(best.get("mapping_kind", "")),
        feature_names=feature_names,
        diagnostics=diagnostics,
        rhs_ast_raw=rhs_ast_raw,
        residual_ast_raw=residual_ast_raw,
        rhs_ast_simplified=rhs_ast_simplified,
        residual_ast_simplified=residual_ast_simplified,
        canonical_equation_raw=canonical_equation_raw,
        canonical_equation_simplified=canonical_equation_simplified,
    )


def factorized_search_report_shortlist(
    report: dict[str, Any],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Serialize a global factorized symbolic search shortlist from a DE report.

    The returned candidates are self-contained enough for
    ``factorized_search_report_to_rhs_callable()`` to reconstruct the RHS without the
    original full report object.
    """

    if not isinstance(report, dict):
        raise TypeError("report must be a dict")

    try:
        limit_i = None if limit is None else max(0, int(limit))
    except Exception:
        limit_i = None

    constants_ordered = _ordered_constants_from_report(report)
    include_x = bool(report.get("include_x", True))
    include_u = bool(report.get("include_u", True))
    include_du = bool(report.get("include_du", True))
    x_axis = int(report.get("x_axis", 0))

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, row, feature_names in _report_global_candidate_rows(report):
        key = _factorized_candidate_key_payload(row)
        if key in seen:
            continue
        seen.add(key)
        candidate_id = _factorized_candidate_id(row)

        mse = _safe_float(row.get("mse", None), default=float("inf"))
        canonical_equation_raw = str(row.get("residual_ast_raw", "") or row.get("residual_ast", "") or row.get("expr", ""))
        canonical_equation_simplified = str(
            row.get("residual_ast_simplified", "") or row.get("residual_ast", "") or row.get("expr", "")
        )
        canonical_equation = canonical_equation_simplified or canonical_equation_raw
        domain_projection = row.get("domain_projection", None)
        mapping_payload = row.get("mapping", None)
        if not isinstance(domain_projection, Mapping) and isinstance(mapping_payload, Mapping):
            mapping_domain_projection = mapping_payload.get("_domain_projection", None)
            if isinstance(mapping_domain_projection, Mapping):
                domain_projection = mapping_domain_projection
        payload = {
            "engine": "factorized_search",
            "kind": "factorized",
            "candidate_id": candidate_id,
            "candidate_key": key,
            "order": int(row.get("order", -1)),
            "x_axis": int(x_axis),
            "include_x": bool(include_x),
            "include_u": bool(include_u),
            "include_du": bool(include_du),
            "constants_ordered": oracle_de._to_jsonable(constants_ordered),
            "feature_names": list(feature_names or []),
            "expr_ast": oracle_de._to_jsonable(row.get("expr_ast", None)),
            "mapping": oracle_de._to_jsonable(row.get("mapping", None)),
            "mapping_kind": str(row.get("mapping_kind", "")),
            "score": oracle_de._to_jsonable(row.get("score", None)),
            "score_raw": oracle_de._to_jsonable(row.get("score_raw", None)),
            "probe_mse": None if not math.isfinite(mse) else float(mse),
            "probe_rms": math.sqrt(max(0.0, mse)) if math.isfinite(mse) else float("inf"),
            "mse": oracle_de._to_jsonable(row.get("mse", None)),
            "size": oracle_de._to_jsonable(row.get("size", None)),
            "symbolic_size_raw": oracle_de._to_jsonable(row.get("symbolic_size_raw", None)),
            "symbolic_size_simplified": oracle_de._to_jsonable(row.get("symbolic_size_simplified", None)),
            "mapping_complexity": oracle_de._to_jsonable(row.get("mapping_complexity", None)),
            "canonical_equation": canonical_equation,
            "canonical_equation_raw": canonical_equation_raw,
            "canonical_equation_simplified": canonical_equation_simplified,
            "residual_ast": row.get("residual_ast", None),
            "residual_ast_raw": row.get("residual_ast_raw", None),
            "residual_ast_simplified": row.get("residual_ast_simplified", None),
            "rhs_ast_raw": row.get("rhs_mapped_ast_raw", row.get("rhs_ast_raw", None)),
            "rhs_ast_simplified": row.get("rhs_mapped_ast_simplified", row.get("rhs_ast_simplified", None)),
            "expr": row.get("expr", None),
            "original_rank": oracle_de._to_jsonable(row.get("original_rank", None)),
            "score_rank": oracle_de._to_jsonable(row.get("score_rank", None)),
            "rerank_rank": oracle_de._to_jsonable(row.get("rerank_rank", None)),
            "integrate_ok": oracle_de._to_jsonable(row.get("integrate_ok", None)),
            "integrate_mse": oracle_de._to_jsonable(row.get("integrate_mse", None)),
            "domain_ok": oracle_de._to_jsonable(row.get("domain_ok", None)),
            "domain_failure_reason": row.get("domain_failure_reason", None),
            "domain_fragility_penalty": oracle_de._to_jsonable(row.get("domain_fragility_penalty", None)),
            "finite_mask": oracle_de._to_jsonable(row.get("finite_mask", None)),
            "domain_projection": oracle_de._to_jsonable(domain_projection),
            "structural_ok": oracle_de._to_jsonable(row.get("structural_ok", None)),
            "structural_hard_reject": oracle_de._to_jsonable(row.get("structural_hard_reject", None)),
            "structural_reasons": oracle_de._to_jsonable(row.get("structural_reasons", None)),
            "structural_gate_version": oracle_de._to_jsonable(row.get("structural_gate_version", None)),
        }
        payload["shortlist_rank"] = int(len(out))
        payload["candidate_rank"] = int(payload["shortlist_rank"])
        out.append(payload)
        if limit_i is not None and int(len(out)) >= int(limit_i):
            break
    return out


def normalized_rmse(y_true: Any, y_pred: Any) -> float:
    y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    num = float(np.linalg.norm(y_true_arr - y_pred_arr))
    den = float(np.linalg.norm(y_true_arr) + 1.0e-12)
    return float(num / den)


def evaluate_factorized_search_candidate(
    candidate: dict[str, Any],
    features: Any,
    *,
    dtype: torch.dtype = torch.float64,
) -> np.ndarray:
    """Evaluate a factorized symbolic search candidate on one row or a batch of feature rows."""
    expr_ast, mapping = _expr_and_mapping_from_candidate(candidate)
    vals = _candidate_eval_array(
        expr_ast,
        mapping,
        features,
        dtype=dtype,
        domain_projection_cfg=_domain_projection_cfg_from_candidate(candidate),
    )
    if not np.isfinite(vals).all():
        raise FloatingPointError("Non-finite factorized symbolic search candidate evaluation")
    return vals


def factorized_search_candidate_to_feature_predictor(
    candidate: dict[str, Any],
    *,
    dtype: torch.dtype = torch.float64,
):
    """Compile a factorized symbolic search candidate into a single-row feature predictor."""

    def _predict(features_1d: Sequence[float]) -> float:
        vals = evaluate_factorized_search_candidate(candidate, list(features_1d), dtype=dtype)
        val = float(vals[0])
        if not math.isfinite(val):
            raise FloatingPointError(f"Non-finite discovered RHS value ({val})")
        return val

    return _predict


def factorized_search_report_to_rhs_callable(report: dict[str, Any]) -> tuple[int, Any]:
    """Compile a factorized symbolic search DE report or serialized candidate into an RHS callable."""

    if not isinstance(report, dict):
        raise TypeError("report must be a dict")

    best = report.get("best", None)
    meta = report
    if not isinstance(best, dict):
        best = report
        diagnostics = report.get("diagnostics", None)
        if isinstance(diagnostics, dict) and isinstance(diagnostics.get("report", None), dict):
            meta = diagnostics["report"]

    order = int(best.get("order", meta.get("order", -1)))
    if order not in (1, 2):
        raise ValueError(f"Unsupported discovered order: {order}")

    const_specs = _ordered_constants_from_report(best) or _ordered_constants_from_report(meta)
    const_vals = [float(c["value"]) for c in const_specs]

    feat_names = list(best.get("feature_names", []) or meta.get("feature_names", []) or [])
    n_const = len(const_specs)
    variable_feat_names = feat_names[: max(0, len(feat_names) - n_const)] if n_const else feat_names
    variable_feat_names_l = [str(name).strip().lower() for name in variable_feat_names]

    def _explicit_switch(name: str) -> Any:
        for source in (best, meta):
            if isinstance(source, dict) and source.get(name, None) is not None:
                return source.get(name)
        return None

    include_x = _explicit_switch("include_x")
    include_u = _explicit_switch("include_u")
    include_du = _explicit_switch("include_du")
    if include_x is None:
        include_x = bool(
            variable_feat_names_l
            and any(name not in {"u", "du", "dudx", "u_dot", "udot"} for name in variable_feat_names_l)
        )
    if include_u is None:
        include_u = (
            any(name in {"u", "y"} for name in variable_feat_names_l)
            if variable_feat_names_l
            else True
        )
    if include_du is None:
        include_du = (
            any(name in {"du", "dudx", "u_dot", "udot"} for name in variable_feat_names_l)
            if variable_feat_names_l
            else True
        )
    include_x = bool(include_x)
    include_u = bool(include_u)
    include_du = bool(include_du)

    predict_rhs = factorized_search_candidate_to_feature_predictor(best, dtype=torch.float64)

    def _predict_rhs(x: float, u: float, du: float) -> float:
        feats: list[float] = []
        if include_x:
            feats.append(float(x))
        if include_u:
            feats.append(float(u))
        if int(order) == 2 and include_du:
            feats.append(float(du))
        feats.extend(const_vals)
        return predict_rhs(feats)

    if int(order) == 1:
        def rhs1(x: float, s: Sequence[float]) -> list[float]:
            return [_predict_rhs(float(x), float(s[0]), 0.0)]

        return 1, rhs1

    def rhs2(x: float, s: Sequence[float]) -> list[float]:
        u = float(s[0])
        du = float(s[1])
        return [du, _predict_rhs(float(x), u, du)]

    return 2, rhs2


def _apply_rescue_cfg_to_hp(rescue_cfg: FactorizedSearchDERescueConfig) -> FactorizedSearchConfig:
    hp = copy.deepcopy(rescue_cfg.hp) if rescue_cfg.hp is not None else default_physics_rescue_hp()
    if bool(rescue_cfg.strict_shared_rhs):
        hp.refine_joint_score_enable = False
        hp.refine_joint_enable = False
        hp.refine_joint_terms_enable = False
    return hp


def _build_feature_group_from_surrogate(
    surrogate,
    train_loader,
    val_loader,
    *,
    group_id: str,
    spec: oracle_de.DELabSpec,
    hp: FactorizedSearchConfig,
    seed: int,
    device,
    dtype: torch.dtype,
) -> DEFeatureGroup:
    provider = oracle_de.SurrogateDerivProvider(
        surrogate,
        y_transform=str(spec.y_transform),
        out_idx=int(spec.out_idx),
    )
    features = provider.build_features_from_loaders(
        train_loader,
        val_loader,
        spec=spec,
        seed=int(seed),
        dtype=dtype,
        device=device,
        n_fit=int(hp.n_fit),
        n_probe=int(hp.n_probe),
    )
    return DEFeatureGroup(id=str(group_id), features=features)


def build_factorized_search_de_feature_groups_from_surrogate(
    surrogate,
    train_loader,
    val_loader,
    *,
    cfg,
    rescue_cfg: FactorizedSearchDERescueConfig,
    device,
    dtype: torch.dtype = torch.float64,
    group_id: str = "dataset0",
) -> list[DEFeatureGroup]:
    """Build reusable DE feature groups once from a surrogate/loaders pair."""

    hp = _apply_rescue_cfg_to_hp(rescue_cfg)
    spec = de_lab_spec_from_de_cfg(cfg)
    group = _build_feature_group_from_surrogate(
        surrogate,
        train_loader,
        val_loader,
        group_id=str(group_id),
        spec=spec,
        hp=hp,
        seed=int(hp.seed),
        device=device,
        dtype=dtype,
    )
    return [group]


def build_factorized_search_de_feature_groups_from_surrogates(
    surrogates,
    train_loaders,
    val_loaders,
    *,
    cfg,
    rescue_cfg: FactorizedSearchDERescueConfig,
    device,
    dataset_ids: Sequence[str] | None = None,
    dtype: torch.dtype = torch.float64,
) -> list[DEFeatureGroup]:
    """Build reusable DE feature groups once from multiple surrogates/loaders."""

    if len(surrogates) != len(train_loaders):
        raise ValueError("surrogates and train_loaders must have the same length")
    if val_loaders is not None and len(val_loaders) != len(surrogates):
        raise ValueError("val_loaders and surrogates must have the same length")
    if len(surrogates) == 0:
        raise ValueError("No surrogates were provided")

    hp = _apply_rescue_cfg_to_hp(rescue_cfg)
    spec = de_lab_spec_from_de_cfg(cfg)
    ids = list(dataset_ids) if dataset_ids is not None else [f"dataset{i}" for i in range(len(surrogates))]
    if len(ids) != len(surrogates):
        raise ValueError("dataset_ids must match the number of surrogates")

    return [
        _build_feature_group_from_surrogate(
            surrogate,
            tr_loader,
            None if val_loaders is None else val_loaders[i],
            group_id=str(ids[i]),
            spec=spec,
            hp=hp,
            seed=int(hp.seed) + i * 1_000_003,
            device=device,
            dtype=dtype,
        )
        for i, (surrogate, tr_loader) in enumerate(zip(surrogates, train_loaders))
    ]


def run_factorized_search_de_from_feature_groups(
    groups: Sequence[DEFeatureGroup],
    *,
    cfg,
    rescue_cfg: FactorizedSearchDERescueConfig,
    dtype: torch.dtype = torch.float64,
) -> FactorizedSearchDEResult:
    """Run DE-facing factorized symbolic search from prebuilt reusable feature groups."""

    hp = _apply_rescue_cfg_to_hp(rescue_cfg)
    spec = de_lab_spec_from_de_cfg(cfg)
    if int(getattr(rescue_cfg, "validate_integrate_topk", 0)) > 0:
        spec = replace(spec, validate_integrate_topk=int(rescue_cfg.validate_integrate_topk))

    report = run_factorized_de_from_feature_groups(
        spec,
        list(groups),
        factorized_search_hp=hp,
        seed=int(hp.seed),
        dtype=dtype,
        enforce_dims=bool(getattr(cfg, "enforce_units", False)),
        verbose=False,
        parallel_orders=True,
        budget_scope=str(getattr(rescue_cfg, "budget_scope", "per_group")),
        coefficient_dim_mode=str(getattr(rescue_cfg, "coefficient_dim_mode", "strict_expression")),
    )
    return factorized_search_report_to_de_result(report)


def _direct_residual_attempt_hp(
    hp: FactorizedSearchConfig,
    *,
    autonomous: bool,
) -> FactorizedSearchConfig:
    out = copy.deepcopy(hp)
    if bool(autonomous):
        out.max_depth = min(int(getattr(out, "max_depth", 5)), 3)
        out.n_iter = min(int(getattr(out, "n_iter", 30_000)), 3_000)
    else:
        out.max_depth = min(int(getattr(out, "max_depth", 5)), 4)
        out.n_iter = min(int(getattr(out, "n_iter", 30_000)), 10_000)
    out.brute_depth = int(out.max_depth)
    out.plateau_stop_enable = True
    out.plateau_stop_max_soft_restarts = 2
    out.plateau_stop_min_evals = 2000
    out.return_topk = max(int(getattr(out, "return_topk", 8)), 16)
    out.brute_max_expressions = max(int(getattr(out, "brute_max_expressions", 50_000)), 50_000)
    out.de_sparse_combo_enable = True
    out.de_sparse_combo_pre_mutation_enable = True
    out.de_sparse_combo_pool_topk = max(
        int(getattr(out, "de_sparse_combo_pool_topk", 8)),
        16 if bool(autonomous) else 32,
    )
    out.de_sparse_combo_max_terms = max(int(getattr(out, "de_sparse_combo_max_terms", 2)), 4)
    out.de_sparse_combo_beam = max(int(getattr(out, "de_sparse_combo_beam", 16)), 32)
    out.de_sparse_combo_backward_prune = True
    out.de_sparse_combo_ridge = 1.0e-8
    out.de_sparse_combo_prune_rel = 1.0e-5
    out.de_sparse_combo_mapping_mode = "affine_only"
    out.de_sparse_combo_corr_eps = max(float(getattr(out, "de_sparse_combo_corr_eps", 1.0e-8)), 1.0e-8)
    out.de_sparse_combo_rank_eps = max(float(getattr(out, "de_sparse_combo_rank_eps", 1.0e-10)), 1.0e-10)
    out.de_sparse_combo_max_condition = min(
        float(getattr(out, "de_sparse_combo_max_condition", 1.0e10)),
        1.0e10,
    )
    # The direct DE lane is intentionally shallow-first.  Cheap prescreening can
    # reject domain-projected atoms such as sqrt(u) before the projection-aware
    # scorer gets to fit the candidate, so score direct grammar atoms fully.
    out.score_prescreen_enable = False
    out.score_prescreen_residual_allow_hint = False
    out.score_prescreen_residual_use_global_best = False
    return out


def _direct_residual_attempt_specs(
    spec: oracle_de.DELabSpec,
    cfg,
) -> list[tuple[str, oracle_de.DELabSpec, bool]]:
    attempts: list[tuple[str, oracle_de.DELabSpec, bool]] = []
    order_candidates = [int(o) for o in getattr(cfg, "order_candidates", getattr(spec, "order_candidates", (1, 2)))]
    for order in order_candidates:
        if int(order) not in (1, 2):
            continue
        if bool(getattr(cfg, "include_u", True)):
            attempts.append(
                (
                    f"order{int(order)}_autonomous",
                    replace(
                        spec,
                        order_candidates=(int(order),),
                        include_x=False,
                        include_u=True,
                        include_du=bool(int(order) == 2 and getattr(cfg, "include_du", True)),
                    ),
                    True,
                )
            )
        attempts.append(
            (
                f"order{int(order)}_full",
                replace(
                    spec,
                    order_candidates=(int(order),),
                    include_x=bool(getattr(cfg, "include_x", True)),
                    include_u=bool(getattr(cfg, "include_u", True)),
                    include_du=bool(int(order) == 2 and getattr(cfg, "include_du", True)),
                ),
                False,
            )
        )

    # Drop unusable attempts that would have no variable feature. Fixed constants
    # remain part of the spec and are appended by oracle_lab_de.
    filtered: list[tuple[str, oracle_de.DELabSpec, bool]] = []
    for label, attempt_spec, autonomous in attempts:
        has_order_feature = bool(attempt_spec.include_x or attempt_spec.include_u or attempt_spec.include_du)
        if has_order_feature or len(attempt_spec.constants) > 0:
            filtered.append((label, attempt_spec, autonomous))
    return filtered


def _direct_residual_canary_diagnostics(
    groups: Sequence[DEFeatureGroup],
    *,
    spec: oracle_de.DELabSpec,
    hp: FactorizedSearchConfig,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    """Score shallow DE atoms as diagnostics only.

    These rows do not enter the FSS archive or benchmark shortlist.  They answer
    whether expected shallow atoms are scoreable under strict, finite-mask, and
    projection policies on the same feature tables used by the direct lane.
    """

    try:
        order = int(next(iter(spec.order_candidates)))
    except Exception:
        order = 1
    try:
        _, fit_groups, probe_groups, _ = _partition_feature_groups(groups, dtype=dtype)
        x_fit_parts = []
        y_fit_parts = []
        x_probe_parts = []
        y_probe_parts = []
        feature_names = None
        for group in fit_groups:
            z_fit, y_fit, names = oracle_de._build_table_from_features(
                spec,
                group.features,
                order=order,
                split="fit",
            )
            x_fit_parts.append(z_fit)
            y_fit_parts.append(y_fit)
            feature_names = list(names)
        for group in probe_groups:
            z_probe, y_probe, names = oracle_de._build_table_from_features(
                spec,
                group.features,
                order=order,
                split="probe",
            )
            x_probe_parts.append(z_probe)
            y_probe_parts.append(y_probe)
            if feature_names is None:
                feature_names = list(names)
        if not x_fit_parts or not x_probe_parts or not feature_names:
            return []
        x_fit = torch.cat(x_fit_parts, dim=0)
        y_fit = torch.cat(y_fit_parts, dim=0)
        x_probe = torch.cat(x_probe_parts, dim=0)
        y_probe = torch.cat(y_probe_parts, dim=0)
    except Exception as exc:
        return [{"status": "canary_table_error", "error": type(exc).__name__}]

    nodes: list[tuple[str, Any]] = []
    try:
        u_idx = list(feature_names).index(str(spec.u_col))
    except Exception:
        u_idx = -1
    if u_idx >= 0:
        u_node = ("var", int(u_idx))
        nodes.append(("u", u_node))
        nodes.append(("sqrt(u)", ("sqrt", u_node)))
        try:
            x_idx = list(feature_names).index(str(spec.x_col))
        except Exception:
            x_idx = -1
        if x_idx >= 0:
            x_node = ("var", int(x_idx))
            x2_node = ("mul", x_node, x_node)
            nodes.append(("u/x", ("div", u_node, x_node)))
            nodes.append(("u/x^2", ("div", u_node, x2_node)))
            nodes.append(("x*u", ("mul", x_node, u_node)))
        try:
            du_idx = list(feature_names).index(str(spec.derivative.du_col or "du"))
        except Exception:
            du_idx = -1
        if du_idx >= 0 and x_idx >= 0:
            du_node = ("var", int(du_idx))
            nodes.append(("du/x", ("div", du_node, ("var", int(x_idx)))))
    if not nodes:
        return [{"status": "canary_no_u_feature", "feature_names": list(feature_names)}]

    gen = torch.Generator(device=x_probe.device)
    gen.manual_seed(8_675_309)
    proj = torch.randn((int(x_probe.shape[0]), 8), generator=gen, dtype=x_probe.dtype, device=x_probe.device)

    def _cfg_variant(kind: str) -> dict[str, Any]:
        cfg = {
            "score_mapping_family_mode": "poly_only",
            "score_prescreen_enable": False,
        }
        if kind in {"finite_mask", "projection"}:
            cfg.update(
                {
                    "score_finite_mask_enable": bool(getattr(hp, "score_finite_mask_enable", False)),
                    "score_finite_mask_min_fit_frac": float(getattr(hp, "score_finite_mask_min_fit_frac", 0.98)),
                    "score_finite_mask_min_probe_frac": float(getattr(hp, "score_finite_mask_min_probe_frac", 0.98)),
                    "score_finite_mask_min_dataset_frac": float(getattr(hp, "score_finite_mask_min_dataset_frac", 0.95)),
                    "score_finite_mask_min_points": int(getattr(hp, "score_finite_mask_min_points", 8)),
                }
            )
        if kind == "projection":
            cfg.update(
                {
                    "score_domain_projection_enable": bool(getattr(hp, "score_domain_projection_enable", False)),
                    "score_domain_projection_abs_tol": float(getattr(hp, "score_domain_projection_abs_tol", 1.0e-8)),
                    "score_domain_projection_rel_tol": float(getattr(hp, "score_domain_projection_rel_tol", 1.0e-8)),
                    "score_domain_projection_max_frac": float(getattr(hp, "score_domain_projection_max_frac", 1.0)),
                    "score_domain_projection_positive_floor": float(
                        getattr(hp, "score_domain_projection_positive_floor", 1.0e-12)
                    ),
                }
            )
        return cfg

    out: list[dict[str, Any]] = []
    for expr_name, node in nodes:
        for kind in ("strict", "finite_mask", "projection"):
            cfg_kind = _cfg_variant(kind)
            row: dict[str, Any] = {
                "expr": str(expr_name),
                "mode": str(kind),
                "diagnostic_only": True,
                "scoreable": False,
            }
            if kind == "projection":
                try:
                    _, dom_fit = eval_node_with_domain_projection(node, x_fit, cfg_kind)
                    _, dom_probe = eval_node_with_domain_projection(node, x_probe, cfg_kind)
                    dom_diag = merge_domain_projection_diagnostics(
                        dom_fit,
                        dom_probe,
                        labels=("fit", "probe"),
                    )
                    row["domain_projection_eval"] = oracle_de._to_jsonable(dom_diag)
                    if not domain_projection_is_acceptable(dom_diag):
                        row["rejection_hint"] = "domain_projection_eval_rejected"
                except Exception as exc:
                    row["domain_projection_eval_error"] = type(exc).__name__
            try:
                sc = _score_expr_diagnostic(
                    node,
                    x_fit,
                    y_fit,
                    x_probe,
                    y_probe,
                    proj,
                    "quant",
                    10.0,
                    15,
                    int(getattr(hp, "poly_degree", 1)),
                    refine_cfg=cfg_kind,
                )
                if sc is not None:
                    mse = _safe_float(sc[0], default=float("inf"))
                    mapping = sc[3] if len(sc) > 3 and isinstance(sc[3], dict) else {}
                    row.update(
                        {
                            "scoreable": True,
                            "mse": None if not math.isfinite(mse) else float(mse),
                            "rms": None if not math.isfinite(mse) else math.sqrt(max(0.0, float(mse))),
                            "mapping_kind": str(mapping.get("kind", "")),
                            "finite_mask": oracle_de._to_jsonable(mapping.get("_finite_mask", None)),
                            "domain_projection": oracle_de._to_jsonable(mapping.get("_domain_projection", None)),
                        }
                    )
            except Exception as exc:
                row["error"] = type(exc).__name__
            out.append(row)
    return out


def _tuple_ast_str(node: Any) -> str:
    try:
        return str(oracle_de.node_str(node))
    except Exception:
        return str(node)


def _tuple_const(value: float) -> tuple[str, float]:
    return ("const", float(value))


def _tuple_add(lhs: Any, rhs: Any) -> Any:
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    return ("add", lhs, rhs)


def _tuple_mul(lhs: Any, rhs: Any) -> Any:
    return ("mul", lhs, rhs)


def _tuple_sum_terms(terms: Sequence[Any]) -> Any:
    out = None
    for term in terms:
        out = _tuple_add(out, term)
    return _tuple_const(0.0) if out is None else out


def _tuple_scale(value: float, node: Any) -> Any:
    value_f = float(value)
    if abs(value_f) <= 1.0e-14:
        return _tuple_const(0.0)
    if abs(value_f - 1.0) <= 1.0e-14:
        return node
    if abs(value_f + 1.0) <= 1.0e-14:
        return _tuple_mul(_tuple_const(-1.0), node)
    return _tuple_mul(_tuple_const(value_f), node)


def _identity_mapping() -> dict[str, Any]:
    return {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}


def _embedded_tuple_ast(node: Any, input_exprs: Sequence[Any]) -> Any:
    inner = factorized_search_to_nestynet(node)
    return embed_mapping_in_ast(inner, _identity_mapping(), list(input_exprs), units_mode="raw")


def _eval_tuple_node_for_implicit(
    node: Any,
    x: torch.Tensor,
    *,
    hp: FactorizedSearchConfig,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    cfg = {
        "score_domain_projection_enable": bool(getattr(hp, "score_domain_projection_enable", False)),
        "score_domain_projection_abs_tol": float(getattr(hp, "score_domain_projection_abs_tol", 1.0e-8)),
        "score_domain_projection_rel_tol": float(getattr(hp, "score_domain_projection_rel_tol", 1.0e-8)),
        "score_domain_projection_max_frac": float(getattr(hp, "score_domain_projection_max_frac", 1.0)),
        "score_domain_projection_positive_floor": float(
            getattr(hp, "score_domain_projection_positive_floor", 1.0e-12)
        ),
    }
    try:
        vals, diag = eval_node_with_domain_projection(node, x, cfg)
    except Exception as exc:
        return None, {"ok": False, "error": type(exc).__name__}
    vals = vals.reshape(-1, 1)
    if not bool(torch.isfinite(vals).all().detach().cpu().item()):
        return None, {"ok": False, "error": "nonfinite"}
    return vals, oracle_de._to_jsonable(diag if isinstance(diag, dict) else {"ok": True})


def _robust_rms_tensor(values: torch.Tensor) -> float:
    t = values.reshape(-1)
    mask = torch.isfinite(t)
    if not bool(mask.any().detach().cpu().item()):
        return float("inf")
    a = torch.abs(t[mask]).to(dtype=torch.float64)
    if int(a.numel()) >= 32:
        try:
            cap = torch.quantile(a, 0.95)
            if bool(torch.isfinite(cap).detach().cpu().item()) and float(cap.detach().cpu()) > 0.0:
                a = torch.clamp(a, max=cap)
        except Exception:
            pass
    return float(torch.sqrt(torch.mean(a * a)).detach().cpu().item())


def _solve_implicit_b_coeffs(
    M: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float,
) -> torch.Tensor | None:
    if M.ndim != 2 or target.ndim != 2 or int(M.shape[0]) != int(target.shape[0]):
        return None
    if int(M.shape[0]) <= 0 or int(M.shape[1]) <= 0:
        return None
    try:
        Mt = M.transpose(0, 1)
        eye = torch.eye(int(M.shape[1]), dtype=M.dtype, device=M.device)
        lhs = Mt @ M + float(max(ridge, 0.0)) * eye
        rhs = Mt @ target
        coeff = torch.linalg.solve(lhs, rhs).reshape(-1)
    except Exception:
        try:
            coeff = torch.linalg.lstsq(M, target).solution.reshape(-1)
        except Exception:
            return None
    if not bool(torch.isfinite(coeff).all().detach().cpu().item()):
        return None
    return coeff


def _implicit_invariant_refit_coeffs(
    groups: Sequence[DEFeatureGroup],
    *,
    spec: oracle_de.DELabSpec,
    order: int,
    a_name: str,
    b_names: Sequence[str],
    rescue_cfg: FactorizedSearchDERescueConfig,
    dtype: torch.dtype,
) -> dict[str, Any] | None:
    """Fit simple first-order implicit coefficients from trajectory invariants.

    For ``A*u' + c*B = 0`` with ``A in {1, x}`` and ``B in {1, u}``, the
    solution has a scalar invariant ``Y + c*S = const`` per trajectory:

    - ``A=1, B=u``: ``Y=log|u|``, ``S=x``
    - ``A=x, B=u``: ``Y=log|u|``, ``S=log|x|``
    - ``A=1, B=1``: ``Y=u``, ``S=x``
    - ``A=x, B=1``: ``Y=u``, ``S=log|x|``

    This deliberately uses only observed trajectory geometry, not surrogate
    derivatives, so coordinate-singular laws are not judged by a noisy
    pointwise derivative estimate near the boundary.
    """

    if int(order) != 1 or len(list(b_names)) != 1:
        return None
    a_key = str(a_name).strip()
    b_key = str(list(b_names)[0]).strip()
    if a_key not in {"1", "x", "x0"} or b_key not in {"1", "u"}:
        return None
    min_points = max(3, int(getattr(rescue_cfg, "regularized_implicit_invariant_min_points", 8) or 8))
    x_axis = int(getattr(spec, "x_axis", 0))

    def _parts(split: str, use_attr: str) -> list[tuple[torch.Tensor, torch.Tensor, str]]:
        out: list[tuple[torch.Tensor, torch.Tensor, str]] = []
        for group in groups:
            if not bool(getattr(group, use_attr, True)):
                continue
            features = getattr(group, "features", None)
            if features is None:
                continue
            x = getattr(features, f"x_{split}", None)
            u = getattr(features, f"u_{split}", None)
            if x is None or u is None:
                continue
            x_t = x.to(dtype=torch.float64).reshape(int(x.shape[0]), -1)[:, x_axis].reshape(-1)
            u_t = u.to(dtype=torch.float64).reshape(-1)
            mask = torch.isfinite(x_t) & torch.isfinite(u_t)
            eps = torch.tensor(1.0e-14, dtype=torch.float64, device=x_t.device)
            if a_key in {"x", "x0"}:
                mask = mask & (torch.abs(x_t) > eps)
            if b_key == "u":
                mask = mask & (torch.abs(u_t) > eps)
            if int(mask.sum().detach().cpu().item()) < min_points:
                continue
            x_v = x_t[mask]
            u_v = u_t[mask]
            if b_key == "u":
                signs = torch.sign(u_v)
                if bool((torch.min(signs) < 0).detach().cpu().item()) and bool((torch.max(signs) > 0).detach().cpu().item()):
                    continue
                y_v = torch.log(torch.abs(u_v))
            else:
                y_v = u_v
            if a_key in {"x", "x0"}:
                s_v = torch.log(torch.abs(x_v))
            else:
                s_v = x_v
            finite = torch.isfinite(s_v) & torch.isfinite(y_v)
            if int(finite.sum().detach().cpu().item()) < min_points:
                continue
            s_v = s_v[finite]
            y_v = y_v[finite]
            if float(torch.std(s_v).detach().cpu().item()) <= 1.0e-12:
                continue
            out.append((s_v, y_v, str(group.id)))
        return out

    fit_parts = _parts("fit", "use_for_fit")
    probe_parts = _parts("probe", "use_for_probe")
    if not fit_parts or not probe_parts:
        return None

    num = torch.tensor(0.0, dtype=torch.float64)
    den = torch.tensor(0.0, dtype=torch.float64)
    n_fit = 0
    for s_v, y_v, _ in fit_parts:
        s_c = s_v - torch.mean(s_v)
        y_c = y_v - torch.mean(y_v)
        num = num + torch.sum(s_c * y_c).detach().cpu()
        den = den + torch.sum(s_c * s_c).detach().cpu()
        n_fit += int(s_v.numel())
    if float(den.item()) <= 1.0e-20:
        return None
    coeff = float((-num / den).item())
    if not math.isfinite(coeff):
        return None

    def _coeff_diagnostics(parts: Sequence[tuple[torch.Tensor, torch.Tensor, str]]) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        coeff_values: list[float] = []
        for s_v, y_v, gid in parts:
            s_c = s_v - torch.mean(s_v)
            y_c = y_v - torch.mean(y_v)
            den_i = float(torch.sum(s_c * s_c).detach().cpu().item())
            if den_i <= 1.0e-20:
                continue
            coeff_i = float((-torch.sum(s_c * y_c) / torch.sum(s_c * s_c)).detach().cpu().item())
            if not math.isfinite(coeff_i):
                continue
            coeff_values.append(float(coeff_i))
            rows.append({"id": str(gid), "n": int(s_v.numel()), "coeff": float(coeff_i)})
        if not coeff_values:
            return {"coeffs": [], "spread_abs": float("inf"), "spread_rel": float("inf")}
        spread_abs = max(abs(float(c) - float(coeff)) for c in coeff_values)
        spread_rel = spread_abs / max(abs(float(coeff)), 1.0e-12)
        return {
            "coeffs": oracle_de._to_jsonable(rows),
            "spread_abs": float(spread_abs),
            "spread_rel": float(spread_rel),
        }

    fit_coeff_diag = _coeff_diagnostics(fit_parts)
    probe_coeff_diag = _coeff_diagnostics(probe_parts)

    def _score(parts: Sequence[tuple[torch.Tensor, torch.Tensor, str]]) -> tuple[float, float, int, list[dict[str, Any]]]:
        residuals: list[torch.Tensor] = []
        y_centered: list[torch.Tensor] = []
        per_traj: list[dict[str, Any]] = []
        n_total = 0
        for s_v, y_v, gid in parts:
            inv = y_v + float(coeff) * s_v
            resid = inv - torch.mean(inv)
            yc = y_v - torch.mean(y_v)
            residuals.append(resid)
            y_centered.append(yc)
            n_total += int(resid.numel())
            rms_i = float(torch.sqrt(torch.mean(resid * resid)).detach().cpu().item())
            denom_i = float(torch.sqrt(torch.mean(yc * yc)).detach().cpu().item())
            per_traj.append(
                {
                    "id": str(gid),
                    "n": int(resid.numel()),
                    "raw_rms": float(rms_i),
                    "score": float(rms_i / max(denom_i, 1.0e-12)),
                }
            )
        if not residuals:
            return float("inf"), float("inf"), 0, per_traj
        r = torch.cat([v.reshape(-1) for v in residuals], dim=0)
        yc_all = torch.cat([v.reshape(-1) for v in y_centered], dim=0)
        raw = float(torch.sqrt(torch.mean(r * r)).detach().cpu().item())
        denom = float(torch.sqrt(torch.mean(yc_all * yc_all)).detach().cpu().item())
        return float(raw / max(denom, 1.0e-12)), float(raw), int(n_total), per_traj

    fit_score, fit_raw, n_fit_score, fit_diag = _score(fit_parts)
    probe_score, probe_raw, n_probe, probe_diag = _score(probe_parts)
    if not math.isfinite(probe_score):
        return None
    return {
        "enabled": True,
        "kind": "separable_invariant",
        "a_expr": str(a_name),
        "b_expr": str(b_key),
        "coordinate": "log_abs_x" if a_key in {"x", "x0"} else "x",
        "state_observable": "log_abs_u" if b_key == "u" else "u",
        "coeffs": [float(coeff)],
        "fit_score": float(fit_score),
        "probe_score": float(probe_score),
        "fit_raw_rms": float(fit_raw),
        "probe_raw_rms": float(probe_raw),
        "n_fit": int(n_fit_score if n_fit_score else n_fit),
        "n_probe": int(n_probe),
        "fit_traj": oracle_de._to_jsonable(fit_diag),
        "probe_traj": oracle_de._to_jsonable(probe_diag),
        "fit_coeffs": oracle_de._to_jsonable(fit_coeff_diag.get("coeffs", [])),
        "probe_coeffs": oracle_de._to_jsonable(probe_coeff_diag.get("coeffs", [])),
        "fit_coeff_spread_abs": _safe_float(fit_coeff_diag.get("spread_abs", float("inf")), default=float("inf")),
        "fit_coeff_spread_rel": _safe_float(fit_coeff_diag.get("spread_rel", float("inf")), default=float("inf")),
        "probe_coeff_spread_abs": _safe_float(probe_coeff_diag.get("spread_abs", float("inf")), default=float("inf")),
        "probe_coeff_spread_rel": _safe_float(probe_coeff_diag.get("spread_rel", float("inf")), default=float("inf")),
    }


def _implicit_multiplier_diagnostics(
    vals_fit: torch.Tensor,
    vals_probe: torch.Tensor,
    *,
    a_name: str,
    rescue_cfg: FactorizedSearchDERescueConfig,
) -> dict[str, Any]:
    vals = torch.cat([vals_fit.reshape(-1), vals_probe.reshape(-1)], dim=0).to(dtype=torch.float64)
    finite = vals[torch.isfinite(vals)]
    if int(finite.numel()) <= 0:
        return {"ok": False, "reason": "nonfinite_multiplier", "a_expr": str(a_name)}
    abs_v = torch.abs(finite)
    max_abs = float(torch.max(abs_v).detach().cpu().item())
    med_abs = float(torch.median(abs_v).detach().cpu().item())
    mean_abs = float(torch.mean(abs_v).detach().cpu().item())
    scale = max(float(med_abs), float(mean_abs), 1.0e-30)
    eps_zero = max(1.0e-12, 1.0e-10 * scale)
    nonzero_frac = float(torch.mean((abs_v > eps_zero).to(dtype=torch.float64)).detach().cpu().item())
    min_abs = float(torch.min(abs_v).detach().cpu().item())
    dynamic_range = float(max_abs / max(min_abs, eps_zero))
    pos = bool(torch.all(finite > 0).detach().cpu().item())
    neg = bool(torch.all(finite < 0).detach().cpu().item())
    sign_ok = bool(pos or neg or torch.all(torch.abs(finite) <= eps_zero).detach().cpu().item())
    max_range = _safe_float(
        getattr(rescue_cfg, "regularized_implicit_max_a_dynamic_range", 1.0e8),
        default=1.0e8,
    )
    min_nonzero = _safe_float(
        getattr(rescue_cfg, "regularized_implicit_min_nonzero_frac", 0.995),
        default=0.995,
    )
    simple_boundary = str(a_name) in {"x", "x0", "sqr(x)", "sqr(x0)", "x^2", "x0^2"}
    ok = bool(nonzero_frac >= min_nonzero and dynamic_range <= max_range and sign_ok)
    if simple_boundary and nonzero_frac >= min_nonzero and sign_ok:
        ok = True
    reason = ""
    if nonzero_frac < min_nonzero:
        reason = "multiplier_zero_fraction"
    elif dynamic_range > max_range and not simple_boundary:
        reason = "multiplier_dynamic_range"
    elif not sign_ok:
        reason = "multiplier_sign_change"
    return {
        "ok": bool(ok),
        "reason": reason,
        "a_expr": str(a_name),
        "min_abs": float(min_abs),
        "median_abs": float(med_abs),
        "max_abs": float(max_abs),
        "dynamic_range": float(dynamic_range),
        "nonzero_frac": float(nonzero_frac),
        "sign_ok": bool(sign_ok),
        "boundary_singular_allowed": bool(simple_boundary and ok),
    }


def _implicit_candidate_atoms(feature_names: Sequence[str], order: int) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]]]:
    names = [str(v) for v in feature_names]
    a_atoms: list[tuple[str, Any]] = [("1", _tuple_const(1.0))]
    b_atoms: list[tuple[str, Any]] = [("1", _tuple_const(1.0))]
    try:
        x_idx = names.index("x0")
    except Exception:
        x_idx = 0 if names and names[0].startswith("x") else -1
    try:
        u_idx = names.index("u")
    except Exception:
        u_idx = -1
    try:
        du_idx = names.index("du")
    except Exception:
        du_idx = -1

    if x_idx >= 0:
        x_node = ("var", int(x_idx))
        a_atoms.append(("x0", x_node))
    if u_idx >= 0:
        u_node = ("var", int(u_idx))
        b_atoms.append(("u", u_node))
        if x_idx >= 0:
            b_atoms.append(("x0*u", ("mul", ("var", int(x_idx)), u_node)))
    if int(order) == 2 and du_idx >= 0:
        b_atoms.append(("du", ("var", int(du_idx))))

    seen_a: set[str] = set()
    a_out: list[tuple[str, Any]] = []
    for name, node in a_atoms:
        key = _tuple_ast_str(node)
        if key in seen_a:
            continue
        seen_a.add(key)
        a_out.append((name, node))
    seen_b: set[str] = set()
    b_out: list[tuple[str, Any]] = []
    for name, node in b_atoms:
        key = _tuple_ast_str(node)
        if key in seen_b:
            continue
        seen_b.add(key)
        b_out.append((name, node))
    return a_out, b_out


def _build_regularized_implicit_result(
    *,
    row: Mapping[str, Any],
    spec: oracle_de.DELabSpec,
    order: int,
    feature_names: Sequence[str],
    input_exprs: Sequence[Any],
    groups: Sequence[DEFeatureGroup],
    diagnostics: Mapping[str, Any],
) -> FactorizedSearchDEResult | None:
    explicit_expr = _jsonable_ast_to_tuple(row.get("explicit_expr_ast", None))
    a_expr = _jsonable_ast_to_tuple(row.get("a_expr_ast", None))
    b_sum_expr = _jsonable_ast_to_tuple(row.get("b_sum_expr_ast", None))
    if explicit_expr is None or a_expr is None or b_sum_expr is None:
        return None
    try:
        rhs_ast = _embedded_tuple_ast(explicit_expr, input_exprs)
        a_ast = _embedded_tuple_ast(a_expr, input_exprs)
        b_ast = _embedded_tuple_ast(b_sum_expr, input_exprs)
        residual_ast = Add(Mul(a_ast, _anchor_for_order(order, x_axis=int(spec.x_axis))), b_ast)
    except Exception:
        return None
    compiled = _compiled_de_ast_payload(rhs_ast=rhs_ast, residual_ast=residual_ast)
    residual_simplified = compiled["residual_ast_simplified"]
    rhs_simplified = compiled["rhs_ast_simplified"]
    probe_rms = _safe_float(row.get("score", None), default=float("inf"))
    probe_mse = float(probe_rms) ** 2 if math.isfinite(probe_rms) else float("inf")
    report = {
        "spec_id": str(spec.id),
        "order_candidates": [int(o) for o in spec.order_candidates],
        "x_axis": int(spec.x_axis),
        "include_x": bool(spec.include_x),
        "include_u": bool(spec.include_u),
        "include_du": bool(spec.include_du),
        "x_col": str(spec.x_col),
        "u_col": str(spec.u_col),
        "constants_ordered": [
            {"name": str(c.name), "value": float(c.value)}
            for c in tuple(spec.constants)
        ],
        "trajectories": [{"id": str(group.id), "csv": None} for group in groups],
        "fit_trajectories": [{"id": str(group.id), "csv": None} for group in groups if group.use_for_fit],
        "probe_trajectories": [{"id": str(group.id), "csv": None} for group in groups if group.use_for_probe],
        "factorized_de_diagnostics": oracle_de._to_jsonable(dict(diagnostics)),
        "best": {
            "order": int(order),
            "expr": _tuple_ast_str(explicit_expr),
            "expr_ast": oracle_de._to_jsonable(explicit_expr),
            "mapping": _identity_mapping(),
            "mapping_kind": "poly",
            "mse": float(probe_mse),
            "score": float(probe_rms),
            "score_raw": float(row.get("raw_probe_rms", probe_rms)),
            "size": int(row.get("size", 0)),
            "candidate_source": "regularized_implicit_residual",
            "structural_ok": True,
            "structural_hard_reject": False,
            "structural_reasons": [],
        },
        "per_order": [
            {
                "order": int(order),
                "feature_names": list(feature_names),
                "results": [],
                "best": None,
            }
        ],
    }
    candidate_id = _factorized_candidate_id(report["best"])
    diag = {
        "score": float(probe_rms),
        "score_raw": float(row.get("raw_probe_rms", probe_rms)),
        "size": int(row.get("size", 0)),
        "symbolic_size_raw": compiled["symbolic_size_raw"],
        "symbolic_size_simplified": compiled["symbolic_size_simplified"],
        "mapping_complexity": 0,
        "canonical_equation_raw": repr(compiled["residual_ast_raw"]),
        "canonical_equation_simplified": repr(residual_simplified),
        "candidate_id": candidate_id,
        "candidate_key": _factorized_candidate_key_payload(report["best"]),
        "candidate_source": "regularized_implicit_residual",
        "probe_rel_rms": float(probe_rms),
        "raw_probe_rms": _safe_float(row.get("raw_probe_rms", None), default=float("inf")),
        "raw_fit_rms": _safe_float(row.get("raw_fit_rms", None), default=float("inf")),
        "normalized_probe_score": float(probe_rms),
        "domain_ok": True,
        "domain_fragility_penalty": 0.0,
        "domain_failure_reason": None,
        "finite_mask": None,
        "domain_projection": None,
        "structural_ok": True,
        "structural_hard_reject": False,
        "structural_reasons": [],
        "structural_gate_version": 1,
        "include_x": bool(spec.include_x),
        "include_u": bool(spec.include_u),
        "include_du": bool(spec.include_du),
        "n_traj_total": int(len(groups)),
        "fit_traj_ids": [str(group.id) for group in groups if group.use_for_fit],
        "probe_traj_ids": [str(group.id) for group in groups if group.use_for_probe],
        "implicit_residual": oracle_de._to_jsonable(dict(row)),
        "factorized_de_diagnostics": oracle_de._to_jsonable(dict(diagnostics)),
        "report": report,
    }
    return FactorizedSearchDEResult(
        order=int(order),
        x_axis=int(spec.x_axis),
        rhs_ast=rhs_simplified,
        residual_ast=residual_simplified,
        canonical_equation=repr(residual_simplified),
        probe_mse=float(probe_mse),
        probe_rms=float(probe_rms),
        expr_ast=explicit_expr,
        mapping=_identity_mapping(),
        mapping_kind="poly",
        feature_names=list(feature_names),
        diagnostics=diag,
        engine="factorized_search",
        rhs_ast_raw=compiled["rhs_ast_raw"],
        residual_ast_raw=compiled["residual_ast_raw"],
        rhs_ast_simplified=rhs_simplified,
        residual_ast_simplified=residual_simplified,
        canonical_equation_raw=repr(compiled["residual_ast_raw"]),
        canonical_equation_simplified=repr(residual_simplified),
    )


def run_regularized_implicit_residual_fss_from_feature_groups(
    groups: Sequence[DEFeatureGroup],
    *,
    cfg,
    rescue_cfg: FactorizedSearchDERescueConfig,
    dtype: torch.dtype = torch.float64,
    verbose: bool = False,
) -> FactorizedSearchDEResult | None:
    """Search constrained quasilinear implicit residuals ``A(z) h + B(z)=0``.

    This lane is intentionally narrow: ``A`` and ``B`` are shallow grammar atoms
    over lower-order variables only, and ``A`` is gauge-checked before a candidate
    can win. The selected result carries the implicit residual for reporting and
    the explicit ``h=-B/A`` equivalent for rollout.
    """

    if not bool(getattr(rescue_cfg, "regularized_implicit_enable", True)):
        return None
    if any(
        getattr(group, "features", None) is None
        or not hasattr(group, "use_for_fit")
        or not hasattr(group, "use_for_probe")
        for group in groups
    ):
        return None
    hp = _apply_rescue_cfg_to_hp(rescue_cfg)
    spec_base = de_lab_spec_from_de_cfg(cfg)
    best: FactorizedSearchDEResult | None = None
    all_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "enabled": True,
        "lane": "regularized_implicit_residual",
        "orders": [],
        "candidate_rows": 0,
    }
    orders = [int(o) for o in tuple(spec_base.order_candidates) if int(o) in (1, 2)]
    for order in orders:
        spec = spec_base
        x_fit_parts: list[torch.Tensor] = []
        y_fit_parts: list[torch.Tensor] = []
        x_probe_parts: list[torch.Tensor] = []
        y_probe_parts: list[torch.Tensor] = []
        feature_names: list[str] | None = None
        for group in groups:
            if bool(group.use_for_fit):
                z_fit, y_fit, names = oracle_de._build_table_from_features(
                    spec,
                    group.features,
                    order=int(order),
                    split="fit",
                )
                x_fit_parts.append(z_fit.to(dtype=dtype))
                y_fit_parts.append(y_fit.to(dtype=dtype))
                feature_names = list(names)
            if bool(group.use_for_probe):
                z_probe, y_probe, names = oracle_de._build_table_from_features(
                    spec,
                    group.features,
                    order=int(order),
                    split="probe",
                )
                x_probe_parts.append(z_probe.to(dtype=dtype))
                y_probe_parts.append(y_probe.to(dtype=dtype))
                if feature_names is None:
                    feature_names = list(names)
        if not x_fit_parts or not x_probe_parts or not feature_names:
            continue
        x_fit = torch.cat(x_fit_parts, dim=0)
        y_fit = torch.cat(y_fit_parts, dim=0).reshape(-1, 1)
        x_probe = torch.cat(x_probe_parts, dim=0)
        y_probe = torch.cat(y_probe_parts, dim=0).reshape(-1, 1)
        input_exprs = _input_exprs_from_report(
            {
                "x_axis": int(spec.x_axis),
                "include_x": bool(spec.include_x),
                "include_u": bool(spec.include_u),
                "include_du": bool(spec.include_du),
                "constants_ordered": [
                    {"name": str(c.name), "value": float(c.value)}
                    for c in tuple(spec.constants)
                ],
            },
            order=int(order),
        )
        a_atoms, b_atoms = _implicit_candidate_atoms(feature_names, int(order))
        b_eval: dict[str, tuple[str, Any, torch.Tensor, torch.Tensor]] = {}
        for b_name, b_node in b_atoms:
            b_fit, b_fit_diag = _eval_tuple_node_for_implicit(b_node, x_fit, hp=hp)
            b_probe, b_probe_diag = _eval_tuple_node_for_implicit(b_node, x_probe, hp=hp)
            if b_fit is None or b_probe is None:
                continue
            b_eval[str(b_name)] = (str(b_name), b_node, b_fit, b_probe)
        max_terms = max(1, int(getattr(rescue_cfg, "regularized_implicit_max_b_terms", 2) or 1))
        order_rows: list[dict[str, Any]] = []
        for a_name, a_node in a_atoms:
            a_fit, a_fit_diag = _eval_tuple_node_for_implicit(a_node, x_fit, hp=hp)
            a_probe, a_probe_diag = _eval_tuple_node_for_implicit(a_node, x_probe, hp=hp)
            if a_fit is None or a_probe is None:
                continue
            mult_diag = _implicit_multiplier_diagnostics(
                a_fit,
                a_probe,
                a_name=str(a_name),
                rescue_cfg=rescue_cfg,
            )
            if not bool(mult_diag.get("ok", False)):
                continue
            keys = list(b_eval.keys())
            for k_terms in range(1, min(max_terms, len(keys)) + 1):
                for combo_keys in itertools.combinations(keys, k_terms):
                    entries = [b_eval[k] for k in combo_keys]
                    M_fit = torch.cat([entry[2] for entry in entries], dim=1)
                    M_probe = torch.cat([entry[3] for entry in entries], dim=1)
                    coeff = _solve_implicit_b_coeffs(
                        M_fit,
                        -a_fit * y_fit,
                        ridge=_safe_float(
                            getattr(rescue_cfg, "regularized_implicit_ridge", 1.0e-10),
                            default=1.0e-10,
                        ),
                    )
                    if coeff is None:
                        continue
                    derivative_coeff = coeff.detach().clone()
                    b_fit_pred = M_fit @ coeff.reshape(-1, 1)
                    b_probe_pred = M_probe @ coeff.reshape(-1, 1)
                    res_fit = a_fit * y_fit + b_fit_pred
                    res_probe = a_probe * y_probe + b_probe_pred
                    derivative_raw_fit = _robust_rms_tensor(res_fit)
                    derivative_raw_probe = _robust_rms_tensor(res_probe)
                    denom_fit = (
                        _robust_rms_tensor(a_fit * y_fit)
                        + _robust_rms_tensor(b_fit_pred)
                        + 1.0e-12
                    )
                    denom_probe = (
                        _robust_rms_tensor(a_probe * y_probe)
                        + _robust_rms_tensor(b_probe_pred)
                        + 1.0e-12
                    )
                    derivative_score_fit = (
                        derivative_raw_fit / denom_fit
                        if math.isfinite(denom_fit) and denom_fit > 0.0
                        else float("inf")
                    )
                    derivative_score_probe = (
                        derivative_raw_probe / denom_probe
                        if math.isfinite(denom_probe) and denom_probe > 0.0
                        else float("inf")
                    )
                    raw_fit = float(derivative_raw_fit)
                    raw_probe = float(derivative_raw_probe)
                    score_fit = float(derivative_score_fit)
                    score_probe = float(derivative_score_probe)
                    coeff_source = "derivative_residual"
                    invariant_refit = None
                    if bool(getattr(rescue_cfg, "regularized_implicit_invariant_refit_enable", True)):
                        invariant_refit = _implicit_invariant_refit_coeffs(
                            groups,
                            spec=spec,
                            order=int(order),
                            a_name=str(a_name),
                            b_names=[entry[0] for entry in entries],
                            rescue_cfg=rescue_cfg,
                            dtype=dtype,
                        )
                    if invariant_refit is not None:
                        coeff = torch.as_tensor(
                            invariant_refit["coeffs"],
                            dtype=M_fit.dtype,
                            device=M_fit.device,
                        ).reshape(-1)
                        b_fit_pred = M_fit @ coeff.reshape(-1, 1)
                        b_probe_pred = M_probe @ coeff.reshape(-1, 1)
                        res_fit = a_fit * y_fit + b_fit_pred
                        res_probe = a_probe * y_probe + b_probe_pred
                        raw_fit = _safe_float(
                            invariant_refit.get("fit_raw_rms", None),
                            default=_robust_rms_tensor(res_fit),
                        )
                        raw_probe = _safe_float(
                            invariant_refit.get("probe_raw_rms", None),
                            default=_robust_rms_tensor(res_probe),
                        )
                        score_fit = _safe_float(invariant_refit.get("fit_score", None), default=float("inf"))
                        score_probe = _safe_float(invariant_refit.get("probe_score", None), default=float("inf"))
                        coeff_source = "separable_invariant_refit"
                    if not math.isfinite(score_probe):
                        continue
                    b_terms = [
                        _tuple_scale(float(coeff[i].detach().cpu().item()), entries[i][1])
                        for i in range(len(entries))
                    ]
                    b_sum = _tuple_sum_terms(b_terms)
                    explicit = ("div", _tuple_mul(_tuple_const(-1.0), b_sum), a_node)
                    residual_tuple = _tuple_add(
                        _tuple_mul(a_node, ("anchor", int(order))),
                        b_sum,
                    )
                    # The tuple evaluator does not know about anchors; this tuple is
                    # only for diagnostics. Build the embedded residual separately.
                    try:
                        a_emb = _embedded_tuple_ast(a_node, input_exprs)
                        b_emb = _embedded_tuple_ast(b_sum, input_exprs)
                        anchor_node = _anchor_for_order(int(order), x_axis=int(spec.x_axis))
                        residual_ast = Add(Mul(a_emb, anchor_node), b_emb)
                        rhs_ast = _embedded_tuple_ast(explicit, input_exprs)
                        compiled_tmp = _compiled_de_ast_payload(rhs_ast=rhs_ast, residual_ast=residual_ast)
                        size = int(compiled_tmp["symbolic_size_simplified"] or compiled_tmp["symbolic_size_raw"] or 0)
                    except Exception:
                        size = 10**9
                    complexity_penalty = 1.0e-5 * max(0, int(size))
                    score_total = float(score_probe + complexity_penalty)
                    row = {
                        "order": int(order),
                        "a_expr": str(a_name),
                        "a_expr_ast": oracle_de._to_jsonable(a_node),
                        "b_exprs": [entry[0] for entry in entries],
                        "b_expr_asts": [oracle_de._to_jsonable(entry[1]) for entry in entries],
                        "b_coeffs": [float(c.detach().cpu().item()) for c in coeff.reshape(-1)],
                        "b_coeff_source": str(coeff_source),
                        "derivative_b_coeffs": [float(c.detach().cpu().item()) for c in derivative_coeff.reshape(-1)],
                        "b_sum_expr_ast": oracle_de._to_jsonable(b_sum),
                        "explicit_expr_ast": explicit,
                        "implicit_residual_tuple": residual_tuple,
                        "raw_fit_rms": float(raw_fit),
                        "raw_probe_rms": float(raw_probe),
                        "normalized_fit_score": float(score_fit),
                        "normalized_probe_score": float(score_probe),
                        "derivative_raw_fit_rms": float(derivative_raw_fit),
                        "derivative_raw_probe_rms": float(derivative_raw_probe),
                        "derivative_normalized_fit_score": float(derivative_score_fit),
                        "derivative_normalized_probe_score": float(derivative_score_probe),
                        "invariant_refit": oracle_de._to_jsonable(invariant_refit),
                        "score": float(score_total),
                        "size": int(size),
                        "multiplier": mult_diag,
                        "a_domain_fit": a_fit_diag,
                        "a_domain_probe": a_probe_diag,
                    }
                    order_rows.append(row)
        order_rows.sort(key=lambda r: (float(r.get("score", float("inf"))), int(r.get("size", 10**9))))
        diagnostics["orders"].append(
            {
                "order": int(order),
                "feature_names": list(feature_names),
                "a_atoms": [name for name, _ in a_atoms],
                "b_atoms": [name for name, _ in b_atoms],
                "rows": int(len(order_rows)),
                "best_score": None if not order_rows else float(order_rows[0]["score"]),
                "best": None if not order_rows else oracle_de._to_jsonable(order_rows[0]),
            }
        )
        all_rows.extend(order_rows)
        if order_rows:
            candidate = _build_regularized_implicit_result(
                row=order_rows[0],
                spec=spec,
                order=int(order),
                feature_names=list(feature_names),
                input_exprs=input_exprs,
                groups=groups,
                diagnostics=diagnostics,
            )
            if candidate is not None and (best is None or float(candidate.probe_rms) < float(best.probe_rms)):
                best = candidate
    diagnostics["candidate_rows"] = int(len(all_rows))
    if best is not None:
        best.diagnostics["regularized_implicit_residual"] = {
            "enabled": True,
            "selected_probe_score": float(best.probe_rms),
            "selected": oracle_de._to_jsonable(best.diagnostics.get("implicit_residual", {})),
            "orders": oracle_de._to_jsonable(diagnostics.get("orders", [])),
        }
        if verbose:
            imp = best.diagnostics.get("implicit_residual", {})
            print(
                "[factorized DE] Regularized implicit lane: "
                f"score={float(best.probe_rms):.6e} "
                f"A={imp.get('a_expr')} B={imp.get('b_exprs')} coeffs={imp.get('b_coeffs')} "
                f"source={imp.get('b_coeff_source')}",
                flush=True,
            )
    elif verbose:
        print("[factorized DE] Regularized implicit lane: no eligible candidate.", flush=True)
    return best


def run_direct_residual_fss_from_feature_groups(
    groups: Sequence[DEFeatureGroup],
    *,
    cfg,
    rescue_cfg: FactorizedSearchDERescueConfig,
    dtype: torch.dtype = torch.float64,
    verbose: bool = False,
    attempt_phase: str = "all",
) -> FactorizedSearchDEResult | None:
    """Run the paper-facing anchored residual/RHS DE-FSS lane.

    This lane searches ``highest_derivative ~= F(features)`` using the generic
    FSS grammar. Dimensional filtering is DE-specific and opt-in: candidate
    structures may have their own dimensions while learned outer coefficients
    carry the target-minus-structure dimension.
    """

    hp_base = _apply_rescue_cfg_to_hp(rescue_cfg)
    spec_base = de_lab_spec_from_de_cfg(cfg)
    if int(getattr(rescue_cfg, "validate_integrate_topk", 0)) > 0:
        spec_base = replace(spec_base, validate_integrate_topk=int(rescue_cfg.validate_integrate_topk))

    attempts = _direct_residual_attempt_specs(spec_base, cfg)
    phase = str(attempt_phase or "all").strip().lower()
    if phase not in {"all", "autonomous", "full"}:
        raise ValueError(f"unknown direct residual attempt phase: {attempt_phase!r}")
    if phase == "autonomous":
        attempts = [attempt for attempt in attempts if bool(attempt[2])]
    elif phase == "full":
        attempts = [attempt for attempt in attempts if not bool(attempt[2])]
    if not attempts:
        return None

    best_res: FactorizedSearchDEResult | None = None
    attempt_logs: list[dict[str, Any]] = []
    trigger = float(getattr(rescue_cfg, "trigger_val_rms", 1.0e-3))
    if not math.isfinite(trigger) or trigger <= 0.0:
        trigger = 1.0e-3
    rel_trigger = float(getattr(rescue_cfg, "trigger_rel_rms", 1.0e-3))
    if not math.isfinite(rel_trigger) or rel_trigger <= 0.0:
        rel_trigger = 1.0e-3
    witness_topk = max(0, int(getattr(rescue_cfg, "direct_generator_witness_topk", 0) or 0))

    for attempt_index, (attempt_label, attempt_spec, autonomous) in enumerate(attempts):
        hp_attempt = _direct_residual_attempt_hp(hp_base, autonomous=bool(autonomous))
        hp_attempt._de_early_stop_val_rms = float(trigger)
        hp_attempt._de_early_stop_rel_rms = float(rel_trigger)
        hp_attempt._de_early_stop_rms_multiplier = 1.0
        canary_diag = _direct_residual_canary_diagnostics(
            groups,
            spec=attempt_spec,
            hp=hp_attempt,
            dtype=dtype,
        )
        if verbose:
            print(
                "[factorized DE] Direct attempt "
                f"{attempt_label}: include_x={int(bool(attempt_spec.include_x))} "
                f"include_u={int(bool(attempt_spec.include_u))} "
                f"include_du={int(bool(attempt_spec.include_du))} "
                f"max_depth={int(getattr(hp_attempt, 'max_depth', -1))} "
                f"brute_depth={int(getattr(hp_attempt, 'brute_depth', -1))} "
                f"early_stop_mse={float(getattr(hp_attempt, 'early_stop_mse', float('nan'))):.3e} "
                f"n_iter={int(getattr(hp_attempt, 'n_iter', -1))}",
                flush=True,
            )
            for canary_row in canary_diag:
                if not isinstance(canary_row, dict):
                    continue
                rms_canary = _safe_float(canary_row.get("rms", None), default=float("inf"))
                rms_text = f"{rms_canary:.3e}" if math.isfinite(rms_canary) else "inf"
                status_text = "scoreable" if bool(canary_row.get("scoreable", False)) else "not-scoreable"
                dom_eval = canary_row.get("domain_projection_eval", None)
                dom_text = ""
                if isinstance(dom_eval, dict) and bool(dom_eval.get("enabled", False)):
                    max_v = _safe_float(dom_eval.get("max_violation", None), default=float("nan"))
                    max_v_text = f"{max_v:.3e}" if math.isfinite(max_v) else "nan"
                    dom_text = (
                        f" domain={dom_eval.get('status')} "
                        f"proj_frac={float(dom_eval.get('projected_frac', 0.0) or 0.0):.3f} "
                        f"max_v={max_v_text}"
                    )
                hint = str(canary_row.get("rejection_hint", "") or "")
                hint_text = f" hint={hint}" if hint else ""
                print(
                    "[factorized DE] Direct canary "
                    f"{attempt_label}: {canary_row.get('expr')} "
                    f"mode={canary_row.get('mode')} {status_text} rms={rms_text}"
                    f"{dom_text}{hint_text}",
                    flush=True,
                )
        started = time.perf_counter()
        report = run_factorized_de_from_feature_groups(
            attempt_spec,
            list(groups),
            factorized_search_hp=hp_attempt,
            seed=int(hp_attempt.seed) + int(attempt_index) * 10_007,
            dtype=dtype,
            enforce_dims=bool(getattr(cfg, "enforce_units", False)),
            verbose=bool(verbose),
            parallel_orders=False,
            budget_scope=str(getattr(rescue_cfg, "budget_scope", "global")),
            coefficient_dim_mode="inferred_outer",
        )
        result = factorized_search_report_to_de_result(report)
        result_diag = getattr(result, "diagnostics", {}) or {}
        probe_rel = _safe_float(result_diag.get("probe_rel_rms", None), default=float("inf"))
        generator_witness: dict[str, Any] | None = None
        generator_accepted = False
        if (
            witness_topk > 0
            and int(getattr(result, "order", -1)) == 2
            and result.expr_ast is not None
            and result.rhs_ast is not None
        ):
            generator_witness = validate_order2_generator_witness(
                result,
                groups,
                spec=attempt_spec,
                rescue_cfg=rescue_cfg,
                dtype=dtype,
            )
            status = str(generator_witness.get("generator_status", "NOT_VIABLE")).strip().upper()
            witness_materialized = False
            diag = getattr(result, "diagnostics", None)
            if isinstance(diag, dict):
                diag["generator_witness"] = generator_witness
                diag["generator_status"] = status
                diag["evidence_tier"] = "generator_witness"
            if status in _GENERATOR_WITNESS_MATERIALIZABLE_STATUSES:
                witness_materialized = _materialize_generator_witness_result(
                    result,
                    generator_witness,
                    spec=attempt_spec,
                )
            generator_accepted = bool(status == "EXACT_STRUCTURAL_GENERATOR" and witness_materialized)
            diag = getattr(result, "diagnostics", None)
            if isinstance(diag, dict) and generator_accepted:
                diag["early_exit_reason"] = "generator_witness_pass"
        elif witness_topk > 0 and int(getattr(result, "order", -1)) != 2:
            diag = getattr(result, "diagnostics", None)
            if isinstance(diag, dict):
                diag["generator_witness"] = {
                    "enabled": False,
                    "generator_status": "NOT_APPLICABLE",
                    "reason": "order_not_2",
                }
        report_diag = report.get("factorized_de_diagnostics", {}) if isinstance(report, dict) else {}
        order_diag = {}
        if isinstance(report_diag, dict):
            orders = [row for row in list(report_diag.get("orders", []) or []) if isinstance(row, dict)]
            order_diag = orders[0] if orders else {}
        diag = getattr(result, "diagnostics", None)
        if isinstance(diag, dict):
            diag["direct_residual_attempt"] = {
                "label": str(attempt_label),
                "autonomous": bool(autonomous),
                "attempt_index": int(attempt_index),
                "include_x": bool(attempt_spec.include_x),
                "include_u": bool(attempt_spec.include_u),
                "include_du": bool(attempt_spec.include_du),
                "order_candidates": [int(o) for o in attempt_spec.order_candidates],
                "coefficient_dim_mode": "inferred_outer",
                "max_depth": int(getattr(hp_attempt, "max_depth", -1)),
                "brute_depth": int(getattr(hp_attempt, "brute_depth", -1)),
                "base_early_stop_mse": float(getattr(hp_attempt, "early_stop_mse", 0.0)),
                "effective_early_stop_mse": order_diag.get("effective_early_stop_mse", None),
                "probe_rel_rms": None if not math.isfinite(float(probe_rel)) else float(probe_rel),
                "target_scale": order_diag.get("target_scale", None),
                "canary": canary_diag,
                "generator_witness": generator_witness,
                "generator_status": None if generator_witness is None else generator_witness.get("generator_status", None),
                "generator_accepted": bool(generator_accepted),
                "wall_seconds": float(time.perf_counter() - started),
            }
        attempt_logs.append(
            {
                "label": str(attempt_label),
                "autonomous": bool(autonomous),
                "order_candidates": [int(o) for o in attempt_spec.order_candidates],
                "include_x": bool(attempt_spec.include_x),
                "include_u": bool(attempt_spec.include_u),
                "include_du": bool(attempt_spec.include_du),
                "max_depth": int(getattr(hp_attempt, "max_depth", -1)),
                "brute_depth": int(getattr(hp_attempt, "brute_depth", -1)),
                "base_early_stop_mse": float(getattr(hp_attempt, "early_stop_mse", 0.0)),
                "effective_early_stop_mse": order_diag.get("effective_early_stop_mse", None),
                "probe_rms": None if not math.isfinite(float(result.probe_rms)) else float(result.probe_rms),
                "probe_rel_rms": None if not math.isfinite(float(probe_rel)) else float(probe_rel),
                "target_scale": order_diag.get("target_scale", None),
                "status": str((getattr(result, "diagnostics", {}) or {}).get("status", "OK")),
                "canary": canary_diag,
                "generator_witness": generator_witness,
                "generator_status": None if generator_witness is None else generator_witness.get("generator_status", None),
                "generator_accepted": bool(generator_accepted),
                "wall_seconds": float(time.perf_counter() - started),
            }
        )
        if verbose:
            expr_text = ""
            try:
                expr_text = str(result.expr_ast)
            except Exception:
                expr_text = ""
            rms_text = f"{float(result.probe_rms):.6e}" if math.isfinite(float(result.probe_rms)) else "inf"
            rel_text = f"{float(probe_rel):.6e}" if math.isfinite(float(probe_rel)) else "inf"
            print(
                "[factorized DE] Direct attempt "
                f"{attempt_label}: probe RMS={rms_text} rel={rel_text} "
                f"best={expr_text} wall={float(time.perf_counter() - started):.2f}s",
                flush=True,
            )
            if generator_witness is not None:
                print(
                    "[factorized DE] Direct generator witness "
                    f"{attempt_label}: status={generator_witness.get('generator_status')} "
                    f"local_rms_z={generator_witness.get('local_rms_z')} "
                    f"rollout_u_nrmse={generator_witness.get('rollout_u_nrmse')}",
                    flush=True,
                )
        if result.expr_ast is None or result.rhs_ast is None or result.residual_ast is None:
            continue
        if best_res is None or float(result.probe_rms) < float(best_res.probe_rms):
            best_res = result
        if generator_accepted:
            best_res = result
            break
        if (
            (math.isfinite(float(result.probe_rms)) and float(result.probe_rms) <= trigger)
            or (math.isfinite(float(probe_rel)) and float(probe_rel) <= float(rel_trigger))
        ):
            break

    if best_res is not None:
        best_res.engine = "factorized_search"
        best_res.diagnostics.setdefault("factorized_de_diagnostics", {})
        best_res.diagnostics["direct_residual_fss"] = {
            "enabled": True,
            "attempts": attempt_logs,
            "selected_probe_rms": float(best_res.probe_rms),
            "selected_probe_rel_rms": (
                None
                if not math.isfinite(_safe_float((best_res.diagnostics or {}).get("probe_rel_rms", None), default=float("inf")))
                else _safe_float((best_res.diagnostics or {}).get("probe_rel_rms", None), default=float("inf"))
            ),
            "coefficient_dim_mode": "inferred_outer",
            "generator_status": (best_res.diagnostics or {}).get("generator_status", None),
            "evidence_tier": (best_res.diagnostics or {}).get("evidence_tier", None),
            "early_exit_reason": (best_res.diagnostics or {}).get("early_exit_reason", None),
        }
    return best_res


def run_factorized_search_de_from_surrogate(
    surrogate,
    train_loader,
    val_loader,
    *,
    cfg,
    rescue_cfg: FactorizedSearchDERescueConfig,
    device,
    dtype: torch.dtype = torch.float64,
) -> FactorizedSearchDEResult:
    """Run DE-facing factorized symbolic search from one surrogate and its train/val loaders."""

    groups = build_factorized_search_de_feature_groups_from_surrogate(
        surrogate,
        train_loader,
        val_loader,
        cfg=cfg,
        rescue_cfg=rescue_cfg,
        device=device,
        dtype=dtype,
    )
    return run_factorized_search_de_from_feature_groups(
        groups,
        cfg=cfg,
        rescue_cfg=rescue_cfg,
        dtype=dtype,
    )


def run_factorized_search_de_from_surrogates(
    surrogates,
    train_loaders,
    val_loaders,
    *,
    cfg,
    rescue_cfg: FactorizedSearchDERescueConfig,
    device,
    dataset_ids: Sequence[str] | None = None,
    dtype: torch.dtype = torch.float64,
) -> FactorizedSearchDEResult:
    """Run DE-facing factorized symbolic search from multiple surrogates with grouped identities."""

    groups = build_factorized_search_de_feature_groups_from_surrogates(
        surrogates,
        train_loaders,
        val_loaders,
        cfg=cfg,
        rescue_cfg=rescue_cfg,
        device=device,
        dataset_ids=dataset_ids,
        dtype=dtype,
    )
    return run_factorized_search_de_from_feature_groups(
        groups,
        cfg=cfg,
        rescue_cfg=rescue_cfg,
        dtype=dtype,
    )

__factorized_de_definitions__ = (
    "default_physics_rescue_hp",
    "run_factorized_de_from_feature_groups",
    "factorized_search_report_to_de_result",
    "factorized_search_report_shortlist",
    "normalized_rmse",
    "evaluate_factorized_search_candidate",
    "factorized_search_candidate_to_feature_predictor",
    "factorized_search_report_to_rhs_callable",
    "_apply_rescue_cfg_to_hp",
    "_build_feature_group_from_surrogate",
    "build_factorized_search_de_feature_groups_from_surrogate",
    "build_factorized_search_de_feature_groups_from_surrogates",
    "run_factorized_search_de_from_feature_groups",
    "_direct_residual_attempt_hp",
    "_direct_residual_attempt_specs",
    "_direct_residual_canary_diagnostics",
    "_tuple_ast_str",
    "_tuple_const",
    "_tuple_add",
    "_tuple_mul",
    "_tuple_sum_terms",
    "_tuple_scale",
    "_identity_mapping",
    "_embedded_tuple_ast",
    "_eval_tuple_node_for_implicit",
    "_robust_rms_tensor",
    "_solve_implicit_b_coeffs",
    "_implicit_invariant_refit_coeffs",
    "_implicit_multiplier_diagnostics",
    "_implicit_candidate_atoms",
    "_build_regularized_implicit_result",
    "run_regularized_implicit_residual_fss_from_feature_groups",
    "run_direct_residual_fss_from_feature_groups",
    "run_factorized_search_de_from_surrogate",
    "run_factorized_search_de_from_surrogates",
)

__factorized_de_constants__ = (

)

__factorized_de_late_bindings__ = (
    "_compiled_de_ast_payload",
)
