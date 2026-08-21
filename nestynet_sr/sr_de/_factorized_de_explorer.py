# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

"""Typed-explorer launch planning, execution, and candidate integration."""

import concurrent.futures
import math
import os
import time
from typing import Any, Mapping, Sequence
import torch
import nestynet_sr.sr_search.factorized_search.oracle_lab_de as oracle_de
from nestynet_sr.sr_core.bridges import AtomNode, ConstNode
from nestynet_sr.sr_search.factorized_search.bridge import run_explorer

from ._factorized_de_frontend import (
    DEFeatureGroup,
    _best_numeric_row_value,
    _diag_inc,
    _diag_number_from_reports,
    _dtype_name,
    _first_explorer_diagnostics,
    _global_group_budgets,
    _memory_value,
    _merge_diagnostics,
    _process_memory_report,
    _record_typed_explorer_task_event,
    _torch_dtype_from_name,
    _typed_explorer_task_identity,
)
from ._factorized_de_operator import (
    FactorizedDERescueConfig,
    TypedExplorerLaunchResult,
    TypedExplorerLaunchState,
    TypedExplorerLaunchTask,
    _best_probe_row,
    _candidate_identity_key,
    _eval_ast_on_features,
    _finite_row_float,
    _finite_xy_rows,
    _fit_original_scale_affine_explorer_head,
    _material_improvement,
    _multiprocessing_start_method_name,
    _pooled_same_coord_coeff_target,
    _pooled_target_mse_from_local_ast,
    _residual_ratio_collapse_diagnostics,
    _safe_ratio_target,
    _select_lane_representative,
    _select_state_lane_candidates,
    _select_x_lane_candidates,
    _typed_explorer_worker_init,
    _typed_explorer_worker_run,
)
from ._factorized_de_lanes import (
    _EXPLORER_REFINE_DIAG_KEYS,
    _build_family_lane_candidates,
    _distill_state_lane_explorer_candidate,
    _distill_x_lane_explorer_candidate,
    _make_candidate_row,
)

def _record_explorer_refine_diagnostics(
    diagnostics: dict[str, Any] | None,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if diagnostics is None or not rows:
        return
    report = rows[0].get("explorer_diagnostics", None)
    if not isinstance(report, Mapping):
        return
    diagnostics.setdefault("explorer_refine_diagnostics", []).append(oracle_de._to_jsonable(dict(report)))
    for key in _EXPLORER_REFINE_DIAG_KEYS:
        value = report.get(key, None)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            _diag_inc(diagnostics, key, float(value))


def _record_explorer_launch_diagnostics(
    diagnostics: dict[str, Any] | None,
    *,
    lane: str,
    base_mode: str,
    order: int,
    carrier_ast,
    coord_ast,
    fit_rows_full: int,
    probe_rows_full: int,
    fit_rows_search: int,
    probe_rows_search: int,
    seed: int,
    sample_seed: int,
    wall_seconds: float,
    rows: Sequence[Mapping[str, Any]],
    memory_before: Mapping[str, Any] | None = None,
    memory_after: Mapping[str, Any] | None = None,
) -> None:
    if diagnostics is None:
        return
    explorer_diag = rows[0].get("explorer_diagnostics", None) if rows else None
    if not isinstance(explorer_diag, Mapping):
        explorer_diag = {}
    phase_diag = explorer_diag.get("phase_diagnostics", None)
    if not isinstance(phase_diag, Mapping):
        phase_diag = {}

    def _num(key: str, default: float | int = 0.0):
        value = explorer_diag.get(key, phase_diag.get(key, default))
        try:
            f = float(value)
        except Exception:
            return default
        if not math.isfinite(f):
            return default
        return f

    rss_before = _memory_value(memory_before, "rss_mb")
    rss_after = _memory_value(memory_after, "rss_mb")
    maxrss_before = _memory_value(memory_before, "maxrss_mb")
    maxrss_after = _memory_value(memory_after, "maxrss_mb")
    report = {
        "lane": str(lane),
        "base_mode": str(base_mode),
        "order": int(order),
        "carrier": repr(carrier_ast),
        "coord": repr(coord_ast),
        "fit_rows_full": int(fit_rows_full),
        "probe_rows_full": int(probe_rows_full),
        "fit_rows_search": int(fit_rows_search),
        "probe_rows_search": int(probe_rows_search),
        "seed": int(seed),
        "sample_seed": int(sample_seed),
        "wall_seconds": float(wall_seconds),
        "search_stop_reason": str(explorer_diag.get("search_stop_reason", phase_diag.get("search_stop_reason", "")) or ""),
        "setup_wall_s": float(_num("setup_wall_s")),
        "pool_eval_wall_s": float(_num("pool_eval_wall_s")),
        "brute_wall_s": float(_num("brute_wall_s")),
        "brute_scored": int(_num("brute_scored", 0)),
        "mutation_wall_s": float(_num("mutation_wall_s")),
        "score_calls": int(_num("score_calls", 0)),
        "prescore_calls": int(_num("prescore_calls", 0)),
        "full_score_calls": int(_num("full_score_calls", 0)),
        "base_score_s": float(_num("base_score_s")),
        "fit_poly_s": float(_num("fit_poly_s", _num("fit_poly_wall_seconds"))),
        "negated_variant_scores": int(_num("negated_variant_scores", 0)),
        "negated_variant_skipped_affine_poly_only": int(_num("negated_variant_skipped_affine_poly_only", 0)),
        "stall_checks": int(_num("stall_checks", 0)),
        "stall_triggered": int(_num("stall_triggered", 0)),
        "soft_restarts": int(_num("soft_restarts", 0)),
        "plateau_stop_requested": bool(explorer_diag.get("plateau_stop_requested", False)),
        "plateau_stop_eval": int(_num("plateau_stop_eval", 0)),
        "plateau_stop_soft_restarts": int(_num("plateau_stop_soft_restarts", 0)),
        "explorer_rows": int(len(rows)),
    }
    if rss_before is not None:
        report["process_rss_mb_before"] = float(rss_before)
    if rss_after is not None:
        report["process_rss_mb_after"] = float(rss_after)
    if rss_before is not None and rss_after is not None:
        report["process_rss_delta_mb"] = float(rss_after - rss_before)
    if maxrss_before is not None:
        report["process_maxrss_mb_before"] = float(maxrss_before)
    if maxrss_after is not None:
        report["process_maxrss_mb_after"] = float(maxrss_after)
    if maxrss_before is not None and maxrss_after is not None:
        report["process_maxrss_delta_mb"] = float(maxrss_after - maxrss_before)
    diagnostics.setdefault("explorer_launch_reports", []).append(oracle_de._to_jsonable(dict(report)))
    if str(lane or "") == "generic_coeff_on_carrier":
        diagnostics.setdefault("generic_explorer_launch_reports", []).append(oracle_de._to_jsonable(dict(report)))
    else:
        diagnostics.setdefault("typed_explorer_launch_reports", []).append(oracle_de._to_jsonable(dict(report)))


def _lane_subsample_xy(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    cap: int | None,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cap is None:
        return x, y
    n = int(x.shape[0])
    cap_i = int(cap)
    if cap_i <= 0 or n <= cap_i:
        return x, y

    x2 = x.reshape(n, -1)
    z_cpu = x2[:, 0].detach().cpu()
    y_cpu = y.reshape(n, -1)[:, 0].detach().cpu()
    finite = torch.isfinite(z_cpu) & torch.isfinite(y_cpu)
    finite_idx = torch.nonzero(finite, as_tuple=False).reshape(-1)
    if int(finite_idx.numel()) <= 0:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        idx = torch.randperm(n, generator=g)[:cap_i]
        return x[idx.to(device=x.device)], y[idx.to(device=y.device)]

    order = finite_idx[torch.argsort(z_cpu[finite_idx])]
    chosen_parts: list[torch.Tensor] = []
    edge = min(max(1, cap_i // 16), 8, int(order.numel()))
    chosen_parts.append(order[:edge])
    chosen_parts.append(order[-edge:])

    q_count = max(cap_i - 2 * edge, 1)
    q_count = min(q_count, int(order.numel()))
    if q_count > 0:
        q_pos = torch.linspace(
            0,
            int(order.numel()) - 1,
            steps=q_count,
            device=order.device,
        ).round().to(dtype=torch.long)
        chosen_parts.append(order[q_pos])

    tail = min(edge, int(finite_idx.numel()), cap_i)
    if tail > 0:
        y_tail = finite_idx[torch.topk(torch.abs(y_cpu[finite_idx]), k=tail, largest=True).indices]
        chosen_parts.append(y_tail)

    idx = torch.unique(torch.cat(chosen_parts)) if chosen_parts else torch.empty(0, dtype=torch.long)
    if int(idx.numel()) < cap_i:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        extra = torch.randperm(n, generator=g)
        if int(idx.numel()) > 0:
            keep = ~torch.isin(extra, idx)
            extra = extra[keep]
        need = cap_i - int(idx.numel())
        idx = torch.cat([idx, extra[:need]])
    idx = idx[:cap_i]
    return x[idx.to(device=x.device)], y[idx.to(device=y.device)]


def _subsample_lane_xy_parts(
    x_parts: Sequence[torch.Tensor],
    y_parts: Sequence[torch.Tensor],
    *,
    cap: int | None,
    seed: int,
    min_rows: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any]]:
    x_list = list(x_parts)
    y_list = list(y_parts)
    rows_before = int(sum(int(x.shape[0]) for x in x_list))
    cap_i = None if cap is None else int(cap)
    if cap_i is None or cap_i <= 0 or rows_before <= cap_i:
        return x_list, y_list, {
            "cap": None if cap_i is None or cap_i <= 0 else int(cap_i),
            "rows_before": rows_before,
            "rows_after": rows_before,
            "parts": [int(x.shape[0]) for x in x_list],
            "sampled": False,
        }

    budgets = _global_group_budgets(int(cap_i), len(x_list))
    sampled_x: list[torch.Tensor] = []
    sampled_y: list[torch.Tensor] = []
    before_parts: list[int] = []
    after_parts: list[int] = []
    for i, (x_part, y_part, budget) in enumerate(zip(x_list, y_list, budgets)):
        n_part = int(x_part.shape[0])
        before_parts.append(n_part)
        if n_part <= 0:
            sampled_x.append(x_part)
            sampled_y.append(y_part)
            after_parts.append(0)
            continue
        part_cap = min(n_part, max(int(budget), min(int(min_rows), n_part)))
        x_s, y_s = _lane_subsample_xy(
            x_part,
            y_part,
            cap=part_cap,
            seed=int(seed) + 1009 * (i + 1),
        )
        sampled_x.append(x_s)
        sampled_y.append(y_s)
        after_parts.append(int(x_s.shape[0]))

    rows_after = int(sum(after_parts))
    return sampled_x, sampled_y, {
        "cap": int(cap_i),
        "rows_before": rows_before,
        "rows_after": rows_after,
        "parts_before": before_parts,
        "parts_after": after_parts,
        "sampled": rows_after < rows_before,
    }


def _record_lane_subsample_diagnostics(
    diagnostics: dict[str, Any] | None,
    *,
    lane: str,
    base_mode: str,
    order: int,
    fit_report: Mapping[str, Any],
    probe_report: Mapping[str, Any],
) -> None:
    if diagnostics is None:
        return
    fit_before = int(fit_report.get("rows_before", 0) or 0)
    fit_after = int(fit_report.get("rows_after", fit_before) or fit_before)
    probe_before = int(probe_report.get("rows_before", 0) or 0)
    probe_after = int(probe_report.get("rows_after", probe_before) or probe_before)
    _diag_inc(diagnostics, "typed_explorer_fit_rows_before", fit_before)
    _diag_inc(diagnostics, "typed_explorer_fit_rows_after", fit_after)
    _diag_inc(diagnostics, "typed_explorer_probe_rows_before", probe_before)
    _diag_inc(diagnostics, "typed_explorer_probe_rows_after", probe_after)
    if bool(fit_report.get("sampled", False)) or bool(probe_report.get("sampled", False)):
        _diag_inc(diagnostics, "typed_explorer_subsampled_launches", 1)
    diagnostics.setdefault("typed_explorer_subsample_reports", []).append(
        {
            "lane": str(lane),
            "base_mode": str(base_mode),
            "order": int(order),
            "fit": oracle_de._to_jsonable(dict(fit_report)),
            "probe": oracle_de._to_jsonable(dict(probe_report)),
        }
    )


def _typed_explorer_caps_from_hp(hp) -> tuple[int | None, int | None]:
    # Correctness first: sampled typed-lane proposal searches can recover the
    # right family but miss coefficients enough to fail rollout validation.
    # Keep the sampler helpers available for explicit experiments, but default
    # production typed lanes to full-row scoring.
    return None, None


def _typed_explorer_caps_for_order(
    order: int,
    fit_cap: int | None,
    probe_cap: int | None,
) -> tuple[int | None, int | None, str]:
    if fit_cap is not None or probe_cap is not None:
        return fit_cap, probe_cap, "explicit"
    if int(order) == 2:
        return 2048, 4096, "second_order_resource_guard"
    return None, None, "disabled_correctness_first"


def _run_one_typed_explorer_launch(
    task: TypedExplorerLaunchTask,
    state: TypedExplorerLaunchState,
) -> TypedExplorerLaunchResult:
    diagnostics: dict[str, Any] = {}
    groups = list(state.groups)
    resid_fit_parts = list(state.resid_fit_parts)
    resid_probe_parts = list(state.resid_probe_parts)
    lane_norm = str(task.lane or "")
    carrier_ast = task.carrier_ast
    coord_ast = task.coord_ast

    if lane_norm == "x_coeff_on_u":
        pooled_fit = _pooled_same_coord_coeff_target(
            groups=groups,
            resid_parts=resid_fit_parts,
            carrier_ast=carrier_ast,
            coord_ast=coord_ast,
            split="fit",
            x_axis=int(task.x_axis),
            rel_eps=float(task.rel_eps),
            min_rows=int(task.min_ratio_rows),
        )
        pooled_probe = _pooled_same_coord_coeff_target(
            groups=groups,
            resid_parts=resid_probe_parts,
            carrier_ast=carrier_ast,
            coord_ast=coord_ast,
            split="probe",
            x_axis=int(task.x_axis),
            rel_eps=float(task.rel_eps),
            min_rows=int(task.min_ratio_rows),
        )
        if pooled_fit is None or pooled_probe is None:
            return TypedExplorerLaunchResult(
                task_id=int(task.task_id),
                ok=True,
                launched=False,
                rows=[],
                diagnostics=diagnostics,
                carrier_ast=carrier_ast,
                coord_ast=coord_ast,
            )
        z_fit_rows = [pooled_fit[0]]
        y_fit_rows = [pooled_fit[1]]
        z_probe_rows = [pooled_probe[0]]
        y_probe_rows = [pooled_probe[1]]
    else:
        z_fit_rows = []
        y_fit_rows = []
        z_probe_rows = []
        y_probe_rows = []

        for group, resid_fit, resid_probe in zip(groups, resid_fit_parts, resid_probe_parts):
            phi_fit = _eval_ast_on_features(
                carrier_ast, features=group.features, split="fit", x_axis=int(task.x_axis)
            ).reshape(-1)
            phi_probe = _eval_ast_on_features(
                carrier_ast, features=group.features, split="probe", x_axis=int(task.x_axis)
            ).reshape(-1)
            z_fit = _eval_ast_on_features(
                coord_ast, features=group.features, split="fit", x_axis=int(task.x_axis)
            ).reshape(-1, 1)
            z_probe = _eval_ast_on_features(
                coord_ast, features=group.features, split="probe", x_axis=int(task.x_axis)
            ).reshape(-1, 1)

            ratio_fit, mask_fit = _safe_ratio_target(resid_fit, phi_fit, rel_eps=float(task.rel_eps))
            ratio_probe, mask_probe = _safe_ratio_target(resid_probe, phi_probe, rel_eps=float(task.rel_eps))
            if ratio_fit is None or ratio_probe is None:
                continue

            z_fit_valid, ratio_fit_valid = _finite_xy_rows(z_fit[mask_fit], ratio_fit)
            z_probe_valid, ratio_probe_valid = _finite_xy_rows(z_probe[mask_probe], ratio_probe)
            if (
                int(z_fit_valid.shape[0]) < int(task.min_ratio_rows)
                or int(z_probe_valid.shape[0]) < int(task.min_ratio_rows)
            ):
                continue

            z_fit_rows.append(z_fit_valid)
            y_fit_rows.append(ratio_fit_valid)
            z_probe_rows.append(z_probe_valid)
            y_probe_rows.append(ratio_probe_valid)

    if not z_fit_rows or not z_probe_rows:
        return TypedExplorerLaunchResult(
            task_id=int(task.task_id),
            ok=True,
            launched=False,
            rows=[],
            diagnostics=diagnostics,
            carrier_ast=carrier_ast,
            coord_ast=coord_ast,
        )

    z_fit_search_rows, y_fit_search_rows, fit_sample_report = _subsample_lane_xy_parts(
        z_fit_rows,
        y_fit_rows,
        cap=task.explorer_fit_cap,
        seed=int(task.sample_seed),
        min_rows=int(task.min_ratio_rows),
    )
    z_probe_search_rows, y_probe_search_rows, probe_sample_report = _subsample_lane_xy_parts(
        z_probe_rows,
        y_probe_rows,
        cap=task.explorer_probe_cap,
        seed=int(task.sample_seed) + 7919,
        min_rows=int(task.min_ratio_rows),
    )
    _record_lane_subsample_diagnostics(
        diagnostics,
        lane=str(task.lane),
        base_mode=str(task.base_mode),
        order=int(task.order),
        fit_report=fit_sample_report,
        probe_report=probe_sample_report,
    )

    dtype = _torch_dtype_from_name(task.dtype_name)
    x_fit_search = torch.cat(z_fit_search_rows, dim=0).to(dtype=dtype)
    y_fit_search = torch.cat(y_fit_search_rows, dim=0).to(dtype=dtype)
    x_probe_search = torch.cat(z_probe_search_rows, dim=0).to(dtype=dtype)
    y_probe_search = torch.cat(y_probe_search_rows, dim=0).to(dtype=dtype)
    fit_rows_full = int(sum(int(x.shape[0]) for x in z_fit_rows))
    probe_rows_full = int(sum(int(x.shape[0]) for x in z_probe_rows))
    task_hash, task_key = _typed_explorer_task_identity(
        lane=str(task.lane),
        base_mode=str(task.base_mode),
        order=int(task.order),
        carrier_ast=carrier_ast,
        coord_ast=coord_ast,
        seed=int(task.seed),
        sample_seed=int(task.sample_seed),
    )
    task_common = {
        "task_id": int(task.task_id),
        "task_hash": str(task_hash),
        "task_key": str(task_key),
        "lane": str(task.lane),
        "base_mode": str(task.base_mode),
        "order": int(task.order),
        "carrier": repr(carrier_ast),
        "coord": repr(coord_ast),
        "fit_rows_full": int(fit_rows_full),
        "probe_rows_full": int(probe_rows_full),
        "fit_rows_search": int(x_fit_search.shape[0]),
        "probe_rows_search": int(x_probe_search.shape[0]),
        "n_iter": int(task.n_iter),
        "max_depth": int(task.max_depth),
        "explorer_topk": int(task.explorer_topk),
        "seed": int(task.seed),
        "sample_seed": int(task.sample_seed),
    }
    _record_typed_explorer_task_event(diagnostics, dict(task_common, event="planned"))
    launch_started = time.perf_counter()
    launch_memory_before = _process_memory_report("typed_explorer_launch_before")
    _record_typed_explorer_task_event(
        diagnostics,
        dict(
            task_common,
            event="started",
            pid=int(os.getpid()),
            rss_mb=_memory_value(launch_memory_before, "rss_mb"),
            maxrss_mb=_memory_value(launch_memory_before, "maxrss_mb"),
        ),
    )
    try:
        rows_raw = run_explorer(
            x_fit_data=x_fit_search,
            y_fit_data=y_fit_search,
            x_probe_data=x_probe_search,
            y_probe_data=y_probe_search,
            n_iter=int(task.n_iter),
            max_depth=int(task.max_depth),
            poly_degree=1,
            score_mapping_family_mode="poly_only",
            brute_score_mapping_family_mode="poly_only",
            score_pade_structural_enable=True,
            plateau_stop_enable=True,
            plateau_stop_max_soft_restarts=2,
            plateau_stop_min_evals=2000,
            return_topk=int(task.explorer_topk),
            seed=int(task.seed),
            dtype=dtype,
        )
        rows = list(rows_raw or [])
    except Exception as exc:
        launch_memory_after = _process_memory_report("typed_explorer_launch_failed")
        _record_typed_explorer_task_event(
            diagnostics,
            dict(
                task_common,
                event="failed",
                pid=int(os.getpid()),
                wall_seconds=float(time.perf_counter() - launch_started),
                rss_mb=_memory_value(launch_memory_after, "rss_mb"),
                maxrss_mb=_memory_value(launch_memory_after, "maxrss_mb"),
                error_type=type(exc).__name__,
                error=str(exc),
            ),
        )
        return TypedExplorerLaunchResult(
            task_id=int(task.task_id),
            ok=False,
            launched=True,
            rows=[],
            diagnostics=diagnostics,
            carrier_ast=carrier_ast,
            coord_ast=coord_ast,
            fit_rows_full=int(fit_rows_full),
            probe_rows_full=int(probe_rows_full),
            fit_rows_search=int(x_fit_search.shape[0]),
            probe_rows_search=int(x_probe_search.shape[0]),
            error=f"{type(exc).__name__}: {exc}",
        )

    launch_memory_after = _process_memory_report("typed_explorer_launch_after")
    launch_wall_seconds = float(time.perf_counter() - launch_started)
    explorer_diag, phase_diag = _first_explorer_diagnostics(rows)
    best_mse = _best_numeric_row_value(rows, ("probe_mse", "local_probe_mse", "mse", "fit_mse"))
    if best_mse is None:
        best_mse = _best_numeric_row_value(rows, ("score",))
    _record_typed_explorer_task_event(
        diagnostics,
        dict(
            task_common,
            event="finished",
            pid=int(os.getpid()),
            wall_seconds=float(launch_wall_seconds),
            rows=int(len(rows)),
            best_mse=best_mse,
            search_stop_reason=str(
                explorer_diag.get("search_stop_reason", phase_diag.get("search_stop_reason", "")) or ""
            ),
            score_calls=int(_diag_number_from_reports(explorer_diag, phase_diag, "score_calls", 0) or 0),
            prescore_calls=int(_diag_number_from_reports(explorer_diag, phase_diag, "prescore_calls", 0) or 0),
            full_score_calls=int(_diag_number_from_reports(explorer_diag, phase_diag, "full_score_calls", 0) or 0),
            base_score_s=float(_diag_number_from_reports(explorer_diag, phase_diag, "base_score_s", 0.0) or 0.0),
            fit_poly_s=float(
                _diag_number_from_reports(
                    explorer_diag,
                    phase_diag,
                    "fit_poly_s",
                    _diag_number_from_reports(explorer_diag, phase_diag, "fit_poly_wall_seconds", 0.0),
                )
                or 0.0
            ),
            negated_variant_scores=int(
                _diag_number_from_reports(explorer_diag, phase_diag, "negated_variant_scores", 0) or 0
            ),
            rss_mb=_memory_value(launch_memory_after, "rss_mb"),
            maxrss_mb=_memory_value(launch_memory_after, "maxrss_mb"),
        ),
    )
    _record_explorer_launch_diagnostics(
        diagnostics,
        lane=str(task.lane),
        base_mode=str(task.base_mode),
        order=int(task.order),
        carrier_ast=carrier_ast,
        coord_ast=coord_ast,
        fit_rows_full=int(fit_rows_full),
        probe_rows_full=int(probe_rows_full),
        fit_rows_search=int(x_fit_search.shape[0]),
        probe_rows_search=int(x_probe_search.shape[0]),
        seed=int(task.seed),
        sample_seed=int(task.sample_seed),
        wall_seconds=float(launch_wall_seconds),
        rows=rows,
        memory_before=launch_memory_before,
        memory_after=launch_memory_after,
    )
    _record_explorer_refine_diagnostics(diagnostics, rows)
    return TypedExplorerLaunchResult(
        task_id=int(task.task_id),
        ok=True,
        launched=True,
        rows=rows,
        diagnostics=diagnostics,
        carrier_ast=carrier_ast,
        coord_ast=coord_ast,
        fit_rows_full=int(fit_rows_full),
        probe_rows_full=int(probe_rows_full),
        fit_rows_search=int(x_fit_search.shape[0]),
        probe_rows_search=int(x_probe_search.shape[0]),
    )


def _finalize_typed_explorer_launch_in_parent(
    result: TypedExplorerLaunchResult,
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
    coord_ast,
    coord_asts: Sequence[Any],
    rel_eps: float,
    min_ratio_rows: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in list(result.rows):
        coeff_ast, mapping, ratio_probe_mse, coeff_local_ast = _fit_original_scale_affine_explorer_head(
            row=row,
            groups=groups,
            order=int(order),
            x_axis=int(x_axis),
            resid_fit_parts=resid_fit_parts,
            resid_probe_parts=resid_probe_parts,
            carrier_ast=carrier_ast,
            coord_ast=coord_ast,
            rel_eps=float(rel_eps),
            min_ratio_rows=int(min_ratio_rows),
        )
        if coeff_ast is None:
            continue
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
        explorer_cand = _make_candidate_row(
            lane=str(lane),
            family="explorer",
            base_mode=str(base_mode),
            groups=groups,
            order=int(order),
            x_axis=int(x_axis),
            base_ast=base_ast,
            carrier_ast=carrier_ast,
            coord_ast=coord_ast,
            coeff_ast=coeff_ast,
            coeff_local_ast=coeff_local_ast,
            coeff_expr=str(row.get("expr", "")),
            mapping=mapping,
            size=int(row.get("size", 0)),
            ratio_probe_mse=float(ratio_probe_mse),
            fit_target_mse=None if fit_target_mse is None else float(fit_target_mse),
            probe_target_mse=None if probe_target_mse is None else float(probe_target_mse),
            resid_probe_parts=resid_probe_parts,
            rel_eps=float(rel_eps),
        )
        if explorer_cand is None:
            continue
        if coeff_local_ast is not None:
            explorer_cand["coeff_local_ast"] = coeff_local_ast
        out.append(explorer_cand)
        if str(lane or "") == "x_coeff_on_u":
            out.extend(
                _distill_x_lane_explorer_candidate(
                    explorer_candidate=explorer_cand,
                    groups=groups,
                    order=int(order),
                    x_axis=int(x_axis),
                    base_ast=base_ast,
                    base_mode=str(base_mode),
                    resid_fit_parts=resid_fit_parts,
                    resid_probe_parts=resid_probe_parts,
                    carrier_ast=carrier_ast,
                    coord_ast=coord_ast,
                    rel_eps=float(rel_eps),
                    min_ratio_rows=int(min_ratio_rows),
                )
            )
        elif str(lane or "") == "state_nonlinearity":
            out.extend(
                _distill_state_lane_explorer_candidate(
                    explorer_candidate=explorer_cand,
                    groups=groups,
                    order=int(order),
                    x_axis=int(x_axis),
                    base_ast=base_ast,
                    base_mode=str(base_mode),
                    resid_fit_parts=resid_fit_parts,
                    resid_probe_parts=resid_probe_parts,
                    carrier_ast=carrier_ast,
                    coord_asts=coord_asts,
                    rel_eps=float(rel_eps),
                    min_ratio_rows=int(min_ratio_rows),
                )
            )
    return out


def _build_typed_explorer_launch_work_plan(
    *,
    lane: str,
    base_mode: str,
    order: int,
    x_axis: int,
    carrier_asts: Sequence[Any],
    coord_asts: Sequence[Any],
    rel_eps: float,
    min_ratio_rows: int,
    n_iter: int,
    max_depth: int,
    explorer_topk: int,
    seed: int,
    dtype: torch.dtype,
    explorer_fit_cap: int | None,
    explorer_probe_cap: int | None,
) -> list[TypedExplorerLaunchTask]:
    tasks: list[TypedExplorerLaunchTask] = []
    task_id = 0
    for carrier_ast in list(carrier_asts):
        for coord_ast in list(coord_asts):
            task_id += 1
            sample_seed = int(seed) + 104729 * int(task_id) + 1000003 * int(order)
            tasks.append(
                TypedExplorerLaunchTask(
                    task_id=int(task_id),
                    lane=str(lane),
                    base_mode=str(base_mode),
                    order=int(order),
                    x_axis=int(x_axis),
                    carrier_ast=carrier_ast,
                    coord_ast=coord_ast,
                    rel_eps=float(rel_eps),
                    min_ratio_rows=int(min_ratio_rows),
                    n_iter=int(n_iter),
                    max_depth=int(max_depth),
                    explorer_topk=int(explorer_topk),
                    seed=int(seed),
                    sample_seed=int(sample_seed),
                    dtype_name=_dtype_name(dtype),
                    explorer_fit_cap=explorer_fit_cap,
                    explorer_probe_cap=explorer_probe_cap,
                )
            )
    return tasks


def _run_typed_explorer_work_plan_serial(
    tasks: Sequence[TypedExplorerLaunchTask],
    state: TypedExplorerLaunchState,
    *,
    diagnostics: dict[str, Any] | None = None,
) -> list[TypedExplorerLaunchResult]:
    results: list[TypedExplorerLaunchResult] = []
    for task in sorted(list(tasks), key=lambda item: int(item.task_id)):
        result = _run_one_typed_explorer_launch(task, state)
        if diagnostics is not None:
            _merge_diagnostics(diagnostics, result.diagnostics)
        if not bool(result.ok):
            raise RuntimeError(
                f"typed explorer launch failed task_id={result.task_id} "
                f"order={task.order} base={task.base_mode} lane={task.lane}: {result.error}"
            )
        results.append(result)
    return results


def _run_typed_explorer_work_plan_process(
    tasks: Sequence[TypedExplorerLaunchTask],
    state: TypedExplorerLaunchState,
    *,
    max_workers: int,
    diagnostics: dict[str, Any] | None = None,
) -> list[TypedExplorerLaunchResult]:
    tasks_sorted = sorted(list(tasks), key=lambda item: int(item.task_id))
    if int(max_workers) <= 1 or len(tasks_sorted) <= 1:
        return _run_typed_explorer_work_plan_serial(tasks_sorted, state, diagnostics=diagnostics)

    worker_count = min(int(max_workers), len(tasks_sorted))
    if diagnostics is not None:
        _diag_inc(diagnostics, "typed_process_pool_launches", 1)
        _diag_inc(diagnostics, "typed_process_pool_tasks", len(tasks_sorted))
        diagnostics["typed_tasks_inflight_peak"] = max(
            int(diagnostics.get("typed_tasks_inflight_peak", 0) or 0),
            int(worker_count),
        )
        diagnostics.setdefault("typed_process_pool_reports", []).append(
            {
                "workers": int(worker_count),
                "tasks": int(len(tasks_sorted)),
                "start_method": _multiprocessing_start_method_name(),
            }
        )

    results_by_id: dict[int, TypedExplorerLaunchResult] = {}
    failed_task_ids: set[int] = set()
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(worker_count),
            initializer=_typed_explorer_worker_init,
            initargs=(state,),
        ) as executor:
            future_to_task = {
                executor.submit(_typed_explorer_worker_run, task): task
                for task in tasks_sorted
            }
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failed_task_ids.add(int(task.task_id))
                    if diagnostics is not None:
                        _diag_inc(diagnostics, "typed_process_pool_future_failures", 1)
                        diagnostics.setdefault("typed_process_pool_errors", []).append(
                            {
                                "task_id": int(task.task_id),
                                "lane": str(task.lane),
                                "base_mode": str(task.base_mode),
                                "order": int(task.order),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                    continue
                if diagnostics is not None:
                    _merge_diagnostics(diagnostics, result.diagnostics)
                if bool(result.ok):
                    results_by_id[int(task.task_id)] = result
                else:
                    failed_task_ids.add(int(task.task_id))
    except Exception as exc:
        if diagnostics is not None:
            _diag_inc(diagnostics, "typed_process_pool_failures", 1)
            diagnostics.setdefault("typed_process_pool_errors", []).append(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    missing_tasks = [task for task in tasks_sorted if int(task.task_id) not in results_by_id]
    if missing_tasks and diagnostics is not None:
        _diag_inc(diagnostics, "typed_tasks_rerun_serial", len(missing_tasks))
    for task in missing_tasks:
        result = _run_one_typed_explorer_launch(task, state)
        if diagnostics is not None:
            _merge_diagnostics(diagnostics, result.diagnostics)
        if not bool(result.ok):
            raise RuntimeError(
                f"typed explorer launch failed after serial rerun task_id={result.task_id} "
                f"order={task.order} base={task.base_mode} lane={task.lane}: {result.error}"
            )
        results_by_id[int(task.task_id)] = result
        failed_task_ids.discard(int(task.task_id))

    return [results_by_id[int(task.task_id)] for task in tasks_sorted]


def _build_explorer_lane_candidates(
    *,
    lane: str,
    base_mode: str,
    groups: Sequence[DEFeatureGroup],
    order: int,
    x_axis: int,
    base_ast,
    resid_fit_parts: Sequence[torch.Tensor],
    resid_probe_parts: Sequence[torch.Tensor],
    carrier_asts: Sequence[Any],
    coord_asts: Sequence[Any],
    rel_eps: float,
    min_ratio_rows: int,
    n_iter: int,
    max_depth: int,
    explorer_topk: int,
    seed: int,
    dtype: torch.dtype,
    explorer_fit_cap: int | None = None,
    explorer_probe_cap: int | None = None,
    explorer_workers: int = 1,
    diagnostics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    out: list[dict[str, Any]] = []
    pairs_with_targets = 0
    launches = 0
    explorer_rows = 0
    carrier_list = list(carrier_asts)
    coord_list = list(coord_asts)
    state = TypedExplorerLaunchState(
        groups=groups,
        resid_fit_parts=resid_fit_parts,
        resid_probe_parts=resid_probe_parts,
    )
    work_plan = _build_typed_explorer_launch_work_plan(
        lane=str(lane),
        base_mode=str(base_mode),
        order=int(order),
        x_axis=int(x_axis),
        carrier_asts=carrier_list,
        coord_asts=coord_list,
        rel_eps=float(rel_eps),
        min_ratio_rows=int(min_ratio_rows),
        n_iter=int(n_iter),
        max_depth=int(max_depth),
        explorer_topk=int(explorer_topk),
        seed=int(seed),
        dtype=dtype,
        explorer_fit_cap=explorer_fit_cap,
        explorer_probe_cap=explorer_probe_cap,
    )
    pairs_considered = int(len(work_plan))
    use_process_backend = int(explorer_workers) > 1 and int(len(work_plan)) > 1
    work_plan_execution = "process" if use_process_backend else "serial"
    process_workers = min(max(1, int(explorer_workers)), max(1, int(len(work_plan))))
    if diagnostics is not None:
        _diag_inc(diagnostics, "explorer_work_plan_tasks", pairs_considered)
        if str(lane or "") == "generic_coeff_on_carrier":
            _diag_inc(diagnostics, "generic_work_plan_tasks", pairs_considered)
        else:
            _diag_inc(diagnostics, "typed_work_plan_tasks", pairs_considered)
        diagnostics.setdefault("explorer_work_plan_reports", []).append(
            {
                "lane": str(lane),
                "base_mode": str(base_mode),
                "order": int(order),
                "tasks": int(pairs_considered),
                "carrier_count": int(len(carrier_list)),
                "coord_count": int(len(coord_list)),
                "execution": str(work_plan_execution),
                "workers": int(process_workers if use_process_backend else 1),
            }
        )

    if use_process_backend:
        results = _run_typed_explorer_work_plan_process(
            work_plan,
            state,
            max_workers=int(process_workers),
            diagnostics=diagnostics,
        )
    else:
        results = _run_typed_explorer_work_plan_serial(work_plan, state, diagnostics=diagnostics)
    for result in sorted(results, key=lambda item: int(item.task_id)):
        if not bool(result.launched):
            continue
        if result.carrier_ast is None or result.coord_ast is None:
            raise RuntimeError(f"typed explorer launch missing AST context for task_id={result.task_id}")
        pairs_with_targets += 1
        launches += 1
        explorer_rows += int(len(result.rows))
        out.extend(
            _finalize_typed_explorer_launch_in_parent(
                result,
                lane=str(lane),
                base_mode=str(base_mode),
                groups=groups,
                order=int(order),
                x_axis=int(x_axis),
                base_ast=base_ast,
                resid_fit_parts=resid_fit_parts,
                resid_probe_parts=resid_probe_parts,
                carrier_ast=result.carrier_ast,
                coord_ast=result.coord_ast,
                coord_asts=coord_list,
                rel_eps=float(rel_eps),
                min_ratio_rows=int(min_ratio_rows),
            )
        )
    if diagnostics is not None:
        elapsed = float(time.perf_counter() - started)
        _diag_inc(diagnostics, "explorer_pairs_considered", pairs_considered)
        _diag_inc(diagnostics, "explorer_pairs_with_targets", pairs_with_targets)
        _diag_inc(diagnostics, "explorer_launches", launches)
        _diag_inc(diagnostics, "explorer_rows", explorer_rows)
        _diag_inc(diagnostics, "explorer_candidates", len(out))
        _diag_inc(diagnostics, "explorer_wall_seconds", elapsed)
        if str(lane or "") == "generic_coeff_on_carrier":
            _diag_inc(diagnostics, "generic_explorer_launches", launches)
            _diag_inc(diagnostics, "generic_explorer_candidates", len(out))
        else:
            _diag_inc(diagnostics, "typed_explorer_launches", launches)
            _diag_inc(diagnostics, "typed_explorer_candidates", len(out))
        diagnostics.setdefault("explorer_lane_calls", []).append(
            {
                "lane": str(lane),
                "base_mode": str(base_mode),
                "order": int(order),
                "carrier_count": int(len(carrier_list)),
                "coord_count": int(len(coord_list)),
                "work_plan_tasks": int(pairs_considered),
                "work_plan_execution": str(work_plan_execution),
                "work_plan_workers": int(process_workers if use_process_backend else 1),
                "pairs_considered": int(pairs_considered),
                "pairs_with_targets": int(pairs_with_targets),
                "launches": int(launches),
                "explorer_rows": int(explorer_rows),
                "candidates": int(len(out)),
                "n_iter": int(n_iter),
                "max_depth": int(max_depth),
                "explorer_topk": int(explorer_topk),
                "explorer_fit_cap": None if explorer_fit_cap is None else int(explorer_fit_cap),
                "explorer_probe_cap": None if explorer_probe_cap is None else int(explorer_probe_cap),
                "wall_seconds": elapsed,
            }
        )
    return out


def _carrier_role(node, *, x_axis: int) -> str:
    if isinstance(node, ConstNode):
        try:
            value = float(node.value)
        except Exception:
            value = float("nan")
        if math.isfinite(value) and abs(value - 1.0) < 1.0e-12:
            return "1"
        return "const"
    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        kwargs = dict(getattr(node, "kwargs", {}) or {})
        if kind in ("u", "field", "state"):
            return "u"
        if kind in ("du", "d1u", "grad_u") and int(kwargs.get("axis", x_axis)) == int(x_axis):
            return "du"
    return repr(node)


def _should_run_two_block_shared_coord(
    mode: str,
    *,
    order: int,
    baseline_probe_rms: float,
    best_single_probe_rms: float,
    replace_rel_factor: float,
) -> bool:
    mode_norm = str(mode or "never").strip().lower()
    if mode_norm == "never":
        return False
    if int(order) != 2:
        return False
    if mode_norm == "always":
        return True
    if not math.isfinite(best_single_probe_rms):
        return True
    if not math.isfinite(baseline_probe_rms):
        return True
    return not bool(best_single_probe_rms < float(replace_rel_factor) * baseline_probe_rms)


def _family_first_gate_decision(
    *,
    lane: str,
    base_mode: str,
    order: int,
    family_rows: Sequence[dict[str, Any]],
    baseline_probe_rms: float,
    replace_rel_factor: float,
    trigger_val_rms: float | None = None,
    min_domain_safe_fraction: float = 0.75,
    min_collapse_coverage: float = 0.75,
    max_collapse_score: float = 0.25,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    rows = [row for row in list(family_rows) if str(row.get("family", "")) != "explorer"]
    best = _best_probe_row(rows)
    reason = "no_family_candidate"
    skip = False
    if best is not None:
        probe_rms = _finite_row_float(best, "probe_rms")
        domain_safe = float(best.get("collapse_domain_safe_fraction", 0.0) or 0.0)
        collapse_confidence = str(best.get("collapse_confidence", "") or "")
        collapse_reason = str(best.get("collapse_reason", "") or "")
        collapse_coverage = float(best.get("collapse_coverage", 0.0) or 0.0)
        collapse_score = _finite_row_float(best, "collapse_score")
        trigger_rms = float(trigger_val_rms) if trigger_val_rms is not None else float("inf")
        absolute_certified = math.isfinite(trigger_rms) and trigger_rms > 0.0 and probe_rms <= trigger_rms
        relative_certified = (
            math.isfinite(float(baseline_probe_rms))
            and float(baseline_probe_rms) > 0.0
            and probe_rms <= 0.25 * float(baseline_probe_rms)
        )
        if not math.isfinite(probe_rms):
            reason = "nonfinite_probe_residual"
        elif not _material_improvement(
            probe_rms,
            float(baseline_probe_rms),
            replace_rel_factor=float(replace_rel_factor),
        ):
            reason = "insufficient_residual_improvement"
        elif collapse_confidence != "high" or collapse_reason != "ok":
            reason = "weak_collapse_evidence"
        elif not math.isfinite(collapse_score) or collapse_score > float(max_collapse_score):
            reason = "weak_collapse_score"
        elif collapse_coverage < float(min_collapse_coverage):
            reason = "low_collapse_coverage"
        elif domain_safe < float(min_domain_safe_fraction):
            reason = "low_domain_safety"
        elif not (absolute_certified or relative_certified):
            reason = "insufficient_certification_margin"
        else:
            skip = True
            reason = "family_certified"

    if diagnostics is not None:
        _diag_inc(diagnostics, "family_gate_evaluations", 1)
        if skip:
            _diag_inc(diagnostics, "family_gate_passes", 1)
            _diag_inc(diagnostics, "explorer_skipped", 1)
        diagnostics.setdefault("family_gate_reports", []).append(
            {
                "lane": str(lane),
                "base_mode": str(base_mode),
                "order": int(order),
                "family_candidates": int(len(rows)),
                "skip_explorer": bool(skip),
                "explorer_skipped_reason": str(reason),
                "best_family": None if best is None else str(best.get("family", "")),
                "best_probe_rms": None
                if best is None or not math.isfinite(_finite_row_float(best, "probe_rms"))
                else _finite_row_float(best, "probe_rms"),
                "collapse_score": None
                if best is None or not math.isfinite(_finite_row_float(best, "collapse_score"))
                else _finite_row_float(best, "collapse_score"),
                "collapse_confidence": "" if best is None else str(best.get("collapse_confidence", "")),
                "collapse_coverage": 0.0 if best is None else float(best.get("collapse_coverage", 0.0) or 0.0),
                "collapse_domain_safe_fraction": 0.0
                if best is None
                else float(best.get("collapse_domain_safe_fraction", 0.0) or 0.0),
                "trigger_val_rms": None
                if trigger_val_rms is None or not math.isfinite(float(trigger_val_rms))
                else float(trigger_val_rms),
            }
        )
    return bool(skip), str(reason), best


def _collapse_scheduler_confidence_rank(confidence: str) -> int:
    return {"high": 0, "weak": 1, "low": 2}.get(str(confidence or ""), 3)


def _schedule_explorer_coord_asts_by_collapse(
    *,
    lane: str,
    base_mode: str,
    order: int,
    groups: Sequence[DEFeatureGroup],
    resid_probe_parts: Sequence[torch.Tensor],
    carrier_ast,
    coord_asts: Sequence[Any],
    x_axis: int,
    rel_eps: float,
    max_high_confidence_coords: int = 2,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    reports: list[dict[str, Any]] = []
    for idx, coord_ast in enumerate(list(coord_asts)):
        collapse = _residual_ratio_collapse_diagnostics(
            groups=groups,
            resid_parts=resid_probe_parts,
            carrier_ast=carrier_ast,
            coord_ast=coord_ast,
            split="probe",
            x_axis=int(x_axis),
            rel_eps=float(rel_eps),
            min_rows=2,
        )
        reports.append(
            {
                "lane": str(lane),
                "base_mode": str(base_mode),
                "order": int(order),
                "coord_index": int(idx),
                "coord_ast": repr(coord_ast),
                **collapse,
            }
        )

    if not reports:
        return [], []

    high_conf = [
        row
        for row in reports
        if str(row.get("collapse_confidence", "")) == "high" and str(row.get("collapse_reason", "")) == "ok"
    ]
    if high_conf:
        ranked = sorted(
            high_conf,
            key=lambda row: (
                _collapse_scheduler_confidence_rank(str(row.get("collapse_confidence", ""))),
                float(row.get("collapse_score", float("inf"))),
                -float(row.get("collapse_coverage", 0.0) or 0.0),
                -float(row.get("collapse_domain_safe_fraction", 0.0) or 0.0),
                int(row.get("coord_index", 0)),
            ),
        )
        if len(ranked) <= 4:
            keep_indices = {int(row["coord_index"]) for row in ranked}
            scheduler_reason = "high_confidence_keep_all_small_pool"
        else:
            keep_indices = {
                int(row["coord_index"])
                for row in ranked[: max(1, int(max_high_confidence_coords))]
            }
            scheduler_reason = "high_confidence_topk"
    else:
        keep_indices = {int(row["coord_index"]) for row in reports}
        scheduler_reason = "inconclusive_keep_all"

    selected = [coord_ast for idx, coord_ast in enumerate(list(coord_asts)) if int(idx) in keep_indices]
    for row in reports:
        row["selected_for_explorer"] = bool(int(row["coord_index"]) in keep_indices)
        row["scheduler_reason"] = str(scheduler_reason)

    if diagnostics is not None:
        considered = int(len(reports))
        skipped = int(sum(1 for row in reports if not bool(row["selected_for_explorer"])))
        _diag_inc(diagnostics, "scheduler_coord_candidates_considered", considered)
        _diag_inc(diagnostics, "scheduler_coord_candidates_skipped", skipped)
        diagnostics.setdefault("lane_scheduler_reports", []).append(
            {
                "lane": str(lane),
                "base_mode": str(base_mode),
                "order": int(order),
                "coord_candidates": int(considered),
                "coords_selected": int(len(selected)),
                "coords_skipped": int(skipped),
                "reason": str(scheduler_reason),
                "coord_reports": reports,
            }
        )

    return selected, reports


def _select_typed_lane_candidates(
    lane: str,
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    lane_norm = str(lane or "")
    if lane_norm == "x_coeff_on_u":
        return _select_x_lane_candidates(rows)
    if lane_norm == "state_nonlinearity":
        return _select_state_lane_candidates(rows)
    representative = _select_lane_representative(rows)
    return representative, [] if representative is None else [representative]


def _build_typed_lane_candidates_with_gate(
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
    baseline_probe_rms: float,
    rescue_cfg: FactorizedDERescueConfig,
    n_iter: int,
    max_depth: int,
    explorer_topk: int,
    seed: int,
    dtype: torch.dtype,
    explorer_fit_cap: int | None = None,
    explorer_probe_cap: int | None = None,
    explorer_workers: int = 1,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    family_rows = _build_family_lane_candidates(
        lane=str(lane),
        base_mode=str(base_mode),
        groups=groups,
        order=int(order),
        x_axis=int(x_axis),
        base_ast=base_ast,
        resid_fit_parts=resid_fit_parts,
        resid_probe_parts=resid_probe_parts,
        carrier_ast=carrier_ast,
        coord_asts=coord_asts,
        family_names=family_names,
        rel_eps=float(rescue_cfg.ratio_rel_eps),
        min_ratio_rows=int(rescue_cfg.min_ratio_rows),
        diagnostics=diagnostics,
    )
    skip_explorer, skip_reason, gate_row = _family_first_gate_decision(
        lane=str(lane),
        base_mode=str(base_mode),
        order=int(order),
        family_rows=family_rows,
        baseline_probe_rms=float(baseline_probe_rms),
        replace_rel_factor=float(getattr(rescue_cfg, "replace_rel_factor", 0.98)),
        trigger_val_rms=float(getattr(rescue_cfg, "trigger_val_rms", 1.0e-3)),
        diagnostics=diagnostics,
    )

    if skip_explorer:
        explorer_rows: list[dict[str, Any]] = []
    else:
        explorer_coord_asts, _scheduler_reports = _schedule_explorer_coord_asts_by_collapse(
            lane=str(lane),
            base_mode=str(base_mode),
            order=int(order),
            groups=groups,
            resid_probe_parts=resid_probe_parts,
            carrier_ast=carrier_ast,
            coord_asts=coord_asts,
            x_axis=int(x_axis),
            rel_eps=float(rescue_cfg.ratio_rel_eps),
            diagnostics=diagnostics,
        )
        explorer_rows = _build_explorer_lane_candidates(
            lane=str(lane),
            base_mode=str(base_mode),
            groups=groups,
            order=int(order),
            x_axis=int(x_axis),
            base_ast=base_ast,
            resid_fit_parts=resid_fit_parts,
            resid_probe_parts=resid_probe_parts,
            carrier_asts=[carrier_ast],
            coord_asts=explorer_coord_asts,
            rel_eps=float(rescue_cfg.ratio_rel_eps),
            min_ratio_rows=int(rescue_cfg.min_ratio_rows),
            n_iter=int(n_iter),
            max_depth=int(max_depth),
            explorer_topk=int(explorer_topk),
            seed=int(seed),
            dtype=dtype,
            explorer_fit_cap=explorer_fit_cap,
            explorer_probe_cap=explorer_probe_cap,
            explorer_workers=int(explorer_workers),
            diagnostics=diagnostics,
        )

    choice, kept_rows = _select_typed_lane_candidates(str(lane), family_rows + explorer_rows)
    rows = list(kept_rows)
    if skip_explorer and gate_row is not None:
        gate_key = _candidate_identity_key(gate_row)
        rows = [gate_row] + [row for row in rows if _candidate_identity_key(row) != gate_key]

    report = {
        "lane": str(lane),
        "base_mode": str(base_mode),
        "order": int(order),
        "family_candidates": int(len(family_rows)),
        "explorer_candidates": int(len(explorer_rows)),
        "kept_candidates": int(len(rows)),
        "skip_explorer": bool(skip_explorer),
        "explorer_skipped_reason": str(skip_reason),
        "selected_family": "" if choice is None else str(choice.get("family", "")),
    }
    return rows, family_rows + explorer_rows, report

__factorized_de_definitions__ = (
    "_record_explorer_refine_diagnostics",
    "_record_explorer_launch_diagnostics",
    "_lane_subsample_xy",
    "_subsample_lane_xy_parts",
    "_record_lane_subsample_diagnostics",
    "_typed_explorer_caps_from_hp",
    "_typed_explorer_caps_for_order",
    "_run_one_typed_explorer_launch",
    "_finalize_typed_explorer_launch_in_parent",
    "_build_typed_explorer_launch_work_plan",
    "_run_typed_explorer_work_plan_serial",
    "_run_typed_explorer_work_plan_process",
    "_build_explorer_lane_candidates",
    "_carrier_role",
    "_should_run_two_block_shared_coord",
    "_family_first_gate_decision",
    "_collapse_scheduler_confidence_rank",
    "_schedule_explorer_coord_asts_by_collapse",
    "_select_typed_lane_candidates",
    "_build_typed_lane_candidates_with_gate",
)

__factorized_de_constants__ = (

)

__factorized_de_late_bindings__ = (

)
