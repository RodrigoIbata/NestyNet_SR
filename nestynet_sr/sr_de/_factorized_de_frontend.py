# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

"""DE-facing types, diagnostics, validation, and feature-group preparation."""

from typing import TYPE_CHECKING
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import numpy as np
import torch
import nestynet_sr.sr_search.factorized_search.oracle_lab_de as oracle_de
from nestynet_sr.sr_core.bridges import AbsNode, AcosNode, Add, AddNode, AsinNode, AtanNode, ConstNode, CosNode, D2U, DU, ExpNode, LogNode, Mul, MulNode, PowNode, SinNode, U, Var
from nestynet_sr.sr_search.factorized_search.bridge import factorized_search_to_nestynet, embed_mapping_in_ast
from nestynet_sr.sr_search.factorized_search.config import FactorizedSearchConfig
from nestynet_sr.sr_search.factorized_search.domain_projection import domain_projection_is_acceptable, eval_node_with_domain_projection
from nestynet_sr.sr_search.factorized_search.expr_ast import dim_round, node_dims


if TYPE_CHECKING:
    from ._factorized_de_operator import (
        _compiled_de_ast_payload,
        _compiled_de_row_payload,
    )
    from ._factorized_de_search import (
        factorized_search_report_shortlist,
        factorized_search_report_to_rhs_callable,
        normalized_rmse,
    )

_DOMAIN_PROJECTION_DEFAULT_ABS_TOL = 1.0e-8


_DOMAIN_PROJECTION_DEFAULT_REL_TOL = 1.0e-2


_DOMAIN_PROJECTION_DEFAULT_MAX_FRAC = 1.0


_DOMAIN_PROJECTION_DEFAULT_POSITIVE_FLOOR = 1.0e-12


@dataclass(frozen=True)
class DEFeatureGroup:
    """One DE feature-table split associated with a trajectory or dataset id."""

    id: str
    features: oracle_de.DEFeatureTensors
    use_for_fit: bool = True
    use_for_probe: bool = True
    surrogate_val_loss: float | None = None


@dataclass
class FactorizedSearchDERescueConfig:
    """Configuration surface for DE-facing factorized symbolic search rescue.

    The trigger fields are consumed by later ``run_de.py`` integration, while
    the public wrappers in this module currently use the embedded factorized symbolic search
    hyperparameter payload and integration-validation setting.
    """

    mode: str = "auto"  # never | auto | always
    trigger_val_rms: float = 1.0e-3
    trigger_rel_rms: float = 1.0e-3
    trigger_cond: float = 1.0e8
    replace_rel_factor: float = 0.98
    strict_shared_rhs: bool = True
    validate_integrate_topk: int = 8
    budget_scope: str = "per_group"  # per_group | global
    max_attempts: int | None = None
    coefficient_dim_mode: str = "strict_expression"  # strict_expression | inferred_outer
    direct_generator_witness_topk: int = 1
    generator_witness_tau_rel: float = 5.0e-2
    generator_witness_tau_abs: float = 1.0e-8
    generator_witness_state_tau_rel: float = 2.0e-2
    generator_witness_velocity_tau_rel: float = 5.0e-2
    generator_witness_max_coeff_spread_rel: float = 5.0e-1
    generator_witness_max_intercept_z: float = 3.0
    generator_witness_max_local_rms_z: float = 2.0
    generator_witness_max_local_q95_z: float = 3.5
    generator_witness_max_rollout_u_rms_z: float = 3.5
    generator_witness_max_rollout_u_nrmse: float = 5.0e-2
    generator_witness_rollout_window_fraction: float = 0.25
    generator_witness_rollout_max_span: float = 5.0
    regularized_implicit_enable: bool = True
    regularized_implicit_max_b_terms: int = 2
    regularized_implicit_max_a_dynamic_range: float = 1.0e8
    regularized_implicit_min_nonzero_frac: float = 0.995
    regularized_implicit_ridge: float = 1.0e-10
    regularized_implicit_invariant_refit_enable: bool = True
    regularized_implicit_invariant_min_points: int = 8
    regularized_implicit_invariant_accept_score: float = 1.0e-2
    regularized_implicit_invariant_max_traj_score: float = 5.0e-2
    regularized_implicit_invariant_max_coeff_spread_rel: float = 1.0e-1
    regularized_implicit_invariant_min_abs_coeff: float = 1.0e-10
    regularized_implicit_invariant_max_abs_coeff: float = 1.0e6
    # Cleanliness certificate for multi-term regularized implicit linear fits.
    # The normalized derivative-residual probe score must beat this threshold;
    # the surrogate-derivative noise floor sits around 1e-3 to 3e-3 on the
    # Feynman-DE benchmark, so exact linear DEs typically land in that band.
    regularized_implicit_clean_score: float = 5.0e-3
    regularized_implicit_clean_max_overfit_ratio: float = 5.0
    # Optional overrides for the challenger-lane cleanliness gate. None falls
    # back to trigger_val_rms / trigger_rel_rms.
    clean_gate_val_rms: float | None = None
    clean_gate_rel_rms: float | None = None
    hp: FactorizedSearchConfig | None = None


@dataclass
class FactorizedSearchDEResult:
    """Normalized DE-facing factorized symbolic search result object."""

    order: int
    x_axis: int
    rhs_ast: Any
    residual_ast: Any
    canonical_equation: str
    probe_mse: float
    probe_rms: float
    expr_ast: Any
    mapping: dict[str, Any]
    mapping_kind: str
    feature_names: list[str]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    engine: str = "factorized_search"
    rhs_ast_raw: Any | None = None
    residual_ast_raw: Any | None = None
    rhs_ast_simplified: Any | None = None
    residual_ast_simplified: Any | None = None
    canonical_equation_raw: str | None = None
    canonical_equation_simplified: str | None = None

    def format_equation(self) -> str:
        return str(self.canonical_equation)


def _diag_inc(diag: dict[str, Any] | None, key: str, amount: int | float = 1) -> None:
    if diag is None:
        return
    prev = diag.get(key, 0)
    try:
        diag[key] = prev + amount
    except Exception:
        diag[key] = amount


_DIAGNOSTIC_MIN_KEYS = {
    "typed_best_probe_mse_so_far",
    "typed_best_probe_rms_so_far",
}


_DIAGNOSTIC_MAX_KEYS = {
    "typed_tasks_inflight_peak",
}


def _diag_set_min(diag: dict[str, Any] | None, key: str, value: Any) -> None:
    if diag is None:
        return
    try:
        f = float(value)
    except Exception:
        return
    if not math.isfinite(f):
        return
    prev = diag.get(key, None)
    try:
        p = float(prev)
    except Exception:
        diag[key] = f
        return
    if not math.isfinite(p) or f < p:
        diag[key] = f


def _diag_set_max(diag: dict[str, Any] | None, key: str, value: Any) -> None:
    if diag is None:
        return
    try:
        f = float(value)
    except Exception:
        return
    if not math.isfinite(f):
        return
    prev = diag.get(key, None)
    try:
        p = float(prev)
    except Exception:
        diag[key] = f
        return
    if not math.isfinite(p) or f > p:
        diag[key] = f


def _merge_diagnostics(dst: dict[str, Any], src: Mapping[str, Any] | None) -> None:
    if not isinstance(src, Mapping):
        return
    for key, value in src.items():
        key_s = str(key)
        if isinstance(value, list):
            dst.setdefault(key_s, [])
            if isinstance(dst[key_s], list):
                dst[key_s].extend(value)
            else:
                dst[key_s] = list(value)
            continue
        if key_s in _DIAGNOSTIC_MIN_KEYS:
            _diag_set_min(dst, key_s, value)
            continue
        if key_s in _DIAGNOSTIC_MAX_KEYS:
            _diag_set_max(dst, key_s, value)
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            prev = dst.get(key_s, 0)
            if isinstance(prev, (int, float)) and not isinstance(prev, bool):
                dst[key_s] = prev + value
            elif key_s not in dst:
                dst[key_s] = value
            continue
        if key_s not in dst:
            dst[key_s] = value


def _resource_maxrss_mb() -> float | None:
    try:
        import resource

        raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if not math.isfinite(raw) or raw < 0.0:
        return None
    # macOS reports bytes; Linux reports KiB.
    if sys.platform == "darwin":
        return raw / (1024.0 * 1024.0)
    return raw / 1024.0


def _current_process_rss_mb() -> float | None:
    # Linux fast path.
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as fh:
            parts = fh.read().strip().split()
        if len(parts) >= 2:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            rss_pages = int(parts[1])
            return float(rss_pages * page_size) / (1024.0 * 1024.0)
    except Exception:
        pass

    # macOS and other Unix fallback. This is sampled only around explorer
    # launches, so the subprocess overhead is acceptable.
    try:
        proc = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
            capture_output=True,
            timeout=1.0,
        )
        if int(proc.returncode) != 0:
            return None
        raw = str(proc.stdout or "").strip().split()
        if not raw:
            return None
        return float(int(raw[0])) / 1024.0
    except Exception:
        return None


def _process_memory_report(label: str) -> dict[str, Any]:
    report: dict[str, Any] = {"label": str(label), "pid": int(os.getpid())}
    rss_mb = _current_process_rss_mb()
    maxrss_mb = _resource_maxrss_mb()
    if rss_mb is not None and math.isfinite(float(rss_mb)):
        report["rss_mb"] = float(rss_mb)
    if maxrss_mb is not None and math.isfinite(float(maxrss_mb)):
        report["maxrss_mb"] = float(maxrss_mb)
    return report


def _memory_value(report: Mapping[str, Any] | None, key: str) -> float | None:
    if not isinstance(report, Mapping) or report.get(key, None) is None:
        return None
    try:
        out = float(report[key])
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    name = str(dtype_name).replace("torch.", "").strip().lower()
    if name in {"float64", "double"}:
        return torch.float64
    if name in {"float32", "float"}:
        return torch.float32
    if name in {"float16", "half"}:
        return torch.float16
    if name in {"bfloat16"}:
        return torch.bfloat16
    return torch.float64


def _typed_explorer_task_identity(
    *,
    lane: str,
    base_mode: str,
    order: int,
    carrier_ast,
    coord_ast,
    seed: int,
    sample_seed: int,
) -> tuple[str, str]:
    key = json.dumps(
        {
            "lane": str(lane),
            "base_mode": str(base_mode),
            "order": int(order),
            "carrier": repr(carrier_ast),
            "coord": repr(coord_ast),
            "seed": int(seed),
            "sample_seed": int(sample_seed),
        },
        sort_keys=True,
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16], key


def _first_explorer_diagnostics(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    explorer_diag = rows[0].get("explorer_diagnostics", None) if rows else None
    if not isinstance(explorer_diag, Mapping):
        explorer_diag = {}
    phase_diag = explorer_diag.get("phase_diagnostics", None)
    if not isinstance(phase_diag, Mapping):
        phase_diag = {}
    return explorer_diag, phase_diag


def _diag_number_from_reports(
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
    key: str,
    default: float | int | None = None,
) -> float | int | None:
    value = primary.get(key, secondary.get(key, default))
    try:
        f = float(value)
    except Exception:
        return default
    if not math.isfinite(f):
        return default
    if isinstance(default, int):
        return int(f)
    return f


def _best_numeric_row_value(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> float | None:
    best = float("inf")
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        for key in keys:
            try:
                value = float(row.get(key, float("nan")))
            except Exception:
                continue
            if math.isfinite(value) and value < best:
                best = value
    return best if math.isfinite(best) else None


def _short_diag_float(value: Any) -> str:
    try:
        f = float(value)
    except Exception:
        return "NA"
    if not math.isfinite(f):
        return "NA"
    if f == 0.0:
        return "0"
    if abs(f) < 1.0e-3 or abs(f) >= 1.0e4:
        return f"{f:.3e}"
    return f"{f:.4g}"


def _log_typed_explorer_task_event(row: Mapping[str, Any]) -> None:
    event = str(row.get("event", ""))
    if event not in {"started", "finished", "failed"}:
        return
    task_id = str(row.get("task_id", "?"))
    lane = str(row.get("lane", "?"))
    base_mode = str(row.get("base_mode", "?"))
    order = str(row.get("order", "?"))
    if event == "started":
        msg = (
            f"[typed-task] started id={task_id} order={order} base={base_mode} lane={lane} "
            f"fit={row.get('fit_rows_search', '?')}/{row.get('fit_rows_full', '?')} "
            f"probe={row.get('probe_rows_search', '?')}/{row.get('probe_rows_full', '?')} "
            f"n_iter={row.get('n_iter', '?')} pid={row.get('pid', '?')} "
            f"rss_mb={_short_diag_float(row.get('rss_mb', None))}"
        )
    elif event == "finished":
        msg = (
            f"[typed-task] finished id={task_id} order={order} base={base_mode} lane={lane} "
            f"wall_s={_short_diag_float(row.get('wall_seconds', None))} rows={row.get('rows', '?')} "
            f"best_mse={_short_diag_float(row.get('best_mse', None))} "
            f"score_calls={row.get('score_calls', '?')}"
        )
    else:
        msg = (
            f"[typed-task] failed id={task_id} order={order} base={base_mode} lane={lane} "
            f"wall_s={_short_diag_float(row.get('wall_seconds', None))} error={row.get('error', '')}"
        )
    print(msg, flush=True)


def _record_typed_explorer_task_event(
    diagnostics: dict[str, Any] | None,
    event_row: Mapping[str, Any],
) -> None:
    if diagnostics is None:
        return
    row = dict(event_row)
    event = str(row.get("event", ""))
    row["event"] = event
    if "pid" not in row:
        row["pid"] = int(os.getpid())
    diagnostics.setdefault("typed_explorer_task_events", []).append(oracle_de._to_jsonable(row))

    try:
        n_iter = int(row.get("n_iter", 0) or 0)
    except Exception:
        n_iter = 0

    if event == "planned":
        _diag_inc(diagnostics, "typed_tasks_planned", 1)
        _diag_inc(diagnostics, "typed_tasks_submitted", 1)
        _diag_inc(diagnostics, "typed_eval_budget_total", n_iter)
    elif event == "started":
        _diag_inc(diagnostics, "typed_tasks_started", 1)
        inflight = max(0, int(diagnostics.get("typed_tasks_inflight", 0) or 0) + 1)
        diagnostics["typed_tasks_inflight"] = int(inflight)
        diagnostics["typed_tasks_inflight_peak"] = max(
            int(diagnostics.get("typed_tasks_inflight_peak", 0) or 0),
            int(inflight),
        )
    elif event == "finished":
        _diag_inc(diagnostics, "typed_tasks_finished", 1)
        _diag_inc(diagnostics, "typed_eval_budget_finished", n_iter)
        _diag_inc(diagnostics, "typed_task_wall_seconds_finished", float(row.get("wall_seconds", 0.0) or 0.0))
        diagnostics["typed_tasks_inflight"] = max(0, int(diagnostics.get("typed_tasks_inflight", 0) or 0) - 1)
        best_mse = row.get("best_mse", None)
        _diag_set_min(diagnostics, "typed_best_probe_mse_so_far", best_mse)
        try:
            best_mse_f = float(best_mse)
        except Exception:
            best_mse_f = float("nan")
        if math.isfinite(best_mse_f) and best_mse_f >= 0.0:
            _diag_set_min(diagnostics, "typed_best_probe_rms_so_far", math.sqrt(best_mse_f))
    elif event == "failed":
        _diag_inc(diagnostics, "typed_tasks_failed", 1)
        diagnostics["typed_tasks_inflight"] = max(0, int(diagnostics.get("typed_tasks_inflight", 0) or 0) - 1)

    _log_typed_explorer_task_event(row)


def _feature_group_row_summary(groups: Sequence[DEFeatureGroup]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group in groups:
        features = group.features
        out.append(
            {
                "id": str(group.id),
                "use_for_fit": bool(group.use_for_fit),
                "use_for_probe": bool(group.use_for_probe),
                "x_fit_rows": int(features.x_fit.shape[0]),
                "x_probe_rows": int(features.x_probe.shape[0]),
                "u_fit_rows": int(features.u_fit.reshape(-1).shape[0]),
                "u_probe_rows": int(features.u_probe.reshape(-1).shape[0]),
            }
        )
    return out


def _global_group_budgets(total_budget: int, n_groups: int) -> list[int]:
    n = max(1, int(n_groups))
    total = max(1, int(total_budget))
    if total < n:
        return [1 for _ in range(n)]
    base = total // n
    rem = total % n
    return [int(base + (1 if i < rem else 0)) for i in range(n)]


def _jsonable_ast_to_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_jsonable_ast_to_tuple(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_jsonable_ast_to_tuple(v) for v in value)
    return value


def _factorized_candidate_key_payload(row: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "order": int(row.get("order", -1)),
            "expr_ast": oracle_de._to_jsonable(row.get("expr_ast", None)),
            "mapping": oracle_de._to_jsonable(row.get("mapping", None)),
        },
        sort_keys=True,
        default=str,
    )


def _factorized_candidate_id(row: Mapping[str, Any]) -> str:
    digest = hashlib.sha1(_factorized_candidate_key_payload(row).encode("utf-8")).hexdigest()[:16]
    return f"fss:{digest}"


def _ordered_constants_from_report(report: dict[str, Any]) -> list[dict[str, float]]:
    raw_ordered = report.get("constants_ordered", None)
    if isinstance(raw_ordered, (list, tuple)):
        out: list[dict[str, float]] = []
        for item in raw_ordered:
            if isinstance(item, dict) and ("value" in item):
                out.append({"name": str(item.get("name", "")), "value": float(item["value"])})
        if out:
            return out

    raw = report.get("constants", None)
    if isinstance(raw, dict):
        return [{"name": str(k), "value": float(v)} for k, v in raw.items()]
    if isinstance(raw, (list, tuple)):
        out = []
        for i, item in enumerate(raw):
            if isinstance(item, dict) and ("value" in item):
                out.append({"name": str(item.get("name", f"c{i}")), "value": float(item["value"])})
            else:
                out.append({"name": f"c{i}", "value": float(item)})
        return out
    return []


def _expr_and_mapping_from_candidate(candidate: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    if not isinstance(candidate, dict):
        raise TypeError("candidate must be a dict")
    expr_raw = candidate.get("expr_ast", None)
    if expr_raw is None:
        expr_raw = candidate.get("expr", None)
    expr_ast = _jsonable_ast_to_tuple(expr_raw)
    mapping = candidate.get("mapping", None)
    if expr_ast is None or mapping is None:
        raise ValueError("factorized symbolic search candidate is missing expr/expr_ast or mapping")
    if isinstance(expr_ast, str):
        raise ValueError("factorized symbolic search candidate expr_ast must be structured AST data, not a display string")
    if not isinstance(mapping, dict):
        raise ValueError("factorized symbolic search candidate mapping must be a dict")
    return expr_ast, mapping


def _float_from_mapping(mapping: Mapping[str, Any], keys: Sequence[str], default: float) -> float:
    for key in keys:
        if key in mapping:
            try:
                out = float(mapping[key])
            except Exception:
                continue
            if math.isfinite(out):
                return float(out)
    return float(default)


def _domain_projection_cfg_from_diag(diag: Any) -> dict[str, Any] | None:
    if not isinstance(diag, Mapping):
        return None

    status = str(diag.get("status", "") or "").strip().lower()
    enabled = bool(diag.get("score_domain_projection_enable", False)) or bool(
        diag.get("enabled", False)
    )
    if not enabled and status in {"projected_within_tube", "rejected_outside_tube"}:
        enabled = True
    if not enabled:
        return None

    return {
        "score_domain_projection_enable": True,
        "score_domain_projection_abs_tol": _float_from_mapping(
            diag,
            ("score_domain_projection_abs_tol", "abs_tol"),
            _DOMAIN_PROJECTION_DEFAULT_ABS_TOL,
        ),
        "score_domain_projection_rel_tol": _float_from_mapping(
            diag,
            ("score_domain_projection_rel_tol", "rel_tol"),
            _DOMAIN_PROJECTION_DEFAULT_REL_TOL,
        ),
        "score_domain_projection_max_frac": _float_from_mapping(
            diag,
            ("score_domain_projection_max_frac", "max_frac"),
            _DOMAIN_PROJECTION_DEFAULT_MAX_FRAC,
        ),
        "score_domain_projection_positive_floor": _float_from_mapping(
            diag,
            ("score_domain_projection_positive_floor", "positive_floor"),
            _DOMAIN_PROJECTION_DEFAULT_POSITIVE_FLOOR,
        ),
    }


def _domain_projection_diags_from_candidate(candidate: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []

    def _add(value: Any) -> None:
        if isinstance(value, Mapping):
            out.append(value)

    _add(candidate.get("domain_projection", None))
    _add(candidate.get("domain_projection_cfg", None))
    mapping = candidate.get("mapping", None)
    if isinstance(mapping, Mapping):
        _add(mapping.get("_domain_projection", None))

    diagnostics = candidate.get("diagnostics", None)
    if isinstance(diagnostics, Mapping):
        _add(diagnostics.get("domain_projection", None))
        report = diagnostics.get("report", None)
        if isinstance(report, Mapping):
            best = report.get("best", None)
            if isinstance(best, Mapping):
                _add(best.get("domain_projection", None))
                best_mapping = best.get("mapping", None)
                if isinstance(best_mapping, Mapping):
                    _add(best_mapping.get("_domain_projection", None))

    best = candidate.get("best", None)
    if isinstance(best, Mapping):
        _add(best.get("domain_projection", None))
        best_mapping = best.get("mapping", None)
        if isinstance(best_mapping, Mapping):
            _add(best_mapping.get("_domain_projection", None))

    return out


def _domain_projection_cfg_from_candidate(candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    for diag in _domain_projection_diags_from_candidate(candidate):
        cfg = _domain_projection_cfg_from_diag(diag)
        if cfg is not None:
            return cfg
    return None


def _input_exprs_from_report(report: dict[str, Any], *, order: int) -> list[Any]:
    x_axis = int(report.get("x_axis", 0))
    exprs: list[Any] = []
    if bool(report.get("include_x", True)):
        exprs.append(Var(x_axis))
    if bool(report.get("include_u", True)):
        exprs.append(U())
    if int(order) == 2 and bool(report.get("include_du", True)):
        exprs.append(DU(x_axis))
    for const in _ordered_constants_from_report(report):
        exprs.append(ConstNode(float(const["value"])))
    return exprs


def _no_candidate_result_from_report(
    report: dict[str, Any],
    *,
    order: int | None = None,
    feature_names: Sequence[str] | None = None,
    reason: str,
) -> FactorizedSearchDEResult:
    per_order = [row for row in list(report.get("per_order", []) or []) if isinstance(row, dict)]
    order_i = int(order) if order is not None else 1
    feature_names_out = list(feature_names or [])
    if order is None or not feature_names_out:
        for row in per_order:
            try:
                cand_order = int(row.get("order", -1))
            except Exception:
                continue
            if cand_order in (1, 2):
                if order is None:
                    order_i = int(cand_order)
                if not feature_names_out:
                    feature_names_out = list(row.get("feature_names", []) or [])
                break

    diagnostics = {
        "status": "NO_CANDIDATE",
        "failure_kind": str(reason),
        "domain_ok": False,
        "structural_ok": False,
        "structural_hard_reject": True,
        "structural_reasons": [str(reason)],
        "structural_gate_version": 1,
        "integrate_ok": False,
        "integrate_mse": float("inf"),
        "domain_fragility_penalty": float("inf"),
        "domain_failure_reason": str(reason),
        "size": 10**9,
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
        "report": report,
    }
    return FactorizedSearchDEResult(
        order=order_i,
        x_axis=int(report.get("x_axis", 0)),
        rhs_ast=None,
        residual_ast=None,
        canonical_equation="",
        probe_mse=float("inf"),
        probe_rms=float("inf"),
        expr_ast=None,
        mapping={},
        mapping_kind="",
        feature_names=feature_names_out,
        diagnostics=diagnostics,
    )


def _anchor_for_order(order: int, *, x_axis: int):
    if int(order) == 1:
        return DU(int(x_axis))
    if int(order) == 2:
        return D2U(int(x_axis), int(x_axis))
    raise ValueError(f"Unsupported DE order: {order}")


def _safe_float(value: Any, *, default: float = float("inf")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _safe_int(value: Any, *, default: int = 10**9) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _score_decade(value: Any) -> int:
    try:
        value_f = float(value)
    except Exception:
        return 10**9
    if not math.isfinite(value_f):
        return 10**9
    return int(math.floor(math.log10(max(float(value_f), 1.0e-300))))


def _row_validation_decade(row: Mapping[str, Any]) -> int:
    for key in ("mse", "probe_mse", "score_raw", "final_validated_mse", "raw_mse", "score"):
        if key not in row:
            continue
        decade = _score_decade(row.get(key))
        if decade < 10**9:
            return int(decade)
    return 10**9


def _real_scalar_or_none(value: Any) -> float | None:
    if isinstance(value, ConstNode):
        value = value.value
    if isinstance(value, np.generic):
        value = value.item()
    if hasattr(value, "detach") and hasattr(value, "numel"):
        try:
            if int(value.numel()) == 1:
                value = value.detach().cpu().item()
        except Exception:
            return None
    if isinstance(value, complex):
        if abs(float(value.imag)) > 1.0e-12:
            return None
        value = value.real
    try:
        return float(value)
    except Exception:
        return None


def _integerish(value: float, *, tol: float = 1.0e-12) -> bool:
    if not math.isfinite(float(value)):
        return False
    return abs(float(value) - round(float(value))) <= float(tol)


def _tuple_static_const_value(node: Any) -> float | None:
    node = _jsonable_ast_to_tuple(node)
    scalar = _real_scalar_or_none(node)
    if scalar is not None:
        return scalar
    if not isinstance(node, tuple) or not node:
        return None

    op = str(node[0])
    if op == "const" and len(node) >= 2:
        return _real_scalar_or_none(node[1])
    if op == "neg" and len(node) >= 2:
        child = _tuple_static_const_value(node[1])
        return None if child is None else -float(child)
    if op in {"add", "sub", "mul", "div"} and len(node) >= 3:
        lhs = _tuple_static_const_value(node[1])
        rhs = _tuple_static_const_value(node[2])
        if lhs is None or rhs is None:
            return None
        try:
            if op == "add":
                return float(lhs) + float(rhs)
            if op == "sub":
                return float(lhs) - float(rhs)
            if op == "mul":
                return float(lhs) * float(rhs)
            if abs(float(rhs)) <= 0.0:
                return None
            return float(lhs) / float(rhs)
        except Exception:
            return None
    if op == "sqr" and len(node) >= 2:
        child = _tuple_static_const_value(node[1])
        return None if child is None else float(child) * float(child)
    if op == "sqrt" and len(node) >= 2:
        child = _tuple_static_const_value(node[1])
        if child is None or float(child) < 0.0:
            return None
        return math.sqrt(float(child))
    if op == "log" and len(node) >= 2:
        child = _tuple_static_const_value(node[1])
        if child is None or float(child) <= 0.0:
            return None
        return math.log(float(child))
    if op == "exp" and len(node) >= 2:
        child = _tuple_static_const_value(node[1])
        if child is None:
            return None
        try:
            return math.exp(float(child))
        except OverflowError:
            return None
    if op == "sin" and len(node) >= 2:
        child = _tuple_static_const_value(node[1])
        return None if child is None else math.sin(float(child))
    if op == "cos" and len(node) >= 2:
        child = _tuple_static_const_value(node[1])
        return None if child is None else math.cos(float(child))
    if op == "asin" and len(node) >= 2:
        child = _tuple_static_const_value(node[1])
        if child is None or not -1.0 <= float(child) <= 1.0:
            return None
        return math.asin(float(child))
    if op == "acos" and len(node) >= 2:
        child = _tuple_static_const_value(node[1])
        if child is None or not -1.0 <= float(child) <= 1.0:
            return None
        return math.acos(float(child))
    if op == "atan" and len(node) >= 2:
        child = _tuple_static_const_value(node[1])
        return None if child is None else math.atan(float(child))
    if op == "abs" and len(node) >= 2:
        child = _tuple_static_const_value(node[1])
        return None if child is None else abs(float(child))
    return None


def _nestynet_static_const_value(node: Any) -> float | None:
    scalar = _real_scalar_or_none(node)
    if scalar is not None:
        return scalar
    if isinstance(node, AddNode):
        lhs = _nestynet_static_const_value(node.left)
        rhs = _nestynet_static_const_value(node.right)
        return None if lhs is None or rhs is None else float(lhs) + float(rhs)
    if isinstance(node, MulNode):
        lhs = _nestynet_static_const_value(node.left)
        rhs = _nestynet_static_const_value(node.right)
        return None if lhs is None or rhs is None else float(lhs) * float(rhs)
    if isinstance(node, PowNode):
        base = _nestynet_static_const_value(node.base)
        exponent = _nestynet_static_const_value(node.exponent)
        if base is None or exponent is None:
            return None
        if float(base) < 0.0 and not _integerish(float(exponent)):
            return None
        if abs(float(base)) <= 0.0 and float(exponent) < 0.0:
            return None
        try:
            return float(base) ** float(exponent)
        except Exception:
            return None
    if isinstance(node, LogNode):
        arg = _nestynet_static_const_value(node.arg)
        if arg is None or float(arg) <= 0.0:
            return None
        return math.log(float(arg))
    if isinstance(node, ExpNode):
        arg = _nestynet_static_const_value(node.arg)
        if arg is None:
            return None
        try:
            return math.exp(float(arg))
        except OverflowError:
            return None
    if isinstance(node, SinNode):
        arg = _nestynet_static_const_value(node.arg)
        return None if arg is None else math.sin(float(arg))
    if isinstance(node, CosNode):
        arg = _nestynet_static_const_value(node.arg)
        return None if arg is None else math.cos(float(arg))
    if isinstance(node, AsinNode):
        arg = _nestynet_static_const_value(node.arg)
        if arg is None or not -1.0 <= float(arg) <= 1.0:
            return None
        return math.asin(float(arg))
    if isinstance(node, AcosNode):
        arg = _nestynet_static_const_value(node.arg)
        if arg is None or not -1.0 <= float(arg) <= 1.0:
            return None
        return math.acos(float(arg))
    if isinstance(node, AtanNode):
        arg = _nestynet_static_const_value(node.arg)
        return None if arg is None else math.atan(float(arg))
    if isinstance(node, AbsNode):
        arg = _nestynet_static_const_value(node.arg)
        return None if arg is None else abs(float(arg))
    return None


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _tuple_structural_reasons(node: Any) -> list[str]:
    reasons: list[str] = []

    def _walk(cur: Any) -> None:
        cur = _jsonable_ast_to_tuple(cur)
        if not isinstance(cur, tuple) or not cur:
            scalar = _real_scalar_or_none(cur)
            if scalar is not None and not math.isfinite(float(scalar)):
                _append_reason(reasons, "nonfinite_constant")
            return

        op = str(cur[0])
        if op == "const" and len(cur) >= 2:
            val = _real_scalar_or_none(cur[1])
            if val is not None and not math.isfinite(float(val)):
                _append_reason(reasons, "nonfinite_constant")
        elif op == "log" and len(cur) >= 2:
            arg = _tuple_static_const_value(cur[1])
            if arg is not None and float(arg) <= 0.0:
                _append_reason(reasons, "log_nonpositive_constant")
        elif op == "sqrt" and len(cur) >= 2:
            arg = _tuple_static_const_value(cur[1])
            if arg is not None and float(arg) < 0.0:
                _append_reason(reasons, "sqrt_negative_constant")
        elif op in {"asin", "acos"} and len(cur) >= 2:
            arg = _tuple_static_const_value(cur[1])
            if arg is not None and not -1.0 <= float(arg) <= 1.0:
                _append_reason(reasons, f"{op}_constant_out_of_domain")
        elif op in {"pow", "power"} and len(cur) >= 3:
            base = _tuple_static_const_value(cur[1])
            exponent = _tuple_static_const_value(cur[2])
            if base is not None and exponent is not None:
                if float(base) < 0.0 and not _integerish(float(exponent)):
                    _append_reason(reasons, "noninteger_power_negative_constant")
                if abs(float(base)) <= 0.0 and float(exponent) < 0.0:
                    _append_reason(reasons, "negative_power_zero_constant")
        elif op == "div" and len(cur) >= 3:
            denom = _tuple_static_const_value(cur[2])
            if denom is not None and abs(float(denom)) <= 0.0:
                _append_reason(reasons, "division_by_zero_constant")

        for child in cur[1:]:
            _walk(child)

    _walk(node)
    return reasons


def _nestynet_structural_reasons(node: Any) -> list[str]:
    reasons: list[str] = []

    def _walk(cur: Any) -> None:
        if isinstance(cur, ConstNode):
            val = _real_scalar_or_none(cur)
            if val is not None and not math.isfinite(float(val)):
                _append_reason(reasons, "nonfinite_constant")
            return
        if isinstance(cur, LogNode):
            arg = _nestynet_static_const_value(cur.arg)
            if arg is not None and float(arg) <= 0.0:
                _append_reason(reasons, "log_nonpositive_constant")
            _walk(cur.arg)
            return
        if isinstance(cur, (AsinNode, AcosNode)):
            arg = _nestynet_static_const_value(cur.arg)
            if arg is not None and not -1.0 <= float(arg) <= 1.0:
                name = "asin" if isinstance(cur, AsinNode) else "acos"
                _append_reason(reasons, f"{name}_constant_out_of_domain")
            _walk(cur.arg)
            return
        if isinstance(cur, PowNode):
            base = _nestynet_static_const_value(cur.base)
            exponent = _nestynet_static_const_value(cur.exponent)
            if base is not None and exponent is not None:
                if float(base) < 0.0 and not _integerish(float(exponent)):
                    if abs(float(exponent) - 0.5) <= 1.0e-12:
                        _append_reason(reasons, "sqrt_negative_constant")
                    else:
                        _append_reason(reasons, "noninteger_power_negative_constant")
                if abs(float(base)) <= 0.0 and float(exponent) < 0.0:
                    _append_reason(reasons, "negative_power_zero_constant")
            _walk(cur.base)
            if hasattr(cur.exponent, "__dict__") or isinstance(cur.exponent, ConstNode):
                _walk(cur.exponent)
            return
        if isinstance(cur, (SinNode, CosNode, ExpNode, AtanNode, AbsNode)):
            _walk(cur.arg)
            return
        if isinstance(cur, AddNode):
            _walk(cur.left)
            _walk(cur.right)
            return
        if isinstance(cur, MulNode):
            _walk(cur.left)
            _walk(cur.right)
            return

    _walk(node)
    return reasons


def _domain_projection_value_ok(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return True
    try:
        return bool(domain_projection_is_acceptable(value))
    except Exception:
        return value.get("ok", None) is not False


def _row_projection_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in ("domain_projection", "domain_projection_eval"):
        diag = row.get(key, None)
        if isinstance(diag, Mapping) and not _domain_projection_value_ok(diag):
            _append_reason(reasons, f"{key}_rejected")
    mapping = row.get("mapping", None)
    if isinstance(mapping, Mapping):
        diag = mapping.get("_domain_projection", None)
        if isinstance(diag, Mapping) and not _domain_projection_value_ok(diag):
            _append_reason(reasons, "mapping_domain_projection_rejected")
    for traj in row.get("mse_traj", []) or []:
        if not isinstance(traj, Mapping):
            continue
        diag = traj.get("domain_projection", None)
        if isinstance(diag, Mapping) and not _domain_projection_value_ok(diag):
            _append_reason(reasons, "trajectory_domain_projection_rejected")
    return reasons


def _broad_row_structural_safety(
    row: Mapping[str, Any],
    *,
    expr_ast: Any | None = None,
    mapping: Mapping[str, Any] | None = None,
    input_exprs: Sequence[Any] | None = None,
    order: int | None = None,
    x_axis: int = 0,
) -> dict[str, Any]:
    """Return hard structural eligibility diagnostics for a broad whole-RHS row."""

    reasons: list[str] = []
    if not isinstance(row, Mapping):
        return {
            "structural_ok": False,
            "structural_hard_reject": True,
            "structural_reasons": ["not_a_mapping_row"],
            "structural_gate_version": 1,
        }

    if row.get("domain_ok", None) is False:
        _append_reason(reasons, "domain_ok_false")
    if bool(row.get("hidden_score_head", False)):
        _append_reason(reasons, "hidden_score_head")
    finite_mask = row.get("finite_mask", None)
    if isinstance(finite_mask, Mapping) and finite_mask.get("ok", None) is False:
        _append_reason(reasons, "finite_mask_rejected")
    for numeric_key in ("mse", "score", "score_raw", "probe_rms"):
        if numeric_key not in row:
            continue
        val = _safe_float(row.get(numeric_key, None), default=float("inf"))
        if not math.isfinite(val):
            _append_reason(reasons, f"nonfinite_{numeric_key}")

    for reason in _row_projection_reasons(row):
        _append_reason(reasons, reason)

    expr_obj = expr_ast if expr_ast is not None else row.get("expr_ast", None)
    for reason in _tuple_structural_reasons(expr_obj):
        _append_reason(reasons, reason)

    payload: dict[str, Any] = {}
    rhs_ast = None
    residual_ast = None
    if expr_ast is not None and mapping is not None and input_exprs is not None and order in (1, 2):
        try:
            inner_nn = factorized_search_to_nestynet(expr_ast)
            rhs_ast = embed_mapping_in_ast(inner_nn, dict(mapping), list(input_exprs), units_mode="raw")
            if rhs_ast is not None:
                residual_ast = Add(_anchor_for_order(int(order), x_axis=int(x_axis)), Mul(ConstNode(-1.0), rhs_ast))
                payload.update(_compiled_de_row_payload(rhs_ast=rhs_ast, residual_ast=residual_ast))
        except Exception:
            rhs_ast = None
            residual_ast = None
            if bool(row.get("hidden_score_head", False)):
                _append_reason(reasons, "hidden_score_head_uncompiled")

    for node in (rhs_ast, residual_ast):
        for reason in _nestynet_structural_reasons(node):
            _append_reason(reasons, reason)

    structural_ok = len(reasons) == 0
    payload.update(
        {
            "structural_ok": bool(structural_ok),
            "structural_hard_reject": bool(not structural_ok),
            "structural_reasons": list(reasons),
            "structural_gate_version": 1,
        }
    )
    return payload


def _row_structurally_eligible(row: Mapping[str, Any] | None) -> bool:
    if not isinstance(row, Mapping):
        return False
    if row.get("structural_ok", None) is False:
        return False
    if row.get("structural_hard_reject", None) is True:
        return False
    if row.get("domain_ok", None) is False:
        return False
    if bool(row.get("hidden_score_head", False)):
        return False
    if _row_projection_reasons(row):
        return False
    if _tuple_structural_reasons(row.get("expr_ast", None)):
        return False
    return True


def _coefficient_dim_mode(value: Any) -> str:
    mode = str(value or "strict_expression").strip().lower()
    if mode in {"strict", "strict_expression", "target", "target_expression"}:
        return "strict_expression"
    if mode in {"inferred", "inferred_outer", "outer", "outer_coefficient"}:
        return "inferred_outer"
    raise ValueError(f"unknown coefficient_dim_mode: {value!r}")


def _dim_sub_tuple(lhs: Sequence[float], rhs: Sequence[float]) -> tuple[float, ...]:
    if len(lhs) != len(rhs):
        raise ValueError("dimension vector lengths do not match")
    return dim_round(tuple(float(a) - float(b) for a, b in zip(lhs, rhs)))


def _node_dim_jsonable(node: Any, var_dims: Sequence[Sequence[float]] | None) -> list[float] | None:
    if var_dims is None or node is None:
        return None
    try:
        dim = node_dims(node, var_dims)
    except Exception:
        dim = None
    if dim is None:
        return None
    return [float(v) for v in dim]


def _coefficient_dim_jsonable(
    *,
    expr_dim: Sequence[float] | None,
    target_dim: Sequence[float] | None,
    coefficient_dim_mode: str,
) -> list[float] | None:
    if str(coefficient_dim_mode) != "inferred_outer":
        return None
    if expr_dim is None or target_dim is None:
        return None
    try:
        return [float(v) for v in _dim_sub_tuple(tuple(target_dim), tuple(expr_dim))]
    except Exception:
        return None


def _uniform_subsample_rows(x: torch.Tensor, *, max_rows: int) -> torch.Tensor:
    if int(x.ndim) != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {tuple(x.shape)}")
    n = int(x.shape[0])
    if n <= int(max_rows):
        return x
    idx = torch.div(
        torch.arange(int(max_rows), device=x.device) * n,
        int(max_rows),
        rounding_mode="floor",
    )
    return x.index_select(0, idx.to(dtype=torch.long))


def _candidate_eval_array(
    expr_ast: Any,
    mapping: dict[str, Any],
    features: Any,
    *,
    dtype: torch.dtype,
    domain_projection_cfg: dict[str, Any] | None = None,
) -> np.ndarray:
    from nestynet_sr.sr_search.factorized_search.explorer import eval_mapping, eval_node

    x = torch.as_tensor(features, dtype=dtype)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError(f"Expected features to be 1D or 2D, got shape {tuple(x.shape)}")

    with torch.no_grad():
        if isinstance(domain_projection_cfg, dict) and bool(
            domain_projection_cfg.get("score_domain_projection_enable", False)
        ):
            pred, domain_diag = eval_node_with_domain_projection(expr_ast, x, domain_projection_cfg)
            if not domain_projection_is_acceptable(domain_diag):
                raise FloatingPointError("domain_projection_rejected")
        else:
            pred = eval_node(expr_ast, x)
        yhat = eval_mapping(pred, mapping).reshape(-1)
    return np.asarray(yhat.detach().cpu().numpy(), dtype=np.float64)


def _observed_derivative_scale(
    y_fit_parts: Sequence[torch.Tensor],
    y_probe_parts: Sequence[torch.Tensor],
) -> float:
    y_all = torch.cat([*y_fit_parts, *y_probe_parts], dim=0).reshape(-1)
    if int(y_all.numel()) == 0:
        return 1.0
    abs_y = torch.abs(y_all)
    mean_abs = float(torch.mean(abs_y).detach().cpu().item())
    max_abs = float(torch.max(abs_y).detach().cpu().item())
    return max(1.0e-8, mean_abs, 0.1 * max_abs)


def _safe_probe_rel_rms(probe_rms: float, target_scale: float) -> float:
    if not math.isfinite(float(probe_rms)):
        return float("inf")
    scale = max(1.0e-8, float(target_scale))
    return float(probe_rms) / scale


def _de_effective_early_stop_mse(
    hp: FactorizedSearchConfig,
    *,
    observed_scale: float,
) -> float:
    base = _safe_float(getattr(hp, "early_stop_mse", 0.0), default=0.0)
    if not math.isfinite(base) or base < 0.0:
        base = 0.0
    val_rms = _safe_float(getattr(hp, "_de_early_stop_val_rms", None), default=float("nan"))
    rel_rms = _safe_float(getattr(hp, "_de_early_stop_rel_rms", None), default=float("nan"))
    multiplier = _safe_float(getattr(hp, "_de_early_stop_rms_multiplier", 1.0), default=1.0)
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        multiplier = 1.0

    stop_rms = 0.0
    if math.isfinite(val_rms) and val_rms > 0.0:
        stop_rms = max(stop_rms, float(val_rms))
    if math.isfinite(rel_rms) and rel_rms > 0.0:
        stop_rms = max(stop_rms, float(rel_rms) * max(1.0e-8, float(observed_scale)))
    if stop_rms <= 0.0:
        return float(base)
    return float(max(base, (float(multiplier) * float(stop_rms)) ** 2))


def _np1(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _finite_np1(value: Any) -> np.ndarray:
    arr = _np1(value)
    return arr[np.isfinite(arr)]


def _robust_rms(value: Any) -> float:
    arr = _finite_np1(value)
    if int(arr.size) == 0:
        return float("inf")
    return float(np.sqrt(np.mean(np.square(arr))))


def _robust_scale(value: Any) -> float:
    arr = _finite_np1(value)
    if int(arr.size) == 0:
        return 1.0
    rms = float(np.sqrt(np.mean(np.square(arr))))
    max_abs = float(np.max(np.abs(arr))) if int(arr.size) else 0.0
    med = float(np.median(np.abs(arr))) if int(arr.size) else 0.0
    return max(1.0e-12, rms, 0.1 * max_abs, med)


def _q_abs(value: Any, q: float) -> float:
    arr = np.abs(_finite_np1(value))
    if int(arr.size) == 0:
        return float("inf")
    return float(np.quantile(arr, float(q)))


def _fit_affine_1d(phi: Any, y: Any) -> dict[str, Any]:
    phi_arr = _np1(phi)
    y_arr = _np1(y)
    m = np.isfinite(phi_arr) & np.isfinite(y_arr)
    phi_f = phi_arr[m]
    y_f = y_arr[m]
    if int(phi_f.size) < 3:
        raise ValueError(f"too few finite rows for affine fit ({int(phi_f.size)})")
    A = np.column_stack([phi_f, np.ones_like(phi_f)])
    coeff, *_ = np.linalg.lstsq(A, y_f, rcond=None)
    a = float(coeff[0])
    b = float(coeff[1])
    pred = a * phi_f + b
    resid = y_f - pred
    try:
        cond = float(np.linalg.cond(A))
    except Exception:
        cond = float("inf")
    return {
        "a": a,
        "b": b,
        "n": int(phi_f.size),
        "pred": pred,
        "resid": resid,
        "phi": phi_f,
        "y": y_f,
        "cond": cond,
    }


def _carrier_eval_array(
    expr_ast: Any,
    features: Any,
    *,
    dtype: torch.dtype,
    domain_projection_cfg: dict[str, Any] | None,
) -> np.ndarray:
    from nestynet_sr.sr_search.factorized_search.explorer import eval_node

    x = torch.as_tensor(features, dtype=dtype)
    if int(x.ndim) == 1:
        x = x.reshape(1, -1)
    if int(x.ndim) != 2:
        raise ValueError(f"Expected carrier feature table to be rank 2, got {tuple(x.shape)}")

    with torch.no_grad():
        if isinstance(domain_projection_cfg, dict) and bool(
            domain_projection_cfg.get("score_domain_projection_enable", False)
        ):
            pred, domain_diag = eval_node_with_domain_projection(expr_ast, x, domain_projection_cfg)
            if not domain_projection_is_acceptable(domain_diag):
                raise FloatingPointError("domain_projection_rejected")
        else:
            pred = eval_node(expr_ast, x)
    return np.asarray(pred.reshape(-1).detach().cpu().numpy(), dtype=np.float64)


def _order2_group_witness_table(
    candidate: Mapping[str, Any],
    group: DEFeatureGroup,
    *,
    spec: oracle_de.DELabSpec,
    dtype: torch.dtype,
) -> dict[str, Any]:
    expr_ast, _ = _expr_and_mapping_from_candidate(dict(candidate))
    domain_cfg = _domain_projection_cfg_from_candidate(candidate)
    z_parts: list[torch.Tensor] = []
    y_parts: list[torch.Tensor] = []
    for split in ("fit", "probe"):
        z, y, _ = oracle_de._build_table_from_features(spec, group.features, order=2, split=split)
        z_parts.append(z)
        y_parts.append(y)
    z_all = torch.cat(z_parts, dim=0)
    y_all = torch.cat(y_parts, dim=0).reshape(-1)
    phi = _carrier_eval_array(expr_ast, z_all, dtype=dtype, domain_projection_cfg=domain_cfg)
    y_np = np.asarray(y_all.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    finite = np.isfinite(phi) & np.isfinite(y_np)
    return {
        "group_id": str(group.id),
        "phi": phi[finite],
        "y": y_np[finite],
        "n": int(np.sum(finite)),
    }


def _d2u_fd_disagreement_scale(
    groups: Sequence[DEFeatureGroup],
    *,
    spec: oracle_de.DELabSpec,
) -> float:
    diffs: list[np.ndarray] = []
    for group in groups:
        try:
            traj = _group_to_validation_trajectory(group, spec=spec)
        except Exception:
            continue
        x = _np1(traj.x)
        du = _np1(traj.du)
        d2u = _np1(traj.d2u)
        m = np.isfinite(x) & np.isfinite(du) & np.isfinite(d2u)
        x = x[m]
        du = du[m]
        d2u = d2u[m]
        if int(x.size) < 5:
            continue
        order = np.argsort(x)
        x = x[order]
        du = du[order]
        d2u = d2u[order]
        keep = np.ones(int(x.size), dtype=bool)
        keep[1:] = np.abs(np.diff(x)) > max(1.0e-12, 1.0e-12 * float(np.max(np.abs(x))))
        x = x[keep]
        du = du[keep]
        d2u = d2u[keep]
        if int(x.size) < 5:
            continue
        try:
            edge_order = 2 if int(x.size) >= 3 else 1
            fd = np.gradient(du, x, edge_order=edge_order)
        except Exception:
            continue
        diff = d2u - fd
        diff = diff[np.isfinite(diff)]
        if int(diff.size):
            diffs.append(diff)
    if not diffs:
        return float("inf")
    return _robust_rms(np.concatenate(diffs))


def _make_candidate_with_witness_mapping(
    result: FactorizedSearchDEResult,
    *,
    spec: oracle_de.DELabSpec,
    theta: Mapping[str, Any],
) -> dict[str, Any]:
    constants = [{"name": str(c.name), "value": float(c.value)} for c in tuple(spec.constants)]
    return {
        "engine": "factorized_search",
        "kind": "factorized",
        "order": int(result.order),
        "x_axis": int(result.x_axis),
        "include_x": bool(spec.include_x),
        "include_u": bool(spec.include_u),
        "include_du": bool(spec.include_du),
        "constants_ordered": constants,
        "feature_names": list(result.feature_names),
        "expr_ast": oracle_de._to_jsonable(result.expr_ast),
        "mapping": {
            "kind": "poly",
            "coeffs": [float(theta["b"]), float(theta["a"])],
            "mu": 0.0,
            "std": 1.0,
        },
        "mapping_kind": "poly",
        "domain_projection": oracle_de._to_jsonable((result.diagnostics or {}).get("domain_projection", None)),
    }


_GENERATOR_WITNESS_MATERIALIZABLE_STATUSES = {
    "EXACT_STRUCTURAL_GENERATOR",
    "DYNAMICALLY_COMPATIBLE",
    "VIABLE_WITH_MODELLING_ERROR",
}


def _input_report_from_direct_spec(
    result: FactorizedSearchDEResult,
    spec: oracle_de.DELabSpec,
) -> dict[str, Any]:
    return {
        "x_axis": int(result.x_axis),
        "include_x": bool(spec.include_x),
        "include_u": bool(spec.include_u),
        "include_du": bool(spec.include_du),
        "constants_ordered": [
            {"name": str(c.name), "value": float(c.value)}
            for c in tuple(spec.constants)
        ],
    }


def _prepend_generator_witness_shortlist_row(
    result: FactorizedSearchDEResult,
    *,
    spec: oracle_de.DELabSpec,
    generator_status: str,
) -> None:
    diag = result.diagnostics if isinstance(result.diagnostics, dict) else {}
    row = {
        "engine": "factorized_search",
        "kind": "factorized",
        "order": int(result.order),
        "x_axis": int(result.x_axis),
        "include_x": bool(spec.include_x),
        "include_u": bool(spec.include_u),
        "include_du": bool(spec.include_du),
        "constants_ordered": [
            {"name": str(c.name), "value": float(c.value)}
            for c in tuple(spec.constants)
        ],
        "feature_names": list(result.feature_names),
        "expr_ast": oracle_de._to_jsonable(result.expr_ast),
        "mapping": oracle_de._to_jsonable(result.mapping),
        "mapping_kind": str(result.mapping_kind),
        "score": float(result.probe_mse),
        "score_raw": float(result.probe_mse),
        "probe_mse": float(result.probe_mse),
        "probe_rms": float(result.probe_rms),
        "mse": float(result.probe_mse),
        "canonical_equation": str(result.canonical_equation),
        "canonical_equation_raw": str(result.canonical_equation_raw or result.canonical_equation),
        "canonical_equation_simplified": str(result.canonical_equation_simplified or result.canonical_equation),
        "rhs_ast_raw": None if result.rhs_ast_raw is None else repr(result.rhs_ast_raw),
        "rhs_ast_simplified": None if result.rhs_ast_simplified is None else repr(result.rhs_ast_simplified),
        "residual_ast_raw": None if result.residual_ast_raw is None else repr(result.residual_ast_raw),
        "residual_ast_simplified": None
        if result.residual_ast_simplified is None
        else repr(result.residual_ast_simplified),
        "residual_ast": None if result.residual_ast is None else repr(result.residual_ast),
        "candidate_source": "direct_generator_witness",
        "source_lane": "direct_residual_fss",
        "generator_status": str(generator_status),
        "evidence_tier": "generator_witness",
        "witness_materialized": True,
    }

    existing: list[dict[str, Any]] = []
    union = diag.get("shortlist_union", None)
    if isinstance(union, list):
        existing = [dict(r) for r in union if isinstance(r, dict)]
    else:
        report = diag.get("report", None)
        if isinstance(report, dict):
            hp = report.get("hp", {}) if isinstance(report.get("hp", None), dict) else {}
            try:
                existing = factorized_search_report_shortlist(report, limit=hp.get("return_topk", None))
            except Exception:
                existing = []

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for payload in [row, *existing]:
        key = json.dumps(
            oracle_de._to_jsonable(
                {
                    "expr_ast": payload.get("expr_ast", None),
                    "mapping": payload.get("mapping", None),
                    "mapping_kind": payload.get("mapping_kind", None),
                    "canonical_equation": payload.get("canonical_equation", None),
                }
            ),
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        payload = dict(payload)
        rank = int(len(rows))
        payload["shortlist_rank"] = rank
        payload["candidate_rank"] = rank
        rows.append(payload)
    diag["shortlist_union"] = rows
    result.diagnostics = diag


def _materialize_generator_witness_result(
    result: FactorizedSearchDEResult,
    generator_witness: Mapping[str, Any],
    *,
    spec: oracle_de.DELabSpec,
) -> bool:
    diag = result.diagnostics if isinstance(result.diagnostics, dict) else {}
    status = str(generator_witness.get("generator_status", "") or "").strip().upper()
    witness_candidate = generator_witness.get("witness_candidate", None)
    if not isinstance(witness_candidate, Mapping):
        diag["witness_materialized"] = False
        diag["witness_materialization_error"] = "missing_witness_candidate"
        result.diagnostics = diag
        return False
    mapping = witness_candidate.get("mapping", None)
    if not isinstance(mapping, Mapping):
        diag["witness_materialized"] = False
        diag["witness_materialization_error"] = "missing_witness_mapping"
        result.diagnostics = diag
        return False

    mapping_kind = str(witness_candidate.get("mapping_kind", "poly") or "poly")
    try:
        input_report = _input_report_from_direct_spec(result, spec)
        inner_nn = factorized_search_to_nestynet(result.expr_ast)
        rhs_ast = embed_mapping_in_ast(
            inner_nn,
            dict(mapping),
            _input_exprs_from_report(input_report, order=int(result.order)),
            units_mode="raw",
        )
        residual_ast = Add(
            _anchor_for_order(int(result.order), x_axis=int(result.x_axis)),
            Mul(ConstNode(-1.0), rhs_ast),
        )
        compiled_payload = _compiled_de_ast_payload(rhs_ast=rhs_ast, residual_ast=residual_ast)
    except Exception as exc:
        diag["witness_materialized"] = False
        diag["witness_materialization_error"] = type(exc).__name__
        result.diagnostics = diag
        return False

    result.mapping = dict(mapping)
    result.mapping_kind = mapping_kind
    result.rhs_ast = compiled_payload["rhs_ast_simplified"]
    result.residual_ast = compiled_payload["residual_ast_simplified"]
    result.rhs_ast_raw = compiled_payload["rhs_ast_raw"]
    result.residual_ast_raw = compiled_payload["residual_ast_raw"]
    result.rhs_ast_simplified = compiled_payload["rhs_ast_simplified"]
    result.residual_ast_simplified = compiled_payload["residual_ast_simplified"]
    result.canonical_equation_raw = repr(result.residual_ast_raw)
    result.canonical_equation_simplified = repr(result.residual_ast_simplified)
    result.canonical_equation = str(result.canonical_equation_simplified)
    diag["witness_materialized"] = True
    diag["witness_materialized_status"] = status
    diag["candidate_source"] = "direct_generator_witness"
    result.diagnostics = diag
    _prepend_generator_witness_shortlist_row(result, spec=spec, generator_status=status)
    return True


def _rollout_order2_generator_witness(
    candidate: dict[str, Any],
    groups: Sequence[DEFeatureGroup],
    *,
    spec: oracle_de.DELabSpec,
    rescue_cfg: FactorizedSearchDERescueConfig,
) -> dict[str, Any]:
    from scipy.integrate import solve_ivp

    order, rhs_fn = factorized_search_report_to_rhs_callable(candidate)
    if int(order) != 2:
        return {"status": "not_order2", "traj_scores": []}

    max_span = _safe_float(getattr(rescue_cfg, "generator_witness_rollout_max_span", 5.0), default=5.0)
    frac = _safe_float(getattr(rescue_cfg, "generator_witness_rollout_window_fraction", 0.25), default=0.25)
    state_tau_rel = _safe_float(getattr(rescue_cfg, "generator_witness_state_tau_rel", 2.0e-2), default=2.0e-2)
    vel_tau_rel = _safe_float(getattr(rescue_cfg, "generator_witness_velocity_tau_rel", 5.0e-2), default=5.0e-2)
    tau_abs = _safe_float(getattr(rescue_cfg, "generator_witness_tau_abs", 1.0e-8), default=1.0e-8)

    traj_scores: list[dict[str, Any]] = []
    for group in groups:
        try:
            traj = _group_to_validation_trajectory(group, spec=spec)
        except Exception as exc:
            traj_scores.append({"traj_id": str(group.id), "status": "ERROR", "error": type(exc).__name__})
            continue
        t = _np1(traj.x)
        u = _np1(traj.u)
        v = _np1(traj.du)
        m = np.isfinite(t) & np.isfinite(u) & np.isfinite(v)
        t = t[m]
        u = u[m]
        v = v[m]
        if int(t.size) < 5:
            traj_scores.append({"traj_id": str(group.id), "status": "ERROR", "error": "too_few_rows"})
            continue
        order_idx = np.argsort(t)
        t = t[order_idx]
        u = u[order_idx]
        v = v[order_idx]
        keep = np.ones(int(t.size), dtype=bool)
        keep[1:] = np.abs(np.diff(t)) > max(1.0e-12, 1.0e-12 * float(np.max(np.abs(t))))
        t = t[keep]
        u = u[keep]
        v = v[keep]
        if int(t.size) < 5:
            traj_scores.append({"traj_id": str(group.id), "status": "ERROR", "error": "too_few_unique_rows"})
            continue

        full_span = float(t[-1] - t[0])
        if not math.isfinite(full_span) or full_span <= 0.0:
            traj_scores.append({"traj_id": str(group.id), "status": "ERROR", "error": "nonpositive_span"})
            continue
        horizon = min(full_span, max_span if max_span > 0 else full_span, max(1.0e-12, frac * full_span))
        end_t = float(t[0] + horizon)
        mask = t <= end_t
        if int(np.sum(mask)) < 5:
            take = min(int(t.size), max(5, min(32, int(t.size))))
            mask = np.zeros(int(t.size), dtype=bool)
            mask[:take] = True
        t_eval = t[mask]
        u_eval = u[mask]
        v_eval = v[mask]
        if int(t_eval.size) < 2 or float(t_eval[-1]) <= float(t_eval[0]):
            traj_scores.append({"traj_id": str(group.id), "status": "ERROR", "error": "bad_rollout_window"})
            continue

        scale_u = _robust_scale(u_eval)
        scale_v = _robust_scale(v_eval)
        sigma_u = max(float(tau_abs), float(state_tau_rel) * scale_u)
        sigma_v = max(float(tau_abs), float(vel_tau_rel) * scale_v)
        u0 = float(u_eval[0])
        v0 = float(v_eval[0])
        offsets = [
            (0.0, 0.0),
            (0.0, -1.0),
            (0.0, 1.0),
            (-0.5, 0.0),
            (0.5, 0.0),
            (-0.5, -0.5),
            (-0.5, 0.5),
            (0.5, -0.5),
            (0.5, 0.5),
        ]
        best: dict[str, Any] | None = None
        for du0_z, dv0_z in offsets:
            y0 = [u0 + float(du0_z) * sigma_u, v0 + float(dv0_z) * sigma_v]
            try:
                sol = solve_ivp(
                    lambda tt, yy: rhs_fn(float(tt), yy),
                    (float(t_eval[0]), float(t_eval[-1])),
                    y0,
                    t_eval=t_eval,
                    method="DOP853",
                    rtol=1.0e-7,
                    atol=1.0e-9,
                    max_step=max(1.0e-9, float(t_eval[-1] - t_eval[0]) / 200.0),
                )
            except Exception:
                continue
            if not bool(getattr(sol, "success", False)) or getattr(sol, "y", np.empty((0, 0))).shape[1] != int(t_eval.size):
                continue
            y_sol = np.asarray(sol.y, dtype=np.float64)
            if y_sol.shape[0] < 2 or not np.isfinite(y_sol).all():
                continue
            err_u = y_sol[0] - u_eval
            err_v = y_sol[1] - v_eval
            u_nrmse = normalized_rmse(u_eval, y_sol[0])
            v_nrmse = normalized_rmse(v_eval, y_sol[1])
            u_rms_z = _robust_rms(err_u / sigma_u)
            v_rms_z = _robust_rms(err_v / sigma_v)
            u_q95_z = _q_abs(err_u / sigma_u, 0.95)
            init_shift_rms_z = float(math.sqrt((float(du0_z) ** 2 + float(dv0_z) ** 2) / 2.0))
            objective = float(u_rms_z + 0.2 * v_rms_z + 0.1 * init_shift_rms_z)
            row = {
                "traj_id": str(group.id),
                "status": "OK",
                "u_nrmse": float(u_nrmse),
                "v_nrmse": float(v_nrmse),
                "u_rms_z": float(u_rms_z),
                "u_q95_z": float(u_q95_z),
                "v_rms_z": float(v_rms_z),
                "init_shift_rms_z": float(init_shift_rms_z),
                "n_points": int(t_eval.size),
                "t0": float(t_eval[0]),
                "t1": float(t_eval[-1]),
                "objective": float(objective),
            }
            if best is None or float(row["objective"]) < float(best["objective"]):
                best = row
        if best is None:
            traj_scores.append({"traj_id": str(group.id), "status": "ERROR", "error": "all_rollouts_failed"})
        else:
            traj_scores.append(best)

    ok_scores = [row for row in traj_scores if str(row.get("status")) == "OK"]
    if not ok_scores:
        return {
            "status": "ERROR",
            "traj_scores": traj_scores,
            "rollout_u_rms_z": float("inf"),
            "rollout_u_q95_z": float("inf"),
            "rollout_v_rms_z": float("inf"),
            "rollout_u_nrmse": float("inf"),
            "init_shift_rms_z": float("inf"),
        }
    return {
        "status": "OK",
        "traj_scores": traj_scores,
        "rollout_u_rms_z": float(max(float(row["u_rms_z"]) for row in ok_scores)),
        "rollout_u_q95_z": float(max(float(row["u_q95_z"]) for row in ok_scores)),
        "rollout_v_rms_z": float(max(float(row["v_rms_z"]) for row in ok_scores)),
        "rollout_u_nrmse": float(max(float(row["u_nrmse"]) for row in ok_scores)),
        "init_shift_rms_z": float(max(float(row["init_shift_rms_z"]) for row in ok_scores)),
        "ok_trajectories": int(len(ok_scores)),
        "total_trajectories": int(len(traj_scores)),
    }


def validate_order2_generator_witness(
    result: FactorizedSearchDEResult,
    groups: Sequence[DEFeatureGroup],
    *,
    spec: oracle_de.DELabSpec,
    rescue_cfg: FactorizedSearchDERescueConfig,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    """Validate an order-2 direct FSS candidate as a trajectory generator."""

    if int(getattr(result, "order", -1)) != 2:
        return {"enabled": False, "generator_status": "NOT_APPLICABLE", "reason": "order_not_2"}
    if result.expr_ast is None:
        return {"enabled": True, "generator_status": "NOT_VIABLE", "reason": "missing_expr_ast"}

    try:
        validated_groups, fit_groups, probe_groups, _ = _partition_feature_groups(groups, dtype=dtype)
    except Exception as exc:
        return {
            "enabled": True,
            "generator_status": "NOT_VIABLE",
            "reason": "feature_group_error",
            "error": type(exc).__name__,
        }

    candidate = _make_candidate_with_witness_mapping(
        result,
        spec=spec,
        theta={"a": 1.0, "b": 0.0},
    )
    candidate["mapping"] = oracle_de._to_jsonable(result.mapping)
    candidate["mapping_kind"] = str(result.mapping_kind)

    group_tables: list[dict[str, Any]] = []
    for group in validated_groups:
        try:
            table = _order2_group_witness_table(candidate, group, spec=spec, dtype=dtype)
        except Exception as exc:
            return {
                "enabled": True,
                "generator_status": "NOT_VIABLE",
                "reason": "candidate_eval_error",
                "error": type(exc).__name__,
                "group_id": str(group.id),
            }
        if int(table.get("n", 0)) >= 3:
            group_tables.append(table)
    if not group_tables:
        return {"enabled": True, "generator_status": "NOT_VIABLE", "reason": "no_finite_witness_rows"}

    phi_all = np.concatenate([_np1(row["phi"]) for row in group_tables])
    y_all = np.concatenate([_np1(row["y"]) for row in group_tables])
    try:
        global_fit = _fit_affine_1d(phi_all, y_all)
    except Exception as exc:
        return {
            "enabled": True,
            "generator_status": "NOT_VIABLE",
            "reason": "global_affine_fit_failed",
            "error": type(exc).__name__,
        }

    per_traj: list[dict[str, Any]] = []
    for row in group_tables:
        try:
            fit_j = _fit_affine_1d(row["phi"], row["y"])
        except Exception as exc:
            per_traj.append({"traj_id": str(row["group_id"]), "status": "ERROR", "error": type(exc).__name__})
            continue
        per_traj.append(
            {
                "traj_id": str(row["group_id"]),
                "status": "OK",
                "a": float(fit_j["a"]),
                "b": float(fit_j["b"]),
                "n": int(fit_j["n"]),
                "rms": _robust_rms(fit_j["resid"]),
            }
        )

    loo: list[dict[str, Any]] = []
    if len(group_tables) > 1:
        for i, row_skip in enumerate(group_tables):
            keep = [row for j, row in enumerate(group_tables) if j != i]
            try:
                fit_loo = _fit_affine_1d(
                    np.concatenate([_np1(row["phi"]) for row in keep]),
                    np.concatenate([_np1(row["y"]) for row in keep]),
                )
            except Exception as exc:
                loo.append({"held_out_traj_id": str(row_skip["group_id"]), "status": "ERROR", "error": type(exc).__name__})
                continue
            loo.append(
                {
                    "held_out_traj_id": str(row_skip["group_id"]),
                    "status": "OK",
                    "a": float(fit_loo["a"]),
                    "b": float(fit_loo["b"]),
                    "n": int(fit_loo["n"]),
                }
            )

    a_global = float(global_fit["a"])
    b_global = float(global_fit["b"])
    target_scale = _robust_scale(y_all)
    fd_sigma = _d2u_fd_disagreement_scale(validated_groups, spec=spec)
    tau_rel = _safe_float(getattr(rescue_cfg, "generator_witness_tau_rel", 5.0e-2), default=5.0e-2)
    tau_abs = _safe_float(getattr(rescue_cfg, "generator_witness_tau_abs", 1.0e-8), default=1.0e-8)
    sigma_u2 = max(float(tau_abs), float(tau_rel) * target_scale)
    if math.isfinite(fd_sigma):
        sigma_u2 = max(sigma_u2, float(fd_sigma))

    local_z = _np1(global_fit["resid"]) / max(float(tau_abs), float(sigma_u2))
    local_rms_z = _robust_rms(local_z)
    local_q95_z = _q_abs(local_z, 0.95)
    intercept_z = abs(float(b_global)) / max(float(tau_abs), float(sigma_u2))

    ok_a = [float(row["a"]) for row in per_traj if str(row.get("status")) == "OK" and math.isfinite(float(row.get("a", float("nan"))))]
    coeff_spread_rel = float("inf")
    sign_consistent = False
    if ok_a:
        denom = max(abs(float(a_global)), 1.0e-12)
        coeff_spread_rel = float(max(abs(a - float(a_global)) for a in ok_a) / denom)
        if abs(float(a_global)) <= 1.0e-12:
            sign_consistent = all(abs(a) <= max(1.0e-10, 1.0e-6 * target_scale) for a in ok_a)
        else:
            sign_global = math.copysign(1.0, float(a_global))
            sign_consistent = all(math.copysign(1.0, a) == sign_global for a in ok_a if abs(a) > 1.0e-12)

    max_coeff_spread = _safe_float(
        getattr(rescue_cfg, "generator_witness_max_coeff_spread_rel", 5.0e-1),
        default=5.0e-1,
    )
    max_intercept_z = _safe_float(getattr(rescue_cfg, "generator_witness_max_intercept_z", 3.0), default=3.0)
    max_local_rms_z = _safe_float(getattr(rescue_cfg, "generator_witness_max_local_rms_z", 2.0), default=2.0)
    max_local_q95_z = _safe_float(getattr(rescue_cfg, "generator_witness_max_local_q95_z", 3.5), default=3.5)

    coeff_ok = bool(sign_consistent and math.isfinite(coeff_spread_rel) and coeff_spread_rel <= max_coeff_spread)
    intercept_ok = bool(math.isfinite(intercept_z) and intercept_z <= max_intercept_z)
    local_ok = bool(
        math.isfinite(local_rms_z)
        and math.isfinite(local_q95_z)
        and local_rms_z <= max_local_rms_z
        and local_q95_z <= max_local_q95_z
    )

    witness_candidate = _make_candidate_with_witness_mapping(
        result,
        spec=spec,
        theta={"a": float(a_global), "b": float(b_global)},
    )
    rollout = _rollout_order2_generator_witness(
        witness_candidate,
        probe_groups or fit_groups,
        spec=spec,
        rescue_cfg=rescue_cfg,
    )
    max_rollout_z = _safe_float(
        getattr(rescue_cfg, "generator_witness_max_rollout_u_rms_z", 3.5),
        default=3.5,
    )
    max_rollout_nrmse = _safe_float(
        getattr(rescue_cfg, "generator_witness_max_rollout_u_nrmse", 5.0e-2),
        default=5.0e-2,
    )
    rollout_ok = bool(
        str(rollout.get("status")) == "OK"
        and _safe_float(rollout.get("rollout_u_rms_z", float("inf")), default=float("inf")) <= max_rollout_z
        and _safe_float(rollout.get("rollout_u_nrmse", float("inf")), default=float("inf")) <= max_rollout_nrmse
    )
    max_rollout_v_z = _safe_float(
        getattr(rescue_cfg, "generator_witness_max_rollout_v_rms_z", max_rollout_z),
        default=max_rollout_z,
    )
    max_init_shift_z = _safe_float(
        getattr(rescue_cfg, "generator_witness_max_init_shift_rms_z", max_rollout_z),
        default=max_rollout_z,
    )
    exact_rollout_ok = bool(
        rollout_ok
        and _safe_float(rollout.get("rollout_v_rms_z", float("inf")), default=float("inf")) <= max_rollout_v_z
        and _safe_float(rollout.get("init_shift_rms_z", float("inf")), default=float("inf")) <= max_init_shift_z
    )

    if coeff_ok and intercept_ok and local_ok and rollout_ok:
        if (
            local_rms_z <= 0.75
            and _safe_float(rollout.get("rollout_u_nrmse", float("inf")), default=float("inf")) <= 1.0e-2
            and exact_rollout_ok
        ):
            status = "EXACT_STRUCTURAL_GENERATOR"
            legacy_status = "EXACT_GENERATOR"
        else:
            status = "DYNAMICALLY_COMPATIBLE"
            legacy_status = "VIABLE_WITH_MODELLING_ERROR"
    elif coeff_ok and (local_ok or rollout_ok):
        status = "AMBIGUOUS_ROLE"
        legacy_status = "AMBIGUOUS_UNDER_ERROR_MODEL"
    else:
        status = "NOT_VIABLE"
        legacy_status = "NOT_VIABLE"

    out = {
        "enabled": True,
        "evidence_tier": "generator_witness",
        "generator_status": status,
        "generator_status_legacy": legacy_status,
        "theta_shared": {"a": float(a_global), "b": float(b_global), "n": int(global_fit["n"])},
        "theta_per_traj": per_traj,
        "theta_loo": loo,
        "theta_spread_rel": None if not math.isfinite(coeff_spread_rel) else float(coeff_spread_rel),
        "theta_sign_consistent": bool(sign_consistent),
        "intercept_abs": float(abs(b_global)),
        "intercept_z": None if not math.isfinite(intercept_z) else float(intercept_z),
        "target_scale": float(target_scale),
        "sigma_u2": float(sigma_u2),
        "sigma_u2_components": {
            "fd_disagreement": None if not math.isfinite(fd_sigma) else float(fd_sigma),
            "tau_rel_floor": float(float(tau_rel) * target_scale),
            "tau_abs": float(tau_abs),
        },
        "local_rms_z": None if not math.isfinite(local_rms_z) else float(local_rms_z),
        "local_q95_z": None if not math.isfinite(local_q95_z) else float(local_q95_z),
        "coeff_ok": bool(coeff_ok),
        "intercept_ok": bool(intercept_ok),
        "local_ok": bool(local_ok),
        "rollout_ok": bool(rollout_ok),
        "exact_rollout_ok": bool(exact_rollout_ok),
        "max_rollout_v_rms_z": float(max_rollout_v_z),
        "max_init_shift_rms_z": float(max_init_shift_z),
        "rollout": oracle_de._to_jsonable(rollout),
        "rollout_u_rms_z": rollout.get("rollout_u_rms_z", None),
        "rollout_u_q95_z": rollout.get("rollout_u_q95_z", None),
        "rollout_v_rms_z": rollout.get("rollout_v_rms_z", None),
        "rollout_u_nrmse": rollout.get("rollout_u_nrmse", None),
        "init_shift_rms_z": rollout.get("init_shift_rms_z", None),
        "witness_candidate": oracle_de._to_jsonable(witness_candidate),
    }
    return oracle_de._to_jsonable(out)


def _row_mse_value(row: dict[str, Any]) -> float:
    mse = _safe_float(row.get("mse", None), default=float("inf"))
    if math.isfinite(mse):
        return float(mse)
    return _safe_float(row.get("score_raw", None), default=float("inf"))


def _build_domain_eval_cloud(
    x_fit_parts: Sequence[torch.Tensor],
    x_probe_parts: Sequence[torch.Tensor],
    *,
    seed: int,
    dtype: torch.dtype,
    max_rows: int = 1024,
    perturb_rel_scale: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    cloud = torch.cat([*x_fit_parts, *x_probe_parts], dim=0)
    cloud = torch.as_tensor(cloud, dtype=dtype)
    cloud = _uniform_subsample_rows(cloud, max_rows=int(max_rows))
    if int(cloud.numel()) == 0:
        return cloud, cloud.clone()

    col_std = torch.std(cloud, dim=0, unbiased=False)
    col_abs = torch.amax(torch.abs(cloud), dim=0)
    col_scale = torch.where(col_std > 0.0, col_std, col_abs)
    col_scale = torch.clamp(col_scale, min=1.0e-6).reshape(1, -1)

    gen = torch.Generator(device=cloud.device)
    gen.manual_seed(int(seed))
    noise = torch.randn(cloud.shape, generator=gen, device=cloud.device, dtype=cloud.dtype)
    perturbed = cloud + float(perturb_rel_scale) * col_scale * noise
    return cloud, perturbed


def _score_candidate_domain_fragility(
    expr_ast: Any,
    mapping: dict[str, Any],
    *,
    actual_cloud: torch.Tensor,
    perturbed_cloud: torch.Tensor,
    observed_scale: float,
    dtype: torch.dtype,
    domain_projection_cfg: dict[str, Any] | None = None,
    soft_magnitude_ratio: float = 50.0,
    hard_magnitude_ratio: float = 1.0e4,
    soft_jump_ratio: float = 25.0,
    hard_jump_ratio: float = 1.0e3,
) -> dict[str, Any]:
    diag: dict[str, Any] = {
        "domain_ok": True,
        "domain_failure_reason": None,
        "domain_fragility_penalty": 0.0,
        "domain_eval_observed_scale": float(max(observed_scale, 1.0e-8)),
        "domain_eval_n_actual": int(actual_cloud.shape[0]),
        "domain_eval_n_perturbed": int(perturbed_cloud.shape[0]),
    }
    scale = float(diag["domain_eval_observed_scale"])
    projection_enabled = isinstance(domain_projection_cfg, dict) and bool(
        domain_projection_cfg.get("score_domain_projection_enable", False)
    )

    try:
        vals_actual = _candidate_eval_array(
            expr_ast,
            mapping,
            actual_cloud,
            dtype=dtype,
            domain_projection_cfg=domain_projection_cfg,
        )
    except FloatingPointError:
        diag["domain_ok"] = False
        diag["domain_failure_reason"] = "nonfinite_actual"
        diag["domain_fragility_penalty"] = float("inf")
        return diag
    except Exception as exc:
        diag["domain_ok"] = False
        diag["domain_failure_reason"] = f"eval_error_actual:{type(exc).__name__}"
        diag["domain_fragility_penalty"] = float("inf")
        return diag
    if not np.isfinite(vals_actual).all():
        diag["domain_ok"] = False
        diag["domain_failure_reason"] = "nonfinite_actual"
        diag["domain_fragility_penalty"] = float("inf")
        return diag

    try:
        vals_perturbed = _candidate_eval_array(
            expr_ast,
            mapping,
            perturbed_cloud,
            dtype=dtype,
            domain_projection_cfg=domain_projection_cfg,
        )
    except FloatingPointError:
        if projection_enabled:
            diag["domain_failure_reason"] = "perturbed_outside_projection_tube"
            diag["domain_eval_perturbed_outside_projection_tube"] = True
            diag["domain_eval_max_abs_actual"] = (
                float(np.max(np.abs(vals_actual))) if vals_actual.size else 0.0
            )
            return diag
        diag["domain_ok"] = False
        diag["domain_failure_reason"] = "nonfinite_perturbed"
        diag["domain_fragility_penalty"] = float("inf")
        return diag
    except Exception as exc:
        if projection_enabled:
            diag["domain_failure_reason"] = f"eval_error_perturbed:{type(exc).__name__}"
            diag["domain_eval_perturbed_outside_projection_tube"] = True
            diag["domain_eval_max_abs_actual"] = (
                float(np.max(np.abs(vals_actual))) if vals_actual.size else 0.0
            )
            return diag
        diag["domain_ok"] = False
        diag["domain_failure_reason"] = f"eval_error_perturbed:{type(exc).__name__}"
        diag["domain_fragility_penalty"] = float("inf")
        return diag
    if not np.isfinite(vals_perturbed).all():
        if projection_enabled:
            diag["domain_failure_reason"] = "nonfinite_perturbed"
            diag["domain_eval_perturbed_outside_projection_tube"] = True
            diag["domain_eval_max_abs_actual"] = (
                float(np.max(np.abs(vals_actual))) if vals_actual.size else 0.0
            )
            return diag
        diag["domain_ok"] = False
        diag["domain_failure_reason"] = "nonfinite_perturbed"
        diag["domain_fragility_penalty"] = float("inf")
        return diag

    max_actual = float(np.max(np.abs(vals_actual))) if vals_actual.size else 0.0
    max_perturbed = float(np.max(np.abs(vals_perturbed))) if vals_perturbed.size else 0.0
    max_ratio = max(max_actual, max_perturbed) / scale
    jump_max = float(np.max(np.abs(vals_perturbed - vals_actual))) if vals_actual.size else 0.0
    jump_ratio = jump_max / scale

    penalty = 0.0
    if max_ratio > float(soft_magnitude_ratio):
        penalty += math.log10(max_ratio / float(soft_magnitude_ratio) + 1.0)
    if jump_ratio > float(soft_jump_ratio):
        penalty += math.log10(jump_ratio / float(soft_jump_ratio) + 1.0)

    diag.update(
        {
            "domain_eval_max_abs_actual": float(max_actual),
            "domain_eval_max_abs_perturbed": float(max_perturbed),
            "domain_eval_jump_max_abs": float(jump_max),
            "domain_eval_max_ratio": float(max_ratio),
            "domain_eval_jump_ratio": float(jump_ratio),
            "domain_fragility_penalty": float(penalty),
        }
    )

    if max_ratio > float(hard_magnitude_ratio):
        diag["domain_ok"] = False
        diag["domain_failure_reason"] = "magnitude_explosion"
        diag["domain_fragility_penalty"] = float("inf")
    elif jump_ratio > float(hard_jump_ratio):
        diag["domain_ok"] = False
        diag["domain_failure_reason"] = "perturbation_explosion"
        diag["domain_fragility_penalty"] = float("inf")

    return diag


def _row_rerank_key(
    row: dict[str, Any],
    *,
    score_scale: float = 1.0,
) -> tuple[int, int, int, float, float, int, int, int, float, int]:
    integrate_ok = row.get("integrate_ok", None)
    integrate_mse = _safe_float(row.get("integrate_mse", None), default=float("inf"))
    if integrate_ok is True and math.isfinite(integrate_mse):
        integration_bucket = 0
    elif integrate_ok is None:
        integration_bucket = 1
    else:
        integration_bucket = 2

    domain_ok = row.get("domain_ok", None)
    domain_bucket = 1 if domain_ok is False else 0
    structural_bucket = 1 if not _row_structurally_eligible(row) else 0
    fragility_penalty = _safe_float(row.get("domain_fragility_penalty", 0.0), default=float("inf"))
    size = _safe_int(row.get("symbolic_size_simplified", row.get("size", None)))
    score = _safe_float(row.get("score", None), default=float("inf")) * float(score_scale)
    score_decade = _score_decade(score)
    original_rank = _safe_int(row.get("original_rank", row.get("score_rank", None)))
    return (
        int(domain_bucket),
        int(structural_bucket),
        int(integration_bucket),
        float(integrate_mse) if int(integration_bucket) == 0 else float("inf"),
        float(fragility_penalty),
        int(_row_validation_decade(row)),
        int(score_decade),
        int(size),
        float(score),
        int(original_rank),
    )


def _select_best_global(spec: oracle_de.DELabSpec, per_order: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    # Prefer lower order via a score adjustment, but only after integration-aware
    # quality signals have already been considered.
    order_preference_factor = 10.0
    if isinstance(getattr(spec, "extra", None), dict):
        order_preference_factor = float(spec.extra.get("order_preference_factor", order_preference_factor))
    min_ord = min(
        (int(po.get("order", 99)) for po in per_order if po.get("best") is not None),
        default=1,
    )

    best_global = None
    best_key = None
    for po in per_order:
        row = po.get("best", None)
        if row is None:
            continue
        order_i = int(po.get("order", 1))
        key = _row_rerank_key(
            row,
            score_scale=float(order_preference_factor) ** max(0, order_i - min_ord),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_global = row
    return best_global


def _report_order_preference_factor(report: dict[str, Any]) -> float:
    extra = report.get("extra", None)
    if isinstance(extra, dict):
        try:
            return float(extra.get("order_preference_factor", 10.0))
        except Exception:
            return 10.0
    return 10.0


def _report_global_candidate_rows(report: dict[str, Any]) -> list[tuple[tuple[Any, ...], dict[str, Any], list[str]]]:
    per_order = list(report.get("per_order", []) or [])
    min_ord = min(
        (
            int(po.get("order", 99))
            for po in per_order
            if isinstance(po, dict) and list(po.get("results", []) or [])
        ),
        default=1,
    )
    order_preference_factor = _report_order_preference_factor(report)
    rows: list[tuple[tuple[Any, ...], dict[str, Any], list[str]]] = []
    for po in per_order:
        if not isinstance(po, dict):
            continue
        order_i = int(po.get("order", 1))
        feature_names = list(po.get("feature_names", []) or [])
        score_scale = float(order_preference_factor) ** max(0, order_i - min_ord)
        for row in list(po.get("results", []) or []):
            if not isinstance(row, dict):
                continue
            rows.append((_row_rerank_key(row, score_scale=score_scale), row, feature_names))
    rows.sort(key=lambda item: item[0])
    return rows


def _group_to_validation_trajectory(
    group: DEFeatureGroup,
    *,
    spec: oracle_de.DELabSpec,
) -> Any:
    feats = group.features
    x_all = torch.cat([feats.x_fit, feats.x_probe], dim=0)
    u_all = torch.cat([feats.u_fit, feats.u_probe], dim=0).reshape(-1)
    du_all = torch.cat([feats.du_fit, feats.du_probe], dim=0).reshape(-1)
    d2u_all = torch.cat([feats.d2u_fit, feats.d2u_probe], dim=0).reshape(-1)

    xa = int(spec.x_axis)
    x_axis = x_all[:, xa].reshape(-1)
    perm = torch.argsort(x_axis)
    x_axis = x_axis.index_select(0, perm)
    u_all = u_all.index_select(0, perm)
    du_all = du_all.index_select(0, perm)
    d2u_all = d2u_all.index_select(0, perm)

    keep = torch.ones_like(x_axis, dtype=torch.bool)
    if int(x_axis.numel()) > 1:
        tol = max(1.0e-12, 1.0e-12 * float(torch.max(torch.abs(x_axis)).detach().cpu().item()))
        keep[1:] = torch.abs(x_axis[1:] - x_axis[:-1]) > tol
    x_axis = x_axis[keep]
    u_all = u_all[keep]
    du_all = du_all[keep]
    d2u_all = d2u_all[keep]

    return oracle_de._Trajectory(
        traj_id=str(group.id),
        path="<feature_group>",
        x=x_axis.detach().cpu().numpy(),
        u=u_all.detach().cpu().numpy(),
        du=du_all.detach().cpu().numpy(),
        d2u=d2u_all.detach().cpu().numpy(),
    )


def _partition_feature_groups(
    groups: Sequence[DEFeatureGroup],
    *,
    dtype: torch.dtype,
) -> tuple[list[DEFeatureGroup], list[DEFeatureGroup], list[DEFeatureGroup], bool]:
    validated_groups = [
        DEFeatureGroup(
            id=str(group.id),
            features=oracle_de._validate_feature_tensors(group.features, dtype=dtype),
            use_for_fit=bool(getattr(group, "use_for_fit", True)),
            use_for_probe=bool(getattr(group, "use_for_probe", True)),
            surrogate_val_loss=(
                float(getattr(group, "surrogate_val_loss"))
                if getattr(group, "surrogate_val_loss", None) is not None
                else None
            ),
        )
        for group in groups
    ]
    if not validated_groups:
        raise ValueError("No DE feature groups were provided")

    fit_groups = [group for group in validated_groups if bool(group.use_for_fit)]
    probe_groups = [group for group in validated_groups if bool(group.use_for_probe)]

    if not fit_groups:
        raise ValueError("At least one DE feature group must be marked use_for_fit=True")
    if not probe_groups:
        probe_groups = list(fit_groups)
        probe_fallback_to_fit = True
    else:
        probe_fallback_to_fit = False

    unused = [str(group.id) for group in validated_groups if not group.use_for_fit and not group.use_for_probe]
    if unused:
        raise ValueError(
            "Every DE feature group must participate in fit and/or probe; "
            f"unused groups: {unused}"
        )

    return validated_groups, fit_groups, probe_groups, bool(probe_fallback_to_fit)


def de_lab_spec_from_de_cfg(
    cfg,
    *,
    include_fixed_constants: bool = True,
    traj_metric: str = "max",
    split_mode: str = "traj_holdout",
) -> oracle_de.DELabSpec:
    """Translate a ``DESearchConfig`` into a DE-facing factorized symbolic search spec."""

    dims = None
    if getattr(cfg, "units_spec", None) is not None:
        us = cfg.units_spec.unit_system
        dims = oracle_de.DimensionSpec(
            basis=tuple(str(b) for b in us.base),
            x_dim=tuple(float(v) for v in cfg.units_spec.x_dims[int(cfg.x_axis)]),
            u_dim=tuple(float(v) for v in cfg.units_spec.y_phi_dim),
        )

    constants = []
    if include_fixed_constants and getattr(cfg, "units_spec", None) is not None:
        for name, val in (cfg.units_spec.fixed_const_values or {}).items():
            dim = (cfg.units_spec.fixed_const_dims or {}).get(name, None)
            constants.append(
                oracle_de.ConstantSpec(
                    name=str(name),
                    value=float(val),
                    dim=None if dim is None else tuple(float(v) for v in dim),
                )
            )

    # symmetry-reduction whole-law proposals and pulled-back rows stashed on
    # the DE config seed the factorized-search additive-combo pool
    # (oracle_lab_de._gs_symmetry_seed_rows); absent by default.
    extra: dict[str, Any] | None = None
    seed_asts = list(getattr(cfg, "gs_de_reduction_seed_asts", None) or [])
    if seed_asts:
        extra = {"gs_symmetry_seed_asts": tuple(seed_asts)}

    return oracle_de.DELabSpec(
        id="surrogate_de",
        csv_paths=(),
        order_candidates=tuple(int(o) for o in getattr(cfg, "order_candidates", (1, 2))),
        x_axis=int(getattr(cfg, "x_axis", 0)),
        include_x=bool(getattr(cfg, "include_x", True)),
        include_u=bool(getattr(cfg, "include_u", True)),
        include_du=bool(getattr(cfg, "include_du", True)),
        x_col=f"x{int(getattr(cfg, 'x_axis', 0))}",
        u_col="u",
        y_transform="identity",
        traj_metric=str(traj_metric),
        split_mode=str(split_mode),
        constants=tuple(constants),
        dims=dims,
        extra=extra,
    )

__factorized_de_definitions__ = (
    "DEFeatureGroup",
    "FactorizedSearchDERescueConfig",
    "FactorizedSearchDEResult",
    "_diag_inc",
    "_diag_set_min",
    "_diag_set_max",
    "_merge_diagnostics",
    "_resource_maxrss_mb",
    "_current_process_rss_mb",
    "_process_memory_report",
    "_memory_value",
    "_dtype_name",
    "_torch_dtype_from_name",
    "_typed_explorer_task_identity",
    "_first_explorer_diagnostics",
    "_diag_number_from_reports",
    "_best_numeric_row_value",
    "_short_diag_float",
    "_log_typed_explorer_task_event",
    "_record_typed_explorer_task_event",
    "_feature_group_row_summary",
    "_global_group_budgets",
    "_jsonable_ast_to_tuple",
    "_factorized_candidate_key_payload",
    "_factorized_candidate_id",
    "_ordered_constants_from_report",
    "_expr_and_mapping_from_candidate",
    "_float_from_mapping",
    "_domain_projection_cfg_from_diag",
    "_domain_projection_diags_from_candidate",
    "_domain_projection_cfg_from_candidate",
    "_input_exprs_from_report",
    "_no_candidate_result_from_report",
    "_anchor_for_order",
    "_safe_float",
    "_safe_int",
    "_score_decade",
    "_row_validation_decade",
    "_real_scalar_or_none",
    "_integerish",
    "_tuple_static_const_value",
    "_nestynet_static_const_value",
    "_append_reason",
    "_tuple_structural_reasons",
    "_nestynet_structural_reasons",
    "_domain_projection_value_ok",
    "_row_projection_reasons",
    "_broad_row_structural_safety",
    "_row_structurally_eligible",
    "_coefficient_dim_mode",
    "_dim_sub_tuple",
    "_node_dim_jsonable",
    "_coefficient_dim_jsonable",
    "_uniform_subsample_rows",
    "_candidate_eval_array",
    "_observed_derivative_scale",
    "_safe_probe_rel_rms",
    "_de_effective_early_stop_mse",
    "_np1",
    "_finite_np1",
    "_robust_rms",
    "_robust_scale",
    "_q_abs",
    "_fit_affine_1d",
    "_carrier_eval_array",
    "_order2_group_witness_table",
    "_d2u_fd_disagreement_scale",
    "_make_candidate_with_witness_mapping",
    "_input_report_from_direct_spec",
    "_prepend_generator_witness_shortlist_row",
    "_materialize_generator_witness_result",
    "_rollout_order2_generator_witness",
    "validate_order2_generator_witness",
    "_row_mse_value",
    "_build_domain_eval_cloud",
    "_score_candidate_domain_fragility",
    "_row_rerank_key",
    "_select_best_global",
    "_report_order_preference_factor",
    "_report_global_candidate_rows",
    "_group_to_validation_trajectory",
    "_partition_feature_groups",
    "de_lab_spec_from_de_cfg",
)

__factorized_de_constants__ = (
    "_DOMAIN_PROJECTION_DEFAULT_ABS_TOL",
    "_DOMAIN_PROJECTION_DEFAULT_REL_TOL",
    "_DOMAIN_PROJECTION_DEFAULT_MAX_FRAC",
    "_DOMAIN_PROJECTION_DEFAULT_POSITIVE_FLOOR",
    "_DIAGNOSTIC_MIN_KEYS",
    "_DIAGNOSTIC_MAX_KEYS",
    "_GENERATOR_WITNESS_MATERIALIZABLE_STATUSES",
)

__factorized_de_late_bindings__ = (
    "_compiled_de_ast_payload",
    "_compiled_de_row_payload",
    "factorized_search_report_shortlist",
    "factorized_search_report_to_rhs_callable",
    "normalized_rmse",
)
