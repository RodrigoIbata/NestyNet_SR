# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Streamlit GUI for oracle DE factorized symbolic search/continuous skeleton refinement experiments.

Run with:

    streamlit run examples/oracle_factorized_search/oracle_lab_de_streamlit.py
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import pathlib
import queue
import re
import threading
import traceback
from typing import Any, Callable

import streamlit as st
import torch

from nestynet_sr.sr_search.factorized_search.oracle_lab_de import (
    default_oracle_de_hyperparams,
    equation_de_spec_from_dict,
    run_oracle_de_equation,
)
from nestynet_sr.sr_search.y_transforms import get_y_transform_registry


_RE_BEST_MSE = re.compile(r"best_mse[=\s]+([0-9eE+\-.]+)")
_RE_ITER_BEST_EXPR = re.compile(r"\bbest (.+?) \|")
_RE_NEW_BEST = re.compile(r"NEW BEST .*->\s*(.+?)\s+\(mse [^>]*->\s*([0-9eE+\-.]+)")
_RE_ORACLE_LINE = re.compile(
    r"\[oracle\].*best_mse=([0-9eE+\-.]+)\s+expr=(.*?)\s+mapping=([^\s]+)"
)
_RE_BASINS = re.compile(r"\bresidual_basins(?:=|\s+)(\d+)\b")
_RE_ORDER = re.compile(r"\border(?:=|\s+)(\d+)\b")


def _split_csv(raw: str) -> list[str]:
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _split_paths(raw: str) -> list[str]:
    toks = re.split(r"[\n,]+", str(raw))
    return [tok.strip() for tok in toks if tok.strip()]


def _is_blank(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return str(v).strip() == ""


def _records(editor_output: Any) -> list[dict[str, Any]]:
    if hasattr(editor_output, "to_dict"):
        return list(editor_output.to_dict(orient="records"))
    if isinstance(editor_output, list):
        return [dict(row) for row in editor_output]
    return []


def _parse_dim_text(raw: Any, *, n_base: int, where: str) -> list[float]:
    toks = _split_csv(str(raw))
    if len(toks) != int(n_base):
        raise ValueError(f"{where}: expected {n_base} comma-separated exponents, got {len(toks)}")
    out: list[float] = []
    for i, tok in enumerate(toks):
        try:
            out.append(float(tok))
        except Exception as exc:
            raise ValueError(f"{where}[{i}]: invalid exponent {tok!r}") from exc
    return out


def _parse_optional_int(raw: Any) -> int | None:
    try:
        if raw is None:
            return None
        return int(raw)
    except Exception:
        return None


def _parse_finite_float(raw: Any) -> float | None:
    try:
        v = float(raw)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _row_to_jsonable(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in dict(row).items():
        if isinstance(value, (dict, list, tuple)):
            out[str(key)] = json.dumps(value, ensure_ascii=True)
        elif isinstance(value, (bytes, bytearray)):
            out[str(key)] = bytes(value).decode("utf-8", errors="replace")
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[str(key)] = value
        else:
            out[str(key)] = str(value)
    return out


def _rows_to_constants(
    rows: list[dict[str, Any]],
    *,
    dims_enable: bool,
    n_base: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if _is_blank(row.get("name")) and _is_blank(row.get("value")) and _is_blank(row.get("dim")):
            continue

        name = str(row.get("name", "")).strip()
        if name == "":
            raise ValueError(f"constants[{i}].name cannot be empty")
        try:
            value = float(row.get("value"))
        except Exception as exc:
            raise ValueError(f"constants[{i}].value must be numeric") from exc

        payload: dict[str, Any] = {"name": name, "value": float(value)}
        if dims_enable:
            payload["dim"] = _parse_dim_text(
                row.get("dim", ""),
                n_base=n_base,
                where=f"constants[{i}].dim",
            )
        out.append(payload)
    return out


def _build_spec_payload(
    *,
    spec_id: str,
    csv_paths_raw: str,
    order_candidates: list[int],
    x_axis: int,
    include_x: bool,
    include_u: bool,
    include_du: bool,
    x_col: str,
    u_col: str,
    out_idx: int,
    y_transform: str,
    deriv_method: str,
    spline_s: float,
    spline_k: int,
    du_col: str,
    d2u_col: str,
    validate_integrate_topk: int,
    constants_rows: list[dict[str, Any]],
    dims_enable: bool,
    basis_raw: str,
    x_dim_raw: str,
    u_dim_raw: str,
) -> dict[str, Any]:
    paths = _split_paths(csv_paths_raw)
    if not paths:
        raise ValueError("At least one CSV path is required.")

    ords = sorted(set(int(o) for o in order_candidates if int(o) in (1, 2)))
    if not ords:
        raise ValueError("Select at least one order from {1,2}.")

    basis = _split_csv(basis_raw) if dims_enable else []
    n_base = len(basis)

    constants = _rows_to_constants(
        constants_rows,
        dims_enable=bool(dims_enable),
        n_base=n_base,
    )

    deriv_payload: dict[str, Any] = {
        "method": str(deriv_method).strip(),
        "s": float(spline_s),
        "k": int(spline_k),
    }
    du_col_s = str(du_col).strip()
    d2u_col_s = str(d2u_col).strip()
    if du_col_s:
        deriv_payload["du_col"] = du_col_s
    if d2u_col_s:
        deriv_payload["d2u_col"] = d2u_col_s

    payload: dict[str, Any] = {
        "id": str(spec_id).strip() or "oracle_de_gui_equation",
        "csv_paths": list(paths),
        "order_candidates": list(ords),
        "x_axis": int(x_axis),
        "include_x": bool(include_x),
        "include_u": bool(include_u),
        "include_du": bool(include_du),
        "x_col": str(x_col).strip() or "x0",
        "u_col": str(u_col).strip() or "y",
        "out_idx": int(out_idx),
        "y_transform": str(y_transform).strip() or "identity",
        "deriv": deriv_payload,
        "constants": constants,
        "validate_integrate_topk": int(validate_integrate_topk),
    }

    if dims_enable:
        if n_base <= 0:
            raise ValueError("dims.basis cannot be empty when dimensions are enabled.")
        payload["dims"] = {
            "basis": basis,
            "x": _parse_dim_text(x_dim_raw, n_base=n_base, where="dims.x"),
            "u": _parse_dim_text(u_dim_raw, n_base=n_base, where="dims.u"),
        }

    return payload


def _feature_names_from_spec(spec: Any, order: int) -> list[str]:
    names: list[str] = []
    if bool(spec.include_x):
        names.append(str(spec.x_col))
    if bool(spec.include_u):
        names.append(str(spec.u_col))
    if int(order) == 2 and bool(spec.include_du):
        names.append(str(spec.derivative.du_col or "du"))
    for c in list(spec.constants):
        names.append(str(c.name))
    return names


def _format_dim(dim: list[float] | tuple[float, ...]) -> str:
    return ",".join(f"{float(x):g}" for x in dim)


def _render_dim_preview(spec: Any) -> None:
    st.subheader("Dimension Preview")
    if spec.dims is None:
        st.info("No dimension metadata in this spec.")
        return

    x_dim = tuple(spec.dims.x_dim)
    u_dim = tuple(spec.dims.u_dim)
    du_dim = tuple(float(a) - float(b) for a, b in zip(u_dim, x_dim))
    d2u_dim = tuple(float(a) - float(b) for a, b in zip(du_dim, x_dim))

    rows = [
        {"symbol": "x", "dim": _format_dim(x_dim)},
        {"symbol": "u", "dim": _format_dim(u_dim)},
        {"symbol": "du", "dim": _format_dim(du_dim)},
        {"symbol": "d2u", "dim": _format_dim(d2u_dim)},
    ]
    for c in list(spec.constants):
        c_dim = tuple(c.dim) if c.dim is not None else tuple(0.0 for _ in range(len(x_dim)))
        rows.append({"symbol": str(c.name), "dim": _format_dim(c_dim)})
    st.dataframe(rows, width="stretch")


def _render_spec_preview(spec: Any) -> None:
    st.subheader("Order Preview")
    rows: list[dict[str, Any]] = []
    for order in list(spec.order_candidates):
        rows.append(
            {
                "order": int(order),
                "features": ", ".join(_feature_names_from_spec(spec, int(order))),
                "target": "du" if int(order) == 1 else "d2u",
            }
        )
    st.dataframe(rows, width="stretch")


def _render_current_best_panel(current_best: dict[str, Any] | None) -> None:
    st.subheader("Current Best Solution")
    if not isinstance(current_best, dict):
        st.info("No best solution yet. Run a search to populate this panel.")
        return

    mse = float(current_best.get("mse", float("nan")))
    expr = str(current_best.get("expr", ""))
    mapping = str(current_best.get("mapping_kind", ""))
    residual_basins = _parse_optional_int(current_best.get("residual_basins"))
    order = _parse_optional_int(current_best.get("order"))

    c1, c2, c3 = st.columns(3)
    c1.metric("Best MSE", f"{mse:.6g}" if math.isfinite(mse) else "nan")
    c2.metric("Order", str(order) if order is not None else "n/a")
    c3.metric("Residual basins", str(residual_basins) if residual_basins is not None else "n/a")
    st.code(expr if expr else "<empty>", language="text")
    st.caption(f"Mapping: {mapping}")


def _render_log_panel(log_text: str) -> None:
    st.subheader("Processing Logs")
    if not str(log_text).strip():
        st.info("No logs captured yet.")
        return
    st.text_area("Captured logs", value=str(log_text), height=360)


def _render_results(report: dict[str, Any]) -> None:
    st.subheader("Results")
    best = report.get("best")
    if not isinstance(best, dict):
        st.warning("No candidate expression found.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best MSE", f"{float(best.get('mse', float('nan'))):.6g}")
    c2.metric("Best Order", str(int(best.get("order", -1))))
    c3.metric("Wall Time (s)", f"{float(report.get('wall_seconds', float('nan'))):.3f}")
    c4.metric("Per-order runs", str(len(report.get("per_order", []))))

    st.code(str(best.get("expr", "")), language="text")
    st.caption(f"Mapping: {best.get('mapping_kind', '')}")

    per_order = report.get("per_order", [])
    for po in per_order:
        order = int(po.get("order", -1))
        with st.expander(f"Order {order} details", expanded=False):
            o1, o2, o3 = st.columns(3)
            o1.metric("nvars", str(int(po.get("nvars", 0))))
            o2.metric("fit/probe", f"{int(po.get('n_points_fit', 0))}/{int(po.get('n_points_probe', 0))}")
            o3.metric(
                "n_seeds ran",
                f"{int(po.get('n_seeds_ran', 0))}/{int(po.get('n_seeds', 0))}",
            )
            st.caption(
                f"features: {', '.join(str(x) for x in po.get('feature_names', []))} "
                f"| target: {po.get('target_name', '')}"
            )

            best_o = po.get("best")
            if isinstance(best_o, dict):
                st.code(str(best_o.get("expr", "")), language="text")
                st.caption(f"Best MSE: {float(best_o.get('mse', float('nan'))):.6g}")
                if best_o.get("integrate_mse", None) is not None:
                    st.caption(
                        f"Integrate MSE: {float(best_o.get('integrate_mse', float('nan'))):.6g} "
                        f"(ok={bool(best_o.get('integrate_ok', False))})"
                    )

            rows = po.get("results", [])
            if rows:
                safe_rows = [_row_to_jsonable(dict(r)) for r in rows]
                st.dataframe(safe_rows, width="stretch")


def _update_live_best_from_line(
    line: str,
    current_best: dict[str, Any] | None,
    *,
    spec_id: str,
) -> tuple[dict[str, Any], bool]:
    best = dict(current_best) if isinstance(current_best, dict) else {}
    if "spec_id" not in best:
        best["spec_id"] = str(spec_id)
    if "expr" not in best:
        best["expr"] = ""
    if "mse" not in best:
        best["mse"] = float("inf")
    if "mapping_kind" not in best:
        best["mapping_kind"] = ""
    if "residual_basins" not in best:
        best["residual_basins"] = None
    if "order" not in best:
        best["order"] = None

    old_expr = str(best.get("expr", ""))
    old_map = str(best.get("mapping_kind", ""))
    old_mse = float(best.get("mse", float("inf")))
    old_residual_basins = _parse_optional_int(best.get("residual_basins"))
    old_order = _parse_optional_int(best.get("order"))

    new_expr = old_expr
    new_map = old_map
    new_mse = old_mse
    new_residual_basins = old_residual_basins
    new_order = old_order

    def _accept_candidate(
        *,
        mse_v: float | None,
        expr_v: str = "",
        map_v: str = "",
    ) -> None:
        nonlocal new_mse, new_expr, new_map
        if mse_v is None:
            return
        if (not math.isfinite(new_mse)) or (float(mse_v) < float(new_mse)):
            new_mse = float(mse_v)
            if str(expr_v).strip():
                new_expr = str(expr_v).strip()
            if str(map_v).strip():
                new_map = str(map_v).strip()
            return
        if float(mse_v) == float(new_mse):
            if (not str(new_expr).strip()) and str(expr_v).strip():
                new_expr = str(expr_v).strip()
            if (not str(new_map).strip()) and str(map_v).strip():
                new_map = str(map_v).strip()

    m_oracle = _RE_ORACLE_LINE.search(line)
    if m_oracle:
        mse_v = _parse_finite_float(m_oracle.group(1))
        expr_v = str(m_oracle.group(2)).strip()
        map_v = str(m_oracle.group(3)).strip()
        _accept_candidate(mse_v=mse_v, expr_v=expr_v, map_v=map_v)
    else:
        m_new_best = _RE_NEW_BEST.search(line)
        if m_new_best:
            expr_v = str(m_new_best.group(1)).strip()
            mse_v = _parse_finite_float(m_new_best.group(2))
            _accept_candidate(mse_v=mse_v, expr_v=expr_v)

        m_best_mse = _RE_BEST_MSE.search(line)
        if m_best_mse:
            mse_v = _parse_finite_float(m_best_mse.group(1))
            _accept_candidate(mse_v=mse_v)

        m_iter_expr = _RE_ITER_BEST_EXPR.search(line)
        if m_iter_expr:
            expr_v = str(m_iter_expr.group(1)).strip()
            if expr_v and (not str(new_expr).strip()) and (not math.isfinite(new_mse)):
                new_expr = expr_v

    m_residual_basins = _RE_BASINS.search(line)
    if m_residual_basins:
        try:
            new_residual_basins = int(m_residual_basins.group(1))
        except Exception:
            pass

    m_order = _RE_ORDER.search(line)
    if m_order:
        try:
            new_order = int(m_order.group(1))
        except Exception:
            pass

    changed = (
        (new_expr != old_expr)
        or (new_map != old_map)
        or (new_residual_basins != old_residual_basins)
        or (new_order != old_order)
        or (math.isfinite(new_mse) and (not math.isfinite(old_mse) or abs(new_mse - old_mse) > 0.0))
    )

    best["expr"] = new_expr
    best["mse"] = float(new_mse)
    best["mapping_kind"] = new_map
    best["residual_basins"] = new_residual_basins
    best["order"] = new_order
    return best, changed


def _run_oracle_de_equation_live(
    spec: Any,
    *,
    factorized_search_hp: Any,
    seed: int,
    dtype: torch.dtype,
    enforce_dims: bool,
    verbose: bool,
    on_line: Callable[[str], None],
) -> tuple[dict[str, Any], str]:
    q: queue.Queue[str | None] = queue.Queue()
    out_chunks: list[str] = []
    worker_err: dict[str, str] = {}
    worker_out: dict[str, Any] = {}

    class _QueueWriter:
        def write(self, s: str) -> int:
            if s:
                q.put(s)
            return len(s)

        def flush(self) -> None:
            return None

    def _worker() -> None:
        try:
            with contextlib.redirect_stdout(_QueueWriter()):
                worker_out["report"] = run_oracle_de_equation(
                    spec,
                    factorized_search_hp=factorized_search_hp,
                    seed=int(seed),
                    dtype=dtype,
                    enforce_dims=bool(enforce_dims),
                    verbose=bool(verbose),
                )
        except Exception:
            worker_err["traceback"] = traceback.format_exc()
        finally:
            q.put(None)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    pending = ""
    while True:
        item = q.get()
        if item is None:
            break
        out_chunks.append(item)
        pending += item
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            on_line(line)

    if pending:
        on_line(pending)

    t.join()

    if "traceback" in worker_err:
        raise RuntimeError(worker_err["traceback"])
    if "report" not in worker_out:
        raise RuntimeError("Live oracle-de run failed without a report.")
    return dict(worker_out["report"]), "".join(out_chunks)


def _default_torch_num_threads() -> int:
    try:
        n = int(torch.get_num_threads())
    except Exception:
        n = int(os.cpu_count() or 1)
    return max(1, n)


def _y_transform_names() -> list[str]:
    names = [str(t.name) for t in get_y_transform_registry()]
    if not names:
        return ["identity"]
    return names


def _ensure_session_defaults() -> None:
    hp_cfg = default_oracle_de_hyperparams()
    defaults: dict[str, Any] = {
        "oracle_de_gui_spec_id": "de000_simple",
        "oracle_de_gui_csv_paths": "data/feynman_de/de000.csv",
        "oracle_de_gui_order_candidates": [1],
        "oracle_de_gui_x_axis": 0,
        "oracle_de_gui_include_x": True,
        "oracle_de_gui_include_u": True,
        "oracle_de_gui_include_du": True,
        "oracle_de_gui_x_col": "x0",
        "oracle_de_gui_u_col": "y",
        "oracle_de_gui_out_idx": 0,
        "oracle_de_gui_y_transform": "identity",
        "oracle_de_gui_deriv_method": "spline",
        "oracle_de_gui_spline_s": 0.0,
        "oracle_de_gui_spline_k": 3,
        "oracle_de_gui_du_col": "",
        "oracle_de_gui_d2u_col": "",
        "oracle_de_gui_validate_integrate_topk": 1,
        "oracle_de_gui_dims_enable": True,
        "oracle_de_gui_basis": "U,T",
        "oracle_de_gui_x_dim": "0,1",
        "oracle_de_gui_u_dim": "1,0",
        "oracle_de_gui_refine_enable": bool(hp_cfg.refine_enable),
        "oracle_de_gui_ignore_dims": False,
        "oracle_de_gui_dtype_name": "float64",
        "oracle_de_gui_quiet": False,
        "oracle_de_gui_seed": int(hp_cfg.seed),
        "oracle_de_gui_n_seeds": int(hp_cfg.n_seeds),
        "oracle_de_gui_split_iter_across_seeds": bool(hp_cfg.split_iter_across_seeds),
        "oracle_de_gui_n_iter": int(hp_cfg.n_iter),
        "oracle_de_gui_max_depth": int(hp_cfg.max_depth),
        "oracle_de_gui_poly_degree": int(hp_cfg.poly_degree),
        "oracle_de_gui_return_topk": int(hp_cfg.return_topk),
        "oracle_de_gui_early_stop_mse": float(hp_cfg.early_stop_mse),
        "oracle_de_gui_n_fit": int(hp_cfg.n_fit),
        "oracle_de_gui_n_probe": int(hp_cfg.n_probe),
        "oracle_de_gui_torch_num_threads": _default_torch_num_threads(),
        "oracle_de_gui_refine_lbfgs_steps": int(hp_cfg.refine_lbfgs_steps),
        "oracle_de_gui_refine_num_restarts": int(hp_cfg.refine_num_restarts),
        "oracle_de_gui_refine_max_variants": int(hp_cfg.refine_max_variants),
        "oracle_de_gui_refine_max_params": int(hp_cfg.refine_max_params),
        "oracle_de_gui_logs": "",
        "oracle_de_gui_editor_version": 0,
        "oracle_de_gui_report": None,
        "oracle_de_gui_current_best": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "oracle_de_gui_const_rows" not in st.session_state:
        st.session_state.oracle_de_gui_const_rows = []


def _apply_preset_de000() -> None:
    hp_cfg = default_oracle_de_hyperparams()
    st.session_state.oracle_de_gui_spec_id = "de000_simple"
    st.session_state.oracle_de_gui_csv_paths = "data/feynman_de/de000.csv"
    st.session_state.oracle_de_gui_order_candidates = [1]
    st.session_state.oracle_de_gui_x_axis = 0
    st.session_state.oracle_de_gui_include_x = True
    st.session_state.oracle_de_gui_include_u = True
    st.session_state.oracle_de_gui_include_du = True
    st.session_state.oracle_de_gui_x_col = "x0"
    st.session_state.oracle_de_gui_u_col = "y"
    st.session_state.oracle_de_gui_out_idx = 0
    st.session_state.oracle_de_gui_y_transform = "identity"
    st.session_state.oracle_de_gui_deriv_method = "spline"
    st.session_state.oracle_de_gui_spline_s = 0.0
    st.session_state.oracle_de_gui_spline_k = 3
    st.session_state.oracle_de_gui_du_col = ""
    st.session_state.oracle_de_gui_d2u_col = ""
    st.session_state.oracle_de_gui_validate_integrate_topk = 1
    st.session_state.oracle_de_gui_dims_enable = True
    st.session_state.oracle_de_gui_basis = "U,T"
    st.session_state.oracle_de_gui_x_dim = "0,1"
    st.session_state.oracle_de_gui_u_dim = "1,0"
    st.session_state.oracle_de_gui_const_rows = []

    st.session_state.oracle_de_gui_refine_enable = bool(hp_cfg.refine_enable)
    st.session_state.oracle_de_gui_ignore_dims = False
    st.session_state.oracle_de_gui_dtype_name = "float64"
    st.session_state.oracle_de_gui_quiet = False
    st.session_state.oracle_de_gui_seed = int(hp_cfg.seed)
    st.session_state.oracle_de_gui_n_seeds = int(hp_cfg.n_seeds)
    st.session_state.oracle_de_gui_split_iter_across_seeds = bool(hp_cfg.split_iter_across_seeds)
    st.session_state.oracle_de_gui_n_iter = int(hp_cfg.n_iter)
    st.session_state.oracle_de_gui_max_depth = int(hp_cfg.max_depth)
    st.session_state.oracle_de_gui_poly_degree = int(hp_cfg.poly_degree)
    st.session_state.oracle_de_gui_return_topk = int(hp_cfg.return_topk)
    st.session_state.oracle_de_gui_early_stop_mse = float(hp_cfg.early_stop_mse)
    st.session_state.oracle_de_gui_n_fit = int(hp_cfg.n_fit)
    st.session_state.oracle_de_gui_n_probe = int(hp_cfg.n_probe)
    st.session_state.oracle_de_gui_torch_num_threads = _default_torch_num_threads()
    st.session_state.oracle_de_gui_refine_lbfgs_steps = int(hp_cfg.refine_lbfgs_steps)
    st.session_state.oracle_de_gui_refine_num_restarts = int(hp_cfg.refine_num_restarts)
    st.session_state.oracle_de_gui_refine_max_variants = int(hp_cfg.refine_max_variants)
    st.session_state.oracle_de_gui_refine_max_params = int(hp_cfg.refine_max_params)

    st.session_state.oracle_de_gui_report = None
    st.session_state.oracle_de_gui_current_best = None
    st.session_state.oracle_de_gui_logs = "[preset] Loaded DE000 spline preset."
    st.session_state.oracle_de_gui_editor_version = int(st.session_state.oracle_de_gui_editor_version) + 1


def _apply_preset_quick_debug() -> None:
    st.session_state.oracle_de_gui_spec_id = "de_quick_debug"
    st.session_state.oracle_de_gui_csv_paths = "data/feynman_de/de000.csv"
    st.session_state.oracle_de_gui_order_candidates = [1]
    st.session_state.oracle_de_gui_x_axis = 0
    st.session_state.oracle_de_gui_include_x = True
    st.session_state.oracle_de_gui_include_u = True
    st.session_state.oracle_de_gui_include_du = False
    st.session_state.oracle_de_gui_x_col = "x0"
    st.session_state.oracle_de_gui_u_col = "y"
    st.session_state.oracle_de_gui_out_idx = 0
    st.session_state.oracle_de_gui_y_transform = "identity"
    st.session_state.oracle_de_gui_deriv_method = "finite_diff"
    st.session_state.oracle_de_gui_spline_s = 0.0
    st.session_state.oracle_de_gui_spline_k = 3
    st.session_state.oracle_de_gui_du_col = ""
    st.session_state.oracle_de_gui_d2u_col = ""
    st.session_state.oracle_de_gui_validate_integrate_topk = 0
    st.session_state.oracle_de_gui_dims_enable = False
    st.session_state.oracle_de_gui_basis = "U,T"
    st.session_state.oracle_de_gui_x_dim = "0,1"
    st.session_state.oracle_de_gui_u_dim = "1,0"
    st.session_state.oracle_de_gui_const_rows = []

    st.session_state.oracle_de_gui_refine_enable = False
    st.session_state.oracle_de_gui_ignore_dims = True
    st.session_state.oracle_de_gui_dtype_name = "float64"
    st.session_state.oracle_de_gui_quiet = False
    st.session_state.oracle_de_gui_seed = 0
    st.session_state.oracle_de_gui_n_seeds = 1
    st.session_state.oracle_de_gui_split_iter_across_seeds = True
    st.session_state.oracle_de_gui_n_iter = 300
    st.session_state.oracle_de_gui_max_depth = 3
    st.session_state.oracle_de_gui_poly_degree = 3
    st.session_state.oracle_de_gui_return_topk = 3
    st.session_state.oracle_de_gui_early_stop_mse = 1.0e-8
    st.session_state.oracle_de_gui_n_fit = 384
    st.session_state.oracle_de_gui_n_probe = 512
    st.session_state.oracle_de_gui_torch_num_threads = _default_torch_num_threads()
    st.session_state.oracle_de_gui_refine_lbfgs_steps = 15
    st.session_state.oracle_de_gui_refine_num_restarts = 2
    st.session_state.oracle_de_gui_refine_max_variants = 12
    st.session_state.oracle_de_gui_refine_max_params = 8

    st.session_state.oracle_de_gui_report = None
    st.session_state.oracle_de_gui_current_best = None
    st.session_state.oracle_de_gui_logs = "[preset] Loaded quick-debug DE preset."
    st.session_state.oracle_de_gui_editor_version = int(st.session_state.oracle_de_gui_editor_version) + 1


def main() -> None:
    st.set_page_config(page_title="Oracle DE continuous skeleton refinement Lab", layout="wide")
    st.title("Oracle DE continuous skeleton refinement Lab (Streamlit)")
    st.caption("Build DE specs, run factorized symbolic search/continuous skeleton refinement, and inspect per-order discovery logs/results.")
    _ensure_session_defaults()

    p1, p2 = st.columns(2)
    if p1.button("Load DE000 Preset", width="stretch", type="primary"):
        _apply_preset_de000()
    if p2.button("Load Quick Debug Preset", width="stretch"):
        _apply_preset_quick_debug()

    y_names = _y_transform_names()

    with st.sidebar:
        st.header("Search Controls")
        refine_enable = st.checkbox("Enable continuous skeleton refinement", key="oracle_de_gui_refine_enable")
        ignore_dims = st.checkbox("Ignore units filtering", key="oracle_de_gui_ignore_dims")
        dtype_name = st.selectbox("dtype", ["float64", "float32"], key="oracle_de_gui_dtype_name")
        quiet = st.checkbox("Quiet mode", key="oracle_de_gui_quiet")

        seed = st.number_input("Seed", min_value=0, step=1, key="oracle_de_gui_seed")
        n_seeds = st.number_input("n_seeds", min_value=1, step=1, key="oracle_de_gui_n_seeds")
        split_iter_across_seeds = st.checkbox(
            "split_iter_across_seeds",
            key="oracle_de_gui_split_iter_across_seeds",
        )
        n_iter = st.number_input("n_iter", min_value=1, step=100, key="oracle_de_gui_n_iter")
        max_depth = st.number_input("max_depth", min_value=1, step=1, key="oracle_de_gui_max_depth")
        poly_degree = st.number_input("poly_degree", min_value=1, step=1, key="oracle_de_gui_poly_degree")
        return_topk = st.number_input("return_topk", min_value=1, step=1, key="oracle_de_gui_return_topk")
        early_stop_mse = st.number_input(
            "early_stop_mse",
            min_value=0.0,
            step=1.0e-8,
            format="%.3e",
            key="oracle_de_gui_early_stop_mse",
        )
        n_fit = st.number_input("n_fit", min_value=8, step=8, key="oracle_de_gui_n_fit")
        n_probe = st.number_input("n_probe", min_value=8, step=8, key="oracle_de_gui_n_probe")
        torch_num_threads = st.number_input(
            "torch_num_threads",
            min_value=1,
            step=1,
            key="oracle_de_gui_torch_num_threads",
        )

        st.caption("continuous skeleton refinement knobs")
        refine_lbfgs_steps = st.number_input(
            "refine_lbfgs_steps",
            min_value=1,
            step=1,
            key="oracle_de_gui_refine_lbfgs_steps",
        )
        refine_num_restarts = st.number_input(
            "refine_num_restarts",
            min_value=1,
            step=1,
            key="oracle_de_gui_refine_num_restarts",
        )
        refine_max_variants = st.number_input(
            "refine_max_variants",
            min_value=1,
            step=1,
            key="oracle_de_gui_refine_max_variants",
        )
        refine_max_params = st.number_input(
            "refine_max_params",
            min_value=1,
            step=1,
            key="oracle_de_gui_refine_max_params",
        )

    left, right = st.columns([1.2, 1.0])

    with left:
        st.subheader("DE Spec")
        spec_id = st.text_input("Equation ID", key="oracle_de_gui_spec_id")
        csv_paths_raw = st.text_area(
            "CSV path(s) (comma/newline separated)",
            key="oracle_de_gui_csv_paths",
            height=96,
        )
        order_candidates = st.multiselect(
            "Order candidates",
            options=[1, 2],
            key="oracle_de_gui_order_candidates",
        )
        x_axis = st.number_input("x_axis", min_value=0, step=1, key="oracle_de_gui_x_axis")

        c1, c2, c3 = st.columns(3)
        x_col = c1.text_input("x_col", key="oracle_de_gui_x_col")
        u_col = c2.text_input("u_col", key="oracle_de_gui_u_col")
        out_idx = c3.number_input("out_idx", min_value=0, step=1, key="oracle_de_gui_out_idx")

        y_transform = st.selectbox("y_transform", options=y_names, key="oracle_de_gui_y_transform")

        f1, f2, f3 = st.columns(3)
        include_x = f1.checkbox("include_x", key="oracle_de_gui_include_x")
        include_u = f2.checkbox("include_u", key="oracle_de_gui_include_u")
        include_du = f3.checkbox("include_du", key="oracle_de_gui_include_du")

        st.markdown("**Derivative Config**")
        d1, d2, d3 = st.columns(3)
        deriv_method = d1.selectbox(
            "method",
            options=["spline", "finite_diff", "precomputed"],
            key="oracle_de_gui_deriv_method",
        )
        spline_s = float(d2.number_input("spline_s", key="oracle_de_gui_spline_s"))
        spline_k = int(d3.number_input("spline_k", min_value=1, step=1, key="oracle_de_gui_spline_k"))

        p1c, p2c = st.columns(2)
        du_col = p1c.text_input("du_col (precomputed)", key="oracle_de_gui_du_col")
        d2u_col = p2c.text_input("d2u_col (precomputed)", key="oracle_de_gui_d2u_col")

        validate_integrate_topk = st.number_input(
            "validate_integrate_topk",
            min_value=0,
            step=1,
            key="oracle_de_gui_validate_integrate_topk",
        )

        st.markdown("**Constants**")
        editor_version = int(st.session_state.oracle_de_gui_editor_version)
        const_editor = st.data_editor(
            st.session_state.oracle_de_gui_const_rows,
            num_rows="dynamic",
            width="stretch",
            key=f"oracle_de_gui_const_editor_{editor_version}",
        )
        st.session_state.oracle_de_gui_const_rows = _records(const_editor)

        dims_enable = st.checkbox("Include dimensions", key="oracle_de_gui_dims_enable")
        basis_raw = ""
        x_dim_raw = ""
        u_dim_raw = ""
        if dims_enable:
            bd1, bd2, bd3 = st.columns(3)
            basis_raw = bd1.text_input("dims.basis", key="oracle_de_gui_basis")
            x_dim_raw = bd2.text_input("dims.x", key="oracle_de_gui_x_dim")
            u_dim_raw = bd3.text_input("dims.u", key="oracle_de_gui_u_dim")
            st.caption("Use comma-separated exponents in basis order.")

        b1, b2 = st.columns(2)
        validate_clicked = b1.button("Validate Spec", width="stretch")
        run_clicked = b2.button("Run Oracle DE Search", width="stretch", type="primary")

    with right:
        current_best_placeholder = st.empty()
        with current_best_placeholder.container():
            _render_current_best_panel(st.session_state.get("oracle_de_gui_current_best"))

    if validate_clicked or run_clicked:
        log_lines: list[str] = []
        try:
            const_rows = list(st.session_state.oracle_de_gui_const_rows)
            payload = _build_spec_payload(
                spec_id=spec_id,
                csv_paths_raw=csv_paths_raw,
                order_candidates=[int(x) for x in order_candidates],
                x_axis=int(x_axis),
                include_x=bool(include_x),
                include_u=bool(include_u),
                include_du=bool(include_du),
                x_col=x_col,
                u_col=u_col,
                out_idx=int(out_idx),
                y_transform=y_transform,
                deriv_method=deriv_method,
                spline_s=float(spline_s),
                spline_k=int(spline_k),
                du_col=du_col,
                d2u_col=d2u_col,
                validate_integrate_topk=int(validate_integrate_topk),
                constants_rows=const_rows,
                dims_enable=bool(dims_enable),
                basis_raw=basis_raw,
                x_dim_raw=x_dim_raw,
                u_dim_raw=u_dim_raw,
            )
            spec = equation_de_spec_from_dict(payload, source="streamlit-ui")
            log_lines.append(
                f"[spec] id={spec.id} paths={len(spec.csv_paths)} orders={list(spec.order_candidates)} "
                f"x_col={spec.x_col} u_col={spec.u_col} deriv={spec.derivative.method}"
            )

            missing_paths = [
                str(p)
                for p in list(spec.csv_paths)
                if not pathlib.Path(str(p)).exists()
            ]
            if missing_paths:
                raise FileNotFoundError(
                    "Missing CSV path(s): " + ", ".join(missing_paths)
                )
            log_lines.append("[spec] all CSV paths exist.")

            st.subheader("Normalized Spec")
            st.json(payload)
            _render_spec_preview(spec)
            _render_dim_preview(spec)

            if run_clicked:
                hp = default_oracle_de_hyperparams()
                hp.n_iter = int(n_iter)
                hp.max_depth = int(max_depth)
                hp.poly_degree = int(poly_degree)
                hp.return_topk = int(return_topk)
                hp.early_stop_mse = float(early_stop_mse)
                hp.n_fit = int(n_fit)
                hp.n_probe = int(n_probe)
                hp.n_seeds = int(n_seeds)
                hp.split_iter_across_seeds = bool(split_iter_across_seeds)
                hp.refine_enable = bool(refine_enable)
                hp.refine_lbfgs_steps = int(refine_lbfgs_steps)
                hp.refine_num_restarts = int(refine_num_restarts)
                hp.refine_max_variants = int(refine_max_variants)
                hp.refine_max_params = int(refine_max_params)

                dtype = torch.float64 if str(dtype_name).lower() == "float64" else torch.float32
                log_lines.append(
                    f"[run] starting n_iter={hp.n_iter} max_depth={hp.max_depth} "
                    f"return_topk={hp.return_topk} n_seeds={hp.n_seeds} "
                    f"refine_enable={hp.refine_enable} enforce_dims={not bool(ignore_dims)}"
                )

                try:
                    torch.set_num_threads(int(torch_num_threads))
                    log_lines.append(f"[run] torch_num_threads={int(torch.get_num_threads())}")
                except Exception as exc:
                    log_lines.append(f"[run] warning: could not set torch_num_threads ({exc})")

                with st.spinner("Running oracle DE search..."):
                    live_best: dict[str, Any] = {
                        "spec_id": str(spec.id),
                        "expr": "",
                        "mse": float("inf"),
                        "mapping_kind": "",
                        "residual_basins": None,
                        "order": None,
                    }
                    with current_best_placeholder.container():
                        _render_current_best_panel(live_best)

                    def _on_live_line(line: str) -> None:
                        nonlocal live_best
                        if str(line).strip() == "":
                            return
                        live_best, changed = _update_live_best_from_line(
                            str(line),
                            live_best if isinstance(live_best, dict) else None,
                            spec_id=str(spec.id),
                        )
                        if changed:
                            with current_best_placeholder.container():
                                _render_current_best_panel(live_best)

                    report, raw_log = _run_oracle_de_equation_live(
                        spec,
                        factorized_search_hp=hp,
                        seed=int(seed),
                        dtype=dtype,
                        enforce_dims=not bool(ignore_dims),
                        verbose=not bool(quiet),
                        on_line=_on_live_line,
                    )

                log_lines.append(
                    "[run] completed "
                    f"wall_seconds={float(report.get('wall_seconds', float('nan'))):.3f}"
                )

                raw_log = str(raw_log).strip()
                if raw_log:
                    log_lines.append("\n[explorer]\n" + raw_log)

                st.session_state.oracle_de_gui_report = report
                best = report.get("best")
                if isinstance(best, dict):
                    st.session_state.oracle_de_gui_current_best = {
                        "spec_id": str(report.get("spec_id", spec_id)),
                        "expr": str(best.get("expr", "")),
                        "mse": float(best.get("mse", float("nan"))),
                        "mapping_kind": str(best.get("mapping_kind", "")),
                        "residual_basins": _parse_optional_int(live_best.get("residual_basins")),
                        "order": _parse_optional_int(best.get("order")),
                    }
                else:
                    st.session_state.oracle_de_gui_current_best = None
            else:
                st.session_state.oracle_de_gui_report = None
                st.session_state.oracle_de_gui_current_best = None

            st.session_state.oracle_de_gui_logs = "\n".join(log_lines)
        except Exception:
            st.session_state.oracle_de_gui_report = None
            st.session_state.oracle_de_gui_current_best = None
            st.session_state.oracle_de_gui_logs = "\n".join(log_lines + ["", traceback.format_exc()])
            st.error("Spec validation or run failed. See logs for details.")

    with current_best_placeholder.container():
        _render_current_best_panel(st.session_state.get("oracle_de_gui_current_best"))

    report = st.session_state.get("oracle_de_gui_report")
    if isinstance(report, dict):
        _render_results(report)
        st.download_button(
            label="Download Report JSON",
            data=json.dumps(report, indent=2),
            file_name=f"{report.get('spec_id', 'oracle_de_report')}.json",
            mime="application/json",
            width="stretch",
        )

    _render_log_panel(str(st.session_state.get("oracle_de_gui_logs", "")))


if __name__ == "__main__":
    main()
