# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Streamlit GUI for oracle factorized symbolic search/continuous skeleton refinement experiments.

Run with:

    streamlit run examples/oracle_factorized_search/oracle_lab_streamlit.py
"""

from __future__ import annotations

import contextlib
import heapq
import io
import json
import math
import os
import queue
import re
import threading
import time
import traceback
from typing import Any, Callable

import streamlit as st
import torch

from nestynet_sr.sr_search.factorized_search.oracle_lab import (
    build_oracle_dataset,
    compile_target_expression,
    default_oracle_hyperparams,
    equation_spec_from_dict,
    run_oracle_equation,
)
from nestynet_sr.sr_search.config import FactorizedSearchConfig
from nestynet_sr.sr_search.factorized_search.explorer import compute_reachable, dim_round


_RE_BEST_MSE = re.compile(r"best_mse[=\s]+([0-9eE+\-.]+)")
_RE_ITER_BEST_EXPR = re.compile(r"\bbest (.+?) \|")
_RE_NEW_BEST = re.compile(r"NEW BEST .*->\s*(.+?)\s+\(mse [^>]*->\s*([0-9eE+\-.]+)")
_RE_ORACLE_LINE = re.compile(
    r"\[oracle\].*best_mse=([0-9eE+\-.]+)\s+expr=(.*?)\s+mapping=([^\s]+)"
)
_RE_BASINS = re.compile(r"\bresidual_basins(?:=|\s+)(\d+)\b")
_RE_ORACLE_SEED = re.compile(
    r"\[oracle\]\s+seed\s+(\d+)/(\d+)\s+seed_search=(\d+)\s+n_iter=(\d+)"
)
_RE_MUTATE_PROGRESS = re.compile(r"\[mutate\]\s+(\d+)/(\d+)\s+evals,\s+residual_basins=(\d+)")
_RE_MUTATE_DONE = re.compile(r"\[mutate\]\s+done:\s+(\d+)\s+evals,\s+residual_basins=(\d+)")
_RE_METRICS = re.compile(
    r"\[metrics\]\s+iter=(\d+)\s+seed_search=(\d+)\s+residual_basins=(\d+)\s+"
    r"new_residual_basin_rate=([0-9eE+\-.]+)\s+proposed=(\d+)\s+accepted=(\d+)"
)


def _split_csv(raw: str) -> list[str]:
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _is_blank(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return str(v).strip() == ""


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


def _records(editor_output: Any) -> list[dict[str, Any]]:
    if hasattr(editor_output, "to_dict"):
        return list(editor_output.to_dict(orient="records"))
    if isinstance(editor_output, list):
        return [dict(row) for row in editor_output]
    return []


def _parse_basis(raw: str) -> list[str]:
    basis = _split_csv(raw)
    if not basis:
        raise ValueError("Basis cannot be empty. Example: L,T,M,I,Theta")
    return basis


def _rows_to_variables(rows: list[dict[str, Any]], *, n_base: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if _is_blank(row.get("name")) and _is_blank(row.get("lo")) and _is_blank(row.get("hi")):
            continue
        name = str(row.get("name", "")).strip()
        if name == "":
            raise ValueError(f"variables[{i}].name cannot be empty")
        lo = float(row.get("lo"))
        hi = float(row.get("hi"))
        dim = _parse_dim_text(row.get("dim", ""), n_base=n_base, where=f"variables[{i}].dim")
        out.append({"name": name, "bounds": [lo, hi], "dim": dim})
    if not out:
        raise ValueError("At least one variable is required")
    return out


def _build_spec_payload(
    *,
    spec_id: str,
    basis_raw: str,
    target_expr: str,
    target_dim_raw: str,
    variable_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    basis = _parse_basis(basis_raw)
    n_base = len(basis)
    payload = {
        "id": str(spec_id).strip() or "oracle_gui_equation",
        "basis": basis,
        "variables": _rows_to_variables(variable_rows, n_base=n_base),
        "constants": [],
        "target": {
            "expr": str(target_expr).strip(),
            "dim": _parse_dim_text(target_dim_raw, n_base=n_base, where="target.dim"),
        },
    }
    if payload["target"]["expr"] == "":
        raise ValueError("Target expression cannot be empty")
    return payload


def _format_dim(dim: list[float] | tuple[float, ...]) -> str:
    return ",".join(f"{float(x):g}" for x in dim)


def _parse_finite_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        v = float(raw)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _parse_optional_int(raw: Any) -> int | None:
    try:
        if raw is None:
            return None
        v = int(raw)
        return int(v)
    except Exception:
        return None


def _tupleify_expr_ast(node: Any) -> Any:
    if isinstance(node, list):
        return tuple(_tupleify_expr_ast(x) for x in node)
    if isinstance(node, tuple):
        return tuple(_tupleify_expr_ast(x) for x in node)
    return node


def _robust_loss_scale(y: torch.Tensor) -> float:
    yy = y.reshape(-1).detach()
    if yy.numel() == 0:
        return 1.0
    med = torch.median(yy)
    mad = torch.median(torch.abs(yy - med))
    scale = float(mad.item())
    if math.isfinite(scale) and scale > 0.0:
        return scale
    rms = float(torch.sqrt((yy * yy).mean()).item())
    if math.isfinite(rms) and rms > 0.0:
        return rms
    return 1.0


def _canonicalize_expr_string_noiseless(expr: str) -> str:
    """Return a shorter but symbolically identical expression string when possible."""
    s = str(expr).strip()
    if not s:
        return s
    try:
        import sympy as sp

        local_dict = {
            "sqrt": sp.sqrt,
            "sin": sp.sin,
            "cos": sp.cos,
            "exp": sp.exp,
            "log": sp.log,
            "pi": sp.pi,
            "E": sp.E,
        }
        old = sp.sympify(s.replace("^", "**"), locals=local_dict)
        new = sp.simplify(old, ratio=1.5)
        delta = sp.simplify(old - new)
        is_exact = bool(delta == 0) or bool(getattr(delta, "is_zero", False))
        if not is_exact:
            return s
        return str(new).replace("**", "^")
    except Exception:
        return s


def _prune_best_solution(
    report: dict[str, Any],
    ds: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype,
) -> tuple[dict[str, Any] | None, str]:
    best = report.get("best")
    if not isinstance(best, dict):
        return None, "[prune] skipped: no best candidate"

    mapping = best.get("mapping")
    expr_ast = best.get("expr_ast")
    if not isinstance(mapping, dict) or expr_ast is None:
        return None, "[prune] skipped: best candidate has no mapping/expr_ast payload"

    try:
        from torch.utils.data import DataLoader, TensorDataset

        from nestynet_sr.sr_search.factorized_search.bridge import (
            factorized_search_to_nestynet,
            embed_mapping_in_ast,
            promote_argument_const_scales,
        )
        from nestynet_sr.sr_search.config import LMHyperparams
        from nestynet_sr.sr_search.representation import pretty_print_state
        from nestynet_sr.sr_search.stageB.fitting import _fit_candidate_root
        from nestynet_sr.sr_search.stageB.pruning import run_stageb_pruning_pipeline
    except Exception as exc:
        return None, f"[prune] import error: {exc}"

    log_buf = io.StringIO()
    try:
        device = torch.device("cpu")
        x_fit = ds["x_fit"].detach().cpu().to(dtype=dtype)
        y_fit = ds["y_fit"].detach().cpu().to(dtype=dtype)
        x_probe = ds["x_probe"].detach().cpu().to(dtype=dtype)
        y_probe = ds["y_probe"].detach().cpu().to(dtype=dtype)

        train_bs = max(1, int(x_fit.shape[0]))
        val_bs = max(1, int(x_probe.shape[0]))

        train_loader = DataLoader(
            TensorDataset(x_fit, y_fit),
            batch_size=train_bs,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(x_probe, y_probe),
            batch_size=val_bs,
            shuffle=False,
        )

        toy_ast = _tupleify_expr_ast(expr_ast)
        f_ast = factorized_search_to_nestynet(toy_ast)
        nvars_total = int(report.get("nvars_total", int(x_fit.shape[1])))
        input_exprs = [factorized_search_to_nestynet(("var", i)) for i in range(nvars_total)]
        root = embed_mapping_in_ast(
            f_ast,
            dict(mapping),
            input_exprs,
            trainable_dimless=True,
            tag_prefix="oracle_prune",
        )
        if root is None:
            return None, "[prune] skipped: could not embed mapping into AST"
        root = promote_argument_const_scales(root, tag_prefix="oracle_prune")

        lm_hp = LMHyperparams()
        lm_hp.prune_sympy_iter_enable = True
        lm_hp.prune_sympy_iter_max_rounds = 2

        loss_scale = _robust_loss_scale(y_fit)

        with contextlib.redirect_stdout(log_buf):
            base_state = _fit_candidate_root(
                root=root,
                reuse={},
                train_loader=train_loader,
                val_loader=val_loader,
                lm_hp=lm_hp,
                device=device,
                dtype=dtype,
                epochs_stageB=int(lm_hp.epochs),
                loss_scale=loss_scale,
                atom_factory=None,
            )
            before_expr = pretty_print_state(base_state, sig=10)
            before_expr_display = _canonicalize_expr_string_noiseless(str(before_expr))
            before_mse = float(base_state.val_loss)

            pruned_state = run_stageb_pruning_pipeline(
                state=base_state,
                train_loader=train_loader,
                val_loader=val_loader,
                lm_hp=lm_hp,
                device=device,
                dtype=dtype,
                loss_scale=loss_scale,
                atom_factory=None,
                verbose=True,
            )

            after_expr = pretty_print_state(pruned_state, sig=10)
            after_expr_display = _canonicalize_expr_string_noiseless(str(after_expr))
            after_mse = float(pruned_state.val_loss)

        out = {
            "before_expr": str(before_expr_display),
            "after_expr": str(after_expr_display),
            "before_expr_raw": str(before_expr),
            "after_expr_raw": str(after_expr),
            "before_mse": float(before_mse),
            "after_mse": float(after_mse),
            "changed": str(before_expr_display) != str(after_expr_display),
        }
        return out, str(log_buf.getvalue())
    except Exception:
        return None, str(log_buf.getvalue()) + "\n" + traceback.format_exc()


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

    old_expr = str(best.get("expr", ""))
    old_map = str(best.get("mapping_kind", ""))
    old_mse = float(best.get("mse", float("inf")))
    old_residual_basins = _parse_optional_int(best.get("residual_basins"))

    new_expr = old_expr
    new_map = old_map
    new_mse = old_mse
    new_residual_basins = old_residual_basins

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
            # Tie-breaker: fill missing metadata without changing the score.
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
            b_now = int(m_residual_basins.group(1))
            if old_residual_basins is None:
                new_residual_basins = b_now
            else:
                new_residual_basins = max(int(old_residual_basins), b_now)
        except Exception:
            pass

    changed = (
        (new_expr != old_expr)
        or (new_map != old_map)
        or (new_residual_basins != old_residual_basins)
        or (math.isfinite(new_mse) and (not math.isfinite(old_mse) or abs(new_mse - old_mse) > 0.0))
    )

    best["expr"] = new_expr
    best["mse"] = float(new_mse)
    best["mapping_kind"] = new_map
    best["residual_basins"] = new_residual_basins
    return best, changed


def _update_live_progress_from_line(
    line: str,
    current_progress: dict[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    prog = dict(current_progress) if isinstance(current_progress, dict) else {}
    keys = (
        "seed_index",
        "seed_total",
        "seed_search",
        "iteration",
        "n_iter_target",
        "residual_basins",
        "new_residual_basin_rate",
        "proposed",
        "accepted",
        "accept_rate",
    )
    for k in keys:
        prog.setdefault(k, None)
    old = {k: prog.get(k) for k in keys}
    prev_seed_search = _parse_optional_int(prog.get("seed_search"))

    m_seed = _RE_ORACLE_SEED.search(line)
    if m_seed:
        try:
            seed_idx = int(m_seed.group(1))
            seed_tot = int(m_seed.group(2))
            seed_search = int(m_seed.group(3))
            n_iter_target = int(m_seed.group(4))

            prog["seed_index"] = seed_idx
            prog["seed_total"] = seed_tot
            prog["seed_search"] = seed_search
            prog["n_iter_target"] = n_iter_target

            # Seed changed: reset per-seed progress counters immediately.
            if prev_seed_search is None or int(seed_search) != int(prev_seed_search):
                prog["iteration"] = 0
                prog["residual_basins"] = 0
                prog["new_residual_basin_rate"] = None
                prog["proposed"] = 0
                prog["accepted"] = 0
                prog["accept_rate"] = None
        except Exception:
            pass

    m_metrics = _RE_METRICS.search(line)
    if m_metrics:
        try:
            prog["iteration"] = int(m_metrics.group(1))
            prog["seed_search"] = int(m_metrics.group(2))
            prog["residual_basins"] = int(m_metrics.group(3))
            prog["new_residual_basin_rate"] = float(m_metrics.group(4))
            prog["proposed"] = int(m_metrics.group(5))
            prog["accepted"] = int(m_metrics.group(6))
        except Exception:
            pass
    else:
        m_prog = _RE_MUTATE_PROGRESS.search(line)
        if m_prog:
            try:
                prog["iteration"] = int(m_prog.group(1))
                prog["n_iter_target"] = int(m_prog.group(2))
                prog["residual_basins"] = int(m_prog.group(3))
            except Exception:
                pass
        m_done = _RE_MUTATE_DONE.search(line)
        if m_done:
            try:
                prog["iteration"] = int(m_done.group(1))
                prog["residual_basins"] = int(m_done.group(2))
            except Exception:
                pass

    p = _parse_optional_int(prog.get("proposed"))
    a = _parse_optional_int(prog.get("accepted"))
    if p is not None and p > 0 and a is not None:
        prog["accept_rate"] = float(a) / float(p)
    else:
        prog["accept_rate"] = None

    changed = any(prog.get(k) != old.get(k) for k in keys)
    return prog, changed


def _run_oracle_equation_live(
    spec: Any,
    *,
    factorized_search_hp: Any,
    seed: int,
    dtype: torch.dtype,
    enforce_dims: bool,
    on_line: Callable[[str], None],
    on_tick: Callable[[], None] | None = None,
    print_every: int | None = None,
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
                worker_out["report"] = run_oracle_equation(
                    spec,
                    factorized_search_hp=factorized_search_hp,
                    seed=int(seed),
                    dtype=dtype,
                    enforce_dims=bool(enforce_dims),
                    print_every=print_every,
                    verbose=True,
                )
        except Exception:
            worker_err["traceback"] = traceback.format_exc()
        finally:
            q.put(None)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    pending = ""
    while True:
        if on_tick is not None:
            on_tick()
        try:
            item = q.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break
        out_chunks.append(item)
        pending += item
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            on_line(line)

    if pending:
        on_line(pending)
    if on_tick is not None:
        on_tick()

    t.join()

    if "traceback" in worker_err:
        raise RuntimeError(worker_err["traceback"])
    if "report" not in worker_out:
        raise RuntimeError("Live oracle run failed without a report.")
    return dict(worker_out["report"]), "".join(out_chunks)


def _combine_depths(depths: list[int]) -> int:
    if not depths:
        return 1
    if len(depths) == 1:
        return int(depths[0])
    h = [int(x) for x in depths]
    heapq.heapify(h)
    while len(h) > 1:
        a = heapq.heappop(h)
        b = heapq.heappop(h)
        heapq.heappush(h, 1 + max(a, b))
    return int(h[0])


def _pow_depth_pos_int(base_depth: int, exponent: int, memo: dict[int, int]) -> int:
    n = int(exponent)
    if n <= 0:
        return 1
    if n == 1:
        return int(base_depth)
    if n in memo:
        return int(memo[n])
    best = 10**9
    # Unary square shortcut when exponent is even.
    if n % 2 == 0:
        best = min(best, 1 + _pow_depth_pos_int(base_depth, n // 2, memo))
    # General multiplication split: x^n = x^a * x^(n-a)
    for a in range(1, n):
        b = n - a
        da = _pow_depth_pos_int(base_depth, a, memo)
        db = _pow_depth_pos_int(base_depth, b, memo)
        best = min(best, 1 + max(da, db))
    memo[n] = int(best)
    return int(best)


def _estimate_expression_depth(target_expr: str, names: list[str]) -> tuple[int, list[str]]:
    try:
        import sympy as sp
    except Exception as exc:
        raise RuntimeError("sympy is required for depth estimation") from exc

    sym_map = {nm: sp.Symbol(nm, real=True) for nm in names}
    local_scope = {
        **sym_map,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
        "pi": sp.pi,
        "E": sp.E,
    }
    expr = sp.sympify(str(target_expr), locals=local_scope)

    unknown = sorted(str(s) for s in expr.free_symbols if str(s) not in sym_map)
    if unknown:
        raise ValueError(f"Unknown symbols in expression: {unknown}")

    notes: set[str] = set()

    def _rec(e: Any) -> int:
        if getattr(e, "is_number", False):
            return 1
        if isinstance(e, sp.Symbol):
            return 1
        if isinstance(e, sp.Add):
            return _combine_depths([_rec(arg) for arg in e.args])
        if isinstance(e, sp.Mul):
            return _combine_depths([_rec(arg) for arg in e.args])
        if isinstance(e, sp.Pow):
            base, exp = e.args
            db = _rec(base)
            if getattr(exp, "is_number", False):
                if exp == sp.Integer(0):
                    return 1
                if exp == sp.Integer(1):
                    return db
                if exp == sp.Integer(2):
                    return 1 + db
                if exp == sp.Rational(1, 2):
                    return 1 + db
                if getattr(exp, "is_Integer", False):
                    memo: dict[int, int] = {}
                    k = int(exp)
                    if k > 0:
                        return _pow_depth_pos_int(db, k, memo)
                    # x^-k = 1 / x^k
                    return 1 + max(1, _pow_depth_pos_int(db, -k, memo))
                # Generic numeric exponent fallback: exp(c * log(x))
                notes.add("Used exp(c*log(x)) rewrite for non-integer power.")
                d_log = 1 + db
                d_mul = 1 + max(1, d_log)
                return 1 + d_mul
            # Symbolic exponent fallback: exp(g(x)*log(f(x)))
            notes.add("Used exp(g(x)*log(f(x))) rewrite for symbolic power.")
            d_log = 1 + db
            d_exp = _rec(exp)
            d_mul = 1 + max(d_log, d_exp)
            return 1 + d_mul
        if e.func in (sp.sin, sp.cos, sp.exp, sp.log, sp.sqrt, sp.Abs):
            if e.func is sp.Abs:
                notes.add("abs(...) is optimistic here; residual_basin core may need extra handling.")
            return 1 + _rec(e.args[0])
        raise ValueError(f"Unsupported operation in depth estimator: {type(e).__name__}: {e}")

    return int(_rec(expr)), sorted(notes)


def _estimate_units_min_depth(
    var_dims: list[tuple[float, ...]],
    y_dims: tuple[float, ...],
    *,
    probe_depth: int,
) -> int | None:
    if not var_dims:
        return None
    depth = max(1, int(probe_depth))
    reach = compute_reachable(var_dims, depth, target_dim=y_dims)
    tgt = dim_round(tuple(y_dims))
    for d in range(1, len(reach)):
        if tgt in reach[d]:
            return int(d)
    return None


def _render_depth_needed_panel(
    *,
    spec_id: str,
    basis_raw: str,
    target_expr: str,
    target_dim_raw: str,
    variable_rows: list[dict[str, Any]],
    max_depth_cfg: int,
    ignore_dims: bool,
) -> None:
    st.subheader("Depth Needed")
    try:
        payload = _build_spec_payload(
            spec_id=spec_id,
            basis_raw=basis_raw,
            target_expr=target_expr,
            target_dim_raw=target_dim_raw,
            variable_rows=variable_rows,
        )
        spec = equation_spec_from_dict(payload, source="depth-panel")
        all_names = [v.name for v in spec.variables] + [c.name for c in spec.constants]
        expr_depth, notes = _estimate_expression_depth(spec.target_expr, all_names)

        var_dims = [tuple(v.dim) for v in spec.variables] + [tuple(c.dim) for c in spec.constants]
        y_dims = tuple(spec.target_dim)
        # Keep this panel responsive even if configured max_depth is very high.
        probe_depth = max(12, min(int(max_depth_cfg), 24))
        units_min_depth = _estimate_units_min_depth(var_dims, y_dims, probe_depth=probe_depth)

        if ignore_dims:
            recommended = int(expr_depth)
        else:
            recommended = int(max(expr_depth, units_min_depth or 1))
        slack = int(max_depth_cfg) - int(recommended)

        c1, c2, c3 = st.columns(3)
        c1.metric("Expr Depth (est)", str(int(expr_depth)))
        c2.metric("Units Min Depth", "n/a" if units_min_depth is None else str(int(units_min_depth)))
        c3.metric("Recommended max_depth", str(int(recommended)))

        if slack >= 0:
            st.success(
                f"Configured max_depth={int(max_depth_cfg)} has {slack} depth slack for this target."
            )
        else:
            st.warning(
                f"Configured max_depth={int(max_depth_cfg)} is likely too shallow by {-slack} levels."
            )

        if ignore_dims:
            st.caption("Units filter is disabled (`ignore_dims=True`), so recommendation is expression-only.")
        elif units_min_depth is None:
            st.caption(
                f"Units target not found up to depth {probe_depth} in heuristic reachability; this may be a hard case."
            )

        if notes:
            for note in notes:
                st.caption(f"Note: {note}")
    except Exception as exc:
        st.info(f"Depth estimate unavailable until spec is valid: {exc}")


def _render_results(report: dict[str, Any]) -> None:
    st.subheader("Results")
    best = report.get("best")
    if best is None:
        st.warning("No candidate expression found.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Best MSE", f"{float(best.get('mse', float('nan'))):.6g}")
    c2.metric("Wall Time (s)", f"{float(report.get('wall_seconds', float('nan'))):.3f}")
    c3.metric("Candidates", str(len(report.get("results", []))))

    st.code(str(best.get("expr", "")), language="text")
    st.caption(f"Mapping: {best.get('mapping_kind', '')}")

    rows = report.get("results", [])
    if rows:
        safe_rows: list[dict[str, Any]] = []
        for row in rows:
            safe_row: dict[str, Any] = {}
            for key, value in dict(row).items():
                if isinstance(value, (dict, list, tuple)):
                    safe_row[str(key)] = json.dumps(value, ensure_ascii=True)
                elif isinstance(value, (bytes, bytearray)):
                    safe_row[str(key)] = bytes(value).decode("utf-8", errors="replace")
                elif isinstance(value, (str, int, float, bool)) or value is None:
                    safe_row[str(key)] = value
                else:
                    safe_row[str(key)] = str(value)
            safe_rows.append(safe_row)
        st.dataframe(safe_rows, width="stretch")

    action_dist = report.get("action_distribution")
    if isinstance(action_dist, dict):
        counts = action_dist.get("counts", {})
        fractions = action_dist.get("fractions", {})
        total_selected = int(action_dist.get("total_selected", 0))
        if isinstance(counts, dict):
            st.subheader("Action Distribution")
            if total_selected <= 0:
                st.info("No mutation actions were selected (likely solved during brute phase).")
            else:
                rows_ad: list[dict[str, Any]] = []
                for name, cnt in counts.items():
                    try:
                        c = int(cnt)
                    except Exception:
                        continue
                    frac = fractions.get(name, None)
                    if frac is None:
                        f = (float(c) / float(total_selected)) if total_selected > 0 else 0.0
                    else:
                        try:
                            f = float(frac)
                        except Exception:
                            f = 0.0
                    rows_ad.append(
                        {
                            "action": str(name),
                            "count": int(c),
                            "fraction": float(f),
                        }
                    )
                rows_ad.sort(key=lambda r: (-int(r["count"]), str(r["action"])))
                st.caption(f"Selected mutation actions (total={total_selected}).")
                try:
                    import pandas as pd

                    df = pd.DataFrame(rows_ad)
                    if not df.empty:
                        st.bar_chart(df.set_index("action")["count"])
                    st.dataframe(df, width="stretch")
                except Exception:
                    st.dataframe(rows_ad, width="stretch")


def _render_current_best_panel(current_best: dict[str, Any] | None) -> None:
    st.subheader("Current Best Solution")
    if not isinstance(current_best, dict):
        st.info("No best solution yet. Run a search to populate this panel.")
        return

    mse = current_best.get("mse", float("nan"))
    expr = str(current_best.get("expr", ""))
    mapping = str(current_best.get("mapping_kind", ""))
    residual_basins = _parse_optional_int(current_best.get("residual_basins"))

    c1, c2 = st.columns([2, 1])
    c1.metric("Best MSE", f"{float(mse):.6g}" if math.isfinite(float(mse)) else "nan")
    c2.metric("Residual basins", str(int(residual_basins)) if residual_basins is not None else "n/a")
    st.code(expr if expr else "<empty>", language="text")
    st.caption(f"Mapping: {mapping}")


def _render_run_timer_panel(elapsed_seconds: float | None, *, running: bool) -> None:
    st.subheader("Run Timer")
    if elapsed_seconds is None:
        st.info("No run started yet.")
        return
    try:
        sec = float(elapsed_seconds)
    except Exception:
        sec = float("nan")
    if not math.isfinite(sec) or sec < 0.0:
        st.info("Timer unavailable.")
        return
    label = "Elapsed (running)" if running else "Elapsed (last run)"
    st.metric(label, f"{sec:.1f} s")


def _render_run_progress_panel(progress: dict[str, Any] | None, *, running: bool) -> None:
    st.subheader("Run Progress")
    if not isinstance(progress, dict):
        st.info("No run progress yet.")
        return

    seed_idx = _parse_optional_int(progress.get("seed_index"))
    seed_tot = _parse_optional_int(progress.get("seed_total"))
    seed_search = _parse_optional_int(progress.get("seed_search"))
    iteration = _parse_optional_int(progress.get("iteration"))
    n_iter_target = _parse_optional_int(progress.get("n_iter_target"))
    residual_basins = _parse_optional_int(progress.get("residual_basins"))
    proposed = _parse_optional_int(progress.get("proposed"))
    accepted = _parse_optional_int(progress.get("accepted"))
    acc_rate = progress.get("accept_rate")
    new_residual_basin_rate = progress.get("new_residual_basin_rate")

    if seed_idx is not None and seed_tot is not None:
        seed_label = f"{seed_idx}/{seed_tot}"
    elif seed_search is not None:
        seed_label = str(seed_search)
    else:
        seed_label = "n/a"

    if iteration is not None and n_iter_target is not None and n_iter_target > 0:
        iter_label = f"{iteration}/{n_iter_target}"
    elif iteration is not None:
        iter_label = str(iteration)
    else:
        iter_label = "n/a"

    c1, c2, c3 = st.columns(3)
    c1.metric("Seed", seed_label)
    c2.metric("Iteration", iter_label)
    c3.metric("Residual basins", str(residual_basins) if residual_basins is not None else "n/a")

    c4, c5, c6 = st.columns(3)
    if new_residual_basin_rate is not None:
        try:
            nbs = float(new_residual_basin_rate)
            c4.metric("New residual-basin rate", f"{nbs:.4f}/iter")
        except Exception:
            c4.metric("New residual-basin rate", "n/a")
    else:
        c4.metric("New residual-basin rate", "n/a")
    c5.metric("Proposed", str(proposed) if proposed is not None else "n/a")
    c6.metric("Accepted", str(accepted) if accepted is not None else "n/a")

    if acc_rate is not None:
        try:
            st.caption(
                f"Acceptance rate (accepted/proposed): {100.0 * float(acc_rate):.1f}% "
                f"{'(running)' if running else '(last run)'}"
            )
        except Exception:
            pass


def _render_log_panel(log_text: str) -> None:
    st.subheader("Processing Logs")
    if not log_text.strip():
        st.info("No logs captured yet.")
        return
    # Use a non-widget renderer so repeated in-run updates don't create
    # duplicate widget IDs during live log streaming.
    st.code(log_text, language="text")


def _render_final_simplified_panel(pruned: dict[str, Any] | None) -> None:
    st.subheader("Final Simplified Solution")
    if not isinstance(pruned, dict):
        st.info("No simplified solution yet. Run a search first.")
        return

    before_expr = str(pruned.get("before_expr", "")).strip()
    after_expr = str(pruned.get("after_expr", "")).strip()
    before_mse = float(pruned.get("before_mse", float("nan")))
    after_mse = float(pruned.get("after_mse", float("nan")))
    changed = bool(pruned.get("changed", False))

    c1, c2, c3 = st.columns(3)
    c1.metric("Before MSE", f"{before_mse:.6g}" if math.isfinite(before_mse) else "nan")
    c2.metric("Final MSE", f"{after_mse:.6g}" if math.isfinite(after_mse) else "nan")
    c3.metric("Changed", "yes" if changed else "no")

    st.code(after_expr if after_expr else "<empty>", language="text")
    if changed and before_expr and before_expr != after_expr:
        with st.expander("Show pre-simplification expression"):
            st.code(before_expr, language="text")


def _config_default_refine_lbfgs_steps() -> int:
    try:
        return int(FactorizedSearchConfig().refine_lbfgs_steps)
    except Exception:
        return 15


def _config_default_n_seeds() -> int:
    try:
        return int(FactorizedSearchConfig().n_seeds)
    except Exception:
        return 10


def _config_default_split_iter_across_seeds() -> bool:
    try:
        return bool(FactorizedSearchConfig().split_iter_across_seeds)
    except Exception:
        return True


def _default_torch_num_threads() -> int:
    try:
        n = int(torch.get_num_threads())
    except Exception:
        n = int(os.cpu_count() or 1)
    return max(1, n)


def _ensure_session_defaults() -> None:
    dim_default = "0,0,0,0,0"
    hp_cfg = FactorizedSearchConfig()
    refine_lbfgs_steps_default = _config_default_refine_lbfgs_steps()
    n_seeds_default = _config_default_n_seeds()
    split_iter_across_seeds_default = _config_default_split_iter_across_seeds()
    defaults: dict[str, Any] = {
        "oracle_gui_spec_id": "trig_constant_demo",
        "oracle_gui_basis": "L,T,M,I,Theta",
        "oracle_gui_target_expr": "cos(2.7*x0*x1) + 1.2*x0*x1",
        "oracle_gui_target_dim": dim_default,
        "oracle_gui_refine_enable": True,
        "oracle_gui_enable_residual": True,
        "oracle_gui_ignore_dims": False,
        "oracle_gui_dtype_name": "float64",
        "oracle_gui_seed": int(hp_cfg.seed),
        "oracle_gui_n_seeds": n_seeds_default,
        "oracle_gui_split_iter_across_seeds": split_iter_across_seeds_default,
        "oracle_gui_n_iter": int(hp_cfg.n_iter),
        "oracle_gui_max_depth": int(hp_cfg.max_depth),
        "oracle_gui_poly_degree": int(hp_cfg.poly_degree),
        "oracle_gui_early_stop_mse": float(hp_cfg.early_stop_mse),
        "oracle_gui_n_fit": int(hp_cfg.n_fit),
        "oracle_gui_n_probe": int(hp_cfg.n_probe),
        "oracle_gui_refine_lbfgs_steps": refine_lbfgs_steps_default,
        "oracle_gui_refine_num_restarts": int(hp_cfg.refine_num_restarts),
        "oracle_gui_refine_max_variants": int(hp_cfg.refine_max_variants),
        "oracle_gui_refine_max_params": int(hp_cfg.refine_max_params),
        "oracle_gui_torch_num_threads": _default_torch_num_threads(),
        "oracle_gui_logs": "",
        "oracle_gui_last_run_seconds": None,
        "oracle_gui_last_progress": None,
        "oracle_gui_editor_version": 0,
        "oracle_gui_report": None,
        "oracle_gui_current_best": None,
        "oracle_gui_pruned_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "oracle_gui_var_rows" not in st.session_state:
        st.session_state.oracle_gui_var_rows = [
            {"name": "x0", "lo": 0.4, "hi": 3.5, "dim": "1,0,0,0,0"},
            {"name": "x1", "lo": 0.4, "hi": 3.5, "dim": "-1,0,0,0,0"},
        ]


def _apply_preset_hello_world() -> None:
    refine_lbfgs_steps_default = _config_default_refine_lbfgs_steps()
    st.session_state.oracle_gui_spec_id = "hello_world_linear"
    st.session_state.oracle_gui_basis = "L"
    st.session_state.oracle_gui_target_expr = "x0"
    st.session_state.oracle_gui_target_dim = "1"
    st.session_state.oracle_gui_var_rows = [{"name": "x0", "lo": 0.0, "hi": 1.0, "dim": "1"}]
    st.session_state.oracle_gui_refine_enable = False
    st.session_state.oracle_gui_enable_residual = True
    st.session_state.oracle_gui_ignore_dims = False
    st.session_state.oracle_gui_dtype_name = "float64"
    st.session_state.oracle_gui_seed = 0
    st.session_state.oracle_gui_n_seeds = 1
    st.session_state.oracle_gui_split_iter_across_seeds = True
    st.session_state.oracle_gui_n_iter = 200
    st.session_state.oracle_gui_max_depth = 3
    st.session_state.oracle_gui_poly_degree = 3
    st.session_state.oracle_gui_early_stop_mse = 1.0e-12
    st.session_state.oracle_gui_n_fit = 64
    st.session_state.oracle_gui_n_probe = 96
    st.session_state.oracle_gui_refine_lbfgs_steps = refine_lbfgs_steps_default
    st.session_state.oracle_gui_refine_num_restarts = 2
    st.session_state.oracle_gui_refine_max_variants = 12
    st.session_state.oracle_gui_refine_max_params = 8
    st.session_state.oracle_gui_torch_num_threads = _default_torch_num_threads()
    st.session_state.oracle_gui_report = None
    st.session_state.oracle_gui_current_best = None
    st.session_state.oracle_gui_pruned_result = None
    st.session_state.oracle_gui_last_run_seconds = None
    st.session_state.oracle_gui_last_progress = None
    st.session_state.oracle_gui_logs = (
        "[preset] Loaded factorized symbolic search hello-world: target = x0 (uses lightweight overrides)"
    )
    st.session_state.oracle_gui_editor_version = int(st.session_state.oracle_gui_editor_version) + 1


def _apply_preset_trig_demo() -> None:
    hp_cfg = FactorizedSearchConfig()
    refine_lbfgs_steps_default = _config_default_refine_lbfgs_steps()
    n_seeds_default = _config_default_n_seeds()
    split_iter_across_seeds_default = _config_default_split_iter_across_seeds()
    st.session_state.oracle_gui_spec_id = "trig_constant_demo"
    st.session_state.oracle_gui_basis = "L,T,M,I,Theta"
    st.session_state.oracle_gui_target_expr = "cos(2.7*x0*x1) + 1.2*x0*x1"
    st.session_state.oracle_gui_target_dim = "0,0,0,0,0"
    st.session_state.oracle_gui_var_rows = [
        {"name": "x0", "lo": 0.4, "hi": 3.5, "dim": "1,0,0,0,0"},
        {"name": "x1", "lo": 0.4, "hi": 3.5, "dim": "-1,0,0,0,0"},
    ]
    st.session_state.oracle_gui_refine_enable = True
    st.session_state.oracle_gui_enable_residual = True
    st.session_state.oracle_gui_ignore_dims = False
    st.session_state.oracle_gui_dtype_name = "float64"
    st.session_state.oracle_gui_seed = int(hp_cfg.seed)
    st.session_state.oracle_gui_n_seeds = n_seeds_default
    st.session_state.oracle_gui_split_iter_across_seeds = split_iter_across_seeds_default
    st.session_state.oracle_gui_n_iter = int(hp_cfg.n_iter)
    st.session_state.oracle_gui_max_depth = int(hp_cfg.max_depth)
    st.session_state.oracle_gui_poly_degree = int(hp_cfg.poly_degree)
    st.session_state.oracle_gui_early_stop_mse = float(hp_cfg.early_stop_mse)
    st.session_state.oracle_gui_n_fit = int(hp_cfg.n_fit)
    st.session_state.oracle_gui_n_probe = int(hp_cfg.n_probe)
    st.session_state.oracle_gui_refine_lbfgs_steps = refine_lbfgs_steps_default
    st.session_state.oracle_gui_refine_num_restarts = int(hp_cfg.refine_num_restarts)
    st.session_state.oracle_gui_refine_max_variants = int(hp_cfg.refine_max_variants)
    st.session_state.oracle_gui_refine_max_params = int(hp_cfg.refine_max_params)
    st.session_state.oracle_gui_torch_num_threads = _default_torch_num_threads()
    st.session_state.oracle_gui_report = None
    st.session_state.oracle_gui_current_best = None
    st.session_state.oracle_gui_pruned_result = None
    st.session_state.oracle_gui_last_run_seconds = None
    st.session_state.oracle_gui_last_progress = None
    st.session_state.oracle_gui_logs = "[preset] Loaded trig-constant demo with factorized symbolic search config defaults"
    st.session_state.oracle_gui_editor_version = int(st.session_state.oracle_gui_editor_version) + 1


def main() -> None:
    st.set_page_config(page_title="Oracle continuous skeleton refinement Lab", layout="wide")
    st.title("Oracle continuous skeleton refinement Lab (Streamlit)")
    st.caption("Enter equation + units, run factorized symbolic search/continuous skeleton refinement, and inspect processing logs.")
    _ensure_session_defaults()

    p1, p2 = st.columns(2)
    if p1.button("Load factorized symbolic search Hello-World", width="stretch", type="primary"):
        _apply_preset_hello_world()
    if p2.button("Load Trig Demo", width="stretch"):
        _apply_preset_trig_demo()

    with st.sidebar:
        st.header("Search Controls")
        refine_enable = st.checkbox("Enable continuous skeleton refinement", key="oracle_gui_refine_enable")
        enable_residual = st.checkbox("Enable residual action", key="oracle_gui_enable_residual")
        ignore_dims = st.checkbox("Ignore units filtering", key="oracle_gui_ignore_dims")
        dtype_name = st.selectbox("dtype", ["float64", "float32"], key="oracle_gui_dtype_name")

        seed = st.number_input("Seed", min_value=0, step=1, key="oracle_gui_seed")
        n_seeds = st.number_input("n_seeds", min_value=1, step=1, key="oracle_gui_n_seeds")
        split_iter_across_seeds = st.checkbox(
            "split_iter_across_seeds",
            key="oracle_gui_split_iter_across_seeds",
        )
        n_iter = st.number_input("n_iter", min_value=1, step=100, key="oracle_gui_n_iter")
        max_depth = st.number_input("max_depth", min_value=1, step=1, key="oracle_gui_max_depth")
        poly_degree = st.number_input("poly_degree", min_value=1, step=1, key="oracle_gui_poly_degree")
        early_stop_mse = st.number_input(
            "early_stop_mse",
            min_value=0.0,
            step=1.0e-8,
            format="%.3e",
            key="oracle_gui_early_stop_mse",
        )
        n_fit = st.number_input("n_fit", min_value=8, step=8, key="oracle_gui_n_fit")
        n_probe = st.number_input("n_probe", min_value=8, step=8, key="oracle_gui_n_probe")
        torch_num_threads = st.number_input(
            "torch_num_threads",
            min_value=1,
            step=1,
            key="oracle_gui_torch_num_threads",
        )

        st.caption("continuous skeleton refinement knobs")
        refine_lbfgs_steps = st.number_input(
            "refine_lbfgs_steps",
            min_value=1,
            step=1,
            key="oracle_gui_refine_lbfgs_steps",
        )
        refine_num_restarts = st.number_input(
            "refine_num_restarts",
            min_value=1,
            step=1,
            key="oracle_gui_refine_num_restarts",
        )
        refine_max_variants = st.number_input(
            "refine_max_variants",
            min_value=1,
            step=1,
            key="oracle_gui_refine_max_variants",
        )
        refine_max_params = st.number_input(
            "refine_max_params",
            min_value=1,
            step=1,
            key="oracle_gui_refine_max_params",
        )

    left, right = st.columns([1.2, 1.0])

    with left:
        st.subheader("Equation Spec")
        spec_id = st.text_input("Equation ID", key="oracle_gui_spec_id")
        basis_raw = st.text_input("Base dimensions", key="oracle_gui_basis")
        target_expr = st.text_input("Target expression", key="oracle_gui_target_expr")
        target_dim_raw = st.text_input("Target dimension exponents", key="oracle_gui_target_dim")
        st.caption("External constants are disabled in this GUI. Use numeric literals in `target.expr`.")

        st.markdown("**Variables**")
        editor_version = int(st.session_state.oracle_gui_editor_version)
        var_editor = st.data_editor(
            st.session_state.oracle_gui_var_rows,
            num_rows="dynamic",
            width="stretch",
            key=f"oracle_gui_var_editor_{editor_version}",
        )
        st.session_state.oracle_gui_var_rows = _records(var_editor)

        b1, b2 = st.columns(2)
        validate_clicked = b1.button("Validate Spec", width="stretch")
        run_clicked = b2.button("Run Oracle Search", width="stretch", type="primary")

    with right:
        st.subheader("Units Processing")
        st.write("Enter dimensions as comma-separated exponents in basis order.")
        st.code("Example basis: L,T,M,I,Theta\nExample dim: 1,0,-2,0,0", language="text")
        _render_depth_needed_panel(
            spec_id=spec_id,
            basis_raw=basis_raw,
            target_expr=target_expr,
            target_dim_raw=target_dim_raw,
            variable_rows=list(st.session_state.oracle_gui_var_rows),
            max_depth_cfg=int(max_depth),
            ignore_dims=bool(ignore_dims),
        )
        run_timer_placeholder = st.empty()
        with run_timer_placeholder.container():
            _render_run_timer_panel(
                st.session_state.get("oracle_gui_last_run_seconds"),
                running=False,
            )
        run_progress_placeholder = st.empty()
        with run_progress_placeholder.container():
            _render_run_progress_panel(
                st.session_state.get("oracle_gui_last_progress"),
                running=False,
            )
        current_best_placeholder = st.empty()
        with current_best_placeholder.container():
            _render_current_best_panel(st.session_state.get("oracle_gui_current_best"))
            _render_final_simplified_panel(st.session_state.get("oracle_gui_pruned_result"))

    results_placeholder = st.empty()
    download_placeholder = st.empty()
    log_placeholder = st.empty()

    if validate_clicked or run_clicked:
        log_lines: list[str] = []
        run_started_at: float | None = None
        live_progress: dict[str, Any] | None = None

        def _flush_logs() -> None:
            st.session_state.oracle_gui_logs = "\n".join(log_lines)
            with log_placeholder.container():
                _render_log_panel(str(st.session_state.get("oracle_gui_logs", "")))

        def _flush_progress(*, running: bool) -> None:
            st.session_state.oracle_gui_last_progress = (
                dict(live_progress) if isinstance(live_progress, dict) else None
            )
            with run_progress_placeholder.container():
                _render_run_progress_panel(
                    st.session_state.get("oracle_gui_last_progress"),
                    running=bool(running),
                )

        try:
            if run_clicked:
                # Clear stale outputs from any previous run before starting a new one.
                st.session_state.oracle_gui_report = None
                st.session_state.oracle_gui_current_best = None
                st.session_state.oracle_gui_pruned_result = None
                st.session_state.oracle_gui_logs = ""
                st.session_state.oracle_gui_last_progress = None
                with current_best_placeholder.container():
                    _render_current_best_panel(None)
                    _render_final_simplified_panel(None)
                with results_placeholder.container():
                    st.subheader("Results")
                    st.info("Run in progress...")
                with download_placeholder.container():
                    st.empty()
                with log_placeholder.container():
                    _render_log_panel("")
                with run_progress_placeholder.container():
                    _render_run_progress_panel(None, running=True)

            var_rows = list(st.session_state.oracle_gui_var_rows)

            payload = _build_spec_payload(
                spec_id=spec_id,
                basis_raw=basis_raw,
                target_expr=target_expr,
                target_dim_raw=target_dim_raw,
                variable_rows=var_rows,
            )
            spec = equation_spec_from_dict(payload, source="streamlit-ui")
            log_lines.append(
                f"[spec] id={spec.id} basis={list(spec.basis)} "
                f"n_vars={len(spec.variables)} n_consts={len(spec.constants)}"
            )

            target_fn = compile_target_expression(spec)
            log_lines.append("[compile] target expression compiled successfully")

            dtype = torch.float64 if dtype_name == "float64" else torch.float32
            ds = build_oracle_dataset(
                spec,
                target_fn,
                n_fit=int(n_fit),
                n_probe=int(n_probe),
                seed=int(seed),
                dtype=dtype,
            )
            log_lines.append(
                "[dataset] "
                f"x_fit={tuple(ds['x_fit'].shape)} x_probe={tuple(ds['x_probe'].shape)} "
                f"y_fit={tuple(ds['y_fit'].shape)}"
            )
            log_lines.append("[units] variable and target dimensions are valid")

            unit_rows = []
            for v in spec.variables:
                unit_rows.append({"symbol": v.name, "kind": "variable", "dim": _format_dim(list(v.dim))})
            for c in spec.constants:
                unit_rows.append({"symbol": c.name, "kind": "constant", "dim": _format_dim(list(c.dim))})
            unit_rows.append({"symbol": "target", "kind": "output", "dim": _format_dim(list(spec.target_dim))})
            st.dataframe(unit_rows, width="stretch")

            st.subheader("Normalized Spec")
            st.json(payload)

            if run_clicked:
                hp = default_oracle_hyperparams()
                hp.n_iter = int(n_iter)
                hp.max_depth = int(max_depth)
                hp.poly_degree = int(poly_degree)
                hp.early_stop_mse = float(early_stop_mse)
                hp.n_fit = int(n_fit)
                hp.n_probe = int(n_probe)
                hp.n_seeds = int(n_seeds)
                hp.split_iter_across_seeds = bool(split_iter_across_seeds)
                hp.no_residual = not bool(enable_residual)
                hp.refine_enable = bool(refine_enable)
                hp.refine_lbfgs_steps = int(refine_lbfgs_steps)
                hp.refine_num_restarts = int(refine_num_restarts)
                hp.refine_max_variants = int(refine_max_variants)
                hp.refine_max_params = int(refine_max_params)

                log_lines.append(
                    f"[run] starting search n_iter={hp.n_iter} max_depth={hp.max_depth} "
                    f"n_seeds={hp.n_seeds} split_iter_across_seeds={hp.split_iter_across_seeds} "
                    f"enable_residual={bool(enable_residual)} "
                    f"early_stop_mse={float(hp.early_stop_mse):.3e} "
                    f"refine_enable={hp.refine_enable} enforce_dims={not ignore_dims}"
                )
                try:
                    torch.set_num_threads(int(torch_num_threads))
                    log_lines.append(f"[run] torch_num_threads={int(torch.get_num_threads())}")
                except Exception as exc:
                    log_lines.append(f"[run] warning: could not set torch_num_threads ({exc})")
                _flush_logs()

                run_started_at = time.perf_counter()
                st.session_state.oracle_gui_last_run_seconds = 0.0
                live_progress = {
                    "seed_index": None,
                    "seed_total": _parse_optional_int(hp.n_seeds),
                    "seed_search": None,
                    "iteration": 0,
                    "n_iter_target": _parse_optional_int(hp.n_iter),
                    "residual_basins": None,
                    "new_residual_basin_rate": None,
                    "proposed": None,
                    "accepted": None,
                    "accept_rate": None,
                }
                with run_timer_placeholder.container():
                    _render_run_timer_panel(0.0, running=True)
                _flush_progress(running=True)
                last_timer_draw = -1.0

                with st.spinner("Running oracle search..."):
                    st.session_state.oracle_gui_pruned_result = None
                    live_best: dict[str, Any] = {
                        "spec_id": str(spec.id),
                        "expr": "",
                        "mse": float("inf"),
                        "mapping_kind": "",
                        "residual_basins": None,
                    }
                    with current_best_placeholder.container():
                        _render_current_best_panel(live_best)
                        _render_final_simplified_panel(st.session_state.get("oracle_gui_pruned_result"))

                    def _on_live_line(line: str) -> None:
                        nonlocal live_best, live_progress
                        line_s = str(line).rstrip("\n")
                        if line_s.strip() == "":
                            return
                        log_lines.append(line_s)
                        _flush_logs()
                        live_updated, live_changed = _update_live_progress_from_line(
                            line_s,
                            live_progress if isinstance(live_progress, dict) else None,
                        )
                        if live_changed:
                            live_progress = live_updated
                            _flush_progress(running=True)
                        live_best, changed = _update_live_best_from_line(
                            line_s,
                            live_best if isinstance(live_best, dict) else None,
                            spec_id=str(spec.id),
                        )
                        if changed:
                            with current_best_placeholder.container():
                                _render_current_best_panel(live_best)
                                _render_final_simplified_panel(
                                    st.session_state.get("oracle_gui_pruned_result")
                                )

                    def _on_tick() -> None:
                        nonlocal last_timer_draw
                        if run_started_at is None:
                            return
                        elapsed = max(0.0, time.perf_counter() - float(run_started_at))
                        if elapsed - float(last_timer_draw) < 0.2:
                            return
                        last_timer_draw = float(elapsed)
                        with run_timer_placeholder.container():
                            _render_run_timer_panel(elapsed, running=True)

                    report, _raw_log = _run_oracle_equation_live(
                        spec,
                        factorized_search_hp=hp,
                        seed=int(seed),
                        dtype=dtype,
                        enforce_dims=not bool(ignore_dims),
                        on_line=_on_live_line,
                        on_tick=_on_tick,
                        print_every=1000,
                    )

                run_elapsed_seconds = (
                    max(0.0, time.perf_counter() - float(run_started_at))
                    if run_started_at is not None
                    else float("nan")
                )
                if math.isfinite(run_elapsed_seconds):
                    st.session_state.oracle_gui_last_run_seconds = float(run_elapsed_seconds)
                    with run_timer_placeholder.container():
                        _render_run_timer_panel(float(run_elapsed_seconds), running=False)
                _flush_progress(running=False)
                log_lines.append(
                    "[run] completed "
                    f"wall_seconds={float(report.get('wall_seconds', float('nan'))):.3f}"
                )
                if math.isfinite(run_elapsed_seconds):
                    log_lines.append(f"[run] timer_elapsed={float(run_elapsed_seconds):.3f}s")
                _flush_logs()

                st.session_state.oracle_gui_logs = "\n".join(log_lines)
                st.session_state.oracle_gui_report = report
                best = report.get("best")
                if isinstance(best, dict):
                    live_residual_basins = None
                    if isinstance(live_best, dict):
                        live_residual_basins = _parse_optional_int(live_best.get("residual_basins"))
                    st.session_state.oracle_gui_current_best = {
                        "spec_id": str(report.get("spec_id", spec_id)),
                        "expr": str(best.get("expr", "")),
                        "mse": float(best.get("mse", float("nan"))),
                        "mapping_kind": str(best.get("mapping_kind", "")),
                        "residual_basins": live_residual_basins,
                    }
                else:
                    st.session_state.oracle_gui_current_best = None

                if isinstance(best, dict):
                    with st.spinner("Pruning best solution..."):
                        pruned_result, prune_log = _prune_best_solution(
                            report,
                            ds,
                            dtype=dtype,
                        )
                    if str(prune_log).strip():
                        log_lines.append("\n[prune]\n" + str(prune_log).strip())
                    st.session_state.oracle_gui_pruned_result = pruned_result
                    if isinstance(pruned_result, dict):
                        log_lines.append(
                            "[prune] completed "
                            f"changed={bool(pruned_result.get('changed', False))} "
                            f"before_mse={float(pruned_result.get('before_mse', float('nan'))):.6g} "
                            f"after_mse={float(pruned_result.get('after_mse', float('nan'))):.6g}"
                        )
                    else:
                        log_lines.append("[prune] skipped")
                    _flush_logs()
                else:
                    st.session_state.oracle_gui_pruned_result = None
            else:
                _flush_logs()
                st.session_state.oracle_gui_report = None
                st.session_state.oracle_gui_current_best = None
                st.session_state.oracle_gui_pruned_result = None

        except Exception:
            st.session_state.oracle_gui_report = None
            st.session_state.oracle_gui_current_best = None
            st.session_state.oracle_gui_pruned_result = None
            if run_started_at is not None:
                elapsed = max(0.0, time.perf_counter() - float(run_started_at))
                st.session_state.oracle_gui_last_run_seconds = float(elapsed)
                with run_timer_placeholder.container():
                    _render_run_timer_panel(float(elapsed), running=False)
                _flush_progress(running=False)
            st.session_state.oracle_gui_logs = "\n".join(log_lines + ["", traceback.format_exc()])
            with log_placeholder.container():
                _render_log_panel(str(st.session_state.get("oracle_gui_logs", "")))
            st.error("Spec validation or run failed. See logs for details.")

    with current_best_placeholder.container():
        _render_current_best_panel(st.session_state.get("oracle_gui_current_best"))
        _render_final_simplified_panel(st.session_state.get("oracle_gui_pruned_result"))

    report = st.session_state.get("oracle_gui_report")
    with results_placeholder.container():
        if isinstance(report, dict):
            _render_results(report)
    with download_placeholder.container():
        if isinstance(report, dict):
            st.download_button(
                label="Download Report JSON",
                data=json.dumps(report, indent=2),
                file_name=f"{report.get('spec_id', 'oracle_report')}.json",
                mime="application/json",
                width="stretch",
            )

    with log_placeholder.container():
        _render_log_panel(str(st.session_state.get("oracle_gui_logs", "")))


if __name__ == "__main__":
    main()
