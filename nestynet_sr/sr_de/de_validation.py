# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""DE rollout validation helpers.

The benchmark runner used to own RHS reconstruction and trajectory rollout
validation.  This module provides the same primitives from library code so the
DE committee can reuse them without importing the benchmark harness.
"""

from __future__ import annotations

import contextlib
import math
import signal
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from nestynet_sr.sr_search.coe_witness import CoEWitnessExecutor, CoEWitnessJob

from .factorized_de import factorized_search_report_to_rhs_callable, normalized_rmse


@dataclass(frozen=True)
class DEWitnessSpec:
    witness_id: str
    tier: str
    traj_id: str | None
    t0: float | None
    t1: float | None
    horizon_kind: str
    noise_floor: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class DEWitnessResult:
    proposal_id: str
    witness_id: str
    tier: str
    status: str
    residual_mse: float | None
    rollout_nrmse: float | None
    max_abs_error: float | None
    blew_up: bool
    solve_time_s: float
    vote_vs_incumbent: str | None
    failure_kind: str | None
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


VALIDATE_SOLVER_TRIALS: tuple[str, ...] = ("DOP853", "Radau", "BDF", "RK45")


class SimValidateTimeout(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _library_validation_payload(candidate: Mapping[str, Any]) -> dict[str, Any]:
    payload = candidate.get("validation_candidate", None)
    if isinstance(payload, Mapping):
        return dict(payload)
    term_asts_json = candidate.get("term_asts_json", None)
    coeffs = candidate.get("coefficients", None)
    if (
        isinstance(term_asts_json, list)
        and isinstance(coeffs, list)
        and all(not isinstance(c, (list, tuple, dict)) for c in coeffs)
    ):
        return {
            "order": candidate.get("order", -1),
            "x_axis": candidate.get("x_axis", 0),
            "coefficients": coeffs,
            "term_asts_json": term_asts_json,
        }
    raise ValueError("Library candidate does not include a validation-ready shared RHS payload")


def _coeff_map_from_candidate(candidate: Mapping[str, Any]) -> dict[str, float]:
    terms = list(candidate.get("terms", []) or [])
    coeffs = list(candidate.get("coefficients", []) or [])
    out: dict[str, float] = {}
    for term, c in zip(terms, coeffs):
        if isinstance(c, (list, tuple, dict)):
            return {}
        out[str(term)] = float(c)
    return out


def _eval_library_ast_json(
    node: Any,
    *,
    x: float,
    u: float,
    du: float,
    order: int,
    x_axis: int,
) -> float:
    if node is None:
        return 1.0
    if not isinstance(node, Mapping):
        raise TypeError(f"Expected AST node dict, got {type(node).__name__}")

    kind = str(node.get("type", "")).lower()
    if kind == "atom":
        atom_kind = str(node.get("kind", "")).lower()
        kwargs = dict(node.get("kwargs", {}) or {})
        if atom_kind in ("var", "x", "input"):
            var_idxs = list(node.get("var_idxs", []) or [])
            if len(var_idxs) != 1:
                raise ValueError(f"Expected one var index, got {var_idxs!r}")
            idx = int(var_idxs[0])
            if idx != int(x_axis):
                raise ValueError(f"Unsupported x-axis index {idx}; expected {x_axis}")
            return float(x)
        if atom_kind in ("u", "field", "state"):
            return float(u)
        if atom_kind in ("du", "d1u", "grad_u"):
            axis = int(kwargs.get("axis", x_axis))
            if int(order) == 1:
                raise ValueError("Encountered du term in first-order explicit RHS validation")
            if axis != int(x_axis):
                raise ValueError(f"Unsupported du axis {axis}; expected {x_axis}")
            return float(du)
        if atom_kind in ("d2u", "ddu", "hess_u"):
            raise ValueError("Encountered d2u term in explicit ODE RHS validation")
        if atom_kind in ("const", "constant"):
            return float(kwargs.get("value", 1.0))
        if atom_kind in ("free_const", "freeconst", "free_constant", "scale"):
            return float(kwargs.get("init", 1.0))
        if atom_kind in ("fixed_const", "fixedconst", "fixed_constant"):
            return float(kwargs.get("value", 1.0))
        raise ValueError(f"Unsupported atom kind in validation AST: {atom_kind!r}")
    if kind == "const":
        return float(node.get("value", 0.0))
    if kind == "add":
        return _eval_library_ast_json(node.get("left"), x=x, u=u, du=du, order=order, x_axis=x_axis) + _eval_library_ast_json(
            node.get("right"), x=x, u=u, du=du, order=order, x_axis=x_axis
        )
    if kind == "mul":
        return _eval_library_ast_json(node.get("left"), x=x, u=u, du=du, order=order, x_axis=x_axis) * _eval_library_ast_json(
            node.get("right"), x=x, u=u, du=du, order=order, x_axis=x_axis
        )
    if kind == "pow":
        base = _eval_library_ast_json(node.get("base"), x=x, u=u, du=du, order=order, x_axis=x_axis)
        exponent = node.get("exponent", 1.0)
        if isinstance(exponent, Mapping):
            exponent = _eval_library_ast_json(exponent, x=x, u=u, du=du, order=order, x_axis=x_axis)
        with np.errstate(all="ignore"):
            return float(np.power(base, float(exponent)))
    if kind == "log":
        with np.errstate(all="ignore"):
            return float(np.log(_eval_library_ast_json(node.get("arg"), x=x, u=u, du=du, order=order, x_axis=x_axis)))
    if kind == "exp":
        with np.errstate(all="ignore"):
            return float(np.exp(_eval_library_ast_json(node.get("arg"), x=x, u=u, du=du, order=order, x_axis=x_axis)))
    if kind == "sin":
        return float(np.sin(_eval_library_ast_json(node.get("arg"), x=x, u=u, du=du, order=order, x_axis=x_axis)))
    if kind == "cos":
        return float(np.cos(_eval_library_ast_json(node.get("arg"), x=x, u=u, du=du, order=order, x_axis=x_axis)))
    if kind == "conj":
        return float(np.real(np.conjugate(_eval_library_ast_json(node.get("arg"), x=x, u=u, du=du, order=order, x_axis=x_axis))))
    if kind == "real":
        return float(np.real(_eval_library_ast_json(node.get("arg"), x=x, u=u, du=du, order=order, x_axis=x_axis)))
    if kind == "imag":
        return float(np.imag(_eval_library_ast_json(node.get("arg"), x=x, u=u, du=du, order=order, x_axis=x_axis)))
    if kind == "abs":
        return float(np.abs(_eval_library_ast_json(node.get("arg"), x=x, u=u, du=du, order=order, x_axis=x_axis)))
    if kind == "arg":
        return float(np.angle(_eval_library_ast_json(node.get("arg"), x=x, u=u, du=du, order=order, x_axis=x_axis)))
    raise ValueError(f"Unsupported AST node type in validation payload: {kind!r}")


def library_candidate_to_rhs_callable(candidate: Mapping[str, Any]) -> tuple[int, Any]:
    payload = _library_validation_payload(candidate)
    order = int(payload.get("order", candidate.get("order", -1)))
    x_axis = int(payload.get("x_axis", candidate.get("x_axis", 0)))
    coeffs = [float(c) for c in list(payload.get("coefficients", []) or [])]
    term_asts_json = list(payload.get("term_asts_json", []) or [])
    if len(coeffs) != len(term_asts_json):
        raise ValueError(
            f"Coefficient/term mismatch in library validation payload: {len(coeffs)} vs {len(term_asts_json)}"
        )

    def _predict_rhs(x: float, u: float, du: float) -> float:
        total = 0.0
        for coeff, term_ast in zip(coeffs, term_asts_json):
            term_val = _eval_library_ast_json(
                term_ast,
                x=float(x),
                u=float(u),
                du=float(du),
                order=int(order),
                x_axis=int(x_axis),
            )
            total += float(coeff) * float(term_val)
        return -float(total)

    if int(order) == 1:
        def rhs1(x: float, s: Sequence[float]) -> list[float]:
            return [_predict_rhs(float(x), float(s[0]), 0.0)]

        return 1, rhs1

    if int(order) != 2:
        raise ValueError(f"Unsupported discovered order: {order}")

    def rhs2(x: float, s: Sequence[float]) -> list[float]:
        u = float(s[0])
        du = float(s[1])
        return [du, _predict_rhs(float(x), u, du)]

    return 2, rhs2


def candidate_to_rhs_callable(
    candidate: Mapping[str, Any],
    *,
    engine: str | None = None,
) -> tuple[int, Any]:
    engine_s = str(engine or candidate.get("engine", "") or "").strip().lower()
    if engine_s in ("factorized_search", "whole_rhs_fss", "factorized_search_only"):
        return factorized_search_report_to_rhs_callable(dict(candidate))
    return library_candidate_to_rhs_callable(candidate)


@contextlib.contextmanager
def walltime_budget(seconds: float | None):
    """Best-effort wall-time limiter for solve_ivp.

    Uses SIGALRM/ITIMER_REAL when available (POSIX). On platforms without
    SIGALRM, this becomes a no-op.
    """

    if seconds is None:
        yield
        return
    try:
        sec_f = float(seconds)
    except Exception:
        sec_f = 0.0
    if sec_f <= 0.0 or not math.isfinite(sec_f):
        yield
        return

    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum: int, frame: Any) -> None:  # pragma: no cover
        raise SimValidateTimeout(f"solve_ivp exceeded {sec_f:.3g}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, sec_f)
        yield
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        except Exception:
            pass
        try:
            signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            pass


def witness_spec_from_run(
    run: Any,
    *,
    tier: str = "short_rollout",
    horizon_kind: str = "prefix_window",
    noise_floor: float = 0.0,
    rollout_window_fraction: float | None = 0.25,
    rollout_max_span: float | None = 5.0,
) -> DEWitnessSpec:
    csv_path = Path(getattr(run, "csv_path"))
    traj_id = str(getattr(run, "traj_id", csv_path.stem))
    t0 = None
    t1 = None
    full_t1 = None
    n_points_full = 0
    try:
        data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)
        if data.ndim == 2 and data.shape[1] >= 2 and data.shape[0] > 0:
            x = np.asarray(data[:, 1], dtype=np.float64)
            n_points_full = int(x.size)
            t0 = float(x[0])
            full_t1 = float(x[-1])
            t1 = full_t1
            full_span = abs(float(full_t1) - float(t0))
            span = full_span
            if rollout_window_fraction is not None and float(rollout_window_fraction) > 0.0:
                span = min(span, full_span * float(rollout_window_fraction))
            if rollout_max_span is not None and float(rollout_max_span) > 0.0:
                span = min(span, float(rollout_max_span))
            if span > 0.0 and span < full_span:
                sign = 1.0 if full_t1 >= t0 else -1.0
                t1 = float(t0) + sign * float(span)
    except Exception:
        pass
    return DEWitnessSpec(
        witness_id=f"{tier}:{traj_id}",
        tier=str(tier),
        traj_id=traj_id,
        t0=t0,
        t1=t1,
        horizon_kind=str(horizon_kind),
        noise_floor=float(noise_floor),
        metadata={
            "csv_path": str(csv_path),
            "u0": _safe_float(getattr(run, "u0", 0.0), 0.0),
            "v0": _safe_float(getattr(run, "v0", 0.0), 0.0),
            "full_t1": full_t1,
            "n_points_full": int(n_points_full),
            "rollout_window_fraction": None if rollout_window_fraction is None else float(rollout_window_fraction),
            "rollout_max_span": None if rollout_max_span is None else float(rollout_max_span),
        },
    )


def witness_specs_from_runs(
    runs: Iterable[Any],
    *,
    tier: str = "short_rollout",
    horizon_kind: str = "prefix_window",
    noise_floor: float = 0.0,
    rollout_window_fraction: float | None = 0.25,
    rollout_max_span: float | None = 5.0,
) -> list[DEWitnessSpec]:
    return [
        witness_spec_from_run(
            run,
            tier=tier,
            horizon_kind=horizon_kind,
            noise_floor=float(noise_floor),
            rollout_window_fraction=rollout_window_fraction,
            rollout_max_span=rollout_max_span,
        )
        for run in runs
    ]


def _traj_score_from_witness_result(result: DEWitnessResult) -> dict[str, Any]:
    row = {
        "traj_id": str(result.metrics.get("traj_id", result.witness_id)),
        "nrmse": float(result.rollout_nrmse) if result.rollout_nrmse is not None else float("inf"),
        "n_points": _safe_int(result.metrics.get("n_points", 0), 0),
        "method": str(result.metrics.get("method", "?")),
    }
    for key in ("horizon_kind", "t0", "t1", "full_t1", "n_points_full"):
        if key in result.metrics:
            row[key] = result.metrics[key]
    if result.metrics.get("nfev", None) is not None:
        row["nfev"] = _safe_int(result.metrics.get("nfev", -1), -1)
    if result.failure_kind or result.status in ("FAIL", "ERROR"):
        row["error"] = str(result.metrics.get("error", result.failure_kind or result.status))
    return row


def _evaluate_rollout_witness(
    spec: DEWitnessSpec,
    *,
    rhs_fn: Any,
    order: int,
    proposal_id: str,
    methods: Sequence[str],
    rtol: float,
    atol: float,
    traj_time_budget_s: float | None,
    blowup_factor: float,
    blowup_abs: float,
) -> DEWitnessResult:
    t_start = time.time()
    csv_path = Path(str(spec.metadata.get("csv_path", "")))
    traj_id = str(spec.traj_id or csv_path.stem)
    try:
        data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)
        if data.ndim != 2 or data.shape[1] < 2:
            return DEWitnessResult(
                proposal_id=proposal_id,
                witness_id=str(spec.witness_id),
                tier=str(spec.tier),
                status="FAIL",
                residual_mse=None,
                rollout_nrmse=float("inf"),
                max_abs_error=None,
                blew_up=False,
                solve_time_s=float(time.time() - t_start),
                vote_vs_incumbent=None,
                failure_kind="invalid_csv",
                metrics={"traj_id": traj_id, "error": f"invalid CSV shape for {csv_path.name}"},
            )
        u_true = np.asarray(data[:, 0], dtype=np.float64)
        x = np.asarray(data[:, 1], dtype=np.float64)
        n_points_full = int(x.size)
        if spec.t1 is not None and x.size > 1:
            t1 = float(spec.t1)
            eps = max(1.0e-12, 1.0e-12 * abs(t1))
            if float(x[-1]) >= float(x[0]):
                mask = x <= t1 + eps
            else:
                mask = x >= t1 - eps
            if int(mask.sum()) >= 2:
                x = x[mask]
                u_true = u_true[mask]
        y0 = [float(spec.metadata.get("u0", 0.0))] if int(order) == 1 else [
            float(spec.metadata.get("u0", 0.0)),
            float(spec.metadata.get("v0", 0.0)),
        ]

        umax = float(np.max(np.abs(u_true))) if u_true.size else 0.0
        blowup_thr = float(max(float(blowup_abs), float(blowup_factor) * umax))

        def _blowup_event(t: float, y: np.ndarray) -> float:
            try:
                return blowup_thr - abs(float(y[0]))
            except Exception:
                return -1.0

        _blowup_event.terminal = True  # type: ignore[attr-defined]
        _blowup_event.direction = -1  # type: ignore[attr-defined]

        sol = None
        last_err: str | None = None
        used_method: str | None = None
        t_budget0 = time.time()
        for method in methods:
            remaining = None
            if traj_time_budget_s is not None:
                elapsed = time.time() - t_budget0
                remaining = float(traj_time_budget_s) - float(elapsed)
                if remaining <= 0.0:
                    last_err = f"timeout (budget {float(traj_time_budget_s):.3g}s)"
                    sol = None
                    break
            try:
                with walltime_budget(remaining):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        sol = solve_ivp(
                            rhs_fn,
                            [float(x[0]), float(x[-1])],
                            y0,
                            t_eval=x,
                            method=method,
                            rtol=rtol,
                            atol=atol,
                            events=_blowup_event,
                        )
                used_method = str(method)
                if sol.success and getattr(sol, "y", None) is not None and sol.y.shape[1] == x.size:
                    break
                if sol.success and sol.y.shape[1] != x.size:
                    last_err = f"terminated early (blow-up > {blowup_thr:.3g})"
                    sol = None
                    continue
                last_err = sol.message if sol is not None else "integration failed"
            except Exception as exc:
                last_err = str(exc)
                sol = None

        if sol is None or not sol.success:
            err_msg = last_err or (sol.message if sol is not None else "all solvers failed")
            failure_kind = _failure_kind_from_text(err_msg)
            return DEWitnessResult(
                proposal_id=proposal_id,
                witness_id=str(spec.witness_id),
                tier=str(spec.tier),
                status="FAIL",
                residual_mse=None,
                rollout_nrmse=float("inf"),
                max_abs_error=None,
                blew_up=failure_kind == "blowup",
                solve_time_s=float(time.time() - t_start),
                vote_vs_incumbent=None,
                failure_kind=failure_kind,
                metrics={
                    "traj_id": traj_id,
                    "n_points": int(x.size),
                    "n_points_full": int(n_points_full),
                    "horizon_kind": str(spec.horizon_kind),
                    "t0": float(x[0]) if x.size else None,
                    "t1": float(x[-1]) if x.size else None,
                    "full_t1": spec.metadata.get("full_t1", None),
                    "method": used_method or "?",
                    "error": str(err_msg),
                },
            )

        u_pred = np.asarray(sol.y[0], dtype=np.float64)
        nrmse = float(normalized_rmse(u_true, u_pred))
        max_abs_error = float(np.max(np.abs(u_true - u_pred))) if u_true.size else 0.0
        return DEWitnessResult(
            proposal_id=proposal_id,
            witness_id=str(spec.witness_id),
            tier=str(spec.tier),
            status="PASS",
            residual_mse=None,
            rollout_nrmse=nrmse,
            max_abs_error=max_abs_error,
            blew_up=False,
            solve_time_s=float(time.time() - t_start),
            vote_vs_incumbent=None,
            failure_kind=None,
            metrics={
                "traj_id": traj_id,
                "n_points": int(x.size),
                "n_points_full": int(n_points_full),
                "horizon_kind": str(spec.horizon_kind),
                "t0": float(x[0]) if x.size else None,
                "t1": float(x[-1]) if x.size else None,
                "full_t1": spec.metadata.get("full_t1", None),
                "method": used_method or "?",
                "nfev": int(getattr(sol, "nfev", -1)),
            },
        )
    except Exception as exc:
        return DEWitnessResult(
            proposal_id=proposal_id,
            witness_id=str(spec.witness_id),
            tier=str(spec.tier),
            status="ERROR",
            residual_mse=None,
            rollout_nrmse=float("inf"),
            max_abs_error=None,
            blew_up=False,
            solve_time_s=float(time.time() - t_start),
            vote_vs_incumbent=None,
            failure_kind="pipeline_error",
            metrics={"traj_id": traj_id, "error": str(exc)},
        )


def _failure_kind_from_text(text: str | None) -> str:
    blob = str(text or "").lower()
    if "timeout (budget" in blob or "exceeded" in blob:
        return "timeout"
    if "terminated early (blow-up" in blob or "blow-up" in blob:
        return "blowup"
    if "non-finite" in blob:
        return "nonfinite_candidate"
    return "integration_failure"


def evaluate_compile_domain_witness(
    candidate: Mapping[str, Any],
    *,
    engine: str | None = None,
    proposal_id: str = "",
    witness_id: str = "compile_domain",
    domain_samples: Sequence[Mapping[str, Any] | Sequence[float]] | None = None,
) -> DEWitnessResult:
    t_start = time.time()
    engine_s = str(engine or candidate.get("engine", "") or "")
    try:
        order, rhs_fn = candidate_to_rhs_callable(candidate, engine=engine_s)
        for idx, sample in enumerate(list(domain_samples or [{"x": 0.0, "u": 1.0, "du": 0.0}])):
            if isinstance(sample, Mapping):
                x = float(sample.get("x", 0.0))
                u = float(sample.get("u", 1.0))
                du = float(sample.get("du", 0.0))
            else:
                vals = list(sample)
                x = float(vals[0]) if len(vals) > 0 else 0.0
                u = float(vals[1]) if len(vals) > 1 else 1.0
                du = float(vals[2]) if len(vals) > 2 else 0.0
            state = [u] if int(order) == 1 else [u, du]
            value = rhs_fn(x, state)
            arr = np.asarray(value, dtype=np.float64)
            if not np.isfinite(arr).all():
                return DEWitnessResult(
                    proposal_id=str(proposal_id),
                    witness_id=str(witness_id),
                    tier="compile_domain",
                    status="FAIL",
                    residual_mse=None,
                    rollout_nrmse=None,
                    max_abs_error=None,
                    blew_up=False,
                    solve_time_s=float(time.time() - t_start),
                    vote_vs_incumbent=None,
                    failure_kind="nonfinite_candidate",
                    metrics={"engine": engine_s, "sample_index": int(idx), "value": _jsonable(arr)},
                )
        return DEWitnessResult(
            proposal_id=str(proposal_id),
            witness_id=str(witness_id),
            tier="compile_domain",
            status="PASS",
            residual_mse=None,
            rollout_nrmse=None,
            max_abs_error=None,
            blew_up=False,
            solve_time_s=float(time.time() - t_start),
            vote_vs_incumbent=None,
            failure_kind=None,
            metrics={"engine": engine_s, "order": int(order), "n_samples": len(list(domain_samples or [None]))},
        )
    except Exception as exc:
        return DEWitnessResult(
            proposal_id=str(proposal_id),
            witness_id=str(witness_id),
            tier="compile_domain",
            status="ERROR",
            residual_mse=None,
            rollout_nrmse=None,
            max_abs_error=None,
            blew_up=False,
            solve_time_s=float(time.time() - t_start),
            vote_vs_incumbent=None,
            failure_kind="compile_error",
            metrics={"engine": engine_s, "error": str(exc)},
        )


def run_rollout_witnesses(
    specs: Sequence[DEWitnessSpec],
    *,
    rhs_fn: Any,
    order: int,
    proposal_id: str = "",
    methods: Sequence[str] = VALIDATE_SOLVER_TRIALS,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
    traj_time_budget_s: float | None = 20.0,
    blowup_factor: float = 100.0,
    blowup_abs: float = 1.0e6,
    executor: CoEWitnessExecutor | None = None,
) -> list[DEWitnessResult]:
    executor = executor or CoEWitnessExecutor(parallelism=1)
    jobs = [
        CoEWitnessJob(
            job_id=f"de_rollout:{idx}",
            slice_id=idx,
            payload=spec,
            metadata={"witness_id": spec.witness_id, "tier": spec.tier},
        )
        for idx, spec in enumerate(specs)
    ]

    def _worker(job: CoEWitnessJob) -> dict[str, Any]:
        result = _evaluate_rollout_witness(
            job.payload,
            rhs_fn=rhs_fn,
            order=int(order),
            proposal_id=str(proposal_id),
            methods=methods,
            rtol=float(rtol),
            atol=float(atol),
            traj_time_budget_s=traj_time_budget_s,
            blowup_factor=float(blowup_factor),
            blowup_abs=float(blowup_abs),
        )
        return result.to_dict()

    rows = executor.run(jobs, _worker)
    return [
        DEWitnessResult(
            proposal_id=str(row.get("proposal_id", proposal_id)),
            witness_id=str(row.get("witness_id", "")),
            tier=str(row.get("tier", "")),
            status=str(row.get("status", "ERROR")),
            residual_mse=row.get("residual_mse", None),
            rollout_nrmse=row.get("rollout_nrmse", None),
            max_abs_error=row.get("max_abs_error", None),
            blew_up=bool(row.get("blew_up", False)),
            solve_time_s=float(row.get("solve_time_s", 0.0) or 0.0),
            vote_vs_incumbent=row.get("vote_vs_incumbent", None),
            failure_kind=row.get("failure_kind", None),
            metrics=dict(row.get("metrics", {}) or {}),
        )
        for row in rows
    ]


def validate_by_simulation(
    runs: Sequence[Any],
    *,
    rhs_fn: Any,
    order: int,
    pass_nrmse: float,
    partial_nrmse: float,
    methods: Sequence[str] = VALIDATE_SOLVER_TRIALS,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
    traj_time_budget_s: float | None = 20.0,
    blowup_factor: float = 100.0,
    blowup_abs: float = 1.0e6,
    executor: CoEWitnessExecutor | None = None,
    rollout_window_fraction: float | None = 0.25,
    rollout_max_span: float | None = 5.0,
) -> tuple[str, str, list[dict[str, Any]]]:
    specs = witness_specs_from_runs(
        runs,
        tier="short_rollout",
        horizon_kind="prefix_window",
        rollout_window_fraction=rollout_window_fraction,
        rollout_max_span=rollout_max_span,
    )
    if executor is not None:
        witness_results = run_rollout_witnesses(
            specs,
            rhs_fn=rhs_fn,
            order=int(order),
            methods=methods,
            rtol=float(rtol),
            atol=float(atol),
            traj_time_budget_s=traj_time_budget_s,
            blowup_factor=float(blowup_factor),
            blowup_abs=float(blowup_abs),
            executor=executor,
        )
    else:
        witness_results = [
            _evaluate_rollout_witness(
                spec,
                rhs_fn=rhs_fn,
                order=int(order),
                proposal_id="",
                methods=methods,
                rtol=float(rtol),
                atol=float(atol),
                traj_time_budget_s=traj_time_budget_s,
                blowup_factor=float(blowup_factor),
                blowup_abs=float(blowup_abs),
            )
            for spec in specs
        ]

    traj_scores = [_traj_score_from_witness_result(result) for result in witness_results]
    for result in witness_results:
        if result.status in ("FAIL", "ERROR"):
            err_msg = str(result.metrics.get("error", result.failure_kind or "all solvers failed"))
            return "FAIL", f"Integration failed on {result.metrics.get('traj_id', result.witness_id)}: {err_msg}", traj_scores

    if not traj_scores:
        return "FAIL", "No trajectories available for simulation validation", traj_scores

    mean_e = float(sum(t["nrmse"] for t in traj_scores) / len(traj_scores))
    max_e = float(max(t["nrmse"] for t in traj_scores))
    msg = f"NRMSE mean={mean_e:.3g} max={max_e:.3g}"

    if max_e < float(pass_nrmse):
        return "PASS", msg, traj_scores
    if max_e < float(partial_nrmse):
        return "PARTIAL", msg, traj_scores
    return "FAIL", msg, traj_scores


def _candidate_rank(row: Mapping[str, Any]) -> int:
    for key in ("candidate_rank", "shortlist_rank", "rerank_rank", "original_rank"):
        try:
            if key in row and row.get(key) is not None:
                return int(row.get(key))
        except Exception:
            continue
    return 10**9


def evaluate_library_candidate_rollout(
    candidate: Mapping[str, Any],
    *,
    probe_runs: Sequence[Any],
    pass_nrmse: float,
    partial_nrmse: float,
    sim_validate_traj_time_budget_s: float | None,
    sim_validate_blowup_factor: float,
    sim_validate_blowup_abs: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "engine": "stlsq",
        "canonical_equation": str(candidate.get("canonical_equation", "")),
        "coeff_map": _coeff_map_from_candidate(candidate),
    }
    try:
        order, rhs_fn = library_candidate_to_rhs_callable(candidate)
    except Exception as exc:
        out["status"] = "ERROR"
        out["message"] = f"Could not build sparse RHS: {exc}"
        out["traj_scores"] = []
        out["discovered_order"] = int(candidate.get("order", -1))
        return out

    status, message, traj_scores = validate_by_simulation(
        probe_runs,
        rhs_fn=rhs_fn,
        order=int(order),
        pass_nrmse=float(pass_nrmse),
        partial_nrmse=float(partial_nrmse),
        traj_time_budget_s=sim_validate_traj_time_budget_s,
        blowup_factor=float(sim_validate_blowup_factor),
        blowup_abs=float(sim_validate_blowup_abs),
    )
    out["status"] = str(status)
    out["message"] = str(message)
    out["traj_scores"] = traj_scores
    out["discovered_order"] = int(order)
    return out


def evaluate_factorized_search_candidate_rollout(
    candidate: Mapping[str, Any],
    *,
    probe_runs: Sequence[Any],
    pass_nrmse: float,
    partial_nrmse: float,
    sim_validate_traj_time_budget_s: float | None,
    sim_validate_blowup_factor: float,
    sim_validate_blowup_abs: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "engine": "factorized_search",
        "canonical_equation": str(candidate.get("canonical_equation", "")),
        "candidate_rank": _candidate_rank(candidate),
    }
    try:
        order, rhs_fn = factorized_search_report_to_rhs_callable(dict(candidate))
    except Exception as exc:
        out["status"] = "ERROR"
        out["message"] = f"Could not build factorized symbolic search RHS: {exc}"
        out["traj_scores"] = []
        out["discovered_order"] = int(candidate.get("order", -1))
        return out

    status, message, traj_scores = validate_by_simulation(
        probe_runs,
        rhs_fn=rhs_fn,
        order=int(order),
        pass_nrmse=float(pass_nrmse),
        partial_nrmse=float(partial_nrmse),
        traj_time_budget_s=sim_validate_traj_time_budget_s,
        blowup_factor=float(sim_validate_blowup_factor),
        blowup_abs=float(sim_validate_blowup_abs),
    )
    out["status"] = str(status)
    out["message"] = str(message)
    out["traj_scores"] = traj_scores
    out["discovered_order"] = int(order)
    return out


def evaluate_factorized_candidate_rollout(
    candidate: Mapping[str, Any],
    *,
    probe_runs: Sequence[Any],
    pass_nrmse: float,
    partial_nrmse: float,
    sim_validate_traj_time_budget_s: float | None,
    sim_validate_blowup_factor: float,
    sim_validate_blowup_abs: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "engine": "factorized",
        "canonical_equation": str(candidate.get("canonical_equation", "")),
        "candidate_rank": _candidate_rank(candidate),
    }
    try:
        order, rhs_fn = library_candidate_to_rhs_callable(candidate)
    except Exception as exc:
        out["status"] = "ERROR"
        out["message"] = f"Could not build factorized RHS: {exc}"
        out["traj_scores"] = []
        out["discovered_order"] = int(candidate.get("order", -1))
        return out

    status, message, traj_scores = validate_by_simulation(
        probe_runs,
        rhs_fn=rhs_fn,
        order=int(order),
        pass_nrmse=float(pass_nrmse),
        partial_nrmse=float(partial_nrmse),
        traj_time_budget_s=sim_validate_traj_time_budget_s,
        blowup_factor=float(sim_validate_blowup_factor),
        blowup_abs=float(sim_validate_blowup_abs),
    )
    out["status"] = str(status)
    out["message"] = str(message)
    out["traj_scores"] = traj_scores
    out["discovered_order"] = int(order)
    return out


__all__ = [
    "DEWitnessResult",
    "DEWitnessSpec",
    "SimValidateTimeout",
    "VALIDATE_SOLVER_TRIALS",
    "candidate_to_rhs_callable",
    "evaluate_compile_domain_witness",
    "evaluate_factorized_candidate_rollout",
    "evaluate_factorized_search_candidate_rollout",
    "evaluate_library_candidate_rollout",
    "factorized_search_report_to_rhs_callable",
    "library_candidate_to_rhs_callable",
    "run_rollout_witnesses",
    "validate_by_simulation",
    "walltime_budget",
    "witness_spec_from_run",
    "witness_specs_from_runs",
]
