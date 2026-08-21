# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Late continuous-skeleton refinement helpers for DE proposal slates."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from nestynet_sr.sr_search.factorized_search.config import apply_refine_profile
from nestynet_sr.sr_search.factorized_search.engine.scoring import score_expr

from .factorized_de import _expr_and_mapping_from_candidate, default_physics_rescue_hp


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        detached = value.detach().cpu()
        if int(detached.ndim) == 0:
            return float(detached.item())
        return detached.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _safe_float(value: Any, default: float = float("inf")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _candidate_feature_switches(candidate: Mapping[str, Any], *, order: int) -> tuple[bool, bool, bool]:
    feature_names = [str(v).strip().lower() for v in list(candidate.get("feature_names", []) or [])]
    include_x = candidate.get("include_x", None)
    include_u = candidate.get("include_u", None)
    include_du = candidate.get("include_du", None)

    if include_x is None:
        include_x = bool(feature_names and any(name not in {"u", "du", "dudx", "u_dot", "udot"} for name in feature_names))
    if include_u is None:
        include_u = any(name in {"u", "y"} for name in feature_names) if feature_names else True
    if include_du is None:
        include_du = any(name in {"du", "dudx", "u_dot", "udot"} for name in feature_names) if feature_names else True

    return bool(include_x), bool(include_u), bool(include_du) if int(order) == 2 else False


def _ordered_constants(candidate: Mapping[str, Any]) -> list[float]:
    raw = candidate.get("constants_ordered", None)
    if isinstance(raw, (list, tuple)):
        out: list[float] = []
        for item in raw:
            if isinstance(item, Mapping) and "value" in item:
                out.append(float(item["value"]))
        if out:
            return out
    raw_constants = candidate.get("constants", None)
    if isinstance(raw_constants, Mapping):
        return [float(v) for v in raw_constants.values()]
    if isinstance(raw_constants, (list, tuple)):
        out = []
        for item in raw_constants:
            if isinstance(item, Mapping) and "value" in item:
                out.append(float(item["value"]))
            else:
                out.append(float(item))
        return out
    return []


def _load_run_feature_rows(
    run: Any,
    candidate: Mapping[str, Any],
    *,
    order: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    csv_path = Path(getattr(run, "csv_path", run))
    data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"invalid trajectory CSV shape for {csv_path}")
    u = np.asarray(data[:, 0], dtype=np.float64).reshape(-1)
    x = np.asarray(data[:, 1], dtype=np.float64).reshape(-1)
    if u.size < 4:
        raise ValueError(f"too few trajectory rows for CSR refinement: {csv_path}")

    edge_order = 2 if int(u.size) >= 3 else 1
    du = np.gradient(u, x, edge_order=edge_order)
    d2u = np.gradient(du, x, edge_order=edge_order)
    y = du if int(order) == 1 else d2u

    include_x, include_u, include_du = _candidate_feature_switches(candidate, order=int(order))
    cols: list[np.ndarray] = []
    if include_x:
        cols.append(x)
    if include_u:
        cols.append(u)
    if int(order) == 2 and include_du:
        cols.append(du)
    for value in _ordered_constants(candidate):
        cols.append(np.full_like(x, float(value), dtype=np.float64))
    if not cols:
        raise ValueError("candidate has no active feature columns")

    X = np.stack(cols, axis=1)
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if int(mask.sum()) < 8:
        raise ValueError(f"too few finite CSR rows for {csv_path}: {int(mask.sum())}")
    return (
        torch.as_tensor(X[mask], dtype=dtype),
        torch.as_tensor(y[mask].reshape(-1, 1), dtype=dtype),
    )


def _stack_run_rows(
    runs: Sequence[Any],
    candidate: Mapping[str, Any],
    *,
    order: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for run in list(runs or []):
        x_i, y_i = _load_run_feature_rows(run, candidate, order=int(order), dtype=dtype)
        xs.append(x_i)
        ys.append(y_i)
    if not xs:
        raise ValueError("no trajectory rows available for CSR refinement")
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def _refine_cfg(*, profile: str, max_trials: int, max_depth: int) -> dict[str, Any]:
    hp = default_physics_rescue_hp(preset="fast")
    hp = apply_refine_profile(hp, str(profile or "rare_final_polish"))
    hp.refine_max_trials = max(0, min(int(getattr(hp, "refine_max_trials", 50)), int(max_trials)))
    hp.refine_slate_budget = max(0, min(int(getattr(hp, "refine_slate_budget", 8)), int(max_trials)))
    hp.refine_max_variants = max(1, min(int(getattr(hp, "refine_max_variants", 1)), 2))
    hp.refine_max_params = max(1, min(int(getattr(hp, "refine_max_params", 1)), 1))
    hp.refine_num_restarts = max(1, min(int(getattr(hp, "refine_num_restarts", 1)), 1))
    hp.refine_lbfgs_steps = max(1, min(int(getattr(hp, "refine_lbfgs_steps", 4)), 4))
    hp.refine_grid_max_evals = max(8, min(int(getattr(hp, "refine_grid_max_evals", 32)), 32))
    hp.refine_fit_subset = max(0, min(int(getattr(hp, "refine_fit_subset", 64)), 256))

    diagnostics: dict[str, Any] = {}
    return {
        "refine_profile": str(profile or "rare_final_polish"),
        "refine_mode": "final_polish",
        "optimizer": str(getattr(hp, "refine_optimizer", "grid_then_lbfgs")),
        "lbfgs_escalate_improve_factor": float(getattr(hp, "refine_lbfgs_escalate_improve_factor", 2.0)),
        "lbfgs_steps": int(hp.refine_lbfgs_steps),
        "fit_subset": int(hp.refine_fit_subset),
        "fit_subset_mode": str(getattr(hp, "refine_fit_subset_mode", "stratified")),
        "num_restarts": int(hp.refine_num_restarts),
        "max_variants": int(hp.refine_max_variants),
        "max_params": int(hp.refine_max_params),
        "linear_combo_enable": bool(getattr(hp, "refine_linear_combo_enable", True)),
        "linear_terms_max": int(getattr(hp, "refine_linear_terms_max", 6)),
        "linear_prune_rel": float(getattr(hp, "refine_linear_prune_rel", 1.0e-10)),
        "linear_ridge": float(getattr(hp, "refine_linear_ridge", 1.0e-8)),
        "slot_sensitivity_enable": bool(getattr(hp, "refine_slot_sensitivity_enable", True)),
        "slot_sensitivity_subset": int(getattr(hp, "refine_slot_sensitivity_subset", 64)),
        "slot_sensitivity_delta": float(getattr(hp, "refine_slot_sensitivity_delta", 0.05)),
        "slot_sensitivity_max_paths": int(getattr(hp, "refine_slot_sensitivity_max_paths", 32)),
        "prune_mapping_equiv_root_slots": bool(getattr(hp, "refine_prune_mapping_equiv_root_slots", True)),
        "attempt_cache_enable": True,
        "attempt_cache_max_entries": int(getattr(hp, "refine_attempt_cache_max_entries", 2048)),
        "attempt_cache": {},
        "diagnostics": diagnostics,
        "gate_best_factor": float(getattr(hp, "refine_gate_best_factor", 2.0)),
        "gate_potential_enable": bool(getattr(hp, "refine_gate_potential_enable", True)),
        "gate_potential_subset": int(getattr(hp, "refine_gate_potential_subset", 64)),
        "gate_potential_improve_factor": float(getattr(hp, "refine_gate_potential_improve_factor", 1.5)),
        "gate_log_min": float(getattr(hp, "refine_gate_log_min", -3.0)),
        "gate_log_max": float(getattr(hp, "refine_gate_log_max", 3.0)),
        "gate_grid_size": int(getattr(hp, "refine_gate_grid_size", 9)),
        "gate_max_evals": int(getattr(hp, "refine_gate_max_evals", 16)),
        "max_refines": int(hp.refine_max_trials),
        "safe_eps": float(getattr(hp, "refine_safe_eps", 1.0e-8)),
        "safe_penalty_weight": float(getattr(hp, "refine_safe_penalty_weight", 1.0e2)),
        "safe_exp_clip": float(getattr(hp, "refine_safe_exp_clip", 20.0)),
        "theta_l2": float(getattr(hp, "refine_theta_l2", 0.0)),
        "init_log_min": float(getattr(hp, "refine_init_log_min", -3.0)),
        "init_log_max": float(getattr(hp, "refine_init_log_max", 3.0)),
        "refine_grid_enable": bool(getattr(hp, "refine_grid_enable", True)),
        "refine_grid_size": int(getattr(hp, "refine_grid_size", 17)),
        "refine_grid_size_2d": int(getattr(hp, "refine_grid_size_2d", 5)),
        "refine_grid_passes": int(getattr(hp, "refine_grid_passes", 1)),
        "refine_grid_topk": int(getattr(hp, "refine_grid_topk", 4)),
        "refine_grid_max_evals": int(hp.refine_grid_max_evals),
        "max_depth": int(max_depth),
        "score_head_enable": False,
        "score_mapping_family_mode": str(getattr(hp, "score_mapping_family_mode", "poly_only")),
        "brute_score_mapping_family_mode": str(getattr(hp, "brute_score_mapping_family_mode", "poly_only")),
        "score_pade_structural_enable": bool(getattr(hp, "score_pade_structural_enable", False)),
        "score_pade_structural_max_degree": int(getattr(hp, "score_pade_structural_max_degree", 2)),
        "score_pade_structural_max_total_degree": int(getattr(hp, "score_pade_structural_max_total_degree", 3)),
        "score_pade_structural_max_depth": int(getattr(hp, "score_pade_structural_max_depth", 8)),
        "score_pade_structural_max_size": int(getattr(hp, "score_pade_structural_max_size", 64)),
        "score_pade_structural_coeff_tol": float(getattr(hp, "score_pade_structural_coeff_tol", 1.0e-10)),
        "score_pade_structural_mse_rel_tol": float(getattr(hp, "score_pade_structural_mse_rel_tol", 1.0e-6)),
        "score_mapping_expensive_gate_best_factor": float(getattr(hp, "score_mapping_expensive_gate_best_factor", 5.0)),
        "score_mapping_expensive_rel_y": float(getattr(hp, "score_mapping_expensive_rel_y", 0.10)),
        "score_prescreen_enable": False,
    }


def refine_factorized_search_candidate_from_runs(
    candidate: Mapping[str, Any],
    *,
    fit_runs: Sequence[Any],
    probe_runs: Sequence[Any],
    max_trials: int = 8,
    profile: str = "rare_final_polish",
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    """Run bounded CSR on one serialized whole-RHS FSS candidate.

    Returns a structured dict with ``accepted`` and ``candidate`` keys.  The
    returned candidate is only different from the input when CSR improves probe
    loss on the same trajectory-derived derivative table.
    """

    cand = dict(candidate)
    if str(cand.get("engine", "factorized_search")) != "factorized_search":
        return {"accepted": False, "candidate": cand, "reason": "unsupported_engine"}
    try:
        expr_ast, _ = _expr_and_mapping_from_candidate(cand)
        order = int(cand.get("order", 1))
        x_fit, y_fit = _stack_run_rows(fit_runs, cand, order=order, dtype=dtype)
        probe_source = list(probe_runs or fit_runs)
        x_probe, y_probe = _stack_run_rows(probe_source, cand, order=order, dtype=dtype)

        max_depth = max(2, int(cand.get("max_depth", 8) or 8))
        cfg = _refine_cfg(profile=str(profile), max_trials=int(max_trials), max_depth=int(max_depth))
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        proj = torch.randn((int(x_probe.shape[0]), 16), generator=gen, dtype=dtype)

        base = score_expr(
            expr_ast,
            x_fit,
            y_fit,
            x_probe,
            y_probe,
            proj,
            "bits",
            2.0,
            6.0,
            1,
            refine_enable=False,
            refine_cfg=dict(cfg),
            return_expr=True,
        )
        if base is None:
            return {"accepted": False, "candidate": cand, "reason": "base_score_failed"}
        base_mse = _safe_float(base[0])
        state = {"trials_done": 0, "window_trials_left": int(max_trials), "gate_relax_factor": 1.0}
        refined = score_expr(
            expr_ast,
            x_fit,
            y_fit,
            x_probe,
            y_probe,
            proj,
            "bits",
            2.0,
            6.0,
            1,
            refine_enable=True,
            refine_cfg=cfg,
            refine_best_mse=base_mse,
            refine_state=state,
            return_expr=True,
        )
        diagnostics = dict(cfg.get("diagnostics", {}) or {})
        if refined is None:
            return {
                "accepted": False,
                "candidate": cand,
                "reason": "refine_score_failed",
                "base_probe_mse": None if not math.isfinite(base_mse) else float(base_mse),
                "trials_used": int(state.get("trials_done", 0)),
                "diagnostics": _jsonable(diagnostics),
            }
        refined_mse = _safe_float(refined[0])
        accepted = bool(
            int(state.get("trials_done", 0)) > 0
            and math.isfinite(refined_mse)
            and math.isfinite(base_mse)
            and refined_mse < base_mse * (1.0 - 1.0e-12)
        )
        if not accepted:
            return {
                "accepted": False,
                "candidate": cand,
                "reason": "not_improved",
                "base_probe_mse": float(base_mse) if math.isfinite(base_mse) else None,
                "refined_probe_mse": float(refined_mse) if math.isfinite(refined_mse) else None,
                "trials_used": int(state.get("trials_done", 0)),
                "diagnostics": _jsonable(diagnostics),
            }

        out = dict(cand)
        out["expr_ast"] = _jsonable(refined[4])
        out["mapping"] = _jsonable(refined[3])
        out["mapping_kind"] = str((refined[3] or {}).get("kind", out.get("mapping_kind", "")))
        out["probe_mse"] = float(refined_mse)
        out["probe_rms"] = math.sqrt(max(0.0, float(refined_mse)))
        out["mse"] = float(refined_mse)
        out["score"] = float(refined_mse)
        out["score_raw"] = float(refined_mse)
        out["de_coe_csr_refined"] = True
        out["source_candidate_rank"] = cand.get("candidate_rank", cand.get("shortlist_rank", None))
        diag = dict(out.get("diagnostics", {}) or {})
        diag["de_coe_csr"] = {
            "accepted": True,
            "profile": str(profile),
            "max_trials": int(max_trials),
            "trials_used": int(state.get("trials_done", 0)),
            "base_probe_mse": float(base_mse),
            "refined_probe_mse": float(refined_mse),
            "diagnostics": _jsonable(diagnostics),
        }
        out["diagnostics"] = _jsonable(diag)
        return {
            "accepted": True,
            "candidate": _jsonable(out),
            "base_probe_mse": float(base_mse),
            "refined_probe_mse": float(refined_mse),
            "trials_used": int(state.get("trials_done", 0)),
            "diagnostics": _jsonable(diagnostics),
        }
    except Exception as exc:
        return {"accepted": False, "candidate": cand, "reason": "exception", "error": str(exc)}


__all__ = [
    "refine_factorized_search_candidate_from_runs",
]
