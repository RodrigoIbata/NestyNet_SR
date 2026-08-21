# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Manifest-driven planted capability benchmark for factorized symbolic search repair methods.

This runner is intentionally separate from:

- ``smoke_inverse_spec_solver.py``: unit-level correctness
- ``stage1_benchmark_harness.py``: controller comparisons

It focuses on conditional capability under planted corrupt-hole / missing-subtree
tasks, and emits machine-readable rows for downstream analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestynet_sr.sr_search.factorized_search import explorer
from nestynet_sr.sr_search.factorized_search.expr_ast import eval_node, get_at, node_depth, node_dims, node_str
from nestynet_sr.sr_search.factorized_search.inverse_action import (
    _estimate_inverse_action_transport,
    _inverse_action_path_mode_beam_states,
)
from nestynet_sr.sr_search.factorized_search.inverse_spec_solver import solve_inverse_spec_preview_rows
from nestynet_sr.sr_search.factorized_search.micro_search import MicroSearchGrammar, run_single_hole_micro_search
from nestynet_sr.sr_search.factorized_search.oracle_lab import equation_spec_from_dict, load_equation_spec
from nestynet_sr.sr_search.factorized_search.config import FactorizedSearchConfig


REPO_ROOT = ROOT
DEFAULT_SUITE_MANIFEST = (
    REPO_ROOT / "examples" / "oracle_factorized_search" / "capability_suites" / "planted_smoke.json"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    if torch.is_tensor(value):
        if value.ndim == 0:
            try:
                return float(value.item())
            except Exception:
                return None
        return None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        return None
    return str(value)


def _write_json(payload: dict[str, Any], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_capability_suite(path: str | pathlib.Path | None = None) -> tuple[pathlib.Path, dict[str, Any]]:
    manifest_path = pathlib.Path(path) if path is not None else DEFAULT_SUITE_MANIFEST
    payload = _load_json(manifest_path)
    cases = list(payload.get("cases") or [])
    if not cases:
        raise ValueError(f"No cases declared in capability suite manifest: {manifest_path}")
    return manifest_path, payload


def _coerce_ast(node: Any) -> Any:
    if isinstance(node, tuple):
        return tuple(_coerce_ast(v) for v in node)
    if isinstance(node, list):
        return tuple(_coerce_ast(v) for v in node)
    return node


def _coerce_dim_tuple(raw: Any) -> tuple[float, ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"Expected dim tuple/list, got {type(raw).__name__}")
    return tuple(float(v) for v in raw)


def _coerce_var_dims(raw: Any) -> tuple[tuple[float, ...], ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"Expected var_dims list/tuple, got {type(raw).__name__}")
    return tuple(_coerce_dim_tuple(dim) or tuple() for dim in raw)


def _safe_pool_phi(pool_nodes: Sequence[Any], x: torch.Tensor) -> torch.Tensor:
    cols: list[torch.Tensor] = []
    for node in pool_nodes:
        try:
            vals = eval_node(node, x).squeeze(-1)
        except Exception:
            vals = torch.zeros((int(x.shape[0]),), dtype=x.dtype)
        if not torch.isfinite(vals).all():
            vals = torch.zeros_like(vals)
        cols.append(vals)
    return torch.stack(cols, dim=1)


def _make_inverse_problem(
    truth_expr: Any,
    candidate_expr: Any,
    *,
    nvars: int,
    var_dims: Sequence[Sequence[float]] | None = None,
    seed: int = 0,
    n_fit: int = 96,
    n_probe: int = 192,
    poly_degree: int = 4,
) -> dict[str, Any]:
    g_fit = torch.Generator().manual_seed(int(seed))
    g_probe = torch.Generator().manual_seed(int(seed) + 17)
    x_fit = 0.5 + 1.5 * torch.rand((int(n_fit), int(nvars)), generator=g_fit, dtype=torch.float64)
    x_probe = 0.5 + 1.5 * torch.rand((int(n_probe), int(nvars)), generator=g_probe, dtype=torch.float64)
    y_fit = eval_node(truth_expr, x_fit)
    y_probe = eval_node(truth_expr, x_probe)
    fit_best = explorer.fit_best(eval_node(candidate_expr, x_fit), y_fit, int(poly_degree))
    if fit_best is None:
        raise RuntimeError("candidate mapping fit failed")
    _fit_mse, mapping = fit_best
    pool_nodes = explorer.build_pool(int(nvars))
    pool_dims = [node_dims(node, var_dims) for node in pool_nodes] if var_dims is not None else [None] * len(pool_nodes)
    return {
        "truth_expr": truth_expr,
        "candidate_expr": candidate_expr,
        "mapping": mapping,
        "x_fit": x_fit,
        "y_fit": y_fit,
        "x_probe": x_probe,
        "y_probe": y_probe,
        "pool_nodes": pool_nodes,
        "pool_dims": pool_dims,
        "pool_phi_fit": _safe_pool_phi(pool_nodes, x_fit),
        "pool_phi_probe": _safe_pool_phi(pool_nodes, x_probe),
    }


def _build_beam_state(problem: Mapping[str, Any], *, path: Sequence[int], var_dims: Any = None) -> dict[str, Any]:
    transport_ctx = _estimate_inverse_action_transport(
        problem["candidate_expr"],
        problem["mapping"],
        problem["x_fit"],
        problem["y_fit"],
        problem["x_probe"],
        problem["y_probe"],
        [tuple(int(v) for v in path)],
        safe_eps=1.0e-12,
    )
    beam_states = _inverse_action_path_mode_beam_states(
        parent_node=problem["candidate_expr"],
        parent_mapping=problem["mapping"],
        x_fit=problem["x_fit"],
        y_fit=problem["y_fit"],
        x_probe=problem["x_probe"],
        y_probe=problem["y_probe"],
        pool_nodes=problem["pool_nodes"],
        pool_phi_fit=problem["pool_phi_fit"],
        pool_phi_probe=problem["pool_phi_probe"],
        pool_dims=problem["pool_dims"],
        all_paths=[tuple(int(v) for v in path)],
        path_target_modes=None,
        transport_ctx=transport_ctx,
        cfg={
            "max_paths": 1,
            "dm": bool(var_dims is not None),
            "var_dims": var_dims,
            "max_depth": 6,
            "poly_degree": 4,
            "topk_terms": 6,
            "shortlist_mult": 4,
            "local_mode": "affine",
            "min_valid_frac": 0.25,
            "min_confidence": 0.10,
            "safe_eps": 1.0e-12,
            "confidence_mode": "conditioning",
            "confidence_target_gain": 4.0,
            "confidence_floor": 0.05,
            "branch_beam_width": 1,
            "micro_search_enable": False,
            "micro_search_max_depth": 3,
            "micro_search_beam_width": 24,
            "micro_search_topk": 16,
            "micro_search_seed_terms": 8,
            "target_mode": "robust",
            "full_mapping_penalty": 0.75,
            "exact_simple_target_bonus": 0.10,
            "additive_descend_penalty": 0.15,
            "nonadditive_leaf_penalty": 0.20,
            "exact_path_eta": 0.98,
            "exact_transport_min_lin_rel": 0.0,
            "periodic_min_valid_scale": 1.25,
            "periodic_min_confidence_scale": 1.35,
            "periodic_path_penalty": 0.65,
            "nonperiodic_muldiv_bonus": 0.10,
            "nonperiodic_explogsqrt_bonus": 0.05,
            "branch_ambiguity_penalty": 0.50,
            "transport_min_lin_rel": 0.02,
            "transport_min_effective_n": 8.0,
        },
        beam_width=1,
    )
    if not beam_states:
        raise RuntimeError("failed to build inverse beam state")
    return dict(beam_states[0])


def _resolve_case_profiles(case: Mapping[str, Any], defaults: Mapping[str, Any]) -> list[str]:
    raw = case.get("profiles", defaults.get("profiles", ["flat", "recursive"]))
    out: list[str] = []
    seen: set[str] = set()
    for item in list(raw or []):
        name = str(item or "").strip().lower()
        if name == "" or name in seen:
            continue
        seen.add(name)
        out.append(name)
    if not out:
        raise ValueError(f"Capability case resolved zero profiles: {case.get('case_id', '')}")
    return out


def _summarize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("case_type", "")), str(row.get("profile", "")))
        groups[key].append(dict(row))
    out: list[dict[str, Any]] = []
    for (case_type, profile), group_rows in sorted(groups.items()):
        truth_ranks = [
            float(row["truth_rank"])
            for row in group_rows
            if row.get("truth_rank", None) is not None and math.isfinite(float(row["truth_rank"]))
        ]
        best_probe_mses = [
            float(row["best_probe_mse"])
            for row in group_rows
            if row.get("best_probe_mse", None) is not None and math.isfinite(float(row["best_probe_mse"]))
        ]
        successes = [1.0 if bool(row.get("success", False)) else 0.0 for row in group_rows]
        truth_present = [1.0 if bool(row.get("truth_present", False)) else 0.0 for row in group_rows]
        out.append(
            {
                "case_type": case_type,
                "profile": profile,
                "n_rows": int(len(group_rows)),
                "success_rate": float(sum(successes) / len(successes)) if successes else float("nan"),
                "truth_present_rate": float(sum(truth_present) / len(truth_present)) if truth_present else float("nan"),
                "mean_truth_rank": float(sum(truth_ranks) / len(truth_ranks)) if truth_ranks else float("nan"),
                "mean_best_probe_mse": float(sum(best_probe_mses) / len(best_probe_mses)) if best_probe_mses else float("nan"),
            }
        )
    return out


def _run_inverse_spec_case(
    case: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any],
    save_dir: pathlib.Path | None,
) -> list[dict[str, Any]]:
    case_id = str(case.get("case_id", "") or "")
    truth_expr = _coerce_ast(case.get("truth_expr"))
    candidate_expr = _coerce_ast(case.get("candidate_expr"))
    if not isinstance(truth_expr, tuple) or not isinstance(candidate_expr, tuple):
        raise ValueError(f"inverse_spec case {case_id} requires tuple-like truth_expr and candidate_expr")
    nvars = int(case.get("nvars", 0) or 0)
    if nvars <= 0:
        raise ValueError(f"inverse_spec case {case_id} requires positive nvars")
    var_dims = _coerce_var_dims(case.get("var_dims", None))
    path = tuple(int(v) for v in list(case.get("path") or ()))
    if not path:
        raise ValueError(f"inverse_spec case {case_id} requires non-empty path")

    inverse_defaults = dict(defaults.get("inverse_spec", {}) or {})
    seed = int(case.get("seed", inverse_defaults.get("seed", 0)) or 0)
    n_fit = int(case.get("n_fit", inverse_defaults.get("n_fit", 96)) or 96)
    n_probe = int(case.get("n_probe", inverse_defaults.get("n_probe", 192)) or 192)
    poly_degree = int(case.get("poly_degree", inverse_defaults.get("poly_degree", 4)) or 4)
    max_depth = int(case.get("max_depth", inverse_defaults.get("max_depth", max(4, node_depth(truth_expr)))) or 4)
    enum_max_depth = int(case.get("enum_max_depth", inverse_defaults.get("enum_max_depth", 2)) or 2)
    enum_max_trees = int(case.get("enum_max_trees", inverse_defaults.get("enum_max_trees", 256)) or 256)
    preview_topk = int(case.get("preview_topk", inverse_defaults.get("preview_topk", 8)) or 8)
    recursive_max_depth = int(case.get("recursive_max_depth", inverse_defaults.get("recursive_max_depth", 2)) or 2)
    recursive_trigger_rel_mse = float(
        case.get("recursive_trigger_rel_mse", inverse_defaults.get("recursive_trigger_rel_mse", 0.0)) or 0.0
    )
    solve_threshold = float(case.get("solve_threshold", defaults.get("solve_threshold", 1.0e-10)) or 1.0e-10)
    profiles = _resolve_case_profiles(case, defaults)

    problem = _make_inverse_problem(
        truth_expr,
        candidate_expr,
        nvars=nvars,
        var_dims=var_dims,
        seed=seed,
        n_fit=n_fit,
        n_probe=n_probe,
        poly_degree=poly_degree,
    )
    beam_state = _build_beam_state(problem, path=path, var_dims=var_dims)
    truth_key = node_str(truth_expr)
    hidden_truth_expr = get_at(truth_expr, path)

    rows: list[dict[str, Any]] = []
    for profile in profiles:
        profile_name = str(profile).strip().lower()
        recursive_enable = profile_name not in {"flat", "nonrecursive", "flat_only"}
        result = solve_inverse_spec_preview_rows(
            parent_node=problem["candidate_expr"],
            beam_state=beam_state,
            beam_rank=0,
            slate_id=f"{case_id}:{profile_name}",
            max_depth=max_depth,
            nvars=nvars,
            poly_degree=poly_degree,
            var_dims=var_dims,
            pool_nodes=problem["pool_nodes"],
            pool_dims=problem["pool_dims"],
            enum_max_depth=enum_max_depth,
            enum_max_trees=enum_max_trees,
            preview_topk=preview_topk,
            recursive_enable=bool(recursive_enable),
            recursive_max_depth=recursive_max_depth,
            recursive_trigger_rel_mse=recursive_trigger_rel_mse,
            recursive_seed_cap=int(inverse_defaults.get("recursive_seed_cap", 6) or 6),
            recursive_branch_topk=int(inverse_defaults.get("recursive_branch_topk", 4) or 4),
            recursive_child_topk=int(inverse_defaults.get("recursive_child_topk", 2) or 2),
            complexity_penalty=float(case.get("complexity_penalty", inverse_defaults.get("complexity_penalty", 0.0)) or 0.0),
        )
        preview_rows = list(result.get("rows") or [])
        solver_meta = dict(result.get("solver_meta", {}) or {})
        truth_rank = None
        for idx, row in enumerate(preview_rows, start=1):
            if str(row.get("child_key", "") or "") == truth_key:
                truth_rank = int(idx)
                break
        best_row = dict(preview_rows[0] if preview_rows else {})
        best_probe_mse = float(best_row.get("local_probe_mse", float("inf")) or float("inf"))
        out_row = {
            "case_id": case_id,
            "case_type": "inverse_spec",
            "profile": profile_name,
            "seed": int(seed),
            "nvars": int(nvars),
            "path": [int(v) for v in path],
            "truth_expr": truth_key,
            "candidate_expr": node_str(candidate_expr),
            "truth_depth": int(node_depth(truth_expr)),
            "candidate_depth": int(node_depth(candidate_expr)),
            "hidden_truth_expr": node_str(hidden_truth_expr),
            "hidden_truth_depth": int(node_depth(hidden_truth_expr)),
            "enum_max_depth": int(enum_max_depth),
            "enum_max_trees": int(enum_max_trees),
            "preview_topk": int(preview_topk),
            "recursive_enable": bool(recursive_enable),
            "recursive_used": bool(solver_meta.get("recursive_used", False)),
            "periodic_forward_used": bool(solver_meta.get("periodic_forward_used", False)),
            "row_count": int(len(preview_rows)),
            "truth_present": bool(truth_rank is not None),
            "truth_rank": None if truth_rank is None else int(truth_rank),
            "best_expr": str(best_row.get("child_key", "") or ""),
            "best_probe_mse": None if not math.isfinite(best_probe_mse) else float(best_probe_mse),
            "best_fit_mse": _jsonable(best_row.get("local_fit_mse", None)),
            "best_is_truth": bool(best_row.get("child_key", "") == truth_key),
            "success": bool(truth_rank is not None),
            "solve_threshold": float(solve_threshold),
            "best_under_threshold": bool(math.isfinite(best_probe_mse) and best_probe_mse <= solve_threshold),
            "candidate_count_raw": int(solver_meta.get("candidate_count_raw", 0) or 0),
            "candidate_count_scored": int(solver_meta.get("candidate_count_scored", 0) or 0),
            "preview_count": int(solver_meta.get("preview_count", len(preview_rows)) or len(preview_rows)),
            "solver_status": str(solver_meta.get("status", "") or ""),
            "wall_seconds": _jsonable(solver_meta.get("wall_seconds", None)),
        }
        rows.append(out_row)
        if save_dir is not None:
            _write_json(
                {
                    "case": dict(case),
                    "row": out_row,
                    "solver_meta": solver_meta,
                    "preview_rows": preview_rows,
                },
                save_dir / f"{case_id}__{profile_name}.json",
            )
    return rows


def _micro_grammar_from_case(case: Mapping[str, Any], defaults: Mapping[str, Any]) -> MicroSearchGrammar:
    grammar_raw = dict(defaults.get("micro_search", {}).get("grammar", {}) or {})
    grammar_raw.update(dict(case.get("grammar", {}) or {}))
    return MicroSearchGrammar(
        max_depth=int(grammar_raw.get("max_depth", 2) or 2),
        unary_ops=tuple(str(v) for v in list(grammar_raw.get("unary_ops", []))),
        binary_ops=tuple(str(v) for v in list(grammar_raw.get("binary_ops", ["add", "mul"]))),
        constant_values=tuple(float(v) for v in list(grammar_raw.get("constant_values", []))),
    )


def _load_micro_spec(case: Mapping[str, Any], *, manifest_path: pathlib.Path) -> Any:
    if case.get("spec_payload", None) is not None:
        return equation_spec_from_dict(dict(case.get("spec_payload") or {}), source=f"capability::{case.get('case_id', '')}")
    raw_path = str(case.get("spec_path", "") or "")
    if raw_path == "":
        raise ValueError(f"micro_search case {case.get('case_id', '')} requires spec_payload or spec_path")
    cand = pathlib.Path(raw_path)
    candidates = [cand] if cand.is_absolute() else [manifest_path.parent / cand, REPO_ROOT / cand]
    for path in candidates:
        if path.is_file():
            return load_equation_spec(path)
    raise FileNotFoundError(f"Could not resolve micro_search spec path: {raw_path}")


def _run_micro_search_case(
    case: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any],
    manifest_path: pathlib.Path,
    save_dir: pathlib.Path | None,
) -> list[dict[str, Any]]:
    case_id = str(case.get("case_id", "") or "")
    micro_defaults = dict(defaults.get("micro_search", {}) or {})
    spec = _load_micro_spec(case, manifest_path=manifest_path)
    hp = FactorizedSearchConfig()
    hp.n_fit = int(case.get("n_fit", micro_defaults.get("n_fit", 64)) or 64)
    hp.n_probe = int(case.get("n_probe", micro_defaults.get("n_probe", 96)) or 96)
    hp.poly_degree = int(case.get("poly_degree", micro_defaults.get("poly_degree", 3)) or 3)
    seed = int(case.get("seed", micro_defaults.get("seed", 0)) or 0)
    solve_threshold = float(case.get("solve_threshold", defaults.get("solve_threshold", 1.0e-12)) or 1.0e-12)
    budget_ladder = tuple(int(v) for v in list(case.get("budget_ladder", micro_defaults.get("budget_ladder", [1, 3, 5, 10]))) if int(v) > 0)
    if not budget_ladder:
        raise ValueError(f"micro_search case {case_id} resolved empty budget_ladder")
    evaluation_budget = int(case.get("evaluation_budget", max(budget_ladder)) or max(budget_ladder))

    report = run_single_hole_micro_search(
        spec,
        factorized_search_hp=hp,
        seed=seed,
        candidate_expr=str(case.get("candidate_expr")) if case.get("candidate_expr", None) is not None else None,
        corrupt_path=case.get("corrupt_path", None),
        replacement_expr=str(case.get("replacement_expr")) if case.get("replacement_expr", None) is not None else None,
        hole_path=case.get("hole_path", None),
        grammar=_micro_grammar_from_case(case, defaults),
        solve_threshold=solve_threshold,
        budget_ladder=budget_ladder,
        report_topk=int(case.get("report_topk", micro_defaults.get("report_topk", 16)) or 16),
        include_samples=False,
    )
    inverse_metrics = dict(((report.get("metrics") or {}).get("inverse") or {}))
    residual_metrics = dict(((report.get("metrics") or {}).get("residual") or {}))
    grammar_info = dict(report.get("grammar", {}) or {})

    def _budget_flag(mapping: Mapping[str, Any] | None, budget: int) -> bool:
        mm = dict(mapping or {})
        if budget in mm:
            return bool(mm[budget])
        return bool(mm.get(str(int(budget)), False))

    common = {
        "case_id": case_id,
        "case_type": "micro_search",
        "seed": int(seed),
        "spec_id": str(report.get("spec_id", "")),
        "truth_expr": str(report.get("truth_expr", "")),
        "candidate_expr": str(report.get("candidate_expr", "")),
        "hole_path": list(report.get("hole_path", []) or []),
        "hole_path_str": str(report.get("hole_path_str", "")),
        "hole_truth_expr": str(report.get("hole_truth_expr", "") or ""),
        "grammar_n_candidates": int(grammar_info.get("n_candidates", 0) or 0),
        "grammar_truth_in_grammar": bool(grammar_info.get("truth_in_grammar", False)),
        "evaluation_budget": int(evaluation_budget),
        "solve_threshold": float(solve_threshold),
        "truth_rank_advantage_vs_residual": _jsonable((report.get("oracle") or {}).get("inverse_truth_rank_advantage", None)),
    }
    rows = [
        {
            **common,
            "profile": "inverse",
            "truth_present": inverse_metrics.get("truth_rank", None) is not None,
            "truth_rank": inverse_metrics.get("truth_rank", None),
            "best_expr": inverse_metrics.get("best_expr", None),
            "best_probe_mse": inverse_metrics.get("best_full_probe_mse", None),
            "success": _budget_flag(inverse_metrics.get("solve_at_budget", {}) or {}, evaluation_budget),
            "truth_seen_at_evaluation_budget": _budget_flag(inverse_metrics.get("truth_seen_at_budget", {}) or {}, evaluation_budget),
            "solve_at_budget": dict(inverse_metrics.get("solve_at_budget", {}) or {}),
            "truth_seen_at_budget": dict(inverse_metrics.get("truth_seen_at_budget", {}) or {}),
        },
        {
            **common,
            "profile": "residual",
            "truth_present": residual_metrics.get("truth_rank", None) is not None,
            "truth_rank": residual_metrics.get("truth_rank", None),
            "best_expr": residual_metrics.get("best_expr", None),
            "best_probe_mse": residual_metrics.get("best_full_probe_mse", None),
            "success": _budget_flag(residual_metrics.get("solve_at_budget", {}) or {}, evaluation_budget),
            "truth_seen_at_evaluation_budget": _budget_flag(residual_metrics.get("truth_seen_at_budget", {}) or {}, evaluation_budget),
            "solve_at_budget": dict(residual_metrics.get("solve_at_budget", {}) or {}),
            "truth_seen_at_budget": dict(residual_metrics.get("truth_seen_at_budget", {}) or {}),
        },
    ]
    if save_dir is not None:
        _write_json({"case": dict(case), "report": report, "rows": rows}, save_dir / f"{case_id}.json")
    return rows


def run_capability_suite(
    manifest: Mapping[str, Any],
    *,
    manifest_path: pathlib.Path,
    output_dir: pathlib.Path,
    save_individual_reports: bool,
) -> dict[str, Any]:
    defaults = dict(manifest.get("defaults", {}) or {})
    suite_id = str(manifest.get("suite_id", "capability_suite") or "capability_suite")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir = output_dir / "cases" if save_individual_reports else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for raw_case in list(manifest.get("cases") or []):
        case = dict(raw_case or {})
        case_type = str(case.get("case_type", "") or "").strip().lower()
        if case_type == "inverse_spec":
            rows.extend(_run_inverse_spec_case(case, defaults=defaults, save_dir=save_dir))
            continue
        if case_type == "micro_search":
            rows.extend(_run_micro_search_case(case, defaults=defaults, manifest_path=manifest_path, save_dir=save_dir))
            continue
        raise ValueError(f"Unknown capability case_type {case_type!r} in {case.get('case_id', '')}")

    summary = _summarize_rows(rows)
    payload = {
        "suite_id": suite_id,
        "suite_manifest": str(manifest_path),
        "n_cases": int(len(list(manifest.get("cases") or []))),
        "rows": rows,
        "summary": summary,
    }
    _write_json(payload, output_dir / "capability_benchmark_results.json")
    _write_json({"suite_id": suite_id, "summary": summary}, output_dir / "capability_benchmark_summary.json")
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Planted conditional-capability benchmark for factorized symbolic search repair methods")
    p.add_argument("--suite_manifest", type=str, default=str(DEFAULT_SUITE_MANIFEST))
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--save_individual_reports", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path, manifest = load_capability_suite(args.suite_manifest)
    suite_id = str(manifest.get("suite_id", "capability_suite") or "capability_suite")
    output_dir = pathlib.Path(args.output_dir or (REPO_ROOT / "results" / f"capability_benchmark_{suite_id}"))
    payload = run_capability_suite(
        manifest,
        manifest_path=manifest_path,
        output_dir=output_dir,
        save_individual_reports=bool(args.save_individual_reports),
    )
    print(
        f"[capability_benchmark] suite={payload['suite_id']} "
        f"cases={payload['n_cases']} rows={len(payload['rows'])} output_dir={output_dir}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
