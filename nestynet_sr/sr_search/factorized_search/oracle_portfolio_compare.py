# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Compare baseline and scaffold oracle experts as a portfolio."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import pathlib
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import torch

from .basis_compile import basis_expr_key, basis_structure_signature_key
from .expr_ast import is_valid_node, node_depth, node_size, node_str
from .oracle_lab import (
    _apply_cli_overrides,
    _parse_args as _parse_oracle_lab_args,
    compile_target_ast,
    default_oracle_hyperparams,
    load_equation_spec,
    run_oracle_equation,
)
from .oracle_regression import (
    DEFAULT_SUITE_MANIFEST,
    load_regression_suite,
    resolve_suite_spec_paths,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
EXPERT_BASELINE = "baseline"
EXPERT_PERIODIC = "periodic_scaffold"
EXPERT_PERIODIC_SPECIALIST = "periodic_specialist"
EXPERT_EXP = "exp_scaffold"
EXPERT_EXP_SPECIALIST = "exp_specialist"
EXPERT_LOG = "log_scaffold"
EXPERT_LOG_SPECIALIST = "log_specialist"
EXPERT_RATIONAL = "rational_scaffold"
EXPERT_RATIONAL_SPECIALIST = "rational_specialist"
SCAFFOLD_EXPERT_FAMILIES = {
    EXPERT_PERIODIC: ("periodic",),
    EXPERT_PERIODIC_SPECIALIST: ("periodic",),
    EXPERT_EXP: ("exp",),
    EXPERT_EXP_SPECIALIST: ("exp",),
    EXPERT_LOG: ("log",),
    EXPERT_LOG_SPECIALIST: ("log",),
    EXPERT_RATIONAL: ("rational",),
    EXPERT_RATIONAL_SPECIALIST: ("rational",),
}
ALL_EXPERTS = (EXPERT_BASELINE,) + tuple(SCAFFOLD_EXPERT_FAMILIES.keys())
DEFAULT_EXPERTS = (EXPERT_BASELINE, EXPERT_PERIODIC)
_STRUCTURE_IGNORE_OPS = {"add", "const", "var"}
DEFAULT_STRUCTURAL_SOLVE_MSE_THRESHOLD = 1.0e-8
_SOLUTION_LABEL_SOLVE = "solve"
_SOLUTION_LABEL_SURROGATE = "surrogate"
_SOLUTION_LABEL_MISS = "miss"
SPECIALIST_EXPERTS = {
    EXPERT_PERIODIC_SPECIALIST,
    EXPERT_EXP_SPECIALIST,
    EXPERT_LOG_SPECIALIST,
    EXPERT_RATIONAL_SPECIALIST,
}
_TORCH_THREAD_CONFIGURED: tuple[int | None, int | None] | None = None


def _mean(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return float(sum(xs) / len(xs))


def _median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    if n % 2 == 1:
        return float(ys[mid])
    return float((ys[mid - 1] + ys[mid]) / 2.0)


def _write_csv(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(payload: dict[str, Any], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _ast_from_jsonable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_ast_from_jsonable(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_ast_from_jsonable(v) for v in value)
    if isinstance(value, dict):
        return {str(k): _ast_from_jsonable(v) for k, v in value.items()}
    return value


def _canonical_expr(node: Any) -> str:
    return basis_expr_key(node)


def _collect_structure_ops(node: Any) -> list[str]:
    ops: set[str] = set()

    def _walk(cur: Any) -> None:
        if not isinstance(cur, tuple) or not cur:
            return
        op = str(cur[0])
        if op == "var":
            return
        if op not in _STRUCTURE_IGNORE_OPS:
            ops.add(op)
        for child in cur[1:]:
            _walk(child)

    _walk(node)
    return sorted(ops)


def _collect_var_indices(node: Any) -> list[int]:
    vars_used: set[int] = set()

    def _walk(cur: Any) -> None:
        if not isinstance(cur, tuple) or not cur:
            return
        op = str(cur[0])
        if op == "var" and len(cur) >= 2:
            try:
                vars_used.add(int(cur[1]))
            except Exception:
                pass
            return
        for child in cur[1:]:
            _walk(child)

    _walk(node)
    return sorted(vars_used)


def score_structure_match(truth_ast: Any, candidate_ast: Any) -> dict[str, Any]:
    truth_node = _ast_from_jsonable(truth_ast)
    cand_node = _ast_from_jsonable(candidate_ast)
    truth_canon = _canonical_expr(truth_node)
    cand_canon = _canonical_expr(cand_node)
    truth_sig = basis_structure_signature_key(truth_node, collapse_scalar_consts=True)
    cand_sig = basis_structure_signature_key(cand_node, collapse_scalar_consts=True)
    truth_ops = _collect_structure_ops(truth_node)
    cand_ops = _collect_structure_ops(cand_node)
    truth_vars = _collect_var_indices(truth_node)
    cand_vars = _collect_var_indices(cand_node)
    truth_op_set = set(truth_ops)
    cand_op_set = set(cand_ops)
    truth_var_set = set(truth_vars)
    cand_var_set = set(cand_vars)
    truth_top_op = str(truth_node[0]) if isinstance(truth_node, tuple) and truth_node else ""
    cand_top_op = str(cand_node[0]) if isinstance(cand_node, tuple) and cand_node else ""
    truth_size = int(node_size(truth_node)) if isinstance(truth_node, tuple) and is_valid_node(truth_node) else -1
    cand_size = int(node_size(cand_node)) if isinstance(cand_node, tuple) and is_valid_node(cand_node) else -1
    truth_depth = int(node_depth(truth_node)) if isinstance(truth_node, tuple) and is_valid_node(truth_node) else -1
    cand_depth = int(node_depth(cand_node)) if isinstance(cand_node, tuple) and is_valid_node(cand_node) else -1
    return {
        "truth_canonical_expr": truth_canon,
        "candidate_canonical_expr": cand_canon,
        "exact_canonical_match": bool(truth_canon != "" and truth_canon == cand_canon),
        "truth_structure_signature": truth_sig,
        "candidate_structure_signature": cand_sig,
        "exact_structure_signature_match": bool(truth_sig != "" and truth_sig == cand_sig),
        "truth_top_op": truth_top_op,
        "candidate_top_op": cand_top_op,
        "top_level_op_match": bool(truth_top_op != "" and truth_top_op == cand_top_op),
        "truth_structure_ops": truth_ops,
        "candidate_structure_ops": cand_ops,
        "structure_ops_hit": bool(truth_op_set.issubset(cand_op_set)),
        "exact_structure_ops_match": bool(truth_op_set == cand_op_set),
        "truth_var_indices": truth_vars,
        "candidate_var_indices": cand_vars,
        "truth_var_coverage": bool(truth_var_set.issubset(cand_var_set)),
        "exact_var_set_match": bool(truth_var_set == cand_var_set),
        "truth_node_size": int(truth_size),
        "candidate_node_size": int(cand_size),
        "same_node_size": bool(truth_size >= 0 and truth_size == cand_size),
        "truth_node_depth": int(truth_depth),
        "candidate_node_depth": int(cand_depth),
        "same_node_depth": bool(truth_depth >= 0 and truth_depth == cand_depth),
    }


def classify_solution_outcome(
    *,
    best_mse: float,
    truth_ast: Any,
    candidate_ast: Any,
    structure: Mapping[str, Any],
    numeric_solve_threshold: float = DEFAULT_STRUCTURAL_SOLVE_MSE_THRESHOLD,
) -> dict[str, Any]:
    mse = float(best_mse)
    numeric_solve = bool(math.isfinite(mse) and mse <= float(numeric_solve_threshold))
    if not numeric_solve:
        return {
            "solution_label": _SOLUTION_LABEL_MISS,
            "solution_label_reason": "mse_above_threshold",
            "numeric_solve": 0,
        }
    if not (isinstance(candidate_ast, tuple) and is_valid_node(candidate_ast)):
        return {
            "solution_label": _SOLUTION_LABEL_SURROGATE,
            "solution_label_reason": "numeric_only_no_candidate_ast",
            "numeric_solve": 1,
        }
    if not (isinstance(truth_ast, tuple) and is_valid_node(truth_ast)):
        return {
            "solution_label": _SOLUTION_LABEL_SURROGATE,
            "solution_label_reason": "numeric_only_truth_unavailable",
            "numeric_solve": 1,
        }
    if bool(structure.get("exact_canonical_match", False)):
        return {
            "solution_label": _SOLUTION_LABEL_SOLVE,
            "solution_label_reason": "exact_canonical",
            "numeric_solve": 1,
        }
    if (
        bool(structure.get("exact_structure_signature_match", False))
        and bool(structure.get("exact_var_set_match", False))
        and bool(structure.get("top_level_op_match", False))
    ):
        return {
            "solution_label": _SOLUTION_LABEL_SOLVE,
            "solution_label_reason": "numeric_refine_structure_signature",
            "numeric_solve": 1,
        }
    if (
        bool(structure.get("top_level_op_match", False))
        and bool(structure.get("exact_structure_ops_match", False))
        and bool(structure.get("exact_var_set_match", False))
        and bool(structure.get("same_node_size", False))
        and bool(structure.get("same_node_depth", False))
    ):
        return {
            "solution_label": _SOLUTION_LABEL_SOLVE,
            "solution_label_reason": "numeric_refine_exact_signature",
            "numeric_solve": 1,
        }
    return {
        "solution_label": _SOLUTION_LABEL_SURROGATE,
        "solution_label_reason": "numeric_only",
        "numeric_solve": 1,
    }


def _solution_label_rank(label: str) -> int:
    if label == _SOLUTION_LABEL_SOLVE:
        return 0
    if label == _SOLUTION_LABEL_SURROGATE:
        return 1
    return 2


def _normalize_experts(values: Sequence[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in list(values or ()):
        tok = str(raw or "").strip()
        if not tok or tok in seen:
            continue
        if tok not in ALL_EXPERTS:
            raise ValueError(
                f"Unknown expert {tok!r}; expected one of {', '.join(ALL_EXPERTS)}"
            )
        seen.add(tok)
        out.append(tok)
    if not out:
        raise ValueError("Expected at least one expert")
    return out


def _resolve_worker_torch_threads(
    *,
    jobs: int,
    torch_num_threads: int | None,
    torch_num_interop_threads: int | None,
) -> tuple[int | None, int | None]:
    num_threads = None if torch_num_threads is None else max(1, int(torch_num_threads))
    num_interop_threads = (
        None if torch_num_interop_threads is None else max(1, int(torch_num_interop_threads))
    )
    if int(jobs) > 1:
        if num_threads is None:
            num_threads = 1
        if num_interop_threads is None:
            num_interop_threads = 1
    return num_threads, num_interop_threads


def _configure_worker_torch_threads(
    *,
    torch_num_threads: int | None,
    torch_num_interop_threads: int | None,
) -> None:
    global _TORCH_THREAD_CONFIGURED
    desired = (
        None if torch_num_threads is None else max(1, int(torch_num_threads)),
        None if torch_num_interop_threads is None else max(1, int(torch_num_interop_threads)),
    )
    if _TORCH_THREAD_CONFIGURED == desired:
        return
    if desired[0] is not None:
        torch.set_num_threads(int(desired[0]))
    if desired[1] is not None:
        try:
            torch.set_num_interop_threads(int(desired[1]))
        except RuntimeError:
            pass
    _TORCH_THREAD_CONFIGURED = desired


def _make_hp(args: argparse.Namespace, *, budget: int, refine_enable: bool, expert: str) -> Any:
    hp = default_oracle_hyperparams()
    hp = _apply_cli_overrides(hp, args)
    hp.n_iter = int(budget)
    hp.refine_enable = bool(refine_enable)
    expert = str(expert)
    if expert in SCAFFOLD_EXPERT_FAMILIES:
        hp.closure_search_enable = True
        family_attr = f"{expert}_families"
        families = list(getattr(args, family_attr, SCAFFOLD_EXPERT_FAMILIES.get(expert, ())) or ())
        if not families:
            families = list(SCAFFOLD_EXPERT_FAMILIES.get(expert, ()))
        hp.closure_search_families = [str(v) for v in families]
        hp.closure_search_max_proposals = int(getattr(args, "periodic_scaffold_max_scaffolds", hp.closure_search_max_proposals))
        hp.closure_search_anchors_per_family = int(getattr(args, "periodic_scaffold_anchors_per_family", hp.closure_search_anchors_per_family))
        hp.closure_search_preview_topk = int(getattr(args, "periodic_scaffold_preview_topk", hp.closure_search_preview_topk))
        hp.closure_search_exact_topk = int(getattr(args, "periodic_scaffold_exact_topk", hp.closure_search_exact_topk))
    if expert in SPECIALIST_EXPERTS:
        # Family specialists run the stripped scaffold-centric profile that
        # produced the original single-family wins, rather than the generic
        # preservation configuration with a family toggle layered on top.
        hp.max_depth = 5
        hp.brute_depth = 0
        hp.refine_enable = False
        hp.inverse_steering_enable = False
        hp.hole_search_enable = False
        hp.repair_pass_enable = False
        hp.inverse_spec_enable = False
        hp.inverse_spec_recursive_enable = False
        hp.no_residual = True
        hp.score_prescreen_enable = False
        hp.score_mapping_family_mode = "cheap"
        # Do not terminate early on a numerically good surrogate; let the direct
        # periodic scaffold route keep searching for the structurally correct form.
        hp.early_stop_mse = 0.0
    return hp


def _run_portfolio_job(job: Mapping[str, Any]) -> dict[str, Any]:
    _configure_worker_torch_threads(
        torch_num_threads=job.get("torch_num_threads", None),
        torch_num_interop_threads=job.get("torch_num_interop_threads", None),
    )
    spec_path = pathlib.Path(str(job["spec_path"]))
    budget = int(job["budget"])
    mode = str(job["mode"])
    rep = int(job["repeat"])
    rep_seed = int(job["seed"])
    expert = str(job["expert"])
    dtype_name = str(job["dtype"])
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    args = _parse_oracle_lab_args(["--spec", str(spec_path)])
    for key, value in dict(job.get("hp_overrides") or {}).items():
        setattr(args, str(key), value)

    spec = load_equation_spec(spec_path)
    truth_ast = None
    truth_parse_error = ""
    try:
        truth_ast = compile_target_ast(spec)
    except Exception as exc:
        truth_parse_error = str(exc)
    hp = _make_hp(args, budget=budget, refine_enable=(mode == "refine_on"), expert=expert)
    report = run_oracle_equation(
        spec,
        factorized_search_hp=hp,
        seed=rep_seed,
        dtype=dtype,
        enforce_dims=bool(job["enforce_dims"]),
        verbose=bool(job["verbose"]),
    )

    best = report.get("best")
    if isinstance(best, Mapping):
        best_mse = float(best.get("mse", float("inf")))
        best_expr = str(best.get("expr", ""))
        best_expr_ast = _ast_from_jsonable(best.get("expr_ast", None))
        mapping_kind = str(best.get("mapping_kind", ""))
    else:
        best_mse = float("inf")
        best_expr = ""
        best_expr_ast = None
        mapping_kind = ""
    structure = score_structure_match(truth_ast, best_expr_ast)
    solution = classify_solution_outcome(
        best_mse=best_mse,
        truth_ast=truth_ast,
        candidate_ast=best_expr_ast,
        structure=structure,
        numeric_solve_threshold=float(job["structural_solve_mse_threshold"]),
    )
    success = bool(math.isfinite(best_mse) and best_mse <= float(job["success_mse_threshold"]))
    row = {
        "spec_id": str(spec.id),
        "spec_path": str(spec_path),
        "expert": expert,
        "mode": str(mode),
        "budget": int(budget),
        "repeat": int(rep),
        "seed": int(rep_seed),
        "best_mse": float(best_mse),
        "success": int(success),
        "best_expr": best_expr,
        "mapping_kind": mapping_kind,
        "wall_seconds": float(report.get("wall_seconds", float("nan"))),
        "truth_expr": str(node_str(truth_ast)) if isinstance(truth_ast, tuple) and is_valid_node(truth_ast) else str(getattr(spec, "target_expr", "")),
        "truth_ast_available": int(isinstance(truth_ast, tuple) and is_valid_node(truth_ast)),
        "truth_ast_parse_error": truth_parse_error,
        "truth_canonical_expr": str(structure["truth_canonical_expr"]),
        "candidate_canonical_expr": str(structure["candidate_canonical_expr"]),
        "exact_canonical_match": int(bool(structure["exact_canonical_match"])),
        "truth_structure_signature": str(structure["truth_structure_signature"]),
        "candidate_structure_signature": str(structure["candidate_structure_signature"]),
        "exact_structure_signature_match": int(bool(structure["exact_structure_signature_match"])),
        "truth_top_op": str(structure["truth_top_op"]),
        "candidate_top_op": str(structure["candidate_top_op"]),
        "top_level_op_match": int(bool(structure["top_level_op_match"])),
        "structure_ops_hit": int(bool(structure["structure_ops_hit"])),
        "exact_structure_ops_match": int(bool(structure["exact_structure_ops_match"])),
        "truth_var_coverage": int(bool(structure["truth_var_coverage"])),
        "exact_var_set_match": int(bool(structure["exact_var_set_match"])),
        "truth_node_size": int(structure["truth_node_size"]),
        "candidate_node_size": int(structure["candidate_node_size"]),
        "same_node_size": int(bool(structure["same_node_size"])),
        "truth_node_depth": int(structure["truth_node_depth"]),
        "candidate_node_depth": int(structure["candidate_node_depth"]),
        "same_node_depth": int(bool(structure["same_node_depth"])),
        "truth_structure_ops": list(structure["truth_structure_ops"]),
        "candidate_structure_ops": list(structure["candidate_structure_ops"]),
        "truth_var_indices": list(structure["truth_var_indices"]),
        "candidate_var_indices": list(structure["candidate_var_indices"]),
        "structural_solve_mse_threshold": float(job["structural_solve_mse_threshold"]),
        "numeric_solve": int(solution["numeric_solve"]),
        "solution_label": str(solution["solution_label"]),
        "solution_label_reason": str(solution["solution_label_reason"]),
    }
    return {
        "job_index": int(job["job_index"]),
        "row": row,
        "report": report,
    }


def aggregate_rows_by_expert(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("spec_id", "")),
            str(row.get("expert", "")),
            str(row.get("mode", "")),
            int(row.get("budget", 0) or 0),
        )
        groups[key].append(row)
    out: list[dict[str, Any]] = []
    for (spec_id, expert, mode, budget), rs in sorted(groups.items(), key=lambda item: (item[0][0], item[0][3], item[0][2], item[0][1])):
        mse_vals = [float(r.get("best_mse", float("inf"))) for r in rs if math.isfinite(float(r.get("best_mse", float("inf"))))]
        wall_vals = [float(r.get("wall_seconds", float("nan"))) for r in rs if math.isfinite(float(r.get("wall_seconds", float("nan"))))]
        out.append(
            {
                "spec_id": spec_id,
                "expert": expert,
                "mode": mode,
                "budget": int(budget),
                "n_runs": int(len(rs)),
                "solve_rate": _mean([float(r.get("success", 0.0) or 0.0) for r in rs]),
                "numeric_solve_rate": _mean([float(r.get("numeric_solve", 0.0) or 0.0) for r in rs]),
                "structural_solve_rate": _mean(
                    [1.0 if str(r.get("solution_label", "")) == _SOLUTION_LABEL_SOLVE else 0.0 for r in rs]
                ),
                "surrogate_rate": _mean(
                    [1.0 if str(r.get("solution_label", "")) == _SOLUTION_LABEL_SURROGATE else 0.0 for r in rs]
                ),
                "miss_rate": _mean(
                    [1.0 if str(r.get("solution_label", "")) == _SOLUTION_LABEL_MISS else 0.0 for r in rs]
                ),
                "exact_match_rate": _mean([float(r.get("exact_canonical_match", 0.0) or 0.0) for r in rs]),
                "exact_structure_signature_rate": _mean(
                    [float(r.get("exact_structure_signature_match", 0.0) or 0.0) for r in rs]
                ),
                "structure_ops_hit_rate": _mean([float(r.get("structure_ops_hit", 0.0) or 0.0) for r in rs]),
                "truth_var_coverage_rate": _mean([float(r.get("truth_var_coverage", 0.0) or 0.0) for r in rs]),
                "best_mse_median": _median(mse_vals),
                "best_mse_mean": _mean(mse_vals),
                "wall_seconds_mean": _mean(wall_vals),
            }
        )
    return out


def build_portfolio_rows(
    rows: Iterable[dict[str, Any]],
    *,
    experts: Sequence[str],
) -> list[dict[str, Any]]:
    experts_norm = _normalize_experts(experts)
    groups: dict[tuple[str, str, str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("spec_id", "")),
            str(row.get("spec_path", "")),
            str(row.get("mode", "")),
            int(row.get("budget", 0) or 0),
            int(row.get("repeat", 0) or 0),
            int(row.get("seed", 0) or 0),
        )
        groups[key].append(row)

    out: list[dict[str, Any]] = []
    for key, rs in sorted(groups.items(), key=lambda item: item[0]):
        arm_map = {str(r.get("expert", "")): r for r in rs}
        ranked = sorted(
            rs,
            key=lambda row: (
                float(row.get("best_mse", float("inf")) or float("inf")),
                -int(row.get("exact_canonical_match", 0) or 0),
                -int(row.get("exact_structure_signature_match", 0) or 0),
                -int(row.get("structure_ops_hit", 0) or 0),
                str(row.get("expert", "")),
            ),
        )
        solved_ranked = sorted(
            rs,
            key=lambda row: (
                _solution_label_rank(str(row.get("solution_label", ""))),
                float(row.get("best_mse", float("inf")) or float("inf")),
                -int(row.get("exact_canonical_match", 0) or 0),
                -int(row.get("exact_structure_signature_match", 0) or 0),
                -int(row.get("structure_ops_hit", 0) or 0),
                str(row.get("expert", "")),
            ),
        )
        best_mse_row = ranked[0] if ranked else {}
        best_solution_row = solved_ranked[0] if solved_ranked else {}
        structure_pref = [r for r in rs if int(r.get("exact_canonical_match", 0) or 0) > 0]
        if not structure_pref:
            structure_pref = [r for r in rs if int(r.get("exact_structure_signature_match", 0) or 0) > 0]
        if not structure_pref:
            structure_pref = [r for r in rs if int(r.get("structure_ops_hit", 0) or 0) > 0]
        structure_row = sorted(
            structure_pref or ranked,
            key=lambda row: (
                float(row.get("best_mse", float("inf")) or float("inf")),
                str(row.get("expert", "")),
            ),
        )[0]
        row_out = {
            "spec_id": key[0],
            "spec_path": key[1],
            "mode": key[2],
            "budget": int(key[3]),
            "repeat": int(key[4]),
            "seed": int(key[5]),
        }
        for expert in experts_norm:
            arm = arm_map.get(str(expert), {})
            prefix = str(expert)
            row_out[f"{prefix}_mse"] = float(arm.get("best_mse", float("inf")) or float("inf"))
            row_out[f"{prefix}_expr"] = str(arm.get("best_expr", ""))
            row_out[f"{prefix}_numeric_solve"] = int(arm.get("numeric_solve", 0) or 0)
            row_out[f"{prefix}_solution_label"] = str(arm.get("solution_label", _SOLUTION_LABEL_MISS))
            row_out[f"{prefix}_solution_label_reason"] = str(arm.get("solution_label_reason", ""))
            row_out[f"{prefix}_exact_canonical_match"] = int(arm.get("exact_canonical_match", 0) or 0)
            row_out[f"{prefix}_exact_structure_signature_match"] = int(
                arm.get("exact_structure_signature_match", 0) or 0
            )
            row_out[f"{prefix}_structure_ops_hit"] = int(arm.get("structure_ops_hit", 0) or 0)
            row_out[f"{prefix}_wall_seconds"] = float(arm.get("wall_seconds", float("nan")))

        row_out["best_of_portfolio_expert"] = str(best_mse_row.get("expert", ""))
        row_out["best_of_portfolio_mse"] = float(best_mse_row.get("best_mse", float("inf")) or float("inf"))
        row_out["best_of_portfolio_expr"] = str(best_mse_row.get("best_expr", ""))
        row_out["best_of_portfolio_solution_expert"] = str(best_solution_row.get("expert", ""))
        row_out["best_of_portfolio_solution_label"] = str(best_solution_row.get("solution_label", _SOLUTION_LABEL_MISS))
        row_out["best_of_portfolio_solution_mse"] = float(best_solution_row.get("best_mse", float("inf")) or float("inf"))
        row_out["best_of_portfolio_solution_expr"] = str(best_solution_row.get("best_expr", ""))
        row_out["best_of_two_expert"] = row_out["best_of_portfolio_expert"]
        row_out["best_of_two_mse"] = row_out["best_of_portfolio_mse"]
        row_out["best_of_two_expr"] = row_out["best_of_portfolio_expr"]
        row_out["best_of_two_solution_expert"] = row_out["best_of_portfolio_solution_expert"]
        row_out["best_of_two_solution_label"] = row_out["best_of_portfolio_solution_label"]
        row_out["best_of_two_solution_mse"] = row_out["best_of_portfolio_solution_mse"]
        row_out["best_of_two_solution_expr"] = row_out["best_of_portfolio_solution_expr"]
        row_out["any_success"] = int(any(int(r.get("success", 0) or 0) > 0 for r in rs))
        row_out["any_numeric_solve"] = int(any(int(r.get("numeric_solve", 0) or 0) > 0 for r in rs))
        row_out["any_structural_solve"] = int(
            any(str(r.get("solution_label", "")) == _SOLUTION_LABEL_SOLVE for r in rs)
        )
        row_out["any_surrogate"] = int(
            any(str(r.get("solution_label", "")) == _SOLUTION_LABEL_SURROGATE for r in rs)
        )
        row_out["all_miss"] = int(
            all(str(r.get("solution_label", _SOLUTION_LABEL_MISS)) == _SOLUTION_LABEL_MISS for r in rs)
        )
        row_out["any_exact_canonical_match"] = int(any(int(r.get("exact_canonical_match", 0) or 0) > 0 for r in rs))
        row_out["any_exact_structure_signature_match"] = int(
            any(int(r.get("exact_structure_signature_match", 0) or 0) > 0 for r in rs)
        )
        row_out["any_structure_ops_hit"] = int(any(int(r.get("structure_ops_hit", 0) or 0) > 0 for r in rs))
        row_out["structure_preferred_expert"] = str(structure_row.get("expert", ""))
        row_out["structure_preferred_mse"] = float(structure_row.get("best_mse", float("inf")) or float("inf"))
        row_out["structure_preferred_expr"] = str(structure_row.get("best_expr", ""))
        out.append(row_out)
    return out


def aggregate_portfolio_rows(
    rows: Iterable[dict[str, Any]],
    *,
    experts: Sequence[str],
) -> list[dict[str, Any]]:
    experts_norm = _normalize_experts(experts)
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("spec_id", "")),
            str(row.get("mode", "")),
            int(row.get("budget", 0) or 0),
        )
        groups[key].append(row)
    out: list[dict[str, Any]] = []
    for (spec_id, mode, budget), rs in sorted(groups.items(), key=lambda item: (item[0][0], item[0][2], item[0][1])):
        row_out = {
            "spec_id": spec_id,
            "mode": mode,
            "budget": int(budget),
            "n_runs": int(len(rs)),
        }
        for expert in experts_norm:
            prefix = str(expert)
            vals = [
                float(r.get(f"{prefix}_mse", float("inf")))
                for r in rs
                if math.isfinite(float(r.get(f"{prefix}_mse", float("inf"))))
            ]
            row_out[f"{prefix}_mse_median"] = _median(vals)
            if prefix != EXPERT_BASELINE:
                row_out[f"{prefix}_adds_solve_rate"] = _mean(
                    [
                        1.0
                        if str(r.get(f"{EXPERT_BASELINE}_solution_label", "")) != _SOLUTION_LABEL_SOLVE
                        and str(r.get(f"{prefix}_solution_label", "")) == _SOLUTION_LABEL_SOLVE
                        else 0.0
                        for r in rs
                    ]
                )
                row_out[f"{prefix}_loses_solve_rate"] = _mean(
                    [
                        1.0
                        if str(r.get(f"{EXPERT_BASELINE}_solution_label", "")) == _SOLUTION_LABEL_SOLVE
                        and str(r.get(f"{prefix}_solution_label", "")) != _SOLUTION_LABEL_SOLVE
                        else 0.0
                        for r in rs
                    ]
                )
        best_vals = [float(r.get("best_of_portfolio_mse", float("inf"))) for r in rs if math.isfinite(float(r.get("best_of_portfolio_mse", float("inf"))))]
        baseline_vals = [
            float(r.get(f"{EXPERT_BASELINE}_mse", float("inf")))
            for r in rs
            if math.isfinite(float(r.get(f"{EXPERT_BASELINE}_mse", float("inf"))))
        ]
        row_out["best_of_portfolio_mse_median"] = _median(best_vals)
        row_out["best_of_portfolio_beats_baseline_rate"] = _mean(
            [
                1.0
                if float(r.get("best_of_portfolio_mse", float("inf"))) < float(r.get(f"{EXPERT_BASELINE}_mse", float("inf"))) - 1.0e-18
                else 0.0
                for r in rs
            ]
        )
        row_out["best_of_two_mse_median"] = row_out["best_of_portfolio_mse_median"]
        row_out["best_of_two_beats_baseline_rate"] = row_out["best_of_portfolio_beats_baseline_rate"]
        row_out["baseline_mse_median"] = _median(baseline_vals)
        row_out["any_success_rate"] = _mean([float(r.get("any_success", 0.0) or 0.0) for r in rs])
        row_out["any_numeric_solve_rate"] = _mean([float(r.get("any_numeric_solve", 0.0) or 0.0) for r in rs])
        row_out["any_structural_solve_rate"] = _mean([float(r.get("any_structural_solve", 0.0) or 0.0) for r in rs])
        row_out["any_surrogate_rate"] = _mean([float(r.get("any_surrogate", 0.0) or 0.0) for r in rs])
        row_out["all_miss_rate"] = _mean([float(r.get("all_miss", 0.0) or 0.0) for r in rs])
        row_out["any_exact_canonical_match_rate"] = _mean(
            [float(r.get("any_exact_canonical_match", 0.0) or 0.0) for r in rs]
        )
        row_out["any_exact_structure_signature_match_rate"] = _mean(
            [float(r.get("any_exact_structure_signature_match", 0.0) or 0.0) for r in rs]
        )
        row_out["any_structure_ops_hit_rate"] = _mean(
            [float(r.get("any_structure_ops_hit", 0.0) or 0.0) for r in rs]
        )
        out.append(row_out)
    return out


def _build_overall_summary(
    portfolio_rows: Sequence[dict[str, Any]],
    *,
    experts: Sequence[str],
) -> dict[str, Any]:
    experts_norm = _normalize_experts(experts)
    if not portfolio_rows:
        out = {
            "n_portfolio_rows": 0,
            "best_of_portfolio_beats_baseline_count": 0,
            "best_of_portfolio_beats_baseline_rate": 0.0,
            "best_of_two_beats_baseline_count": 0,
            "best_of_two_beats_baseline_rate": 0.0,
            "any_numeric_solve_count": 0,
            "any_numeric_solve_rate": 0.0,
            "any_structural_solve_count": 0,
            "any_structural_solve_rate": 0.0,
            "any_surrogate_count": 0,
            "any_surrogate_rate": 0.0,
            "all_miss_count": 0,
            "all_miss_rate": 0.0,
            "scaffold_adds_solve_count": 0,
            "scaffold_adds_solve_rate": 0.0,
            "scaffold_loses_solve_count": 0,
            "scaffold_loses_solve_rate": 0.0,
            "any_exact_canonical_match_count": 0,
            "any_exact_canonical_match_rate": 0.0,
            "any_exact_structure_signature_match_count": 0,
            "any_exact_structure_signature_match_rate": 0.0,
            "any_structure_ops_hit_count": 0,
            "any_structure_ops_hit_rate": 0.0,
        }
        for expert in experts_norm:
            if expert == EXPERT_BASELINE:
                continue
            out[f"{expert}_adds_solve_count"] = 0
            out[f"{expert}_adds_solve_rate"] = 0.0
            out[f"{expert}_loses_solve_count"] = 0
            out[f"{expert}_loses_solve_rate"] = 0.0
        return out
    n = len(portfolio_rows)
    beat = sum(
        1
        for r in portfolio_rows
        if float(r.get("best_of_portfolio_mse", float("inf"))) < float(r.get(f"{EXPERT_BASELINE}_mse", float("inf"))) - 1.0e-18
    )
    numeric_solve = sum(1 for r in portfolio_rows if int(r.get("any_numeric_solve", 0) or 0) > 0)
    structural_solve = sum(1 for r in portfolio_rows if int(r.get("any_structural_solve", 0) or 0) > 0)
    surrogate = sum(1 for r in portfolio_rows if int(r.get("any_surrogate", 0) or 0) > 0)
    all_miss = sum(1 for r in portfolio_rows if int(r.get("all_miss", 0) or 0) > 0)
    exact = sum(1 for r in portfolio_rows if int(r.get("any_exact_canonical_match", 0) or 0) > 0)
    exact_sig = sum(1 for r in portfolio_rows if int(r.get("any_exact_structure_signature_match", 0) or 0) > 0)
    family = sum(1 for r in portfolio_rows if int(r.get("any_structure_ops_hit", 0) or 0) > 0)
    out = {
        "n_portfolio_rows": int(n),
        "best_of_portfolio_beats_baseline_count": int(beat),
        "best_of_portfolio_beats_baseline_rate": float(beat / float(n)),
        "best_of_two_beats_baseline_count": int(beat),
        "best_of_two_beats_baseline_rate": float(beat / float(n)),
        "any_numeric_solve_count": int(numeric_solve),
        "any_numeric_solve_rate": float(numeric_solve / float(n)),
        "any_structural_solve_count": int(structural_solve),
        "any_structural_solve_rate": float(structural_solve / float(n)),
        "any_surrogate_count": int(surrogate),
        "any_surrogate_rate": float(surrogate / float(n)),
        "all_miss_count": int(all_miss),
        "all_miss_rate": float(all_miss / float(n)),
        "any_exact_canonical_match_count": int(exact),
        "any_exact_canonical_match_rate": float(exact / float(n)),
        "any_exact_structure_signature_match_count": int(exact_sig),
        "any_exact_structure_signature_match_rate": float(exact_sig / float(n)),
        "any_structure_ops_hit_count": int(family),
        "any_structure_ops_hit_rate": float(family / float(n)),
    }
    for expert in experts_norm:
        if expert == EXPERT_BASELINE:
            continue
        adds_solve = sum(
            1
            for r in portfolio_rows
            if str(r.get(f"{EXPERT_BASELINE}_solution_label", "")) != _SOLUTION_LABEL_SOLVE
            and str(r.get(f"{expert}_solution_label", "")) == _SOLUTION_LABEL_SOLVE
        )
        loses_solve = sum(
            1
            for r in portfolio_rows
            if str(r.get(f"{EXPERT_BASELINE}_solution_label", "")) == _SOLUTION_LABEL_SOLVE
            and str(r.get(f"{expert}_solution_label", "")) != _SOLUTION_LABEL_SOLVE
        )
        out[f"{expert}_adds_solve_count"] = int(adds_solve)
        out[f"{expert}_adds_solve_rate"] = float(adds_solve / float(n))
        out[f"{expert}_loses_solve_count"] = int(loses_solve)
        out[f"{expert}_loses_solve_rate"] = float(loses_solve / float(n))
    return out


def run_oracle_portfolio_compare(
    spec_paths: Sequence[str | pathlib.Path],
    *,
    experts: Sequence[str],
    budgets: Sequence[int],
    modes: Sequence[str],
    n_repeats: int,
    seed: int,
    dtype: torch.dtype,
    enforce_dims: bool,
    success_mse_threshold: float,
    verbose: bool,
    structural_solve_mse_threshold: float,
    hp_overrides: argparse.Namespace,
    output_dir: str | pathlib.Path,
    save_individual_reports: bool,
    jobs: int = 1,
    torch_num_threads: int | None = None,
    torch_num_interop_threads: int | None = None,
) -> dict[str, Any]:
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experts_norm = _normalize_experts(experts)
    spec_files = [pathlib.Path(p) for p in spec_paths]
    worker_torch_num_threads, worker_torch_num_interop_threads = _resolve_worker_torch_threads(
        jobs=jobs,
        torch_num_threads=torch_num_threads,
        torch_num_interop_threads=torch_num_interop_threads,
    )
    job_payloads: list[dict[str, Any]] = []
    hp_override_map = dict(vars(hp_overrides))
    dtype_name = "float64" if dtype == torch.float64 else "float32"
    job_index = 0
    for sp in spec_files:
        for budget in budgets:
            for mode in modes:
                for rep in range(int(n_repeats)):
                    rep_seed = int(seed) + int(rep) * 1_000_003
                    for expert in experts_norm:
                        job_payloads.append(
                            {
                                "job_index": int(job_index),
                                "spec_path": str(sp),
                                "budget": int(budget),
                                "mode": str(mode),
                                "repeat": int(rep),
                                "seed": int(rep_seed),
                                "expert": str(expert),
                                "dtype": dtype_name,
                                "enforce_dims": bool(enforce_dims),
                                "verbose": bool(verbose),
                                "success_mse_threshold": float(success_mse_threshold),
                                "structural_solve_mse_threshold": float(structural_solve_mse_threshold),
                                "hp_overrides": hp_override_map,
                                "torch_num_threads": worker_torch_num_threads,
                                "torch_num_interop_threads": worker_torch_num_interop_threads,
                            }
                        )
                        job_index += 1

    rows_by_index: dict[int, dict[str, Any]] = {}
    reports_by_index: dict[int, dict[str, Any]] = {}
    max_workers = max(1, int(jobs))
    if max_workers <= 1 or len(job_payloads) <= 1:
        for job in job_payloads:
            result = _run_portfolio_job(job)
            rows_by_index[int(result["job_index"])] = dict(result["row"])
            reports_by_index[int(result["job_index"])] = dict(result["report"])
    else:
        def _run_parallel(executor_factory) -> None:
            with executor_factory(max_workers=max_workers) as ex:
                futures = [ex.submit(_run_portfolio_job, job) for job in job_payloads]
                for fut in concurrent.futures.as_completed(futures):
                    result = fut.result()
                    rows_by_index[int(result["job_index"])] = dict(result["row"])
                    reports_by_index[int(result["job_index"])] = dict(result["report"])

        try:
            _run_parallel(concurrent.futures.ProcessPoolExecutor)
        except (PermissionError, OSError):
            _run_parallel(concurrent.futures.ThreadPoolExecutor)

    rows: list[dict[str, Any]] = []
    for job in job_payloads:
        idx = int(job["job_index"])
        row = dict(rows_by_index[idx])
        report = dict(reports_by_index[idx])
        report_path = None
        if save_individual_reports:
            ind_dir = out_dir / "individual_reports"
            ind_dir.mkdir(parents=True, exist_ok=True)
            report_path = ind_dir / (
                f"{row['spec_id']}.{row['expert']}.{row['mode']}.n{row['budget']}.r{row['repeat']}.json"
            )
            _write_json(report, report_path)
            row["report_path"] = str(report_path)
        rows.append(row)

    expert_summary = aggregate_rows_by_expert(rows)
    portfolio_rows = build_portfolio_rows(rows, experts=experts_norm)
    portfolio_summary = aggregate_portfolio_rows(portfolio_rows, experts=experts_norm)
    overall = _build_overall_summary(portfolio_rows, experts=experts_norm)

    payload = {
        "n_specs": int(len(spec_files)),
        "spec_paths": [str(p) for p in spec_files],
        "budgets": [int(v) for v in budgets],
        "modes": [str(v) for v in modes],
        "experts": [str(v) for v in experts_norm],
        "n_repeats": int(n_repeats),
        "seed": int(seed),
        "dtype": str(dtype),
        "enforce_dims": bool(enforce_dims),
        "success_mse_threshold": float(success_mse_threshold),
        "structural_solve_mse_threshold": float(structural_solve_mse_threshold),
        "rows": rows,
        "expert_summary": expert_summary,
        "portfolio_rows": portfolio_rows,
        "portfolio_summary": portfolio_summary,
        "overall_summary": overall,
    }

    _write_json(payload, out_dir / "oracle_portfolio_results.json")
    _write_csv(rows, out_dir / "oracle_portfolio_rows.csv")
    _write_csv(expert_summary, out_dir / "oracle_portfolio_expert_summary.csv")
    _write_csv(portfolio_rows, out_dir / "oracle_portfolio_portfolio_rows.csv")
    _write_csv(portfolio_summary, out_dir / "oracle_portfolio_portfolio_summary.csv")
    return payload


def _parse_int_list(raw: str | None, default: Sequence[int]) -> list[int]:
    if raw is None:
        return [int(v) for v in default]
    vals = [int(tok.strip()) for tok in str(raw).split(",") if tok.strip()]
    out = sorted(set(v for v in vals if v > 0))
    if not out:
        raise ValueError("Expected at least one positive budget")
    return out


def _parse_modes(raw: str | None, default: Sequence[str]) -> list[str]:
    if raw is None:
        vals = [str(v) for v in default]
    else:
        vals = [tok.strip().lower() for tok in str(raw).split(",") if tok.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for val in vals:
        if val in {"refine_off", "refine_on"} and val not in seen:
            out.append(val)
            seen.add(val)
    if not out:
        raise ValueError("Expected at least one mode from refine_off,refine_on")
    return out


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare baseline and scaffold oracle experts")
    p.add_argument("--suite_manifest", type=str, default=str(DEFAULT_SUITE_MANIFEST))
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--experts", type=str, default=None)
    p.add_argument("--budgets", type=str, default=None)
    p.add_argument("--modes", type=str, default=None)
    p.add_argument("--n_repeats", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dtype", type=str, choices=["float32", "float64"], default=None)
    p.add_argument("--ignore_dims", action="store_true", default=None)
    p.add_argument("--success_mse", type=float, default=None)
    p.add_argument("--structural_solve_mse", type=float, default=None)
    p.add_argument("--quiet", action="store_true", default=None)
    p.add_argument("--fast_benchmark", action="store_true", default=None)
    p.add_argument("--save_individual_reports", action="store_true")
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--torch_num_threads", type=int, default=None)
    p.add_argument("--torch_num_interop_threads", type=int, default=None)

    p.add_argument("--periodic_scaffold_families", type=str, default="periodic")
    p.add_argument("--exp_scaffold_families", type=str, default="exp")
    p.add_argument("--log_scaffold_families", type=str, default="log")
    p.add_argument("--rational_scaffold_families", type=str, default="rational")
    p.add_argument("--periodic_scaffold_max_scaffolds", type=int, default=16)
    p.add_argument("--periodic_scaffold_anchors_per_family", type=int, default=4)
    p.add_argument("--periodic_scaffold_preview_topk", type=int, default=8)
    p.add_argument("--periodic_scaffold_exact_topk", type=int, default=4)

    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--poly_degree", type=int, default=None)
    p.add_argument("--return_topk", type=int, default=None)
    p.add_argument("--n_fit", type=int, default=None)
    p.add_argument("--n_probe", type=int, default=None)
    p.add_argument("--n_iter", type=int, default=None)
    p.add_argument("--brute_depth", type=int, default=None)
    p.add_argument("--wall_time_limit_s", type=float, default=None)
    p.add_argument("--no_brute_force", action="store_true", default=None)
    p.add_argument("--n_seeds", type=int, default=None)

    split_g = p.add_mutually_exclusive_group()
    split_g.add_argument("--split_iter_across_seeds", dest="split_iter_across_seeds", action="store_true")
    split_g.add_argument("--no_split_iter_across_seeds", dest="split_iter_across_seeds", action="store_false")
    p.set_defaults(split_iter_across_seeds=None)

    p.add_argument("--refine_lbfgs_steps", type=int, default=None)
    p.add_argument("--refine_num_restarts", type=int, default=None)
    p.add_argument("--refine_max_variants", type=int, default=None)
    p.add_argument("--refine_max_params", type=int, default=None)
    linear_g = p.add_mutually_exclusive_group()
    linear_g.add_argument("--refine_linear_combo", dest="refine_linear_combo_enable", action="store_true")
    linear_g.add_argument("--no_refine_linear_combo", dest="refine_linear_combo_enable", action="store_false")
    p.set_defaults(refine_linear_combo_enable=None)
    p.add_argument("--refine_gate_best_factor", type=float, default=None)
    p.add_argument("--refine_max_trials", type=int, default=None)

    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path, manifest = load_regression_suite(args.suite_manifest)
    spec_paths = resolve_suite_spec_paths(manifest, manifest_path=manifest_path)
    defaults = dict(manifest.get("defaults") or {})

    budgets = _parse_int_list(args.budgets, defaults.get("budgets", [1000]))
    modes = _parse_modes(args.modes, defaults.get("modes", ["refine_off"]))
    n_repeats = int(args.n_repeats if args.n_repeats is not None else defaults.get("n_repeats", 1))
    seed = int(args.seed if args.seed is not None else defaults.get("seed", 0))
    dtype_name = str(args.dtype or defaults.get("dtype", "float64")).lower()
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    ignore_dims = bool(args.ignore_dims if args.ignore_dims is not None else defaults.get("ignore_dims", False))
    quiet = bool(args.quiet if args.quiet is not None else defaults.get("quiet", True))
    success_mse = float(args.success_mse if args.success_mse is not None else defaults.get("success_mse", 1.0e-6))
    raw_experts = args.experts
    if raw_experts is None:
        raw_experts = defaults.get("experts", ",".join(DEFAULT_EXPERTS))
    experts = _normalize_experts([tok.strip() for tok in str(raw_experts).split(",") if tok.strip()])
    structural_solve_mse = float(
        args.structural_solve_mse
        if args.structural_solve_mse is not None
        else defaults.get("structural_solve_mse", DEFAULT_STRUCTURAL_SOLVE_MSE_THRESHOLD)
    )
    fast_benchmark = bool(args.fast_benchmark if args.fast_benchmark is not None else defaults.get("fast_benchmark", False))
    if args.wall_time_limit_s is None and defaults.get("wall_time_limit_s", None) is not None:
        args.wall_time_limit_s = float(defaults.get("wall_time_limit_s"))
    if fast_benchmark:
        args.no_brute_force = True
    if args.n_seeds is None and defaults.get("n_seeds", None) is not None:
        args.n_seeds = int(defaults.get("n_seeds"))
    if args.split_iter_across_seeds is None and defaults.get("split_iter_across_seeds", None) is not None:
        args.split_iter_across_seeds = bool(defaults.get("split_iter_across_seeds"))

    args.periodic_scaffold_families = [
        tok.strip() for tok in str(args.periodic_scaffold_families or "periodic").split(",") if tok.strip()
    ]
    args.exp_scaffold_families = [
        tok.strip() for tok in str(args.exp_scaffold_families or "exp").split(",") if tok.strip()
    ]
    args.log_scaffold_families = [
        tok.strip() for tok in str(args.log_scaffold_families or "log").split(",") if tok.strip()
    ]
    args.rational_scaffold_families = [
        tok.strip() for tok in str(args.rational_scaffold_families or "rational").split(",") if tok.strip()
    ]

    suite_id = str(manifest.get("suite_id", "portfolio_suite") or "portfolio_suite")
    output_dir = pathlib.Path(args.output_dir or (REPO_ROOT / "results" / f"oracle_portfolio_{suite_id}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = run_oracle_portfolio_compare(
        spec_paths,
        experts=experts,
        budgets=budgets,
        modes=modes,
        n_repeats=n_repeats,
        seed=seed,
        dtype=dtype,
        enforce_dims=not ignore_dims,
        success_mse_threshold=success_mse,
        verbose=not quiet,
        structural_solve_mse_threshold=structural_solve_mse,
        hp_overrides=args,
        output_dir=output_dir,
        save_individual_reports=bool(args.save_individual_reports),
        jobs=int(args.jobs),
        torch_num_threads=args.torch_num_threads,
        torch_num_interop_threads=args.torch_num_interop_threads,
    )

    overall = dict(payload.get("overall_summary", {}) or {})
    print(
        f"[portfolio] suite={suite_id} specs={len(spec_paths)} experts={experts} budgets={budgets} modes={modes} "
        f"best_of_portfolio_beats_baseline={int(overall.get('best_of_portfolio_beats_baseline_count', 0))}/"
        f"{int(overall.get('n_portfolio_rows', 0))}"
    )
    print(f"[portfolio] outputs written to {output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
