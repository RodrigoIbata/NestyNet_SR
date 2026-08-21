# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Single-hole micro-search toy for inverse-guided symbolic completion.

This module is intentionally narrower than the production factorized symbolic search explorer.
It keeps the inverse-steering semantics, typed tuple-ASTs, and exact scoring
against oracle data, while stripping away the large grammar, mutation archive,
and Stage-B integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .config import FactorizedSearchConfig
from .expr_ast import (
    BINARY_OPS,
    UNARY_OPS,
    collect_paths,
    dims_eq,
    eval_node,
    get_at,
    node_depth,
    node_dims,
    node_size,
    node_str,
    replace_at,
)
from .explorer import eval_mapping
from .oracle_lab import (
    EquationSpec,
    _finite_mask,
    _mask_fraction,
    _prepare_local_target_slices,
    _score_expr_against_target,
    _score_node_on_local_target,
    _to_jsonable,
    build_candidate_ast_for_inverse_lab,
    build_oracle_dataset,
    compile_target_ast,
    compile_target_expression,
    default_oracle_hyperparams,
    invert_context_target,
    load_equation_spec,
)


_COMMUTATIVE_BINARY_OPS = frozenset({"add", "mul"})
DEFAULT_MICRO_SEARCH_SPLITS = ("train", "val", "test")
DEFAULT_MICRO_SEARCH_SPLIT_FRACTIONS = (0.7, 0.15, 0.15)


@dataclass(frozen=True)
class MicroSearchGrammar:
    """Tiny grammar used by the single-hole micro-search environment."""

    max_depth: int = 2
    unary_ops: tuple[str, ...] = ()
    binary_ops: tuple[str, ...] = ("add", "mul")
    constant_values: tuple[float, ...] = ()


def _normalize_path(path: str | Sequence[int] | None) -> tuple[int, ...] | None:
    if path is None:
        return None
    if isinstance(path, str):
        text = path.strip()
        if text in ("", "root"):
            return ()
        return tuple(int(tok) for tok in text.split("/") if tok.strip())
    return tuple(int(v) for v in path)


def _format_path(path: Sequence[int] | None) -> str:
    if path is None:
        return "none"
    pp = tuple(int(v) for v in path)
    if not pp:
        return "root"
    return "/".join(str(v) for v in pp)


def _stable_id(*parts: Any) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(str(part).encode("utf-8", errors="ignore"))
        digest.update(b"\x1f")
    return digest.hexdigest()[:16]


def _normalize_split_fractions(split_fractions: Sequence[float] | None) -> tuple[float, float, float]:
    raw = DEFAULT_MICRO_SEARCH_SPLIT_FRACTIONS if split_fractions is None else split_fractions
    if len(raw) != 3:
        raise ValueError(f"Expected 3 split fractions for train/val/test, got {len(raw)}")
    vals = [max(0.0, float(v)) for v in raw]
    total = float(sum(vals))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"Invalid split fractions: {raw!r}")
    return (vals[0] / total, vals[1] / total, vals[2] / total)


def assign_micro_search_split(
    split_key: Any,
    *,
    split_fractions: Sequence[float] | None = None,
    split_names: Sequence[str] = DEFAULT_MICRO_SEARCH_SPLITS,
) -> str:
    """Assign a deterministic train/val/test split from a stable key."""

    names = tuple(str(name) for name in split_names)
    if len(names) != 3:
        raise ValueError(f"Expected exactly 3 split names, got {names!r}")
    fractions = _normalize_split_fractions(split_fractions)
    token = int.from_bytes(hashlib.blake2b(str(split_key).encode("utf-8"), digest_size=8).digest(), "big")
    u = float(token) / float((1 << 64) - 1)
    if u < fractions[0]:
        return names[0]
    if u < fractions[0] + fractions[1]:
        return names[1]
    return names[2]


def _dataset_split_key(*, spec_id: str, seed: int, state_id: str, split_unit: str) -> str:
    unit = str(split_unit or "spec").strip().lower()
    if unit == "spec":
        return str(spec_id)
    if unit == "spec_seed":
        return f"{str(spec_id)}::seed::{int(seed)}"
    if unit == "state":
        return str(state_id)
    raise ValueError(f"Unsupported split_unit {split_unit!r}; expected one of 'spec', 'spec_seed', 'state'")


def _default_hole_path(node: tuple) -> tuple[int, ...]:
    paths = [tuple(int(v) for v in p) for p in collect_paths(node) if tuple(int(v) for v in p)]
    if not paths:
        return ()
    return max(paths, key=lambda p: (len(p), p))


def _normalize_grammar(grammar: MicroSearchGrammar | None) -> MicroSearchGrammar:
    grammar = MicroSearchGrammar() if grammar is None else grammar
    unary_ops: list[str] = []
    seen_unary: set[str] = set()
    for op in grammar.unary_ops:
        name = str(op).strip()
        if name == "":
            continue
        if name not in UNARY_OPS:
            raise ValueError(f"Unsupported unary op {name!r}; allowed={UNARY_OPS}")
        if name not in seen_unary:
            unary_ops.append(name)
            seen_unary.add(name)

    binary_ops: list[str] = []
    seen_binary: set[str] = set()
    for op in grammar.binary_ops:
        name = str(op).strip()
        if name == "":
            continue
        if name not in BINARY_OPS:
            raise ValueError(f"Unsupported binary op {name!r}; allowed={BINARY_OPS}")
        if name not in seen_binary:
            binary_ops.append(name)
            seen_binary.add(name)

    consts: list[float] = []
    seen_consts: set[float] = set()
    for raw in grammar.constant_values:
        value = float(raw)
        if value not in seen_consts:
            consts.append(value)
            seen_consts.add(value)

    if int(grammar.max_depth) < 1:
        raise ValueError(f"grammar.max_depth must be >= 1, got {grammar.max_depth!r}")

    return MicroSearchGrammar(
        max_depth=int(grammar.max_depth),
        unary_ops=tuple(unary_ops),
        binary_ops=tuple(binary_ops),
        constant_values=tuple(consts),
    )


def _canonicalize_binary(op: str, left: tuple, right: tuple) -> tuple:
    if op in _COMMUTATIVE_BINARY_OPS and node_str(left) > node_str(right):
        left, right = right, left
    return (op, left, right)


def enumerate_micro_search_expressions(
    nvars: int,
    *,
    grammar: MicroSearchGrammar | None = None,
    var_dims: Sequence[Sequence[float]] | None = None,
    target_dim: Sequence[float] | None = None,
) -> list[tuple]:
    """Enumerate all unique expressions in a tiny grammar up to ``max_depth``."""

    grammar_eff = _normalize_grammar(grammar)
    by_depth: dict[int, list[tuple]] = {}
    dim_cache: dict[str, tuple[float, ...] | None] = {}
    seen: set[str] = set()
    all_nodes: list[tuple] = []

    def _maybe_add(node: tuple, depth: int) -> None:
        if var_dims is not None:
            nd = node_dims(node, var_dims)
            if nd is None:
                return
        key = node_str(node)
        if key in seen:
            return
        seen.add(key)
        if var_dims is not None:
            dim_cache[key] = tuple(float(v) for v in nd)
        by_depth.setdefault(depth, []).append(node)
        all_nodes.append(node)

    for var_idx in range(int(nvars)):
        _maybe_add(("var", int(var_idx)), depth=1)
    for value in grammar_eff.constant_values:
        _maybe_add(("const", float(value)), depth=1)

    for depth in range(2, int(grammar_eff.max_depth) + 1):
        for op in grammar_eff.unary_ops:
            for child in by_depth.get(depth - 1, []):
                _maybe_add((op, child), depth=depth)
        for op in grammar_eff.binary_ops:
            for left_depth in range(1, depth):
                for right_depth in range(1, depth):
                    if 1 + max(left_depth, right_depth) != depth:
                        continue
                    for left in by_depth.get(left_depth, []):
                        for right in by_depth.get(right_depth, []):
                            _maybe_add(_canonicalize_binary(op, left, right), depth=depth)
    if target_dim is None or var_dims is None:
        return all_nodes
    target_dim_eff = tuple(float(v) for v in target_dim)
    return [node for node in all_nodes if dims_eq(dim_cache.get(node_str(node), None), target_dim_eff)]


def _score_completion_table(
    *,
    candidate_ast: tuple,
    truth_node: tuple | None,
    hole_path: Sequence[int],
    completions: Sequence[tuple],
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    t_fit: torch.Tensor,
    mask_fit: torch.Tensor,
    t_probe: torch.Tensor,
    mask_probe: torch.Tensor,
    poly_degree: int,
) -> list[dict[str, Any]]:
    xf, tf, xp, tp = _prepare_local_target_slices(
        x_fit,
        t_fit,
        mask_fit,
        x_probe,
        t_probe,
        mask_probe,
    )
    if int(xf.shape[0]) < 4 or int(xp.shape[0]) < 4:
        return []

    path_t = tuple(int(v) for v in hole_path)
    rows: list[dict[str, Any]] = []
    for node in completions:
        local = _score_node_on_local_target(
            node,
            x_fit=xf,
            t_fit=tf,
            x_probe=xp,
            t_probe=tp,
            poly_degree=int(poly_degree),
            local_score_mode="affine",
        )
        if local is None:
            continue

        repaired_ast = replace_at(candidate_ast, path_t, node)
        full = _score_expr_against_target(
            repaired_ast,
            x_fit=x_fit,
            y_fit=y_fit,
            x_probe=x_probe,
            y_probe=y_probe,
            poly_degree=int(poly_degree),
        )
        if full is None:
            continue

        row = {
            "expr": node_str(node),
            "expr_ast": node,
            "expr_depth": int(node_depth(node)),
            "expr_size": int(node_size(node)),
            "is_truth": bool(truth_node is not None and node == truth_node),
            "local_probe_mse": float(local["local_probe_mse"]),
            "local_fit_mse": float(local["local_fit_mse"]),
            "local_corr_probe": float(local["local_corr_probe"]),
            "local_mapping": local["local_mapping"],
            "local_mapping_kind": str(local["local_mapping_kind"]),
            "full_probe_mse": float(full["probe_mse"]),
            "full_fit_mse": float(full["fit_mse"]),
            "full_mapping": full["mapping"],
            "full_mapping_kind": str(full["mapping_kind"]),
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            float(row["local_probe_mse"]),
            float(row["full_probe_mse"]),
            int(row["expr_size"]),
            str(row["expr"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = int(rank)
    return rows


def _budget_metrics(
    rows: Sequence[dict[str, Any]],
    *,
    solve_threshold: float,
    budgets: Sequence[int],
) -> dict[str, Any]:
    truth_row = next((row for row in rows if bool(row.get("is_truth", False))), None)
    best_row = rows[0] if rows else None

    metrics: dict[str, Any] = {
        "n_ranked": int(len(rows)),
        "truth_rank": None if truth_row is None else int(truth_row["rank"]),
        "truth_expr": None if truth_row is None else str(truth_row["expr"]),
        "truth_local_probe_mse": None if truth_row is None else float(truth_row["local_probe_mse"]),
        "truth_full_probe_mse": None if truth_row is None else float(truth_row["full_probe_mse"]),
        "best_expr": None if best_row is None else str(best_row["expr"]),
        "best_local_probe_mse": None if best_row is None else float(best_row["local_probe_mse"]),
        "best_full_probe_mse": None if best_row is None else float(best_row["full_probe_mse"]),
        "solve_threshold": float(solve_threshold),
        "truth_seen_at_budget": {},
        "solve_at_budget": {},
        "best_full_probe_mse_at_budget": {},
        "best_expr_at_budget": {},
    }
    for budget in budgets:
        budget_eff = max(1, int(budget))
        prefix = list(rows[:budget_eff])
        truth_seen = truth_row is not None and int(truth_row["rank"]) <= budget_eff
        best_prefix = None if not prefix else min(prefix, key=lambda row: float(row["full_probe_mse"]))
        solved = bool(best_prefix is not None and float(best_prefix["full_probe_mse"]) <= float(solve_threshold))
        metrics["truth_seen_at_budget"][budget_eff] = bool(truth_seen)
        metrics["solve_at_budget"][budget_eff] = bool(solved)
        metrics["best_full_probe_mse_at_budget"][budget_eff] = (
            None if best_prefix is None else float(best_prefix["full_probe_mse"])
        )
        metrics["best_expr_at_budget"][budget_eff] = None if best_prefix is None else str(best_prefix["expr"])
    return metrics


def run_single_hole_micro_search(
    spec: EquationSpec,
    *,
    factorized_search_hp: FactorizedSearchConfig | None = None,
    seed: int | None = None,
    dtype: torch.dtype = torch.float64,
    enforce_dims: bool = True,
    candidate_expr: str | None = None,
    corrupt_path: str | Sequence[int] | None = None,
    replacement_expr: str | None = None,
    hole_path: str | Sequence[int] | None = None,
    grammar: MicroSearchGrammar | None = None,
    solve_threshold: float = 1.0e-12,
    budget_ladder: Sequence[int] = (1, 3, 5, 10),
    report_topk: int | None = 16,
    include_samples: bool = False,
) -> dict[str, Any]:
    """Run the single-hole micro-search toy and return an exact oracle report."""

    hp = default_oracle_hyperparams() if factorized_search_hp is None else factorized_search_hp
    run_seed = int(hp.seed if seed is None else seed)
    target_fn = compile_target_expression(spec)
    ds = build_oracle_dataset(
        spec,
        target_fn,
        n_fit=int(hp.n_fit),
        n_probe=int(hp.n_probe),
        seed=run_seed,
        dtype=dtype,
    )
    var_dims = ds["var_dims"] if enforce_dims else None

    candidate_ast, truth_ast, used_corrupt_path = build_candidate_ast_for_inverse_lab(
        spec,
        candidate_expr=candidate_expr,
        corrupt_path=corrupt_path,
        replacement_expr=replacement_expr,
        var_dims=var_dims,
    )
    hole_path_eff = _normalize_path(hole_path)
    if hole_path_eff is None:
        if used_corrupt_path is not None:
            hole_path_eff = tuple(int(v) for v in used_corrupt_path)
        else:
            hole_path_eff = _default_hole_path(candidate_ast)

    current_node = get_at(candidate_ast, hole_path_eff)
    truth_paths = {tuple(int(v) for v in p) for p in collect_paths(truth_ast)}
    truth_node = get_at(truth_ast, hole_path_eff) if hole_path_eff in truth_paths else None
    truth_target_dim = None if truth_node is None or var_dims is None else node_dims(truth_node, var_dims)
    current_target_dim = None if var_dims is None else node_dims(current_node, var_dims)
    target_dim = truth_target_dim if truth_target_dim is not None else current_target_dim

    base = _score_expr_against_target(
        candidate_ast,
        x_fit=ds["x_fit"],
        y_fit=ds["y_fit"],
        x_probe=ds["x_probe"],
        y_probe=ds["y_probe"],
        poly_degree=int(hp.poly_degree),
    )
    if base is None:
        raise RuntimeError(f"candidate AST could not be scored: {node_str(candidate_ast)}")

    inv_fit = invert_context_target(
        candidate_ast,
        hole_path_eff,
        ds["x_fit"],
        ds["y_fit"],
        mapping=base["mapping"],
        safe_eps=float(getattr(hp, "inverse_safe_eps", 1.0e-12) or 1.0e-12),
        allow_identity_fallback=True,
        confidence_mode=str(getattr(hp, "inverse_confidence_mode", "conditioning")),
        confidence_target_gain=float(getattr(hp, "inverse_confidence_target_gain", 4.0)),
        confidence_floor=float(getattr(hp, "inverse_confidence_floor", 0.05)),
        branch_beam_width=max(1, int(getattr(hp, "inverse_branch_beam_width", 1))),
    )
    inv_probe = invert_context_target(
        candidate_ast,
        hole_path_eff,
        ds["x_probe"],
        ds["y_probe"],
        mapping=base["mapping"],
        safe_eps=float(getattr(hp, "inverse_safe_eps", 1.0e-12) or 1.0e-12),
        allow_identity_fallback=True,
        confidence_mode=str(getattr(hp, "inverse_confidence_mode", "conditioning")),
        confidence_target_gain=float(getattr(hp, "inverse_confidence_target_gain", 4.0)),
        confidence_floor=float(getattr(hp, "inverse_confidence_floor", 0.05)),
        branch_beam_width=max(1, int(getattr(hp, "inverse_branch_beam_width", 1))),
    )

    pred_fit = eval_node(candidate_ast, ds["x_fit"])
    pred_probe = eval_node(candidate_ast, ds["x_probe"])
    y_hat_fit = eval_mapping(pred_fit, base["mapping"])
    y_hat_probe = eval_mapping(pred_probe, base["mapping"])
    resid_fit = ds["y_fit"] - y_hat_fit
    resid_probe = ds["y_probe"] - y_hat_probe
    resid_mask_fit = _finite_mask(resid_fit)
    resid_mask_probe = _finite_mask(resid_probe)

    grammar_eff = _normalize_grammar(grammar)
    completions = enumerate_micro_search_expressions(
        int(ds["x_fit"].shape[1]),
        grammar=grammar_eff,
        var_dims=var_dims,
        target_dim=target_dim,
    )

    inverse_rows_all = _score_completion_table(
        candidate_ast=candidate_ast,
        truth_node=truth_node,
        hole_path=hole_path_eff,
        completions=completions,
        x_fit=ds["x_fit"],
        y_fit=ds["y_fit"],
        x_probe=ds["x_probe"],
        y_probe=ds["y_probe"],
        t_fit=inv_fit.target,
        mask_fit=inv_fit.valid_mask,
        t_probe=inv_probe.target,
        mask_probe=inv_probe.valid_mask,
        poly_degree=int(hp.poly_degree),
    )
    residual_rows_all = _score_completion_table(
        candidate_ast=candidate_ast,
        truth_node=truth_node,
        hole_path=hole_path_eff,
        completions=completions,
        x_fit=ds["x_fit"],
        y_fit=ds["y_fit"],
        x_probe=ds["x_probe"],
        y_probe=ds["y_probe"],
        t_fit=resid_fit,
        mask_fit=resid_mask_fit,
        t_probe=resid_probe,
        mask_probe=resid_mask_probe,
        poly_degree=int(hp.poly_degree),
    )

    truth_repair = None
    if truth_node is not None:
        truth_repair = _score_expr_against_target(
            replace_at(candidate_ast, hole_path_eff, truth_node),
            x_fit=ds["x_fit"],
            y_fit=ds["y_fit"],
            x_probe=ds["x_probe"],
            y_probe=ds["y_probe"],
            poly_degree=int(hp.poly_degree),
        )

    inverse_metrics = _budget_metrics(
        inverse_rows_all,
        solve_threshold=float(solve_threshold),
        budgets=budget_ladder,
    )
    residual_metrics = _budget_metrics(
        residual_rows_all,
        solve_threshold=float(solve_threshold),
        budgets=budget_ladder,
    )

    if report_topk is None:
        inverse_rows_report = inverse_rows_all
        residual_rows_report = residual_rows_all
    else:
        topk_eff = max(1, int(report_topk))
        inverse_rows_report = inverse_rows_all[:topk_eff]
        residual_rows_report = residual_rows_all[:topk_eff]

    inverse_truth_rank = inverse_metrics.get("truth_rank", None)
    residual_truth_rank = residual_metrics.get("truth_rank", None)
    truth_rank_gap = None
    if inverse_truth_rank is not None and residual_truth_rank is not None:
        truth_rank_gap = int(residual_truth_rank) - int(inverse_truth_rank)

    report = {
        "mode": "micro_search_single_hole",
        "spec_id": str(spec.id),
        "target_expr": str(spec.target_expr),
        "truth_expr": node_str(truth_ast),
        "truth_expr_ast": truth_ast,
        "candidate_expr": node_str(candidate_ast),
        "candidate_expr_ast": candidate_ast,
        "hole_path": [int(v) for v in hole_path_eff],
        "hole_path_str": _format_path(hole_path_eff),
        "hole_current_expr": node_str(current_node),
        "hole_current_expr_ast": current_node,
        "hole_truth_expr": None if truth_node is None else node_str(truth_node),
        "hole_truth_expr_ast": truth_node,
        "candidate_score": {
            "fit_mse": float(base["fit_mse"]),
            "probe_mse": float(base["probe_mse"]),
            "mapping": base["mapping"],
            "mapping_kind": str(base["mapping_kind"]),
        },
        "inverse_target": {
            "confidence": float(inv_fit.confidence),
            "mapping_inverted": bool(inv_fit.mapping_inverted),
            "mapping_kind": str(inv_fit.mapping_kind),
            "valid_fraction_fit": _mask_fraction(inv_fit.valid_mask),
            "valid_fraction_probe": _mask_fraction(inv_probe.valid_mask),
            "steps": [
                {
                    "parent_path": [int(v) for v in step.parent_path],
                    "parent_path_str": _format_path(step.parent_path),
                    "op": str(step.op),
                    "child_slot": int(step.child_slot),
                    "valid_fraction": float(step.valid_fraction),
                    "confidence": float(step.confidence),
                    "note": str(step.note),
                }
                for step in inv_fit.steps
            ],
        },
        "residual_target": {
            "valid_fraction_fit": _mask_fraction(resid_mask_fit),
            "valid_fraction_probe": _mask_fraction(resid_mask_probe),
        },
        "grammar": {
            "max_depth": int(grammar_eff.max_depth),
            "unary_ops": [str(op) for op in grammar_eff.unary_ops],
            "binary_ops": [str(op) for op in grammar_eff.binary_ops],
            "constant_values": [float(v) for v in grammar_eff.constant_values],
            "n_candidates": int(len(completions)),
            "truth_in_grammar": bool(truth_node is not None and any(node == truth_node for node in completions)),
            "current_in_grammar": bool(any(node == current_node for node in completions)),
            "target_dim": None if target_dim is None else [float(v) for v in target_dim],
        },
        "units": {
            "var_dims": [[float(v) for v in dim] for dim in list(ds["var_dims"] or ())],
            "y_dims": [float(v) for v in list(ds["y_dims"] or ())],
            "hole_dim": None if target_dim is None else [float(v) for v in target_dim],
        },
        "oracle": {
            "truth_repair_full_expr": truth_repair,
            "inverse_truth_rank_advantage": truth_rank_gap,
        },
        "metrics": {
            "inverse": inverse_metrics,
            "residual": residual_metrics,
        },
        "rankings": {
            "inverse": inverse_rows_report,
            "residual": residual_rows_report,
        },
        "hp": {
            "n_fit": int(hp.n_fit),
            "n_probe": int(hp.n_probe),
            "poly_degree": int(hp.poly_degree),
            "seed": int(run_seed),
            "enforce_dims": bool(enforce_dims),
        },
    }
    if include_samples:
        current_fit = eval_node(current_node, ds["x_fit"])
        current_probe = eval_node(current_node, ds["x_probe"])
        samples = {
            "x_fit": ds["x_fit"],
            "x_probe": ds["x_probe"],
            "y_fit": ds["y_fit"],
            "y_probe": ds["y_probe"],
            "hole_current_fit": current_fit,
            "hole_current_probe": current_probe,
            "inverse_target_fit": inv_fit.target,
            "inverse_target_probe": inv_probe.target,
            "inverse_valid_mask_fit": inv_fit.valid_mask,
            "inverse_valid_mask_probe": inv_probe.valid_mask,
            "residual_target_fit": resid_fit,
            "residual_target_probe": resid_probe,
            "residual_valid_mask_fit": resid_mask_fit,
            "residual_valid_mask_probe": resid_mask_probe,
        }
        if truth_node is not None:
            samples["hole_truth_fit"] = eval_node(truth_node, ds["x_fit"])
            samples["hole_truth_probe"] = eval_node(truth_node, ds["x_probe"])
        report["samples"] = samples
    return _to_jsonable(report)


def generate_micro_search_dataset(
    specs: Sequence[EquationSpec | str | pathlib.Path],
    *,
    factorized_search_hp: FactorizedSearchConfig | None = None,
    seeds: Sequence[int] = (0,),
    dtype: torch.dtype = torch.float64,
    enforce_dims: bool = True,
    depth_min: int = 2,
    depth_max: int = 8,
    max_corrupt_paths_per_spec: int | None = None,
    grammar: MicroSearchGrammar | None = None,
    solve_threshold: float = 1.0e-12,
    budget_ladder: Sequence[int] = (1, 3, 5, 10),
    split_unit: str = "spec",
    split_fractions: Sequence[float] | None = None,
    include_samples: bool = True,
    include_completion_tables: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Generate a deterministic micro-search dataset over corrupted truth holes."""

    spec_objs: list[EquationSpec] = []
    for item in specs:
        if isinstance(item, EquationSpec):
            spec_objs.append(item)
        else:
            spec_objs.append(load_equation_spec(item))

    grammar_eff = _normalize_grammar(grammar)
    split_fracs = _normalize_split_fractions(split_fractions)
    rows: list[dict[str, Any]] = []
    skipped_specs: list[dict[str, Any]] = []

    for spec in spec_objs:
        truth_ast = compile_target_ast(spec)
        truth_depth = int(node_depth(truth_ast))
        if truth_depth < int(depth_min) or truth_depth > int(depth_max):
            skipped_specs.append({
                "spec_id": str(spec.id),
                "truth_depth": int(truth_depth),
                "reason": "truth_depth_out_of_range",
            })
            continue

        corrupt_paths = [tuple(int(v) for v in p) for p in collect_paths(truth_ast) if tuple(int(v) for v in p)]
        corrupt_paths.sort(key=lambda p: (len(p), tuple(p)))
        if max_corrupt_paths_per_spec is not None:
            corrupt_paths = corrupt_paths[: max(1, int(max_corrupt_paths_per_spec))]

        for seed in seeds:
            for corrupt_path in corrupt_paths:
                state_id = _stable_id(
                    "micro_search",
                    spec.id,
                    int(seed),
                    _format_path(corrupt_path),
                    grammar_eff.max_depth,
                    grammar_eff.unary_ops,
                    grammar_eff.binary_ops,
                    grammar_eff.constant_values,
                    int(enforce_dims),
                )
                split_key = _dataset_split_key(
                    spec_id=str(spec.id),
                    seed=int(seed),
                    state_id=state_id,
                    split_unit=str(split_unit),
                )
                report = run_single_hole_micro_search(
                    spec,
                    factorized_search_hp=factorized_search_hp,
                    seed=int(seed),
                    dtype=dtype,
                    enforce_dims=bool(enforce_dims),
                    corrupt_path=corrupt_path,
                    grammar=grammar_eff,
                    solve_threshold=float(solve_threshold),
                    budget_ladder=budget_ladder,
                    report_topk=None if include_completion_tables else 16,
                    include_samples=bool(include_samples),
                )
                if not include_completion_tables:
                    report["rankings"] = {"inverse": [], "residual": []}
                report["state_id"] = str(state_id)
                report["split"] = assign_micro_search_split(
                    split_key,
                    split_fractions=split_fracs,
                    split_names=DEFAULT_MICRO_SEARCH_SPLITS,
                )
                report["split_key"] = str(split_key)
                report["split_unit"] = str(split_unit)
                report["truth_depth"] = int(truth_depth)
                report["corrupt_path"] = [int(v) for v in corrupt_path]
                report["corrupt_path_str"] = _format_path(corrupt_path)
                rows.append(report)
                if verbose:
                    print(
                        f"[micro-search-dataset] {spec.id} seed={int(seed)} hole={_format_path(corrupt_path)} "
                        f"split={report['split']} candidates={int(report['grammar']['n_candidates'])}"
                    )

    rows.sort(key=lambda row: (str(row.get("split", "")), str(row.get("spec_id", "")), int(row.get("seed", 0)), str(row.get("corrupt_path_str", ""))))
    split_counts = {
        split: int(sum(1 for row in rows if str(row.get("split", "")) == split))
        for split in DEFAULT_MICRO_SEARCH_SPLITS
    }
    truth_in_grammar_counts = {
        split: int(
            sum(
                1
                for row in rows
                if str(row.get("split", "")) == split and bool((row.get("grammar", {}) or {}).get("truth_in_grammar", False))
            )
        )
        for split in DEFAULT_MICRO_SEARCH_SPLITS
    }
    return _to_jsonable({
        "mode": "micro_search_dataset",
        "n_rows": int(len(rows)),
        "config": {
            "seeds": [int(s) for s in seeds],
            "depth_min": int(depth_min),
            "depth_max": int(depth_max),
            "max_corrupt_paths_per_spec": None if max_corrupt_paths_per_spec is None else int(max_corrupt_paths_per_spec),
            "grammar": {
                "max_depth": int(grammar_eff.max_depth),
                "unary_ops": [str(op) for op in grammar_eff.unary_ops],
                "binary_ops": [str(op) for op in grammar_eff.binary_ops],
                "constant_values": [float(v) for v in grammar_eff.constant_values],
            },
            "solve_threshold": float(solve_threshold),
            "budget_ladder": [int(v) for v in budget_ladder],
            "split_unit": str(split_unit),
            "split_fractions": [float(v) for v in split_fracs],
            "include_samples": bool(include_samples),
            "include_completion_tables": bool(include_completion_tables),
            "enforce_dims": bool(enforce_dims),
        },
        "split_counts": split_counts,
        "truth_in_grammar_counts": truth_in_grammar_counts,
        "rows": rows,
        "skipped_specs": skipped_specs,
    })


def _parse_ops(raw: str) -> tuple[str, ...]:
    return tuple(tok.strip() for tok in str(raw).split(",") if tok.strip())


def _parse_constants(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for tok in str(raw).split(","):
        text = tok.strip()
        if text == "":
            continue
        values.append(float(text))
    return tuple(values)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-hole micro-search toy runner")
    parser.add_argument("--spec", required=True, help="Equation spec JSON/YAML file")
    parser.add_argument("--candidate_expr", default=None, help="Optional candidate expression")
    parser.add_argument("--corrupt_path", default=None, help="Truth path to corrupt, e.g. 1/2")
    parser.add_argument("--replacement_expr", default=None, help="Replacement expression for --corrupt_path")
    parser.add_argument("--hole_path", default=None, help="Hole path to solve; defaults to corrupt path")
    parser.add_argument("--max_depth", type=int, default=2, help="Grammar max depth")
    parser.add_argument("--unary_ops", default="", help="Comma-separated unary ops, e.g. exp,sin")
    parser.add_argument("--binary_ops", default="add,mul", help="Comma-separated binary ops")
    parser.add_argument("--const_values", default="", help="Comma-separated literal constants")
    parser.add_argument("--report_topk", type=int, default=16, help="How many ranked rows to emit")
    parser.add_argument("--n_fit", type=int, default=None, help="Override fit sample count")
    parser.add_argument("--n_probe", type=int, default=None, help="Override probe sample count")
    parser.add_argument("--poly_degree", type=int, default=None, help="Override mapping poly degree")
    parser.add_argument("--seed", type=int, default=0, help="Oracle sampling seed")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument(
        "--no_enforce_dims",
        action="store_true",
        help="Disable dimensional filtering during enumeration",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    spec = load_equation_spec(args.spec)
    hp = default_oracle_hyperparams()
    if args.n_fit is not None:
        hp.n_fit = int(args.n_fit)
    if args.n_probe is not None:
        hp.n_probe = int(args.n_probe)
    if args.poly_degree is not None:
        hp.poly_degree = int(args.poly_degree)

    grammar = MicroSearchGrammar(
        max_depth=int(args.max_depth),
        unary_ops=_parse_ops(args.unary_ops),
        binary_ops=_parse_ops(args.binary_ops),
        constant_values=_parse_constants(args.const_values),
    )
    report = run_single_hole_micro_search(
        spec,
        factorized_search_hp=hp,
        seed=int(args.seed),
        candidate_expr=args.candidate_expr,
        corrupt_path=args.corrupt_path,
        replacement_expr=args.replacement_expr,
        hole_path=args.hole_path,
        grammar=grammar,
        enforce_dims=not bool(args.no_enforce_dims),
        report_topk=None if int(args.report_topk) <= 0 else int(args.report_topk),
    )

    payload = json.dumps(report, indent=2)
    if args.output:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return report


if __name__ == "__main__":  # pragma: no cover
    main()
