# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Deterministic micro-benchmark for inline skeleton refinement.

This is intentionally a fixed-skeleton benchmark.  The open-ended oracle search
can miss the intended scaffold under a small wall budget, which makes it a poor
first gate for refactoring where continuous refinement is scheduled.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import pathlib
import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from nestynet_sr.sr_search.factorized_search.explorer import node_str, score_expr


Node = tuple
TargetFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class MicroCase:
    case_id: str
    name: str
    nvars: int
    bounds: tuple[tuple[float, float], ...]
    target_expr: str
    skeleton_expr: str
    skeleton_node: Node
    target_fn: TargetFn
    gate_kind: str
    cfg_overrides: dict[str, object] = field(default_factory=dict)
    max_ratio: float | None = None
    max_refined_mse: float | None = None


def _default_refine_cfg(*, family_mode: str = "full") -> dict[str, object]:
    return {
        "score_mapping_family_mode": family_mode,
        "score_prescreen_enable": False,
        "lbfgs_steps": 10,
        "fit_subset": 128,
        "fit_subset_mode": "stride",
        "num_restarts": 2,
        "max_variants": 8,
        "max_params": 2,
        "linear_combo_enable": True,
        "linear_terms_max": 6,
        "linear_prune_rel": 1.0e-10,
        "gate_best_factor": 100.0,
        "max_refines": 20,
        "theta_l2": 1.0e-5,
        "init_log_min": -2.0,
        "init_log_max": 2.0,
        "grid_enable": True,
        "grid_size": 17,
        "grid_size_2d": 7,
        "grid_passes": 1,
        "grid_max_evals": 64,
        "safe_eps": 1.0e-6,
    }


_REFINE_COUNT_KEYS = (
    "score_calls",
    "refine_score_calls",
    "refinement_attempts",
    "hparam_optimizations",
    "grid_evals",
    "lbfgs_runs",
    "lbfgs_closures",
    "linear_solves",
    "linear_solves_multi",
    "materialized_rescores",
    "accepted_refinements",
    "attempt_cache_hits",
    "attempt_cache_misses",
    "attempt_cache_stores",
    "attempt_cache_size",
    "mapping_equiv_root_slots_pruned",
    "brute_refinement_attempts",
    "mutation_refinement_attempts",
    "slate_refinement_attempts",
    "controller_slate_refinement_attempts",
    "external_refinement_attempts",
)


def _merge_numeric_stats(total: dict[str, object], row: dict[str, object]) -> None:
    for key, value in row.items():
        if isinstance(value, bool) or value is None:
            continue
        try:
            if isinstance(value, int):
                total[key] = int(total.get(key, 0) or 0) + int(value)
            else:
                fv = float(value)
                if math.isfinite(fv):
                    total[key] = float(total.get(key, 0.0) or 0.0) + fv
        except Exception:
            continue


def _diag_number(diag: dict[str, object], key: str) -> float:
    try:
        value = float(diag.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0
    return value if math.isfinite(value) else 0.0


def _refine_cost_summary(diag: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {
        key: int(round(_diag_number(diag, key)))
        for key in _REFINE_COUNT_KEYS
        if key in diag
    }
    for key in ("base_score_s", "hparam_optimization_s"):
        if key in diag:
            out[key] = float(_diag_number(diag, key))
    attempts = max(_diag_number(diag, "refinement_attempts"), _diag_number(diag, "hparam_optimizations"))
    hits = _diag_number(diag, "attempt_cache_hits")
    misses = _diag_number(diag, "attempt_cache_misses")
    lookups = hits + misses
    out["accepted_per_attempt"] = None if attempts <= 0.0 else _diag_number(diag, "accepted_refinements") / attempts
    out["attempt_cache_hit_rate"] = None if lookups <= 0.0 else hits / lookups
    out["grid_evals_per_attempt"] = None if attempts <= 0.0 else _diag_number(diag, "grid_evals") / attempts
    out["lbfgs_closures_per_attempt"] = None if attempts <= 0.0 else _diag_number(diag, "lbfgs_closures") / attempts
    linear_solves = _diag_number(diag, "linear_solves") + _diag_number(diag, "linear_solves_multi")
    out["linear_solves_per_attempt"] = None if attempts <= 0.0 else linear_solves / attempts
    return out


def _make_grid(case: MicroCase, *, n: int, seed: int) -> torch.Tensor:
    dtype = torch.float64
    if case.nvars == 1:
        lo, hi = case.bounds[0]
        return torch.linspace(lo, hi, n, dtype=dtype).unsqueeze(-1)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    cols = []
    for lo, hi in case.bounds:
        cols.append(lo + (hi - lo) * torch.rand((n,), generator=g, dtype=dtype))
    return torch.stack(cols, dim=1)


def _cases() -> list[MicroCase]:
    return [
        MicroCase(
            case_id="000",
            name="sin_scale_1d",
            nvars=1,
            bounds=((0.25, 3.0),),
            target_expr="sin(2.7*x0)",
            skeleton_expr="sin(x0)",
            skeleton_node=("sin", ("var", 0)),
            target_fn=lambda x: torch.sin(2.7 * x[:, 0:1]),
            gate_kind="hard_improve",
            cfg_overrides={"score_mapping_family_mode": "poly_only"},
            max_ratio=1.0e-5,
            max_refined_mse=1.0e-7,
        ),
        MicroCase(
            case_id="001",
            name="trig_product_carrier",
            nvars=2,
            bounds=((0.2, 3.0), (-1.5, 1.5)),
            target_expr="cos(1.9*x0*x1)",
            skeleton_expr="cos(x0*x1)",
            skeleton_node=("cos", ("mul", ("var", 0), ("var", 1))),
            target_fn=lambda x: torch.cos(1.9 * x[:, 0:1] * x[:, 1:2]),
            gate_kind="hard_improve",
            cfg_overrides={"score_mapping_family_mode": "poly_only"},
            max_ratio=1.0e-5,
            max_refined_mse=1.0e-7,
        ),
        MicroCase(
            case_id="002",
            name="sin_plus_cos_2freq",
            nvars=2,
            bounds=((0.2, 3.0), (-1.5, 1.5)),
            target_expr="sin(2.7*x0)+cos(1.4*x1)",
            skeleton_expr="sin(x0)+cos(x1)",
            skeleton_node=("add", ("sin", ("var", 0)), ("cos", ("var", 1))),
            target_fn=lambda x: torch.sin(2.7 * x[:, 0:1]) + torch.cos(1.4 * x[:, 1:2]),
            gate_kind="soft_improve",
            cfg_overrides={
                "lbfgs_steps": 16,
                "fit_subset": 160,
                "num_restarts": 5,
                "theta_l2": 1.0e-6,
                "grid_size": 21,
                "grid_size_2d": 9,
                "grid_max_evals": 128,
            },
            max_ratio=2.0e-3,
        ),
        MicroCase(
            case_id="003",
            name="trigprod_plus_trig",
            nvars=2,
            bounds=((0.2, 3.0), (-1.5, 1.5)),
            target_expr="cos(1.9*x0*x1)+sin(1.3*x0)",
            skeleton_expr="cos(x0*x1)+sin(x0)",
            skeleton_node=(
                "add",
                ("cos", ("mul", ("var", 0), ("var", 1))),
                ("sin", ("var", 0)),
            ),
            target_fn=lambda x: torch.cos(1.9 * x[:, 0:1] * x[:, 1:2])
            + torch.sin(1.3 * x[:, 0:1]),
            gate_kind="hard_improve",
            max_ratio=1.0e-5,
            max_refined_mse=1.0e-7,
        ),
        MicroCase(
            case_id="004",
            name="exp_plus_sin",
            nvars=2,
            bounds=((0.2, 3.0), (-1.5, 1.5)),
            target_expr="x1*exp(0.8*x0)+sin(1.6*x1)",
            skeleton_expr="x1*exp(x0)+sin(x1)",
            skeleton_node=(
                "add",
                ("mul", ("var", 1), ("exp", ("var", 0))),
                ("sin", ("var", 1)),
            ),
            target_fn=lambda x: x[:, 1:2] * torch.exp(0.8 * x[:, 0:1])
            + torch.sin(1.6 * x[:, 1:2]),
            gate_kind="hard_improve",
            max_ratio=1.0e-5,
            max_refined_mse=1.0e-7,
        ),
        MicroCase(
            case_id="005",
            name="log_trig_internal",
            nvars=1,
            bounds=((0.35, 3.6),),
            target_expr="cos(2.7*log(1.8*x0))",
            skeleton_expr="cos(log(x0))",
            skeleton_node=("cos", ("log", ("var", 0))),
            target_fn=lambda x: torch.cos(2.7 * torch.log(1.8 * x[:, 0:1])),
            gate_kind="soft_improve",
            cfg_overrides={
                "score_mapping_family_mode": "poly_only",
                "lbfgs_steps": 20,
                "fit_subset": 180,
                "num_restarts": 9,
                "max_variants": 12,
                "theta_l2": 1.0e-6,
                "grid_size": 21,
                "grid_size_2d": 9,
                "grid_max_evals": 96,
            },
            max_ratio=5.0e-3,
        ),
        MicroCase(
            case_id="006",
            name="root_log_scale_control",
            nvars=1,
            bounds=((0.35, 3.6),),
            target_expr="log(2.4*x0)+0.3",
            skeleton_expr="log(x0)",
            skeleton_node=("log", ("var", 0)),
            target_fn=lambda x: torch.log(2.4 * x[:, 0:1]) + 0.3,
            gate_kind="mapping_equivalent",
        ),
        MicroCase(
            case_id="007",
            name="ineligible_product",
            nvars=2,
            bounds=((-2.0, 2.0), (-1.5, 1.5)),
            target_expr="x0*x1",
            skeleton_expr="x0*x1",
            skeleton_node=("mul", ("var", 0), ("var", 1)),
            target_fn=lambda x: x[:, 0:1] * x[:, 1:2],
            gate_kind="no_slot",
            cfg_overrides={"linear_combo_enable": False},
        ),
    ]


def run_micro_case(case: MicroCase, *, seed: int = 2026, verbose: bool = False) -> dict[str, object]:
    x_fit = _make_grid(case, n=220 if case.nvars > 1 else 192, seed=seed + int(case.case_id))
    x_probe = _make_grid(case, n=420 if case.nvars > 1 else 384, seed=seed + 100 + int(case.case_id))
    y_fit = case.target_fn(x_fit)
    y_probe = case.target_fn(x_probe)
    proj = torch.randn(
        (x_probe.shape[0], 16),
        generator=torch.Generator(device="cpu").manual_seed(seed + 200 + int(case.case_id)),
        dtype=torch.float64,
    )
    cfg = _default_refine_cfg(family_mode=str(case.cfg_overrides.get("score_mapping_family_mode", "full")))
    cfg.update(case.cfg_overrides)
    diagnostics: dict[str, object] = {}
    cfg["diagnostics"] = diagnostics
    cfg["attempt_cache"] = {}

    stream = None if verbose else io.StringIO()
    with contextlib.nullcontext() if verbose else contextlib.redirect_stdout(stream):
        t0 = time.perf_counter()
        base = score_expr(
            case.skeleton_node,
            x_fit,
            y_fit,
            x_probe,
            y_probe,
            proj,
            "bits",
            2.0,
            6,
            4,
            refine_enable=False,
            refine_cfg=cfg,
            return_expr=True,
        )
        t1 = time.perf_counter()
        state = {"trials_done": 0}
        refined = score_expr(
            case.skeleton_node,
            x_fit,
            y_fit,
            x_probe,
            y_probe,
            proj,
            "bits",
            2.0,
            6,
            4,
            refine_enable=True,
            refine_cfg=cfg,
            refine_best_mse=float("inf"),
            refine_state=state,
            return_expr=True,
        )
        t2 = time.perf_counter()

    if base is None or refined is None:
        raise AssertionError(f"{case.name}: score_expr returned None")
    base_mse = float(base[0])
    refined_mse = float(refined[0])
    ratio = refined_mse / max(base_mse, 1.0e-300)
    diagnostics["attempt_cache_size"] = len(cfg.get("attempt_cache", {}) or {})
    return {
        "id": case.case_id,
        "case": case.name,
        "gate_kind": case.gate_kind,
        "target": case.target_expr,
        "skeleton": case.skeleton_expr,
        "base_mse": base_mse,
        "refined_mse": refined_mse,
        "improvement_ratio": ratio,
        "base_expr": node_str(base[4]),
        "refined_expr": node_str(refined[4]),
        "trials_done": int(state.get("trials_done", 0)),
        "elapsed_base_s": float(t1 - t0),
        "elapsed_refine_s": float(t2 - t1),
        "refine_diagnostics": dict(diagnostics),
        "refine_cost_summary": _refine_cost_summary(diagnostics),
        "config": {
            "score_mapping_family_mode": str(cfg.get("score_mapping_family_mode", "full")),
            "refine_optimizer": str(cfg.get("optimizer", cfg.get("refine_optimizer", "lbfgs"))),
            "refine_max_variants": int(cfg.get("max_variants", 0)),
            "refine_max_params": int(cfg.get("max_params", 0)),
            "refine_num_restarts": int(cfg.get("num_restarts", 0)),
            "refine_lbfgs_steps": int(cfg.get("lbfgs_steps", 0)),
            "refine_grid_enable": bool(cfg.get("grid_enable", False)),
        },
    }


def run_microbench(*, seed: int = 2026, verbose: bool = False) -> dict[str, object]:
    rows = [run_micro_case(case, seed=seed, verbose=verbose) for case in _cases()]
    diagnostics_total: dict[str, object] = {}
    for row in rows:
        diag = row.get("refine_diagnostics", {})
        if isinstance(diag, dict):
            _merge_numeric_stats(diagnostics_total, diag)
    return {
        "benchmark": "skeleton_refinement_microbench",
        "seed": int(seed),
        "refine_diagnostics": diagnostics_total,
        "refine_cost_summary": _refine_cost_summary(diagnostics_total),
        "rows": rows,
    }


def _assert_case_gates(row: dict[str, object], case: MicroCase) -> None:
    base_mse = float(row["base_mse"])
    refined_mse = float(row["refined_mse"])
    ratio = float(row["improvement_ratio"])
    trials = int(row["trials_done"])
    assert math.isfinite(base_mse)
    assert math.isfinite(refined_mse)
    if case.gate_kind == "hard_improve":
        assert trials > 0
        assert ratio <= float(case.max_ratio)
        assert refined_mse <= float(case.max_refined_mse)
    elif case.gate_kind == "soft_improve":
        assert trials > 0
        assert ratio <= float(case.max_ratio)
    elif case.gate_kind == "mapping_equivalent":
        assert refined_mse <= max(base_mse * 3.0, base_mse + 1.0e-18)
    elif case.gate_kind == "no_slot":
        assert trials == 0
        assert abs(refined_mse - base_mse) <= 1.0e-18


def test_skeleton_refinement_microbench_cases():
    payload = run_microbench()
    rows = {str(row["id"]): row for row in payload["rows"]}
    assert int(payload["refine_cost_summary"]["score_calls"]) >= len(rows) * 2
    assert int(payload["refine_cost_summary"]["refinement_attempts"]) >= 1
    for case in _cases():
        _assert_case_gates(rows[case.case_id], case)
    assert rows["006"]["refine_cost_summary"]["mapping_equiv_root_slots_pruned"] == 1


def _load_rows(path: pathlib.Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["id"]): dict(row) for row in payload.get("rows", [])}


def _compare_to_baseline(rows: list[dict[str, object]], baseline_path: pathlib.Path) -> list[str]:
    baseline = _load_rows(baseline_path)
    failures: list[str] = []
    for row in rows:
        case_id = str(row["id"])
        old = baseline.get(case_id)
        if old is None:
            continue
        cur_mse = float(row["refined_mse"])
        old_mse = float(old["refined_mse"])
        floor = 1.0e-12
        if cur_mse > max(old_mse, floor) * 3.0:
            failures.append(
                f"{case_id} {row['case']}: refined_mse regressed "
                f"{old_mse:.3e} -> {cur_mse:.3e}"
            )
        cur_trials = int(row["trials_done"])
        old_trials = int(old["trials_done"])
        if cur_trials > int(math.ceil(old_trials * 1.25 + 2)):
            failures.append(
                f"{case_id} {row['case']}: trials_done regressed "
                f"{old_trials} -> {cur_trials}"
            )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-json", type=pathlib.Path, default=None)
    parser.add_argument("--baseline", type=pathlib.Path, default=None)
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    payload = run_microbench(seed=int(args.seed), verbose=bool(args.verbose))
    rows = list(payload["rows"])
    case_by_id = {case.case_id: case for case in _cases()}
    gate_failures: list[str] = []
    for row in rows:
        try:
            _assert_case_gates(row, case_by_id[str(row["id"])])
        except AssertionError as exc:
            gate_failures.append(f"{row['id']} {row['case']}: {exc}")

    baseline_failures: list[str] = []
    if args.baseline is not None:
        baseline_failures = _compare_to_baseline(rows, args.baseline)

    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    failures = gate_failures + baseline_failures
    if failures:
        for failure in failures:
            print(f"REGRESSION: {failure}")
        return 1 if args.fail_on_regression else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
