# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Promotion-grade attribution analysis for oracle factorized symbolic search regression runs."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .oracle_regression import aggregate_rows_by_spec
from .oracle_suite import aggregate_rows


@dataclass(frozen=True)
class ArmRef:
    profile: str
    mode: str


@dataclass(frozen=True)
class ComparisonRef:
    comparison_id: str
    label: str
    candidate: ArmRef
    baseline: ArmRef
    description: str = ""


DEFAULT_PROMOTE_MIN_SOLVE_DELTA = 0.05
DEFAULT_PROMOTE_MAX_SUCCESS_LOSSES = 0
DEFAULT_PROMOTE_MAX_WALL_RATIO = 2.0
DEFAULT_PROMOTE_MAX_MSE_RATIO = 1.25
DEFAULT_TIE_FACTOR = 1.05


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        return None
    return str(value)


def _load_json(path: str | pathlib.Path) -> dict[str, Any]:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def _write_json(payload: dict[str, Any], path: str | pathlib.Path) -> None:
    out_path = pathlib.Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return float(out)


def _ratio(candidate: float, baseline: float) -> float | None:
    if not math.isfinite(candidate) or not math.isfinite(baseline):
        return None
    if abs(baseline) <= 1.0e-30:
        if abs(candidate) <= 1.0e-30:
            return 1.0
        return float("inf")
    return float(candidate / baseline)


def _geom_mean(values: Iterable[float]) -> float | None:
    logs: list[float] = []
    for raw in values:
        val = float(raw)
        if not math.isfinite(val) or val <= 0.0:
            continue
        logs.append(math.log(val))
    if not logs:
        return None
    return float(math.exp(sum(logs) / len(logs)))


def _parse_comparison(raw: str) -> ComparisonRef:
    text = str(raw).strip()
    if text == "":
        raise ValueError("Comparison string must be non-empty")
    if "=" in text:
        comp_id, spec = text.split("=", 1)
    else:
        spec = text
        comp_id = text.replace("->", "__").replace(":", "_")
    lhs, rhs = spec.split("->", 1)
    cand_profile, cand_mode = lhs.split(":", 1)
    base_profile, base_mode = rhs.split(":", 1)
    return ComparisonRef(
        comparison_id=str(comp_id).strip(),
        label=str(comp_id).strip(),
        candidate=ArmRef(profile=str(cand_profile).strip(), mode=str(cand_mode).strip()),
        baseline=ArmRef(profile=str(base_profile).strip(), mode=str(base_mode).strip()),
    )


def default_suite_comparisons(suite_id: str) -> list[ComparisonRef]:
    sid = str(suite_id or "").strip()
    if sid == "quick12_method_attribution":
        return [
            ComparisonRef(
                comparison_id="inverse_steering",
                label="Inverse Steering over factorized symbolic search",
                candidate=ArmRef("inverse_spec", "refine_off"),
                baseline=ArmRef("residual_basin_only", "refine_off"),
                description="Marginal value of inverse steering plus direct-spec repair over plain factorized symbolic search.",
            ),
            ComparisonRef(
                comparison_id="hole_fixing",
                label="Hole Fixing over Inverse Steering",
                candidate=ArmRef("hole_fix", "refine_off"),
                baseline=ArmRef("inverse_spec", "refine_off"),
                description="Marginal value of hole search on top of inverse steering.",
            ),
            ComparisonRef(
                comparison_id="repair_stack",
                label="Repair Stack over factorized symbolic search",
                candidate=ArmRef("hole_fix", "refine_off"),
                baseline=ArmRef("residual_basin_only", "refine_off"),
                description="Combined value of inverse steering and hole fixing over plain factorized symbolic search.",
            ),
        ]
    if sid == "quick12_inverse_compare":
        return [
            ComparisonRef(
                comparison_id="inverse_steering",
                label="Inverse Steering over No-Inverse",
                candidate=ArmRef("current", "refine_off"),
                baseline=ArmRef("no_inverse", "refine_off"),
                description="Explicit inverse-steering attribution run without hole search.",
            )
        ]
    return []


def _build_summary_maps(payload: dict[str, Any]) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[tuple[str, str, str, int], dict[str, Any]]]:
    rows = list(payload.get("rows") or [])
    overall_rows = list(payload.get("summary") or [])
    if not overall_rows:
        overall_rows = aggregate_rows(rows)
    spec_rows = list(payload.get("spec_summary") or [])
    if not spec_rows:
        spec_rows = aggregate_rows_by_spec(rows)

    overall_map = {
        (
            str(row.get("profile", "")),
            str(row.get("mode", "")),
            int(row.get("budget", 0) or 0),
        ): dict(row)
        for row in overall_rows
    }
    spec_map = {
        (
            str(row.get("spec_id", "")),
            str(row.get("profile", "")),
            str(row.get("mode", "")),
            int(row.get("budget", 0) or 0),
        ): dict(row)
        for row in spec_rows
    }
    return overall_map, spec_map


def _arm_budgets(overall_map: dict[tuple[str, str, int], dict[str, Any]], arm: ArmRef) -> set[int]:
    return {
        int(budget)
        for (profile, mode, budget) in overall_map
        if profile == str(arm.profile) and mode == str(arm.mode)
    }


def _compare_ratio(candidate: float | None, baseline: float | None, *, tie_factor: float) -> str:
    if candidate is None or baseline is None:
        return "missing"
    ratio = _ratio(float(candidate), float(baseline))
    if ratio is None:
        return "missing"
    lo = 1.0 / max(1.0, float(tie_factor))
    hi = max(1.0, float(tie_factor))
    if ratio < lo:
        return "win"
    if ratio > hi:
        return "loss"
    return "tie"


def _recommend_comparison(
    aggregate: dict[str, Any],
    *,
    promote_min_solve_delta: float,
    promote_max_success_losses: int,
    promote_max_wall_ratio: float,
    promote_max_mse_ratio: float,
) -> dict[str, Any]:
    solve_delta = _safe_float(aggregate.get("solve_rate_delta", 0.0), 0.0)
    success_losses = int(aggregate.get("success_loss_specs", 0) or 0)
    wall_ratio = aggregate.get("wall_seconds_ratio", None)
    mse_ratio = aggregate.get("mse_ratio_geomean", None)

    reasons: list[str] = []
    solve_ok = solve_delta >= float(promote_min_solve_delta)
    if solve_ok:
        reasons.append(f"solve_rate_delta={solve_delta:.3f} meets threshold {float(promote_min_solve_delta):.3f}")
    else:
        reasons.append(f"solve_rate_delta={solve_delta:.3f} below threshold {float(promote_min_solve_delta):.3f}")

    success_ok = success_losses <= int(promote_max_success_losses)
    if success_ok:
        reasons.append(f"success_loss_specs={success_losses} within limit {int(promote_max_success_losses)}")
    else:
        reasons.append(f"success_loss_specs={success_losses} exceeds limit {int(promote_max_success_losses)}")

    wall_ok = wall_ratio is None or float(wall_ratio) <= float(promote_max_wall_ratio)
    if wall_ratio is None:
        reasons.append("wall_seconds_ratio unavailable")
    elif wall_ok:
        reasons.append(f"wall_seconds_ratio={float(wall_ratio):.3f} within limit {float(promote_max_wall_ratio):.3f}")
    else:
        reasons.append(f"wall_seconds_ratio={float(wall_ratio):.3f} exceeds limit {float(promote_max_wall_ratio):.3f}")

    mse_ok = mse_ratio is None or float(mse_ratio) <= float(promote_max_mse_ratio)
    if mse_ratio is None:
        reasons.append("mse_ratio_geomean unavailable")
    elif mse_ok:
        reasons.append(f"mse_ratio_geomean={float(mse_ratio):.3f} within limit {float(promote_max_mse_ratio):.3f}")
    else:
        reasons.append(f"mse_ratio_geomean={float(mse_ratio):.3f} exceeds limit {float(promote_max_mse_ratio):.3f}")

    meets_bar = bool(solve_ok and success_ok and wall_ok and mse_ok)
    return {
        "decision": "promote" if meets_bar else "further_study",
        "meets_promotion_bar": bool(meets_bar),
        "reasons": reasons,
    }


def _summarize_budget_comparison(
    *,
    comparison: ComparisonRef,
    budget: int,
    overall_map: dict[tuple[str, str, int], dict[str, Any]],
    spec_map: dict[tuple[str, str, str, int], dict[str, Any]],
    tie_factor: float,
    promote_min_solve_delta: float,
    promote_max_success_losses: int,
    promote_max_wall_ratio: float,
    promote_max_mse_ratio: float,
) -> dict[str, Any]:
    candidate_key = (str(comparison.candidate.profile), str(comparison.candidate.mode), int(budget))
    baseline_key = (str(comparison.baseline.profile), str(comparison.baseline.mode), int(budget))
    candidate_row = dict(overall_map[candidate_key])
    baseline_row = dict(overall_map[baseline_key])

    spec_ids = sorted(
        {
            spec_id
            for (spec_id, profile, mode, row_budget) in spec_map
            if row_budget == int(budget)
            and (
                (profile == candidate_key[0] and mode == candidate_key[1])
                or (profile == baseline_key[0] and mode == baseline_key[1])
            )
        }
    )

    per_spec: list[dict[str, Any]] = []
    mse_ratios: list[float] = []
    counts = {
        "success_win_specs": 0,
        "success_loss_specs": 0,
        "mse_win_specs": 0,
        "mse_loss_specs": 0,
        "time_win_specs": 0,
        "time_loss_specs": 0,
    }

    for spec_id in spec_ids:
        cand = spec_map.get((spec_id, candidate_key[0], candidate_key[1], int(budget)))
        base = spec_map.get((spec_id, baseline_key[0], baseline_key[1], int(budget)))
        if cand is None or base is None:
            continue
        cand_solve = _safe_float(cand.get("solve_rate", 0.0), 0.0)
        base_solve = _safe_float(base.get("solve_rate", 0.0), 0.0)
        cand_mse = _safe_float(cand.get("best_mse_median", float("inf")), float("inf"))
        base_mse = _safe_float(base.get("best_mse_median", float("inf")), float("inf"))
        cand_time = _safe_float(cand.get("wall_seconds_mean", float("nan")), float("nan"))
        base_time = _safe_float(base.get("wall_seconds_mean", float("nan")), float("nan"))

        success_cmp = "tie"
        if cand_solve > base_solve + 1.0e-12:
            success_cmp = "win"
            counts["success_win_specs"] += 1
        elif cand_solve + 1.0e-12 < base_solve:
            success_cmp = "loss"
            counts["success_loss_specs"] += 1

        mse_cmp = _compare_ratio(cand_mse, base_mse, tie_factor=tie_factor)
        if mse_cmp == "win":
            counts["mse_win_specs"] += 1
        elif mse_cmp == "loss":
            counts["mse_loss_specs"] += 1

        time_cmp = _compare_ratio(cand_time, base_time, tie_factor=tie_factor)
        if time_cmp == "win":
            counts["time_win_specs"] += 1
        elif time_cmp == "loss":
            counts["time_loss_specs"] += 1

        mse_ratio = _ratio(cand_mse, base_mse)
        if mse_ratio is not None and math.isfinite(float(mse_ratio)) and float(mse_ratio) > 0.0:
            mse_ratios.append(float(mse_ratio))

        if success_cmp != "tie":
            primary_outcome = f"success_{success_cmp}"
        elif mse_cmp in {"win", "loss"}:
            primary_outcome = f"mse_{mse_cmp}"
        elif time_cmp in {"win", "loss"}:
            primary_outcome = f"time_{time_cmp}"
        else:
            primary_outcome = "tie"

        per_spec.append(
            {
                "spec_id": str(spec_id),
                "candidate_solve_rate": float(cand_solve),
                "baseline_solve_rate": float(base_solve),
                "solve_rate_delta": float(cand_solve - base_solve),
                "candidate_best_mse_median": float(cand_mse),
                "baseline_best_mse_median": float(base_mse),
                "mse_ratio": None if mse_ratio is None or not math.isfinite(float(mse_ratio)) else float(mse_ratio),
                "candidate_wall_seconds_mean": None if not math.isfinite(cand_time) else float(cand_time),
                "baseline_wall_seconds_mean": None if not math.isfinite(base_time) else float(base_time),
                "wall_seconds_ratio": None
                if _ratio(cand_time, base_time) is None or not math.isfinite(float(_ratio(cand_time, base_time)))
                else float(_ratio(cand_time, base_time)),
                "primary_outcome": primary_outcome,
            }
        )

    aggregate = {
        "candidate_solve_rate": float(_safe_float(candidate_row.get("solve_rate", 0.0), 0.0)),
        "baseline_solve_rate": float(_safe_float(baseline_row.get("solve_rate", 0.0), 0.0)),
        "solve_rate_delta": float(
            _safe_float(candidate_row.get("solve_rate", 0.0), 0.0)
            - _safe_float(baseline_row.get("solve_rate", 0.0), 0.0)
        ),
        "candidate_best_mse_median": float(_safe_float(candidate_row.get("best_mse_median", float("inf")), float("inf"))),
        "baseline_best_mse_median": float(_safe_float(baseline_row.get("best_mse_median", float("inf")), float("inf"))),
        "candidate_wall_seconds_mean": float(_safe_float(candidate_row.get("wall_seconds_mean", float("nan")), float("nan"))),
        "baseline_wall_seconds_mean": float(_safe_float(baseline_row.get("wall_seconds_mean", float("nan")), float("nan"))),
        "wall_seconds_ratio": _ratio(
            _safe_float(candidate_row.get("wall_seconds_mean", float("nan")), float("nan")),
            _safe_float(baseline_row.get("wall_seconds_mean", float("nan")), float("nan")),
        ),
        "mse_ratio_geomean": _geom_mean(mse_ratios),
        "paired_spec_count": int(len(per_spec)),
        **counts,
    }
    recommendation = _recommend_comparison(
        aggregate,
        promote_min_solve_delta=promote_min_solve_delta,
        promote_max_success_losses=promote_max_success_losses,
        promote_max_wall_ratio=promote_max_wall_ratio,
        promote_max_mse_ratio=promote_max_mse_ratio,
    )

    per_spec.sort(
        key=lambda row: (
            str(row.get("primary_outcome", "")),
            str(row.get("spec_id", "")),
        )
    )
    return {
        "comparison_id": str(comparison.comparison_id),
        "label": str(comparison.label),
        "description": str(comparison.description),
        "candidate_arm": {
            "profile": candidate_key[0],
            "mode": candidate_key[1],
            "budget": int(budget),
        },
        "baseline_arm": {
            "profile": baseline_key[0],
            "mode": baseline_key[1],
            "budget": int(budget),
        },
        "aggregate": aggregate,
        "recommendation": recommendation,
        "per_spec": per_spec,
    }


def summarize_method_attribution(
    payload: dict[str, Any],
    *,
    comparisons: Sequence[ComparisonRef] | None = None,
    tie_factor: float = DEFAULT_TIE_FACTOR,
    promote_min_solve_delta: float = DEFAULT_PROMOTE_MIN_SOLVE_DELTA,
    promote_max_success_losses: int = DEFAULT_PROMOTE_MAX_SUCCESS_LOSSES,
    promote_max_wall_ratio: float = DEFAULT_PROMOTE_MAX_WALL_RATIO,
    promote_max_mse_ratio: float = DEFAULT_PROMOTE_MAX_MSE_RATIO,
) -> dict[str, Any]:
    overall_map, spec_map = _build_summary_maps(payload)
    suite_id = str(payload.get("suite_id", "") or "")
    comparison_list = list(comparisons) if comparisons is not None else default_suite_comparisons(suite_id)
    missing: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []

    for comparison in comparison_list:
        candidate_budgets = _arm_budgets(overall_map, comparison.candidate)
        baseline_budgets = _arm_budgets(overall_map, comparison.baseline)
        shared_budgets = sorted(candidate_budgets & baseline_budgets)
        if not shared_budgets:
            missing.append(
                {
                    "comparison_id": str(comparison.comparison_id),
                    "candidate": _jsonable(comparison.candidate.__dict__),
                    "baseline": _jsonable(comparison.baseline.__dict__),
                }
            )
            continue
        for budget in shared_budgets:
            resolved.append(
                _summarize_budget_comparison(
                    comparison=comparison,
                    budget=int(budget),
                    overall_map=overall_map,
                    spec_map=spec_map,
                    tie_factor=float(tie_factor),
                    promote_min_solve_delta=float(promote_min_solve_delta),
                    promote_max_success_losses=int(promote_max_success_losses),
                    promote_max_wall_ratio=float(promote_max_wall_ratio),
                    promote_max_mse_ratio=float(promote_max_mse_ratio),
                )
            )

    return _jsonable(
        {
            "mode": "oracle_method_attribution",
            "suite_id": suite_id,
            "suite_manifest": str(payload.get("suite_manifest", "") or ""),
            "promotion_bar": {
                "promote_min_solve_delta": float(promote_min_solve_delta),
                "promote_max_success_losses": int(promote_max_success_losses),
                "promote_max_wall_ratio": float(promote_max_wall_ratio),
                "promote_max_mse_ratio": float(promote_max_mse_ratio),
                "tie_factor": float(tie_factor),
            },
            "comparisons": resolved,
            "missing_comparisons": missing,
        }
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze oracle_regression results into promotion-grade method comparisons")
    parser.add_argument("--results", required=True, help="oracle_regression_results.json path")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    parser.add_argument(
        "--comparison",
        action="append",
        default=None,
        help="Optional comparison override of the form id=profile:mode->profile:mode",
    )
    parser.add_argument("--tie_factor", type=float, default=DEFAULT_TIE_FACTOR)
    parser.add_argument("--promote_min_solve_delta", type=float, default=DEFAULT_PROMOTE_MIN_SOLVE_DELTA)
    parser.add_argument("--promote_max_success_losses", type=int, default=DEFAULT_PROMOTE_MAX_SUCCESS_LOSSES)
    parser.add_argument("--promote_max_wall_ratio", type=float, default=DEFAULT_PROMOTE_MAX_WALL_RATIO)
    parser.add_argument("--promote_max_mse_ratio", type=float, default=DEFAULT_PROMOTE_MAX_MSE_RATIO)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = _load_json(args.results)
    comparisons = None if not args.comparison else [_parse_comparison(raw) for raw in args.comparison]
    report = summarize_method_attribution(
        payload,
        comparisons=comparisons,
        tie_factor=float(args.tie_factor),
        promote_min_solve_delta=float(args.promote_min_solve_delta),
        promote_max_success_losses=int(args.promote_max_success_losses),
        promote_max_wall_ratio=float(args.promote_max_wall_ratio),
        promote_max_mse_ratio=float(args.promote_max_mse_ratio),
    )
    if args.output:
        _write_json(report, args.output)
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
