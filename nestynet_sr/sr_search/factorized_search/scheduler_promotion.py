# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_ELIGIBILITY_FLOOR = 0.0
DEFAULT_PROMOTE_MIN_ELIGIBLE_UTILITY_LIFT = 0.0
DEFAULT_PROMOTE_MAX_INELIGIBLE_MEAN_BUDGET = 1.25
DEFAULT_PROMOTE_MAX_CALIBRATION_ERROR = 0.15
DEFAULT_PROMOTE_MIN_ONLINE_SOLVE_DELTA = -0.01
DEFAULT_PROMOTE_MAX_ONLINE_WALL_RATIO = 1.10


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _mean_or_none(values: Sequence[Any]) -> float | None:
    xs = [float(v) for v in values if math.isfinite(_safe_float(v))]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _ratio_or_none(numer: Any, denom: Any) -> float | None:
    numer_f = _safe_float(numer)
    denom_f = _safe_float(denom)
    if not math.isfinite(numer_f) or not math.isfinite(denom_f) or abs(denom_f) <= 1.0e-30:
        return None
    return float(numer_f / denom_f)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_json(payload: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _arm_map(report: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    if isinstance(report.get("arm_overall", None), Mapping):
        return dict(report.get("arm_overall", {}) or {}), "controller"
    if isinstance(report.get("overall", None), Mapping):
        return dict(report.get("overall", {}) or {}), "stage1"
    return {}, ""


def _resolve_arm_name(
    arms: Mapping[str, Any],
    *,
    explicit_name: str | None,
    prefix_candidates: Sequence[str],
) -> tuple[str | None, str | None]:
    if explicit_name:
        name = str(explicit_name)
        if name in arms:
            return name, None
        return None, f"requested arm {name!r} missing"
    matches = [
        str(name)
        for name in arms
        if any(str(name).startswith(str(prefix)) for prefix in prefix_candidates)
    ]
    if not matches:
        return None, f"no arm matched prefixes {list(prefix_candidates)!r}"
    if len(matches) > 1:
        return None, f"ambiguous arm match for prefixes {list(prefix_candidates)!r}: {sorted(matches)!r}"
    return matches[0], None


def summarize_scheduler_online_comparison(
    report: Mapping[str, Any],
    *,
    candidate_arm_name: str | None = None,
    baseline_arm_name: str | None = None,
) -> dict[str, Any]:
    arms, source = _arm_map(report)
    if not arms:
        return {
            "available": False,
            "source": "",
            "reason": "report does not contain arm_overall or overall summaries",
        }

    candidate_arm, candidate_err = _resolve_arm_name(
        arms,
        explicit_name=candidate_arm_name,
        prefix_candidates=("scheduler_control", "stage1_scheduler_control"),
    )
    baseline_arm, baseline_err = _resolve_arm_name(
        arms,
        explicit_name=baseline_arm_name,
        prefix_candidates=("macro", "stage1_macro"),
    )
    if candidate_arm is None or baseline_arm is None:
        reasons = [msg for msg in (candidate_err, baseline_err) if msg]
        return {
            "available": False,
            "source": str(source),
            "reason": "; ".join(reasons) if reasons else "unable to resolve scheduler and macro arms",
            "available_arms": sorted(str(name) for name in arms),
        }

    candidate = dict(arms.get(candidate_arm, {}) or {})
    baseline = dict(arms.get(baseline_arm, {}) or {})
    median_key = "median_eff_mse" if "median_eff_mse" in candidate or "median_eff_mse" in baseline else "median_mse"
    return {
        "available": True,
        "source": str(source),
        "candidate_arm": str(candidate_arm),
        "baseline_arm": str(baseline_arm),
        "candidate_solve_rate": _safe_float(candidate.get("solve_rate", float("nan"))),
        "baseline_solve_rate": _safe_float(baseline.get("solve_rate", float("nan"))),
        "solve_rate_delta": _safe_float(candidate.get("solve_rate", float("nan")))
        - _safe_float(baseline.get("solve_rate", float("nan"))),
        "candidate_mean_wall_s": _safe_float(candidate.get("mean_wall_s", float("nan"))),
        "baseline_mean_wall_s": _safe_float(baseline.get("mean_wall_s", float("nan"))),
        "wall_ratio": _ratio_or_none(candidate.get("mean_wall_s", None), baseline.get("mean_wall_s", None)),
        "candidate_median_mse": _safe_float(candidate.get(median_key, float("nan"))),
        "baseline_median_mse": _safe_float(baseline.get(median_key, float("nan"))),
        "median_mse_ratio": _ratio_or_none(candidate.get(median_key, None), baseline.get(median_key, None)),
        "candidate_mean_exact_eval_count": _safe_float(candidate.get("mean_exact_eval_count", float("nan"))),
        "baseline_mean_exact_eval_count": _safe_float(baseline.get("mean_exact_eval_count", float("nan"))),
        "exact_eval_ratio": _ratio_or_none(
            candidate.get("mean_exact_eval_count", None),
            baseline.get("mean_exact_eval_count", None),
        ),
        "candidate_route_usage": dict(candidate.get("route_usage", {}) or {}),
        "baseline_route_usage": dict(baseline.get("route_usage", {}) or {}),
    }


def _summarize_calibration_axis(axis_payload: Mapping[str, Any]) -> dict[str, Any]:
    total_count = 0
    weighted_abs_error = 0.0
    weighted_brier = 0.0
    bucket_count = 0
    for payload in dict(axis_payload or {}).values():
        if not isinstance(payload, Mapping):
            continue
        count = max(0, _safe_int(payload.get("count", 0), 0))
        mean_prob = _safe_float(payload.get("mean_prob", float("nan")))
        empirical_rate = _safe_float(payload.get("empirical_rate", float("nan")))
        brier = _safe_float(payload.get("brier", float("nan")))
        if count <= 0 or not math.isfinite(mean_prob) or not math.isfinite(empirical_rate):
            continue
        bucket_count += 1
        total_count += int(count)
        weighted_abs_error += float(count) * abs(float(mean_prob) - float(empirical_rate))
        if math.isfinite(brier):
            weighted_brier += float(count) * float(brier)
    if total_count <= 0:
        return {
            "bucket_count": int(bucket_count),
            "sample_count": 0,
            "weighted_abs_error": None,
            "weighted_brier": None,
        }
    return {
        "bucket_count": int(bucket_count),
        "sample_count": int(total_count),
        "weighted_abs_error": float(weighted_abs_error / total_count),
        "weighted_brier": float(weighted_brier / total_count),
    }


def summarize_scheduler_replay_promotion(
    stage1_report: Mapping[str, Any],
    *,
    eligibility_floor: float = DEFAULT_ELIGIBILITY_FLOOR,
) -> dict[str, Any]:
    replay = stage1_report.get("scheduler_replay", None)
    if not isinstance(replay, Mapping) or not bool(replay.get("trained", False)):
        return {
            "available": False,
            "reason": "scheduler_replay missing or untrained",
        }

    decision_rows = [
        dict(row)
        for row in list(replay.get("decision_rows", []) or [])
        if isinstance(row, Mapping)
    ]
    eligible_rows = []
    ineligible_rows = []
    eligible_utility_lifts: list[float] = []
    eligible_regrets: list[float] = []
    eligible_actual_regrets: list[float] = []
    ineligible_predicted_budgets: list[float] = []
    ineligible_actual_budgets: list[float] = []

    for row in decision_rows:
        oracle_utility = _safe_float(row.get("oracle_utility", float("nan")))
        predicted_utility = _safe_float(row.get("predicted_utility", float("nan")))
        predicted_budget = _safe_float(row.get("predicted_budget", float("nan")))
        actual_utility = _safe_float(row.get("actual_utility", float("nan")))
        actual_budget = _safe_float(row.get("actual_budget", float("nan")))
        if math.isfinite(oracle_utility) and oracle_utility > float(eligibility_floor):
            eligible_rows.append(row)
            if math.isfinite(predicted_utility):
                eligible_regrets.append(max(0.0, float(oracle_utility) - float(predicted_utility)))
                if math.isfinite(actual_utility):
                    eligible_actual_regrets.append(max(0.0, float(oracle_utility) - float(actual_utility)))
                    eligible_utility_lifts.append(float(predicted_utility) - float(actual_utility))
        else:
            ineligible_rows.append(row)
            if math.isfinite(predicted_budget):
                ineligible_predicted_budgets.append(float(predicted_budget))
            if math.isfinite(actual_budget) and float(actual_budget) > 0.0:
                ineligible_actual_budgets.append(float(actual_budget))

    route_cal = _summarize_calibration_axis(dict(replay.get("calibration_by_route", {}) or {}))
    depth_cal = _summarize_calibration_axis(dict(replay.get("calibration_by_depth", {}) or {}))
    budget_cal = _summarize_calibration_axis(dict(replay.get("calibration_by_budget", {}) or {}))
    cal_errors = [
        metric["weighted_abs_error"]
        for metric in (route_cal, depth_cal, budget_cal)
        if metric["weighted_abs_error"] is not None
    ]

    return {
        "available": True,
        "eligibility_floor": float(eligibility_floor),
        "groups_replayed": int(replay.get("groups_replayed", 0) or 0),
        "groups_with_actual_choice": int(replay.get("groups_with_actual_choice", 0) or 0),
        "top1_hit_rate": replay.get("top1_hit_rate", None),
        "mean_regret": replay.get("mean_regret", None),
        "actual_mean_regret": replay.get("actual_mean_regret", None),
        "mean_wasted_budget": replay.get("mean_wasted_budget", None),
        "actual_mean_wasted_budget": replay.get("actual_mean_wasted_budget", None),
        "eligible_group_count": int(len(eligible_rows)),
        "ineligible_group_count": int(len(ineligible_rows)),
        "eligible_groups_with_actual_choice": int(len(eligible_utility_lifts)),
        "eligible_mean_utility_lift_vs_actual": _mean_or_none(eligible_utility_lifts),
        "eligible_mean_regret": _mean_or_none(eligible_regrets),
        "eligible_actual_mean_regret": _mean_or_none(eligible_actual_regrets),
        "eligible_regret_reduction_vs_actual": None
        if _mean_or_none(eligible_actual_regrets) is None or _mean_or_none(eligible_regrets) is None
        else float(_mean_or_none(eligible_actual_regrets) - _mean_or_none(eligible_regrets)),
        "ineligible_mean_predicted_budget": _mean_or_none(ineligible_predicted_budgets),
        "ineligible_mean_actual_budget": _mean_or_none(ineligible_actual_budgets),
        "ineligible_budget_tax_vs_actual": None
        if _mean_or_none(ineligible_predicted_budgets) is None or _mean_or_none(ineligible_actual_budgets) is None
        else float(_mean_or_none(ineligible_predicted_budgets) - _mean_or_none(ineligible_actual_budgets)),
        "ineligible_share_above_min_budget": None
        if not ineligible_predicted_budgets
        else float(sum(1 for budget in ineligible_predicted_budgets if float(budget) > 1.0)) / float(len(ineligible_predicted_budgets)),
        "calibration": {
            "by_route": route_cal,
            "by_depth": depth_cal,
            "by_budget": budget_cal,
            "max_weighted_abs_error": None if not cal_errors else float(max(cal_errors)),
            "mean_weighted_abs_error": _mean_or_none(cal_errors),
        },
    }


def recommend_scheduler_promotion(
    *,
    controller_report: Mapping[str, Any] | None = None,
    stage1_report: Mapping[str, Any] | None = None,
    candidate_arm_name: str | None = None,
    baseline_arm_name: str | None = None,
    eligibility_floor: float = DEFAULT_ELIGIBILITY_FLOOR,
    promote_min_eligible_utility_lift: float = DEFAULT_PROMOTE_MIN_ELIGIBLE_UTILITY_LIFT,
    promote_max_ineligible_mean_budget: float = DEFAULT_PROMOTE_MAX_INELIGIBLE_MEAN_BUDGET,
    promote_max_calibration_error: float = DEFAULT_PROMOTE_MAX_CALIBRATION_ERROR,
    promote_min_online_solve_delta: float = DEFAULT_PROMOTE_MIN_ONLINE_SOLVE_DELTA,
    promote_max_online_wall_ratio: float = DEFAULT_PROMOTE_MAX_ONLINE_WALL_RATIO,
) -> dict[str, Any]:
    online = summarize_scheduler_online_comparison(
        controller_report or stage1_report or {},
        candidate_arm_name=candidate_arm_name,
        baseline_arm_name=baseline_arm_name,
    )
    replay = summarize_scheduler_replay_promotion(
        stage1_report or {},
        eligibility_floor=float(eligibility_floor),
    )

    gates: dict[str, dict[str, Any]] = {}

    eligible_lift = replay.get("eligible_mean_utility_lift_vs_actual", None) if replay.get("available", False) else None
    if replay.get("available", False) and eligible_lift is not None and math.isfinite(float(eligible_lift)):
        passed = float(eligible_lift) >= float(promote_min_eligible_utility_lift)
        reason = (
            f"eligible_mean_utility_lift_vs_actual={float(eligible_lift):.4f} "
            f"{'meets' if passed else 'below'} threshold {float(promote_min_eligible_utility_lift):.4f}"
        )
    elif replay.get("available", False):
        passed = False
        reason = "eligible opportunity utility lift unavailable"
    else:
        passed = False
        reason = str(replay.get("reason", "scheduler replay unavailable"))
    gates["eligible_utility"] = {
        "passed": bool(passed),
        "metric": eligible_lift,
        "threshold": float(promote_min_eligible_utility_lift),
        "reason": reason,
    }

    ineligible_budget = replay.get("ineligible_mean_predicted_budget", None) if replay.get("available", False) else None
    if replay.get("available", False) and int(replay.get("ineligible_group_count", 0) or 0) <= 0:
        passed = True
        reason = "no ineligible opportunities observed"
    elif replay.get("available", False) and ineligible_budget is not None and math.isfinite(float(ineligible_budget)):
        passed = float(ineligible_budget) <= float(promote_max_ineligible_mean_budget)
        reason = (
            f"ineligible_mean_predicted_budget={float(ineligible_budget):.4f} "
            f"{'within' if passed else 'above'} limit {float(promote_max_ineligible_mean_budget):.4f}"
        )
    elif replay.get("available", False):
        passed = False
        reason = "ineligible budget tax unavailable"
    else:
        passed = False
        reason = str(replay.get("reason", "scheduler replay unavailable"))
    gates["ineligible_tax"] = {
        "passed": bool(passed),
        "metric": ineligible_budget,
        "threshold": float(promote_max_ineligible_mean_budget),
        "reason": reason,
    }

    calibration_error = None
    if replay.get("available", False):
        calibration_error = dict(replay.get("calibration", {}) or {}).get("max_weighted_abs_error", None)
    if replay.get("available", False) and calibration_error is not None and math.isfinite(float(calibration_error)):
        passed = float(calibration_error) <= float(promote_max_calibration_error)
        reason = (
            f"max_weighted_calibration_error={float(calibration_error):.4f} "
            f"{'within' if passed else 'above'} limit {float(promote_max_calibration_error):.4f}"
        )
    elif replay.get("available", False):
        passed = False
        reason = "calibration summary unavailable"
    else:
        passed = False
        reason = str(replay.get("reason", "scheduler replay unavailable"))
    gates["calibration"] = {
        "passed": bool(passed),
        "metric": calibration_error,
        "threshold": float(promote_max_calibration_error),
        "reason": reason,
    }

    solve_delta = online.get("solve_rate_delta", None) if online.get("available", False) else None
    wall_ratio = online.get("wall_ratio", None) if online.get("available", False) else None
    if (
        online.get("available", False)
        and solve_delta is not None
        and wall_ratio is not None
        and math.isfinite(float(solve_delta))
        and math.isfinite(float(wall_ratio))
    ):
        solve_ok = float(solve_delta) >= float(promote_min_online_solve_delta)
        wall_ok = float(wall_ratio) <= float(promote_max_online_wall_ratio)
        passed = bool(solve_ok and wall_ok)
        reason = (
            f"solve_rate_delta={float(solve_delta):.4f} "
            f"{'meets' if solve_ok else 'below'} threshold {float(promote_min_online_solve_delta):.4f}; "
            f"wall_ratio={float(wall_ratio):.4f} "
            f"{'within' if wall_ok else 'above'} limit {float(promote_max_online_wall_ratio):.4f}"
        )
    elif online.get("available", False):
        passed = False
        reason = "online solve or wall metrics unavailable"
    else:
        passed = False
        reason = str(online.get("reason", "online controller comparison unavailable"))
    gates["online_noninferior"] = {
        "passed": bool(passed),
        "solve_rate_delta": solve_delta,
        "solve_delta_threshold": float(promote_min_online_solve_delta),
        "wall_ratio": wall_ratio,
        "wall_ratio_limit": float(promote_max_online_wall_ratio),
        "reason": reason,
    }

    reasons = [str(payload.get("reason", "")) for payload in gates.values() if str(payload.get("reason", ""))]
    meets_bar = all(bool(payload.get("passed", False)) for payload in gates.values())
    return {
        "mode": "scheduler_promotion",
        "promotion_bar": {
            "eligibility_floor": float(eligibility_floor),
            "promote_min_eligible_utility_lift": float(promote_min_eligible_utility_lift),
            "promote_max_ineligible_mean_budget": float(promote_max_ineligible_mean_budget),
            "promote_max_calibration_error": float(promote_max_calibration_error),
            "promote_min_online_solve_delta": float(promote_min_online_solve_delta),
            "promote_max_online_wall_ratio": float(promote_max_online_wall_ratio),
        },
        "inputs": {
            "controller_report_present": isinstance(controller_report, Mapping),
            "stage1_report_present": isinstance(stage1_report, Mapping),
        },
        "online_comparison": online,
        "replay_summary": replay,
        "gates": gates,
        "decision": "promote" if meets_bar else "hold",
        "meets_promotion_bar": bool(meets_bar),
        "reasons": reasons,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate scheduler benchmark reports against promotion gates")
    parser.add_argument("--controller_report", default=None, help="Optional controller_harness JSON report")
    parser.add_argument("--stage1_report", default=None, help="Optional stage1_benchmark_harness JSON report")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    parser.add_argument("--candidate_arm_name", default=None)
    parser.add_argument("--baseline_arm_name", default=None)
    parser.add_argument("--eligibility_floor", type=float, default=DEFAULT_ELIGIBILITY_FLOOR)
    parser.add_argument("--promote_min_eligible_utility_lift", type=float, default=DEFAULT_PROMOTE_MIN_ELIGIBLE_UTILITY_LIFT)
    parser.add_argument("--promote_max_ineligible_mean_budget", type=float, default=DEFAULT_PROMOTE_MAX_INELIGIBLE_MEAN_BUDGET)
    parser.add_argument("--promote_max_calibration_error", type=float, default=DEFAULT_PROMOTE_MAX_CALIBRATION_ERROR)
    parser.add_argument("--promote_min_online_solve_delta", type=float, default=DEFAULT_PROMOTE_MIN_ONLINE_SOLVE_DELTA)
    parser.add_argument("--promote_max_online_wall_ratio", type=float, default=DEFAULT_PROMOTE_MAX_ONLINE_WALL_RATIO)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.controller_report and not args.stage1_report:
        raise ValueError("scheduler_promotion requires --controller_report and/or --stage1_report")
    controller_report = _load_json(args.controller_report) if args.controller_report else None
    stage1_report = _load_json(args.stage1_report) if args.stage1_report else None
    report = recommend_scheduler_promotion(
        controller_report=controller_report,
        stage1_report=stage1_report,
        candidate_arm_name=args.candidate_arm_name,
        baseline_arm_name=args.baseline_arm_name,
        eligibility_floor=float(args.eligibility_floor),
        promote_min_eligible_utility_lift=float(args.promote_min_eligible_utility_lift),
        promote_max_ineligible_mean_budget=float(args.promote_max_ineligible_mean_budget),
        promote_max_calibration_error=float(args.promote_max_calibration_error),
        promote_min_online_solve_delta=float(args.promote_min_online_solve_delta),
        promote_max_online_wall_ratio=float(args.promote_max_online_wall_ratio),
    )
    if args.output:
        _write_json(report, args.output)
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
