# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Build a deterministic, truth-blind list of cases to promote to CoE.

The ordinary campaign escalator handles completed cheap runs that have no
eligible symbolic selection.  This module preserves those hard-failure
decisions and adds a second opportunity class: an eligible final-polish result
whose full-dataset residual is significantly above the declared noise
variance.

No truth/canary field participates in either decision.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nestynet_sr.campaign_escalation import (
    ACTION_RETRY_COE,
    ACTION_RUN_COE,
    ACTION_SKIP,
    ACTION_TERMINAL_FAILURE,
    build_escalation_manifest,
    ignored_problem_ids,
    inspect_phase_artifacts,
    parse_problem_ids,
    problem_id_range,
    write_manifest_atomic,
)


PROMOTION_SCHEMA_VERSION = 1
PROMOTION_POLICY_NAME = "truth_blind_coe_promotion_v1"
DEFAULT_MIN_MSE_RATIO = 1.0
DEFAULT_MIN_EXCESS_Z = 5.0

REASON_HARD_FAILURE = "internal_failure_requires_coe"
REASON_RETRY_COE = "retry_incomplete_coe"
REASON_RESIDUAL_ABOVE_NOISE = "eligible_residual_above_noise"
REASON_RESIDUAL_WITHIN_BUDGET = "eligible_residual_within_noise_budget"
REASON_RESIDUAL_UNAVAILABLE = "eligible_residual_evidence_unavailable"
REASON_NOT_READY = "cheap_phase_not_ready"
REASON_ALREADY_SETTLED = "coe_or_campaign_already_settled"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _validated_thresholds(
    min_mse_ratio: float,
    min_excess_z: float,
) -> tuple[float, float]:
    ratio = float(min_mse_ratio)
    excess_z = float(min_excess_z)
    if not math.isfinite(ratio) or ratio < 1.0:
        raise ValueError("min_mse_ratio must be finite and at least 1")
    if not math.isfinite(excess_z) or excess_z < 0.0:
        raise ValueError("min_excess_z must be finite and non-negative")
    return ratio, excess_z


def residual_promotion_evidence(
    report: Mapping[str, Any],
    *,
    min_mse_ratio: float = DEFAULT_MIN_MSE_RATIO,
    min_excess_z: float = DEFAULT_MIN_EXCESS_Z,
) -> dict[str, Any]:
    """Evaluate full-data residual excess using only internal report fields."""

    ratio_threshold, z_threshold = _validated_thresholds(
        min_mse_ratio,
        min_excess_z,
    )
    final_selection = _as_mapping(report.get("final_selection"))
    final_polish = _as_mapping(report.get("final_polish"))
    full_snap = _as_mapping(final_polish.get("full_dataset_snap"))
    recommended = _as_mapping(final_polish.get("recommended"))

    def unavailable(detail: str) -> dict[str, Any]:
        return {
            "assessable": False,
            "promote": False,
            "reason_code": REASON_RESIDUAL_UNAVAILABLE,
            "reason": detail,
            "min_mse_ratio": ratio_threshold,
            "min_excess_z": z_threshold,
        }

    if final_selection.get("eligible_for_success") is not True:
        return unavailable("the final selection is not explicitly eligible")
    if str(final_selection.get("source") or "") != "final_polish":
        return unavailable("the eligible selection has no final-polish residual ballot")

    selected_expr = final_selection.get("expr")
    metric_expr = full_snap.get("selected_expr") or recommended.get("expr")
    if not isinstance(selected_expr, str) or not selected_expr.strip():
        return unavailable("the eligible final selection has no expression")
    if not isinstance(metric_expr, str) or not metric_expr.strip():
        return unavailable("the full-data residual metric has no selected expression")
    if metric_expr.strip() != selected_expr.strip():
        return unavailable("the full-data residual metric belongs to a different expression")

    selected_mse = _finite_float(full_snap.get("selected_full_mse"))
    if selected_mse is None:
        selected_mse = _finite_float(recommended.get("full_dataset_mse"))
    n_full_value = _finite_float(full_snap.get("n_full"))
    noise_mse_se = _finite_float(full_snap.get("loss_equiv_abs_floor"))
    noise_marker = _finite_float(final_polish.get("noise_loss_equiv_abs_floor"))
    if selected_mse is None or selected_mse < 0.0:
        return unavailable("selected full-dataset MSE is unavailable")
    if n_full_value is None or n_full_value < 1.0 or not n_full_value.is_integer():
        return unavailable("full-dataset row count is unavailable")
    if noise_mse_se is None or noise_mse_se <= 0.0:
        return unavailable("full-dataset noise-equivalence uncertainty is unavailable")
    if noise_marker is None or noise_marker <= 0.0:
        return unavailable("the report has no active declared-noise marker")

    n_full = int(n_full_value)
    declared_noise_mse = noise_mse_se * math.sqrt(float(n_full) / 2.0)
    if not math.isfinite(declared_noise_mse) or declared_noise_mse <= 0.0:
        return unavailable("declared noise variance could not be recovered")

    mse_ratio = selected_mse / declared_noise_mse
    excess_z = (selected_mse - declared_noise_mse) / noise_mse_se
    promote = bool(
        mse_ratio >= ratio_threshold
        and excess_z >= z_threshold
    )
    reason_code = (
        REASON_RESIDUAL_ABOVE_NOISE
        if promote
        else REASON_RESIDUAL_WITHIN_BUDGET
    )
    reason = (
        "eligible selected-expression residual exceeds both CoE promotion thresholds"
        if promote
        else "eligible selected-expression residual does not exceed both CoE promotion thresholds"
    )
    return {
        "assessable": True,
        "promote": promote,
        "reason_code": reason_code,
        "reason": reason,
        "metric_source": "final_polish.full_dataset_snap",
        "selected_expr": selected_expr.strip(),
        "selected_full_mse": selected_mse,
        "declared_noise_mse": declared_noise_mse,
        "noise_mse_se": noise_mse_se,
        "n_full": n_full,
        "mse_ratio": mse_ratio,
        "excess_z": excess_z,
        "min_mse_ratio": ratio_threshold,
        "min_excess_z": z_threshold,
    }


def _load_report(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def build_coe_promotion_manifest(
    *,
    cheap_results: str | os.PathLike[str],
    coe_results: str | os.PathLike[str] | None,
    problem_ids: Sequence[Any],
    excluded_ids: Sequence[Any] = (),
    min_mse_ratio: float = DEFAULT_MIN_MSE_RATIO,
    min_excess_z: float = DEFAULT_MIN_EXCESS_Z,
) -> dict[str, Any]:
    """Combine hard internal failures with eligible residual opportunities."""

    ratio_threshold, z_threshold = _validated_thresholds(
        min_mse_ratio,
        min_excess_z,
    )
    base = build_escalation_manifest(
        cheap_results=cheap_results,
        coe_results=coe_results,
        problem_ids=problem_ids,
        excluded_ids=excluded_ids,
    )
    cheap_path = Path(cheap_results)
    coe_path = None if coe_results is None else Path(coe_results)
    rows: list[dict[str, Any]] = []
    for problem in base["problems"]:
        problem_id = str(problem["id"])
        base_action = str(problem["action"])
        base_reason_code = str(problem["reason_code"])
        row: dict[str, Any] = {
            "id": problem_id,
            "promote": False,
            "promotion_class": None,
            "reason_code": REASON_ALREADY_SETTLED,
            "reason": "the case does not currently require a CoE promotion",
            "base_action": base_action,
            "base_reason_code": base_reason_code,
            "cheap_report": _as_mapping(problem.get("cheap")).get("report"),
            "coe": problem.get("coe"),
            "residual_evidence": None,
        }

        if base_action == ACTION_RUN_COE:
            row.update(
                promote=True,
                promotion_class="hard_failure",
                reason_code=REASON_HARD_FAILURE,
                reason=str(problem.get("reason") or base_reason_code),
            )
        elif base_action == ACTION_RETRY_COE:
            row.update(
                promote=True,
                promotion_class="retry_coe",
                reason_code=REASON_RETRY_COE,
                reason=str(problem.get("reason") or base_reason_code),
            )
        elif (
            base_action == ACTION_SKIP
            and base_reason_code == "cheap_selection_eligible"
        ):
            report_name = row["cheap_report"]
            report = None
            if isinstance(report_name, str) and Path(report_name).name == report_name:
                report = _load_report(cheap_path / report_name)
            if report is None:
                evidence = {
                    "assessable": False,
                    "promote": False,
                    "reason_code": REASON_RESIDUAL_UNAVAILABLE,
                    "reason": "the eligible cheap report could not be loaded",
                    "min_mse_ratio": ratio_threshold,
                    "min_excess_z": z_threshold,
                }
            else:
                evidence = residual_promotion_evidence(
                    report,
                    min_mse_ratio=ratio_threshold,
                    min_excess_z=z_threshold,
                )
            row["residual_evidence"] = evidence
            row["reason_code"] = str(evidence["reason_code"])
            row["reason"] = str(evidence["reason"])
            if evidence["promote"] is True:
                coe_state = "not_run"
                if coe_path is not None:
                    coe, coe_summary_row, coe_report = inspect_phase_artifacts(
                        results_dir=coe_path,
                        problem_id=problem_id,
                    )
                    row["coe"] = coe
                    if coe["summary_status"] == "missing":
                        coe_state = "not_run"
                    elif (
                        coe["summary_status"] != "ok"
                        or coe_summary_row is None
                        or coe["process_status"] == "failed"
                        or coe["report_status"] != "ok"
                        or coe_report is None
                    ):
                        coe_state = "retry"
                    elif coe["selection_status"] == "eligible":
                        coe_state = "settled_eligible"
                    else:
                        coe_state = "settled_terminal"
                if coe_state == "not_run":
                    row["promote"] = True
                    row["promotion_class"] = "eligible_residual_above_noise"
                elif coe_state == "retry":
                    row.update(
                        promote=True,
                        promotion_class="retry_coe",
                        reason_code=REASON_RETRY_COE,
                        reason="the residual-triggered CoE phase is incomplete",
                    )
                else:
                    row.update(
                        reason_code=REASON_ALREADY_SETTLED,
                        reason="the residual-triggered CoE phase already settled",
                    )
        elif base_action not in (ACTION_SKIP, ACTION_TERMINAL_FAILURE):
            row.update(
                reason_code=REASON_NOT_READY,
                reason="the cheap or CoE process must settle before promotion",
            )
        rows.append(row)

    promote_ids = [row["id"] for row in rows if row["promote"]]
    counts = {
        "promote": len(promote_ids),
        "hard_failure": sum(
            row["promotion_class"] == "hard_failure" for row in rows
        ),
        "retry_coe": sum(row["promotion_class"] == "retry_coe" for row in rows),
        "eligible_residual_above_noise": sum(
            row["promotion_class"] == "eligible_residual_above_noise"
            for row in rows
        ),
        "not_promoted": sum(not row["promote"] for row in rows),
    }
    return {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "policy": {
            "name": PROMOTION_POLICY_NAME,
            "truth_blind": True,
            "hard_failures": "reuse deterministic cheap-to-CoE escalation",
            "eligible_residual_rule": {
                "min_mse_ratio": ratio_threshold,
                "min_excess_z": z_threshold,
                "requires_both": True,
                "metric_source": "final_polish.full_dataset_snap",
            },
            "timestamps_in_manifest": False,
        },
        "requested_ids": list(base["requested_ids"]),
        "excluded_ids": list(base["excluded_ids"]),
        "active_ids": list(base["active_ids"]),
        "counts": counts,
        "promote_ids": promote_ids,
        "problems": rows,
    }


def promotion_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Serialize a promotion manifest canonically for reproducibility."""

    return (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _write_text_atomic(path: str | os.PathLike[str], payload: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        "--cheap-results",
        dest="cheap_results",
        default="results",
        help="directory containing completed cheap-run summaries and reports",
    )
    parser.add_argument(
        "--coe-results",
        help="optional CoE results directory used to suppress settled/retry-classify runs",
    )
    parser.add_argument("--ids", help="comma/whitespace-separated problem IDs")
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=119)
    parser.add_argument("--skip-ids", default="")
    parser.add_argument("--ignore-file")
    parser.add_argument(
        "--min-mse-ratio",
        type=float,
        default=DEFAULT_MIN_MSE_RATIO,
        help="minimum selected-MSE / declared-noise-MSE ratio (default: 1.0)",
    )
    parser.add_argument(
        "--min-excess-z",
        type=float,
        default=DEFAULT_MIN_EXCESS_Z,
        help="minimum residual excess in noise-MSE standard errors (default: 5)",
    )
    parser.add_argument("--lines", action="store_true", help="print one ID per line")
    parser.add_argument("--output", help="optional path for the plain ID list")
    parser.add_argument("--json-output", help="optional detailed evidence manifest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``scripts/list_coe_promotions.py``."""

    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    try:
        if args.ids is not None:
            requested = parse_problem_ids(args.ids)
            if not requested:
                raise ValueError("--ids was provided but contained no problem IDs")
        else:
            requested = problem_id_range(args.start_id, args.end_id)
        excluded = parse_problem_ids(args.skip_ids)
        excluded.extend(ignored_problem_ids(args.ignore_file))
        manifest = build_coe_promotion_manifest(
            cheap_results=args.cheap_results,
            coe_results=args.coe_results,
            problem_ids=requested,
            excluded_ids=excluded,
            min_mse_ratio=args.min_mse_ratio,
            min_excess_z=args.min_excess_z,
        )
        separator = "\n" if args.lines else " "
        ids_payload = separator.join(manifest["promote_ids"]) + "\n"
        if args.output:
            _write_text_atomic(args.output, ids_payload)
        else:
            sys.stdout.write(ids_payload)
        if args.json_output:
            write_manifest_atomic(args.json_output, manifest)
        return 0
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 2


__all__ = [
    "DEFAULT_MIN_EXCESS_Z",
    "DEFAULT_MIN_MSE_RATIO",
    "PROMOTION_POLICY_NAME",
    "PROMOTION_SCHEMA_VERSION",
    "REASON_RESIDUAL_ABOVE_NOISE",
    "build_coe_promotion_manifest",
    "promotion_manifest_bytes",
    "residual_promotion_evidence",
]


if __name__ == "__main__":
    raise SystemExit(main())
