# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Deterministic, truth-blind cheap-run to CoE campaign escalation.

The campaign policy deliberately ignores ``truth_eval`` and every nested truth
canary.  Escalation is based on process completion and the pipeline's own
selection/unit certificates.  This keeps benchmark truth out of model
selection while making the expensive committee phase reproducible and
resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nestynet_sr.sr_core.coefficient_metadata import (
    coefficient_symbol_values_for_expression,
)


MANIFEST_SCHEMA_VERSION = 1
OUTCOME_SCHEMA_VERSION = 1
POLICY_NAME = "truth_blind_cheap_then_coe_v1"
_NOISELESS_GENERIC_NUMERICAL_ZERO_RATIO = 1.0e-12

ACTION_SKIP = "skip"
ACTION_RUN_COE = "run_coe"
ACTION_RETRY_CHEAP = "retry_cheap"
ACTION_PENDING = "pending"
ACTION_RETRY_COE = "retry_coe"
ACTION_TERMINAL_FAILURE = "terminal_failure"

MANIFEST_ACTIONS = (
    ACTION_SKIP,
    ACTION_RUN_COE,
    ACTION_RETRY_CHEAP,
    ACTION_PENDING,
    ACTION_RETRY_COE,
    ACTION_TERMINAL_FAILURE,
)

_REASON_TEXT = {
    "selection_eligible": "the report contains an eligible final symbolic selection",
    "final_selection_ineligible": "the report explicitly marks the final selection ineligible",
    "final_selection_not_applied": "the report says the final selection was not applied",
    "final_selection_missing_expression": "the final selection has no symbolic expression",
    "final_selection_invalid_coefficient_metadata": "the final selection has invalid or incomplete coefficient metadata",
    "final_selection_unit_risk": "the final selection lacks the required checked-valid unit certificate",
    "final_polish_no_safe_unit_valid_replacement": "final polish found no safe unit-valid replacement",
    "final_polish_requested_escalation": "final polish explicitly requested campaign escalation",
    "stagec_uncertified": "Stage C did not certify its symbolic expression",
    "stagec_invalid_coefficient_metadata": "Stage C has invalid or incomplete coefficient metadata",
    "stagec_selection_eligible": "legacy Stage C output contains a certified symbolic expression",
    "noiseless_generic_approximant": (
        "the noiseless cheap selection is a generic approximant without "
        "numerical-zero validation error"
    ),
    "no_symbolic_selection": "the run completed without an eligible symbolic selection",
    "cheap_summary_missing": "the cheap per-problem suite summary is not present",
    "cheap_summary_invalid": "the cheap per-problem suite summary is malformed",
    "cheap_process_failed": "the cheap process exited unsuccessfully",
    "cheap_report_missing": "the cheap process succeeded but its report is missing",
    "cheap_report_invalid": "the cheap report is malformed or ambiguous",
    "cheap_selection_eligible": "the cheap report contains an eligible final symbolic selection",
    "cheap_final_selection_ineligible": "the cheap final selection is explicitly ineligible",
    "cheap_final_selection_not_applied": "the cheap final selection was not applied",
    "cheap_final_selection_missing_expression": "the cheap final selection has no symbolic expression",
    "cheap_final_selection_invalid_coefficient_metadata": "the cheap final selection has invalid or incomplete coefficient metadata",
    "cheap_final_selection_unit_risk": "the cheap final selection has unresolved unit risk",
    "cheap_final_polish_no_safe_unit_valid_replacement": "cheap final polish found no safe unit-valid replacement",
    "cheap_final_polish_requested_escalation": "cheap final polish explicitly requested CoE escalation",
    "cheap_stagec_uncertified": "cheap Stage C did not certify its symbolic expression",
    "cheap_stagec_invalid_coefficient_metadata": "cheap Stage C has invalid or incomplete coefficient metadata",
    "cheap_noiseless_generic_approximant": (
        "the noiseless cheap selection is a generic approximant without "
        "numerical-zero validation error"
    ),
    "cheap_no_symbolic_selection": "the cheap run produced no eligible symbolic selection",
    "coe_summary_invalid": "the CoE per-problem suite summary is malformed",
    "coe_process_failed": "the CoE process exited unsuccessfully",
    "coe_report_missing": "the CoE process succeeded but its report is missing",
    "coe_report_invalid": "the CoE report is malformed or ambiguous",
    "coe_selection_eligible": "the CoE report contains an eligible final symbolic selection",
    "coe_no_eligible_selection": "the completed CoE run produced no eligible final selection",
}


def reason_text(code: str) -> str:
    """Return stable human-readable text for a machine reason code."""

    return _REASON_TEXT.get(str(code), str(code).replace("_", " "))


def normalize_problem_id(value: Any) -> str:
    """Normalize ``7``/``"7"``/``"pb007"`` to ``"pb007"``."""

    if isinstance(value, bool):
        raise ValueError("boolean values are not problem IDs")
    raw = str(value).strip().lower()
    if raw.startswith("pb"):
        raw = raw[2:]
    if not re.fullmatch(r"[0-9]+", raw):
        raise ValueError(f"invalid problem ID {value!r}")
    number = int(raw)
    if number < 0:
        raise ValueError(f"invalid problem ID {value!r}")
    return f"pb{number:03d}"


def parse_problem_ids(values: Any) -> list[str]:
    """Parse comma/whitespace-separated or iterable IDs into numeric order."""

    if values is None:
        return []
    if isinstance(values, str):
        raw_values: Iterable[Any] = re.split(r"[\s,]+", values.strip())
    else:
        raw_values = values
    normalized = {
        normalize_problem_id(value)
        for value in raw_values
        if str(value).strip()
    }
    return sorted(normalized, key=lambda item: int(item[2:]))


def problem_id_range(start_id: int = 0, end_id: int = 119) -> list[str]:
    """Return an inclusive, normalized problem-ID range."""

    start = int(start_id)
    end = int(end_id)
    if start < 0 or end < start:
        raise ValueError(f"invalid problem range {start_id!r}..{end_id!r}")
    return [normalize_problem_id(value) for value in range(start, end + 1)]


def ignored_problem_ids(path: str | os.PathLike[str] | None) -> list[str]:
    """Read the campaign ``PROBLEMS.ignore`` syntax without requiring the file."""

    if path is None:
        return []
    ignore_path = Path(path)
    if not ignore_path.is_file():
        return []
    values: list[str] = []
    for line in ignore_path.read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip()
        if value:
            values.append(value)
    return parse_problem_ids(values)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _unit_certificate_has_risk(certificate: Any) -> bool:
    certificate = _as_mapping(certificate)
    return bool(
        certificate
        and (
            (
                certificate.get("checked") is True
                and certificate.get("valid") is not True
            )
            or str(certificate.get("code") or "") == "expression_unavailable"
        )
    )


def _stagec_has_unit_risk(stagec: Any) -> bool:
    stagec = _as_mapping(stagec)
    sympy_meta = _as_mapping(stagec.get("sympy_meta"))
    return bool(
        _unit_certificate_has_risk(stagec.get("unit_admissibility"))
        or _unit_certificate_has_risk(sympy_meta.get("unit_admissibility"))
        or str(sympy_meta.get("kind") or "")
        == "unit_check_expression_unavailable"
        or (
            stagec.get("units_checked") is True
            and stagec.get("units_ok") is not True
        )
        or str(stagec.get("symbolic_status") or "") == "unit_invalid"
    )


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _coe_was_attempted(report: Mapping[str, Any]) -> bool:
    coe_committee = _as_mapping(report.get("coe_committee"))
    final_selection = _as_mapping(report.get("final_selection"))
    return bool(
        coe_committee.get("enabled") is True
        or str(final_selection.get("source") or "") == "coe_committee"
    )


def _noiseless_generic_approximant_decision(
    report: Mapping[str, Any],
    *,
    final_selection: Mapping[str, Any],
) -> ReportDecision | None:
    """Reject a merely target-good generic fit from the cheap noiseless phase."""

    if _coe_was_attempted(report):
        return None
    stageb = _as_mapping(report.get("stageB"))
    metrics = _as_mapping(stageb.get("candidate_metrics"))
    num_nn = _finite_float(metrics.get("num_nn"))
    if not (
        metrics.get("full_rewrite") is True
        and metrics.get("generic_approximant") is True
        and num_nn == 0.0
    ):
        return None

    use_original_y = metrics.get("has_original_y_validation") is True
    if use_original_y:
        noise_floor = _finite_float(metrics.get("original_y_noise_floor_raw"))
        good_enough = _finite_float(
            metrics.get("original_y_loss_good_enough_eff")
        )
        selected_mse = _finite_float(metrics.get("original_y_val_loss"))
    else:
        noise_floor = _finite_float(
            metrics.get(
                "portfolio_noise_floor_raw",
                metrics.get("noise_floor_raw"),
            )
        )
        good_enough = _finite_float(metrics.get("loss_good_enough_eff"))
        selected_mse = _finite_float(
            metrics.get("portfolio_val_loss", metrics.get("val_loss"))
        )

    if noise_floor is None or noise_floor != 0.0:
        return None

    final_polish = _as_mapping(report.get("final_polish"))
    recommended = _as_mapping(final_polish.get("recommended"))
    if (
        str(final_selection.get("source") or "") == "final_polish"
        and recommended.get("expr") == final_selection.get("expr")
    ):
        polished_mse = _finite_float(recommended.get("full_dataset_mse"))
        if polished_mse is None:
            polished_mse = _finite_float(recommended.get("val_mse"))
        if polished_mse is not None:
            selected_mse = polished_mse

    if (
        selected_mse is None
        or selected_mse < 0.0
        or good_enough is None
        or good_enough <= 0.0
    ):
        return None
    if (
        selected_mse / good_enough
        <= _NOISELESS_GENERIC_NUMERICAL_ZERO_RATIO
    ):
        return None
    code = "noiseless_generic_approximant"
    return ReportDecision(False, code, reason_text(code))


@dataclass(frozen=True)
class ReportDecision:
    """Truth-blind eligibility decision for one completed SR report."""

    eligible: bool
    reason_code: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": bool(self.eligible),
            "reason_code": self.reason_code,
            "reason": self.reason,
        }


def _embedded_report_decision(report: Mapping[str, Any]) -> ReportDecision | None:
    outcome = _as_mapping(report.get("campaign_outcome"))
    if not (
        outcome.get("schema_version") == OUTCOME_SCHEMA_VERSION
        and outcome.get("policy") == POLICY_NAME
        and outcome.get("truth_blind") is True
        and isinstance(outcome.get("selection_eligible"), bool)
    ):
        return None
    code = str(outcome.get("reason_code") or "")
    if not code:
        return None
    return ReportDecision(
        eligible=bool(outcome["selection_eligible"]),
        reason_code=code,
        reason=reason_text(code),
    )


def _classify_report_structure(report: Mapping[str, Any]) -> ReportDecision:
    final_polish = _as_mapping(report.get("final_polish"))
    no_safe_polish = (
        str(final_polish.get("status") or "")
        == "no_safe_unit_valid_replacement"
    )
    stagec = _as_mapping(report.get("stageC"))
    stagec_unit_risk = _stagec_has_unit_risk(stagec)
    final_selection = _as_mapping(report.get("final_selection"))

    if final_selection:
        if final_selection.get("eligible_for_success") is False:
            if final_polish.get("needs_escalation") is True or no_safe_polish:
                code = str(final_polish.get("escalation_reason") or "")
                if code != "final_polish_no_safe_unit_valid_replacement":
                    code = "final_polish_requested_escalation"
                return ReportDecision(False, code, reason_text(code))
            return ReportDecision(
                False,
                "final_selection_ineligible",
                reason_text("final_selection_ineligible"),
            )
        if final_selection.get("applied") is False:
            if final_polish.get("needs_escalation") is True or no_safe_polish:
                code = str(final_polish.get("escalation_reason") or "")
                if code != "final_polish_no_safe_unit_valid_replacement":
                    code = "final_polish_requested_escalation"
                return ReportDecision(False, code, reason_text(code))
            return ReportDecision(
                False,
                "final_selection_not_applied",
                reason_text("final_selection_not_applied"),
            )
        expression = final_selection.get("expr")
        if not isinstance(expression, str) or not expression.strip():
            return ReportDecision(
                False,
                "final_selection_missing_expression",
                reason_text("final_selection_missing_expression"),
            )
        try:
            coefficient_symbol_values_for_expression(
                final_selection.get("coefficient_metadata"),
                expression,
            )
        except Exception:
            return ReportDecision(
                False,
                "final_selection_invalid_coefficient_metadata",
                reason_text("final_selection_invalid_coefficient_metadata"),
            )
        certificate = _as_mapping(final_selection.get("unit_admissibility"))
        if _unit_certificate_has_risk(certificate):
            return ReportDecision(
                False,
                "final_selection_unit_risk",
                reason_text("final_selection_unit_risk"),
            )
        if no_safe_polish or stagec_unit_risk:
            if not (
                certificate.get("checked") is True
                and certificate.get("valid") is True
            ):
                return ReportDecision(
                    False,
                    "final_selection_unit_risk",
                    reason_text("final_selection_unit_risk"),
                )
        generic_decision = _noiseless_generic_approximant_decision(
            report,
            final_selection=final_selection,
        )
        if generic_decision is not None:
            return generic_decision
        return ReportDecision(
            True,
            "selection_eligible",
            reason_text("selection_eligible"),
        )

    if final_polish.get("needs_escalation") is True:
        code = str(final_polish.get("escalation_reason") or "")
        if code != "final_polish_no_safe_unit_valid_replacement":
            code = "final_polish_requested_escalation"
        return ReportDecision(False, code, reason_text(code))
    if no_safe_polish:
        code = "final_polish_no_safe_unit_valid_replacement"
        return ReportDecision(False, code, reason_text(code))
    if stagec.get("certified") is False or stagec_unit_risk:
        code = "stagec_uncertified"
        return ReportDecision(False, code, reason_text(code))
    stagec_expr = stagec.get("y_expr_str") or stagec.get("phi_expr_str")
    if isinstance(stagec_expr, str) and stagec_expr.strip():
        try:
            coefficient_symbol_values_for_expression(
                stagec.get("coefficient_metadata"),
                stagec_expr,
            )
        except Exception:
            return ReportDecision(
                False,
                "stagec_invalid_coefficient_metadata",
                reason_text("stagec_invalid_coefficient_metadata"),
            )
        generic_decision = _noiseless_generic_approximant_decision(
            report,
            final_selection={
                "source": "stageB",
                "expr": stagec_expr,
            },
        )
        if generic_decision is not None:
            return generic_decision
        return ReportDecision(
            True,
            "stagec_selection_eligible",
            reason_text("stagec_selection_eligible"),
        )
    return ReportDecision(
        False,
        "no_symbolic_selection",
        reason_text("no_symbolic_selection"),
    )


def classify_report_selection(
    report: Mapping[str, Any],
    *,
    prefer_embedded: bool = True,
) -> ReportDecision:
    """Classify a report without reading truth-canary fields.

    New reports carry a settled ``campaign_outcome``. Its stable reason is
    reused only while its eligibility and reason agree with a fresh structural
    audit, so stale or subsequently corrupted report metadata fails closed.
    """

    structural = _classify_report_structure(report)
    if prefer_embedded:
        embedded = _embedded_report_decision(report)
        if (
            embedded is not None
            and embedded.eligible == structural.eligible
            and embedded.reason_code == structural.reason_code
        ):
            return embedded
    return structural


def report_campaign_outcome(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the stable campaign-outcome block persisted in new reports."""

    decision = classify_report_selection(report, prefer_embedded=False)

    coe_attempted = _coe_was_attempted(report)
    if decision.eligible:
        action = "complete"
    elif coe_attempted:
        action = ACTION_TERMINAL_FAILURE
    else:
        action = ACTION_RUN_COE
    return {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "policy": POLICY_NAME,
        "truth_blind": True,
        "phase": "coe" if coe_attempted else "cheap",
        "selection_eligible": bool(decision.eligible),
        "action": action,
        "reason_code": decision.reason_code,
        "reason": decision.reason,
    }


@dataclass(frozen=True)
class _JsonLoad:
    status: str
    payload: Mapping[str, Any] | None


def _load_json_mapping(path: Path) -> _JsonLoad:
    if not path.is_file():
        return _JsonLoad("missing", None)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return _JsonLoad("invalid", None)
    if not isinstance(payload, Mapping):
        return _JsonLoad("invalid", None)
    return _JsonLoad("ok", payload)


def _summary_result(
    summary: Mapping[str, Any],
    problem_id: str,
) -> Mapping[str, Any] | None:
    results = summary.get("results")
    if not isinstance(results, list):
        return None
    matching = []
    for row in results:
        if not isinstance(row, Mapping):
            continue
        try:
            stem = normalize_problem_id(row.get("stem"))
        except ValueError:
            continue
        if stem == problem_id:
            matching.append(row)
    if len(matching) != 1:
        return None
    return matching[0]


def _report_path(
    results_dir: Path,
    problem_id: str,
    summary_row: Mapping[str, Any],
) -> Path | None:
    filepath = summary_row.get("filepath")
    if isinstance(filepath, str) and filepath.strip():
        exact = results_dir / f"{Path(filepath).stem}.report.json"
        if exact.is_file():
            return exact
    matches = sorted(results_dir.glob(f"{problem_id}*.report.json"))
    return matches[0] if len(matches) == 1 else None


def _path_name(path: Path | None) -> str | None:
    return None if path is None else path.name


def _decision_digest(problem: Mapping[str, Any]) -> str:
    """Hash only the truth-blind decision projection, never the full report."""

    projection = {
        "id": problem.get("id"),
        "action": problem.get("action"),
        "reason_code": problem.get("reason_code"),
        "cheap": problem.get("cheap"),
        "coe": problem.get("coe"),
    }
    encoded = json.dumps(
        projection,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def inspect_phase_artifacts(
    *,
    results_dir: Path,
    problem_id: str,
) -> tuple[dict[str, Any], Mapping[str, Any] | None, Mapping[str, Any] | None]:
    """Inspect one phase's summary and report using campaign semantics.

    The returned tuple contains the JSON-ready phase record, its matching
    summary row (when valid), and the parsed report (when valid).  Keeping this
    as a public helper lets adjacent campaign policies reuse exactly the same
    artifact validation without depending on private implementation details.
    """

    summary_path = results_dir / f"allstages_suite_summary_{problem_id}.json"
    loaded_summary = _load_json_mapping(summary_path)
    record: dict[str, Any] = {
        "summary": summary_path.name,
        "summary_status": loaded_summary.status,
        "process_status": "unknown",
        "report": None,
        "report_status": "not_checked",
        "selection_status": "not_checked",
        "selection_reason_code": None,
    }
    if loaded_summary.status != "ok" or loaded_summary.payload is None:
        return record, None, None
    row = _summary_result(loaded_summary.payload, problem_id)
    if row is None or not isinstance(row.get("success"), bool):
        record["summary_status"] = "invalid"
        return record, None, None
    record["process_status"] = "success" if row["success"] else "failed"
    if row["success"] is not True:
        return record, row, None
    report_path = _report_path(results_dir, problem_id, row)
    record["report"] = _path_name(report_path)
    if report_path is None:
        record["report_status"] = "missing_or_ambiguous"
        return record, row, None
    loaded_report = _load_json_mapping(report_path)
    record["report_status"] = loaded_report.status
    if loaded_report.status != "ok" or loaded_report.payload is None:
        return record, row, None
    decision = classify_report_selection(loaded_report.payload)
    record["selection_status"] = "eligible" if decision.eligible else "ineligible"
    record["selection_reason_code"] = decision.reason_code
    return record, row, loaded_report.payload


def _prefix_reason(prefix: str, report_code: str) -> str:
    if report_code == "selection_eligible" or report_code == "stagec_selection_eligible":
        return f"{prefix}_selection_eligible"
    return f"{prefix}_{report_code}"


def _classify_problem(
    problem_id: str,
    *,
    cheap_results: Path,
    coe_results: Path | None,
) -> dict[str, Any]:
    cheap, cheap_row, cheap_report = inspect_phase_artifacts(
        results_dir=cheap_results,
        problem_id=problem_id,
    )
    problem: dict[str, Any] = {
        "id": problem_id,
        "action": ACTION_PENDING,
        "reason_code": "cheap_summary_missing",
        "reason": reason_text("cheap_summary_missing"),
        "cheap": cheap,
        "coe": None,
    }

    if cheap["summary_status"] == "missing":
        pass
    elif cheap["summary_status"] != "ok":
        problem.update(
            action=ACTION_RETRY_CHEAP,
            reason_code="cheap_summary_invalid",
            reason=reason_text("cheap_summary_invalid"),
        )
    elif cheap_row is None:
        problem.update(
            action=ACTION_RETRY_CHEAP,
            reason_code="cheap_summary_invalid",
            reason=reason_text("cheap_summary_invalid"),
        )
    elif cheap["process_status"] == "failed":
        problem.update(
            action=ACTION_RETRY_CHEAP,
            reason_code="cheap_process_failed",
            reason=reason_text("cheap_process_failed"),
        )
    elif cheap["report_status"] == "missing_or_ambiguous":
        problem.update(
            action=ACTION_RETRY_CHEAP,
            reason_code="cheap_report_missing",
            reason=reason_text("cheap_report_missing"),
        )
    elif cheap["report_status"] != "ok" or cheap_report is None:
        problem.update(
            action=ACTION_RETRY_CHEAP,
            reason_code="cheap_report_invalid",
            reason=reason_text("cheap_report_invalid"),
        )
    elif cheap["selection_status"] == "eligible":
        problem.update(
            action=ACTION_SKIP,
            reason_code="cheap_selection_eligible",
            reason=reason_text("cheap_selection_eligible"),
        )
    else:
        report_code = str(cheap.get("selection_reason_code") or "no_symbolic_selection")
        prefixed = _prefix_reason("cheap", report_code)
        problem.update(
            action=ACTION_RUN_COE,
            reason_code=prefixed,
            reason=reason_text(prefixed),
        )

    if problem["action"] == ACTION_RUN_COE and coe_results is not None:
        coe, coe_row, coe_report = inspect_phase_artifacts(
            results_dir=coe_results,
            problem_id=problem_id,
        )
        problem["coe"] = coe
        if coe["summary_status"] == "missing":
            pass
        elif coe["summary_status"] != "ok" or coe_row is None:
            problem.update(
                action=ACTION_RETRY_COE,
                reason_code="coe_summary_invalid",
                reason=reason_text("coe_summary_invalid"),
            )
        elif coe["process_status"] == "failed":
            problem.update(
                action=ACTION_RETRY_COE,
                reason_code="coe_process_failed",
                reason=reason_text("coe_process_failed"),
            )
        elif coe["report_status"] == "missing_or_ambiguous":
            problem.update(
                action=ACTION_RETRY_COE,
                reason_code="coe_report_missing",
                reason=reason_text("coe_report_missing"),
            )
        elif coe["report_status"] != "ok" or coe_report is None:
            problem.update(
                action=ACTION_RETRY_COE,
                reason_code="coe_report_invalid",
                reason=reason_text("coe_report_invalid"),
            )
        elif coe["selection_status"] == "eligible":
            problem.update(
                action=ACTION_SKIP,
                reason_code="coe_selection_eligible",
                reason=reason_text("coe_selection_eligible"),
            )
        else:
            problem.update(
                action=ACTION_TERMINAL_FAILURE,
                reason_code="coe_no_eligible_selection",
                reason=reason_text("coe_no_eligible_selection"),
            )

    problem["decision_digest"] = _decision_digest(problem)
    return problem


def build_escalation_manifest(
    *,
    cheap_results: str | os.PathLike[str],
    coe_results: str | os.PathLike[str] | None,
    problem_ids: Sequence[Any],
    excluded_ids: Sequence[Any] = (),
) -> dict[str, Any]:
    """Build a deterministic manifest for one cheap-first campaign."""

    requested = parse_problem_ids(problem_ids)
    excluded = set(parse_problem_ids(excluded_ids))
    active = [problem_id for problem_id in requested if problem_id not in excluded]
    cheap_path = Path(cheap_results)
    coe_path = None if coe_results is None else Path(coe_results)
    problems = [
        _classify_problem(
            problem_id,
            cheap_results=cheap_path,
            coe_results=coe_path,
        )
        for problem_id in active
    ]
    counts = {
        action: sum(1 for problem in problems if problem["action"] == action)
        for action in MANIFEST_ACTIONS
    }
    ids_by_action = {
        f"{action}_ids": [
            problem["id"] for problem in problems if problem["action"] == action
        ]
        for action in MANIFEST_ACTIONS
    }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "policy": {
            "name": POLICY_NAME,
            "truth_blind": True,
            "process_failures": "retry_same_phase",
            "coe_trigger": (
                "completed cheap report has no eligible symbolic selection "
                "under the truth-blind quality and unit policy"
            ),
            "timestamps_in_manifest": False,
        },
        "requested_ids": requested,
        "excluded_ids": sorted(excluded, key=lambda item: int(item[2:])),
        "active_ids": active,
        "counts": counts,
        **ids_by_action,
        "problems": problems,
    }


def manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Serialize a manifest canonically for byte-for-byte reproducibility."""

    return (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def write_manifest_atomic(
    path: str | os.PathLike[str],
    manifest: Mapping[str, Any],
) -> None:
    """Atomically replace a manifest after fully serializing it."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest_bytes(manifest)
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.",
        dir=output.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _manifest_problem_ids(
    *,
    ids: str | None,
    start_id: int,
    end_id: int,
) -> list[str]:
    if ids is not None:
        parsed = parse_problem_ids(ids)
        if not parsed:
            raise ValueError("--ids was provided but contained no problem IDs")
        return parsed
    return problem_id_range(start_id, end_id)


def _load_manifest(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    loaded = _load_json_mapping(Path(path))
    if loaded.status != "ok" or loaded.payload is None:
        raise ValueError(f"invalid escalation manifest: {path}")
    if loaded.payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "unsupported escalation manifest schema: "
            f"{loaded.payload.get('schema_version')!r}"
        )
    return loaded.payload


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build",
        help="classify campaign artifacts and atomically write the manifest",
    )
    build.add_argument("--cheap-results", required=True)
    build.add_argument("--coe-results")
    build.add_argument("--output", required=True)
    build.add_argument("--ids")
    build.add_argument("--start-id", type=int, default=0)
    build.add_argument("--end-id", type=int, default=119)
    build.add_argument("--skip-ids", default="")
    build.add_argument("--ignore-file")
    build.add_argument("--quiet", action="store_true")

    list_parser = subparsers.add_parser(
        "list",
        help="print IDs having one or more requested actions",
    )
    list_parser.add_argument("--manifest", required=True)
    list_parser.add_argument(
        "--action",
        action="append",
        choices=MANIFEST_ACTIONS,
        required=True,
    )
    list_parser.add_argument(
        "--lines",
        action="store_true",
        help="print one ID per line instead of one shell-friendly line",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point used by the campaign wrapper."""

    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            requested = _manifest_problem_ids(
                ids=args.ids,
                start_id=args.start_id,
                end_id=args.end_id,
            )
            excluded = parse_problem_ids(args.skip_ids)
            excluded.extend(ignored_problem_ids(args.ignore_file))
            manifest = build_escalation_manifest(
                cheap_results=args.cheap_results,
                coe_results=args.coe_results,
                problem_ids=requested,
                excluded_ids=excluded,
            )
            write_manifest_atomic(args.output, manifest)
            if not args.quiet:
                counts = ", ".join(
                    f"{action}={manifest['counts'][action]}"
                    for action in MANIFEST_ACTIONS
                    if manifest["counts"][action]
                )
                print(
                    f"Wrote {args.output}: {counts or 'no active problems'}",
                    file=sys.stderr,
                )
            return 0

        manifest = _load_manifest(args.manifest)
        wanted = set(args.action)
        problem_rows = manifest.get("problems")
        if not isinstance(problem_rows, list):
            raise ValueError("invalid escalation manifest: problems must be a list")
        selected: list[str] = []
        for row in problem_rows:
            if not isinstance(row, Mapping) or row.get("action") not in wanted:
                continue
            selected.append(normalize_problem_id(row.get("id")))
        selected = parse_problem_ids(selected)
        separator = "\n" if args.lines else " "
        if selected:
            print(separator.join(selected))
        return 0
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 2


__all__ = [
    "ACTION_PENDING",
    "ACTION_RETRY_CHEAP",
    "ACTION_RETRY_COE",
    "ACTION_RUN_COE",
    "ACTION_SKIP",
    "ACTION_TERMINAL_FAILURE",
    "MANIFEST_ACTIONS",
    "MANIFEST_SCHEMA_VERSION",
    "OUTCOME_SCHEMA_VERSION",
    "POLICY_NAME",
    "ReportDecision",
    "build_escalation_manifest",
    "classify_report_selection",
    "inspect_phase_artifacts",
    "ignored_problem_ids",
    "manifest_bytes",
    "normalize_problem_id",
    "parse_problem_ids",
    "problem_id_range",
    "reason_text",
    "report_campaign_outcome",
    "write_manifest_atomic",
]


if __name__ == "__main__":
    raise SystemExit(main())
