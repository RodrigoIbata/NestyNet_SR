# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""DE committee audit/adjudication logic.

The committee consumes proposal/rollout summaries, applies one ranking policy,
and can either report an audit recommendation or drive final adjudication.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np

from .de_validation import DEWitnessResult, evaluate_compile_domain_witness
from .proposals import canonicalize_de_equation


@dataclass
class DECommitteeDecision:
    selected_id: str | None
    status: str
    selection_basis: str
    candidate_summary: list[dict[str, Any]]
    witness_results: list[dict[str, Any]]
    warnings: list[str]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


_STATUS_RANK = {
    "PASS": 0,
    "PARTIAL": 1,
    "FAIL": 2,
    "UNVERIFIED": 3,
    "ERROR": 4,
}


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


def _safe_float(value: Any, default: float = float("inf")) -> float:
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


def _support_count(proposal: Mapping[str, Any] | None) -> int:
    if not isinstance(proposal, Mapping):
        return 0
    support = proposal.get("support", {})
    if isinstance(support, Mapping):
        try:
            return int(support.get("support_count", 0) or 0)
        except Exception:
            return 0
    return 0


def _typed_confidence(proposal: Mapping[str, Any] | None, row: Mapping[str, Any] | None = None) -> str:
    for source in (row, (proposal or {}).get("rhs_payload", None), proposal):
        if not isinstance(source, Mapping):
            continue
        typed = source.get("typed_metadata", None)
        if isinstance(typed, Mapping):
            confidence = str(typed.get("collapse_confidence", "") or "").strip().lower()
            reason = str(typed.get("collapse_reason", "") or "").strip().lower()
        else:
            confidence = str(source.get("collapse_confidence", "") or "").strip().lower()
            reason = str(source.get("collapse_reason", "") or "").strip().lower()
        if confidence == "high" and reason in {"", "ok"}:
            return "high"
        if confidence in {"weak", "low"}:
            return confidence
    return ""


def _typed_confidence_rank(confidence: str | None) -> int:
    return {"high": 0, "weak": 1, "low": 2}.get(str(confidence or "").strip().lower(), 3)


def _pointwise_score(proposal: Mapping[str, Any] | None, row: Mapping[str, Any] | None = None) -> float:
    for source in (row, proposal):
        if not isinstance(source, Mapping):
            continue
        metrics = source.get("pointwise_metrics", source)
        if not isinstance(metrics, Mapping):
            continue
        for key in ("residual_mse", "probe_mse", "mse", "score"):
            val = _safe_float(metrics.get(key, None), float("inf"))
            if math.isfinite(val):
                return val
        for key in ("probe_rms", "rms_val", "rms_train"):
            val = _safe_float(metrics.get(key, None), float("inf"))
            if math.isfinite(val):
                return float(val * val)
    return float("inf")


def _traj_nrmse_values(row: Mapping[str, Any] | None) -> list[float]:
    vals: list[float] = []
    if not isinstance(row, Mapping):
        return vals
    for score in list(row.get("traj_scores", []) or []):
        if not isinstance(score, Mapping):
            continue
        val = _safe_float(score.get("nrmse", float("inf")), float("inf"))
        if math.isfinite(val):
            vals.append(float(val))
    return vals


def _worst_nrmse(row: Mapping[str, Any] | None) -> float:
    vals = _traj_nrmse_values(row)
    return float(max(vals)) if vals else float("inf")


def _median_nrmse(row: Mapping[str, Any] | None) -> float:
    vals = _traj_nrmse_values(row)
    return float(median(vals)) if vals else float("inf")


def _loss_bucket(loss: float, tolerance: float) -> float | int:
    if not math.isfinite(loss):
        return float("inf")
    tol = float(tolerance)
    if tol <= 0.0 or not math.isfinite(tol):
        return float(loss)
    return int(math.floor(float(loss) / tol))


def _compile_status_rank(status: str | None) -> int:
    status_s = str(status or "UNVERIFIED").upper()
    if status_s == "PASS":
        return 0
    if status_s in ("UNVERIFIED", "NONE"):
        return 1
    if status_s == "FAIL":
        return 2
    return 3


def _proposal_lookup_key(proposal: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        proposal.get("canonical_key"),
        _safe_int(proposal.get("order", 0), 0),
        _safe_int(proposal.get("x_axis", 0), 0),
    )


def _row_lookup_key(row: Mapping[str, Any]) -> tuple[Any, ...] | None:
    canonical_key = row.get("canonical_key", None)
    if canonical_key is None:
        eq = row.get("canonical_equation", None)
        if eq not in (None, ""):
            canonical_key = canonicalize_de_equation(eq)
    if canonical_key is None:
        return None
    return (
        canonical_key,
        _safe_int(row.get("discovered_order", row.get("order", 0)), 0),
        _safe_int(row.get("x_axis", 0), 0),
    )


def _index_proposals(proposals: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Mapping[str, Any]], dict[tuple[Any, ...], Mapping[str, Any]]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    by_key: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for proposal in proposals:
        if not isinstance(proposal, Mapping):
            continue
        pid = str(proposal.get("proposal_id", "") or "")
        if pid:
            by_id[pid] = proposal
        by_key.setdefault(_proposal_lookup_key(proposal), proposal)
    return by_id, by_key


def _proposal_for_rollout_row(
    row: Mapping[str, Any],
    *,
    by_id: Mapping[str, Mapping[str, Any]],
    by_key: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    pid = str(row.get("proposal_id", "") or "")
    if pid and pid in by_id:
        return by_id[pid]
    key = _row_lookup_key(row)
    if key is not None and key in by_key:
        return by_key[key]
    return None


def _summary_from_proposal(
    proposal: Mapping[str, Any],
    *,
    compile_result: DEWitnessResult | None,
    selected_engine: str | None,
) -> dict[str, Any]:
    pointwise = _pointwise_score(proposal)
    compile_status = str(compile_result.status if compile_result is not None else "UNVERIFIED")
    status = "UNVERIFIED" if compile_status == "PASS" else compile_status
    support_count = _support_count(proposal)
    typed_confidence = _typed_confidence(proposal)
    return {
        "proposal_id": str(proposal.get("proposal_id", "")),
        "engine": str(proposal.get("engine", "")),
        "role_signature": str(proposal.get("role_signature", "")),
        "canonical_key": str(proposal.get("canonical_key", "")),
        "canonical_equation": str(proposal.get("canonical_equation", "")),
        "order": _safe_int(proposal.get("order", 0), 0),
        "x_axis": _safe_int(proposal.get("x_axis", 0), 0),
        "status": status,
        "compile_status": compile_status,
        "rollout_status": None,
        "worst_rollout_nrmse": None,
        "median_rollout_nrmse": None,
        "pointwise_residual": pointwise if math.isfinite(pointwise) else None,
        "complexity": _safe_float(proposal.get("complexity", 0.0), 0.0),
        "support_count": support_count,
        "typed_confidence": typed_confidence,
        "legacy_selected_engine": bool(str(proposal.get("engine", "")) == str(selected_engine or "")),
        "fatal_failures": 0 if compile_status == "PASS" else 1,
    }


def _summary_from_rollout_row(
    row: Mapping[str, Any],
    *,
    proposal: Mapping[str, Any] | None,
    selected_engine: str | None,
) -> dict[str, Any]:
    status = str(row.get("status", "ERROR") or "ERROR").upper()
    proposal_id = str(row.get("proposal_id", "") or "")
    if not proposal_id and isinstance(proposal, Mapping):
        proposal_id = str(proposal.get("proposal_id", "") or "")
    if not proposal_id:
        proposal_id = f"rollout:{row.get('engine', 'candidate')}:{row.get('candidate_rank', '')}"
    engine = str(row.get("engine", "") or (proposal or {}).get("engine", ""))
    worst = _worst_nrmse(row)
    med = _median_nrmse(row)
    pointwise = _pointwise_score(proposal, row)
    compile_status = str(row.get("compile_status", "") or "")
    if not compile_status and isinstance(proposal, Mapping):
        compile_status = str(proposal.get("compile_status", "") or "UNVERIFIED")
    if not compile_status:
        compile_status = "UNVERIFIED"
    typed_confidence = _typed_confidence(proposal, row)
    return {
        "proposal_id": proposal_id,
        "engine": engine,
        "role_signature": str((proposal or {}).get("role_signature", row.get("role_signature", ""))),
        "canonical_key": str((proposal or {}).get("canonical_key", row.get("canonical_key", ""))),
        "canonical_equation": str(row.get("canonical_equation", (proposal or {}).get("canonical_equation", ""))),
        "order": _safe_int(row.get("discovered_order", row.get("order", (proposal or {}).get("order", 0))), 0),
        "x_axis": _safe_int(row.get("x_axis", (proposal or {}).get("x_axis", 0)), 0),
        "status": status,
        "compile_status": compile_status,
        "rollout_status": status,
        "worst_rollout_nrmse": worst if math.isfinite(worst) else None,
        "median_rollout_nrmse": med if math.isfinite(med) else None,
        "pointwise_residual": pointwise if math.isfinite(pointwise) else None,
        "complexity": _safe_float(row.get("complexity", (proposal or {}).get("complexity", 0.0)), 0.0),
        "support_count": _support_count(proposal),
        "typed_confidence": typed_confidence,
        "legacy_selected_engine": bool(engine == str(selected_engine or "")),
        "candidate_rank": row.get("candidate_rank", None),
        "fatal_failures": 1 if status == "ERROR" else 0,
    }


def _rank_summary(row: Mapping[str, Any], *, tolerance_nrmse: float) -> tuple[Any, ...]:
    status = str(row.get("status", "ERROR") or "ERROR").upper()
    worst = _safe_float(row.get("worst_rollout_nrmse", None), float("inf"))
    med = _safe_float(row.get("median_rollout_nrmse", None), float("inf"))
    pointwise = _safe_float(row.get("pointwise_residual", None), float("inf"))
    return (
        _safe_int(row.get("fatal_failures", 0), 0),
        int(_STATUS_RANK.get(status, len(_STATUS_RANK) + 1)),
        _loss_bucket(worst, tolerance_nrmse),
        _loss_bucket(med, tolerance_nrmse),
        pointwise,
        _typed_confidence_rank(str(row.get("typed_confidence", ""))),
        -_safe_int(row.get("support_count", 0), 0),
        _safe_float(row.get("complexity", 0.0), 0.0),
        _compile_status_rank(str(row.get("compile_status", "UNVERIFIED"))),
        str(row.get("proposal_id", "")),
    )


def _compile_proposal(
    proposal: Mapping[str, Any],
    *,
    domain_samples: Sequence[Mapping[str, Any] | Sequence[float]] | None,
) -> DEWitnessResult:
    payload = proposal.get("rhs_payload", proposal)
    if not isinstance(payload, Mapping):
        payload = proposal
    return evaluate_compile_domain_witness(
        payload,
        engine=str(proposal.get("engine", "")),
        proposal_id=str(proposal.get("proposal_id", "")),
        witness_id=f"compile_domain:{proposal.get('proposal_id', '')}",
        domain_samples=domain_samples,
    )


def run_de_committee_audit(
    proposal_slate: Sequence[Mapping[str, Any]] | None,
    *,
    rollout_candidates: Sequence[Mapping[str, Any]] | None = None,
    selected_engine: str | None = None,
    config: Mapping[str, Any] | None = None,
    run_compile_domain: bool = True,
    domain_samples: Sequence[Mapping[str, Any] | Sequence[float]] | None = None,
) -> DECommitteeDecision:
    proposals = [p for p in list(proposal_slate or []) if isinstance(p, Mapping)]
    cfg = dict(config or {})
    tolerance_nrmse = _safe_float(cfg.get("tolerance_nrmse", 0.0), 0.0)
    warnings: list[str] = []
    witness_results: list[dict[str, Any]] = []

    compile_by_id: dict[str, DEWitnessResult] = {}
    if run_compile_domain:
        for proposal in proposals:
            result = _compile_proposal(proposal, domain_samples=domain_samples)
            pid = str(proposal.get("proposal_id", "") or "")
            if pid:
                compile_by_id[pid] = result
            witness_results.append(result.to_dict())

    summaries: list[dict[str, Any]] = []
    rollout_rows = [row for row in list(rollout_candidates or []) if isinstance(row, Mapping)]
    by_id, by_key = _index_proposals(proposals)
    if rollout_rows:
        for idx, row in enumerate(rollout_rows):
            proposal = _proposal_for_rollout_row(row, by_id=by_id, by_key=by_key)
            summary = _summary_from_rollout_row(
                row,
                proposal=proposal,
                selected_engine=selected_engine,
            )
            if not summary.get("proposal_id"):
                summary["proposal_id"] = f"rollout:{idx}"
            pid = str(summary.get("proposal_id", ""))
            if pid in compile_by_id:
                summary["compile_status"] = str(compile_by_id[pid].status)
                if str(compile_by_id[pid].status) != "PASS":
                    summary["fatal_failures"] = 1
            summaries.append(summary)
            witness_results.append(
                {
                    "proposal_id": pid,
                    "witness_id": f"rollout:{idx}",
                    "tier": "rollout_summary",
                    "status": str(row.get("status", "ERROR")),
                    "residual_mse": None,
                    "rollout_nrmse": summary.get("worst_rollout_nrmse", None),
                    "max_abs_error": None,
                    "blew_up": False,
                    "solve_time_s": 0.0,
                    "vote_vs_incumbent": None,
                    "failure_kind": row.get("failure_kind", None),
                    "metrics": {
                        "engine": summary.get("engine", ""),
                        "traj_scores": row.get("traj_scores", []),
                        "message": row.get("message", ""),
                    },
                }
            )
    else:
        warnings.append("rollout witnesses unavailable; audit ranks compile/domain and pointwise residual only")
        for proposal in proposals:
            pid = str(proposal.get("proposal_id", "") or "")
            summaries.append(
                _summary_from_proposal(
                    proposal,
                    compile_result=compile_by_id.get(pid, None),
                    selected_engine=selected_engine,
                )
            )

    valid = [row for row in summaries if _safe_int(row.get("fatal_failures", 0), 0) <= 0]
    selected_id = None
    status = "no_candidates"
    if valid:
        valid.sort(key=lambda row: _rank_summary(row, tolerance_nrmse=tolerance_nrmse))
        selected_id = str(valid[0].get("proposal_id", "")) or None
        status = "selected"
    elif summaries:
        status = "no_valid_candidates"

    summaries_sorted = sorted(summaries, key=lambda row: _rank_summary(row, tolerance_nrmse=tolerance_nrmse))
    for rank, row in enumerate(summaries_sorted):
        row["committee_rank"] = int(rank)
        row["committee_selected"] = bool(selected_id is not None and str(row.get("proposal_id", "")) == str(selected_id))

    basis = "rollout_worst_nrmse" if rollout_rows else "compile_domain_pointwise_residual"
    return DECommitteeDecision(
        selected_id=selected_id,
        status=status,
        selection_basis=basis,
        candidate_summary=_jsonable(summaries_sorted),
        witness_results=_jsonable(witness_results),
        warnings=warnings,
        config=_jsonable({"mode": "audit", **cfg}),
    )


def selected_engine_from_decision(decision: Mapping[str, Any] | DECommitteeDecision | None) -> str | None:
    row = selected_summary_from_decision(decision)
    if row is None:
        return None
    engine = row.get("engine", None)
    return None if engine is None else str(engine)


def selected_summary_from_decision(
    decision: Mapping[str, Any] | DECommitteeDecision | None,
) -> dict[str, Any] | None:
    payload = decision.to_dict() if isinstance(decision, DECommitteeDecision) else decision
    if not isinstance(payload, Mapping):
        return None
    selected_id = str(payload.get("selected_id", "") or "")
    if not selected_id:
        return None
    for row in list(payload.get("candidate_summary", []) or []):
        if isinstance(row, Mapping) and str(row.get("proposal_id", "")) == selected_id:
            return dict(row)
    return None


def tied_candidate_summaries_from_decision(
    decision: Mapping[str, Any] | DECommitteeDecision | None,
    *,
    tolerance_nrmse: float = 1.0e-3,
    max_candidates: int = 4,
    max_per_role: int = 2,
) -> list[dict[str, Any]]:
    """Return committee summaries tied closely enough to justify late CSR."""

    payload = decision.to_dict() if isinstance(decision, DECommitteeDecision) else decision
    if not isinstance(payload, Mapping):
        return []
    selected = selected_summary_from_decision(payload)
    if not isinstance(selected, dict):
        return []

    selected_status = str(selected.get("status", "ERROR") or "ERROR").upper()
    selected_status_rank = int(_STATUS_RANK.get(selected_status, len(_STATUS_RANK) + 1))
    selected_worst = _safe_float(selected.get("worst_rollout_nrmse", None), float("inf"))
    selected_med = _safe_float(selected.get("median_rollout_nrmse", None), float("inf"))
    tol = max(0.0, _safe_float(tolerance_nrmse, 0.0))

    def _within(candidate_value: float, selected_value: float) -> bool:
        if not math.isfinite(candidate_value) or not math.isfinite(selected_value):
            return candidate_value == selected_value
        allowed = max(float(tol), float(tol) * max(1.0, abs(selected_value)))
        return candidate_value <= selected_value + allowed

    tied: list[dict[str, Any]] = []
    per_role: dict[str, int] = {}
    for row in list(payload.get("candidate_summary", []) or []):
        if not isinstance(row, Mapping):
            continue
        if _safe_int(row.get("fatal_failures", 0), 0) > 0:
            continue
        status = str(row.get("status", "ERROR") or "ERROR").upper()
        if int(_STATUS_RANK.get(status, len(_STATUS_RANK) + 1)) != selected_status_rank:
            continue
        worst = _safe_float(row.get("worst_rollout_nrmse", None), float("inf"))
        med = _safe_float(row.get("median_rollout_nrmse", None), float("inf"))
        if not _within(worst, selected_worst):
            continue
        if math.isfinite(selected_med) and not _within(med, selected_med):
            continue
        role = str(row.get("role_signature", "") or row.get("engine", ""))
        if int(max_per_role) > 0 and int(per_role.get(role, 0)) >= int(max_per_role):
            continue
        per_role[role] = int(per_role.get(role, 0)) + 1
        tied.append(dict(row))
        if int(max_candidates) > 0 and len(tied) >= int(max_candidates):
            break
    return tied


__all__ = [
    "DECommitteeDecision",
    "run_de_committee_audit",
    "selected_engine_from_decision",
    "selected_summary_from_decision",
    "tied_candidate_summaries_from_decision",
]
