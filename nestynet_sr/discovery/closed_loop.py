# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from nestynet_sr.sr_core.units import UnitsSpec

from .active_design import ExperimentCandidate, resolve_disagreement_mode, select_next_experiment
from .committee import CommitteeMember, CommitteeState, build_committee_state, canonicalize_candidate_law
from .constant_lift import parameter_samples_from_local_constants
from .physics_tests import score_physics_consistency


@dataclass(frozen=True)
class ClosedLoopIterationResult:
    committee_state: CommitteeState
    selected_experiment: Mapping[str, Any] | None
    ranked_experiments: tuple[Mapping[str, Any], ...]
    physics_reports: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def _coerce_committee_member(candidate: CommitteeMember | Mapping[str, Any], index: int) -> CommitteeMember:
    if isinstance(candidate, CommitteeMember):
        return candidate
    row = dict(candidate)
    info = canonicalize_candidate_law(row.get("symbolic_structure", row.get("expr", row.get("law", None))))
    return CommitteeMember(
        member_id=str(row.get("member_id", "") or f"member_{int(index)}"),
        symbolic_structure=row.get("symbolic_structure", row.get("expr", row.get("law", None))),
        fitted_constants=dict(row.get("fitted_constants", {}) or {}),
        shared_constants=dict(row.get("shared_constants", {}) or {}),
        local_constants_by_experiment=dict(row.get("local_constants_by_experiment", {}) or {}),
        train_error=float(row.get("train_error", float("nan"))),
        validation_error=float(row.get("validation_error", float("nan"))),
        regime_holdout_error=row.get("regime_holdout_error", None),
        simplicity_score=float(row.get("simplicity_score", 1.0)),
        physics_consistency_score=float(row.get("physics_consistency_score", 1.0)),
        committee_weight=float(row.get("committee_weight", 0.0)),
        canonical_key=str(row.get("canonical_key", "") or info["canonical_key"]),
        display_expr=str(row.get("display_expr", "") or info["display_expr"]),
        metadata=dict(row.get("metadata", {}) or {}),
    )


def run_closed_loop_iteration(
    candidate_laws: Sequence[CommitteeMember | Mapping[str, Any]],
    experiment_candidates: Sequence[ExperimentCandidate],
    *,
    units_spec: UnitsSpec | None = None,
    symmetry_tests: Sequence[Any] | None = None,
    committee_max_members: int | None = None,
    weight_temperature: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    disagreement_mode: str | None = None,
    lambda_cost: float = 1.0,
    lambda_noise: float = 1.0,
    lambda_feasibility: float = 1.0,
) -> ClosedLoopIterationResult:
    physics_reports: dict[str, Mapping[str, Any]] = {}
    enriched: list[CommitteeMember] = []
    for index, candidate in enumerate(list(candidate_laws or [])):
        member = _coerce_committee_member(candidate, index)
        report = score_physics_consistency(
            {
                "symbolic_structure": member.symbolic_structure,
                "train_error": member.train_error,
                "validation_error": member.validation_error,
                "metadata": dict(member.metadata or {}),
            },
            units_spec=units_spec,
            symmetry_tests=symmetry_tests,
            parameter_samples=parameter_samples_from_local_constants(
                member.local_constants_by_experiment,
                regime_ids=dict(member.metadata or {}).get("dataset_ids", None),
            ),
        )
        physics_reports[str(member.member_id)] = report
        enriched.append(
            CommitteeMember(
                member_id=member.member_id,
                symbolic_structure=member.symbolic_structure,
                fitted_constants=member.fitted_constants,
                shared_constants=member.shared_constants,
                local_constants_by_experiment=member.local_constants_by_experiment,
                train_error=member.train_error,
                validation_error=member.validation_error,
                regime_holdout_error=member.regime_holdout_error,
                simplicity_score=member.simplicity_score,
                physics_consistency_score=float(report["overall_score"]),
                committee_weight=member.committee_weight,
                canonical_key=member.canonical_key,
                display_expr=member.display_expr,
                metadata=member.metadata,
            )
        )
    committee = build_committee_state(
        enriched,
        max_members=committee_max_members,
        deduplicate=True,
        weight_temperature=float(weight_temperature),
    )
    mode_name = resolve_disagreement_mode(disagreement_mode)
    selection = select_next_experiment(
        committee,
        experiment_candidates,
        beta=float(beta),
        gamma=float(gamma),
        disagreement_mode=mode_name,
        lambda_cost=float(lambda_cost),
        lambda_noise=float(lambda_noise),
        lambda_feasibility=float(lambda_feasibility),
    )
    return ClosedLoopIterationResult(
        committee_state=committee,
        selected_experiment=selection.get("selected", None),
        ranked_experiments=tuple(selection.get("ranking", []) or []),
        physics_reports=physics_reports,
    )


__all__ = [
    "ClosedLoopIterationResult",
    "run_closed_loop_iteration",
]
