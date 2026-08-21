# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Statistical certification for frozen SR/DE candidate archives."""

from .archive import CandidateArchive, CandidateSpec, candidate_id_for
from .audit import AuditDesign, LossAudit, UnitLossRecord
from .certificate import ParetoCertificate, build_certificate
from .complexity import ComplexityVector, validate_complexity_collection
from .sr_pipeline import (
    DeploymentNoninferiorityResult,
    NoPortableAnalyticCandidatesError,
    SRAuditEvaluation,
    SRAuditPlan,
    SRArchiveBuild,
    SRSelectionOutcome,
    build_sr_candidate_archive,
    certify_sr_archive,
    evaluate_sr_archive,
    format_sr_statistical_selection,
    prepare_sr_audit_plan,
    run_sr_statistical_selection,
    update_report_with_sr_statistical_selection,
)
from .pareto import (
    ConfidenceParetoResult,
    PairwiseRiskComparison,
    bootstrap_front_inclusion_frequencies,
    confidence_pareto,
    point_pareto_front,
)

__all__ = [
    "CalibrationProfile",
    "CalibrationCell",
    "MAXT_PROFILE_V1",
    "select_inference_method",
    "CandidateArchive",
    "CandidateSpec",
    "AuditDesign",
    "ComplexityVector",
    "ConfidenceParetoResult",
    "DeploymentNoninferiorityResult",
    "LossAudit",
    "NoPortableAnalyticCandidatesError",
    "PairwiseRiskComparison",
    "ParetoCertificate",
    "SRAuditEvaluation",
    "SRAuditPlan",
    "SRArchiveBuild",
    "SRSelectionOutcome",
    "UnitLossRecord",
    "bootstrap_front_inclusion_frequencies",
    "build_sr_candidate_archive",
    "certify_sr_archive",
    "evaluate_sr_archive",
    "format_sr_statistical_selection",
    "prepare_sr_audit_plan",
    "run_sr_statistical_selection",
    "update_report_with_sr_statistical_selection",
    "build_certificate",
    "candidate_id_for",
    "confidence_pareto",
    "point_pareto_front",
    "validate_complexity_collection",
]

from .de_pipeline import DEAuditPlan, prepare_de_audit_plan, build_de_archive, run_de_statistical_selection
from .calibration_profile import (
    MAXT_PROFILE_V1,
    CalibrationCell,
    CalibrationProfile,
    select_inference_method,
)
from .uncertainty import CoherentLossDraws, calibration_smoke_test, structural_rediscovery_summary
from .committee_inference import (
    CommitteeMaxTDecision,
    CommitteeMemberVerdict,
    CommitteeShardStats,
    committee_maxt_decision,
    committee_shard_stats,
    reduce_committee_decision,
)

__all__ += [
    "CommitteeMaxTDecision",
    "CommitteeMemberVerdict",
    "CommitteeShardStats",
    "committee_maxt_decision",
    "committee_shard_stats",
    "reduce_committee_decision",
]
