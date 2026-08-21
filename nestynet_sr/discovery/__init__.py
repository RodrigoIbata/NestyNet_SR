# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from importlib import import_module

active_design = import_module(".active_design", __name__)
closed_loop_driver = import_module(".closed_loop_driver", __name__)
closed_loop = import_module(".closed_loop", __name__)
committee = import_module(".committee", __name__)
constant_lift = import_module(".constant_lift", __name__)
experiment_opt = import_module(".experiment_opt", __name__)
integration = import_module(".integration", __name__)
physics_tests = import_module(".physics_tests", __name__)

from .active_design import (
    ExperimentCandidate,
    committee_disagreement,
    resolve_disagreement_mode,
    resolve_surface_disagreement_mode,
    score_experiment_candidate,
    select_next_experiment,
)
from .closed_loop_driver import run_closed_loop_driver
from .closed_loop import ClosedLoopIterationResult, run_closed_loop_iteration
from .committee import (
    CommitteeMember,
    CommitteeState,
    build_committee_state,
    canonicalize_candidate_law,
)
from .constant_lift import (
    apply_constant_lift_proposals,
    discover_constant_lifts,
    parameter_samples_from_local_constants,
)
from .experiment_opt import optimize_continuous_experiment_candidates
from .integration import (
    RuntimeDiscoveryCandidate,
    build_sr_experiment_candidates,
    deserialize_committee_members,
    deserialize_experiment_candidates,
    discovery_summary_from_payload,
    run_closed_loop_from_discovery_payload,
    run_sr_discovery_integration,
    serialize_committee_member,
    serialize_experiment_candidate,
)
from .physics_tests import (
    PhysicsCheckResult,
    check_dimensional_consistency,
    check_parameter_stability,
    check_regime_generalization,
    check_residual_structure,
    score_physics_consistency,
)

__all__ = [
    "CommitteeMember",
    "CommitteeState",
    "build_committee_state",
    "canonicalize_candidate_law",
    "apply_constant_lift_proposals",
    "discover_constant_lifts",
    "optimize_continuous_experiment_candidates",
    "PhysicsCheckResult",
    "check_dimensional_consistency",
    "check_parameter_stability",
    "check_regime_generalization",
    "check_residual_structure",
    "parameter_samples_from_local_constants",
    "score_physics_consistency",
    "ExperimentCandidate",
    "committee_disagreement",
    "resolve_disagreement_mode",
    "resolve_surface_disagreement_mode",
    "score_experiment_candidate",
    "select_next_experiment",
    "run_closed_loop_driver",
    "RuntimeDiscoveryCandidate",
    "build_sr_experiment_candidates",
    "deserialize_committee_members",
    "deserialize_experiment_candidates",
    "discovery_summary_from_payload",
    "run_closed_loop_from_discovery_payload",
    "run_sr_discovery_integration",
    "serialize_committee_member",
    "serialize_experiment_candidate",
    "ClosedLoopIterationResult",
    "run_closed_loop_iteration",
]
