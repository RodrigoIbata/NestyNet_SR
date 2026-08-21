# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Compatibility facade for split factorized symbolic search repair feature modules."""

from __future__ import annotations

from .engine.signals import (
    CandidateStateFeatures,
    InverseSteeringPotential,
    ModeStateFeatures,
    PathStateFeatures,
    path_concentration,
    path_distribution_metrics,
    path_summary_stats,
    summarize_path_rows,
)
from .policy.features import (
    ParentStateFeatures,
    RepairControllerFeatureRecord,
    build_controller_state_record,
    coerce_repair_feature_record,
    coerce_repair_feature_row,
)

__all__ = [
    "CandidateStateFeatures",
    "InverseSteeringPotential",
    "ModeStateFeatures",
    "ParentStateFeatures",
    "PathStateFeatures",
    "RepairControllerFeatureRecord",
    "build_controller_state_record",
    "coerce_repair_feature_record",
    "coerce_repair_feature_row",
    "path_concentration",
    "path_distribution_metrics",
    "path_summary_stats",
    "summarize_path_rows",
]
