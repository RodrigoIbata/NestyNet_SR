# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Core factorized symbolic search engine contracts and search-signal types."""

from __future__ import annotations

from importlib import import_module

from .api import ArchiveRecord, EngineRequest, EngineResult, SearchPolicy
from .actions import apply_action_impl, apply_crossover_action_impl, apply_residual_action_impl
from .archive import ResidualBasinArchive, Elite, Rec
from .signals import (
    CandidateStateFeatures,
    InverseSteeringPotential,
    ModeStateFeatures,
    PathStateFeatures,
    path_concentration,
    path_distribution_metrics,
    path_summary_stats,
    summarize_path_rows,
)

_LAZY_EXPORTS = {
    "Explorer": (".search", "Explorer"),
    "_eval_node_hparam_safe": (".scoring", "_eval_node_hparam_safe"),
    "_harvest_pool_from_archive": (".scoring", "_harvest_pool_from_archive"),
    "_mapping_equiv_root": (".scoring", "_mapping_equiv_root"),
    "fingerprint": (".scoring", "fingerprint"),
    "run_explorer_core": (".search", "run_explorer_core"),
    "score_expr": (".scoring", "score_expr"),
}

__all__ = [
    "apply_action_impl",
    "apply_crossover_action_impl",
    "apply_residual_action_impl",
    "ArchiveRecord",
    "ResidualBasinArchive",
    "CandidateStateFeatures",
    "EngineRequest",
    "EngineResult",
    "Explorer",
    "Elite",
    "InverseSteeringPotential",
    "ModeStateFeatures",
    "PathStateFeatures",
    "Rec",
    "SearchPolicy",
    "_eval_node_hparam_safe",
    "_harvest_pool_from_archive",
    "_mapping_equiv_root",
    "fingerprint",
    "path_concentration",
    "path_distribution_metrics",
    "path_summary_stats",
    "run_explorer_core",
    "score_expr",
    "summarize_path_rows",
]


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
