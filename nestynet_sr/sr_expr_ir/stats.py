# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Lightweight counters for expression IR ablations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExpressionIRStats:
    raw_candidates_seen: int = 0
    canonicalized_candidates: int = 0
    canonical_key_hits: int = 0
    duplicate_candidates_dropped: int = 0
    evals_skipped_duplicate: int = 0
    fallback_count: int = 0
    qdag_nodes_created: int = 0
    qdag_intern_hits: int = 0
    add_flattened: int = 0
    mul_flattened: int = 0
    like_terms_combined: int = 0
    powers_combined: int = 0
    constants_folded: int = 0
    signature_rejected: int = 0
    signature_unknown: int = 0
    invariant_seed_count: int = 0
    egraph_runs: int = 0
    egraph_time_ms_total: float = 0.0
    egraph_limit_hits: int = 0
    raw_projected_trees: int = 0
    unique_canonical_trees: int = 0
    raw_depth_reached: int = 0
    lowered_depth_max: int = 0
    lowered_size_max: int = 0
    gs_diagnostic_count: int = 0
    gs_constraint_count: int = 0
    gs_de_term_generator_count: int = 0
    gs_fss_score_count: int = 0
    gs_fss_aux_generator_count: int = 0
    examples: list[dict[str, Any]] = field(default_factory=list)

    def record_example(self, entry: dict[str, Any], *, limit: int = 0) -> None:
        if int(limit) <= 0 or len(self.examples) >= int(limit):
            return
        self.examples.append(dict(entry))

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        if self.unique_canonical_trees > 0:
            out["collapse_ratio"] = float(self.raw_projected_trees) / max(float(self.unique_canonical_trees), 1.0)
        else:
            out["collapse_ratio"] = None
        return out


__all__ = ["ExpressionIRStats"]
