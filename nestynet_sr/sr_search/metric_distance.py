# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Compatibility shim for metric-distance compound proposals.

The metric-distance generator now lives in
``sr_search.compound_proposals.families.metric`` and emits shared
``CompoundProposal`` objects.  This module preserves the older
``MetricDistanceProposal`` API used by Stage A, Stage B, and tests while the
remaining compound families migrate to the shared proposal layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from nestynet_sr.sr_core.bridges import AtomNode, Node, clone_ast
from nestynet_sr.sr_search.compound_proposals import (
    CompoundProposal,
    compound_proposal_from_metric,
    stageA_tuple_from_proposal,
)
from nestynet_sr.sr_search.compound_proposals.families.metric import (
    build_metric_distance_compound_proposals,
)


@dataclass(frozen=True)
class MetricDistanceProposal:
    """Legacy metric-distance proposal shape."""

    label: str
    q_ast: Node
    z_ast: Node
    pattern: Tuple[int, ...]
    wrapper: str
    family: str
    confidence: float
    q_dim: Optional[Tuple[Any, ...]] = None
    z_dim: Optional[Tuple[Any, ...]] = None
    meta: Optional[dict] = None

    def to_compound_proposal(self) -> CompoundProposal:
        """Return the shared proposal-layer representation."""

        return compound_proposal_from_metric(self)


def _legacy_metric_from_compound(prop: CompoundProposal) -> MetricDistanceProposal:
    """Convert a shared metric proposal to the legacy public shape."""

    meta = dict(prop.meta or {})
    meta["kind"] = "metric_distance"
    meta["metric_family"] = str(prop.family)
    meta["metric_wrapper"] = str(prop.wrapper or "q")
    return MetricDistanceProposal(
        label=str(prop.label),
        q_ast=clone_ast(prop.base_ast if prop.base_ast is not None else prop.z_ast),
        z_ast=clone_ast(prop.z_ast),
        pattern=tuple(int(v) for v in prop.consumed_pattern),
        wrapper=str(prop.wrapper or "q"),
        family=str(prop.family),
        confidence=float(prop.confidence),
        q_dim=prop.base_dim,
        z_dim=prop.z_dim,
        meta=meta,
    )


def build_metric_distance_proposals(
    atom_or_inputs: AtomNode | Sequence[Node],
    *,
    units_spec: Any = None,
    include_polar: bool = True,
    include_cartesian: bool = True,
    wrappers: Sequence[str] = ("q", "sqrt_q", "inv_sqrt_q", "inv_q"),
    max_cartesian_pairs: int = 3,
    max_proposals: int = 16,
) -> List[MetricDistanceProposal]:
    """Build bounded metric-distance proposals over effective coordinates.

    This compatibility function returns ``MetricDistanceProposal`` objects, but
    the actual construction and unit filtering are delegated to the shared
    ``CompoundProposal`` family implementation.
    """

    props = build_metric_distance_compound_proposals(
        atom_or_inputs,
        units_spec=units_spec,
        include_polar=include_polar,
        include_cartesian=include_cartesian,
        wrappers=wrappers,
        max_cartesian_pairs=max_cartesian_pairs,
        max_proposals=max_proposals,
    )
    return [_legacy_metric_from_compound(p) for p in props]


def metric_stageA_tuple(prop: MetricDistanceProposal):
    """Convert a metric proposal to Stage A's legacy proposal tuple shape."""

    return stageA_tuple_from_proposal(prop.to_compound_proposal())
