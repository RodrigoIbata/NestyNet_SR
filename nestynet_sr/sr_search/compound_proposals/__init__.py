# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://www.mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Shared compound proposal objects and adapters.

This package is intentionally proposal-only.  Stage A and Stage B consume these
objects through their own validation/acceptance policies.
"""

from .core import (
    CompoundProposal,
    ProposalDim,
    compound_proposal_from_metric,
    proposal_signature,
    stageA_tuple_from_proposal,
    stageB_meta_from_proposal,
)
from .families import (
    build_barycentric_compound_proposals,
    build_logexp_compound_proposals,
    build_metric_distance_compound_proposals,
)
from .wrappers import WrappedExpression, apply_compound_wrapper, build_compound_wrappers, canonical_wrapper_name

__all__ = [
    "CompoundProposal",
    "ProposalDim",
    "WrappedExpression",
    "apply_compound_wrapper",
    "build_barycentric_compound_proposals",
    "build_logexp_compound_proposals",
    "build_metric_distance_compound_proposals",
    "build_compound_wrappers",
    "canonical_wrapper_name",
    "compound_proposal_from_metric",
    "proposal_signature",
    "stageA_tuple_from_proposal",
    "stageB_meta_from_proposal",
]
