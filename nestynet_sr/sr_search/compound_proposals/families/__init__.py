# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://www.mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Shared compound proposal families."""

from .barycentric import build_barycentric_compound_proposals
from .logexp import build_logexp_compound_proposals
from .metric import build_metric_distance_compound_proposals

__all__ = [
    "build_barycentric_compound_proposals",
    "build_logexp_compound_proposals",
    "build_metric_distance_compound_proposals",
]
