# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Bounded e-graph facade.

This module intentionally does not add a required external e-graph dependency.
When enabled, it records a bounded run and falls back to QDAG canonicalization.
"""

from __future__ import annotations

import time
from typing import Any

from .config import coerce_expr_ir_config
from .stats import ExpressionIRStats


def normalize_with_egraph_tuple(node: Any, cfg: object | None = None, stats: ExpressionIRStats | None = None, signature_context: Any = None):
    from .tuple_bridge import canonicalize_tuple_ast

    c = coerce_expr_ir_config(cfg)
    if isinstance(stats, ExpressionIRStats):
        stats.egraph_runs += 1
    t0 = time.perf_counter()
    try:
        from nestynet_sr.sr_search.factorized_search.expr_ast import node_size

        if int(node_size(node)) > int(c.egraph_max_input_size):
            if isinstance(stats, ExpressionIRStats):
                stats.egraph_limit_hits += 1
            return canonicalize_tuple_ast(node, c, stats=stats, signature_context=signature_context)
    finally:
        if isinstance(stats, ExpressionIRStats):
            stats.egraph_time_ms_total += 1000.0 * (time.perf_counter() - t0)
    return canonicalize_tuple_ast(node, c, stats=stats, signature_context=signature_context)


def normalize_with_egraph_core(node: Any, cfg: object | None = None, stats: ExpressionIRStats | None = None, signature_context: Any = None):
    from .core_bridge import canonicalize_core_ast

    c = coerce_expr_ir_config(cfg)
    if isinstance(stats, ExpressionIRStats):
        stats.egraph_runs += 1
    t0 = time.perf_counter()
    try:
        # Core node size is not centralized here; QDAG bounds still apply.
        return canonicalize_core_ast(node, c, stats=stats, signature_context=signature_context)
    finally:
        if isinstance(stats, ExpressionIRStats):
            stats.egraph_time_ms_total += 1000.0 * (time.perf_counter() - t0)


__all__ = ["normalize_with_egraph_core", "normalize_with_egraph_tuple"]
