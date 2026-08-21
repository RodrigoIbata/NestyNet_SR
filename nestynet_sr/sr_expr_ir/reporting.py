# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""JSON-friendly reporting helpers for expression IR experiments."""

from __future__ import annotations

from typing import Any

from .config import coerce_expr_ir_config, expr_ir_active, expression_ir_config_to_dict
from .stats import ExpressionIRStats


def _stats_dict(stats: ExpressionIRStats | None) -> dict[str, Any]:
    return stats.to_dict() if isinstance(stats, ExpressionIRStats) else ExpressionIRStats().to_dict()


def expression_ir_report(cfg: object | None, stats: ExpressionIRStats | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    c = coerce_expr_ir_config(cfg)
    payload = {
        "enabled": bool(expr_ir_active(c)),
        "mode": str(c.expr_ir),
        "canonicalize": str(c.canonicalize),
        "domain_mode": str(c.domain_mode),
        "active_reason": "active" if expr_ir_active(c) else "expr_ir_ast_or_canonicalize_off",
        "config": expression_ir_config_to_dict(c),
        "stats": _stats_dict(stats),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def attach_expression_ir_report(result_dict: dict[str, Any], cfg: object | None, stats: ExpressionIRStats | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    result_dict["expr_ir"] = expression_ir_report(cfg, stats, extra=extra)
    return result_dict


__all__ = ["attach_expression_ir_report", "expression_ir_report"]
