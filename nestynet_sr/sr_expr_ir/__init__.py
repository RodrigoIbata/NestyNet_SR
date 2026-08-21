# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Opt-in expression IR utilities shared by SR, FSS, DE, and GS paths."""

from .config import (
    ExpressionIRConfig,
    add_expr_ir_cli_args,
    apply_expr_ir_args_to_obj,
    coerce_expr_ir_config,
    expr_ir_active,
    expression_ir_config_to_dict,
)
from .stats import ExpressionIRStats

__all__ = [
    "ExpressionIRConfig",
    "ExpressionIRStats",
    "add_expr_ir_cli_args",
    "apply_expr_ir_args_to_obj",
    "coerce_expr_ir_config",
    "expr_ir_active",
    "expression_ir_config_to_dict",
]
