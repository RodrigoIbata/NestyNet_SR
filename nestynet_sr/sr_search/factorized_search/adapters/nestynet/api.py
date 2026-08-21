# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Transitional NestyNet adapter API backed by the existing bridge module."""

from ...bridge import (
    factorized_search_to_nestynet,
    bounds_from_data,
    dims_to_fraction,
    dims_to_units_spec,
    embed_mapping_in_ast,
    fraction_to_dims,
    nestynet_to_factorized_search,
    promote_argument_const_scales,
    promote_const_to_scale,
    remap_var_to_exprs,
    run_explorer,
)

__all__ = [
    "factorized_search_to_nestynet",
    "bounds_from_data",
    "dims_to_fraction",
    "dims_to_units_spec",
    "embed_mapping_in_ast",
    "fraction_to_dims",
    "nestynet_to_factorized_search",
    "promote_argument_const_scales",
    "promote_const_to_scale",
    "remap_var_to_exprs",
    "run_explorer",
]
