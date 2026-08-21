# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""NestyNet-specific factorized symbolic search adapter APIs."""

from .api import (
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
from .stageb_prep import (
    StageBExplorerPrep,
    _append_declared_constant_columns,
    _append_declared_constant_dims,
    _atom_dims_for_explorer,
    _build_input_exprs_with_declared_constants,
    _declared_constant_specs_for_explorer,
    prepare_stageb_explorer_inputs,
)
from .stageb_runner import (
    StageBEmbedContext,
    build_stageb_main_candidates,
    build_stageb_probe_jobs,
    has_structural_solved_result,
    pool_stageb_results,
    prepare_stageb_embed_context,
    row_raw_mse,
    run_stageb_explorer_jobs,
    run_stageb_wrapper_pass,
)
from .wrapper_utils import (
    _normalize_outer_wrapper_name,
    _outer_wrapper_forward,
    _outer_wrapper_inverse_ast,
    _outer_wrapper_transformed_y_dims,
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
    "StageBExplorerPrep",
    "_append_declared_constant_columns",
    "_append_declared_constant_dims",
    "_atom_dims_for_explorer",
    "_build_input_exprs_with_declared_constants",
    "_declared_constant_specs_for_explorer",
    "prepare_stageb_explorer_inputs",
    "StageBEmbedContext",
    "build_stageb_main_candidates",
    "build_stageb_probe_jobs",
    "has_structural_solved_result",
    "pool_stageb_results",
    "prepare_stageb_embed_context",
    "row_raw_mse",
    "run_stageb_explorer_jobs",
    "run_stageb_wrapper_pass",
    "_normalize_outer_wrapper_name",
    "_outer_wrapper_forward",
    "_outer_wrapper_inverse_ast",
    "_outer_wrapper_transformed_y_dims",
]
