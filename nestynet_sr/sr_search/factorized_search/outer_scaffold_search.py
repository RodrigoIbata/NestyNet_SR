# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Backward-compatibility redirect -- all logic now lives in closure_search_compat.py."""

from .closure_search_compat import *  # noqa: F401,F403
from .closure_search_compat import (  # noqa: F401 -- explicit re-exports for attribute access
    LEGACY_SCAFFOLD_ADAPTER_ENABLED,
    OperatorApplication,
    OuterScaffoldSpec,
    ScaffoldPreviewCandidate,
    _build_scaffold_beam_state,
    _collect_direct_hole_candidates,
    _debug_scaffold_beam_state_failure,
    _direct_exp_scaffold_kind,
    _direct_log_scaffold_kind,
    _direct_periodic_scaffold_kind,
    _direct_power_scaffold_kind,
    _direct_quadratic_scaffold_kind,
    _direct_rational_scaffold_kind,
    _fit_scaffold_mapping,
    _solve_direct_affine_preview_rows,
    _solve_direct_exp_preview_rows,
    _solve_direct_log_preview_rows,
    _solve_direct_operator_preview_rows,
    _solve_direct_periodic_add_preview_rows,
    _solve_direct_power_preview_rows,
    _solve_direct_quadratic_preview_rows,
    _solve_direct_rational_affine_preview_rows,
    enumerate_closure_search_specs,
    enumerate_operator_applications,
    enumerate_outer_scaffold_specs,
    node_size,
    node_str,
    run_closure_search_pass,
    run_closure_search_pass_impl,
    run_outer_scaffold_pass,
    solve_inverse_spec_preview_rows,
)
