# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Compatibility wrapper for closure search (formerly outer scaffold search).

The implementation now lives under ``proposal_families/`` so family experts can
be imported and steered independently.  This module keeps the historical import
surface used by tests and the engine.

The inverse fallback path has been removed -- the closure machine now uses only
the direct operator route.  Legacy symbols (``OuterScaffoldSpec``,
``solve_inverse_spec_preview_rows``, etc.) are still re-exported for callers
that reference them, but they are no longer wired into the live proposal loop.
"""

from __future__ import annotations

from .expr_ast import node_size, node_str
from .inverse_spec_solver import solve_inverse_spec_preview_rows  # re-export only
from .proposal_families.compat import enumerate_closure_search_specs
from .proposal_families import (
    OperatorApplication,
    OuterScaffoldSpec,
    ScaffoldPreviewCandidate,
    build_scaffold_beam_state as _build_scaffold_beam_state,
    debug_scaffold_beam_state_failure as _debug_scaffold_beam_state_failure,
    direct_exp_scaffold_kind as _direct_exp_scaffold_kind,
    direct_log_scaffold_kind as _direct_log_scaffold_kind,
    direct_power_scaffold_kind as _direct_power_scaffold_kind,
    direct_periodic_scaffold_kind as _direct_periodic_scaffold_kind,
    direct_quadratic_scaffold_kind as _direct_quadratic_scaffold_kind,
    direct_rational_scaffold_kind as _direct_rational_scaffold_kind,
    enumerate_operator_applications,
    operator_application_from_scaffold,
    render_operator_as_scaffold,
    fit_scaffold_mapping as _fit_scaffold_mapping,
    run_closure_search_pass_impl,
    solve_direct_affine_preview_rows as _solve_direct_affine_preview_rows_impl,
    solve_direct_operator_preview_rows as _solve_direct_operator_preview_rows_impl,
    solve_direct_exp_preview_rows as _solve_direct_exp_preview_rows_impl,
    solve_direct_log_preview_rows as _solve_direct_log_preview_rows_impl,
    solve_direct_power_preview_rows as _solve_direct_power_preview_rows_impl,
    solve_direct_periodic_add_preview_rows as _solve_direct_periodic_add_preview_rows_impl,
    solve_direct_quadratic_preview_rows as _solve_direct_quadratic_preview_rows_impl,
    solve_direct_rational_affine_preview_rows as _solve_direct_rational_affine_preview_rows_impl,
)
from .proposal_families.direct import collect_direct_hole_candidates as _collect_direct_hole_candidates

# Legacy gate -- kept for backward compatibility but always False.
LEGACY_SCAFFOLD_ADAPTER_ENABLED = False


# ---------------------------------------------------------------------------
# Per-family solver wrappers (inject collect_direct_hole_candidates_fn default)
# ---------------------------------------------------------------------------

def _solve_direct_periodic_add_preview_rows(*args, **kwargs):
    kwargs.setdefault("collect_direct_hole_candidates_fn", _collect_direct_hole_candidates)
    return _solve_direct_periodic_add_preview_rows_impl(*args, **kwargs)


def _solve_direct_exp_preview_rows(*args, **kwargs):
    kwargs.setdefault("collect_direct_hole_candidates_fn", _collect_direct_hole_candidates)
    return _solve_direct_exp_preview_rows_impl(*args, **kwargs)


def _solve_direct_log_preview_rows(*args, **kwargs):
    kwargs.setdefault("collect_direct_hole_candidates_fn", _collect_direct_hole_candidates)
    return _solve_direct_log_preview_rows_impl(*args, **kwargs)


def _solve_direct_power_preview_rows(*args, **kwargs):
    kwargs.setdefault("collect_direct_hole_candidates_fn", _collect_direct_hole_candidates)
    return _solve_direct_power_preview_rows_impl(*args, **kwargs)


def _solve_direct_rational_affine_preview_rows(*args, **kwargs):
    kwargs.setdefault("collect_direct_hole_candidates_fn", _collect_direct_hole_candidates)
    return _solve_direct_rational_affine_preview_rows_impl(*args, **kwargs)


def _solve_direct_quadratic_preview_rows(*args, **kwargs):
    return _solve_direct_quadratic_preview_rows_impl(*args, **kwargs)


def _solve_direct_affine_preview_rows(*args, **kwargs):
    return _solve_direct_affine_preview_rows_impl(*args, **kwargs)


def _solve_direct_operator_preview_rows(*args, **kwargs):
    """Unified direct operator dispatcher -- no legacy compat dispatch."""
    kwargs.setdefault("collect_direct_hole_candidates_fn", _collect_direct_hole_candidates)
    return _solve_direct_operator_preview_rows_impl(*args, **kwargs)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_closure_search_pass(**kwargs):
    """Thin wrapper that wires the direct-only operator route into the impl."""
    kwargs = dict(kwargs)
    # Pop legacy parameters that callers may still pass but are now ignored.
    kwargs.pop("prefer_legacy_scaffold_enumeration", None)
    kwargs.pop("enumerate_outer_scaffold_specs_fn", None)
    kwargs.pop("enumerate_closure_search_specs_fn", None)
    kwargs.pop("legacy_scaffold_to_operator_fn", None)
    return run_closure_search_pass_impl(
        **kwargs,
        enumerate_operator_applications_fn=enumerate_operator_applications,
        solve_direct_operator_preview_rows_fn=_solve_direct_operator_preview_rows,
    )


# Backward-compat aliases
run_outer_scaffold_pass = run_closure_search_pass
enumerate_outer_scaffold_specs = enumerate_closure_search_specs


__all__ = [
    "LEGACY_SCAFFOLD_ADAPTER_ENABLED",
    "OperatorApplication",
    "OuterScaffoldSpec",
    "ScaffoldPreviewCandidate",
    "operator_application_from_scaffold",
    "render_operator_as_scaffold",
    "_build_scaffold_beam_state",
    "_debug_scaffold_beam_state_failure",
    "_direct_exp_scaffold_kind",
    "_direct_log_scaffold_kind",
    "_direct_power_scaffold_kind",
    "_direct_periodic_scaffold_kind",
    "_direct_quadratic_scaffold_kind",
    "_direct_rational_scaffold_kind",
    "_fit_scaffold_mapping",
    "_collect_direct_hole_candidates",
    "_solve_direct_exp_preview_rows",
    "_solve_direct_affine_preview_rows",
    "_solve_direct_log_preview_rows",
    "_solve_direct_operator_preview_rows",
    "_solve_direct_power_preview_rows",
    "_solve_direct_periodic_add_preview_rows",
    "_solve_direct_quadratic_preview_rows",
    "_solve_direct_rational_affine_preview_rows",
    "enumerate_operator_applications",
    "enumerate_closure_search_specs",
    "enumerate_outer_scaffold_specs",
    "node_size",
    "node_str",
    "run_closure_search_pass",
    "run_closure_search_pass_impl",
    "run_outer_scaffold_pass",
    "solve_inverse_spec_preview_rows",
]
