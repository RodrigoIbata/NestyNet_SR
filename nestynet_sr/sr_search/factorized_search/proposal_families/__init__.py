# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from .direct import (
    DIRECT_OPERATOR_PLANNERS,
    DirectOperatorPlanner,
    direct_affine_scaffold_kind,
    direct_exp_scaffold_kind,
    direct_log_scaffold_kind,
    direct_power_scaffold_kind,
    direct_periodic_scaffold_kind,
    direct_quadratic_scaffold_kind,
    direct_rational_scaffold_kind,
    resolve_direct_operator_planner,
    solve_direct_affine_preview_rows,
    solve_direct_operator_preview_rows,
    solve_direct_exp_preview_rows,
    solve_direct_log_preview_rows,
    solve_direct_periodic_add_preview_rows,
    solve_direct_power_preview_rows,
    solve_direct_quadratic_preview_rows,
    solve_direct_rational_affine_preview_rows,
)
from .compat import (
    OuterScaffoldSpec,
    enumerate_closure_search_specs,
    operator_application_from_scaffold,
    render_operator_as_scaffold,
)
from .inverse_route import build_scaffold_beam_state, debug_scaffold_beam_state_failure, fit_scaffold_mapping
from .operator_specs import OperatorSpec, family_operator_preset_ids, family_operator_specs, operator_algebra_specs
from .runner import run_closure_search_pass_impl
from .scaffold_enum import enumerate_operator_applications, normalize_families
from .seed_blocks import (
    SeedBlock,
    extend_seed_blocks_with_basis,
    seed_anchor_blocks,
    seed_blocks_from_basis_state,
)
from .steering import (
    FamilyBudgetPlanEntry,
    allocate_family_budgets,
    heuristic_family_priority_scores,
)
from .slot_binding import bind_slot_candidates, family_anchor_blocks, pick_placeholder_block
from .types import OperatorApplication, ScaffoldPreviewCandidate

__all__ = [
    "FamilyBudgetPlanEntry",
    "DIRECT_OPERATOR_PLANNERS",
    "DirectOperatorPlanner",
    "OperatorSpec",
    "OperatorApplication",
    "OuterScaffoldSpec",
    "ScaffoldPreviewCandidate",
    "SeedBlock",
    "allocate_family_budgets",
    "bind_slot_candidates",
    "build_scaffold_beam_state",
    "debug_scaffold_beam_state_failure",
    "extend_seed_blocks_with_basis",
    "direct_affine_scaffold_kind",
    "direct_exp_scaffold_kind",
    "direct_log_scaffold_kind",
    "direct_power_scaffold_kind",
    "direct_periodic_scaffold_kind",
    "direct_quadratic_scaffold_kind",
    "direct_rational_scaffold_kind",
    "enumerate_operator_applications",
    "enumerate_closure_search_specs",
    "family_anchor_blocks",
    "family_operator_preset_ids",
    "family_operator_specs",
    "fit_scaffold_mapping",
    "heuristic_family_priority_scores",
    "normalize_families",
    "operator_algebra_specs",
    "pick_placeholder_block",
    "operator_application_from_scaffold",
    "resolve_direct_operator_planner",
    "render_operator_as_scaffold",
    "run_closure_search_pass_impl",
    "seed_anchor_blocks",
    "seed_blocks_from_basis_state",
    "solve_direct_affine_preview_rows",
    "solve_direct_operator_preview_rows",
    "solve_direct_exp_preview_rows",
    "solve_direct_log_preview_rows",
    "solve_direct_periodic_add_preview_rows",
    "solve_direct_power_preview_rows",
    "solve_direct_quadratic_preview_rows",
    "solve_direct_rational_affine_preview_rows",
]
