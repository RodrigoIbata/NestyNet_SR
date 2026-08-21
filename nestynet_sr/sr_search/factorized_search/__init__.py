# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""factorized symbolic search public surface with lazy exports."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "factorized_search_to_nestynet",
    "nestynet_to_factorized_search",
    "dims_to_fraction",
    "fraction_to_dims",
    "dims_to_units_spec",
    "run_explorer",
    "embed_mapping_in_ast",
    "remap_var_to_exprs",
    "promote_argument_const_scales",
    "promote_const_to_scale",
    "bounds_from_data",
    "discover_gs_carrier_seeds",
    "nonlinear_invariant_carrier_seeds",
    "FactorizedSearchConfig",
    "ArchiveRecord",
    "EngineRequest",
    "EngineResult",
    "SearchPolicy",
    "MacroActionDecision",
    "MacroController",
    "MacroControllerConfig",
    "MacroControllerState",
    "build_macro_controller_state",
    "FeatureBlock",
    "BasisState",
    "SlotSpec",
    "ClosureSpec",
    "ClosureDesign",
    "BoundClosure",
    "ProposalCandidate",
    "ProposalContext",
    "ProposalFamily",
]

_EXPORT_MAP = {
    "factorized_search_to_nestynet": (".bridge", "factorized_search_to_nestynet"),
    "nestynet_to_factorized_search": (".bridge", "nestynet_to_factorized_search"),
    "dims_to_fraction": (".bridge", "dims_to_fraction"),
    "fraction_to_dims": (".bridge", "fraction_to_dims"),
    "dims_to_units_spec": (".bridge", "dims_to_units_spec"),
    "run_explorer": (".bridge", "run_explorer"),
    "embed_mapping_in_ast": (".bridge", "embed_mapping_in_ast"),
    "remap_var_to_exprs": (".bridge", "remap_var_to_exprs"),
    "promote_argument_const_scales": (".bridge", "promote_argument_const_scales"),
    "promote_const_to_scale": (".bridge", "promote_const_to_scale"),
    "bounds_from_data": (".bridge", "bounds_from_data"),
    "discover_gs_carrier_seeds": (".gs_carrier_seed", "discover_gs_carrier_seeds"),
    "nonlinear_invariant_carrier_seeds": (
        ".gs_carrier_seed",
        "nonlinear_invariant_carrier_seeds",
    ),
    "FactorizedSearchConfig": (".config", "FactorizedSearchConfig"),
    "ArchiveRecord": (".engine", "ArchiveRecord"),
    "EngineRequest": (".engine", "EngineRequest"),
    "EngineResult": (".engine", "EngineResult"),
    "SearchPolicy": (".engine", "SearchPolicy"),
    "MacroActionDecision": (".controller", "MacroActionDecision"),
    "MacroController": (".controller", "MacroController"),
    "MacroControllerConfig": (".controller", "MacroControllerConfig"),
    "MacroControllerState": (".controller", "MacroControllerState"),
    "build_macro_controller_state": (".controller", "build_macro_controller_state"),
    "FeatureBlock": (".basis_state", "FeatureBlock"),
    "BasisState": (".basis_state", "BasisState"),
    "SlotSpec": (".closures", "SlotSpec"),
    "ClosureSpec": (".closures", "ClosureSpec"),
    "ClosureDesign": (".closures", "ClosureDesign"),
    "BoundClosure": (".closures", "BoundClosure"),
    "ProposalCandidate": (".basis_state", "ProposalCandidate"),
    "ProposalContext": (".basis_state", "ProposalContext"),
    "ProposalFamily": (".basis_state", "ProposalFamily"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORT_MAP[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
