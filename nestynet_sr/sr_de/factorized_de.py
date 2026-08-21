# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.
# ruff: noqa: F822

"""Compatibility facade for DE-facing factorized symbolic search."""

from __future__ import annotations

import inspect
from functools import wraps
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

import nestynet_sr.sr_search.factorized_search.oracle_lab_de as oracle_de
from nestynet_sr.sr_de.de_search import DESearchResult, DESearchResultMulti
from nestynet_sr.sr_search.factorized_search.bridge import run_explorer
from nestynet_sr.sr_search.factorized_search.config import FactorizedSearchConfig

from . import _factorized_de_explorer as _explorer
from . import _factorized_de_frontend as _frontend
from . import _factorized_de_lanes as _lanes
from . import _factorized_de_operator as _operator
from . import _factorized_de_rescue as _rescue
from . import _factorized_de_search as _search

_GROUP_MODULES = (_frontend, _search, _operator, _lanes, _explorer, _rescue)
_IMPLEMENTATIONS = {
    name: getattr(module, name)
    for module in _GROUP_MODULES
    for name in module.__factorized_de_definitions__
}

# Deferred annotations copied by ``functools.wraps`` resolve in this facade.
_ANNOTATION_GLOBALS = (
    Any,
    DESearchResult,
    DESearchResultMulti,
    FactorizedSearchConfig,
    Mapping,
    Sequence,
    np,
    oracle_de,
    run_explorer,
    torch,
)

_PATCHABLE_GLOBAL_NAMES = (
    "_score_candidate_domain_fragility",
    "factorized_search_candidate_to_feature_predictor",
    "run_explorer",
    "run_factorized_de_from_feature_groups",
    "validate_order2_generator_witness",
)


def _sync_patchable_globals() -> None:
    facade_globals = globals()
    for module in _GROUP_MODULES:
        for name in _PATCHABLE_GLOBAL_NAMES:
            if hasattr(module, name):
                setattr(module, name, facade_globals[name])


def _forward(module: ModuleType, name: str) -> Callable[..., Any]:
    implementation = _IMPLEMENTATIONS[name]

    @wraps(implementation)
    def forwarded(*args, **kwargs):
        _sync_patchable_globals()
        return implementation(*args, **kwargs)

    forwarded.__signature__ = inspect.signature(implementation)
    forwarded.__defaults__ = implementation.__defaults__
    forwarded.__kwdefaults__ = implementation.__kwdefaults__
    forwarded.__module__ = __name__
    forwarded.__qualname__ = name
    return forwarded


for _module in _GROUP_MODULES:
    for _name in _module.__factorized_de_constants__:
        globals()[_name] = getattr(_module, _name)

for _module in _GROUP_MODULES:
    for _name in _module.__factorized_de_definitions__:
        _definition = getattr(_module, _name)
        if inspect.isclass(_definition):
            _definition.__module__ = __name__
            for _member in _definition.__dict__.values():
                if inspect.isfunction(_member):
                    _member.__module__ = __name__
            globals()[_name] = _definition
        else:
            globals()[_name] = _forward(_module, _name)

for _module in _GROUP_MODULES:
    for _name in _module.__factorized_de_late_bindings__:
        setattr(_module, _name, globals()[_name])

_sync_patchable_globals()

__all__ = [
    "DEFeatureGroup",
    "FactorizedSearchDERescueConfig",
    "FactorizedSearchDEResult",
    "de_lab_spec_from_de_cfg",
    "default_physics_rescue_hp",
    "run_factorized_de_from_feature_groups",
    "run_direct_residual_fss_from_feature_groups",
    "run_regularized_implicit_residual_fss_from_feature_groups",
    "factorized_search_report_to_de_result",
    "normalized_rmse",
    "evaluate_factorized_search_candidate",
    "factorized_search_candidate_to_feature_predictor",
    "factorized_search_report_shortlist",
    "factorized_search_report_to_rhs_callable",
    "validate_order2_generator_witness",
    "build_factorized_search_de_feature_groups_from_surrogate",
    "build_factorized_search_de_feature_groups_from_surrogates",
    "run_factorized_search_de_from_feature_groups",
    "run_factorized_search_de_from_surrogate",
    "run_factorized_search_de_from_surrogates",
    "FactorizedDEBlock",
    "FactorizedDERescueConfig",
    "FactorizedDEResult",
    "run_factorized_coeff_rescue_from_feature_groups",
]

del _definition, _member, _module, _name
