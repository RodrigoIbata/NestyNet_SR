# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""NestyNet_SR: Symbolic Regression and Differential Equation Discovery

This package provides tools for discovering analytical expressions and differential
equations from data using neural network surrogates built on NestyNet.

Main Components:
- sr_core: AST compiler, bridges, and separability mathematics
- sr_search: Separability search and Stage B rewrites
- stat_selection: Frozen-archive confidence Pareto certification
- sr_de: Differential equation discovery engine
- adaptors: Specialized adaptors for optimization

Command-line Tools:
- nestynet-sr: Symbolic regression (discovers y = f(x) from data)
- nestynet-de: DE discovery (discovers differential equations from data)
"""

__version__ = "0.1.0"

# Expose the always-available core SR packages eagerly. Keep discovery lazy so
# importing ``nestynet_sr`` does not immediately pull in optional downstream
# search helpers.
from . import sr_core, sr_search, stat_selection

__all__ = ["sr_core", "sr_search", "stat_selection", "discovery"]


def __getattr__(name: str):
    if name == "discovery":
        from importlib import import_module

        module = import_module(".discovery", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ``sr_de`` depends on the adaptor stack (and therefore on ``nestynet``).
# Keep it optional so users can import factorized symbolic search / core AST tools without the
# full NestyNet runtime.
try:  # pragma: no cover - depends on external optional dependency
    from . import sr_de
except ModuleNotFoundError as e:
    # Only suppress the optional dependency failure. Other import errors should
    # surface normally.
    if getattr(e, "name", None) != "nestynet":
        raise
    sr_de = None  # type: ignore[assignment]
else:
    __all__.append("sr_de")

try:  # pragma: no cover - depends on external optional dependency
    from . import adaptors
except ModuleNotFoundError as e:
    if getattr(e, "name", None) != "nestynet":
        raise
    adaptors = None  # type: ignore[assignment]
else:
    __all__.append("adaptors")
