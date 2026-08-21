# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Compatibility shim for the legacy controller module."""

from __future__ import annotations

from importlib import import_module as _import_module

_legacy_mod = _import_module(".legacy.controller", __package__)
for _name in dir(_legacy_mod):
    if _name.startswith("__") and _name not in {"__all__"}:
        continue
    globals()[_name] = getattr(_legacy_mod, _name)

__all__ = getattr(
    _legacy_mod,
    "__all__",
    tuple(name for name in globals() if not name.startswith("__")),
)
