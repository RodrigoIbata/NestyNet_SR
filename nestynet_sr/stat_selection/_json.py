# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Deterministic, deeply immutable JSON-compatible records."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from types import MappingProxyType
from typing import Any, Optional


_JSON_PRIMITIVES = (str, int, bool, type(None))


def freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): freeze_json(item)
                for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON records may not contain NaN or infinity")
        return value
    if isinstance(value, _JSON_PRIMITIVES):
        return value
    try:
        scalar = value.item()
    except Exception:
        scalar = None
    else:
        if scalar is not value:
            return freeze_json(scalar)
    try:
        sequence = value.tolist()
    except Exception:
        sequence = None
    else:
        if sequence is not value:
            return freeze_json(sequence)
    raise TypeError(
        "statistical-audit records must be JSON-compatible; "
        f"received {type(value).__name__}"
    )


def freeze_mapping(value: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TypeError("expected a mapping")
    return freeze_json(dict(value))


def thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        thaw_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
