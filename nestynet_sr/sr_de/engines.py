# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Shared proposal-engine result types for DE discovery.

The current concrete engines still live mostly in ``run_de.py`` and
``factorized_de.py``.  This module provides the small common result shape that
future engine wrappers can return into the DE ladder.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .proposals import merge_proposal_slates


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


@dataclass
class DEEngineOutput:
    engine: str
    proposal_slate: list[dict[str, Any]] = field(default_factory=list)
    selected_payload: dict[str, Any] | None = None
    selected_engine: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def merge_engine_proposal_slates(
    outputs: Sequence[DEEngineOutput | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge proposal slates emitted by proposal producers."""

    slates: list[list[dict[str, Any]]] = []
    namespaces: list[str | None] = []
    for idx, output in enumerate(list(outputs or [])):
        if isinstance(output, DEEngineOutput):
            engine = str(output.engine or f"engine{idx}")
            slate = output.proposal_slate
        elif isinstance(output, Mapping):
            engine = str(output.get("engine", f"engine{idx}") or f"engine{idx}")
            slate = output.get("proposal_slate", []) or []
        else:
            continue
        rows = [row for row in list(slate or []) if isinstance(row, dict)]
        if not rows:
            continue
        slates.append(rows)
        namespaces.append(engine)
    return merge_proposal_slates(slates, source_namespaces=namespaces)


__all__ = [
    "DEEngineOutput",
    "merge_engine_proposal_slates",
]
