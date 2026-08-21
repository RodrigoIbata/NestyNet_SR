# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Explicit engine-facing factorized symbolic search contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class ArchiveRecord:
    """Stable archive/result record shape for future engine consumers."""

    expr: Any = None
    mapping: Mapping[str, Any] = field(default_factory=dict)
    mse_raw: float = float("inf")
    mse_eff: float = float("inf")
    size: int = 0
    depth: int = 0
    lineage: Mapping[str, Any] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EngineRequest:
    """Stable engine input contract used by adapters and harnesses."""

    target_fn: Any = None
    nvars: int | None = None
    x_fit_data: Any = None
    y_fit_data: Any = None
    x_probe_data: Any = None
    y_probe_data: Any = None
    var_dims: Sequence[Any] | None = None
    y_dims: Any = None
    declared_constants: Sequence[Mapping[str, Any]] = ()
    engine_config: Any = None
    policy: SearchPolicy | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EngineResult:
    """Stable engine output contract used by adapters and harnesses."""

    records: tuple[ArchiveRecord, ...] = ()
    best: ArchiveRecord | None = None
    archive: Any = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class SearchPolicy(Protocol):
    """Pluggable policy surface layered on top of the core engine."""

    def choose_parent(self, *, archive: Any, state: Mapping[str, Any]) -> Any:
        """Choose the next parent candidate or return ``None``."""

    def choose_action_slate(
        self,
        *,
        parent: Any,
        archive: Any,
        state: Mapping[str, Any],
    ) -> Sequence[Any]:
        """Choose the build/repair action slate for a parent candidate."""

    def rank_inverse_paths(
        self,
        *,
        parent: Any,
        path_rows: Sequence[Mapping[str, Any]],
        state: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]:
        """Order inverse/repair paths without mutating engine state."""

    def rank_repair_routes(
        self,
        *,
        parent: Any,
        route_rows: Sequence[Mapping[str, Any]],
        state: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]:
        """Order repair routes without mutating engine state."""

    def update(self, *, observation: Mapping[str, Any]) -> None:
        """Consume search observations after an engine step."""


__all__ = [
    "ArchiveRecord",
    "EngineRequest",
    "EngineResult",
    "SearchPolicy",
]
