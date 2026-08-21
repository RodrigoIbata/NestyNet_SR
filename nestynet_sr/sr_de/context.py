# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Shared context object for DE discovery ladders.

This module is intentionally small.  It gives the DE stack one place to carry
datasets, surrogate-derived feature groups, diagnostics, and policy metadata
without forcing the current CLI pipeline to be rewritten in one patch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach") and callable(getattr(value, "detach")):
        detached = value.detach().cpu()
        if getattr(detached, "ndim", 0) == 0:
            return float(detached.item())
        return detached.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


@dataclass
class DiscoveryContext:
    """Reusable state for one DE discovery run.

    The context is deliberately permissive about object types because current
    callers pass trained surrogate modules, data loaders, config dataclasses,
    and benchmark metadata.  Expensive derived objects, such as feature groups,
    are cached by stable string keys.
    """

    filepaths: tuple[str, ...] = ()
    dataset_ids: tuple[str, ...] = ()
    cfg: Any = None
    rescue_cfg: Any = None
    surrogates: tuple[Any, ...] = ()
    train_loaders: tuple[Any, ...] = ()
    validation_loaders: tuple[Any, ...] = ()
    device: Any = None
    dtype: Any = None
    surrogate_val_losses: tuple[float, ...] | None = None
    feature_groups: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_components(
        cls,
        *,
        filepaths: Sequence[Any] | None = None,
        dataset_ids: Sequence[Any] | None = None,
        cfg: Any = None,
        rescue_cfg: Any = None,
        surrogates: Sequence[Any] | None = None,
        train_loaders: Sequence[Any] | None = None,
        validation_loaders: Sequence[Any] | None = None,
        device: Any = None,
        dtype: Any = None,
        surrogate_val_losses: Sequence[float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "DiscoveryContext":
        return cls(
            filepaths=tuple(str(path) for path in list(filepaths or [])),
            dataset_ids=tuple(str(dataset_id) for dataset_id in list(dataset_ids or [])),
            cfg=cfg,
            rescue_cfg=rescue_cfg,
            surrogates=tuple(surrogates or ()),
            train_loaders=tuple(train_loaders or ()),
            validation_loaders=tuple(validation_loaders or ()),
            device=device,
            dtype=dtype,
            surrogate_val_losses=None if surrogate_val_losses is None else tuple(float(v) for v in surrogate_val_losses),
            metadata=dict(metadata or {}),
        )

    def get_cached(self, key: str) -> Any:
        return self.feature_groups.get(str(key), None)

    def set_cached(self, key: str, value: Any) -> Any:
        self.feature_groups[str(key)] = value
        return value

    def get_or_build(self, key: str, builder: Callable[[], Any]) -> Any:
        key_s = str(key)
        if key_s not in self.feature_groups:
            self.feature_groups[key_s] = builder()
            self.diagnostics.setdefault("cache_builds", {})[key_s] = (
                int(self.diagnostics.setdefault("cache_builds", {}).get(key_s, 0)) + 1
            )
        else:
            self.diagnostics.setdefault("cache_hits", {})[key_s] = (
                int(self.diagnostics.setdefault("cache_hits", {}).get(key_s, 0)) + 1
            )
        return self.feature_groups[key_s]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "filepaths": list(self.filepaths),
                "dataset_ids": list(self.dataset_ids),
                "num_surrogates": int(len(self.surrogates)),
                "num_train_loaders": int(len(self.train_loaders)),
                "num_validation_loaders": int(len(self.validation_loaders)),
                "feature_group_keys": sorted(str(k) for k in self.feature_groups.keys()),
                "diagnostics": self.diagnostics,
                "metadata": self.metadata,
            }
        )


def build_factorized_search_feature_groups(
    context: DiscoveryContext,
    *,
    cache_key: str = "factorized_search",
) -> Any:
    """Build or return cached DE-facing FSS feature groups for a context."""

    def _build() -> Any:
        from dataclasses import replace

        from .factorized_de import (
            build_factorized_search_de_feature_groups_from_surrogate,
            build_factorized_search_de_feature_groups_from_surrogates,
        )

        if len(context.surrogates) != len(context.train_loaders) or len(context.surrogates) != len(context.validation_loaders):
            raise ValueError("surrogates, train_loaders, and validation_loaders must have matching lengths")
        if len(context.surrogates) <= 0:
            raise ValueError("no surrogates available for factorized-search feature groups")

        if len(context.surrogates) == 1:
            groups = build_factorized_search_de_feature_groups_from_surrogate(
                context.surrogates[0],
                context.train_loaders[0],
                context.validation_loaders[0],
                cfg=context.cfg,
                rescue_cfg=context.rescue_cfg,
                device=context.device,
                dtype=context.dtype,
                group_id=str(context.dataset_ids[0]) if context.dataset_ids else "dataset0",
            )
        else:
            groups = build_factorized_search_de_feature_groups_from_surrogates(
                list(context.surrogates),
                list(context.train_loaders),
                list(context.validation_loaders),
                cfg=context.cfg,
                rescue_cfg=context.rescue_cfg,
                device=context.device,
                dataset_ids=list(context.dataset_ids),
                dtype=context.dtype,
            )

        losses = list(context.surrogate_val_losses or [])
        if losses:
            if len(losses) != len(groups):
                raise ValueError("surrogate_val_losses must match the number of DE feature groups")
            groups = [
                replace(group, surrogate_val_loss=float(losses[i]))
                for i, group in enumerate(groups)
            ]
        return groups

    return context.get_or_build(cache_key, _build)


__all__ = [
    "DiscoveryContext",
    "build_factorized_search_feature_groups",
]
