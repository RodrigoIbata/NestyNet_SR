# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

from pathlib import Path

from nestynet_sr.sr_de.context import DiscoveryContext


def test_discovery_context_caches_expensive_feature_objects_once():
    ctx = DiscoveryContext.from_components(
        filepaths=[Path("traj0.csv")],
        dataset_ids=["traj0"],
        surrogate_val_losses=[1.0e-4],
        metadata={"engine": "test"},
    )
    calls = {"n": 0}

    def _builder():
        calls["n"] += 1
        return ["feature-group"]

    first = ctx.get_or_build("factorized_search", _builder)
    second = ctx.get_or_build("factorized_search", _builder)

    assert first is second
    assert calls["n"] == 1
    assert ctx.diagnostics["cache_builds"]["factorized_search"] == 1
    assert ctx.diagnostics["cache_hits"]["factorized_search"] == 1

    payload = ctx.to_dict()
    assert payload["filepaths"] == ["traj0.csv"]
    assert payload["dataset_ids"] == ["traj0"]
    assert payload["feature_group_keys"] == ["factorized_search"]
    assert payload["metadata"]["engine"] == "test"
