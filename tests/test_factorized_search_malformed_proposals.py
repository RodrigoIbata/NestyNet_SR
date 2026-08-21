# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch

import nestynet_sr.sr_search.factorized_search.explorer as explorer_mod


def test_run_explorer_core_drops_malformed_action_proposals(monkeypatch):
    original_apply_action = explorer_mod.apply_action
    calls = {"n": 0}

    def _apply_action_malformed_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return ("add", None, ("var", 0))
        return original_apply_action(*args, **kwargs)

    monkeypatch.setattr(explorer_mod, "apply_action", _apply_action_malformed_once)

    arch = explorer_mod.run_explorer_core(
        target_fn=lambda x: x[:, 0:1] * x[:, 1:2],
        nvars=2,
        n_iter=4,
        max_depth=3,
        n_fit=16,
        n_probe=24,
        brute_depth=1,
        no_residual=True,
        print_every=0,
        dtype=torch.float64,
        seed=0,
    )

    assert calls["n"] >= 1
    best = arch.best(1)
    assert len(best) == 1
