# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from types import SimpleNamespace

import torch

import nestynet_sr.sr_search.factorized_search.explorer as explorer_mod


def test_archive_repair_pass_repairs_seeded_elite_without_online_inverse(monkeypatch):
    init_expr = ("add", ("var", 0), ("const", 1.0))
    repaired_expr = ("sin", ("var", 0))
    captured = {}

    def _fake_run_brute_phase(arch, *args, **kwargs):
        x_probe = args[3]
        z = torch.zeros((x_probe.shape[0], 1), dtype=x_probe.dtype)
        mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
        arch.update(("expr", "seed"), 1.0, init_expr, z, mapping, raw_mse=1.0)
        return False

    def _fake_score_expr(node, x_fit, y_fit, x_probe, y_probe, *args, **kwargs):
        score_map = {
            init_expr: 1.0,
            repaired_expr: 0.1,
        }
        mse = float(score_map.get(node, 2.0))
        key = ("expr", str(node))
        z = torch.zeros((x_probe.shape[0], 1), dtype=x_probe.dtype)
        mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
        if kwargs.get("return_expr", False):
            return mse, key, z, mapping, node
        return mse, key, z, mapping

    def _fake_estimate_inverse_steering_potential(*args, **kwargs):
        return SimpleNamespace(
            allowed=True,
            reason="ok",
            best_path=(1,),
            candidate_paths=((1,), (2,)),
        )

    def _fake_apply_inverse_steering_action(parent_node, parent_mapping, *args, **kwargs):
        captured["parent_node"] = parent_node
        captured["candidate_paths"] = [tuple(path) for path in list(kwargs.get("candidate_paths", []) or [])]
        return repaired_expr, {
            "status": "ok",
            "selected_path": [1],
        }

    monkeypatch.setattr(explorer_mod, "_run_brute_phase", _fake_run_brute_phase)
    monkeypatch.setattr(explorer_mod, "estimate_inverse_steering_potential", _fake_estimate_inverse_steering_potential)
    monkeypatch.setattr(explorer_mod, "apply_inverse_steering_action", _fake_apply_inverse_steering_action)

    arch = explorer_mod.run_explorer_core(
        target_fn=lambda x: x[:, :1],
        nvars=1,
        n_iter=0,
        max_depth=3,
        poly_degree=2,
        lo=0.2,
        hi=0.8,
        seed=0,
        dtype=torch.float64,
        brute_depth=1,
        no_residual=True,
        no_crossover=True,
        p_restart=0.0,
        print_every=0,
        verbose=False,
        inverse_steering_enable=False,
        repair_pass_enable=True,
        repair_pass_elite_k=1,
        repair_pass_paths_per_elite=2,
        repair_pass_rounds=1,
        _score_expr_fn=_fake_score_expr,
    )

    stats = getattr(arch, "repair_pass_stats", {})
    best = arch.best(1)[0]

    assert captured["parent_node"] == init_expr
    assert captured["candidate_paths"] == [(1,)]
    assert best.best_expr == repaired_expr
    assert float(best.best_mse) == 0.1
    assert bool(stats.get("enabled", False)) is True
    assert int(stats.get("elites_selected", 0)) == 1
    assert int(stats.get("solver_calls", 0)) == 1
    assert int(stats.get("solver_ok", 0)) == 1
    assert int(stats.get("accepted_repairs", 0)) == 1
    assert int(stats.get("evals_used", 0)) == 1
    assert int(stats.get("elites_improved", 0)) == 1
