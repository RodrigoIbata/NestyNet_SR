# SPDX-License-Identifier: MPL-2.0

import random

import torch

import nestynet_sr.sr_search.factorized_search.explorer as explorer_mod


def test_run_explorer_tracks_legacy_crossover_policy(monkeypatch):
    ctr = {"score": 0}

    def _fake_score_expr(*args, **kwargs):
        node = args[0]
        i = ctr["score"]
        ctr["score"] += 1
        z = torch.tensor([float(i), float(i % 2), 0.0, 0.0], dtype=torch.float64)
        mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
        mse = 1.0 + 0.01 * float(i)
        key = ("k", i)
        if kwargs.get("return_expr", False):
            return mse, key, z, mapping, node
        return mse, key, z, mapping

    def _fake_select_action(self, s_key, rng, allowed_actions=None):
        return explorer_mod.A_CROSSOVER

    def _fake_crossover(
        recipient, arch, parent_key, rng, max_depth, nvars, var_dims=None, **kwargs
    ):
        return ("add", recipient, ("var", 0))

    monkeypatch.setattr(explorer_mod, "score_expr", _fake_score_expr)
    monkeypatch.setattr(explorer_mod.Explorer, "select_action", _fake_select_action)
    monkeypatch.setattr(explorer_mod, "apply_crossover_action", _fake_crossover)

    def _target_fn(x):
        return x[:, :1]

    arch = explorer_mod.run_explorer_core(
        _target_fn,
        1,
        n_iter=8,
        max_depth=2,
        poly_degree=2,
        lo=0.2,
        hi=0.8,
        seed=0,
        dtype=torch.float64,
        brute_depth=0,
        no_residual=True,
        p_restart=0.0,
        print_every=0,
        verbose=False,
    )

    cps = getattr(arch, "crossover_policy_stats", None)
    assert isinstance(cps, dict)
    assert int(cps.get("legacy", {}).get("selected", 0)) > 0
    assert int(cps.get("legacy", {}).get("proposed", 0)) > 0

    ad = getattr(arch, "action_distribution", None)
    assert isinstance(ad, dict)
    counts = ad.get("counts", {})
    assert isinstance(counts, dict)
    assert int(ad.get("total_selected", 0)) > 0
    assert int(counts.get("crossover", 0)) > 0


def test_residual_selection_uses_complexity_penalty(monkeypatch):
    x_fit = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    y_fit = x_fit.clone()
    x_probe = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float64)
    y_probe = x_probe.clone()

    def _fake_fit_best(pred, y, poly_degree):
        v = pred.squeeze(-1)
        if torch.allclose(v, 2.0 * x_fit.squeeze(-1)):
            return 0.0, {"scale": 0.5}
        return 0.0, {"scale": 0.9998}

    def _fake_eval_mapping(pred, mapping):
        return pred * float(mapping.get("scale", 1.0))

    monkeypatch.setattr(explorer_mod, "fit_best", _fake_fit_best)
    monkeypatch.setattr(explorer_mod, "eval_mapping", _fake_eval_mapping)

    parent = ("var", 0)
    parent_mapping = {"scale": 1.0}
    pool_nodes = [("var", 0)]
    pool_phi = torch.ones((x_probe.shape[0], 1), dtype=torch.float64)
    pool_norms = torch.ones((1,), dtype=torch.float64)
    pool_dims = [None]
    rng = random.Random(0)

    expr_no_penalty = explorer_mod.apply_residual_action(
        parent,
        parent_mapping,
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        pool_nodes,
        pool_phi,
        pool_norms,
        pool_dims,
        rng,
        max_depth=3,
        nvars=1,
        poly_degree=2,
        var_dims=None,
        topk=1,
        complexity_penalty=0.0,
    )
    expr_with_penalty = explorer_mod.apply_residual_action(
        parent,
        parent_mapping,
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        pool_nodes,
        pool_phi,
        pool_norms,
        pool_dims,
        random.Random(0),
        max_depth=3,
        nvars=1,
        poly_degree=2,
        var_dims=None,
        topk=1,
        complexity_penalty=1.0e-3,
    )

    assert expr_no_penalty == ("add", ("var", 0), ("var", 0))
    assert expr_with_penalty == ("var", 0)
