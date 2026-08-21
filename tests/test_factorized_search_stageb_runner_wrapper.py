# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from types import SimpleNamespace

import torch

from nestynet_sr.sr_core.bridges import AtomNode, Var
from nestynet_sr.sr_search import fitting_utils
from nestynet_sr.sr_search.factorized_search.adapters.nestynet import stageb_runner
from nestynet_sr.sr_search.factorized_search.config import FactorizedSearchConfig
from nestynet_sr.sr_search.stageB.rule_factorized_search import RuleFactorizedSearchFallback


def test_stageb_factorized_search_rule_exposes_inverse_spec_kwargs():
    hp = FactorizedSearchConfig()
    hp.inverse_spec_enable = True
    hp.inverse_spec_enum_max_depth = 3
    hp.refine_profile = "rare_final_polish"
    hp.refine_mode = "final_polish"
    hp.refine_during_brute = False
    hp.refine_during_mutation = False
    hp.refine_during_controller_slate = False
    hp.refine_during_slate = True
    hp.refine_slate_after_brute = False
    hp.refine_slate_period = 11
    hp.refine_final_polish = True
    hp.refine_slate_k = 5
    hp.refine_slate_diverse_k = 2
    hp.refine_slate_budget = 7

    rule = RuleFactorizedSearchFallback(factorized_search_hp=hp)
    kwargs = stageb_runner._build_stageb_explorer_kwargs_from_rule(
        rule,
        var_dims=None,
        y_dims=None,
        n_iter=17,
    )

    assert kwargs["inverse_spec_enable"] is True
    assert kwargs["inverse_spec_enum_max_depth"] == 3
    assert kwargs["inverse_spec_preview_topk"] == hp.inverse_spec_preview_topk
    assert kwargs["refine_optimizer"] == hp.refine_optimizer
    assert (
        kwargs["refine_lbfgs_escalate_improve_factor"]
        == hp.refine_lbfgs_escalate_improve_factor
    )
    assert kwargs["refine_prune_mapping_equiv_root_slots"] == hp.refine_prune_mapping_equiv_root_slots
    assert kwargs["refine_attempt_cache_enable"] == hp.refine_attempt_cache_enable
    assert kwargs["refine_attempt_cache_max_entries"] == hp.refine_attempt_cache_max_entries
    assert kwargs["refine_profile"] == hp.refine_profile
    assert kwargs["refine_mode"] == hp.refine_mode
    assert kwargs["refine_during_brute"] == hp.refine_during_brute
    assert kwargs["refine_during_mutation"] == hp.refine_during_mutation
    assert kwargs["refine_during_controller_slate"] == hp.refine_during_controller_slate
    assert kwargs["refine_during_slate"] == hp.refine_during_slate
    assert kwargs["refine_slate_after_brute"] == hp.refine_slate_after_brute
    assert kwargs["refine_slate_period"] == hp.refine_slate_period
    assert kwargs["refine_final_polish"] == hp.refine_final_polish
    assert kwargs["refine_slate_k"] == hp.refine_slate_k
    assert kwargs["refine_slate_diverse_k"] == hp.refine_slate_diverse_k
    assert kwargs["refine_slate_budget"] == hp.refine_slate_budget


def test_run_stageb_wrapper_pass_builds_wrapped_candidate(monkeypatch):
    captured = {}

    def fake_run_explorer(**kwargs):
        captured["run_explorer"] = kwargs
        return [
            {
                "expr": "x0",
                "nestynet_ast": Var(0),
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mse_raw": 1.0e-9,
                "mse_eff": 1.0e-9,
            }
        ]

    monkeypatch.setattr(stageb_runner, "run_explorer", fake_run_explorer)
    monkeypatch.setattr(stageb_runner, "embed_mapping_in_ast", lambda *args, **kwargs: Var(0))
    monkeypatch.setattr(stageb_runner, "replace_atom_in_ast", lambda root, target, repl: repl)
    monkeypatch.setattr(stageb_runner, "_mapping_cost", lambda mapping: 0.25)
    monkeypatch.setattr(fitting_utils, "_rational_probe_nd", lambda *args, **kwargs: 1.0e-6)
    monkeypatch.setattr(
        fitting_utils,
        "_nonlinear_substitution_screen",
        lambda *args, **kwargs: [{"error": 1.0e-6}],
    )

    rule = SimpleNamespace(
        outer_wrapper_enable=True,
        outer_wrapper_max_arity=2,
        outer_wrapper_transforms=["log"],
        outer_wrapper_probe_max_points=8,
        outer_wrapper_min_points=4,
        outer_wrapper_min_domain_frac=0.90,
        outer_wrapper_screen_rational_err_max=0.02,
        outer_wrapper_screen_nls_err_max=0.02,
        outer_wrapper_topk=1,
        seed=7,
        n_fit=2,
        n_probe=2,
        n_iter=500,
        outer_wrapper_iter_scale=0.20,
        outer_wrapper_n_seeds=1,
        max_depth=3,
        poly_degree=2,
        outer_wrapper_return_topk=1,
        brute_depth=1,
        early_stop_mse=1.0e-6,
        brute_max_expressions=100,
        refine_enable=False,
        refine_stageb_promote_consts=False,
    )

    target = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    x_all = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float64)
    y_all = torch.tensor([2.0, 4.0, 8.0, 16.0], dtype=torch.float64)
    monkeypatch.setattr(stageb_runner, "_gather_stageb_atom_teacher_data", lambda **kwargs: (x_all, y_all))

    candidates = stageb_runner.run_stageb_wrapper_pass(
        rule=rule,
        root=target,
        target=target,
        probe_jobs=[(0, "ds0", object(), object())],
        declared_consts=[],
        var_dims=None,
        y_dims=None,
        input_exprs=[Var(0)],
        embed_ctx=stageb_runner.StageBEmbedContext(tag_prefix="factorized_search__leaf0"),
        main_structurally_solved=False,
        enforce_units=False,
        device=torch.device("cpu"),
        dtype=torch.float64,
        log_fn=None,
    )

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.label == "factorized_wrap_log(x0)"
    assert str(type(cand.root).__name__) == "ExpNode"
    assert cand.meta["factorized_search_wrapper_transform"] == "log"
    assert cand.meta["factorized_search_wrapper_screen_cheap_solved"] is True
    assert abs(float(cand.meta["mapping_cost"]) - 1.25) < 1.0e-12
    assert captured["run_explorer"]["refine_enable"] is False

    perm = torch.randperm(
        int(y_all.shape[0]),
        generator=torch.Generator(device="cpu").manual_seed(int(rule.seed)),
    )
    expected_y_fit = torch.log(y_all[perm[: int(rule.n_fit)]])
    assert torch.allclose(
        captured["run_explorer"]["y_fit_data"],
        expected_y_fit,
        atol=1.0e-12,
        rtol=0.0,
    )


def test_build_stageb_probe_jobs_single_dataset(monkeypatch):
    target = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    state = SimpleNamespace(root=target, model=object(), models=None)
    ctx = SimpleNamespace(
        state=state,
        train_loader_probe="loader0",
        train_loader_probes=None,
        dataset_ids=None,
    )

    monkeypatch.setattr(
        stageb_runner,
        "_build_atom_to_leaf_map",
        lambda root, model: {id(target): "teacher0"},
    )

    jobs = stageb_runner.build_stageb_probe_jobs(ctx=ctx, target=target, log_fn=None)

    assert jobs == [(0, "ds0", "loader0", "teacher0")]


def test_build_stageb_probe_jobs_multi_dataset_skips_missing_teachers(monkeypatch):
    target = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    models = [object(), object()]
    state = SimpleNamespace(root=target, model=models[0], models=models)
    ctx = SimpleNamespace(
        state=state,
        train_loader_probe="loader0",
        train_loader_probes=["loader0", "loader1"],
        dataset_ids=["A", "B"],
    )

    def fake_build_atom_to_leaf_map(root, model):
        if model is models[0]:
            return {id(target): "teacherA"}
        return {}

    monkeypatch.setattr(stageb_runner, "_build_atom_to_leaf_map", fake_build_atom_to_leaf_map)

    jobs = stageb_runner.build_stageb_probe_jobs(ctx=ctx, target=target, log_fn=None)

    assert jobs == [(0, "A", "loader0", "teacherA")]
