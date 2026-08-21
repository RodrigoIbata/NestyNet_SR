# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import math

import torch

import nestynet_sr.sr_search.factorized_search.explorer as explorer_mod
from nestynet_sr.sr_search.factorized_search.explorer import score_expr, _eval_node_hparam_safe


def _has_const_scaled_trig(node):
    op = node[0]
    if op in ("sin", "cos"):
        arg = node[1]
        if arg[0] == "mul":
            l, r = arg[1], arg[2]
            if (l[0] == "const" and r[0] != "const") or (r[0] == "const" and l[0] != "const"):
                return True
    if op in ("var", "const", "hparam"):
        return False
    if op in ("sin", "cos", "exp", "log", "sqrt", "sqr", "neg"):
        return _has_const_scaled_trig(node[1])
    return _has_const_scaled_trig(node[1]) or _has_const_scaled_trig(node[2])


def _make_problem():
    dtype = torch.float64
    x_fit = torch.linspace(0.2, 3.2, 320, dtype=dtype).unsqueeze(-1)
    x_probe = torch.linspace(0.25, 3.1, 448, dtype=dtype).unsqueeze(-1)
    y_fit = torch.cos(3.0 * x_fit) + 0.5 * x_fit
    y_probe = torch.cos(3.0 * x_probe) + 0.5 * x_probe
    g = torch.Generator(device="cpu").manual_seed(123)
    proj = torch.randn((x_probe.shape[0], 16), generator=g, dtype=dtype)
    node = ("add", ("cos", ("var", 0)), ("mul", ("const", 0.5), ("var", 0)))
    return node, x_fit, y_fit, x_probe, y_probe, proj


def _make_phase1_pure_trig_problem():
    dtype = torch.float64
    x_fit = torch.linspace(0.2, 3.0, 320, dtype=dtype).unsqueeze(-1)
    x_probe = torch.linspace(0.25, 2.95, 420, dtype=dtype).unsqueeze(-1)
    y_fit = torch.cos(2.7 * x_fit)
    y_probe = torch.cos(2.7 * x_probe)
    g = torch.Generator(device="cpu").manual_seed(7)
    proj = torch.randn((x_probe.shape[0], 16), generator=g, dtype=dtype)
    node = ("cos", ("var", 0))
    return node, x_fit, y_fit, x_probe, y_probe, proj


def _make_grid_optimizer_problem(scale=math.e):
    dtype = torch.float64
    x_fit = torch.linspace(0.2, 3.0, 220, dtype=dtype).unsqueeze(-1)
    x_probe = torch.linspace(0.25, 2.95, 320, dtype=dtype).unsqueeze(-1)
    y_fit = torch.sin(float(scale) * x_fit)
    y_probe = torch.sin(float(scale) * x_probe)
    g = torch.Generator(device="cpu").manual_seed(17)
    proj = torch.randn((x_probe.shape[0], 16), generator=g, dtype=dtype)
    node = ("sin", ("var", 0))
    return node, x_fit, y_fit, x_probe, y_probe, proj


def _grid_optimizer_cfg(**overrides):
    cfg = {
        "fit_subset": 160,
        "fit_subset_mode": "stride",
        "num_restarts": 1,
        "max_variants": 1,
        "max_params": 1,
        "linear_combo_enable": True,
        "gate_best_factor": 100.0,
        "max_refines": 4,
        "theta_l2": 1.0e-6,
        "init_log_min": -2.0,
        "init_log_max": 2.0,
        "refine_grid_enable": True,
        "refine_grid_size": 9,
        "refine_grid_size_2d": 5,
        "refine_grid_passes": 0,
        "refine_grid_topk": 1,
        "refine_grid_max_evals": 16,
        "safe_eps": 1.0e-6,
    }
    cfg.update(overrides)
    return cfg


def _make_phase2_problem():
    dtype = torch.float64
    x_fit = torch.linspace(0.35, 3.6, 360, dtype=dtype).unsqueeze(-1)
    x_probe = torch.linspace(0.4, 3.5, 520, dtype=dtype).unsqueeze(-1)
    y_fit = torch.cos(2.7 * torch.log(1.8 * x_fit))
    y_probe = torch.cos(2.7 * torch.log(1.8 * x_probe))
    g = torch.Generator(device="cpu").manual_seed(321)
    proj = torch.randn((x_probe.shape[0], 16), generator=g, dtype=dtype)
    node = ("cos", ("log", ("var", 0)))
    return node, x_fit, y_fit, x_probe, y_probe, proj


def _make_phase3_problem():
    dtype = torch.float64
    x_fit = torch.linspace(0.4, 3.8, 380, dtype=dtype).unsqueeze(-1)
    x_probe = torch.linspace(0.45, 3.7, 540, dtype=dtype).unsqueeze(-1)
    y_fit = torch.cos(2.7 * torch.log(1.8 * x_fit)) + 0.35 * x_fit
    y_probe = torch.cos(2.7 * torch.log(1.8 * x_probe)) + 0.35 * x_probe
    g = torch.Generator(device="cpu").manual_seed(456)
    proj = torch.randn((x_probe.shape[0], 16), generator=g, dtype=dtype)
    node = ("add", ("cos", ("log", ("var", 0))), ("var", 0))
    return node, x_fit, y_fit, x_probe, y_probe, proj


def _make_phase3_product_problem():
    dtype = torch.float64
    x_fit = torch.linspace(0.4, 3.8, 380, dtype=dtype).unsqueeze(-1)
    x_probe = torch.linspace(0.45, 3.7, 540, dtype=dtype).unsqueeze(-1)
    y_fit = torch.cos(2.5 * x_fit) * torch.log(1.8 * x_fit) + 0.35 * x_fit
    y_probe = torch.cos(2.5 * x_probe) * torch.log(1.8 * x_probe) + 0.35 * x_probe
    g = torch.Generator(device="cpu").manual_seed(8)
    proj = torch.randn((x_probe.shape[0], 16), generator=g, dtype=dtype)
    node = ("add", ("mul", ("cos", ("var", 0)), ("log", ("var", 0))), ("var", 0))
    return node, x_fit, y_fit, x_probe, y_probe, proj


def _make_linear_combo_only_problem():
    dtype = torch.float64
    x_fit = torch.linspace(-2.0, 2.0, 401, dtype=dtype).unsqueeze(-1)
    x_probe = torch.linspace(-1.95, 1.95, 509, dtype=dtype).unsqueeze(-1)
    y_fit = 2.0 * x_fit + 3.0 * (x_fit ** 2)
    y_probe = 2.0 * x_probe + 3.0 * (x_probe ** 2)
    g = torch.Generator(device="cpu").manual_seed(19)
    proj = torch.randn((x_probe.shape[0], 16), generator=g, dtype=dtype)
    node = ("add", ("var", 0), ("sqr", ("var", 0)))
    return node, x_fit, y_fit, x_probe, y_probe, proj


def _make_score_head_basis_admission_problem():
    dtype = torch.float64
    g = torch.Generator(device="cpu").manual_seed(29)
    x_fit = 0.4 + 2.4 * torch.rand((401, 3), generator=g, dtype=dtype)
    x_probe = 0.4 + 2.4 * torch.rand((503, 3), generator=g, dtype=dtype)
    y_fit = (x_fit[:, 0] * torch.sqrt(x_fit[:, 1]) + x_fit[:, 2]).unsqueeze(-1)
    y_probe = (x_probe[:, 0] * torch.sqrt(x_probe[:, 1]) + x_probe[:, 2]).unsqueeze(-1)
    proj = torch.randn((x_probe.shape[0], 16), generator=g, dtype=dtype)
    node = ("mul", ("var", 0), ("sqrt", ("var", 1)))
    return node, x_fit, y_fit, x_probe, y_probe, proj


def test_score_expr_refine_disabled_matches_baseline():
    node, x_fit, y_fit, x_probe, y_probe, proj = _make_problem()

    sc_base = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4
    )
    sc_refine_off = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=False,
        refine_cfg={"max_refines": 100},
        refine_state={"trials_done": 0},
    )

    assert sc_base is not None
    assert sc_refine_off is not None
    assert abs(float(sc_base[0]) - float(sc_refine_off[0])) < 1.0e-12
    assert sc_base[1] == sc_refine_off[1]


def test_score_expr_refine_phase1_improves_trig_frequency_mismatch():
    node, x_fit, y_fit, x_probe, y_probe, proj = _make_problem()

    sc_off = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=False,
    )
    refine_state = {"trials_done": 0}
    sc_on = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=True,
        refine_cfg={
            "lbfgs_steps": 14,
            "fit_subset": 256,
            "num_restarts": 5,
            "max_variants": 4,
            "linear_combo_enable": False,
            "gate_best_factor": 100.0,
            "max_refines": 20,
            "theta_l2": 1.0e-5,
            "init_log_min": -2.0,
            "init_log_max": 2.0,
        },
        refine_best_mse=float("inf"),
        refine_state=refine_state,
    )

    assert sc_off is not None
    assert sc_on is not None
    assert refine_state["trials_done"] > 0
    assert math.isfinite(float(sc_on[0]))
    assert float(sc_on[0]) < float(sc_off[0]) * 1.0e-2


def test_score_expr_refine_phase1_recovers_cos_a_x0():
    node, x_fit, y_fit, x_probe, y_probe, proj = _make_phase1_pure_trig_problem()
    sc_off = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=False,
    )
    sc_on = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=True,
        refine_cfg={
            "lbfgs_steps": 14,
            "fit_subset": 256,
            "num_restarts": 5,
            "max_variants": 4,
            "linear_combo_enable": False,
            "gate_best_factor": 100.0,
            "max_refines": 20,
            "theta_l2": 1.0e-5,
            "init_log_min": -2.0,
            "init_log_max": 2.0,
        },
        refine_best_mse=float("inf"),
        refine_state={"trials_done": 0},
        return_expr=True,
    )
    assert sc_off is not None
    assert sc_on is not None
    assert math.isfinite(float(sc_on[0]))
    assert float(sc_on[0]) < float(sc_off[0]) * 1.0e-2
    assert _has_const_scaled_trig(sc_on[4])


def test_score_expr_refine_phase1_returns_refined_expression_for_archive():
    node, x_fit, y_fit, x_probe, y_probe, proj = _make_problem()
    sc = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=True,
        refine_cfg={
            "lbfgs_steps": 14,
            "fit_subset": 256,
            "num_restarts": 5,
            "max_variants": 4,
            "linear_combo_enable": False,
            "gate_best_factor": 100.0,
            "max_refines": 20,
            "theta_l2": 1.0e-5,
            "init_log_min": -2.0,
            "init_log_max": 2.0,
        },
        refine_best_mse=float("inf"),
        refine_state={"trials_done": 0},
        return_expr=True,
    )
    assert sc is not None
    refined = sc[4]
    assert refined != node
    assert _has_const_scaled_trig(refined)


def test_refine_optimizer_grid_skips_lbfgs(monkeypatch):
    node, x_fit, y_fit, x_probe, y_probe, proj = _make_grid_optimizer_problem()

    def _fail_lbfgs(*args, **kwargs):
        raise AssertionError("grid optimizer should not construct LBFGS")

    monkeypatch.setattr(torch.optim, "LBFGS", _fail_lbfgs)
    sc_off = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=False,
        return_expr=True,
    )
    state = {"trials_done": 0}
    sc_on = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=True,
        refine_cfg=_grid_optimizer_cfg(optimizer="grid"),
        refine_best_mse=float("inf"),
        refine_state=state,
        return_expr=True,
    )

    assert sc_off is not None
    assert sc_on is not None
    assert state["trials_done"] == 1
    assert float(sc_on[0]) < float(sc_off[0]) * 1.0e-4
    assert _has_const_scaled_trig(sc_on[4])


def test_refine_optimizer_grid_then_lbfgs_respects_escalation_gate(monkeypatch):
    node, x_fit, y_fit, x_probe, y_probe, proj = _make_grid_optimizer_problem(scale=2.0)

    def _fail_lbfgs(*args, **kwargs):
        raise AssertionError("grid_then_lbfgs should skip LBFGS when the grid gate fails")

    monkeypatch.setattr(torch.optim, "LBFGS", _fail_lbfgs)
    state = {"trials_done": 0}
    sc = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=True,
        refine_cfg=_grid_optimizer_cfg(
            optimizer="grid_then_lbfgs",
            lbfgs_escalate_improve_factor=1.0e12,
        ),
        refine_best_mse=float("inf"),
        refine_state=state,
        return_expr=True,
    )

    assert sc is not None
    assert state["trials_done"] == 1


def test_refine_optimizer_grid_then_lbfgs_escalates_after_grid_improvement(monkeypatch):
    node, x_fit, y_fit, x_probe, y_probe, proj = _make_grid_optimizer_problem()
    calls = {"count": 0}

    class _CountingLBFGS:
        def __init__(self, params, **kwargs):
            self.params = list(params)
            calls["count"] += 1

        def zero_grad(self):
            for param in self.params:
                param.grad = None

        def step(self, closure):
            return closure()

    monkeypatch.setattr(torch.optim, "LBFGS", _CountingLBFGS)
    state = {"trials_done": 0}
    sc = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=True,
        refine_cfg=_grid_optimizer_cfg(
            optimizer="grid_then_lbfgs",
            lbfgs_escalate_improve_factor=1.0,
        ),
        refine_best_mse=float("inf"),
        refine_state=state,
        return_expr=True,
    )

    assert sc is not None
    assert state["trials_done"] == 1
    assert calls["count"] == 1
    assert _has_const_scaled_trig(sc[4])


def test_refine_slot_sensitivity_ranks_even_when_under_variant_cap(monkeypatch):
    node = ("add", ("sin", ("var", 0)), ("sin", ("var", 1)))
    x_fit = torch.zeros((8, 2), dtype=torch.float64)
    y_fit = torch.zeros((8, 1), dtype=torch.float64)

    def _fake_sensitivity(_node, _slot_kind, path, _x_fit, _y_fit, _cfg):
        return 10.0 if tuple(path) == (2,) else 1.0

    monkeypatch.setattr(explorer_mod, "_slot_sensitivity_score", _fake_sensitivity)

    variants = explorer_mod._decorate_refine_variants(
        node,
        max_variants=2,
        max_params=1,
        x_fit=x_fit,
        y_fit=y_fit,
        cfg={"slot_sensitivity_enable": True},
    )

    assert len(variants) == 2
    assert "sin((hp0*x1))" in explorer_mod.node_str(variants[0][0])


def test_refine_prunes_mapping_equivalent_root_log_slot():
    x_fit = torch.ones((8, 1), dtype=torch.float64)
    y_fit = torch.ones((8, 1), dtype=torch.float64)
    diag = {}

    root_variants = explorer_mod._decorate_refine_variants(
        ("log", ("var", 0)),
        max_variants=4,
        max_params=1,
        x_fit=x_fit,
        y_fit=y_fit,
        cfg={"prune_mapping_equiv_root_slots": True, "diagnostics": diag},
    )
    nested_variants = explorer_mod._decorate_refine_variants(
        ("sqr", ("log", ("var", 0))),
        max_variants=4,
        max_params=1,
        x_fit=x_fit,
        y_fit=y_fit,
        cfg={"prune_mapping_equiv_root_slots": True},
    )

    assert root_variants == []
    assert diag["mapping_equiv_root_slots_pruned"] == 1
    assert any("log((hp0*x0))" in explorer_mod.node_str(v[0]) for v in nested_variants)


def test_refine_attempt_cache_reuses_successful_materialized_candidate(monkeypatch):
    node, x_fit, y_fit, x_probe, y_probe, proj = _make_grid_optimizer_problem(scale=2.0)
    calls = {"count": 0}

    def _fake_refine_hparams(*args, **kwargs):
        calls["count"] += 1
        return [2.0]

    monkeypatch.setattr(explorer_mod, "_refine_hparams", _fake_refine_hparams)
    cfg = _grid_optimizer_cfg(
        optimizer="grid",
        diagnostics={},
        attempt_cache={},
        attempt_cache_enable=True,
        attempt_cache_max_entries=8,
    )

    state1 = {"trials_done": 0}
    sc1 = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=True,
        refine_cfg=cfg,
        refine_best_mse=float("inf"),
        refine_state=state1,
        return_expr=True,
    )
    state2 = {"trials_done": 0}
    sc2 = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=True,
        refine_cfg=cfg,
        refine_best_mse=float("inf"),
        refine_state=state2,
        return_expr=True,
    )

    assert sc1 is not None
    assert sc2 is not None
    assert calls["count"] == 1
    assert state1["trials_done"] == 1
    assert state2["trials_done"] == 0
    assert cfg["diagnostics"]["attempt_cache_hits"] >= 1
    assert cfg["diagnostics"]["attempt_cache_stores"] == 1


def test_score_expr_refine_phase2_improves_log_and_trig_scales():
    node, x_fit, y_fit, x_probe, y_probe, proj = _make_phase2_problem()

    sc_off = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=False,
    )
    refine_state = {"trials_done": 0}
    sc_on = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=True,
        refine_cfg={
            "lbfgs_steps": 20,
            "fit_subset": 300,
            "num_restarts": 9,
            "max_variants": 12,
            "max_params": 2,
            "linear_combo_enable": False,
            "gate_best_factor": 100.0,
            "max_refines": 40,
            "theta_l2": 1.0e-6,
            "init_log_min": -2.0,
            "init_log_max": 2.0,
        },
        refine_best_mse=float("inf"),
        refine_state=refine_state,
    )

    assert sc_off is not None
    assert sc_on is not None
    assert refine_state["trials_done"] > 0
    assert math.isfinite(float(sc_on[0]))
    assert float(sc_on[0]) < float(sc_off[0]) * 1.0e-2


def test_score_expr_refine_phase3_additive_linear_basis_improves():
    node, x_fit, y_fit, x_probe, y_probe, proj = _make_phase3_problem()

    sc_phase2_style = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=True,
        refine_cfg={
            "lbfgs_steps": 20,
            "fit_subset": 320,
            "num_restarts": 9,
            "max_variants": 12,
            "max_params": 2,
            "linear_combo_enable": False,
            "gate_best_factor": 100.0,
            "max_refines": 40,
            "theta_l2": 1.0e-6,
            "init_log_min": -2.0,
            "init_log_max": 2.0,
        },
        refine_best_mse=float("inf"),
        refine_state={"trials_done": 0},
    )
    sc_phase3 = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=True,
        refine_cfg={
            "lbfgs_steps": 20,
            "fit_subset": 320,
            "num_restarts": 9,
            "max_variants": 12,
            "max_params": 2,
            "linear_combo_enable": True,
            "linear_terms_max": 6,
            "linear_prune_rel": 1.0e-12,
            "gate_best_factor": 100.0,
            "max_refines": 40,
            "theta_l2": 1.0e-6,
            "init_log_min": -2.0,
            "init_log_max": 2.0,
        },
        refine_best_mse=float("inf"),
        refine_state={"trials_done": 0},
    )

    assert sc_phase2_style is not None
    assert sc_phase3 is not None
    assert math.isfinite(float(sc_phase3[0]))
    assert float(sc_phase3[0]) < float(sc_phase2_style[0]) * 1.0e-2


def test_score_expr_refine_phase3_recovers_cos_a_x_log_b_x_refine_cx():
    node, x_fit, y_fit, x_probe, y_probe, proj = _make_phase3_product_problem()
    sc_off = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=False,
    )
    sc_phase2_style = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=True,
        refine_cfg={
            "lbfgs_steps": 20,
            "fit_subset": 320,
            "num_restarts": 9,
            "max_variants": 12,
            "max_params": 2,
            "linear_combo_enable": False,
            "gate_best_factor": 100.0,
            "max_refines": 40,
            "theta_l2": 1.0e-6,
            "init_log_min": -2.0,
            "init_log_max": 2.0,
        },
        refine_best_mse=float("inf"),
        refine_state={"trials_done": 0},
    )
    sc_phase3 = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=True,
        refine_cfg={
            "lbfgs_steps": 20,
            "fit_subset": 320,
            "num_restarts": 9,
            "max_variants": 12,
            "max_params": 2,
            "linear_combo_enable": True,
            "linear_terms_max": 6,
            "linear_prune_rel": 1.0e-12,
            "gate_best_factor": 100.0,
            "max_refines": 40,
            "theta_l2": 1.0e-6,
            "init_log_min": -2.0,
            "init_log_max": 2.0,
        },
        refine_best_mse=float("inf"),
        refine_state={"trials_done": 0},
    )
    assert sc_off is not None
    assert sc_phase2_style is not None
    assert sc_phase3 is not None
    assert math.isfinite(float(sc_phase3[0]))
    assert float(sc_phase3[0]) < float(sc_off[0]) * 0.05
    assert float(sc_phase3[0]) < float(sc_phase2_style[0]) * 0.2


def test_score_expr_refine_linear_combo_runs_without_trig_variants():
    node, x_fit, y_fit, x_probe, y_probe, proj = _make_linear_combo_only_problem()

    sc_off = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=False,
    )
    sc_on = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=True,
        refine_cfg={
            "linear_combo_enable": True,
            "linear_terms_max": 6,
            "linear_prune_rel": 1.0e-12,
            "max_variants": 4,
            "max_refines": 8,
            "fit_subset_mode": "stride",
            "gate_best_factor": 100.0,
        },
        refine_best_mse=float("inf"),
        refine_state={"trials_done": 0},
        return_expr=True,
    )

    assert sc_off is not None
    assert sc_on is not None
    assert math.isfinite(float(sc_on[0]))
    assert float(sc_on[0]) < float(sc_off[0]) * 1.0e-4
    assert sc_on[4] != node


def test_score_expr_materializes_additive_basis_admission_from_head_terms():
    node, x_fit, y_fit, x_probe, y_probe, proj = _make_score_head_basis_admission_problem()
    sc = score_expr(
        node,
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        proj,
        "bits",
        2.0,
        6,
        4,
        refine_enable=False,
        refine_cfg={
            "max_depth": 6,
            "score_head_enable": True,
            "score_head_vars_enable": True,
            "score_head_var_terms": [("var", 0), ("var", 1), ("var", 2)],
            "score_head_direct_combo_enable": True,
            "score_head_direct_combo_prune_rel": 1.0e-10,
        },
        return_expr=True,
    )

    assert sc is not None
    assert math.isfinite(float(sc[0]))
    assert float(sc[0]) < 1.0e-18
    assert "x2" in explorer_mod.node_str(sc[4])
    mapping = sc[3]
    assert isinstance(mapping, dict)
    assert mapping.get("kind") == "poly"
    assert "_lin_head" not in mapping
    transition = mapping.get("_basis_transition", None)
    assert isinstance(transition, dict)
    assert transition.get("kind") == "additive_basis_admission"
    assert explorer_mod.node_str(transition["core_expr"]) in {"(x0*sqrt(x1))", "(sqrt(x1)*x0)"}
    assert "x2" in [explorer_mod.node_str(node) for node in list(transition.get("term_nodes", []))]
    assert explorer_mod.node_str(transition["compiled_expr"]) in {"((x0*sqrt(x1))+x2)", "((sqrt(x1)*x0)+x2)"}


def test_score_expr_refine_gate_potential_unlocks_refine_when_base_gate_fails():
    node, x_fit, y_fit, x_probe, y_probe, proj = _make_phase1_pure_trig_problem()

    cfg_common = {
        "lbfgs_steps": 16,
        "fit_subset": 256,
        "fit_subset_mode": "stride",
        "num_restarts": 4,
        "max_variants": 6,
        "max_params": 1,
        "linear_combo_enable": False,
        "gate_best_factor": 2.0,
        "max_refines": 20,
        "theta_l2": 1.0e-5,
        "init_log_min": -2.0,
        "init_log_max": 2.0,
    }

    sc_blocked = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=True,
        refine_cfg={
            **cfg_common,
            "gate_potential_enable": False,
        },
        refine_best_mse=1.0e-6,
        refine_state={"trials_done": 0},
    )
    st = {"trials_done": 0}
    sc_unlocked = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=True,
        refine_cfg={
            **cfg_common,
            "gate_potential_enable": True,
            "gate_potential_improve_factor": 1.2,
            "gate_log_min": -2.0,
            "gate_log_max": 2.0,
            "gate_grid_size": 8,
        },
        refine_best_mse=1.0e-6,
        refine_state=st,
    )

    assert sc_blocked is not None
    assert sc_unlocked is not None
    assert st["trials_done"] > 0
    assert float(sc_unlocked[0]) < float(sc_blocked[0]) * 1.0e-2


def test_score_expr_refine_respects_phase4_budgets():
    node, x_fit, y_fit, x_probe, y_probe, proj = _make_problem()

    sc_base = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=False,
    )
    sc_blocked = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=True,
        refine_cfg={
            "lbfgs_steps": 14,
            "fit_subset": 256,
            "num_restarts": 5,
            "max_variants": 4,
            "linear_combo_enable": False,
            "gate_best_factor": 100.0,
            "max_refines": 100,
        },
        refine_best_mse=float("inf"),
        refine_state={"trials_done": 0, "depth_trials_left": 0, "window_trials_left": 0},
    )
    assert sc_base is not None
    assert sc_blocked is not None
    assert abs(float(sc_blocked[0]) - float(sc_base[0])) < 1.0e-12

    st = {"trials_done": 0, "depth_trials_left": 1, "window_trials_left": 1}
    sc_budgeted = score_expr(
        node, x_fit, y_fit, x_probe, y_probe, proj, "bits", 2.0, 6, 4,
        refine_enable=True,
        refine_cfg={
            "lbfgs_steps": 14,
            "fit_subset": 256,
            "num_restarts": 5,
            "max_variants": 4,
            "linear_combo_enable": False,
            "gate_best_factor": 100.0,
            "max_refines": 100,
        },
        refine_best_mse=float("inf"),
        refine_state=st,
    )
    assert sc_budgeted is not None
    assert st["trials_done"] == 1
    assert st["depth_trials_left"] == 0
    assert st["window_trials_left"] == 0


def test_refine_safe_eval_handles_domain_stress_without_nan():
    x = torch.tensor([[-2.0], [0.0], [1.0e-9], [0.2], [1.0]], dtype=torch.float64)
    expr = (
        "add",
        ("sqrt", ("var", 0)),
        ("add", ("log", ("var", 0)), ("div", ("const", 1.0), ("var", 0))),
    )
    v, pen = _eval_node_hparam_safe(
        expr,
        x,
        [],
        {"safe_eps": 1.0e-6, "safe_exp_clip": 20.0},
    )
    assert torch.isfinite(v).all()
    assert torch.isfinite(pen)
    assert float(pen) > 0.0


def test_phase6_brute_archives_refined_expression(monkeypatch):
    refined_expr = ("cos", ("mul", ("const", 3.0), ("var", 0)))

    def _fake_score_expr(*args, **kwargs):
        z = torch.zeros((4,), dtype=torch.float64)
        mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
        if kwargs.get("return_expr", False):
            return 1.0e-3, ("k",), z, mapping, refined_expr
        return 1.0e-3, ("k",), z, mapping

    monkeypatch.setattr(explorer_mod, "score_expr", _fake_score_expr)

    x = torch.linspace(0.2, 0.8, 32, dtype=torch.float64).unsqueeze(-1)
    y = x.clone()
    proj = torch.randn((x.shape[0], 4), generator=torch.Generator(device="cpu").manual_seed(1), dtype=torch.float64)
    arch = explorer_mod.ResidualBasinArchive()

    explorer_mod._run_brute_phase(
        arch=arch,
        nvars=1,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        proj=proj,
        fp_mode="bits",
        q_scale=2.0,
        q_clip=6,
        poly_degree=4,
        brute_depth=1,
        max_expressions=4,
        refine_enable=True,
        refine_cfg={"trials_per_brute_depth": 1},
        refine_state={"trials_done": 0},
        early_stop_mse=1.0e-12,
    )

    assert len(arch.d) > 0
    assert arch.best(1)[0].best_expr == refined_expr


def test_phase6_mutation_archives_refined_expression(monkeypatch):
    refined_expr = ("cos", ("mul", ("const", 2.5), ("var", 0)))

    def _fake_score_expr(*args, **kwargs):
        z = torch.zeros((4,), dtype=torch.float64)
        mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
        if kwargs.get("return_expr", False):
            return 1.0e-3, ("k_mut",), z, mapping, refined_expr
        return 1.0e-3, ("k_mut",), z, mapping

    monkeypatch.setattr(explorer_mod, "score_expr", _fake_score_expr)

    def _target_fn(x):
        return x[:, :1]

    arch = explorer_mod.run_explorer_core(
        _target_fn,
        1,
        n_iter=1,
        max_depth=1,
        poly_degree=4,
        lo=0.2,
        hi=0.8,
        seed=0,
        dtype=torch.float64,
        brute_depth=0,
        refine_enable=True,
    )

    assert len(arch.d) > 0
    assert arch.best(1)[0].best_expr == refined_expr


def _run_search_recording_refine_flags(*, monkeypatch=None, patch_brute_score_expr=False, **kwargs):
    calls = []

    def _fake_score_expr(node, x_fit, y_fit, x_probe, y_probe, *args, **score_kwargs):
        calls.append(bool(score_kwargs.get("refine_enable", False)))
        z = torch.zeros((x_probe.shape[0], 1), dtype=torch.float64)
        mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
        key = ("placement", len(calls), bool(score_kwargs.get("refine_enable", False)))
        if score_kwargs.get("return_expr", False):
            return 1.0e-3, key, z, mapping, node
        return 1.0e-3, key, z, mapping

    def _target_fn(x):
        return x[:, :1]

    if patch_brute_score_expr:
        assert monkeypatch is not None
        monkeypatch.setattr(explorer_mod, "score_expr", _fake_score_expr)

    arch = explorer_mod.run_explorer_core(
        _target_fn,
        1,
        max_depth=1,
        poly_degree=4,
        lo=0.2,
        hi=0.8,
        seed=0,
        dtype=torch.float64,
        print_every=0,
        verbose=False,
        no_residual=True,
        _score_expr_fn=_fake_score_expr,
        **kwargs,
    )
    return arch, calls


def test_refine_enable_defaults_to_scheduled_slate(monkeypatch):
    arch, calls = _run_search_recording_refine_flags(
        monkeypatch=monkeypatch,
        patch_brute_score_expr=True,
        n_iter=0,
        brute_depth=1,
        brute_max_expressions=4,
        refine_enable=True,
        refine_final_polish=False,
        refine_slate_budget=1,
        refine_max_trials=1,
    )

    assert calls
    assert False in calls
    assert True in calls
    cfg = getattr(arch, "refine_runtime_config", {})
    assert cfg["refine_mode"] == "slate"
    assert cfg["brute_refine_enable"] is False
    assert cfg["mutation_refine_enable"] is False
    assert cfg["scheduled_slate_refine_enable"] is True
    assert cfg["after_brute_slate_refine_enable"] is True


def test_refine_placement_disables_brute_inline_refinement(monkeypatch):
    arch, calls = _run_search_recording_refine_flags(
        monkeypatch=monkeypatch,
        patch_brute_score_expr=True,
        n_iter=0,
        brute_depth=1,
        brute_max_expressions=4,
        refine_enable=True,
        refine_mode="inline",
        refine_during_brute=False,
        refine_trials_per_brute_depth=1,
    )

    assert calls
    assert set(calls) == {False}
    cfg = getattr(arch, "refine_runtime_config", {})
    assert cfg["brute_refine_enable"] is False
    assert cfg["mutation_refine_enable"] is True


def test_brute_trials_per_depth_zero_means_zero_refinement_budget(monkeypatch):
    depth_left_seen = []

    def _fake_score_expr(*args, **kwargs):
        state = kwargs.get("refine_state", {}) or {}
        depth_left_seen.append(state.get("depth_trials_left", None))
        z = torch.zeros((4,), dtype=torch.float64)
        mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
        if kwargs.get("return_expr", False):
            return 1.0e-3, ("k_zero_budget",), z, mapping, args[0]
        return 1.0e-3, ("k_zero_budget",), z, mapping

    monkeypatch.setattr(explorer_mod, "score_expr", _fake_score_expr)

    x = torch.linspace(0.2, 0.8, 32, dtype=torch.float64).unsqueeze(-1)
    y = x.clone()
    proj = torch.randn((x.shape[0], 4), generator=torch.Generator(device="cpu").manual_seed(1), dtype=torch.float64)
    arch = explorer_mod.ResidualBasinArchive()

    explorer_mod._run_brute_phase(
        arch=arch,
        nvars=1,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        proj=proj,
        fp_mode="bits",
        q_scale=2.0,
        q_clip=6,
        poly_degree=4,
        brute_depth=1,
        max_expressions=4,
        refine_enable=True,
        refine_cfg={"trials_per_brute_depth": 0},
        refine_state={"trials_done": 0},
        early_stop_mse=1.0e-12,
    )

    assert depth_left_seen
    assert set(depth_left_seen) == {0}


def test_refine_placement_disables_mutation_inline_refinement():
    arch, calls = _run_search_recording_refine_flags(
        n_iter=1,
        brute_depth=0,
        refine_enable=True,
        refine_mode="inline",
        refine_during_mutation=False,
    )

    assert calls
    assert set(calls) == {False}
    cfg = getattr(arch, "refine_runtime_config", {})
    assert cfg["brute_refine_enable"] is True
    assert cfg["mutation_refine_enable"] is False


def test_refine_mode_off_disables_inline_refinement_everywhere(monkeypatch):
    arch, calls = _run_search_recording_refine_flags(
        monkeypatch=monkeypatch,
        patch_brute_score_expr=True,
        n_iter=1,
        brute_depth=1,
        brute_max_expressions=4,
        refine_enable=True,
        refine_mode="off",
        refine_during_brute=True,
        refine_during_mutation=True,
    )

    assert calls
    assert set(calls) == {False}
    cfg = getattr(arch, "refine_runtime_config", {})
    assert cfg["refine_mode"] == "off"
    assert cfg["refine_active"] is False
    assert cfg["brute_refine_enable"] is False
    assert cfg["mutation_refine_enable"] is False


def test_refine_mode_slate_runs_after_brute_slate_pass(monkeypatch):
    calls = []
    refined_expr = ("mul", ("const", 2.0), ("var", 0))

    def _fake_score_expr(node, x_fit, y_fit, x_probe, y_probe, *args, **kwargs):
        refine = bool(kwargs.get("refine_enable", False))
        calls.append(refine)
        state = kwargs.get("refine_state", None)
        if refine and isinstance(state, dict):
            state["trials_done"] = int(state.get("trials_done", 0)) + 1
            if state.get("window_trials_left", None) is not None:
                state["window_trials_left"] = int(state["window_trials_left"]) - 1
        z = torch.zeros((x_probe.shape[0], 1), dtype=torch.float64)
        mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
        if refine:
            return 1.0e-6, ("slate_refined",), z, mapping, refined_expr
        return 1.0e-3, ("slate_raw",), z, mapping, node

    monkeypatch.setattr(explorer_mod, "score_expr", _fake_score_expr)

    def _target_fn(x):
        return x[:, :1]

    arch = explorer_mod.run_explorer_core(
        _target_fn,
        1,
        n_iter=0,
        max_depth=1,
        poly_degree=4,
        lo=0.2,
        hi=0.8,
        seed=0,
        dtype=torch.float64,
        print_every=0,
        verbose=False,
        no_residual=True,
        brute_depth=1,
        brute_max_expressions=4,
        refine_enable=True,
        refine_mode="slate",
        refine_slate_after_brute=True,
        refine_final_polish=False,
        refine_slate_k=4,
        refine_slate_diverse_k=0,
        refine_slate_budget=2,
        refine_max_trials=10,
        _score_expr_fn=_fake_score_expr,
    )

    assert calls
    assert False in calls
    assert True in calls
    assert arch.best(1)[0].best_expr == refined_expr
    cfg = getattr(arch, "refine_runtime_config", {})
    assert cfg["brute_refine_enable"] is False
    assert cfg["mutation_refine_enable"] is False
    assert cfg["scheduled_slate_refine_enable"] is True
    assert cfg["after_brute_slate_refine_enable"] is True
    stats = getattr(arch, "refine_slate_stats", {})
    assert stats["total_passes"] == 1
    assert stats["total_accepted"] == 1
    assert stats["total_trials_used"] == 1
    assert stats["passes"][0]["source"] == "after_brute"
    diag = getattr(arch, "refine_diagnostics", {})
    assert isinstance(diag, dict)
    assert diag["attempt_cache_size"] == 0


def test_refine_mode_final_polish_runs_only_at_end():
    calls = []
    refined_expr = ("mul", ("const", 3.0), ("var", 0))

    def _fake_score_expr(node, x_fit, y_fit, x_probe, y_probe, *args, **kwargs):
        refine = bool(kwargs.get("refine_enable", False))
        calls.append(refine)
        state = kwargs.get("refine_state", None)
        if refine and isinstance(state, dict):
            state["trials_done"] = int(state.get("trials_done", 0)) + 1
            if state.get("window_trials_left", None) is not None:
                state["window_trials_left"] = int(state["window_trials_left"]) - 1
        z = torch.zeros((x_probe.shape[0], 1), dtype=torch.float64)
        mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
        if refine:
            return 1.0e-6, ("final_refined",), z, mapping, refined_expr
        return 1.0e-3, ("final_raw",), z, mapping, node

    def _target_fn(x):
        return x[:, :1]

    arch = explorer_mod.run_explorer_core(
        _target_fn,
        1,
        n_iter=1,
        max_depth=1,
        poly_degree=4,
        lo=0.2,
        hi=0.8,
        seed=0,
        dtype=torch.float64,
        print_every=0,
        verbose=False,
        no_residual=True,
        brute_depth=0,
        periodic_seed_enable=False,
        refine_enable=True,
        refine_mode="final_polish",
        refine_final_polish=True,
        refine_slate_k=4,
        refine_slate_diverse_k=0,
        refine_slate_budget=2,
        refine_max_trials=10,
        _score_expr_fn=_fake_score_expr,
    )

    assert calls
    assert calls[-1] is True
    assert any(call is False for call in calls[:-1])
    assert arch.best(1)[0].best_expr == refined_expr
    cfg = getattr(arch, "refine_runtime_config", {})
    assert cfg["after_brute_slate_refine_enable"] is False
    assert cfg["final_polish_refine_enable"] is True
    stats = getattr(arch, "refine_slate_stats", {})
    assert stats["total_passes"] == 1
    assert stats["total_accepted"] == 1
    assert stats["passes"][0]["source"] == "final_polish"


def test_mutation_early_stop_threshold_breaks_without_stop_event(monkeypatch):
    refined_expr = ("cos", ("mul", ("const", 2.5), ("var", 0)))

    def _fake_score_expr(*args, **kwargs):
        z = torch.zeros((4,), dtype=torch.float64)
        mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
        if kwargs.get("return_expr", False):
            return 1.0e-12, ("k_early",), z, mapping, refined_expr
        return 1.0e-12, ("k_early",), z, mapping

    monkeypatch.setattr(explorer_mod, "score_expr", _fake_score_expr)

    def _target_fn(x):
        return x[:, :1]

    arch = explorer_mod.run_explorer_core(
        _target_fn,
        1,
        n_iter=200,
        max_depth=1,
        poly_degree=4,
        lo=0.2,
        hi=0.8,
        seed=0,
        dtype=torch.float64,
        brute_depth=0,
        periodic_seed_enable=False,
        early_stop_mse=1.0e-8,
        refine_enable=True,
        print_every=0,
    )

    assert len(arch.d) > 0
    assert arch.n_eval == 1
    assert arch.best(1)[0].best_expr == refined_expr


def test_mutation_early_stop_ignores_nonstructural_mapping(monkeypatch):
    refined_expr = ("cos", ("mul", ("const", 2.5), ("var", 0)))

    def _fake_score_expr(*args, **kwargs):
        z = torch.zeros((4,), dtype=torch.float64)
        # degree-4 polynomial mapping (non-structural for solved-gating)
        mapping = {"kind": "poly", "coeffs": [0.0, 1.0, 0.0, 0.0, 0.0], "mu": 0.0, "std": 1.0}
        if kwargs.get("return_expr", False):
            return 1.0e-12, ("k_early_nonstruct",), z, mapping, refined_expr
        return 1.0e-12, ("k_early_nonstruct",), z, mapping

    monkeypatch.setattr(explorer_mod, "score_expr", _fake_score_expr)

    def _target_fn(x):
        return x[:, :1]

    arch = explorer_mod.run_explorer_core(
        _target_fn,
        1,
        n_iter=25,
        max_depth=1,
        poly_degree=4,
        lo=0.2,
        hi=0.8,
        seed=0,
        dtype=torch.float64,
        brute_depth=0,
        early_stop_mse=1.0e-8,
        refine_enable=True,
        print_every=0,
    )

    assert len(arch.d) > 0
    assert arch.n_eval > 1
    assert arch.best(1)[0].best_expr == refined_expr


def test_brute_early_stop_ignores_nonstructural_mapping(monkeypatch):
    refined_expr = ("cos", ("mul", ("const", 3.0), ("var", 0)))

    def _fake_score_expr(*args, **kwargs):
        z = torch.zeros((4,), dtype=torch.float64)
        # degree-4 polynomial mapping (non-structural for solved-gating)
        mapping = {"kind": "poly", "coeffs": [0.0, 1.0, 0.0, 0.0, 0.0], "mu": 0.0, "std": 1.0}
        if kwargs.get("return_expr", False):
            return 1.0e-12, ("k_brute_nonstruct",), z, mapping, refined_expr
        return 1.0e-12, ("k_brute_nonstruct",), z, mapping

    monkeypatch.setattr(explorer_mod, "score_expr", _fake_score_expr)

    x = torch.linspace(0.2, 0.8, 32, dtype=torch.float64).unsqueeze(-1)
    y = x.clone()
    proj = torch.randn((x.shape[0], 4), generator=torch.Generator(device="cpu").manual_seed(1), dtype=torch.float64)
    arch = explorer_mod.ResidualBasinArchive()

    solved = explorer_mod._run_brute_phase(
        arch=arch,
        nvars=1,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        proj=proj,
        fp_mode="bits",
        q_scale=2.0,
        q_clip=6,
        poly_degree=4,
        brute_depth=1,
        max_expressions=4,
        refine_enable=True,
        refine_cfg={"trials_per_brute_depth": 1},
        refine_state={"trials_done": 0},
        early_stop_mse=1.0e-8,
    )

    assert len(arch.d) > 0
    assert solved is False
    assert arch.best(1)[0].best_expr == refined_expr
