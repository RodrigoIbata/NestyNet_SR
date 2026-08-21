# SPDX-License-Identifier: MPL-2.0

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nestynet_sr.sr_core.ast_simplify import SimplifyOptions, simplify_ast
from nestynet_sr.sr_core.bridges import Mul, Pow, Var
from nestynet_sr.sr_de.de_search import DESearchConfig, _prolongation_penalty
from nestynet_sr.sr_de.de_search import build_de_library_terms_with_sources
from nestynet_sr.sr_de.de_search import _score_de_lie_prolongation_multi
from nestynet_sr.sr_expr_ir.config import ExpressionIRConfig
from nestynet_sr.sr_expr_ir.tuple_bridge import canonical_key_tuple_ast, maybe_canonicalize_tuple_ast
from nestynet_sr.sr_gs import GeneralizedSymmetryConfig
from nestynet_sr.sr_gs.stagea_bridge import stageA_generalized_symmetry_proposals
from nestynet_sr.sr_gs.unit_torus import enumerate_nullspace_exponents, projective_exponent_key
from nestynet_sr.sr_search.factorized_search.proposal_families.gs import (
    build_gs_fss_context,
    extend_pool_with_gs_atoms,
)


def test_prolongation_is_audit_only_by_default_and_missing_metric_penalized_when_selected():
    audit_cfg = DESearchConfig(gs_enable=True, gs_de_lie_prolongation=True)
    assert _prolongation_penalty(audit_cfg, 0.7) == 0.0
    assert _prolongation_penalty(audit_cfg, None) == 0.0

    selection_cfg = DESearchConfig(
        gs_enable=True,
        gs_de_lie_prolongation=True,
        gs_de_lie_use_for_selection=True,
        gs_de_lie_prolongation_weight=0.05,
    )
    assert _prolongation_penalty(selection_cfg, 0.7) == pytest.approx(0.035)
    assert _prolongation_penalty(selection_cfg, None) == 0.5


def test_gs_fss_score_only_does_not_mutate_pool():
    cfg = type(
        "Cfg",
        (),
        {
            "expr_gs_fss_score": True,
            "expr_gs_fss_aux_generator": False,
            "expr_gs_fss_max_aux_atoms": 8,
        },
    )()
    ctx = build_gs_fss_context(nvars=2, feature_names=("x0", "x1"), cfg=cfg)
    pool = [("var", 0), ("var", 1)]

    out = extend_pool_with_gs_atoms(pool, ctx)

    assert out == pool
    assert ctx.score_enabled is True
    assert ctx.aux_generator_enabled is False
    assert ctx.reason == "score_only_not_wired"


def test_neutral_hard_tail_templates_do_not_require_gs_enable():
    cfg = DESearchConfig(
        gs_enable=False,
        de_hard_tail_templates=True,
        de_hard_tail_velocity_templates=True,
        order_candidates=(2,),
    )

    _terms, sources = build_de_library_terms_with_sources(cfg, order=2)

    assert "de_prior_hard_tail" in sources
    assert not any(str(source).startswith("gs_") for source in sources)


def test_gs_template_flag_no_longer_activates_hard_tail_for_real_de_config():
    cfg = DESearchConfig(
        gs_enable=True,
        gs_de_templates=True,
        de_hard_tail_templates=False,
        order_candidates=(2,),
    )

    _terms, sources = build_de_library_terms_with_sources(cfg, order=2)

    assert "de_prior_hard_tail" not in sources


def test_stagea_leaf_eval_failure_does_not_fabricate_zero_target():
    class BadLeaf:
        def __call__(self, _x):
            raise RuntimeError("boom")

    X = np.ones((16, 2), dtype=float)
    G = np.ones((16, 2), dtype=float)
    cfg = GeneralizedSymmetryConfig(enabled=True, mode="propose")

    proposals, diag = stageA_generalized_symmetry_proposals(
        atom=None,
        leaf=BadLeaf(),
        x_vals=X,
        dydx_vals=G,
        cols=(0, 1),
        cfg=cfg,
    )

    assert proposals == []
    assert diag and diag[0]["reason"] == "leaf_evaluation_failed"


def test_unit_torus_dedupes_projective_exponent_rays():
    assert projective_exponent_key((1, -1)) == projective_exponent_key((2, -2))
    exponents = enumerate_nullspace_exponents(
        [(1,), (1,)],
        max_exponent=3,
        max_l1=6,
        max_proposals=20,
        rational_denom=1,
    )
    keys = [projective_exponent_key(q) for q in exponents]
    assert len(keys) == len(set(keys))


def test_strict_ast_simplify_does_not_merge_fractional_power_domains():
    expr = Mul(Pow(Var(0), 0.5), Pow(Var(0), 0.5))
    simplified, stats = simplify_ast(
        expr,
        SimplifyOptions(enabled=True, domain_policy="strict"),
    )

    assert repr(simplified) == repr(expr)
    assert "mul_collect_powers" not in stats.rules_fired


def test_strict_qdag_does_not_merge_fractional_power_domains():
    expr = ("mul", ("sqrt", ("var", 0)), ("sqrt", ("var", 0)))
    cfg = ExpressionIRConfig(expr_ir="qdag", canonicalize="safe", domain_mode="strict")
    out = maybe_canonicalize_tuple_ast(expr, cfg)

    assert out != ("var", 0)
    assert "sqrt" in repr(out)
    assert canonical_key_tuple_ast(expr, cfg) == canonical_key_tuple_ast(out, cfg)


def test_strict_qdag_preserves_zero_times_singular_domain():
    expr = ("mul", ("const", 0.0), ("div", ("const", 1.0), ("var", 0)))
    cfg = ExpressionIRConfig(expr_ir="qdag", canonicalize="safe", domain_mode="strict")
    out = maybe_canonicalize_tuple_ast(expr, cfg)

    assert out != ("const", 0.0)
    assert "div" in repr(out)


def test_run_de_dirty_env_mode_off_records_provenance_without_enabling_gs(monkeypatch):
    import nestynet_sr.run_de as run_de

    monkeypatch.setenv("NESTYNET_GS_ENABLE", "1")
    monkeypatch.setattr(sys, "argv", ["run_de.py", "--filepath", "dummy.csv", "--gs-mode", "off"])
    args = run_de.parse_args()

    assert args.gs_enable is False
    assert any(row["name"] == "NESTYNET_GS_ENABLE" for row in args.gs_env_activation_provenance)


def test_run_de_legacy_gs_template_aliases_normalize_to_neutral_priors(monkeypatch):
    import nestynet_sr.run_de as run_de

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_de.py",
            "--filepath",
            "dummy.csv",
            "--gs-de-templates",
            "--gs-de-velocity-templates",
            "--gs-de-no-radial-templates",
        ],
    )
    args = run_de.parse_args()

    assert args.gs_enable is False
    assert args.gs_de_templates is False
    assert args.gs_de_velocity_templates is False
    assert args.de_hard_tail_templates is True
    assert args.de_hard_tail_velocity_templates is True
    assert args.de_hard_tail_radial_templates is False
    assert args.gs_legacy_alias_provenance


def test_run_de_rejects_lie_selection_without_active_scorer(monkeypatch):
    import nestynet_sr.run_de as run_de

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_de.py",
            "--filepath",
            "dummy.csv",
            "--gs-enable",
            "--gs-de-lie-use-for-selection",
        ],
    )

    with pytest.raises(SystemExit):
        run_de.parse_args()


def test_plan_gs_matched_ablation_marks_unverified_d_and_active_selection_e():
    from scripts.plan_gs_matched_ablation import build_plan

    args = SimpleNamespace(
        cmd="python examples/feynman_de/run_benchmark.py --engine sparse",
        output_root="/tmp/gs_plan_test",
        output_flag="--results_dir",
        neutral_library_args="--de-hard-tail-templates --de-hard-tail-velocity-templates",
        gs_audit_args="--gs-enable --gs-mode audit",
        verified_gs_args="--gs-enable --gs-mode propose",
        gs_selection_args="--gs-de-lie-prolongation --gs-de-lie-use-for-selection",
        oracle_args="",
        allow_collapsed_arms=False,
    )

    plan = build_plan(args)
    arms = {arm["id"]: arm for arm in plan["arms"]}

    assert arms["D"]["name"] == "gs_proposal_mode_unverified"
    assert arms["D"]["mechanisms"]["heldout_verified_proposals"] is False
    assert "--gs-de-lie-prolongation" in arms["E"]["command"]
    assert arms["E"]["mechanisms"]["lie_prolongation_selection"] is True
    assert len({arms[k]["mechanisms"]["mechanism_hash"] for k in ("C", "D", "E")}) == 3


def test_multi_dataset_prolongation_requires_common_generator(monkeypatch):
    fake = types.ModuleType("nestynet_sr.sr_gs.prolongation")

    def fake_score_de_lie_prolongation(*, cache, **_kwargs):
        if getattr(cache, "dataset", "") == "a":
            rows = [
                {"name": "translation", "metric_eligible": True, "on_shell_metric": 0.01, "accepted": True},
                {"name": "scaling", "metric_eligible": True, "on_shell_metric": 1.0, "accepted": False},
            ]
        else:
            rows = [
                {"name": "translation", "metric_eligible": True, "on_shell_metric": 1.0, "accepted": False},
                {"name": "scaling", "metric_eligible": True, "on_shell_metric": 0.01, "accepted": True},
            ]
        return {
            "enabled": True,
            "status": "scored",
            "best_metric": 0.01,
            "accepted_generator_names": [row["name"] for row in rows if row["accepted"]],
            "generators": rows,
        }

    fake.score_de_lie_prolongation = fake_score_de_lie_prolongation
    monkeypatch.setitem(sys.modules, "nestynet_sr.sr_gs.prolongation", fake)
    cfg = DESearchConfig(
        gs_enable=True,
        gs_de_lie_prolongation=True,
        gs_de_lie_use_for_selection=True,
        gs_de_lie_prolongation_weight=0.05,
    )
    X = torch.zeros((4, 1), dtype=torch.float64)
    caches = [SimpleNamespace(dataset="a"), SimpleNamespace(dataset="b")]

    meta, penalty = _score_de_lie_prolongation_multi(
        order=1,
        Xs=[X, X],
        caches=caches,
        term_asts=[],
        coeffs=torch.zeros((2, 0), dtype=torch.float64),
        cfg=cfg,
        dataset_ids=["a", "b"],
    )

    assert meta["best_metric"] == pytest.approx(0.505)
    assert meta["accepted_generator_names"] == []
    assert penalty == pytest.approx(0.05 * 0.505)


def test_multi_dataset_prolongation_selection_fails_closed_on_partial_failure(monkeypatch):
    fake = types.ModuleType("nestynet_sr.sr_gs.prolongation")

    def fake_score_de_lie_prolongation(*, cache, **_kwargs):
        if getattr(cache, "fail", False):
            raise RuntimeError("boom")
        return {
            "enabled": True,
            "status": "scored",
            "best_metric": 0.01,
            "accepted_generator_names": ["translation"],
            "generators": [
                {"name": "translation", "metric_eligible": True, "on_shell_metric": 0.01, "accepted": True}
            ],
        }

    fake.score_de_lie_prolongation = fake_score_de_lie_prolongation
    monkeypatch.setitem(sys.modules, "nestynet_sr.sr_gs.prolongation", fake)
    cfg = DESearchConfig(
        gs_enable=True,
        gs_de_lie_prolongation=True,
        gs_de_lie_use_for_selection=True,
        gs_de_lie_prolongation_weight=0.05,
    )
    X = torch.zeros((4, 1), dtype=torch.float64)

    meta, penalty = _score_de_lie_prolongation_multi(
        order=1,
        Xs=[X, X],
        caches=[SimpleNamespace(), SimpleNamespace(fail=True)],
        term_asts=[],
        coeffs=torch.zeros((2, 0), dtype=torch.float64),
        cfg=cfg,
        dataset_ids=["ok", "bad"],
    )

    assert meta["best_metric"] is None
    assert meta["num_failed_required_datasets"] == 1
    assert meta["selection_penalty_reason"] == "missing_or_nonfinite_metric_or_no_common_generator"
    assert penalty == pytest.approx(0.5)
