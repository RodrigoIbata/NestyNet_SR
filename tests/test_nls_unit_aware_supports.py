# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import math
from fractions import Fraction
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from nestynet_sr.sr_core.bridges import AtomNode, CosNode, Var
from nestynet_sr.sr_core.coefficient_units import monomial_dimension
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec, check_units_ast
from nestynet_sr.sr_search import candidate_builders
from nestynet_sr.sr_search import fitting_utils
from nestynet_sr.sr_search.config import LMHyperparams
from nestynet_sr.sr_search.fitting_utils import _nonlinear_substitution_screen
from nestynet_sr.sr_search.rational_supports import (
    plan_unit_consistent_rational_supports,
)
from nestynet_sr.sr_search.stageB import rules_compound


def _unit_system():
    unit_system = UnitSystem(("L", "T"))
    return (
        unit_system,
        unit_system.dimless(),
        unit_system.dim({"L": 1}),
    )


def _unitful_nls_data(n: int = 900):
    generator = torch.Generator().manual_seed(218)
    theta = torch.rand(n, generator=generator, dtype=torch.float64) * (2 * math.pi) - math.pi
    length = torch.rand(n, generator=generator, dtype=torch.float64) + 0.5
    wave = torch.cos(theta)
    output = length * (1.0 + wave) / (2.0 + wave)
    return torch.stack((theta, length), dim=1), output


def _reciprocal_klein_nishina_leaf_data(n: int = 3000):
    generator = torch.Generator().manual_seed(119)
    ratio = torch.rand(n, generator=generator, dtype=torch.float64) * 4.8 + 0.2
    theta = torch.rand(n, generator=generator, dtype=torch.float64) * (2 * math.pi)
    output = ratio + ratio**3 - ratio.square() * torch.sin(theta).square()
    return torch.stack((ratio, theta), dim=1), output


def test_nls_proposal_contract_has_explicit_reproducible_defaults():
    config = LMHyperparams()

    assert config.stageB_nls_requested_count == 3
    assert config.stageB_nls_max_attempts == 1024
    assert config.stageB_nls_max_support_attempts == 2048


def test_nls_extra_degree_pair_does_not_open_a_degree_rectangle(monkeypatch):
    attempted = set()

    def fake_probe(*_args, **kwargs):
        attempted.add((int(kwargs["deg_num"]), int(kwargs["deg_den"])))
        return float("inf")

    monkeypatch.setattr(fitting_utils, "_rational_probe_nd", fake_probe)
    X = torch.linspace(0.2, 1.0, 220, dtype=torch.float64).unsqueeze(1)

    _nonlinear_substitution_screen(
        X,
        X[:, 0].square(),
        max_deg_num=3,
        max_deg_den=3,
        extra_degree_pairs=[(4, 1)],
        min_points=200,
    )

    assert (4, 1) in attempted
    assert (4, 2) not in attempted
    assert (4, 3) not in attempted


def test_support_planner_uses_one_path_for_unitless_and_unitful_dimensions():
    _units, zero, length = _unit_system()

    unitless = plan_unit_consistent_rational_supports(
        target_dim=zero,
        input_dims=(zero, zero),
        max_deg_num=2,
        max_deg_den=2,
    )
    unitful = plan_unit_consistent_rational_supports(
        target_dim=length,
        input_dims=(zero, length),
        max_deg_num=2,
        max_deg_den=2,
    )

    assert unitless.supports
    assert unitful.supports
    assert unitless.raw_attempted > 0
    assert unitful.raw_attempted > 0
    assert all(item.certificate.ok for item in unitless.supports)
    assert all(item.certificate.ok for item in unitful.supports)

    for support in unitful.supports:
        num_dims = {
            monomial_dimension(row, (zero, length))
            for row in support.numerator_exponents
        }
        den_dims = {
            monomial_dimension(row, (zero, length))
            for row in support.denominator_exponents
        }
        assert len(num_dims) == 1
        assert len(den_dims) == 1
        assert tuple(a - b for a, b in zip(next(iter(num_dims)), next(iter(den_dims)))) == length


def test_support_planner_cannot_invent_anonymous_unitful_coefficient():
    _units, zero, length = _unit_system()

    plan = plan_unit_consistent_rational_supports(
        target_dim=length,
        input_dims=(zero, zero),
        max_deg_num=3,
        max_deg_den=3,
    )

    assert plan.supports == ()
    assert plan.diagnostics()["exhaustion_reason"] == "candidate_space_exhausted"


def test_support_planner_reports_attempt_budget_exhaustion():
    _units, zero, _length = _unit_system()

    plan = plan_unit_consistent_rational_supports(
        target_dim=zero,
        input_dims=(zero,),
        max_deg_num=2,
        max_deg_den=2,
        max_attempts=0,
    )

    diagnostics = plan.diagnostics()
    assert plan.supports == ()
    assert diagnostics["raw_attempted"] == 0
    assert diagnostics["truncated_by_attempt_budget"] is True
    assert diagnostics["exhaustion_reason"] == "attempt_budget_exhausted"


def test_unit_aware_screen_constructs_valid_supports_before_ranking():
    _units, zero, length = _unit_system()
    X, output = _unitful_nls_data()
    diagnostics = {}

    hits = _nonlinear_substitution_screen(
        X,
        output,
        teacher=None,
        threshold=0.03,
        max_points=900,
        min_points=200,
        trig_hints={0: "cos"},
        target_dim=length,
        input_dims=(zero, length),
        max_attempts=256,
        diagnostics=diagnostics,
    )

    assert hits
    assert hits[0]["transform"] == "cos"
    assert hits[0]["col_idx"] == 0
    assert hits[0]["unit_support_rank"] == 0
    assert diagnostics["unit_aware"] is True
    assert diagnostics["unit_rejected"] >= 2
    assert diagnostics["reason_counts"]["non_dimensionless_transform_argument"] >= 2
    for hit in hits:
        assert hit["col_idx"] == 0
        assert hit["coefficient_unit_certificate"]["valid"] is True
        assert hit["exps_num_override"]
        assert hit["exps_den_override"]


def test_unit_aware_screen_reports_numeric_attempt_exhaustion():
    _units, zero, length = _unit_system()
    X, output = _unitful_nls_data(500)
    diagnostics = {}

    _nonlinear_substitution_screen(
        X,
        output,
        threshold=0.03,
        max_points=500,
        min_points=200,
        trig_hints={0: "cos"},
        target_dim=length,
        input_dims=(zero, length),
        max_attempts=1,
        diagnostics=diagnostics,
    )

    assert diagnostics["raw_attempted"] == 1
    assert diagnostics["numeric_attempt_budget_exhausted"] is True
    assert diagnostics["truncated_by_attempt_budget"] is True
    assert diagnostics["exhaustion_reason"] == "attempt_budget_exhausted"


def test_unit_aware_screen_applies_one_global_structural_attempt_budget():
    _units, zero, length = _unit_system()
    X, output = _unitful_nls_data(500)
    diagnostics = {}

    _nonlinear_substitution_screen(
        X,
        output,
        threshold=0.03,
        max_points=500,
        min_points=200,
        trig_hints={0: "cos"},
        outer_transforms=["square", "reciprocal"],
        target_dim=length,
        input_dims=(zero, length),
        max_attempts=32,
        max_support_attempts=1,
        diagnostics=diagnostics,
    )

    assert diagnostics["support_raw_attempted"] == 1
    assert diagnostics["support_raw_attempted"] <= diagnostics["max_support_attempts"]
    assert diagnostics["support_attempt_budget_exhausted"] is True
    assert diagnostics["truncated_by_attempt_budget"] is True
    assert diagnostics["reason_counts"]["support_attempt_budget_exhausted"] == 1


def test_dimensionless_screen_uses_the_same_exact_support_path():
    unit_system = UnitSystem(("L",))
    zero = unit_system.dimless()
    generator = torch.Generator().manual_seed(19)
    theta = torch.rand(700, generator=generator, dtype=torch.float64) * (2 * math.pi) - math.pi
    x = torch.rand(700, generator=generator, dtype=torch.float64) * 1.6 - 0.8
    output = (1.0 - x.square()) / (1.0 + x * torch.cos(theta))
    diagnostics = {}

    hits = _nonlinear_substitution_screen(
        torch.stack((theta, x), dim=1),
        output,
        threshold=0.05,
        max_points=700,
        min_points=200,
        trig_hints={0: "cos"},
        target_dim=zero,
        input_dims=(zero, zero),
        max_attempts=256,
        diagnostics=diagnostics,
    )

    assert hits
    assert diagnostics["unit_aware"] is True
    assert hits[0]["unit_support_planned"] is True
    assert hits[0]["coefficient_unit_certificate"]["valid"] is True


def test_outer_square_screen_solves_in_doubled_target_dimension():
    unit_system = UnitSystem(("L",))
    zero = unit_system.dimless()
    length = unit_system.dim({"L": 1})
    sqrt_length = unit_system.dim({"L": Fraction(1, 2)})
    generator = torch.Generator().manual_seed(31)
    theta = torch.rand(900, generator=generator, dtype=torch.float64) * 6.0 - 3.0
    x = torch.rand(900, generator=generator, dtype=torch.float64) + 0.5
    wave = torch.cos(theta)
    output = torch.sqrt(x * (2.0 + wave) / (3.0 + wave))

    hits = _nonlinear_substitution_screen(
        torch.stack((theta, x), dim=1),
        output,
        threshold=0.03,
        max_points=900,
        min_points=200,
        trig_hints={0: "cos"},
        outer_transforms=["square"],
        target_dim=sqrt_length,
        input_dims=(zero, length),
        max_attempts=512,
    )

    square_hits = [hit for hit in hits if hit.get("outer_transform") == "square"]
    assert square_hits
    assert square_hits[0]["rational_target_dim"] == length
    assert square_hits[0]["coefficient_unit_certificate"]["target_dim"] == ["1"]


def test_ast_checker_uses_exact_overrides_and_rejects_mixed_unit_support():
    _units, zero, length = _unit_system()
    spec = UnitsSpec(
        unit_system=_units,
        x_dims=(zero, length),
        y_dim=length,
    )
    inputs = (CosNode(Var(0)), Var(1))
    valid = AtomNode(
        kind="ratpoly",
        var_idxs=(0, 1),
        kwargs={
            "deg_num": 2,
            "deg_den": 1,
            "exps_num_override": [[0, 1], [1, 1]],
            "exps_den_override": [[0, 0], [1, 0]],
        },
        inputs=inputs,
    )
    invalid = AtomNode(
        kind="ratpoly",
        var_idxs=(0, 1),
        kwargs={
            "deg_num": 1,
            "deg_den": 1,
            "exps_num_override": [[0, 0], [0, 1]],
            "exps_den_override": [[0, 0], [1, 0]],
        },
        inputs=inputs,
    )

    assert check_units_ast(valid, spec).ok is True
    invalid_result = check_units_ast(invalid, spec)
    assert invalid_result.ok is False
    assert "exact coefficient support is inconsistent" in invalid_result.reason


def test_candidate_builder_refits_only_the_planned_support():
    _units, zero, length = _unit_system()
    X, output = _unitful_nls_data(700)
    hit = _nonlinear_substitution_screen(
        X,
        output,
        threshold=0.03,
        max_points=700,
        min_points=200,
        trig_hints={0: "cos"},
        target_dim=length,
        input_dims=(zero, length),
        max_attempts=256,
    )[0]

    class Teacher(torch.nn.Module):
        def forward(self, values):
            wave = torch.cos(values[:, 0])
            return (values[:, 1] * (1.0 + wave) / (2.0 + wave)).unsqueeze(1)

    target = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="nls_units")
    result = candidate_builders._build_nonlinear_sub_candidate(
        root=target,
        target=target,
        reuse={"nls_units": Teacher()},
        train_loader=DataLoader(
            TensorDataset(X, output.unsqueeze(1)), batch_size=len(X), shuffle=False
        ),
        device=torch.device("cpu"),
        dtype=torch.float64,
        hit=hit,
    )

    assert result is not None
    root, _init, metadata = result
    assert metadata["unit_support_planned"] is True
    certificate = metadata["coefficient_unit_certificate"]
    assert certificate["valid"] is True
    assert certificate["target_dim"] == ["1", "0"]
    assert metadata["n_terms_num"] <= len(hit["exps_num_override"])
    assert metadata["n_terms_den"] <= len(hit["exps_den_override"])
    assert check_units_ast(
        root,
        UnitsSpec(unit_system=_units, x_dims=(zero, length), y_dim=length),
    ).ok


def test_candidate_builder_fails_closed_on_tampered_planned_support():
    _units, zero, _length = _unit_system()
    target = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="nls_tamper")
    hit = {
        "col_idx": 0,
        "transform": "cos",
        "deg_num": 1,
        "deg_den": 1,
        "error": 0.0,
        "unit_support_planned": True,
        "coefficient_unit_certificate": {"valid": True},
        "coefficient_policy": "free_const_only",
        "rational_target_dim": zero,
        "transformed_input_dims": (zero, zero),
        "exps_num_override": [[0.5, 0]],
        "exps_den_override": [[0, 0]],
    }

    assert candidate_builders._build_nonlinear_sub_candidate(
        root=target,
        target=target,
        reuse={"nls_tamper": torch.nn.Identity()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        hit=hit,
    ) is None


def test_rule_emits_requested_real_unitful_candidates():
    units, zero, length = _unit_system()
    X, output = _unitful_nls_data(700)

    class Teacher(torch.nn.Module):
        def forward(self, values):
            wave = torch.cos(values[:, 0])
            return (values[:, 1] * (1.0 + wave) / (2.0 + wave)).unsqueeze(1)

    target = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="nls_real")
    ctx = SimpleNamespace(
        state=SimpleNamespace(root=target, reuse={"nls_real": Teacher()}),
        train_loader_probe=DataLoader(
            TensorDataset(X, output.unsqueeze(1)), batch_size=len(X), shuffle=False
        ),
        device=torch.device("cpu"),
        dtype=torch.float64,
        trig_by_axis={0: SimpleNamespace(phase=0.0)},
        lm_hp=SimpleNamespace(
            stageB_nls_requested_count=3,
            stageB_nls_max_attempts=256,
        ),
        enforce_units=True,
        units_spec=UnitsSpec(
            unit_system=units,
            x_dims=(zero, length),
            y_dim=length,
        ),
        infer_target_dim=lambda _target: length,
        _cache={},
        log=lambda _message: None,
    )

    candidates = rules_compound.RuleNonlinearSubstitution().propose(ctx, target)

    assert len(candidates) == 3
    assert len({candidate.signature for candidate in candidates}) == 3
    for candidate in candidates:
        assert candidate.meta["coefficient_unit_certificate"]["valid"] is True
        assert candidate.meta["unit_admissibility"]["valid"] is True
        assert check_units_ast(candidate.root, ctx.units_spec).ok is True
    budget = candidates[0].meta["proposal_budget"]
    assert budget["requested_count"] == 3
    assert budget["emitted"] == 3
    assert budget["exhausted"] is False


def test_rule_closes_reciprocal_klein_nishina_leaf_with_degree_four_numerator():
    units, zero, _length = _unit_system()
    X, output = _reciprocal_klein_nishina_leaf_data()

    degree_three_hits = _nonlinear_substitution_screen(
        X,
        output,
        threshold=0.02,
        max_points=len(X),
        min_points=200,
        trig_hints={1: "sin"},
        max_deg_num=3,
        max_deg_den=3,
        outer_transforms=["square", "reciprocal"],
        target_dim=zero,
        input_dims=(zero, zero),
        max_attempts=1024,
        max_support_attempts=2048,
    )
    assert min(hit["error"] for hit in degree_three_hits) > 1.0e-4

    special_hits = _nonlinear_substitution_screen(
        X,
        output,
        threshold=0.02,
        max_points=len(X),
        min_points=200,
        trig_hints={1: "sin"},
        max_deg_num=3,
        max_deg_den=3,
        extra_degree_pairs=[(4, 1)],
        outer_transforms=["square", "reciprocal"],
        target_dim=zero,
        input_dims=(zero, zero),
        max_attempts=1024,
        max_support_attempts=2048,
    )
    assert any(
        hit["deg_num"] == 4
        and hit["deg_den"] == 1
        and hit["error"] < 1.0e-8
        for hit in special_hits
    )
    assert all(
        hit["deg_num"] <= 3
        or (hit["deg_num"], hit["deg_den"]) == (4, 1)
        for hit in special_hits
    )

    class Teacher(torch.nn.Module):
        def forward(self, values):
            ratio = values[:, 0]
            theta = values[:, 1]
            return (
                ratio
                + ratio**3
                - ratio.square() * torch.sin(theta).square()
            ).unsqueeze(1)

    target = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="kn_reciprocal")
    ctx = SimpleNamespace(
        state=SimpleNamespace(root=target, reuse={"kn_reciprocal": Teacher()}),
        train_loader_probe=DataLoader(
            TensorDataset(X, output.unsqueeze(1)), batch_size=len(X), shuffle=False
        ),
        device=torch.device("cpu"),
        dtype=torch.float64,
        trig_by_axis={1: SimpleNamespace(phase=math.pi / 2)},
        lm_hp=SimpleNamespace(
            stageB_nls_requested_count=3,
            stageB_nls_max_attempts=1024,
            stageB_nls_max_support_attempts=2048,
        ),
        enforce_units=True,
        units_spec=UnitsSpec(
            unit_system=units,
            x_dims=(zero, zero),
            y_dim=zero,
        ),
        infer_target_dim=lambda _target: zero,
        _cache={},
        log=lambda _message: None,
    )

    candidates = rules_compound.RuleNonlinearSubstitution().propose(ctx, target)

    assert len(candidates) == 3
    exact = [
        candidate
        for candidate in candidates
        if candidate.meta["deg_num"] == 4
        and candidate.meta["precheck_rel_rms"] < 1.0e-8
    ]
    assert exact, [
        {
            "label": candidate.label,
            "deg_num": candidate.meta["deg_num"],
            "deg_den": candidate.meta["deg_den"],
            "error": candidate.meta["precheck_rel_rms"],
        }
        for candidate in candidates
    ]
    assert all(candidate.meta["deg_den"] == 0 for candidate in exact)
    assert all(
        candidate.meta["exps_den_override"] == [[0, 0]]
        for candidate in exact
    )
    budget = candidates[0].meta["proposal_budget"]
    assert budget["raw_attempted"] < budget["max_attempts"]
    assert budget["support_raw_attempted"] < budget["max_support_attempts"]
    assert budget["truncated_by_attempt_budget"] is False
    assert budget["screen"]["extra_degree_pairs"] == [[4, 1]]


def test_rule_counts_built_admissible_emissions_not_prefilter_hits(monkeypatch):
    target = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="nls_count")
    X = torch.ones((220, 2), dtype=torch.float64)
    output = torch.ones(220, dtype=torch.float64)

    monkeypatch.setattr(
        candidate_builders,
        "_gather_atom_teacher_data",
        lambda **_kwargs: (X, output),
    )
    monkeypatch.setattr(
        rules_compound,
        "_effective_input_dims_for_atom",
        lambda _target, _spec: [(Fraction(0),), (Fraction(0),)],
    )

    hits = [
        {
            "col_idx": 0,
            "transform": "cos",
            "parity": "even",
            "deg_num": index + 1,
            "deg_den": 1,
            "error": 1.0e-6 * (index + 1),
            "unit_support_planned": True,
            "exps_num_override": [[index + 1, 0]],
            "exps_den_override": [[index, 0]],
        }
        for index in range(5)
    ]

    def fake_screen(*_args, **kwargs):
        kwargs["diagnostics"].update(
            {
                "raw_attempted": 9,
                "unit_rejected": 4,
                "numeric_rejected": 2,
                "deduplicated": 1,
                "emitted": len(hits),
                "truncated_by_attempt_budget": False,
                "exhaustion_reason": "candidate_space_exhausted",
            }
        )
        return hits

    monkeypatch.setattr(
        "nestynet_sr.sr_search.fitting_utils._nonlinear_substitution_screen",
        fake_screen,
    )

    build_calls = []

    def fake_builder(**kwargs):
        hit = kwargs["hit"]
        build_calls.append(hit["deg_num"])
        if hit["deg_num"] == 1:
            return None
        final_power = 1 if hit["deg_num"] in (2, 3) else hit["deg_num"] - 2
        return (
            AtomNode(kind="var", var_idxs=(0,), kwargs={}),
            None,
            {
                "transform": "cos",
                "col_idx": 0,
                "deg_num": final_power,
                "deg_den": 0,
                "leaf_kind": "ratpoly",
                "exps_num_override": [[final_power, 0]],
                "exps_den_override": [[0, 0]],
                "coefficient_unit_certificate": {
                    "checked": True,
                    "valid": True,
                }
            },
        )

    monkeypatch.setattr(
        candidate_builders,
        "_build_nonlinear_sub_candidate",
        fake_builder,
    )
    monkeypatch.setattr(
        "nestynet_sr.sr_core.units.check_units_ast",
        lambda *_args, **_kwargs: SimpleNamespace(ok=True, reason=""),
    )

    logs = []
    ctx = SimpleNamespace(
        state=SimpleNamespace(root=target, reuse={"nls_count": torch.nn.Identity()}),
        train_loader_probe=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        trig_by_axis={},
        lm_hp=SimpleNamespace(stageB_nls_requested_count=3),
        enforce_units=True,
        units_spec=SimpleNamespace(policy="free_const_only"),
        infer_target_dim=lambda _target: (Fraction(0),),
        _cache={},
        log=logs.append,
    )

    candidates = rules_compound.RuleNonlinearSubstitution().propose(ctx, target)

    assert len(candidates) == 3
    assert len({candidate.signature for candidate in candidates}) == 3
    assert build_calls == [1, 2, 3, 4, 5]
    budget = candidates[0].meta["proposal_budget"]
    assert budget["requested_count"] == 3
    assert budget["candidate_build_attempted"] == 5
    assert budget["build_rejected"] == 1
    assert budget["candidate_deduplicated"] == 1
    assert budget["deduplicated"] == 2
    assert budget["emitted"] == 3
    assert budget["exhausted"] is False
    assert budget["exhaustion_reason"] is None
