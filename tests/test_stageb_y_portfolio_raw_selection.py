# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from types import SimpleNamespace

from nestynet_sr.run_SR import (
    _stageB_adjudication_key,
    _stageB_raw_y_branch_family_signature,
    _stageB_y_branch_artifact,
)
from nestynet_sr.sr_search.coe_committee import collect_final_candidates


class _TinyModel:
    def num_parameters(self):
        return 3


def _state(expr, labels, *, val_loss):
    return SimpleNamespace(
        num_nn_atoms=0,
        num_multivar_nn_atoms=0,
        max_nn_arity=0,
        model=_TinyModel(),
        val_loss=float(val_loss),
        acceptance_noise_floor_raw=1.76e-6,
        acceptance_noise_n_eff=2000,
        loss_acceptable_eff=1.0e-5,
        loss_good_enough_eff=1.76e-6,
        original_y_val_loss=float(val_loss),
        original_y_loss_acceptable_eff=1.0e-5,
        original_y_loss_good_enough_eff=1.76e-6,
        original_y_noise_floor_raw=1.76e-6,
        enabled_patterns=list(labels),
        sympy_meta={"accepted": True, "complexity_score": 10.0},
        y_expr_raw_str=str(expr),
        root=None,
    )


def test_raw_y_branch_classifier_protects_sparse_rational_and_flags_inverse_wrapper():
    rat = _stageB_raw_y_branch_family_signature(
        "(-10*x0*x1 - 15)/(5*x0*x1 - 6*sqrt(2)*sqrt(pi))",
        accepted_labels=("ratpoly_1d",),
        y_name="identity",
        explicit_simplified_expr=True,
    )
    wrapped = _stageB_raw_y_branch_family_signature(
        "tan(5*tanh(pi*x0*x1/8 + 7*exp(-1)/5)/3)",
        accepted_labels=("tanh",),
        y_name="arctan",
        explicit_simplified_expr=True,
    )

    assert rat["raw_protected_family"] is True
    assert rat["raw_generic_approximant"] is False
    assert wrapped["raw_protected_family"] is False
    assert wrapped["raw_generic_approximant"] is True


def test_raw_y_branch_classifier_does_not_flag_analytic_inverse_transform():
    analytic = _stageB_raw_y_branch_family_signature(
        "sqrt(2)*sqrt(exp(-x1**2/x0**2)/x0**2)/(2*sqrt(pi))",
        accepted_labels=("homogeneity_peel", "exp"),
        y_name="square",
        explicit_simplified_expr=True,
    )

    assert analytic["inverse_y_transform_wrapped"] is True
    assert analytic["raw_generic_approximant"] is False


def test_noisy_y_portfolio_prefers_raw_sparse_rational_inside_loss_tie():
    identity = _state(
        "(-10*x0*x1 - 15)/(5*x0*x1 - 6*sqrt(2)*sqrt(pi))",
        ("ratpoly_1d",),
        val_loss=1.773e-6,
    )
    arctan = _state(
        "tan(5*tanh(pi*x0*x1/8 + 7*exp(-1)/5)/3)",
        ("tanh",),
        val_loss=1.769e-6,
    )

    assert _stageB_adjudication_key(identity, y_name="identity", rank=0) < _stageB_adjudication_key(
        arctan,
        y_name="arctan",
        rank=1,
    )


def test_final_candidates_include_y_branch_artifacts_before_pareto_fill():
    candidates = collect_final_candidates(
        stageB_data={
            "y_branch_artifacts": [
                {
                    "expr": "(x0 + 1)/(x0 - 1)",
                    "source": "stageB_y_branch",
                    "label": "y_branch:identity",
                    "complexity": 5.0,
                    "metadata": {"raw_protected_family": True},
                }
            ],
            "sympy_meta": {},
        },
        final_polish_summary={
            "recommended": {"expr": "tan(tanh(x0))", "complexity": 10.0},
            "strict_pareto": [{"expr": f"x0 + {i}", "complexity": float(i)} for i in range(8)],
        },
        max_candidates=3,
        include_reservoir=False,
    )

    assert [c.source for c in candidates][:2] == [
        "final_polish:recommended",
        "stageB_y_branch",
    ]


def test_terminal_y_branch_artifact_allows_zero_nn_state():
    state = _state(
        "exp(-x0**2/2)/sqrt(2*pi)",
        ("exp",),
        val_loss=1.0e-6,
    )

    artifact = _stageB_y_branch_artifact(
        state,
        y_name="identity",
        rank=0,
        y_sources=["baseline"],
    )

    assert artifact is not None
    assert artifact["expr"] == "exp(-x0**2/2)/sqrt(2*pi)"
    assert artifact["metadata"]["branch_id"] == "identity"
