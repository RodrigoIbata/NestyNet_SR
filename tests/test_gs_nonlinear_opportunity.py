# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import numpy as np
import pytest

from nestynet_sr.sr_gs.nonlinear_opportunity import (
    OpportunityAggregateConfig,
    OpportunityArmMetrics,
    OpportunityAttributionConfig,
    OPPORTUNITY_CASE_REGISTRY,
    aggregate_opportunity_results,
    evaluate_matched_opportunity,
    get_opportunity_case,
    list_opportunity_cases,
    sample_opportunity_case,
)


_VOCABULARY = ("I:projective_ratio", "I:quadratic_orbit")
_CERTIFICATES = {
    "determining": True,
    "off_shell": True,
    "bootstrap": True,
    "flow": True,
    "invariant": True,
    "support": True,
}


def _arm_a() -> OpportunityArmMetrics:
    return OpportunityArmMetrics(
        arm="A",
        determining_residual=2.0e-5,
        subspace_stability=0.72,
        false_generator_rate=0.08,
        simple_invariant_recovery=0.05,
        heldout_equation_gain=0.01,
    )


def _arm_b(*, vocabulary=_VOCABULARY) -> dict:
    return {
        "arm": "B",
        "carrier_vocabulary": list(vocabulary),
        "metrics": {
            "simple_invariant_recovery": 0.10,
            "arity_reduction": 0.0,
            "order_reduction": 0.0,
            "heldout_equation_gain": 0.02,
            "heldout_rollout_gain": 0.01,
        },
        "generator_discovered": False,
    }


def _arm_c(*, useful: bool = True, vocabulary=_VOCABULARY) -> dict:
    if useful:
        downstream = {
            "simple_invariant_recovery": 0.80,
            "arity_reduction": 1.0,
            "order_reduction": 0.0,
            "heldout_equation_gain": 0.08,
            "heldout_rollout_gain": 0.03,
        }
    else:
        downstream = {
            "simple_invariant_recovery": 0.10,
            "arity_reduction": 0.0,
            "order_reduction": 0.0,
            "heldout_equation_gain": 0.02,
            "heldout_rollout_gain": 0.01,
        }
    return {
        "arm": "C",
        "extra_carrier_vocabulary": list(vocabulary),
        "metrics": {
            "determining_residual": 2.0e-9,
            "subspace_stability": 0.98,
            "false_generator_rate": 0.01,
            **downstream,
        },
        "generator_discovered": True,
        "certificates": dict(_CERTIFICATES),
    }


def _positive_result(
    *,
    useful: bool = True,
    condition_id: str = "clean",
    noise_level: float = 0.0,
    support_fraction: float = 1.0,
):
    return evaluate_matched_opportunity(
        "free_particle_projective",
        _arm_a(),
        _arm_b(),
        _arm_c(useful=useful),
        condition_id=condition_id,
        noise_level=noise_level,
        support_fraction=support_fraction,
    )


def _negative_result(*, false_positive: bool = False):
    vocabulary = ("I:neutral_probe",)
    c = {
        "arm": "C",
        "extra_carrier_vocabulary": vocabulary,
        "false_generator_rate": 0.25 if false_positive else 0.0,
        "generator_discovered": bool(false_positive),
    }
    return evaluate_matched_opportunity(
        "generic_negative_control",
        OpportunityArmMetrics(arm="A"),
        {"arm": "B", "extra_carrier_vocabulary": vocabulary},
        c,
    )


def test_matched_opportunity_credits_certified_downstream_gain_and_serializes():
    result = _positive_result()

    assert result.credited
    assert result.status == "credited"
    assert result.vocabulary_matched
    assert result.certificate_gates_passed
    assert result.comparison_gates_passed
    assert result.deltas_vs_b["simple_invariant_recovery"] == pytest.approx(0.70)
    assert result.deltas_vs_b["arity_reduction"] == pytest.approx(1.0)
    assert result.deltas_vs_b["heldout_equation_gain"] == pytest.approx(0.06)

    report = result.to_report()
    assert report["arms"]["B"]["extra_carrier_vocabulary"] == sorted(_VOCABULARY)
    assert report["comparison_gates"]["reduction_gain_vs_b"]
    assert report["credited"] is True


def test_small_determining_residual_alone_gets_no_credit_over_matched_baseline():
    b = _arm_b()
    c = _arm_c(useful=False)
    b["metrics"]["simple_invariant_recovery"] = 0.60
    c["metrics"]["simple_invariant_recovery"] = 0.60
    result = evaluate_matched_opportunity(
        "free_particle_projective",
        _arm_a(),
        b,
        c,
    )

    assert not result.credited
    assert result.certificate_gates_passed
    assert not result.comparison_gates_passed
    assert "comparison_gate_failed:invariant_gain_vs_b" in result.reasons
    assert "comparison_gate_failed:reduction_gain_vs_b" in result.reasons
    assert "comparison_gate_failed:heldout_gain_vs_b" in result.reasons


def test_vocabulary_matching_is_exact_order_independent_and_mismatch_is_invalid():
    reordered = tuple(reversed(_VOCABULARY))
    valid = evaluate_matched_opportunity(
        "free_particle_projective",
        _arm_a(),
        _arm_b(vocabulary=_VOCABULARY),
        _arm_c(vocabulary=reordered),
    )
    assert valid.credited

    invalid = evaluate_matched_opportunity(
        "free_particle_projective",
        _arm_a(),
        _arm_b(vocabulary=_VOCABULARY),
        _arm_c(vocabulary=("I:quadratic_orbit", "I:unmatched")),
    )
    assert invalid.status == "invalid"
    assert not invalid.credited
    assert not invalid.vocabulary_matched
    assert invalid.reasons == ("unmatched_extra_carrier_vocabulary",)


def test_configured_pairwise_quality_margin_is_enforced_when_requested():
    config = OpportunityAttributionConfig(min_subspace_stability_gain_vs_b=0.05)
    b = _arm_b()
    b["metrics"]["subspace_stability"] = 0.97
    result = evaluate_matched_opportunity(
        "free_particle_projective",
        _arm_a(),
        b,
        _arm_c(),
        config=config,
    )

    assert not result.credited
    assert not result.comparison_gates["subspace_stability_gain_vs_b"]


def test_negative_control_measures_rejection_instead_of_awarding_credit():
    passed = _negative_result(false_positive=False)
    failed = _negative_result(false_positive=True)

    assert passed.status == "negative_control_passed"
    assert passed.negative_control_passed is True
    assert not passed.credited
    assert failed.status == "negative_control_failed"
    assert failed.negative_control_passed is False
    assert "negative_control_generator_not_rejected" in failed.reasons


def test_registry_samples_are_deterministic_support_aware_and_mark_pdes_deferred():
    assert {
        "free_particle_projective",
        "riccati_mobius",
        "generic_negative_control",
    } == {case.case_id for case in list_opportunity_cases()}
    assert "wave_conformal_deferred" in OPPORTUNITY_CASE_REGISTRY
    assert get_opportunity_case("wave_conformal_deferred").deferred

    clean_1 = sample_opportunity_case("free_particle_projective", n_samples=32, seed=17)
    clean_2 = sample_opportunity_case("free_particle_projective", n_samples=32, seed=17)
    restricted = sample_opportunity_case(
        "free_particle_projective",
        n_samples=32,
        seed=17,
        support_fraction=0.2,
    )
    noisy = sample_opportunity_case(
        "riccati_mobius",
        n_samples=32,
        seed=8,
        noise_std=1.0e-2,
    )

    for key in clean_1.on_shell:
        np.testing.assert_allclose(clean_1.on_shell[key], clean_2.on_shell[key])
    assert np.max(np.abs(restricted.on_shell["x"])) <= 0.4
    assert noisy.noise_std == pytest.approx(1.0e-2)
    assert noisy.n_on_shell == 32
    with pytest.raises(ValueError, match="deferred"):
        sample_opportunity_case("wave_conformal_deferred")


def test_noisy_and_restricted_support_aggregation_requires_matched_robustness():
    results = [
        _positive_result(condition_id="clean"),
        _positive_result(condition_id="noise_1e-2", noise_level=1.0e-2),
        _positive_result(condition_id="restricted_good", support_fraction=0.25),
        _positive_result(
            useful=False,
            condition_id="restricted_failure",
            support_fraction=0.15,
        ),
        _negative_result(false_positive=False),
    ]
    aggregate = aggregate_opportunity_results(
        results,
        config=OpportunityAggregateConfig(
            min_overall_credit_rate=0.75,
            min_clean_credit_rate=1.0,
            min_noisy_credit_rate=1.0,
            min_restricted_credit_rate=0.5,
            require_clean_condition=True,
            require_noisy_condition=True,
            require_restricted_condition=True,
            require_negative_control=True,
        ),
    )

    assert aggregate.robustly_attributed
    assert aggregate.n_positive_trials == 4
    assert aggregate.overall_credit_rate == pytest.approx(0.75)
    assert aggregate.clean_credit_rate == pytest.approx(1.0)
    assert aggregate.noisy_credit_rate == pytest.approx(1.0)
    assert aggregate.restricted_credit_rate == pytest.approx(0.5)
    assert aggregate.negative_control_pass_rate == pytest.approx(1.0)
    assert aggregate.to_report()["gates"]["restricted_credit_rate"]


def test_aggregation_fails_when_restricted_trials_do_not_generalize():
    aggregate = aggregate_opportunity_results(
        [
            _positive_result(),
            _positive_result(
                useful=False,
                condition_id="restricted",
                support_fraction=0.2,
            ),
        ],
        config=OpportunityAggregateConfig(
            min_overall_credit_rate=0.5,
            min_restricted_credit_rate=0.5,
            require_restricted_condition=True,
        ),
    )

    assert not aggregate.robustly_attributed
    assert aggregate.restricted_credit_rate == pytest.approx(0.0)
    assert "aggregate_gate_failed:restricted_credit_rate" in aggregate.reasons
