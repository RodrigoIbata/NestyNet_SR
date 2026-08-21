# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import pytest

from nestynet_sr.sr_search.truth_eval import evaluate_against_truth


def test_constant_discovered_expression_is_broadcast_to_samples():
    result = evaluate_against_truth(
        "0",
        "x0",
        {"x0": (1.0, 2.0)},
        n_samples=100,
    )

    assert result["success"] is True
    assert result["n_valid"] == 100
    assert result["rmse_abs"] > 0.0


def test_constant_truth_and_discovered_expressions_are_broadcast():
    result = evaluate_against_truth(
        "2",
        "2",
        {"x0": (1.0, 2.0)},
        n_samples=100,
    )

    assert result["success"] is True
    assert result["n_valid"] == 100
    assert result["rmse_abs"] == pytest.approx(0.0)


def test_constant_denominator_masks_are_broadcast():
    result = evaluate_against_truth(
        "x0/2",
        "x0/2",
        {"x0": (1.0, 2.0)},
        n_samples=100,
    )

    assert result["success"] is True
    assert result["n_valid"] == 100
    assert result["rmse_abs"] == pytest.approx(0.0)
