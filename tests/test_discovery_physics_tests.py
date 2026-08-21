# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from nestynet_sr.discovery.physics_tests import (
    check_dimensional_consistency,
    check_parameter_stability,
    score_physics_consistency,
)
from nestynet_sr.sr_core.bridges import AddNode, MulNode, Var
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec


def _units_spec() -> UnitsSpec:
    us = UnitSystem(("L", "T"))
    return UnitsSpec(
        unit_system=us,
        x_dims=(us.dim((1, 0)), us.dim((0, 1))),
        y_dim=us.dim((1, 1)),
    )


def test_check_dimensional_consistency_passes_for_matching_product_units():
    spec = _units_spec()
    expr = MulNode(Var(0), Var(1))

    result = check_dimensional_consistency(expr, spec)

    assert result.passed is True
    assert result.score == 1.0


def test_check_dimensional_consistency_rejects_mismatched_addition():
    spec = _units_spec()
    expr = AddNode(Var(0), Var(1))

    result = check_dimensional_consistency(expr, spec)

    assert result.passed is False
    assert result.score == 0.0


def test_score_physics_consistency_combines_units_and_parameter_stability():
    spec = _units_spec()
    candidate = {
        "expr": ("mul", ("var", 0), ("var", 1)),
        "train_error": 0.05,
        "validation_error": 0.07,
        "metadata": {
            "parameter_samples": [
                {"k": 1.0, "b": 2.0},
                {"k": 1.1, "b": 2.1},
                {"k": 0.9, "b": 1.9},
            ]
        },
    }

    report = score_physics_consistency(candidate, units_spec=spec)
    stability = check_parameter_stability(candidate["metadata"]["parameter_samples"])

    assert report["passed"] is True
    assert report["overall_score"] > 0.5
    assert report["checks"]["dimensional_consistency"]["passed"] is True
    assert stability.passed is True


def test_check_parameter_stability_reports_per_parameter_cvs():
    result = check_parameter_stability(
        [
            {"k": 1.0},
            {"k": 2.0},
            {"k": 3.0},
        ],
        max_mean_cv=0.2,
    )

    assert result.passed is False
    assert result.details["parameter_sample_counts"]["k"] == 3
    assert result.details["parameter_cvs"]["k"] > 0.2
