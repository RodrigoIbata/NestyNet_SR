# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "examples" / "kepler_ephemeris_real"
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

_MODULE_PATH = _EXAMPLE_DIR / "discover_third_body_residuals.py"
_SPEC = importlib.util.spec_from_file_location("_kepler_third_body_residuals_test_utils", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"unable to load {_MODULE_PATH}")
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)

CircularPerturber = _MOD.CircularPerturber
KeplerianPerturber = _MOD.KeplerianPerturber
ObservationBlock = _MOD.ObservationBlock
_phase_distance = _MOD._phase_distance
circular_source_positions = _MOD.circular_source_positions
circular_third_body_template = _MOD.circular_third_body_template
evaluate_circular_model = _MOD.evaluate_circular_model
evaluate_keplerian_model = _MOD.evaluate_keplerian_model
keplerian_source_positions = _MOD.keplerian_source_positions
refine_keplerian_perturber = _MOD.refine_keplerian_perturber
scan_circular_perturber = _MOD.scan_circular_perturber
split_observation_blocks = _MOD.split_observation_blocks
stack_observations = _MOD.stack_observations
third_body_template = _MOD.third_body_template


def _synthetic_asteroid_blocks(
    *,
    mu_sun: float,
    perturber_a: float,
    perturber_phase: float,
    perturber_mu: float,
) -> list:
    t_day = np.linspace(0.0, 2200.0, 480, dtype=np.float64)
    source = circular_source_positions(
        t_day,
        a_au=perturber_a,
        phase_rad=perturber_phase,
        mu_sun=mu_sun,
    )
    blocks = []
    for idx, (a_body, phase_body) in enumerate([(2.05, 0.2), (2.45, 1.1), (2.9, 2.4), (3.2, 4.2)]):
        mean_motion = math.sqrt(mu_sun / (a_body ** 3))
        theta = mean_motion * t_day + phase_body
        position = np.column_stack(
            [
                a_body * np.cos(theta),
                a_body * np.sin(theta),
                0.03 * a_body * np.sin(theta + 0.4 * idx),
            ]
        )
        residual = perturber_mu * third_body_template(position, source)
        blocks.append(
            ObservationBlock(
                orbit_id=f"synthetic_{idx}",
                body_name=f"Synthetic {idx}",
                split="candidate",
                t_day=t_day,
                position_au=position,
                residual_accel_au_per_d2=residual,
            )
        )
    return blocks


def test_circular_scan_recovers_hidden_third_body_template():
    mu_sun = 2.9591220819207774e-4
    true_a = 5.2
    true_phase = 11.0 * 2.0 * math.pi / 96.0
    true_mu = 9.55e-4 * mu_sun
    blocks = _synthetic_asteroid_blocks(
        mu_sun=mu_sun,
        perturber_a=true_a,
        perturber_phase=true_phase,
        perturber_mu=true_mu,
    )
    train_blocks, test_blocks = split_observation_blocks(blocks, holdout_fraction=0.25)
    train_obs = stack_observations(train_blocks)
    test_obs = stack_observations(test_blocks)

    a_grid = np.linspace(4.8, 5.6, 41, dtype=np.float64)
    phase_grid = np.linspace(0.0, 2.0 * math.pi, 96, endpoint=False, dtype=np.float64)
    best = scan_circular_perturber(
        train_obs,
        a_grid=a_grid,
        phase_grid=phase_grid,
        mu_sun=mu_sun,
    )
    fitted_train, _pred_train, train_metrics = evaluate_circular_model(train_obs, [best], mu_sun=mu_sun)
    _fitted_test, _pred_test, test_metrics = evaluate_circular_model(
        test_obs,
        fitted_train,
        mu_sun=mu_sun,
        refit_coefficients=False,
    )
    recovered = fitted_train[0]

    assert abs(float(recovered.a_au) - true_a) < 0.03
    assert _phase_distance(float(recovered.phase_wrapped), true_phase) < 0.04
    assert abs(float(recovered.mu_au3_per_d2) - true_mu) / true_mu < 0.02
    assert float(train_metrics["rel_rmse"]) < 2.0e-3
    assert float(test_metrics["rel_rmse"]) < 3.0e-3


def test_evaluate_circular_model_refits_shared_linear_mass():
    mu_sun = 2.9591220819207774e-4
    true_a = 5.2
    true_phase = 1.3
    true_mu = 9.55e-4 * mu_sun
    blocks = _synthetic_asteroid_blocks(
        mu_sun=mu_sun,
        perturber_a=true_a,
        perturber_phase=true_phase,
        perturber_mu=true_mu,
    )
    obs = stack_observations(blocks)

    initial = CircularPerturber(
        a_au=true_a,
        phase_rad=true_phase,
        mu_au3_per_d2=0.0,
        train_sse=float("inf"),
    )
    fitted, _pred, metrics = evaluate_circular_model(obs, [initial], mu_sun=mu_sun)

    assert abs(float(fitted[0].mu_au3_per_d2) - true_mu) / true_mu < 1.0e-12
    assert float(metrics["rel_rmse"]) < 1.0e-12
    assert int(metrics["k_params"]) == 3


def _synthetic_asteroid_blocks_from_source(
    *,
    t_day: np.ndarray,
    source_position: np.ndarray,
    perturber_mu: float,
    mu_sun: float,
) -> list:
    blocks = []
    for idx, (a_body, phase_body) in enumerate([(2.05, 0.2), (2.45, 1.1), (2.9, 2.4), (3.2, 4.2)]):
        mean_motion = math.sqrt(mu_sun / (a_body ** 3))
        theta = mean_motion * t_day + phase_body
        position = np.column_stack(
            [
                a_body * np.cos(theta),
                a_body * np.sin(theta),
                0.03 * a_body * np.sin(theta + 0.4 * idx),
            ]
        )
        residual = perturber_mu * third_body_template(position, source_position)
        blocks.append(
            ObservationBlock(
                orbit_id=f"synthetic_kepler_{idx}",
                body_name=f"Synthetic Kepler {idx}",
                split="candidate",
                t_day=t_day,
                position_au=position,
                residual_accel_au_per_d2=residual,
            )
        )
    return blocks


def test_keplerian_refinement_recovers_eccentric_inclined_source():
    pytest.importorskip("scipy.optimize")
    mu_sun = 2.9591220819207774e-4
    true_mu = 9.55e-4 * mu_sun
    t_day = np.linspace(0.0, 2600.0, 520, dtype=np.float64)
    source = keplerian_source_positions(
        t_day,
        a_au=5.2,
        eccentricity=0.07,
        inclination_rad=0.05,
        node_rad=0.4,
        arg_peri_rad=1.1,
        mean_anomaly0_rad=0.8,
        mu_sun=mu_sun,
    )
    blocks = _synthetic_asteroid_blocks_from_source(
        t_day=t_day,
        source_position=source,
        perturber_mu=true_mu,
        mu_sun=mu_sun,
    )
    train_blocks, test_blocks = split_observation_blocks(blocks, holdout_fraction=0.25)
    train_obs = stack_observations(train_blocks)
    test_obs = stack_observations(test_blocks)
    initial = KeplerianPerturber(
        a_au=5.15,
        eccentricity=0.04,
        inclination_rad=0.03,
        node_rad=0.35,
        arg_peri_rad=1.0,
        mean_anomaly0_rad=0.9,
        mu_au3_per_d2=0.0,
        train_sse=float("inf"),
    )

    refined = refine_keplerian_perturber(
        train_obs,
        initial,
        mu_sun=mu_sun,
        a_bounds=(4.8, 5.6),
        e_bounds=(0.0, 0.2),
        inclination_bounds=(0.0, 0.12),
        maxiter=90,
        maxfev=900,
    )
    fitted_train, _pred_train, train_metrics = evaluate_keplerian_model(train_obs, [refined], mu_sun=mu_sun)
    _fitted_test, _pred_test, test_metrics = evaluate_keplerian_model(
        test_obs,
        fitted_train,
        mu_sun=mu_sun,
        refit_coefficients=False,
    )
    recovered = fitted_train[0]

    assert abs(float(recovered.a_au) - 5.2) < 0.03
    assert abs(float(recovered.eccentricity) - 0.07) < 0.03
    assert abs(float(recovered.inclination_rad) - 0.05) < 0.03
    assert abs(float(recovered.mu_au3_per_d2) - true_mu) / true_mu < 0.03
    assert float(train_metrics["rel_rmse"]) < 1.0e-2
    assert float(test_metrics["rel_rmse"]) < 2.0e-2
