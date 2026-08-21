# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.
"""Fast guards on the calibration laboratory itself.

These are not the calibration campaign, which needs thousands of replicates and
runs from ``scripts/run_calibration_lab.py``.  They check that the laboratory
measures what it claims to: that the declared population risks are the risks the
generator actually produces, that the population front is computed from truth,
and that the headline qualitative findings reproduce at small scale.

The most important of these is `test_declared_risks_are_the_generated_risks`.
An earlier version of the generator silently violated it under the chi-square
construction, which presented as a catastrophic coverage failure rather than as
the setup error it was.
"""

import numpy as np
import pytest

from nestynet_sr.stat_selection.calibration_lab import (
    critical_value_strategies,
    cluster_calibration,
    dominance_power,
    draw_loss_matrix,
    make_lab_population,
    null_front_coverage,
    population_front,
)

# numpy on Apple Accelerate raises spurious matmul FP flags on finite operands.
pytestmark = pytest.mark.filterwarnings("ignore:.*encountered in matmul.*")

@pytest.mark.parametrize("distribution", ["gaussian", "chisq"])
def test_declared_risks_are_the_generated_risks(distribution):
    """The generator must reproduce the risks the population declares.

    If it does not, every downstream coverage number is measured against the
    wrong truth and a setup error masquerades as a method failure.
    """
    population = make_lab_population(seed=0)
    rng = np.random.default_rng(3)
    means = np.stack([
        draw_loss_matrix(population, 4000, rng, distribution=distribution).mean(axis=0)
        for _ in range(30)
    ]).mean(axis=0)
    assert np.allclose(means, population.risks, atol=0.02), (
        f"{distribution}: generated risks depart from declared risks"
    )

def test_chisq_construction_keeps_variance_below_risk():
    """The chi-square shift must stay real, which caps the marginal variance."""
    population = make_lab_population(seed=0)
    assert float(np.max(np.diag(population.cov))) < float(np.min(population.risks))

def test_population_front_is_computed_from_truth():
    """Dominated candidates are absent from the population front by construction."""
    population = make_lab_population(seed=0)
    front = set(population_front(population))
    dominated = {
        cid for cid, grp in zip(population.candidate_ids, population.group)
        if grp == "dominated"
    }
    assert front, "population front must be non-empty"
    assert front.isdisjoint(dominated)
    null_ids = {
        cid for cid, grp in zip(population.candidate_ids, population.group)
        if grp == "null"
    }
    assert null_ids <= front, "equal-risk equal-complexity candidates are non-dominated"

def test_generated_losses_are_finite_and_correlated():
    """Correlation is the structure a paired analysis exploits; assert it exists."""
    population = make_lab_population(rho=0.85, seed=0)
    losses = draw_loss_matrix(population, 3000, np.random.default_rng(1))
    assert np.isfinite(losses).all()
    corr = np.corrcoef(losses, rowvar=False)
    off_diagonal = corr[~np.eye(corr.shape[0], dtype=bool)]
    assert float(off_diagonal.mean()) > 0.5

def test_null_coverage_is_in_a_sane_range_at_small_scale():
    """A cheap sanity band, not a calibration measurement.

    The full campaign shows the familywise rate converging to nominal from above
    as units grow.  Here we only assert the procedure is neither wildly
    anti-conservative nor vacuously conservative.
    """
    population = make_lab_population(
        n_null=4, n_dominated=0, n_near=0, n_decoy=0, seed=0
    )
    result = null_front_coverage(
        population, n_units=60, n_replicates=60, n_resamples=200, seed=5
    )
    assert 0.0 <= result["familywise_false_edge_rate"] <= 0.35
    assert result["front_coverage"] >= 0.65
    assert result["population_front_size"] == population.n_candidates

def test_row_level_resampling_degrades_with_sampling_density():
    """Dense within-group sampling must not manufacture independent evidence."""
    rows = cluster_calibration(
        n_groups=10, samples_per_group=(1, 64), n_replicates=120, seed=4
    )
    sparse, dense = rows[0], rows[1]
    assert dense["group_level_coverage"] > 0.80
    assert dense["row_level_coverage"] < sparse["row_level_coverage"] - 0.20

def test_dominance_power_increases_with_the_risk_gap():
    """The procedure must be powerful, not merely conservative."""
    rows = dominance_power(
        gaps=(0.0, 1.0), n_units=60, n_replicates=120, n_resamples=200, seed=6
    )
    null_row, wide_row = rows[0], rows[1]
    assert null_row["removal_probability"] < 0.25
    assert wide_row["removal_probability"] > 0.75

def test_population_construction_rejects_bad_arguments():
    with pytest.raises(ValueError):
        make_lab_population(n_null=1)
    with pytest.raises(ValueError):
        make_lab_population(rho=1.0)
    with pytest.raises(ValueError):
        make_lab_population(noise_fraction=1.5)
    with pytest.raises(ValueError):
        draw_loss_matrix(make_lab_population(), 1, np.random.default_rng(0))
    with pytest.raises(ValueError):
        draw_loss_matrix(
            make_lab_population(), 10, np.random.default_rng(0), distribution="poisson"
        )


def test_critical_value_strategies_expose_the_small_unit_repair():
    """Changing the multiplier does not fix small-G anti-conservatism.

    The full campaign shows the Gaussian and Rademacher multipliers producing
    nearly identical critical values, with Rademacher slightly worse, while a
    Bonferroni-t bound restores validity at a power cost.  This asserts only the
    structural facts that make that comparison meaningful.
    """
    rows = critical_value_strategies(
        unit_grid=(30,), n_null=4, n_replicates=40, n_resamples=400, seed=2
    )
    row = rows[0]
    # The conservative bound must never be smaller than the bootstrap value,
    # otherwise the hybrid is not a fallback at all.
    assert row["hybrid_mean_critical_value"] >= row["multiplier_normal_mean_critical_value"]
    assert row["hybrid_false_edge_rate"] <= row["multiplier_normal_false_edge_rate"] + 1e-9
    for name in ("multiplier_normal", "multiplier_rademacher", "bonferroni_t", "hybrid"):
        assert row[f"{name}_mean_critical_value"] > 0.0
        assert 0.0 <= row[f"{name}_false_edge_rate"] <= 1.0
