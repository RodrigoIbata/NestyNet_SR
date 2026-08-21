# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import unittest
from types import SimpleNamespace

import numpy as np

from nestynet_sr.sr_core.bridges import ConstNode
from nestynet_sr.sr_search.model_selection import (
    apply_noise_floor_to_acceptance_thresholds,
    clamp_threshold_to_noise_floor,
    compute_accept_threshold,
    estimate_transform_noise_floor_raw,
    loss_excess_above_floor,
    loss_within_floor_or_noise_equivalent,
    noisy_rel_rms_threshold,
    noise_equivalent,
    noise_equivalence_tolerance,
    resolve_acceptance_noise_floor_raw,
)


DUMMY_AST = ConstNode(1.0)


class AcceptanceNoiseFloorTests(unittest.TestCase):
    def _base_kwargs(self):
        return dict(
            base_loss=0.55,
            best_loss=0.52,
            base_ast=DUMMY_AST,
            cand_ast=DUMMY_AST,
            base_params=10,
            cand_params=8,
            loss_floor=0.1,
            loss_cap=10.0,
            count_weight=1.0,
            struct_gamma=0.0,
            param_gamma=0.0,
            base_bonus_decades=0.0,
            sep_bonus_decades=0.0,
            partial_sep_bonus_decades=0.0,
            is_separability=False,
            is_partial_separability=False,
            extra_bonus_decades=0.0,
            max_worsening_factor=None,
            worsening_floor=None,
            hard_ceiling=None,
        )

    def test_default_noise_floor_preserves_previous_behavior(self):
        kwargs = self._base_kwargs()
        thr = compute_accept_threshold(**kwargs)
        self.assertEqual(thr, compute_accept_threshold(**kwargs, noise_floor=None))
        self.assertEqual(thr, compute_accept_threshold(**kwargs, noise_floor=0.0))

    def test_noise_floor_acts_in_excess_loss_space(self):
        kwargs = self._base_kwargs()
        thr = compute_accept_threshold(**kwargs, noise_floor=0.5)
        self.assertAlmostEqual(thr, 0.6, places=12)

    def test_hard_ceiling_conversion_uses_full_raw_ceiling(self):
        global_best_raw = 2.0
        max_worsening_factor = 100.0
        noise_floor = 1.5
        global_ceil_raw = global_best_raw * max_worsening_factor
        hard_ceiling_excess = loss_excess_above_floor(global_ceil_raw, noise_floor)
        self.assertAlmostEqual(hard_ceiling_excess, 198.5, places=12)

    def test_hard_ceiling_applies_in_excess_space(self):
        kwargs = self._base_kwargs()
        kwargs.update(base_loss=5.0, best_loss=5.0, loss_cap=100.0)
        hard_ceiling_excess = loss_excess_above_floor(3.25, 1.0)
        kwargs["hard_ceiling"] = hard_ceiling_excess
        thr = compute_accept_threshold(**kwargs, noise_floor=1.0)
        self.assertAlmostEqual(thr, 3.25, places=12)

    def test_resolve_acceptance_noise_floor_raw_prefers_explicit_raw(self):
        lm_hp = SimpleNamespace(
            acceptance_noise_floor_raw=0.75,
            acceptance_noise_floor=0.2,
            loss_in_MAD_units=True,
        )
        self.assertAlmostEqual(resolve_acceptance_noise_floor_raw(lm_hp, loss_scale=9.0), 0.75, places=12)

    def test_resolve_acceptance_noise_floor_raw_scales_normalized_floor(self):
        lm_hp = SimpleNamespace(
            acceptance_noise_floor_raw=None,
            acceptance_noise_floor=0.2,
            loss_in_MAD_units=True,
        )
        self.assertAlmostEqual(resolve_acceptance_noise_floor_raw(lm_hp, loss_scale=9.0), 1.8, places=12)

    def test_clamp_threshold_to_noise_floor(self):
        self.assertAlmostEqual(
            clamp_threshold_to_noise_floor(0.25, 0.4, min_factor=3.0),
            1.2,
            places=12,
        )

    def test_apply_noise_floor_to_acceptance_thresholds(self):
        target, acceptable, candidate = apply_noise_floor_to_acceptance_thresholds(
            loss_target_raw=1.0e-4,
            loss_acceptable_raw=2.0e-3,
            accept_threshold_raw=5.0e-4,
            noise_floor_raw=1.0e-2,
        )
        self.assertAlmostEqual(target, 5.0e-3, places=12)
        self.assertAlmostEqual(acceptable, 3.0e-2, places=12)
        self.assertAlmostEqual(candidate, 2.0e-2, places=12)

    def test_noise_equivalent_uses_raw_noise_floor(self):
        self.assertTrue(noise_equivalent(5.825e-5, 5.894e-5, noise_floor=1.03e-5))
        self.assertFalse(noise_equivalent(1.0e-4, 1.5e-4, noise_floor=1.0e-5))
        self.assertAlmostEqual(
            noise_equivalence_tolerance(1.0e-4, 1.5e-4, noise_floor=1.0e-5),
            1.0e-5,
            places=12,
        )

    def test_noise_equivalent_uses_chisq_sampling_band_when_n_eff_supplied(self):
        noise_floor = 1.432e-8
        n_eff = 2000
        self.assertAlmostEqual(
            noise_equivalence_tolerance(1.0e-8, 1.1e-8, noise_floor=noise_floor, n_eff=n_eff),
            noise_floor * np.sqrt(2.0 / n_eff),
            places=18,
        )
        self.assertTrue(
            noise_equivalent(
                1.5224e-8,
                1.54e-8,
                noise_floor=noise_floor,
                n_eff=n_eff,
            )
        )
        self.assertFalse(
            noise_equivalent(
                1.5224e-8,
                1.7112e-8,
                noise_floor=noise_floor,
                n_eff=n_eff,
            )
        )

    def test_loss_within_floor_or_noise_equivalent_uses_boundary_band(self):
        noise_floor = 1.0e-4
        floor = 1.0e-3
        n_eff = 10000
        boundary = noise_floor + floor
        self.assertTrue(
            loss_within_floor_or_noise_equivalent(
                boundary + 5.0e-7,
                floor,
                noise_floor=noise_floor,
                n_eff=n_eff,
            )
        )
        self.assertFalse(
            loss_within_floor_or_noise_equivalent(
                boundary + 5.0e-6,
                floor,
                noise_floor=noise_floor,
                n_eff=n_eff,
            )
        )

    def test_noisy_rel_rms_threshold_preserves_noiseless_and_lifts_noisy(self):
        self.assertAlmostEqual(
            noisy_rel_rms_threshold(1.0e-3, noise_floor=0.0, y_rms=1.0),
            1.0e-3,
            places=12,
        )
        self.assertAlmostEqual(
            noisy_rel_rms_threshold(1.0e-3, noise_floor=1.0e-4, y_rms=2.0, noise_mult=2.0),
            1.0e-2,
            places=12,
        )

    def test_estimate_transform_noise_floor_identity_matches_sigma_squared(self):
        y = np.linspace(-3.0, 3.0, 4096, dtype=np.float64)
        sigma = 0.1
        est = estimate_transform_noise_floor_raw(
            y,
            None,
            sigma,
            n_mc=128,
            seed=7,
        )
        self.assertIsNotNone(est)
        self.assertAlmostEqual(float(est), sigma * sigma, delta=2.5e-3)


if __name__ == "__main__":
    unittest.main()
