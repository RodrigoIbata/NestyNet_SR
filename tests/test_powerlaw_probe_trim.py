# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""power_product probe hardening (pb115): additive teacher pollution sinks the
full-set log-space R2 but the magnitude-trimmed refit recovers the true
power law without loosening the 0.98 gate."""

import numpy as np

from nestynet_sr.sr_search.stageB.rules import _powerlaw_logfit, _powerlaw_probe

RNG = np.random.default_rng(7)


def _make_data(n=2000):
    X = RNG.uniform(0.5, 3.0, size=(n, 2))
    F = 2.5 * X[:, 0] * X[:, 1] ** 2
    return X, F


def test_clean_power_law_passes_on_full_set():
    X, F = _make_data()
    probe = _powerlaw_probe(X, F)
    assert probe is not None and probe["passed"]
    assert probe["subset"] == "full"
    assert probe["r2_full"] > 0.999
    exps = probe["coeffs"][:2]
    assert abs(exps[0] - 1.0) < 0.02 and abs(exps[1] - 2.0) < 0.02


def test_additive_pollution_recovered_by_trim():
    """The pb115 mechanism in miniature: exact x0*x1^2 sub-leaf plus additive
    contamination from an imperfect upstream split. Full-set R2 drops below
    the gate; the top-70%-by-|F| refit passes it and recovers the exponents."""
    X, F = _make_data()
    F_polluted = np.abs(F + 0.6 * RNG.standard_normal(F.shape))
    F_polluted = np.clip(F_polluted, 1e-9, None)
    probe = _powerlaw_probe(X, F_polluted)
    assert probe is not None
    assert probe["r2_full"] < 0.98, f"pollution too weak: r2_full={probe['r2_full']}"
    assert probe["passed"], f"trim did not recover: r2_trim={probe['r2_trim']}"
    assert probe["subset"] != "full"
    assert probe["r2_trim"] >= 0.98
    exps = probe["coeffs"][:2]
    assert abs(exps[0] - 1.0) < 0.15 and abs(exps[1] - 2.0) < 0.15


def test_genuinely_non_powerlaw_still_rejected():
    """The gate must not be loosened: a non-power-law target fails both the
    full fit and the trimmed refit."""
    X, _ = _make_data()
    F = 1.0 + np.exp(0.8 * X[:, 0]) + np.sin(3.0 * X[:, 1]) ** 2
    probe = _powerlaw_probe(X, F)
    assert probe is not None
    assert not probe["passed"]
    assert probe["r2_full"] < 0.98
    assert probe["r2_trim"] is None or probe["r2_trim"] < 0.98


def test_logfit_degenerate_inputs():
    X = np.full((300, 2), 2.0)
    F = np.full(300, 8.0)
    # constant log_F -> ss_tot ~ 0 -> None (no spurious perfect fit)
    assert _powerlaw_logfit(np.log(X), np.log(F)) is None
