# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""GS-for-compound enablement, opt-outs, and ordinary acceptance authority."""

import sys

from nestynet_sr.run_SR import _STRUCTURAL_LANES, _buckingham_retry_trigger
from nestynet_sr.run_sr_args import parse_args


def _parse(extra=()):
    old = sys.argv
    try:
        sys.argv = ["run_SR.py", "--filepath", "dummy.csv", *extra]
        return parse_args()
    finally:
        sys.argv = old


def test_gs_compound_dials_default_on():
    args = _parse()
    assert args.gs_stagea is True
    assert args.gs_charts == "identity,log,reciprocal"
    assert args.gs_noise_calibrated_promotion is True
    assert args.gs_pairwise_composition is True
    assert args.gs_recursive_composition is True
    assert args.gs_recursive_max_depth == 3
    assert args.gs_recursive_beam_width == 2
    assert args.gs_stagea_proposal_budget == 6
    assert args.gs_decisive_min_confidence == 0.995
    assert args.gs_decisive_max_trials == 1
    # deliberately NOT default-on: broader-than-compound / units-side dials
    assert args.gs_general_affine is False
    assert args.gs_lorentz_boosts is False


def test_gs_opt_outs():
    args = _parse(
        [
            "--gs-no-stagea",
            "--gs-no-pairwise-composition",
            "--gs-no-recursive-composition",
            "--gs-no-noise-calibrated-promotion",
            "--gs-charts",
            "identity",
            "--gs-recursive-max-depth",
            "2",
            "--gs-recursive-beam-width",
            "1",
            "--gs-stagea-proposal-budget",
            "3",
            "--gs-decisive-max-trials",
            "0",
        ]
    )
    assert args.gs_stagea is False
    assert args.gs_pairwise_composition is False
    assert args.gs_recursive_composition is False
    assert args.gs_noise_calibrated_promotion is False
    assert args.gs_charts == "identity"
    assert args.gs_recursive_max_depth == 2
    assert args.gs_recursive_beam_width == 1
    assert args.gs_stagea_proposal_budget == 3
    assert args.gs_decisive_max_trials == 0


def test_gs_proposal_does_not_trigger_provisional_whole_run_retry():
    assert "gs_compound_lane" not in _STRUCTURAL_LANES
    rep = {
        "stageC": {"certified": False},
        "gate_telemetry": {"records": [
            {"rule": "gs_compound_lane", "gate": "proposed", "accepted": True},
        ]},
    }
    ok, reason, envs = _buckingham_retry_trigger(rep)
    assert not ok and reason == "lane_not_fired"
    assert envs == []
    rep["gate_telemetry"]["records"].append(
        {"rule": "visible_buckingham_lane", "accepted": True}
    )
    ok, _, envs = _buckingham_retry_trigger(rep)
    assert ok
    assert envs == ["NNSR_SUPPRESS_VISIBLE_BUCKINGHAM"]
