# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import pytest
from types import SimpleNamespace

from nestynet_sr.sr_search.y_transforms import build_default_y_transforms
from nestynet_sr.sr_search.ysearch_ranker import (
    VirtualProbeHint,
    derive_joint_homogeneity_certificate,
    rank_virtual_hints,
    select_virtual_indices,
    select_virtual_portfolio,
)


def test_ranker_keeps_non_candidate_eligible():
    hints = [
        VirtualProbeHint(
            idx=1,
            name="square",
            domain_ok_frac=1.0,
            candidate_flag=False,
            sep_has_split=True,
            sep_proposals=3,
        ),
        VirtualProbeHint(
            idx=2,
            name="log",
            domain_ok_frac=1.0,
            candidate_flag=True,
            sep_has_split=False,
            sep_proposals=0,
        ),
    ]
    ranked = rank_virtual_hints(hints)
    assert ranked[0].name == "square"
    assert ranked[0].candidate_flag is False


def test_ranker_is_deterministic_on_ties():
    hints = [
        VirtualProbeHint(3, "sin", 0.99, True, False, 0),
        VirtualProbeHint(1, "cos", 0.99, True, False, 0),
        VirtualProbeHint(2, "cos", 0.99, True, False, 0),
    ]
    ranked = rank_virtual_hints(hints)
    assert [h.idx for h in ranked] == [1, 2, 3]


def test_select_virtual_indices_top_k():
    hints = [
        VirtualProbeHint(0, "a", 1.0, True, False, 0),
        VirtualProbeHint(1, "b", 1.0, True, True, 1),
        VirtualProbeHint(2, "c", 1.0, True, True, 2),
    ]
    selected = select_virtual_indices(hints, expand_k=2)
    assert selected == [2, 1]


def test_ranker_prefers_trig_strength_when_sep_ties():
    hints = [
        VirtualProbeHint(0, "square", 1.0, False, False, 0, trig_strength=2.0),
        VirtualProbeHint(1, "log", 1.0, False, False, 0, trig_strength=7.5),
    ]
    ranked = rank_virtual_hints(hints)
    assert ranked[0].name == "log"


def test_ranker_prefers_lower_virtual_mse_when_structure_ties():
    hints = [
        VirtualProbeHint(
            idx=0,
            name="square",
            domain_ok_frac=1.0,
            candidate_flag=False,
            sep_has_split=False,
            sep_proposals=0,
            trig_strength=0.0,
            virtual_mse=1.0e-2,
        ),
        VirtualProbeHint(
            idx=1,
            name="reciprocal",
            domain_ok_frac=1.0,
            candidate_flag=False,
            sep_has_split=False,
            sep_proposals=0,
            trig_strength=0.0,
            virtual_mse=1.0e-3,
        ),
    ]
    ranked = rank_virtual_hints(hints)
    assert ranked[0].name == "reciprocal"


def _pb001_hpc_hints():
    return [
        # The log prints trig strengths to three significant digits.  These
        # within-rounding-bin values preserve its observed tie order.
        VirtualProbeHint(1, "square", 1.0, True, False, 0, 35.11, 4.92e-17),
        VirtualProbeHint(2, "log", 1.0, True, False, 0, 29.5, 4.36e-12),
        VirtualProbeHint(4, "reciprocal", 1.0, True, False, 0, 65.0, 5.35e-8),
        VirtualProbeHint(5, "sqrt", 1.0, False, False, 0, 30.0, 1.29e-14),
        VirtualProbeHint(6, "sqrt1p", 1.0, True, False, 0, 35.11, 4.92e-17),
        VirtualProbeHint(7, "exp", 1.0, False, False, 0, 34.3, 1.54e-15),
        VirtualProbeHint(8, "expneg", 1.0, False, False, 0, 36.1, 1.14e-15),
        VirtualProbeHint(9, "arctan", 1.0, False, False, 0, 35.3, 1.29e-15),
        VirtualProbeHint(10, "arcsin", 1.0, False, False, 0, 34.7, 1.33e-15),
        VirtualProbeHint(11, "arccos", 1.0, False, False, 0, 34.7, 1.33e-15),
        VirtualProbeHint(12, "sin", 1.0, True, False, 0, 35.14, 1.30e-15),
        VirtualProbeHint(13, "cos", 1.0, True, False, 0, 35.15, 1.22e-17),
        VirtualProbeHint(14, "tan", 1.0, False, False, 0, 34.5, 1.34e-15),
    ]


def test_pb001_hpc_replay_places_square_sixth():
    """Replay the HPC pb001 ranking without depending on its copied log."""
    hints = _pb001_hpc_hints()
    ranked = rank_virtual_hints(hints)
    assert [hint.name for hint in ranked[:6]] == [
        "expneg",
        "arctan",
        "cos",
        "sin",
        "sqrt1p",
        "square",
    ]


def test_pb001_portfolio_reserves_one_joint_homogeneity_transform():
    hints = _pb001_hpc_hints()
    # Desired producer/consumer contract: a direct joint scaling check can
    # reserve this transform even when the soft ranker puts it below top-k.
    object.__setattr__(hints[0], "joint_homogeneity_verified", True)
    object.__setattr__(hints[2], "joint_homogeneity_verified", True)

    selected = select_virtual_indices(hints, expand_k=3, max_k=3)

    assert selected == [8, 9, 13, 1]
    assert len(selected) == 4  # top-k plus a bounded single reserve

    selected_with_reasons, reasons = select_virtual_portfolio(
        hints, expand_k=3, max_k=3
    )
    assert selected_with_reasons == selected
    assert reasons[1] == "joint_homogeneity_reserve"


def test_joint_reserve_is_not_added_when_top_k_already_has_one():
    hints = [
        VirtualProbeHint(0, "a", 1.0, True, True, 2, joint_homogeneity_verified=True),
        VirtualProbeHint(1, "b", 1.0, True, True, 1),
        VirtualProbeHint(2, "c", 1.0, True, False, 0, joint_homogeneity_verified=True),
    ]
    assert select_virtual_indices(hints, expand_k=2, max_k=2) == [0, 1]


def test_exact_transform_homogeneity_metadata_is_conservative():
    transforms = {
        transform.name: transform
        for transform in build_default_y_transforms(
            ["identity", "square", "sqrt", "reciprocal", "sqrt1p", "log"]
        )
    }
    assert transforms["identity"].homogeneity_power == 1.0
    assert transforms["square"].homogeneity_power == 2.0
    assert transforms["sqrt"].homogeneity_power == 0.5
    assert transforms["reciprocal"].homogeneity_power == -1.0
    assert transforms["sqrt1p"].homogeneity_power is None
    assert transforms["log"].homogeneity_power is None


def test_joint_certificate_transports_degree_through_square():
    base = SimpleNamespace(
        indices=[0, 1],
        oracle_verified=True,
        oracle_k=-1.0,
        oracle_rel_std=2.0e-4,
        n_points=1900,
    )
    cert = derive_joint_homogeneity_certificate([base], 2.0)
    assert cert is not None
    assert cert["indices"] == (0, 1)
    assert cert["degree"] == pytest.approx(-2.0)
    assert cert["rel_std"] == pytest.approx(2.0e-4)
    assert cert["n_points"] == 1900


def test_nonhomogeneous_output_map_cannot_derive_joint_certificate():
    base = SimpleNamespace(
        indices=[0, 1],
        oracle_verified=True,
        oracle_k=-1.0,
        oracle_rel_std=0.0,
        n_points=1000,
    )
    assert derive_joint_homogeneity_certificate([base], None) is None


def test_json_report_persists_virtual_portfolio_provenance(tmp_path):
    import json

    import torch

    from nestynet_sr.run_sr_reports import write_json_report

    records = [
        {
            "rank": 6,
            "idx": 1,
            "name": "square",
            "selected": True,
            "selection_reason": "joint_homogeneity_reserve",
            "joint_homogeneity_verified": True,
            "joint_homogeneity_indices": [0, 1],
            "joint_homogeneity_degree": -2.0,
            "joint_homogeneity_rel_std": 2.0e-4,
            "joint_homogeneity_n_points": 1900,
        }
    ]
    report_path = tmp_path / "pb001.report.json"
    write_json_report(
        filepath="pb001.csv",
        report_path=str(report_path),
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=0,
        walltime=0.0,
        stageA_data={"stageB_virtual_portfolio": records},
        enable_truth_eval=False,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["stageA"]["stageB_virtual_portfolio"] == records
