# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

from nestynet_sr.sr_de.de_ladder import (
    DELadderPolicy,
    LegacyDEResultPayloads,
    run_legacy_de_ladder,
)


def _payload(engine: str, equation: str, *, probe_rms: float) -> dict:
    return {
        "engine": engine,
        "kind": "library" if engine == "stlsq" else "factorized_blocks",
        "order": 1,
        "x_axis": 0,
        "canonical_equation": equation,
        "probe_rms": float(probe_rms),
        "probe_mse": float(probe_rms) ** 2,
        "num_terms": 1,
        "validation_candidate": {
            "order": 1,
            "x_axis": 0,
            "coefficients": [1.0],
            "term_asts_json": [{"type": "atom", "kind": "u", "kwargs": {}}],
        },
    }


def test_legacy_ladder_builds_slate_without_changing_off_selection():
    first_line = _payload("stlsq", "u_x + u = 0", probe_rms=1.0e-2)
    factorized = _payload("factorized", "u_x + x*u = 0", probe_rms=1.0e-4)

    report = run_legacy_de_ladder(
        LegacyDEResultPayloads(
            first_line=first_line,
            factorized=factorized,
            selected=first_line,
            selected_engine="stlsq",
        ),
        policy=DELadderPolicy(coe_mode="off"),
    )

    assert report.selected_engine == "stlsq"
    assert report.internal_selected_engine == "stlsq"
    assert report.committee_decision is None
    assert report.committee_adjudicated is False
    assert {row["engine"] for row in report.proposal_slate} == {"stlsq", "factorized"}


def test_legacy_ladder_adjudicate_can_select_committee_payload():
    first_line = _payload("stlsq", "u_x + u = 0", probe_rms=1.0e-2)
    factorized = _payload("factorized", "u_x + x*u = 0", probe_rms=1.0e-4)

    report = run_legacy_de_ladder(
        LegacyDEResultPayloads(
            first_line=first_line,
            factorized=factorized,
            selected=first_line,
            selected_engine="stlsq",
        ),
        policy=DELadderPolicy(coe_mode="adjudicate"),
    )

    assert report.selected_engine == "factorized"
    assert report.internal_selected_engine == "stlsq"
    assert report.selected_payload["canonical_equation"] == "u_x + x*u = 0"
    assert report.committee_adjudicated is True
    assert report.committee_adjudication_fallback is False
    assert report.committee_decision["config"]["mode"] == "adjudicate"


def test_legacy_ladder_reservoir_mode_reports_without_selecting_in_run_de():
    first_line = _payload("stlsq", "u_x + u = 0", probe_rms=1.0e-2)
    factorized = _payload("factorized", "u_x + x*u = 0", probe_rms=1.0e-4)

    report = run_legacy_de_ladder(
        LegacyDEResultPayloads(
            first_line=first_line,
            factorized=factorized,
            selected=first_line,
            selected_engine="stlsq",
        ),
        policy=DELadderPolicy(
            coe_mode="reservoir",
            reservoir_scouts_requested=2,
        ),
    )

    assert report.selected_engine == "stlsq"
    assert report.committee_adjudicated is False
    assert report.committee_decision["status"] == "selected"
    assert report.committee_decision["config"]["mode"] == "reservoir"
    assert report.committee_decision["config"]["reservoir_scouts_requested"] == 2
