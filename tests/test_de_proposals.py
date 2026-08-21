# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import json

from nestynet_sr.sr_de.proposals import (
    build_proposal_slate,
    canonicalize_de_equation,
    merge_proposal_slates,
    proposal_from_factorized_result,
    proposal_from_factorized_search_result,
    proposal_from_stlsq_result,
)


def test_canonicalize_de_equation_normalizes_sign_scale_and_explicit_rhs():
    key = canonicalize_de_equation("u_x + sin(u) = 0")

    assert canonicalize_de_equation("-u_x - sin(u) = 0") == key
    assert canonicalize_de_equation("2*u_x + 2*sin(u) = 0") == key
    assert canonicalize_de_equation("u_x = -sin(u)") == key
    assert key == "residual:du+sin(u)"


def test_build_proposal_slate_dedupes_equivalent_sign_scaled_candidates():
    first_line = {
        "engine": "stlsq",
        "kind": "library",
        "order": 1,
        "x_axis": 0,
        "canonical_equation": "u_x + sin(u) = 0",
        "num_terms": 2,
        "probe_rms": 1.0e-4,
    }
    selected = {
        "engine": "stlsq",
        "kind": "library",
        "order": 1,
        "x_axis": 0,
        "canonical_equation": "-2*u_x - 2*sin(u) = 0",
        "num_terms": 2,
        "probe_rms": 1.0e-4,
    }

    slate = build_proposal_slate(
        first_line=first_line,
        selected=selected,
        selected_engine="stlsq",
    )

    assert len(slate) == 1
    proposal = slate[0]
    assert proposal["canonical_key"] == "residual:du+sin(u)"
    assert proposal["support"]["sources"] == ["first_line", "selected"]
    assert proposal["support"]["selected"] is True


def test_proposal_constructors_preserve_compile_payloads_and_round_trip_json():
    stlsq_payload = {
        "engine": "stlsq",
        "kind": "library",
        "order": 1,
        "x_axis": 0,
        "canonical_equation": "u_x + u = 0",
        "terms": ["u"],
        "coefficients": [1.0],
        "num_terms": 1,
        "rms_val": 1.0e-5,
        "validation_candidate": {"order": 1, "term_asts_json": [{"kind": "u"}]},
    }
    factorized_payload = {
        "engine": "factorized",
        "kind": "factorized_blocks",
        "order": 1,
        "x_axis": 0,
        "canonical_equation": "u_x + x*u = 0",
        "residual_ast": "Add(DU(0), Mul(Var(0), U()))",
        "probe_rms": 2.0e-5,
        "lane": "x_coeff_on_u",
        "family": "linear",
        "validation_candidate": {"order": 1, "term_asts_json": [{"kind": "mul"}]},
    }
    fss_payload = {
        "engine": "factorized_search",
        "kind": "factorized",
        "order": 2,
        "x_axis": 0,
        "canonical_equation": "u_xx + u = 0",
        "feature_names": ["x0", "u", "du"],
        "expr_ast": ("var", 1),
        "mapping": {"kind": "poly", "coeffs": [0.0, -1.0]},
        "mapping_kind": "poly",
        "probe_rms": 3.0e-5,
    }

    proposals = [
        proposal_from_stlsq_result(stlsq_payload),
        proposal_from_factorized_result(factorized_payload),
        proposal_from_factorized_search_result(fss_payload),
    ]

    encoded = json.dumps([proposal.to_dict() for proposal in proposals])
    decoded = json.loads(encoded)

    assert decoded[0]["engine"] == "stlsq"
    assert decoded[0]["rhs_payload"]["validation_candidate"]["order"] == 1
    assert decoded[1]["role_signature"] == "typed:x_coeff_on_u:linear"
    assert decoded[1]["residual_payload"]["residual_ast"] == factorized_payload["residual_ast"]
    assert decoded[2]["engine"] == "factorized_search"
    assert decoded[2]["role_signature"] == "whole_rhs_fss:poly"
    assert decoded[2]["rhs_payload"]["mapping"]["coeffs"] == [0.0, -1.0]


def test_build_proposal_slate_includes_factorized_and_fss_shortlists():
    factorized = {
        "engine": "factorized",
        "kind": "factorized_blocks",
        "order": 1,
        "x_axis": 0,
        "canonical_equation": "u_x + u = 0",
        "probe_rms": 1.0e-4,
        "shortlist": [
            {
                "engine": "factorized",
                "kind": "factorized_blocks",
                "order": 1,
                "x_axis": 0,
                "canonical_equation": "u_x + x*u = 0",
                "candidate_rank": 0,
                "lane": "x_coeff_on_u",
            }
        ],
    }
    fss = {
        "engine": "factorized_search",
        "kind": "factorized",
        "order": 1,
        "x_axis": 0,
        "canonical_equation": "u_x + exp(u) = 0",
        "mapping_kind": "poly",
        "probe_rms": 2.0e-4,
        "shortlist": [
            {
                "engine": "factorized_search",
                "kind": "factorized",
                "order": 1,
                "x_axis": 0,
                "canonical_equation": "u_x + log(u) = 0",
                "mapping_kind": "poly",
                "candidate_rank": 0,
            }
        ],
    }

    slate = build_proposal_slate(factorized=factorized, factorized_search=fss)

    canonical_keys = {proposal["canonical_key"] for proposal in slate}
    assert canonical_keys == {
        "residual:du+u",
        "residual:du+u*x",
        "residual:du+exp(u)",
        "residual:du+log(u)",
    }
    assert {proposal["engine"] for proposal in slate} == {"factorized", "factorized_search"}


def test_merge_proposal_slates_namespaces_scout_support_without_duplicate_inflation():
    main = build_proposal_slate(
        first_line={
            "engine": "stlsq",
            "kind": "library",
            "order": 1,
            "x_axis": 0,
            "canonical_equation": "u_x + u = 0",
            "probe_rms": 1.0e-4,
        }
    )
    scout = build_proposal_slate(
        first_line={
            "engine": "stlsq",
            "kind": "library",
            "order": 1,
            "x_axis": 0,
            "canonical_equation": "-2*u_x - 2*u = 0",
            "probe_rms": 2.0e-4,
        }
    )

    merged = merge_proposal_slates(
        [main, scout, scout],
        source_namespaces=[None, "scout0", "scout0"],
    )

    assert len(merged) == 1
    support = merged[0]["support"]
    assert support["support_count"] == 2
    assert support["sources"] == ["first_line", "scout0:first_line"]
    assert support["source_namespaces"] == ["scout0"]
