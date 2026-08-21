# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import pytest
import torch

from nestynet_sr.sr_core.bridges import Pow, U, Var
from nestynet_sr.sr_de import (
    DEFeatureGroup,
    DESearchConfig,
    DESearchResult,
    assemble_explicit_supports,
    assemble_implicit_rational_supports,
    build_de_candidate_eval_report,
    build_de_term_bank,
)
from nestynet_sr.sr_search.factorized_search.oracle_lab_de import DEFeatureTensors


def _linear_rhs_group(*, scale: float = 1.0, group_id: str = "g0") -> DEFeatureGroup:
    x_fit = torch.linspace(0.1, 1.0, 96, dtype=torch.float64).reshape(-1, 1)
    x_probe = torch.linspace(1.1, 2.0, 96, dtype=torch.float64).reshape(-1, 1)
    u_fit = float(scale) * torch.exp(-0.25 * x_fit)
    u_probe = float(scale) * torch.exp(-0.25 * x_probe)
    du_fit = -(2.0 * u_fit + 3.0 * x_fit)
    du_probe = -(2.0 * u_probe + 3.0 * x_probe)
    return DEFeatureGroup(
        id=group_id,
        features=DEFeatureTensors(
            x_fit=x_fit,
            u_fit=u_fit,
            du_fit=du_fit,
            d2u_fit=torch.zeros_like(u_fit),
            x_probe=x_probe,
            u_probe=u_probe,
            du_probe=du_probe,
            d2u_probe=torch.zeros_like(u_probe),
        ),
    )


def _fss_identity_candidate(feature_index: int, *, rank: int = 0) -> dict:
    return {
        "engine": "factorized_search",
        "order": 1,
        "x_axis": 0,
        "feature_names": ["x0", "u"],
        "expr_ast": ["var", int(feature_index)],
        "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
        "mapping_kind": "poly",
        "canonical_equation": f"fss_var_{feature_index}",
        "shortlist_rank": int(rank),
    }


def _rational_rhs_group(*, group_id: str = "rat") -> DEFeatureGroup:
    x_fit = torch.linspace(0.1, 1.0, 128, dtype=torch.float64).reshape(-1, 1)
    x_probe = torch.linspace(1.1, 2.0, 128, dtype=torch.float64).reshape(-1, 1)
    u_fit = 1.0 / (1.0 + x_fit)
    u_probe = 1.0 / (1.0 + x_probe)
    du_fit = -u_fit / (1.0 + x_fit)
    du_probe = -u_probe / (1.0 + x_probe)
    return DEFeatureGroup(
        id=group_id,
        features=DEFeatureTensors(
            x_fit=x_fit,
            u_fit=u_fit,
            du_fit=du_fit,
            d2u_fit=torch.zeros_like(u_fit),
            x_probe=x_probe,
            u_probe=u_probe,
            du_probe=du_probe,
            d2u_probe=torch.zeros_like(u_probe),
        ),
    )


def _pivoted_mass_group(*, scale: float = 1.0, group_id: str = "pivot") -> DEFeatureGroup:
    x_fit = torch.linspace(0.2, 1.1, 128, dtype=torch.float64).reshape(-1, 1)
    x_probe = torch.linspace(1.2, 2.1, 128, dtype=torch.float64).reshape(-1, 1)
    u_fit = float(scale) * (1.0 + x_fit**2)
    u_probe = float(scale) * (1.0 + x_probe**2)
    du_fit = -u_fit / (x_fit + u_fit)
    du_probe = -u_probe / (x_probe + u_probe)
    return DEFeatureGroup(
        id=group_id,
        features=DEFeatureTensors(
            x_fit=x_fit,
            u_fit=u_fit,
            du_fit=du_fit,
            d2u_fit=torch.zeros_like(u_fit),
            x_probe=x_probe,
            u_probe=u_probe,
            du_probe=du_probe,
            d2u_probe=torch.zeros_like(u_probe),
        ),
    )


def test_de_term_bank_normalizes_library_and_fss_terms_and_dedupes_columns():
    groups = [_linear_rhs_group()]
    stlsq = DESearchResult(
        order=1,
        x_axis=0,
        term_asts=[U()],
        coeffs=torch.tensor([2.0], dtype=torch.float64),
        rms_train=1.0,
    )

    bank = build_de_term_bank(
        groups,
        order=1,
        x_axis=0,
        library_results=[stlsq],
        factorized_search_candidates=[
            _fss_identity_candidate(0, rank=0),
            _fss_identity_candidate(1, rank=1),
        ],
    )

    sources = {term.source for term in bank.terms}
    displays = [term.metadata.get("display", "") for term in bank.terms]

    assert sources == {"stlsq", "factorized_search"}
    assert any("fss_var_0" in str(display) for display in displays)
    assert bank.diagnostics["source_counts"]["stlsq"] == 1
    assert bank.diagnostics["source_counts"]["factorized_search"] == 1
    assert any(row["reason"] == "duplicate_column" for row in bank.diagnostics["rejected"])


def test_role_shadow_affine_denominator_terms_are_visible_in_term_bank():
    groups = [_linear_rhs_group()]
    stlsq = DESearchResult(
        order=1,
        x_axis=0,
        term_asts=[U(), Var(0)],
        coeffs=torch.zeros(2, dtype=torch.float64),
        rms_train=1.0,
    )

    bank = build_de_term_bank(
        groups,
        order=1,
        x_axis=0,
        library_results=[stlsq],
        enable_role_shadow=True,
        max_role_shadow_base_terms=2,
        max_role_shadow_terms=4,
    )

    shadow_terms = [term for term in bank.terms if term.source == "role_shadow_affine"]

    assert shadow_terms
    assert bank.diagnostics["role_shadow"]["enabled"] is True
    assert bank.diagnostics["role_shadow"]["kept"] == len(shadow_terms)
    assert all(term.role_hint == "denominator" for term in shadow_terms)
    assert all(term.metadata.get("source_term_id") for term in shadow_terms)
    assert any("role_shadow_affine" in opp["source"] for opp in bank.diagnostics["role_shadow"]["opportunities"])


def test_explicit_support_assembly_recovers_shared_two_term_rhs():
    groups = [
        _linear_rhs_group(scale=1.0, group_id="g0"),
        _linear_rhs_group(scale=1.7, group_id="g1"),
    ]
    stlsq = DESearchResult(
        order=1,
        x_axis=0,
        term_asts=[U(), Var(0), Pow(Var(0), 2.0)],
        coeffs=torch.zeros(3, dtype=torch.float64),
        rms_train=1.0,
    )
    bank = build_de_term_bank(groups, order=1, x_axis=0, library_results=[stlsq])

    supports = assemble_explicit_supports(
        bank,
        groups,
        max_support_width=2,
        beam_width=8,
        expansions_per_support=4,
        shortlist_topk=8,
    )

    assert supports
    best = supports[0]
    best_displays = [row["display"] for row in best.metadata["term_summaries"]]
    coeff_by_display = dict(zip(best_displays, best.coefficients))

    assert best.probe_rms_max < 1.0e-10
    assert coeff_by_display[repr(U())] == pytest.approx(2.0, abs=1.0e-8)
    assert coeff_by_display[repr(Var(0))] == pytest.approx(3.0, abs=1.0e-8)


def test_implicit_rational_support_recovers_safe_denominator():
    groups = [_rational_rhs_group()]
    stlsq = DESearchResult(
        order=1,
        x_axis=0,
        term_asts=[U(), Var(0), Pow(Var(0), 2.0)],
        coeffs=torch.zeros(3, dtype=torch.float64),
        rms_train=1.0,
    )
    bank = build_de_term_bank(groups, order=1, x_axis=0, library_results=[stlsq])
    explicit = assemble_explicit_supports(
        bank,
        groups,
        max_support_width=1,
        beam_width=4,
        expansions_per_support=2,
        shortlist_topk=4,
    )

    implicit = assemble_implicit_rational_supports(
        bank,
        groups,
        explicit,
        max_denominator_width=1,
        max_denominator_terms=3,
        max_implicit_candidates=8,
        ast_serializer=lambda node: {"repr": repr(node)},
    )

    assert implicit
    best = implicit[0]
    assert best.denominator_safety["safe"] is True
    assert best.implicit_probe_rms_max < 1.0e-10
    assert best.explicit_probe_rms_max < 1.0e-10
    assert best.numerator_term_ids == (bank.terms[0].term_id,)
    assert best.denominator_term_ids == (bank.terms[1].term_id,)
    assert best.numerator_coefficients[0] == pytest.approx(1.0, abs=1.0e-8)
    assert best.denominator_coefficients[0] == pytest.approx(1.0, abs=1.0e-8)
    assert best.validation_candidate is not None


def test_pivoted_implicit_role_shadow_recovers_two_term_mass_factor():
    groups = [
        _pivoted_mass_group(scale=1.0, group_id="p0"),
        _pivoted_mass_group(scale=1.7, group_id="p1"),
    ]
    stlsq = DESearchResult(
        order=1,
        x_axis=0,
        term_asts=[U(), Var(0)],
        coeffs=torch.zeros(2, dtype=torch.float64),
        rms_train=1.0,
    )
    bank = build_de_term_bank(
        groups,
        order=1,
        x_axis=0,
        library_results=[stlsq],
        enable_role_shadow=True,
    )
    explicit = assemble_explicit_supports(
        bank,
        groups,
        max_support_width=1,
        beam_width=4,
        expansions_per_support=2,
        shortlist_topk=4,
    )

    const_pivot = assemble_implicit_rational_supports(
        bank,
        groups,
        explicit,
        max_denominator_width=1,
        max_denominator_terms=4,
        max_implicit_candidates=8,
        enable_pivoted=False,
        ast_serializer=lambda node: {"repr": repr(node)},
    )
    pivoted = assemble_implicit_rational_supports(
        bank,
        groups,
        explicit,
        max_denominator_width=1,
        max_denominator_terms=4,
        max_implicit_candidates=16,
        enable_pivoted=True,
        max_pivots=4,
        ast_serializer=lambda node: {"repr": repr(node)},
    )

    assert const_pivot
    assert pivoted
    assert min(cand.explicit_probe_rms_max for cand in const_pivot) > 1.0e-4
    assert pivoted[0].pivot != "const_one"
    assert pivoted[0].explicit_probe_rms_max < 1.0e-10
    assert pivoted[0].denominator_safety["safe"] is True
    assert pivoted[0].validation_candidate is not None
    assert pivoted[0].metadata["role_shadow_pivoted"] is True


def test_de_candidate_eval_report_is_diagnostics_only_and_jsonable():
    groups = [_linear_rhs_group()]
    stlsq = DESearchResult(
        order=1,
        x_axis=0,
        term_asts=[U(), Var(0)],
        coeffs=torch.zeros(2, dtype=torch.float64),
        rms_train=1.0,
    )

    report = build_de_candidate_eval_report(
        groups,
        cfg=DESearchConfig(x_axis=0, order_candidates=(1,)),
        primary_result=stlsq,
        order=1,
        x_axis=0,
        enable_role_shadow=False,
    )

    assert report["status"] == "OK"
    assert report["mode"] == "diagnostics_only"
    assert report["term_bank"]["term_count"] == 2
    assert report["selected_explicit_support"]["probe_rms_max"] < 1.0e-10


def test_de_candidate_eval_report_includes_implicit_rollout_shortlist():
    groups = [_rational_rhs_group()]
    stlsq = DESearchResult(
        order=1,
        x_axis=0,
        term_asts=[U(), Var(0)],
        coeffs=torch.zeros(2, dtype=torch.float64),
        rms_train=1.0,
    )

    report = build_de_candidate_eval_report(
        groups,
        cfg=DESearchConfig(x_axis=0, order_candidates=(1,)),
        primary_result=stlsq,
        order=1,
        x_axis=0,
        max_support_width=1,
        ast_serializer=lambda node: {"repr": repr(node)},
    )

    assert report["status"] == "OK"
    assert report["implicit_rational_supports"]
    assert report["selected_implicit_rational_support"]["denominator_safety"]["safe"] is True
    assert any(row["candidate_family"] == "implicit_rational" for row in report["rollout_shortlist"])
