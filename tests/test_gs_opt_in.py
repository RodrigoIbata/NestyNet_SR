# SPDX-License-Identifier: MPL-2.0

from nestynet_sr.sr_core.bridges import Add, U, Var
from nestynet_sr.sr_de.de_search import (
    DESearchConfig,
    _gs_dim_policy_active,
    _maybe_expr_ir_de_rows,
    _maybe_simplify_de_rows,
    build_de_library_terms_with_sources,
)
from nestynet_sr.sr_gs.reporting import build_gs_payload, reset_gs_reporter


def _policy_event_count() -> int:
    return int(build_gs_payload(mode="test")["summary"]["policy_events"])


def test_de_library_ignores_gs_knobs_until_gs_enable():
    cfg = DESearchConfig(
        gs_enable=False,
        gs_policy="gs-only-affine",
        gs_de_templates=True,
        gs_de_all_upgrades=True,
        gs_de_determining_equations=True,
        gs_de_contact_templates=True,
        gs_de_noether_templates=True,
        gs_de_discrete_symmetry_templates=True,
        gs_de_weighted_scaling_templates=True,
        gs_de_radial_reduction_templates=True,
        gs_de_lie_prolongation=True,
        gs_unit_torus=True,
        gs_pi_invariants=True,
        order_candidates=(2,),
        max_x_power=1,
        max_u_power=1,
    )

    terms, sources = build_de_library_terms_with_sources(cfg, order=2)

    assert terms
    assert sources
    assert all(not str(source).startswith("gs") for source in sources)
    assert not _gs_dim_policy_active(cfg)


def test_expr_ir_without_gs_does_not_record_gs_policy_events():
    reset_gs_reporter({"case": "expr_ir_without_gs"})
    cfg = DESearchConfig(expr_ir="qdag", expr_canonicalize="safe", gs_enable=False)
    rows = [
        (Add(Var(0), U()), "baseline", "manual"),
        (Add(U(), Var(0)), "baseline", "manual"),
    ]

    out = _maybe_expr_ir_de_rows(rows, cfg, order=1)

    assert len(out) == 1
    assert _policy_event_count() == 0


def test_ast_simplify_without_gs_does_not_record_gs_policy_events():
    reset_gs_reporter({"case": "ast_simplify_without_gs"})
    cfg = DESearchConfig(ast_simplify=True, gs_enable=False)
    rows = [(Add(Var(0), U()), "baseline", "manual")]

    out = _maybe_simplify_de_rows(rows, cfg, order=1)

    assert out
    assert _policy_event_count() == 0


def test_expr_ir_records_gs_policy_event_when_gs_is_enabled():
    reset_gs_reporter({"case": "expr_ir_with_gs"})
    cfg = DESearchConfig(expr_ir="qdag", expr_canonicalize="safe", gs_enable=True)
    rows = [
        (Add(Var(0), U()), "baseline", "manual"),
        (Add(U(), Var(0)), "gs:test", "manual_gs"),
    ]

    out = _maybe_expr_ir_de_rows(rows, cfg, order=1)

    assert len(out) == 1
    assert _policy_event_count() == 1
