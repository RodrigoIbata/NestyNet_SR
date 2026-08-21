# SPDX-License-Identifier: MPL-2.0

from nestynet_sr.sr_core.bridges import Mul, U, Var
from nestynet_sr.sr_de.de_search import DESearchConfig, _maybe_simplify_de_rows


def test_de_ast_simplify_switch_off_leaves_rows_unchanged():
    rows = [(Mul(Var(1), Var(0)), "baseline", "manual")]
    out = _maybe_simplify_de_rows(rows, DESearchConfig(ast_simplify=False), order=2)
    assert out is rows


def test_de_ast_simplify_collapses_commutative_duplicate_and_preserves_gs_source():
    rows = [
        (Mul(Var(0), U()), "baseline", "x_u_cross"),
        (Mul(U(), Var(0)), "gs_template", "known_lie"),
    ]
    cfg = DESearchConfig(ast_simplify=True, ast_simplify_domain_policy="strict")
    out = _maybe_simplify_de_rows(rows, cfg, order=2)
    assert len(out) == 1
    _term, source, family = out[0]
    assert source.startswith("gs_template")
    assert "baseline" in source
    assert "known_lie" in family
    assert "x_u_cross" in family
