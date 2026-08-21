# SPDX-License-Identifier: MPL-2.0

from nestynet_sr.sr_core.ast_simplify import SimplifyOptions, simplify_ast, stable_ast_key
from nestynet_sr.sr_core.bridges import Add, ConstNode, Mul, Pow, Var
from nestynet_sr.sr_search.ast_utils import check_ast_is_tree


def _safe_opts(**kwargs):
    opts = {
        "enabled": True,
        "level": "safe",
        "domain_policy": "strict",
        "context": "generic",
    }
    opts.update(kwargs)
    return SimplifyOptions(**opts)


def test_neutral_add_mul_and_pow_one_simplify_to_var():
    x0 = Var(0)
    for expr in (Add(x0, ConstNode(0)), Mul(ConstNode(1), x0), Pow(x0, 1.0)):
        out, stats = simplify_ast(expr, _safe_opts())
        assert stable_ast_key(out) == stable_ast_key(x0)
        assert stats.changed
        ok, detail = check_ast_is_tree(out)
        assert ok, detail


def test_commutative_products_have_same_simplified_key():
    a, _ = simplify_ast(Mul(Var(1), Var(0)), _safe_opts(context="de_term"))
    b, _ = simplify_ast(Mul(Var(0), Var(1)), _safe_opts(context="de_term"))
    assert stable_ast_key(a, context="de_term") == stable_ast_key(b, context="de_term")


def test_nested_products_flatten_and_are_idempotent():
    expr = Mul(Var(2), Mul(Var(1), Var(0)))
    once, _ = simplify_ast(expr, _safe_opts(context="de_term"))
    twice, _ = simplify_ast(once, _safe_opts(context="de_term"))
    assert stable_ast_key(once, context="de_term") == stable_ast_key(twice, context="de_term")
    ok, detail = check_ast_is_tree(once)
    assert ok, detail


def test_complex_constants_preserve_complex_identity():
    expr = Add(ConstNode(1 + 0j), ConstNode(2 + 0j))
    out, _ = simplify_ast(expr, _safe_opts())
    assert isinstance(out, ConstNode)
    assert isinstance(out.value, complex)
    assert out.value == 3 + 0j


def test_strict_mode_does_not_cancel_singular_factor_to_one():
    expr = Mul(Var(0), Pow(Var(0), -1.0))
    out, _ = simplify_ast(expr, _safe_opts(context="de_term"))
    assert stable_ast_key(out, context="de_term") != stable_ast_key(ConstNode(1), context="de_term")
