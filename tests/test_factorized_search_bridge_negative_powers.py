# SPDX-License-Identifier: MPL-2.0

import pytest

pytest.importorskip("sympy")

from nestynet_sr.sr_core.bridges import PowNode, Var
from nestynet_sr.sr_search.factorized_search.bridge import nestynet_to_factorized_search
from nestynet_sr.sr_search.factorized_search.oracle_lab import compile_target_ast, equation_spec_from_dict


def _spec_payload(spec_id: str, expr: str) -> dict:
    return {
        "id": spec_id,
        "basis": ["D"],
        "variables": [
            {"name": "x", "bounds": [0.3, 3.0], "dim": [0]},
            {"name": "y", "bounds": [0.3, 3.0], "dim": [0]},
        ],
        "constants": [],
        "target": {"expr": expr, "dim": [0]},
    }


@pytest.mark.parametrize(
    ("exp", "expected"),
    [
        (-1.0, ("div", ("const", 1.0), ("var", 0))),
        (-0.5, ("div", ("const", 1.0), ("sqrt", ("var", 0)))),
        (-2.0, ("div", ("const", 1.0), ("sqr", ("var", 0)))),
    ],
)
def test_nestynet_to_factorized_search_supports_selected_negative_powers(exp, expected):
    assert nestynet_to_factorized_search(PowNode(Var(0), exp)) == expected


def test_compile_target_ast_supports_invsqrt_demo_shape():
    spec = equation_spec_from_dict(
        _spec_payload("invsqrt_demo", "x / sqrt(1 + y**2)"),
        source="unit-test",
    )
    ast = compile_target_ast(spec)
    assert ast[0] == "div"
    assert ast[1] == ("var", 0)
    assert ast[2][0] == "sqrt"


def test_compile_target_ast_supports_rational_shift_demo_shape():
    spec = equation_spec_from_dict(
        _spec_payload("rational_shift_demo", "(1 + x) / (1 + x*y)"),
        source="unit-test",
    )
    ast = compile_target_ast(spec)
    assert ast[0] == "div"
    assert ast[1][0] == "add"
    assert ast[2][0] == "add"


def test_compile_target_ast_supports_inverse_square_shift_demo_shape():
    payload = {
        "id": "sqr_shift_demo",
        "basis": ["D"],
        "variables": [
            {"name": "z", "bounds": [1.5, 5.0], "dim": [0]},
        ],
        "constants": [],
        "target": {"expr": "z / (z**2 - 1)**2", "dim": [0]},
    }
    spec = equation_spec_from_dict(payload, source="unit-test")
    ast = compile_target_ast(spec)
    assert ast[0] == "div"
    assert ast[1] == ("var", 0)
    assert ast[2][0] == "sqr"
