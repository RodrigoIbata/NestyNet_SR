import numpy as np
import torch

from nestynet_sr.sr_core.bridges import AddNode, ConstNode, MulNode, PowNode, Var, ast_to_human_readable, eval_input_expr
from nestynet_sr.sr_gs.quotient import (
    CoordinateSpec,
    DomainSpec,
    compose_coordinate_spec_with_inputs,
    substitute_local_coordinate_ast,
)


def test_local_coordinate_ast_substitutes_atom_inputs_before_global_use():
    local_expr = MulNode(Var(0), PowNode(Var(1), -1.0))
    local_inputs = (
        MulNode(Var(0), Var(2)),
        AddNode(Var(1), ConstNode(3.0)),
    )

    raw_expr = substitute_local_coordinate_ast(local_expr, local_inputs)
    assert set(raw_expr.raw_var_idxs if hasattr(raw_expr, "raw_var_idxs") else ()) == set()
    assert "x2" in ast_to_human_readable(raw_expr)

    X = torch.tensor(
        [[2.0, 1.0, 5.0], [3.0, -0.5, 4.0]],
        dtype=torch.float64,
    )
    expected = (X[:, 0] * X[:, 2]) / (X[:, 1] + 3.0)
    actual = eval_input_expr(raw_expr, X).reshape(-1)
    assert torch.allclose(actual, expected)


def test_coordinate_spec_composition_recomputes_raw_support_and_provenance():
    coord = CoordinateSpec(
        name="local_ratio",
        kind="ratio",
        ast=MulNode(Var(0), PowNode(Var(1), -1.0)),
        coordinate_map=None,
        domain=DomainSpec(exclusions=("z1 == 0",)),
        provenance={"source": "test"},
        raw_support=(0, 1),
    )
    local_inputs = (
        MulNode(Var(0), Var(2)),
        AddNode(Var(1), ConstNode(3.0)),
    )

    raw = compose_coordinate_spec_with_inputs(coord, local_inputs)

    assert raw.coordinate_map is None
    assert raw.raw_support == (0, 1, 2)
    assert raw.provenance["local_coordinate_namespace"]
    assert raw.provenance["substituted_to_raw_ast"]
    assert "x2" in raw.provenance["raw_human"]

    X = np.asarray([[2.0, 1.0, 5.0], [3.0, -0.5, 4.0]], dtype=float)
    np.testing.assert_allclose(raw.evaluate(X), (X[:, 0] * X[:, 2]) / (X[:, 1] + 3.0))
