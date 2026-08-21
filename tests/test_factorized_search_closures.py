# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch

from nestynet_sr.sr_search.factorized_search.basis_scoring import score_bound_closure
from nestynet_sr.sr_search.factorized_search.closures import (
    ClosureDesign,
    bound_closure_identity_key,
    make_direct_affine_closure,
    make_direct_linear_wrap_closure,
    make_direct_exp_closure,
    make_direct_periodic_closure,
    make_direct_power_closure,
    make_direct_quadratic_closure,
    make_direct_rational_closure,
)
from nestynet_sr.sr_search.factorized_search.expr_ast import eval_node, node_depth, node_str
from nestynet_sr.sr_search.factorized_search.proposal_families.closure_eval import make_direct_preview_row


def test_score_bound_closure_linear_materializes_expected_expression():
    x = torch.tensor(
        [
            [0.2, 0.3, 0.1],
            [0.4, 0.5, 0.2],
            [0.6, 0.7, 0.3],
            [0.8, 0.9, 0.4],
        ],
        dtype=torch.float64,
    )
    feat = torch.exp(x[:, 0] * x[:, 1])
    anchor = x[:, 2]
    y = (1.3 * feat + anchor).unsqueeze(1)

    closure = make_direct_exp_closure(
        scaffold_id="exp:add:x2",
        exp_kind="add",
        hole_node=("mul", ("var", 0), ("var", 1)),
        feature_node=("exp", ("mul", ("var", 0), ("var", 1))),
        anchor_node=("var", 2),
    )
    scored = score_bound_closure(
        closure,
        design=ClosureDesign(
            fit_matrix=torch.stack(
                [
                    feat,
                    anchor,
                    torch.ones(int(x.shape[0]), dtype=x.dtype),
                ],
                dim=1,
            ),
            probe_matrix=torch.stack(
                [
                    feat,
                    anchor,
                    torch.ones(int(x.shape[0]), dtype=x.dtype),
                ],
                dim=1,
            ),
            materializer="linear_combo",
            materializer_payload={
                "terms": [
                    ("exp", ("mul", ("var", 0), ("var", 1))),
                    ("var", 2),
                ],
                "bias_index": 2,
            },
        ),
        y_fit=y,
        y_probe=y,
    )

    assert scored is not None
    assert float(scored["probe_mse"]) < 1.0e-20
    # Materialized expression is structural — coefficients live in the linear
    # head mapping, not embedded as const nodes in the AST.
    assert node_str(scored["expr"]) == "(exp((x0*x1))+x2)"


def test_make_direct_linear_wrap_closure_captures_roles_and_domain_rules():
    exp_mul = make_direct_linear_wrap_closure(
        scaffold_id="exp:mul",
        family="exp",
        wrap_kind="mul",
        wrap_op="exp",
        hole_node=("var", 0),
        anchor_node=("var", 1),
    )
    log_add = make_direct_linear_wrap_closure(
        scaffold_id="log:add",
        family="log",
        wrap_kind="add",
        wrap_op="log",
        hole_node=("var", 0),
        anchor_node=("var", 1),
        carrier_domain_rule="positive_output",
        anchor_role="companion",
    )

    exp_roles = {slot.name: slot.role for slot in exp_mul.spec.slot_specs}
    log_slots = {slot.name: slot for slot in log_add.spec.slot_specs}

    assert exp_roles["anchor"] == "envelope"
    assert log_slots["carrier"].domain_rule == "positive_output"
    assert log_slots["anchor"].role == "companion"


def test_bound_closure_identity_uses_bound_slots_not_rendered_expr_view():
    closure_a = make_direct_periodic_closure(
        scaffold_id="periodic:cos_add:view_a",
        periodic_kind="cos",
        hole_node=("var", 2),
        feature_node=("cos", ("var", 2)),
        anchor_node=("var", 0),
        envelope_node=("sqrt", ("mul", ("var", 0), ("var", 1))),
        companion_nodes=(("var", 0), ("var", 1)),
        expr=("add", ("cos", ("var", 2)), ("var", 0)),
    )
    closure_b = make_direct_periodic_closure(
        scaffold_id="periodic:cos_add:view_b",
        periodic_kind="cos",
        hole_node=("var", 2),
        feature_node=("cos", ("var", 2)),
        anchor_node=("var", 0),
        envelope_node=("sqrt", ("mul", ("var", 0), ("var", 1))),
        companion_nodes=(("var", 0), ("var", 1)),
        expr=("add", ("mul", ("cos", ("var", 2)), ("const", 1.0)), ("var", 0)),
    )
    closure_c = make_direct_periodic_closure(
        scaffold_id="periodic:cos_add:view_c",
        periodic_kind="cos",
        hole_node=("var", 1),
        feature_node=("cos", ("var", 1)),
        anchor_node=("var", 0),
        envelope_node=("sqrt", ("mul", ("var", 0), ("var", 1))),
        companion_nodes=(("var", 0), ("var", 1)),
        expr=("add", ("cos", ("var", 1)), ("var", 0)),
    )

    assert bound_closure_identity_key(closure_a) == bound_closure_identity_key(closure_b)
    assert bound_closure_identity_key(closure_a) != bound_closure_identity_key(closure_c)


def test_direct_preview_row_dedupes_by_closure_identity_not_rendered_expr():
    seen: set[str] = set()
    row_a = make_direct_preview_row(
        bound_closure=make_direct_power_closure(
            scaffold_id="power:sqrt:x0",
            power_kind="sqrt",
            exponent=0.5,
            hole_node=("var", 0),
        ),
        child_expr=("sqrt", ("var", 0)),
        fit_mse=0.0,
        probe_mse=0.0,
        max_depth=6,
        var_dims=[(0.0,)],
        y_dims=(0.0,),
        candidate_subtree_node=("var", 0),
        parent_sub_size=1,
        parent_sub_depth=1,
        parent_size=1,
        parent_depth=1,
        generation_source="test",
        tuple_provenance="test",
        proposal_family="power",
        local_mapping_kind="discrete_power",
        direct_metadata={},
        seen_child_keys=seen,
    )
    row_b = make_direct_preview_row(
        bound_closure=make_direct_power_closure(
            scaffold_id="power:sqrt:x1",
            power_kind="sqrt",
            exponent=0.5,
            hole_node=("var", 1),
        ),
        child_expr=("sqrt", ("var", 0)),
        fit_mse=0.0,
        probe_mse=0.0,
        max_depth=6,
        var_dims=[(0.0,), (0.0,)],
        y_dims=(0.0,),
        candidate_subtree_node=("var", 1),
        parent_sub_size=1,
        parent_sub_depth=1,
        parent_size=1,
        parent_depth=1,
        generation_source="test",
        tuple_provenance="test",
        proposal_family="power",
        local_mapping_kind="discrete_power",
        direct_metadata={},
        seen_child_keys=seen,
    )

    assert row_a is not None
    assert row_b is not None
    assert row_a["proposal_key"] != row_b["proposal_key"]
    assert row_a["child_key"] == row_b["child_key"]


def test_direct_preview_row_allows_structural_depth_slack_for_quadratic_sqrt_mul():
    row = make_direct_preview_row(
        bound_closure=make_direct_quadratic_closure(
            scaffold_id="quadratic:sqrt_mul:x0",
            quadratic_kind="sqrt_mul",
            base_nodes=(("var", 1), ("var", 2), ("var", 3)),
            anchor_node=("var", 0),
        ),
        child_expr=(
            "mul",
            ("sqrt", ("add", ("add", ("sqr", ("var", 1)), ("sqr", ("var", 2))), ("sqr", ("var", 3)))),
            ("var", 0),
        ),
        fit_mse=0.0,
        probe_mse=0.0,
        max_depth=5,
        var_dims=[(0.0,), (0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
        candidate_subtree_node=("var", 1),
        parent_sub_size=1,
        parent_sub_depth=1,
        parent_size=1,
        parent_depth=1,
        generation_source="test",
        tuple_provenance="test",
        proposal_family="quadratic",
        local_mapping_kind="direct_quadratic_sqrt_head",
        direct_metadata={
            "quadratic_kind": "sqrt_mul",
            "quadratic_base_nodes": [("var", 1), ("var", 2), ("var", 3)],
            "anchor_node": ("var", 0),
        },
        seen_child_keys=set(),
    )

    assert row is not None
    assert row["proposal_family"] == "quadratic"


def test_direct_preview_row_allows_scaled_quadratic_sqrt_depth_slack():
    row = make_direct_preview_row(
        bound_closure=make_direct_quadratic_closure(
            scaffold_id="quadratic:sqrt:diffsq",
            quadratic_kind="sqrt",
            base_nodes=(("div", ("var", 0), ("var", 1)), ("div", ("const", 1.0), ("var", 2))),
        ),
        child_expr=(
            "sqrt",
            (
                "sub",
                ("sqr", ("div", ("var", 0), ("var", 1))),
                ("mul", ("const", 9.869604401089358), ("sqr", ("div", ("const", 1.0), ("var", 2)))),
            ),
        ),
        fit_mse=0.0,
        probe_mse=0.0,
        max_depth=5,
        var_dims=[(0.0,), (0.0,), (0.0,)],
        y_dims=(0.0,),
        candidate_subtree_node=("div", ("var", 0), ("var", 1)),
        parent_sub_size=1,
        parent_sub_depth=1,
        parent_size=1,
        parent_depth=1,
        generation_source="test",
        tuple_provenance="test",
        proposal_family="quadratic",
        local_mapping_kind="direct_quadratic_sqrt_head",
        direct_metadata={
            "quadratic_kind": "sqrt",
            "quadratic_base_nodes": [("div", ("var", 0), ("var", 1)), ("div", ("const", 1.0), ("var", 2))],
        },
        seen_child_keys=set(),
        local_mapping_coeffs=[1.0, -9.869604401089358],
    )

    assert row is not None
    assert int(row["candidate_child_depth"]) == 6


def test_score_bound_closure_affine_latent_materializes_expected_expression():
    x = torch.tensor(
        [
            [0.2, 0.3],
            [0.4, 0.55],
            [0.6, 0.65],
            [0.8, 1.05],
        ],
        dtype=torch.float64,
    )
    y = (1.5 * x[:, 0] - 2.0 * x[:, 1] + 0.25).unsqueeze(1)

    closure = make_direct_affine_closure(
        scaffold_id="affine:latent:x0+x1",
        term_nodes=(("var", 0), ("var", 1)),
    )
    scored = score_bound_closure(
        closure,
        design=ClosureDesign(
            fit_matrix=torch.stack(
                [
                    x[:, 0],
                    x[:, 1],
                    torch.ones(int(x.shape[0]), dtype=x.dtype),
                ],
                dim=1,
            ),
            probe_matrix=torch.stack(
                [
                    x[:, 0],
                    x[:, 1],
                    torch.ones(int(x.shape[0]), dtype=x.dtype),
                ],
                dim=1,
            ),
            materializer="linear_combo",
            materializer_payload={
                "terms": [("var", 0), ("var", 1)],
                "bias_index": 2,
            },
        ),
        y_fit=y,
        y_probe=y,
    )

    assert scored is not None
    assert float(scored["probe_mse"]) < 1.0e-20
    expr_text = node_str(scored["expr"])
    assert "x0" in expr_text
    assert "x1" in expr_text


def test_score_bound_closure_fractional_materializes_expected_expression():
    x = torch.tensor(
        [
            [0.2, 0.3],
            [0.4, 0.5],
            [0.6, 0.7],
            [0.8, 0.9],
        ],
        dtype=torch.float64,
    )
    u = x[:, 0:1]
    v = (x[:, 0] * x[:, 1]).unsqueeze(1)
    y = (1.0 + u) / (1.0 + v)

    closure = make_direct_rational_closure(
        scaffold_id="rational:affine",
        u_node=("var", 0),
        v_node=("mul", ("var", 0), ("var", 1)),
    )
    scored = score_bound_closure(
        closure,
        design=ClosureDesign(
            payload={
                "u_fit": u,
                "u_probe": u,
                "v_fit": v,
                "v_probe": v,
                "safe_eps": 1.0e-6,
            },
            materializer="rational_affine",
            materializer_payload={
                "u_node": ("var", 0),
                "v_node": ("mul", ("var", 0), ("var", 1)),
                "max_depth": 6,
                "var_dims": [(0.0,), (0.0,)],
                "y_dims": (0.0,),
            },
        ),
        y_fit=y,
        y_probe=y,
    )

    assert scored is not None
    assert float(scored["probe_mse"]) < 1.0e-20
    assert node_str(scored["expr"]) == "((1+x0)/((x0*x1)+1))"


def test_score_bound_closure_harmonic_materializes_expected_expression():
    x = torch.tensor(
        [
            [0.25, 0.36, 0.2],
            [0.49, 0.72, 0.5],
            [0.81, 1.05, 0.8],
            [1.21, 1.37, 1.1],
            [1.44, 1.78, 1.4],
            [1.69, 2.05, 1.7],
        ],
        dtype=torch.float64,
    )
    envelope = torch.sqrt(x[:, 0] * x[:, 1])
    carrier = x[:, 2]
    cos_term = envelope * torch.cos(carrier)
    sin_term = envelope * torch.sin(carrier)
    y = (2.0 * cos_term + x[:, 0] + x[:, 1]).unsqueeze(1)

    closure = make_direct_periodic_closure(
        scaffold_id="periodic:cos_add:x0",
        periodic_kind="cos",
        hole_node=("var", 2),
        feature_node=("cos", ("var", 2)),
        anchor_node=("var", 0),
        envelope_node=("sqrt", ("mul", ("var", 0), ("var", 1))),
        companion_nodes=(("var", 0), ("var", 1)),
        harmonic_feature_nodes=(("cos", ("var", 2)), ("sin", ("var", 2))),
    )
    scored = score_bound_closure(
        closure,
        design=ClosureDesign(
            fit_matrix=torch.stack(
                [
                    cos_term,
                    sin_term,
                    x[:, 0],
                    x[:, 1],
                    torch.ones(int(x.shape[0]), dtype=x.dtype),
                ],
                dim=1,
            ),
            probe_matrix=torch.stack(
                [
                    cos_term,
                    sin_term,
                    x[:, 0],
                    x[:, 1],
                    torch.ones(int(x.shape[0]), dtype=x.dtype),
                ],
                dim=1,
            ),
            materializer="linear_combo",
            materializer_payload={
                "terms": [
                    ("mul", ("sqrt", ("mul", ("var", 0), ("var", 1))), ("cos", ("var", 2))),
                    ("mul", ("sqrt", ("mul", ("var", 0), ("var", 1))), ("sin", ("var", 2))),
                    ("var", 0),
                    ("var", 1),
                ],
                "bias_index": 4,
            },
        ),
        y_fit=y,
        y_probe=y,
    )

    assert scored is not None
    assert float(scored["probe_mse"]) < 1.0e-20
    expr_text = node_str(scored["expr"])
    assert "sqrt((x0*x1))" in expr_text
    assert "cos(x2)" in expr_text
    assert "x0" in expr_text
    assert "x1" in expr_text


def test_score_bound_closure_quadratic_sqrt_materializes_expected_expression():
    x = torch.tensor(
        [
            [1.1, 0.7, 0.9, 1.3],
            [0.8, 1.2, 0.6, 1.0],
            [1.4, 0.9, 1.1, 0.5],
            [0.6, 1.3, 1.4, 0.8],
        ],
        dtype=torch.float64,
    )
    q = x[:, 0]
    energies = x[:, 1:]
    y = (q * torch.sqrt(torch.sum(energies**2, dim=1))).unsqueeze(1)

    closure = make_direct_quadratic_closure(
        scaffold_id="quadratic:sqrt_mul:x0",
        quadratic_kind="sqrt_mul",
        base_nodes=(("var", 1), ("var", 2), ("var", 3)),
        anchor_node=("var", 0),
    )
    scored = score_bound_closure(
        closure,
        design=ClosureDesign(
            payload={
                "quad_fit": energies**2,
                "quad_probe": energies**2,
                "anchor_fit": q.unsqueeze(1),
                "anchor_probe": q.unsqueeze(1),
                "safe_eps": 1.0e-8,
            },
            materializer="quadratic_sqrt",
            materializer_payload={
                "base_nodes": [("var", 1), ("var", 2), ("var", 3)],
                "anchor_node": ("var", 0),
                "max_depth": 8,
                "var_dims": [(0.0,), (0.0,), (0.0,), (0.0,)],
                "y_dims": (0.0,),
            },
        ),
        y_fit=y,
        y_probe=y,
    )

    assert scored is not None
    assert float(scored["probe_mse"]) < 1.0e-20
    expr_text = node_str(scored["expr"])
    assert "sqrt(" in expr_text
    assert "x0" in expr_text
    assert "sqr(x1)" in expr_text
    assert "sqr(x2)" in expr_text
    assert "sqr(x3)" in expr_text


def test_score_bound_closure_quadratic_sqrt_allows_signed_difference_of_squares():
    x = torch.tensor(
        [
            [2.5, 1.0, 4.0],
            [3.1, 1.2, 5.0],
            [3.8, 1.4, 5.5],
            [4.4, 1.5, 6.2],
        ],
        dtype=torch.float64,
    )
    ratio = x[:, 0] / x[:, 1]
    inv = 1.0 / x[:, 2]
    y = torch.sqrt(torch.square(ratio) - (torch.pi**2) * torch.square(inv)).unsqueeze(1)

    closure = make_direct_quadratic_closure(
        scaffold_id="quadratic:sqrt:diffsq",
        quadratic_kind="sqrt",
        base_nodes=(("div", ("var", 0), ("var", 1)), ("div", ("const", 1.0), ("var", 2))),
    )
    scored = score_bound_closure(
        closure,
        design=ClosureDesign(
            payload={
                "quad_fit": torch.stack([torch.square(ratio), torch.square(inv)], dim=1),
                "quad_probe": torch.stack([torch.square(ratio), torch.square(inv)], dim=1),
                "safe_eps": 1.0e-8,
            },
            materializer="quadratic_sqrt",
            materializer_payload={
                "base_nodes": [("div", ("var", 0), ("var", 1)), ("div", ("const", 1.0), ("var", 2))],
                "anchor_node": None,
                "max_depth": 8,
                "var_dims": [(0.0,), (0.0,), (0.0,)],
                "y_dims": (0.0,),
            },
        ),
        y_fit=y,
        y_probe=y,
    )

    assert scored is not None
    assert float(scored["probe_mse"]) < 1.0e-20
    assert float(scored["coeffs"][0]) > 0.0
    assert float(scored["coeffs"][1]) < -1.0
    expr_text = node_str(scored["expr"])
    assert "sqrt(" in expr_text
    assert "sqr((x0/x1))" in expr_text
    assert "9.8696" in expr_text
    assert "sqr((1/x2))" in expr_text
    pred = eval_node(scored["expr"], x)
    assert float(torch.mean((pred - y) ** 2).item()) < 1.0e-20


def test_score_bound_closure_quadratic_sqrt_mul_allows_structural_depth_slack():
    x = torch.tensor(
        [
            [1.1, 0.7, 0.9, 1.3],
            [0.8, 1.2, 0.6, 1.0],
            [1.4, 0.9, 1.1, 0.5],
            [0.6, 1.3, 1.4, 0.8],
        ],
        dtype=torch.float64,
    )
    q = x[:, 0]
    energies = x[:, 1:]
    y = (q * torch.sqrt(torch.sum(energies**2, dim=1))).unsqueeze(1)

    closure = make_direct_quadratic_closure(
        scaffold_id="quadratic:sqrt_mul:x0",
        quadratic_kind="sqrt_mul",
        base_nodes=(("var", 1), ("var", 2), ("var", 3)),
        anchor_node=("var", 0),
    )
    scored = score_bound_closure(
        closure,
        design=ClosureDesign(
            payload={
                "quad_fit": energies**2,
                "quad_probe": energies**2,
                "anchor_fit": q.unsqueeze(1),
                "anchor_probe": q.unsqueeze(1),
                "safe_eps": 1.0e-8,
            },
            materializer="quadratic_sqrt",
            materializer_payload={
                "base_nodes": [("var", 1), ("var", 2), ("var", 3)],
                "anchor_node": ("var", 0),
                "max_depth": 5,
                "var_dims": [(0.0,), (0.0,), (0.0,), (0.0,)],
                "y_dims": (0.0,),
            },
        ),
        y_fit=y,
        y_probe=y,
    )

    assert scored is not None
    assert int(node_depth(scored["expr"])) == 6
    assert float(scored["probe_mse"]) < 1.0e-20


def test_score_bound_closure_power_neg2_materializes_expected_expression():
    z = torch.tensor([[1.6], [2.0], [2.8], [3.5]], dtype=torch.float64)
    hole = z**2
    y = z / ((hole - 1.0) ** 2)

    closure = make_direct_power_closure(
        scaffold_id="power:neg2_mul:x0",
        power_kind="neg2_mul",
        exponent=-2.0,
        hole_node=("sqr", ("var", 0)),
        anchor_node=("var", 0),
    )
    scored = score_bound_closure(
        closure,
        design=ClosureDesign(
            payload={
                "h_fit": hole,
                "h_probe": hole,
                "anchor_fit": z,
                "anchor_probe": z,
                "exponent": -2.0,
                "safe_eps": 1.0e-8,
            },
            materializer="affine_power",
            materializer_payload={
                "hole_node": ("sqr", ("var", 0)),
                "anchor_node": ("var", 0),
                "exponent": -2.0,
                "max_depth": 8,
                "var_dims": [(0.0,)],
                "y_dims": (0.0,),
            },
        ),
        y_fit=y,
        y_probe=y,
    )

    assert scored is not None
    assert float(scored["probe_mse"]) < 1.0e-20
    expr_text = node_str(scored["expr"])
    assert "x0" in expr_text
    assert "sqr(" in expr_text
    assert "sqr(x0)" in expr_text
    assert "-1" in expr_text or "-1+" in expr_text
