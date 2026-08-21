# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from nestynet_sr.sr_search.factorized_search.basis_state import BasisState, FeatureBlock
from nestynet_sr.sr_search.factorized_search.expr_ast import node_str
from nestynet_sr.sr_search.factorized_search.closures import (
    make_direct_linear_wrap_closure,
    make_direct_periodic_closure,
    make_direct_rational_closure,
)
from nestynet_sr.sr_search.factorized_search.closures import SlotSpec
from nestynet_sr.sr_search.factorized_search.proposal_families import (
    direct_exp_scaffold_kind,
    direct_log_scaffold_kind,
    direct_rational_scaffold_kind,
    resolve_direct_operator_planner,
)
from nestynet_sr.sr_search.factorized_search.proposal_families.periodic_search import (
    direct_periodic_scaffold_kind,
    periodic_scaffold_mode,
)
from nestynet_sr.sr_search.factorized_search.proposal_families.scaffold_enum import (
    build_operator_bound_closure,
    enumerate_operator_applications,
)
from nestynet_sr.sr_search.factorized_search.proposal_families.compat import (
    OuterScaffoldSpec,
    enumerate_closure_search_specs,
    operator_application_from_scaffold,
    render_operator_as_scaffold,
)
from nestynet_sr.sr_search.factorized_search.proposal_families.operator_specs import family_operator_specs
from nestynet_sr.sr_search.factorized_search.proposal_families.seed_blocks import (
    build_recursive_seed_pool,
    extend_seed_blocks_with_basis,
    generate_affine_seed_blocks,
    generate_monomial_seed_blocks,
    generate_quadratic_seed_blocks,
    seed_anchor_blocks,
    seed_blocks_from_basis_state,
)
from nestynet_sr.sr_search.factorized_search.proposal_families.slot_binding import (
    bind_slot_candidates,
    family_anchor_blocks,
    validate_bound_closure_bindings,
)
from nestynet_sr.sr_search.factorized_search.proposal_families.types import OperatorApplication
from nestynet_sr.sr_search.factorized_search.proposal_families.types import OperatorCompatState


def test_seed_anchor_blocks_emit_const_vars_and_small_pool_nodes():
    rows = seed_anchor_blocks(
        nvars=2,
        pool_nodes=[
            ("mul", ("var", 0), ("var", 1)),
            ("mul", ("mul", ("var", 0), ("var", 1)), ("var", 1)),
        ],
        pool_dims=[(0.0,), (0.0,)],
        var_dims=[(0.0,), (0.0,)],
        max_count=2,
    )
    exprs = [row.to_dict()["expr"] for row in rows]

    assert "1" in exprs
    assert "x0" in exprs
    assert "x1" in exprs
    assert "(x0*x1)" in exprs
    assert "((x0*x1)*x1)" not in exprs


def test_bind_slot_candidates_filters_by_dim_and_arity():
    rows = seed_anchor_blocks(
        nvars=3,
        pool_nodes=[("mul", ("var", 0), ("var", 1))],
        pool_dims=[(1.0,)],
        var_dims=[(0.0,), (1.0,), (0.0,)],
        max_count=3,
    )
    slot = SlotSpec(name="carrier", role="carrier", dim_rule=(0.0,), arity_cap=1)
    bound = bind_slot_candidates(slot, rows, var_dims=[(0.0,), (1.0,), (0.0,)])
    exprs = [row.to_dict()["expr"] for row in bound]

    assert "1" in exprs
    assert "x0" in exprs
    assert "x2" in exprs
    assert "x1" not in exprs
    assert "(x0*x1)" not in exprs


def test_bind_slot_candidates_filters_by_domain_and_reuse_policy():
    rows = seed_anchor_blocks(
        nvars=1,
        pool_nodes=[("exp", ("var", 0)), ("sqrt", ("var", 0))],
        pool_dims=[(0.0,), (0.0,)],
        var_dims=[(0.0,)],
        max_count=2,
    )
    exp_block = next(row for row in rows if row.to_dict()["expr"] == "exp(x0)")
    slot = SlotSpec(
        name="carrier",
        role="carrier",
        domain_rule="positive_output",
        reuse_policy="distinct_from:anchor",
    )
    bound = bind_slot_candidates(
        slot,
        rows,
        var_dims=[(0.0,)],
        existing_bindings={"anchor": exp_block},
    )
    exprs = [row.to_dict()["expr"] for row in bound]

    assert "exp(x0)" not in exprs
    assert "sqrt(x0)" not in exprs
    assert "1" in exprs
    assert "x0" in exprs


def test_bind_slot_candidates_filters_by_recursive_slot_metadata():
    rows = seed_anchor_blocks(
        nvars=2,
        pool_nodes=[],
        pool_dims=[],
        var_dims=[(0.0,), (0.0,)],
        max_count=2,
    )
    pool = build_recursive_seed_pool(
        rows,
        rounds=2,
        include_product=True,
        include_monomial=True,
        include_quadratic=True,
        product_max_arity=2,
        product_limit=8,
        monomial_limit=12,
        quadratic_limit=8,
        max_builder_depth=3,
        max_nonlinear_depth=2,
    )
    slot = SlotSpec(
        name="carrier",
        role="carrier",
        metadata={
            "allowed_builders": ("identity", "product", "monomial"),
            "disallow_builders": ("quadratic",),
            "max_builder_depth": 2,
            "max_nonlinear_depth": 1,
            "max_product_arity": 2,
        },
    )
    bound = bind_slot_candidates(slot, pool, var_dims=[(0.0,), (0.0,)])
    exprs = {row.to_dict()["expr"] for row in bound}
    builders = {row.builder for row in bound}

    assert "(x0*x1)" in exprs
    assert any("sqrt(" in expr and "(x0*x1)" in expr for expr in exprs)
    assert "quadratic" not in builders
    assert all(row.builder in {"identity", "product", "monomial"} for row in bound)


def test_family_anchor_blocks_and_operator_apps_preserve_exp_var_priority():
    rows = seed_anchor_blocks(
        nvars=2,
        pool_nodes=[("mul", ("var", 0), ("var", 1))],
        pool_dims=[(0.0,)],
        var_dims=[(0.0,), (0.0,)],
        max_count=3,
    )
    exp_rows = family_anchor_blocks("exp", rows, anchor_cap=2)
    assert [row.to_dict()["expr"] for row in exp_rows] == ["x0", "x1"]

    apps = enumerate_operator_applications(
        families=["exp"],
        nvars=2,
        y_dims=(0.0,),
        var_dims=[(0.0,), (0.0,)],
        pool_nodes=[("mul", ("var", 0), ("var", 1))],
        pool_dims=[(0.0,)],
        anchors_per_family=2,
        max_scaffolds=8,
    )
    add_apps = [app for app in apps if app.operator_id == "exp:add"]
    assert add_apps
    assert any(app.metadata.get("slot_bindings", {}).get("anchor", {}).get("expr") == "x0" for app in add_apps)


def test_family_anchor_blocks_preserve_quadratic_var_priority():
    rows = seed_anchor_blocks(
        nvars=3,
        pool_nodes=[
            ("mul", ("var", 0), ("var", 1)),
            ("sqr", ("var", 1)),
        ],
        pool_dims=[(0.0,), (0.0,)],
        var_dims=[(0.0,), (0.0,), (0.0,)],
        max_count=3,
    )
    quad_rows = family_anchor_blocks("quadratic", rows, anchor_cap=2)
    assert [row.to_dict()["expr"] for row in quad_rows] == ["x0", "x1"]


def test_family_anchor_blocks_preserve_power_var_priority():
    rows = seed_anchor_blocks(
        nvars=3,
        pool_nodes=[
            ("mul", ("var", 0), ("var", 1)),
            ("sqr", ("var", 1)),
        ],
        pool_dims=[(0.0,), (0.0,)],
        var_dims=[(0.0,), (0.0,), (0.0,)],
        max_count=3,
    )
    power_rows = family_anchor_blocks("power", rows, anchor_cap=2)
    assert [row.to_dict()["expr"] for row in power_rows] == ["x0", "x1"]


def test_generate_monomial_seed_blocks_emits_inverse_and_sqrt_forms():
    rows = seed_anchor_blocks(
        nvars=1,
        pool_nodes=[],
        pool_dims=[],
        var_dims=[(0.0,)],
        max_count=1,
    )
    monos = generate_monomial_seed_blocks(rows, limit=8)
    exprs = {row.to_dict()["expr"] for row in monos}

    assert "(1/x0)" in exprs or "(1/sqrt(x0))" in exprs or "sqrt(x0)" in exprs


def test_generate_affine_seed_blocks_emits_additive_latents():
    rows = seed_anchor_blocks(
        nvars=2,
        pool_nodes=[],
        pool_dims=[],
        var_dims=[(0.0,), (0.0,)],
        max_count=2,
    )
    affines = generate_affine_seed_blocks(rows, limit=8, max_arity=2)
    exprs = {row.to_dict()["expr"] for row in affines}

    assert "(x0+x1)" in exprs
    assert "(x1-x0)" in exprs
    assert all(row.builder == "affine" for row in affines)


def test_recursive_seed_pool_builds_compound_monomial_latents():
    rows = seed_anchor_blocks(
        nvars=2,
        pool_nodes=[],
        pool_dims=[],
        var_dims=[(0.0,), (0.0,)],
        max_count=2,
    )
    pool = build_recursive_seed_pool(
        rows,
        rounds=2,
        include_product=True,
        include_monomial=True,
        include_quadratic=False,
        product_max_arity=2,
        product_limit=8,
        monomial_limit=12,
        max_builder_depth=2,
        max_nonlinear_depth=1,
    )
    exprs = {row.to_dict()["expr"] for row in pool}

    assert "(x0*x1)" in exprs
    assert any("sqrt(" in expr and "(x0*x1)" in expr for expr in exprs)


def test_recursive_seed_pool_builds_recursive_quadratic_latents_over_products():
    rows = seed_anchor_blocks(
        nvars=2,
        pool_nodes=[],
        pool_dims=[],
        var_dims=[(0.0,), (0.0,)],
        max_count=2,
    )
    pool = build_recursive_seed_pool(
        rows,
        rounds=2,
        include_product=True,
        include_monomial=False,
        include_quadratic=True,
        product_max_arity=2,
        product_limit=8,
        quadratic_limit=16,
        max_builder_depth=3,
        max_nonlinear_depth=2,
        var_dims=[(0.0,), (0.0,)],
    )
    exprs = {row.to_dict()["expr"] for row in pool}

    assert "(x0*x1)" in exprs
    # sqr(x0*x1) can appear standalone or inside a sum — the quadratic
    # builder now prioritises multi-variable sums, so with a small budget
    # it may only appear as a term in a larger sum-of-squares.
    assert any("sqr((x0*x1))" in e for e in exprs)


def test_recursive_seed_pool_can_include_affine_latents():
    rows = seed_anchor_blocks(
        nvars=2,
        pool_nodes=[],
        pool_dims=[],
        var_dims=[(0.0,), (0.0,)],
        max_count=2,
    )
    pool = build_recursive_seed_pool(
        rows,
        rounds=1,
        include_product=False,
        include_monomial=False,
        include_quadratic=False,
        include_affine=True,
        affine_limit=8,
        max_builder_depth=2,
        max_nonlinear_depth=1,
    )
    exprs = {row.to_dict()["expr"] for row in pool}

    assert "(x0+x1)" in exprs
    assert "(x1-x0)" in exprs


def test_recursive_seed_pool_preserves_raw_variable_differences_under_pool_pressure():
    pool_nodes = [
        ("mul", ("var", 0), ("var", 1)),
        ("mul", ("var", 0), ("var", 2)),
        ("mul", ("var", 0), ("var", 3)),
        ("mul", ("var", 1), ("var", 2)),
        ("mul", ("var", 1), ("var", 3)),
        ("mul", ("var", 2), ("var", 3)),
        ("sqrt", ("var", 0)),
        ("sqrt", ("var", 1)),
        ("sqrt", ("var", 2)),
        ("sqrt", ("var", 3)),
        ("sqr", ("var", 0)),
        ("sqr", ("var", 1)),
        ("sqr", ("var", 2)),
        ("sqr", ("var", 3)),
    ]
    pool_dims = [
        (2.0,),
        (2.0,),
        (2.0,),
        (2.0,),
        (2.0,),
        (2.0,),
        (0.5,),
        (0.5,),
        (0.5,),
        (0.5,),
        (2.0,),
        (2.0,),
        (2.0,),
        (2.0,),
    ]
    rows = seed_anchor_blocks(
        nvars=4,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        var_dims=[(1.0,), (1.0,), (1.0,), (1.0,)],
        max_count=8,
    )
    pool = build_recursive_seed_pool(
        rows,
        rounds=2,
        include_product=True,
        include_monomial=True,
        include_quadratic=False,
        include_affine=True,
        product_max_arity=3,
        product_limit=24,
        monomial_limit=32,
        affine_limit=24,
        max_builder_depth=3,
        max_nonlinear_depth=2,
        var_dims=[(1.0,), (1.0,), (1.0,), (1.0,)],
    )
    exprs = {row.to_dict()["expr"] for row in pool}

    assert "(x1-x0)" in exprs
    assert "(x3-x2)" in exprs


def test_recursive_seed_pool_builds_quadratic_latents_over_affine_differences():
    rows = seed_anchor_blocks(
        nvars=4,
        pool_nodes=[],
        pool_dims=[],
        var_dims=[(0.0,), (0.0,), (0.0,), (0.0,)],
        max_count=4,
    )
    pool = build_recursive_seed_pool(
        rows,
        rounds=2,
        include_product=False,
        include_monomial=False,
        include_quadratic=True,
        include_affine=True,
        quadratic_limit=32,
        affine_limit=24,
        max_builder_depth=3,
        max_nonlinear_depth=1,
        var_dims=[(0.0,), (0.0,), (0.0,), (0.0,)],
    )
    quad_rows = [row for row in pool if row.builder == "quadratic"]

    def _pair_key(node):
        if not (
            isinstance(node, tuple)
            and len(node) >= 3
            and str(node[0]) == "sub"
            and isinstance(node[1], tuple)
            and isinstance(node[2], tuple)
            and str(node[1][0]) == "var"
            and str(node[2][0]) == "var"
        ):
            return None
        return frozenset((int(node[1][1]), int(node[2][1])))

    matched = [
        row
        for row in quad_rows
        if {
            pair
            for pair in [
                _pair_key(node)
                for node in list(dict(row.metadata or {}).get("base_nodes", ()) or [])
            ]
            if pair is not None
        }
        == {frozenset((0, 1)), frozenset((2, 3))}
    ]

    assert matched


def test_enumerate_operator_applications_includes_quadratic_difference_norm():
    apps = enumerate_operator_applications(
        families=["quadratic"],
        nvars=4,
        y_dims=(0.0,),
        var_dims=[(0.0,), (0.0,), (0.0,), (0.0,)],
        pool_nodes=[],
        pool_dims=[],
        anchors_per_family=8,
        max_scaffolds=64,
    )

    def _pair_key(node):
        if not (
            isinstance(node, tuple)
            and len(node) >= 3
            and str(node[0]) == "sub"
            and isinstance(node[1], tuple)
            and isinstance(node[2], tuple)
            and str(node[1][0]) == "var"
            and str(node[2][0]) == "var"
        ):
            return None
        return frozenset((int(node[1][1]), int(node[2][1])))

    matched = []
    for app in apps:
        bound = getattr(app, "bound_closure", None)
        bindings = dict(getattr(bound, "bindings", {}) or {})
        bases = []
        for raw in list(bindings.get("bases", ()) or ()):
            node = getattr(raw, "node", raw)
            if isinstance(node, tuple):
                bases.append(node)
        pairs = {pair for pair in (_pair_key(node) for node in bases) if pair is not None}
        if pairs == {frozenset((0, 1)), frozenset((2, 3))}:
            matched.append(app)

    assert matched


def test_enumerate_operator_applications_keeps_quadratic_difference_norm_under_pool_pressure():
    pool_nodes = [
        ("mul", ("var", 0), ("var", 1)),
        ("mul", ("var", 0), ("var", 2)),
        ("mul", ("var", 0), ("var", 3)),
        ("mul", ("var", 1), ("var", 2)),
        ("mul", ("var", 1), ("var", 3)),
        ("mul", ("var", 2), ("var", 3)),
        ("sqrt", ("var", 0)),
        ("sqrt", ("var", 1)),
        ("sqrt", ("var", 2)),
        ("sqrt", ("var", 3)),
        ("sqr", ("var", 0)),
        ("sqr", ("var", 1)),
        ("sqr", ("var", 2)),
        ("sqr", ("var", 3)),
    ]
    pool_dims = [
        (2.0,),
        (2.0,),
        (2.0,),
        (2.0,),
        (2.0,),
        (2.0,),
        (0.5,),
        (0.5,),
        (0.5,),
        (0.5,),
        (2.0,),
        (2.0,),
        (2.0,),
        (2.0,),
    ]
    apps = enumerate_operator_applications(
        families=["quadratic"],
        nvars=4,
        y_dims=(1.0,),
        var_dims=[(1.0,), (1.0,), (1.0,), (1.0,)],
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        anchors_per_family=8,
        max_scaffolds=48,
    )
    app = next(app for app in apps if app.scaffold_id == "quadratic:sqrt:(sqr((x1-x0))+sqr((x3-x2)))")
    bindings = dict(getattr(app.bound_closure, "bindings", {}) or {})
    base_exprs = [node_str(getattr(block, "node", block)) for block in list(bindings.get("bases", ()) or ())]

    assert base_exprs == ["(x1-x0)", "(x3-x2)"]


def test_generate_quadratic_seed_blocks_emit_sum_of_squares_latents():
    rows = seed_anchor_blocks(
        nvars=2,
        pool_nodes=[],
        pool_dims=[],
        var_dims=[(0.0,), (0.0,)],
        max_count=2,
    )
    quads = generate_quadratic_seed_blocks(
        rows,
        limit=8,
        max_arity=2,
        max_builder_depth=2,
        max_nonlinear_depth=1,
    )
    exprs = {row.to_dict()["expr"] for row in quads}

    assert any("sqr(x0)" in expr for expr in exprs)
    assert any("sqr(x0)" in expr and "sqr(x1)" in expr for expr in exprs)


def test_operator_application_is_native_and_renders_scaffold_compatibly():
    app = OperatorApplication(
        family="exp",
        operator_id="exp:add",
        scaffold_id="exp:add:x0",
        parent_node=("add", ("exp", ("const", 1.0)), ("var", 0)),
        hole_path=(1, 1),
        target_mode="robust",
        anchor_node=("var", 0),
        bindings={"carrier": ("const", 1.0)},
        metadata={"operator": "exp:add"},
    )

    assert not isinstance(app, OuterScaffoldSpec)

    scaffold = render_operator_as_scaffold(app)
    assert isinstance(scaffold, OuterScaffoldSpec)
    assert scaffold.scaffold_id == "exp:add:x0"
    assert tuple(scaffold.hole_path) == (1, 1)
    assert scaffold.metadata.get("slot_bindings", {}).get("carrier") == ["const", 1.0]

    roundtrip = operator_application_from_scaffold(scaffold)
    assert isinstance(roundtrip, OperatorApplication)
    assert roundtrip.operator_id == "exp:add"
    assert roundtrip.scaffold_id == app.scaffold_id


def test_linear_wrap_planner_resolution_works_for_generic_bound_closures():
    exp_app = OperatorApplication(
        family="generic",
        operator_id="generic:closure",
        scaffold_id="exp:base",
        parent_node=("exp", ("var", 0)),
        hole_path=(1,),
        target_mode="robust",
        bound_closure=make_direct_linear_wrap_closure(
            scaffold_id="exp:base",
            family="exp",
            wrap_kind="base",
            wrap_op="exp",
            hole_node=("var", 0),
        ),
        metadata={},
    )
    log_app = OperatorApplication(
        family="generic",
        operator_id="generic:closure",
        scaffold_id="log:add",
        parent_node=("add", ("log", ("var", 0)), ("var", 1)),
        hole_path=(1, 1),
        target_mode="robust",
        anchor_node=("var", 1),
        bound_closure=make_direct_linear_wrap_closure(
            scaffold_id="log:add",
            family="log",
            wrap_kind="add",
            wrap_op="log",
            hole_node=("var", 0),
            anchor_node=("var", 1),
            carrier_domain_rule="positive_output",
            anchor_role="companion",
        ),
        metadata={},
    )

    assert direct_exp_scaffold_kind(exp_app) == "base"
    assert direct_log_scaffold_kind(log_app) == "add"
    exp_planner = resolve_direct_operator_planner(exp_app)
    log_planner = resolve_direct_operator_planner(log_app)

    assert exp_planner is not None
    assert exp_planner.planner_id == "linear_wrap"
    assert exp_planner.operator_kinds == ("unary_wrap", "anchored_unary_wrap")
    assert exp_planner.composition_modes == ("base", "companion", "prefactor")

    assert log_planner is not None
    assert log_planner.planner_id == "linear_wrap"
    assert log_planner.operator_kinds == ("unary_wrap", "anchored_unary_wrap")
    assert log_planner.composition_modes == ("base", "companion", "prefactor")


def test_build_operator_bound_closure_uses_shared_linear_wrap_constructor():
    bound = build_operator_bound_closure(
        family="log",
        operator_id="log:add",
        scaffold_id="log:add",
        parent_node=("add", ("log", ("var", 0)), ("var", 1)),
        anchor_node=("var", 1),
        bindings={"carrier": ("var", 0)},
        metadata={"form": "log_add"},
    )
    carrier_slot = next(slot for slot in bound.spec.slot_specs if slot.name == "carrier")

    assert bound.spec.family == "log"
    assert bound.metadata["wrap_op"] == "log"
    assert bound.metadata["log_kind"] == "add"
    assert carrier_slot.domain_rule == "positive_output"


def test_validate_bound_closure_bindings_uses_recursive_slot_metadata_on_raw_seed_bindings():
    rows = seed_anchor_blocks(
        nvars=2,
        pool_nodes=[],
        pool_dims=[],
        var_dims=[(0.0,), (0.0,)],
        max_count=2,
    )
    pool = build_recursive_seed_pool(
        rows,
        rounds=2,
        include_product=True,
        include_monomial=True,
        include_quadratic=True,
        product_max_arity=2,
        product_limit=8,
        monomial_limit=12,
        quadratic_limit=8,
        max_builder_depth=3,
        max_nonlinear_depth=2,
    )
    quadratic_carrier = next(row for row in pool if row.builder == "quadratic")
    monomial_envelope = next(row for row in pool if row.builder == "monomial")
    closure = make_direct_periodic_closure(
        scaffold_id="periodic:cos_mul:test",
        periodic_kind="cos",
        hole_node=quadratic_carrier.node,
        feature_node=("cos", quadratic_carrier.node),
        anchor_node=monomial_envelope.node,
        envelope_node=monomial_envelope.node,
        expr=("mul", ("cos", quadratic_carrier.node), monomial_envelope.node),
    )

    assert not validate_bound_closure_bindings(
        closure,
        var_dims=[(0.0,), (0.0,)],
        binding_values={
            "carrier": quadratic_carrier,
            "envelope": monomial_envelope,
        },
    )


def test_seed_blocks_from_basis_state_exposes_block_latents():
    sqrt_xy = ("sqrt", ("mul", ("var", 0), ("var", 1)))
    cos_xy = ("cos", ("mul", ("var", 0), ("var", 1)))
    state = BasisState(
        blocks=(
            FeatureBlock(
                family="basis",
                atoms=(sqrt_xy, cos_xy),
                head_type="linear",
                block_id="basis:block:env",
                latent_bundle_nodes=(sqrt_xy,),
                latent_bundle_roles=("envelope",),
                head_bundle_nodes=(cos_xy, sqrt_xy),
                head_bundle_roles=("wrapped_feature", "companion_term"),
                metadata={"block_expr_obj": sqrt_xy},
            ),
        ),
        compiled_expr=sqrt_xy,
    )
    rows = seed_blocks_from_basis_state(state, var_dims=[(0.0,), (0.0,), (0.0,)])
    assert rows
    assert any(row.node == sqrt_xy for row in rows)
    assert any(row.node == cos_xy for row in rows)
    assert any(":head:wrapped_feature" in str(row.source) for row in rows)
    assert any(":head:companion_term" in str(row.source) for row in rows)
    assert any(str(row.source).startswith("basis:active") for row in rows)
    assert all(str(dict(row.metadata or {}).get("basis_block_id", "")) == "basis:block:env" for row in rows)


def test_seed_blocks_from_basis_state_preserve_parent_block_ids():
    x0 = ("var", 0)
    sqrt_shift = ("sqrt", ("add", ("const", 1.0), ("sqr", x0)))
    state = BasisState(
        blocks=(
            FeatureBlock(
                family="basis",
                atoms=(x0,),
                head_type="linear",
                block_id="basis:block:x0",
                head_bundle_nodes=(x0,),
                head_bundle_roles=("primary",),
                metadata={"block_expr_obj": x0},
            ),
            FeatureBlock(
                family="basis",
                atoms=(sqrt_shift,),
                head_type="linear",
                block_id="basis:block:sqrt_shift",
                parent_block_ids=("basis:block:x0",),
                head_bundle_nodes=(sqrt_shift,),
                head_bundle_roles=("primary",),
                metadata={"block_expr_obj": sqrt_shift},
            ),
        ),
        compiled_expr=sqrt_shift,
    )

    rows = seed_blocks_from_basis_state(state, var_dims=[(0.0,)])
    derived_row = next(row for row in rows if row.node == sqrt_shift)

    assert dict(derived_row.metadata or {}).get("basis_block_id") == "basis:block:sqrt_shift"
    assert dict(derived_row.metadata or {}).get("basis_parent_block_ids") == ["basis:block:x0"]


def test_extend_seed_blocks_with_basis_prepends_active_basis_rows():
    sqrt_xy = ("sqrt", ("mul", ("var", 0), ("var", 1)))
    cos_xy = ("cos", ("mul", ("var", 0), ("var", 1)))
    state = BasisState(
        blocks=(
            FeatureBlock(
                family="basis",
                atoms=(sqrt_xy, cos_xy),
                head_type="linear",
                latent_bundle_nodes=(sqrt_xy,),
                latent_bundle_roles=("envelope",),
                head_bundle_nodes=(cos_xy, sqrt_xy),
                head_bundle_roles=("wrapped_feature", "companion_term"),
                metadata={"block_expr_obj": sqrt_xy},
            ),
        ),
        compiled_expr=sqrt_xy,
    )
    base_rows = seed_anchor_blocks(
        nvars=2,
        pool_nodes=[],
        pool_dims=[],
        var_dims=[(0.0,), (0.0,)],
        max_count=2,
    )
    rows = extend_seed_blocks_with_basis(
        base_rows,
        basis_state=state,
        basis_state_beam=(state,),
        var_dims=[(0.0,), (0.0,)],
        limit=8,
    )
    assert rows
    # Basis seeds are now appended after core seeds (not prepended),
    # so core seeds come first and basis augmentation doesn't crowd them out.
    basis_nodes = {row.node for row in rows if "basis" in str(row.source)}
    assert cos_xy in basis_nodes


def test_operator_enumerator_uses_current_basis_blocks_as_anchor_seeds():
    sqrt_xy = ("sqrt", ("mul", ("var", 0), ("var", 1)))
    cos_xy = ("cos", ("mul", ("var", 0), ("var", 1)))
    state = BasisState(
        blocks=(
            FeatureBlock(
                family="basis",
                atoms=(sqrt_xy, cos_xy),
                head_type="linear",
                latent_bundle_nodes=(sqrt_xy,),
                latent_bundle_roles=("envelope",),
                head_bundle_nodes=(cos_xy, sqrt_xy),
                head_bundle_roles=("wrapped_feature", "companion_term"),
                metadata={"block_expr_obj": sqrt_xy},
            ),
        ),
        compiled_expr=sqrt_xy,
    )
    # Basis-derived seeds only appear in the augmented lane, not the core lane.
    # Use a generous budget so basis-derived anchors are reachable.
    apps = enumerate_operator_applications(
        families=["periodic"],
        nvars=3,
        y_dims=(0.0,),
        var_dims=[(0.0,), (0.0,), (0.0,)],
        pool_nodes=(),
        pool_dims=(),
        anchors_per_family=6,
        max_scaffolds=48,
        basis_state=state,
        basis_state_beam=(state,),
        basis_seed_mode="basis_augmented",
    )
    assert any(app.anchor_node == sqrt_xy for app in apps)


def test_periodic_kind_and_mode_are_recognized_from_bound_closure():
    carrier = ("var", 2)
    app = OperatorApplication(
        family="periodic",
        operator_id="periodic:cos_base",
        scaffold_id="periodic:cos",
        parent_node=("cos", carrier),
        hole_path=(1,),
        target_mode="robust",
        metadata={"form": "cos_base"},
        bound_closure=make_direct_periodic_closure(
            scaffold_id="periodic:cos",
            periodic_kind="cos",
            hole_node=carrier,
            feature_node=("cos", carrier),
            anchor_node=None,
            expr=("cos", carrier),
        ),
    )
    assert direct_periodic_scaffold_kind(app) == "cos"
    assert periodic_scaffold_mode(app) == "base"


def test_direct_planner_resolution_prefers_bound_closure_head_solver():
    app = OperatorApplication(
        family="generic",
        operator_id="generic:closure",
        scaffold_id="generic:closure",
        parent_node=("div", ("const", 1.0), ("const", 1.0)),
        hole_path=(),
        target_mode="full",
        bound_closure=make_direct_rational_closure(
            scaffold_id="rational:affine",
            u_node=("var", 0),
            v_node=("var", 1),
        ),
        metadata={},
    )
    assert direct_rational_scaffold_kind(app) == "affine"
    planner = resolve_direct_operator_planner(app)
    assert planner is not None
    assert planner.planner_id == "fractional_head"
    assert planner.operator_kinds == ("fractional_head",)
    assert planner.composition_modes == (
        "fractional",
        "denominator_companion",
        "numerator_companion",
    )


def test_family_operator_specs_use_generic_operator_kinds():
    affine_kinds = {spec.operator_kind for spec in family_operator_specs("affine")}
    exp_kinds = {spec.operator_kind for spec in family_operator_specs("exp")}
    log_kinds = {spec.operator_kind for spec in family_operator_specs("log")}
    periodic_kinds = {spec.operator_kind for spec in family_operator_specs("periodic")}
    rational_kinds = {spec.operator_kind for spec in family_operator_specs("rational")}
    power_kinds = {spec.operator_kind for spec in family_operator_specs("power")}
    quadratic_kinds = {spec.operator_kind for spec in family_operator_specs("quadratic")}

    assert affine_kinds == {"affine_latent"}
    assert exp_kinds <= {"unary_wrap", "anchored_unary_wrap"}
    assert log_kinds <= {"unary_wrap", "anchored_unary_wrap"}
    assert periodic_kinds == {"harmonic_wrap"}
    assert rational_kinds == {"fractional_head"}
    assert power_kinds == {"power_wrap"}
    assert quadratic_kinds == {"quadratic_wrap"}


def test_operator_specs_expose_composable_roles():
    spec_by_id = {
        spec.operator_id: spec
        for family in ("affine", "periodic", "exp", "log", "rational", "power", "quadratic")
        for spec in family_operator_specs(family)
    }

    assert spec_by_id["affine:latent"].composition_roles == ("affine_latent",)
    assert spec_by_id["affine:latent"].subset_role == "affine_term"

    assert spec_by_id["exp:add"].composition_roles == ("wrapper", "companion")
    assert spec_by_id["exp:add"].anchor_role == "companion"
    assert spec_by_id["exp:mul"].composition_roles == ("wrapper", "prefactor")
    assert spec_by_id["exp:mul"].anchor_role == "prefactor"

    assert spec_by_id["periodic:cos_add"].composition_roles == ("wrapper", "companion")
    assert spec_by_id["periodic:cos_add"].anchor_role == "companion"
    assert spec_by_id["periodic:cos_mul"].composition_roles == ("wrapper", "prefactor")
    assert spec_by_id["periodic:cos_mul"].anchor_role == "prefactor"

    assert spec_by_id["rational:num_over_anchor"].carrier_role == "numerator"
    assert spec_by_id["rational:num_over_anchor"].anchor_role == "denominator_companion"
    assert spec_by_id["rational:anchor_over_den"].carrier_role == "denominator"
    assert spec_by_id["rational:anchor_over_den"].anchor_role == "numerator_companion"
    assert spec_by_id["exp:add"].composition_mode == "companion"
    assert spec_by_id["exp:mul"].composition_mode == "prefactor"
    assert spec_by_id["rational:affine"].composition_mode == "fractional"
    assert spec_by_id["rational:num_over_anchor"].composition_mode == "denominator_companion"
    assert spec_by_id["rational:anchor_over_den"].composition_mode == "numerator_companion"


def test_operator_algebra_registry_is_filtered_by_family_presets():
    from nestynet_sr.sr_search.factorized_search.proposal_families.operator_specs import (
        family_operator_preset_ids,
        operator_algebra_specs,
    )

    registry = {spec.operator_id for spec in operator_algebra_specs()}
    power_preset_ids = set(family_operator_preset_ids("power"))

    assert "affine:latent" in registry
    assert "periodic:cos_mul" in registry
    assert power_preset_ids <= registry
    assert {spec.operator_id for spec in family_operator_specs("power")} == power_preset_ids


def test_closure_search_compatibility_lives_in_explicit_adapter_module():
    assert enumerate_closure_search_specs.__module__.endswith("proposal_families.compat")


def test_operator_application_stores_scaffold_shape_in_explicit_compat_state():
    app = OperatorApplication(
        family="exp",
        operator_id="exp:base",
        scaffold_id="exp:base:x0",
        parent_node=("exp", ("var", 0)),
        hole_path=(1,),
        target_mode="robust",
        anchor_node=None,
        bound_closure=make_direct_linear_wrap_closure(
            scaffold_id="exp:base:x0",
            family="exp",
            wrap_kind="base",
            wrap_op="exp",
            hole_node=("var", 0),
            feature_node=("exp", ("var", 0)),
        ),
        metadata={},
    )

    assert isinstance(app.compat_state, OperatorCompatState)
    assert app.scaffold_id == "exp:base:x0"
    assert app.parent_node == ("exp", ("var", 0))
    rendered = app.to_scaffold_spec()
    assert rendered.scaffold_id == "exp:base:x0"
    assert rendered.parent_node == ("exp", ("var", 0))


def test_operator_application_can_be_native_without_explicit_compat_state():
    app = OperatorApplication(
        family="rational",
        operator_id="rational:affine",
        target_mode="full",
        bound_closure=make_direct_rational_closure(
            scaffold_id="rational:affine",
            u_node=("var", 0),
            v_node=("var", 1),
        ),
        metadata={"operator_kind": "fractional_head", "form": "rational_affine"},
    )

    assert app.compat_state is None
    assert app.scaffold_id == "rational:affine"
    planner = resolve_direct_operator_planner(app)
    assert planner is not None
    assert planner.planner_id == "fractional_head"
    assert planner.operator_kinds == ("fractional_head",)


def test_operator_applications_emit_generic_operator_metadata():
    expected = {
        "affine": {"affine_latent"},
        "periodic": {"harmonic_wrap"},
        "exp": {"unary_wrap", "anchored_unary_wrap"},
        "rational": {"fractional_head"},
        "power": {"power_wrap"},
        "quadratic": {"quadratic_wrap"},
    }

    for family, kinds in expected.items():
        apps = enumerate_operator_applications(
            families=[family],
            nvars=2,
            y_dims=(0.0,),
            var_dims=[(0.0,), (0.0,)],
            pool_nodes=[("mul", ("var", 0), ("var", 1))],
            pool_dims=[(0.0,)],
            anchors_per_family=2,
            max_scaffolds=32,
        )
        assert apps, family
        seen_kinds = {app.metadata.get("operator_kind") for app in apps}
        assert seen_kinds <= kinds
        assert seen_kinds


def test_operator_applications_emit_composition_role_metadata():
    apps = enumerate_operator_applications(
        families=["exp", "periodic", "rational", "affine"],
        nvars=2,
        y_dims=(0.0,),
        var_dims=[(0.0,), (0.0,)],
        pool_nodes=[("mul", ("var", 0), ("var", 1))],
        pool_dims=[(0.0,)],
        anchors_per_family=2,
        max_scaffolds=64,
    )
    affine_apps = enumerate_operator_applications(
        families=["affine"],
        nvars=2,
        y_dims=(0.0,),
        var_dims=[(0.0,), (0.0,)],
        pool_nodes=[("mul", ("var", 0), ("var", 1))],
        pool_dims=[(0.0,)],
        anchors_per_family=2,
        max_scaffolds=16,
    )

    exp_add = next(app for app in apps if app.operator_id == "exp:add")
    periodic_mul = next(app for app in apps if app.operator_id == "periodic:cos_mul")
    rational_num = next(app for app in apps if app.operator_id == "rational:num_over_anchor")
    affine = next(app for app in affine_apps if app.operator_id == "affine:latent")

    assert exp_add.metadata.get("composition_roles") == ["wrapper", "companion"]
    assert exp_add.metadata.get("composition_mode") == "companion"
    assert exp_add.metadata.get("anchor_role") == "companion"
    assert periodic_mul.metadata.get("composition_roles") == ["wrapper", "prefactor"]
    assert periodic_mul.metadata.get("composition_mode") == "prefactor"
    assert periodic_mul.metadata.get("anchor_role") == "prefactor"
    assert rational_num.metadata.get("composition_roles") == ["fractional_head", "denominator_companion"]
    assert rational_num.metadata.get("composition_mode") == "denominator_companion"
    assert rational_num.metadata.get("carrier_role") == "numerator"
    assert rational_num.metadata.get("anchor_role") == "denominator_companion"
    assert affine.metadata.get("composition_roles") == ["affine_latent"]
    assert affine.metadata.get("composition_mode") == "latent"
    assert affine.metadata.get("subset_role") == "affine_term"


def test_build_operator_bound_closure_uses_operator_kind_and_composition_mode():
    exp_bound = build_operator_bound_closure(
        family="exp",
        operator_id="exp:add",
        scaffold_id="exp:add:x0",
        parent_node=("add", ("exp", ("var", 0)), ("var", 1)),
        anchor_node=("var", 1),
        bindings={"carrier": ("var", 0)},
        metadata={
            "operator_kind": "anchored_unary_wrap",
            "composition_mode": "companion",
            "wrap_op": "exp",
        },
    )
    frac_bound = build_operator_bound_closure(
        family="rational",
        operator_id="rational:num_over_anchor",
        scaffold_id="rational:num:x0",
        parent_node=("div", ("var", 0), ("var", 1)),
        anchor_node=("var", 1),
        bindings={"numerator": ("var", 0)},
        metadata={
            "operator_kind": "fractional_head",
            "composition_mode": "denominator_companion",
        },
    )

    assert exp_bound.metadata["exp_kind"] == "add"
    assert exp_bound.spec.head_solver == "linear"
    assert frac_bound.spec.head_solver == "fractional_linear"
    assert frac_bound.bindings["denominator"] == ("var", 1)


def test_operator_applications_bind_real_carriers_for_unary_and_periodic_wraps():
    cases = {
        "periodic": {"periodic:sin_base", "periodic:cos_base"},
        "exp": {"exp:base", "exp:add", "exp:mul"},
        "log": {"log:base", "log:add"},
        "power": {"power:invsqrt", "power:inv", "power:sqrt", "power:sqr"},
    }

    for family, operator_ids in cases.items():
        apps = enumerate_operator_applications(
            families=[family],
            nvars=2,
            y_dims=(0.0,),
            var_dims=[(0.0,), (0.0,)],
            pool_nodes=[("mul", ("var", 0), ("var", 1))],
            pool_dims=[(0.0,)],
            anchors_per_family=2,
            max_scaffolds=32,
        )
        selected = [app for app in apps if app.operator_id in operator_ids]
        assert selected, family
        assert all(app.bound_closure is not None for app in selected)
        assert all("carrier" in app.bindings for app in selected)
        assert all(app.bindings["carrier"].node != ("const", 1.0) for app in selected)
        assert all("__CARRIER__" not in str(app.parent_node) for app in selected)


def test_periodic_operator_apps_prefer_unwrapped_phase_carriers_when_available():
    apps = enumerate_operator_applications(
        families=["periodic"],
        nvars=3,
        y_dims=(0.0,),
        var_dims=[(1.0,), (2.0,), (0.0,)],
        pool_nodes=[("cos", ("var", 2)), ("sin", ("var", 2))],
        pool_dims=[(0.0,), (0.0,)],
        anchors_per_family=4,
        max_scaffolds=48,
    )

    periodic_apps = [app for app in apps if str(app.family) == "periodic"]
    assert periodic_apps
    carrier_nodes = [getattr(app.bindings.get("carrier", None), "node", None) for app in periodic_apps]
    assert ("var", 2) in carrier_nodes
    assert ("cos", ("var", 2)) not in carrier_nodes
    assert ("sin", ("var", 2)) not in carrier_nodes


def test_affine_operator_applications_bind_real_term_sets_up_front():
    apps = enumerate_operator_applications(
        families=["affine"],
        nvars=3,
        y_dims=(0.0,),
        var_dims=[(0.0,), (0.0,), (0.0,)],
        pool_nodes=[("mul", ("var", 0), ("var", 1))],
        pool_dims=[(0.0,)],
        anchors_per_family=3,
        max_scaffolds=24,
    )

    assert apps
    assert all(app.metadata.get("operator_kind") == "affine_latent" for app in apps)
    multi_term = [app for app in apps if len(tuple(app.bindings.get("terms", ()) or ())) >= 2]
    assert multi_term
    for app in multi_term:
        terms = tuple(app.bindings.get("terms", ()) or ())
        assert all(getattr(term, "node", None) not in {None, ("const", 1.0)} for term in terms)
        assert "__CARRIER__" not in str(app.parent_node)
        assert app.bound_closure is not None
        assert len(tuple(app.bound_closure.bindings.get("terms", ()) or ())) == len(terms)


def test_anchored_unary_operator_apps_respect_slot_reuse_policy():
    apps = enumerate_operator_applications(
        families=["exp", "log"],
        nvars=2,
        y_dims=(0.0,),
        var_dims=[(0.0,), (0.0,)],
        pool_nodes=[("mul", ("var", 0), ("var", 1))],
        pool_dims=[(0.0,)],
        anchors_per_family=3,
        max_scaffolds=32,
    )

    anchored = [app for app in apps if app.operator_id in {"exp:add", "exp:mul", "log:add"}]
    assert anchored
    for app in anchored:
        carrier = app.bindings.get("carrier")
        anchor = app.bindings.get("anchor")
        assert carrier is not None
        assert anchor is not None
        assert carrier.node != anchor.node


def test_periodic_prefactor_operator_apps_can_bind_recursive_prefactors():
    apps = enumerate_operator_applications(
        families=["periodic"],
        nvars=3,
        y_dims=(0.0,),
        var_dims=[(0.0,), (0.0,), (0.0,)],
        pool_nodes=[],
        pool_dims=[],
        anchors_per_family=4,
        max_scaffolds=96,
    )

    prefactor_apps = [app for app in apps if app.operator_id in {"periodic:cos_mul", "periodic:sin_mul"}]
    assert prefactor_apps
    assert any(
        getattr(app.bindings.get("envelope", None), "builder", "") in {"product", "monomial"}
        for app in prefactor_apps
    )


def test_exp_operator_apps_reject_quadratic_recursive_carriers_when_quadratic_pool_is_enabled():
    apps = enumerate_operator_applications(
        families=["exp", "quadratic"],
        nvars=2,
        y_dims=(0.0,),
        var_dims=[(0.0,), (0.0,)],
        pool_nodes=[],
        pool_dims=[],
        anchors_per_family=4,
        max_scaffolds=96,
    )

    exp_apps = [app for app in apps if str(app.family) == "exp"]
    assert exp_apps
    assert all(getattr(app.bindings.get("carrier", None), "builder", "") != "quadratic" for app in exp_apps)


def test_quadratic_operator_applications_bind_real_latent_bases_up_front():
    apps = enumerate_operator_applications(
        families=["quadratic"],
        nvars=3,
        y_dims=(0.0,),
        var_dims=[(0.0,), (0.0,), (0.0,)],
        pool_nodes=[("mul", ("var", 0), ("var", 1)), ("sqr", ("var", 2))],
        pool_dims=[(0.0,), (0.0,)],
        anchors_per_family=3,
        max_scaffolds=24,
    )

    assert apps
    sqrt_apps = [app for app in apps if app.operator_id == "quadratic:sqrt"]
    assert sqrt_apps
    assert all(app.bound_closure is not None for app in sqrt_apps)
    assert all(app.bindings.get("carrier", None) is not None for app in sqrt_apps)
    assert all(app.bindings["carrier"].builder == "quadratic" for app in sqrt_apps)
    assert all(app.bindings.get("bases") for app in sqrt_apps)
    assert all(any(base.node != ("const", 1.0) for base in app.bindings["bases"]) for app in sqrt_apps)
    assert all(any(node != ("const", 1.0) for node in app.bound_closure.bindings.get("bases", ())) for app in sqrt_apps)


def test_power_operator_apps_can_surface_recursive_basis_carriers():
    target_expr = ("add", ("const", 1.0), ("sqr", ("var", 0)))
    basis_state = BasisState(
        blocks=(
            FeatureBlock(
                family="affine",
                atoms=(target_expr,),
                head_type="linear",
                latent_bundle_nodes=(target_expr,),
                latent_bundle_roles=("latent",),
                head_bundle_nodes=(target_expr,),
                head_bundle_roles=("primary",),
                active_vars=(0,),
                metadata={"source": "test"},
            ),
        ),
        compiled_expr=target_expr,
    )

    # Basis-derived carriers only appear in the augmented lane.
    apps = enumerate_operator_applications(
        families=["power"],
        nvars=1,
        y_dims=(0.0,),
        var_dims=[(0.0,)],
        pool_nodes=[],
        pool_dims=[],
        anchors_per_family=4,
        max_scaffolds=32,
        basis_state=basis_state,
        basis_seed_mode="basis_augmented",
    )

    sqrt_apps = [app for app in apps if app.operator_id == "power:sqrt"]
    assert sqrt_apps
    # The basis-derived expression should appear as a carrier somewhere in the
    # power family (possibly via a product or recursive seed, not necessarily
    # as a standalone sqrt carrier).
    all_carriers = {
        getattr(app.bindings.get("carrier", None), "node", None)
        for app in apps
        if getattr(app.bindings.get("carrier", None), "node", None) is not None
    }
    assert any("sqr(x0)" in node_str(c) for c in all_carriers)


def test_quadratic_operator_apps_surface_multi_base_latents_under_small_anchor_budget():
    apps = enumerate_operator_applications(
        families=["quadratic"],
        nvars=3,
        y_dims=(0.0,),
        var_dims=[(0.0,), (0.0,), (0.0,)],
        pool_nodes=[],
        pool_dims=[],
        anchors_per_family=1,
        max_scaffolds=12,
    )

    sqrt_apps = [app for app in apps if app.operator_id == "quadratic:sqrt"]
    assert sqrt_apps
    assert any(len(tuple(app.bindings.get("bases", ()) or ())) >= 2 for app in sqrt_apps)
