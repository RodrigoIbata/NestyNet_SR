# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch

import nestynet_sr.sr_search.factorized_search.basis_state as basis_mod
from nestynet_sr.sr_search.factorized_search.basis_head import fit_basis_state_head
import nestynet_sr.sr_search.factorized_search.closures as closure_mod
import nestynet_sr.sr_search.factorized_search.closure_search_compat as scaffold_mod
from nestynet_sr.sr_search.factorized_search.proposal_families.compat import operator_application_from_scaffold


def test_feature_block_and_basis_state_from_closure_candidate():
    hole = ("mul", ("var", 0), ("var", 1))
    anchor = ("mul", ("var", 0), ("var", 1))
    expr = ("add", ("cos", hole), anchor)

    block = basis_mod.feature_block_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:cos_add:(x0*x1)",
        expr=expr,
        anchor_node=anchor,
        scaffold_metadata={"form": "cos_add"},
        local_mapping_kind="direct_linear_head",
        local_mapping_coeffs=[1.0, 1.0, 0.0],
        direct_metadata={
            "feature_kind": "cos",
            "hole_node": hole,
            "feature_node": ("cos", hole),
        },
    )
    state = basis_mod.basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:cos_add:(x0*x1)",
        expr=expr,
        anchor_node=anchor,
        scaffold_metadata={"form": "cos_add"},
        local_fit_mse=1.0e-9,
        local_probe_mse=2.0e-9,
        local_mapping_kind="direct_linear_head",
        local_mapping_coeffs=[1.0, 1.0, 0.0],
        direct_metadata={
            "feature_kind": "cos",
            "hole_node": hole,
            "feature_node": ("cos", hole),
        },
    )

    assert block.family == "periodic"
    assert block.head_type == "linear"
    assert block.active_vars == (0, 1)
    assert [scaffold_mod.node_str(atom) for atom in block.atoms] == ["cos((x0*x1))", "(x0*x1)"]
    assert block.to_dict()["atom_exprs"] == ["cos((x0*x1))", "(x0*x1)"]
    latent_bundle = {
        (row["role"], row["expr"])
        for row in block.to_dict()["latent_bundle"]
    }
    head_bundle = {
        (row["role"], row["expr"])
        for row in block.to_dict()["head_bundle"]
    }
    assert ("feature", "cos((x0*x1))") in latent_bundle
    assert ("carrier", "(x0*x1)") in latent_bundle
    assert ("companion", "(x0*x1)") in latent_bundle
    assert ("wrapped_feature", "cos((x0*x1))") in head_bundle
    assert ("companion_term", "(x0*x1)") in head_bundle

    state_dict = state.to_dict()
    assert state.blocks == (block,)
    assert float(state.fit_loss) == 1.0e-9
    assert float(state.probe_loss) == 2.0e-9
    assert state_dict["block_count"] == 1
    assert state_dict["block_families"] == ["periodic"]
    assert state_dict["compiled_expr"] == "((x0*x1)+cos((x0*x1)))"

    closure = closure_mod.bound_closure_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:cos_add:(x0*x1)",
        expr=expr,
        anchor_node=anchor,
        scaffold_metadata={"form": "cos_add"},
        direct_metadata={
            "feature_kind": "cos",
            "hole_node": hole,
            "feature_node": ("cos", hole),
        },
    )
    assert closure.spec.family == "periodic"
    assert closure.spec.head_solver == "harmonic_linear"
    assert closure.to_dict()["bindings"]["carrier"] == "(x0*x1)"


def test_feature_block_retains_harmonic_latent_bundle_channels():
    hole = ("mul", ("var", 0), ("var", 1))
    envelope = ("sqrt", ("mul", ("var", 0), ("var", 1)))
    companion = ("add", ("var", 0), ("var", 1))
    expr = ("add", ("mul", envelope, ("cos", hole)), companion)
    bound_closure = closure_mod.make_direct_periodic_closure(
        scaffold_id="periodic:cos_mul:sqrtxy",
        periodic_kind="cos",
        hole_node=hole,
        feature_node=("cos", hole),
        anchor_node=envelope,
        envelope_node=envelope,
        companion_nodes=(companion,),
        harmonic_feature_nodes=(("cos", hole), ("sin", hole)),
        expr=expr,
    )

    block = basis_mod.feature_block_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:cos_mul:sqrtxy",
        expr=expr,
        anchor_node=envelope,
        scaffold_metadata={"form": "cos_mul"},
        local_mapping_kind="direct_harmonic_head",
        local_mapping_coeffs=[1.0, 0.0, 1.0, 0.0],
        direct_metadata={
            "feature_kind": "cos",
            "hole_node": hole,
            "feature_node": ("cos", hole),
            "harmonic_feature_nodes": [("cos", hole), ("sin", hole)],
            "envelope_node": envelope,
            "companion_nodes": [companion],
        },
        bound_closure=bound_closure,
    )

    bundle_rows = {
        (row["role"], row["expr"])
        for row in block.to_dict()["latent_bundle"]
    }
    head_bundle_rows = {
        (row["role"], row["expr"])
        for row in block.to_dict()["head_bundle"]
    }
    assert ("carrier", "(x0*x1)") in bundle_rows
    assert ("harmonic_feature", "cos((x0*x1))") in bundle_rows
    assert ("harmonic_feature", "sin((x0*x1))") in bundle_rows
    assert ("envelope", "sqrt((x0*x1))") in bundle_rows
    assert ("companion", "(x0+x1)") in bundle_rows
    assert ("harmonic_term", "(sqrt((x0*x1))*cos((x0*x1)))") in head_bundle_rows
    assert ("harmonic_term", "(sqrt((x0*x1))*sin((x0*x1)))") in head_bundle_rows
    assert ("companion_term", "(x0+x1)") in head_bundle_rows


def test_ensure_feature_block_head_bundle_upgrades_legacy_block():
    expr = ("sqrt", ("var", 0))
    legacy = basis_mod.FeatureBlock(
        family="basis",
        atoms=(expr,),
        head_type="linear",
        metadata={"block_expr_obj": expr},
    )

    upgraded = basis_mod.ensure_feature_block_head_bundle(legacy)

    assert upgraded is not None
    assert upgraded.head_bundle_nodes == (expr,)
    assert upgraded.head_bundle_roles == ("primary",)
    assert upgraded.metadata["head_bundle_exprs"] == ["sqrt(x0)"]


def test_power_square_variants_have_support_id_and_masked_basis_identity():
    carrier = ("cos", ("mul", ("var", 1), ("var", 2)))
    anchor = ("mul", ("var", 0), ("var", 3))
    full_expr = ("mul", anchor, ("add", carrier, ("sqr", carrier)))
    square_expr = ("mul", anchor, ("sqr", carrier))

    full_block = basis_mod.feature_block_from_closure_candidate(
        family="power",
        scaffold_id="power:sqr_mul:debug",
        expr=full_expr,
        anchor_node=anchor,
        scaffold_metadata={"form": "sqr_mul"},
        local_mapping_kind="direct_power_head",
        local_mapping_coeffs=[0.0, 0.5, 1.0],
        direct_metadata={
            "power_kind": "sqr_mul",
            "power_exponent": 2.0,
            "power_variant": "full_quadratic",
            "hole_node": carrier,
            "power_inner_node": carrier,
            "anchor_node": anchor,
        },
        bound_closure=closure_mod.make_direct_power_closure(
            scaffold_id="power:sqr_mul:debug",
            power_kind="sqr_mul",
            exponent=2.0,
            hole_node=carrier,
            anchor_node=anchor,
        ),
    )
    square_block = basis_mod.feature_block_from_closure_candidate(
        family="power",
        scaffold_id="power:sqr_mul:debug",
        expr=square_expr,
        anchor_node=anchor,
        scaffold_metadata={"form": "sqr_mul"},
        local_mapping_kind="direct_power_head",
        local_mapping_coeffs=[0.0, 0.0, 1.0],
        direct_metadata={
            "power_kind": "sqr_mul",
            "power_exponent": 2.0,
            "power_variant": "square_only",
            "hole_node": carrier,
            "power_inner_node": carrier,
            "anchor_node": anchor,
        },
        bound_closure=closure_mod.make_direct_power_closure(
            scaffold_id="power:sqr_mul:debug",
            power_kind="sqr_mul",
            exponent=2.0,
            hole_node=carrier,
            anchor_node=anchor,
        ),
    )
    linear_block = basis_mod.feature_block_from_closure_candidate(
        family="power",
        scaffold_id="power:sqr_mul:debug",
        expr=full_expr,
        anchor_node=anchor,
        scaffold_metadata={"form": "sqr_mul"},
        local_mapping_kind="direct_power_head",
        local_mapping_coeffs=[0.0, 0.5, 1.0],
        direct_metadata={
            "power_kind": "sqr_mul",
            "power_exponent": 2.0,
            "power_variant": "linear_square",
            "hole_node": carrier,
            "power_inner_node": carrier,
            "anchor_node": anchor,
        },
        bound_closure=closure_mod.make_direct_power_closure(
            scaffold_id="power:sqr_mul:debug",
            power_kind="sqr_mul",
            exponent=2.0,
            hole_node=carrier,
            anchor_node=anchor,
        ),
    )

    assert full_block.metadata["support_id"] == square_block.metadata["support_id"]
    assert basis_mod.feature_block_id(full_block) != basis_mod.feature_block_id(square_block)

    full_head = {
        (row["role"], row["expr"])
        for row in full_block.to_dict()["head_bundle"]
    }
    square_head = {
        (row["role"], row["expr"])
        for row in square_block.to_dict()["head_bundle"]
    }
    assert ("power_bias_term", "(x0*x3)") in full_head
    assert ("power_linear_term", "((x0*x3)*cos((x1*x2)))") in full_head or ("power_linear_term", "((cos((x1*x2))*x0)*x3)") in full_head
    assert ("power_square_term", "((x0*x3)*sqr(cos((x1*x2))))") in square_head or ("power_square_term", "((sqr(cos((x1*x2)))*x0)*x3)") in square_head
    assert {role for role, _expr in square_head} == {"power_square_term"}

    square_state = basis_mod.BasisState(
        blocks=(square_block,),
        fit_loss=1.0,
        probe_loss=1.0,
        complexity=square_block.complexity(),
        compiled_expr=square_expr,
    )

    assert basis_mod.basis_state_covers_feature_block(square_state, square_block) is True
    assert basis_mod.basis_state_covers_feature_block(square_state, linear_block) is False


def test_power_decomposed_head_refits_on_rebound_carrier_terms():
    x = torch.tensor(
        [
            [1.10, 1.15, 1.20, 1.25],
            [1.20, 1.25, 1.30, 1.35],
            [1.30, 1.35, 1.40, 1.45],
            [1.40, 1.45, 1.50, 1.55],
            [1.50, 1.55, 1.60, 1.65],
            [1.60, 1.65, 1.70, 1.75],
        ],
        dtype=torch.float64,
    )
    carrier = ("cos", ("mul", ("var", 1), ("var", 2)))
    anchor = ("mul", ("var", 0), ("var", 3))
    h = torch.cos(x[:, 1] * x[:, 2])
    y = (x[:, 0] * x[:, 3] * (0.5 * h + h * h)).unsqueeze(-1)
    expr = ("mul", ("add", ("mul", ("const", 0.5), carrier), ("sqr", carrier)), anchor)

    state = basis_mod.basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:sqr_mul:debug",
        expr=expr,
        anchor_node=anchor,
        scaffold_metadata={"form": "sqr_mul"},
        local_fit_mse=0.1,
        local_probe_mse=0.1,
        local_mapping_kind="direct_power_head",
        local_mapping_coeffs=[0.0, 0.5, 1.0],
        direct_metadata={
            "power_kind": "sqr_mul",
            "power_exponent": 2.0,
            "power_variant": "full_quadratic",
            "hole_node": carrier,
            "power_inner_node": carrier,
            "anchor_node": anchor,
        },
        bound_closure=closure_mod.make_direct_power_closure(
            scaffold_id="power:sqr_mul:debug",
            power_kind="sqr_mul",
            exponent=2.0,
            hole_node=carrier,
            anchor_node=anchor,
        ),
    )

    scored = fit_basis_state_head(
        state,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        route_name="unit_power_rebound",
    )

    assert scored is not None
    compiled = scaffold_mod.node_str(scored.compiled_expr)
    assert "cos((x1*x2))" in compiled
    assert "sqr((x1*x2))+sqr(x3)" not in compiled
    assert float(scored.probe_loss) < 0.3


def test_run_closure_search_pass_enriches_candidate_rows_with_basis_snapshots(monkeypatch):
    anchor = ("mul", ("var", 0), ("var", 1))
    hole = ("mul", ("var", 0), ("var", 1))
    expr = ("add", ("cos", hole), anchor)
    spec = scaffold_mod.OuterScaffoldSpec(
        family="periodic",
        scaffold_id="periodic:cos_add:(x0*x1)",
        parent_node=("add", ("cos", ("const", 1.0)), anchor),
        hole_path=(1, 1),
        target_mode="robust",
        anchor_node=anchor,
        metadata={"form": "cos_add"},
    )

    def _fake_enumerate_closure_search_specs(**kwargs):
        return [spec]

    def _fake_direct_periodic(*args, **kwargs):
        return (
            [
                {
                    "expr": expr,
                    "child_key": scaffold_mod.node_str(expr),
                    "local_probe_mse": 1.0e-6,
                    "local_fit_mse": 1.0e-6,
                    "local_mapping_kind": "direct_linear_head",
                    "local_mapping_coeffs": [1.0, 1.0, 0.0],
                    "candidate_child_size": scaffold_mod.node_size(expr),
                    "direct_metadata": {
                        "feature_kind": "cos",
                        "hole_node": hole,
                        "feature_node": ("cos", hole),
                    },
                }
            ],
            "direct_ok",
            {},
        )

    op_app = operator_application_from_scaffold(spec)

    def _fake_enumerate_operator_applications(**kwargs):
        return [op_app]

    monkeypatch.setattr(scaffold_mod, "enumerate_operator_applications", _fake_enumerate_operator_applications)
    monkeypatch.setattr(scaffold_mod, "_solve_direct_operator_preview_rows", _fake_direct_periodic)

    x = torch.linspace(0.2, 0.8, steps=8, dtype=torch.float64).unsqueeze(1).repeat(1, 2)
    y = torch.ones((8, 1), dtype=torch.float64)
    ret = scaffold_mod.run_closure_search_pass(
        families=["periodic"],
        nvars=2,
        max_scaffolds=1,
        anchors_per_family=1,
        max_depth=4,
        poly_degree=2,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        var_dims=[(0.0,), (0.0,)],
        y_dims=(0.0,),
        pool_nodes=[anchor],
        pool_phi_fit=torch.zeros((8, 1), dtype=torch.float64),
        pool_phi_probe=torch.zeros((8, 1), dtype=torch.float64),
        pool_dims=[(0.0,)],
        safe_eps=1.0e-6,
        preview_topk=1,
        beam_cfg={},
        solver_kwargs={"enum_max_depth": 3, "enum_max_trees": 16},
    )

    rows = list(ret.get("candidate_rows", []) or [])
    assert len(rows) == 1
    row = rows[0]
    assert row["feature_block_obj"].family == "periodic"
    assert row["feature_block_dict"]["atom_exprs"] == ["cos((x0*x1))", "(x0*x1)"]
    assert row["basis_state_dict"]["block_count"] == 1
    assert row["basis_state_dict"]["compiled_expr"] == "((x0*x1)+cos((x0*x1)))"
    assert row["proposal_candidate_obj"].family == "periodic"
    assert row["proposal_candidate_obj"].bound_closure.spec.head_solver == "harmonic_linear"
    assert row["proposal_candidate_dict"]["scaffold_id"] == "periodic:cos_add:(x0*x1)"
    assert row["proposal_candidate_dict"]["feature_block"]["atom_exprs"] == ["cos((x0*x1))", "(x0*x1)"]
    assert row["proposal_candidate_dict"]["bound_closure"]["spec"]["closure_id"] == "periodic:cos:harmonic_linear"


def test_basis_state_extend_beam_and_cover_checks():
    block_a = basis_mod.feature_block_from_closure_candidate(
        family="power",
        scaffold_id="power:sqrt",
        expr=("sqrt", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "power_sqrt"},
        local_mapping_kind="direct_power_head",
        direct_metadata={"hole_node": ("var", 0), "power_inner_node": ("var", 0)},
    )
    block_b = basis_mod.feature_block_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:cos",
        expr=("cos", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_mapping_kind="direct_linear_head",
        direct_metadata={"hole_node": ("var", 1), "feature_node": ("cos", ("var", 1))},
    )
    state_a = basis_mod.BasisState(
        blocks=(block_a,),
        fit_loss=0.2,
        probe_loss=0.2,
        complexity=block_a.complexity(),
        compiled_expr=("sqrt", ("var", 0)),
    )
    state_b = basis_mod.BasisState(
        blocks=(block_b,),
        fit_loss=0.1,
        probe_loss=0.1,
        complexity=block_b.complexity(),
        compiled_expr=("cos", ("var", 1)),
    )

    merged = basis_mod.basis_state_extend(
        state_a,
        state_b,
        route_name="test_merge",
        fit_loss=0.05,
        probe_loss=0.05,
        compiled_expr=("add", ("sqrt", ("var", 0)), ("cos", ("var", 1))),
    )
    assert merged is not None
    assert len(merged.blocks) == 2
    assert basis_mod.basis_state_covers_feature_block(merged, block_a) is True
    assert basis_mod.basis_state_covers_feature_block(merged, block_b) is True

    beam = basis_mod.admit_basis_state_to_beam([], state_a, beam_width=2)
    beam = basis_mod.admit_basis_state_to_beam(beam, merged, beam_width=2)
    beam = basis_mod.admit_basis_state_to_beam(beam, state_b, beam_width=2)
    assert len(beam) == 2
    assert float(beam[0].probe_loss) <= float(beam[1].probe_loss)


def test_admit_basis_state_to_beam_prunes_dominated_superset():
    block_a = basis_mod.feature_block_from_closure_candidate(
        family="power",
        scaffold_id="power:sqrt",
        expr=("sqrt", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "power_sqrt"},
        local_mapping_kind="direct_power_head",
        direct_metadata={"hole_node": ("var", 0), "power_inner_node": ("var", 0)},
    )
    block_b = basis_mod.feature_block_from_closure_candidate(
        family="log",
        scaffold_id="log:base",
        expr=("log", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "log_base"},
        local_mapping_kind="direct_linear_head",
        direct_metadata={"hole_node": ("var", 1), "feature_node": ("log", ("var", 1))},
    )
    base_state = basis_mod.BasisState(
        blocks=(block_a,),
        fit_loss=0.01,
        probe_loss=0.01,
        complexity=block_a.complexity(),
        compiled_expr=("sqrt", ("var", 0)),
    )
    superset_state = basis_mod.basis_state_extend(
        base_state,
        basis_mod.BasisState(
            blocks=(block_b,),
            fit_loss=0.010000000001,
            probe_loss=0.010000000001,
            complexity=block_b.complexity(),
            compiled_expr=("log", ("var", 1)),
        ),
        route_name="test_superset",
        fit_loss=0.010000000001,
        probe_loss=0.010000000001,
        compiled_expr=("add", ("sqrt", ("var", 0)), ("log", ("var", 1))),
    )

    beam = basis_mod.admit_basis_state_to_beam([], superset_state, beam_width=3)
    beam = basis_mod.admit_basis_state_to_beam(beam, base_state, beam_width=3)

    assert len(beam) == 1
    assert beam[0].to_dict()["block_count"] == 1
