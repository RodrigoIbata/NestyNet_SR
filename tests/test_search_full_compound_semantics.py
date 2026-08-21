# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import numpy as np
import torch

from nestynet_sr.sr_core import ast_to_human_readable, build_monomial_ast, collect_nn_atoms
from nestynet_sr.sr_core.bridges import AtomNode, ConstNode, FreeConst, MulNode, PowNode, SinNode, Var, clone_ast, get_input_exprs, is_pure_1d_full_compound_ast
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec
from nestynet_sr.sr_core.separability_math import build_linear_ast
from nestynet_sr.sr_search.ast_utils import compact_expression_repr
from nestynet_sr.sr_search.search import (
    _analytic_units_rejection,
    _append_compound_extra_input_asts,
    _atom_compound_cols,
    _build_compound_candidate_ast,
    _build_monomial_ast_from_cols,
    _compound_candidate_new_arity,
    _compound_extra_input_asts_after_prefactor_peel,
    _detect_compound_variable_for_atom,
    _detect_pure_difference_compounds,
    _is_ast_noop_candidate,
    _is_passthrough_noop_candidate,
    _is_pure_1d_full_compound_ast,
    _select_compound_z_variant_shortlist,
)


def _make_compound_nn(inputs):
    return AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2),
        kwargs={"num_segments": 8, "dual_layer": False},
        inputs=inputs,
        tag="nn_test",
    )


def test_is_pure_1d_full_compound_true_for_f_of_z():
    z_ast = build_monomial_ast((0, 1, 2), (1, -1, 1))
    ast = _make_compound_nn((z_ast,))
    assert _is_pure_1d_full_compound_ast(ast, Nxvars=3)
    assert is_pure_1d_full_compound_ast(ast, Nxvars=3)


def test_is_pure_1d_full_compound_false_for_mixed_inputs():
    z_ast = build_monomial_ast((0, 1), (1, -1))
    ast = _make_compound_nn((z_ast, Var(2)))
    assert not _is_pure_1d_full_compound_ast(ast, Nxvars=3)
    assert not is_pure_1d_full_compound_ast(ast, Nxvars=3)


def test_is_pure_1d_full_compound_false_with_outer_prefactor():
    z_ast = build_monomial_ast((0, 1, 2), (1, -1, 1))
    nn_atom = _make_compound_nn((z_ast,))
    ast = MulNode(Var(0), nn_atom)
    assert not _is_pure_1d_full_compound_ast(ast, Nxvars=3)
    assert not is_pure_1d_full_compound_ast(ast, Nxvars=3)


def test_passthrough_noop_candidate_detection():
    z_ast = build_monomial_ast((0, 1), (1, -1))
    atom = _make_compound_nn((z_ast, Var(2)))
    assert _is_passthrough_noop_candidate(atom, clone_ast(z_ast), [2])
    assert not _is_passthrough_noop_candidate(atom, clone_ast(z_ast), [1])
    z_alt = build_monomial_ast((0, 1), (1, 1))
    assert not _is_passthrough_noop_candidate(atom, z_alt, [2])


def test_ast_noop_candidate_detection_before_training():
    z_ast = build_monomial_ast((0, 1), (1, -1))
    atom = _make_compound_nn((z_ast, Var(2)))
    cand_same = _build_compound_candidate_ast(
        atom,
        atom,
        clone_ast(z_ast),
        exponents=(1,),
        extra_var_idxs_override=[2],
    )
    assert _is_ast_noop_candidate(atom, cand_same)

    cand_changed = _build_compound_candidate_ast(
        atom,
        atom,
        clone_ast(z_ast),
        exponents=(1,),
        extra_var_idxs_override=[],
    )
    assert not _is_ast_noop_candidate(atom, cand_changed)


def test_compound_cols_preserve_multiple_compound_inputs():
    z0 = build_monomial_ast((0, 1), (1, -1))
    z1 = build_monomial_ast((2, 3), (1, -1))
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3, 4),
        kwargs={"num_segments": 8, "dual_layer": False},
        inputs=(z0, z1, Var(4)),
        tag="nn_multi_z",
    )

    cols, z_map = _atom_compound_cols(atom)

    assert cols == ["z", "z1", 4]
    assert z_map["z"] is z0
    assert z_map["z1"] is z1
    combined = _build_monomial_ast_from_cols(cols, (1, -1, 0), z_ast=z_map)
    assert combined is not None


def test_append_compound_extra_input_asts_handles_token_map_and_dedups():
    z0 = build_monomial_ast((0, 1), (1, -1))
    z1 = build_monomial_ast((2, 3), (1, -1))
    out = []
    seen = set()

    _append_compound_extra_input_asts(out, {"z": z0, "z1": z1}, seen=seen)
    _append_compound_extra_input_asts(out, z0, seen=seen)

    assert len(out) == 2
    rendered = [ast_to_human_readable(ast) for ast in out]
    assert rendered[0] == ast_to_human_readable(z0)
    assert rendered[1] == ast_to_human_readable(z1)


def test_visible_prefactor_is_removed_from_residual_compound_extras():
    prefactor = build_monomial_ast((0, 1, 2), (1, -1, -1))
    residual = build_monomial_ast((3, 4), (1, -1))
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3, 4),
        inputs=(prefactor, Var(3), Var(4)),
        tag="nn_pb070",
    )

    extras = _compound_extra_input_asts_after_prefactor_peel(
        atom,
        [prefactor, residual],
        prefactor_exponents=(1, 0, 0),
        prefactor_ast=prefactor,
    )

    assert [ast_to_human_readable(expr) for expr in extras] == [
        ast_to_human_readable(residual)
    ]


def test_pb070_prefactor_transaction_builds_p_times_one_dimensional_nn():
    prefactor = build_monomial_ast((0, 1, 2), (1, -1, -1))
    residual = build_monomial_ast((3, 4), (1, -1))
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3, 4),
        inputs=(prefactor, Var(3), Var(4)),
        tag="nn_pb070",
    )

    cand = _build_compound_candidate_ast(
        atom,
        atom,
        residual,
        exponents=(0, 1, -1),
        extra_var_idxs_override=[],
        prefactor_exponents=(1, 0, 0),
        prefactor_ast=prefactor,
        extra_input_asts=[prefactor],
    )

    nn_atoms = collect_nn_atoms(cand)
    assert len(nn_atoms) == 1
    assert len(get_input_exprs(nn_atoms[0])) == 1
    assert ast_to_human_readable(get_input_exprs(nn_atoms[0])[0]) == ast_to_human_readable(
        residual
    )

    us = UnitSystem(("L", "T", "M"))
    units = UnitsSpec(
        unit_system=us,
        x_dims=(
            us.dim({"L": 1}),
            us.dim({"T": 1}),
            us.dim({"M": 1}),
            us.dim({"L": 1}),
            us.dim({"L": 1}),
        ),
        y_dim=us.dim({"L": 1, "T": -1, "M": -1}),
    )
    assert _analytic_units_rejection(cand, units, enforce_units=True) is None


def test_pure_difference_detection_handles_two_compound_tokens(capsys):
    z0 = build_monomial_ast((0, 1), (1, -1))
    z1 = build_monomial_ast((2, 3), (1, -1))
    rng = np.random.default_rng(123)
    x_vals = rng.uniform(0.5, 2.0, size=(256, 2))
    grad = 1.0 + 0.1 * x_vals[:, 0]
    dydx_vals = np.column_stack([
        grad,
        -grad,
    ])

    proposals = _detect_pure_difference_compounds(
        x_vals=x_vals,
        dydx_vals=dydx_vals,
        var_idxs=("z", "z1"),
        precision=0.1,
        z_ast_existing={"z": z0, "z1": z1},
    )

    assert proposals
    captured = capsys.readouterr()
    assert "z - z1" in captured.out


def test_compound_candidate_new_arity_counts_preserved_compound_inputs():
    z_old = build_monomial_ast((0, 1), (1, -1))

    assert _compound_candidate_new_arity(
        extra_var_count=0,
        extra_input_asts=None,
    ) == 1
    assert _compound_candidate_new_arity(
        extra_var_count=0,
        extra_input_asts=[z_old],
    ) == 2
    assert _compound_candidate_new_arity(
        extra_var_count=1,
        extra_input_asts=[z_old],
    ) == 3


def test_bundle_candidate_builds_multiple_compound_inputs():
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3),
        kwargs={"num_segments": 8, "dual_layer": False},
        tag="nn_bundle",
    )
    z01 = build_linear_ast((0, 1), (1, -1))
    z23 = build_linear_ast((2, 3), (1, -1))

    cand = _build_compound_candidate_ast(
        atom,
        atom,
        z01,
        exponents=(1, -1, 1, -1),
        extra_var_idxs_override=[],
        extra_input_asts=[z23],
    )

    assert isinstance(cand, AtomNode)
    assert cand.n_in == 2
    assert cand.raw_var_idxs == (0, 1, 2, 3)


def test_compound_candidate_builder_uses_local_inputs_for_zero_pattern():
    q = build_monomial_ast((2, 3), (1, 1))
    p = build_monomial_ast((0, 1), (1, 1))
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3),
        kwargs={"num_segments": 8, "dual_layer": False},
        tag="nn_pb086_shape",
        inputs=(q, Var(0), Var(1)),
    )

    cand = _build_compound_candidate_ast(
        atom,
        atom,
        p,
        exponents=(0, 1, 1),
        extra_input_asts=[q],
    )

    assert isinstance(cand, AtomNode)
    assert cand.n_in == 2
    rendered_inputs = [ast_to_human_readable(inp) for inp in cand.inputs]
    assert any("x0" in s and "x1" in s for s in rendered_inputs)
    assert any("x2" in s and "x3" in s for s in rendered_inputs)
    assert not any(s.strip() == "x0" for s in rendered_inputs)


def test_compact_repr_shows_multiple_compound_inputs():
    z23 = build_linear_ast((2, 3), (1, -1))
    z01 = build_linear_ast((0, 1), (1, -1))
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3),
        kwargs={"num_segments": 8, "dual_layer": False},
        tag="nn_bundle",
        inputs=(z23, z01),
    )

    rendered = compact_expression_repr(atom, use_color=False)

    assert "z0=" in rendered
    assert "z1=" in rendered
    assert "x2" in rendered and "x3" in rendered
    assert "x0" in rendered and "x1" in rendered


def test_radial_detector_works_on_compound_inputs():
    class RadiusLeaf(torch.nn.Module):
        def forward(self, x):
            return x[:, 0:1] ** 2 + x[:, 1:2] ** 2

        def grad(self, cache):
            return (2.0 * cache["x"]).unsqueeze(1)

    rng = np.random.default_rng(2468)
    x_raw = torch.tensor(
        rng.uniform(0.5, 3.0, size=(256, 4)),
        dtype=torch.float64,
    )
    y_dummy = torch.zeros((x_raw.shape[0], 1), dtype=torch.float64)

    z23 = build_linear_ast((2, 3), (1, -1))
    z01 = build_linear_ast((0, 1), (1, -1))
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3),
        kwargs={"num_segments": 8, "dual_layer": False},
        tag="nn_radius",
        inputs=(z23, z01),
    )

    proposals, _ = _detect_compound_variable_for_atom(
        model=object(),
        atom=atom,
        leaf=RadiusLeaf(),
        datagen_train=[(x_raw, y_dummy)],
        device=torch.device("cpu"),
        max_exponent=2,
        precision=0.05,
        max_batches=1,
        enable_linear=False,
        enable_radial=True,
        enable_shift=False,
        enable_mixed_compound=False,
        trig_axis_specs=None,
        scaling_features=None,
    )

    radial = [
        p for p in proposals
        if len(p) >= 5 and (p[4] or {}).get("kind") == "radial"
    ]
    assert radial
    rendered = ast_to_human_readable(radial[0][1])
    assert "x2" in rendered and "x3" in rendered
    assert "x0" in rendered and "x1" in rendered


def test_radial_variant_shortlist_retains_z_and_sqrt_under_screen_cap():
    variants = [
        ("z", Var(0), 0.12),
        ("sqrt", PowNode(Var(0), 0.5), 0.10),
        ("rat_inv_zp1", PowNode(Var(0), -1.0), 0.99),
        ("rat_inv", PowNode(Var(0), -1.0), 0.98),
        ("rat_inv_z2p1", PowNode(Var(0), -2.0), 0.97),
    ]

    kept = _select_compound_z_variant_shortlist(
        variants,
        kind="radial",
        screen_gate=0.15,
        max_variants_to_try=3,
    )

    assert [name for name, _, _ in kept] == ["z", "sqrt", "rat_inv_zp1"]


def test_nonradial_variant_shortlist_uses_screened_ranking():
    variants = [
        ("z", Var(0), 0.12),
        ("sqrt", PowNode(Var(0), 0.5), 0.10),
        ("rat_inv_zp1", PowNode(Var(0), -1.0), 0.99),
        ("rat_inv", PowNode(Var(0), -1.0), 0.98),
    ]

    kept = _select_compound_z_variant_shortlist(
        variants,
        kind="monomial",
        screen_gate=0.15,
        max_variants_to_try=2,
    )

    assert [name for name, _, _ in kept] == ["rat_inv_zp1", "rat_inv"]


def test_analytic_units_gate_rejects_numeric_trig_of_unitful_variable():
    us = UnitSystem(("L", "T"))
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim({"L": 1}),),
        y_dim=us.dimless(),
    )
    expr = SinNode(MulNode(ConstNode(1.0), Var(0)))

    reason = _analytic_units_rejection(expr, spec, enforce_units=True)

    assert reason is not None


def test_analytic_units_gate_allows_dimensionless_trig_ratio():
    us = UnitSystem(("L", "T"))
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim({"L": 1}), us.dim({"L": 1})),
        y_dim=us.dimless(),
    )
    ratio = MulNode(Var(0), PowNode(Var(1), -1.0))
    expr = SinNode(ratio)

    assert _analytic_units_rejection(expr, spec, enforce_units=True) is None


def test_analytic_units_gate_allows_explicit_declared_unitful_frequency():
    us = UnitSystem(("L", "T"))
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim({"L": 1}),),
        y_dim=us.dimless(),
        free_const_dims={"k": us.dim({"L": -1})},
    )
    expr = SinNode(MulNode(FreeConst("k"), Var(0)))

    assert _analytic_units_rejection(expr, spec, enforce_units=True) is None


def test_stagea_search_path_consumes_real_general_affine_proposal():
    from nestynet_sr.sr_gs import GeneralizedSymmetryConfig

    class ObliqueLeaf(torch.nn.Module):
        def forward(self, x):
            z = np.sqrt(2.0) * x[:, 0:1] - x[:, 1:2]
            return torch.sin(z)

        def grad(self, cache):
            x = cache["x"]
            z = np.sqrt(2.0) * x[:, 0] - x[:, 1]
            c = torch.cos(z)
            return torch.stack([np.sqrt(2.0) * c, -c], dim=1).unsqueeze(1)

    rng = np.random.default_rng(97531)
    x_raw = torch.tensor(rng.normal(size=(512, 2)), dtype=torch.float64)
    y_dummy = torch.zeros((x_raw.shape[0], 1), dtype=torch.float64)
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1),
        kwargs={"num_segments": 8, "dual_layer": False},
        tag="nn_gs_oblique",
    )
    cfg = GeneralizedSymmetryConfig(
        enabled=True,
        mode="propose",
        policy="augment",
        known_generators=False,
        known_lie=False,
        general_affine=True,
        translations=False,
        diagonal_translations=False,
        scalings=False,
        rotations=False,
        output_equivariance=False,
        residual_tol=1.0e-8,
        audit_residual_tol=1.0e-6,
        min_confidence=0.5,
    )

    proposals, _ = _detect_compound_variable_for_atom(
        model=object(),
        atom=atom,
        leaf=ObliqueLeaf(),
        datagen_train=[(x_raw, y_dummy)],
        device=torch.device("cpu"),
        max_batches=1,
        enable_linear=False,
        enable_radial=False,
        enable_shift=False,
        enable_mixed_compound=False,
        enable_retained_axis_wrappers=False,
        trig_axis_specs=None,
        scaling_features=None,
        invariance_features=None,
        gs_cfg=cfg,
    )

    gs_rows = [p for p in proposals if len(p) >= 5 and (p[4] or {}).get("source") == "generalized_symmetry"]
    row = next(p for p in gs_rows if (p[4] or {}).get("gs_family") == "general_affine")
    meta = row[4] or {}
    assert meta["gs_kind"] == "linear_distribution_annihilator"
    assert meta["gs_promotion_state"] == "promoted"

    # f = sin(sqrt(2)*x0 - x1) is constant along v ∝ (1, sqrt(2)), so the orbit is a
    # line and the quotient has codimension 1.
    reduction = meta["gs_reduction"]
    assert reduction["generic_orbit_rank"] == 1
    assert reduction["quotient_codimension"] == 1

    # The annihilating covector spans the orthogonal complement of v, so the recovered
    # invariant coordinate is proportional to sqrt(2)*x0 - x1: the argument of the sine.
    covector = np.asarray(meta["gs_linear_covector"], dtype=float)
    assert np.allclose(covector / covector[1], np.asarray([-np.sqrt(2.0), 1.0]), atol=1.0e-9)

    rendered = ast_to_human_readable(row[1])
    assert "x0" in rendered and "x1" in rendered
