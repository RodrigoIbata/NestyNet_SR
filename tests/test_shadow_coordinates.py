# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from types import SimpleNamespace

from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    ConstNode,
    CosNode,
    ExpNode,
    LogNode,
    MulNode,
    PowNode,
    SinNode,
    Var,
    ast_to_human_readable,
)
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec
from nestynet_sr.sr_search.compound_proposals import build_logexp_compound_proposals
from nestynet_sr.sr_search.search import (
    _stageA_compound_variant_shadow_only,
    _stageA_record_logexp_shadows,
    _stageA_reset_shadow_registry,
    _stageA_shadow_ast_present_in_inputs,
    _stageA_shadow_composite_proposals,
    _stageA_shadow_preserved_factor_proposals,
    _stageA_shadow_promotion_audit,
    _stageA_shadow_promotion_payoff_reason,
    _stageA_shadow_registry,
    _stageA_shadow_trig_factor_peel_proposals,
    _stageA_shadow_trig_composite_proposals,
    _stageA_sync_shadow_registry,
)
from nestynet_sr.sr_search.shadow_coordinates import (
    ShadowCoordinate,
    ShadowRegistry,
    shadow_parent_key,
)


def test_shadow_registry_merges_duplicate_leaf_local_coordinates():
    atom = AtomNode(kind="nn", var_idxs=(0,), tag="leaf0")
    reg = ShadowRegistry()
    s1 = ShadowCoordinate(
        parent_key=shadow_parent_key(atom),
        parent_atom_tag="leaf0",
        base_ast=Var(0),
        shadow_ast=SinNode(Var(0)),
        transform_kind="sin",
        source="oracle",
        confidence=0.7,
        evidence={"a": 1},
    )
    s2 = ShadowCoordinate(
        parent_key=shadow_parent_key(atom),
        parent_atom_tag="leaf0",
        base_ast=Var(0),
        shadow_ast=SinNode(Var(0)),
        transform_kind="sin",
        source="oracle",
        confidence=0.9,
        evidence={"b": 2},
    )

    _, created1 = reg.add(s1)
    stored, created2 = reg.add(s2)

    assert created1
    assert not created2
    assert stored.confidence == 0.9
    assert stored.evidence["a"] == 1
    assert stored.evidence["b"] == 2
    assert len(reg.local_for(shadow_parent_key(atom))) == 1
    assert reg.global_for_ast(SinNode(Var(0))) is not None


def test_shadow_registry_prunes_deleted_and_incompatible_leaf_scopes():
    atom_keep = AtomNode(kind="nn", var_idxs=(0, 4), tag="leaf_keep")
    atom_drop = AtomNode(kind="nn", var_idxs=(0, 9), tag="leaf_drop")
    atom_missing = AtomNode(kind="nn", var_idxs=(2,), tag="leaf_missing")
    reg = ShadowRegistry()
    reg.add(
        ShadowCoordinate(
            parent_key=shadow_parent_key(atom_keep),
            parent_atom_tag="leaf_keep",
            base_ast=Var(4),
            shadow_ast=SinNode(Var(4)),
            transform_kind="sin",
            source="oracle",
            confidence=0.9,
        )
    )
    reg.add(
        ShadowCoordinate(
            parent_key=shadow_parent_key(atom_drop),
            parent_atom_tag="leaf_drop",
            base_ast=Var(9),
            shadow_ast=SinNode(Var(9)),
            transform_kind="sin",
            source="oracle",
            confidence=0.8,
        )
    )
    reg.add(
        ShadowCoordinate(
            parent_key=shadow_parent_key(atom_missing),
            parent_atom_tag="leaf_missing",
            base_ast=Var(2),
            shadow_ast=SinNode(Var(2)),
            transform_kind="sin",
            source="oracle",
            confidence=0.7,
        )
    )

    removed_parents, removed_shadows = reg.prune_for_live_parent_vars(
        {
            shadow_parent_key(atom_keep): (0, 4),
            shadow_parent_key(atom_drop): (0,),
        }
    )

    assert removed_parents == 2
    assert removed_shadows == 2
    assert len(reg.local_for(shadow_parent_key(atom_keep))) == 1
    assert reg.local_for(shadow_parent_key(atom_drop)) == []
    assert reg.local_for(shadow_parent_key(atom_missing)) == []
    assert reg.global_for_ast(SinNode(Var(4))) is not None
    assert reg.global_for_ast(SinNode(Var(9))) is None


def test_stageA_shadow_registry_reset_scopes_independent_runs():
    hp = SimpleNamespace()
    atom = AtomNode(kind="nn", var_idxs=(0, 1), tag="leaf_reused")
    reg = _stageA_shadow_registry(hp)
    reg.add(
        ShadowCoordinate(
            parent_key=shadow_parent_key(atom),
            parent_atom_tag="leaf_reused",
            base_ast=Var(1),
            shadow_ast=SinNode(Var(1)),
            transform_kind="sin",
            source="old_run",
            confidence=0.99,
        )
    )
    assert reg.count() == 1

    fresh = _stageA_reset_shadow_registry(hp, reason="unit test")

    assert fresh.count() == 0
    assert _stageA_shadow_registry(hp) is fresh
    proposals = [
        ((1, 0), Var(0), 0.95, [1], {"kind": "monomial"}),
    ]
    assert (
        _stageA_shadow_composite_proposals(
            proposals,
            atom=atom,
            cols=[0, 1],
            shadow_registry=fresh,
        )
        == []
    )


def test_stageA_shadow_sync_prunes_consumed_coordinates():
    hp = SimpleNamespace()
    consumed = SinNode(Var(1))
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1),
        tag="leaf_consumed_sync",
        inputs=(MulNode(Var(0), consumed), Var(1)),
    )
    reg = _stageA_shadow_registry(hp)
    reg.add(
        ShadowCoordinate(
            parent_key=shadow_parent_key(atom),
            parent_atom_tag="leaf_consumed_sync",
            base_ast=Var(1),
            shadow_ast=SinNode(Var(1)),
            transform_kind="sin",
            source="oracle",
            confidence=0.99,
        )
    )
    assert reg.count() == 1

    _stageA_sync_shadow_registry(hp, atom, reason="unit test")

    assert reg.count() == 0


def test_stageA_logexp_proposals_are_recorded_as_shadows_only():
    us = UnitSystem(("L",))
    dimless = us.dimless()
    spec = UnitsSpec(unit_system=us, x_dims=(dimless,), y_dim=dimless)
    atom = AtomNode(kind="nn", var_idxs=(0,), tag="leaf0")
    props = build_logexp_compound_proposals(
        atom,
        units_spec=spec,
        wrappers=("log", "exp"),
        max_proposals=4,
    )
    assert any(isinstance(p.z_ast, LogNode) for p in props)
    assert any(isinstance(p.z_ast, ExpNode) for p in props)

    hp = SimpleNamespace()
    reg = _stageA_shadow_registry(hp)
    _stageA_record_logexp_shadows(
        atom=atom,
        proposals=props,
        shadow_registry=reg,
        units_spec=spec,
        enforce_units=True,
    )

    shadows = reg.local_for(shadow_parent_key(atom))
    assert len(shadows) == len(props)
    assert {s.transform_kind for s in shadows} == {"log", "exp"}
    assert {s.source for s in shadows} == {"stageA_logexp"}


def test_stageA_logexp_shadow_recomputes_strict_unit_status():
    us = UnitSystem(("L",))
    length = us.dim((1,))
    dimless = us.dimless()
    spec = UnitsSpec(unit_system=us, x_dims=(length,), y_dim=dimless)
    atom = AtomNode(kind="nn", var_idxs=(0,), tag="leaf_unitful_log")
    prop = SimpleNamespace(
        z_ast=LogNode(Var(0)),
        base_ast=Var(0),
        wrapper="log",
        family="unit-test",
        confidence=0.9,
        meta={},
    )
    hp = SimpleNamespace()
    reg = _stageA_shadow_registry(hp)

    _stageA_record_logexp_shadows(
        atom=atom,
        proposals=[prop],
        shadow_registry=reg,
        units_spec=spec,
        enforce_units=True,
    )

    shadows = reg.local_for(shadow_parent_key(atom))
    assert len(shadows) == 1
    assert shadows[0].unit_status == "unit_invalid"


def test_stageA_trig_wrapper_variants_are_shadow_only():
    assert _stageA_compound_variant_shadow_only("sin")
    assert _stageA_compound_variant_shadow_only("cos")
    assert _stageA_compound_variant_shadow_only("one_minus_cos")
    assert not _stageA_compound_variant_shadow_only("z")
    assert not _stageA_compound_variant_shadow_only("rat_inv")


def test_stageA_shadow_trig_promotes_only_with_real_compound_lane():
    atom = AtomNode(kind="nn", var_idxs=(1, 2, 3, 4), tag="leaf_pb011")
    reg = ShadowRegistry()
    reg.add(
        ShadowCoordinate(
            parent_key=shadow_parent_key(atom),
            parent_atom_tag="leaf_pb011",
            base_ast=Var(4),
            shadow_ast=SinNode(Var(4)),
            transform_kind="sin",
            source="local_oracle_trig",
            confidence=0.99,
            unit_status="unit_valid",
        )
    )
    base_z = MulNode(Var(2), Var(3))
    proposals = [
        (
            (0, 1, 1, 0),
            base_z,
            0.98,
            None,
            {"kind": "monomial", "clean_monomial_product": True},
        )
    ]

    promoted = _stageA_shadow_trig_composite_proposals(
        proposals,
        atom=atom,
        cols=[1, 2, 3, 4],
        shadow_registry=reg,
    )

    assert len(promoted) == 1
    pattern, z_ast, conf, extras, meta = promoted[0]
    assert pattern == (0, 1, 1, "shadow")
    assert extras == [1]
    assert conf > 0.9
    assert meta["kind"] == "shadow_composite"
    assert meta["shadow_visible_ast"] is True
    assert meta["shadow_requires_payoff"] is True
    assert meta["hidden_shadow_only"] is False
    assert meta["shadow_base_lane_class"] == "compound_lane"
    human = ast_to_human_readable(z_ast)
    assert "x2" in human
    assert "x3" in human
    assert "sin(x4)" in human


def test_stageA_shadow_does_not_repromote_consumed_coordinate():
    consumed = SinNode(Var(4))
    atom = AtomNode(
        kind="nn",
        var_idxs=(2, 4),
        tag="leaf_consumed",
        inputs=(MulNode(Var(2), consumed), Var(4)),
    )
    assert _stageA_shadow_ast_present_in_inputs(atom, consumed)

    reg = ShadowRegistry()
    reg.add(
        ShadowCoordinate(
            parent_key=shadow_parent_key(atom),
            parent_atom_tag="leaf_consumed",
            base_ast=Var(4),
            shadow_ast=SinNode(Var(4)),
            transform_kind="sin",
            source="local_oracle_trig",
            confidence=0.99,
            unit_status="unit_valid",
        )
    )
    proposals = [
        (
            (1, 0),
            Var(2),
            0.98,
            [4],
            {"kind": "monomial", "clean_monomial_product": True},
        )
    ]

    assert (
        _stageA_shadow_composite_proposals(
            proposals,
            atom=atom,
            cols=[2, 4],
            shadow_registry=reg,
        )
        == []
    )


def test_stageA_shadow_promotions_require_valid_units_under_strict_policy():
    atom = AtomNode(kind="nn", var_idxs=(0, 1), tag="leaf_unknown_units")
    reg = ShadowRegistry()
    reg.add(
        ShadowCoordinate(
            parent_key=shadow_parent_key(atom),
            parent_atom_tag="leaf_unknown_units",
            base_ast=Var(1),
            shadow_ast=SinNode(Var(1)),
            transform_kind="sin",
            source="oracle",
            confidence=0.99,
            unit_status="unchecked",
        )
    )
    proposals = [
        ((1, 0), Var(0), 0.95, [1], {"kind": "monomial"}),
    ]

    assert (
        _stageA_shadow_composite_proposals(
            proposals,
            atom=atom,
            cols=[0, 1],
            shadow_registry=reg,
            enforce_units=True,
        )
        == []
    )
    loose = _stageA_shadow_composite_proposals(
        proposals,
        atom=atom,
        cols=[0, 1],
        shadow_registry=reg,
        enforce_units=False,
    )
    assert len(loose) == 1
    assert loose[0][4]["shadow_base_lane_class"] == "raw_factor"


def test_stageA_shadow_trig_does_not_promote_without_compound_lane():
    atom = AtomNode(kind="nn", var_idxs=(4,), tag="leaf_trig_only")
    reg = ShadowRegistry()
    reg.add(
        ShadowCoordinate(
            parent_key=shadow_parent_key(atom),
            parent_atom_tag="leaf_trig_only",
            base_ast=Var(4),
            shadow_ast=SinNode(Var(4)),
            transform_kind="sin",
            source="local_oracle_trig",
            confidence=0.99,
        )
    )

    assert (
        _stageA_shadow_trig_composite_proposals(
            [],
            atom=atom,
            cols=[4],
            shadow_registry=reg,
        )
        == []
    )


def test_stageA_shadow_promotes_log_ratio_after_compound_lanes_exist():
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 2, 3, 4), tag="leaf_pb047")
    reg = ShadowRegistry()
    log_ratio = LogNode(MulNode(Var(4), PowNode(Var(3), -1.0)))
    reg.add(
        ShadowCoordinate(
            parent_key=shadow_parent_key(atom),
            parent_atom_tag="leaf_pb047",
            base_ast=MulNode(Var(4), PowNode(Var(3), -1.0)),
            shadow_ast=log_ratio,
            transform_kind="log",
            source="stageA_logexp",
            confidence=0.93,
            unit_status="unit_valid",
        )
    )
    base_z = MulNode(MulNode(Var(0), Var(1)), Var(2))
    proposals = [
        (
            (1, 1, 1, 0, 0),
            base_z,
            0.97,
            None,
            {"kind": "monomial", "clean_monomial_product": True},
        )
    ]

    promoted = _stageA_shadow_composite_proposals(
        proposals,
        atom=atom,
        cols=[0, 1, 2, 3, 4],
        shadow_registry=reg,
    )

    assert len(promoted) == 1
    pattern, z_ast, _conf, extras, meta = promoted[0]
    assert pattern == (1, 1, 1, "shadow", "shadow")
    assert extras is None
    assert meta["kind"] == "shadow_composite"
    assert meta["shadow_transform"] == "log"
    assert meta["shadow_base_lane_class"] == "compound_lane"
    human = ast_to_human_readable(z_ast)
    assert "x0" in human
    assert "x1" in human
    assert "x2" in human
    assert "log" in human
    assert "x3" in human
    assert "x4" in human


def test_stageA_shadow_preserved_factor_promotes_log_ratio_after_product_coordinate():
    product = MulNode(MulNode(Var(0), Var(1)), Var(2))
    ratio = MulNode(Var(4), PowNode(Var(3), -1.0))
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3, 4),
        tag="leaf_pb047_preserved",
        inputs=(product, Var(3), Var(4)),
    )
    reg = ShadowRegistry()
    reg.add(
        ShadowCoordinate(
            parent_key=shadow_parent_key(atom),
            parent_atom_tag="leaf_pb047_preserved",
            base_ast=ratio,
            shadow_ast=LogNode(ratio),
            transform_kind="log",
            source="stageA_logexp",
            confidence=0.93,
            unit_status="unit_valid",
        )
    )

    promoted = _stageA_shadow_preserved_factor_proposals(
        atom=atom,
        cols=["z", 3, 4],
        shadow_registry=reg,
    )

    assert len(promoted) == 1
    pattern, z_ast, conf, extras, meta = promoted[0]
    assert pattern == ("factor", "shadow", "shadow")
    assert extras is None
    assert conf > 0.8
    assert meta["kind"] == "shadow_preserved_factor"
    assert meta["shadow_visible_ast"] is True
    assert meta["shadow_requires_payoff"] is True
    assert meta["hidden_shadow_only"] is False
    assert meta["shadow_base_lane_class"] == "preserved_factor"
    human = ast_to_human_readable(z_ast)
    assert "x0" in human
    assert "x1" in human
    assert "x2" in human
    assert "log" in human
    assert "x3" in human
    assert "x4" in human


def test_stageA_shadow_preserved_factor_promotes_shadow_of_existing_ratio_input():
    product = MulNode(MulNode(Var(0), Var(1)), Var(2))
    ratio = MulNode(Var(4), PowNode(Var(3), -1.0))
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3, 4),
        tag="leaf_pb047_ratio_product",
        inputs=(ratio, product),
    )
    reg = ShadowRegistry()
    reg.add(
        ShadowCoordinate(
            parent_key=shadow_parent_key(atom),
            parent_atom_tag="leaf_pb047_ratio_product",
            base_ast=ratio,
            shadow_ast=LogNode(ratio),
            transform_kind="log",
            source="stageA_logexp",
            confidence=0.93,
            unit_status="unit_valid",
        )
    )

    promoted = _stageA_shadow_preserved_factor_proposals(
        atom=atom,
        cols=["z", "z1"],
        shadow_registry=reg,
    )

    assert len(promoted) == 1
    pattern, z_ast, _conf, extras, meta = promoted[0]
    assert pattern == ("shadow", "factor")
    assert extras is None
    assert meta["shadow_consumed_positions"] == (0,)
    assert meta["shadow_preserved_positions"] == (1,)
    assert meta["shadow_base_lane_class"] == "preserved_factor"
    human = ast_to_human_readable(z_ast)
    assert "x0" in human
    assert "x1" in human
    assert "x2" in human
    assert "log" in human
    assert "x3" in human
    assert "x4" in human


def test_stageA_shadow_preserved_factor_does_not_replace_normal_raw_compound_lane():
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 2, 3, 4), tag="leaf_pb047_raw")
    ratio = MulNode(Var(4), PowNode(Var(3), -1.0))
    reg = ShadowRegistry()
    reg.add(
        ShadowCoordinate(
            parent_key=shadow_parent_key(atom),
            parent_atom_tag="leaf_pb047_raw",
            base_ast=ratio,
            shadow_ast=LogNode(ratio),
            transform_kind="log",
            source="stageA_logexp",
            confidence=0.93,
            unit_status="unit_valid",
        )
    )

    assert (
        _stageA_shadow_preserved_factor_proposals(
            atom=atom,
            cols=[0, 1, 2, 3, 4],
            shadow_registry=reg,
        )
        == []
    )


def test_stageA_shadow_allows_explicit_single_raw_factor_lane_for_pb025_shape():
    atom = AtomNode(kind="nn", var_idxs=(0, 1), tag="leaf_pb025")
    reg = ShadowRegistry()
    reg.add(
        ShadowCoordinate(
            parent_key=shadow_parent_key(atom),
            parent_atom_tag="leaf_pb025",
            base_ast=Var(1),
            shadow_ast=SinNode(Var(1)),
            transform_kind="sin",
            source="local_oracle_trig",
            confidence=0.99,
            unit_status="unit_valid",
        )
    )
    proposals = [((1, 0), Var(0), 0.95, None, {"kind": "monomial"})]

    promoted = _stageA_shadow_composite_proposals(
        proposals,
        atom=atom,
        cols=[0, 1],
        shadow_registry=reg,
    )

    assert len(promoted) == 1
    pattern, z_ast, _conf, extras, meta = promoted[0]
    assert pattern == (1, "shadow")
    assert extras is None
    assert meta["shadow_base_lane_class"] == "raw_factor"
    human = ast_to_human_readable(z_ast)
    assert "x0" in human
    assert "sin(x1)" in human


def test_stageA_shadow_trig_factor_peel_materializes_visible_prefactor():
    product = MulNode(Var(2), Var(3))
    atom = AtomNode(
        kind="nn",
        var_idxs=(1, 2, 3, 4),
        tag="leaf_pb113_shape",
        inputs=(Var(1), product, Var(4)),
    )
    reg = ShadowRegistry()
    reg.add(
        ShadowCoordinate(
            parent_key=shadow_parent_key(atom),
            parent_atom_tag="leaf_pb113_shape",
            base_ast=Var(1),
            shadow_ast=CosNode(Var(1)),
            transform_kind="cos",
            source="local_oracle_trig",
            confidence=0.99,
            unit_status="unit_valid",
        )
    )

    promoted = _stageA_shadow_trig_factor_peel_proposals(
        atom=atom,
        cols=[1, "z", 4],
        shadow_registry=reg,
    )

    assert len(promoted) == 1
    pattern, z_ast, conf, extras, meta = promoted[0]
    assert pattern == ("shadow", 1, 0)
    assert extras == [4]
    assert conf > 0.9
    assert meta["kind"] == "shadow_trig_factor_peel"
    assert meta["shadow_visible_ast"] is True
    assert meta["shadow_requires_payoff"] is True
    assert meta["hidden_shadow_only"] is False
    assert meta["shadow_base_lane_class"] == "factor_peel"
    assert isinstance(meta["prefactor_ast"], CosNode)
    assert "x2" in ast_to_human_readable(z_ast)
    assert "x3" in ast_to_human_readable(z_ast)


def test_stageA_shadow_trig_factor_peel_supports_one_minus_cos():
    atom = AtomNode(kind="nn", var_idxs=(0, 1), tag="leaf_one_minus_cos")
    one_minus = AddNode(ConstNode(1.0), MulNode(ConstNode(-1.0), CosNode(Var(1))))
    reg = ShadowRegistry()
    reg.add(
        ShadowCoordinate(
            parent_key=shadow_parent_key(atom),
            parent_atom_tag="leaf_one_minus_cos",
            base_ast=Var(1),
            shadow_ast=one_minus,
            transform_kind="one_minus_cos",
            source="local_oracle_trig",
            confidence=0.98,
            unit_status="unit_valid",
        )
    )

    promoted = _stageA_shadow_trig_factor_peel_proposals(
        atom=atom,
        cols=[0, 1],
        shadow_registry=reg,
    )

    assert len(promoted) == 1
    pattern, z_ast, _conf, extras, meta = promoted[0]
    assert pattern == (1, "shadow")
    assert extras is None
    assert meta["shadow_transform"] == "one_minus_cos"
    assert "cos(x1)" in ast_to_human_readable(meta["prefactor_ast"])
    assert ast_to_human_readable(z_ast) == "x0"


def test_stageA_shadow_promotion_payoff_requires_visible_ast_and_burden_reduction():
    base = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="leaf")
    reduced = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2),
        tag="leaf",
        inputs=(MulNode(Var(0), SinNode(Var(1))), Var(2)),
    )

    ok, reason = _stageA_shadow_promotion_payoff_reason(
        base_ast=base,
        cand_ast=reduced,
        old_arity=3,
        new_arity=2,
        enables_sep=False,
        meta={"shadow_visible_ast": True, "shadow_requires_payoff": True},
    )
    assert ok
    assert "visible NN burden" in reason


def test_stageA_terminal_shadow_closure_uses_same_audit_vocabulary():
    base = AtomNode(kind="nn", var_idxs=(0, 1), tag="leaf")
    terminal = MulNode(Var(0), SinNode(Var(1)))

    ok, reason = _stageA_shadow_promotion_audit(
        base_ast=base,
        cand_ast=terminal,
        old_arity=2,
        new_arity=0,
        enables_sep=False,
        meta={
            "kind": "shadow_composite",
            "shadow_visible_ast": True,
            "shadow_requires_payoff": True,
            "hidden_shadow_only": False,
        },
    )
    assert ok
    assert "visible NN burden" in reason

    ok_hidden, reason_hidden = _stageA_shadow_promotion_audit(
        base_ast=base,
        cand_ast=terminal,
        old_arity=2,
        new_arity=0,
        enables_sep=False,
        meta={
            "kind": "shadow_composite",
            "shadow_visible_ast": True,
            "shadow_requires_payoff": True,
            "hidden_shadow_only": True,
        },
    )
    assert not ok_hidden
    assert "hidden shadow-only" in reason_hidden

    ok_invisible, reason_invisible = _stageA_shadow_promotion_audit(
        base_ast=base,
        cand_ast=terminal,
        old_arity=2,
        new_arity=0,
        enables_sep=False,
        meta={
            "kind": "shadow_composite",
            "shadow_visible_ast": False,
            "shadow_requires_payoff": True,
            "hidden_shadow_only": False,
        },
    )
    assert not ok_invisible
    assert "did not produce a visible AST" in reason_invisible


def test_stageA_shadow_promotion_payoff_rejects_hidden_or_no_payoff_state():
    base = AtomNode(kind="nn", var_idxs=(0, 1), tag="leaf")
    same = AtomNode(
        kind="nn",
        var_idxs=(0, 1),
        tag="leaf",
        inputs=(MulNode(Var(0), SinNode(Var(1))), Var(1)),
    )

    ok_hidden, reason_hidden = _stageA_shadow_promotion_payoff_reason(
        base_ast=base,
        cand_ast=same,
        old_arity=2,
        new_arity=2,
        enables_sep=True,
        meta={"shadow_visible_ast": True, "hidden_shadow_only": True},
    )
    assert not ok_hidden
    assert "hidden shadow-only" in reason_hidden

    ok_no_payoff, reason_no_payoff = _stageA_shadow_promotion_payoff_reason(
        base_ast=base,
        cand_ast=same,
        old_arity=2,
        new_arity=2,
        enables_sep=False,
        meta={"shadow_visible_ast": True, "shadow_requires_payoff": True},
    )
    assert not ok_no_payoff
    assert "no visible NN-burden payoff" in reason_no_payoff

    ok_sep, reason_sep = _stageA_shadow_promotion_payoff_reason(
        base_ast=base,
        cand_ast=same,
        old_arity=2,
        new_arity=2,
        enables_sep=True,
        meta={"shadow_visible_ast": True, "shadow_requires_payoff": True},
    )
    assert ok_sep
    assert "confirmed separability" in reason_sep
