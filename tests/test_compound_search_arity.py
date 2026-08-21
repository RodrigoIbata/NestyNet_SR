# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import warnings
import copy
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import nestynet_sr.sr_search.search as search_mod
from nestynet_sr.sr_core.bridges import AtomNode, MulNode, PowNode, Var
from nestynet_sr.sr_search.features import ScaleSpec
from nestynet_sr.sr_search.search import (
    _compound_absorbed_effective_inputs,
    _compound_candidate_new_arity,
    _compound_candidate_default_extra_var_idxs,
    _compound_candidate_has_confirmed_payoff,
    _compound_candidate_payoff_policy,
    _compound_candidate_preserves_separated_coordinate,
    _stageA_compound_is_structurally_protected,
    _compound_overlapping_raw_extras,
    _compound_proposal_support_arity,
    _check_early_compound_from_scaling,
    _clean_monomial_product_proposal_from_scaling,
    _detect_compound_variable_for_atom,
    _retained_axis_overlap_split_confirmed,
    _should_skip_compound_extension_after_sep,
    _stageA_leaf_prune_acceptance_gate,
    _stageA_noisy_overlap_mul_split_gate,
    _stageA_append_compound_replay_proposals,
    _stageA_build_compound_replay_descriptor,
    _shortlist_compound_proposals_with_pair_backup,
    _stageA_leaf_projection_nonregression_override,
    _stageA_classify_iso_z_result,
    _stageA_provisional_full_refit_failure_status,
    _stageA_provisional_move_reason,
    _stageA_terminal_closure_rejection_reason,
)


def _compound_atom_with_extra():
    z_expr = MulNode(Var(0), PowNode(Var(1), -1.0))
    return AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2),
        tag="leaf0",
        inputs=(z_expr, Var(2)),
    )


def test_compound_candidate_new_arity_counts_preserved_ast_extras():
    z_expr = MulNode(Var(0), PowNode(Var(1), -1.0))

    assert _compound_candidate_new_arity(extra_var_count=0, extra_input_asts=None) == 1
    assert _compound_candidate_new_arity(extra_var_count=1, extra_input_asts=None) == 2
    assert _compound_candidate_new_arity(extra_var_count=0, extra_input_asts=[z_expr]) == 2


def test_stageA_provisional_classifier_marks_exploratory_moves_only():
    assert (
        _stageA_provisional_move_reason(
            "compound_exhaustion",
            {"compound_coordinate"},
            {"full_compound": True, "old_arity": 4, "new_arity": 1},
        )
        == "full_compound_compression_requires_stageB_confirmation"
    )
    assert (
        _stageA_provisional_move_reason(
            "compound_exhaustion",
            {"compound_coordinate"},
            {"old_arity": 2, "new_arity": 2},
        )
        == "same_arity_compound_requires_downstream_confirmation"
    )
    assert (
        _stageA_provisional_move_reason(
            "x_preconditioning_trig",
            {"x_transform_active"},
            {"x_transform_map": {1: "cos(x1)"}},
        )
        == "x_transform_branch_requires_downstream_confirmation"
    )
    assert (
        _stageA_provisional_move_reason(
            "compound_exhaustion",
            {"compound_coordinate"},
            {"old_arity": 3, "new_arity": 2},
        )
        is None
    )
    assert (
        _stageA_provisional_move_reason(
            "leaf_axis_projection_prune",
            {"destructive_prune"},
            {"old_arity": 3, "new_arity": 2},
        )
        is None
    )
    assert (
        _stageA_provisional_move_reason(
            "early_compound_validation",
            {"compound_coordinate", "soft_monomial_compound"},
            {"soft_monomial_compound": True, "old_arity": 6, "new_arity": 3},
        )
        == "soft_monomial_compound_requires_full_refit_confirmation"
    )
    assert (
        _stageA_provisional_move_reason(
            "early_compound_validation",
            {"compound_coordinate", "soft_monomial_compound"},
            {
                "soft_monomial_compound": True,
                "old_arity": 6,
                "new_arity": 3,
                "full_refit_confirmed": True,
            },
        )
        is None
    )
    assert (
        _stageA_provisional_move_reason(
            "early_compound_validation",
            {"compound_coordinate"},
            {"old_arity": 6, "new_arity": 3, "null_verified": False},
        )
        == "noisy_provisional_compound_requires_full_refit_confirmation"
    )


def test_stageA_iso_z_classifier_preserves_clean_certificate():
    out = _stageA_classify_iso_z_result(
        ratio=0.02,
        y_scale=1.0,
        noise_floor_screen=0.0,
        clean_threshold=0.03,
    )

    assert out["status"] == "certified"
    assert out["decision"] == "allow"
    assert out["iso_z_clean_certified"] is True
    assert out["iso_z_uncertified"] is False


def test_stageA_iso_z_classifier_keeps_pb010_like_noisy_product_as_proposal():
    out = _stageA_classify_iso_z_result(
        ratio=0.0543,
        y_scale=3.47,
        noise_floor_screen=1.08,
        clean_threshold=0.03,
        noise_mult=2.0,
        noise_cap=0.25,
        confidence=0.956,
        min_confidence=0.75,
    )

    assert out["status"] == "provisional"
    assert out["decision"] == "allow"
    assert out["iso_z_clean_certified"] is False
    assert out["iso_z_noise_compatible"] is True
    assert out["iso_z_uncertified"] is True
    assert out["proposal_lane_protected"] is True
    assert out["iso_z_struct_ratio"] == 0.0
    assert out["iso_z_threshold_eff"] == 0.25


def test_stageA_iso_z_classifier_rejects_above_clean_threshold_without_noise():
    out = _stageA_classify_iso_z_result(
        ratio=0.0543,
        y_scale=3.47,
        noise_floor_screen=0.0,
        clean_threshold=0.03,
    )

    assert out["status"] == "reject"
    assert out["decision"] == "reject"
    assert out["iso_z_reject_reason"] == "clean_threshold_failed_without_noise_floor"


def test_stageA_iso_z_classifier_caps_noisy_proposal_threshold():
    out = _stageA_classify_iso_z_result(
        ratio=0.5,
        y_scale=3.47,
        noise_floor_screen=1.08,
        clean_threshold=0.03,
        noise_mult=2.0,
        noise_cap=0.25,
        confidence=0.956,
    )

    assert out["status"] == "reject"
    assert out["iso_z_threshold_eff"] == 0.25
    assert out["iso_z_reject_reason"] == "noise_adjusted_threshold_failed"


def test_stageA_iso_z_classifier_requires_confidence_for_noisy_path():
    out = _stageA_classify_iso_z_result(
        ratio=0.0543,
        y_scale=3.47,
        noise_floor_screen=1.08,
        clean_threshold=0.03,
        confidence=0.50,
        min_confidence=0.75,
    )

    assert out["status"] == "reject"
    assert out["iso_z_reject_reason"] == "confidence_below_noisy_iso_z_minimum"


def test_stageA_structural_protection_honors_noise_compatible_iso_z_flag():
    assert _stageA_compound_is_structurally_protected(
        {
            "kind": "monomial",
            "old_arity": 2,
            "new_arity": 1,
            "structural_protected": True,
            "iso_z_uncertified": True,
        }
    )

    assert not _stageA_compound_is_structurally_protected(
        {
            "kind": "monomial",
            "old_arity": 2,
            "new_arity": 2,
            "structural_protected": True,
        }
    )


def test_stageA_provisional_full_refit_failure_classifies_severity():
    severe = _stageA_provisional_full_refit_failure_status(
        candidate_loss=1.0e-2,
        parent_loss=1.0e-4,
        acceptable_loss=2.0e-4,
        noise_floor_raw=1.0e-6,
        n_eff=2000,
    )
    assert severe["decision"] == "rollback"
    assert severe["status"] == "severe"
    assert "candidate_exceeds_bad_loss_multiplier" in severe["reasons"]

    ambiguous = _stageA_provisional_full_refit_failure_status(
        candidate_loss=2.004e-4,
        parent_loss=2.0e-4,
        acceptable_loss=2.0e-4,
        noise_floor_raw=1.0e-6,
        n_eff=2000,
    )
    assert ambiguous["decision"] == "rollback"
    assert ambiguous["status"] == "ambiguous"


def test_compound_shortlist_preserves_arity2_backup_after_greedy_cap():
    greedy_full = []
    for i in range(6):
        z_expr = MulNode(MulNode(Var(0), Var(1)), MulNode(PowNode(Var(2), -1.0), PowNode(Var(3), -1.0)))
        greedy_full.append(
            (
                (1, 1, -1, -1),
                z_expr,
                0.99 - 0.01 * i,
                [i % 4],
                {
                    "kind": "monomial",
                    "retained_axis_wrapper": True,
                    "retained_axis": i % 4,
                    "prefactor_exponents": (i, 0, 0, 0),
                },
            )
        )
    pair_fallback = ((0, 0, 1, 1), MulNode(Var(2), Var(3)), 0.80, None, {"kind": "monomial"})

    shortlist = _shortlist_compound_proposals_with_pair_backup(
        greedy_full + [pair_fallback],
        max_proposals_to_try=6,
    )

    assert len(shortlist) == 7
    assert pair_fallback in shortlist


def test_compound_shortlist_preserves_clean_product_after_greedy_cap():
    greedy_full = []
    for i in range(6):
        z_expr = MulNode(MulNode(Var(0), Var(1)), MulNode(PowNode(Var(2), -1.0), PowNode(Var(3), -1.0)))
        greedy_full.append(
            (
                (1, 1, -1, -1),
                z_expr,
                0.99 - 0.01 * i,
                [i % 4],
                {
                    "kind": "monomial",
                    "retained_axis_wrapper": True,
                    "retained_axis": i % 4,
                    "prefactor_exponents": (i, 0, 0, 0),
                },
            )
        )
    clean_product = (
        (1, 1, 1, 1, 1, -1, 0),
        MulNode(MulNode(MulNode(MulNode(MulNode(Var(0), Var(1)), Var(2)), Var(3)), Var(4)), PowNode(Var(5), -1.0)),
        0.995,
        None,
        {"kind": "monomial", "clean_monomial_product": True},
    )

    shortlist = _shortlist_compound_proposals_with_pair_backup(
        greedy_full + [clean_product],
        max_proposals_to_try=6,
    )

    assert len(shortlist) == 7
    assert clean_product in shortlist


def _replay_lm_hp():
    return SimpleNamespace(
        fit_y_link=None,
        fit_y_link_scale=None,
        coe_current_y_transform_name="identity",
    )


def _replay_search_hp(payload=None):
    return SimpleNamespace(
        coe_problem_id="pbReplay",
        coe_current_y_transform_name="identity",
        coe_stageA_replay_reservoir=payload,
        coe_stageA_replay_log=[],
        coe_stageA_replay_scout_lane_k=2,
    )


def _compound_replay_descriptor_for(atom, current_ast=None):
    current_ast = atom if current_ast is None else current_ast
    return _stageA_build_compound_replay_descriptor(
        current_ast=current_ast,
        atom=atom,
        pattern=(1, 1, 0),
        z_expr=MulNode(Var(0), Var(1)),
        extra_var_idxs=[2],
        extra_input_asts=None,
        meta={"kind": "monomial"},
        old_arity=3,
        new_arity=2,
        confidence=0.97,
        z_name="z",
        search_hp=_replay_search_hp(),
        lm_hp=_replay_lm_hp(),
        Nxvars=3,
        x_transform_map={},
        units_spec=None,
    )


def test_stageA_compound_replay_injects_only_on_exact_parent_context():
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="leaf", inputs=(Var(0), Var(1), Var(2)))
    desc = _compound_replay_descriptor_for(atom)
    assert isinstance(desc, dict)

    payload = {
        "candidates": [
            {
                "kind": "compound_coordinate_replay",
                "payload": desc,
                "reservoir_id": "a000",
                "support_count": 1,
                "score": 0.97,
            }
        ]
    }
    search_hp = _replay_search_hp(payload)

    proposals = _stageA_append_compound_replay_proposals(
        [],
        search_hp=search_hp,
        lm_hp=_replay_lm_hp(),
        current_ast=atom,
        atom=atom,
        Nxvars=3,
        x_transform_map={},
        units_spec=None,
    )

    assert len(proposals) == 1
    pattern, z_expr, conf, extras, meta = proposals[0]
    assert tuple(pattern) == (1, 1, 0)
    assert extras == [2]
    assert conf >= 0.97
    assert meta["coe_scout_replay"] is True
    assert "matched_and_injected" in {row["status"] for row in search_hp.coe_stageA_replay_log}


def test_stageA_compound_replay_dedupes_same_coordinate_across_detector_kinds():
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="leaf", inputs=(Var(0), Var(1), Var(2)))
    desc_a = _compound_replay_descriptor_for(atom)
    desc_b = copy.deepcopy(desc_a)
    desc_b["candidate_descriptor"]["proposal_kind"] = "var_times_var"
    desc_b["replay_key"]["candidate_key"]["proposal_kind"] = "var_times_var"
    payload = {
        "candidates": [
            {"kind": "compound_coordinate_replay", "payload": desc_a, "reservoir_id": "a010"},
            {"kind": "compound_coordinate_replay", "payload": desc_b, "reservoir_id": "a011"},
        ]
    }
    search_hp = _replay_search_hp(payload)

    proposals = _stageA_append_compound_replay_proposals(
        [],
        search_hp=search_hp,
        lm_hp=_replay_lm_hp(),
        current_ast=atom,
        atom=atom,
        Nxvars=3,
        x_transform_map={},
        units_spec=None,
    )

    assert len(proposals) == 1
    reasons = {row.get("reason") for row in search_hp.coe_stageA_replay_log}
    assert "duplicate_candidate" in reasons


def test_stageA_compound_replay_rejects_wrong_problem_id():
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="leaf", inputs=(Var(0), Var(1), Var(2)))
    desc = _compound_replay_descriptor_for(atom)
    desc["problem_id"] = "other_problem"
    desc["replay_key"]["problem_id"] = "other_problem"
    payload = {"candidates": [{"kind": "compound_coordinate_replay", "payload": desc, "reservoir_id": "a012"}]}
    search_hp = _replay_search_hp(payload)

    proposals = _stageA_append_compound_replay_proposals(
        [],
        search_hp=search_hp,
        lm_hp=_replay_lm_hp(),
        current_ast=atom,
        atom=atom,
        Nxvars=3,
        x_transform_map={},
        units_spec=None,
    )

    assert proposals == []
    reasons = {row.get("reason") for row in search_hp.coe_stageA_replay_log}
    assert "problem_id_mismatch" in reasons


def test_stageA_compound_replay_accepts_legacy_stat_view_problem_id():
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="leaf", inputs=(Var(0), Var(1), Var(2)))
    desc = _compound_replay_descriptor_for(atom)
    original = "pb079_II_35_18_data"
    decorated = f"{original}.stat-search-n80000.df9f46dd4c58"
    desc["problem_id"] = decorated
    desc["replay_key"]["problem_id"] = decorated
    payload = {"candidates": [{"kind": "compound_coordinate_replay", "payload": desc, "reservoir_id": "a079"}]}
    search_hp = _replay_search_hp(payload)
    search_hp.coe_problem_id = original

    proposals = _stageA_append_compound_replay_proposals(
        [],
        search_hp=search_hp,
        lm_hp=_replay_lm_hp(),
        current_ast=atom,
        atom=atom,
        Nxvars=3,
        x_transform_map={},
        units_spec=None,
    )

    assert len(proposals) == 1
    injected = next(row for row in search_hp.coe_stageA_replay_log if row["status"] == "matched_and_injected")
    assert injected["descriptor_problem_id_raw"] == decorated
    assert injected["descriptor_problem_id_canonical"] == original
    assert injected["current_problem_id_raw"] == original
    assert injected["current_problem_id_canonical"] == original


def test_stageA_compound_replay_rejects_same_raw_support_different_effective_inputs():
    scout_atom = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="scout", inputs=(Var(0), Var(1), Var(2)))
    desc = _compound_replay_descriptor_for(scout_atom)
    assert isinstance(desc, dict)
    ref_atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2),
        tag="ref",
        inputs=(MulNode(Var(0), PowNode(Var(1), -1.0)), Var(1), Var(2)),
    )
    payload = {"candidates": [{"kind": "compound_coordinate_replay", "payload": desc, "reservoir_id": "a001"}]}
    search_hp = _replay_search_hp(payload)

    proposals = _stageA_append_compound_replay_proposals(
        [],
        search_hp=search_hp,
        lm_hp=_replay_lm_hp(),
        current_ast=ref_atom,
        atom=ref_atom,
        Nxvars=3,
        x_transform_map={},
        units_spec=None,
    )

    assert proposals == []
    reasons = {row.get("reason") for row in search_hp.coe_stageA_replay_log}
    assert "parent_effective_input_fps_mismatch" in reasons


def test_stageA_compound_replay_rejects_same_leaf_different_hole_context():
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="leaf", inputs=(Var(0), Var(1), Var(2)))
    desc = _compound_replay_descriptor_for(atom, current_ast=atom)
    assert isinstance(desc, dict)
    current_ast = MulNode(Var(1), atom)
    payload = {"candidates": [{"kind": "compound_coordinate_replay", "payload": desc, "reservoir_id": "a002"}]}
    search_hp = _replay_search_hp(payload)

    proposals = _stageA_append_compound_replay_proposals(
        [],
        search_hp=search_hp,
        lm_hp=_replay_lm_hp(),
        current_ast=current_ast,
        atom=atom,
        Nxvars=3,
        x_transform_map={},
        units_spec=None,
    )

    assert proposals == []
    reasons = {row.get("reason") for row in search_hp.coe_stageA_replay_log}
    assert "parent_hole_context_fp_mismatch" in reasons


def test_stageA_compound_replay_rejects_ambiguous_identical_parent_match():
    left = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="left", inputs=(Var(0), Var(1), Var(2)))
    right = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="right", inputs=(Var(0), Var(1), Var(2)))
    current_ast = MulNode(left, right)
    desc = _compound_replay_descriptor_for(left, current_ast=current_ast)
    assert isinstance(desc, dict)
    payload = {"candidates": [{"kind": "compound_coordinate_replay", "payload": desc, "reservoir_id": "a003"}]}
    search_hp = _replay_search_hp(payload)

    proposals = _stageA_append_compound_replay_proposals(
        [],
        search_hp=search_hp,
        lm_hp=_replay_lm_hp(),
        current_ast=current_ast,
        atom=left,
        Nxvars=3,
        x_transform_map={},
        units_spec=None,
    )

    assert proposals == []
    reasons = {row.get("reason") for row in search_hp.coe_stageA_replay_log}
    assert "ambiguous_parent_match" in reasons


def test_stageA_compound_replay_descriptor_rejects_shadow_and_prefactor_lanes():
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="leaf", inputs=(Var(0), Var(1), Var(2)))

    hidden_shadow = _stageA_build_compound_replay_descriptor(
        current_ast=atom,
        atom=atom,
        pattern=(1, 1, 0),
        z_expr=MulNode(Var(0), Var(1)),
        extra_var_idxs=[2],
        meta={"kind": "shadow_composite", "hidden_shadow_only": False},
        old_arity=3,
        new_arity=2,
        confidence=0.97,
        z_name="z",
        search_hp=_replay_search_hp(),
        lm_hp=_replay_lm_hp(),
        Nxvars=3,
        x_transform_map={},
        units_spec=None,
    )
    assert hidden_shadow is None

    prefactor = _stageA_build_compound_replay_descriptor(
        current_ast=atom,
        atom=atom,
        pattern=(1, 1, 0),
        z_expr=MulNode(Var(0), Var(1)),
        extra_var_idxs=[2],
        meta={"kind": "monomial", "prefactor_exponents": (1, 0, 0)},
        old_arity=3,
        new_arity=2,
        confidence=0.97,
        z_name="z",
        search_hp=_replay_search_hp(),
        lm_hp=_replay_lm_hp(),
        Nxvars=3,
        x_transform_map={},
        units_spec=None,
    )
    assert prefactor is None


def test_stageA_terminal_caps_inverse_trig_rational_degree():
    cand_deg2 = SimpleNamespace(
        label="inverse_trig_outer_rational_arcsin_deg2",
        meta={
            "pattern": "inverse_trig_outer_rational_closure",
            "rational_degree": 2,
        },
    )
    cand_deg1 = SimpleNamespace(
        label="inverse_trig_outer_rational_arcsin_deg1",
        meta={
            "pattern": "inverse_trig_outer_rational_closure",
            "rational_degree": 1,
        },
    )
    cand_other = SimpleNamespace(
        label="phase_hint_trig_closure",
        meta={"pattern": "phase_hint_trig_closure", "rational_degree": 4},
    )

    reason = _stageA_terminal_closure_rejection_reason(cand_deg2)
    assert reason is not None
    assert "degree=2" in reason
    assert _stageA_terminal_closure_rejection_reason(cand_deg1) is None
    assert _stageA_terminal_closure_rejection_reason(cand_other) is None
    assert (
        _stageA_terminal_closure_rejection_reason(
            cand_deg2,
            max_inverse_trig_rational_degree=2,
        )
        is None
    )


def test_compound_support_arity_counts_effective_inputs_not_raw_vars():
    z0 = MulNode(Var(0), Var(1))
    z1 = MulNode(Var(2), Var(3))
    z_pair = MulNode(z0, z1)

    assert _compound_proposal_support_arity((1, 1, 0), z_pair, {"kind": "monomial"}) == 2


def test_leaf_projection_nonregression_override_accepts_strict_axis_drop():
    z_expr = MulNode(Var(0), Var(1))
    base = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2),
        tag="leaf_projection",
        inputs=(z_expr, Var(2)),
    )
    cand = AtomNode(
        kind="nn",
        var_idxs=(0, 1),
        tag="leaf_projection",
        inputs=(z_expr,),
    )

    ok, reason = _stageA_leaf_projection_nonregression_override(
        base_ast=base,
        cand_ast=cand,
        base_val_loss=1.0e-5,
        cand_val_loss=9.0e-6,
        loss_floor=1.0e-8,
        noise_floor=0.0,
        base_train_loss=1.0e-7,
        cand_train_loss=1.0e-7,
        max_train_degradation=100.0,
        axes_to_drop=[2],
    )

    assert ok, reason
    assert "strict NN input projection" in reason


def test_leaf_projection_nonregression_override_rejects_validation_regression():
    z_expr = MulNode(Var(0), Var(1))
    base = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2),
        tag="leaf_projection",
        inputs=(z_expr, Var(2)),
    )
    cand = AtomNode(
        kind="nn",
        var_idxs=(0, 1),
        tag="leaf_projection",
        inputs=(z_expr,),
    )

    ok, reason = _stageA_leaf_projection_nonregression_override(
        base_ast=base,
        cand_ast=cand,
        base_val_loss=1.0e-5,
        cand_val_loss=1.1e-5,
        loss_floor=1.0e-8,
        noise_floor=0.0,
        base_train_loss=1.0e-7,
        cand_train_loss=1.0e-7,
        max_train_degradation=100.0,
        axes_to_drop=[2],
    )

    assert not ok
    assert "validation regression" in reason


def test_leaf_prune_gate_rejects_noisy_global_axis_erasure_regression():
    base = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="leaf_projection")
    cand = AtomNode(kind="nn", var_idxs=(0, 1), tag="leaf_projection")

    ok, reason = _stageA_leaf_prune_acceptance_gate(
        base_ast=base,
        cand_ast=cand,
        axes_to_drop=[2],
        base_val_loss=9.3e-5,
        cand_val_loss=2.45e-3,
        loss_floor=4.6e-3,
        noise_floor=1.0e-2,
        n_eff=2000,
    )

    assert not ok
    assert "global raw-axis erasure" in reason
    assert "requires validation improvement" in reason


def test_leaf_prune_gate_requires_target_quality_for_noisy_global_erasure():
    base = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="leaf_projection")
    cand = AtomNode(kind="nn", var_idxs=(0, 1), tag="leaf_projection")

    ok, reason = _stageA_leaf_prune_acceptance_gate(
        base_ast=base,
        cand_ast=cand,
        axes_to_drop=[2],
        base_val_loss=1.0e-1,
        cand_val_loss=5.0e-2,
        loss_floor=1.0e-2,
        noise_floor=1.0e-2,
        n_eff=2000,
    )

    assert not ok
    assert "target-quality" in reason


def test_leaf_prune_gate_allows_noiseless_below_floor_equivalent_global_erasure():
    base = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="leaf_projection")
    cand = AtomNode(kind="nn", var_idxs=(0, 1), tag="leaf_projection")

    ok, reason = _stageA_leaf_prune_acceptance_gate(
        base_ast=base,
        cand_ast=cand,
        axes_to_drop=[2],
        base_val_loss=1.0e-12,
        cand_val_loss=1.0e-12,
        loss_floor=1.0e-10,
        noise_floor=0.0,
        n_eff=None,
    )

    assert ok, reason
    assert "noiseless equivalent" in reason


def test_leaf_prune_gate_allows_local_prune_when_axis_remains_elsewhere():
    base = MulNode(
        AtomNode(kind="nn", var_idxs=(0, 1), tag="leaf_projection"),
        Var(1),
    )
    cand = MulNode(
        AtomNode(kind="nn", var_idxs=(0,), tag="leaf_projection"),
        Var(1),
    )

    ok, reason = _stageA_leaf_prune_acceptance_gate(
        base_ast=base,
        cand_ast=cand,
        axes_to_drop=[1],
        base_val_loss=1.0e-5,
        cand_val_loss=1.000000001e-5,
        loss_floor=1.0e-8,
        noise_floor=0.0,
        n_eff=None,
    )

    assert ok, reason
    assert "local axis prune" in reason


def test_noisy_overlap_mul_split_gate_rejects_noise_equivalent_sideways_move():
    ok, reason = _stageA_noisy_overlap_mul_split_gate(
        is_multiplicative=True,
        has_overlap=True,
        base_val_loss=3.40e-5,
        cand_val_loss=3.29e-5,
        noise_floor=3.10e-5,
        n_eff=2000,
    )

    assert not ok
    assert "noisy overlapping multiplicative split rejected" in reason


def test_noisy_overlap_mul_split_gate_allows_material_improvement():
    ok, reason = _stageA_noisy_overlap_mul_split_gate(
        is_multiplicative=True,
        has_overlap=True,
        base_val_loss=1.0e-2,
        cand_val_loss=1.0e-4,
        noise_floor=3.10e-5,
        n_eff=2000,
    )

    assert ok, reason
    assert "materially improves" in reason


def test_noisy_overlap_mul_split_gate_leaves_noiseless_behavior_unchanged():
    ok, reason = _stageA_noisy_overlap_mul_split_gate(
        is_multiplicative=True,
        has_overlap=True,
        base_val_loss=1.0e-12,
        cand_val_loss=1.0e-12,
        noise_floor=0.0,
        n_eff=None,
    )

    assert ok, reason
    assert "no positive noise floor" in reason


def test_clean_monomial_product_uses_raw_clean_axes_and_leaves_dirty_extra():
    specs = [
        ScaleSpec(f"x{i}", [i], 2.0, 2.0, 0.0, 0.003, 2000)
        for i in range(5)
    ]
    specs.append(ScaleSpec("x5", [5], -2.0, -2.0, 0.0, 0.004, 2000))
    specs.append(ScaleSpec("x6_noisy", [6], -2.0, -2.0, 0.0, 0.20, 2000))

    prop = _clean_monomial_product_proposal_from_scaling(
        specs,
        tuple(range(7)),
        rel_std_threshold=0.05,
        k_int_threshold=0.15,
    )

    assert prop is not None
    pattern, _z_ast, _conf, extra, meta = prop
    assert pattern == (1, 1, 1, 1, 1, -1, 0)
    assert extra is None
    assert meta["clean_extras"] == (6,)


def test_early_compound_preserves_raw_clean_product_when_oracle_is_too_strict():
    specs = [
        ScaleSpec(f"x{i}", [i], 2.0, 2.0, 0.0, 0.003, 2000)
        for i in range(5)
    ]
    specs.append(ScaleSpec("x5", [5], -2.0, -2.0, 0.0, 0.004, 2000))
    specs.append(ScaleSpec("x6_noisy", [6], -2.0, -2.0, 0.0, 0.20, 2000))

    # Only x0 and x5 survive the stricter oracle lane; the raw clean-product
    # lane should still be first, with x6 left as the only dirty extra.
    specs[0].oracle_verified = True
    specs[0].oracle_k = 2.0
    specs[0].oracle_rel_std = 0.01
    specs[5].oracle_verified = True
    specs[5].oracle_k = -2.0
    specs[5].oracle_rel_std = 0.01

    candidates = _check_early_compound_from_scaling(
        specs,
        7,
        rel_std_threshold=0.05,
        k_int_threshold=0.15,
        require_oracle=True,
    )

    assert candidates[0] == ((0, 1, 2, 3, 4, 5), (1, 1, 1, 1, 1, -1), (6,))
    assert ((0, 5), (1, -1), (1, 2, 3, 4, 6)) in candidates


def test_compound_absorbed_effective_inputs_uses_effective_arity_for_compound_atoms():
    atom = _compound_atom_with_extra()

    # Old raw-var accounting would have treated this as absorbing one input
    # because len(var_idxs)==3. Effective arity says the candidate is a no-op.
    assert _compound_absorbed_effective_inputs(atom, new_arity=2) == 0
    assert _compound_absorbed_effective_inputs(atom, new_arity=1) == 1


def test_compound_absorbed_effective_inputs_matches_simple_atoms():
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="leaf1")

    assert _compound_absorbed_effective_inputs(atom, new_arity=2) == 1
    assert _compound_absorbed_effective_inputs(atom, new_arity=1) == 2


def test_compound_same_arity_candidates_require_confirmed_payoff():
    assert _compound_candidate_payoff_policy(old_arity=2, new_arity=3) == "reject"
    assert _compound_candidate_payoff_policy(old_arity=2, new_arity=2) == "require_sep"
    assert _compound_candidate_payoff_policy(old_arity=3, new_arity=2) == "arity_reduction"

    assert not _compound_candidate_has_confirmed_payoff(
        old_arity=2,
        new_arity=3,
        enables_sep=True,
    )
    assert not _compound_candidate_has_confirmed_payoff(
        old_arity=2,
        new_arity=2,
        enables_sep=False,
    )
    assert _compound_candidate_has_confirmed_payoff(
        old_arity=2,
        new_arity=2,
        enables_sep=True,
    )
    assert _compound_candidate_has_confirmed_payoff(
        old_arity=3,
        new_arity=2,
        enables_sep=False,
    )


def test_already_separable_compound_still_checks_arity_reducing_extensions():
    z_expr = MulNode(Var(0), PowNode(Var(1), -1.0))

    assert not _should_skip_compound_extension_after_sep(
        already_sep=True,
        extra_var_idxs=[2],
        extra_input_asts=None,
    )
    assert not _should_skip_compound_extension_after_sep(
        already_sep=True,
        extra_var_idxs=[0, 1],
        extra_input_asts=None,
    )
    assert not _should_skip_compound_extension_after_sep(
        already_sep=True,
        extra_var_idxs=[0],
        extra_input_asts=[z_expr],
    )
    assert not _should_skip_compound_extension_after_sep(
        already_sep=False,
        extra_var_idxs=[2],
        extra_input_asts=None,
    )
    assert _should_skip_compound_extension_after_sep(
        already_sep=True,
        extra_var_idxs=[],
        extra_input_asts=None,
    )


def test_already_separable_compound_blocks_preserved_coordinate_refinements():
    old_z = MulNode(Var(0), Var(1))
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3),
        tag="leaf_sep_refinement",
        inputs=(old_z, Var(2), Var(3)),
    )

    # Explicit preserved-coordinate metadata from the extras-only scan is the
    # pb032 failure mode: NN[old_z, x2, x3] -> NN[z(x2,x3), old_z].
    assert _compound_candidate_preserves_separated_coordinate(
        already_sep=True,
        atom=atom,
        pattern=(0, 1, -1),
        preserve_z_ast=old_z,
    )

    # The same proposal is a legitimate arity reduction before the old
    # coordinate has been certified separable.
    assert not _compound_candidate_preserves_separated_coordinate(
        already_sep=False,
        atom=atom,
        pattern=(0, 1, -1),
        preserve_z_ast=old_z,
    )

    # Auto-preservation via a zero exponent on the compound token is blocked in
    # an already-separated context.
    assert _compound_candidate_preserves_separated_coordinate(
        already_sep=True,
        atom=atom,
        pattern=(0, 1, -1),
        preserve_z_ast=None,
    )

    # Bundled proposal families may preserve the old coordinate through
    # explicit compound-expression extras rather than preserve_z_ast.
    assert _compound_candidate_preserves_separated_coordinate(
        already_sep=True,
        atom=atom,
        pattern=(1, 1, -1),
        preserve_z_ast=None,
        extra_input_asts=[old_z],
    )

    # Consuming the existing coordinate into the new compound is still allowed.
    assert not _compound_candidate_preserves_separated_coordinate(
        already_sep=True,
        atom=atom,
        pattern=(1, 1, 0),
        preserve_z_ast=None,
    )


def test_stageA_compound_during_sep_forwards_pending_split_guard(monkeypatch):
    old_z = MulNode(Var(0), Var(1))
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3),
        tag="leaf_pending_split",
        inputs=(old_z, Var(2), Var(3)),
    )
    proposal = (
        (0, 1, -1),
        MulNode(Var(2), PowNode(Var(3), -1.0)),
        0.99,
        None,
        {"kind": "monomial", "preserve_z_ast": old_z},
    )
    captured = {}

    monkeypatch.setattr(search_mod, "_quick_separability_candidates", lambda **kwargs: ["sep"])
    monkeypatch.setattr(search_mod, "_stageA_split_simplicity_score", lambda **kwargs: ("score",))
    monkeypatch.setattr(
        search_mod,
        "_detect_compound_variable_for_atom",
        lambda **kwargs: ([proposal], None),
    )

    def _fake_try_compound_candidates_for_atom(**kwargs):
        captured.update(kwargs)
        return False, None, None, None, False, False

    monkeypatch.setattr(
        search_mod,
        "_try_compound_candidates_for_atom",
        _fake_try_compound_candidates_for_atom,
    )

    search_hp = SimpleNamespace(
        enable_compound_detection=True,
        compound_max_vars=4,
        compound_confidence_gate=0.85,
    )

    search_mod._try_stageA_compound_during_sep_for_atom(
        model=object(),
        current_ast=atom,
        atom=atom,
        tag_to_leaf={"leaf_pending_split": object()},
        datagen_train_noshuffle=None,
        datagen_val_noshuffle=None,
        device=torch.device("cpu"),
        dtype=torch.float64,
        leaf_builder=None,
        dual_layer_used=True,
        search_hp=search_hp,
        lm_hp=SimpleNamespace(),
        loss_target_eff=1.0e-8,
        accept_threshold_eff_cand=1.0e-6,
        best_val_loss=1.0e-6,
        best_train_loss=1.0e-6,
        loss_scale=1.0,
        model_sep_output=None,
        y_op=None,
        y_op_inv=None,
        Nxvars=4,
        x_transform_map=None,
        trig_spec=None,
        scale_specs=[],
        invariance_feats=[],
        trig_axis_specs_all=[],
        units_spec=None,
        enforce_units=False,
    )

    assert captured.get("skip_same_arity_if_already_sep") is True
    assert captured.get("baseline_split_score") == ("score",)


def test_compound_detection_can_multiply_existing_z_by_one_extra():
    class ProductAllLeaf(torch.nn.Module):
        def forward(self, x):
            return (x[:, 0] * x[:, 1]).view(-1, 1)

        def grad(self, cache):
            x = cache["x"]
            g = torch.zeros((x.shape[0], 2), dtype=x.dtype, device=x.device)
            g[:, 0] = x[:, 1]
            g[:, 1] = x[:, 0]
            return g.unsqueeze(1)

    rng = np.random.default_rng(24680)
    x_raw = torch.tensor(
        rng.uniform(0.5, 2.0, size=(512, 3)),
        dtype=torch.float64,
    )
    datagen = DataLoader(
        TensorDataset(x_raw, torch.zeros((x_raw.shape[0], 1), dtype=torch.float64)),
        batch_size=x_raw.shape[0],
    )
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2),
        tag="leaf_product_extension",
        inputs=(MulNode(Var(0), Var(1)), Var(2)),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        proposals, _ = _detect_compound_variable_for_atom(
            model=type("Model", (), {})(),
            atom=atom,
            leaf=ProductAllLeaf(),
            datagen_train=datagen,
            device=torch.device("cpu"),
            max_exponent=2,
            precision=0.05,
            max_batches=1,
            enable_linear=False,
            enable_radial=False,
            enable_shift=False,
            enable_mixed_compound=False,
        )

    assert any(tuple(prop[0]) == (1, 1) for prop in proposals)


def test_retained_axis_wrappers_are_generated_for_simple_raw_ratios():
    class RatioLeaf(torch.nn.Module):
        def forward(self, x):
            return (x[:, 1] / x[:, 0]).view(-1, 1)

        def grad(self, cache):
            x = cache["x"]
            g = torch.zeros((x.shape[0], 2), dtype=x.dtype, device=x.device)
            g[:, 0] = -x[:, 1] / (x[:, 0] ** 2)
            g[:, 1] = 1.0 / x[:, 0]
            return g.unsqueeze(1)

    rng = np.random.default_rng(97531)
    x_raw = torch.tensor(
        rng.uniform(0.5, 2.0, size=(512, 2)),
        dtype=torch.float64,
    )
    datagen = DataLoader(
        TensorDataset(x_raw, torch.zeros((x_raw.shape[0], 1), dtype=torch.float64)),
        batch_size=x_raw.shape[0],
    )
    atom = AtomNode(kind="nn", var_idxs=(0, 1), tag="leaf_ratio")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        proposals, _ = _detect_compound_variable_for_atom(
            model=type("Model", (), {})(),
            atom=atom,
            leaf=RatioLeaf(),
            datagen_train=datagen,
            device=torch.device("cpu"),
            max_exponent=2,
            precision=0.05,
            max_batches=1,
            enable_linear=False,
            enable_radial=False,
            enable_shift=False,
            enable_mixed_compound=False,
        )

    retained = [
        prop for prop in proposals
        if len(prop) >= 5 and bool((prop[4] or {}).get("retained_axis_wrapper", False))
    ]
    assert retained
    assert {tuple(prop[3] or ()) for prop in retained} & {(0,), (1,)}


def test_retained_axis_wrappers_are_generated_for_high_arity_compounds():
    class RatioProductLeaf(torch.nn.Module):
        def forward(self, x):
            return ((x[:, 0] * x[:, 1]) / (x[:, 2] * x[:, 3])).view(-1, 1)

        def grad(self, cache):
            x = cache["x"]
            z = (x[:, 0] * x[:, 1]) / (x[:, 2] * x[:, 3])
            g = torch.zeros((x.shape[0], 4), dtype=x.dtype, device=x.device)
            g[:, 0] = z / x[:, 0]
            g[:, 1] = z / x[:, 1]
            g[:, 2] = -z / x[:, 2]
            g[:, 3] = -z / x[:, 3]
            return g.unsqueeze(1)

    rng = np.random.default_rng(86420)
    x_raw = torch.tensor(
        rng.uniform(0.5, 2.0, size=(512, 4)),
        dtype=torch.float64,
    )
    datagen = DataLoader(
        TensorDataset(x_raw, torch.zeros((x_raw.shape[0], 1), dtype=torch.float64)),
        batch_size=x_raw.shape[0],
    )
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 2, 3), tag="leaf_ratio_product")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        proposals, _ = _detect_compound_variable_for_atom(
            model=type("Model", (), {})(),
            atom=atom,
            leaf=RatioProductLeaf(),
            datagen_train=datagen,
            device=torch.device("cpu"),
            max_exponent=2,
            precision=0.05,
            max_batches=1,
            enable_linear=False,
            enable_radial=False,
            enable_shift=False,
            enable_mixed_compound=False,
        )

    retained = [
        prop for prop in proposals
        if len(prop) >= 5 and bool((prop[4] or {}).get("retained_axis_wrapper", False))
    ]
    assert retained


def test_compound_overlap_raw_extra_detection():
    z_expr = MulNode(MulNode(Var(3), Var(4)), PowNode(Var(6), -2.0))

    assert _compound_overlapping_raw_extras(z_expr, [6]) == (6,)
    assert _compound_overlapping_raw_extras(z_expr, [0, 1]) == ()


def test_retained_axis_overlap_certificate_accepts_power_factor():
    class PowerFactorLeaf(torch.nn.Module):
        def forward(self, x):
            z = x[:, 0]
            u = x[:, 1]
            return (z * u ** 2).view(-1, 1)

        def grad(self, cache):
            x = cache["x"]
            z = x[:, 0]
            u = x[:, 1]
            g = torch.zeros((x.shape[0], 2), dtype=x.dtype, device=x.device)
            g[:, 0] = u ** 2
            g[:, 1] = 2.0 * z * u
            return g.unsqueeze(1)

    class HP:
        early_compound_rel_std = 0.05
        compound_max_exponent = 5
        compound_iso_z_min_valid = 64
        compound_pretrain_max_points = 512

    rng = np.random.default_rng(12345)
    x_raw = torch.tensor(
        rng.uniform(1.0, 2.0, size=(512, 7)),
        dtype=torch.float64,
    )
    datagen = DataLoader(
        TensorDataset(x_raw, torch.zeros((x_raw.shape[0], 1), dtype=torch.float64)),
        batch_size=x_raw.shape[0],
    )
    z_expr = MulNode(Var(3), PowNode(Var(6), -1.0))
    sep_cands = [[torch.multiply, ["z"], [6], None, 0.0]]

    ok, reason = _retained_axis_overlap_split_confirmed(
        sep_cands=sep_cands,
        leaf=PowerFactorLeaf(),
        z_expr=z_expr,
        extra_var_idxs=[6],
        retained_axis=6,
        datagen_train=datagen,
        device=torch.device("cpu"),
        dtype=torch.float64,
        search_hp=HP(),
    )

    assert ok, reason
    assert "power-like" in reason


def test_retained_axis_overlap_certificate_marks_constant_factor():
    class ConstantAxisLeaf(torch.nn.Module):
        def forward(self, x):
            z = x[:, 0]
            return z.view(-1, 1)

        def grad(self, cache):
            x = cache["x"]
            g = torch.zeros((x.shape[0], 2), dtype=x.dtype, device=x.device)
            g[:, 0] = 1.0
            g[:, 1] = 0.0
            return g.unsqueeze(1)

    class HP:
        early_compound_rel_std = 0.05
        early_compound_k_int = 0.15
        compound_max_exponent = 5
        compound_iso_z_min_valid = 64
        compound_pretrain_max_points = 512

    rng = np.random.default_rng(24680)
    x_raw = torch.tensor(
        rng.uniform(1.0, 2.0, size=(512, 7)),
        dtype=torch.float64,
    )
    datagen = DataLoader(
        TensorDataset(x_raw, torch.zeros((x_raw.shape[0], 1), dtype=torch.float64)),
        batch_size=x_raw.shape[0],
    )
    z_expr = MulNode(Var(3), PowNode(Var(6), -1.0))
    sep_cands = [[torch.multiply, ["z"], [6], None, 0.0]]

    ok, reason = _retained_axis_overlap_split_confirmed(
        sep_cands=sep_cands,
        leaf=ConstantAxisLeaf(),
        z_expr=z_expr,
        extra_var_idxs=[6],
        retained_axis=6,
        datagen_train=datagen,
        device=torch.device("cpu"),
        dtype=torch.float64,
        search_hp=HP(),
    )

    assert ok, reason
    assert "effectively constant" in reason


def test_retained_axis_overlap_certificate_rejects_nontrivial_axis_factor():
    class AngularFactorLeaf(torch.nn.Module):
        def forward(self, x):
            z = x[:, 0]
            u = x[:, 1]
            b = (u ** 4) / (torch.sin(0.5 * u) ** 4)
            return (z * b).view(-1, 1)

        def grad(self, cache):
            x = cache["x"]
            z = x[:, 0]
            u = x[:, 1]
            s = torch.sin(0.5 * u)
            c = torch.cos(0.5 * u)
            b = (u ** 4) / (s ** 4)
            db = b * (4.0 / u - 2.0 * c / s)
            g = torch.zeros((x.shape[0], 2), dtype=x.dtype, device=x.device)
            g[:, 0] = b
            g[:, 1] = z * db
            return g.unsqueeze(1)

    class HP:
        early_compound_rel_std = 0.05
        compound_max_exponent = 5
        compound_iso_z_min_valid = 64
        compound_pretrain_max_points = 512

    rng = np.random.default_rng(54321)
    x_raw = torch.tensor(
        rng.uniform(1.0, 2.0, size=(512, 7)),
        dtype=torch.float64,
    )
    datagen = DataLoader(
        TensorDataset(x_raw, torch.zeros((x_raw.shape[0], 1), dtype=torch.float64)),
        batch_size=x_raw.shape[0],
    )
    z_expr = MulNode(Var(3), PowNode(Var(6), -2.0))
    sep_cands = [[torch.multiply, ["z"], [6], None, 0.0]]

    ok, reason = _retained_axis_overlap_split_confirmed(
        sep_cands=sep_cands,
        leaf=AngularFactorLeaf(),
        z_expr=z_expr,
        extra_var_idxs=[6],
        retained_axis=6,
        datagen_train=datagen,
        device=torch.device("cpu"),
        dtype=torch.float64,
        search_hp=HP(),
    )

    assert not ok
    assert "not power-like" in reason


def test_compound_pattern_zero_on_z_token_is_not_raw_extra():
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3),
        tag="leaf_extra_map",
        inputs=(MulNode(Var(2), Var(3)), Var(0), Var(1)),
    )

    assert _compound_candidate_default_extra_var_idxs(atom, (0, 1, 1)) == []
    assert _compound_candidate_default_extra_var_idxs(atom, (0, 1, 0)) == [1]


def test_compound_detection_can_compress_pb086_shape_extras_while_preserving_existing_z():
    class ProductRatioLeaf(torch.nn.Module):
        def forward(self, x):
            q = x[:, 0]
            p = x[:, 1] * x[:, 2]
            return (q * torch.sin(p / q)).view(-1, 1)

        def grad(self, cache):
            x = cache["x"]
            q = x[:, 0]
            p = x[:, 1] * x[:, 2]
            u = p / q
            g = torch.zeros((x.shape[0], 3), dtype=x.dtype, device=x.device)
            c = torch.cos(u)
            g[:, 0] = torch.sin(u) - u * c
            g[:, 1] = x[:, 2] * c
            g[:, 2] = x[:, 1] * c
            return g.unsqueeze(1)

    rng = np.random.default_rng(13579)
    x_raw = torch.tensor(
        rng.uniform(0.5, 2.0, size=(512, 4)),
        dtype=torch.float64,
    )
    datagen = DataLoader(
        TensorDataset(x_raw, torch.zeros((x_raw.shape[0], 1), dtype=torch.float64)),
        batch_size=x_raw.shape[0],
    )
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3),
        tag="leaf_pb086_shape",
        inputs=(MulNode(Var(2), Var(3)), Var(0), Var(1)),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        proposals, _ = _detect_compound_variable_for_atom(
            model=type("Model", (), {})(),
            atom=atom,
            leaf=ProductRatioLeaf(),
            datagen_train=datagen,
            device=torch.device("cpu"),
            max_exponent=2,
            precision=0.05,
            max_batches=1,
            enable_linear=False,
            enable_radial=False,
            enable_shift=False,
            enable_mixed_compound=False,
        )

    assert any(
        tuple(prop[0]) == (0, 1, 1)
        and (prop[4] or {}).get("kind") == "monomial"
        and bool((prop[4] or {}).get("extra_only", False))
        for prop in proposals
    )
