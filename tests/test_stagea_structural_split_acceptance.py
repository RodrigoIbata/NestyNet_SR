# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from nestynet_sr.sr_core.bridges import AddNode, AtomNode, MulNode
from nestynet_sr.sr_search.search import _is_confirmed_nn_split_simplification
from nestynet_sr.sr_search.search import _stageA_noisy_overlap_split_gate
from nestynet_sr.sr_search.search import _stageA_under_protest_threshold_cap


def test_multivariate_atom_split_to_univariate_atoms_is_structural_simplification():
    base = AtomNode(kind="nn", var_idxs=(0, 1), tag="leaf0")
    cand = MulNode(
        AtomNode(kind="nn", var_idxs=(0,), tag="leaf1"),
        AtomNode(kind="nn", var_idxs=(1,), tag="leaf2"),
    )

    assert _is_confirmed_nn_split_simplification(base, cand)


def test_partial_arity_reduction_is_structural_simplification():
    base = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="leaf0")
    cand = MulNode(
        AtomNode(kind="nn", var_idxs=(0, 1), tag="leaf1"),
        AtomNode(kind="nn", var_idxs=(2,), tag="leaf2"),
    )

    assert _is_confirmed_nn_split_simplification(base, cand)


def test_same_multivariate_arity_is_not_structural_simplification():
    base = AtomNode(kind="nn", var_idxs=(0, 1), tag="leaf0")
    cand = AtomNode(kind="nn", var_idxs=(0, 1), tag="leaf1")

    assert not _is_confirmed_nn_split_simplification(base, cand)


def test_noisy_overlapping_additive_split_rejects_pb006_sideways_structure():
    base = AtomNode(kind="nn", var_idxs=(0, 1, 2, 3, 4, 5), tag="leaf0")
    cand = AddNode(
        AtomNode(kind="nn", var_idxs=(0, 1, 2, 3, 5), tag="leaf1"),
        AtomNode(kind="nn", var_idxs=(1, 2, 3, 4), tag="leaf2"),
    )

    ok, reason = _stageA_noisy_overlap_split_gate(
        split_kind="add",
        has_overlap=True,
        base_ast=base,
        cand_ast=cand,
        base_val_loss=0.10099,
        cand_val_loss=0.14471,
        noise_floor=0.01,
        n_eff=2000,
    )

    assert not ok
    assert "unresolved NN arity is not strictly simpler" in reason


def test_noisy_overlapping_additive_split_allows_true_arity_pareto_gain():
    base = AtomNode(kind="nn", var_idxs=(0, 1, 2, 3), tag="leaf0")
    cand = AddNode(
        AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="leaf1"),
        AtomNode(kind="nn", var_idxs=(2,), tag="leaf2"),
    )

    ok, reason = _stageA_noisy_overlap_split_gate(
        split_kind="add",
        has_overlap=True,
        base_ast=base,
        cand_ast=cand,
        base_val_loss=0.1,
        cand_val_loss=0.10001,
        noise_floor=0.01,
        n_eff=2000,
    )

    assert ok, reason
    assert "strictly simplifies unresolved NN arity" in reason


def test_under_protest_caps_stageA_acceptance_to_current_validation_loss():
    capped, did_cap = _stageA_under_protest_threshold_cap(
        accept_threshold=1.0,
        current_val_loss=1.0e-3,
        loss_floor=1.0e-8,
        under_protest=True,
        label="test",
    )

    assert did_cap
    assert capped == 1.0e-3


def test_under_protest_preserves_existing_floor_equivalence():
    capped, did_cap = _stageA_under_protest_threshold_cap(
        accept_threshold=1.0,
        current_val_loss=1.0e-10,
        loss_floor=1.0e-8,
        under_protest=True,
        label="test",
    )

    assert did_cap
    assert capped == 1.0e-8


def test_normal_stageA_path_keeps_existing_acceptance_budget():
    threshold, did_cap = _stageA_under_protest_threshold_cap(
        accept_threshold=1.0,
        current_val_loss=1.0e-3,
        loss_floor=1.0e-8,
        under_protest=False,
        label="test",
    )

    assert not did_cap
    assert threshold == 1.0
