# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
"""Regression tests for Stage-B multi-dataset class-tag sharing."""

import torch

from nestynet_sr.sr_core.bridges import AddNode, FreeConst, MulNode
from nestynet_sr.sr_search.stageB import fitting


class _DummyLeaf:
    def __init__(self, name: str):
        self.name = str(name)


class _DummyComposite:
    def __init__(self, leaves):
        self.leaf = list(leaves)


def _make_root_with_repeated_class_tag():
    # Two occurrences of the same class tag ("k") plus one experiment tag ("c").
    k1 = FreeConst("k", tag="k", init=1.0, scope="class")
    k2 = FreeConst("k", tag="k", init=1.0, scope="class")
    c = FreeConst("c", tag="c", init=1.0, scope="experiment")
    return AddNode(MulNode(k1, k2), c)


def _fake_build_composite_from_ast(root, *, reuse, **_kwargs):
    """Mimic tag-based local leaf reuse inside one compiled composite."""
    atoms = fitting.collect_all_atoms(root)
    tag_to_leaf = {}
    leaves = []
    for idx, atom in enumerate(atoms):
        tag = getattr(atom, "tag", None) or f"leaf{idx}"
        leaf = reuse.get(tag, tag_to_leaf.get(tag, None))
        if leaf is None:
            leaf = _DummyLeaf(f"{tag}_{idx}")
        tag_to_leaf[tag] = leaf
        leaves.append(leaf)
    return _DummyComposite(leaves)


def _patch_joint_builder_dependencies(monkeypatch):
    monkeypatch.setattr(
        fitting,
        "_clone_reuse",
        lambda reuse, _device, _dtype: dict(reuse or {}),
    )
    monkeypatch.setattr(
        fitting,
        "make_reuse_only_nn_factory",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        fitting,
        "build_composite_from_ast",
        _fake_build_composite_from_ast,
    )


def _patch_joint_builder_dependencies_with_capture(monkeypatch, calls):
    def _fake_build_with_capture(root, *, reuse, **kwargs):
        calls.append(kwargs.get("atom_factory", None))
        return _fake_build_composite_from_ast(root, reuse=reuse, **kwargs)

    monkeypatch.setattr(
        fitting,
        "_clone_reuse",
        lambda reuse, _device, _dtype: dict(reuse or {}),
    )
    monkeypatch.setattr(
        fitting,
        "make_reuse_only_nn_factory",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        fitting,
        "build_composite_from_ast",
        _fake_build_with_capture,
    )


def test_joint_builder_preserves_repeated_class_tag_ties(monkeypatch):
    _patch_joint_builder_dependencies(monkeypatch)
    root = _make_root_with_repeated_class_tag()

    composites, shared_by_tag = fitting._build_joint_composites_for_class_sharing(
        root=root,
        reuses=[{}, {}],
        class_set={"k"},
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    atoms = fitting.collect_all_atoms(root)
    k_idxs = [i for i, a in enumerate(atoms) if getattr(a, "tag", None) == "k"]
    c_idx = next(i for i, a in enumerate(atoms) if getattr(a, "tag", None) == "c")
    assert len(k_idxs) == 2

    shared_k = shared_by_tag["k"]
    # Dataset 0 keeps local repeated-tag tying.
    assert composites[0].leaf[k_idxs[0]] is composites[0].leaf[k_idxs[1]]
    # Dataset 1 receives injected shared tag and preserves repeated-tag tying.
    assert composites[1].leaf[k_idxs[0]] is shared_k
    assert composites[1].leaf[k_idxs[1]] is shared_k
    # Experiment-scoped tag remains dataset-specific.
    assert composites[0].leaf[c_idx] is not composites[1].leaf[c_idx]


def test_joint_builder_overrides_only_class_tag_reuse(monkeypatch):
    _patch_joint_builder_dependencies(monkeypatch)
    root = _make_root_with_repeated_class_tag()

    k_ds0 = _DummyLeaf("k_ds0")
    c_ds0 = _DummyLeaf("c_ds0")
    k_ds1 = _DummyLeaf("k_ds1")
    c_ds1 = _DummyLeaf("c_ds1")
    reuses = [
        {"k": k_ds0, "c": c_ds0},
        {"k": k_ds1, "c": c_ds1},
    ]

    composites, shared_by_tag = fitting._build_joint_composites_for_class_sharing(
        root=root,
        reuses=reuses,
        class_set={"k"},
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    atoms = fitting.collect_all_atoms(root)
    k_idxs = [i for i, a in enumerate(atoms) if getattr(a, "tag", None) == "k"]
    c_idx = next(i for i, a in enumerate(atoms) if getattr(a, "tag", None) == "c")
    assert len(k_idxs) == 2

    # Class tag in later datasets is forced to the dataset-0 shared leaf.
    assert shared_by_tag["k"] is k_ds0
    assert composites[1].leaf[k_idxs[0]] is k_ds0
    assert composites[1].leaf[k_idxs[1]] is k_ds0
    assert composites[1].leaf[k_idxs[0]] is not k_ds1

    # Experiment tag keeps dataset-specific reuse.
    assert composites[1].leaf[c_idx] is c_ds1
    assert composites[1].leaf[c_idx] is not c_ds0


def test_joint_builder_forwards_per_dataset_atom_factory(monkeypatch):
    calls = []
    _patch_joint_builder_dependencies_with_capture(monkeypatch, calls)
    root = _make_root_with_repeated_class_tag()

    atom_factory_0 = lambda atom, existing: None
    atom_factory_1 = lambda atom, existing: None

    fitting._build_joint_composites_for_class_sharing(
        root=root,
        reuses=[{}, {}],
        class_set={"k"},
        device=torch.device("cpu"),
        dtype=torch.float64,
        atom_factory=[atom_factory_0, atom_factory_1],
    )

    assert calls == [atom_factory_0, atom_factory_1]


def test_joint_builder_forwards_single_atom_factory_callable(monkeypatch):
    calls = []
    _patch_joint_builder_dependencies_with_capture(monkeypatch, calls)
    root = _make_root_with_repeated_class_tag()

    atom_factory = lambda atom, existing: None

    fitting._build_joint_composites_for_class_sharing(
        root=root,
        reuses=[{}, {}],
        class_set={"k"},
        device=torch.device("cpu"),
        dtype=torch.float64,
        atom_factory=atom_factory,
    )

    assert calls == [atom_factory, atom_factory]
