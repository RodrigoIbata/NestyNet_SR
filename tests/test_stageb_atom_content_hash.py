from __future__ import annotations

import pytest

from nestynet_sr.sr_core import MulNode, Var
from nestynet_sr.sr_core.bridges import AtomNode
from nestynet_sr.sr_search.stageB._engine_support import atom_content_hash


def _pb066_like_atom(*, reverse_handoff: bool = False) -> AtomNode:
    handoff_items = [
        ("decision", "DEFERRED_UNTIL_OUTER_MAP"),
        ("carrier_dim", [0.0, 0.0, 0.0, 0.0, 0.0]),
        ("target_dim", [0.0, 0.0, 0.0, 0.0, 0.0]),
        ("carrier_certified", True),
        ("outer_map_pending", True),
    ]
    if reverse_handoff:
        handoff_items.reverse()
    return AtomNode(
        kind="nn",
        var_idxs=(1, 2),
        kwargs={
            "num_segments": 16,
            "dual_layer": True,
            "_unit_handoff": dict(handoff_items),
        },
        inputs=(MulNode(Var(1), Var(2)),),
    )


def test_atom_content_hash_accepts_pb066_nested_unit_handoff():
    assert isinstance(atom_content_hash(_pb066_like_atom()), int)


def test_atom_content_hash_is_independent_of_mapping_insertion_order():
    assert atom_content_hash(_pb066_like_atom()) == atom_content_hash(
        _pb066_like_atom(reverse_handoff=True)
    )


def test_atom_content_hash_changes_when_nested_metadata_changes():
    base = _pb066_like_atom()
    changed = _pb066_like_atom()
    changed.kwargs["_unit_handoff"]["outer_map_pending"] = False
    assert atom_content_hash(base) != atom_content_hash(changed)


def test_atom_content_hash_handles_nested_sequences_and_sets():
    atom = _pb066_like_atom()
    atom.kwargs["metadata"] = {
        "ordered": ("a", [1, 2]),
        "unordered": {"left", "right"},
    }
    assert isinstance(atom_content_hash(atom), int)


@pytest.mark.parametrize("unsupported", [bytearray(b"mutable"), object()])
def test_atom_content_hash_rejects_unknown_object_metadata(unsupported):
    atom = _pb066_like_atom()
    atom.kwargs["unsupported"] = unsupported
    with pytest.raises(TypeError, match="unsupported AtomNode kwarg value"):
        atom_content_hash(atom)
