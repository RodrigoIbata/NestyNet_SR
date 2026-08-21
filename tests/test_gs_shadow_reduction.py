import numpy as np

from nestynet_sr.sr_core.bridges import AddNode, AtomNode, ConstNode, MulNode, Var
from nestynet_sr.sr_gs.affine_algebra import discover_affine_algebra
from nestynet_sr.sr_gs.quotient import compile_reduction_plan
from nestynet_sr.sr_search.shadow_coordinates import ShadowRegistry, shadow_parent_key, shadow_reduction_from_plan


def _scaling_fixture(n=96):
    rng = np.random.default_rng(601)
    X = rng.uniform(0.4, 2.0, size=(n, 2))
    z = X[:, 0] / X[:, 1]
    y = np.sin(z) + 0.17 * z**3
    grad_z = np.cos(z) + 0.51 * z**2
    grad = np.stack([grad_z / X[:, 1], -grad_z * X[:, 0] / X[:, 1] ** 2], axis=1)
    return X, y, grad


def test_shadow_reduction_composes_local_plan_and_does_not_create_coordinate_proposal():
    X, y, grad = _scaling_fixture()
    plan = compile_reduction_plan(discover_affine_algebra(X, y, grad))
    parent = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2),
        tag="leaf0",
        inputs=(MulNode(Var(0), Var(2)), AddNode(Var(1), ConstNode(1.0))),
    )

    shadow = shadow_reduction_from_plan(
        parent_atom=parent,
        reduction_plan=plan,
        local_inputs=parent.inputs,
        confidence=0.75,
        evidence={"case": "compound_local"},
    )
    registry = ShadowRegistry()
    stored, created = registry.add_reduction(shadow)

    assert created
    assert registry.count() == 0
    assert registry.reduction_count() == 1
    assert stored.raw_var_idxs == (0, 1, 2)
    assert stored.provenance["shadow_only"]
    assert not stored.provenance["active_candidate"]
    assert registry.reductions_local_for(shadow_parent_key(parent))[0] is stored

    report = stored.to_report()
    assert report["status"] == "shadow"
    assert report["raw_var_idxs"] == [0, 1, 2]
    assert report["coordinates"]
    assert any("x2" in str(coord.get("human", "")) for coord in report["coordinates"])


def test_shadow_registry_merges_duplicate_reductions_by_confidence_and_prunes_support():
    X, y, grad = _scaling_fixture()
    plan = compile_reduction_plan(discover_affine_algebra(X, y, grad))
    parent = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2),
        tag="leaf0",
        inputs=(MulNode(Var(0), Var(2)), AddNode(Var(1), ConstNode(1.0))),
    )
    registry = ShadowRegistry()

    first = shadow_reduction_from_plan(parent_atom=parent, reduction_plan=plan, local_inputs=parent.inputs, confidence=0.25)
    second = shadow_reduction_from_plan(parent_atom=parent, reduction_plan=plan, local_inputs=parent.inputs, confidence=0.9)
    registry.add_reduction(first)
    stored, created = registry.add_reduction(second)

    assert not created
    assert stored.confidence == 0.9
    assert registry.reduction_count() == 1

    removed_parents, removed_records = registry.prune_for_live_parent_vars({shadow_parent_key(parent): (0, 1)})
    assert removed_parents == 1
    assert removed_records == 1
    assert registry.reduction_count() == 0
