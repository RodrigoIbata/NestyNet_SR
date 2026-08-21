import numpy as np

from nestynet_sr.sr_gs import GeneralizedSymmetryConfig
from nestynet_sr.sr_gs.stagea_bridge import stageA_generalized_symmetry_proposals


def _scaling_fixture(n=144):
    rng = np.random.default_rng(901)
    X = rng.uniform(0.5, 2.0, size=(n, 2))
    z = X[:, 0] / X[:, 1]
    y = np.sin(z) + 0.17 * z**3
    grad_z = np.cos(z) + 0.51 * z**2
    grad = np.stack([grad_z / X[:, 1], -grad_z * X[:, 0] / X[:, 1] ** 2], axis=1)
    return X, y, grad


def _generic_no_symmetry_fixture(n=160):
    rng = np.random.default_rng(902)
    X = rng.uniform(-1.2, 1.2, size=(n, 2))
    y = (
        np.sin(X[:, 0] + 0.31 * X[:, 1] ** 2)
        + 0.17 * X[:, 0] * X[:, 1]
        + 0.11 * X[:, 0] ** 3
    )
    dy_dx0 = (
        np.cos(X[:, 0] + 0.31 * X[:, 1] ** 2)
        + 0.17 * X[:, 1]
        + 0.33 * X[:, 0] ** 2
    )
    dy_dx1 = (
        0.62 * X[:, 1] * np.cos(X[:, 0] + 0.31 * X[:, 1] ** 2)
        + 0.17 * X[:, 0]
    )
    grad = np.stack([dy_dx0, dy_dx1], axis=1)
    return X, y, grad


def _affine_cfg(mode):
    return GeneralizedSymmetryConfig(
        enabled=True,
        mode=mode,
        known_generators=False,
        known_lie=False,
        general_affine=True,
        residual_tol=1.0e-8,
        general_affine_promotion_residual_tol=1.0e-8,
    )


def test_audit_mode_keeps_general_affine_reduction_shadow_only():
    X, y, grad = _scaling_fixture()
    proposals, diagnostics = stageA_generalized_symmetry_proposals(
        atom=None,
        leaf=None,
        x_vals=X,
        y_vals=y,
        dydx_vals=grad,
        cols=(0, 1),
        cfg=_affine_cfg("audit"),
    )

    assert proposals == []
    rows = [row for row in diagnostics if row.get("kind") == "shadow_reduction"]
    assert len(rows) == 1
    assert rows[0]["promotion_state"] == "audit"
    assert rows[0]["shadow_only"]
    assert not rows[0]["used_for_proposal"]


def test_promoted_general_affine_reduction_enters_stagea_as_visible_coordinate():
    X, y, grad = _scaling_fixture()
    proposals, diagnostics = stageA_generalized_symmetry_proposals(
        atom=None,
        leaf=None,
        x_vals=X,
        y_vals=y,
        dydx_vals=grad,
        cols=(0, 1),
        cfg=_affine_cfg("propose"),
    )

    promoted = [p for p in proposals if p[4].get("kind") == "gs_promoted_reduction"]
    assert len(promoted) == 1
    pattern, _z_ast, confidence, _extra, meta = promoted[0]
    assert pattern == (1, 1)
    assert confidence > 0.99
    assert meta["gs_promotion_state"] == "promoted"
    assert meta["gs_promotion"]["accepted"]
    assert meta["gs_reduction"]["status"] == "compiled"
    assert meta["gs_reduction"]["invariant_coordinates"]
    assert "x0" in meta["z_human"]
    assert "x1" in meta["z_human"]

    rows = [row for row in diagnostics if row.get("kind") == "shadow_reduction"]
    assert len(rows) == 1
    assert rows[0]["promotion_state"] == "promoted"
    assert rows[0]["used_for_proposal"]
    assert rows[0]["used_for_selection"]
    assert rows[0]["active_candidate"]
    assert not rows[0]["shadow_only"]


def test_generic_unhelpful_affine_probe_is_rejected_and_does_not_flood_stagea():
    X, y, grad = _generic_no_symmetry_fixture()
    proposals, diagnostics = stageA_generalized_symmetry_proposals(
        atom=None,
        leaf=None,
        x_vals=X,
        y_vals=y,
        dydx_vals=grad,
        cols=(0, 1),
        cfg=_affine_cfg("propose"),
    )

    assert proposals == []
    rows = [row for row in diagnostics if row.get("kind") == "shadow_reduction"]
    assert len(rows) == 1
    assert rows[0]["promotion_state"] == "rejected"
    assert not rows[0]["used_for_proposal"]
    assert not rows[0]["active_candidate"]
    assert rows[0]["promotion"]["reason"] != "passed_promotion_gate"


def test_gs_disabled_leaves_stagea_bridge_baseline_empty():
    X, y, grad = _scaling_fixture()
    proposals, diagnostics = stageA_generalized_symmetry_proposals(
        atom=None,
        leaf=None,
        x_vals=X,
        y_vals=y,
        dydx_vals=grad,
        cols=(0, 1),
        cfg=GeneralizedSymmetryConfig(enabled=False),
    )

    assert proposals == []
    assert diagnostics == []
