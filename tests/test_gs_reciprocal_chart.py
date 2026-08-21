import numpy as np
import pytest

from nestynet_sr.sr_gs import GeneralizedSymmetryConfig
from nestynet_sr.sr_gs.affine_algebra import discover_affine_algebra
from nestynet_sr.sr_gs.charts import RECIPROCAL_CHART, snap_log_chart_algebra
from nestynet_sr.sr_gs.quotient import compile_reduction_plan
from nestynet_sr.sr_gs.stagea_bridge import stageA_generalized_symmetry_proposals
from nestynet_sr.sr_core.bridges import ast_to_human_readable


def _reciprocal_diff_fixture(sign=-1.0, n=200, seed=303):
    """f = g(1/x0 + sign*1/x1) with generic g and analytic gradients."""

    rng = np.random.default_rng(seed)
    X = rng.uniform(0.5, 3.0, size=(n, 2))
    z = 1.0 / X[:, 0] + sign / X[:, 1]
    y = np.sin(z) + 0.17 * z**3
    gz = np.cos(z) + 0.51 * z**2
    grad = np.stack([gz * (-1.0 / X[:, 0] ** 2), gz * (sign * -1.0 / X[:, 1] ** 2)], axis=1)
    return X, y, grad


def _cfg(charts=("identity", "reciprocal")):
    return GeneralizedSymmetryConfig(
        enabled=True,
        mode="propose",
        known_generators=False,
        known_lie=False,
        general_affine=True,
        residual_tol=1.0e-8,
        general_affine_promotion_residual_tol=1.0e-8,
        general_affine_charts=charts,
    )


def _promoted(props):
    return [p for p in props if p[4].get("kind") == "gs_promoted_reduction"]


def test_reciprocal_chart_reduction_contract():
    X, y, grad = _reciprocal_diff_fixture(sign=-1.0)
    u, grad_u = RECIPROCAL_CHART.transform(X, grad)
    algebra = discover_affine_algebra(u, y, grad_u, heldout_fraction=0.25, acceptance_residual_tol=1.0e-8)
    snapped, report = snap_log_chart_algebra(algebra, grad_u=grad_u, chart_name="reciprocal")
    assert report["status"] == "snapped"
    assert report["chart"] == "reciprocal"
    assert tuple(report["exponents"]) == (1, -1)

    plan = compile_reduction_plan(snapped)
    assert plan.status == "compiled"
    assert plan.reason == "reciprocal_linear_invariant"
    coord = plan.invariant_coordinates[0]
    assert coord.kind == "reciprocal_linear"
    assert coord.provenance["chart"] == "reciprocal"
    assert tuple(coord.provenance["coefficients"]) == (1, -1)
    assert coord.domain.excludes("x1 == 0")
    z_true = 1.0 / X[:, 0] - 1.0 / X[:, 1]
    assert np.corrcoef(coord.evaluate(X), z_true)[0, 1] > 1.0 - 1.0e-10
    # Not a monomial and not a bare linear projection.
    assert all(c.kind not in ("monomial", "linear_projection") for c in plan.invariant_coordinates)


def test_reciprocal_difference_promotes_through_bridge():
    X, y, grad = _reciprocal_diff_fixture(sign=-1.0)
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1), y_vals=y, cfg=_cfg()
    )
    promoted = _promoted(proposals)
    assert len(promoted) == 1
    meta = promoted[0][4]
    assert meta["gs_chart"] == "reciprocal"
    z_human = ast_to_human_readable(promoted[0][1])
    assert "(x0)**-1" in z_human and "(x1)**-1" in z_human


def test_reciprocal_sum_promotes():
    """1/x0 + 1/x1 — the parallel-resistor / reduced-mass family."""

    X, y, grad = _reciprocal_diff_fixture(sign=+1.0)
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1), y_vals=y, cfg=_cfg()
    )
    promoted = _promoted(proposals)
    assert len(promoted) == 1
    z_human = ast_to_human_readable(promoted[0][1])
    assert z_human.count("**-1") == 2


def test_reciprocal_chart_skipped_on_sign_crossing():
    rng = np.random.default_rng(304)
    X = rng.uniform(-2.0, 2.0, size=(200, 2))  # crosses zero
    z = 1.0 / X[:, 0] - 1.0 / X[:, 1]
    y = np.sin(z)
    gz = np.cos(z)
    grad = np.stack([gz * (-1.0 / X[:, 0] ** 2), gz * (1.0 / X[:, 1] ** 2)], axis=1)
    ok, reason = RECIPROCAL_CHART.eligibility(X)
    assert not ok
    assert reason in ("sign_crossing_samples", "near_zero_samples")
    proposals, diagnostics = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1), y_vals=y, cfg=_cfg()
    )
    rows = [r for r in diagnostics if r.get("kind") == "shadow_reduction" and r.get("chart") == "reciprocal"]
    assert rows and rows[0]["promotion_state"] == "skipped"


def test_negative_control_not_promoted_in_reciprocal_chart():
    rng = np.random.default_rng(305)
    X = rng.uniform(0.5, 3.0, size=(200, 2))
    # No reciprocal-linear symmetry: generic coupling.
    y = np.sin(X[:, 0] + 0.31 * X[:, 1] ** 2) + 0.17 * X[:, 0] * X[:, 1]
    d0 = np.cos(X[:, 0] + 0.31 * X[:, 1] ** 2) + 0.17 * X[:, 1]
    d1 = 0.62 * X[:, 1] * np.cos(X[:, 0] + 0.31 * X[:, 1] ** 2) + 0.17 * X[:, 0]
    grad = np.stack([d0, d1], axis=1)
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1), y_vals=y, cfg=_cfg()
    )
    assert _promoted(proposals) == []


def test_identity_default_still_excludes_reciprocal():
    cfg = GeneralizedSymmetryConfig(enabled=True, general_affine=True)
    assert "reciprocal" not in cfg.general_affine_chart_names()
    assert cfg.general_affine_chart_names() == ("identity",)


def test_chart_names_accepts_reciprocal():
    names = _cfg(charts=("reciprocal", "identity", "log")).general_affine_chart_names()
    assert names[0] == "identity"  # identity is always first
    assert set(names) == {"identity", "log", "reciprocal"}


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
