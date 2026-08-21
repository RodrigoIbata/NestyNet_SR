import numpy as np
import pytest

from nestynet_sr.sr_gs import GeneralizedSymmetryConfig
from nestynet_sr.sr_gs.affine_algebra import discover_affine_algebra
from nestynet_sr.sr_gs.charts import LOG_CHART, resolve_charts, snap_log_chart_algebra
from nestynet_sr.sr_gs.quotient import compile_reduction_plan
from nestynet_sr.sr_gs.stagea_bridge import stageA_generalized_symmetry_proposals


def _monomial_fixture(n=160):
    """f = g(x0**2 / x1) with generic g and analytic gradients."""

    rng = np.random.default_rng(311)
    X = rng.uniform(0.5, 2.0, size=(n, 2))
    z = X[:, 0] ** 2 / X[:, 1]
    y = np.sin(z) + 0.17 * z**3
    gz = np.cos(z) + 0.51 * z**2
    grad = np.stack(
        [gz * 2.0 * X[:, 0] / X[:, 1], -gz * X[:, 0] ** 2 / X[:, 1] ** 2],
        axis=1,
    )
    return X, y, grad


def _ratio_fixture(n=144):
    """f = g(x0 / x1): promotable in both the identity and log charts."""

    rng = np.random.default_rng(901)
    X = rng.uniform(0.5, 2.0, size=(n, 2))
    z = X[:, 0] / X[:, 1]
    y = np.sin(z) + 0.17 * z**3
    gz = np.cos(z) + 0.51 * z**2
    grad = np.stack([gz / X[:, 1], -gz * X[:, 0] / X[:, 1] ** 2], axis=1)
    return X, y, grad


def _generic_positive_fixture(n=160):
    """No symmetry, strictly positive domain: the log chart runs and must reject."""

    rng = np.random.default_rng(312)
    X = rng.uniform(0.5, 2.0, size=(n, 2))
    y = np.sin(X[:, 0] + 0.31 * X[:, 1] ** 2) + 0.17 * X[:, 0] * X[:, 1] + 0.11 * X[:, 0] ** 3
    dy_dx0 = np.cos(X[:, 0] + 0.31 * X[:, 1] ** 2) + 0.17 * X[:, 1] + 0.33 * X[:, 0] ** 2
    dy_dx1 = 0.62 * X[:, 1] * np.cos(X[:, 0] + 0.31 * X[:, 1] ** 2) + 0.17 * X[:, 0]
    grad = np.stack([dy_dx0, dy_dx1], axis=1)
    return X, y, grad


def _chart_cfg(mode, charts=("identity", "log")):
    return GeneralizedSymmetryConfig(
        enabled=True,
        mode=mode,
        known_generators=False,
        known_lie=False,
        general_affine=True,
        residual_tol=1.0e-8,
        general_affine_promotion_residual_tol=1.0e-8,
        general_affine_charts=charts,
    )


def _log_chart_algebra(X, y, grad, *, max_denominator=4):
    u, grad_u = LOG_CHART.transform(X, grad)
    algebra = discover_affine_algebra(
        u, y, grad_u, heldout_fraction=0.25, bootstrap=0, acceptance_residual_tol=1.0e-8
    )
    return snap_log_chart_algebra(
        algebra, grad_u=grad_u, max_denominator=max_denominator, residual_tol=1.0e-8
    )


def test_log_chart_monomial_reduction_contract():
    X, y, grad = _monomial_fixture()
    snapped, snap_report = _log_chart_algebra(X, y, grad)

    assert snap_report["status"] == "snapped"
    assert tuple(snap_report["exponents"]) == (2, -1)
    assert snap_report["residual_rel"] <= 1.0e-10

    plan = compile_reduction_plan(snapped)
    assert plan.status == "compiled"
    assert plan.reason == "log_monomial_invariant"
    # Regression: the log-chart algebra must never leak into the identity-chart
    # linear-projection branch (which would render a bogus linear-in-log AST).
    assert all(coord.kind != "linear_projection" for coord in plan.invariant_coordinates)

    coord = plan.invariant_coordinates[0]
    assert coord.kind == "monomial"
    assert coord.provenance["chart"] == "log"
    assert coord.provenance["source"] == "data"
    assert tuple(coord.provenance["exponents"]) == (2, -1)
    assert coord.raw_support == (0, 1)
    assert coord.ast is not None
    assert coord.coordinate_map is not None
    assert coord.domain.excludes("x1 <= 0")

    z_true = X[:, 0] ** 2 / X[:, 1]
    z_coord = coord.evaluate(X)
    assert np.corrcoef(z_coord, z_true)[0, 1] > 1.0 - 1.0e-10

    assert not plan.output_action.is_equivariant
    assert plan.normal_form is not None
    assert plan.normal_form.kind == "invariant_residual"


def test_log_chart_promotion_in_propose_mode():
    X, y, grad = _monomial_fixture()
    proposals, diagnostics = stageA_generalized_symmetry_proposals(
        atom=None,
        leaf=None,
        x_vals=X,
        y_vals=y,
        dydx_vals=grad,
        cols=(0, 1),
        cfg=_chart_cfg("propose"),
    )

    promoted = [p for p in proposals if p[4].get("kind") == "gs_promoted_reduction"]
    assert len(promoted) == 1
    pattern, _z_ast, confidence, _extra, meta = promoted[0]
    assert pattern == (1, 1)
    assert confidence > 0.99
    assert meta["gs_chart"] == "log"
    assert tuple(meta["gs_monomial_exponents_key"]) == (2, -1)
    assert tuple(meta["gs_monomial_exponents"]) == (2, -1)
    assert "x0" in meta["z_human"] and "x1" in meta["z_human"]

    rows = [row for row in diagnostics if row.get("kind") == "shadow_reduction"]
    assert len(rows) == 2
    by_chart = {row["chart"]: row for row in rows}
    assert by_chart["log"]["promotion_state"] == "promoted"
    assert by_chart["log"]["used_for_proposal"]
    # x0**2/x1 has a full-rank identity-chart distribution: no supported chart.
    assert by_chart["identity"]["promotion_state"] == "rejected"
    assert not by_chart["identity"]["used_for_proposal"]


def test_log_chart_audit_is_shadow_only():
    X, y, grad = _monomial_fixture()
    proposals, diagnostics = stageA_generalized_symmetry_proposals(
        atom=None,
        leaf=None,
        x_vals=X,
        y_vals=y,
        dydx_vals=grad,
        cols=(0, 1),
        cfg=_chart_cfg("audit"),
    )

    assert proposals == []
    rows = [row for row in diagnostics if row.get("kind") == "shadow_reduction"]
    assert len(rows) == 2
    for row in rows:
        assert row["chart"] in {"identity", "log"}
        assert row["shadow_only"]
        assert not row["used_for_proposal"]


def test_negative_control_generic_positive_function_not_promoted():
    X, y, grad = _generic_positive_fixture()
    proposals, diagnostics = stageA_generalized_symmetry_proposals(
        atom=None,
        leaf=None,
        x_vals=X,
        y_vals=y,
        dydx_vals=grad,
        cols=(0, 1),
        cfg=_chart_cfg("propose"),
    )

    assert proposals == []
    rows = [row for row in diagnostics if row.get("kind") == "shadow_reduction"]
    assert len(rows) == 2
    for row in rows:
        assert row["promotion_state"] in {"rejected", "skipped"}
        assert not row["used_for_proposal"]


def test_log_chart_skipped_on_nonpositive_data():
    rng = np.random.default_rng(313)
    X = rng.uniform(-1.2, 1.2, size=(160, 2))
    z = X[:, 0] - X[:, 1]
    y = np.sin(z) + 0.17 * z**3
    gz = np.cos(z) + 0.51 * z**2
    grad = np.stack([gz, -gz], axis=1)

    proposals, diagnostics = stageA_generalized_symmetry_proposals(
        atom=None,
        leaf=None,
        x_vals=X,
        y_vals=y,
        dydx_vals=grad,
        cols=(0, 1),
        cfg=_chart_cfg("propose"),
    )

    rows = [row for row in diagnostics if row.get("kind") == "shadow_reduction"]
    by_chart = {row["chart"]: row for row in rows}
    assert by_chart["log"]["promotion_state"] == "skipped"
    assert by_chart["log"]["promotion_reason"].startswith("chart_ineligible:")
    # The identity chart is unaffected: the translation invariant x0-x1 promotes.
    assert by_chart["identity"]["promotion_state"] == "promoted"
    assert len([p for p in proposals if p[4].get("kind") == "gs_promoted_reduction"]) == 1


def test_identity_default_bundle_unchanged():
    """Default (chart list unset) reproduces the pre-chart-loop behavior."""

    X, y, grad = _ratio_fixture()
    cfg = GeneralizedSymmetryConfig(
        enabled=True,
        mode="propose",
        known_generators=False,
        known_lie=False,
        general_affine=True,
        residual_tol=1.0e-8,
        general_affine_promotion_residual_tol=1.0e-8,
    )
    assert resolve_charts(cfg) == resolve_charts(_chart_cfg("propose", charts=("identity",)))

    proposals, diagnostics = stageA_generalized_symmetry_proposals(
        atom=None,
        leaf=None,
        x_vals=X,
        y_vals=y,
        dydx_vals=grad,
        cols=(0, 1),
        cfg=cfg,
    )

    promoted = [p for p in proposals if p[4].get("kind") == "gs_promoted_reduction"]
    assert len(promoted) == 1
    pattern, _z_ast, confidence, _extra, meta = promoted[0]
    assert pattern == (1, 1)
    assert confidence > 0.99
    assert meta["gs_promotion_state"] == "promoted"
    assert meta["gs_chart"] == "identity"

    rows = [row for row in diagnostics if row.get("kind") == "shadow_reduction"]
    assert len(rows) == 1
    assert rows[0]["chart"] == "identity"
    assert rows[0]["promotion_state"] == "promoted"


def test_ratio_cross_chart_dedup():
    """x0/x1 promotes in both charts but must be proposed only once."""

    X, y, grad = _ratio_fixture()
    proposals, diagnostics = stageA_generalized_symmetry_proposals(
        atom=None,
        leaf=None,
        x_vals=X,
        y_vals=y,
        dydx_vals=grad,
        cols=(0, 1),
        cfg=_chart_cfg("propose"),
    )

    promoted = [p for p in proposals if p[4].get("kind") == "gs_promoted_reduction"]
    assert len(promoted) == 1
    assert promoted[0][4]["gs_chart"] == "identity"

    rows = [row for row in diagnostics if row.get("kind") == "shadow_reduction"]
    by_chart = {row["chart"]: row for row in rows}
    assert by_chart["identity"]["used_for_proposal"]
    assert by_chart["log"].get("cross_chart_duplicate") is True
    assert not by_chart["log"]["used_for_proposal"]


def test_snap_rejects_non_rational_covector():
    """An oblique log-chart ray (e.g. exponent sqrt(2)) must fail the snap."""

    rng = np.random.default_rng(314)
    X = rng.uniform(0.5, 2.0, size=(200, 2))
    z = X[:, 0] ** np.sqrt(2.0) / X[:, 1]
    y = np.sin(z) + 0.17 * z**3
    gz = np.cos(z) + 0.51 * z**2
    grad = np.stack(
        [gz * np.sqrt(2.0) * z / X[:, 0], -gz * z / X[:, 1]],
        axis=1,
    )
    snapped, snap_report = _log_chart_algebra(X, y, grad)

    assert snapped is None
    assert snap_report["status"] == "rejected"
    assert snap_report["reason"] == "snapped_residual_exceeds_tol"


def test_half_integer_exponents_snap_to_primitive_integer_ray():
    """f = g(x0 / sqrt(x1)) canonicalizes to the (2, -1) integer ray."""

    rng = np.random.default_rng(315)
    X = rng.uniform(0.5, 2.0, size=(200, 2))
    z = X[:, 0] / np.sqrt(X[:, 1])
    y = np.sin(z) + 0.17 * z**3
    gz = np.cos(z) + 0.51 * z**2
    grad = np.stack(
        [gz / np.sqrt(X[:, 1]), -0.5 * gz * X[:, 0] * X[:, 1] ** -1.5],
        axis=1,
    )
    snapped, snap_report = _log_chart_algebra(X, y, grad)

    assert snap_report["status"] == "snapped"
    assert tuple(snap_report["exponents"]) == (2, -1)
    plan = compile_reduction_plan(snapped)
    assert plan.status == "compiled"
    z_coord = plan.invariant_coordinates[0].evaluate(X)
    # x0**2/x1 is a monotone recoding of x0/sqrt(x1) on the positive domain.
    assert np.corrcoef(z_coord, z**2)[0, 1] > 1.0 - 1.0e-10


def test_unsnapped_log_algebra_stays_audit():
    """compile_reduction_plan never renders monomials from unsnapped floats."""

    X, y, grad = _monomial_fixture()
    u, grad_u = LOG_CHART.transform(X, grad)
    algebra = discover_affine_algebra(
        u, y, grad_u, heldout_fraction=0.25, bootstrap=0, acceptance_residual_tol=1.0e-8
    )
    import dataclasses

    tagged = dataclasses.replace(algebra, chart="log")
    plan = compile_reduction_plan(tagged)
    assert plan.status == "audit"
    assert plan.reason == "log_chart_covector_not_snapped"
    assert plan.invariant_coordinates == ()


def test_gs_disabled_bridge_stays_empty_with_charts_configured():
    X, y, grad = _monomial_fixture()
    proposals, diagnostics = stageA_generalized_symmetry_proposals(
        atom=None,
        leaf=None,
        x_vals=X,
        y_vals=y,
        dydx_vals=grad,
        cols=(0, 1),
        cfg=GeneralizedSymmetryConfig(enabled=False, general_affine_charts=("identity", "log")),
    )
    assert proposals == []
    assert diagnostics == []


def test_chart_names_canonicalization():
    cfg = _chart_cfg("propose", charts=("log",))
    assert cfg.general_affine_chart_names() == ("identity", "log")
    cfg2 = _chart_cfg("propose", charts="log,identity,log")
    assert cfg2.general_affine_chart_names() == ("identity", "log")
    cfg3 = _chart_cfg("propose", charts=())
    assert cfg3.general_affine_chart_names() == ("identity",)


def test_log_chart_promotion_residual_gate_respected():
    """Noisy gradients must fail the snap/promotion residual gates."""

    X, y, grad = _monomial_fixture()
    rng = np.random.default_rng(316)
    grad_noisy = grad * (1.0 + 3.0e-2 * rng.standard_normal(grad.shape))
    proposals, _diagnostics = stageA_generalized_symmetry_proposals(
        atom=None,
        leaf=None,
        x_vals=X,
        y_vals=y,
        dydx_vals=grad_noisy,
        cols=(0, 1),
        cfg=_chart_cfg("propose"),
    )
    promoted = [p for p in proposals if p[4].get("kind") == "gs_promoted_reduction"]
    assert promoted == []


def test_log_chart_homogeneous_prefactor_stays_audit_only():
    """f = x0 * g(x0**2/x1) is output-equivariant in the log chart: no promotion.

    The scaling algebra acts on the output (beta != 0), so slice 1 keeps the
    reduction audit-only per the existing output-equivariance promotion gate.
    """

    rng = np.random.default_rng(317)
    X = rng.uniform(0.5, 2.0, size=(200, 2))
    z = X[:, 0] ** 2 / X[:, 1]
    g = np.sin(z) + 0.17 * z**3
    gz = np.cos(z) + 0.51 * z**2
    y = X[:, 0] * g
    grad = np.stack(
        [g + X[:, 0] * gz * 2.0 * X[:, 0] / X[:, 1], -X[:, 0] * gz * X[:, 0] ** 2 / X[:, 1] ** 2],
        axis=1,
    )
    proposals, diagnostics = stageA_generalized_symmetry_proposals(
        atom=None,
        leaf=None,
        x_vals=X,
        y_vals=y,
        dydx_vals=grad,
        cols=(0, 1),
        cfg=_chart_cfg("propose"),
    )
    promoted = [p for p in proposals if p[4].get("kind") == "gs_promoted_reduction"]
    assert promoted == []
    rows = [row for row in diagnostics if row.get("kind") == "shadow_reduction"]
    log_rows = [row for row in rows if row.get("chart") == "log"]
    assert log_rows
    assert all(not row["used_for_proposal"] for row in log_rows)


def test_snap_rejects_pure_single_axis_dependence():
    """f = g(x0) alone must not produce a degenerate one-axis 'monomial'."""

    rng = np.random.default_rng(318)
    X = rng.uniform(0.5, 2.0, size=(160, 2))
    z = X[:, 0]
    y = np.sin(z) + 0.17 * z**3
    gz = np.cos(z) + 0.51 * z**2
    grad = np.stack([gz, np.zeros_like(gz)], axis=1)

    proposals, _diagnostics = stageA_generalized_symmetry_proposals(
        atom=None,
        leaf=None,
        x_vals=X,
        y_vals=y,
        dydx_vals=grad,
        cols=(0, 1),
        cfg=_chart_cfg("propose"),
    )
    monomials = [
        p
        for p in proposals
        if p[4].get("kind") == "gs_promoted_reduction" and p[4].get("gs_monomial_exponents_key")
    ]
    assert monomials == []


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
