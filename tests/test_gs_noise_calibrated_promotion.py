import numpy as np
import pytest

from nestynet_sr.sr_gs import GeneralizedSymmetryConfig
from nestynet_sr.sr_gs.affine_algebra import discover_affine_algebra
from nestynet_sr.sr_gs.charts import LOG_CHART, snap_log_chart_algebra
from nestynet_sr.sr_gs.stagea_bridge import stageA_generalized_symmetry_proposals


def _monomial_fixture(sigma=0.0, n=400, seed=311):
    """f = g(x0**2/x1) with generic g; optional relative gradient noise."""

    rng = np.random.default_rng(seed)
    X = rng.uniform(0.5, 2.0, size=(n, 2))
    z = X[:, 0] ** 2 / X[:, 1]
    y = np.sin(z) + 0.17 * z**3
    gz = np.cos(z) + 0.51 * z**2
    grad = np.stack(
        [gz * 2.0 * X[:, 0] / X[:, 1], -gz * X[:, 0] ** 2 / X[:, 1] ** 2],
        axis=1,
    )
    if sigma > 0.0:
        grad = grad * (1.0 + sigma * rng.standard_normal(grad.shape))
    return X, y, grad


def _generic_noisy_fixture(sigma=1.0e-3, n=400, seed=902):
    """No symmetry, strictly positive domain, noisy gradients."""

    rng = np.random.default_rng(seed)
    X = rng.uniform(0.5, 2.0, size=(n, 2))
    y = np.sin(X[:, 0] + 0.31 * X[:, 1] ** 2) + 0.17 * X[:, 0] * X[:, 1] + 0.11 * X[:, 0] ** 3
    dy_dx0 = np.cos(X[:, 0] + 0.31 * X[:, 1] ** 2) + 0.17 * X[:, 1] + 0.33 * X[:, 0] ** 2
    dy_dx1 = 0.62 * X[:, 1] * np.cos(X[:, 0] + 0.31 * X[:, 1] ** 2) + 0.17 * X[:, 0]
    grad = np.stack([dy_dx0, dy_dx1], axis=1)
    grad = grad * (1.0 + sigma * rng.standard_normal(grad.shape))
    return X, y, grad


def _cfg(calibrated, policy="augment"):
    return GeneralizedSymmetryConfig(
        enabled=True,
        mode="propose",
        policy=policy,
        known_generators=False,
        known_lie=False,
        general_affine=True,
        residual_tol=1.0e-8,
        general_affine_promotion_residual_tol=1.0e-8,
        general_affine_charts=("identity", "log"),
        general_affine_promotion_noise_calibrated=calibrated,
    )


def _promoted(proposals):
    return [p for p in proposals if p[4].get("kind") == "gs_promoted_reduction"]


def _bridge(X, y, grad, cfg):
    return stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1), y_vals=y, cfg=cfg
    )


def test_absolute_mode_rejects_noisy_monomial():
    """Default (absolute) gates only pass with oracle-exact gradients."""

    X, y, grad = _monomial_fixture(sigma=1.0e-3)
    proposals, _ = _bridge(X, y, grad, _cfg(calibrated=False))
    assert _promoted(proposals) == []


def test_calibrated_mode_promotes_noisy_monomial():
    X, y, grad = _monomial_fixture(sigma=1.0e-3)
    proposals, diagnostics = _bridge(X, y, grad, _cfg(calibrated=True))

    promoted = _promoted(proposals)
    assert len(promoted) == 1
    pattern, _z_ast, confidence, _extra, meta = promoted[0]
    assert pattern == (1, 1)
    assert tuple(meta["gs_monomial_exponents_key"]) == (2, -1)
    assert meta["gs_chart"] == "log"
    evidence = meta["gs_promotion"]["evidence"]
    assert evidence["promotion_tier"] == "noise_calibrated"
    calibration = evidence["noise_calibration"]
    assert calibration["spectral_gap"] >= calibration["min_spectral_gap"]
    assert calibration["nullity_strategy"] == "spectral_gap"
    assert 0.0 < confidence < 1.0

    rows = [r for r in diagnostics if r.get("kind") == "shadow_reduction"]
    assert all(r.get("noise_calibrated") for r in rows)


def test_calibrated_mode_oracle_promotes_via_absolute_tier():
    """Exact gradients keep the strict absolute tier; calibration changes nothing."""

    X, y, grad = _monomial_fixture(sigma=0.0)
    proposals, _ = _bridge(X, y, grad, _cfg(calibrated=True))
    promoted = _promoted(proposals)
    assert len(promoted) == 1
    meta = promoted[0][4]
    assert tuple(meta["gs_monomial_exponents_key"]) == (2, -1)
    assert meta["gs_promotion"]["evidence"]["promotion_tier"] == "absolute"
    assert promoted[0][2] > 0.99


def test_calibrated_negative_control_not_promoted():
    X, y, grad = _generic_noisy_fixture(sigma=1.0e-3)
    proposals, _ = _bridge(X, y, grad, _cfg(calibrated=True))
    assert _promoted(proposals) == []


def test_calibrated_mode_declines_at_high_noise():
    """At ~1e-2 relative gradient noise the spectrum blurs: no promotion."""

    X, y, grad = _monomial_fixture(sigma=1.0e-2)
    proposals, _ = _bridge(X, y, grad, _cfg(calibrated=True))
    assert _promoted(proposals) == []


def test_calibrated_homogeneous_prefactor_stays_audit_only():
    """f = x0 * g(z): the relative alpha/beta cut must NOT swallow genuine
    equivariance, which stays audit-only per the slice-1 promotion gate."""

    rng = np.random.default_rng(317)
    X = rng.uniform(0.5, 2.0, size=(400, 2))
    z = X[:, 0] ** 2 / X[:, 1]
    g = np.sin(z) + 0.17 * z**3
    gz = np.cos(z) + 0.51 * z**2
    y = X[:, 0] * g
    grad = np.stack(
        [g + X[:, 0] * gz * 2.0 * X[:, 0] / X[:, 1], -X[:, 0] * gz * X[:, 0] ** 2 / X[:, 1] ** 2],
        axis=1,
    )
    proposals, _ = _bridge(X, y, grad, _cfg(calibrated=True))
    assert _promoted(proposals) == []


def test_calibrated_replace_shadowed_meta_supports_suppression():
    """A calibrated-tier promotion carries the same suppression-matching meta."""

    from nestynet_sr.sr_gs import suppress_shadowed_stagea_proposals

    X, y, grad = _monomial_fixture(sigma=1.0e-3)
    proposals, _ = _bridge(X, y, grad, _cfg(calibrated=True, policy="replace-shadowed"))
    promoted = _promoted(proposals)
    assert len(promoted) == 1

    legacy_bare = ((2, -1), promoted[0][1], 0.9)
    filtered, events = suppress_shadowed_stagea_proposals(
        [legacy_bare], proposals, cols=(0, 1), cfg=_cfg(True, policy="replace-shadowed")
    )
    assert filtered == []
    assert len(events) == 1


def test_spectral_gap_solve_requires_opt_in():
    """Without the calibrated kwargs, discover keeps the absolute rank cut."""

    X, y, grad = _monomial_fixture(sigma=1.0e-3)
    u, grad_u = LOG_CHART.transform(X, grad)
    algebra = discover_affine_algebra(u, y, grad_u, heldout_fraction=0.25, bootstrap=0)
    assert algebra.nullity == 0
    assert algebra.evidence["nullity_strategy"] == "rank_tol"

    calibrated = discover_affine_algebra(
        u,
        y,
        grad_u,
        heldout_fraction=0.25,
        bootstrap=2,
        nullity_strategy="spectral_gap",
        min_spectral_gap=10.0,
        closure_tol=3.0e-2,
        bootstrap_angle_tol=0.10,
        heldout_consistency_factor=3.0,
    )
    assert calibrated.nullity == 3
    assert calibrated.evidence["nullity_strategy"] == "spectral_gap"
    assert calibrated.evidence["spectral_gap"] >= 10.0
    assert calibrated.certificate is not None and calibrated.certificate.quotient_ready


def test_snap_calibration_factor_scales_with_baseline():
    """Snapping may not degrade the residual by more than the factor."""

    X, y, grad = _monomial_fixture(sigma=1.0e-3)
    u, grad_u = LOG_CHART.transform(X, grad)
    algebra = discover_affine_algebra(
        u,
        y,
        grad_u,
        heldout_fraction=0.25,
        bootstrap=2,
        nullity_strategy="spectral_gap",
        min_spectral_gap=10.0,
        closure_tol=3.0e-2,
        bootstrap_angle_tol=0.10,
        heldout_consistency_factor=3.0,
    )
    # Absolute tolerance alone rejects the noisy snap...
    rejected, report_abs = snap_log_chart_algebra(
        algebra, grad_u=grad_u, max_denominator=4, residual_tol=1.0e-8
    )
    assert rejected is None
    assert report_abs["reason"] == "snapped_residual_exceeds_tol"
    # ...the calibrated factor accepts it relative to the unsnapped baseline.
    snapped, report_cal = snap_log_chart_algebra(
        algebra, grad_u=grad_u, max_denominator=4, residual_tol=1.0e-8, calibration_factor=3.0
    )
    assert snapped is not None
    assert report_cal["status"] == "snapped"
    assert tuple(report_cal["exponents"]) == (2, -1)
    assert report_cal["residual_rel"] <= 3.0 * report_cal["baseline_residual_rel"] + 1.0e-12


def test_snap_calibration_still_rejects_wrong_ray():
    """An irrational exponent ray must fail even with the calibrated factor:
    the snapped residual is far above the unsnapped baseline."""

    rng = np.random.default_rng(314)
    X = rng.uniform(0.5, 2.0, size=(400, 2))
    z = X[:, 0] ** np.sqrt(2.0) / X[:, 1]
    y = np.sin(z) + 0.17 * z**3
    gz = np.cos(z) + 0.51 * z**2
    grad = np.stack([gz * np.sqrt(2.0) * z / X[:, 0], -gz * z / X[:, 1]], axis=1)
    grad = grad * (1.0 + 1.0e-4 * rng.standard_normal(grad.shape))
    u, grad_u = LOG_CHART.transform(X, grad)
    algebra = discover_affine_algebra(
        u,
        y,
        grad_u,
        heldout_fraction=0.25,
        bootstrap=2,
        nullity_strategy="spectral_gap",
        min_spectral_gap=10.0,
        closure_tol=3.0e-2,
        bootstrap_angle_tol=0.10,
        heldout_consistency_factor=3.0,
    )
    if algebra.nullity == 0:
        return  # spectrum too blurred to even hypothesize: equally safe
    snapped, report = snap_log_chart_algebra(
        algebra, grad_u=grad_u, max_denominator=4, residual_tol=1.0e-8, calibration_factor=3.0
    )
    assert snapped is None
    assert report["reason"] == "snapped_residual_exceeds_tol"


def test_three_var_feynman_carrier_promotes_with_noise():
    """AI-Feynman I.30.3-style carrier: f = sin(2*pi*x0*x1/x2)**2.

    The invariance algebra of a single covector has dimension n**2-1 (8 for
    three inputs), so the spectral-gap search window must scale with the
    input dimension — a fixed window of 6 finds nothing (regression)."""

    rng = np.random.default_rng(88)
    X = np.column_stack(
        [rng.uniform(1.0, 2.0, 500), rng.uniform(1.0, 2.0, 500), rng.uniform(1.0, 4.0, 500)]
    )
    z = X[:, 0] * X[:, 1] / X[:, 2]
    w = 2.0 * np.pi
    y = np.sin(w * z) ** 2
    gz = 2.0 * w * np.sin(w * z) * np.cos(w * z)
    grad = np.stack(
        [gz * X[:, 1] / X[:, 2], gz * X[:, 0] / X[:, 2], -gz * z / X[:, 2]],
        axis=1,
    )
    grad = grad * (1.0 + 1.0e-3 * rng.standard_normal(grad.shape))

    cfg = GeneralizedSymmetryConfig(
        enabled=True,
        mode="propose",
        known_generators=False,
        known_lie=False,
        general_affine=True,
        residual_tol=1.0e-8,
        general_affine_promotion_residual_tol=1.0e-8,
        general_affine_charts=("identity", "log"),
        general_affine_promotion_noise_calibrated=True,
    )
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1, 2), y_vals=y, cfg=cfg
    )
    promoted = _promoted(proposals)
    assert len(promoted) == 1
    meta = promoted[0][4]
    assert tuple(meta["gs_monomial_exponents_key"]) == (1, 1, -1)
    assert meta["gs_promotion"]["evidence"]["promotion_tier"] == "noise_calibrated"


def test_four_var_feynman_carrier_promotes_at_oracle():
    """Bose-Einstein carrier x0*x1/(x2*x3): 4 inputs, algebra dim 15."""

    rng = np.random.default_rng(85)
    X = rng.uniform(1.0, 5.0, size=(600, 4))
    z = X[:, 0] * X[:, 1] / (2.0 * np.pi * X[:, 2] * X[:, 3])
    y = 1.0 / (np.exp(z) - 1.0)
    gz = -np.exp(z) / (np.exp(z) - 1.0) ** 2
    grad = np.stack(
        [gz * z / X[:, 0], gz * z / X[:, 1], -gz * z / X[:, 2], -gz * z / X[:, 3]],
        axis=1,
    )
    cfg = GeneralizedSymmetryConfig(
        enabled=True,
        mode="propose",
        known_generators=False,
        known_lie=False,
        general_affine=True,
        residual_tol=1.0e-8,
        general_affine_promotion_residual_tol=1.0e-8,
        general_affine_charts=("identity", "log"),
        general_affine_promotion_noise_calibrated=True,
    )
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1, 2, 3), y_vals=y, cfg=cfg
    )
    promoted = _promoted(proposals)
    assert len(promoted) == 1
    assert tuple(promoted[0][4]["gs_monomial_exponents_key"]) == (1, 1, -1, -1)


def test_covector_shrink_rescues_half_integer_ray_at_noise():
    """f = g(x0/sqrt(x1)) at 1e-3 gradient noise: the gap split overshoots by
    absorbing an approximate direction, and the covector shrink rescue must
    drop it and recover the (2, -1) ray."""

    rng = np.random.default_rng(311)
    X = rng.uniform(0.5, 2.0, size=(400, 2))
    z = X[:, 0] / np.sqrt(X[:, 1])
    y = np.sin(z) + 0.17 * z**3
    gz = np.cos(z) + 0.51 * z**2
    grad = np.stack(
        [gz / np.sqrt(X[:, 1]), -0.5 * gz * X[:, 0] * X[:, 1] ** -1.5],
        axis=1,
    )
    grad = grad * (1.0 + 1.0e-3 * rng.standard_normal(grad.shape))
    proposals, _ = _bridge(X, y, grad, _cfg(calibrated=True))
    promoted = _promoted(proposals)
    assert len(promoted) == 1
    meta = promoted[0][4]
    assert tuple(meta["gs_monomial_exponents_key"]) == (2, -1)
    algebra_report = meta["gs_reduction"]["provenance"]
    del algebra_report  # provenance content asserted via evidence below
    evidence = meta["gs_promotion"]["evidence"]
    assert evidence["promotion_tier"] == "noise_calibrated"


def test_covector_shrink_records_steps_in_algebra_evidence():
    rng = np.random.default_rng(311)
    X = rng.uniform(0.5, 2.0, size=(400, 2))
    z = X[:, 0] / np.sqrt(X[:, 1])
    y = np.sin(z) + 0.17 * z**3
    gz = np.cos(z) + 0.51 * z**2
    grad = np.stack(
        [gz / np.sqrt(X[:, 1]), -0.5 * gz * X[:, 0] * X[:, 1] ** -1.5],
        axis=1,
    )
    grad = grad * (1.0 + 1.0e-3 * rng.standard_normal(grad.shape))
    u, grad_u = LOG_CHART.transform(X, grad)
    algebra = discover_affine_algebra(
        u,
        y,
        grad_u,
        heldout_fraction=0.25,
        bootstrap=4,
        nullity_strategy="spectral_gap",
        min_spectral_gap=10.0,
        closure_tol=3.0e-2,
        bootstrap_angle_tol=0.10,
        heldout_consistency_factor=3.0,
    )
    assert algebra.evidence["covector_shrink_steps"] >= 1
    assert algebra.discovered_nullity > algebra.nullity
    assert algebra.linear_invariant_covectors.shape[0] == 1


def test_underspanned_quotient_is_not_promoted():
    """An exact 2-of-4-variable product symmetry inside a non-factoring target:
    the quotient is 3-dim but only 2 linear invariants exist, so promoting the
    first coordinate would propose something the target does not factor
    through."""

    rng = np.random.default_rng(414)
    X = rng.uniform(1.0, 5.0, size=(600, 4))
    y = np.sin(X[:, 0] + 0.31 * X[:, 1] ** 2) + 0.17 * X[:, 2] * X[:, 3] + 0.11 * X[:, 0] ** 3
    d0 = np.cos(X[:, 0] + 0.31 * X[:, 1] ** 2) + 0.33 * X[:, 0] ** 2
    d1 = 0.62 * X[:, 1] * np.cos(X[:, 0] + 0.31 * X[:, 1] ** 2)
    d2 = 0.17 * X[:, 3]
    d3 = 0.17 * X[:, 2]
    grad = np.stack([d0, d1, d2, d3], axis=1)
    grad = grad * (1.0 + 1.0e-3 * rng.standard_normal(grad.shape))
    proposals, diagnostics = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1, 2, 3), y_vals=y, cfg=_cfg(True)
    )
    assert _promoted(proposals) == []
    reasons = {
        row.get("promotion_reason", "")
        for row in diagnostics
        if row.get("kind") == "shadow_reduction"
    }
    assert any(
        "underspan" in reason or "not_compiled" in reason or "snap" in reason for reason in reasons
    )


def test_bose_einstein_carrier_promotes_at_1em4_noise_via_shrink():
    """pb085 carrier at 1e-4 gradient noise: the shrink drops the approximate
    near-monomial direction (1/(e^z - 1) ~ 1/z at small z) and recovers the
    (1, 1, -1, -1) ray."""

    rng = np.random.default_rng(85)
    X = rng.uniform(1.0, 5.0, size=(600, 4))
    z = X[:, 0] * X[:, 1] / (2.0 * np.pi * X[:, 2] * X[:, 3])
    y = 1.0 / (np.exp(z) - 1.0)
    gz = -np.exp(z) / (np.exp(z) - 1.0) ** 2
    grad = np.stack(
        [gz * z / X[:, 0], gz * z / X[:, 1], -gz * z / X[:, 2], -gz * z / X[:, 3]],
        axis=1,
    )
    grad = grad * (1.0 + 1.0e-4 * rng.standard_normal(grad.shape))
    proposals, _ = _bridge_n(X, y, grad, _cfg(True), n=4)
    promoted = _promoted(proposals)
    assert len(promoted) == 1
    meta = promoted[0][4]
    assert tuple(meta["gs_monomial_exponents_key"]) == (1, 1, -1, -1)
    assert meta["gs_promotion"]["evidence"]["promotion_tier"] == "noise_calibrated"


def _bridge_n(X, y, grad, cfg, n):
    return stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=tuple(range(n)), y_vals=y, cfg=cfg
    )


def test_default_config_leaves_calibration_off():
    cfg = GeneralizedSymmetryConfig(enabled=True, general_affine=True)
    assert not cfg.general_affine_promotion_noise_calibrated
    assert cfg.noise_calibrated_promotion_active() is False


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
