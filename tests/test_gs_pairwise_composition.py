import numpy as np
import pytest

from nestynet_sr.sr_gs import GeneralizedSymmetryConfig, suppress_shadowed_stagea_proposals
from nestynet_sr.sr_gs.pairwise_composition import compose_pairwise_monomial_proposals
from nestynet_sr.sr_gs.stagea_bridge import stageA_generalized_symmetry_proposals


class _Spec:
    """Minimal GeneratorSpec stand-in for helper-level tests."""

    def __init__(self, kind, axes, residual_rel=1.0e-3, accepted=True, alpha=0.0, beta=0.0):
        self.family = "scaling"
        self.kind = kind
        self.axes = tuple(axes)
        self.accepted = accepted
        self.residual_rel = residual_rel
        self.output_alpha = alpha
        self.output_beta = beta


def _cfg(pairwise=True, calibrated=True, policy="replace-shadowed"):
    return GeneralizedSymmetryConfig(
        enabled=True,
        mode="propose",
        policy=policy,
        general_affine=True,
        general_affine_charts=("identity", "log"),
        general_affine_promotion_noise_calibrated=calibrated,
        pairwise_composition=pairwise,
    )


def _bose_einstein(sigma=0.0, n=600, seed=85):
    rng = np.random.default_rng(seed)
    X = rng.uniform(1.0, 5.0, size=(n, 4))
    z = X[:, 0] * X[:, 1] / (2.0 * np.pi * X[:, 2] * X[:, 3])
    y = 1.0 / (np.exp(z) - 1.0)
    gz = -np.exp(z) / (np.exp(z) - 1.0) ** 2
    grad = np.stack(
        [gz * z / X[:, 0], gz * z / X[:, 1], -gz * z / X[:, 2], -gz * z / X[:, 3]],
        axis=1,
    )
    if sigma > 0.0:
        grad = grad * (1.0 + sigma * rng.standard_normal(grad.shape))
    return X, y, grad


def _promoted(proposals):
    return [p for p in proposals if p[4].get("kind") == "gs_promoted_reduction"]


# ---------------------------------------------------------------------------
# Sign-propagation contracts (helper level, fabricated specs)
# ---------------------------------------------------------------------------


def _fake_grad_for_ray(X, exponents):
    """Gradients of f = g(prod x**e) with generic g, for fabricated-spec tests."""

    e = np.asarray(exponents, dtype=float)
    z = np.prod(np.power(X, e.reshape(1, -1)), axis=1)
    gz = np.cos(z) + 0.51 * z**2
    return gz[:, None] * e.reshape(1, -1) * z[:, None] / X


def test_consistent_pairs_compose_into_unique_ray():
    rng = np.random.default_rng(1)
    X = rng.uniform(1.0, 3.0, size=(300, 4))
    grad = _fake_grad_for_ray(X, (1, 1, -1, -1))
    specs = [
        _Spec("opposite_pair", (0, 1)),   # e0 = e1
        _Spec("common_pair", (1, 2)),     # e1 = -e2
        _Spec("opposite_pair", (2, 3)),   # e2 = e3
    ]
    proposals, diagnostics = compose_pairwise_monomial_proposals(
        specs, x_vals=X, dydx_vals=grad, cols=(0, 1, 2, 3), cfg=_cfg()
    )
    assert len(proposals) == 1
    pattern, _z, confidence, _extra, meta = proposals[0]
    assert pattern == (1, 1, 1, 1)
    assert tuple(meta["gs_monomial_exponents_key"]) == (1, 1, -1, -1)
    assert meta["gs_promotion"]["evidence"]["promotion_tier"] == "pairwise_composition"
    assert confidence > 0.0
    assert any(r.get("status") == "promoted" for r in diagnostics)


def test_contradictory_pairs_are_rejected():
    rng = np.random.default_rng(2)
    X = rng.uniform(1.0, 3.0, size=(200, 3))
    grad = _fake_grad_for_ray(X, (1, 1, -1))
    specs = [
        _Spec("opposite_pair", (0, 1)),  # e0 = e1
        _Spec("common_pair", (1, 2)),    # e1 = -e2
        _Spec("opposite_pair", (0, 2)),  # e0 = e2 -- contradicts the above
    ]
    proposals, diagnostics = compose_pairwise_monomial_proposals(
        specs, x_vals=X, dydx_vals=grad, cols=(0, 1, 2), cfg=_cfg()
    )
    assert proposals == []
    assert any(r.get("reason") == "inconsistent_pair_constraints" for r in diagnostics)


def test_two_variable_components_are_skipped():
    rng = np.random.default_rng(3)
    X = rng.uniform(1.0, 3.0, size=(200, 4))
    grad = _fake_grad_for_ray(X, (0, 0, 1, 1))
    specs = [_Spec("opposite_pair", (2, 3))]
    proposals, diagnostics = compose_pairwise_monomial_proposals(
        specs, x_vals=X, dydx_vals=grad, cols=(0, 1, 2, 3), cfg=_cfg()
    )
    assert proposals == []
    assert any(str(r.get("reason", "")).startswith("support_below_min") for r in diagnostics)


def test_joint_validation_rejects_false_pair_claims():
    """Fabricated pair claims on a function with no monomial carrier must fail
    the joint determining-residual test."""

    rng = np.random.default_rng(4)
    X = rng.uniform(1.0, 3.0, size=(300, 3))
    y = np.sin(X[:, 0] + 0.31 * X[:, 1] ** 2) + 0.17 * X[:, 2] ** 3
    del y
    grad = np.stack(
        [
            np.cos(X[:, 0] + 0.31 * X[:, 1] ** 2),
            0.62 * X[:, 1] * np.cos(X[:, 0] + 0.31 * X[:, 1] ** 2),
            0.51 * X[:, 2] ** 2,
        ],
        axis=1,
    )
    specs = [
        _Spec("opposite_pair", (0, 1), residual_rel=1.0e-3),
        _Spec("common_pair", (1, 2), residual_rel=1.0e-3),
    ]
    proposals, diagnostics = compose_pairwise_monomial_proposals(
        specs, x_vals=X, dydx_vals=grad, cols=(0, 1, 2), cfg=_cfg()
    )
    assert proposals == []
    assert any(r.get("reason") == "joint_ray_residual_exceeds_tol" for r in diagnostics)


def test_equivariant_pairs_are_excluded():
    rng = np.random.default_rng(5)
    X = rng.uniform(1.0, 3.0, size=(200, 3))
    grad = _fake_grad_for_ray(X, (1, 1, -1))
    specs = [
        _Spec("opposite_pair", (0, 1), beta=0.5),  # equivariant: excluded
        _Spec("common_pair", (1, 2)),
    ]
    proposals, _ = compose_pairwise_monomial_proposals(
        specs, x_vals=X, dydx_vals=grad, cols=(0, 1, 2), cfg=_cfg()
    )
    assert proposals == []  # remaining component has support 2


def test_monomial_composition_is_sign_agnostic():
    """Integer monomial rays validate through weighted gradients (x .* grad),
    which never take logs — any-sign domains compose (unlike the global
    log-chart route, which requires positivity)."""

    rng = np.random.default_rng(6)
    X = rng.uniform(-2.0, 2.0, size=(300, 3))
    z = X[:, 0] * X[:, 1] / X[:, 2]
    gz = np.cos(z) + 0.51 * z**2
    grad = np.stack([gz * z / X[:, 0], gz * z / X[:, 1], -gz * z / X[:, 2]], axis=1)
    specs = [
        _Spec("opposite_pair", (0, 1)),
        _Spec("common_pair", (1, 2)),
    ]
    proposals, _ = compose_pairwise_monomial_proposals(
        specs, x_vals=X, dydx_vals=grad, cols=(0, 1, 2), cfg=_cfg()
    )
    assert len(proposals) == 1
    assert tuple(proposals[0][4]["gs_monomial_exponents_key"]) == (1, 1, -1)

    # ...while mismatched gradients on the same specs are still rejected.
    proposals_bad, diagnostics_bad = compose_pairwise_monomial_proposals(
        specs, x_vals=X, dydx_vals=np.ones_like(X), cols=(0, 1, 2), cfg=_cfg()
    )
    assert proposals_bad == []
    assert any(r.get("reason") == "joint_ray_residual_exceeds_tol" for r in diagnostics_bad)


# ---------------------------------------------------------------------------
# Bridge-level contracts (real witness discovery)
# ---------------------------------------------------------------------------


def test_bose_einstein_composes_at_pipeline_noise():
    """The 4-var Bose-Einstein ray promotes via composition at 3e-3 gradient
    noise, where the global determining solve loses bracket closure."""

    X, y, grad = _bose_einstein(sigma=3.0e-3)
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1, 2, 3), y_vals=y, cfg=_cfg()
    )
    promoted = _promoted(proposals)
    assert len(promoted) == 1
    meta = promoted[0][4]
    assert tuple(meta["gs_monomial_exponents_key"]) == (1, 1, -1, -1)
    assert meta["gs_promotion"]["evidence"]["promotion_tier"] == "pairwise_composition"
    assert meta["gs_kind"] == "pairwise_composed_monomial"


def test_oracle_both_routes_dedup_to_one_proposal():
    X, y, grad = _bose_einstein(sigma=0.0)
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1, 2, 3), y_vals=y, cfg=_cfg()
    )
    promoted = _promoted(proposals)
    assert len(promoted) == 1
    # The global reduction (absolute tier) wins; composition dedups against it.
    assert promoted[0][4]["gs_promotion"]["evidence"]["promotion_tier"] == "absolute"


def test_flag_off_produces_no_composition_rows():
    X, y, grad = _bose_einstein(sigma=3.0e-3)
    proposals, diagnostics = stageA_generalized_symmetry_proposals(
        atom=None,
        leaf=None,
        x_vals=X,
        dydx_vals=grad,
        cols=(0, 1, 2, 3),
        y_vals=y,
        cfg=_cfg(pairwise=False),
    )
    assert all(r.get("kind") != "pairwise_composition" for r in diagnostics)
    assert _promoted(proposals) == []  # global solve fails at this noise


def test_negative_control_composes_nothing():
    rng = np.random.default_rng(414)
    X = rng.uniform(1.0, 5.0, size=(600, 4))
    y = np.sin(X[:, 0] + 0.31 * X[:, 1] ** 2) + 0.17 * X[:, 2] * X[:, 3] + 0.11 * X[:, 0] ** 3
    d0 = np.cos(X[:, 0] + 0.31 * X[:, 1] ** 2) + 0.33 * X[:, 0] ** 2
    d1 = 0.62 * X[:, 1] * np.cos(X[:, 0] + 0.31 * X[:, 1] ** 2)
    d2 = 0.17 * X[:, 3]
    d3 = 0.17 * X[:, 2]
    grad = np.stack([d0, d1, d2, d3], axis=1)
    grad = grad * (1.0 + 3.0e-3 * rng.standard_normal(grad.shape))
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1, 2, 3), y_vals=y, cfg=_cfg()
    )
    assert _promoted(proposals) == []


def test_composed_proposal_drives_replace_shadowed_suppression():
    X, y, grad = _bose_einstein(sigma=3.0e-3)
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1, 2, 3), y_vals=y, cfg=_cfg()
    )
    promoted = _promoted(proposals)
    assert promoted
    legacy_bare = ((1, 1, -1, -1), promoted[0][1], 0.95)
    filtered, events = suppress_shadowed_stagea_proposals(
        [legacy_bare], proposals, cols=(0, 1, 2, 3), cfg=_cfg()
    )
    assert filtered == []
    assert len(events) == 1
    assert events[0]["gs_chart"] == "log"


# ---------------------------------------------------------------------------
# Linear (translation-pair) composition
# ---------------------------------------------------------------------------


def _linear_fixture(sigma=0.0, n=500, seed=21):
    """f = g(x0 - x1 + x2) on an any-sign domain with analytic gradients."""

    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n, 3))
    z = X[:, 0] - X[:, 1] + X[:, 2]
    y = np.sin(z) + 0.17 * z**3
    gz = np.cos(z) + 0.51 * z**2
    grad = np.stack([gz, -gz, gz], axis=1)
    if sigma > 0.0:
        grad = grad * (1.0 + sigma * rng.standard_normal(grad.shape))
    return X, y, grad


def test_fabricated_translation_pairs_compose_linear_ray():
    rng = np.random.default_rng(7)
    X = rng.uniform(-2.0, 2.0, size=(300, 3))
    gz = np.ones(300)
    grad = np.stack([gz, -gz, gz], axis=1)  # exact gradients of x0 - x1 + x2

    class _TSpec(_Spec):
        def __init__(self, kind, axes, **kw):
            super().__init__(kind, axes, **kw)
            self.family = "translation"

    specs = [
        _TSpec("diagonal_plus", (0, 1)),   # c1 = -c0 (invariant x0 - x1)
        _TSpec("diagonal_minus", (1, 2)),  # c2 = +c1 -> wait: invariant x1 + x2
    ]
    # diagonal_minus(1,2) gives c2 = c1 = -1 which contradicts the target
    # gradients (x0 - x1 + x2 needs c2 = +1): the joint validation must reject.
    proposals, diagnostics = compose_pairwise_monomial_proposals(
        specs, x_vals=X, dydx_vals=grad, cols=(0, 1, 2), cfg=_cfg()
    )
    assert proposals == []
    assert any(r.get("reason") == "joint_ray_residual_exceeds_tol" for r in diagnostics)

    specs_ok = [
        _TSpec("diagonal_plus", (0, 1)),   # c1 = -c0
        _TSpec("diagonal_plus", (1, 2)),   # c2 = -c1 = +c0
    ]
    proposals, _ = compose_pairwise_monomial_proposals(
        specs_ok, x_vals=X, dydx_vals=grad, cols=(0, 1, 2), cfg=_cfg()
    )
    assert len(proposals) == 1
    meta = proposals[0][4]
    assert meta["gs_kind"] == "pairwise_composed_linear"
    assert tuple(meta["gs_linear_ray_key"]) == (1, -1, 1)
    assert tuple(meta["gs_linear_covector"]) == (1.0, -1.0, 1.0)


def test_linear_composition_promotes_on_any_sign_domain_with_noise():
    X, y, grad = _linear_fixture(sigma=1.0e-3)
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1, 2), y_vals=y, cfg=_cfg()
    )
    promoted = _promoted(proposals)
    assert len(promoted) == 1
    meta = promoted[0][4]
    # The composed integer coordinate replaces the global route's raw float
    # covector for the same ray.
    assert meta["gs_kind"] == "pairwise_composed_linear"
    assert meta["gs_promotion"]["evidence"]["promotion_tier"] == "pairwise_composition"
    assert tuple(meta["gs_linear_ray_key"]) == (1, -1, 1)
    assert "0.57" not in str(meta.get("z_human", ""))


def test_four_var_linear_ray_composes_at_pipeline_noise():
    rng = np.random.default_rng(22)
    X = rng.uniform(-2.0, 2.0, size=(600, 4))
    z = X[:, 0] - X[:, 1] + X[:, 2] - X[:, 3]
    y = np.sin(z) + 0.17 * z**3
    gz = np.cos(z) + 0.51 * z**2
    grad = np.stack([gz, -gz, gz, -gz], axis=1)
    grad = grad * (1.0 + 3.0e-3 * rng.standard_normal(grad.shape))
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1, 2, 3), y_vals=y, cfg=_cfg()
    )
    promoted = _promoted(proposals)
    assert len(promoted) == 1
    assert tuple(promoted[0][4]["gs_linear_ray_key"]) == (1, -1, 1, -1)


def test_oblique_learned_ratio_does_not_compose():
    """An irrational learned pair ratio (e.g. sqrt(2)) must not snap into an
    integer ray — genuinely oblique directions stay the global solve's job."""

    rng = np.random.default_rng(23)
    X = rng.uniform(-2.0, 2.0, size=(300, 3))
    grad = np.ones_like(X)

    class _ASpec(_Spec):
        def __init__(self, axes, coeffs, **kw):
            super().__init__("affine_translation_pair", axes, **kw)
            self.family = "general_affine"
            self.xi_coeffs = coeffs

    specs = [
        _ASpec((0, 1), (1.0, np.sqrt(2.0)), residual_rel=1.0e-8),
        _ASpec((1, 2), (1.0, np.sqrt(3.0)), residual_rel=1.0e-8),
    ]
    proposals, _ = compose_pairwise_monomial_proposals(
        specs, x_vals=X, dydx_vals=grad, cols=(0, 1, 2), cfg=_cfg()
    )
    assert proposals == []


def test_composed_linear_drives_replace_shadowed_suppression_of_legacy_linear():
    X, y, grad = _linear_fixture(sigma=1.0e-3)
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1, 2), y_vals=y, cfg=_cfg()
    )
    promoted = _promoted(proposals)
    assert promoted
    legacy_linear = ((1, -1, 1), promoted[0][1], 0.9, None, {"kind": "linear"})
    filtered, events = suppress_shadowed_stagea_proposals(
        [legacy_linear], proposals, cols=(0, 1, 2), cfg=_cfg()
    )
    assert filtered == []
    assert len(events) == 1
    assert events[0]["legacy_kind"] == "linear"


# ---------------------------------------------------------------------------
# Radial (rotation-pair) composition
# ---------------------------------------------------------------------------


class _RSpec(_Spec):
    def __init__(self, axes, **kw):
        super().__init__("so2_pair", axes, **kw)
        self.family = "rotation"


def _radial_fixture(sigma=0.0, n=500, seed=31):
    """f = g(x0**2 + x1**2 + x2**2) on an any-sign domain."""

    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n, 3))
    r2 = np.sum(X**2, axis=1)
    y = np.sin(r2) + 0.17 * r2**1.5
    gz = np.cos(r2) + 0.255 * np.sqrt(r2)
    grad = 2.0 * gz[:, None] * X
    if sigma > 0.0:
        grad = grad * (1.0 + sigma * rng.standard_normal(grad.shape))
    return X, y, grad


def test_rotation_pairs_compose_radial_ray_including_bracket_implied_plane():
    """Edges (0,1) and (1,2) alone determine SO(3) on {0,1,2} (the (0,2)
    rotation is bracket-implied); the joint alignment test covers it."""

    X, _y, grad = _radial_fixture()
    specs = [_RSpec((0, 1)), _RSpec((1, 2))]
    proposals, _ = compose_pairwise_monomial_proposals(
        specs, x_vals=X, dydx_vals=grad, cols=(0, 1, 2), cfg=_cfg()
    )
    assert len(proposals) == 1
    pattern, _z, _conf, _extra, meta = proposals[0]
    assert pattern == (1, 1, 1)
    assert meta["gs_kind"] == "pairwise_composed_radial"
    assert meta["gs_coordinate_kind"] == "radial"
    assert meta["form"] == "r2" and meta["allow_sqrt"] is True
    assert "x0" in meta["z_human"] and "x2" in meta["z_human"]


def test_radial_composition_promotes_with_noise():
    X, y, grad = _radial_fixture(sigma=1.0e-3)
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1, 2), y_vals=y, cfg=_cfg()
    )
    promoted = _promoted(proposals)
    assert len(promoted) == 1
    meta = promoted[0][4]
    assert meta["gs_kind"] == "pairwise_composed_radial"
    assert meta["gs_promotion"]["evidence"]["promotion_tier"] == "pairwise_composition"


def test_false_radial_pair_claim_rejected_by_alignment():
    """A fabricated (1,2) rotation witness on data that is only radial in
    (0,1) must fail the joint radial-alignment test."""

    rng = np.random.default_rng(32)
    X = rng.uniform(-2.0, 2.0, size=(400, 3))
    r12 = X[:, 0] ** 2 + X[:, 1] ** 2
    grad = np.stack(
        [2 * X[:, 0] * np.cos(r12), 2 * X[:, 1] * np.cos(r12), 0.9 * X[:, 2] ** 2],
        axis=1,
    )
    specs = [_RSpec((0, 1)), _RSpec((1, 2))]
    proposals, diagnostics = compose_pairwise_monomial_proposals(
        specs, x_vals=X, dydx_vals=grad, cols=(0, 1, 2), cfg=_cfg()
    )
    assert proposals == []
    assert any(r.get("reason") == "joint_ray_residual_exceeds_tol" for r in diagnostics)


def test_composed_radial_drives_suppression_of_legacy_radial():
    X, y, grad = _radial_fixture(sigma=1.0e-3)
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1, 2), y_vals=y, cfg=_cfg()
    )
    promoted = _promoted(proposals)
    assert promoted
    legacy_radial = (
        (1, 1, 1),
        promoted[0][1],
        0.9,
        None,
        {"kind": "radial", "form": "r2", "allow_sqrt": True},
    )
    sqrt_only_variant = ((1, 1, 1), promoted[0][1], 0.9, None, {"kind": "radial", "form": "r"})
    filtered, events = suppress_shadowed_stagea_proposals(
        [legacy_radial, sqrt_only_variant], proposals, cols=(0, 1, 2), cfg=_cfg()
    )
    assert legacy_radial not in filtered
    assert sqrt_only_variant in filtered  # different hypothesis: never suppressed
    assert len(events) == 1
    assert events[0]["legacy_kind"] == "radial"


def test_composed_radial_receives_kind_aware_wrappers():
    """The wrapper policy must treat GS promoted reductions by their
    coordinate family: composed radial gets the sqrt(r^2) variant."""

    from types import SimpleNamespace

    from nestynet_sr.sr_search.wrapper_policy import compound_z_wrapper_policy

    X, y, grad = _radial_fixture(sigma=1.0e-3)
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1, 2), y_vals=y, cfg=_cfg()
    )
    promoted = _promoted(proposals)
    assert promoted
    meta = promoted[0][4]
    policy = compound_z_wrapper_policy(
        kind=meta["kind"], pattern=promoted[0][0], meta=meta, search_hp=SimpleNamespace()
    )
    assert bool(getattr(policy, "sqrt"))


# ---------------------------------------------------------------------------
# Quadratic-form (rotation + boost) composition
# ---------------------------------------------------------------------------


class _BSpec(_Spec):
    def __init__(self, axes, **kw):
        super().__init__("boost_pair", axes, **kw)
        self.family = "lorentz"


def _boost_cfg():
    return GeneralizedSymmetryConfig(
        enabled=True,
        mode="propose",
        policy="replace-shadowed",
        general_affine=True,
        lorentz_boosts=True,
        general_affine_charts=("identity", "log"),
        general_affine_promotion_noise_calibrated=True,
        pairwise_composition=True,
    )


def test_mixed_rotation_boost_pairs_compose_minkowski_signature():
    """rotation(0,1) + boost(1,2) propagate signature (+1, +1, -1)."""

    rng = np.random.default_rng(41)
    X = rng.uniform(-2.0, 2.0, size=(400, 3))
    q = X[:, 0] ** 2 + X[:, 1] ** 2 - X[:, 2] ** 2
    gq = np.cos(q) + 0.51 * q**2
    grad = 2.0 * gq[:, None] * X * np.array([1.0, 1.0, -1.0]).reshape(1, -1)
    specs = [_RSpec((0, 1)), _BSpec((1, 2))]
    proposals, _ = compose_pairwise_monomial_proposals(
        specs, x_vals=X, dydx_vals=grad, cols=(0, 1, 2), cfg=_boost_cfg()
    )
    assert len(proposals) == 1
    pattern, _z, _conf, _extra, meta = proposals[0]
    assert pattern == (1, 1, 1)
    assert meta["gs_kind"] == "pairwise_composed_quadratic"
    assert meta["gs_coordinate_kind"] == "quadratic_form"
    assert tuple(meta["gs_quadratic_signature"]) == (1, 1, -1)
    # Indefinite form: no sqrt wrapper contract.
    assert "allow_sqrt" not in meta


def test_spacetime_interval_composes_at_noise():
    """u^2 - x^2 - y^2 - z^2 (signature (1,-1,-1,-1)) via real witnesses."""

    rng = np.random.default_rng(42)
    X = rng.uniform(-2.0, 2.0, size=(700, 4))
    q = X[:, 0] ** 2 - X[:, 1] ** 2 - X[:, 2] ** 2 - X[:, 3] ** 2
    y = np.sin(q) + 0.17 * q**3
    gq = np.cos(q) + 0.51 * q**2
    grad = 2.0 * gq[:, None] * X * np.array([1.0, -1.0, -1.0, -1.0]).reshape(1, -1)
    grad = grad * (1.0 + 1.0e-3 * rng.standard_normal(grad.shape))
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1, 2, 3), y_vals=y, cfg=_boost_cfg()
    )
    promoted = _promoted(proposals)
    assert len(promoted) == 1
    meta = promoted[0][4]
    assert meta["gs_kind"] == "pairwise_composed_quadratic"
    assert tuple(meta["gs_quadratic_signature"]) == (1, -1, -1, -1)
    assert meta["gs_promotion"]["evidence"]["promotion_tier"] == "pairwise_composition"


def test_false_boost_claim_rejected_by_signed_alignment():
    rng = np.random.default_rng(43)
    X = rng.uniform(-2.0, 2.0, size=(400, 3))
    q01 = X[:, 0] ** 2 + X[:, 1] ** 2
    grad = np.stack(
        [2 * X[:, 0] * np.cos(q01), 2 * X[:, 1] * np.cos(q01), 0.9 * X[:, 2] ** 2],
        axis=1,
    )
    specs = [_RSpec((0, 1)), _BSpec((1, 2))]
    proposals, diagnostics = compose_pairwise_monomial_proposals(
        specs, x_vals=X, dydx_vals=grad, cols=(0, 1, 2), cfg=_boost_cfg()
    )
    assert proposals == []
    assert any(r.get("reason") == "joint_ray_residual_exceeds_tol" for r in diagnostics)


# ---------------------------------------------------------------------------
# Two-level composition: products of linear invariants with raw axes
# ---------------------------------------------------------------------------


def _difference_product_fixture(sigma=0.0, n=500, seed=51):
    """f = g((x0 - x1) * x2) on an any-sign domain with analytic gradients."""

    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n, 3))
    w = X[:, 0] - X[:, 1]
    z = w * X[:, 2]
    y = np.sin(z) + 0.17 * z**3
    gz = np.cos(z) + 0.51 * z**2
    grad = np.stack([gz * X[:, 2], -gz * X[:, 2], gz * w], axis=1)
    if sigma > 0.0:
        grad = grad * (1.0 + sigma * rng.standard_normal(grad.shape))
    return X, y, grad


def test_virtual_linear_axis_composes_difference_product():
    """A translation-pair witness becomes a virtual axis whose scaling pair
    with a disjoint raw axis composes (x0 - x1) * x2."""

    X, _y, grad = _difference_product_fixture()

    class _TSpec(_Spec):
        def __init__(self, kind, axes, **kw):
            super().__init__(kind, axes, **kw)
            self.family = "translation"

    specs = [_TSpec("diagonal_plus", (0, 1))]
    proposals, diagnostics = compose_pairwise_monomial_proposals(
        specs, x_vals=X, dydx_vals=grad, cols=(0, 1, 2), cfg=_cfg()
    )
    assert len(proposals) == 1
    pattern, _z, confidence, _extra, meta = proposals[0]
    assert pattern == (1, 1, 1)
    assert meta["gs_kind"] == "pairwise_composed_difference_product"
    assert meta["gs_coordinate_kind"] == "monomial_of_linear"
    assert confidence > 0.9
    z_human = meta["z_human"]
    assert "x2" in z_human and "x0" in z_human and "x1" in z_human
    assert any(r.get("ray_family") == "virtual_monomial" and r.get("status") == "promoted" for r in diagnostics)


def test_difference_product_promotes_with_noise_via_real_witnesses():
    X, y, grad = _difference_product_fixture(sigma=1.0e-3)
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1, 2), y_vals=y, cfg=_cfg()
    )
    promoted = _promoted(proposals)
    assert len(promoted) == 1
    meta = promoted[0][4]
    assert meta["gs_kind"] == "pairwise_composed_difference_product"
    assert meta["gs_promotion"]["evidence"]["promotion_tier"] == "pairwise_composition"


def test_four_var_chain_uses_raw_raw_edges():
    """f = g((x0 - x1) * x2 / x3): the virtual-axis star plus the raw-raw
    scaling edge compose a three-factor chain."""

    rng = np.random.default_rng(52)
    X = rng.uniform(-2.0, 2.0, size=(600, 4))
    w = X[:, 0] - X[:, 1]
    z = w * X[:, 2] / X[:, 3]
    y = np.sin(z) + 0.17 * z**3
    gz = np.cos(z) + 0.51 * z**2
    grad = np.stack(
        [gz * X[:, 2] / X[:, 3], -gz * X[:, 2] / X[:, 3], gz * w / X[:, 3], -gz * z / X[:, 3]],
        axis=1,
    )
    grad = grad * (1.0 + 1.0e-3 * rng.standard_normal(grad.shape))
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1, 2, 3), y_vals=y, cfg=_cfg()
    )
    promoted = _promoted(proposals)
    assert len(promoted) == 1
    meta = promoted[0][4]
    assert meta["gs_kind"] == "pairwise_composed_difference_product"
    assert promoted[0][0] == (1, 1, 1, 1)
    z_human = meta["z_human"]
    assert "x3" in z_human


def test_additive_target_composes_linear_not_product():
    """f = g((x0 - x1) + x2): the correct coordinate is the 3-var linear ray;
    the virtual scaling test must not fabricate a product."""

    rng = np.random.default_rng(53)
    X = rng.uniform(-2.0, 2.0, size=(400, 3))
    z = X[:, 0] - X[:, 1] + X[:, 2]
    y = np.sin(z) + 0.17 * z**3
    gz = np.cos(z) + 0.51 * z**2
    grad = np.stack([gz, -gz, gz], axis=1)
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1, 2), y_vals=y, cfg=_cfg()
    )
    promoted = _promoted(proposals)
    kinds = {p[4].get("gs_kind") for p in promoted}
    assert "pairwise_composed_linear" in kinds
    assert "pairwise_composed_difference_product" not in kinds


# ---------------------------------------------------------------------------
# Generalized virtual axes
# ---------------------------------------------------------------------------


def test_euclidean_distance_carrier_composes_via_virtual_rotation():
    """pb003-style: two translation-pair invariants form a virtual plane whose
    rotation pair composes (x0-x1)**2 + (x2-x3)**2, with the sqrt wrapper."""

    rng = np.random.default_rng(71)
    X = rng.uniform(-2.0, 2.0, size=(600, 4))
    w1 = X[:, 1] - X[:, 0]
    w2 = X[:, 3] - X[:, 2]
    q = w1**2 + w2**2
    y = np.sqrt(q)
    gq = 0.5 / np.sqrt(np.maximum(q, 1.0e-12))
    grad = np.stack([-2 * w1 * gq, 2 * w1 * gq, -2 * w2 * gq, 2 * w2 * gq], axis=1)
    grad = grad * (1.0 + 1.0e-3 * rng.standard_normal(grad.shape))
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1, 2, 3), y_vals=y, cfg=_cfg()
    )
    promoted = _promoted(proposals)
    assert len(promoted) == 1
    meta = promoted[0][4]
    assert meta["gs_kind"] == "pairwise_composed_virtual_quadratic"
    assert meta["gs_coordinate_kind"] == "quadratic_form"
    assert meta.get("allow_sqrt") is True and meta.get("form") == "r2"
    assert promoted[0][0] == (1, 1, 1, 1)
    z_human = meta["z_human"]
    assert z_human.count("**2") == 2


def test_quadratic_virtual_axis_composes_product_with_raw_axis():
    """f = g((x0**2 + x1**2) * x2): the rotation-pair invariant becomes a
    virtual axis whose scaling pair with x2 composes the product."""

    rng = np.random.default_rng(72)
    X = rng.uniform(-2.0, 2.0, size=(500, 3))
    q = X[:, 0] ** 2 + X[:, 1] ** 2
    z = q * X[:, 2]
    y = np.sin(z) + 0.17 * z**2
    gz = np.cos(z) + 0.34 * z
    grad = np.stack([2 * X[:, 0] * X[:, 2] * gz, 2 * X[:, 1] * X[:, 2] * gz, q * gz], axis=1)
    grad = grad * (1.0 + 1.0e-3 * rng.standard_normal(grad.shape))
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1, 2), y_vals=y, cfg=_cfg()
    )
    kinds = {p[4].get("gs_kind"): p[4].get("z_human") for p in _promoted(proposals)}
    assert "pairwise_composed_virtual_product" in kinds
    assert "x2" in kinds["pairwise_composed_virtual_product"]


def test_product_of_two_linear_virtual_axes():
    rng = np.random.default_rng(73)
    X = rng.uniform(-2.0, 2.0, size=(500, 4))
    z = (X[:, 0] - X[:, 1]) * (X[:, 2] + X[:, 3])
    y = np.sin(z) + 0.17 * z**3
    gz = np.cos(z) + 0.51 * z**2
    grad = np.stack(
        [gz * (X[:, 2] + X[:, 3]), -gz * (X[:, 2] + X[:, 3]), gz * (X[:, 0] - X[:, 1]), gz * (X[:, 0] - X[:, 1])],
        axis=1,
    )
    grad = grad * (1.0 + 1.0e-3 * rng.standard_normal(grad.shape))
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1, 2, 3), y_vals=y, cfg=_cfg()
    )
    promoted = _promoted(proposals)
    assert len(promoted) == 1
    meta = promoted[0][4]
    assert meta["gs_kind"] == "pairwise_composed_virtual_product"
    assert promoted[0][0] == (1, 1, 1, 1)


# ---------------------------------------------------------------------------
# Centered pair composition (the geometric shift/preferred-origin lane)
# ---------------------------------------------------------------------------


def _centered_radial_fixture(sigma=0.0, n=600, seed=81):
    """f = g((x0-1)**2 + (x1+2)**2): rotation about a non-origin center."""

    rng = np.random.default_rng(seed)
    X = rng.uniform(-3.0, 3.0, size=(n, 2))
    q = (X[:, 0] - 1.0) ** 2 + (X[:, 1] + 2.0) ** 2
    y = np.sin(q) + 0.17 * np.sqrt(q)
    gq = np.cos(q) + 0.085 / np.sqrt(np.maximum(q, 1.0e-9))
    grad = 2.0 * gq[:, None] * np.stack([X[:, 0] - 1.0, X[:, 1] + 2.0], axis=1)
    if sigma > 0.0:
        grad = grad * (1.0 + sigma * rng.standard_normal(grad.shape))
    return X, y, grad


def test_centered_radial_composes_with_recovered_centers():
    X, y, grad = _centered_radial_fixture(sigma=1.0e-3)
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1), y_vals=y, cfg=_cfg()
    )
    promoted = _promoted(proposals)
    assert len(promoted) == 1
    meta = promoted[0][4]
    assert meta["gs_kind"] == "pairwise_composed_quadratic"
    centers = dict(meta["gs_centers"])
    assert abs(centers[0] - 1.0) < 0.05 and abs(centers[1] + 2.0) < 0.05
    # Definite centered form: sqrt wrapper allowed, but the centered
    # hypothesis must not enter legacy-radial or boost suppression matching.
    assert meta.get("allow_sqrt") is True
    assert "gs_radial_support" not in meta
    assert "gs_quadratic_signature" not in meta
    z_human = meta["z_human"]
    assert "x0 + -0.99" in z_human or "x0 + -1" in z_human
    assert "x1 + 1.99" in z_human or "x1 + 2" in z_human


def test_centered_product_composes_without_suppression_key():
    rng = np.random.default_rng(82)
    X = rng.uniform(-3.0, 3.0, size=(600, 2))
    z = (X[:, 0] - 1.0) * (X[:, 1] + 0.5)
    y = np.sin(z) + 0.17 * z**3
    gz = np.cos(z) + 0.51 * z**2
    grad = np.stack([gz * (X[:, 1] + 0.5), gz * (X[:, 0] - 1.0)], axis=1)
    grad = grad * (1.0 + 1.0e-3 * rng.standard_normal(grad.shape))
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1), y_vals=y, cfg=_cfg()
    )
    promoted = _promoted(proposals)
    assert len(promoted) == 1
    meta = promoted[0][4]
    assert meta["gs_kind"] == "pairwise_composed_monomial"
    centers = dict(meta["gs_centers"])
    assert abs(centers[0] - 1.0) < 0.05 and abs(centers[1] + 0.5) < 0.05
    assert "gs_monomial_exponents_key" not in meta  # origin-only suppression


def test_three_var_centered_radial_requires_consistent_centers():
    rng = np.random.default_rng(83)
    X = rng.uniform(-3.0, 3.0, size=(700, 3))
    c = np.array([1.0, -0.5, 2.0])
    q = np.sum((X - c.reshape(1, -1)) ** 2, axis=1)
    y = np.sin(q) + 0.17 * np.sqrt(q)
    gq = np.cos(q) + 0.085 / np.sqrt(np.maximum(q, 1.0e-9))
    grad = 2.0 * gq[:, None] * (X - c.reshape(1, -1))
    grad = grad * (1.0 + 1.0e-3 * rng.standard_normal(grad.shape))
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1, 2), y_vals=y, cfg=_cfg()
    )
    promoted = _promoted(proposals)
    assert len(promoted) == 1
    centers = dict(promoted[0][4]["gs_centers"])
    for axis, expected in enumerate(c):
        assert abs(centers[axis] - expected) < 0.05


def test_conflicting_pair_centers_reject_component():
    class _USpec(_Spec):
        def __init__(self, axes, coeffs6, **kw):
            super().__init__("unclassified_pair", axes, **kw)
            self.family = "general_affine"
            self.xi_coeffs = coeffs6

    rng = np.random.default_rng(84)
    X = rng.uniform(-3.0, 3.0, size=(300, 3))
    grad = np.ones_like(X)
    # Two centered rotation pairs sharing axis 1 but implying different
    # centers for it: (1, -2) from pair (0,1) and (1, +3) from pair (1,2).
    # Coefficient layout: [b_i, b_j, a_ii, a_ij, a_ji, a_jj], rotation form
    # a_ij = -w, a_ji = +w, centers p = -b_j/a_ji, q = -b_i/a_ij.
    pair01 = _USpec((0, 1), (-2.0 * 0.5, -1.0 * 0.5, 0.0, -0.5, 0.5, 0.0))
    pair12 = _USpec((1, 2), (-1.0 * 0.5, 3.0 * 0.5, 0.0, -0.5, 0.5, 0.0))
    del pair12  # centers for axis 1: -(-1*0.5)/0.5 = 1 vs q = ... construct directly below
    pair12 = _USpec((1, 2), (2.0 * 0.5, -1.0 * 0.5, 0.0, -0.5, 0.5, 0.0))
    proposals, diagnostics = compose_pairwise_monomial_proposals(
        [pair01, pair12], x_vals=X, dydx_vals=grad, cols=(0, 1, 2), cfg=_cfg()
    )
    assert proposals == []
    assert any(
        r.get("reason") in ("inconsistent_pair_centers", "joint_ray_residual_exceeds_tol")
        for r in diagnostics
    )


# ---------------------------------------------------------------------------
# Difference-family suppression matching
# ---------------------------------------------------------------------------


def test_two_var_boost_composes_and_suppresses_power_difference_n2():
    rng = np.random.default_rng(61)
    X = rng.uniform(-2.0, 2.0, size=(500, 2))
    q = X[:, 0] ** 2 - X[:, 1] ** 2
    y = np.sin(q) + 0.17 * q**3
    gq = np.cos(q) + 0.51 * q**2
    grad = 2.0 * gq[:, None] * X * np.array([1.0, -1.0]).reshape(1, -1)
    grad = grad * (1.0 + 1.0e-3 * rng.standard_normal(grad.shape))
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1), y_vals=y, cfg=_boost_cfg()
    )
    promoted = _promoted(proposals)
    assert len(promoted) == 1
    assert tuple(promoted[0][4]["gs_quadratic_signature"]) == (1, -1)

    legacy_n2 = ((2, -2), promoted[0][1], 0.9, None, {"kind": "power_difference", "power": 2, "indices": (0, 1)})
    legacy_n3 = ((3, -3), promoted[0][1], 0.9, None, {"kind": "power_difference", "power": 3, "indices": (0, 1)})
    legacy_recip = (
        (1, -1),
        promoted[0][1],
        0.9,
        None,
        {"kind": "power_pair_sumdiff", "power": 1, "op": "-", "left_inverse": True, "right_inverse": True, "indices": (0, 1)},
    )
    filtered, events = suppress_shadowed_stagea_proposals(
        [legacy_n2, legacy_n3, legacy_recip], proposals, cols=(0, 1), cfg=_boost_cfg()
    )
    assert legacy_n2 not in filtered
    assert legacy_n3 in filtered  # cubic differences are not GS-covered
    assert legacy_recip in filtered  # reciprocal variants are non-affine: never ours
    assert len(events) == 1


def test_power_difference_n1_suppressed_via_promoted_linear():
    rng = np.random.default_rng(62)
    X = rng.uniform(-2.0, 2.0, size=(400, 2))
    z = X[:, 0] - X[:, 1]
    y = np.sin(z) + 0.17 * z**3
    gz = np.cos(z) + 0.51 * z**2
    grad = np.stack([gz, -gz], axis=1)
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1), y_vals=y, cfg=_cfg()
    )
    legacy_n1 = ((1, -1), proposals[0][1], 0.9, None, {"kind": "power_difference", "power": 1, "indices": (0, 1)})
    legacy_sum = (
        (1, 1),
        proposals[0][1],
        0.9,
        None,
        {"kind": "power_pair_sumdiff", "power": 1, "op": "+", "left_inverse": False, "right_inverse": False, "indices": (0, 1)},
    )
    filtered, events = suppress_shadowed_stagea_proposals(
        [legacy_n1, legacy_sum], proposals, cols=(0, 1), cfg=_cfg()
    )
    assert legacy_n1 not in filtered
    assert legacy_sum in filtered  # wrong direction: covector does not match
    assert len(events) == 1


def test_power_diffprod_suppressed_by_composed_difference_product():
    X, y, grad = _difference_product_fixture(sigma=1.0e-3)
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=(0, 1, 2), y_vals=y, cfg=_cfg()
    )
    promoted = _promoted(proposals)
    assert promoted
    legacy_dp = (
        (1, -1, 1),
        promoted[0][1],
        0.9,
        None,
        {"kind": "power_diffprod", "power": 1, "indices": (0, 1, 2), "outer_power": 1, "prefactor_exponents": (0, 0, 1)},
    )
    legacy_dp_other = (
        (1, -1, 1),
        promoted[0][1],
        0.9,
        None,
        {"kind": "power_diffprod", "power": 1, "indices": (0, 2, 1), "outer_power": 1, "prefactor_exponents": (0, 1, 0)},
    )
    legacy_dp_n2 = (
        (2, -2, 1),
        promoted[0][1],
        0.9,
        None,
        {"kind": "power_diffprod", "power": 2, "indices": (0, 1, 2), "outer_power": 1, "prefactor_exponents": (0, 0, 1)},
    )
    filtered, events = suppress_shadowed_stagea_proposals(
        [legacy_dp, legacy_dp_other, legacy_dp_n2], proposals, cols=(0, 1, 2), cfg=_cfg()
    )
    assert legacy_dp not in filtered
    assert legacy_dp_other in filtered  # different pairing: not covered
    assert legacy_dp_n2 in filtered  # squared difference factor: not covered
    assert len(events) == 1


def test_merge_level_composed_proposal_survives_variant_dedup():
    """An ordinary copy of the AST must not erase the certified GS lane."""

    import torch

    from nestynet_sr.sr_core.bridges import AtomNode
    from nestynet_sr.sr_search.search import _detect_compound_variable_for_atom

    class BoseLeaf(torch.nn.Module):
        def forward(self, x):
            z = x[:, 0:1] * x[:, 1:2] / (2 * np.pi * x[:, 2:3] * x[:, 3:4])
            return 1.0 / (torch.exp(z) - 1.0)

        def grad(self, cache):
            x = cache["x"]
            z = x[:, 0:1] * x[:, 1:2] / (2 * np.pi * x[:, 2:3] * x[:, 3:4])
            gz = -torch.exp(z) / (torch.exp(z) - 1.0) ** 2
            d = torch.cat(
                [gz * z / x[:, 0:1], gz * z / x[:, 1:2], -gz * z / x[:, 2:3], -gz * z / x[:, 3:4]],
                dim=1,
            )
            return d.unsqueeze(1)

    rng = np.random.default_rng(2469)
    x_raw = torch.tensor(rng.uniform(1.0, 5.0, size=(400, 4)), dtype=torch.float64)
    y_dummy = torch.zeros((400, 1), dtype=torch.float64)
    atom = AtomNode(
        kind="nn", var_idxs=(0, 1, 2, 3), kwargs={"num_segments": 8, "dual_layer": False}, tag="nn_bose"
    )
    proposals, _ = _detect_compound_variable_for_atom(
        model=object(),
        atom=atom,
        leaf=BoseLeaf(),
        datagen_train=[(x_raw, y_dummy)],
        device=torch.device("cpu"),
        max_batches=1,
        enable_linear=False,
        enable_radial=False,
        enable_shift=False,
        enable_mixed_compound=False,
        trig_axis_specs=None,
        scaling_features=None,
        gs_cfg=_cfg(),
    )
    gs_promoted = [
        p
        for p in proposals
        if len(p) >= 5 and isinstance(p[4], dict) and p[4].get("kind") == "gs_promoted_reduction"
    ]
    assert len(gs_promoted) == 1
    assert tuple(gs_promoted[0][4]["gs_monomial_exponents_key"]) == (1, 1, -1, -1)
    # The ordinary bare monomial and the certified GS copy must coexist; the
    # retained-axis variants (same AST, nonempty extras) survive as well.
    bare_legacy = [
        p
        for p in proposals
        if len(p) >= 5
        and isinstance(p[4], dict)
        and p[4].get("kind") == "monomial"
        and not (p[3] if len(p) >= 4 else None)
        and not p[4].get("retained_axis_wrapper")
        and tuple(int(c) for c, v in zip((0, 1, 2, 3), p[0]) if float(v) != 0.0) == (0, 1, 2, 3)
    ]
    assert len(bare_legacy) == 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
