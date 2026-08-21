"""Contract for the standalone warp-discovery experiment (sr_gs/warp_discovery.py).

Discovers the per-axis coordinate warp that makes a hidden generalized-additive
symmetry affine: certifies via the pair-independent normalized Hessian, recovers
per-axis warps from pairwise log-gradient differences, and validates the
recovered coordinate (including its affine covector) by a rank-1 test.
"""

import numpy as np
import torch

from nestynet_sr.sr_gs.warp_discovery import (
    certify_warped_additivity,
    discover_warp,
    numerical_grad_hess,
)


def _gh(fn, X):
    _f, g, h = numerical_grad_hess(fn, X)
    return g, h


def _X(n=500, d=3, lo=0.6, hi=1.8, seed=0):
    return np.random.default_rng(seed).uniform(lo, hi, size=(n, d))


def test_radial_warp_recovered_as_square():
    X = _X()
    g, h = _gh(lambda X: torch.sin((X**2).sum(1)) + 0.3 * (X**2).sum(1), X)
    cert = discover_warp(X, g, h)
    assert cert.is_separable_after_warp
    assert cert.reason == "warp_recovered"
    assert [w.kind for w in cert.warps] == ["square", "square", "square"]
    assert cert.warp_validation_residual < 1e-6


def test_multiplicative_warp_recovered_as_log():
    X = _X()
    g, h = _gh(lambda X: torch.sin(X.prod(1)) + 0.3 * X.prod(1), X)
    cert = discover_warp(X, g, h)
    assert cert.reason == "warp_recovered"
    assert [w.kind for w in cert.warps] == ["log", "log", "log"]
    assert cert.warp_validation_residual < 1e-6


def test_reciprocal_warp_recovered():
    X = _X()
    s = lambda X: 1 / X[:, 0] + 1 / X[:, 1] + 1 / X[:, 2]
    g, h = _gh(lambda X: torch.sin(s(X)) + 0.2 * s(X) ** 2, X)
    cert = discover_warp(X, g, h)
    assert cert.reason == "warp_recovered"
    assert [w.kind for w in cert.warps] == ["reciprocal", "reciprocal", "reciprocal"]


def test_linear_warp_is_identity_and_recovers_covector():
    X = _X()
    g, h = _gh(lambda X: torch.sin(X[:, 0] - 2 * X[:, 1] + X[:, 2]), X)
    cert = discover_warp(X, g, h)
    assert cert.warp_is_trivial  # identity chart; no warp needed
    assert [w.kind for w in cert.warps] == ["identity"] * 3
    # covector recovered up to overall sign/scale: proportional to (1, -2, 1)
    c = np.array(cert.evidence["covector"], dtype=float)
    c = c / c[0]
    assert np.allclose(c, [1.0, -2.0, 1.0], atol=1e-3)


def test_nonseparable_control_is_rejected_and_localized():
    X = _X()
    s = lambda X: X[:, 0] * X[:, 1] + X[:, 0] + X[:, 2]
    g, h = _gh(lambda X: torch.sin(s(X)) + 0.2 * s(X) ** 2, X)
    cert = discover_warp(X, g, h)
    assert not cert.is_separable_after_warp
    assert cert.warps is None
    assert (0, 1) in cert.interacting_pairs  # the coupled pair is fingered
    assert (0, 2) not in cert.interacting_pairs and (1, 2) not in cert.interacting_pairs


def test_partial_power_axis_flagged_empirical():
    """A non-power axis (sin x0) is certified additive but flagged empirical."""
    X = _X()
    s = lambda X: torch.sin(X[:, 0]) + X[:, 1] ** 3 + torch.log(X[:, 2])
    g, h = _gh(lambda X: torch.sin(s(X)) + 0.2 * s(X) ** 2, X)
    cert = discover_warp(X, g, h)
    assert cert.is_separable_after_warp  # certificate passes (it IS additive)
    kinds = {w.axis: w.kind for w in cert.warps}
    assert kinds[0] == "empirical"  # sin is not a power warp
    assert kinds[1] == "power" and abs(cert.warps[1].exponent - 3.0) < 1e-6
    assert kinds[2] == "log"
    assert cert.reason == "warp_empirical_or_unvalidated"
    assert cert.coordinate_human is None  # not a clean full-dictionary discovery


def test_certificate_survives_mild_noise():
    X = _X()
    g, h = _gh(lambda X: torch.sin(X.prod(1)) + 0.3 * X.prod(1), X)
    rng = np.random.default_rng(1)
    gn = g + 1e-3 * np.std(g) * rng.standard_normal(g.shape)
    hn = h + 1e-3 * np.std(h) * rng.standard_normal(h.shape)
    cert = certify_warped_additivity(X, gn, hn)
    assert cert.is_separable_after_warp
    assert cert.pair_consistency < 0.05


# --- Bridge integration: the discovered-warp chart (--gs-charts ...,warp) ------

from nestynet_sr.sr_gs import GeneralizedSymmetryConfig  # noqa: E402
from nestynet_sr.sr_gs.stagea_bridge import (  # noqa: E402
    stageA_generalized_symmetry_proposals,
)


def _cfg(charts, enabled=True):
    return GeneralizedSymmetryConfig(
        enabled=enabled, mode="propose", policy="augment", general_affine=True,
        general_affine_charts=charts, general_affine_promotion_noise_calibrated=True,
    )


def _run(leaf, X, charts, cols=(0, 1, 2), enabled=True):
    Xt = torch.tensor(X, requires_grad=True)
    (g,) = torch.autograd.grad(leaf(Xt).sum(), Xt)
    return stageA_generalized_symmetry_proposals(
        atom=None, leaf=leaf, x_vals=X, dydx_vals=g.detach().numpy(),
        cols=cols, cfg=_cfg(charts, enabled),
    )


def _warp_props(props):
    return [p for p in props if p[4].get("gs_chart") == "warp"]


def test_warp_chart_promotes_radial_through_bridge():
    X = _X()
    props, _d = _run(lambda X: torch.sin((X**2).sum(1)) + 0.3 * (X**2).sum(1), X, ("identity", "warp"))
    wp = _warp_props(props)
    assert len(wp) == 1
    m = wp[0][4]
    assert m["kind"] == "gs_promoted_reduction" and m["gs_source_family"] == "warp_discovery"
    assert m["gs_warp_kinds"] == ("square", "square", "square")
    assert m["gs_linear_covector"] is not None  # covector carried for suppression


def test_warp_chart_promotes_mixed_power_coordinate():
    """The genuinely-new capability: a mixed-power warp no fixed chart reaches."""
    X = _X()
    s = lambda X: X[:, 0] ** 2 + X[:, 1] ** 3 + X[:, 2] ** 2
    props, _d = _run(lambda X: torch.sin(s(X)) + 0.2 * s(X) ** 2, X, ("identity", "warp"))
    wp = _warp_props(props)
    assert len(wp) == 1
    assert wp[0][4]["gs_warp_kinds"] == ("square", "power", "square")


def test_warp_chart_default_off_and_gs_off():
    X = _X()
    leaf = lambda X: torch.sin((X**2).sum(1)) + 0.3 * (X**2).sum(1)
    assert _warp_props(_run(leaf, X, ("identity",))[0]) == []  # default charts
    assert _warp_props(_run(leaf, X, ("identity", "warp"), enabled=False)[0]) == []  # GS off


def test_warp_chart_rejects_nonseparable():
    X = _X()
    s = lambda X: X[:, 0] * X[:, 1] + X[:, 0] + X[:, 2]
    props, diag = _run(lambda X: torch.sin(s(X)) + 0.2 * s(X) ** 2, X, ("identity", "warp"))
    assert _warp_props(props) == []
    wrow = [d for d in diag if d.get("chart") == "warp"][0]
    assert wrow["promotion_reason"] == "warp_not_pair_consistent"


def test_warp_chart_cross_chart_dedup_with_log():
    """An all-log warp reproduces the log-chart monomial; it must dedup, not double."""
    X = _X()
    props, diag = _run(lambda X: torch.sin(X.prod(1)) + 0.3 * X.prod(1), X, ("identity", "log", "warp"))
    promoted_charts = [p[4].get("gs_chart") for p in props if p[4].get("kind") == "gs_promoted_reduction"]
    assert "log" in promoted_charts and "warp" not in promoted_charts
    dup = [d for d in diag if d.get("chart") == "warp" and d.get("cross_chart_duplicate")]
    assert dup


def test_warp_chart_needs_three_vars():
    X = _X(d=2)
    props, diag = _run(lambda X: torch.sin((X**2).sum(1)), X, ("identity", "warp"), cols=(0, 1))
    assert _warp_props(props) == []
    wrow = [d for d in diag if d.get("chart") == "warp"][0]
    assert wrow["promotion_reason"] == "warp_needs_3plus_vars"
