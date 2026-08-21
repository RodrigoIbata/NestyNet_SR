import numpy as np

from nestynet_sr.sr_gs import GeneralizedSymmetryConfig, discover_generator_specs


def test_gs_rotation_generator_detects_radial_invariant():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(512, 2))
    y = X[:, 0] ** 2 + X[:, 1] ** 2
    G = np.stack([2 * X[:, 0], 2 * X[:, 1]], axis=1)
    cfg = GeneralizedSymmetryConfig(enabled=True, rotations=True, translations=False, scalings=False)
    specs = discover_generator_specs(X, y, G, cols=(0, 1), cfg=cfg)
    assert any(s.family == "rotation" and s.kind == "so2_pair" for s in specs)


def test_gs_scaling_generator_detects_ratio_invariant():
    rng = np.random.default_rng(1)
    X = rng.uniform(0.5, 2.0, size=(512, 2))
    z = X[:, 0] / X[:, 1]
    y = np.sin(z)
    # f_x = cos(z)/x1 ; f_y = -x0*cos(z)/x1^2
    G = np.stack([np.cos(z) / X[:, 1], -X[:, 0] * np.cos(z) / (X[:, 1] ** 2)], axis=1)
    cfg = GeneralizedSymmetryConfig(enabled=True, scalings=True, translations=False, rotations=False)
    specs = discover_generator_specs(X, y, G, cols=(0, 1), cfg=cfg)
    assert any(s.family == "scaling" and s.kind == "common_pair" for s in specs)


def test_gs_diagonal_translation_detects_difference_invariant():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(512, 2))
    z = X[:, 0] - X[:, 1]
    y = z ** 3
    G = np.stack([3 * z ** 2, -3 * z ** 2], axis=1)
    cfg = GeneralizedSymmetryConfig(enabled=True, diagonal_translations=True, translations=False, scalings=False, rotations=False)
    specs = discover_generator_specs(X, y, G, cols=(0, 1), cfg=cfg)
    assert any(s.family == "translation" and s.kind == "diagonal_plus" for s in specs)


def test_stagea_bridge_does_not_emit_quotient_for_pure_equivariance():
    from nestynet_sr.sr_gs.stagea_bridge import stageA_generalized_symmetry_proposals

    class Leaf:
        def __call__(self, x):
            return (x[:, 0:1] ** 2 + x[:, 1:2] ** 2)

    rng = np.random.default_rng(3)
    X = rng.normal(size=(256, 2))
    G = np.stack([2 * X[:, 0], 2 * X[:, 1]], axis=1)
    cfg = GeneralizedSymmetryConfig(
        enabled=True,
        rotations=False,
        scalings=True,
        translations=False,
        diagonal_translations=False,
        output_equivariance=True,
    )
    props, diag = stageA_generalized_symmetry_proposals(
        atom=None, leaf=Leaf(), x_vals=X, dydx_vals=G, cols=(0, 1), cfg=cfg
    )
    assert diag  # sees the homogeneity witness
    assert props == []  # but does not pretend y descends to x0/x1


def test_general_affine_detects_rotation_without_named_generators():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(768, 2))
    y = X[:, 0] ** 2 + X[:, 1] ** 2
    G = np.stack([2 * X[:, 0], 2 * X[:, 1]], axis=1)
    cfg = GeneralizedSymmetryConfig(
        enabled=True,
        known_generators=False,
        known_lie=False,
        general_affine=True,
        rotations=False,
        scalings=False,
        translations=False,
        diagonal_translations=False,
        residual_tol=0.05,
        audit_residual_tol=0.12,
        min_confidence=0.5,
    )
    specs = discover_generator_specs(X, y, G, cols=(0, 1), cfg=cfg)
    assert any(s.family == "general_affine" and s.kind == "affine_rotation_pair" for s in specs)


def test_jet_additive_separability_wraps_hessian_condition():
    import torch
    from nestynet_sr.sr_gs import jet_separability_candidates

    x0 = torch.linspace(-1.0, 1.0, 128)
    # f(x0,x1)=x0^2+sin(x1); mixed Hessian vanishes.
    y = torch.zeros_like(x0)
    grad = torch.stack([2 * x0, torch.cos(x0)], dim=1)
    hess = torch.zeros((x0.numel(), 2, 2), dtype=torch.float64)
    hess[:, 0, 0] = 2.0
    hess[:, 1, 1] = -torch.sin(x0)
    proposals, _ra, _rm, diagnostics = jet_separability_candidates(
        symb=[0, 1],
        y_norm=y,
        dydx_norm=grad,
        d2ydx2_norm=hess,
        precision_sum=1.0e-6,
        precision_mult=1.0e-6,
        jet_separability=True,
        jet_multiplicative=False,
    )
    assert proposals
    assert any(d.get("kind") == "additive_hessian_block" and d.get("accepted") for d in diagnostics)


def test_named_output_equivariance_is_stable_under_large_output_offset():
    X = np.linspace(-1.0, 1.0, 512)[:, None]
    y = 1.0e9 + X[:, 0]
    G = np.ones_like(X)
    cfg = GeneralizedSymmetryConfig(
        enabled=True,
        translations=True,
        diagonal_translations=False,
        scalings=False,
        rotations=False,
        output_equivariance=True,
        residual_tol=1.0e-8,
        audit_residual_tol=1.0e-6,
        min_confidence=0.5,
    )
    spec = next(s for s in discover_generator_specs(X, y, G, cfg=cfg) if s.family == "translation")
    assert spec.accepted
    assert abs(spec.output_alpha - 1.0) < 1.0e-12
    assert abs(spec.output_beta) < 1.0e-12
    assert spec.evidence["used_output_action"] is True


def test_general_affine_discovers_output_equivariance_via_projected_nullspace():
    rng = np.random.default_rng(123)
    X = rng.uniform(-1.0, 1.0, size=(1024, 2))
    e0 = np.exp(X[:, 0])
    e1 = np.exp(2.0 * X[:, 1])
    y = e0 + e1
    G = np.stack([e0, 2.0 * e1], axis=1)
    cfg = GeneralizedSymmetryConfig(
        enabled=True,
        known_generators=False,
        known_lie=False,
        general_affine=True,
        translations=False,
        diagonal_translations=False,
        scalings=False,
        rotations=False,
        output_equivariance=True,
        residual_tol=1.0e-8,
        audit_residual_tol=1.0e-6,
        min_confidence=0.5,
    )
    spec = next(s for s in discover_generator_specs(X, y, G, cfg=cfg) if s.family == "general_affine")
    assert spec.accepted
    assert spec.kind == "affine_translation_pair"
    assert np.allclose(np.asarray(spec.xi_coeffs), np.asarray([1.0, 0.5]), atol=1.0e-10)
    assert abs(spec.output_beta - 1.0) < 1.0e-12
    assert spec.residual_rel < 1.0e-12
    assert spec.evidence["svd_objective"] == "projected_output_equivariance"


def test_general_affine_preserves_noninteger_translation_direction():
    rng = np.random.default_rng(321)
    X = rng.normal(size=(1024, 2))
    a = np.sqrt(2.0)
    z = a * X[:, 0] - X[:, 1]
    y = np.sin(z)
    c = np.cos(z)
    G = np.stack([a * c, -c], axis=1)
    cfg = GeneralizedSymmetryConfig(
        enabled=True,
        known_generators=False,
        known_lie=False,
        general_affine=True,
        translations=False,
        diagonal_translations=False,
        scalings=False,
        rotations=False,
        output_equivariance=False,
        residual_tol=1.0e-8,
        audit_residual_tol=1.0e-6,
        min_confidence=0.5,
    )
    spec = next(s for s in discover_generator_specs(X, y, G, cfg=cfg) if s.kind == "affine_translation_pair")
    assert spec.accepted
    b = np.asarray(spec.xi_coeffs)
    assert np.allclose(b / b[1], np.asarray([1.0 / np.sqrt(2.0), 1.0]), atol=1.0e-10)
    assert spec.evidence["structured_family"] == "translation"
    assert spec.residual_rel < 1.0e-12
