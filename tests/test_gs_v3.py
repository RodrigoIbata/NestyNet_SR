import numpy as np
import torch

from nestynet_sr.sr_gs import GeneralizedSymmetryConfig, discover_generator_specs
from nestynet_sr.sr_gs.jet import jet_separability_candidates


def test_general_affine_finds_ratio_with_known_lie_disabled():
    rng = np.random.default_rng(123)
    X = rng.uniform(0.5, 2.0, size=(256, 2))
    G = np.stack([1.0 / X[:, 1], -X[:, 0] / (X[:, 1] ** 2)], axis=1)
    y = X[:, 0] / X[:, 1]
    cfg = GeneralizedSymmetryConfig(
        enabled=True,
        mode="auto",
        known_lie=False,
        general_affine=True,
        min_confidence=0.3,
        residual_tol=0.05,
        audit_residual_tol=0.2,
    )
    specs = discover_generator_specs(X, y, G, cols=(0, 1), cfg=cfg, include_rejected=True)
    accepted = [s for s in specs if s.accepted]
    assert any(s.family == "general_affine" and s.kind == "affine_common_scaling_pair" for s in accepted)


def test_jet_additive_separability_detects_mixed_hessian_zero():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(128, 2))
    y = torch.as_tensor(X[:, 0] ** 2 + np.sin(X[:, 1]), dtype=torch.float64)
    G = torch.as_tensor(np.stack([2 * X[:, 0], np.cos(X[:, 1])], axis=1), dtype=torch.float64)
    H = np.zeros((X.shape[0], 2, 2))
    H[:, 0, 0] = 2.0
    H[:, 1, 1] = -np.sin(X[:, 1])
    H = torch.as_tensor(H, dtype=torch.float64)
    proposed, _ra, _rm, diagnostics = jet_separability_candidates(
        symb=(0, 1),
        y_norm=y,
        dydx_norm=G,
        d2ydx2_norm=H,
        precision_sum=0.03,
        precision_mult=0.03,
    )
    assert proposed
    assert any(d["kind"].startswith("additive") for d in diagnostics)
