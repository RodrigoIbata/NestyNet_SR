import numpy as np
import torch

from nestynet_sr.sr_gs import (
    GeneralizedSymmetryConfig,
    discover_generator_specs,
    jet_separability_candidates,
)
from nestynet_sr.sr_gs.policy import (
    canonical_gs_policy,
    policy_replaces_affine_shadow,
    policy_replaces_jet_separability,
)


def test_v3_policy_aliases_and_replacement_semantics():
    assert canonical_gs_policy("augment") == "augment"
    assert canonical_gs_policy("replace_baseline") == "replace-shadowed"
    assert canonical_gs_policy("gs_only") == "gs-only-affine"
    assert policy_replaces_affine_shadow("replace-shadowed")
    assert policy_replaces_affine_shadow("gs-only-affine")
    assert policy_replaces_jet_separability("replace-shadowed")


def test_v3_jet_additive_separability_is_mixed_hessian_condition():
    n = 32
    symb = [0, 1]
    y_norm = torch.ones(n)
    dydx_norm = torch.zeros(n, 2)
    hess_norm = torch.zeros(n, 2, 2)
    hess_norm[:, 0, 0] = 2.0
    hess_norm[:, 1, 1] = 6.0

    proposals, rest_add, rest_mult, diagnostics = jet_separability_candidates(
        symb=symb,
        y_norm=y_norm,
        dydx_norm=dydx_norm,
        d2ydx2_norm=hess_norm,
        precision_sum=1e-8,
        precision_mult=1e-8,
        very_verbose=False,
    )

    assert any(p[0] is torch.add and set(p[1]) == {0} and set(p[2]) == {1} for p in proposals)
    assert any(d.get("kind") == "additive_hessian_block" and d.get("accepted") for d in diagnostics)
    assert rest_add is None
    assert rest_mult is None


def test_v3_general_affine_detects_ratio_without_named_scaling():
    rng = np.random.default_rng(123)
    X = rng.uniform(0.5, 2.0, size=(384, 2))
    z = X[:, 0] / X[:, 1]
    y = np.sin(z)
    G = np.stack([
        np.cos(z) / X[:, 1],
        -X[:, 0] * np.cos(z) / (X[:, 1] ** 2),
    ], axis=1)

    cfg = GeneralizedSymmetryConfig(
        enabled=True,
        mode="audit",
        policy="gs-only-affine",
        known_generators=False,
        known_lie=False,
        general_affine=True,
        output_equivariance=False,
        min_confidence=0.5,
        affine_max_generators=4,
    )
    specs = discover_generator_specs(X, y, G, cols=(0, 1), cfg=cfg)
    assert any(s.family == "general_affine" and s.kind == "affine_common_scaling_pair" and s.accepted for s in specs)
