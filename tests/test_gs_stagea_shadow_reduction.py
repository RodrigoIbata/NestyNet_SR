import numpy as np

from nestynet_sr.sr_gs import GeneralizedSymmetryConfig
from nestynet_sr.sr_gs.stagea_bridge import stageA_generalized_symmetry_proposals


class RatioLeaf:
    def __call__(self, x):
        return x[:, 0:1] / x[:, 1:2]


def test_stagea_general_affine_shadow_reduction_is_diagnostic_only_in_audit_mode():
    rng = np.random.default_rng(701)
    X = rng.uniform(0.5, 2.0, size=(128, 2))
    z = X[:, 0] / X[:, 1]
    y = np.sin(z) + 0.17 * z**3
    grad_z = np.cos(z) + 0.51 * z**2
    grad = np.stack([grad_z / X[:, 1], -grad_z * X[:, 0] / X[:, 1] ** 2], axis=1)
    cfg = GeneralizedSymmetryConfig(
        enabled=True,
        mode="audit",
        known_generators=False,
        known_lie=False,
        general_affine=True,
        residual_tol=1.0e-8,
    )

    proposals, diag = stageA_generalized_symmetry_proposals(
        atom=None,
        leaf=RatioLeaf(),
        x_vals=X,
        y_vals=y,
        dydx_vals=grad,
        cols=(0, 1),
        cfg=cfg,
    )

    assert proposals == []
    shadow_rows = [d for d in diag if d.get("kind") == "shadow_reduction"]
    assert len(shadow_rows) == 1
    row = shadow_rows[0]
    assert row["shadow_only"]
    assert not row["used_for_proposal"]
    assert not row["used_for_selection"]
    assert not row["active_candidate"]
    assert row["reduction"]["status"] == "compiled"
    assert row["reduction"]["invariant_coordinates"]
