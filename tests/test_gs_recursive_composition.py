"""Recursive coordinate composition (gated, default-off).

After first-level GS promotes a coordinate ``z1 = g(x)`` (e.g. a monomial), the
recursion treats it as a virtual axis and re-runs the pairwise-witness
composition to discover nested coordinates such as ``(x0*x1/x2) - x3``.
"""

import numpy as np
import torch

from nestynet_sr.sr_gs import (
    GeneralizedSymmetryConfig,
    compose_recursive_coordinate_proposals,
)
from nestynet_sr.sr_gs.stagea_bridge import stageA_generalized_symmetry_proposals
from nestynet_sr.sr_core.bridges import MulNode, Var, eval_input_expr


def _cfg(recursive=True, pairwise=True):
    return GeneralizedSymmetryConfig(
        enabled=True,
        mode="propose",
        policy="augment",
        general_affine=True,
        general_affine_charts=("identity", "log"),
        general_affine_promotion_noise_calibrated=True,
        pairwise_composition=pairwise,
        recursive_composition=recursive,
    )


def _nested_monomial_minus_axis(n=400, seed=404):
    """f = g((x0*x1/x2) - x3) with generic g and analytic gradients."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.7, 3.0, size=(n, 4))
    w = X[:, 0] * X[:, 1] / X[:, 2]
    z = w - X[:, 3]
    y = np.sin(z) + 0.17 * z**3
    gz = np.cos(z) + 0.51 * z**2
    grad = np.stack(
        [gz * X[:, 1] / X[:, 2], gz * X[:, 0] / X[:, 2], -gz * w / X[:, 2], -gz],
        axis=1,
    )
    return X, y, grad, z


def _additively_separable(n=400, seed=404):
    """f = sin(x0*x1/x2) + cos(x3): w is promoted but does NOT compose with x3."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.7, 3.0, size=(n, 4))
    w = X[:, 0] * X[:, 1] / X[:, 2]
    y = np.sin(w) + np.cos(X[:, 3])
    cw = np.cos(w)
    grad = np.stack(
        [cw * X[:, 1] / X[:, 2], cw * X[:, 0] / X[:, 2], -cw * w / X[:, 2], -np.sin(X[:, 3])],
        axis=1,
    )
    return X, y, grad


def _depth2(props):
    return [p for p in props if p[4].get("gs_recursive_depth") == 2]


def _run(cfg, X, y, grad, cols=(0, 1, 2, 3)):
    props, _diag = stageA_generalized_symmetry_proposals(
        atom=None, leaf=None, x_vals=X, dydx_vals=grad, cols=cols, y_vals=y, cfg=cfg
    )
    return props


def test_recursive_composition_contract():
    """Nested (x0*x1/x2) - x3 yields a depth-2 proposal matching the true coordinate."""
    X, y, grad, z_true = _nested_monomial_minus_axis()
    props = _run(_cfg(recursive=True), X, y, grad)
    rec = _depth2(props)
    assert rec, "expected a depth-2 recursive proposal"
    p = rec[0]
    meta = p[4]
    assert meta["gs_source_family"] == "recursive_composition"
    assert meta["gs_recursive_inner"], "inner coordinate should be recorded"
    # The inner coordinate is the first-level monomial.
    assert "x0" in meta["gs_recursive_inner"][0] and "x2" in meta["gs_recursive_inner"][0]
    # The composed coordinate reproduces (x0*x1/x2) - x3 up to affine equivalence.
    z_vals = eval_input_expr(p[1], torch.as_tensor(X, dtype=torch.float64)).detach().numpy().reshape(-1)
    assert abs(np.corrcoef(z_vals, z_true)[0, 1]) > 1.0 - 1.0e-8
    # Support covers all four raw axes.
    assert tuple(i for i, v in enumerate(p[0]) if v) == (0, 1, 2, 3)


def test_recursive_composition_five_var_free_axis():
    """An unused free axis (x4) does not block the nested (x0*x1/x2) - x3 discovery."""
    rng = np.random.default_rng(404)
    X = rng.uniform(0.7, 3.0, size=(400, 5))
    w = X[:, 0] * X[:, 1] / X[:, 2]
    z = w - X[:, 3]
    y = np.sin(z) + 0.17 * z**3
    gz = np.cos(z) + 0.51 * z**2
    grad = np.stack(
        [gz * X[:, 1] / X[:, 2], gz * X[:, 0] / X[:, 2], -gz * w / X[:, 2], -gz, np.zeros(400)],
        axis=1,
    )
    props = _run(_cfg(recursive=True), X, y, grad, cols=(0, 1, 2, 3, 4))
    rec = _depth2(props)
    assert rec, "expected a depth-2 proposal with a spectator axis present"
    z_vals = eval_input_expr(rec[0][1], torch.as_tensor(X, dtype=torch.float64)).detach().numpy().reshape(-1)
    assert abs(np.corrcoef(z_vals, z)[0, 1]) > 1.0 - 1.0e-8


def test_recursive_composition_supports_noncontiguous_global_columns():
    rng = np.random.default_rng(406)
    X = rng.uniform(0.7, 3.0, size=(500, 3))
    inner = X[:, 0] * X[:, 1]
    z = inner + X[:, 2]
    y = np.sin(z) + 0.1 * z**2
    outer_grad = np.cos(z) + 0.2 * z
    grad = np.column_stack(
        (outer_grad * X[:, 1], outer_grad * X[:, 0], outer_grad)
    )
    inner_ast = MulNode(Var(2), Var(5))
    primitive = (
        (1, 1, 0),
        inner_ast,
        1.0,
        None,
        {
            "kind": "gs_scaling",
            "source": "generalized_symmetry",
            "candidate_role": "inner_coordinate",
            "carrier_certified": True,
            "gs_carrier_depth": 1,
            "gs_carrier_fingerprint": "x2*x5",
        },
    )

    proposals, _ = compose_recursive_coordinate_proposals(
        [primitive],
        x_vals=X,
        y_vals=y,
        dydx_vals=grad,
        cols=(2, 5, 7),
        cfg=_cfg(recursive=True),
    )

    assert proposals
    X_global = np.zeros((X.shape[0], 8))
    X_global[:, (2, 5, 7)] = X
    values = (
        eval_input_expr(
            proposals[0][1],
            torch.as_tensor(X_global, dtype=torch.float64),
        )
        .detach()
        .numpy()
        .reshape(-1)
    )
    assert abs(np.corrcoef(values, z)[0, 1]) > 1.0 - 1.0e-8


def test_recursive_composition_negative_control():
    """Additively separable f = sin(w) + cos(x3) yields no depth-2 proposal."""
    X, y, grad = _additively_separable()
    props = _run(_cfg(recursive=True), X, y, grad)
    assert not _depth2(props), "additive separability must not compose a nested coordinate"


def test_recursive_composition_default_off():
    """With the flag off (pairwise still on) no depth-2 proposal is produced."""
    X, y, grad, _z = _nested_monomial_minus_axis()
    props_off = _run(_cfg(recursive=False, pairwise=True), X, y, grad)
    assert not _depth2(props_off)
    # And the first-level monomial is still promoted (baseline unchanged).
    assert any(p[4].get("kind") == "gs_promoted_reduction" for p in props_off)


def test_recursive_composition_gate_requires_pairwise():
    """recursive_composition_active() is False unless pairwise composition is on."""
    cfg = _cfg(recursive=True, pairwise=False)
    assert not cfg.recursive_composition_active()
    X, y, grad, _z = _nested_monomial_minus_axis()
    props = _run(cfg, X, y, grad)
    assert not _depth2(props)
