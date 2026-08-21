# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
"""Regression tests for the H^2 (curvature) Sobolev residual.

Covers the machinery added for curvature-supervised surrogate training:
  * ASTCompositeAdaptor.grad_grad_jvp / grad_grad_vjp routing to the leaf's
    analytic Hessian parameter-Jacobian (the dual-layer composition, previously
    untested vs the single-model audits);
  * SobolevGradientAdaptor H^2 block: jvp/vjp adjoint consistency over the full
    value+gradient+Hessian output stack, out_dim routing, and backward
    compatibility when H^2 is disabled.
"""
from __future__ import annotations

import torch

from nestynet_sr.adaptors.sobolev_residual import SobolevGradientAdaptor
from nestynet_sr.sr_core import build_initial_ast
from nestynet_sr.sr_search.config import ModelHyperparams
from nestynet_sr.sr_search.model_builders import LeafBuilder, build_composite_ast

DT = torch.float64


def _build_composite(Nx: int = 4, S: int = 8):
    mhp = ModelHyperparams(
        double_precision=True, repeatable_runs=True, model_base_name="G_Model",
        num_segments_min=S, num_segments_max=S, Nout_size=1,
    )
    lb = LeafBuilder(mhp, torch.device("cpu"), DT)
    ast0 = build_initial_ast(Nxvars=Nx, num_segments=S, dual_layer=True, tag="A0")
    comp, _npar, _ = build_composite_ast(
        ast0, S, dual_layer=True, leaf_builder=lb, device=torch.device("cpu"), dtype=DT
    )
    comp.declare_global_input_dim(Nx)
    return comp


def test_composite_grad_grad_jvp_matches_leaf_autograd():
    torch.manual_seed(0)
    Nx, B = 4, 5
    comp = _build_composite(Nx)
    x = torch.randn(B, Nx, dtype=DT)
    cache = comp.build_cache((x, comp(x).detach()))
    v = torch.randn(comp.num_parameters(), dtype=DT)
    Hv = comp.grad_grad_jvp(cache, v, out_dim=None)
    ref = comp.leaf[0]._autograd_grad_grad_jvp_from_cache(
        cache, v, out_dim=None
    )
    rel = (Hv - ref).norm() / (ref.norm() + 1e-30)
    assert float(rel) < 1e-7, f"grad_grad_jvp vs autograd rel={float(rel):.3e}"


def test_composite_grad_grad_vjp_adjoint():
    torch.manual_seed(1)
    Nx, B = 4, 5
    comp = _build_composite(Nx)
    x = torch.randn(B, Nx, dtype=DT)
    cache = comp.build_cache((x, comp(x).detach()))
    v = torch.randn(comp.num_parameters(), dtype=DT)
    Hv = comp.grad_grad_jvp(cache, v, out_dim=None)
    if Hv.dim() == 4:
        Hv = Hv[:, 0]
    w = torch.randn(B, 1, Nx, Nx, dtype=DT)
    w = 0.5 * (w + w.transpose(-1, -2))
    JTw = comp.grad_grad_vjp(cache, w, out_dim=None).reshape(-1)
    lhs = (w[:, 0] * Hv).sum()
    rhs = (JTw * v).sum()
    rel = (lhs - rhs).abs() / (rhs.abs() + 1e-30)
    assert float(rel) < 1e-9, f"grad_grad vjp adjoint rel={float(rel):.3e}"


def test_composite_grad_grad_jvp_guard_on_nontrivial_ast():
    """The single-feature-atom guard must fire (loudly) for a non-atom root."""
    import pytest
    from nestynet_sr.sr_core.bridges import AddNode
    comp = _build_composite(4)
    comp.ast_root = AddNode(comp.ast_root, comp.ast_root)  # pretend a 2-node AST
    x = torch.randn(3, 4, dtype=DT)
    cache = comp.build_cache((x, comp(x).detach())) if False else {"x": x, "leaves": []}
    with pytest.raises(NotImplementedError):
        comp.grad_grad_jvp(cache, torch.zeros(comp.num_parameters(), dtype=DT))


def test_sobolev_h2_adjoint_and_out_dim():
    torch.manual_seed(2)
    Nx, B = 4, 6
    comp = _build_composite(Nx)
    sob = SobolevGradientAdaptor(
        comp, axes=(0, 1, 2, 3), value_weight=1.0, grad_weight=0.7,
        grad_scales=[1.0, 1.1, 0.9, 1.2],
        hess_pairs=[(1, 1), (2, 2), (3, 3)], hess_weight=0.5, hess_scales=[1.3, 0.8, 1.0],
    )
    assert sob.n_outputs == 1 + 4 + 3
    x = torch.randn(B, Nx, dtype=DT)
    y_aug = torch.randn(B, sob.n_outputs, dtype=DT)
    cache = sob.build_cache((x, y_aug))
    assert tuple(sob.residuals(cache).shape) == (B, sob.n_outputs)

    v = torch.randn(sob.num_parameters(), dtype=DT)
    w = torch.randn(B, sob.n_outputs, dtype=DT)
    Jv = sob.jvp(cache, v)
    JTw = sob.vjp(cache, w).sum(dim=0)  # stacked per-output rows -> summed
    lhs = (w * Jv).sum()
    rhs = (JTw * v).sum()
    assert float((lhs - rhs).abs() / (rhs.abs() + 1e-30)) < 1e-9
    for k in range(sob.n_outputs):
        jk = sob.jvp(cache, v, out_dim=k).reshape(-1)
        assert float((jk - Jv[:, k]).abs().max()) < 1e-11


def test_sobolev_h2_fast_diagonal_jacobian_matches_jvp():
    """The fast dense `_jacobian_fast` H² block (leaf selected-diagonal Jacobian)
    must equal the residual Jacobian.  Cross-check: jvp()'s hess block uses the
    *full-Hessian* path (grad_grad_jvp), independent of the diagonal Jacobian path,
    so agreement validates both.  residuals() fixes r_grad=(y−g), r_hess=(y−H), so
    the residual Jacobian is −(prediction jvp) on the grad+hess blocks."""
    torch.manual_seed(5)
    Nx, B = 4, 6
    comp = _build_composite(Nx)
    sob = SobolevGradientAdaptor(
        comp, axes=(0, 1, 2, 3), value_weight=1.0, grad_weight=0.7,
        grad_scales=[1.0, 1.1, 0.9, 1.2],
        hess_pairs=[(1, 1), (2, 2), (3, 3)], hess_weight=0.5, hess_scales=[1.3, 0.8, 1.0],
    )
    assert sob._hess_all_diag and sob._hess_diag_dims == [1, 2, 3]
    x = torch.randn(B, Nx, dtype=DT)
    cache = sob.build_cache((x, torch.randn(B, sob.n_outputs, dtype=DT)))

    J = sob.jacobian(cache)                       # (B, O, P) via the fast diagonal path
    assert tuple(J.shape) == (B, sob.n_outputs, sob.num_parameters())
    # confirm we exercised the fast path, not the generic jvp-loop fallback
    assert sob._jacobian_fast(cache) is not None
    v = torch.randn(sob.num_parameters(), dtype=DT)
    Jv = torch.einsum("bop,p->bo", J, v)
    jvp = sob.jvp(cache, v)
    # Dense Jacobian and jvp are the SAME operator on EVERY block (value, grad,
    # and the curvature rows): both are residual-signed.  The hess block is the
    # cross-check -- jvp's hess path uses the full Hessian (grad_grad_jvp) while
    # the dense Jacobian uses the selected-diagonal route, two independent codes.
    assert float((Jv - jvp).abs().max()) < 1e-9


def test_sobolev_h2_trace_channel():
    """Laplacian-trace mode: one channel = Σ_a ∂²f/∂x_a².  Dense Jacobian must equal
    the jvp (cross-checked via the full-Hessian path), and the trace value/Jacobian
    must equal the SUM of the individual diagonal channels."""
    torch.manual_seed(6)
    Nx, B = 4, 6
    comp = _build_composite(Nx)
    sob = SobolevGradientAdaptor(
        comp, axes=(0, 1, 2, 3), value_weight=1.0, grad_weight=0.7,
        grad_scales=[1.0, 1.1, 0.9, 1.2],
        hess_trace_dims=(1, 2, 3), hess_weight=0.5, hess_scales=[1.0],
    )
    assert sob.n_outputs == 1 + 4 + 1 and sob._hess_trace and sob._n_hess == 1
    x = torch.randn(B, Nx, dtype=DT)
    cache = sob.build_cache((x, torch.randn(B, sob.n_outputs, dtype=DT)))
    assert tuple(sob.residuals(cache).shape) == (B, 6)
    # dense Jacobian (diagonal-sum route) == jvp (full-Hessian-sum route) on every block
    assert sob._jacobian_fast(cache) is not None
    v = torch.randn(sob.num_parameters(), dtype=DT)
    Jv = torch.einsum("bop,p->bo", sob.jacobian(cache), v)
    assert float((Jv - sob.jvp(cache, v)).abs().max()) < 1e-9
    # adjoint
    w = torch.randn(B, sob.n_outputs, dtype=DT)
    JTw = sob.vjp(cache, w).sum(dim=0)
    assert float(((w * Jv).sum() - (JTw * v).sum()).abs() / ((JTw * v).sum().abs() + 1e-30)) < 1e-9
    # trace value/jvp == sum of the three diagonal-pair channels (unit scales)
    pp = SobolevGradientAdaptor(
        comp, axes=(0, 1, 2, 3), value_weight=1.0, grad_weight=0.7, grad_scales=[1.0, 1.1, 0.9, 1.2],
        hess_pairs=[(1, 1), (2, 2), (3, 3)], hess_weight=0.5, hess_scales=[1.0, 1.0, 1.0],
    )
    cpp = pp.build_cache((x, torch.randn(B, pp.n_outputs, dtype=DT)))
    assert float((sob._hess_pred(cache["base"])[:, 0] - pp._hess_pred(cpp["base"]).sum(1)).abs().max()) < 1e-12
    tr = sob.jvp(cache, v, out_dim=5)
    dg = sum(pp.jvp(cpp, v, out_dim=5 + k) for k in range(3))
    assert float((tr - dg).abs().max()) < 1e-9


def test_sobolev_h2_pairs_trace_mutually_exclusive():
    import pytest
    comp = _build_composite(4)
    with pytest.raises(ValueError):
        SobolevGradientAdaptor(comp, axes=(0, 1, 2, 3), hess_pairs=[(1, 1)],
                               hess_trace_dims=(1, 2, 3), hess_weight=1.0)


def test_sobolev_h2_disabled_is_unchanged():
    """With H^2 off (default), outputs/residuals match a value+gradient adaptor."""
    torch.manual_seed(3)
    Nx, B = 4, 5
    comp = _build_composite(Nx)
    sob = SobolevGradientAdaptor(comp, axes=(0, 1, 2, 3), value_weight=1.0, grad_weight=1.0)
    assert sob.n_outputs == 1 + 4
    assert len(sob.hess_pairs) == 0
    x = torch.randn(B, Nx, dtype=DT)
    y_aug = torch.randn(B, sob.n_outputs, dtype=DT)
    cache = sob.build_cache((x, y_aug))
    assert tuple(sob.residuals(cache).shape) == (B, 1 + 4)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
