# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

#
# Adjoint audit for function-level JVP/VJP of ASTCompositeAdaptor:
#   <A, J v>  ==  <J^T A, v>
#
# Notes:
#   * Scalar-output composite (O=1).
#   * Uses SegmentedAdaptor leaves (analytic primitives).
#   * Uses a domain-safe AST so log/exp never see invalid inputs:
#       F = sin(a0(x0)*a1(x1)) + log( exp(a2(x0,x2)^2) + a3(x1)^2 ) + cos(a4(x2))
#     log arg >= 1 always.

import torch, pytest
torch.set_default_dtype(torch.float64)

try:
    import nestynet
    from nestynet.adaptors.adaptors import SegmentedAdaptor
except Exception:
    pytest.skip("nestynet not importable", allow_module_level=True)

try:
    from symbolic_regression_DE.adaptors.ast_composite import ASTCompositeAdaptor
    from symbolic_regression_DE.sr_core.bridges import (
        AtomNode, AddNode, MulNode, PowNode, SinNode, CosNode, ExpNode, LogNode
    )
except Exception:
    from nestynet_sr.adaptors.ast_composite import ASTCompositeAdaptor
    from nestynet_sr.sr_core.bridges import (
        AtomNode, AddNode, MulNode, PowNode, SinNode, CosNode, ExpNode, LogNode
    )

def _orthonormal_columns(n, m, dtype, device):
    X = torch.randn(n, m, dtype=dtype, device=device)
    Q, _ = torch.linalg.qr(X, mode="reduced")
    return Q[:, :m]

def _mk_leaf(Nx_leaf, S, *, seg_width=2, device="cpu"):
    kw = dict(model_base_name="G_Model", model_scale=0.1,
              dtype=torch.float64, device=device, num_segments=S, seg_width=seg_width)
    net = nestynet.nets.NestyNet_Model(Nout_size=1, Nx_size=Nx_leaf, **kw)
    seg = torch.arange(net.base_model.num_segments, device=device)
    return SegmentedAdaptor(net, segments=seg)

def _mk_composite(S, device="cpu"):
    # Global input space uses indices (0,1,2); Nx_global can be >=3.
    a0 = AtomNode("nn", (0,))
    a1 = AtomNode("nn", (1,))
    a2 = AtomNode("nn", (0, 2))
    a3 = AtomNode("nn", (1,))
    a4 = AtomNode("nn", (2,))

    term1 = SinNode(MulNode(a0, a1))
    term2 = LogNode(AddNode(ExpNode(PowNode(a2, 2.0)), PowNode(a3, 2.0)))
    term3 = CosNode(a4)
    ast = AddNode(AddNode(term1, term2), term3)

    leaves = [
        _mk_leaf(1, S, seg_width=2, device=device),  # x0
        _mk_leaf(1, S, seg_width=2, device=device),  # x1
        _mk_leaf(2, S, seg_width=2, device=device),  # (x0,x2)
        _mk_leaf(1, S, seg_width=2, device=device),  # x1
        _mk_leaf(1, S, seg_width=2, device=device),  # x2
    ]
    return ASTCompositeAdaptor(ast, leaves)

def _canon_Y(Y, B):
    if Y.ndim == 2 and Y.size(1) == 1: return Y.reshape(B)
    if Y.ndim == 1: return Y
    raise ValueError(f"Unexpected Y shape {tuple(Y.shape)}")

def _canon_A(A, B):
    if A.ndim == 2 and A.size(1) == 1: return A.reshape(B)
    if A.ndim == 1: return A
    raise ValueError(f"Unexpected A shape {tuple(A.shape)}")

def _canon_g(g, P):
    if g.ndim == 2 and g.size(0) == 1 and g.size(1) == P: return g.reshape(P)
    if g.ndim == 1 and g.numel() == P: return g
    return g.reshape(-1)

@pytest.mark.parametrize("S", [1, 2])
@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("Nx_global", [3, 4])
def test_ast_composite_jvp_vjp_bilinear_adjoint(S, B, Nx_global):
    torch.manual_seed(0)
    model = _mk_composite(S)

    x = 0.3 * torch.randn(B, Nx_global)
    with torch.no_grad():
        y = model.forward(x).detach()
    cache = model.build_cache((x, y))

    P = model.num_parameters()
    assert P > 0

    mA = min(B, 8)
    mV = min(P, 8)

    # Orthonormal A_i in output space (B,)
    QA = _orthonormal_columns(B, mA, x.dtype, x.device)  # (B, mA)
    A_list = [QA[:, i].clone().reshape(B) for i in range(mA)]

    # Orthonormal v_j in parameter space (P,)
    QP = _orthonormal_columns(P, mV, x.dtype, x.device)  # (P, mV)
    v_list = [QP[:, j].clone().reshape(P) for j in range(mV)]

    # Precompute J v_j
    Y_list = [_canon_Y(model.jvp(cache, vj, out_dim=None), B) for vj in v_list]

    L = torch.zeros(mA, mV, dtype=x.dtype)
    R = torch.zeros_like(L)

    for i, A in enumerate(A_list):
        A = _canon_A(A, B)
        g = _canon_g(model.vjp(cache, A, out_dim=None), P)  # (P,)
        for j, (vj, Yj) in enumerate(zip(v_list, Y_list)):
            L[i, j] = (A * Yj).sum()
            R[i, j] = (g * vj).sum()

    diff = L - R
    tol = 1e-12 * (1.0 + float(L.norm()))
    assert diff.norm() <= tol, f"Adjoint mismatch: ||L-R||={float(diff.norm())}, ||L||={float(L.norm())}"

@pytest.mark.parametrize("S", [1, 2])
@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("Nx_global", [3, 4])
def test_ast_composite_jvp_vjp_bilinear_adjoint_out_dim0(S, B, Nx_global):
    torch.manual_seed(1)
    model = _mk_composite(S)

    x = 0.3 * torch.randn(B, Nx_global)
    with torch.no_grad():
        y = model.forward(x).detach()
    cache = model.build_cache((x, y))

    P = model.num_parameters()
    mA = min(B, 8)
    mV = min(P, 8)

    QA = _orthonormal_columns(B, mA, x.dtype, x.device)
    A_list = [QA[:, i].clone().reshape(B) for i in range(mA)]

    QP = _orthonormal_columns(P, mV, x.dtype, x.device)
    v_list = [QP[:, j].clone().reshape(P) for j in range(mV)]

    Y_list = [_canon_Y(model.jvp(cache, vj, out_dim=0), B) for vj in v_list]

    L = torch.zeros(mA, mV, dtype=x.dtype)
    R = torch.zeros_like(L)

    for i, A in enumerate(A_list):
        g = _canon_g(model.vjp(cache, A, out_dim=0), P)
        for j, (vj, Yj) in enumerate(zip(v_list, Y_list)):
            L[i, j] = (A * Yj).sum()
            R[i, j] = (g * vj).sum()

    diff = L - R
    tol = 1e-12 * (1.0 + float(L.norm()))
    assert diff.norm() <= tol, f"Adjoint(out_dim) mismatch: ||L-R||={float(diff.norm())}, ||L||={float(L.norm())}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
