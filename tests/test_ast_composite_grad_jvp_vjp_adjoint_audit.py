# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

#
# Adjoint audit for gradient-level JVP/VJP of ASTCompositeAdaptor:
#   <V, (∂_θ ∇_x r)·v>  ==  <(∂_θ ∇_x r)ᵀ·V, v>
#
# Notes:
#   * Assumes scalar-output composite (O=1).
#   * Uses SegmentedAdaptor leaves (analytic primitives).
#   * Uses a domain-safe expression so log/exp never see invalid inputs:
#       F = sin(a0(x0)*a1(x1)) + log( exp(a2(x0,x2)^2) + a3(x1)^2 ) + cos(a4(x2))
#     Here exp(·^2) >= 1 and (·^2) >= 0, so log argument >= 1.

import torch, pytest
torch.set_default_dtype(torch.float64)

# --- imports (tolerate either package layout) ---

try:
    import nestynet
    from nestynet.adaptors.adaptors import SegmentedAdaptor
except Exception:
    pytest.skip("nestynet not importable in this environment", allow_module_level=True)

try:
    from symbolic_regression_DE.adaptors.ast_composite import ASTCompositeAdaptor
    from symbolic_regression_DE.sr_core.bridges import (
        AtomNode, AddNode, MulNode, PowNode, SinNode, CosNode, ExpNode, LogNode
    )
except Exception:
    # alternate package name used in some installs
    from nestynet_sr.adaptors.ast_composite import ASTCompositeAdaptor
    from nestynet_sr.sr_core.bridges import (
        AtomNode, AddNode, MulNode, PowNode, SinNode, CosNode, ExpNode, LogNode
    )

# ---------- helpers ----------

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
    # Global input space: (x0, x1, x2)
    # Expression (domain-safe):
    #   F = sin(a0(x0) * a1(x1)) + log( exp( a2(x0,x2)^2 ) + a3(x1)^2 ) + cos(a4(x2))
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

# ---------- adjoint audit: out_dim=None (canonical O=1 path) ----------

@pytest.mark.parametrize("S", [1, 2])
@pytest.mark.parametrize("B", [1, 3])
def test_ast_composite_grad_jvp_vjp_bilinear_adjoint(S, B):
    torch.manual_seed(0)
    model = _mk_composite(S)
    Nx = 3

    # small inputs to keep exp benign
    x = 0.3 * torch.randn(B, Nx)
    with torch.no_grad():
        y = model.forward(x).detach()
    cache = model.build_cache((x, y))

    Ptot = model.num_parameters()
    assert Ptot > 0

    mV = min(Ptot, 8)
    mG = min(B * Nx, 8)

    # Orthonormal V_i in gradient-output space (B,1,Nx)
    QG = _orthonormal_columns(B * Nx, mG, x.dtype, x.device)
    V_list = [QG[:, i].clone().reshape(B, 1, Nx) for i in range(mG)]

    # Orthonormal v_j in parameter space (P,)
    QP = _orthonormal_columns(Ptot, mV, x.dtype, x.device)
    v_list = [QP[:, j].clone().reshape(Ptot) for j in range(mV)]

    # Precompute (∂_θ ∇_x r)·v_j
    Jv_list = [model.grad_jvp(cache, vj, out_dim=None) for vj in v_list]

    L = torch.zeros(mG, mV, dtype=x.dtype)
    R = torch.zeros_like(L)

    for i, V in enumerate(V_list):
        g = model.grad_vjp(cache, V, out_dim=None)  # expected (1,Ptot)
        for j, (vj, Jv) in enumerate(zip(v_list, Jv_list)):
            L[i, j] = (V * Jv).sum()
            R[i, j] = (g * vj).sum()

    diff = L - R
    tol = 1e-12 * (1.0 + float(L.norm()))
    assert diff.norm() <= tol, f"Grad-adjoint mismatch: ||L-R||={float(diff.norm())}, ||L||={float(L.norm())}"

# ---------- adjoint audit: out_dim=0 (exercise “single output” plumbing) ----------

@pytest.mark.parametrize("S", [1, 2])
@pytest.mark.parametrize("B", [1, 3])
def test_ast_composite_grad_jvp_vjp_bilinear_adjoint_out_dim0(S, B):
    torch.manual_seed(1)
    model = _mk_composite(S)
    Nx = 3

    x = 0.3 * torch.randn(B, Nx)
    with torch.no_grad():
        y = model.forward(x).detach()
    cache = model.build_cache((x, y))

    Ptot = model.num_parameters()
    mV = min(Ptot, 8)
    mG = min(B * Nx, 8)

    # out_dim=0: treat V in (B,Nx)
    QG = _orthonormal_columns(B * Nx, mG, x.dtype, x.device)
    V_list = [QG[:, i].clone().reshape(B, Nx) for i in range(mG)]

    QP = _orthonormal_columns(Ptot, mV, x.dtype, x.device)
    v_list = [QP[:, j].clone().reshape(Ptot) for j in range(mV)]

    Jv_list = [model.grad_jvp(cache, vj, out_dim=0) for vj in v_list]  # (B,Nx)

    L = torch.zeros(mG, mV, dtype=x.dtype)
    R = torch.zeros_like(L)

    for i, V in enumerate(V_list):
        g = model.grad_vjp(cache, V, out_dim=0)  # (1,Ptot)
        for j, (vj, Jv) in enumerate(zip(v_list, Jv_list)):
            L[i, j] = (V * Jv).sum()
            R[i, j] = (g * vj).sum()

    diff = L - R
    tol = 1e-12 * (1.0 + float(L.norm()))
    assert diff.norm() <= tol, f"Grad-adjoint(out_dim) mismatch: ||L-R||={float(diff.norm())}, ||L||={float(L.norm())}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
