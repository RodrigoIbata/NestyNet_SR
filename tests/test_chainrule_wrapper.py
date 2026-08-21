# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Test that ChainRuleYModel produces correct derivatives via chain rule."""

import sys
import torch

# ── tiny helpers ──────────────────────────────────────────────────────
def _rel_err(a, b, eps=1e-12):
    """Element-wise relative error."""
    return (a - b).abs() / (b.abs().clamp(min=eps))


def _check(tag, rel, tol=1e-4):
    mx = rel.max().item()
    ok = mx < tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {tag}: max rel err = {mx:.2e}  (tol {tol:.0e})")
    return ok


# ── main test ─────────────────────────────────────────────────────────
def main():
    from nestynet_sr.sr_search.chainrule_wrapper import ChainRuleYModel
    from nestynet_sr.sr_search.y_transforms import get_y_transform_registry

    torch.set_default_dtype(torch.float64)
    B, Nx = 200, 3

    # ---- Build a simple model with known analytic derivatives --------
    # Use a small polynomial: f(x) = 2*x0^2 + 3*x1 - x2 + 1
    class ToyModel(torch.nn.Module):
        """Toy scalar model with exact forward/grad/grad_grad."""

        def forward(self, x):
            # (B, 1)
            return (2.0 * x[:, 0] ** 2 + 3.0 * x[:, 1] - x[:, 2] + 1.0).unsqueeze(-1)

        def grad(self, x, out_dim=None):
            B = x.size(0)
            g = torch.stack(
                [4.0 * x[:, 0], 3.0 * torch.ones(B, dtype=x.dtype), -torch.ones(B, dtype=x.dtype)],
                dim=-1,
            )  # (B, 3)
            # Match ASTComposite convention: (B, O, Nx)
            g = g.unsqueeze(1)  # (B, 1, Nx)
            if out_dim is not None:
                return g[:, out_dim]  # (B, Nx)
            return g

        def grad_grad(self, x, out_dim=None):
            B = x.size(0)
            h = torch.zeros(B, Nx, Nx, dtype=x.dtype)
            h[:, 0, 0] = 4.0  # d²f/dx0² = 4
            # All other second derivatives are 0
            # Match ASTComposite convention: (B, O, Nx, Nx)
            h = h.unsqueeze(1)  # (B, 1, Nx, Nx)
            if out_dim is not None:
                return h[:, out_dim]  # (B, Nx, Nx)
            return h

        def parameters(self, recurse=True):
            return iter([torch.nn.Parameter(torch.zeros(1))])

    toy = ToyModel()
    x = torch.randn(B, Nx)

    # ---- Reference: compute φ(f(x)) derivatives with autograd -------
    registry = get_y_transform_registry()
    transforms_to_test = ["square", "log", "reciprocal", "exp", "sin"]
    name_to_tf = {t.name: t for t in registry}

    all_pass = True
    for tname in transforms_to_test:
        yt = name_to_tf[tname]
        print(f"\nTesting transform: {tname}")

        # Filter x to valid domain
        f_vals = toy(x)[:, 0]  # (B,)
        if tname in ("log", "sqrt"):
            mask = f_vals > 1e-6
        elif tname == "reciprocal":
            mask = f_vals.abs() > 1e-6
        else:
            mask = torch.ones_like(f_vals, dtype=torch.bool)

        if mask.sum() < 10:
            print(f"  [SKIP] Not enough valid samples ({mask.sum()} < 10)")
            continue

        x_valid = x[mask]

        # -- Autograd reference --
        x_ag = x_valid.clone().requires_grad_(True)
        f_ag = toy(x_ag)[:, 0]  # (B_valid,)
        phi_f = yt.torch_op(f_ag)  # (B_valid,)
        # First derivatives via autograd
        grad_phi_f = torch.autograd.grad(
            phi_f.sum(), x_ag, create_graph=True
        )[0]  # (B_valid, Nx)
        # Second derivatives via autograd (Hessian diagonal + off-diag)
        hess_rows = []
        for i in range(Nx):
            g_i = torch.autograd.grad(
                grad_phi_f[:, i].sum(), x_ag, retain_graph=True
            )[0]
            hess_rows.append(g_i)
        hess_ag = torch.stack(hess_rows, dim=1)  # (B_valid, Nx, Nx)

        # -- Chain-rule wrapper --
        wrapper = ChainRuleYModel(toy, yt)
        with torch.no_grad():
            cr_f = wrapper(x_valid)  # (B_valid, 1)
            cr_g = wrapper.grad(x_valid)  # (B_valid, 1, Nx)
            cr_h = wrapper.grad_grad(x_valid)  # (B_valid, 1, Nx, Nx)

        # -- Compare --
        ok_f = _check(
            "forward φ(f)",
            _rel_err(cr_f[:, 0], phi_f.detach()),
        )
        ok_g = _check(
            "grad d/dx[φ(f)]",
            _rel_err(cr_g[:, 0, :], grad_phi_f.detach()),
        )
        ok_h = _check(
            "grad_grad d²/dx²[φ(f)]",
            _rel_err(cr_h[:, 0, :, :], hess_ag.detach()),
            tol=1e-3,
        )

        # Also test out_dim variant
        with torch.no_grad():
            cr_g_0 = wrapper.grad(x_valid, out_dim=0)  # (B_valid, Nx)
            cr_h_0 = wrapper.grad_grad(x_valid, out_dim=0)  # (B_valid, Nx, Nx)
        ok_g0 = _check(
            "grad(out_dim=0)",
            _rel_err(cr_g_0, grad_phi_f.detach()),
        )
        ok_h0 = _check(
            "grad_grad(out_dim=0)",
            _rel_err(cr_h_0, hess_ag.detach()),
            tol=1e-3,
        )

        all_pass = all_pass and ok_f and ok_g and ok_h and ok_g0 and ok_h0

    print(f"\n{'='*50}")
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
