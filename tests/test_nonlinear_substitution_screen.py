# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Unit test for _nonlinear_substitution_screen.

Creates synthetic data for f(z, x1) = (1 - x1**2) / (x1*cos(z) + 1)
and verifies that the screen detects cos on column 0 with low error.
"""

import math
import torch

from nestynet_sr.sr_search.fitting_utils import _nonlinear_substitution_screen


def test_pb102_leaf():
    """The 2D leaf from pb102: f(z, x1) = (1 - x1^2) / (x1*cos(z) + 1).

    After substituting w = cos(z), this becomes (1 - x1^2) / (x1*w + 1),
    a rational function of degree (2,1) in (w, x1).
    """
    torch.manual_seed(42)
    N = 2000

    z = torch.rand(N, dtype=torch.float64) * 2 * math.pi - math.pi
    x1 = torch.rand(N, dtype=torch.float64) * 1.6 - 0.8  # keep |x1| < 1 for safe denom

    denom = x1 * torch.cos(z) + 1.0
    # Ensure denominator is positive (should be for |x1| < 1)
    mask = denom.abs() > 0.1
    z, x1, denom = z[mask], x1[mask], denom[mask]

    F = (1.0 - x1 ** 2) / denom
    X = torch.stack([z, x1], dim=1)

    hits = _nonlinear_substitution_screen(X, F, teacher=None, threshold=0.05)

    assert len(hits) > 0, "Expected at least one hit from the screen"

    best = hits[0]
    print(f"Best hit: transform={best['transform']}, col={best['col_idx']}, "
          f"deg={best['deg_num']}/{best['deg_den']}, err={best['error']:.6f}")

    assert best["transform"] == "cos", f"Expected cos, got {best['transform']}"
    assert best["col_idx"] == 0, f"Expected col 0, got {best['col_idx']}"
    assert best["error"] < 0.01, f"Error {best['error']:.4f} too high (expected < 0.01)"

    print("PASSED: cos substitution on column 0 detected correctly")


def test_parity_prescreen():
    """Verify parity pre-screen narrows candidates when a teacher is available.

    Uses symmetric z-samples so the parity check sees even symmetry clearly.
    """
    torch.manual_seed(123)
    N = 1500

    # Symmetric z around 0 so parity check works reliably
    z = torch.linspace(-math.pi, math.pi, N, dtype=torch.float64)
    x1 = torch.rand(N, dtype=torch.float64) * 1.6 - 0.8

    denom = x1 * torch.cos(z) + 1.0
    mask = denom.abs() > 0.1
    z, x1, denom = z[mask], x1[mask], denom[mask]
    F = (1.0 - x1 ** 2) / denom
    X = torch.stack([z, x1], dim=1)

    # Build a simple "teacher" that evaluates the same function
    class MockTeacher(torch.nn.Module):
        def forward(self, x):
            z_in = x[:, 0]
            x1_in = x[:, 1]
            d = x1_in * torch.cos(z_in) + 1.0
            return ((1.0 - x1_in ** 2) / d).unsqueeze(1)

    teacher = MockTeacher().double()

    hits = _nonlinear_substitution_screen(X, F, teacher=teacher, threshold=0.05)

    assert len(hits) > 0, "Expected at least one hit"
    best = hits[0]

    # With symmetric data, the screen should detect even symmetry in z
    assert best["parity"] == "even", f"Expected even parity, got {best['parity']}"
    assert best["transform"] == "cos", f"Expected cos, got {best['transform']}"
    assert best["error"] < 0.01, f"Error too high: {best['error']:.4f}"

    print(f"PASSED: parity={best['parity']}, transform={best['transform']}, err={best['error']:.6f}")


def test_trig_hints():
    """Verify trig_hints overrides parity without needing a teacher.

    Passes trig_hints={0: "cos"} with no teacher and verifies that
    cos is detected on column 0 with parity="even".
    """
    torch.manual_seed(99)
    N = 2000

    z = torch.rand(N, dtype=torch.float64) * 2 * math.pi - math.pi
    x1 = torch.rand(N, dtype=torch.float64) * 1.6 - 0.8

    denom = x1 * torch.cos(z) + 1.0
    mask = denom.abs() > 0.1
    z, x1, denom = z[mask], x1[mask], denom[mask]
    F = (1.0 - x1 ** 2) / denom
    X = torch.stack([z, x1], dim=1)

    hits = _nonlinear_substitution_screen(
        X, F, teacher=None, threshold=0.05, trig_hints={0: "cos"},
    )

    assert len(hits) > 0, "Expected at least one hit"
    best = hits[0]

    print(f"Best hit: transform={best['transform']}, col={best['col_idx']}, "
          f"parity={best['parity']}, err={best['error']:.6f}")

    assert best["transform"] == "cos", f"Expected cos, got {best['transform']}"
    assert best["col_idx"] == 0, f"Expected col 0, got {best['col_idx']}"
    assert best["parity"] == "even", f"Expected even parity, got {best['parity']}"
    assert best["error"] < 0.01, f"Error too high: {best['error']:.4f}"

    print("PASSED: trig_hints correctly overrides parity without teacher")


def test_pb109_multivariate_substitution():
    """The pb109 inner after outer peel:
    t = (x0*cos(x2) - x1) / (x0 - x1*cos(x2)).

    This is rational in (x0, x1, w) with w = cos(x2), so the screen should
    recover a cos substitution on column 2 even in multivariate mode.
    """
    torch.manual_seed(7)
    N = 3000

    x0 = torch.rand(N, dtype=torch.float64) * 1.5 + 0.5  # keep away from zero
    x1 = torch.rand(N, dtype=torch.float64) * 1.6 - 0.8
    x2 = torch.rand(N, dtype=torch.float64) * 2 * math.pi - math.pi

    w = torch.cos(x2)
    denom = x0 - x1 * w
    mask = denom.abs() > 0.1
    x0, x1, x2, w, denom = x0[mask], x1[mask], x2[mask], w[mask], denom[mask]

    F = (x0 * w - x1) / denom
    X = torch.stack([x0, x1, x2], dim=1)

    hits = _nonlinear_substitution_screen(X, F, teacher=None, threshold=0.05)

    assert len(hits) > 0, "Expected at least one multivariate substitution hit"
    best = hits[0]
    print(f"Best hit: transform={best['transform']}, col={best['col_idx']}, "
          f"deg={best['deg_num']}/{best['deg_den']}, err={best['error']:.6f}")

    assert best["transform"] == "cos", f"Expected cos, got {best['transform']}"
    assert best["col_idx"] == 2, f"Expected col 2, got {best['col_idx']}"
    assert best["error"] < 0.01, f"Error too high: {best['error']:.4f}"

    print("PASSED: multivariate cos substitution detected for pb109-style leaf")


def test_pb109_candidate_builder_cos_wrap():
    """Verify _build_nonlinear_sub_candidate wraps the extra var with CosNode.

    Simulates the pb109 pattern: compound atom NN(x0/x1, x2) where the screen
    detects cos on column 1 (the extra variable x2).  After building, the
    resulting ratpoly atom's inputs tuple must contain CosNode(Var(2)).
    """
    from torch.utils.data import DataLoader, TensorDataset

    from nestynet_sr.sr_core.bridges import (
        AtomNode, CosNode, MulNode, PowNode, Var,
    )
    from nestynet_sr.sr_search.candidate_builders import (
        _build_nonlinear_sub_candidate,
    )

    torch.manual_seed(42)
    N = 2000

    x0 = torch.rand(N, dtype=torch.float64) * 1.5 + 0.5
    x1 = torch.rand(N, dtype=torch.float64) * 1.5 + 0.5
    x2 = torch.rand(N, dtype=torch.float64) * 2 * math.pi - math.pi

    z = x0 / x1
    w = torch.cos(x2)
    F = (z * w - 1.0) / (z - w)
    X = torch.stack([x0, x1, x2], dim=1)

    # Build a compound atom: NN(x0/x1, x2) with var_idxs=(0,1,2)
    # inputs = (MulNode(Var(0), PowNode(Var(1), -1)), Var(2))
    compound_expr = MulNode(left=Var(0), right=PowNode(base=Var(1), exponent=-1))
    target = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2),
        kwargs={},
        tag="leaf0",
        inputs=(compound_expr, Var(2)),
    )

    # Build a simple teacher that maps (z, x2) -> F
    class MockTeacher(torch.nn.Module):
        def forward(self, x_in):
            z_val = x_in[:, 0]
            x2_val = x_in[:, 1]
            w_val = torch.cos(x2_val)
            return ((z_val * w_val - 1.0) / (z_val - w_val)).unsqueeze(1)

    teacher = MockTeacher().double()
    reuse = {"leaf0": teacher}

    dataset = TensorDataset(X, F.unsqueeze(1))
    train_loader = DataLoader(dataset, batch_size=N, shuffle=False)

    # The hit says: cos substitution on col 1 (the extra var x2)
    hit = {
        "col_idx": 1,
        "transform": "cos",
        "deg_num": 2,
        "deg_den": 1,
        "error": 1e-5,
        "parity": "even",
    }

    root = target  # Single-atom AST

    result = _build_nonlinear_sub_candidate(
        root=root,
        target=target,
        reuse=reuse,
        train_loader=train_loader,
        device=torch.device("cpu"),
        dtype=torch.float64,
        hit=hit,
    )

    assert result is not None, "Builder returned None — candidate build failed"
    cand_root, init_fn, meta = result

    # The analytic leaf may be either a plain ratpoly or scale()*rratpoly.
    rat_atom = cand_root
    if isinstance(cand_root, MulNode):
        assert isinstance(cand_root.right, AtomNode)
        assert cand_root.right.kind in ("ratpoly", "rratpoly")
        rat_atom = cand_root.right
    else:
        assert isinstance(cand_root, AtomNode), (
            f"Expected AtomNode or MulNode, got {type(cand_root).__name__}"
        )
        assert cand_root.kind in ("ratpoly", "rratpoly")

    # Check inputs tuple: should be (compound_expr, CosNode(Var(2)))
    inputs = rat_atom.inputs
    assert inputs is not None and len(inputs) == 2, (
        f"Expected 2 inputs, got {len(inputs) if inputs else 'None'}"
    )

    # First input: the compound expression (x0/x1) — should be MulNode
    assert isinstance(inputs[0], MulNode), (
        f"Expected MulNode for input[0], got {type(inputs[0]).__name__}"
    )

    # Second input: must be CosNode(Var(2)), NOT plain Var(2)
    assert isinstance(inputs[1], CosNode), (
        f"Expected CosNode for input[1], got {type(inputs[1]).__name__}. "
        f"This is the bug: the extra var x2 was not wrapped with cos()."
    )
    inner = inputs[1].arg
    assert isinstance(inner, AtomNode) and inner.kind == "var", (
        f"Expected Var inside CosNode, got {type(inner).__name__}"
    )
    assert inner.var_idxs[0] == 2, (
        f"Expected Var(2) inside CosNode, got Var({inner.var_idxs[0]})"
    )

    print("PASSED: candidate builder correctly wraps extra var x2 with CosNode")


if __name__ == "__main__":
    test_pb102_leaf()
    print()
    test_parity_prescreen()
    print()
    test_trig_hints()
    print()
    test_pb109_multivariate_substitution()
    print()
    test_pb109_candidate_builder_cos_wrap()
    print("\nAll tests passed!")
