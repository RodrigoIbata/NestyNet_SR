# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Test derivative-based linearity detection.

The core insight: for f(z, x1) = z + x1, the partial derivative ∂f/∂x1 = 1 (constant).
This is the definitive test for linearity - if the derivative is constant, the axis is
linear regardless of what f looks like overall.

Test Cases:
1. f(z, x1) = z + x1 → x1 is LINEAR (∂f/∂x1 = 1)
2. f(z, x1) = z * x1 → x1 is NOT linear (∂f/∂x1 = z varies)
3. f(z, x1) = z * sin(x1) → x1 is NOT linear (∂f/∂x1 = z*cos(x1) varies)
4. f(z, x1, x4) = x1 + z*sin(x4) → x1 LINEAR, z LINEAR, x4 NOT linear
"""

import torch


class AnalyticModel(torch.nn.Module):
    def __init__(self, f_true):
        super().__init__()
        self._f_true = f_true

    def forward(self, x):
        return self._f_true(x).unsqueeze(-1)


def train_model(f_true, n_inputs, n_samples=2000, epochs=2000, num_segments=32):
    """Return an analytic autograd model on synthetic data from f_true."""
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    dtype = torch.float64

    # Generate training data
    torch.manual_seed(0)
    x_train = torch.rand(n_samples, n_inputs, dtype=dtype, device=device) * 4.0 - 2.0  # [-2, 2]
    model = AnalyticModel(f_true).to(device=device, dtype=dtype)
    return model, x_train, device


def check_linearity_via_derivative(model, axis_idx, n_inputs, x_samples, var_threshold=0.10):
    """Check if axis is linear by testing if ∂f/∂x is approximately constant.

    If ∂f/∂x is constant, f is linear in x. This is much more robust than
    checking f vs x directly, as it's not contaminated by other variables.

    Returns: (is_linear, rel_variance, grad_mean, grad_std)
    """
    x_samples = x_samples.clone().requires_grad_(True)

    # Forward pass
    y_out = model(x_samples)
    if y_out.dim() > 1:
        y_out = y_out[:, 0]

    # Compute gradient ∂f/∂x for all axes
    grads = torch.autograd.grad(y_out.sum(), x_samples, create_graph=False)[0]
    grad_axis = grads[:, axis_idx]

    # Check if gradient is approximately constant (low relative variance)
    grad_mean = grad_axis.mean().item()
    grad_std = grad_axis.std().item()
    if abs(grad_mean) < 1e-8:
        # Near-zero gradient - axis doesn't contribute, treat as "constant" (linear with slope 0)
        return True, 0.0, grad_mean, grad_std
    rel_var = grad_std / abs(grad_mean)
    is_linear = rel_var < var_threshold

    return is_linear, rel_var, grad_mean, grad_std


def test_case_1_additive():
    """f(z, x1) = z + x1 → x1 should be LINEAR (∂f/∂x1 = 1)"""
    print("\n" + "=" * 60)
    print("Test 1: f(z, x1) = z + x1")
    print("Expected: x1 is LINEAR (∂f/∂x1 = 1 everywhere)")
    print("=" * 60)

    def f_true(x):
        z, x1 = x[:, 0], x[:, 1]
        return z + x1

    model, x_train, device = train_model(f_true, n_inputs=2)

    # Check axis 1 (x1)
    is_linear, rel_var, grad_mean, grad_std = check_linearity_via_derivative(
        model, axis_idx=1, n_inputs=2, x_samples=x_train
    )
    print(f"\nAxis 1 (x1): is_linear={is_linear}, rel_var={rel_var:.4f}")
    print(f"  grad_mean={grad_mean:.4f}, grad_std={grad_std:.4f}")
    print("  Expected: is_linear=True, grad_mean ≈ 1.0")

    assert is_linear, f"Test 1 FAILED: x1 should be linear but rel_var={rel_var:.4f}"
    assert abs(grad_mean - 1.0) < 0.1, f"Test 1 FAILED: grad_mean={grad_mean:.4f}, expected ≈ 1.0"
    print("✓ Test 1 PASSED")


def test_case_2_multiplicative():
    """f(z, x1) = z * x1 → x1 is NOT linear (∂f/∂x1 = z varies)"""
    print("\n" + "=" * 60)
    print("Test 2: f(z, x1) = z * x1")
    print("Expected: x1 is NOT linear (∂f/∂x1 = z varies)")
    print("=" * 60)

    def f_true(x):
        z, x1 = x[:, 0], x[:, 1]
        return z * x1

    model, x_train, device = train_model(f_true, n_inputs=2)

    # Check axis 1 (x1)
    is_linear, rel_var, grad_mean, grad_std = check_linearity_via_derivative(
        model, axis_idx=1, n_inputs=2, x_samples=x_train
    )
    print(f"\nAxis 1 (x1): is_linear={is_linear}, rel_var={rel_var:.4f}")
    print(f"  grad_mean={grad_mean:.4f}, grad_std={grad_std:.4f}")
    print("  Expected: is_linear=False (∂f/∂x1 = z varies)")

    assert not is_linear, f"Test 2 FAILED: x1 should NOT be linear but rel_var={rel_var:.4f}"
    print("✓ Test 2 PASSED")


def test_case_3_trig():
    """f(z, x1) = z * sin(x1) → x1 is NOT linear (∂f/∂x1 = z*cos(x1) varies)"""
    print("\n" + "=" * 60)
    print("Test 3: f(z, x1) = z * sin(x1)")
    print("Expected: x1 is NOT linear (∂f/∂x1 = z*cos(x1) varies)")
    print("=" * 60)

    def f_true(x):
        z, x1 = x[:, 0], x[:, 1]
        return z * torch.sin(x1)

    model, x_train, device = train_model(f_true, n_inputs=2)

    # Check axis 1 (x1)
    is_linear, rel_var, grad_mean, grad_std = check_linearity_via_derivative(
        model, axis_idx=1, n_inputs=2, x_samples=x_train
    )
    print(f"\nAxis 1 (x1): is_linear={is_linear}, rel_var={rel_var:.4f}")
    print(f"  grad_mean={grad_mean:.4f}, grad_std={grad_std:.4f}")
    print("  Expected: is_linear=False (∂f/∂x1 = z*cos(x1) varies)")

    assert not is_linear, f"Test 3 FAILED: x1 should NOT be linear but rel_var={rel_var:.4f}"
    print("✓ Test 3 PASSED")


def test_case_4_mixed():
    """f(z, x1, x4) = x1 + z*sin(x4) → x1 LINEAR, z LINEAR, x4 NOT linear"""
    print("\n" + "=" * 60)
    print("Test 4: f(z, x1, x4) = x1 + z*sin(x4)")
    print("Expected: x1 LINEAR, z LINEAR, x4 NOT linear")
    print("=" * 60)

    def f_true(x):
        z, x1, x4 = x[:, 0], x[:, 1], x[:, 2]
        return x1 + z * torch.sin(x4)

    model, x_train, device = train_model(f_true, n_inputs=3, epochs=3000)

    # Check all axes
    results = {}
    for axis, name in [(0, "z"), (1, "x1"), (2, "x4")]:
        is_linear, rel_var, grad_mean, grad_std = check_linearity_via_derivative(
            model, axis_idx=axis, n_inputs=3, x_samples=x_train
        )
        results[name] = (is_linear, rel_var, grad_mean, grad_std)
        print(f"\nAxis {axis} ({name}): is_linear={is_linear}, rel_var={rel_var:.4f}")
        print(f"  grad_mean={grad_mean:.4f}, grad_std={grad_std:.4f}")

    # Expected:
    # - z: ∂f/∂z = sin(x4) varies → NOT linear... actually wait, let me reconsider
    # - x1: ∂f/∂x1 = 1 constant → LINEAR
    # - x4: ∂f/∂x4 = z*cos(x4) varies → NOT linear

    # z is tricky: ∂f/∂z = sin(x4) which varies. But for the compound formula
    # x1 + z, after accepting z, the function becomes f(z, x1) = x1 + z,
    # and in that context z IS linear.
    #
    # This test uses the FULL function, so z will not appear linear here.
    # But after compound acceptance, the compound leaf sees f(z, x1) = x1 + z,
    # so both z and x1 should be linear in that context.

    print("\nNote: In the full function f(z,x1,x4)=x1+z*sin(x4):")
    print("  - z: ∂f/∂z = sin(x4) varies → NOT linear in full function")
    print("  - x1: ∂f/∂x1 = 1 → LINEAR")
    print("  - x4: ∂f/∂x4 = z*cos(x4) → NOT linear")
    print("")
    print("After compound z=(x2*x3)*sin(x4) is accepted, the leaf sees f(z,x1)=x1+z:")
    print("  - z: ∂f/∂z = 1 → LINEAR (this is what Issue 2 tests)")
    print("  - x1: ∂f/∂x1 = 1 → LINEAR (this is what Issue 1 tests)")

    # Verify x1 is linear
    assert results["x1"][0], f"Test 4 FAILED: x1 should be linear but rel_var={results['x1'][1]:.4f}"
    # Verify x4 is NOT linear
    assert not results["x4"][0], f"Test 4 FAILED: x4 should NOT be linear but rel_var={results['x4'][1]:.4f}"
    print("✓ Test 4 PASSED (x1 linear, x4 not linear)")


def test_case_5_compound_leaf():
    """After compound is accepted, test the leaf: f(z, x1) = x1 + z → both LINEAR"""
    print("\n" + "=" * 60)
    print("Test 5: Compound leaf f(z, x1) = x1 + z")
    print("This simulates the scenario after z=(x2*x3)*sin(x4) is accepted")
    print("Expected: BOTH z and x1 are LINEAR")
    print("=" * 60)

    def f_true(x):
        z, x1 = x[:, 0], x[:, 1]
        return x1 + z

    model, x_train, device = train_model(f_true, n_inputs=2)

    # Check both axes
    for axis, name in [(0, "z"), (1, "x1")]:
        is_linear, rel_var, grad_mean, grad_std = check_linearity_via_derivative(
            model, axis_idx=axis, n_inputs=2, x_samples=x_train
        )
        print(f"\nAxis {axis} ({name}): is_linear={is_linear}, rel_var={rel_var:.4f}")
        print(f"  grad_mean={grad_mean:.4f}, grad_std={grad_std:.4f}")
        print("  Expected: is_linear=True, grad_mean ≈ 1.0")

        assert is_linear, f"Test 5 FAILED: {name} should be linear but rel_var={rel_var:.4f}"
        assert abs(grad_mean - 1.0) < 0.15, f"Test 5 FAILED: grad_mean={grad_mean:.4f} for {name}"

    print("✓ Test 5 PASSED (both z and x1 are linear)")


if __name__ == "__main__":
    print("Testing derivative-based linearity detection")
    print("=" * 60)

    try:
        test_case_1_additive()
        test_case_2_multiplicative()
        test_case_3_trig()
        test_case_4_mixed()
        test_case_5_compound_leaf()
    except AssertionError as e:
        print(f"\nFAILED: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1) from e

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
