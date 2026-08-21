# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Tests for the oracle-based scaling probe and compound null-test.

Tests:
1. test_pure_power:            f = x^2        → k_hat ≈ 2.0
2. test_product:               f = x_0 * x_1  → k_0 ≈ 1, k_1 ≈ 1
3. test_ratio:                 f = x_0 / x_1  → k_0 ≈ 1, k_1 ≈ −1
4. test_compound_null_verified: f = sin(x_0·x_1) → null test passes for z = x_0·x_1
5. test_compound_null_rejected: f = x_0 + x_1   → null test rejects z = x_0·x_1
"""

import pytest
import torch

torch.set_default_dtype(torch.float64)

from nestynet_sr.sr_search.features import (
    ScaleSpec,
    discover_scaling_features,
    probe_oracle_scaling,
    probe_oracle_scaling_groups,
    verify_compound_null_test,
)

# ---------------------------------------------------------------------------
# Mock model: wraps a pure function so probe_oracle_scaling can call
# model(X) and next(model.parameters()).device
# ---------------------------------------------------------------------------


class _MockModel(torch.nn.Module):
    """Lightweight model that evaluates a user-supplied function f(X) -> [N]."""

    def __init__(self, func):
        super().__init__()
        self._func = func
        # Need at least one parameter so next(model.parameters()).device works
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, X):
        return self._func(X).unsqueeze(-1)  # [N, 1]

    def grad(self, X):
        X_req = X.detach().clone().requires_grad_(True)
        values = self._func(X_req)
        grad = torch.autograd.grad(values.sum(), X_req)[0]
        return grad.unsqueeze(1)


def _make_datagen(N, Nx, lo=0.5, hi=2.0, seed=42):
    """Return a callable datagen that yields a single batch of random X."""
    def datagen():
        gen = torch.Generator()
        gen.manual_seed(seed)
        X = lo + (hi - lo) * torch.rand(N, Nx, generator=gen)
        yield (X,)
    return datagen


# ---------------------------------------------------------------------------
# Tests: probe_oracle_scaling
# ---------------------------------------------------------------------------


def test_pure_power():
    """f = x^2 → oracle should find k ≈ 2.0 for variable 0."""
    model = _MockModel(lambda X: X[:, 0] ** 2)
    datagen = _make_datagen(2000, 1)

    specs = probe_oracle_scaling(model, datagen, Nxvars=1)

    assert len(specs) >= 1, "Should find at least one ScaleSpec"
    sp = next(s for s in specs if s.indices == [0])
    assert sp.oracle_verified
    assert abs(sp.oracle_k - 2.0) < 0.15, f"Expected k≈2.0, got {sp.oracle_k}"
    print(f"[PASS] test_pure_power: k={sp.oracle_k:.4f}")


def test_product():
    """f = x_0 * x_1 → k_0 ≈ 1, k_1 ≈ 1."""
    model = _MockModel(lambda X: X[:, 0] * X[:, 1])
    datagen = _make_datagen(2000, 2)

    specs = probe_oracle_scaling(model, datagen, Nxvars=2)

    singles = {s.indices[0]: s for s in specs if len(s.indices) == 1 and s.oracle_verified}
    assert 0 in singles, "Should find scaling for x_0"
    assert 1 in singles, "Should find scaling for x_1"
    assert abs(singles[0].oracle_k - 1.0) < 0.15, f"k_0={singles[0].oracle_k}"
    assert abs(singles[1].oracle_k - 1.0) < 0.15, f"k_1={singles[1].oracle_k}"
    print(f"[PASS] test_product: k_0={singles[0].oracle_k:.4f}, k_1={singles[1].oracle_k:.4f}")


def test_ratio():
    """f = x_0 / x_1 → k_0 ≈ 1, k_1 ≈ −1."""
    model = _MockModel(lambda X: X[:, 0] / X[:, 1])
    datagen = _make_datagen(2000, 2)

    specs = probe_oracle_scaling(model, datagen, Nxvars=2)

    singles = {s.indices[0]: s for s in specs if len(s.indices) == 1 and s.oracle_verified}
    assert 0 in singles, "Should find scaling for x_0"
    assert 1 in singles, "Should find scaling for x_1"
    assert abs(singles[0].oracle_k - 1.0) < 0.15, f"k_0={singles[0].oracle_k}"
    assert abs(singles[1].oracle_k - (-1.0)) < 0.15, f"k_1={singles[1].oracle_k}"
    print(f"[PASS] test_ratio: k_0={singles[0].oracle_k:.4f}, k_1={singles[1].oracle_k:.4f}")


def _pb001_model_and_datagen():
    model = _MockModel(
        lambda X: torch.exp(-0.5 * (X[:, 1] / X[:, 0]) ** 2) / X[:, 0]
    )
    return model, _make_datagen(2500, 2, lo=1.1, hi=2.9)


def test_pb001_is_not_homogeneous_on_either_single_axis():
    model, datagen = _pb001_model_and_datagen()
    specs = probe_oracle_scaling(model, datagen, Nxvars=2)
    assert not any(len(sp.indices) == 1 for sp in specs)


def test_pb001_joint_only_homogeneity_is_oracle_verified():
    """pb001 scales jointly as degree -1, but along neither axis alone."""
    model, datagen = _pb001_model_and_datagen()
    gradient_joint = ScaleSpec(
        name="scale_group_0_1",
        indices=[0, 1],
        k_hat=-1.0,
        mean=-1.0,
        std=0.0,
        rel_std=0.0,
        n_points=2000,
    )

    specs = probe_oracle_scaling(
        model,
        datagen,
        Nxvars=2,
        gradient_specs=[gradient_joint],
    )

    joint = next(
        (sp for sp in specs if sp.indices == [0, 1] and sp.oracle_verified),
        None,
    )
    assert joint is not None
    assert joint.oracle_k == pytest.approx(-1.0, abs=0.03)
    assert joint.oracle_rel_std is not None
    assert joint.oracle_rel_std < 1.0e-3


def test_pb001_gradient_discovery_to_direct_joint_certificate():
    model, datagen = _pb001_model_and_datagen()
    proposals = discover_scaling_features(
        model,
        datagen,
        Nxvars=2,
        max_batches=2,
        max_points=2048,
        max_group_size=2,
    )
    assert not any(len(sp.indices) == 1 for sp in proposals)
    joint_proposals = [sp for sp in proposals if sp.indices == [0, 1]]
    assert len(joint_proposals) == 1

    verified = probe_oracle_scaling_groups(
        model,
        datagen,
        Nxvars=2,
        group_specs=joint_proposals,
        max_batches=2,
        max_points=2048,
    )
    assert len(verified) == 1
    assert verified[0].oracle_k == pytest.approx(-1.0, abs=0.03)


def test_joint_gradient_hint_does_not_bypass_oracle_verification():
    model = _MockModel(lambda X: X[:, 0] + X[:, 1] ** 2)
    datagen = _make_datagen(2500, 2, lo=1.1, hi=2.9)
    misleading_hint = ScaleSpec(
        name="misleading_group_hint",
        indices=[0, 1],
        k_hat=1.0,
        mean=1.0,
        std=0.0,
        rel_std=0.0,
        n_points=2000,
    )

    specs = probe_oracle_scaling(
        model,
        datagen,
        Nxvars=2,
        gradient_specs=[misleading_hint],
    )

    assert not any(sp.indices == [0, 1] and sp.oracle_verified for sp in specs)


def test_direct_group_oracle_supports_more_than_two_axes():
    model = _MockModel(lambda X: X[:, 0] * X[:, 1] * X[:, 2])
    datagen = _make_datagen(2500, 3, lo=1.1, hi=2.9)
    proposal = ScaleSpec("all_axes", [0, 1, 2], 3.0, 3.0, 0.0, 0.0, 2000)

    specs = probe_oracle_scaling_groups(model, datagen, 3, [proposal])

    assert len(specs) == 1
    assert specs[0].indices == [0, 1, 2]
    assert specs[0].oracle_k == pytest.approx(3.0, abs=0.03)


def test_direct_group_oracle_ignores_invalid_axis_groups():
    model = _MockModel(lambda X: X[:, 0] * X[:, 1])
    datagen = _make_datagen(1000, 2)
    invalid = ScaleSpec("invalid", [0, 2], 2.0, 2.0, 0.0, 0.0, 1000)
    assert probe_oracle_scaling_groups(model, datagen, 2, [invalid]) == []


# ---------------------------------------------------------------------------
# Tests: verify_compound_null_test
# ---------------------------------------------------------------------------


def test_compound_null_verified():
    """f = sin(x_0 * x_1) → null test should pass for z = x_0·x_1."""
    model = _MockModel(lambda X: torch.sin(X[:, 0] * X[:, 1]))
    datagen = _make_datagen(3000, 2)

    result = verify_compound_null_test(
        model, datagen,
        z_var_idxs=(0, 1),
        z_exponents=(1, 1),
        Nxvars=2,
    )

    assert result.verified, (
        f"Expected null test to pass, got median_dev={result.median_dev:.6f}, n_valid={result.n_valid}"
    )
    print(f"[PASS] test_compound_null_verified: median_dev={result.median_dev:.6f}")


def test_compound_null_rejected():
    """f = x_0 + x_1 → null test should reject z = x_0·x_1."""
    model = _MockModel(lambda X: X[:, 0] + X[:, 1])
    datagen = _make_datagen(3000, 2, lo=0.2, hi=5.0)

    result = verify_compound_null_test(
        model, datagen,
        z_var_idxs=(0, 1),
        z_exponents=(1, 1),
        Nxvars=2,
        lambda_values=(0.5, 0.7, 1.5, 2.0),
    )

    assert not result.verified, (
        f"Expected null test to reject, but got verified=True with median_dev={result.median_dev:.6f}"
    )
    print(f"[PASS] test_compound_null_rejected: median_dev={result.median_dev:.6f}")


# ---------------------------------------------------------------------------
# Main: run as standalone script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Oracle Scaling Probe Tests")
    print("=" * 60)

    test_pure_power()
    test_product()
    test_ratio()
    test_compound_null_verified()
    test_compound_null_rejected()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
