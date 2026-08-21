# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Tests for compound variable trig z-scaling probe.

Tests that probe_trig_scaling can detect trig dependencies on compound
variables like cos(x2 - x3), sin(x0 + x1), cos(x0 * x1), etc.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nestynet_sr.sr_search.features import (
    TrigProbeTarget,
    _classify_compound_expr,
    _perturb_for_compound,
    probe_trig_scaling,
)
from nestynet_sr.sr_core.bridges import AtomNode, AddNode, MulNode, PowNode, ConstNode


# ---------------------------------------------------------------------------
# Test models that depend on compound variables
# ---------------------------------------------------------------------------

class _CompoundTrigModel(nn.Module):
    """f(x) = trig(omega * compound)^k where compound = x_i op x_j."""

    def __init__(
        self, omega: float, k: float, trig_fn: str = "cos",
        compound_kind: str = "difference", indices: tuple = (0, 1)
    ):
        super().__init__()
        self.omega = omega
        self.k = k
        self.trig_fn = trig_fn
        self.compound_kind = compound_kind
        self.indices = indices
        self._dummy = nn.Parameter(torch.zeros(1))

    def _compute_compound(self, x: torch.Tensor) -> torch.Tensor:
        i, j = self.indices
        if self.compound_kind == "difference":
            return x[:, i] - x[:, j]
        elif self.compound_kind == "sum":
            return x[:, i] + x[:, j]
        elif self.compound_kind == "product":
            return x[:, i] * x[:, j]
        elif self.compound_kind == "ratio":
            return x[:, i] / (x[:, j] + 1e-10)
        else:
            return x[:, i]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_compound = self._compute_compound(x)
        if self.trig_fn == "cos":
            z_trig = torch.cos(self.omega * z_compound)
        else:
            z_trig = torch.sin(self.omega * z_compound)
        return (z_trig.abs() ** self.k * z_trig.sign()).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Helper: build a datagen callable
# ---------------------------------------------------------------------------

def _make_datagen(N: int = 2000, x_lo: float = -2.0, x_hi: float = 2.0, n_vars: int = 2):
    """Return a callable that yields one batch of uniform random x in [x_lo, x_hi]."""
    def datagen():
        X = torch.rand(N, n_vars) * (x_hi - x_lo) + x_lo
        dataset = TensorDataset(X)
        return DataLoader(dataset, batch_size=N, shuffle=False)
    return datagen


def _make_compound_target(kind: str, indices: tuple) -> TrigProbeTarget:
    """Create a TrigProbeTarget for testing."""
    i, j = indices
    if kind == "difference":
        # z = x_i - x_j
        var_i = AtomNode("var", (i,))
        var_j = AtomNode("var", (j,))
        neg_one = ConstNode(-1.0)
        expr = AddNode(var_i, MulNode(neg_one, var_j))
        name = f"x{i}-x{j}"
    elif kind == "sum":
        # z = x_i + x_j
        var_i = AtomNode("var", (i,))
        var_j = AtomNode("var", (j,))
        expr = AddNode(var_i, var_j)
        name = f"x{i}+x{j}"
    elif kind == "product":
        # z = x_i * x_j
        var_i = AtomNode("var", (i,))
        var_j = AtomNode("var", (j,))
        expr = MulNode(var_i, var_j)
        name = f"x{i}*x{j}"
    elif kind == "ratio":
        # z = x_i / x_j
        var_i = AtomNode("var", (i,))
        var_j = AtomNode("var", (j,))
        expr = MulNode(var_i, PowNode(var_j, -1.0))
        name = f"x{i}/x{j}"
    else:
        raise ValueError(f"Unknown compound kind: {kind}")

    return TrigProbeTarget(
        name=name,
        indices=indices,
        expr=expr,
        kind=kind,
        pivot_idx=i,
    )


# ---------------------------------------------------------------------------
# Tests for compound AST classification
# ---------------------------------------------------------------------------

class TestCompoundClassification:

    def test_classify_trivial(self):
        """Single variable x_0 → trivial."""
        expr = AtomNode("var", (0,))
        result = _classify_compound_expr(expr)
        assert result is not None
        kind, indices, pivot = result
        assert kind == "trivial"
        assert indices == (0,)
        assert pivot == 0

    def test_classify_difference(self):
        """x_0 + (-1) * x_1 → difference."""
        var0 = AtomNode("var", (0,))
        var1 = AtomNode("var", (1,))
        neg_one = ConstNode(-1.0)
        expr = AddNode(var0, MulNode(neg_one, var1))
        result = _classify_compound_expr(expr)
        assert result is not None
        kind, indices, pivot = result
        assert kind == "difference"
        assert indices == (0, 1)
        assert pivot == 0

    def test_classify_sum(self):
        """x_0 + x_1 → sum."""
        var0 = AtomNode("var", (0,))
        var1 = AtomNode("var", (1,))
        expr = AddNode(var0, var1)
        result = _classify_compound_expr(expr)
        assert result is not None
        kind, indices, pivot = result
        assert kind == "sum"
        assert indices == (0, 1)
        assert pivot == 0

    def test_classify_product(self):
        """x_0 * x_1 → product."""
        var0 = AtomNode("var", (0,))
        var1 = AtomNode("var", (1,))
        expr = MulNode(var0, var1)
        result = _classify_compound_expr(expr)
        assert result is not None
        kind, indices, pivot = result
        assert kind == "product"
        assert indices == (0, 1)
        assert pivot == 0

    def test_classify_ratio(self):
        """x_0 * x_1^(-1) → ratio."""
        var0 = AtomNode("var", (0,))
        var1 = AtomNode("var", (1,))
        expr = MulNode(var0, PowNode(var1, -1.0))
        result = _classify_compound_expr(expr)
        assert result is not None
        kind, indices, pivot = result
        assert kind == "ratio"
        assert indices == (0, 1)
        assert pivot == 0


# ---------------------------------------------------------------------------
# Tests for compound perturbation
# ---------------------------------------------------------------------------

class TestCompoundPerturbation:

    def test_perturb_trivial(self):
        """Trivial target: just set X[:, j] = target_z."""
        target = TrigProbeTarget(
            name="x0", indices=(0,), expr=None, kind="trivial", pivot_idx=0
        )
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target_z = torch.tensor([10.0, 20.0])
        X_new = _perturb_for_compound(X, target_z, target)
        assert torch.allclose(X_new[:, 0], target_z)
        assert torch.allclose(X_new[:, 1], X[:, 1])  # unchanged

    def test_perturb_difference(self):
        """Difference z = x_i - x_j → set x_i = target_z + x_j."""
        target = _make_compound_target("difference", (0, 1))
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target_z = torch.tensor([5.0, 6.0])  # z = x0 - x1
        X_new = _perturb_for_compound(X, target_z, target)
        # x0_new = target_z + x1 = [5+2, 6+4] = [7, 10]
        assert torch.allclose(X_new[:, 0], torch.tensor([7.0, 10.0]))
        assert torch.allclose(X_new[:, 1], X[:, 1])  # unchanged
        # Verify: x0_new - x1 = target_z
        z_result = X_new[:, 0] - X_new[:, 1]
        assert torch.allclose(z_result, target_z)

    def test_perturb_sum(self):
        """Sum z = x_i + x_j → set x_i = target_z - x_j."""
        target = _make_compound_target("sum", (0, 1))
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target_z = torch.tensor([5.0, 6.0])
        X_new = _perturb_for_compound(X, target_z, target)
        # x0_new = target_z - x1 = [5-2, 6-4] = [3, 2]
        assert torch.allclose(X_new[:, 0], torch.tensor([3.0, 2.0]))
        # Verify: x0_new + x1 = target_z
        z_result = X_new[:, 0] + X_new[:, 1]
        assert torch.allclose(z_result, target_z)

    def test_perturb_product(self):
        """Product z = x_i * x_j → set x_i = target_z / x_j."""
        target = _make_compound_target("product", (0, 1))
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target_z = torch.tensor([6.0, 12.0])
        X_new = _perturb_for_compound(X, target_z, target)
        # x0_new = target_z / x1 = [6/2, 12/4] = [3, 3]
        assert torch.allclose(X_new[:, 0], torch.tensor([3.0, 3.0]))
        # Verify: x0_new * x1 = target_z
        z_result = X_new[:, 0] * X_new[:, 1]
        assert torch.allclose(z_result, target_z)

    def test_perturb_ratio(self):
        """Ratio z = x_i / x_j → set x_i = target_z * x_j."""
        target = _make_compound_target("ratio", (0, 1))
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target_z = torch.tensor([3.0, 2.0])
        X_new = _perturb_for_compound(X, target_z, target)
        # x0_new = target_z * x1 = [3*2, 2*4] = [6, 8]
        assert torch.allclose(X_new[:, 0], torch.tensor([6.0, 8.0]))
        # Verify: x0_new / x1 = target_z
        z_result = X_new[:, 0] / X_new[:, 1]
        assert torch.allclose(z_result, target_z)


# ---------------------------------------------------------------------------
# Tests for compound trig scaling detection
# ---------------------------------------------------------------------------

class TestCompoundTrigScaling:

    def test_cos_squared_difference(self):
        """f(x) = cos^2(2*(x0 - x1)) → detect k=2, omega≈2 on compound x0-x1."""
        model = _CompoundTrigModel(
            omega=2.0, k=2.0, trig_fn="cos",
            compound_kind="difference", indices=(0, 1)
        )
        datagen = _make_datagen(n_vars=2)
        compound_target = _make_compound_target("difference", (0, 1))

        results = probe_trig_scaling(
            model=model, datagen=datagen, Nxvars=2, device="cpu",
            compound_targets=[compound_target],
        )

        # Should detect trig scaling on the compound
        compound_results = [r for r in results if r.compound_name]
        assert len(compound_results) >= 1, f"Expected compound result, got {results}"
        r = compound_results[0]
        assert r.trig_fn == "cos"
        assert abs(r.omega - 2.0) < 0.2, f"omega={r.omega}, expected ~2.0"
        assert abs(r.k_hat - 2.0) < 0.2, f"k_hat={r.k_hat}, expected ~2.0"
        assert r.rel_std < 0.15

    def test_sin_cubed_sum(self):
        """f(x) = sin^3(3*(x0 + x1)) → detect k=3, omega≈3 on compound x0+x1."""
        model = _CompoundTrigModel(
            omega=3.0, k=3.0, trig_fn="sin",
            compound_kind="sum", indices=(0, 1)
        )
        datagen = _make_datagen(n_vars=2)
        compound_target = _make_compound_target("sum", (0, 1))

        results = probe_trig_scaling(
            model=model, datagen=datagen, Nxvars=2, device="cpu",
            compound_targets=[compound_target],
        )

        compound_results = [r for r in results if r.compound_name]
        assert len(compound_results) >= 1, f"Expected compound result, got {results}"
        r = compound_results[0]
        assert r.trig_fn == "sin"
        assert abs(r.omega - 3.0) < 0.2, f"omega={r.omega}, expected ~3.0"
        assert abs(r.k_hat - 3.0) < 0.2, f"k_hat={r.k_hat}, expected ~3.0"
        assert r.rel_std < 0.15

    def test_cos_product(self):
        """f(x) = cos(x0 * x1) → detect k=1 on compound x0*x1."""
        model = _CompoundTrigModel(
            omega=1.0, k=1.0, trig_fn="cos",
            compound_kind="product", indices=(0, 1)
        )
        # Use positive range for product to avoid sign issues
        datagen = _make_datagen(n_vars=2, x_lo=0.5, x_hi=2.0)
        compound_target = _make_compound_target("product", (0, 1))

        results = probe_trig_scaling(
            model=model, datagen=datagen, Nxvars=2, device="cpu",
            compound_targets=[compound_target],
        )

        compound_results = [r for r in results if r.compound_name]
        assert len(compound_results) >= 1, f"Expected compound result, got {results}"
        r = compound_results[0]
        assert r.trig_fn == "cos"
        assert abs(r.k_hat - 1.0) < 0.2, f"k_hat={r.k_hat}, expected ~1.0"
        assert r.rel_std < 0.15

    def test_trivial_axes_still_work(self):
        """Verify that trivial axes (single variables) still work when compounds are passed."""
        # Model depends on x0 only
        class _SimpleTrigModel(nn.Module):
            def __init__(self):
                super().__init__()
                self._dummy = nn.Parameter(torch.zeros(1))

            def forward(self, x):
                return torch.cos(2.0 * x[:, 0]).unsqueeze(-1)

        model = _SimpleTrigModel()
        datagen = _make_datagen(n_vars=2)

        # Pass an irrelevant compound target
        compound_target = _make_compound_target("sum", (0, 1))

        results = probe_trig_scaling(
            model=model, datagen=datagen, Nxvars=2, device="cpu",
            compound_targets=[compound_target],
        )

        # Should still detect x0 as a trivial trig axis
        trivial_results = [r for r in results if not r.compound_name]
        x0_results = [r for r in trivial_results if r.axis == 0]
        assert len(x0_results) >= 1, f"Expected x0 result, got {results}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
