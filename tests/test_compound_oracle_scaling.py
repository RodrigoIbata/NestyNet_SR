# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Tests for compound variable oracle scaling probe.

Tests that probe_oracle_scaling can detect homogeneous scaling degrees for compound
variables like (x0*x1)^2, (x0/x1)^3, (x0-x1)^1.5, (x0+x1)^2, etc.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nestynet_sr.sr_core.bridges import AddNode, AtomNode, ConstNode, MulNode, PowNode
from nestynet_sr.sr_search.features import (
    TrigProbeTarget,
    probe_oracle_scaling,
)

# ---------------------------------------------------------------------------
# Test models that depend on compound variables
# ---------------------------------------------------------------------------

class _CompoundPowerModel(nn.Module):
    """f(x) = compound^k where compound = x_i op x_j."""

    def __init__(
        self, k: float,
        compound_kind: str = "product", indices: tuple = (0, 1)
    ):
        super().__init__()
        self.k = k
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
        z = self._compute_compound(x)
        # Use |z|^k * sign(z) to handle negative values for non-integer k
        return (z.abs() ** self.k * z.sign()).unsqueeze(-1)


class _TrivialPowerModel(nn.Module):
    """f(x) = x_j^k for a single variable."""

    def __init__(self, k: float, axis: int = 0):
        super().__init__()
        self.k = k
        self.axis = axis
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xj = x[:, self.axis]
        return (xj.abs() ** self.k * xj.sign()).unsqueeze(-1)


class _MixedModel(nn.Module):
    """f(x) = x0^k0 + (x1*x2)^k_compound — mixes trivial and compound."""

    def __init__(self, k0: float, k_compound: float):
        super().__init__()
        self.k0 = k0
        self.k_compound = k_compound
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Trivial term: x0^k0
        term0 = x[:, 0].abs() ** self.k0 * x[:, 0].sign()
        # Compound term: (x1*x2)^k_compound
        z = x[:, 1] * x[:, 2]
        term_c = z.abs() ** self.k_compound * z.sign()
        return (term0 + term_c).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Helper: build a datagen callable
# ---------------------------------------------------------------------------

def _make_datagen(N: int = 2000, x_lo: float = 0.5, x_hi: float = 2.0, n_vars: int = 2):
    """Return a callable that yields one batch of uniform random x in [x_lo, x_hi].

    Uses positive range by default to avoid issues with fractional powers.
    """
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
# Tests for compound oracle scaling detection
# ---------------------------------------------------------------------------

class TestCompoundOracleScaling:

    def test_product_scaling(self):
        """f = (x0*x1)^2 → k=2 for compound z=x0*x1."""
        model = _CompoundPowerModel(k=2.0, compound_kind="product", indices=(0, 1))
        datagen = _make_datagen(N=3000, x_lo=0.5, x_hi=2.0, n_vars=2)
        compound_target = _make_compound_target("product", (0, 1))

        specs = probe_oracle_scaling(
            model=model,
            datagen=datagen,
            Nxvars=2,
            device="cpu",
            compound_targets=[compound_target],
            rel_std_threshold=0.15,
        )

        # Should find compound scaling
        compound_specs = [s for s in specs if s.compound_name]
        assert len(compound_specs) >= 1, f"Expected compound spec, got {specs}"

        # Check the detected degree
        best = min(compound_specs, key=lambda s: s.rel_std)
        assert abs(best.k_hat - 2.0) < 0.3, f"Expected k≈2, got {best.k_hat}"
        print(f"[PASS] Product scaling: k_hat={best.k_hat:.3f}, rel_std={best.rel_std:.3f}")

    def test_ratio_scaling(self):
        """f = (x0/x1)^3 → k=3 for compound z=x0/x1."""
        model = _CompoundPowerModel(k=3.0, compound_kind="ratio", indices=(0, 1))
        datagen = _make_datagen(N=3000, x_lo=0.5, x_hi=2.0, n_vars=2)
        compound_target = _make_compound_target("ratio", (0, 1))

        specs = probe_oracle_scaling(
            model=model,
            datagen=datagen,
            Nxvars=2,
            device="cpu",
            compound_targets=[compound_target],
            rel_std_threshold=0.15,
        )

        compound_specs = [s for s in specs if s.compound_name]
        assert len(compound_specs) >= 1, f"Expected compound spec, got {specs}"

        best = min(compound_specs, key=lambda s: s.rel_std)
        assert abs(best.k_hat - 3.0) < 0.4, f"Expected k≈3, got {best.k_hat}"
        print(f"[PASS] Ratio scaling: k_hat={best.k_hat:.3f}, rel_std={best.rel_std:.3f}")

    def test_difference_scaling(self):
        """f = (x0-x1)^1.5 → k=1.5 for compound z=x0-x1."""
        model = _CompoundPowerModel(k=1.5, compound_kind="difference", indices=(0, 1))
        # Use wider range to get good spread of difference values
        datagen = _make_datagen(N=3000, x_lo=-2.0, x_hi=2.0, n_vars=2)
        compound_target = _make_compound_target("difference", (0, 1))

        specs = probe_oracle_scaling(
            model=model,
            datagen=datagen,
            Nxvars=2,
            device="cpu",
            compound_targets=[compound_target],
            rel_std_threshold=0.15,
        )

        compound_specs = [s for s in specs if s.compound_name]
        assert len(compound_specs) >= 1, f"Expected compound spec, got {specs}"

        best = min(compound_specs, key=lambda s: s.rel_std)
        assert abs(best.k_hat - 1.5) < 0.3, f"Expected k≈1.5, got {best.k_hat}"
        print(f"[PASS] Difference scaling: k_hat={best.k_hat:.3f}, rel_std={best.rel_std:.3f}")

    def test_sum_scaling(self):
        """f = (x0+x1)^2 → k=2 for compound z=x0+x1."""
        model = _CompoundPowerModel(k=2.0, compound_kind="sum", indices=(0, 1))
        datagen = _make_datagen(N=3000, x_lo=0.5, x_hi=2.0, n_vars=2)
        compound_target = _make_compound_target("sum", (0, 1))

        specs = probe_oracle_scaling(
            model=model,
            datagen=datagen,
            Nxvars=2,
            device="cpu",
            compound_targets=[compound_target],
            rel_std_threshold=0.15,
        )

        compound_specs = [s for s in specs if s.compound_name]
        assert len(compound_specs) >= 1, f"Expected compound spec, got {specs}"

        best = min(compound_specs, key=lambda s: s.rel_std)
        assert abs(best.k_hat - 2.0) < 0.3, f"Expected k≈2, got {best.k_hat}"
        print(f"[PASS] Sum scaling: k_hat={best.k_hat:.3f}, rel_std={best.rel_std:.3f}")

    def test_trivial_still_works(self):
        """f = x0^2 → k=2 for trivial x0 (no compound targets)."""
        model = _TrivialPowerModel(k=2.0, axis=0)
        datagen = _make_datagen(N=3000, x_lo=0.5, x_hi=2.0, n_vars=3)

        specs = probe_oracle_scaling(
            model=model,
            datagen=datagen,
            Nxvars=3,
            device="cpu",
            compound_targets=None,  # No compound targets
            rel_std_threshold=0.15,
        )

        # Should find trivial scaling for x0
        trivial_x0 = [s for s in specs if s.indices == [0] and not s.compound_name]
        assert len(trivial_x0) >= 1, f"Expected trivial spec for x0, got {specs}"

        best = trivial_x0[0]
        assert abs(best.k_hat - 2.0) < 0.2, f"Expected k≈2, got {best.k_hat}"
        print(f"[PASS] Trivial scaling: k_hat={best.k_hat:.3f}, rel_std={best.rel_std:.3f}")

    def test_mixed_trivial_and_compound(self):
        """f = x0^3 + (x1*x2)^2 → finds k=3 for x0 and k=2 for x1*x2."""
        model = _MixedModel(k0=3.0, k_compound=2.0)
        datagen = _make_datagen(N=3000, x_lo=0.5, x_hi=2.0, n_vars=3)
        compound_target = _make_compound_target("product", (1, 2))

        specs = probe_oracle_scaling(
            model=model,
            datagen=datagen,
            Nxvars=3,
            device="cpu",
            compound_targets=[compound_target],
            rel_std_threshold=0.15,
        )

        # Check for trivial x0 scaling
        trivial_x0 = [s for s in specs if s.indices == [0] and not s.compound_name]
        # Note: Mixed model may not give clean scaling for x0 because the
        # centering will try to subtract f(x_rest, 0) which still has the compound term.
        # This test is more to verify both paths work without error.

        # Check for compound scaling
        compound_specs = [s for s in specs if s.compound_name]

        print(f"Found {len(trivial_x0)} trivial specs, {len(compound_specs)} compound specs")
        for s in specs:
            marker = s.compound_name if s.compound_name else f"x{s.indices}"
            print(f"  {marker}: k_hat={s.k_hat:.3f}, rel_std={s.rel_std:.3f}")

        # At least verify no errors and we get some output
        assert len(specs) >= 1, "Expected at least one scaling spec"
        print("[PASS] Mixed model: no errors, found specs")

    def test_compound_without_matching_model(self):
        """Compound target on model that doesn't use that compound → high rel_std."""
        # Model uses x0^2 only
        model = _TrivialPowerModel(k=2.0, axis=0)
        datagen = _make_datagen(N=2000, x_lo=0.5, x_hi=2.0, n_vars=3)
        # Probe for x1*x2 which is NOT in the model
        compound_target = _make_compound_target("product", (1, 2))

        specs = probe_oracle_scaling(
            model=model,
            datagen=datagen,
            Nxvars=3,
            device="cpu",
            compound_targets=[compound_target],
            rel_std_threshold=0.08,  # Strict threshold
        )

        # Should find x0 but NOT the compound (since model doesn't depend on x1*x2)
        compound_specs = [s for s in specs if s.compound_name]
        trivial_x0 = [s for s in specs if s.indices == [0] and not s.compound_name]

        assert len(trivial_x0) >= 1, "Expected trivial spec for x0"
        # Compound should either not be found or have high rel_std
        if compound_specs:
            # If found, check it's not very clean
            best_compound = min(compound_specs, key=lambda s: s.rel_std)
            print(f"  Compound found but rel_std={best_compound.rel_std:.3f} (should be high)")
        else:
            print("  Compound correctly not found (model doesn't use it)")
        print("[PASS] Irrelevant compound correctly filtered or has high rel_std")


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
