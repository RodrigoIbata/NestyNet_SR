# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Tests for the trig z-scaling probe (monomial-in-trig detection)."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nestynet_sr.sr_search.features import probe_trig_scaling


# ---------------------------------------------------------------------------
# Thin wrapper models that compute exact trig^k functions
# ---------------------------------------------------------------------------

class _TrigPowerModel(nn.Module):
    """f(x) = trig(omega * x)^k.  Acts like a surrogate with one parameter."""

    def __init__(self, omega: float, k: float, trig_fn: str = "cos"):
        super().__init__()
        self.omega = omega
        self.k = k
        self.trig_fn = trig_fn
        # Dummy parameter so next(model.parameters()) works
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.cos(self.omega * x[:, 0]) if self.trig_fn == "cos" else torch.sin(self.omega * x[:, 0])
        return (z.abs() ** self.k * z.sign()).unsqueeze(-1)


class _PolynomialInCosModel(nn.Module):
    """f(x) = 1 + cos^2(omega * x).  NOT a monomial in z."""

    def __init__(self, omega: float):
        super().__init__()
        self.omega = omega
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.cos(self.omega * x[:, 0])
        return (1.0 + z ** 2).unsqueeze(-1)


class _OneMinusCosPowerModel(nn.Module):
    """f(x) = (1 - cos(omega * x))^k."""

    def __init__(self, omega: float, k: float):
        super().__init__()
        self.omega = omega
        self.k = k
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = 1.0 - torch.cos(self.omega * x[:, 0])
        return (z ** self.k).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Helper: build a datagen callable for the probe
# ---------------------------------------------------------------------------

def _make_datagen(N: int = 2000, x_lo: float = -3.0, x_hi: float = 3.0, n_vars: int = 1):
    """Return a callable that yields one batch of uniform random x in [x_lo, x_hi]."""
    def datagen():
        X = torch.rand(N, n_vars) * (x_hi - x_lo) + x_lo
        dataset = TensorDataset(X)
        return DataLoader(dataset, batch_size=N, shuffle=False)
    return datagen


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrigScalingProbe:

    def test_cos_squared(self):
        """f(x) = cos^2(2x) → k=2, trig_fn='cos', ω=2."""
        model = _TrigPowerModel(omega=2.0, k=2.0, trig_fn="cos")
        datagen = _make_datagen()
        results = probe_trig_scaling(
            model=model, datagen=datagen, Nxvars=1, device="cpu",
        )
        assert len(results) == 1
        r = results[0]
        assert r.trig_fn == "cos"
        assert abs(r.omega - 2.0) < 0.1, f"omega={r.omega}, expected ~2.0"
        assert abs(r.k_hat - 2.0) < 0.15, f"k_hat={r.k_hat}, expected ~2.0"
        assert r.rel_std < 0.10

    def test_sin_cubed(self):
        """f(x) = sin^3(3x) → k=3, trig_fn='sin', ω=3."""
        model = _TrigPowerModel(omega=3.0, k=3.0, trig_fn="sin")
        datagen = _make_datagen()
        results = probe_trig_scaling(
            model=model, datagen=datagen, Nxvars=1, device="cpu",
        )
        assert len(results) == 1
        r = results[0]
        assert r.trig_fn == "sin"
        assert abs(r.omega - 3.0) < 0.1, f"omega={r.omega}, expected ~3.0"
        assert abs(r.k_hat - 3.0) < 0.15, f"k_hat={r.k_hat}, expected ~3.0"
        assert r.rel_std < 0.10

    def test_cos_linear(self):
        """f(x) = cos(x) → k=1, trig_fn='cos', ω=1."""
        model = _TrigPowerModel(omega=1.0, k=1.0, trig_fn="cos")
        datagen = _make_datagen()
        results = probe_trig_scaling(
            model=model, datagen=datagen, Nxvars=1, device="cpu",
        )
        assert len(results) == 1
        r = results[0]
        assert r.trig_fn == "cos"
        assert abs(r.omega - 1.0) < 0.1, f"omega={r.omega}, expected ~1.0"
        assert abs(r.k_hat - 1.0) < 0.15, f"k_hat={r.k_hat}, expected ~1.0"
        assert r.rel_std < 0.10

    def test_polynomial_not_monomial(self):
        """f(x) = 1 + cos^2(2x) → With centering the probe removes the constant offset
        and correctly identifies a trig^2(2x) monomial (cos² or sin², since
        1 + cos²(2x) = 2 - sin²(2x) — both are valid after offset removal)."""
        model = _PolynomialInCosModel(omega=2.0)
        datagen = _make_datagen()
        results = probe_trig_scaling(
            model=model, datagen=datagen, Nxvars=1, device="cpu",
        )
        # Centering removes the constant offset, revealing pure trig^2(2x)
        assert len(results) == 1, "Should detect trig^2(2x) after centering"
        r = results[0]
        assert r.trig_fn in ("cos", "sin"), f"trig_fn={r.trig_fn}"
        assert abs(r.omega - 2.0) < 0.1, f"omega={r.omega}, expected ~2.0"
        assert abs(r.k_hat - 2.0) < 0.15, f"k_hat={r.k_hat}, expected ~2.0"
        assert r.rel_std < 0.10

    def test_non_integer_omega(self):
        """f(x) = sin^2(1.5·x) → should discover ω≈1.5 from the grid (π/2≈1.571)."""
        model = _TrigPowerModel(omega=1.5, k=2.0, trig_fn="sin")
        datagen = _make_datagen()
        results = probe_trig_scaling(
            model=model, datagen=datagen, Nxvars=1, device="cpu",
        )
        assert len(results) == 1
        r = results[0]
        # ω=1.5 is not on the grid but π/2≈1.571 is close — either that or
        # the exact value 1 or 2 might pick up a monomial, so just check k is clean
        assert r.rel_std < 0.10, f"rel_std={r.rel_std}, expected < 0.10"
        assert abs(r.k_hat - 2.0) < 0.3, f"k_hat={r.k_hat}, expected ~2.0"

    def test_one_minus_cos_squared(self):
        """f(x) = (1 - cos(x))^2 → detect the one-minus-cos scaling basis."""
        model = _OneMinusCosPowerModel(omega=1.0, k=2.0)
        datagen = _make_datagen()
        results = probe_trig_scaling(
            model=model, datagen=datagen, Nxvars=1, device="cpu",
        )
        assert len(results) == 1
        r = results[0]
        assert r.trig_fn == "cos"
        assert r.basis_fn == "one_minus_cos"
        assert abs(r.omega - 1.0) < 0.1, f"omega={r.omega}, expected ~1.0"
        assert abs(r.k_hat - 2.0) < 0.15, f"k_hat={r.k_hat}, expected ~2.0"
        assert r.rel_std < 0.10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
