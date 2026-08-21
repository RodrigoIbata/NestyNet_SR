# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Hello-world test for vector DE discovery with order=0.

Ground truth:  u0 = x0^2 - x1^2,  u1 = 2*x0*x1
(i.e. the real and imaginary parts of (x0 + i*x1)^2).

We create a simple mock surrogate that returns the exact polynomial,
then run discover_system_de_from_surrogate with order_candidates=(0,)
and verify the discovered coefficients match.
"""

import torch
import numpy as np

torch.set_default_dtype(torch.float64)


# ── Mock surrogate ──────────────────────────────────────────

class _MockSurrogate(torch.nn.Module):
    """Exact polynomial surrogate: u0 = x0^2 - x1^2, u1 = 2*x0*x1."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0, x1 = x[:, 0], x[:, 1]
        u0 = x0**2 - x1**2
        u1 = 2.0 * x0 * x1
        return torch.stack([u0, u1], dim=1)

    def parameters(self):
        return iter([torch.zeros(1)])  # dummy so device detection works


# ── Simple dataloader stand-in ──────────────────────────────

def _make_dataloader(N=500, seed=42):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-2.0, 2.0, (N, 2))
    X = torch.tensor(x, dtype=torch.float64)
    return [(X,)]  # single-batch iterable


# ── Test ────────────────────────────────────────────────────

def test_order0_complex_square():
    from nestynet_sr.sr_de.system_de_search import (
        discover_system_de_from_surrogate,
        SystemDESearchConfig,
    )
    from nestynet_sr.sr_core.bridges import Var, Mul, Pow

    surrogate = _MockSurrogate()
    dl = _make_dataloader()

    # Build explicit library: x0, x1, x0^2, x1^2, x0*x1
    x0, x1 = Var(0), Var(1)
    lib = [
        x0,
        x1,
        Pow(x0, 2),
        Pow(x1, 2),
        Mul(x0, x1),
    ]

    cfg = SystemDESearchConfig(
        order_candidates=(0,),
        include_const=True,
        stlsq_lambda=1e-4,
    )

    result = discover_system_de_from_surrogate(
        surrogate=surrogate,
        train_dataloader=dl,
        cfg=cfg,
        library_terms=lib,
    )

    assert result.order == 0, f"Expected order=0, got {result.order}"
    assert len(result.out_idxs) == 2

    # Print discovered equations for debugging
    for i in range(len(result.out_idxs)):
        print(result.format_equation(i))

    # Check RMS is very small (exact surrogate, polynomial library)
    for i, rms in enumerate(result.rms_train):
        print(f"  eq{i} RMS = {rms:.2e}")
        assert rms < 1e-10, f"RMS too large for eq{i}: {rms}"

    print("\nPASSED: order=0 vector DE discovery (complex-square)")


if __name__ == "__main__":
    test_order0_complex_square()
