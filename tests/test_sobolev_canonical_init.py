# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
"""Tests for value-projected canonical initialization of the Sobolev adaptor.

The generic greedy canonical initializer fits a scalar value residual and cannot
consume the augmented [value, d/dt, d/dx, ...] target the Sobolev adaptor
presents.  The adaptor therefore intercepts canonical init (high dispatch
priority) and projects the target to its value column before delegating.  These
tests cover the projection and the dispatch wiring; the full end-to-end canonical
init on a real segmented model is exercised by the Maxwell surrogate runs.
"""

from __future__ import annotations

import torch

from nestynet_sr.adaptors.sobolev_residual import SobolevGradientAdaptor


def _adaptor(axes=(0, 1, 2, 3)) -> SobolevGradientAdaptor:
    base = torch.nn.Linear(4, 1).double()  # minimal base; projection never calls it
    return SobolevGradientAdaptor(base, axes=axes)


def test_dispatch_attributes_present():
    a = _adaptor()
    assert int(a.canonical_init_dispatch_priority) == 110
    assert callable(a.canonical_initialize)
    assert callable(a.canonical_default_segments)


def test_value_projection_slices_value_column():
    a = _adaptor(axes=(0, 1, 2, 3))  # n_outputs == 5
    n = 9
    y_aug = torch.randn(n, 5, dtype=torch.float64)
    Yv, Sv = a._canonical_project_target(y_aug)
    assert Yv.shape == (n, 1)
    assert torch.allclose(Yv, y_aug[:, :1])
    assert Sv is None


def test_value_projection_handles_sigma():
    a = _adaptor(axes=(0, 1, 2, 3))
    n = 6
    y_aug = torch.randn(n, 5, dtype=torch.float64)
    # 2D sigma -> value column
    sig = torch.rand(n, 5, dtype=torch.float64) + 0.1
    _, Sv = a._canonical_project_target(y_aug, sig)
    assert Sv.shape == (n, 1)
    assert torch.allclose(Sv, sig[:, :1])
    # scalar sigma passes through unchanged
    _, Sv0 = a._canonical_project_target(y_aug, torch.tensor(2.0, dtype=torch.float64))
    assert float(Sv0) == 2.0


def test_priority_provider_is_discovered():
    from nestynet.training_utils.canonical_init import _find_priority_canonical_provider

    a = _adaptor()
    found = _find_priority_canonical_provider(a, None)
    assert found is a


if __name__ == "__main__":
    test_dispatch_attributes_present()
    test_value_projection_slices_value_column()
    test_value_projection_handles_sigma()
    test_priority_provider_is_discovered()
    print("OK: sobolev canonical-init tests passed")
