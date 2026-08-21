#!/usr/bin/env python
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Tests for the additive gauge-fix mechanism.

The gauge-fix penalises non-zero leaf output when private variables are at
their median values, pinning the additive gauge freedom that arises when
two leaves share variables.
"""

import torch
import torch.nn as nn


def _make_mock_composite(*leaves):
    """Build a mock composite model wrapping the given leaf modules."""
    class MockComposite(nn.Module):
        def __init__(self):
            super().__init__()
            self.leaf = nn.ModuleList(list(leaves))
    return MockComposite()


class _FormulaLeaf(nn.Module):
    """Small analytic leaf used for overlap truth and init tests."""

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def forward(self, x):
        y = self._fn(x)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        return y


def test_gauge_fix_wrapper_forward():
    """_GaugeFixWrapper replaces private columns with reference values."""
    from nestynet_sr.adaptors.gauge_fix_adaptor import _GaugeFixWrapper

    # Simple leaf: 2-input NN
    leaf = nn.Linear(2, 1, bias=True)
    nn.init.ones_(leaf.weight)
    nn.init.zeros_(leaf.bias)

    composite = _make_mock_composite(leaf)

    # Private variable at local index 1, reference value = 5.0
    wrapper = _GaugeFixWrapper(
        composite, leaf_idx=0,
        private_local_idxs=[1], ref_values=torch.tensor([5.0]), weight=2.0,
    )

    x = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    out = wrapper(x)

    # With private col replaced: x_mod = [[1, 5], [2, 5]]
    # leaf(x_mod) = 1*x0 + 1*5 = [6, 7] => weight * [6, 7] = [12, 14]
    expected = 2.0 * leaf(torch.tensor([[1.0, 5.0], [2.0, 5.0]]))
    assert torch.allclose(out, expected, atol=1e-6), f"{out} != {expected}"
    print("PASSED: wrapper forward")


def test_gauge_fix_wrapper_shares_params():
    """Wrapper exposes the composite model's full parameter set."""
    from nestynet_sr.adaptors.gauge_fix_adaptor import _GaugeFixWrapper

    leaf0 = nn.Linear(3, 1)
    leaf1 = nn.Linear(2, 1)
    composite = _make_mock_composite(leaf0, leaf1)

    wrapper = _GaugeFixWrapper(
        composite, leaf_idx=0,
        private_local_idxs=[2], ref_values=torch.tensor([0.0]),
    )

    composite_params = set(id(p) for p in composite.parameters())
    wrapper_params = set(id(p) for p in wrapper.parameters())

    # Wrapper must expose ALL composite parameters (not just one leaf)
    assert composite_params == wrapper_params, (
        f"Wrapper should expose all composite params: "
        f"{len(composite_params)} composite vs {len(wrapper_params)} wrapper"
    )
    print("PASSED: parameter sharing (full composite)")


def test_build_gauge_fix_factory():
    """build_gauge_fix_factory produces a working ResidualsModule factory."""
    from nestynet_sr.adaptors.gauge_fix_adaptor import build_gauge_fix_factory
    from nestynet_sr.sr_core.bridges import AtomNode

    # Create a simple 2-variable atom
    atom = AtomNode(var_idxs=[0, 1], kind="nn")

    leaf = nn.Linear(2, 1)
    nn.init.zeros_(leaf.weight)
    nn.init.zeros_(leaf.bias)

    model = _make_mock_composite(leaf)

    x_train = torch.randn(50, 3, dtype=torch.float64)
    factory = build_gauge_fix_factory(
        model, leaf_idx=0, atom=atom,
        private_global_idxs=[1],  # x1 is private
        x_train=x_train,
        device=torch.device("cpu"),
        dtype=torch.float64,
        weight=1.0,
    )

    assert factory is not None, "Factory should not be None"

    # Call the factory
    rm = factory(None)
    assert rm is not None, "ResidualsModule should not be None"
    print("PASSED: build_gauge_fix_factory")


def test_gauge_fix_metrics_raw_vs_weighted():
    """Raw gauge metrics should not depend on the wrapper penalty weight."""
    from types import SimpleNamespace

    from nestynet_sr.adaptors.gauge_fix_adaptor import _GaugeFixWrapper, gauge_fix_metrics

    leaf = nn.Linear(2, 1, bias=True, dtype=torch.float64)
    nn.init.ones_(leaf.weight)
    nn.init.zeros_(leaf.bias)
    composite = _make_mock_composite(leaf)

    x_leaf = torch.tensor([[1.0, 3.0], [2.0, 4.0]], dtype=torch.float64)
    wrapper = _GaugeFixWrapper(
        composite,
        leaf_idx=0,
        private_local_idxs=[1],
        ref_values=torch.tensor([5.0], dtype=torch.float64),
        weight=0.25,
    )
    fac = SimpleNamespace(_gauge_wrapper=wrapper, _gauge_x_leaf=x_leaf)

    weighted = gauge_fix_metrics([fac], raw=False)[0]
    raw = gauge_fix_metrics([fac], raw=True)[0]

    expected_raw = leaf(torch.tensor([[1.0, 5.0], [2.0, 5.0]], dtype=torch.float64))
    expected_raw_rms = float((expected_raw ** 2).mean().sqrt())

    assert abs(raw["rms"] - expected_raw_rms) < 1e-10
    assert abs(weighted["rms"] - 0.25 * expected_raw_rms) < 1e-10
    print("PASSED: gauge_fix_metrics raw vs weighted")


def test_overlap_truth_metric_additive_exact():
    """Gauge-invariant additive overlap residual should vanish for a true split."""
    from torch.utils.data import DataLoader, TensorDataset

    from nestynet_sr.sr_core.bridges import AtomNode
    from nestynet_sr.sr_search.search import _evaluate_overlap_truth_metric

    dtype = torch.float64
    device = torch.device("cpu")
    x = torch.randn(256, 3, dtype=dtype)
    y = torch.zeros(256, 1, dtype=dtype)
    dl = DataLoader(TensorDataset(x, y), batch_size=256, shuffle=False)

    parent_ast = AtomNode(var_idxs=[0, 1, 2], kind="nn", tag="A0")
    parent_leaf = _FormulaLeaf(lambda t: t[:, 0] + 2.0 * t[:, 1] + 3.0 * t[:, 2])
    parent_model = _make_mock_composite(parent_leaf)

    metric = _evaluate_overlap_truth_metric(
        parent_model=parent_model,
        current_ast=parent_ast,
        parent_tag="A0",
        g1=[0, 1],
        g2=[0, 2],
        datagen=dl,
        device=device,
        dtype=dtype,
        op=torch.add,
    )

    assert metric is not None
    assert metric["normalized_rms"] < 1.0e-12, metric
    print("PASSED: additive overlap truth metric")


def test_overlap_truth_metric_multiplicative_exact_and_false():
    """Multiplicative overlap truth metric should separate true and false splits."""
    from torch.utils.data import DataLoader, TensorDataset

    from nestynet_sr.sr_core.bridges import AtomNode
    from nestynet_sr.sr_search.search import _evaluate_overlap_truth_metric

    dtype = torch.float64
    device = torch.device("cpu")
    x = 0.25 + torch.rand(256, 3, dtype=dtype)
    y = torch.zeros(256, 1, dtype=dtype)
    dl = DataLoader(TensorDataset(x, y), batch_size=256, shuffle=False)

    parent_ast = AtomNode(var_idxs=[0, 1, 2], kind="nn", tag="A0")

    true_leaf = _FormulaLeaf(lambda t: (1.0 + t[:, 0] + t[:, 1]) * (2.0 + t[:, 2]))
    true_model = _make_mock_composite(true_leaf)
    true_metric = _evaluate_overlap_truth_metric(
        parent_model=true_model,
        current_ast=parent_ast,
        parent_tag="A0",
        g1=[0, 1],
        g2=[0, 2],
        datagen=dl,
        device=device,
        dtype=dtype,
        op=torch.multiply,
    )

    false_leaf = _FormulaLeaf(
        lambda t: (1.0 + t[:, 0]) * t[:, 1] * t[:, 2] + (2.0 - t[:, 0]) * (t[:, 1] ** 2)
    )
    false_model = _make_mock_composite(false_leaf)
    false_metric = _evaluate_overlap_truth_metric(
        parent_model=false_model,
        current_ast=parent_ast,
        parent_tag="A0",
        g1=[0, 1],
        g2=[0, 2],
        datagen=dl,
        device=device,
        dtype=dtype,
        op=torch.multiply,
    )

    assert true_metric is not None
    assert false_metric is not None
    assert true_metric["normalized_rms"] < 1.0e-12, true_metric
    assert false_metric["normalized_rms"] > 1.0e-2, false_metric
    print("PASSED: multiplicative overlap truth metric")


def test_build_additive_gauge_fix_factories():
    """_build_additive_gauge_fix_factories produces factories for overlapping splits."""
    from nestynet_sr.sr_search.search import _build_additive_gauge_fix_factories
    from nestynet_sr.sr_core.bridges import AddNode, AtomNode

    # Build a simple AST: NN[x0,x1,x2] + NN[x0,x1,x3]
    # shared: {x0, x1}, left private: {x2}, right private: {x3}
    left = AtomNode(var_idxs=[0, 1, 2], kind="nn", tag="A0_L")
    right = AtomNode(var_idxs=[0, 1, 3], kind="nn", tag="A0_R")
    root = AddNode(left, right)

    left_leaf = nn.Linear(3, 1, dtype=torch.float64)
    right_leaf = nn.Linear(3, 1, dtype=torch.float64)
    model = _make_mock_composite(left_leaf, right_leaf)

    # Training data
    N = 100
    x_train = torch.randn(N, 4, dtype=torch.float64)
    from torch.utils.data import DataLoader, TensorDataset
    y_train = torch.randn(N, 1, dtype=torch.float64)
    dl = DataLoader(TensorDataset(x_train, y_train), batch_size=N)

    factories = _build_additive_gauge_fix_factories(
        model, root,
        g1=[0, 1, 2], g2=[0, 1, 3],
        parent_tag="A0",
        datagen=dl,
        device=torch.device("cpu"),
        dtype=torch.float64,
        weight=1.0,
    )

    assert len(factories) == 1, f"Expected 1 factory, got {len(factories)}"
    # Verify the factory produces a ResidualsModule
    rm = factories[0](None)
    assert rm is not None
    print("PASSED: _build_additive_gauge_fix_factories")


def test_teacher_init_additive_canonical_overlap():
    """Additive teacher init should anchor the right overlap leaf at zero on v0."""
    from torch.utils.data import DataLoader, TensorDataset

    from nestynet_sr.sr_core.bridges import AddNode, AtomNode
    from nestynet_sr.sr_search.search import _teacher_init_additive

    torch.manual_seed(7)
    dtype = torch.float64
    device = torch.device("cpu")

    parent_ast = AtomNode(var_idxs=[0, 1, 2], kind="nn", tag="A0")
    cand_ast = AddNode(
        AtomNode(var_idxs=[0, 1], kind="nn", tag="A0_L"),
        AtomNode(var_idxs=[0, 2], kind="nn", tag="A0_R"),
    )

    parent_leaf = _FormulaLeaf(lambda t: t[:, 0] + 2.0 * t[:, 1] + 3.0 * t[:, 2])
    parent_model = _make_mock_composite(parent_leaf)

    left_leaf = nn.Linear(2, 1, dtype=dtype)
    right_leaf = nn.Linear(2, 1, dtype=dtype)
    nn.init.zeros_(left_leaf.weight)
    nn.init.zeros_(left_leaf.bias)
    nn.init.zeros_(right_leaf.weight)
    nn.init.zeros_(right_leaf.bias)
    cand_model = _make_mock_composite(left_leaf, right_leaf)

    x = 0.25 + torch.rand(256, 3, dtype=dtype)
    y = torch.zeros(256, 1, dtype=dtype)
    dl = DataLoader(TensorDataset(x, y), batch_size=256, shuffle=False)

    _teacher_init_additive(
        cand_model,
        cand_ast,
        parent_model,
        parent_ast,
        dl,
        device,
        dtype,
        parent_tag="A0",
    )

    with torch.no_grad():
        x_left = x[:, [0, 1]]
        x_right = x[:, [0, 2]]
        x_right_ref = x_right.clone()
        x_right_ref[:, 1] = torch.median(x[:, 2])
        pred = left_leaf(x_left) + right_leaf(x_right)
        ref_out = right_leaf(x_right_ref)
        target = parent_leaf(x)

    fit_rms = float(((pred - target) ** 2).mean().sqrt())
    target_rms = float((target ** 2).mean().sqrt())
    anchor_rms = float((ref_out ** 2).mean().sqrt())
    assert fit_rms < 0.5 * target_rms, (fit_rms, target_rms)
    assert anchor_rms < 1.0e-1, anchor_rms
    print("PASSED: additive teacher init canonical overlap")


def test_no_factories_for_disjoint_split():
    """Disjoint splits (no shared vars) produce no gauge-fix factories."""
    from nestynet_sr.sr_search.search import _build_additive_gauge_fix_factories
    from nestynet_sr.sr_core.bridges import AddNode, AtomNode

    left = AtomNode(var_idxs=[0, 1], kind="nn", tag="A0_L")
    right = AtomNode(var_idxs=[2, 3], kind="nn", tag="A0_R")
    root = AddNode(left, right)

    left_leaf = nn.Linear(2, 1, dtype=torch.float64)
    right_leaf = nn.Linear(2, 1, dtype=torch.float64)
    model = _make_mock_composite(left_leaf, right_leaf)

    N = 50
    x_train = torch.randn(N, 4, dtype=torch.float64)
    from torch.utils.data import DataLoader, TensorDataset
    dl = DataLoader(TensorDataset(x_train, torch.randn(N, 1, dtype=torch.float64)), batch_size=N)

    factories = _build_additive_gauge_fix_factories(
        model, root,
        g1=[0, 1], g2=[2, 3],
        parent_tag="A0",
        datagen=dl,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert len(factories) == 0, f"Expected 0 factories for disjoint split, got {len(factories)}"
    print("PASSED: no factories for disjoint split")


def test_gauge_fix_reduces_gauge_contamination():
    """Gauge-fix penalty should reduce leaf output at the reference point.

    f(x0, x1, x2) = x0 + x2, split g(x0, x1) + h(x1, x2), shared={x1}.
    With gauge fix on right leaf: h(x1, x2_ref) ~ 0.
    """
    from nestynet_sr.adaptors.gauge_fix_adaptor import _GaugeFixWrapper

    torch.manual_seed(42)

    N = 200
    x = 0.5 + torch.rand(N, 3, dtype=torch.float64)
    y = (x[:, 0] + x[:, 2]).unsqueeze(1)

    left_leaf = nn.Sequential(
        nn.Linear(2, 32, dtype=torch.float64),
        nn.Tanh(),
        nn.Linear(32, 1, dtype=torch.float64),
    )
    right_leaf = nn.Sequential(
        nn.Linear(2, 32, dtype=torch.float64),
        nn.Tanh(),
        nn.Linear(32, 1, dtype=torch.float64),
    )

    composite = _make_mock_composite(left_leaf, right_leaf)

    x_left = x[:, [0, 1]]
    x_right = x[:, [1, 2]]

    ref_x2 = float(torch.median(x[:, 2]))
    wrapper = _GaugeFixWrapper(
        composite, leaf_idx=1,
        private_local_idxs=[1],
        ref_values=torch.tensor([ref_x2], dtype=torch.float64),
        weight=1.0,
    )

    optimizer = torch.optim.Adam(composite.parameters(), lr=1e-3)

    for epoch in range(3000):
        pred = left_leaf(x_left) + right_leaf(x_right)
        data_loss = ((pred - y) ** 2).mean()

        gauge_out = wrapper(x_right)
        gauge_loss = (gauge_out ** 2).mean()

        loss = data_loss + gauge_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        gauge_out = wrapper(x_right)
        gauge_rms = (gauge_out ** 2).mean().sqrt().item()

    with torch.no_grad():
        pred = left_leaf(x_left) + right_leaf(x_right)
        data_rms = ((pred - y) ** 2).mean().sqrt().item()

    print(f"  data_rms={data_rms:.4e}, gauge_rms={gauge_rms:.4e}")
    assert data_rms < 0.05, f"Data fit too poor: {data_rms:.4e}"
    assert gauge_rms < 0.05, f"Gauge penalty too large: {gauge_rms:.4e}"
    print("PASSED: gauge fix reduces contamination")


def test_overlap_gauge_stage_feasibility_policy():
    """Non-zero gauge stages must improve gauge while keeping data loss bounded."""
    from nestynet_sr.sr_search.search import _overlap_gauge_stage_is_feasible

    feasible, data_cap, gauge_cap = _overlap_gauge_stage_is_feasible(
        baseline_val_loss=1.0e-6,
        stage_val_loss=4.0e-6,
        accept_threshold=3.0e-5,
        baseline_gauge_rms=10.0,
        stage_gauge_rms=2.0,
        max_data_regress_factor=10.0,
        required_improve_factor=0.3,
    )
    assert feasible
    assert abs(data_cap - 1.0e-5) < 1e-12
    assert abs(gauge_cap - 3.0) < 1e-12

    feasible_bad, _, _ = _overlap_gauge_stage_is_feasible(
        baseline_val_loss=1.0e-6,
        stage_val_loss=2.0e-6,
        accept_threshold=3.0e-5,
        baseline_gauge_rms=10.0,
        stage_gauge_rms=8.0,
        max_data_regress_factor=10.0,
        required_improve_factor=0.3,
    )
    assert not feasible_bad, "Gauge improvement should be required for non-zero stages"

    feasible_tiny, _, gauge_cap_tiny = _overlap_gauge_stage_is_feasible(
        baseline_val_loss=1.0e-6,
        stage_val_loss=1.2e-6,
        accept_threshold=3.0e-5,
        baseline_gauge_rms=0.0,
        stage_gauge_rms=5.0e-11,
        max_data_regress_factor=10.0,
        required_improve_factor=0.3,
        tiny_baseline_relax_factor=1.25,
        tiny_baseline_eps=1.0e-10,
    )
    assert feasible_tiny
    assert abs(gauge_cap_tiny - 1.0e-10) < 1e-20
    print("PASSED: overlap gauge stage feasibility policy")


def test_gauge_fix_with_segmented_models():
    """Gauge-fix with real NestyNet segmented models, not just nn.Linear.

    Builds an additive AST: NN[x0,x1] + NN[x1,x2] with shared x1,
    wraps the right leaf (private var x2) in a _GaugeFixWrapper,
    then verifies:
      a) Parameter count consistency between wrapper and composite
      b) AutogradAdaptor forward pass through the segmented model
      c) AutogradAdaptor JVP/VJP through the segmented model
      d) ResidualsModule can be built and fun_residuals called
    """
    import nestynet.nets
    import nestynet.optimizer
    from nestynet.adaptors.adaptors import AutogradAdaptor, SegmentedAdaptor
    from nestynet_sr.adaptors.ast_composite import ASTCompositeAdaptor
    from nestynet_sr.adaptors.gauge_fix_adaptor import _GaugeFixWrapper
    from nestynet_sr.sr_core.bridges import AddNode, AtomNode
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(99)
    dtype = torch.float64
    device = "cpu"
    num_segments = 4
    seg_width = 2
    N = 50  # batch size

    # --- 1. Create two leaf NestyNet segmented models ---
    kw = dict(
        model_base_name="G_Model",
        model_scale=0.1,
        dtype=dtype,
        device=device,
        num_segments=num_segments,
        seg_width=seg_width,
    )
    net_left = nestynet.nets.NestyNet_Model(Nout_size=1, Nx_size=2, **kw)
    net_right = nestynet.nets.NestyNet_Model(Nout_size=1, Nx_size=2, **kw)

    seg = torch.arange(num_segments, device=device)
    adapt_left = SegmentedAdaptor(net_left, segments=seg)
    adapt_right = SegmentedAdaptor(net_right, segments=seg)

    # --- 2. Build ASTCompositeAdaptor: NN[x0,x1] + NN[x1,x2] ---
    atom_left = AtomNode(var_idxs=[0, 1], kind="nn", tag="A0_L")
    atom_right = AtomNode(var_idxs=[1, 2], kind="nn", tag="A0_R")
    ast_root = AddNode(atom_left, atom_right)

    composite = ASTCompositeAdaptor(ast_root, [adapt_left, adapt_right])

    # --- 3. Create _GaugeFixWrapper for right leaf (private var x2) ---
    # In the right leaf NN[x1, x2], x2 is at local index 1.
    ref_x2 = torch.tensor([0.5], dtype=dtype)
    wrapper = _GaugeFixWrapper(
        composite, leaf_idx=1,
        private_local_idxs=[1],
        ref_values=ref_x2,
        weight=1.0,
    )

    # --- Assertion (a): Parameter count matches ---
    composite_nparams = sum(p.numel() for p in composite.parameters())
    wrapper_nparams = sum(p.numel() for p in wrapper.parameters())
    assert composite_nparams == wrapper_nparams, (
        f"Parameter count mismatch: composite has {composite_nparams}, "
        f"wrapper has {wrapper_nparams}"
    )
    assert composite_nparams > 0, "Models should have parameters"
    print(f"  (a) param count OK: {composite_nparams}")

    # --- 4. Wrap in AutogradAdaptor ---
    gauge_adaptor = AutogradAdaptor(wrapper)

    # --- Assertion (b): Forward pass through segmented model ---
    x_leaf = torch.randn(N, 2, dtype=dtype)
    with torch.no_grad():
        fwd_out = gauge_adaptor(x_leaf)
    assert fwd_out.shape == (N, 1), f"Expected shape ({N}, 1), got {fwd_out.shape}"
    assert torch.isfinite(fwd_out).all(), "Forward pass produced non-finite values"
    print(f"  (b) forward pass OK: shape={fwd_out.shape}")

    # --- Assertion (c): JVP and VJP work through the segmented model ---
    y_dummy = torch.zeros(N, 1, dtype=dtype)
    cache = gauge_adaptor.build_cache((x_leaf, y_dummy))
    assert cache is not None, "build_cache returned None"
    assert "f" in cache and cache["f"] is not None, "cache missing 'f'"
    assert "r" in cache and cache["r"] is not None, "cache missing 'r'"

    P = gauge_adaptor.num_parameters()
    assert P == composite_nparams, (
        f"AutogradAdaptor num_parameters ({P}) != composite ({composite_nparams})"
    )

    # JVP: J * v where v is a random parameter-direction vector
    v_rand = torch.randn(P, dtype=dtype)
    jvp_out = gauge_adaptor.jvp(cache, v_rand)
    assert jvp_out is not None, "JVP returned None"
    assert torch.isfinite(jvp_out).all(), "JVP produced non-finite values"
    print(f"  (c.1) JVP OK: shape={jvp_out.shape}")

    # VJP: J^T * w where w is a random residual-direction vector
    r = cache["r"]
    w_rand = torch.randn_like(r.reshape(-1))
    vjp_out = gauge_adaptor.vjp(cache, w_rand)
    assert vjp_out is not None, "VJP returned None"
    assert torch.isfinite(vjp_out).all(), "VJP produced non-finite values"
    # AutogradAdaptor.vjp returns [1, P]
    assert vjp_out.shape == (1, P), f"VJP shape mismatch: {vjp_out.shape} vs (1, {P})"
    print(f"  (c.2) VJP OK: shape={vjp_out.shape}")

    # --- Assertion (d): ResidualsModule can be built and called ---
    gauge_dl = DataLoader(
        TensorDataset(x_leaf, y_dummy),
        batch_size=N, shuffle=False,
    )
    rm = nestynet.optimizer.ResidualsModule(
        providers=[gauge_adaptor],
        dataloader=gauge_dl,
        device=torch.device(device),
    )
    assert rm is not None, "ResidualsModule is None"
    assert rm.fun_residuals is not None, "fun_residuals is None"

    # Call fun_residuals with a data batch
    batch = next(iter(gauge_dl))
    x_batch, y_batch = rm.get_data_batch(batch, torch.device(device))[:2]
    residuals = rm.fun_residuals(None, None, (x_batch, y_batch))
    assert residuals is not None, "fun_residuals returned None"
    assert residuals.shape[0] == N, f"Residuals batch dim mismatch: {residuals.shape[0]} vs {N}"
    assert torch.isfinite(residuals).all(), "fun_residuals produced non-finite values"
    print(f"  (d) ResidualsModule OK: residuals shape={residuals.shape}")

    print("PASSED: gauge fix with segmented models")


def test_mul_gauge_fix_wrapper_forward():
    """In multiplicative mode, wrapper output has zero mean (mean subtracted)."""
    from nestynet_sr.adaptors.gauge_fix_adaptor import _GaugeFixWrapper

    # Leaf: 2-input linear, output varies with both inputs
    leaf = nn.Linear(2, 1, bias=True, dtype=torch.float64)
    nn.init.ones_(leaf.weight)
    nn.init.zeros_(leaf.bias)

    composite = _make_mock_composite(leaf)

    wrapper = _GaugeFixWrapper(
        composite, leaf_idx=0,
        private_local_idxs=[1], ref_values=torch.tensor([5.0]),
        weight=1.0, mode="multiplicative",
    )

    x = torch.tensor([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]], dtype=torch.float64)
    out = wrapper(x)

    # Mean of the output should be ~0 (deviation-from-mean formulation)
    assert abs(out.mean().item()) < 1e-10, f"Mean should be ~0, got {out.mean().item()}"

    # Additive mode should NOT subtract mean
    wrapper_add = _GaugeFixWrapper(
        composite, leaf_idx=0,
        private_local_idxs=[1], ref_values=torch.tensor([5.0]),
        weight=1.0, mode="additive",
    )
    out_add = wrapper_add(x)
    assert abs(out_add.mean().item()) > 0.1, "Additive mode should not subtract mean"
    print("PASSED: multiplicative wrapper forward (zero-mean)")


def test_build_multiplicative_gauge_fix_factories():
    """_build_multiplicative_gauge_fix_factories produces factories for overlapping mul splits."""
    from nestynet_sr.sr_search.search import _build_multiplicative_gauge_fix_factories
    from nestynet_sr.sr_core.bridges import MulNode, AtomNode

    # Build a simple AST: NN[x0,x1,x2] * NN[x0,x1,x3]
    # shared: {x0, x1}, left private: {x2}, right private: {x3}
    left = AtomNode(var_idxs=[0, 1, 2], kind="nn", tag="A0_L")
    right = AtomNode(var_idxs=[0, 1, 3], kind="nn", tag="A0_R")
    root = MulNode(left, right)

    left_leaf = nn.Linear(3, 1, dtype=torch.float64)
    right_leaf = nn.Linear(3, 1, dtype=torch.float64)
    model = _make_mock_composite(left_leaf, right_leaf)

    N = 100
    x_train = torch.randn(N, 4, dtype=torch.float64)
    from torch.utils.data import DataLoader, TensorDataset
    y_train = torch.randn(N, 1, dtype=torch.float64)
    dl = DataLoader(TensorDataset(x_train, y_train), batch_size=N)

    factories = _build_multiplicative_gauge_fix_factories(
        model, root,
        g1=[0, 1, 2], g2=[0, 1, 3],
        parent_tag="A0",
        datagen=dl,
        device=torch.device("cpu"),
        dtype=torch.float64,
        weight=1.0,
    )

    assert len(factories) == 1, f"Expected 1 factory, got {len(factories)}"
    rm = factories[0](None)
    assert rm is not None

    # Verify the wrapper is in multiplicative mode
    wrapper = getattr(factories[0], "_gauge_wrapper", None)
    assert wrapper is not None
    assert wrapper._mode == "multiplicative", f"Expected multiplicative mode, got {wrapper._mode}"
    print("PASSED: _build_multiplicative_gauge_fix_factories")


def test_teacher_init_multiplicative_canonical_overlap():
    """Multiplicative teacher init should anchor the right overlap leaf at one on v0."""
    from torch.utils.data import DataLoader, TensorDataset

    from nestynet_sr.sr_core.bridges import AtomNode, MulNode
    from nestynet_sr.sr_search.search import _teacher_init_multiplicative

    torch.manual_seed(11)
    dtype = torch.float64
    device = torch.device("cpu")

    parent_ast = AtomNode(var_idxs=[0, 1, 2], kind="nn", tag="A0")
    cand_ast = MulNode(
        AtomNode(var_idxs=[0, 1], kind="nn", tag="A0_L"),
        AtomNode(var_idxs=[0, 2], kind="nn", tag="A0_R"),
    )

    parent_leaf = _FormulaLeaf(lambda t: (1.0 + t[:, 0] + t[:, 1]) * (2.0 + t[:, 2]))
    parent_model = _make_mock_composite(parent_leaf)

    left_leaf = nn.Linear(2, 1, dtype=dtype)
    right_leaf = nn.Linear(2, 1, dtype=dtype)
    nn.init.zeros_(left_leaf.weight)
    nn.init.zeros_(left_leaf.bias)
    nn.init.zeros_(right_leaf.weight)
    nn.init.zeros_(right_leaf.bias)
    cand_model = _make_mock_composite(left_leaf, right_leaf)

    x = 0.25 + torch.rand(256, 3, dtype=dtype)
    y = torch.zeros(256, 1, dtype=dtype)
    dl = DataLoader(TensorDataset(x, y), batch_size=256, shuffle=False)

    _teacher_init_multiplicative(
        cand_model,
        cand_ast,
        parent_model,
        parent_ast,
        dl,
        device,
        dtype,
        parent_tag="A0",
    )

    with torch.no_grad():
        x_left = x[:, [0, 1]]
        x_right = x[:, [0, 2]]
        x_right_ref = x_right.clone()
        x_right_ref[:, 1] = torch.median(x[:, 2])
        pred = left_leaf(x_left) * right_leaf(x_right)
        ref_out = right_leaf(x_right_ref)
        target = parent_leaf(x)

    fit_rms = float(((pred - target) ** 2).mean().sqrt())
    target_rms = float((target ** 2).mean().sqrt())
    anchor_rms = float(((ref_out - 1.0) ** 2).mean().sqrt())
    assert fit_rms < 0.5 * target_rms, (fit_rms, target_rms)
    assert anchor_rms < 1.0e-1, anchor_rms
    print("PASSED: multiplicative teacher init canonical overlap")


def test_mul_gauge_fix_reduces_contamination():
    """Multiplicative gauge-fix should make leaf approximately constant at reference.

    f(x0, x1, x2) = x0 * exp(x2), split g(x0, x1) * h(x1, x2), shared={x1}.
    Without gauge fix, h could absorb arbitrary h_g(x1) from g.
    With gauge fix on right leaf: h(x1, x2_ref) ≈ constant.
    """
    from nestynet_sr.adaptors.gauge_fix_adaptor import _GaugeFixWrapper

    torch.manual_seed(42)

    N = 200
    x = 0.5 + torch.rand(N, 3, dtype=torch.float64)
    y = (x[:, 0] * torch.exp(x[:, 2])).unsqueeze(1)

    left_leaf = nn.Sequential(
        nn.Linear(2, 32, dtype=torch.float64),
        nn.Tanh(),
        nn.Linear(32, 1, dtype=torch.float64),
    )
    right_leaf = nn.Sequential(
        nn.Linear(2, 32, dtype=torch.float64),
        nn.Tanh(),
        nn.Linear(32, 1, dtype=torch.float64),
    )

    composite = _make_mock_composite(left_leaf, right_leaf)

    x_left = x[:, [0, 1]]
    x_right = x[:, [1, 2]]

    ref_x2 = float(torch.median(x[:, 2]))
    wrapper = _GaugeFixWrapper(
        composite, leaf_idx=1,
        private_local_idxs=[1],
        ref_values=torch.tensor([ref_x2], dtype=torch.float64),
        weight=1.0,
        mode="multiplicative",
    )

    optimizer = torch.optim.Adam(composite.parameters(), lr=1e-3)

    for epoch in range(3000):
        pred = left_leaf(x_left) * right_leaf(x_right)
        data_loss = ((pred - y) ** 2).mean()

        gauge_out = wrapper(x_right)
        gauge_loss = (gauge_out ** 2).mean()

        loss = data_loss + gauge_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate: h(x1, x2_ref) should be approximately constant
    with torch.no_grad():
        x_right_ref = x_right.clone()
        x_right_ref[:, 1] = ref_x2
        h_at_ref = right_leaf(x_right_ref)
        # Coefficient of variation: std/|mean| should be small
        h_std = h_at_ref.std().item()
        h_mean = h_at_ref.mean().abs().item()
        cv = h_std / max(h_mean, 1e-12)

    with torch.no_grad():
        pred = left_leaf(x_left) * right_leaf(x_right)
        data_rms = ((pred - y) ** 2).mean().sqrt().item()

    print(f"  data_rms={data_rms:.4e}, h_at_ref: mean={h_mean:.4e}, std={h_std:.4e}, CV={cv:.4f}")
    assert data_rms < 0.1, f"Data fit too poor: {data_rms:.4e}"
    assert cv < 0.15, f"Gauge fix ineffective: CV={cv:.4f} (should be < 0.15)"
    print("PASSED: multiplicative gauge fix reduces contamination")


def test_stageb_main_gauge_fix_updates_state_reuse():
    """Stage B should gauge-fix the reuse map it actually uses for proposals."""
    from torch.utils.data import DataLoader, TensorDataset

    from nestynet_sr.adaptors.fixed_shift import FixedOutputShiftAdaptor
    from nestynet_sr.sr_core.bridges import AddNode, AtomNode
    from nestynet_sr.sr_search.stageB.engine import StageBState
    from nestynet_sr.sr_search.stageB.main import _apply_additive_gauge_fix_to_state_reuse

    dtype = torch.float64
    x = torch.randn(64, 2, dtype=dtype)
    y = torch.zeros(64, 1, dtype=dtype)
    dl = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=False)

    left = _FormulaLeaf(lambda t: t[:, 0] * 0.0 - 10.0)
    right = _FormulaLeaf(lambda t: t[:, 0] * 0.0 + 12.0)

    root = AddNode(
        AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0"),
        AtomNode(kind="nn", var_idxs=(1,), kwargs={}, tag="leaf1"),
    )
    state = StageBState(
        root=root,
        model=_make_mock_composite(left, right),
        reuse={"leaf0": left, "leaf1": right},
        val_loss=1.0,
    )

    state = _apply_additive_gauge_fix_to_state_reuse(
        root=root,
        state=state,
        train_loader_probe=dl,
        device=torch.device("cpu"),
        dtype=dtype,
        cancel_ratio_thresh=8.0,
    )

    assert isinstance(state.reuse["leaf0"], FixedOutputShiftAdaptor)
    assert isinstance(state.reuse["leaf1"], FixedOutputShiftAdaptor)

    med_left = torch.median(state.reuse["leaf0"](x[:, [0]])[:, 0]).item()
    med_right = torch.median(state.reuse["leaf1"](x[:, [1]])[:, 0]).item()

    assert abs(med_left - med_right) < 1.0e-10
    assert abs(med_left - 1.0) < 1.0e-10
    print("PASSED: stageB main updates state.reuse during gauge-fix")


if __name__ == "__main__":
    test_gauge_fix_wrapper_forward()
    test_gauge_fix_wrapper_shares_params()
    test_build_gauge_fix_factory()
    test_gauge_fix_metrics_raw_vs_weighted()
    test_overlap_truth_metric_additive_exact()
    test_overlap_truth_metric_multiplicative_exact_and_false()
    test_build_additive_gauge_fix_factories()
    test_teacher_init_additive_canonical_overlap()
    test_no_factories_for_disjoint_split()
    test_gauge_fix_reduces_gauge_contamination()
    test_overlap_gauge_stage_feasibility_policy()
    test_gauge_fix_with_segmented_models()
    test_mul_gauge_fix_wrapper_forward()
    test_build_multiplicative_gauge_fix_factories()
    test_teacher_init_multiplicative_canonical_overlap()
    test_mul_gauge_fix_reduces_contamination()
    test_stageb_main_gauge_fix_updates_state_reuse()
    print("\nAll gauge-fix tests passed!")
