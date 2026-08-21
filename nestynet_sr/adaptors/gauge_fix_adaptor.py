# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""
Gauge-fixing adaptor for overlapping additive and multiplicative splits.

**Additive** splits f = g(x_shared, x_A) + h(x_shared, x_B) have gauge
freedom: any φ(x_shared) can shift between the two leaves.  The penalty
targets zero: sqrt(λ) * leaf(x_shared, x_B_ref).

**Multiplicative** splits f = g(x_shared, x_A) * h(x_shared, x_B) have gauge
freedom: any h_g(x_shared) can move between factors as g*h_g, h/h_g.  The
penalty targets *constant* output: sqrt(λ) * [leaf(x_shared, x_B_ref) - mean],
where mean is detached so gradients don't couple samples.
"""

from typing import List, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _GaugeFixWrapper(nn.Module):
    """Evaluate a leaf at modified inputs (private vars = ref).

    Registers the *entire* composite model as a submodule so that
    ``named_parameters()`` exposes the full parameter set.  The NestyNet
    LM optimizer requires every ``ResidualsModule`` provider to have the
    same parameter dimension; exposing only a single leaf's parameters
    would cause a size-mismatch crash.

    The forward pass touches only one leaf, so autograd naturally
    produces zero gradients for parameters of other leaves.
    """

    def __init__(
        self,
        composite_model: nn.Module,
        leaf_idx: int,
        private_local_idxs: List[int],
        ref_values: torch.Tensor,
        weight: float = 1.0,
        mode: str = "additive",
    ):
        super().__init__()
        # Register the composite model so named_parameters() returns the
        # full parameter set with correct names for functional_call.
        self._composite = composite_model
        self._leaf_idx = leaf_idx
        self._private_local_idxs = list(private_local_idxs)
        self.register_buffer("_ref_values", ref_values.detach().clone())
        self._weight = weight
        self._mode = mode

    @property
    def Nout_size(self):
        return 1

    def _evaluate_leaf(self, x_leaf: torch.Tensor, raw: bool = False) -> torch.Tensor:
        """Evaluate the constrained leaf, optionally without penalty scaling."""
        leaf = self._composite.leaf[self._leaf_idx]
        x_mod = x_leaf.clone()
        for k, idx in enumerate(self._private_local_idxs):
            x_mod[:, idx] = self._ref_values[k]
        out = leaf(x_mod)
        if self._mode == "multiplicative":
            # Penalise deviation from constant, not deviation from zero.
            # Detach the mean so gradients don't couple samples.
            out = out - out.mean(dim=0, keepdim=True).detach()
        if not raw:
            out = self._weight * out
        return out

    def forward(self, x_leaf: torch.Tensor) -> torch.Tensor:
        """Evaluate leaf at x_leaf with private columns replaced by ref."""
        return self._evaluate_leaf(x_leaf, raw=False)


def build_gauge_fix_factory(
    composite_model: nn.Module,
    leaf_idx: int,
    atom,
    private_global_idxs: Sequence[int],
    x_train: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    weight: float = 1.0,
    mode: str = "additive",
):
    """Build a ResidualsModule factory for gauge-fixing a single leaf.

    Parameters
    ----------
    composite_model : nn.Module
        The ASTCompositeAdaptor (shares parameters with the gauge wrapper).
    leaf_idx : int
        Index of the leaf to constrain (in DFS order).
    atom : AtomNode
        The AST atom corresponding to this leaf.
    private_global_idxs : sequence of int
        Global variable indices that are *private* to this leaf (not shared).
    x_train : Tensor [N, Nx]
        Full training input (for computing reference values and building the
        gauge-fix dataloader).
    device, dtype
        Compute device and precision.
    weight : float
        Penalty strength sqrt(λ).  The gauge penalty is λ·Σ leaf(x_ref)².
    mode : str
        ``"additive"`` penalises deviation from zero; ``"multiplicative"``
        penalises deviation from the (detached) batch mean.

    Returns
    -------
    factory : callable
        A factory ``factory(_) -> ResidualsModule`` suitable for appending to
        the optimizer's ``residual_module_factories`` list.
    """
    import nestynet.optimizer

    from nestynet.adaptors.adaptors import AutogradAdaptor
    from nestynet_sr.sr_core.bridges import get_input_exprs, is_trivial_input

    # --- Map private global indices to local leaf-input indices ---
    # The leaf's inputs come from eval_inputs(atom, x).  For simple atoms,
    # local index k corresponds to atom.var_idxs[k].  For compound atoms
    # with input_expr, we need to check which input expressions reference
    # each private global variable.
    all_inputs = get_input_exprs(atom)
    private_local = []
    for k, inp in enumerate(all_inputs):
        if is_trivial_input(inp):
            gidx = int(inp.var_idxs[0])
            if gidx in private_global_idxs:
                private_local.append(k)

    if not private_local:
        return None  # Can't gauge-fix: no trivial private inputs found

    # --- Compute reference values (median of each private variable) ---
    ref_vals = []
    for k in private_local:
        inp = all_inputs[k]
        gidx = int(inp.var_idxs[0])
        ref_vals.append(float(torch.median(x_train[:, gidx])))
    ref_tensor = torch.tensor(ref_vals, dtype=dtype, device=device)

    # --- Build leaf-input tensor from training data ---
    # We need to feed the leaf the same inputs it sees during normal forward.
    # For trivial inputs this is just x[:, atom.var_idxs].
    var_idxs = list(atom.var_idxs)
    x_leaf = x_train[:, var_idxs].to(device=device, dtype=dtype)

    # Build gauge-fix dataloader: x = leaf inputs, y = zeros (target)
    y_zeros = torch.zeros(x_leaf.shape[0], 1, dtype=dtype, device=device)
    gauge_ds = TensorDataset(x_leaf, y_zeros)
    gauge_dl = DataLoader(gauge_ds, batch_size=x_leaf.shape[0], shuffle=False)

    # --- Build the wrapper and adaptor ---
    wrapper = _GaugeFixWrapper(
        composite_model, leaf_idx, private_local, ref_tensor,
        weight=weight, mode=mode,
    )
    adaptor = AutogradAdaptor(wrapper)

    def factory(_):
        return nestynet.optimizer.ResidualsModule(
            providers=[adaptor],
            dataloader=gauge_dl,
            device=device,
        )

    # Attach the wrapper for post-training diagnostics
    factory._gauge_wrapper = wrapper
    factory._gauge_x_leaf = x_leaf

    return factory


@torch.no_grad()
def gauge_fix_metrics(gauge_factories, raw: bool = False):
    """Return RMS/peak diagnostics for gauge-fix factories."""
    metrics = []
    for fac in gauge_factories or []:
        wrapper = getattr(fac, "_gauge_wrapper", None)
        x_leaf = getattr(fac, "_gauge_x_leaf", None)
        if wrapper is None or x_leaf is None:
            continue
        out = wrapper._evaluate_leaf(x_leaf, raw=raw)
        metrics.append(
            {
                "leaf_idx": int(wrapper._leaf_idx),
                "mode": str(getattr(wrapper, "_mode", "additive")),
                "weight": float(getattr(wrapper, "_weight", 1.0)),
                "rms": float((out ** 2).mean().sqrt()),
                "peak": float(out.abs().max()),
                "private_local": list(getattr(wrapper, "_private_local_idxs", [])),
                "ref_values": [float(v) for v in wrapper._ref_values.tolist()],
            }
        )
    return metrics


@torch.no_grad()
def gauge_fix_diagnostic(gauge_factories, label=""):
    """Evaluate gauge-fix residuals and print a diagnostic summary.

    Call after training to check whether the gauge penalty was effective.

    Parameters
    ----------
    gauge_factories : list of callables
        The factories returned by ``_build_additive_gauge_fix_factories``.
        Each must have ``_gauge_wrapper`` and ``_gauge_x_leaf`` attributes
        (attached by ``build_gauge_fix_factory``).
    label : str
        Optional prefix for the log line.
    """
    if not gauge_factories:
        return
    for m in gauge_fix_metrics(gauge_factories, raw=False):
        tag = f"{label} " if label else ""
        print(
            f"[Gauge fix ({m['mode']})] {tag}leaf {m['leaf_idx']} post-train: "
            f"RMS={m['rms']:.3e}, peak={m['peak']:.3e} "
            f"(private_local={m['private_local']}, "
            f"ref={[f'{v:.4f}' for v in m['ref_values']]})"
        )
