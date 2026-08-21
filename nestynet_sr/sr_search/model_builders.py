# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Helper utilities to build segmented / dual-segmented leaves and composite adaptors.
"""

import nestynet
import torch

from nestynet_sr.sr_core.bridges import (
    AtomNode,
    Node,
    build_composite_from_ast,
    make_stage_a_nn_factory,
    update_ast_nn_kwargs,
)


class LeafBuilder(object):
    """
    Factory for building SegmentedAdaptor or DualSegmentedAdaptor leaves from
    tokens describing input variable groups.
    """

    def __init__(self, model_hp, device, dtype):
        self.hp = model_hp
        self.device = device
        self.dtype = dtype

    def _gseg(self, nout, nin, num_segments, seg_width=None):
        """
        Build a single SegmentedAdaptor wrapping a NestyNet model.
        """
        g_net = getattr(nestynet.nets, self.hp.model_base_name)(
            Nout_size=nout,
            Nx_size=nin,
            num_segments=num_segments,
            model_scale=self.hp.Gmodel_scale,
            dtype=self.dtype,
            device=self.device,
            seg_width=seg_width,
        ).to(self.device)

        # Back-compat shim: give the raw model a *self* handle
        if not hasattr(g_net, "model_attr_name"):
            g_net.model_attr_name = g_net.__class__.__name__
            object.__setattr__(g_net, g_net.model_attr_name, g_net)

        return nestynet.adaptors.SegmentedAdaptor(
            g_net,
            segments=torch.arange(num_segments, device=self.device),
            block_size_target=self.hp.block_size_target,
        ).to(self.device)

    def build_leaf(self, token, num_segments, dual_layer):
        """
        Build a single leaf (SegmentedAdaptor or DualSegmentedAdaptor).

        Parameters
        ----------
        token        : list[int] | tuple[int] | 1-D LongTensor
                       Column indices for this leaf.
        num_segments : int
        dual_layer   : bool
                       If True, build a DualSegmentedAdaptor.

        Returns
        -------
        leaf         : SegmentedAdaptor | DualSegmentedAdaptor
        n_params     : int
                       Number of trainable parameters inside the leaf.
        """
        Nx_in = len(token)

        # 1. Single-layer (also used when Nx_in == 1)
        # if (not dual_layer) or (Nx_in == 1):
        if not dual_layer:
            leaf = self._gseg(self.hp.Nout_size, Nx_in, num_segments)
            # Make CompositeSegmentedAdaptor happy
            object.__setattr__(leaf, "base_model", leaf.base_model)
            # Public aliases for the linear–algebra helpers
            for _name in ("jacobian", "grad", "grad_grad", "jvp", "vjp", "diag", "dense"):
                if not hasattr(leaf, _name) and hasattr(leaf, "_" + _name):
                    object.__setattr__(leaf, _name, getattr(leaf, "_" + _name))
            n_params = leaf.num_parameters()
            return leaf, n_params

        # 2. Dual-layer
        Nmid = Nx_in + 2
        seg1 = self._gseg(Nmid, Nx_in, num_segments, seg_width=1)
        seg2 = self._gseg(self.hp.Nout_size, Nmid, num_segments)
        leaf = nestynet.adaptors.DualSegmentedAdaptor(seg1, seg2)
        # Public aliases for the linear–algebra helpers
        for _name in ("jacobian", "grad", "grad_grad", "jvp", "vjp", "diag", "dense"):
            if not hasattr(leaf, _name) and hasattr(leaf, "_" + _name):
                object.__setattr__(leaf, _name, getattr(leaf, "_" + _name))
        n_params = seg1.num_parameters() + seg2.num_parameters()
        return leaf, n_params


def is_minimal_ast(root: Node) -> bool:
    """
    AST equivalent of is_minimal_expression.
    Detects if AST is just a single NN atom on variable 0: AtomNode('nn', (0,))
    """
    return isinstance(root, AtomNode) and root.kind.lower() == "nn" and root.var_idxs == (0,)


# ──────────────────────────────────────────────────────────────
# AST-based model building (preferred API for Stage A)
# ──────────────────────────────────────────────────────────────


def build_composite_ast(
    root: Node,
    num_segments: int,
    dual_layer: bool,
    leaf_builder: "LeafBuilder",
    device: torch.device,
    dtype: torch.dtype,
    reuse_leaves: dict = None,
    freeze_non_nn: bool = False,
):
    """
    Build a ASTCompositeAdaptor from AST for Stage A.

    This is the preferred API for building models in Stage A. It uses the
    unified AST representation and automatically handles NN leaf construction.

    Parameters
    ----------
    root : Node
        AST root node (AtomNode, AddNode, or MulNode).
    num_segments : int | None
        Number of segments for NN leaves. If None, preserve existing kwargs
        in the AST (useful when loading from checkpoint).
    dual_layer : bool | None
        Whether to use dual-layer architecture for NN leaves. If None,
        preserve existing kwargs in the AST.
    leaf_builder : LeafBuilder
        Factory for building segmented adaptors.
    device : torch.device
        Device to place model on.
    dtype : torch.dtype
        Data type for model parameters.
    reuse_leaves : dict, optional
        Mapping from tag -> trained leaf module for reuse. Atoms with tags
        in this dict will preserve their original num_segments/dual_layer.

    Returns
    -------
    model : ASTCompositeAdaptor
        Compiled composite model.
    nparam : int
        Total number of trainable parameters.
    updated_ast : Node
        The AST with updated kwargs. This should be saved in checkpoints
        to ensure correct model reconstruction.

    Example
    -------
    >>> from sr_core import build_initial_ast
    >>> ast = build_initial_ast(Nxvars=3, num_segments=32, dual_layer=False)
    >>> model, nparam, updated_ast = build_composite_ast(
    ...     ast, num_segments=32, dual_layer=False,
    ...     leaf_builder=builder, device=device, dtype=dtype
    ... )
    """
    # Update NN atoms with current num_segments/dual_layer
    # Skip atoms whose tags are in reuse_leaves - they should preserve their original kwargs
    skip_tags = set(reuse_leaves.keys()) if reuse_leaves else None
    root_updated = update_ast_nn_kwargs(root, num_segments, dual_layer, skip_tags=skip_tags)

    # Build Stage A nn_factory
    nn_factory = make_stage_a_nn_factory(leaf_builder)

    # Compile AST to ASTCompositeAdaptor
    model = build_composite_from_ast(
        root_updated, dtype=dtype, device=device, nn_factory=nn_factory, reuse=reuse_leaves
    )

    if freeze_non_nn:
        from nestynet_sr.sr_search.training import freeze_non_nn_leaves

        freeze_non_nn_leaves(model, root_updated)

    nparam = model.num_parameters()
    return model, nparam, root_updated
