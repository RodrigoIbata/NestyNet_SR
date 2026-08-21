# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
AST-native CompositeAdaptor for symbolic regression.

This adaptor uses Abstract Syntax Trees (AST) natively instead of converting
to/from postfix notation, making it more natural for symbolic regression tasks.
"""

import math
from typing import List, Sequence, Tuple

import torch

# Import nestynet components
try:
    from nestynet.adaptors.adaptors import AutogradAdaptor, LAProvider, SegmentedAdaptor
    from nestynet.adaptors.stacking_adaptors import DualSegmentedAdaptor
except ImportError:
    import os
    import sys

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from nestynet.adaptors.adaptors import AutogradAdaptor, LAProvider, SegmentedAdaptor
    from nestynet.adaptors.stacking_adaptors import DualSegmentedAdaptor

# Import AST node types
try:
    from nestynet_sr.sr_core.bridges import (
        AbsNode,
        AcosNode,
        AddNode,
        ArgNode,
        AsinNode,
        AtanNode,
        AtomNode,
        ConjNode,
        ConstNode,
        CosNode,
        ExpNode,
        ImagNode,
        LogNode,
        MulNode,
        Node,
        PowNode,
        RealNode,
        SinNode,
        const_full_like,
        eval_inputs,
    )
    from nestynet_sr.sr_core.fit_links import (
        canonical_fit_link_name,
        fit_link_torch,
        fit_link_torch_d1,
    )
except ImportError:
    import os
    import sys

    sr_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if sr_dir not in sys.path:
        sys.path.insert(0, sr_dir)
    from nestynet_sr.sr_core.bridges import (
        AbsNode,
        AcosNode,
        AddNode,
        ArgNode,
        AsinNode,
        AtanNode,
        AtomNode,
        ConjNode,
        ConstNode,
        CosNode,
        ExpNode,
        ImagNode,
        LogNode,
        MulNode,
        Node,
        PowNode,
        RealNode,
        SinNode,
        const_full_like,
        eval_inputs,
    )
    from nestynet_sr.sr_core.fit_links import (
        canonical_fit_link_name,
        fit_link_torch,
        fit_link_torch_d1,
    )


def _safe_grad(g, x, expected_K, O=1):
    """Safely handle None gradient from allow_unused=True.

    When torch.autograd.grad returns None for an input with no gradient path,
    replace it with a zeros tensor of the expected shape.

    Parameters
    ----------
    g : Tensor or None
        Gradient tensor from leaf.grad() or None if unused
    x : Tensor
        Input tensor, used to get batch size, device, and dtype
    expected_K : int
        Expected number of columns (input dimensions for this leaf)
    O : int
        Number of output dimensions

    Returns
    -------
    Tensor of shape (B, O, expected_K)
    """
    if g is None:
        B = x.size(0)
        return torch.zeros(B, O, expected_K, device=x.device, dtype=x.dtype)
    if g.ndim == 2:
        g = g.unsqueeze(1)
    return g


def _safe_hess(gg, x, expected_K, O=1):
    """Safely handle None Hessian from allow_unused=True.

    Parameters
    ----------
    gg : Tensor or None
        Hessian tensor from leaf.grad_grad() or None if unused
    x : Tensor
        Input tensor, used to get batch size, device, and dtype
    expected_K : int
        Expected number of columns (input dimensions for this leaf)
    O : int
        Number of output dimensions

    Returns
    -------
    Tensor of shape (B, O, expected_K, expected_K)
    """
    if gg is None:
        B = x.size(0)
        return torch.zeros(B, O, expected_K, expected_K, device=x.device, dtype=x.dtype)
    if gg.ndim == 2:
        gg = gg.unsqueeze(-1).unsqueeze(-1)
    elif gg.ndim == 3:
        gg = gg.unsqueeze(1)
    return gg


class ASTCompositeAdaptor(torch.nn.Module, LAProvider):
    """
    AST-native adaptor that composes several SegmentedAdaptor/AutogradAdaptor
    instances according to an Abstract Syntax Tree.

    This is the preferred adaptor for symbolic regression, providing cleaner
    semantics and easier manipulation than postfix/prefix notation.

    Parameters
    ----------
    ast_root : Node
        Root of the AST (AtomNode, AddNode, or MulNode).
    leaves : Sequence[torch.nn.Module]
        Leaf adaptors (SegmentedAdaptor, DualSegmentedAdaptor, or AutogradAdaptor).
        Must be in left-to-right depth-first order matching the AST.

    Note
    ----
    Building an explicit Gauss-Newton matrix for a composite expression
    needs O(P**2) forward passes. Therefore run the optimizer with
    LM_strategy='matfree' for best performance.

    Example
    -------
    >>> from nestynet_sr.sr_core.bridges import AtomNode, MulNode
    >>> ast = MulNode(AtomNode('nn', (0,)), AtomNode('nn', (1,)))
    >>> leaves = [seg_adaptor_x0, seg_adaptor_x1]
    >>> model = ASTCompositeAdaptor(ast, leaves)
    """

    def __init__(self, ast_root: Node, leaves: Sequence[torch.nn.Module]):
        super().__init__()

        # Store AST root
        self.ast_root = ast_root
        self.skip_jac_sanity = True

        # Type checking for leaves
        #
        # By default we accept the standard nestynet adaptor types. For native DE/PDE
        # discovery we also allow lightweight "feature" leaves via duck-typing
        # (e.g. leaves that expose u(x), u_x, u_xx with no trainable parameters).
        def _is_leaf(obj) -> bool:
            if isinstance(obj, (SegmentedAdaptor, DualSegmentedAdaptor, AutogradAdaptor)):
                return True
            # blocks() is required by the LM optimiser interface (parameter blocking).
            req = ("num_parameters", "build_cache", "jvp", "vjp", "blocks")
            return all(hasattr(obj, name) for name in req)

        bad = [i for i, l in enumerate(leaves) if not _is_leaf(l)]
        if bad:
            idx_str = ", ".join(map(str, bad))
            raise TypeError(
                f"ASTCompositeAdaptor: leaves must be standard nestynet adaptors "
                f"(SegmentedAdaptor/DualSegmentedAdaptor/AutogradAdaptor) or implement "
                f"the minimal LAProvider leaf interface (num_parameters/build_cache/jvp/vjp). "
                f"Bad positions [{idx_str}] are {', '.join(type(leaves[i]).__name__ for i in bad)}"
            )

        # Check that all leaves output exactly one scalar per sample
        def _nout1(leaf):
            base = getattr(leaf, "base_model", getattr(leaf, "model", None))
            return getattr(base, "Nout_size", 1)

        multi = [i for i, l in enumerate(leaves) if _nout1(l) != 1]
        if multi:
            idx = ", ".join(map(str, multi))
            outs = ", ".join(str(_nout1(leaves[i])) for i in multi)
            raise ValueError(
                f"ASTCompositeAdaptor: leaves [{idx}] have "
                f"Nout_size {outs}; only scalar-output leaves are supported."
            )

        self.leaf = torch.nn.ModuleList(leaves)

        # Validate that AST leaf count matches provided leaves
        n_atoms = self._count_atoms(ast_root)
        if n_atoms != len(leaves):
            raise ValueError(
                f"AST contains {n_atoms} atoms but {len(leaves)} leaves provided. "
                f"Leaves must be in left-to-right depth-first order."
            )

        # Flat parameter slices
        off = 0
        self._slice = []
        for m in self.leaf:
            n = m.num_parameters()
            self._slice.append(slice(off, off + n))
            off += n
        self._total_params = off

        # Extract global constants from first leaf
        self._device = self._infer_device()
        self.O = getattr(self.leaf[0], "O", 1)  # outputs per sample

        # Compute Nx (max input dimension across all atoms)
        self.Nx = self._compute_max_input_dim(ast_root)
        # The AST records only referenced coordinates, so it cannot by itself
        # distinguish a full mapping (x0, x1) from a prefix slice of wider data
        # (x0, x1, x2).  Training declares the actual data width before LM;
        # identity transparency stays disabled until that declaration exists.
        self._global_input_dim = None

        # Evidence compatibility path: when this composite has exactly one
        # segmented leaf and every other leaf is zero-parameter structure
        # (e.g. a raw variable prefactor), expose that segmented base model at
        # the top level so evidence-mode plumbing can reuse the normal
        # segmented-provider interface.  Do not do this for composites with
        # trainable analytic leaves, because their parameter vector is not the
        # segmented base-model vector that evidence priors index.
        self.base_model = None
        self.segments = None
        segmented_leaf = None
        segmented_base = None
        n_segmented_leaves = 0
        for leaf in self.leaf:
            base = getattr(leaf, "base_model", None)
            if base is not None and hasattr(base, "num_segments"):
                n_segmented_leaves += 1
                segmented_leaf = leaf
                segmented_base = base
        if n_segmented_leaves == 1 and segmented_leaf is not None:
            try:
                segmented_params = int(segmented_leaf.num_parameters())
            except Exception:
                segmented_params = -1
            if segmented_params == int(self._total_params):
                self.base_model = segmented_base
                self.segments = getattr(segmented_leaf, "segments", None)

        # Fit-only output link (numeric conditioning).
        # When enabled, the optimiser sees residuals in fit-space:
        #     r = t(y) - t(f)
        # but all SR analysis continues to operate in y-space.
        self.fit_y_link = None
        self.fit_y_link_scale = 1.0

    def _transparent_identity_leaf(self):
        """Return the sole leaf when this AST is an exact identity wrapper.

        A one-atom AST over every input coordinate in native order has no
        composition to perform.  Without a fit-only output link, changing the
        leaf's optimiser contract is therefore incorrect: parameter names,
        blocks, residual operators, and optional refinements must remain those
        of the leaf itself.

        Input slicing/reordering, compound inputs, feature leaves, and linked
        residuals deliberately retain the normal AST chain-rule path.
        """
        root = self.__dict__.get("ast_root", None)
        leaves = self._modules.get("leaf", None)
        if not isinstance(root, AtomNode) or leaves is None or len(leaves) != 1:
            return None
        if canonical_fit_link_name(self.__dict__.get("fit_y_link", None)) is not None:
            return None
        leaf = leaves[0]
        if not isinstance(leaf, (SegmentedAdaptor, DualSegmentedAdaptor, AutogradAdaptor)):
            return None
        global_input_dim = self.__dict__.get("_global_input_dim", None)
        if global_input_dim is None:
            return None
        base = getattr(leaf, "base_model", None)
        native_input_dim = getattr(base, "Nx_size", None)
        if native_input_dim is None:
            model = getattr(leaf, "model", None)
            native_input_dim = getattr(model, "Nx_size", getattr(model, "Nx", None))
        try:
            native_input_dim = int(native_input_dim)
        except (TypeError, ValueError):
            return None
        if native_input_dim != int(global_input_dim):
            return None
        simple_var_idxs = root.simple_var_idxs()
        if simple_var_idxs is None:
            return None
        if tuple(simple_var_idxs) != tuple(range(int(global_input_dim))):
            return None
        required = (
            "build_cache",
            "residuals",
            "residuals_lm",
            "jvp",
            "vjp",
            "jacobian",
            "diag",
            "dense",
            "blocks",
            "pre_block",
        )
        if not all(callable(getattr(leaf, name, None)) for name in required):
            return None
        return leaf

    def declare_global_input_dim(self, input_dim: int) -> None:
        """Declare the full data width used to interpret AST variable indices."""
        input_dim = int(input_dim)
        if input_dim <= 0:
            raise ValueError(f"global input dimension must be positive, got {input_dim}")
        previous = self.__dict__.get("_global_input_dim", None)
        if previous is not None and int(previous) != input_dim:
            raise ValueError(
                "ASTCompositeAdaptor cannot be reused with conflicting global input "
                f"dimensions ({previous} and {input_dim})"
            )
        self._global_input_dim = input_dim

    def __getattr__(self, name):
        try:
            return torch.nn.Module.__getattr__(self, name)
        except AttributeError as exc:
            leaf = self._transparent_identity_leaf()
            if leaf is not None:
                try:
                    return getattr(leaf, name)
                except AttributeError:
                    pass
            raise exc

    @staticmethod
    def _is_fractional_exponent(exponent: float) -> bool:
        """Return True when exponent is non-integer in real arithmetic."""
        try:
            c = float(exponent)
        except Exception:
            return False
        if not math.isfinite(c):
            return False
        return abs(c - round(c)) > 1e-12

    @classmethod
    def _pow_safe_base(cls, base_val: torch.Tensor, exponent: float) -> torch.Tensor:
        """Mirror forward-domain safety for numerically fragile powers.

        For real arithmetic we guard:
          * fractional powers (domain: base>0 for the real branch)
          * negative powers (domain: base!=0)

        The guards are intentionally tiny (eps≈1e-12) and primarily exist to
        prevent NaNs/Infs during LM perturbations.
        """
        if base_val.is_complex():
            return base_val

        eps = 1e-12

        # Fractional exponents: clamp to positive to stay on the real branch.
        if cls._is_fractional_exponent(exponent):
            return base_val.clamp(min=eps)

        # Negative integer powers: keep away from zero while preserving sign.
        try:
            c = float(exponent)
        except Exception:
            return base_val
        if math.isfinite(c) and c < 0.0 and abs(c - round(c)) <= 1e-12:
            sgn = torch.where(base_val >= 0, torch.ones_like(base_val), -torch.ones_like(base_val))
            return sgn * base_val.abs().clamp(min=eps)

        return base_val

    @staticmethod
    def _asin_acos_safe_arg(arg_val: torch.Tensor) -> torch.Tensor:
        if arg_val.is_complex():
            return arg_val
        return torch.clamp(arg_val, min=-1.0 + 1.0e-12, max=1.0 - 1.0e-12)

    @classmethod
    def _inv_trig_value(cls, node: Node, arg_val: torch.Tensor) -> torch.Tensor:
        if isinstance(node, AsinNode):
            return torch.asin(cls._asin_acos_safe_arg(arg_val))
        if isinstance(node, AcosNode):
            return torch.acos(cls._asin_acos_safe_arg(arg_val))
        if isinstance(node, AtanNode):
            return torch.atan(arg_val)
        raise TypeError(f"Unsupported inverse trig node: {type(node)}")

    @classmethod
    def _inv_trig_d1(cls, node: Node, arg_val: torch.Tensor) -> torch.Tensor:
        if isinstance(node, (AsinNode, AcosNode)):
            u = cls._asin_acos_safe_arg(arg_val)
            denom = torch.clamp(1.0 - u * u, min=1.0e-24)
            d1 = torch.rsqrt(denom)
            return -d1 if isinstance(node, AcosNode) else d1
        if isinstance(node, AtanNode):
            return 1.0 / (1.0 + arg_val * arg_val)
        raise TypeError(f"Unsupported inverse trig node: {type(node)}")

    @classmethod
    def _inv_trig_d2(cls, node: Node, arg_val: torch.Tensor) -> torch.Tensor:
        if isinstance(node, (AsinNode, AcosNode)):
            u = cls._asin_acos_safe_arg(arg_val)
            denom = torch.clamp(1.0 - u * u, min=1.0e-24)
            d2 = u * torch.pow(denom, -1.5)
            return -d2 if isinstance(node, AcosNode) else d2
        if isinstance(node, AtanNode):
            denom = 1.0 + arg_val * arg_val
            return -2.0 * arg_val / (denom * denom)
        raise TypeError(f"Unsupported inverse trig node: {type(node)}")

    def _infer_device(self):
        """
        Infer device from leaves, handling cases where leaves may have no parameters.
        """
        for leaf in self.leaf:
            # Try to get device from wrapped model first
            base_model = getattr(leaf, "base_model", None) or getattr(leaf, "model", None)
            if base_model is not None:
                # Try parameters
                try:
                    return next(base_model.parameters()).device
                except StopIteration:
                    pass
                # Try buffers
                try:
                    return next(base_model.buffers()).device
                except StopIteration:
                    pass

            # Try direct leaf parameters/buffers
            try:
                return next(leaf.parameters()).device
            except StopIteration:
                pass
            try:
                return next(leaf.buffers()).device
            except StopIteration:
                pass

        # Default to CPU if no device found
        return torch.device("cpu")

    def _count_atoms(self, node: Node) -> int:
        """Count AtomNodes in the AST (left-to-right depth-first)."""
        if isinstance(node, AtomNode):
            return 1
        elif isinstance(node, (AddNode, MulNode)):
            return self._count_atoms(node.left) + self._count_atoms(node.right)
        elif isinstance(node, PowNode):
            return self._count_atoms(node.base)
        elif isinstance(node, (LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode, ConjNode, RealNode, ImagNode, AbsNode, ArgNode)):
            return self._count_atoms(node.arg)
        elif isinstance(node, ConstNode):
            return 0
        else:
            raise TypeError(f"Unknown node type: {type(node)}")

    def _compute_max_input_dim(self, node: Node) -> int:
        """Compute maximum input dimension across all atoms."""
        if isinstance(node, AtomNode):
            return max(node.var_idxs) + 1 if node.var_idxs else 0
        elif isinstance(node, (AddNode, MulNode)):
            left_dim = self._compute_max_input_dim(node.left)
            right_dim = self._compute_max_input_dim(node.right)
            return max(left_dim, right_dim)
        elif isinstance(node, PowNode):
            return self._compute_max_input_dim(node.base)
        elif isinstance(node, (LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode, ConjNode, RealNode, ImagNode, AbsNode, ArgNode)):
            return self._compute_max_input_dim(node.arg)
        elif isinstance(node, ConstNode):
            return 0
        else:
            raise TypeError(f"Unknown node type: {type(node)}")

    # ──────────────────────────────────────────────────────────────
    # Compatibility helpers
    # ──────────────────────────────────────────────────────────────

    @property
    def expressions(self):
        """
        Back-compat: return postfix representation of AST.
        Note: This is a computed property, not stored.
        """
        # Convert AST to postfix for legacy code
        return self._ast_to_postfix(self.ast_root)

    def _ast_to_postfix(self, node: Node) -> List:
        """Convert AST to postfix list notation."""
        if isinstance(node, AtomNode):
            return [list(node.var_idxs)]
        elif isinstance(node, AddNode):
            return self._ast_to_postfix(node.left) + self._ast_to_postfix(node.right) + [torch.add]
        elif isinstance(node, MulNode):
            return (
                self._ast_to_postfix(node.left)
                + self._ast_to_postfix(node.right)
                + [torch.multiply]
            )
        elif isinstance(node, (PowNode, LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode)):
            raise NotImplementedError(
                "PowNode/LogNode/ExpNode/SinNode/CosNode/inverse-trig not supported in postfix notation"
            )
        else:
            raise TypeError(f"Unknown node type: {type(node)}")

    @property
    def module_list(self):
        """
        Back-compat: return unwrapped leaf models.
        """
        mods = []
        for leaf in self.leaf:
            base = getattr(leaf, "base_model", None)
            if base is not None:
                mods.append(base)
            else:
                mods.append(getattr(leaf, "model", leaf))
        return torch.nn.ModuleList(mods)

    @property
    def n_params(self):
        return self.num_parameters()

    @property
    def n_outputs(self):
        return int(self.O)

    def base_models(self):
        """Return the exposed top-level base model when one exists."""
        if self.base_model is None:
            return []
        return [self.base_model]

    def _compose_from_leaf_outputs(self, leaf_outputs, *, ref_x=None):
        """Rebuild the composite output from cached per-leaf forward values."""
        values = list(leaf_outputs)

        def eval_node(node: Node, leaf_idx: int):
            if isinstance(node, AtomNode):
                return values[leaf_idx], leaf_idx + 1
            elif isinstance(node, AddNode):
                left_val, idx = eval_node(node.left, leaf_idx)
                right_val, idx = eval_node(node.right, idx)
                return left_val + right_val, idx
            elif isinstance(node, MulNode):
                left_val, idx = eval_node(node.left, leaf_idx)
                right_val, idx = eval_node(node.right, idx)
                return left_val * right_val, idx
            elif isinstance(node, PowNode):
                base_val, idx = eval_node(node.base, leaf_idx)
                base_safe = self._pow_safe_base(base_val, node.exponent)
                return base_safe.pow(node.exponent), idx
            elif isinstance(node, LogNode):
                arg_val, idx = eval_node(node.arg, leaf_idx)
                if not arg_val.is_complex():
                    arg_val = torch.clamp(arg_val, min=1e-12)
                return torch.log(arg_val), idx
            elif isinstance(node, ExpNode):
                arg_val, idx = eval_node(node.arg, leaf_idx)
                return torch.exp(arg_val), idx
            elif isinstance(node, SinNode):
                arg_val, idx = eval_node(node.arg, leaf_idx)
                return torch.sin(arg_val), idx
            elif isinstance(node, CosNode):
                arg_val, idx = eval_node(node.arg, leaf_idx)
                return torch.cos(arg_val), idx
            elif isinstance(node, (AsinNode, AcosNode, AtanNode)):
                arg_val, idx = eval_node(node.arg, leaf_idx)
                return self._inv_trig_value(node, arg_val), idx
            elif isinstance(node, ConjNode):
                arg_val, idx = eval_node(node.arg, leaf_idx)
                return torch.conj(arg_val), idx
            elif isinstance(node, RealNode):
                arg_val, idx = eval_node(node.arg, leaf_idx)
                return torch.real(arg_val), idx
            elif isinstance(node, ImagNode):
                arg_val, idx = eval_node(node.arg, leaf_idx)
                return torch.imag(arg_val), idx
            elif isinstance(node, AbsNode):
                arg_val, idx = eval_node(node.arg, leaf_idx)
                return torch.abs(arg_val), idx
            elif isinstance(node, ArgNode):
                arg_val, idx = eval_node(node.arg, leaf_idx)
                return torch.angle(arg_val), idx
            elif isinstance(node, ConstNode):
                ref = values[0] if values else ref_x
                if ref is None:
                    raise ValueError("Const-only AST cache rebuild requires a reference tensor.")
                B = ref.shape[0]
                return const_full_like(ref, (B, self.O), node.value), leaf_idx
            else:
                raise TypeError(f"Unknown node type: {type(node)}")

        result, _ = eval_node(self.ast_root, 0)
        return result

    def _cached_forward_value(self, cache, x, *, track_grad: bool):
        """Return a segment-aware forward value from cache when available."""
        if cache is not None:
            if (not track_grad) and ("f" in cache) and (cache["f"] is not None):
                return cache["f"]
            leaf_caches = cache.get("leaves", None)
            if leaf_caches is not None:
                f = self._compose_from_leaf_outputs(
                    [leaf_cache["f"] for leaf_cache in leaf_caches],
                    ref_x=x,
                )
                return f if track_grad else f.detach()

        f = self.forward(x)
        return f if track_grad else f.detach()

    def _split_dir(self, v):
        if v is None:
            return [None] * len(self.leaf)
        if v.ndim == 1:
            return [v[s] for s in self._slice]
        if v.ndim == 2:
            return [v[..., s] for s in self._slice]
        raise ValueError(f"_split_dir: expected v.ndim in {{1,2}}, got shape={tuple(v.shape)}")

    def _fit_link_cfg(self):
        """Return (name, scale) for the fit-only output link."""
        name = canonical_fit_link_name(getattr(self, "fit_y_link", None))
        scale = float(getattr(self, "fit_y_link_scale", 1.0))
        return name, scale

    def _normalize_sigma_like(self, sigma, ref):
        """Broadcast observational sigma to the target shape."""
        if sigma is None:
            return None

        rdtype = ref.real.dtype if ref.is_complex() else ref.dtype
        sig = torch.as_tensor(sigma, device=ref.device, dtype=rdtype)

        if sig.ndim == 0:
            sig = sig.expand_as(ref)
        else:
            if ref.ndim >= 2 and sig.ndim == 1:
                sig = sig.unsqueeze(-1)
            if sig.ndim > ref.ndim:
                reduce_dims = tuple(range(ref.ndim, sig.ndim))
                sig = torch.sqrt((sig * sig).mean(dim=reduce_dims))
            while sig.ndim < ref.ndim:
                sig = sig.unsqueeze(-1)
            try:
                sig = sig.expand(ref.shape)
            except RuntimeError as exc:
                raise ValueError(
                    f"Could not broadcast y_sigma shape {tuple(sig.shape)} to "
                    f"target shape {tuple(ref.shape)}."
                ) from exc

        eps = torch.finfo(sig.dtype).eps
        return sig.clamp_min(eps).contiguous()

    def _target_sigma_from_observations(self, y, y_sigma):
        """Return the effective residual-space sigma for the current fit-link."""
        sigma = self._normalize_sigma_like(y_sigma, y)
        if sigma is None:
            return None

        link, scale = self._fit_link_cfg()
        if link is None:
            return sigma

        y_ref = y.detach() if torch.is_tensor(y) and y.requires_grad else y
        tprime = fit_link_torch_d1(y_ref, link, scale).abs().to(
            device=sigma.device, dtype=sigma.dtype
        )
        eps = torch.finfo(sigma.dtype).eps
        return (tprime * sigma).clamp_min(eps).contiguous()

    def _sample_weights_from_sigma(self, target_sigma):
        """Return 1/sigma^2 row weights or None when sigma is absent."""
        if target_sigma is None:
            return None
        return (1.0 / (target_sigma * target_sigma)).contiguous()

    def _trace_sample_weights(self, cache, out_dim=None, *, device=None, dtype=None):
        """Broadcast cache sample weights to the Jacobian layout."""
        w = cache.get("sample_weights", None)
        if w is None:
            return None

        x = cache["x"]
        B = int(x.shape[0])
        O = int(cache["O"])
        device = x.device if device is None else device
        dtype = x.dtype if dtype is None else dtype

        w = torch.as_tensor(w, device=device, dtype=dtype)
        if out_dim is None:
            if w.ndim == 1 and w.numel() == B:
                return w.view(B, 1).expand(B, O)
            if w.ndim == 2 and w.size(0) == B and w.size(1) == 1:
                return w.expand(B, O)
            if w.ndim == 2 and w.size(0) == B and w.size(1) == O:
                return w
        else:
            if w.ndim == 1 and w.numel() == B:
                return w
            if w.ndim == 2 and w.size(0) == B and w.size(1) == 1:
                return w[:, 0]
            if w.ndim == 2 and w.size(0) == B and w.size(1) == O:
                return w[:, int(out_dim)]

        raise RuntimeError(
            f"Unexpected sample_weights shape {tuple(w.shape)} for B={B}, O={O}, out_dim={out_dim}"
        )

    def _apply_sample_weights_to_jacobian(self, J, cache, *, out_dim=None):
        """Scale Jacobian rows by sqrt(sample_weights) when present."""
        rdtype = J.real.dtype if J.is_complex() else J.dtype
        w = self._trace_sample_weights(cache, out_dim=out_dim, device=J.device, dtype=rdtype)
        if w is None:
            return J

        sqrt_w = torch.sqrt(w).to(device=J.device, dtype=J.dtype)
        if out_dim is None:
            return J * sqrt_w.unsqueeze(-1)
        return J * sqrt_w.unsqueeze(-1)

    # ──────────────────────────────────────────────────────────────
    # Core forward computation
    # ──────────────────────────────────────────────────────────────

    def forward(self, x):
        """
        Evaluate the AST on input x.

        Parameters
        ----------
        x : torch.Tensor, shape (B, Nx)
            Input data.

        Returns
        -------
        torch.Tensor, shape (B, O)
            Output values (O=1 for scalar output).
        """

        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            return transparent_leaf(x)

        def eval_node(node: Node, leaf_idx: int) -> Tuple[torch.Tensor, int]:
            if isinstance(node, AtomNode):
                kind = str(getattr(node, "kind", "")).lower()
                feature_kinds = {
                    "u",
                    "field",
                    "state",
                    "du",
                    "d1u",
                    "grad_u",
                    "d2u",
                    "ddu",
                    "hess_u",
                }

                # Use unified eval_inputs for compound and simple atoms
                if kind in feature_kinds:
                    # Feature atoms use the full x
                    x_in = x
                elif node.n_in > 0:
                    # Use eval_inputs for both simple and compound atoms
                    x_in, _, _ = eval_inputs(node, x, need_grad=False, need_hess=False)
                else:
                    # Fallback for atoms with no inputs (shouldn't happen normally)
                    x_in = x[:, node.var_idxs] if node.var_idxs else x

                result = self.leaf[leaf_idx](x_in)
                return result, leaf_idx + 1
            elif isinstance(node, AddNode):
                left_val, idx = eval_node(node.left, leaf_idx)
                right_val, idx = eval_node(node.right, idx)
                return left_val + right_val, idx
            elif isinstance(node, MulNode):
                left_val, idx = eval_node(node.left, leaf_idx)
                right_val, idx = eval_node(node.right, idx)
                return left_val * right_val, idx
            elif isinstance(node, PowNode):
                base_val, idx = eval_node(node.base, leaf_idx)
                base_safe = self._pow_safe_base(base_val, node.exponent)
                return base_safe.pow(node.exponent), idx
            elif isinstance(node, LogNode):
                arg_val, idx = eval_node(node.arg, leaf_idx)
                if not arg_val.is_complex():
                    arg_val = torch.clamp(arg_val, min=1e-12)
                return torch.log(arg_val), idx
            elif isinstance(node, ExpNode):
                arg_val, idx = eval_node(node.arg, leaf_idx)
                return torch.exp(arg_val), idx
            elif isinstance(node, SinNode):
                arg_val, idx = eval_node(node.arg, leaf_idx)
                return torch.sin(arg_val), idx
            elif isinstance(node, CosNode):
                arg_val, idx = eval_node(node.arg, leaf_idx)
                return torch.cos(arg_val), idx
            elif isinstance(node, (AsinNode, AcosNode, AtanNode)):
                arg_val, idx = eval_node(node.arg, leaf_idx)
                return self._inv_trig_value(node, arg_val), idx
            elif isinstance(node, ConjNode):
                arg_val, idx = eval_node(node.arg, leaf_idx)
                return torch.conj(arg_val), idx
            elif isinstance(node, RealNode):
                arg_val, idx = eval_node(node.arg, leaf_idx)
                return torch.real(arg_val), idx
            elif isinstance(node, ImagNode):
                arg_val, idx = eval_node(node.arg, leaf_idx)
                return torch.imag(arg_val), idx
            elif isinstance(node, AbsNode):
                arg_val, idx = eval_node(node.arg, leaf_idx)
                return torch.abs(arg_val), idx
            elif isinstance(node, ArgNode):
                arg_val, idx = eval_node(node.arg, leaf_idx)
                return torch.angle(arg_val), idx
            elif isinstance(node, ConstNode):
                B = x.shape[0]
                v = const_full_like(x, (B, 1), node.value)
                return v, leaf_idx
            else:
                raise TypeError(f"Unknown node type: {type(node)}")

        result, _ = eval_node(self.ast_root, 0)
        return result

    # ──────────────────────────────────────────────────────────────
    # Residuals
    # ──────────────────────────────────────────────────────────────

    def residuals(self, cache, data=None, *, track_grad=False):
        """Compute residuals.

        Default: r = y - f(x)
        If fit_y_link is set: r = t(y) - t(f(x))
        """
        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            return transparent_leaf.residuals(cache, data=data, track_grad=track_grad)

        x, y, *_ = (cache["x"], cache["y"]) if data is None else data
        with torch.set_grad_enabled(track_grad):
            f = self._cached_forward_value(cache, x, track_grad=track_grad)
            link, scale = self._fit_link_cfg()
            if link is None:
                r = y - f
            else:
                r = fit_link_torch(y, link, scale) - fit_link_torch(f, link, scale)
        return r if track_grad else r.detach()

    def residuals_lm(self, _p, model_fn, data, *, track_grad=False):
        """Compute residuals with external model function (for LM trial steps).

        Default: r = y - f(x)
        If fit_y_link is set: r = t(y) - t(f(x))
        """
        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            return transparent_leaf.residuals_lm(
                _p,
                model_fn,
                data,
                track_grad=track_grad,
            )

        x, y, *_ = data
        with torch.set_grad_enabled(track_grad):
            f = model_fn(x)
            link, scale = self._fit_link_cfg()
            if link is None:
                r = y - f
            else:
                r = fit_link_torch(y, link, scale) - fit_link_torch(f, link, scale)
        return r if track_grad else r.detach()

    # ──────────────────────────────────────────────────────────────
    # Gradients and Hessians
    # ──────────────────────────────────────────────────────────────

    def grad(self, cache_or_x, out_dim=None):
        """First derivative w.r.t. inputs."""
        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            return transparent_leaf.grad(cache_or_x, out_dim=out_dim)
        if isinstance(cache_or_x, dict):
            x = cache_or_x["x"]
        else:
            x = cache_or_x
        _, g, _ = self._value_grad_grad(x, need_gg=False)
        if out_dim is not None:
            return g[:, out_dim]
        return g

    def grad_grad(self, cache_or_x, out_dim=None):
        """Second derivative w.r.t. inputs."""
        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            return transparent_leaf.grad_grad(cache_or_x, out_dim=out_dim)
        if isinstance(cache_or_x, dict):
            x = cache_or_x["x"]
        else:
            x = cache_or_x
        _, _, gg = self._value_grad_grad(x, need_gg=True)
        if out_dim is not None:
            return gg[:, out_dim]
        return gg

    def _value_grad_grad(self, x, *, need_gg):
        """
        Compute value, gradient, and optionally Hessian w.r.t. inputs.

        Returns
        -------
        tuple : (value, grad, grad_grad)
            - value: (B, O)
            - grad: (B, O, Nx)
            - grad_grad: (B, O, Nx, Nx) or None
        """
        # Use the runtime input dimension for gradients rather than the AST-inferred
        # maximum var index. This makes the adaptor robust for feature atoms
        # (e.g. u, u_x, u_xx) that depend on the full x even when var_idxs=().
        B, Nx, O = x.size(0), x.size(1), self.O

        def eval_node(node: Node, leaf_idx: int):
            if isinstance(node, AtomNode):
                leaf = self.leaf[leaf_idx]
                kind = str(getattr(node, "kind", "")).lower()
                cols = tuple(getattr(node, "var_idxs", ()))
                feature_kinds = {
                    "u",
                    "field",
                    "state",
                    "du",
                    "d1u",
                    "grad_u",
                    "d2u",
                    "ddu",
                    "hess_u",
                }

                # Feature atoms (u, u_x, u_xx) depend on the *full* x.
                if kind in feature_kinds:
                    x_sub = x
                    v_leaf = leaf(x_sub)
                    cache_like = {"x": x_sub}
                    if not hasattr(leaf, "grad"):
                        raise NotImplementedError(
                            f"Leaf type {type(leaf).__name__} does not implement grad(); "
                            f"cannot compute composite input gradients with feature atom kind={kind!r}."
                        )
                    try:
                        g = _safe_grad(
                            leaf.grad(cache_like, allow_unused=True), x_sub, Nx, O
                        )
                    except RuntimeError as e:
                        if "does not require grad" in str(e):
                            # Diagnostic: identify which leaf caused the failure
                            n_params = sum(p.numel() for p in leaf.parameters())
                            raise RuntimeError(
                                f"leaf.grad() failed for leaf type={type(leaf).__name__}, "
                                f"atom kind={kind}, var_idxs={cols}, num_params={n_params}: {e}"
                            ) from e
                        raise

                    gg = None
                    if need_gg and hasattr(leaf, "grad_grad"):
                        try:
                            gg = _safe_hess(
                                leaf.grad_grad(cache_like, allow_unused=True), x_sub, Nx, O
                            )
                        except RuntimeError as e:
                            if "does not require grad" in str(e):
                                # Diagnostic: identify which leaf caused the failure
                                n_params = sum(p.numel() for p in leaf.parameters())
                                raise RuntimeError(
                                    f"leaf.grad_grad() failed for leaf type={type(leaf).__name__}, "
                                    f"atom kind={kind}, var_idxs={cols}, num_params={n_params}: {e}"
                                ) from e
                            raise

                    return (v_leaf, g, gg), leaf_idx + 1

                # UNIFIED PATH: use eval_inputs() for all non-feature atoms
                # This handles both simple atoms (column selection) and compound
                # atoms (arbitrary input expressions) uniformly via chain rule.
                x_in, input_grad, input_hess = eval_inputs(
                    node, x, need_grad=True, need_hess=need_gg
                )
                # x_in: (B, n_in), input_grad: (B, n_in, Nx), input_hess: (B, n_in, Nx, Nx) or None

                n_in = x_in.size(1)
                v_leaf = leaf(x_in)  # [B, O]

                # Leaf derivatives w.r.t. its inputs
                cache_like = {"x": x_in}
                try:
                    dv_dinputs = _safe_grad(
                        leaf.grad(cache_like, allow_unused=True), x_in, n_in, O
                    )  # [B, O, n_in]
                except RuntimeError as e:
                    if "does not require grad" in str(e):
                        n_params = sum(p.numel() for p in leaf.parameters())
                        raise RuntimeError(
                            f"leaf.grad() failed: leaf type={type(leaf).__name__}, "
                            f"n_in={n_in}, num_params={n_params}: {e}"
                        ) from e
                    raise

                # Chain rule for first derivative:
                # g[b,o,x] = Σ_j dv_dinputs[b,o,j] * input_grad[b,j,x]
                g = torch.einsum('boj,bjx->box', dv_dinputs, input_grad)

                gg = None
                if need_gg:
                    # Leaf Hessian w.r.t. inputs (if available)
                    d2v_dinputs2 = None
                    if hasattr(leaf, "grad_grad"):
                        try:
                            d2v_dinputs2 = _safe_hess(
                                leaf.grad_grad(cache_like, allow_unused=True),
                                x_in, n_in, O
                            )  # [B, O, n_in, n_in]
                        except RuntimeError as e:
                            if "does not require grad" in str(e):
                                n_params = sum(p.numel() for p in leaf.parameters())
                                raise RuntimeError(
                                    f"leaf.grad_grad() failed: leaf type={type(leaf).__name__}, "
                                    f"n_in={n_in}, num_params={n_params}: {e}"
                                ) from e
                            raise

                    gg = torch.zeros(B, O, Nx, Nx, device=x.device, dtype=x.dtype)

                    # Term 1: leaf Hessian contracted with input grad outer products
                    # gg += Σ_jk d2v/dinputs_j dinputs_k * dinputs_j/dx * dinputs_k/dx
                    if d2v_dinputs2 is not None:
                        # outer[b,j,k,x,y] = input_grad[b,j,x] * input_grad[b,k,y]
                        outer = input_grad.unsqueeze(2).unsqueeze(-1) * input_grad.unsqueeze(1).unsqueeze(-2)
                        # [B, n_in, n_in, Nx, Nx]
                        gg = gg + torch.einsum('bojk,bjkxy->boxy', d2v_dinputs2, outer)

                    # Term 2: leaf grad contracted with input Hessian
                    # gg += Σ_j dv/dinputs_j * d2inputs_j/dx2
                    if input_hess is not None:
                        gg = gg + torch.einsum('boj,bjxy->boxy', dv_dinputs, input_hess)

                return (v_leaf, g, gg), leaf_idx + 1

            elif isinstance(node, AddNode):
                (v1, g1, gg1), idx = eval_node(node.left, leaf_idx)
                (v2, g2, gg2), idx = eval_node(node.right, idx)

                v = v1 + v2
                g = g1 + g2

                if need_gg:
                    if gg1 is None:
                        gg1 = g1.new_zeros(B, O, Nx, Nx)
                    if gg2 is None:
                        gg2 = g2.new_zeros(B, O, Nx, Nx)
                    gg = gg1 + gg2
                else:
                    gg = None

                return (v, g, gg), idx

            elif isinstance(node, MulNode):
                (v1, g1, gg1), idx = eval_node(node.left, leaf_idx)
                (v2, g2, gg2), idx = eval_node(node.right, idx)

                v = v1 * v2
                g = v2[..., None] * g1 + v1[..., None] * g2

                if need_gg:
                    if gg1 is None:
                        gg1 = g1.new_zeros(B, O, Nx, Nx)
                    if gg2 is None:
                        gg2 = g2.new_zeros(B, O, Nx, Nx)
                    outer = g1.unsqueeze(-1) * g2.unsqueeze(-2)
                    outer += g2.unsqueeze(-1) * g1.unsqueeze(-2)
                    gg = v2[..., None, None] * gg1 + v1[..., None, None] * gg2 + outer
                else:
                    gg = None

                return (v, g, gg), idx

            elif isinstance(node, PowNode):
                (v1, g1, gg1), idx = eval_node(node.base, leaf_idx)

                c = node.exponent
                v1_safe = self._pow_safe_base(v1, c)
                v = v1_safe.pow(c)
                # d/dx[f^c] = c*f^(c-1) * f'
                u1 = c * v1_safe.pow(c - 1)
                g = u1[..., None] * g1

                if need_gg:
                    if gg1 is None:
                        gg1 = g1.new_zeros(B, O, Nx, Nx)
                    # d²/dx²[f^c] = c*f^(c-1)*f'' + c*(c-1)*f^(c-2)*(f')²
                    term1 = u1[..., None, None] * gg1
                    outer = g1.unsqueeze(-1) * g1.unsqueeze(-2)
                    u2 = c * (c - 1) * v1_safe.pow(c - 2)
                    term2 = u2[..., None, None] * outer
                    gg = term1 + term2
                else:
                    gg = None

                return (v, g, gg), idx

            elif isinstance(node, LogNode):
                (v1, g1, gg1), idx = eval_node(node.arg, leaf_idx)

                v1_safe = v1
                if not v1_safe.is_complex():
                    v1_safe = torch.clamp(v1_safe, min=1e-12)

                v = torch.log(v1_safe)
                # d/dx[log(f)] = (1/f) * f'
                g = (1.0 / v1_safe)[..., None] * g1

                if need_gg:
                    if gg1 is None:
                        gg1 = g1.new_zeros(B, O, Nx, Nx)
                    # d²/dx²[log(f)] = (1/f)*f'' - (1/f²)*(f')²
                    term1 = (1.0 / v1_safe)[..., None, None] * gg1
                    outer = g1.unsqueeze(-1) * g1.unsqueeze(-2)
                    term2 = (1.0 / v1_safe.square())[..., None, None] * outer
                    gg = term1 - term2
                else:
                    gg = None

                return (v, g, gg), idx

            elif isinstance(node, ExpNode):
                (v1, g1, gg1), idx = eval_node(node.arg, leaf_idx)

                v = torch.exp(v1)
                # d/dx[exp(f)] = exp(f) * f'
                g = v[..., None] * g1

                if need_gg:
                    if gg1 is None:
                        gg1 = g1.new_zeros(B, O, Nx, Nx)
                    # d²/dx²[exp(f)] = exp(f) * (f'' + (f')²)
                    term1 = v[..., None, None] * gg1
                    outer = g1.unsqueeze(-1) * g1.unsqueeze(-2)
                    term2 = v[..., None, None] * outer
                    gg = term1 + term2
                else:
                    gg = None

                return (v, g, gg), idx

            elif isinstance(node, SinNode):
                (v1, g1, gg1), idx = eval_node(node.arg, leaf_idx)

                v = torch.sin(v1)
                # d/dx[sin(f)] = cos(f) * f'
                cos_v1 = torch.cos(v1)
                g = cos_v1[..., None] * g1

                if need_gg:
                    if gg1 is None:
                        gg1 = g1.new_zeros(B, O, Nx, Nx)
                    # d²/dx²[sin(f)] = cos(f)*f'' - sin(f)*(f')²
                    term1 = cos_v1[..., None, None] * gg1
                    outer = g1.unsqueeze(-1) * g1.unsqueeze(-2)
                    term2 = v[..., None, None] * outer  # v = sin(f)
                    gg = term1 - term2
                else:
                    gg = None

                return (v, g, gg), idx

            elif isinstance(node, CosNode):
                (v1, g1, gg1), idx = eval_node(node.arg, leaf_idx)

                v = torch.cos(v1)
                # d/dx[cos(f)] = -sin(f) * f'
                sin_v1 = torch.sin(v1)
                g = -sin_v1[..., None] * g1

                if need_gg:
                    if gg1 is None:
                        gg1 = g1.new_zeros(B, O, Nx, Nx)
                    # d²/dx²[cos(f)] = -sin(f)*f'' - cos(f)*(f')²
                    term1 = -sin_v1[..., None, None] * gg1
                    outer = g1.unsqueeze(-1) * g1.unsqueeze(-2)
                    term2 = v[..., None, None] * outer  # v = cos(f)
                    gg = term1 - term2
                else:
                    gg = None

                return (v, g, gg), idx

            elif isinstance(node, (AsinNode, AcosNode, AtanNode)):
                (v1, g1, gg1), idx = eval_node(node.arg, leaf_idx)

                v = self._inv_trig_value(node, v1)
                d1 = self._inv_trig_d1(node, v1)
                g = d1[..., None] * g1

                if need_gg:
                    if gg1 is None:
                        gg1 = g1.new_zeros(B, O, Nx, Nx)
                    d2 = self._inv_trig_d2(node, v1)
                    outer = g1.unsqueeze(-1) * g1.unsqueeze(-2)
                    gg = d1[..., None, None] * gg1 + d2[..., None, None] * outer
                else:
                    gg = None

                return (v, g, gg), idx

            elif isinstance(node, ConjNode):
                (v1, g1, gg1), idx = eval_node(node.arg, leaf_idx)
                v = torch.conj(v1)
                g = torch.conj(g1)
                gg = torch.conj(gg1) if (need_gg and gg1 is not None) else (v.new_zeros(B, O, Nx, Nx) if need_gg else None)
                return (v, g, gg), idx

            elif isinstance(node, RealNode):
                (v1, g1, gg1), idx = eval_node(node.arg, leaf_idx)
                v = torch.real(v1)
                g = torch.real(g1)
                gg = torch.real(gg1) if (need_gg and gg1 is not None) else (v.new_zeros(B, O, Nx, Nx) if need_gg else None)
                return (v, g, gg), idx

            elif isinstance(node, ImagNode):
                (v1, g1, gg1), idx = eval_node(node.arg, leaf_idx)
                v = torch.imag(v1)
                g = torch.imag(g1)
                gg = torch.imag(gg1) if (need_gg and gg1 is not None) else (v.new_zeros(B, O, Nx, Nx) if need_gg else None)
                return (v, g, gg), idx

            elif isinstance(node, AbsNode) or isinstance(node, ArgNode):
                (v1, g1, gg1), idx = eval_node(node.arg, leaf_idx)
                phi = torch.abs if isinstance(node, AbsNode) else torch.angle
                v = phi(v1)
                g = torch.stack([torch.func.jvp(phi, (v1,), (g1[..., i],))[1] for i in range(Nx)], dim=-1) if Nx > 0 else v.new_zeros(B, O, 0)
                if need_gg:
                    if gg1 is None:
                        gg1 = g1.new_zeros(B, O, Nx, Nx)
                    def g2(z, t):
                        return torch.func.jvp(phi, (z,), (t,))[1]
                    gg = v.new_zeros(B, O, Nx, Nx)
                    for i in range(Nx):
                        ti = g1[..., i]
                        for j in range(Nx):
                            gg[..., i, j] = torch.func.jvp(g2, (v1, ti), (g1[..., j], gg1[..., i, j]))[1]
                else:
                    gg = None
                return (v, g, gg), idx

            elif isinstance(node, ConstNode):
                # Constant node: value, zero gradient, zero Hessian
                v = const_full_like(x, (B, O), node.value)
                g = v.new_zeros(B, O, Nx)
                gg = None
                if need_gg:
                    gg = v.new_zeros(B, O, Nx, Nx)
                return (v, g, gg), leaf_idx

            else:
                raise TypeError(f"Unknown node type: {type(node)}")

        result, _ = eval_node(self.ast_root, 0)
        return result

    # ──────────────────────────────────────────────────────────────
    # Cache building
    # ──────────────────────────────────────────────────────────────

    def build_cache(self, data, **kw):
        """
        Build cache for optimization with per-leaf caches.
        """
        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            return transparent_leaf.build_cache(data, **kw)

        x, y, *rest = data
        y_sigma = rest[0] if rest else None
        need_derivatives = bool(kw.get("need_derivatives", True))
        seg_arg = kw.get("segments", None)
        if seg_arg is None:
            seg_arg = getattr(self, "segments", None)

        target_sigma = self._target_sigma_from_observations(y, y_sigma)
        sample_weights = self._sample_weights_from_sigma(target_sigma)
        y_sigma_norm = self._normalize_sigma_like(y_sigma, y)

        # Collect leaf atoms in left-to-right depth-first order
        leaf_atoms = self._collect_atoms(self.ast_root)

        leaf_caches = []
        for atom, leaf in zip(leaf_atoms, self.leaf):
            kind = str(getattr(atom, "kind", "")).lower()
            feature_kinds = {"u", "field", "state", "du", "d1u", "grad_u", "d2u", "ddu", "hess_u"}

            # Use unified eval_inputs for computing leaf inputs
            if kind in feature_kinds:
                x_in = x
            elif atom.n_in > 0:
                x_in, _, _ = eval_inputs(atom, x, need_grad=False, need_hess=False)
            else:
                x_in = x[:, atom.var_idxs] if atom.var_idxs else x

            # Handle zero-parameter leaves (e.g., VarLeaf) that would crash in
            # AutogradAdaptor.build_cache when it tries next(self.model.parameters())
            npar = getattr(leaf, "num_parameters", None)
            npar = int(npar()) if callable(npar) else sum(p.numel() for p in leaf.parameters())
            if npar == 0:
                f = leaf(x_in)
                if f.ndim == 1:
                    f = f.unsqueeze(1)
                leaf_cache = {
                    "x": x_in,
                    "y": y,
                    "y_sigma": y_sigma_norm,
                    "target_sigma": target_sigma,
                    "sample_weights": sample_weights,
                    "f": f.detach(),
                }
                if need_derivatives:
                    # Jacobian shape must be (B, O, 0) to match expected format.
                    leaf_cache["jac"] = x_in.new_zeros(f.shape[0], f.shape[1], 0)
                leaf_caches.append(leaf_cache)
                continue

            leaf_kw = dict(kw)
            if seg_arg is not None and hasattr(leaf, "segments") and "segments" not in leaf_kw:
                leaf_kw["segments"] = seg_arg

            c = leaf.build_cache((x_in, y), **leaf_kw)

            # Ensure derivative-aware leaf caches have Jacobians.  Loss-only
            # cache builds are allowed to omit derivative tensors.
            if need_derivatives and "jac" not in c:
                if hasattr(leaf, "jacobian"):
                    c["jac"] = leaf.jacobian(c)
                else:
                    # Autograd fallback
                    c["jac"] = self._autograd_param_jacobian(leaf, c)
            leaf_caches.append(c)

        return {
            "leaves": leaf_caches,
            "x": x,
            "y": y,
            "y_sigma": y_sigma_norm,
            "target_sigma": target_sigma,
            "sample_weights": sample_weights,
            "SegmentedModel": False,
            "S": 1,
            "Pseg": self.num_parameters(),
            "O": self.O,
            "f": self._compose_from_leaf_outputs(
                [leaf_cache["f"] for leaf_cache in leaf_caches],
                ref_x=x,
            ).detach(),
        }

    def _collect_atoms(self, node: Node) -> List[AtomNode]:
        """Collect AtomNodes in left-to-right depth-first order."""
        if isinstance(node, AtomNode):
            return [node]
        elif isinstance(node, (AddNode, MulNode)):
            return self._collect_atoms(node.left) + self._collect_atoms(node.right)
        elif isinstance(node, PowNode):
            return self._collect_atoms(node.base)
        elif isinstance(node, (LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode, ConjNode, RealNode, ImagNode, AbsNode, ArgNode)):
            return self._collect_atoms(node.arg)
        elif isinstance(node, ConstNode):
            return []
        else:
            raise TypeError(f"Unknown node type: {type(node)}")

    def _autograd_param_jacobian(self, leaf, cache_leaf):
        """
        Fallback: compute Jacobian w.r.t. parameters using autograd.
        Returns (B, 1, P_leaf).
        """
        model = getattr(leaf, "model", None)
        x = cache_leaf["x"]
        y = cache_leaf["y"]
        if model is None:
            return x.new_zeros(x.shape[0], 1, 0)

        buffers = {k: b.detach() for k, b in model.named_buffers()}
        p_base = {k: p.detach() for k, p in model.named_parameters()}

        def res_flat(pars):
            f = torch.func.functional_call(model, (pars, buffers), (x,))
            return (y - f).reshape(-1)

        r0 = res_flat(p_base)
        R = int(r0.numel())
        param_items = list(model.named_parameters())
        param_sizes = [p.numel() for _, p in param_items]
        cum_sizes = [0]
        for n in param_sizes:
            cum_sizes.append(cum_sizes[-1] + int(n))
        P = int(cum_sizes[-1])

        if P <= R:
            # The Stage-B analytic leaves often have only a few parameters but
            # tens of thousands of residuals. Reverse-mode jacrev scales with
            # the residual dimension there, so build J by forward-mode columns.
            flat0 = torch.cat([p.reshape(-1) for _, p in param_items]) if param_items else x.new_zeros(0)
            cols = []
            for j in range(P):
                tangent_flat = torch.zeros_like(flat0)
                tangent_flat[j] = 1
                tangent = {
                    key: tangent_flat[cum_sizes[i]:cum_sizes[i + 1]].view_as(param)
                    for i, (key, param) in enumerate(param_items)
                }
                _, jv = torch.func.jvp(res_flat, (p_base,), (tangent,))
                cols.append(jv.reshape(R))
            J = torch.stack(cols, dim=1) if cols else x.new_zeros(R, 0)
        else:
            J_cols = torch.func.jacrev(res_flat)(p_base)
            cols = [J_cols[k].reshape(R, -1) for k, _ in param_items]
            J = torch.cat(cols, dim=1) if cols else x.new_zeros(R, 0)
        return J.reshape(x.shape[0], 1, -1)

    # ──────────────────────────────────────────────────────────────
    # Jacobian-vector products (JVP)
    # ──────────────────────────────────────────────────────────────

    def jvp(self, cache, v, out_dim=None):
        """
        Compute J*v (Jacobian-vector product).

        Parameters
        ----------
        cache : dict
            Cache from build_cache.
        v : torch.Tensor, shape (P,)
            Direction vector in parameter space.
        out_dim : int, optional
            Output dimension (unused for scalar output).

        Returns
        -------
        torch.Tensor, shape (B,) or (B, O)
            J*v result.
        """
        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            return transparent_leaf.jvp(cache, v, out_dim=out_dim)

        vs = self._split_dir(v)
        caches = cache["leaves"]

        def eval_node(node: Node, leaf_idx: int):
            if isinstance(node, AtomNode):
                f = caches[leaf_idx]["f"]
                leaf = self.leaf[leaf_idx]
                # Zero-param leaves contribute zero to the JVP
                npar = getattr(leaf, "num_parameters", lambda: 0)
                npar = int(npar()) if callable(npar) else 0
                if npar == 0:
                    jv_leaf = f.new_zeros(f.shape)
                else:
                    jv_leaf = leaf.jvp(caches[leaf_idx], vs[leaf_idx], out_dim)

                if jv_leaf.ndim == 1:
                    jv_leaf = jv_leaf.unsqueeze(1)

                return (f, jv_leaf), leaf_idx + 1

            elif isinstance(node, AddNode):
                (f1, jv1), idx = eval_node(node.left, leaf_idx)
                (f2, jv2), idx = eval_node(node.right, idx)
                return (f1 + f2, jv1 + jv2), idx

            elif isinstance(node, MulNode):
                (f1, jv1), idx = eval_node(node.left, leaf_idx)
                (f2, jv2), idx = eval_node(node.right, idx)
                return (f1 * f2, f2 * jv1 + f1 * jv2), idx

            elif isinstance(node, PowNode):
                (f1, jv1), idx = eval_node(node.base, leaf_idx)
                c = node.exponent
                f1_safe = self._pow_safe_base(f1, c)
                # d/dp[f^c] = c*f^(c-1) * df/dp
                return (f1_safe.pow(c), c * f1_safe.pow(c - 1) * jv1), idx

            elif isinstance(node, LogNode):
                (f1, jv1), idx = eval_node(node.arg, leaf_idx)
                # d/dp[log(f)] = (1/f) * df/dp
                f1_safe = f1
                if not f1_safe.is_complex():
                    f1_safe = torch.clamp(f1_safe, min=1e-12)
                return (torch.log(f1_safe), (1.0 / f1_safe) * jv1), idx

            elif isinstance(node, ExpNode):
                (f1, jv1), idx = eval_node(node.arg, leaf_idx)
                # d/dp[exp(f)] = exp(f) * df/dp
                v = torch.exp(f1)
                return (v, v * jv1), idx

            elif isinstance(node, SinNode):
                (f1, jv1), idx = eval_node(node.arg, leaf_idx)
                # d/dp[sin(f)] = cos(f) * df/dp
                return (torch.sin(f1), torch.cos(f1) * jv1), idx

            elif isinstance(node, CosNode):
                (f1, jv1), idx = eval_node(node.arg, leaf_idx)
                # d/dp[cos(f)] = -sin(f) * df/dp
                return (torch.cos(f1), -torch.sin(f1) * jv1), idx

            elif isinstance(node, (AsinNode, AcosNode, AtanNode)):
                (f1, jv1), idx = eval_node(node.arg, leaf_idx)
                d1 = self._inv_trig_d1(node, f1)
                return (self._inv_trig_value(node, f1), d1 * jv1), idx

            elif isinstance(node, ConjNode):
                (f1, jv1), idx = eval_node(node.arg, leaf_idx)
                return (torch.conj(f1), torch.conj(jv1)), idx

            elif isinstance(node, RealNode):
                (f1, jv1), idx = eval_node(node.arg, leaf_idx)
                return (torch.real(f1), torch.real(jv1)), idx

            elif isinstance(node, ImagNode):
                (f1, jv1), idx = eval_node(node.arg, leaf_idx)
                return (torch.imag(f1), torch.imag(jv1)), idx

            elif isinstance(node, AbsNode):
                (f1, jv1), idx = eval_node(node.arg, leaf_idx)
                return torch.func.jvp(torch.abs, (f1,), (jv1,)), idx

            elif isinstance(node, ArgNode):
                (f1, jv1), idx = eval_node(node.arg, leaf_idx)
                return torch.func.jvp(torch.angle, (f1,), (jv1,)), idx

            elif isinstance(node, ConstNode):
                # Constant node: no parameters, zero JVP
                c0 = caches[0] if caches else cache
                ref = c0["f"]
                B = ref.shape[0]
                v_const = const_full_like(ref, (B, 1), node.value)
                jv_const = v_const.new_zeros(B, 1)
                return (v_const, jv_const), leaf_idx

            else:
                raise TypeError(f"Unknown node type: {type(node)}")

        (f, Jv), _ = eval_node(self.ast_root, 0)

        # If using a fit-link r=t(y)-t(f), then J_r = t'(f) * J_{y-f}
        link, scale = self._fit_link_cfg()
        if link is not None:
            Jv = fit_link_torch_d1(f, link, scale) * Jv

        return Jv.squeeze(-1) if Jv.shape[-1] == 1 else Jv

    # ──────────────────────────────────────────────────────────────
    # Vector-Jacobian products (VJP / adjoint)
    # ──────────────────────────────────────────────────────────────

    def vjp(self, cache, v, out_dim=None):
        """
        Compute J^T*v (vector-Jacobian product / adjoint).

        This uses reverse-mode automatic differentiation on the AST.

        Parameters
        ----------
        cache : dict
            Cache from build_cache.
        v : torch.Tensor, shape (B,) or (B, O)
            Upstream gradient.
        out_dim : int, optional
            Output dimension (unused for scalar output).

        Returns
        -------
        torch.Tensor, shape (O, P)
            J^T*v result in OSTP order.
        """
        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            return transparent_leaf.vjp(cache, v, out_dim=out_dim)

        if v is None:
            return torch.zeros(
                self.O,
                self.num_parameters(),
                dtype=cache["leaves"][0]["f"].dtype,
                device=self._device,
            )

        # Ensure v has shape (B, O)
        if v.ndim == 1:
            v = v.unsqueeze(1)

        # If using a fit-link r=t(y)-t(f), then J_r^T v = J_{y-f}^T (t'(f) * v)
        link, scale = self._fit_link_cfg()
        if link is not None:
            f = cache.get("f", None)
            if f is None:
                # Fallback: compute forward output from cached x
                f = self.forward(cache["x"]).detach()
            v = fit_link_torch_d1(f, link, scale) * v

        _leaf_atoms = self._collect_atoms(self.ast_root)
        caches = cache["leaves"]

        # Forward pass: compute values
        def forward_pass(node: Node, leaf_idx: int):
            if isinstance(node, AtomNode):
                return caches[leaf_idx]["f"], leaf_idx + 1
            elif isinstance(node, AddNode):
                v1, idx = forward_pass(node.left, leaf_idx)
                v2, idx = forward_pass(node.right, idx)
                return v1 + v2, idx
            elif isinstance(node, MulNode):
                v1, idx = forward_pass(node.left, leaf_idx)
                v2, idx = forward_pass(node.right, idx)
                return v1 * v2, idx
            elif isinstance(node, PowNode):
                v1, idx = forward_pass(node.base, leaf_idx)
                v1_safe = self._pow_safe_base(v1, node.exponent)
                return v1_safe.pow(node.exponent), idx
            elif isinstance(node, LogNode):
                v1, idx = forward_pass(node.arg, leaf_idx)
                if not v1.is_complex():
                    v1 = torch.clamp(v1, min=1e-12)
                return torch.log(v1), idx
            elif isinstance(node, ExpNode):
                v1, idx = forward_pass(node.arg, leaf_idx)
                return torch.exp(v1), idx
            elif isinstance(node, SinNode):
                v1, idx = forward_pass(node.arg, leaf_idx)
                return torch.sin(v1), idx
            elif isinstance(node, CosNode):
                v1, idx = forward_pass(node.arg, leaf_idx)
                return torch.cos(v1), idx
            elif isinstance(node, (AsinNode, AcosNode, AtanNode)):
                v1, idx = forward_pass(node.arg, leaf_idx)
                return self._inv_trig_value(node, v1), idx
            elif isinstance(node, ConjNode):
                v1, idx = forward_pass(node.arg, leaf_idx)
                return torch.conj(v1), idx
            elif isinstance(node, RealNode):
                v1, idx = forward_pass(node.arg, leaf_idx)
                return torch.real(v1), idx
            elif isinstance(node, ImagNode):
                v1, idx = forward_pass(node.arg, leaf_idx)
                return torch.imag(v1), idx
            elif isinstance(node, AbsNode):
                v1, idx = forward_pass(node.arg, leaf_idx)
                return torch.abs(v1), idx
            elif isinstance(node, ArgNode):
                v1, idx = forward_pass(node.arg, leaf_idx)
                return torch.angle(v1), idx
            elif isinstance(node, ConstNode):
                # Constant node: return constant value, don't advance leaf_idx
                ref = caches[0]["f"] if caches else v.new_zeros(1)
                B = ref.shape[0]
                v_const = const_full_like(ref, (B, 1), node.value)
                return v_const, leaf_idx
            else:
                raise TypeError(f"Unknown node type: {type(node)}")

        # Backward pass: propagate adjoints
        leaf_adjoints = [None] * len(self.leaf)

        def backward_pass(node: Node, leaf_idx: int, adjoint):
            if isinstance(node, AtomNode):
                leaf_adjoints[leaf_idx] = adjoint
                return leaf_idx + 1

            elif isinstance(node, AddNode):
                idx = backward_pass(node.left, leaf_idx, adjoint)
                idx = backward_pass(node.right, idx, adjoint)
                return idx

            elif isinstance(node, MulNode):
                # Compute values for left and right subtrees
                v1, idx_after_left = forward_pass(node.left, leaf_idx)
                v2, _ = forward_pass(node.right, idx_after_left)

                # Propagate: ∂L/∂left = adjoint * right, ∂L/∂right = adjoint * left
                adj_left = adjoint * v2
                adj_right = adjoint * v1

                idx = backward_pass(node.left, leaf_idx, adj_left)
                idx = backward_pass(node.right, idx, adj_right)
                return idx

            elif isinstance(node, PowNode):
                # Compute base value
                v1, _ = forward_pass(node.base, leaf_idx)

                # Propagate: ∂L/∂base = adjoint * c * base^(c-1)
                c = node.exponent
                v1_safe = self._pow_safe_base(v1, c)
                adj_base = adjoint * c * v1_safe.pow(c - 1)

                idx = backward_pass(node.base, leaf_idx, adj_base)
                return idx

            elif isinstance(node, LogNode):
                # Compute arg value
                v1, _ = forward_pass(node.arg, leaf_idx)
                if not v1.is_complex():
                    v1 = torch.clamp(v1, min=1e-12)

                # Propagate: ∂L/∂arg = adjoint * (1/arg)
                adj_arg = adjoint * (1.0 / v1)

                idx = backward_pass(node.arg, leaf_idx, adj_arg)
                return idx

            elif isinstance(node, ExpNode):
                # Compute arg value
                v1, _ = forward_pass(node.arg, leaf_idx)

                # Propagate: ∂L/∂arg = adjoint * exp(arg)
                adj_arg = adjoint * torch.exp(v1)

                idx = backward_pass(node.arg, leaf_idx, adj_arg)
                return idx

            elif isinstance(node, SinNode):
                # Compute arg value
                v1, _ = forward_pass(node.arg, leaf_idx)

                # Propagate: ∂L/∂arg = adjoint * cos(arg)
                adj_arg = adjoint * torch.cos(v1)

                idx = backward_pass(node.arg, leaf_idx, adj_arg)
                return idx

            elif isinstance(node, CosNode):
                # Compute arg value
                v1, _ = forward_pass(node.arg, leaf_idx)

                # Propagate: ∂L/∂arg = adjoint * (-sin(arg))
                adj_arg = adjoint * (-torch.sin(v1))

                idx = backward_pass(node.arg, leaf_idx, adj_arg)
                return idx

            elif isinstance(node, (AsinNode, AcosNode, AtanNode)):
                v1, _ = forward_pass(node.arg, leaf_idx)
                adj_arg = adjoint * self._inv_trig_d1(node, v1)
                idx = backward_pass(node.arg, leaf_idx, adj_arg)
                return idx

            elif isinstance(node, ConjNode):
                # Propagate: ∂L/∂arg = conj(adjoint)
                adj_arg = torch.conj(adjoint)
                idx = backward_pass(node.arg, leaf_idx, adj_arg)
                return idx

            elif isinstance(node, RealNode):
                # Propagate: ∂L/∂arg = adjoint (real part passes through)
                idx = backward_pass(node.arg, leaf_idx, adjoint)
                return idx

            elif isinstance(node, ImagNode):
                # Propagate: ∂L/∂arg = adjoint * 1j
                adj_arg = adjoint * 1j
                idx = backward_pass(node.arg, leaf_idx, adj_arg)
                return idx

            elif isinstance(node, AbsNode):
                # Compute arg value
                v1, _ = forward_pass(node.arg, leaf_idx)
                # d|z|/dz = z / |z| (for complex), sign(z) for real
                adj_arg = adjoint * torch.where(v1.abs() > 1e-12, v1 / v1.abs(), v1.new_zeros(v1.shape))
                idx = backward_pass(node.arg, leaf_idx, adj_arg)
                return idx

            elif isinstance(node, ArgNode):
                # Compute arg value
                v1, _ = forward_pass(node.arg, leaf_idx)
                # d(arg(z))/dz = -i/(z) (for complex)
                adj_arg = adjoint * torch.where(v1.abs() > 1e-12, -1j / v1, v1.new_zeros(v1.shape))
                idx = backward_pass(node.arg, leaf_idx, adj_arg)
                return idx

            elif isinstance(node, ConstNode):
                # Constant has no leaves to propagate to
                return leaf_idx

            else:
                raise TypeError(f"Unknown node type: {type(node)}")

        # Run forward pass to populate values, then backward pass
        forward_pass(self.ast_root, 0)
        backward_pass(self.ast_root, 0, v)

        # Accumulate VJPs from each leaf
        outs = []
        for leaf_idx, (leaf, adj) in enumerate(zip(self.leaf, leaf_adjoints)):
            if adj is None:
                # This leaf doesn't contribute (shouldn't happen in valid AST)
                adj = v.new_zeros(v.shape)
            # Zero-param leaves contribute zero-column VJP
            npar = getattr(leaf, "num_parameters", lambda: 0)
            npar = int(npar()) if callable(npar) else 0
            if npar == 0:
                outs.append(caches[leaf_idx]["f"].new_zeros(self.O, 0))
            else:
                outs.append(leaf.vjp(caches[leaf_idx], adj, out_dim))

        return torch.cat(outs, dim=-1)

    # ──────────────────────────────────────────────────────────────
    # grad_jvp / grad_vjp above are residual‑oriented
    # (matching SegmentedAdaptor.grad_jvp/grad_vjp signs)
    # ──────────────────────────────────────────────────────────────

    def grad_jvp(self, cache, v, out_dim=None):
        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            method = getattr(transparent_leaf, "grad_jvp", None)
            if not callable(method):
                raise NotImplementedError(
                    f"{type(transparent_leaf).__name__} missing grad_jvp()"
                )
            return method(cache, v, out_dim=out_dim)

        x = cache["x"]
        B, Nx = x.size(0), x.size(1)
        caches = cache["leaves"]
        vs = self._split_dir(v)
        feature_kinds = {"u", "field", "state", "du", "d1u", "grad_u", "d2u", "ddu", "hess_u"}

        def _as_BO(t):
            return t.unsqueeze(1) if t.ndim == 1 else t

        def _as_BON(t):
            if t.ndim == 2:
                return t.unsqueeze(1)
            if t.ndim == 3:
                return t
            raise ValueError(f"grad_jvp: expected 2D/3D, got {tuple(t.shape)}")

        def eval_node(node, leaf_idx):
            if isinstance(node, AtomNode):
                leaf = self.leaf[leaf_idx]
                c = caches[leaf_idx]
                f = _as_BO(c["f"])
                kind = str(getattr(node, "kind", "")).lower()
                tuple(getattr(node, "var_idxs", ()))
                if not hasattr(leaf, "grad"):
                    raise NotImplementedError(f"{type(leaf).__name__} missing grad()")
                g_leaf = _as_BON(leaf.grad(c, out_dim=out_dim))
                # Zero-param leaves contribute zero to JVP and grad_jvp
                npar = getattr(leaf, "num_parameters", lambda: 0)
                npar = int(npar()) if callable(npar) else 0
                if npar == 0:
                    jv_leaf = f.new_zeros(f.shape)
                    jg_leaf = g_leaf.new_zeros(g_leaf.shape)
                elif vs[leaf_idx] is None:
                    jv_leaf = f.new_zeros(f.shape)
                    jg_leaf = g_leaf.new_zeros(g_leaf.shape)
                else:
                    jv_leaf = _as_BO(leaf.jvp(c, vs[leaf_idx], out_dim))
                    if not hasattr(leaf, "grad_jvp"):
                        raise NotImplementedError(f"{type(leaf).__name__} missing grad_jvp()")
                    jg_leaf = _as_BON(leaf.grad_jvp(c, vs[leaf_idx], out_dim=out_dim))

                # UNIFIED PATH: map leaf-space gradients to global x via eval_inputs()
                if kind in feature_kinds:
                    # Feature atoms use full x directly
                    g = g_leaf
                    jg = jg_leaf
                else:
                    # Get input Jacobian from eval_inputs (handles both simple and compound)
                    _, input_grad, _ = eval_inputs(node, x, need_grad=True, need_hess=False)
                    # input_grad: [B, n_in, Nx]
                    # g_leaf/jg_leaf: [B, O, n_in]
                    g = torch.einsum('boj,bjx->box', g_leaf, input_grad)
                    jg = torch.einsum('boj,bjx->box', jg_leaf, input_grad)
                return (f, g, jv_leaf, jg), leaf_idx + 1

            if isinstance(node, AddNode):
                (f1, g1, jv1, jg1), idx = eval_node(node.left, leaf_idx)
                (f2, g2, jv2, jg2), idx = eval_node(node.right, idx)
                return (f1 + f2, g1 + g2, jv1 + jv2, jg1 + jg2), idx

            if isinstance(node, MulNode):
                (f1, g1, jv1, jg1), idx = eval_node(node.left, leaf_idx)
                (f2, g2, jv2, jg2), idx = eval_node(node.right, idx)
                f = f1 * f2
                g = f2[..., None] * g1 + f1[..., None] * g2
                jv = f2 * jv1 + f1 * jv2
                jg = (
                    f2[..., None] * jg1
                    + jv2[..., None] * g1
                    + jv1[..., None] * g2
                    + f1[..., None] * jg2
                )
                return (f, g, jv, jg), idx

            if isinstance(node, PowNode):
                (f1, g1, jv1, jg1), idx = eval_node(node.base, leaf_idx)
                c = node.exponent
                f1_safe = self._pow_safe_base(f1, c)
                f = f1_safe.pow(c)
                u1 = c * f1_safe.pow(c - 1)
                u2 = c * (c - 1) * f1_safe.pow(c - 2)
                g = u1[..., None] * g1
                jv = u1 * jv1
                jg = (u2 * jv1)[..., None] * g1 + u1[..., None] * jg1
                return (f, g, jv, jg), idx

            if isinstance(node, LogNode):
                (f1, g1, jv1, jg1), idx = eval_node(node.arg, leaf_idx)
                f1_safe = f1
                if not f1_safe.is_complex():
                    f1_safe = torch.clamp(f1_safe, min=1e-12)
                f = torch.log(f1_safe)
                u1 = 1.0 / f1_safe
                u2 = -1.0 / f1_safe.square()
                g = u1[..., None] * g1
                jv = u1 * jv1
                jg = (u2 * jv1)[..., None] * g1 + u1[..., None] * jg1
                return (f, g, jv, jg), idx

            if isinstance(node, ExpNode):
                (f1, g1, jv1, jg1), idx = eval_node(node.arg, leaf_idx)
                f = torch.exp(f1)
                u1 = f
                u2 = f
                g = u1[..., None] * g1
                jv = u1 * jv1
                jg = (u2 * jv1)[..., None] * g1 + u1[..., None] * jg1
                return (f, g, jv, jg), idx

            if isinstance(node, SinNode):
                (f1, g1, jv1, jg1), idx = eval_node(node.arg, leaf_idx)
                sinf = torch.sin(f1)
                cosf = torch.cos(f1)
                f = sinf
                u1 = cosf
                u2 = -sinf
                g = u1[..., None] * g1
                jv = u1 * jv1
                jg = (u2 * jv1)[..., None] * g1 + u1[..., None] * jg1
                return (f, g, jv, jg), idx

            if isinstance(node, CosNode):
                (f1, g1, jv1, jg1), idx = eval_node(node.arg, leaf_idx)
                cosf = torch.cos(f1)
                sinf = torch.sin(f1)
                f = cosf
                u1 = -sinf
                u2 = -cosf
                g = u1[..., None] * g1
                jv = u1 * jv1
                jg = (u2 * jv1)[..., None] * g1 + u1[..., None] * jg1
                return (f, g, jv, jg), idx

            if isinstance(node, (AsinNode, AcosNode, AtanNode)):
                (f1, g1, jv1, jg1), idx = eval_node(node.arg, leaf_idx)
                f = self._inv_trig_value(node, f1)
                u1 = self._inv_trig_d1(node, f1)
                u2 = self._inv_trig_d2(node, f1)
                g = u1[..., None] * g1
                jv = u1 * jv1
                jg = (u2 * jv1)[..., None] * g1 + u1[..., None] * jg1
                return (f, g, jv, jg), idx

            if isinstance(node, ConjNode):
                (f1, g1, jv1, jg1), idx = eval_node(node.arg, leaf_idx)
                f = torch.conj(f1)
                g = torch.conj(g1)
                jv = torch.conj(jv1)
                jg = torch.conj(jg1)
                return (f, g, jv, jg), idx

            if isinstance(node, RealNode):
                (f1, g1, jv1, jg1), idx = eval_node(node.arg, leaf_idx)
                f = torch.real(f1)
                g = torch.real(g1)
                jv = torch.real(jv1)
                jg = torch.real(jg1)
                return (f, g, jv, jg), idx

            if isinstance(node, ImagNode):
                (f1, g1, jv1, jg1), idx = eval_node(node.arg, leaf_idx)
                f = torch.imag(f1)
                g = torch.imag(g1)
                jv = torch.imag(jv1)
                jg = torch.imag(jg1)
                return (f, g, jv, jg), idx

            if isinstance(node, AbsNode) or isinstance(node, ArgNode):
                (f1, g1, jv1, jg1), idx = eval_node(node.arg, leaf_idx)
                phi = torch.abs if isinstance(node, AbsNode) else torch.angle
                f, jv = torch.func.jvp(phi, (f1,), (jv1,))
                g = torch.stack([torch.func.jvp(phi, (f1,), (g1[..., i],))[1] for i in range(Nx)], dim=-1) if Nx > 0 else f.new_zeros(B, self.O, 0)
                # For jg, we need second-order JVP which is complex; use finite diff or simplified approach
                jg = g1.new_zeros(B, self.O, Nx)  # Simplified: assume jg≈0 for these functions
                return (f, g, jv, jg), idx

            if isinstance(node, ConstNode):
                # Constant node: value, zero gradient, zero JVP, zero grad_jvp
                O_proc = self.O
                f = const_full_like(x, (B, O_proc), node.value)
                g = f.new_zeros(B, O_proc, Nx)
                jv = f.new_zeros(B, O_proc)
                jg = f.new_zeros(B, O_proc, Nx)
                return (f, g, jv, jg), leaf_idx

            raise TypeError(f"Unknown node type: {type(node)}")

        (_, _, _, jg), _ = eval_node(self.ast_root, 0)
        return jg.squeeze(1) if out_dim is not None else jg

    def grad_vjp(self, cache, v, out_dim=None):
        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            method = getattr(transparent_leaf, "grad_vjp", None)
            if not callable(method):
                raise NotImplementedError(
                    f"{type(transparent_leaf).__name__} missing grad_vjp()"
                )
            return method(cache, v, out_dim=out_dim)

        if v is None:
            return torch.zeros(
                self.O,
                self.num_parameters(),
                dtype=cache["leaves"][0]["f"].dtype,
                device=self._device,
            )
        x = cache["x"]
        B, Nx = x.size(0), x.size(1)
        caches = cache["leaves"]
        if v.ndim == 2:
            v = v.unsqueeze(1)
        if v.ndim != 3:
            raise ValueError(f"grad_vjp: expected 2D/3D, got {tuple(v.shape)}")
        O_proc = v.size(1)
        feature_kinds = {"u", "field", "state", "du", "d1u", "grad_u", "d2u", "ddu", "hess_u"}

        def _as_BO(t):
            return t.unsqueeze(1) if t.ndim == 1 else t

        def _as_BON(t):
            if t.ndim == 2:
                return t.unsqueeze(1)
            if t.ndim == 3:
                return t
            raise ValueError(f"grad_vjp: expected 2D/3D, got {tuple(t.shape)}")

        memo = {}

        # Store input_grad for each leaf to use in the backward pass
        # Key: leaf_idx, Value: input_grad tensor [B, n_in, Nx]
        input_grad_cache = {}

        def fwd(node, leaf_idx):
            k = id(node)
            if isinstance(node, AtomNode):
                leaf = self.leaf[leaf_idx]
                c = caches[leaf_idx]
                f = _as_BO(c["f"])
                kind = str(getattr(node, "kind", "")).lower()
                if not hasattr(leaf, "grad"):
                    raise NotImplementedError(f"{type(leaf).__name__} missing grad()")
                g_leaf = _as_BON(leaf.grad(c, out_dim=out_dim))

                # UNIFIED PATH: map leaf-space gradients to global x via eval_inputs()
                if kind in feature_kinds:
                    # Feature atoms use full x directly
                    g = g_leaf
                else:
                    # Get input Jacobian from eval_inputs (handles both simple and compound)
                    _, input_grad, _ = eval_inputs(node, x, need_grad=True, need_hess=False)
                    # input_grad: [B, n_in, Nx]
                    # g_leaf: [B, O, n_in]
                    g = torch.einsum('boj,bjx->box', g_leaf, input_grad)
                    # Store for backward pass
                    input_grad_cache[leaf_idx] = input_grad
                memo[k] = (f, g)
                return (f, g), leaf_idx + 1

            if isinstance(node, AddNode):
                (f1, g1), idx = fwd(node.left, leaf_idx)
                (f2, g2), idx = fwd(node.right, idx)
                memo[k] = (f1 + f2, g1 + g2)
                return memo[k], idx

            if isinstance(node, MulNode):
                (f1, g1), idx = fwd(node.left, leaf_idx)
                (f2, g2), idx = fwd(node.right, idx)
                memo[k] = (f1 * f2, f2[..., None] * g1 + f1[..., None] * g2)
                return memo[k], idx

            if isinstance(node, PowNode):
                (f1, g1), idx = fwd(node.base, leaf_idx)
                c = node.exponent
                f1_safe = self._pow_safe_base(f1, c)
                u1 = c * f1_safe.pow(c - 1)
                memo[k] = (f1_safe.pow(c), u1[..., None] * g1)
                return memo[k], idx

            if isinstance(node, LogNode):
                (f1, g1), idx = fwd(node.arg, leaf_idx)
                f1_safe = f1
                if not f1_safe.is_complex():
                    f1_safe = torch.clamp(f1_safe, min=1e-12)
                u1 = 1.0 / f1_safe
                memo[k] = (torch.log(f1_safe), u1[..., None] * g1)
                return memo[k], idx

            if isinstance(node, ExpNode):
                (f1, g1), idx = fwd(node.arg, leaf_idx)
                f = torch.exp(f1)
                memo[k] = (f, f[..., None] * g1)
                return memo[k], idx

            if isinstance(node, SinNode):
                (f1, g1), idx = fwd(node.arg, leaf_idx)
                cosf = torch.cos(f1)
                memo[k] = (torch.sin(f1), cosf[..., None] * g1)
                return memo[k], idx

            if isinstance(node, CosNode):
                (f1, g1), idx = fwd(node.arg, leaf_idx)
                sinf = torch.sin(f1)
                memo[k] = (torch.cos(f1), (-sinf)[..., None] * g1)
                return memo[k], idx

            if isinstance(node, (AsinNode, AcosNode, AtanNode)):
                (f1, g1), idx = fwd(node.arg, leaf_idx)
                d1 = self._inv_trig_d1(node, f1)
                memo[k] = (self._inv_trig_value(node, f1), d1[..., None] * g1)
                return memo[k], idx

            if isinstance(node, ConjNode):
                (f1, g1), idx = fwd(node.arg, leaf_idx)
                memo[k] = (torch.conj(f1), torch.conj(g1))
                return memo[k], idx

            if isinstance(node, RealNode):
                (f1, g1), idx = fwd(node.arg, leaf_idx)
                memo[k] = (torch.real(f1), torch.real(g1))
                return memo[k], idx

            if isinstance(node, ImagNode):
                (f1, g1), idx = fwd(node.arg, leaf_idx)
                memo[k] = (torch.imag(f1), torch.imag(g1))
                return memo[k], idx

            if isinstance(node, AbsNode):
                (f1, g1), idx = fwd(node.arg, leaf_idx)
                f = torch.abs(f1)
                g = torch.stack([torch.func.jvp(torch.abs, (f1,), (g1[..., i],))[1] for i in range(Nx)], dim=-1) if Nx > 0 else f.new_zeros(B, O_proc, 0)
                memo[k] = (f, g)
                return memo[k], idx

            if isinstance(node, ArgNode):
                (f1, g1), idx = fwd(node.arg, leaf_idx)
                f = torch.angle(f1)
                g = torch.stack([torch.func.jvp(torch.angle, (f1,), (g1[..., i],))[1] for i in range(Nx)], dim=-1) if Nx > 0 else f.new_zeros(B, O_proc, 0)
                memo[k] = (f, g)
                return memo[k], idx

            if isinstance(node, ConstNode):
                # Constant node: value, zero gradient
                f = const_full_like(x, (B, O_proc), node.value)
                g = f.new_zeros(B, O_proc, Nx)
                memo[k] = (f, g)
                return memo[k], leaf_idx

            raise TypeError(f"Unknown node type: {type(node)}")

        fwd(self.ast_root, 0)

        leaf_adj_f = [None] * len(self.leaf)
        leaf_adj_g = [None] * len(self.leaf)

        def _acc(old, new):
            return new if old is None else old + new

        def _dot(a, b):
            return (a * b).sum(dim=-1)

        def bwd(node, leaf_idx, adj_f, adj_g):
            if isinstance(node, AtomNode):
                leaf_adj_f[leaf_idx] = _acc(leaf_adj_f[leaf_idx], adj_f)
                leaf_adj_g[leaf_idx] = _acc(leaf_adj_g[leaf_idx], adj_g)
                return leaf_idx + 1

            if isinstance(node, AddNode):
                idx = bwd(node.left, leaf_idx, adj_f, adj_g)
                idx = bwd(node.right, idx, adj_f, adj_g)
                return idx

            if isinstance(node, MulNode):
                (f1, g1) = memo[id(node.left)]
                (f2, g2) = memo[id(node.right)]
                adj_f1 = adj_f * f2 + _dot(adj_g, g2)
                adj_f2 = adj_f * f1 + _dot(adj_g, g1)
                adj_g1 = adj_g * f2[..., None]
                adj_g2 = adj_g * f1[..., None]
                idx = bwd(node.left, leaf_idx, adj_f1, adj_g1)
                idx = bwd(node.right, idx, adj_f2, adj_g2)
                return idx

            if isinstance(node, PowNode):
                (f1, g1) = memo[id(node.base)]
                c = node.exponent
                f1_safe = self._pow_safe_base(f1, c)
                u1 = c * f1_safe.pow(c - 1)
                u2 = c * (c - 1) * f1_safe.pow(c - 2)
                adj_f1 = adj_f * u1 + _dot(adj_g, u2[..., None] * g1)
                adj_g1 = adj_g * u1[..., None]
                return bwd(node.base, leaf_idx, adj_f1, adj_g1)

            if isinstance(node, LogNode):
                (f1, g1) = memo[id(node.arg)]
                f1_safe = f1
                if not f1_safe.is_complex():
                    f1_safe = torch.clamp(f1_safe, min=1e-12)
                u1 = 1.0 / f1_safe
                u2 = -1.0 / f1_safe.square()
                adj_f1 = adj_f * u1 + _dot(adj_g, u2[..., None] * g1)
                adj_g1 = adj_g * u1[..., None]
                return bwd(node.arg, leaf_idx, adj_f1, adj_g1)

            if isinstance(node, ExpNode):
                (f1, g1) = memo[id(node.arg)]
                u1 = torch.exp(f1)
                adj_f1 = adj_f * u1 + _dot(adj_g, u1[..., None] * g1)
                adj_g1 = adj_g * u1[..., None]
                return bwd(node.arg, leaf_idx, adj_f1, adj_g1)

            if isinstance(node, SinNode):
                (f1, g1) = memo[id(node.arg)]
                u1 = torch.cos(f1)
                u2 = -torch.sin(f1)
                adj_f1 = adj_f * u1 + _dot(adj_g, u2[..., None] * g1)
                adj_g1 = adj_g * u1[..., None]
                return bwd(node.arg, leaf_idx, adj_f1, adj_g1)

            if isinstance(node, CosNode):
                (f1, g1) = memo[id(node.arg)]
                u1 = -torch.sin(f1)
                u2 = -torch.cos(f1)
                adj_f1 = adj_f * u1 + _dot(adj_g, u2[..., None] * g1)
                adj_g1 = adj_g * u1[..., None]
                return bwd(node.arg, leaf_idx, adj_f1, adj_g1)

            if isinstance(node, (AsinNode, AcosNode, AtanNode)):
                (f1, g1) = memo[id(node.arg)]
                u1 = self._inv_trig_d1(node, f1)
                u2 = self._inv_trig_d2(node, f1)
                adj_f1 = adj_f * u1 + _dot(adj_g, u2[..., None] * g1)
                adj_g1 = adj_g * u1[..., None]
                return bwd(node.arg, leaf_idx, adj_f1, adj_g1)

            if isinstance(node, ConjNode):
                adj_f1 = torch.conj(adj_f)
                adj_g1 = torch.conj(adj_g)
                return bwd(node.arg, leaf_idx, adj_f1, adj_g1)

            if isinstance(node, RealNode):
                # Real passes through
                return bwd(node.arg, leaf_idx, adj_f, adj_g)

            if isinstance(node, ImagNode):
                # Imag: adjoint is multiplied by 1j
                adj_f1 = adj_f * 1j
                adj_g1 = adj_g * 1j
                return bwd(node.arg, leaf_idx, adj_f1, adj_g1)

            if isinstance(node, (AbsNode, ArgNode)):
                # For abs/arg, use simplified approximation
                (f1, g1) = memo[id(node.arg)]
                if isinstance(node, AbsNode):
                    scale = torch.where(f1.abs() > 1e-12, f1 / f1.abs(), f1.new_zeros(f1.shape))
                else:
                    scale = torch.where(f1.abs() > 1e-12, -1j / f1, f1.new_zeros(f1.shape))
                adj_f1 = adj_f * scale
                adj_g1 = adj_g * scale[..., None]
                return bwd(node.arg, leaf_idx, adj_f1, adj_g1)

            if isinstance(node, ConstNode):
                # Constant has no leaves to propagate to
                return leaf_idx

            raise TypeError(f"Unknown node type: {type(node)}")

        adj_f0 = v.new_zeros(B, O_proc)
        bwd(self.ast_root, 0, adj_f0, v)

        leaf_atoms = self._collect_atoms(self.ast_root)
        outs = []
        for leaf_idx, (atom, leaf) in enumerate(zip(leaf_atoms, self.leaf)):
            c = caches[leaf_idx]
            af = leaf_adj_f[leaf_idx]
            ag = leaf_adj_g[leaf_idx]
            if af is None:
                af = v.new_zeros(B, O_proc)
            if ag is None:
                ag = v.new_zeros(B, O_proc, Nx)

            # Zero-param leaves contribute zero-column VJP and grad_VJP
            npar = getattr(leaf, "num_parameters", lambda: 0)
            npar = int(npar()) if callable(npar) else 0
            if npar == 0:
                outs.append(c["f"].new_zeros(self.O, 0))
                continue

            out_val = leaf.vjp(c, af, out_dim)

            if False:  # Zero-param case already handled above
                out_grad = out_val.new_zeros(out_val.shape)
            else:
                if not hasattr(leaf, "grad_vjp"):
                    raise NotImplementedError(f"{type(leaf).__name__} missing grad_vjp()")
                kind = str(getattr(atom, "kind", "")).lower()

                # UNIFIED PATH: map global-space adjoints to leaf-input space
                if kind in feature_kinds:
                    # Feature atoms use full x directly
                    ag_in = ag
                elif leaf_idx in input_grad_cache:
                    # Use stored input_grad to map adjoints back to leaf-input space
                    # ag: [B, O, Nx], input_grad: [B, n_in, Nx]
                    # ag_in[b,o,j] = Σ_x ag[b,o,x] * input_grad[b,j,x]
                    input_grad = input_grad_cache[leaf_idx]
                    ag_in = torch.einsum('box,bjx->boj', ag, input_grad)
                else:
                    # Fallback for atoms with no inputs (should be rare)
                    ag_in = ag[:, :, :0]

                out_grad = leaf.grad_vjp(c, ag_in, out_dim=out_dim)

            outs.append(out_val + out_grad)

        return torch.cat(outs, dim=-1)

    # ──────────────────────────────────────────────────────────────
    # Hessian parameter-Jacobian (for H^2 / curvature-aware residuals)
    # ──────────────────────────────────────────────────────────────

    def _require_single_feature_atom(self, who: str) -> None:
        """Hessian parameter-Jacobians are implemented only for a composite that
        is a single feature atom (e.g. a trained surrogate field).  The general
        multi-node AST case needs second-order product rules and is not yet
        supported; we fail loudly rather than silently returning a wrong tensor."""
        feature_kinds = {"nn", "u", "field", "state", "du", "d1u", "grad_u",
                         "d2u", "ddu", "hess_u"}
        root = self.ast_root
        ok = (
            isinstance(root, AtomNode)
            and str(getattr(root, "kind", "")).lower() in feature_kinds
            and len(self.leaf) == 1
        )
        if not ok:
            raise NotImplementedError(
                f"{who} is implemented only for a single feature-atom composite "
                f"(e.g. a trained surrogate field); general multi-node AST "
                f"Hessian parameter-Jacobians are not yet supported."
            )

    def grad_grad_jvp(self, cache, v, out_dim=None):
        """(∂_θ Hessian)·v, i.e. the parameter-Jacobian of ∂²f applied to ``v``.

        Routes to the leaf provider's analytic
        ``grad_grad_analytic_jvp_from_cache`` (verified to machine precision for
        the dual segmented adaptor).  Single-feature-atom composites only."""
        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            leaf = transparent_leaf
            leaf_cache = cache
            direction = v
        else:
            self._require_single_feature_atom("grad_grad_jvp")
            leaf = self.leaf[0]
            leaf_cache = cache["leaves"][0]
            direction = self._split_dir(v)[0]
        if not hasattr(leaf, "grad_grad_analytic_jvp_from_cache"):
            raise NotImplementedError(
                f"{type(leaf).__name__} missing grad_grad_analytic_jvp_from_cache()"
            )
        return leaf.grad_grad_analytic_jvp_from_cache(
            leaf_cache, direction, out_dim=out_dim
        )

    def grad_grad_vjp(self, cache, w, out_dim=None):
        """(∂_θ Hessian)^T·w, the adjoint of :meth:`grad_grad_jvp`.

        Routes to the leaf provider's analytic
        ``grad_grad_analytic_vjp_from_cache``.  Single-feature-atom composites
        only (see :meth:`grad_grad_jvp`)."""
        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            leaf = transparent_leaf
            leaf_cache = cache
        else:
            self._require_single_feature_atom("grad_grad_vjp")
            leaf = self.leaf[0]
            leaf_cache = cache["leaves"][0]
        if not hasattr(leaf, "grad_grad_analytic_vjp_from_cache"):
            raise NotImplementedError(
                f"{type(leaf).__name__} missing grad_grad_analytic_vjp_from_cache()"
            )
        return leaf.grad_grad_analytic_vjp_from_cache(leaf_cache, w, out_dim=out_dim)

    # ── Selected input-Hessian diagonal (no full Nx×Nx Hessian) ────
    # These route to the leaf's analytic selected-diagonal methods (verified to
    # machine precision for the dual segmented adaptor): the diagonal entries
    # ∂²f/∂x_a² for a in ``input_dims`` and their parameter Jacobian/jvp/vjp.
    # The Laplacian (the ∇² decoy) is the sum of these spatial diagonals, so the
    # H² Sobolev residual uses them instead of building the full Hessian.

    def grad_grad_diag(self, cache, input_dims=None, out_dim=None):
        """Selected diagonal of the input Hessian, ``(B[,O],D)`` for ``input_dims``."""
        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            leaf = transparent_leaf
            leaf_cache = cache
        else:
            self._require_single_feature_atom("grad_grad_diag")
            leaf = self.leaf[0]
            leaf_cache = cache["leaves"][0]
        if not hasattr(leaf, "grad_grad_diag_from_cache"):
            raise NotImplementedError(
                f"{type(leaf).__name__} missing grad_grad_diag_from_cache()"
            )
        return leaf.grad_grad_diag_from_cache(
            leaf_cache, out_dim=out_dim, input_dims=input_dims
        )

    def grad_grad_diag_jacobian(self, cache, input_dims=None, out_dim=None):
        """Dense parameter Jacobian of the selected Hessian diagonal, ``(B[,O],D,P)``.

        Never materializes the ``Nx×Nx`` Hessian or its full Jacobian -- this is the
        direct-solve-compatible object the H² residual's fast Jacobian uses."""
        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            leaf = transparent_leaf
            leaf_cache = cache
        else:
            self._require_single_feature_atom("grad_grad_diag_jacobian")
            leaf = self.leaf[0]
            leaf_cache = cache["leaves"][0]
        if not hasattr(leaf, "grad_grad_diag_analytic_jacobian_from_cache"):
            raise NotImplementedError(
                f"{type(leaf).__name__} missing grad_grad_diag_analytic_jacobian_from_cache()"
            )
        return leaf.grad_grad_diag_analytic_jacobian_from_cache(
            leaf_cache, out_dim=out_dim, input_dims=input_dims
        )

    def grad_grad_diag_jvp(self, cache, v, input_dims=None, out_dim=None):
        """(∂_θ diag Hessian)·v for the selected ``input_dims``."""
        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            leaf = transparent_leaf
            leaf_cache = cache
            direction = v
        else:
            self._require_single_feature_atom("grad_grad_diag_jvp")
            leaf = self.leaf[0]
            leaf_cache = cache["leaves"][0]
            direction = self._split_dir(v)[0]
        return leaf.grad_grad_diag_analytic_jvp_from_cache(
            leaf_cache, direction, out_dim=out_dim, input_dims=input_dims
        )

    def grad_grad_diag_vjp(self, cache, w, input_dims=None, out_dim=None):
        """(∂_θ diag Hessian)^T·w, the adjoint of :meth:`grad_grad_diag_jvp`."""
        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            leaf = transparent_leaf
            leaf_cache = cache
        else:
            self._require_single_feature_atom("grad_grad_diag_vjp")
            leaf = self.leaf[0]
            leaf_cache = cache["leaves"][0]
        return leaf.grad_grad_diag_analytic_vjp_from_cache(
            leaf_cache, w, out_dim=out_dim, input_dims=input_dims
        )

    # ──────────────────────────────────────────────────────────────
    # Explicit Jacobian
    # ──────────────────────────────────────────────────────────────

    def jacobian(self, cache, out_dim=None):
        """
        Compute explicit Jacobian w.r.t. parameters.

        Returns
        -------
        torch.Tensor, shape (B, O, P)
            Jacobian matrix.
        """

        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            return transparent_leaf.jacobian(cache, out_dim=out_dim)

        def eval_node(node: Node, leaf_idx: int):
            if isinstance(node, AtomNode):
                c = cache["leaves"][leaf_idx]
                J_loc = c["jac"]  # (B, O, P_leaf)
                B, O, _ = J_loc.shape

                J = J_loc.new_zeros(B, O, self._total_params)
                s = self._slice[leaf_idx]
                J[..., s] = J_loc

                return (c["f"], J), leaf_idx + 1

            elif isinstance(node, AddNode):
                (v1, J1), idx = eval_node(node.left, leaf_idx)
                (v2, J2), idx = eval_node(node.right, idx)
                return (v1 + v2, J1 + J2), idx

            elif isinstance(node, MulNode):
                (v1, J1), idx = eval_node(node.left, leaf_idx)
                (v2, J2), idx = eval_node(node.right, idx)
                v1e, v2e = v1.unsqueeze(-1), v2.unsqueeze(-1)
                return (v1 * v2, v2e * J1 + v1e * J2), idx

            elif isinstance(node, PowNode):
                (v1, J1), idx = eval_node(node.base, leaf_idx)
                c = node.exponent
                v1_safe = self._pow_safe_base(v1, c)
                # d/dp[f^c] = c*f^(c-1) * df/dp
                v_pow = v1_safe.pow(c)
                J_pow = c * v1_safe.pow(c - 1).unsqueeze(-1) * J1
                return (v_pow, J_pow), idx

            elif isinstance(node, LogNode):
                (v1, J1), idx = eval_node(node.arg, leaf_idx)
                # d/dp[log(f)] = (1/f) * df/dp
                v1_safe = v1
                if not v1_safe.is_complex():
                    v1_safe = torch.clamp(v1_safe, min=1e-12)
                v_log = torch.log(v1_safe)
                J_log = (1.0 / v1_safe).unsqueeze(-1) * J1
                return (v_log, J_log), idx

            elif isinstance(node, ExpNode):
                (v1, J1), idx = eval_node(node.arg, leaf_idx)
                # d/dp[exp(f)] = exp(f) * df/dp
                v_exp = torch.exp(v1)
                J_exp = v_exp.unsqueeze(-1) * J1
                return (v_exp, J_exp), idx

            elif isinstance(node, SinNode):
                (v1, J1), idx = eval_node(node.arg, leaf_idx)
                # d/dp[sin(f)] = cos(f) * df/dp
                v_sin = torch.sin(v1)
                J_sin = torch.cos(v1).unsqueeze(-1) * J1
                return (v_sin, J_sin), idx

            elif isinstance(node, CosNode):
                (v1, J1), idx = eval_node(node.arg, leaf_idx)
                # d/dp[cos(f)] = -sin(f) * df/dp
                v_cos = torch.cos(v1)
                J_cos = (-torch.sin(v1)).unsqueeze(-1) * J1
                return (v_cos, J_cos), idx

            elif isinstance(node, (AsinNode, AcosNode, AtanNode)):
                (v1, J1), idx = eval_node(node.arg, leaf_idx)
                v_inv = self._inv_trig_value(node, v1)
                J_inv = self._inv_trig_d1(node, v1).unsqueeze(-1) * J1
                return (v_inv, J_inv), idx

            elif isinstance(node, ConjNode):
                (v1, J1), idx = eval_node(node.arg, leaf_idx)
                return (torch.conj(v1), torch.conj(J1)), idx

            elif isinstance(node, RealNode):
                (v1, J1), idx = eval_node(node.arg, leaf_idx)
                return (torch.real(v1), torch.real(J1)), idx

            elif isinstance(node, ImagNode):
                (v1, J1), idx = eval_node(node.arg, leaf_idx)
                return (torch.imag(v1), torch.imag(J1)), idx

            elif isinstance(node, AbsNode):
                (v1, J1), idx = eval_node(node.arg, leaf_idx)
                f = torch.abs(v1)
                # d|z|/dp = Re(conj(z)/|z| * dz/dp)
                scale = torch.where(f > 1e-12, v1 / f, v1.new_zeros(v1.shape))
                J = torch.real(scale.unsqueeze(-1).conj() * J1) if J1.is_complex() else scale.unsqueeze(-1) * J1
                return (f, J), idx

            elif isinstance(node, ArgNode):
                (v1, J1), idx = eval_node(node.arg, leaf_idx)
                f = torch.angle(v1)
                # d(arg(z))/dp = Im(-i/z * dz/dp) = Im(-i * conj(z)/|z|² * dz/dp)
                scale = torch.where(v1.abs() > 1e-12, -1j / v1, v1.new_zeros(v1.shape))
                J = torch.imag(scale.unsqueeze(-1) * J1) if J1.is_complex() else scale.unsqueeze(-1).imag * J1
                return (f, J), idx

            elif isinstance(node, ConstNode):
                # Constant node: value, zero Jacobian
                ref_cache = cache["leaves"][0] if cache["leaves"] else cache
                ref = ref_cache["f"]
                B, O = ref.shape[0], ref.shape[1] if ref.ndim > 1 else 1
                v_const = const_full_like(ref, (B, O), node.value)
                J_const = v_const.new_zeros(B, O, self._total_params)
                return (v_const, J_const), leaf_idx

            else:
                raise TypeError(f"Unknown node type: {type(node)}")

        (f, J), _ = eval_node(self.ast_root, 0)

        # Fit-link scaling: if r=t(y)-t(f), then J_r = t'(f) * J_{y-f}
        link, scale = self._fit_link_cfg()
        if link is not None:
            w = fit_link_torch_d1(f, link, scale)  # (B, O)
            J = w.unsqueeze(-1) * J

        if out_dim is not None:
            o = int(out_dim)
            J = J[:, o : o + 1, :]

        return J

    # ──────────────────────────────────────────────────────────────
    # Diagonal of J^T J
    # ──────────────────────────────────────────────────────────────

    def diag(self, cache, out_dim=None):
        """
        Compute diagonal of J^T*J efficiently.

        Returns
        -------
        torch.Tensor, shape (P,)
            Diagonal of J^T*J.
        """

        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            return transparent_leaf.diag(cache, out_dim=out_dim)

        def eval_node(node: Node, leaf_idx: int):
            if isinstance(node, AtomNode):
                c = cache["leaves"][leaf_idx]
                v_leaf = c["f"]  # (B, O)

                # Squared column norms
                J_leaf = c["jac"]  # (B, O, P_leaf)
                d_local = J_leaf.square().sum(1)  # (B, P_leaf)

                # Scatter into global P-vector
                d_leaf = J_leaf.new_zeros(J_leaf.size(0), self._total_params)
                s = self._slice[leaf_idx]
                d_leaf[:, s] = d_local

                return (v_leaf, d_leaf), leaf_idx + 1

            elif isinstance(node, AddNode):
                (v1, d1), idx = eval_node(node.left, leaf_idx)
                (v2, d2), idx = eval_node(node.right, idx)
                return (v1 + v2, d1 + d2), idx

            elif isinstance(node, MulNode):
                (v1, d1), idx = eval_node(node.left, leaf_idx)
                (v2, d2), idx = eval_node(node.right, idx)

                s1 = (v2.square()).sum(-1, keepdim=True)  # (B, 1)
                s2 = (v1.square()).sum(-1, keepdim=True)
                v = v1 * v2
                d = s1 * d1 + s2 * d2

                return (v, d), idx

            elif isinstance(node, PowNode):
                (v1, d1), idx = eval_node(node.base, leaf_idx)
                c = node.exponent
                v1_safe = self._pow_safe_base(v1, c)
                # ||d/dp[f^c]||² = (c*f^(c-1))² * ||df/dp||²
                u1 = c * v1_safe.pow(c - 1)
                scale = u1.square().sum(-1, keepdim=True)  # (B, 1)
                v = v1_safe.pow(c)
                d = scale * d1

                return (v, d), idx

            elif isinstance(node, LogNode):
                (v1, d1), idx = eval_node(node.arg, leaf_idx)
                v1_safe = v1
                if not v1_safe.is_complex():
                    v1_safe = torch.clamp(v1_safe, min=1e-12)
                # ||d/dp[log(f)]||² = (1/f)² * ||df/dp||²
                scale = (1.0 / v1_safe).square().sum(-1, keepdim=True)  # (B, 1)
                v = torch.log(v1_safe)
                d = scale * d1

                return (v, d), idx

            elif isinstance(node, ExpNode):
                (v1, d1), idx = eval_node(node.arg, leaf_idx)
                # ||d/dp[exp(f)]||² = exp(f)² * ||df/dp||²
                v = torch.exp(v1)
                scale = v.square().sum(-1, keepdim=True)  # (B, 1)
                d = scale * d1

                return (v, d), idx

            elif isinstance(node, SinNode):
                (v1, d1), idx = eval_node(node.arg, leaf_idx)
                # ||d/dp[sin(f)]||² = cos(f)² * ||df/dp||²
                scale = torch.cos(v1).square().sum(-1, keepdim=True)  # (B, 1)
                v = torch.sin(v1)
                d = scale * d1

                return (v, d), idx

            elif isinstance(node, CosNode):
                (v1, d1), idx = eval_node(node.arg, leaf_idx)
                # ||d/dp[cos(f)]||² = sin(f)² * ||df/dp||²
                scale = torch.sin(v1).square().sum(-1, keepdim=True)  # (B, 1)
                v = torch.cos(v1)
                d = scale * d1

                return (v, d), idx

            elif isinstance(node, (AsinNode, AcosNode, AtanNode)):
                (v1, d1), idx = eval_node(node.arg, leaf_idx)
                d1_arg = self._inv_trig_d1(node, v1)
                scale = d1_arg.square().sum(-1, keepdim=True)
                v = self._inv_trig_value(node, v1)
                d = scale * d1

                return (v, d), idx

            elif isinstance(node, ConjNode):
                (v1, d1), idx = eval_node(node.arg, leaf_idx)
                return (torch.conj(v1), d1), idx

            elif isinstance(node, RealNode):
                (v1, d1), idx = eval_node(node.arg, leaf_idx)
                return (torch.real(v1), d1), idx

            elif isinstance(node, ImagNode):
                (v1, d1), idx = eval_node(node.arg, leaf_idx)
                return (torch.imag(v1), d1), idx

            elif isinstance(node, (AbsNode, ArgNode)):
                (v1, d1), idx = eval_node(node.arg, leaf_idx)
                phi = torch.abs if isinstance(node, AbsNode) else torch.angle
                v = phi(v1)
                # Simplified: pass through diagonal
                return (v, d1), idx

            elif isinstance(node, ConstNode):
                # Constant node: value, zero diagonal
                ref_cache = cache["leaves"][0] if cache["leaves"] else cache
                ref = ref_cache["f"]
                B, O = ref.shape[0], ref.shape[1] if ref.ndim > 1 else 1
                v_const = const_full_like(ref, (B, O), node.value)
                d_const = v_const.new_zeros(B, self._total_params)
                return (v_const, d_const), leaf_idx

            else:
                raise TypeError(f"Unknown node type: {type(node)}")

        sample_weights = self._trace_sample_weights(
            cache,
            out_dim=out_dim,
            device=cache["x"].device,
            dtype=cache["x"].real.dtype if cache["x"].is_complex() else cache["x"].dtype,
        )
        if sample_weights is not None:
            J = self.jacobian(cache)
            if out_dim is not None:
                J = J[:, out_dim]
            J = self._apply_sample_weights_to_jacobian(J, cache, out_dim=out_dim)
            if out_dim is None:
                J = J.reshape(-1, J.size(-1))
            return J.square().sum(0)

        (f, diag_sq), _ = eval_node(self.ast_root, 0)

        # Fit-link scaling: if r=t(y)-t(f), then J_r rows are scaled by t'(f).
        # Therefore diag(J_r^T J_r) applies a per-sample factor (t'(f))^2.
        link, scale = self._fit_link_cfg()
        if link is not None:
            w2 = fit_link_torch_d1(f, link, scale).square()  # (B, O)
            if w2.ndim == 2 and w2.shape[1] == 1:
                diag_sq = diag_sq * w2
            else:
                diag_sq = diag_sq * w2.mean(dim=1, keepdim=True)

        diag_result = diag_sq.sum(0)  # Sum over batch

        if out_dim is not None:
            return diag_result

        return diag_result

    # ──────────────────────────────────────────────────────────────
    # Dense J^T J
    # ──────────────────────────────────────────────────────────────

    def dense(self, cache, out_dim=None):
        """
        Compute dense J^T*J matrix.

        Returns
        -------
        torch.Tensor, shape (P, P)
            J^T*J matrix.
        """
        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            return transparent_leaf.dense(cache, out_dim=out_dim)

        J = self.jacobian(cache)

        if out_dim is not None:
            J = J[:, out_dim]
        else:
            J = self._apply_sample_weights_to_jacobian(J, cache, out_dim=None)
            J = J.reshape(-1, J.size(-1))
            return J.t().matmul(J)

        J = self._apply_sample_weights_to_jacobian(J, cache, out_dim=out_dim)
        return J.t().matmul(J)

    # ──────────────────────────────────────────────────────────────
    # Parameter management
    # ──────────────────────────────────────────────────────────────

    def num_parameters(self):
        """Return total number of parameters."""
        return self._total_params

    def named_parameters(self, *a, **k):
        """Iterate over named parameters."""
        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            yield from transparent_leaf.named_parameters(*a, **k)
            return
        for i, leaf in enumerate(self.leaf):
            for n, p in leaf.named_parameters(*a, **k):
                yield f"leaf{i}.{n}", p

    def named_buffers(self, *a, **k):
        """Iterate over buffers with native names for an identity wrapper."""
        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            yield from transparent_leaf.named_buffers(*a, **k)
            return
        yield from torch.nn.Module.named_buffers(self, *a, **k)

    # ──────────────────────────────────────────────────────────────
    # Optimizer callbacks
    # ──────────────────────────────────────────────────────────────

    def pre_block(self, block=None, *, theta=None, segments=None):
        """Compatibility shim for provider wrappers that expect `pre_block`."""
        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            if segments is not None:
                if isinstance(block, dict):
                    block = {**block, "segments": segments}
                elif block is None:
                    block = {"segments": segments}
            return transparent_leaf.pre_block(block=block, theta=theta)

        seg_arg = segments
        if seg_arg is None and isinstance(block, dict):
            seg_arg = block.get("segments", None)
        if seg_arg is not None:
            self.segments = torch.as_tensor(seg_arg, device=self._device, dtype=torch.long)

        p_vec = theta
        if p_vec is None and torch.is_tensor(block):
            p_vec = block
        if p_vec is not None:
            self.pre_block_hook(p_vec)
        return None

    def pre_block_hook(self, p_vec, *, a_idx=None, b_idx=None, c_idx=None, K_idx=None):
        """
        Distribute parameter updates to leaves.

        This is called by the optimizer before each block update.
        """
        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            hook = getattr(transparent_leaf, "pre_block_hook", None)
            if callable(hook):
                return hook(
                    p_vec,
                    a_idx=a_idx,
                    b_idx=b_idx,
                    c_idx=c_idx,
                    K_idx=K_idx,
                )
            return transparent_leaf.pre_block(p_vec)

        offset = 0
        for leaf in self.leaf:
            P_leaf = leaf.num_parameters()
            if P_leaf == 0:
                continue

            p_leaf = p_vec[offset : offset + P_leaf]

            def _subset(idxs):
                if idxs is None:
                    return None
                mask = (idxs >= offset) & (idxs < offset + P_leaf)
                if not mask.any():
                    return None
                return idxs[mask] - offset

            a_sub, b_sub, c_sub, K_sub = map(_subset, (a_idx, b_idx, c_idx, K_idx))

            if any(x is not None for x in (a_sub, b_sub, c_sub, K_sub)):
                leaf.pre_block_hook(p_leaf, a_idx=a_sub, b_idx=b_sub, c_idx=c_sub, K_idx=K_sub)

            offset += P_leaf

    def blocks(self, *args, shuffle: bool = False, **kwargs):
        """
        Yield parameter blocks for optimizer.

        Returns single block covering all parameters.
        """
        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            leaf_kwargs = dict(kwargs)
            if len(args) < 2:
                leaf_kwargs["shuffle"] = shuffle
            for raw_block in transparent_leaf.blocks(*args, **leaf_kwargs):
                if isinstance(raw_block, dict):
                    block = dict(raw_block)
                    block["owner"] = id(self)
                else:
                    block = raw_block
                    try:
                        block.owner = id(self)
                    except Exception:
                        pass
                yield block
            return

        P = self._total_params
        dev = self._device

        # Build composite-local analytic/global maps
        ana_parts, glob_parts = [], []
        cursor = 0

        for leaf in self.leaf:
            for blk_leaf in leaf.blocks():
                ana = blk_leaf.get("analytic_map", None)
                gmp = blk_leaf.get("global_map", None)

                if isinstance(ana, slice):
                    ana = torch.arange(ana.start, ana.stop, device=dev, dtype=torch.long)
                elif isinstance(ana, torch.Tensor):
                    ana = ana.to(device=dev, dtype=torch.long)
                else:
                    raise RuntimeError("ASTCompositeAdaptor.blocks: missing analytic_map")

                if isinstance(gmp, slice):
                    gmp = torch.arange(gmp.start, gmp.stop, device=dev, dtype=torch.long)
                elif isinstance(gmp, torch.Tensor):
                    gmp = gmp.to(device=dev, dtype=torch.long)
                else:
                    raise RuntimeError("ASTCompositeAdaptor.blocks: missing global_map")

                ana_parts.append(ana + cursor)
                glob_parts.append(gmp + cursor)
                cursor += int(ana.numel())

        if cursor != P:
            raise RuntimeError(f"ASTCompositeAdaptor.blocks: size mismatch {cursor} != {P}")

        analytic_map = torch.cat(ana_parts).contiguous()
        global_map = torch.cat(glob_parts).contiguous()
        dimension_map = torch.zeros_like(analytic_map)
        param_idx = torch.arange(P, device=dev, dtype=torch.long)

        yield dict(
            owner=id(self),
            segments=None,
            global_map=global_map,
            analytic_map=analytic_map,
            dimension_map=dimension_map,
            param_idx=param_idx,
            a_fit_indices=None,
            b_fit_indices=None,
            c_fit_indices=None,
            K_fit_indices=None,
            weight=1.0,
        )

    def linear_refinement(self, residual_modules, device, lam_LM, ridge_val=1e-5):
        """
        Composite-aware linear refinement.

        We generalize leaf-level closed-form 'a' solves to AST composites by
        transforming the global data residual r = y - f(x) into per-leaf targets
        using the chain rule.

        For composite output f and leaf outputs u_i, define chain factors
        g_i = ∂f/∂u_i (evaluated at current parameters). A first-order update is

            f_new ≈ f + Σ_i g_i · (u_i,new - u_i,old)

        We choose a *simultaneous* (Jacobi-style) update in leaf-output space:

            Δu_i = g_i · r / (Σ_j g_j² + λ)

        which guarantees Σ_i g_i Δu_i = r when λ=0, avoiding the "everyone fits
        the full residual" overshoot when multiple leaves carry linear degrees
        of freedom.

        Each leaf then solves for its linear 'a' parameters to fit

            u_target_i = u_i,old + Δu_i

        under row-weights scaled by g_i² (implemented via y_sigma / |g_i|).

        Returns a single concatenated (idx, new_vals) update spanning all leaves
        that successfully produced a linear refinement.
        """
        transparent_leaf = self._transparent_identity_leaf()
        if transparent_leaf is not None:
            method = getattr(transparent_leaf, "linear_refinement", None)
            if not callable(method):
                return None, None
            return method(
                residual_modules,
                device,
                lam_LM,
                ridge_val=ridge_val,
            )

        if not self.leaf:
            return None, None

        leaf_atoms = self._collect_atoms(self.ast_root)
        if not leaf_atoms or len(leaf_atoms) != len(self.leaf):
            return None, None

        feature_kinds = {
            "u", "field", "state", "du", "d1u", "grad_u",
            "d2u", "ddu", "hess_u",
        }

        lam_damp = float(lam_LM) if lam_LM is not None else 0.0
        g_min = 1e-12

        def _leaf_has_linear_a(lf) -> bool:
            bm = None
            if hasattr(lf, "stage1") and hasattr(lf.stage1, "base_model"):
                bm = lf.stage1.base_model
            elif hasattr(lf, "base_model"):
                bm = lf.base_model
            if bm is None:
                return False
            return float(getattr(bm, "a_size", 0.0)) > 0.0

        cand = [i for i, lf in enumerate(self.leaf)
                if hasattr(lf, "linear_refinement") and _leaf_has_linear_a(lf)]
        if not cand:
            return None, None

        def _split_x(x):
            if torch.is_tensor(x):
                return x, (), False
            if isinstance(x, (tuple, list)) and len(x) > 0 and torch.is_tensor(x[0]):
                return x[0], tuple(x[1:]), True
            return None, (), False

        # ---- helpers -------------------------------------------------
        def _atom_input(atom: AtomNode, x_raw):
            x_main, extras, is_tuple = _split_x(x_raw)
            if x_main is None:
                return None
            kind = str(getattr(atom, "kind", "")).lower()

            # Use unified eval_inputs for both simple and compound atoms
            if kind in feature_kinds:
                return x_raw
            elif atom.n_in > 0:
                x_in, _, _ = eval_inputs(atom, x_main, need_grad=False, need_hess=False)
                if is_tuple:
                    return (x_in, *extras)
                return x_in
            else:
                x_sel = x_main[:, atom.var_idxs] if atom.var_idxs else x_main
                if is_tuple:
                    return (x_sel, *extras)
                return x_sel

        def _eval_ast_and_chain(x_raw):
            # Forward values for every node + leaf outputs in DFS order.
            vals = {}
            leaf_vals = [None] * len(self.leaf)

            def _eval(node: Node, leaf_idx: int):
                key = id(node)
                if key in vals:
                    return vals[key], leaf_idx
                if isinstance(node, AtomNode):
                    xin = _atom_input(node, x_raw)
                    if xin is None:
                        return None, leaf_idx
                    v = self.leaf[leaf_idx](xin)
                    leaf_vals[leaf_idx] = v
                    vals[key] = v
                    return v, leaf_idx + 1
                if isinstance(node, AddNode):
                    vL, idx = _eval(node.left, leaf_idx)
                    vR, idx = _eval(node.right, idx)
                    if vL is None or vR is None:
                        return None, idx
                    v = vL + vR
                    vals[key] = v
                    return v, idx
                if isinstance(node, MulNode):
                    vL, idx = _eval(node.left, leaf_idx)
                    vR, idx = _eval(node.right, idx)
                    if vL is None or vR is None:
                        return None, idx
                    v = vL * vR
                    vals[key] = v
                    return v, idx
                if isinstance(node, PowNode):
                    vB, idx = _eval(node.base, leaf_idx)
                    if vB is None:
                        return None, idx
                    vB_safe = self._pow_safe_base(vB, node.exponent)
                    if vB_safe is not vB:
                        vals[id(node.base)] = vB_safe
                    v = vB_safe.pow(node.exponent)
                    vals[key] = v
                    return v, idx
                if isinstance(node, LogNode):
                    vA, idx = _eval(node.arg, leaf_idx)
                    if vA is None:
                        return None, idx
                    if not vA.is_complex():
                        vA = torch.clamp(vA, min=1e-12)
                    v = torch.log(vA)
                    vals[key] = v
                    return v, idx
                if isinstance(node, ExpNode):
                    vA, idx = _eval(node.arg, leaf_idx)
                    if vA is None:
                        return None, idx
                    v = torch.exp(vA)
                    vals[key] = v
                    return v, idx
                if isinstance(node, SinNode):
                    vA, idx = _eval(node.arg, leaf_idx)
                    if vA is None:
                        return None, idx
                    v = torch.sin(vA)
                    vals[key] = v
                    return v, idx
                if isinstance(node, CosNode):
                    vA, idx = _eval(node.arg, leaf_idx)
                    if vA is None:
                        return None, idx
                    v = torch.cos(vA)
                    vals[key] = v
                    return v, idx
                if isinstance(node, (AsinNode, AcosNode, AtanNode)):
                    vA, idx = _eval(node.arg, leaf_idx)
                    if vA is None:
                        return None, idx
                    v = self._inv_trig_value(node, vA)
                    vals[key] = v
                    return v, idx
                if isinstance(node, ConjNode):
                    vA, idx = _eval(node.arg, leaf_idx)
                    if vA is None:
                        return None, idx
                    v = torch.conj(vA)
                    vals[key] = v
                    return v, idx
                if isinstance(node, RealNode):
                    vA, idx = _eval(node.arg, leaf_idx)
                    if vA is None:
                        return None, idx
                    v = torch.real(vA)
                    vals[key] = v
                    return v, idx
                if isinstance(node, ImagNode):
                    vA, idx = _eval(node.arg, leaf_idx)
                    if vA is None:
                        return None, idx
                    v = torch.imag(vA)
                    vals[key] = v
                    return v, idx
                if isinstance(node, AbsNode):
                    vA, idx = _eval(node.arg, leaf_idx)
                    if vA is None:
                        return None, idx
                    v = torch.abs(vA)
                    vals[key] = v
                    return v, idx
                if isinstance(node, ArgNode):
                    vA, idx = _eval(node.arg, leaf_idx)
                    if vA is None:
                        return None, idx
                    v = torch.angle(vA)
                    vals[key] = v
                    return v, idx
                if isinstance(node, ConstNode):
                    # Constant node: return constant value, don't advance leaf_idx
                    x_main, _, _ = _split_x(x_raw)
                    if x_main is None:
                        return None, leaf_idx
                    B = x_main.shape[0]
                    v = const_full_like(x_main, (B, 1), node.value)
                    vals[key] = v
                    return v, leaf_idx
                raise TypeError(f"Unknown node type: {type(node)}")

            f_pred, next_idx = _eval(self.ast_root, 0)
            if f_pred is None or next_idx != len(self.leaf):
                return None, None, None

            chain = [None] * len(self.leaf)

            def _backprop(node: Node, leaf_idx: int, upstream: torch.Tensor):
                if isinstance(node, AtomNode):
                    chain[leaf_idx] = upstream
                    return leaf_idx + 1
                if isinstance(node, AddNode):
                    idx = _backprop(node.left, leaf_idx, upstream)
                    idx = _backprop(node.right, idx, upstream)
                    return idx
                if isinstance(node, MulNode):
                    vL = vals[id(node.left)]
                    vR = vals[id(node.right)]
                    idx = _backprop(node.left, leaf_idx, upstream * vR)
                    idx = _backprop(node.right, idx, upstream * vL)
                    return idx
                if isinstance(node, PowNode):
                    vB = vals[id(node.base)]
                    c = node.exponent
                    vB_safe = self._pow_safe_base(vB, c)
                    child_up = upstream * c * vB_safe.pow(c - 1)
                    return _backprop(node.base, leaf_idx, child_up)
                if isinstance(node, LogNode):
                    vA = vals[id(node.arg)]
                    if not vA.is_complex():
                        vA = torch.clamp(vA, min=1e-12)
                    child_up = upstream / vA
                    return _backprop(node.arg, leaf_idx, child_up)
                if isinstance(node, ExpNode):
                    v = vals[id(node)]
                    child_up = upstream * v
                    return _backprop(node.arg, leaf_idx, child_up)
                if isinstance(node, SinNode):
                    vA = vals[id(node.arg)]
                    child_up = upstream * torch.cos(vA)
                    return _backprop(node.arg, leaf_idx, child_up)
                if isinstance(node, CosNode):
                    vA = vals[id(node.arg)]
                    child_up = upstream * (-torch.sin(vA))
                    return _backprop(node.arg, leaf_idx, child_up)
                if isinstance(node, (AsinNode, AcosNode, AtanNode)):
                    vA = vals[id(node.arg)]
                    child_up = upstream * self._inv_trig_d1(node, vA)
                    return _backprop(node.arg, leaf_idx, child_up)
                if isinstance(node, ConjNode):
                    child_up = torch.conj(upstream)
                    return _backprop(node.arg, leaf_idx, child_up)
                if isinstance(node, RealNode):
                    return _backprop(node.arg, leaf_idx, upstream)
                if isinstance(node, ImagNode):
                    child_up = upstream * 1j
                    return _backprop(node.arg, leaf_idx, child_up)
                if isinstance(node, AbsNode):
                    vA = vals[id(node.arg)]
                    child_up = upstream * torch.where(vA.abs() > 1e-12, vA / vA.abs(), vA.new_zeros(vA.shape))
                    return _backprop(node.arg, leaf_idx, child_up)
                if isinstance(node, ArgNode):
                    vA = vals[id(node.arg)]
                    child_up = upstream * torch.where(vA.abs() > 1e-12, -1j / vA, vA.new_zeros(vA.shape))
                    return _backprop(node.arg, leaf_idx, child_up)
                if isinstance(node, ConstNode):
                    # Constant has no leaves to propagate to
                    return leaf_idx
                raise TypeError(f"Unknown node type: {type(node)}")

            _backprop(self.ast_root, 0, torch.ones_like(f_pred))
            return f_pred, chain, leaf_vals

        # ---- run per-leaf refinements --------------------------------
        all_idx = []
        all_vals = []

        for leaf_idx in cand:
            leaf = self.leaf[leaf_idx]
            atom = leaf_atoms[leaf_idx]

            class _ChainScaledResidualModule:
                def __init__(_self, base_mod):
                    _self._base_mod = base_mod
                    _self.dataloader = getattr(base_mod, "dataloader", None)
                    _self._parent = getattr(base_mod, "_parent", None)
                    _self.weight = float(getattr(base_mod, "weight", 1.0) or 1.0)
                    _self.normalization = str(getattr(base_mod, "normalization", "mean") or "mean")

                def _normalization_denominator(_self, nres_total):
                    fn = getattr(_self._base_mod, "_normalization_denominator", None)
                    if callable(fn):
                        return fn(nres_total)
                    if _self.normalization == "sum":
                        return 1.0
                    return float(max(1, int(nres_total)))

                def get_data_batch(_self, batch, device):
                    x_raw, y, y_sigma = _self._base_mod.get_data_batch(batch, device)
                    x_main, _, _ = _split_x(x_raw)
                    if x_main is None:
                        return None, y, y_sigma
                    with torch.no_grad():
                        f_pred, chain, leaf_vals = _eval_ast_and_chain(x_raw)
                        if chain is None:
                            return None, y, y_sigma
                        g_i = chain[leaf_idx]
                        u_old = leaf_vals[leaf_idx]
                        if g_i is None or u_old is None:
                            return None, y, y_sigma

                        r = y - f_pred

                        sum_g2 = None
                        for j in cand:
                            g_j = chain[j]
                            if g_j is None:
                                continue
                            term = g_j * g_j
                            sum_g2 = term if sum_g2 is None else (sum_g2 + term)
                        if sum_g2 is None:
                            return None, y, y_sigma
                        denom = (sum_g2 + lam_damp).clamp_min(g_min)

                        u_target = u_old + (g_i * r) / denom

                        if y_sigma is None:
                            sig = torch.ones_like(u_target)
                        else:
                            sig = torch.as_tensor(y_sigma, device=device, dtype=u_target.dtype)

                        g_abs = g_i.abs().clamp_min(g_min)
                        g_b = g_abs
                        while g_b.ndim < sig.ndim:
                            g_b = g_b.unsqueeze(-1)
                        sig_eff = sig / g_b

                        x_in = _atom_input(atom, x_raw)
                        return x_in, u_target, sig_eff

            wrapped = [_ChainScaledResidualModule(m) for m in residual_modules]
            idx_leaf, new_vals = leaf.linear_refinement(
                wrapped, device, lam_LM, ridge_val=ridge_val
            )
            if idx_leaf is None:
                continue

            try:
                n_leaf = int(leaf.num_parameters())
            except Exception:
                n_leaf = None
            if torch.is_tensor(idx_leaf) and n_leaf is not None and idx_leaf.numel() > 0:
                if int(idx_leaf.max().item()) < n_leaf:
                    idx_leaf = idx_leaf + int(self._slice[leaf_idx].start)

            all_idx.append(idx_leaf)
            all_vals.append(new_vals)

        if not all_idx:
            return None, None

        idx = torch.cat(all_idx)
        vals = torch.cat(all_vals)

        sort_perm = torch.argsort(idx)
        return idx[sort_perm], vals[sort_perm]
