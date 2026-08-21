# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Outer-transform and NN-leaf separability Stage-B fallback rules."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from nestynet_sr.sr_core import check_separability
from nestynet_sr.sr_core.bridges import (
    AcosNode,
    AddNode,
    AsinNode,
    AtanNode,
    AtomNode,
    ConstNode,
    CosNode,
    ExpNode,
    LogNode,
    MulNode,
    Node,
    PowNode,
    SinNode,
    clone_inputs,
    effective_arity,
    replace_atom_in_ast,
)
from nestynet_sr.sr_search.candidate_builders import _build_atom_input_tensor
from nestynet_sr.sr_search.y_transforms import get_separability_y_ops

from .engine import Candidate, StageBContext, StageBRule, atom_content_hash
from .helpers import (
    _build_gauge_split_candidates,
    _build_subtree_separability_candidate,
    _build_subtree_separability_outer_transform_candidates,
    _collect_all_atoms,
    _collect_multivariate_nn_atoms,
    _find_nns_in_add_chain,
    _find_nns_in_mul_chain,
    build_atom_to_leaf_map,
    run_subtree_separability,
)
from .models import _SubtreeModel
from .rules_common import _subtree_content_hash

# Full canonical unary set (includes AbsNode etc., which sympy round-trips
# can introduce into accepted states — see bridges.UNARY_NODE_TYPES).
from nestynet_sr.sr_core.bridges import AbsNode
from nestynet_sr.sr_core.bridges import UNARY_NODE_TYPES as _UNARY_AST_NODES


class _OuterTransformedSubtreeModel(torch.nn.Module):
    """A tiny wrapper: v(x) = T(u(x)) for a given subtree u(x).

    This exposes analytic grad / grad_grad w.r.t. inputs via the chain rule:
        v   = T(u)
        v'  = T'(u) u'
        v'' = T''(u) u' u'^T + T'(u) u''

    We cache (u, u', u'') per-x call to avoid recomputing the subtree 3×
    inside check_separability().
    """

    def __init__(self, u_model: _SubtreeModel, torch_op, d1, d2):
        super().__init__()
        self.u_model = u_model
        self.torch_op = torch_op
        self.d1 = d1
        self.d2 = d2
        self._last_x_id = None
        self._last_u = None
        self._last_g = None
        self._last_gg = None

    def _ensure_cache(self, x: torch.Tensor):
        x_id = id(x)
        if x_id == self._last_x_id and self._last_u is not None:
            return
        u, g, gg = self.u_model._value_grad_grad(x, need_gg=True)
        self._last_x_id = x_id
        self._last_u = u
        self._last_g = g
        self._last_gg = gg

    def forward(self, x):
        if isinstance(x, dict):
            x = x["x"]
        self._ensure_cache(x)
        return self.torch_op(self._last_u)

    def grad(self, cache_or_x, out_dim=None):
        x = cache_or_x["x"] if isinstance(cache_or_x, dict) else cache_or_x
        self._ensure_cache(x)
        u = self._last_u
        g = self._last_g
        v_g = self.d1(u)[..., None] * g
        if out_dim is not None:
            return v_g[:, out_dim]
        return v_g

    def grad_grad(self, cache_or_x, out_dim=None):
        x = cache_or_x["x"] if isinstance(cache_or_x, dict) else cache_or_x
        self._ensure_cache(x)
        u = self._last_u
        g = self._last_g
        gg = self._last_gg
        # Outer products u' u'^T
        outer = g.unsqueeze(-1) * g.unsqueeze(-2)
        v_gg = self.d1(u)[..., None, None] * gg + self.d2(u)[..., None, None] * outer
        if out_dim is not None:
            return v_gg[:, out_dim]
        return v_gg


class RuleOuterTransformSplitNN(StageBRule):
    """Fallback rule: try to split a stubborn multivariate NN leaf by
    searching over a small set of *outer transforms* T(u) that may render
    the leaf additively / multiplicatively separable.

    This is intentionally placed late in the pipeline (after RuleUniNN),
    because it is more expensive and may propose uglier expressions.

    Currently tries transforms whose inverse is "safe" to represent in the AST:
        - identity:          u = a ⊕ b
        - log:               log(u) = a ⊕ b   =>  u = exp(a) * exp(b)
        - sqrt:              sqrt(u) = a ⊕ b  =>  u = (a ⊕ b)^2

    where ⊕ is either + or * depending on detected separability.
    """

    name = "outer_transform_split_nn"

    # Conservative transform set (extend later if needed)
    _TRANSFORM_NAMES = ("identity", "log", "sqrt")

    @staticmethod
    def _is_separability_like_outer_split(tname: str, op_kind: str) -> bool:
        """Return True only when the lifted split stays separable in original space."""
        if tname == "identity":
            return op_kind in {"add", "mul"}
        if tname == "log":
            return op_kind == "add"
        if tname == "sqrt":
            return op_kind == "mul"
        return False

    def iter_targets(self, ctx: StageBContext):
        return _collect_multivariate_nn_atoms(ctx.state.root)

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        if not isinstance(target, AtomNode):
            return []
        if str(getattr(target, "kind", "")).lower() != "nn":
            return []
        if len(getattr(target, "var_idxs", ()) or ()) < 2:
            return []

        st = ctx.state

        # Build a subtree model for this single NN atom
        atom_to_leaf = build_atom_to_leaf_map(st.root, st.model)
        u_model = _SubtreeModel(root=target, atom_to_leaf=atom_to_leaf)
        u_model = u_model.to(device=ctx.device).to(dtype=ctx.dtype)

        symb = [int(j) for j in target.var_idxs]
        if len(symb) > 10:
            # Avoid combinatorial explosion in check_additivity (3^n). This is a fallback rule.
            return []

        # Sample a modest number of points to define a "valid" domain for transforms.
        max_points = 4096
        xs = []
        n = 0
        for batch in ctx.train_loader_probe:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            xs.append(x.detach().cpu())
            n += x.shape[0]
            if n >= max_points:
                break
        if not xs:
            return []
        X = torch.cat(xs, dim=0)[:max_points]
        Xd = X.to(device=ctx.device, dtype=ctx.dtype)

        with torch.no_grad():
            u_vals = u_model(Xd).view(-1)

        # Pull the requested transform specs.
        specs, y_ops, dy_ops, d2y_ops = get_separability_y_ops(list(self._TRANSFORM_NAMES))

        # Map transform name -> inverse AST builder.
        def _inv_builder(name: str, inner: Node) -> Optional[Node]:
            # identity: u = inner
            if name == "identity":
                return inner
            # log: log(u)=inner => u = exp(inner); for additive split we prefer exp(a)*exp(b)
            if name == "log":
                if isinstance(inner, AddNode):
                    return MulNode(ExpNode(inner.left), ExpNode(inner.right))
                return ExpNode(inner)
            # sqrt: sqrt(u)=inner => u = inner^2
            if name == "sqrt":
                if isinstance(inner, MulNode):
                    return MulNode(PowNode(inner.left, 2.0), PowNode(inner.right, 2.0))
                return PowNode(inner, 2.0)
            return None

        cands: List[Candidate] = []

        # Run separability checks for each transform (on the leaf output)
        for spec, op, d1, d2 in zip(specs, y_ops, dy_ops, d2y_ops):
            tname = str(getattr(spec, "name", ""))
            if tname not in self._TRANSFORM_NAMES:
                continue

            # Domain mask for this transform (avoid NaNs in check_separability)
            with torch.no_grad():
                try:
                    v = op(u_vals)
                    m = torch.isfinite(v)
                    m = m & torch.isfinite(d1(u_vals)) & torch.isfinite(d2(u_vals))
                except Exception:
                    continue

            frac = float(m.float().mean().item())
            if frac < 0.90:
                continue
            if int(m.sum().item()) < 256:
                continue

            X_valid = X[m.cpu()].contiguous()
            datagen = [(X_valid, torch.zeros(X_valid.shape[0], 1))]

            v_model = _OuterTransformedSubtreeModel(u_model, op, d1, d2)

            # Suppress the very chatty prints inside check_separability unless verbose.
            import contextlib
            import io

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    cand_list, _, _, _, _ = check_separability(
                        symb=symb,
                        index=0,
                        model=v_model,
                        datagen=datagen,
                        precision_sum=1e-3,
                        precision_mult=1e-3,
                        device=ctx.device,
                        very_verbose=ctx.verbose_separabilities,
                    )
                except Exception:
                    cand_list = []
            if ctx.verbose and buf.getvalue().strip():
                ctx.log(buf.getvalue().strip())

            if not cand_list:
                continue

            # Determine NN hyperparams from the parent atom
            kw = dict(getattr(target, "kwargs", {}) or {})
            num_segments = int(kw.get("num_segments", 32))
            dual_layer = bool(kw.get("dual_layer", False))
            nn_kwargs = {"num_segments": num_segments, "dual_layer": dual_layer}

            for proposal in cand_list:
                if not proposal or len(proposal) < 3:
                    continue
                sep_op, g1, g2 = proposal[0], proposal[1], proposal[2]
                if not g1 or not g2:
                    continue
                if set(g1) & set(g2):
                    # Skip overlapping/partial splits in this fallback rule.
                    continue

                # Build the inner separable structure in transformed space.
                tag = getattr(target, "tag", None)
                op_kind = "add" if sep_op is torch.add else "mul"
                left = AtomNode(
                    "nn", tuple(g1), kwargs=nn_kwargs, tag=(f"{tag}_L" if tag else None)
                )
                right = AtomNode(
                    "nn", tuple(g2), kwargs=nn_kwargs, tag=(f"{tag}_R" if tag else None)
                )
                inner = AddNode(left, right) if op_kind == "add" else MulNode(left, right)

                new_sub = _inv_builder(tname, inner)
                if new_sub is None:
                    continue

                new_root = replace_atom_in_ast(st.root, target, new_sub)
                lbl = f"outer_{tname}_{op_kind}"
                cands.append(
                    Candidate(
                        lbl,
                        new_root,
                        meta={
                            "log": (
                                f"[Stage B]  Outer-transform split: T={tname}, op={'+' if op_kind == 'add' else '*'}, "
                                f"vars {tuple(symb)} -> {tuple(g1)} | {tuple(g2)} (domain_ok={frac:.2f})"
                            ),
                            "structural": True,
                            "separability_like": self._is_separability_like_outer_split(tname, op_kind),
                        },
                    )
                )

        return cands


def _iter_add_nodes(root: Node) -> List[AddNode]:
    """Collect all AddNode objects in DFS order (object identity preserved)."""
    out: List[AddNode] = []

    def _walk(n: Node):
        if isinstance(n, AddNode):
            out.append(n)
            _walk(n.left)
            _walk(n.right)
        elif isinstance(n, MulNode):
            _walk(n.left)
            _walk(n.right)
        elif isinstance(n, PowNode):
            _walk(n.base)
        elif isinstance(n, _UNARY_AST_NODES):
            _walk(n.arg)
        elif isinstance(n, AtomNode):
            return
        elif isinstance(n, ConstNode):
            return  # Constants have no children
        else:
            raise TypeError(f"Unexpected node type in AST: {type(n)}")

    _walk(root)
    return out


def _iter_mul_nodes(root: Node) -> List[MulNode]:
    """Collect all MulNode objects in DFS order (object identity preserved)."""
    out: List[MulNode] = []

    def _walk(n: Node):
        if isinstance(n, AddNode):
            _walk(n.left)
            _walk(n.right)
        elif isinstance(n, MulNode):
            out.append(n)
            _walk(n.left)
            _walk(n.right)
        elif isinstance(n, PowNode):
            _walk(n.base)
        elif isinstance(n, _UNARY_AST_NODES):
            _walk(n.arg)
        elif isinstance(n, AtomNode):
            return
        elif isinstance(n, ConstNode):
            return  # Constants have no children
        else:
            raise TypeError(f"Unexpected node type in AST: {type(n)}")

    _walk(root)
    return out


def _contains_node(root: Node, target: Node) -> bool:
    """Return True if `target` appears in `root` (by object identity)."""
    if root is target:
        return True
    if isinstance(root, AtomNode):
        return False
    if isinstance(root, ConstNode):
        return False  # Constants have no children
    if isinstance(root, (AddNode, MulNode)):
        return _contains_node(root.left, target) or _contains_node(root.right, target)
    if isinstance(root, PowNode):
        return _contains_node(root.base, target)
    if isinstance(root, _UNARY_AST_NODES):
        return _contains_node(root.arg, target)
    raise TypeError(f"Unexpected node type in AST: {type(root)}")


def _flatten_mul(node: Node) -> List[Node]:
    """Flatten a multiplication tree into a list of factors (left-to-right)."""
    if isinstance(node, MulNode):
        return _flatten_mul(node.left) + _flatten_mul(node.right)
    return [node]


def _rebuild_mul(factors: List[Node]) -> Node:
    """Rebuild a left-associated multiplication tree from factors."""
    if not factors:
        # Rare edge-case: caller removed the only factor.
        return ConstNode(1.0)
    cur = factors[0]
    for f in factors[1:]:
        cur = MulNode(cur, f)
    return cur


def _replace_node_in_ast(root: Node, old_node: Node, new_subtree: Node) -> Node:
    """Pure functional replacement of an arbitrary subtree (by object identity)."""
    if root is old_node:
        return new_subtree
    if isinstance(root, AtomNode):
        return root
    if isinstance(root, ConstNode):
        return root  # Constants are unchanged
    if isinstance(root, AddNode):
        return AddNode(
            left=_replace_node_in_ast(root.left, old_node, new_subtree),
            right=_replace_node_in_ast(root.right, old_node, new_subtree),
        )
    if isinstance(root, MulNode):
        return MulNode(
            left=_replace_node_in_ast(root.left, old_node, new_subtree),
            right=_replace_node_in_ast(root.right, old_node, new_subtree),
        )
    if isinstance(root, PowNode):
        return PowNode(base=_replace_node_in_ast(root.base, old_node, new_subtree), exponent=root.exponent)
    if isinstance(root, _UNARY_AST_NODES):
        return type(root)(arg=_replace_node_in_ast(root.arg, old_node, new_subtree))
    raise TypeError(f"Unexpected node type in AST: {type(root)}")


def _vars_in_subtree_simple(node: Node) -> List[int]:
    """Collect global variable indices used anywhere inside `node`."""
    s: set[int] = set()

    def _walk(n: Node):
        if isinstance(n, AtomNode):
            for j in n.var_idxs:
                s.add(int(j))
        elif isinstance(n, (AddNode, MulNode)):
            _walk(n.left)
            _walk(n.right)
        elif isinstance(n, PowNode):
            _walk(n.base)
        elif isinstance(n, _UNARY_AST_NODES):
            _walk(n.arg)
        elif isinstance(n, ConstNode):
            pass  # Constants have no variables
        else:
            raise TypeError(f"Unexpected node type in AST: {type(n)}")

    _walk(node)
    return sorted(s)


def _subtree_size(node: Node) -> int:
    if isinstance(node, AtomNode):
        return 1
    if isinstance(node, ConstNode):
        return 1  # Constant is a single node
    if isinstance(node, (AddNode, MulNode)):
        return 1 + _subtree_size(node.left) + _subtree_size(node.right)
    if isinstance(node, PowNode):
        return 1 + _subtree_size(node.base)
    if isinstance(node, _UNARY_AST_NODES):
        return 1 + _subtree_size(node.arg)
    raise TypeError(f"Unexpected node type in AST: {type(node)}")


def _eval_subtree_with_leaf_map(node: Node, x: torch.Tensor, atom_to_leaf: dict[int, nn.Module]) -> torch.Tensor:
    """Evaluate a subtree using an id(atom)->leaf map from build_atom_to_leaf_map."""

    def _eval(n: Node) -> torch.Tensor:
        if isinstance(n, AtomNode):
            kind = str(getattr(n, "kind", "")).lower()
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
            if kind in feature_kinds:
                x_in = x
            else:
                # Crucial: handle compound atoms via their [z, extras...] input tensor.
                x_in = _build_atom_input_tensor(n, x)
            leaf = atom_to_leaf.get(id(n), None)
            if leaf is None:
                raise KeyError(f"Missing leaf for atom id={id(n)} tag={getattr(n, 'tag', None)}")
            return leaf(x_in)
        if isinstance(n, AddNode):
            return _eval(n.left) + _eval(n.right)
        if isinstance(n, MulNode):
            return _eval(n.left) * _eval(n.right)
        if isinstance(n, PowNode):
            return _eval(n.base).pow(float(n.exponent))
        if isinstance(n, LogNode):
            return torch.log(_eval(n.arg))
        if isinstance(n, ExpNode):
            return torch.exp(_eval(n.arg))
        if isinstance(n, SinNode):
            return torch.sin(_eval(n.arg))
        if isinstance(n, CosNode):
            return torch.cos(_eval(n.arg))
        if isinstance(n, AsinNode):
            return torch.asin(torch.clamp(_eval(n.arg), -1.0, 1.0))
        if isinstance(n, AcosNode):
            return torch.acos(torch.clamp(_eval(n.arg), -1.0, 1.0))
        if isinstance(n, AtanNode):
            return torch.atan(_eval(n.arg))
        if isinstance(n, AbsNode):
            return torch.abs(_eval(n.arg))
        if isinstance(n, ConstNode):
            B = x.shape[0]
            return torch.full((B, 1), n.value, device=x.device, dtype=x.dtype)
        raise TypeError(f"Unexpected node type in AST: {type(n)}")

    y = _eval(node)
    if y.dim() == 2 and y.shape[1] == 1:
        return y[:, 0]
    return y.view(-1)


def _build_peel_known_factor_candidates(
    *,
    root: Node,
    model: nn.Module,
    target: AtomNode,
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    max_points: int = 4096,
    min_points: int = 256,
    denom_eps: float = 1e-10,
    max_factor_cands: int = 3,
) -> List[Candidate]:
    """If an additive sibling term already contains a multivariate factor F(vars(target)),
    propose factoring it out and replacing the NN leaf with NN'(vars)=NN/F.

    This is a targeted way to handle cases like

        A(x_other)*NN(xS) + B(x_other)*F(xS)

    where NN(xS) itself is not separable, but NN/F is.
    """

    # Allow multi-dataset contexts by defaulting to the first loader.
    if isinstance(train_loader, (list, tuple)):
        if not train_loader:
            return []
        train_loader = train_loader[0]

    my_vars = tuple(int(v) for v in target.var_idxs)
    my_set = set(my_vars)

    # Leaf map from the *current* model (teacher)
    atom_to_leaf = build_atom_to_leaf_map(root, model)
    leaf_u = atom_to_leaf.get(id(target), None)
    if leaf_u is None:
        return []

    out: List[Candidate] = []

    for add in _iter_add_nodes(root):
        # Identify which side contains the target
        if _contains_node(add.left, target):
            u_term = add.left
            other_term = add.right
            u_on_left = True
        elif _contains_node(add.right, target):
            u_term = add.right
            other_term = add.left
            u_on_left = False
        else:
            continue

        # We only support the simple-but-common case where the NN appears as a top-level
        # multiplicative factor of the addend.
        u_factors = _flatten_mul(u_term)
        if not any(f is target for f in u_factors):
            continue

        other_factors = _flatten_mul(other_term)
        if not other_factors:
            continue

        # Candidate factors in the sibling term that depend only on target vars.
        # IMPORTANT: require an *exact* var-set match, and do NOT peel NN factors.
        cand_factors = []
        for fac in other_factors:
            vset = set(_vars_in_subtree_simple(fac))
            if not vset:
                continue
            if vset != my_set:
                continue
            # Avoid peeling learned NN factors; we only want a "known"/analytic factor.
            if any(
                isinstance(a, AtomNode) and str(a.kind).lower() == "nn"
                for a in _collect_all_atoms(fac)
            ):
                continue
            cand_factors.append((fac, vset))

        if not cand_factors:
            continue

        # Prefer exact var-set match; then larger var coverage; then larger subtree.
        cand_factors.sort(
            key=lambda t: (
                0 if t[1] == my_set else 1,
                -len(t[1]),
                -_subtree_size(t[0]),
            )
        )

        for fac, vset in cand_factors[: max_factor_cands]:
            # Remove `fac` from the sibling multiplicative chain (one occurrence).
            reduced_other_factors = []
            removed = False
            for f in other_factors:
                if (not removed) and (f is fac):
                    removed = True
                    continue
                reduced_other_factors.append(f)
            if not removed:
                continue
            other_reduced = _rebuild_mul(reduced_other_factors)

            # Replace target by a fresh NN leaf that should learn target/fac.
            base_tag = (getattr(target, "tag", None) or f"nn_{abs(atom_content_hash(target))%100000}")
            tag_new = f"{base_tag}_div_{abs(_subtree_content_hash(fac))%100000}"
            nn_kwargs = dict(getattr(target, "kwargs", {}) or {})
            parent_inputs = clone_inputs(target)
            new_nn = AtomNode("nn", tuple(target.var_idxs), kwargs=nn_kwargs, tag=tag_new,
                              inputs=parent_inputs)

            # Replace one occurrence of the target factor.
            new_u_factors = []
            replaced = False
            for f in u_factors:
                if (not replaced) and (f is target):
                    new_u_factors.append(new_nn)
                    replaced = True
                else:
                    new_u_factors.append(f)
            if not replaced:
                continue
            u_replaced = _rebuild_mul(new_u_factors)

            # Build factored subtree: fac * (u_term/fac + other_term/fac)
            # Ordering inside the parentheses follows the original add order.
            inside = AddNode(u_replaced, other_reduced) if u_on_left else AddNode(other_reduced, u_replaced)
            new_add_subtree = MulNode(fac, inside)

            cand_root = _replace_node_in_ast(root, add, new_add_subtree)

            # ------------------------------------------------------------
            # Teacher data: y = target(xS) / fac(xS)
            # ------------------------------------------------------------
            xs, ys = [], []
            n_collected = 0
            with torch.no_grad():
                for batch in train_loader:
                    if isinstance(batch, (list, tuple)):
                        x = batch[0]
                    else:
                        x = batch
                    x = x.to(device=device, dtype=dtype)
                    xS = _build_atom_input_tensor(target, x)

                    u = leaf_u(xS)
                    if u.dim() == 2 and u.shape[1] == 1:
                        u = u[:, 0]
                    else:
                        u = u.view(-1)

                    fval = _eval_subtree_with_leaf_map(fac, x, atom_to_leaf)

                    mask = torch.isfinite(u) & torch.isfinite(fval) & (fval.abs() > denom_eps)
                    if mask.any():
                        ratio = (u[mask] / fval[mask]).detach().cpu()
                        xs.append(xS[mask].detach().cpu())
                        ys.append(ratio)
                        n_collected += int(mask.sum().item())

                    if n_collected >= max_points:
                        break

            if n_collected < min_points:
                continue

            X_teacher = torch.cat(xs, dim=0)[:max_points]
            Y_teacher = torch.cat(ys, dim=0)[:max_points]

            # ------------------------------------------------------------
            # Init: quick-fit the new NN leaf to the teacher ratio
            # ------------------------------------------------------------
            def _init_fn(root_new: Node, model_new: nn.Module, *, _tag=tag_new, _X=X_teacher, _Y=Y_teacher):
                try:
                    atom_to_leaf_new = build_atom_to_leaf_map(root_new, model_new)
                    # Find the corresponding AtomNode by tag
                    leaf_new = None
                    for atom in _collect_all_atoms(root_new):
                        if (
                            isinstance(atom, AtomNode)
                            and str(atom.kind).lower() == "nn"
                            and getattr(atom, "tag", None) == _tag
                        ):
                            leaf_new = atom_to_leaf_new.get(id(atom), None)
                            break
                    if leaf_new is None:
                        return

                    params = list(leaf_new.parameters())
                    if not params:
                        return

                    dev = params[0].device
                    dt = params[0].dtype
                    X = _X.to(device=dev, dtype=dt)
                    Y = _Y.to(device=dev, dtype=dt)
                    n = X.shape[0]
                    if n <= 0:
                        return

                    # Small subsample for speed
                    m = min(1024, n)
                    idx = torch.randperm(n, device=dev)[:m]
                    Xs = X[idx]
                    Ys = Y[idx]

                    leaf_new.train()
                    opt = torch.optim.Adam(params, lr=5e-2)
                    for _ in range(80):
                        opt.zero_grad(set_to_none=True)
                        pred = leaf_new(Xs)
                        if pred.dim() == 2 and pred.shape[1] == 1:
                            pred = pred[:, 0]
                        else:
                            pred = pred.view(-1)
                        loss = (pred - Ys).pow(2).mean()
                        if not torch.isfinite(loss):
                            break
                        loss.backward()
                        opt.step()
                    leaf_new.eval()
                except Exception:
                    # Best effort: if init fails, LM will still run from a random init.
                    return

            sig = (atom_content_hash(target), _subtree_content_hash(fac), 91733)
            fac_vars = sorted(vset)
            msg_fac = repr(fac)
            if len(msg_fac) > 80:
                msg_fac = msg_fac[:77] + "..."

            out.append(
                Candidate(
                    "peel_known_factor",
                    cand_root,
                    _init_fn,
                    meta={
                        "log": (
                            f"[Stage B]  Trying peel_known_factor on NN vars={my_vars}: "
                            f"factor vars={tuple(fac_vars)} factor={msg_fac}"
                        ),
                        "structural": True,
                    },
                    signature=sig,
                )
            )

    return out


class RuleNNLeafSeparability(StageBRule):
    """
    Rule for separability analysis on multivariate NN leaves.

    This is a late-stage fallback rule that applies separability analysis
    directly to multivariate NN atoms that haven't been rewritten by earlier rules.
    It enables iterative refinement by detecting separability in stubborn NN leaves
    and proposing Add(NN, NN) or Mul(NN, NN) splits.

    This makes Stage B the single iterative driver, eliminating the need for
    cross-script Stage A ↔ Stage B loops.

    Pattern label: nn_leaf_separability
    """

    name = "nn_leaf_separability"

    def iter_targets(self, ctx: StageBContext):
        """
        Return all multivariate NN atoms in the current AST.

        Note: Always returns targets regardless of run_subtree_separability availability,
        because gauge-aware splits don't require separability detection.
        """
        return _collect_multivariate_nn_atoms(ctx.state.root)

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        """
        Generate separability candidate for a multivariate NN leaf.

        For NNs in an additive context with shared variables (gauge context),
        we directly propose splits without detection, since gauge freedom
        between additive siblings masks the true structure.

        Args:
            ctx: Stage B context
            target: Multivariate NN atom to split

        Returns:
            List of Candidates (gauge splits, detection-based, or outer-transform)
        """
        # Only target multivariate NN atoms
        if not isinstance(target, AtomNode):
            return []
        if str(target.kind).lower() != "nn":
            return []
        if effective_arity(target) < 2:
            return []

        st = ctx.state
        cands: List[Candidate] = []

        # ============================================================
        # 0a) Check for ADDITIVE gauge context: additive siblings with shared vars
        # ============================================================
        add_sibling_nns = _find_nns_in_add_chain(st.root, target)

        if add_sibling_nns:
            my_vars = set(int(v) for v in target.var_idxs)
            sibling_vars = set()
            for sib in add_sibling_nns:
                sibling_vars.update(int(v) for v in sib.var_idxs)

            shared = my_vars & sibling_vars
            unique = my_vars - sibling_vars

            if unique and shared:
                # Additive gauge context detected: propose splits directly
                # Additive gauge masks multiplicative separability → try mul first
                ctx.log(
                    f"[Stage B] Additive gauge context for NN{sorted(my_vars)}: "
                    f"unique={sorted(unique)}, shared={sorted(shared)}"
                )
                ctx.log(
                    "[Stage B] Proposing multiplicative split first (additive gauge masks mul separability)"
                )

                # Get num_segments from target's kwargs if available
                num_segments = target.kwargs.get("num_segments", 16) if target.kwargs else 16
                dual_layer = target.kwargs.get("dual_layer", False) if target.kwargs else False

                gauge_cands = _build_gauge_split_candidates(
                    root=st.root,
                    target=target,
                    unique_vars=unique,
                    shared_vars=shared,
                    context="additive",
                    num_segments=num_segments,
                    dual_layer=dual_layer,
                )
                for label, root_new, meta in gauge_cands:
                    cands.append(Candidate(label, root_new, meta=meta))

                # Return gauge candidates only - no fallback to detection
                return cands

        # ============================================================
        # 0b) Check for MULTIPLICATIVE gauge context: multiplicative siblings
        # ============================================================
        mul_sibling_nns = _find_nns_in_mul_chain(st.root, target)

        if mul_sibling_nns:
            my_vars = set(int(v) for v in target.var_idxs)
            sibling_vars = set()
            for sib in mul_sibling_nns:
                sibling_vars.update(int(v) for v in sib.var_idxs)

            shared = my_vars & sibling_vars
            unique = my_vars - sibling_vars

            if unique and shared:
                # Multiplicative gauge context detected: propose splits directly
                # Multiplicative gauge masks additive separability → try add first
                ctx.log(
                    f"[Stage B] Multiplicative gauge context for NN{sorted(my_vars)}: "
                    f"unique={sorted(unique)}, shared={sorted(shared)}"
                )
                ctx.log(
                    "[Stage B] Proposing additive split first (multiplicative gauge masks add separability)"
                )

                # Get num_segments from target's kwargs if available
                num_segments = target.kwargs.get("num_segments", 16) if target.kwargs else 16
                dual_layer = target.kwargs.get("dual_layer", False) if target.kwargs else False

                gauge_cands = _build_gauge_split_candidates(
                    root=st.root,
                    target=target,
                    unique_vars=unique,
                    shared_vars=shared,
                    context="multiplicative",
                    num_segments=num_segments,
                    dual_layer=dual_layer,
                )
                for label, root_new, meta in gauge_cands:
                    cands.append(Candidate(label, root_new, meta=meta))

                # Return gauge candidates only - no fallback to detection
                return cands

        # ============================================================
        # 1) No gauge context: use standard detection-based probing
        # ============================================================
        if run_subtree_separability is not None:
            cand_root, init_fn = _build_subtree_separability_candidate(
                root=st.root,
                u_node=target,
                model=st.model,
                reuse=st.reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
                very_verbose=ctx.verbose_separabilities,
            )
            if cand_root is not None:
                cands.append(
                    Candidate(
                        "nn_leaf_separability",
                        cand_root,
                        init_fn,
                        meta={
                            "log": f"[Stage B]  Trying separability split on NN vars={target.var_idxs}",
                            "structural": True,
                        },
                    )
                )

        # ============================================================
        # 2) Outer-transform sweep (log/sqrt) on the leaf teacher
        # ============================================================
        if run_subtree_separability is not None:
            infos = _build_subtree_separability_outer_transform_candidates(
                root=st.root,
                u_node=target,
                model=st.model,
                reuse=st.reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
                transforms=("square", "log", "sqrt", "arcsin"),  # square first for sqrt-of-sum patterns
                domain_ok_frac_min=0.90,
                eps=1e-12,
                very_verbose=ctx.verbose_separabilities,
            )
            for label, root_new, init_fn2, meta in infos:
                # Mark outer-transform candidates as structural
                meta_updated = dict(meta) if meta else {}
                meta_updated["structural"] = True
                cands.append(Candidate(label, root_new, init_fn2, meta=meta_updated))


        # ============================================================
        # 3) Peel a known multivariate factor from an additive sibling term
        # ============================================================
        # This targets hidden common-factor cases like:
        #   A(x_other)*NN(xS) + B(x_other)*F(xS)
        # where NN itself is not separable, but NN/F is.
        # We only apply this when F is an explicit, non-NN factor in the sibling addend.
        cands.extend(
            _build_peel_known_factor_candidates(
                root=st.root,
                model=st.model,
                target=target,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
                max_points=4096,
                min_points=256,
                denom_eps=1e-10,
                max_factor_cands=3,
            )
        )

        return cands
