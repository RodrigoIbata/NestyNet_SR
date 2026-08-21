# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Stage-A overlap evaluation, teacher initialization, and local structure probes."""

import math
from typing import List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from nestynet_sr.sr_core import Var, ast_to_human_readable, collect_nn_atoms, replace_atom_in_ast
from nestynet_sr.sr_core.bridges import AddNode, AtomNode, ConstNode, CosNode, ExpNode, LogNode, MulNode, PowNode, SinNode, ast_equals, clone_ast, compound_input_expr, effective_arity, eval_inputs, extra_input_var_idxs, get_input_exprs, has_nontrivial_input, is_trivial_input
from nestynet_sr.sr_core.fit_links import canonical_fit_link_name, fit_link_torch
from .candidate_builders import _build_atom_input_tensor
from .features import _compound_to_probe_target
from .model_builders import build_composite_ast
from .stagea_fit_tournament import fit_initial_model_with_tournament

from ._search_shadow import (
    RED,
    RESET,
    YELLOW,
    _apply_fit_link_to_model,
)

def _flatten_additive_terms(root):
    """Flatten nested AddNodes into a list of additive sub-trees.

    E.g., AddNode(AddNode(A, B), C) -> [A, B, C].
    Non-AddNode roots return [root].
    """
    if isinstance(root, AddNode):
        return _flatten_additive_terms(root.left) + _flatten_additive_terms(root.right)
    return [root]


def _rebuild_additive_chain(terms):
    """Rebuild a left-associative AddNode chain from a list of terms."""
    if not terms:
        return ConstNode(0.0)
    out = terms[0]
    for t in terms[1:]:
        out = AddNode(out, t)
    return out


def _subtree_contains_atom_tag(subtree, tag):
    """Check whether any AtomNode in *subtree* has the given tag."""
    if isinstance(subtree, AtomNode):
        return getattr(subtree, "tag", None) == tag
    for child_attr in ("left", "right", "base", "arg"):
        child = getattr(subtree, child_attr, None)
        if child is not None and _subtree_contains_atom_tag(child, tag):
            return True
    return False


def _is_subtree_fully_decomposed(subtree):
    """Return True if every NN atom in *subtree* has effective_arity <= 1.

    A subtree is "fully decomposed" when all its NN leaves are univariate
    (single raw variable or single compound variable).  Non-NN atoms
    (e.g. FreeConst, scale) are ignored.
    """
    if isinstance(subtree, AtomNode):
        if str(getattr(subtree, "kind", "")).lower() == "nn":
            return int(effective_arity(subtree)) <= 1
        return True  # non-NN atoms (const, scale, etc.) are fine
    for child_attr in ("left", "right", "base", "arg"):
        child = getattr(subtree, child_attr, None)
        if child is not None and not _is_subtree_fully_decomposed(child):
            return False
    return True


@torch.no_grad()
def _eval_ast_subtree_on_data(subtree, tag_to_leaf, x):
    """Evaluate an AST sub-tree on data *x* using trained leaves from *tag_to_leaf*.

    Returns a 1-D tensor of shape [B].
    """

    def _ev(node):
        if isinstance(node, AtomNode):
            leaf = tag_to_leaf.get(getattr(node, "tag", None))
            if leaf is None:
                raise KeyError(
                    f"Missing leaf for atom tag={getattr(node, 'tag', None)}"
                )
            x_in = _build_atom_input_tensor(node, x)
            out = leaf(x_in)
            return out[:, 0] if (out.dim() == 2 and out.shape[1] == 1) else out.view(-1)
        if isinstance(node, ConstNode):
            return torch.full((x.shape[0],), node.value, device=x.device, dtype=x.dtype)
        if isinstance(node, AddNode):
            return _ev(node.left) + _ev(node.right)
        if isinstance(node, MulNode):
            return _ev(node.left) * _ev(node.right)
        if isinstance(node, PowNode):
            return _ev(node.base).pow(float(node.exponent))
        if isinstance(node, LogNode):
            return torch.log(_ev(node.arg))
        if isinstance(node, ExpNode):
            return torch.exp(_ev(node.arg))
        if isinstance(node, SinNode):
            return torch.sin(_ev(node.arg))
        if isinstance(node, CosNode):
            return torch.cos(_ev(node.arg))
        raise TypeError(f"_eval_ast_subtree_on_data: unexpected node {type(node)}")

    return _ev(subtree)


def _find_residual_refit_context(current_ast, atom):
    """Determine whether *atom* should be re-trained on an explicit residual.

    Conditions (all must hold):
    1. The AST root is an additive chain (top-level AddNode).
    2. The atom is a *direct* additive term (not embedded inside a MulNode, etc.).
    3. All other additive terms are fully decomposed (effective_arity <= 1 for
       every NN leaf they contain).

    Returns
    -------
    list[Node] | None
        The list of sibling sub-trees whose outputs should be subtracted from y
        to produce the clean residual.  ``None`` if the conditions are not met.
    """
    add_terms = _flatten_additive_terms(current_ast)
    if len(add_terms) < 2:
        return None

    atom_tag = getattr(atom, "tag", None)
    if atom_tag is None:
        return None

    # Find which additive term is exactly the target atom.
    target_idx = None
    for i, term in enumerate(add_terms):
        if isinstance(term, AtomNode) and getattr(term, "tag", None) == atom_tag:
            target_idx = i
            break

    if target_idx is None:
        # Atom is not a direct additive term (it's inside a MulNode or similar).
        return None

    siblings = [t for i, t in enumerate(add_terms) if i != target_idx]
    if all(_is_subtree_fully_decomposed(sib) for sib in siblings):
        return siblings
    return None


def _build_tag_to_leaf_map(ast_root, model):
    """Build a robust tag -> leaf mapping using DFS order over ALL atoms.

    This is critical for correctness when the AST contains non-NN atoms (e.g.,
    FreeConst from offset-aware multiplicativity). The model's .leaf list is
    in DFS order over ALL atoms, not just NN atoms. Using collect_nn_atoms()
    and enumerating would give wrong indices when FreeConst atoms are present.

    Args:
        ast_root: The AST root node.
        model: The model (typically ASTCompositeAdaptor) with a .leaf attribute.

    Returns:
        Dict mapping tag (str) -> leaf module (nn.Module).
    """
    from nestynet_sr.sr_search.stageB.atom_mapping import _collect_all_atoms

    atoms = _collect_all_atoms(ast_root)
    leaves = list(model.leaf)
    if len(atoms) != len(leaves):
        print(f"[Warning] atom count {len(atoms)} != leaf count {len(leaves)}")

    tag_to_leaf = {}
    for atom, leaf in zip(atoms, leaves):
        tag = getattr(atom, "tag", None)
        if tag is not None:
            tag_to_leaf[tag] = leaf
    return tag_to_leaf


def _find_tagged_nn_atom(ast_root, tag):
    """Return the NN atom carrying ``tag`` or ``None`` if absent."""
    if tag is None:
        return None
    for atom in collect_nn_atoms(ast_root):
        if getattr(atom, "tag", None) == tag:
            return atom
    return None


def _get_parent_leaf_context(parent_model, current_ast, parent_tag):
    """Resolve the parent atom and leaf for a tagged Stage-A split."""
    parent_atom = _find_tagged_nn_atom(current_ast, parent_tag)
    if parent_atom is None:
        return None, None
    tag_to_leaf_parent = _build_tag_to_leaf_map(current_ast, parent_model)
    parent_leaf = tag_to_leaf_parent.get(parent_tag)
    if parent_leaf is None:
        return None, None
    return parent_atom, parent_leaf


def _input_exprs_match(a, b) -> bool:
    """Return True when two effective-input ASTs represent the same coordinate."""
    try:
        return bool(ast_equals(a, b))
    except Exception:
        pass
    try:
        return ast_to_human_readable(a) == ast_to_human_readable(b)
    except Exception:
        return False


def _child_local_indices_in_parent(parent_atom, child_atom) -> Optional[List[int]]:
    """Map child effective inputs to their column indices in the parent leaf input."""
    try:
        parent_inputs = tuple(get_input_exprs(parent_atom))
        child_inputs = tuple(get_input_exprs(child_atom))
    except Exception:
        return None
    out: List[int] = []
    used: set[int] = set()
    for child_inp in child_inputs:
        match_idx = None
        for i, parent_inp in enumerate(parent_inputs):
            if i in used:
                continue
            if _input_exprs_match(child_inp, parent_inp):
                match_idx = int(i)
                break
        if match_idx is None:
            return None
        used.add(match_idx)
        out.append(match_idx)
    return out


def _effective_input_labels(atom, local_idxs) -> List[str]:
    """Compact labels for diagnostics in parent effective-input coordinates."""
    labels: List[str] = []
    inputs = tuple(get_input_exprs(atom))
    for idx in local_idxs:
        try:
            inp = inputs[int(idx)]
            if is_trivial_input(inp):
                labels.append(f"x{int(inp.var_idxs[0])}")
            else:
                labels.append(ast_to_human_readable(inp))
        except Exception:
            labels.append(f"axis{int(idx)}")
    return labels


@torch.no_grad()
def _evaluate_overlap_truth_metric(
    *,
    parent_model,
    current_ast,
    parent_tag,
    g1,
    g2,
    datagen,
    device,
    dtype,
    op,
    offset_value=None,
    max_batches: int = 4,
    anchor_rel_eps: float = 1.0e-8,
):
    """Evaluate a gauge-invariant overlap residual on the current parent leaf.

    The split truth test is performed in function space:

      additive:        F - F(s,u,v0) - F(s,u0,v) + F(s,u0,v0)
      multiplicative:  F - F(s,u,v0) F(s,u0,v) / F(s,u0,v0)

    Returns ``None`` when the split cannot be screened in this helper
    (for example, compound-token groups or missing leaf context).
    """
    if parent_tag is None:
        return None

    parent_atom, parent_leaf = _get_parent_leaf_context(parent_model, current_ast, parent_tag)
    if parent_atom is None or parent_leaf is None:
        return None

    parent_cols = [int(c) for c in getattr(parent_atom, "var_idxs", ())]
    parent_col_set = set(parent_cols)

    try:
        left_cols = [int(c) for c in g1]
        right_cols = [int(c) for c in g2]
    except Exception:
        return None

    if (
        any(c not in parent_col_set for c in left_cols)
        or any(c not in parent_col_set for c in right_cols)
    ):
        return None

    left_set = set(left_cols)
    right_set = set(right_cols)
    shared = left_set & right_set
    if not shared:
        return None

    left_private = [c for c in left_cols if c not in right_set]
    right_private = [c for c in right_cols if c not in left_set]
    if not left_private or not right_private:
        return None

    parent_cols_list = list(parent_cols)
    left_excl_local = [parent_cols_list.index(c) for c in left_private]
    right_excl_local = [parent_cols_list.index(c) for c in right_private]

    dl = datagen() if callable(datagen) else datagen
    if dl is None:
        return None

    residual_chunks = []
    n_total = 0
    n_valid = 0
    batches_seen = 0
    anchor_scale_max = 0.0

    for batch in dl:
        x_full = batch[0] if isinstance(batch, (tuple, list)) else batch
        if x_full is None:
            continue
        x_full = x_full.to(device=device, dtype=dtype).view(x_full.shape[0], -1)
        x_parent = x_full[:, parent_cols_list]

        med = torch.median(x_parent, dim=0).values

        x_for_left = x_parent.clone()
        for idx in right_excl_local:
            x_for_left[:, idx] = med[idx]

        x_for_right = x_parent.clone()
        for idx in left_excl_local:
            x_for_right[:, idx] = med[idx]

        x_anchor = x_parent.clone()
        for idx in left_excl_local:
            x_anchor[:, idx] = med[idx]
        for idx in right_excl_local:
            x_anchor[:, idx] = med[idx]

        f = parent_leaf(x_parent).squeeze(-1)
        f_left = parent_leaf(x_for_left).squeeze(-1)
        f_right = parent_leaf(x_for_right).squeeze(-1)
        f_anchor = parent_leaf(x_anchor).squeeze(-1)

        if offset_value is not None:
            offset = float(offset_value)
            f = f - offset
            f_left = f_left - offset
            f_right = f_right - offset
            f_anchor = f_anchor - offset

        mad = torch.median(torch.abs(f - torch.median(f)))
        rms = torch.sqrt(torch.mean(f ** 2))
        scale_ref = float(max(float(mad), float(rms), 1.0e-12))
        anchor_scale_max = max(anchor_scale_max, scale_ref)

        if op is torch.add:
            resid = f - f_left - f_right + f_anchor
            residual_chunks.append((resid / scale_ref).reshape(-1).detach().cpu())
            n_valid += int(resid.numel())
            n_total += int(resid.numel())
        elif op in (torch.mul, torch.multiply):
            denom_eps = max(float(anchor_rel_eps), 0.0) * scale_ref
            valid = torch.isfinite(f_anchor) & (torch.abs(f_anchor) > denom_eps)
            n_total += int(f_anchor.numel())
            if bool(valid.any()):
                resid = f[valid] - (f_left[valid] * f_right[valid]) / f_anchor[valid]
                residual_chunks.append((resid / scale_ref).reshape(-1).detach().cpu())
                n_valid += int(valid.sum().item())
        else:
            return None

        batches_seen += 1
        if max_batches and batches_seen >= int(max_batches):
            break

    if not residual_chunks or n_valid <= 0:
        return None

    residuals = torch.cat(residual_chunks, dim=0)
    return {
        "normalized_rms": float((residuals ** 2).mean().sqrt()),
        "normalized_peak": float(residuals.abs().max()),
        "valid_fraction": float(n_valid / max(n_total, 1)),
        "n_valid": int(n_valid),
        "n_total": int(n_total),
        "anchor_scale_ref": float(anchor_scale_max),
        "shared": sorted(int(c) for c in shared),
        "left_private": sorted(int(c) for c in left_private),
        "right_private": sorted(int(c) for c in right_private),
    }


def _overlap_truth_metric_is_acceptable(metric, *, op, precision: float, search_hp) -> tuple[bool, float]:
    """Return whether an overlap truth metric is good enough to try fitting."""
    if metric is None:
        return True, float("inf")
    if op is torch.add:
        factor = float(getattr(search_hp, "overlap_truth_add_rms_factor", 5.0))
    else:
        factor = float(getattr(search_hp, "overlap_truth_mul_rms_factor", 5.0))
    tol = max(float(precision), 0.0) * max(factor, 0.0)
    return bool(float(metric["normalized_rms"]) <= tol), float(tol)


class _XTransformDataset(Dataset):
    """Wrap a base Dataset and apply an x-op to the *input* part of each sample."""

    def __init__(self, base_ds, x_op):
        self.base_ds = base_ds
        self.x_op = x_op

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        item = self.base_ds[idx]
        # Common case: (x, y) or (x, y, ...)
        if isinstance(item, tuple):
            x = item[0]
            x2 = self.x_op(x)
            return (x2, *item[1:])
        if isinstance(item, list):
            x = item[0]
            x2 = self.x_op(x)
            return [x2, *item[1:]]
        if isinstance(item, dict):
            # Try the common key name first; otherwise, transform the first tensor-like value.
            if "x" in item:
                out_d = dict(item)
                out_d["x"] = self.x_op(item["x"])
                return out_d
            # Fallback: transform first value
            out_d = dict(item)
            for k, v in item.items():
                out_d[k] = self.x_op(v)
                break
            return out_d
        # Last resort: assume item *is* x
        return self.x_op(item)


def _teacher_init_multiplicative(model, candidate_ast, parent_model, current_ast, datagen, device, dtype, parent_tag=None, offset_value=None):
    """
    Initialize multiplicative factors using teacher profiles from the parent NN.

    For a split NN[x0, x3] -> NN[x0] * NN[x3], we:
    1. Evaluate parent at (x0, x3_median) to get a 1D profile for left factor
    2. Evaluate parent at (x0_median, x3) to get a 1D profile for right factor
    3. Train each child NN briefly on its profile

    This gives the child NNs a much better starting point than random init.

    Args:
        parent_tag: The tag of the parent atom being split (e.g., 'A0_L_R').
                    This is needed to find the correct MulNode when there are
                    multiple multiplicative splits in the AST.
        offset_value: If provided, subtract this from parent output before
                      computing profiles (for offset-multiplicative splits).
    """
    from nestynet_sr.sr_core.bridges import AtomNode, MulNode

    if parent_tag is None:
        return  # Need to know which split to initialize

    # Expected child tags
    expected_left_tag = parent_tag + '_L'
    expected_right_tag = parent_tag + '_R'

    def find_mul_for_parent(node, target_parent_tag):
        """Find MulNode where children have tags derived from target_parent_tag."""
        if isinstance(node, MulNode):
            left_is_nn = isinstance(node.left, AtomNode) and node.left.kind == "nn"
            right_is_nn = isinstance(node.right, AtomNode) and node.right.kind == "nn"
            if left_is_nn and right_is_nn:
                # Check if this is the split we're looking for
                left_tag = node.left.tag
                right_tag = node.right.tag
                if left_tag == expected_left_tag and right_tag == expected_right_tag:
                    return node
        return None

    def search_ast(node, target_parent_tag):
        mul = find_mul_for_parent(node, target_parent_tag)
        if mul is not None:
            return mul
        if isinstance(node, MulNode):
            left_result = search_ast(node.left, target_parent_tag)
            if left_result:
                return left_result
            return search_ast(node.right, target_parent_tag)
        if hasattr(node, 'left'):
            left_result = search_ast(node.left, target_parent_tag)
            if left_result:
                return left_result
        if hasattr(node, 'right'):
            return search_ast(node.right, target_parent_tag)
        if hasattr(node, 'arg'):
            return search_ast(node.arg, target_parent_tag)
        if hasattr(node, 'base'):
            return search_ast(node.base, target_parent_tag)
        return None

    mul_node = search_ast(candidate_ast, parent_tag)
    if mul_node is None:
        print(f"[Teacher init] Could not find MulNode for parent {parent_tag}")
        return  # No multiplicative split found for this parent

    left_tag = mul_node.left.tag
    right_tag = mul_node.right.tag
    if left_tag is None or right_tag is None:
        return

    parent_atom, parent_leaf = _get_parent_leaf_context(parent_model, current_ast, parent_tag)
    if parent_atom is None or parent_leaf is None:
        return

    # Build tag->leaf maps using ALL atoms (robust to FreeConst atoms)
    tag_to_leaf_cand = _build_tag_to_leaf_map(candidate_ast, model)
    left_leaf = tag_to_leaf_cand.get(left_tag)
    right_leaf = tag_to_leaf_cand.get(right_tag)

    if left_leaf is None or right_leaf is None:
        return

    # Get training data
    try:
        dl = datagen() if callable(datagen) else datagen
        batch = next(iter(dl))
        x_full = batch[0].to(device=device, dtype=dtype)
    except (StopIteration, IndexError):
        return

    left_local = _child_local_indices_in_parent(parent_atom, mul_node.left)
    right_local = _child_local_indices_in_parent(parent_atom, mul_node.right)
    if left_local is None or right_local is None:
        print(f"[Teacher init mul] Could not map split children into parent effective inputs for {parent_tag}")
        return

    x_parent, _, _ = eval_inputs(parent_atom, x_full, need_grad=False, need_hess=False)

    # Identify shared vs exclusive effective inputs for overlapping splits.  A
    # discovered coordinate z(x) and a raw x_i are both just parent leaf columns.
    left_cols_set = set(int(i) for i in left_local)
    right_cols_set = set(int(i) for i in right_local)
    right_excl_local = [i for i in right_local if int(i) not in left_cols_set]
    left_excl_local = [i for i in left_local if int(i) not in right_cols_set]

    if left_cols_set & right_cols_set:
        print(f"[Teacher init mul] Overlapping split detected in effective inputs: shared "
              f"{_effective_input_labels(parent_atom, sorted(left_cols_set & right_cols_set))}, "
              f"left_excl={_effective_input_labels(parent_atom, left_excl_local)}, "
              f"right_excl={_effective_input_labels(parent_atom, right_excl_local)}")

    x_left = x_parent[:, left_local]
    x_right = x_parent[:, right_local]

    with torch.no_grad():
        med = torch.median(x_parent, dim=0).values

        # Canonical overlap gauge:
        #   L(s,u) = F(s,u,v0)
        #   R(s,v) = F(s,u0,v) / F(s,u0,v0)
        x_for_left = x_parent.clone()
        for idx in right_excl_local:
            x_for_left[:, idx] = med[idx]
        profile_left = parent_leaf(x_for_left).squeeze(-1)

        x_for_right = x_parent.clone()
        for idx in left_excl_local:
            x_for_right[:, idx] = med[idx]
        profile_right_num = parent_leaf(x_for_right).squeeze(-1)

        x_anchor = x_parent.clone()
        for idx in left_excl_local:
            x_anchor[:, idx] = med[idx]
        for idx in right_excl_local:
            x_anchor[:, idx] = med[idx]
        profile_anchor = parent_leaf(x_anchor).squeeze(-1)

        # Subtract offset if provided (for y = c + f(x_A) * g(x_B) cases)
        if offset_value is not None:
            offset = float(offset_value)
            profile_left = profile_left - offset
            profile_right_num = profile_right_num - offset
            profile_anchor = profile_anchor - offset

        # Use MAD (median absolute deviation) of parent output for scale reference.
        # This is more robust than evaluating at a single point (medians) which
        # can be near zero by chance.
        parent_full_output = parent_leaf(x_parent).squeeze(-1)
        if offset_value is not None:
            parent_full_output = parent_full_output - offset_value
        parent_median = torch.median(parent_full_output)
        parent_mad = torch.median(torch.abs(parent_full_output - parent_median)).item()
        parent_rms = torch.sqrt(torch.mean(parent_full_output ** 2)).item()
        scale_ref = max(parent_rms, parent_mad, 1e-10)

        denom_eps = 1.0e-8 * scale_ref
        denom_safe = torch.where(
            profile_anchor.abs() > denom_eps,
            profile_anchor,
            profile_anchor.sign() * denom_eps + (profile_anchor == 0).to(profile_anchor.dtype) * denom_eps,
        )
        profile_right = profile_right_num / denom_safe

        has_overlap = bool(left_cols_set & right_cols_set)
        if has_overlap:
            # Preserve the canonical anchor R(s, v0) = 1 exactly for overlap splits.
            profile_left_scaled = profile_left
            profile_right_scaled = profile_right
        else:
            # Disjoint multiplicative splits can still benefit from mild scale balancing.
            left_rms = max(torch.sqrt(torch.mean(profile_left ** 2)).item(), 1.0e-10)
            right_rms = max(torch.sqrt(torch.mean(profile_right ** 2)).item(), 1.0e-10)
            pair_balance = (right_rms / left_rms) ** 0.5
            profile_left_scaled = profile_left * pair_balance
            profile_right_scaled = profile_right / pair_balance

    # Now train each child NN briefly on its profile using Adam
    print(f"[Teacher init] Training left factor on {len(x_left)} profile samples...")

    def train_factor_on_profile(leaf, x_in, y_target, n_iters=200, lr=0.01):
        """Train a leaf NN to match a 1D profile using Adam."""
        y_target = y_target.unsqueeze(-1) if y_target.ndim == 1 else y_target
        optimizer = torch.optim.Adam(leaf.parameters(), lr=lr)

        best_loss = float('inf')
        for i in range(n_iters):
            optimizer.zero_grad()
            y_pred = leaf(x_in)
            loss = torch.mean((y_pred - y_target) ** 2)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
            if loss_val < 1e-8:
                break

        return best_loss

    try:
        loss_left = train_factor_on_profile(left_leaf, x_left, profile_left_scaled)
        print(f"[Teacher init] Left factor trained, final loss: {loss_left:.3e}")
    except Exception as e:
        print(f"[Teacher init] Left factor training failed: {e}")

    # Train right factor
    print(f"[Teacher init] Training right factor on {len(x_right)} profile samples...")

    try:
        loss_right = train_factor_on_profile(right_leaf, x_right, profile_right_scaled)
        print(f"[Teacher init] Right factor trained, final loss: {loss_right:.3e}")
    except Exception as e:
        print(f"[Teacher init] Right factor training failed: {e}")

    if bool(left_cols_set & right_cols_set):
        print("[Teacher init] Overlap canonical gauge preserved; skipping post-init scale rebalance")
        return

    # Final gauge normalization - balance scales using RMS
    with torch.no_grad():
        left_out = left_leaf(x_left)
        right_out = right_leaf(x_right)
        # Use RMS for more robust scale estimation
        left_scale = torch.sqrt(torch.mean(left_out ** 2)).item()
        right_scale = torch.sqrt(torch.mean(right_out ** 2)).item()

        if left_scale > 1e-10 and right_scale > 1e-10:
            ratio = max(left_scale, right_scale) / min(left_scale, right_scale)
            print(f"[Teacher init] Scale check: left_rms={left_scale:.3e}, right_rms={right_scale:.3e}, ratio={ratio:.2f}")
            geom_mean = (left_scale * right_scale) ** 0.5
            # Only rescale if significantly imbalanced (more than 1.5x ratio)
            if ratio > 1.5:
                left_factor = geom_mean / left_scale
                right_factor = geom_mean / right_scale

                def rescale_leaf(leaf, factor, label=""):
                    # For Segmented_Base_Model / G_Model:
                    #   f(x) = Σ_w a_w * Softplus(x·K_w + b_w)
                    # Output rescaling must act on `a` only. `b` is an internal bias
                    # inside Softplus and scaling it would distort the function.

                    def _scale_base(b, fac, route):
                        scaled = False
                        with torch.no_grad():
                            if hasattr(b, 'a_fit') and b.a_fit is not None:
                                b.a_fit.data.mul_(fac)
                                scaled = True
                                print(f"[rescale_leaf] {label}: scaled a_fit via {route}")
                            if hasattr(b, 'a_pieces_fixed'):
                                for i, a_piece in enumerate(b.a_pieces_fixed):
                                    a_piece.mul_(fac)
                                scaled = True
                                print(f"[rescale_leaf] {label}: scaled a_pieces_fixed[{i}] via {route}")
                        return scaled

                    print(f"[rescale_leaf] {label}: leaf type={type(leaf).__name__}, factor={factor:.4f}")

                    # Prefer a dedicated API if the adaptor provides one.
                    if hasattr(leaf, 'scale_output') and callable(getattr(leaf, 'scale_output')):
                        try:
                            leaf.scale_output(factor)
                            print(f"[rescale_leaf] {label}: used scale_output() API")
                            return
                        except Exception as e:
                            print(f"[rescale_leaf] {label}: scale_output() failed: {e}")

                    # Common case: SegmentedAdaptor exposes `base_model`.
                    base = getattr(leaf, 'base_model', None)
                    if base is not None:
                        if _scale_base(base, factor, "base_model"):
                            return
                        print(f"[rescale_leaf] {label}: base_model exists but no a_fit/a_pieces_fixed")

                    # Heuristic for stacked/dual adaptors: try to locate the *final*
                    # stage (whose `a` coefficients scale the overall output).
                    # DualSegmentedAdaptor uses stage0/stage1; stage1 is the output stage.
                    for attr in ('stage1', 'seg2', 'stage2', 'adaptor2', 'model2', 'net2',
                                 'second', 'right', 'tail', 'last'):
                        sub = getattr(leaf, attr, None)
                        if sub is None:
                            continue
                        print(f"[rescale_leaf] {label}: found attr '{attr}' -> {type(sub).__name__}")
                        sub_base = getattr(sub, 'base_model', sub)
                        if hasattr(sub_base, 'a_fit') or hasattr(sub_base, 'a_pieces_fixed'):
                            if _scale_base(sub_base, factor, f"{attr}.base_model"):
                                return

                    # Fallback: try scaling the leaf object itself if it quacks like a base model.
                    print(f"[rescale_leaf] {label}: fallback to scaling leaf directly")
                    if not _scale_base(leaf, factor, "leaf (fallback)"):
                        print(f"[rescale_leaf] {label}: WARNING - no scaling applied!")

                rescale_leaf(left_leaf, left_factor, label="left")
                rescale_leaf(right_leaf, right_factor, label="right")
                print(f"[Teacher init] Final gauge norm: left x{left_factor:.2f}, right x{right_factor:.2f}")
            else:
                print(f"[Teacher init] Scales balanced (ratio={ratio:.2f} <= 1.5), no rescaling needed")
        else:
            print(f"[Teacher init] Scale too small (left={left_scale:.3e}, right={right_scale:.3e}), skipping gauge norm")


def _teacher_init_additive(model, candidate_ast, parent_model, current_ast, datagen, device, dtype, parent_tag=None):
    """
    Initialize additive factors using teacher profiles from the parent NN.

    For a split NN[x0, x3] -> NN[x0] + NN[x3], we:
    1. Compute constant: c = parent(x0_median, x3_median)
    2. Evaluate parent at (x0, x3_median) - c to get left profile
    3. Evaluate parent at (x0_median, x3) - c to get right profile
    4. Train each child NN briefly on its profile

    This gives the child NNs a much better starting point than random init.

    Args:
        parent_tag: The tag of the parent atom being split (e.g., 'A0_L_R').
                    This is needed to find the correct AddNode when there are
                    multiple additive splits in the AST.
    """
    from nestynet_sr.sr_core.bridges import AddNode, AtomNode

    if parent_tag is None:
        return  # Need to know which split to initialize

    # Expected child tags
    expected_left_tag = parent_tag + '_L'
    expected_right_tag = parent_tag + '_R'

    def find_add_for_parent(node, target_parent_tag):
        """Find AddNode where children have tags derived from target_parent_tag."""
        if isinstance(node, AddNode):
            left_is_nn = isinstance(node.left, AtomNode) and node.left.kind == "nn"
            right_is_nn = isinstance(node.right, AtomNode) and node.right.kind == "nn"
            if left_is_nn and right_is_nn:
                # Check if this is the split we're looking for
                left_tag = node.left.tag
                right_tag = node.right.tag
                if left_tag == expected_left_tag and right_tag == expected_right_tag:
                    return node
        return None

    def search_ast(node, target_parent_tag):
        add = find_add_for_parent(node, target_parent_tag)
        if add is not None:
            return add
        if isinstance(node, AddNode):
            left_result = search_ast(node.left, target_parent_tag)
            if left_result:
                return left_result
            return search_ast(node.right, target_parent_tag)
        if hasattr(node, 'left'):
            left_result = search_ast(node.left, target_parent_tag)
            if left_result:
                return left_result
        if hasattr(node, 'right'):
            return search_ast(node.right, target_parent_tag)
        if hasattr(node, 'arg'):
            return search_ast(node.arg, target_parent_tag)
        if hasattr(node, 'base'):
            return search_ast(node.base, target_parent_tag)
        return None

    add_node = search_ast(candidate_ast, parent_tag)
    if add_node is None:
        print(f"[Teacher init add] Could not find AddNode for parent {parent_tag}")
        return  # No additive split found for this parent

    left_tag = add_node.left.tag
    right_tag = add_node.right.tag
    if left_tag is None or right_tag is None:
        return

    parent_atom, parent_leaf = _get_parent_leaf_context(parent_model, current_ast, parent_tag)
    if parent_atom is None or parent_leaf is None:
        return

    # Build tag->leaf maps using ALL atoms (robust to FreeConst atoms)
    tag_to_leaf_cand = _build_tag_to_leaf_map(candidate_ast, model)
    left_leaf = tag_to_leaf_cand.get(left_tag)
    right_leaf = tag_to_leaf_cand.get(right_tag)

    if left_leaf is None or right_leaf is None:
        return

    # Get training data
    try:
        dl = datagen() if callable(datagen) else datagen
        batch = next(iter(dl))
        x_full = batch[0].to(device=device, dtype=dtype)
    except (StopIteration, IndexError):
        return

    left_local = _child_local_indices_in_parent(parent_atom, add_node.left)
    right_local = _child_local_indices_in_parent(parent_atom, add_node.right)
    if left_local is None or right_local is None:
        print(f"[Teacher init add] Could not map split children into parent effective inputs for {parent_tag}")
        return

    x_parent, _, _ = eval_inputs(parent_atom, x_full, need_grad=False, need_hess=False)

    # Identify shared vs exclusive effective inputs for overlapping splits.
    left_cols_set = set(int(i) for i in left_local)
    right_cols_set = set(int(i) for i in right_local)
    shared_vars = left_cols_set & right_cols_set

    right_excl_local = [i for i in right_local if int(i) not in left_cols_set]
    left_excl_local = [i for i in left_local if int(i) not in right_cols_set]

    if shared_vars:
        print(f"[Teacher init add] Overlapping split detected in effective inputs: shared "
              f"{_effective_input_labels(parent_atom, sorted(shared_vars))}, "
              f"left_excl={_effective_input_labels(parent_atom, left_excl_local)}, "
              f"right_excl={_effective_input_labels(parent_atom, right_excl_local)}")

    x_left = x_parent[:, left_local]
    x_right = x_parent[:, right_local]

    with torch.no_grad():
        med = torch.median(x_parent, dim=0).values

        # Canonical overlap gauge:
        #   L(s,u) = F(s,u,v0)
        #   R(s,v) = F(s,u0,v) - F(s,u0,v0)
        x_for_left = x_parent.clone()
        for idx in right_excl_local:
            x_for_left[:, idx] = med[idx]
        profile_left = parent_leaf(x_for_left).squeeze(-1)

        x_for_right = x_parent.clone()
        for idx in left_excl_local:
            x_for_right[:, idx] = med[idx]
        profile_right = parent_leaf(x_for_right).squeeze(-1)

        x_anchor = x_parent.clone()
        for idx in left_excl_local:
            x_anchor[:, idx] = med[idx]
        for idx in right_excl_local:
            x_anchor[:, idx] = med[idx]
        profile_anchor = parent_leaf(x_anchor).squeeze(-1)

        profile_right = profile_right - profile_anchor

    # Now train each child NN briefly on its profile using Adam
    print(f"[Teacher init add] Training left factor on {len(x_left)} profile samples...")

    def train_factor_on_profile(leaf, x_in, y_target, n_iters=200, lr=0.01):
        """Train a leaf NN to match a 1D profile using Adam."""
        y_target = y_target.unsqueeze(-1) if y_target.ndim == 1 else y_target
        optimizer = torch.optim.Adam(leaf.parameters(), lr=lr)

        best_loss = float('inf')
        for i in range(n_iters):
            optimizer.zero_grad()
            y_pred = leaf(x_in)
            loss = torch.mean((y_pred - y_target) ** 2)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
            if loss_val < 1e-8:
                break

        return best_loss

    try:
        loss_left = train_factor_on_profile(left_leaf, x_left, profile_left)
        print(f"[Teacher init add] Left factor trained, final loss: {loss_left:.3e}")
    except Exception as e:
        print(f"[Teacher init add] Left factor training failed: {e}")

    # Train right factor
    print(f"[Teacher init add] Training right factor on {len(x_right)} profile samples...")

    try:
        loss_right = train_factor_on_profile(right_leaf, x_right, profile_right)
        print(f"[Teacher init add] Right factor trained, final loss: {loss_right:.3e}")
    except Exception as e:
        print(f"[Teacher init add] Right factor training failed: {e}")


def _build_additive_gauge_fix_factories(
    temp_model, candidate_ast, g1, g2, parent_tag, datagen, device, dtype,
    weight=1.0,
):
    """Build gauge-fix factories for an overlapping additive split.

    When g1 and g2 share variables, an additive decomposition has gauge
    freedom: any φ(x_shared) can shift between the two leaves.  To pin
    the gauge we constrain ONE leaf to be ≈ 0 when its private variables
    are at their median values.

    We fix the leaf with the fewest private variables (it is most
    "contaminated" by the shared part, so pinning it is cheapest).

    Returns a (possibly empty) list of ResidualsModule factories.
    """
    from nestynet_sr.sr_search.stageB.atom_mapping import _collect_all_atoms
    from nestynet_sr.adaptors.gauge_fix_adaptor import build_gauge_fix_factory

    g1_set, g2_set = set(g1), set(g2)
    shared = g1_set & g2_set
    if not shared:
        return []  # disjoint — no gauge freedom

    left_private = list(g1_set - g2_set)
    right_private = list(g2_set - g1_set)
    if not left_private and not right_private:
        return []  # fully overlapping — no private vars to anchor

    # Choose which leaf to constrain: the one with fewer private vars.
    # If tied, constrain the right leaf (arbitrary but consistent).
    if left_private and (not right_private or len(left_private) < len(right_private)):
        fix_tag = parent_tag + "_L"
        fix_private = left_private
    else:
        fix_tag = parent_tag + "_R"
        fix_private = right_private

    # Get x_train from dataloader
    try:
        dl = datagen() if callable(datagen) else datagen
        batch = next(iter(dl))
        x_train = batch[0].to(device=device, dtype=dtype)
    except (StopIteration, IndexError):
        return []

    # Find leaf index by tag
    atoms = _collect_all_atoms(candidate_ast)
    fix_idx = None
    fix_atom = None
    for i, atom in enumerate(atoms):
        if getattr(atom, "tag", None) == fix_tag:
            fix_idx = i
            fix_atom = atom
            break

    if fix_idx is None or fix_atom is None:
        return []

    factory = build_gauge_fix_factory(
        temp_model, fix_idx, fix_atom, fix_private,
        x_train, device, dtype, weight=weight,
    )
    if factory is None:
        return []

    # Report reference values from the wrapper
    wrapper = getattr(factory, "_gauge_wrapper", None)
    ref_str = ""
    if wrapper is not None:
        ref = wrapper._ref_values.tolist()
        ref_str = f", ref_values={[f'{v:.4f}' for v in ref]}"

    # Pre-training gauge magnitude
    pre_rms = ""
    x_leaf = getattr(factory, "_gauge_x_leaf", None)
    if wrapper is not None and x_leaf is not None:
        with torch.no_grad():
            out = wrapper._evaluate_leaf(x_leaf, raw=True)
            pre_rms = f", pre-train raw RMS={float((out**2).mean().sqrt()):.3e}"

    print(
        f"[Gauge fix (additive)] Constraining leaf '{fix_tag}' "
        f"(private vars {fix_private}, shared {sorted(shared)}, "
        f"weight={weight:.2f}{ref_str}{pre_rms})"
    )
    return [factory]


def _build_multiplicative_gauge_fix_factories(
    temp_model, candidate_ast, g1, g2, parent_tag, datagen, device, dtype,
    weight=1.0,
):
    """Build gauge-fix factories for an overlapping multiplicative split.

    When g1 and g2 share variables, a multiplicative decomposition has gauge
    freedom: any h(x_shared) can move between factors as f*h, g/h.  To pin
    the gauge we constrain ONE leaf to be approximately *constant* when its
    private variables are at their median values.

    We fix the leaf with the fewest private variables (it is most
    "contaminated" by the shared part, so pinning it is cheapest).

    Returns a (possibly empty) list of ResidualsModule factories.
    """
    from nestynet_sr.sr_search.stageB.atom_mapping import _collect_all_atoms
    from nestynet_sr.adaptors.gauge_fix_adaptor import build_gauge_fix_factory

    g1_set, g2_set = set(g1), set(g2)
    shared = g1_set & g2_set
    if not shared:
        return []  # disjoint — no gauge freedom

    left_private = list(g1_set - g2_set)
    right_private = list(g2_set - g1_set)
    if not left_private and not right_private:
        return []  # fully overlapping — no private vars to anchor

    # Choose which leaf to constrain: the one with fewer private vars.
    # If tied, constrain the right leaf (arbitrary but consistent).
    if left_private and (not right_private or len(left_private) < len(right_private)):
        fix_tag = parent_tag + "_L"
        fix_private = left_private
    else:
        fix_tag = parent_tag + "_R"
        fix_private = right_private

    # Get x_train from dataloader
    try:
        dl = datagen() if callable(datagen) else datagen
        batch = next(iter(dl))
        x_train = batch[0].to(device=device, dtype=dtype)
    except (StopIteration, IndexError):
        return []

    # Find leaf index by tag
    atoms = _collect_all_atoms(candidate_ast)
    fix_idx = None
    fix_atom = None
    for i, atom in enumerate(atoms):
        if getattr(atom, "tag", None) == fix_tag:
            fix_idx = i
            fix_atom = atom
            break

    if fix_idx is None or fix_atom is None:
        return []

    factory = build_gauge_fix_factory(
        temp_model, fix_idx, fix_atom, fix_private,
        x_train, device, dtype, weight=weight,
        mode="multiplicative",
    )
    if factory is None:
        return []

    # Report reference values from the wrapper
    wrapper = getattr(factory, "_gauge_wrapper", None)
    ref_str = ""
    if wrapper is not None:
        ref = wrapper._ref_values.tolist()
        ref_str = f", ref_values={[f'{v:.4f}' for v in ref]}"

    # Pre-training gauge magnitude
    pre_rms = ""
    x_leaf = getattr(factory, "_gauge_x_leaf", None)
    if wrapper is not None and x_leaf is not None:
        with torch.no_grad():
            out = wrapper._evaluate_leaf(x_leaf, raw=True)
            pre_rms = f", pre-train raw RMS={float((out**2).mean().sqrt()):.3e}"

    print(
        f"[Gauge fix (multiplicative)] Constraining leaf '{fix_tag}' "
        f"(private vars {fix_private}, shared {sorted(shared)}, "
        f"weight={weight:.2f}{ref_str}{pre_rms})"
    )
    return [factory]


def _overlap_gauge_stage_is_feasible(
    baseline_val_loss: float,
    stage_val_loss: float,
    accept_threshold: float,
    baseline_gauge_rms: float,
    stage_gauge_rms: float,
    max_data_regress_factor: float = 10.0,
    required_improve_factor: float = 0.3,
    tiny_baseline_relax_factor: float = 1.25,
    tiny_baseline_eps: float = 1.0e-10,
) -> tuple[bool, float, float]:
    """Decide whether a non-zero gauge stage is acceptable.

    The stage must preserve the data fit relative to both the global
    acceptance threshold and the ungauged warm-start fit, and it must improve
    the raw gauge metric unless the ungauged metric was already tiny.
    """
    eps = max(float(tiny_baseline_eps), 0.0)
    base_val = max(float(baseline_val_loss), eps)
    data_cap = min(float(accept_threshold), float(max_data_regress_factor) * base_val)

    base_gauge = max(float(baseline_gauge_rms), 0.0)
    if base_gauge <= eps:
        gauge_cap = base_gauge * float(tiny_baseline_relax_factor) + eps
    else:
        gauge_cap = float(required_improve_factor) * base_gauge

    ok = float(stage_val_loss) <= data_cap and float(stage_gauge_rms) <= gauge_cap
    return bool(ok), float(data_cap), float(gauge_cap)


def _compute_atom_scale(parent_model, current_ast, datagen, parent_tag, device, dtype):
    """
    Compute affine scaling: y_sub ≈ α * u + β
    where y_sub is frozen submodel output and u is parent atom output.

    When checking separability for an atom that is nested inside a multiplicative
    or additive structure, the offset detected is in the submodel's output units,
    not the atom's own output units. This function computes the affine relationship
    between the two so we can convert offsets correctly.

    Returns (alpha, beta) or (1.0, 0.0) if can't be computed.
    """
    # Find parent atom
    nn_atoms = collect_nn_atoms(current_ast)
    parent_atom = None
    for atom in nn_atoms:
        if atom.tag == parent_tag:
            parent_atom = atom
            break
    if parent_atom is None:
        return 1.0, 0.0

    # Get parent leaf
    tag_to_leaf = _build_tag_to_leaf_map(current_ast, parent_model)
    parent_leaf = tag_to_leaf.get(parent_tag)
    if parent_leaf is None:
        return 1.0, 0.0

    # Get data
    try:
        dl = datagen() if callable(datagen) else datagen
        batch = next(iter(dl))
        x_full = batch[0].to(device=device, dtype=dtype)
    except (StopIteration, IndexError):
        return 1.0, 0.0

    with torch.no_grad():
        parent_cols = list(parent_atom.var_idxs)
        Nx = x_full.size(1)
        mask_parent = torch.zeros(Nx, dtype=torch.bool, device=x_full.device)
        mask_parent[parent_cols] = True

        # Freeze all non-parent variables to medians (constants)
        # This ensures α and β are computed with fixed context for nested atoms
        x_fixed = x_full.clone()
        med = torch.median(x_full, dim=0).values
        x_fixed[:, ~mask_parent] = med[~mask_parent].unsqueeze(0)

        # y_sub: frozen submodel output with non-parent vars frozen
        y_sub = parent_model(x_fixed).squeeze(-1)

        # u: parent atom output (same frozen context)
        x_parent = x_fixed[:, parent_cols]
        u = parent_leaf(x_parent).squeeze(-1)

        # Fit y_sub ≈ α * u + β via least squares
        # [u, 1] @ [α, β]ᵀ = y_sub
        ones = torch.ones_like(u)
        A = torch.stack([u, ones], dim=1)  # [N, 2]
        b = y_sub  # [N]

        # Solve via normal equations (robust enough for this)
        ATA = A.T @ A  # [2, 2]
        ATb = A.T @ b  # [2]
        try:
            coeffs = torch.linalg.solve(ATA, ATb)  # [α, β]
            alpha = coeffs[0].item()
            beta = coeffs[1].item()
        except Exception:
            # Fallback to median ratio (simple version)
            safe_mask = torch.abs(u) > 1e-6
            if safe_mask.sum() > 10:
                alpha = torch.median(y_sub[safe_mask] / u[safe_mask]).item()
            else:
                alpha = 1.0
            beta = 0.0

        # Reliability check: is the affine fit actually good?
        residual = y_sub - (alpha * u + beta)
        y_spread = torch.median(torch.abs(y_sub - torch.median(y_sub)))
        rel_resid = torch.median(torch.abs(residual)) / (y_spread + 1e-12)
        if rel_resid > 0.05:
            return 1.0, 0.0  # affine assumption doesn't hold

        return alpha, beta


def _snap_omega(omega: float):
    """Snap omega to a simple nearby value when it is clearly close."""
    if omega is None:
        return 1.0
    try:
        w = float(omega)
    except Exception:
        return omega
    # Keep candidate list centralised to avoid drift across Stage-A/B rules.
    try:
        from .feature_grammar import OMEGA_SNAP_CANDS, snap_to_scales

        return snap_to_scales(w, OMEGA_SNAP_CANDS, rel_tol=0.25, abs_tol=0.25)
    except Exception:
        # Fallback to local list.
        cands = [0.5, 1.0, 2.0, math.pi / 2, math.pi, 2.0 * math.pi]
        best = min(cands, key=lambda c: abs(w - c))
        if abs(w - best) <= max(0.25, 0.25 * abs(best)):
            return best
    return w


def _compound_exponents_ratio_like(exponents) -> bool:
    """Heuristic: treat a compound monomial as ratio-like if it mixes +/- exponents."""
    try:
        exps = [float(e) for e in (exponents or ())]
    except Exception:
        return False
    has_pos = any(e > 0 for e in exps)
    has_neg = any(e < 0 for e in exps)
    return bool(has_pos and has_neg)



def _build_xtransformed_loaders(dataset_train, dataset_val, data_hp, x_op):
    train_dl = DataLoader(
        _XTransformDataset(dataset_train, x_op),
        batch_size=data_hp.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )
    val_dl = DataLoader(
        _XTransformDataset(dataset_val, x_op),
        batch_size=data_hp.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )
    return train_dl, val_dl


# -----------------------------------------------------------------------------
# Stage A x-preconditioning helpers
# -----------------------------------------------------------------------------


def _nn_multivar_axes(ast):
    """Collect axes that are part of truly multivariate NN atoms.

    Uses effective_arity to exclude compound atoms (which have multiple var_idxs
    but operate on a scalar compound variable like z=x1*x2).
    """
    axes = set()
    for a in collect_nn_atoms(ast):
        if effective_arity(a) > 1:  # Use effective_arity, not len(var_idxs)
            axes.update(list(a.var_idxs))
    return axes

def _axis_is_inside_compound_input(ast, axis: int) -> bool:
    """True if `axis` is consumed inside a compound atom's input_expr.

    For compound NN atoms, `var_idxs` still lists the original raw input axes,
    while `kwargs['extra_var_idxs']` lists axes that remain as explicit inputs
    in addition to the compound scalar `z`. Any remaining axes are *inside*
    the compound expression and should generally not be targeted by per-axis
    x-preconditioning (e.g. x2 -> cos(ω x2)), since that tends to fight the
    already-learned compound mapping.
    """
    for a in collect_nn_atoms(ast):
        try:
            if not has_nontrivial_input(a):
                continue
            if axis not in a.var_idxs:
                continue
            extras = extra_input_var_idxs(a)
            extra_set = set(int(i) for i in extras)
            if axis not in extra_set:
                return True
        except Exception:
            continue
    return False

def _axis_is_coupled_by_invariance(axis: int, feats) -> bool:
    """Heuristic: avoid per-axis trig replacement when axis participates in a
    multi-axis approximate invariance direction (likely a coupled argument such as x2-x3).
    """
    if not feats:
        return False
    for f in feats:
        if getattr(f, 'kind', None) != 'integer_linear':
            continue
        coeffs = getattr(f, 'coeffs', None)
        if coeffs is None:
            continue
        if axis >= len(coeffs):
            continue
        if abs(float(coeffs[axis])) < 0.5:
            continue
        # coupled if any other coefficient is also non-zero-ish
        for j, c in enumerate(coeffs):
            if j == axis:
                continue
            if abs(float(c)) >= 0.5:
                return True
    return False


def _extract_compound_targets_from_ast(root) -> list:
    """Extract compound variable targets from AST's NN atoms for trig probing.

    Iterates over all NN atoms in the AST and extracts any compound variables
    (input_expr) that can be used as trig probe targets.

    Returns
    -------
    list[TrigProbeTarget]
        List of compound probe targets (excludes trivial single-variable compounds).
    """
    targets = []
    seen_names: set = set()

    for atom in collect_nn_atoms(root):
        if not has_nontrivial_input(atom):
            continue
        z_expr = compound_input_expr(atom)

        # Try to convert compound expression to a TrigProbeTarget
        target = _compound_to_probe_target(z_expr, tuple(atom.var_idxs))
        if target is None:
            continue

        # Skip trivial compounds (single variables)
        if target.kind == "trivial":
            continue

        # Skip duplicates
        if target.name in seen_names:
            continue
        seen_names.add(target.name)

        targets.append(target)

    return targets


def _propose_reciprocal_x_map(scale_specs, multivar_axes, x_transform_map, tol=0.35):
    """Propose per-axis x-transforms based on detected scaling exponents.

    Currently supports:
      k ~ -1  -> recip(x)
      k ~ -2  -> 1/x^2 via (square -> recip)

    Only proposes transforms for axes that are still inside a multivariate NN leaf,
    and not already transformed.
    """
    if not scale_specs:
        return {}
    out = {}
    for sp in scale_specs:
        idxs = getattr(sp, 'indices', None)
        if not idxs or len(idxs) != 1:
            continue
        j = int(idxs[0])
        if j not in multivar_axes:
            continue
        if j in x_transform_map:
            continue
        k = float(getattr(sp, 'k_hat', 0.0))
        if abs(k + 1.0) <= tol:
            pipe = [{'kind': 'recip'}]
        elif abs(k + 2.0) <= tol:
            pipe = [{'kind': 'square'}, {'kind': 'recip'}]
        else:
            continue
        out[j] = {'pipeline': pipe, 'meta': {'source': 'stageA_precondition', 'k_hat': k}}
    return out


def _loader_all_finite(dl) -> bool:
    """Check if all data in a dataloader is finite (no NaN/Inf)."""
    for batch in dl:
        xb = batch[0] if isinstance(batch, (tuple, list)) else batch
        if not torch.isfinite(xb).all():
            return False
    return True


def _compute_y_med_mad_from_loader(
    dl,
    device,
    *,
    fit_y_link: Optional[str] = None,
    fit_y_link_scale: float = 1.0,
):
    """Compute median and MAD of y values from a DataLoader.

    If fit_y_link is set, statistics are computed in the transformed (fit-link) space,
    which is appropriate for MAD-based loss scaling when using a fit-link.
    """
    ys = []
    for batch in dl:
        if isinstance(batch, (list, tuple)):
            _, y = batch
        else:
            y = batch
        y = y.to(device)

        # Optional fit-only output link: compute scale statistics in the same
        # space as the optimiser residuals.
        link = canonical_fit_link_name(fit_y_link)
        if link is not None:
            y = fit_link_torch(y, link, float(fit_y_link_scale))

        # assume scalar target per sample; keep first column if 2D
        if y.dim() > 1:
            y = y[:, 0]
        ys.append(y.detach().cpu())
    if not ys:
        return None, None
    y_all = torch.cat(ys, dim=0)
    med = torch.median(y_all)
    mad = torch.median(torch.abs(y_all - med))
    return float(med), float(mad)


def _eval_yspace_mse(model: torch.nn.Module, dl, device: torch.device) -> float:
    """Compute MSE in *model output space* (i.e., pre fit-link).

    Important: This must NOT apply the fit-link transform. The fit-link is only
    used for residual conditioning in the optimiser; forward() remains in the
    modelling space (typically y_op-space).

    This is used as a sanity gate when fit_y_link='asinh', because asinh can
    make very poor y-space fits look deceptively good in fit-space.
    """
    if model is None or dl is None:
        return float("inf")
    model.eval()
    se_sum = 0.0
    n_total = 0
    with torch.no_grad():
        for batch in dl:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                continue
            x, y = batch[0], batch[1]
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            if y_pred.dim() == 2:
                y_pred = y_pred[:, 0]
            else:
                y_pred = y_pred.view(-1)
            if y.dim() == 2:
                y_true = y[:, 0]
            else:
                y_true = y.view(-1)
            diff = y_pred - y_true
            se_sum += float((diff * diff).sum().detach().cpu())
            n_total += int(diff.numel())
    if n_total <= 0:
        return float("inf")
    return se_sum / float(n_total)


def _asinh_yspace_scale_from_loader(
    dl, device: torch.device, s: float, q: float = 0.90, max_points: int = 20000
) -> float:
    """Compute D_ref = quantile_q(s^2 + y^2) for asinh-space ↔ y-space scaling.

    For small errors, the asinh residual behaves like:
        r_asinh ≈ (Δy) / sqrt(s^2 + y^2)
    so:
        E[r_asinh^2] ≈ E[(Δy)^2 / (s^2 + y^2)].

    Therefore a natural y-space scale for an asinh-space MSE is:
        y_mse ≈ asinh_mse * (typical s^2 + y^2).

    We use a high quantile of (s^2 + y^2) as a robust "typical" scale.
    """
    ys = []
    with torch.no_grad():
        for batch in dl:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                continue
            y = batch[1].to(device)
            if y.dim() == 2:
                y = y[:, 0]
            ys.append(y.detach().flatten())
            if sum(t.numel() for t in ys) >= int(max_points):
                break
    if not ys:
        return float("nan")
    y_all = torch.cat(ys, dim=0)
    D = (float(s) ** 2) + (y_all * y_all)
    try:
        return float(torch.quantile(D, float(q)).item())
    except Exception:
        # Fallback: median if quantile fails for any reason
        return float(torch.median(D).item())


def _check_asinh_yspace_sanity(
    *,
    model: torch.nn.Module,
    dl_val,
    device: torch.device,
    asinh_loss: float,
    lm_hp,
    base_model: Optional[torch.nn.Module] = None,
) -> Tuple[bool, float, float, float, Optional[float]]:
    """Return (ok, y_mse, y_mse_allowed, D_ref, base_y_mse).

    Strategy A (Jacobian-consistent):
        y_mse_allowed_A = alpha * asinh_loss * D_ref

    Strategy B (baseline-relative guard, optional):
        y_mse_allowed_B = beta * base_y_mse

    Combined:
        y_mse_allowed = max(y_mse_allowed_A, y_mse_allowed_B)
    """
    y_mse = float(_eval_yspace_mse(model, dl_val, device))
    s = float(getattr(lm_hp, "fit_y_link_scale", 1.0))
    q = float(getattr(lm_hp, "asinh_yspace_sanity_quantile", 0.90))
    alpha = float(getattr(lm_hp, "asinh_yspace_sanity_factor", 20.0))
    beta = float(getattr(lm_hp, "asinh_yspace_regress_factor", 5.0))

    D_ref = float(_asinh_yspace_scale_from_loader(dl_val, device, s, q))
    y_mse_allowed_A = alpha * max(float(asinh_loss), 1e-30) * max(float(D_ref), 1e-30)

    base_y_mse = None
    y_mse_allowed = y_mse_allowed_A
    if base_model is not None:
        base_y_mse = float(_eval_yspace_mse(base_model, dl_val, device))
        y_mse_allowed_B = beta * max(float(base_y_mse), 1e-30)
        y_mse_allowed = max(float(y_mse_allowed_A), float(y_mse_allowed_B))

    ok = (math.isfinite(y_mse) and math.isfinite(y_mse_allowed) and (y_mse <= y_mse_allowed))
    return ok, y_mse, float(y_mse_allowed), float(D_ref), (None if base_y_mse is None else float(base_y_mse))


def _try_asinh_fit(
    *,
    current_ast,
    num_segments,
    dual_layer,
    leaf_builder,
    device,
    dtype,
    lm_hp,
    y_abs_median,
    datagen_train_noshuffle,
    datagen_val_noshuffle,
    loss_target_eff,
    base_model,
    y_op=None,
    y_op_inv=None,
):
    """Build asinh-conditioned model, train, run y-space sanity check.

    Sets lm_hp.fit_y_link='asinh' and lm_hp.fit_y_link_scale.
    Caller must revert these if result is rejected.

    Returns (asinh_model, asinh_ast, asinh_val_loss, asinh_y_ok).
    """
    lm_hp.fit_y_link = "asinh"
    lm_hp.fit_y_link_scale = float(y_abs_median) if (y_abs_median is not None and y_abs_median > 1e-30) else 1.0

    asinh_model, _, asinh_ast = build_composite_ast(
        current_ast,
        num_segments,
        dual_layer=dual_layer,
        leaf_builder=leaf_builder,
        device=device,
        dtype=dtype,
    )
    asinh_model = _apply_fit_link_to_model(asinh_model, lm_hp)

    asinh_val_loss, _, asinh_val_p, asinh_lm_opt = fit_initial_model_with_tournament(
        asinh_model,
        datagen_train_noshuffle,
        datagen_val_noshuffle,
        epochs=lm_hp.epochs,
        LM_strategy=lm_hp.strategy,
        nval_patience=lm_hp.nval_patience,
        loss_target=loss_target_eff,
        epochs_min=lm_hp.epochs_min,
        chisq_tol=lm_hp.chisq_tol,
        device=device,
        epochs_awful_check=lm_hp.epochs_awful_check,
        awful_threshold=lm_hp.awful_threshold,
        log_file=lm_hp.log_file,
        log_to_console=lm_hp.log_to_console,
        log_level=lm_hp.log_level,
        lm_verbose=lm_hp.LM_verbose,
        y_op=y_op,
        y_op_inv=y_op_inv,
        lm_hp=lm_hp,
    )
    asinh_lm_opt._update_param_groups(asinh_val_p)

    asinh_y_ok = True
    if asinh_model is not None:
        try:
            asinh_y_ok, asinh_y_mse, asinh_y_allow, D_ref, base_y_mse = _check_asinh_yspace_sanity(
                model=asinh_model,
                dl_val=datagen_val_noshuffle,
                device=device,
                asinh_loss=float(asinh_val_loss),
                lm_hp=lm_hp,
                base_model=base_model,
            )
            if not asinh_y_ok:
                base_str = "" if base_y_mse is None else f", base_y_mse={base_y_mse:.3e}"
                print(
                    f"{RED}[Stage A] asinh conditioning fails y-space sanity: "
                    f"y-MSE={float(asinh_y_mse):.3e} > allowed={float(asinh_y_allow):.3e} "
                    f"(asinh={float(asinh_val_loss):.3e}, D_ref={float(D_ref):.3e}{base_str}){RESET}"
                )
        except Exception as e:
            print(f"{YELLOW}[Stage A] Warning: asinh y-space sanity check error: {e}{RESET}")
            asinh_y_ok = True

    return asinh_model, asinh_ast, asinh_val_loss, asinh_y_ok


def _stageA_identity_target_good(
    *,
    val_loss: Optional[float],
    train_loss: Optional[float],
    loss_target_eff: Optional[float],
    rel_tol: float = 1.0e-9,
) -> Tuple[bool, str]:
    """Return whether a raw identity Stage-A fit has already reached target quality.

    Opportunistic fit-links such as ``asinh`` are numerical conditioning devices.
    Once the raw identity model is already at the meaningful target floor, lower
    fit-link loss should not by itself displace the identity branch, because the
    fit-link can distort separability geometry.
    """
    try:
        target = float(loss_target_eff)
    except Exception:
        return False, ""
    if (not math.isfinite(target)) or target < 0.0:
        return False, ""
    tol_target = target * (1.0 + max(0.0, float(rel_tol)))
    for label, value in (("validation", val_loss), ("training", train_loss)):
        try:
            loss = float(value)
        except Exception:
            continue
        if math.isfinite(loss) and loss <= tol_target:
            return True, f"{label}-loss {loss:.4e} <= target {target:.4e}"
    return False, ""


def _stageA_initial_fit_restart_allowed(
    *,
    y_op_is_identity: bool,
    is_multi: bool,
    skip_initial_fit: bool,
    restart_used: bool,
    has_previous_model: bool,
    fit_y_link_active: bool,
) -> bool:
    """Return whether Stage A may spend its single initial-fit random restart.

    This is deliberately scoped to the first identity baseline fit, before the
    reference model is written.  It is a numerical rescue for a bad initial NN
    surrogate, not a general rewrite-search restart policy.
    """
    return bool(
        y_op_is_identity
        and (not is_multi)
        and (not skip_initial_fit)
        and (not restart_used)
        and (not has_previous_model)
        and (not fit_y_link_active)
    )


def _detect_leaf_nondep_axes_for_atom(
    model,
    atom,
    leaf,
    datagen_train,
    device,
    base_precision: float,
    max_batches: int = 4,
    eps: float = 1e-12,
):
    """
    Estimate which input axes this NN leaf is (numerically) independent of,
    by probing its own gradient with respect to its *leaf-input* coordinates.

    For a regular NN atom, the leaf input is x[:, atom.var_idxs].

    For a compound NN atom (those with kwargs['input_expr']), the leaf input is
    constructed exactly as in ASTCompositeAdaptor.forward():

        t = [ z(x), x[extra_var_idxs...] ]

    where z(x) is a scalar compound feature (possibly already wrapped in sin/cos/etc)
    and extra_var_idxs are additional raw variables passed alongside z.

    We:
      * feed the leaf-input tensor into the leaf;
      * compute leaf.grad(...) wrt those coordinates;
      * normalise by MAD of the leaf output;
      * mark axes with median(|∂leaf/∂t_j| / MAD(f_leaf)) < base_precision.

    Returns
    -------
    axes_to_drop : list[int]
        List of *global* axis indices whose influence appears negligible.

        Note: for compound atoms we only ever propose dropping *extra* axes.
        The compound scalar z (local coordinate 0) is treated as indivisible
        for pruning in this heuristic.
    """
    kind = getattr(atom, "kind", None)
    if kind is None or str(kind).lower() != "nn":
        return []

    if leaf is None:
        return []

    if atom.n_in <= 1:
        # Univariate leaf — nothing to prune.
        return []

    # Build human-readable labels from atom.inputs.
    local_labels = []
    for inp in (atom.inputs or ()):
        kind = str(getattr(inp, "kind", "")).lower()
        if kind in ("var", "x", "input") and len(getattr(inp, "var_idxs", ())) == 1:
            local_labels.append(f"x{int(inp.var_idxs[0])}")
        else:
            local_labels.append("z")

    grads = []
    vals = []

    n_batches = 0
    for batch in datagen_train:
        if isinstance(batch, (list, tuple)):
            x, _ = batch
        else:
            x = batch
        x = x.to(device)

        # Build the *leaf-input* tensor (unified for compound and simple atoms).
        with torch.no_grad():
            x_sub = _build_atom_input_tensor(atom, x)

        with torch.no_grad():
            f = leaf(x_sub)
            if f.dim() == 1:
                f = f.view(-1, 1)
            cache = {"x": x_sub}
            g = leaf.grad(cache)
            # Expected shape [B, O, k]; fall back gracefully from [B, k].
            if g.dim() == 2:
                g = g.unsqueeze(1)
            g = g[:, 0, :]  # [B, k]

        vals.append(f[:, 0].detach().cpu())
        grads.append(g.detach().cpu())

        n_batches += 1
        if n_batches >= max_batches:
            break

    if not grads:
        return []

    F = torch.cat(vals, dim=0)  # [N]
    G = torch.cat(grads, dim=0)  # [N, k]

    med = torch.median(F)
    mad = torch.median(torch.abs(F - med))
    if mad.abs() <= eps:
        mad = med.abs() + eps
        if mad <= eps:
            # Leaf is essentially constant everywhere; don't prune based on
            # this heuristic (it would be better expressed as a constant leaf).
            return []

    G_norm = G / mad
    med_abs_grad = torch.median(torch.abs(G_norm), dim=0).values  # [k]

    grad_tol = float(base_precision)
    drop_local = [j for j in range(G_norm.shape[1]) if float(med_abs_grad[j]) < grad_tol]
    if not drop_local:
        return []

    # Map local axis → global variable index (None for compound z).
    inputs = atom.inputs or ()
    local_to_global = []
    for inp in inputs:
        kind = str(getattr(inp, "kind", "")).lower()
        if kind in ("var", "x", "input") and len(getattr(inp, "var_idxs", ())) == 1:
            local_to_global.append(int(inp.var_idxs[0]))
        else:
            local_to_global.append(None)  # compound z — not prunable

    drop_global = [
        local_to_global[j] for j in drop_local if local_to_global[j] is not None
    ]
    if not drop_global:
        return []

    print(
        "Leaf non-dependency probe on NN({}): median(|∂f/∂t_j|/MAD(f)) = [{}] "
        "→ dropping axes {}".format(
            ",".join(local_labels),
            ", ".join(f"{float(v):.3e}" for v in med_abs_grad),
            drop_global,
        )
    )

    return drop_global


def _build_leaf_prune_candidate_ast(current_ast, atom, axes_to_drop):
    """
    Given an NN AtomNode and a list of global axes to drop, build a new
    AST where that atom has a reduced var_idxs with those axes removed.

    For compound NN atoms (kwargs['input_expr']): also updates extra_var_idxs
    so the leaf-input dimensionality matches the pruned coordinate set.

    Returns a new AST or None if no change is applicable.
    """
    if not axes_to_drop:
        return None

    kind = getattr(atom, "kind", None)
    if kind is None or str(kind).lower() != "nn":
        return None

    old_vars = [int(j) for j in atom.var_idxs]
    axes_set = set(int(a) for a in axes_to_drop)

    # Copy kwargs and, for compound atoms, only prune among extra_var_idxs.
    new_kwargs = dict(getattr(atom, "kwargs", {}) or {})
    new_kwargs.pop("input_expr", None)
    new_kwargs.pop("extra_var_idxs", None)
    new_kwargs.pop("compound", None)
    new_inputs = None
    if has_nontrivial_input(atom):
        extra = list(extra_input_var_idxs(atom))
        axes_set = set(a for a in axes_set if a in set(extra))
        if not axes_set:
            return None
        new_extra = [j for j in extra if j not in axes_set]
        # Rebuild inputs tuple: z expression + remaining extras
        z_expr = compound_input_expr(atom)
        new_inputs = tuple([clone_ast(z_expr)] + [Var(int(v)) for v in new_extra])

    new_vars = [j for j in old_vars if j not in axes_set]

    # Either nothing to remove, or would remove *all* inputs → skip.
    if not new_vars or new_vars == old_vars:
        return None

    new_atom = AtomNode(
        kind=atom.kind,
        var_idxs=new_vars,
        kwargs=new_kwargs,
        tag=getattr(atom, "tag", None),
        inputs=new_inputs,
    )
    cand_ast = replace_atom_in_ast(current_ast, atom, new_atom)
    return cand_ast


def _test_difference_product_structure(x_vals, dydx_vals, i, j, k, precision=0.1, f_vals=None):
    """
    Test if gradients are consistent with f = g((xi-xj)*xk) or f = xk^p * g((xi-xj)*xk).

    Key relation: df/dxi = -df/dxj (they should cancel)
    Secondary: df/dxi / xk = df/dxk / (xi-xj) for pure f = g(z)

    For f = xk^p * g(z) where z = (xi-xj)*xk:
        df/dxk * xk - p * f = df/dxi * (xi-xj)
    This alternative test handles cases like f = x2 * sinc²((x4-x5)*x2/2).

    Parameters
    ----------
    x_vals : np.ndarray of shape [N, num_vars]
        Variable values.
    dydx_vals : np.ndarray of shape [N, num_vars]
        Gradient values df/dx for each variable.
    i, j, k : int
        Local indices for variables xi, xj, xk within the atom.
    precision : float
        Tolerance threshold.
    f_vals : np.ndarray of shape [N] or [N, 1], optional
        Function values. If provided, enables alternative Test 2 for f = xk^p * g(z).

    Returns
    -------
    tuple of (float, int)
        (Confidence score in [0, 1], detected outer power p).
        p=0 means pure g(z), p!=0 means f = xk^p * g(z).
    """
    import numpy as np

    gi = dydx_vals[:, i]
    gj = dydx_vals[:, j]
    gk = dydx_vals[:, k]

    xi = x_vals[:, i]
    xj = x_vals[:, j]
    xk = x_vals[:, k]

    # Test 1: gi + gj ≈ 0 (should cancel for (xi-xj) structure)
    sum_ij = gi + gj
    norm_gi = np.linalg.norm(gi)
    norm_gj = np.linalg.norm(gj)
    rel_residual_1 = np.linalg.norm(sum_ij) / (norm_gi + norm_gj + 1e-12)

    # Test 2 (standard): gi * (xi - xj) ≈ gk * xk
    # For z = (xi-xj)*xk: dz/dxi = xk, dz/dxj = -xk, dz/dxk = xi-xj
    # If f = g(z), then df/dxi = g'*xk, df/dxj = -g'*xk, df/dxk = g'*(xi-xj)
    # So gi * (xi - xj) should equal gk * xk
    lhs_std = gi * (xi - xj)
    rhs_std = gk * xk
    rel_residual_2_std = np.linalg.norm(lhs_std - rhs_std) / (
        np.linalg.norm(lhs_std) + np.linalg.norm(rhs_std) + 1e-12
    )

    best_residual_2 = rel_residual_2_std
    best_power = 0  # 0 means pure g(z), nonzero means xk^p * g(z)

    # Alternative Test 2 for f = xk^p * g(z) patterns
    # For f = xk^p * g(z) where z = (xi-xj)*xk:
    #   df/dxi = xk^(p+1) * g'(z)
    #   df/dxk = p * xk^(p-1) * g(z) + xk^p * g'(z) * (xi-xj)
    # Rearranging: df/dxk * xk - p * f = df/dxi * (xi - xj)
    if f_vals is not None and rel_residual_2_std > 0.2:
        f = f_vals.squeeze() if f_vals.ndim > 1 else f_vals
        diff_ij = xi - xj
        lhs_alt = gi * diff_ij

        for p in [1, 2, -1]:  # Try xk, xk², 1/xk prefactors
            rhs_alt = gk * xk - p * f
            residual = np.linalg.norm(lhs_alt - rhs_alt) / (
                np.linalg.norm(lhs_alt) + np.linalg.norm(rhs_alt) + 1e-12
            )
            if residual < best_residual_2:
                best_residual_2 = residual
                best_power = p

    # Combined confidence
    conf = 1.0 - max(rel_residual_1, best_residual_2)
    return max(0.0, conf), best_power


def _test_difference_product_power_structure(x_vals, dydx_vals, i, j, k, p, precision=0.1):
    """
    Test if gradients are consistent with f = g((xi-xj) * xk^p) for a small integer power p.

    For z = (xi-xj) * xk^p:
        df/dxi = g'(z) * xk^p
        df/dxj = -g'(z) * xk^p
        df/dxk = g'(z) * p * (xi-xj) * xk^(p-1)

    Rearranged relation (for p != 0):
        df/dxi * (xi-xj) ≈ (df/dxk * xk) / p

    Parameters
    ----------
    x_vals : np.ndarray [N, k]
    dydx_vals : np.ndarray [N, k]
    i, j, k : int
        Local indices for xi, xj, xk.
    p : int
        Power on xk inside the compound.
    precision : float

    Returns
    -------
    float
        Confidence score in [0, 1].
    """
    import numpy as np

    try:
        p = int(p)
    except Exception:
        p = 1
    if p == 0:
        return 0.0

    gi = dydx_vals[:, i]
    gj = dydx_vals[:, j]
    gk = dydx_vals[:, k]

    xi = x_vals[:, i]
    xj = x_vals[:, j]
    xk = x_vals[:, k]

    # Test 1: gi + gj ≈ 0 (difference structure)
    sum_ij = gi + gj
    norm_gi = np.linalg.norm(gi)
    norm_gj = np.linalg.norm(gj)
    rel_residual_1 = np.linalg.norm(sum_ij) / (norm_gi + norm_gj + 1e-12)

    # Test 2: gi*(xi-xj) ≈ (gk*xk)/p
    lhs = gi * (xi - xj)
    rhs = (gk * xk) / float(p)
    rel_residual_2 = np.linalg.norm(lhs - rhs) / (
        np.linalg.norm(lhs) + np.linalg.norm(rhs) + 1e-12
    )

    conf = 1.0 - max(rel_residual_1, rel_residual_2)
    return max(0.0, float(conf))


# ──────────────────────────────────────────────────────────────────────────────
# Power-difference detection via gradient ratio  (z = xi^n - xj^n)
# ──────────────────────────────────────────────────────────────────────────────

__search_definitions__ = (
    "_flatten_additive_terms",
    "_rebuild_additive_chain",
    "_subtree_contains_atom_tag",
    "_is_subtree_fully_decomposed",
    "_eval_ast_subtree_on_data",
    "_find_residual_refit_context",
    "_build_tag_to_leaf_map",
    "_find_tagged_nn_atom",
    "_get_parent_leaf_context",
    "_input_exprs_match",
    "_child_local_indices_in_parent",
    "_effective_input_labels",
    "_evaluate_overlap_truth_metric",
    "_overlap_truth_metric_is_acceptable",
    "_XTransformDataset",
    "_teacher_init_multiplicative",
    "_teacher_init_additive",
    "_build_additive_gauge_fix_factories",
    "_build_multiplicative_gauge_fix_factories",
    "_overlap_gauge_stage_is_feasible",
    "_compute_atom_scale",
    "_snap_omega",
    "_compound_exponents_ratio_like",
    "_build_xtransformed_loaders",
    "_nn_multivar_axes",
    "_axis_is_inside_compound_input",
    "_axis_is_coupled_by_invariance",
    "_extract_compound_targets_from_ast",
    "_propose_reciprocal_x_map",
    "_loader_all_finite",
    "_compute_y_med_mad_from_loader",
    "_eval_yspace_mse",
    "_asinh_yspace_scale_from_loader",
    "_check_asinh_yspace_sanity",
    "_try_asinh_fit",
    "_stageA_identity_target_good",
    "_stageA_initial_fit_restart_allowed",
    "_detect_leaf_nondep_axes_for_atom",
    "_build_leaf_prune_candidate_ast",
    "_test_difference_product_structure",
    "_test_difference_product_power_structure",
)

__search_constants__ = (

)

__search_late_bindings__ = (

)
