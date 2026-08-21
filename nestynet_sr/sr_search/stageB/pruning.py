# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Final pruning passes for Stage B.

Two complementary pruning strategies:

1. **Per-parameter pruning** (``prune_insignificant_parameters``):
   Iteratively zero individual polynomial / rational-polynomial coefficients,
   one at a time (least significant first), refitting after each removal.
   Acceptance is based on AIC.

2. **Additive-term pruning** (``prune_small_additive_terms``):
   Drop entire additive sub-expressions whose RMS contribution is negligible.

Per-parameter pruning runs first (finer-grained), then additive-term pruning
cleans up any terms that became all-zeros.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional

import torch

from nestynet_sr.sr_core.bridges import (
    AbsNode,
    AcosNode,
    AddNode,
    ArgNode,
    AsinNode,
    AtanNode,
    AtomNode,
    ConjNode,
    CosNode,
    ConstNode,
    ExpNode,
    ImagNode,
    LogNode,
    MulNode,
    Node,
    PowNode,
    RealNode,
    SinNode,
    build_composite_from_ast,
    clone_ast,
    collect_nn_atoms,
    eval_inputs,
    make_reuse_only_nn_factory,
)
from nestynet_sr.sr_core.sympy_bridge import (
    coefficient_symbol_nodes_from_ast,
    sympy_to_nestynet,
)

from .atom_mapping import _collect_all_atoms, build_atom_to_leaf_map
from .engine import StageBState
from .fitting import _fit_candidate_root


# Non-finite term contributions are treated as "very large" so they are
# never selected as prune candidates.
_NONFINITE_TERM_RMS = 1.0e30


def _inherit_simplification_path(
    source_state: StageBState,
    refit_state: StageBState,
) -> StageBState:
    """Carry the accepted analytic lineage across a fresh refit state.

    ``_fit_candidate_root`` intentionally returns a fresh ``StageBState``.
    Pruning changes the fitted model but does not invalidate the sequence of
    accepted structural rewrites that led to it, so losing that sequence makes
    good historical expressions unavailable to final adjudication.
    """
    source_path = copy.deepcopy(
        list(getattr(source_state, "simplification_path", []) or [])
    )
    refit_path = copy.deepcopy(
        list(getattr(refit_state, "simplification_path", []) or [])
    )
    if source_path and refit_path[: len(source_path)] != source_path:
        refit_path = source_path + refit_path
    elif source_path and not refit_path:
        refit_path = source_path
    refit_state.simplification_path = refit_path
    return refit_state


# ---------------------------------------------------------------------------
# AIC helper
# ---------------------------------------------------------------------------

def _compute_aic(mse: float, n_data: int, n_params: int) -> float:
    """Compute AIC = n * ln(MSE) + 2*k."""
    if mse <= 0 or not math.isfinite(mse):
        return float("inf")
    return n_data * math.log(mse) + 2 * n_params


# ===================================================================
# Per-parameter pruning
# ===================================================================

@dataclass
class PrunableParam:
    """Descriptor for a single prunable scalar coefficient."""
    atom_tag: str           # tag of the atom this param belongs to
    param_name: str         # attribute name: "coeffs", "coeffs_num", "coeffs_den"
    index: int              # scalar index into the parameter tensor
    value: float            # current coefficient value
    significance: float     # |c_i| * RMS(phi_i) on validation data
    is_den_constant: bool   # True if this is the constant term of a denominator
    leaf_type: str          # for logging (e.g. "poly", "ratpoly")


def _get_leaf_core(leaf):
    """Unwrap adaptor to get inner core (PolyLeaf, RationalPolyLeaf, etc.)."""
    return getattr(leaf, "model", getattr(leaf, "base_model", leaf))


def _gather_val_data(val_loader, device):
    """Collect all validation data into single tensors (x, y)."""
    xs, ys = [], []
    for batch in val_loader:
        if isinstance(batch, (list, tuple)):
            x, y = batch
            xs.append(x.to(device))
            ys.append(y.to(device))
        else:
            xs.append(batch.to(device))
    x_val = torch.cat(xs, dim=0)
    y_val = torch.cat(ys, dim=0) if ys else None
    return x_val, y_val


def _collect_prunable_params(
    root: Node,
    model,
    x_val: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    protect_den_const: bool = True,
) -> List[PrunableParam]:
    """Collect all prunable scalar parameters from poly/ratpoly leaves.

    Walks the AST, finds polynomial-family leaves, and builds a descriptor
    for each individual trainable coefficient together with its significance
    score (|c_i| * RMS(phi_i) on validation data).
    """
    from nestynet_sr.sr_core.atoms import (
        ExpPolyLeaf,
        ExpRationalPolyLeaf,
        PolyLeaf,
        RPolyLeaf,
        RRationalPolyLeaf,
        RationalPolyLeaf,
        _eval_monomials,
    )

    atom_to_leaf = build_atom_to_leaf_map(root, model)
    all_atoms = _collect_all_atoms(root)

    params: List[PrunableParam] = []

    for atom in all_atoms:
        leaf = atom_to_leaf.get(id(atom))
        if leaf is None:
            continue
        core = _get_leaf_core(leaf)
        tag = getattr(atom, "tag", None) or str(id(atom))

        # Compute leaf input z from validation data via eval_inputs
        try:
            x_in = eval_inputs(atom, x_val, need_grad=False, need_hess=False)[0]
        except Exception:
            continue
        if x_in is None or x_in.numel() == 0:
            continue

        # Dispatch by leaf type
        if isinstance(core, (PolyLeaf, RPolyLeaf)):
            _collect_poly_params(core, tag, x_in, params, _eval_monomials, "poly")
        elif isinstance(core, (RationalPolyLeaf, RRationalPolyLeaf)):
            _collect_ratpoly_params(
                core, tag, x_in, params, _eval_monomials,
                protect_den_const, leaf_type="ratpoly",
            )
        elif isinstance(core, ExpPolyLeaf):
            _collect_poly_params(core, tag, x_in, params, _eval_monomials, "exp_poly")
        elif isinstance(core, ExpRationalPolyLeaf):
            _collect_ratpoly_params(
                core, tag, x_in, params, _eval_monomials,
                protect_den_const, leaf_type="exp_ratpoly",
            )
        # SinLinearLeaf, PowerLeaf, PlanckLeaf, etc.: skip (tightly coupled params)

    return params


def _collect_poly_params(core, tag, x_in, params, _eval_monomials, leaf_type):
    """Collect prunable params from a poly-like leaf (PolyLeaf / RPolyLeaf / ExpPolyLeaf)."""
    coeffs = core.coeffs
    if coeffs.numel() <= 1:
        return  # single-param leaf: additive-term pruning's job

    exps = core.exps
    n_in = int(getattr(core, "n_in", x_in.shape[1]))
    z = x_in[..., :n_in]

    with torch.no_grad():
        if exps.numel() == 0:
            return
        Phi = _eval_monomials(z, exps)  # [B, M]
        for i in range(coeffs.numel()):
            if i >= Phi.shape[1]:
                break
            c_val = float(coeffs[i].detach().cpu())
            phi_rms = float(torch.sqrt((Phi[:, i] ** 2).mean()).cpu())
            sig = abs(c_val) * phi_rms
            params.append(PrunableParam(
                atom_tag=tag,
                param_name="coeffs",
                index=i,
                value=c_val,
                significance=sig,
                is_den_constant=False,
                leaf_type=leaf_type,
            ))


def _collect_ratpoly_params(core, tag, x_in, params, _eval_monomials,
                            protect_den_const, leaf_type="ratpoly"):
    """Collect prunable params from a rational poly leaf."""
    # Determine z input for the rational poly
    if hasattr(core, "indices") and core.indices:
        idx_list = list(core.indices)
        z = x_in[..., idx_list] if max(idx_list) < x_in.shape[-1] else x_in
    else:
        n_in = int(getattr(core, "n_in", x_in.shape[1]))
        z = x_in[..., :n_in]

    coeffs_num = core.coeffs_num
    coeffs_den = core.coeffs_den
    total_scalars = coeffs_num.numel() + coeffs_den.numel()
    if total_scalars <= 1:
        return  # single-param leaf

    with torch.no_grad():
        # Evaluate monomial bases
        Phi_num = (_eval_monomials(z, core.exps_num)
                   if core.exps_num.numel() > 0
                   else z.new_zeros(z.shape[0], 0))
        Phi_den = (_eval_monomials(z, core.exps_den)
                   if core.exps_den.numel() > 0
                   else z.new_zeros(z.shape[0], 0))

        # Q(z) for denominator significance
        eps = float(getattr(core, "eps", 1e-8))
        Q = (Phi_den @ coeffs_den if coeffs_den.numel() > 0
             else z.new_ones(z.shape[0]))
        Q = Q.clamp_min(eps)

        # P(z) for denominator coefficient significance
        if hasattr(core, "lead_exp_num"):
            # RRationalPolyLeaf: has pinned leading term
            lead = _eval_monomials(z, core.lead_exp_num.view(1, -1)).squeeze(1)
            P = lead + (Phi_num @ coeffs_num if coeffs_num.numel() > 0 else 0.0)
        else:
            P = (Phi_num @ coeffs_num if coeffs_num.numel() > 0
                 else z.new_zeros(z.shape[0]))

        # --- Numerator coefficients ---
        if coeffs_num.numel() > 0:
            for i in range(coeffs_num.numel()):
                if i >= Phi_num.shape[1]:
                    break
                c_val = float(coeffs_num[i].detach().cpu())
                # significance = |c_i| * RMS(phi_i / Q)
                phi_over_Q = Phi_num[:, i] / Q
                phi_rms = float(torch.sqrt((phi_over_Q ** 2).mean()).cpu())
                sig = abs(c_val) * phi_rms
                params.append(PrunableParam(
                    atom_tag=tag,
                    param_name="coeffs_num",
                    index=i,
                    value=c_val,
                    significance=sig,
                    is_den_constant=False,
                    leaf_type=leaf_type,
                ))

        # --- Denominator coefficients ---
        if coeffs_den.numel() > 0:
            for j in range(coeffs_den.numel()):
                if j >= Phi_den.shape[1]:
                    break
                # Check if this is the constant term (exponent sum = 0)
                exp_sum = int(core.exps_den[j].sum().item())
                is_den_const = (exp_sum == 0)

                if is_den_const and protect_den_const:
                    continue  # never prune denominator constant

                b_val = float(coeffs_den[j].detach().cpu())
                # d(P/Q)/db_j = -P * phi_j / Q^2
                # significance = |b_j| * RMS(P * phi_j / Q^2)
                deriv = P * Phi_den[:, j] / (Q ** 2)
                phi_rms = float(torch.sqrt((deriv ** 2).mean()).cpu())
                sig = abs(b_val) * phi_rms
                params.append(PrunableParam(
                    atom_tag=tag,
                    param_name="coeffs_den",
                    index=j,
                    value=b_val,
                    significance=sig,
                    is_den_constant=is_den_const,
                    leaf_type=leaf_type,
                ))


def prune_insignificant_parameters(
    state: StageBState,
    train_loader,
    val_loader,
    lm_hp,
    device: torch.device,
    dtype: torch.dtype,
    loss_scale: float,
    atom_factory=None,
    verbose: bool = True,
) -> StageBState:
    """Iteratively prune insignificant poly/ratpoly coefficients.

    Removes individual polynomial and rational-polynomial coefficients,
    one at a time from least significant upward, with a short LM refit
    after each removal.  Acceptance is based on AIC.

    Parameters
    ----------
    state : StageBState
        Current Stage B state (AST + model + val_loss).
    train_loader : DataLoader
        Training data (single dataset).
    val_loader : DataLoader
        Validation data (single dataset).
    lm_hp : LMHyperparams
        Hyperparameters (includes prune_param_* fields).
    device, dtype : torch device/dtype.
    loss_scale : float
        MAD-based loss scale.
    atom_factory : callable, optional
        Atom factory for building composites.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    StageBState
        Possibly pruned state.
    """
    # Gate: skip if disabled
    if not getattr(lm_hp, "prune_param_enable", True):
        return state

    # Gate: skip if any NN atoms remain
    num_nn = len(collect_nn_atoms(state.root))
    if num_nn > 0:
        if verbose:
            print(f"[PruneParam] Skipping: {num_nn} NN atoms remain")
        return state

    # Config
    aic_tol = float(getattr(lm_hp, "prune_param_aic_tolerance", 2.0))
    refit_epochs = int(getattr(lm_hp, "prune_param_refit_epochs", 300))
    max_pruned = int(getattr(lm_hp, "prune_param_max_pruned", 20))
    protect_den = getattr(lm_hp, "prune_param_protect_denominator_constant", True)

    # Gather validation data
    x_val, _ = _gather_val_data(val_loader, device)
    n_data = x_val.shape[0]

    # Collect prunable params
    prunable = _collect_prunable_params(
        state.root, state.model, x_val, device, dtype, protect_den,
    )

    if len(prunable) <= 1:
        if verbose:
            print(f"[PruneParam] Skipping: only {len(prunable)} prunable parameter(s)")
        return state

    # Original metrics
    orig_params = sum(p.numel() for p in state.model.parameters() if p.requires_grad)
    current_mse = float(state.val_loss)
    current_aic = _compute_aic(current_mse, n_data, orig_params)

    if verbose:
        print(
            f"[PruneParam] 0 NN atoms, {len(prunable)} prunable parameters "
            f"— running per-parameter pruning"
        )

    current_state = state
    n_pruned = 0
    pruned_set: set = set()  # (atom_tag, param_name, index)

    for _iteration in range(max_pruned):
        # Re-collect and re-rank after first iteration
        if _iteration > 0:
            prunable = _collect_prunable_params(
                current_state.root, current_state.model,
                x_val, device, dtype, protect_den,
            )
            # Filter out already-pruned params
            prunable = [
                p for p in prunable
                if (p.atom_tag, p.param_name, p.index) not in pruned_set
            ]

        if not prunable:
            break

        # Sort by significance (ascending) — least significant first
        prunable.sort(key=lambda p: p.significance)
        target = prunable[0]

        if verbose:
            print(
                f"[PruneParam]   Param {target.param_name}[{target.index}] "
                f"in {target.atom_tag} ({target.leaf_type}): "
                f"val={target.value:.3e}, sig={target.significance:.3e}"
            )

        # Deep-copy reuse map and zero the target coefficient
        reuse = current_state.reuse or {}
        trial_reuse = _clone_reuse_safe(reuse, device, dtype)

        if target.atom_tag in trial_reuse:
            leaf = trial_reuse[target.atom_tag]
            core = _get_leaf_core(leaf)
            param_tensor = getattr(core, target.param_name, None)
            if param_tensor is not None and target.index < param_tensor.numel():
                with torch.no_grad():
                    param_tensor[target.index] = 0.0

        # Refit with zeroed coefficient
        trial_state = _fit_candidate_root(
            root=clone_ast(current_state.root),
            reuse=trial_reuse,
            train_loader=train_loader,
            val_loader=val_loader,
            lm_hp=lm_hp,
            device=device,
            dtype=dtype,
            epochs_stageB=refit_epochs,
            loss_scale=loss_scale,
            atom_factory=atom_factory,
        )

        # Evaluate
        trial_mse = float(trial_state.val_loss)
        trial_param_count = orig_params - (n_pruned + 1)
        trial_aic = _compute_aic(trial_mse, n_data, trial_param_count)

        accepted = trial_aic <= current_aic + aic_tol

        if verbose:
            print(
                f"[PruneParam]   Zeroed {target.param_name}[{target.index}] "
                f"in {target.atom_tag}, refit {refit_epochs} epochs"
            )
            print(
                f"[PruneParam]   MSE: {current_mse:.3e} -> {trial_mse:.3e}, "
                f"AIC: {current_aic:.1f} -> {trial_aic:.1f} "
                f"— {'ACCEPTED' if accepted else 'REJECTED (stop)'}"
            )

        if accepted:
            current_state = _inherit_simplification_path(current_state, trial_state)
            current_mse = trial_mse
            current_aic = trial_aic
            n_pruned += 1
            pruned_set.add((target.atom_tag, target.param_name, target.index))
        else:
            break  # Stop on first rejection

    if verbose:
        if n_pruned > 0:
            total_orig = len(prunable) + n_pruned
            print(
                f"[PruneParam] Pruned {n_pruned} of {total_orig} parameters. "
                f"Final: {orig_params - n_pruned} params, "
                f"MSE={current_mse:.3e}, AIC={current_aic:.1f}"
            )
        else:
            print("[PruneParam] No parameters pruned")

    return current_state


# ---------------------------------------------------------------------------
# Additive-term helpers (local copies to avoid importing from search.py
# which would create a heavy cross-dependency)
# ---------------------------------------------------------------------------

def _flatten_additive_terms(root: Node):
    """Flatten nested AddNodes into a list of additive sub-trees."""
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


# ---------------------------------------------------------------------------
# Nested additive-site discovery
# ---------------------------------------------------------------------------

def _find_additive_sites(root: Node, min_terms: int = 2) -> List[tuple]:
    """Find all AddNode chains in the AST with >= *min_terms* terms.

    Returns a list of ``(top_node_id, terms)`` where *top_node_id* is the
    ``id()`` of the topmost AddNode in the chain and *terms* is the flattened
    list of non-Add sub-trees.

    The function recurses into the non-Add children of each chain, so nested
    chains (e.g. inside a MulNode child) are also found.
    """
    sites: List[tuple] = []

    def _walk(node: Node):
        if isinstance(node, AddNode):
            # Flatten the entire chain rooted here
            top_id = id(node)
            terms = _flatten_additive_terms(node)
            if len(terms) >= min_terms:
                sites.append((top_id, terms))
            # Recurse into each term's non-Add sub-expressions
            for t in terms:
                _walk_children(t)
        else:
            _walk_children(node)

    def _walk_children(node: Node):
        """Recurse into the children of a non-AddNode."""
        if isinstance(node, (MulNode,)):
            _walk(node.left)
            _walk(node.right)
        elif isinstance(node, PowNode):
            _walk(node.base)
        elif isinstance(node, (LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode,
                               ConjNode, RealNode, ImagNode, AbsNode, ArgNode)):
            _walk(node.arg)
        # AtomNode, ConstNode: leaf — nothing to descend into

    _walk(root)
    return sites


def _clone_ast_with_replacement(root: Node, target_id: int, replacement: Node) -> Node:
    """Clone *root*, substituting *replacement* wherever ``id(node) == target_id``."""
    if id(root) == target_id:
        return clone_ast(replacement)

    if isinstance(root, AddNode):
        return AddNode(
            left=_clone_ast_with_replacement(root.left, target_id, replacement),
            right=_clone_ast_with_replacement(root.right, target_id, replacement),
        )
    if isinstance(root, MulNode):
        return MulNode(
            left=_clone_ast_with_replacement(root.left, target_id, replacement),
            right=_clone_ast_with_replacement(root.right, target_id, replacement),
        )
    if isinstance(root, PowNode):
        return PowNode(
            base=_clone_ast_with_replacement(root.base, target_id, replacement),
            exponent=float(root.exponent),
        )
    if isinstance(root, LogNode):
        return LogNode(arg=_clone_ast_with_replacement(root.arg, target_id, replacement))
    if isinstance(root, ExpNode):
        return ExpNode(arg=_clone_ast_with_replacement(root.arg, target_id, replacement))
    if isinstance(root, SinNode):
        return SinNode(arg=_clone_ast_with_replacement(root.arg, target_id, replacement))
    if isinstance(root, CosNode):
        return CosNode(arg=_clone_ast_with_replacement(root.arg, target_id, replacement))
    if isinstance(root, AsinNode):
        return AsinNode(arg=_clone_ast_with_replacement(root.arg, target_id, replacement))
    if isinstance(root, AcosNode):
        return AcosNode(arg=_clone_ast_with_replacement(root.arg, target_id, replacement))
    if isinstance(root, AtanNode):
        return AtanNode(arg=_clone_ast_with_replacement(root.arg, target_id, replacement))
    if isinstance(root, ConjNode):
        return ConjNode(arg=_clone_ast_with_replacement(root.arg, target_id, replacement))
    if isinstance(root, RealNode):
        return RealNode(arg=_clone_ast_with_replacement(root.arg, target_id, replacement))
    if isinstance(root, ImagNode):
        return ImagNode(arg=_clone_ast_with_replacement(root.arg, target_id, replacement))
    if isinstance(root, AbsNode):
        return AbsNode(arg=_clone_ast_with_replacement(root.arg, target_id, replacement))
    if isinstance(root, ArgNode):
        return ArgNode(arg=_clone_ast_with_replacement(root.arg, target_id, replacement))
    # Leaf nodes: ConstNode, AtomNode — just clone
    return clone_ast(root)


# ---------------------------------------------------------------------------
# Term contribution estimation (in-context ablation)
# ---------------------------------------------------------------------------

def _flatten_pred(y_pred: torch.Tensor) -> torch.Tensor:
    if y_pred.dim() == 2:
        return y_pred[:, 0]
    return y_pred.view(-1)


def _build_eval_model(
    root: Node,
    *,
    reuse: dict,
    device: torch.device,
    dtype: torch.dtype,
    atom_factory=None,
):
    reuse_build = {}
    for k, v in (reuse or {}).items():
        mm = copy.deepcopy(v)
        mm.to(device=device, dtype=dtype)
        reuse_build[k] = mm

    nn_factory = make_reuse_only_nn_factory(device=device, dtype=dtype)
    model = build_composite_from_ast(
        clone_ast(root),
        dtype=dtype,
        device=device,
        nn_factory=nn_factory,
        atom_factory=atom_factory,
        reuse=reuse_build,
    )
    model.eval()
    return model


def _estimate_ablation_delta_rms(
    *,
    base_model,
    ablated_root: Node,
    reuse: dict,
    device: torch.device,
    dtype: torch.dtype,
    val_loader,
    atom_factory=None,
) -> float:
    """Estimate RMS change in prediction from dropping one term in-context."""
    try:
        ablated_model = _build_eval_model(
            ablated_root,
            reuse=reuse,
            device=device,
            dtype=dtype,
            atom_factory=atom_factory,
        )
    except Exception:
        return _NONFINITE_TERM_RMS

    base_model.eval()
    ss = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                x, _y = batch
            else:
                x = batch
            x = x.to(device)
            y_base = _flatten_pred(base_model(x))
            y_drop = _flatten_pred(ablated_model(x))
            if (not torch.isfinite(y_base).all()) or (not torch.isfinite(y_drop).all()):
                return _NONFINITE_TERM_RMS
            d = y_base - y_drop
            if not torch.isfinite(d).all():
                return _NONFINITE_TERM_RMS
            ss += float((d * d).sum().cpu())
            n += d.numel()
    if n == 0:
        return _NONFINITE_TERM_RMS
    mean_sq = ss / n
    if (not math.isfinite(mean_sq)) or mean_sq < 0.0:
        return _NONFINITE_TERM_RMS
    rms = math.sqrt(mean_sq)
    if not math.isfinite(rms):
        return _NONFINITE_TERM_RMS
    return rms


def _count_n_data(val_loader) -> int:
    """Count total data points in a validation loader."""
    n = 0
    for batch in val_loader:
        if isinstance(batch, (list, tuple)):
            x, _ = batch
        else:
            x = batch
        n += x.shape[0]
    return n


def _infer_nvars_from_loader(val_loader) -> Optional[int]:
    """Infer number of input variables from a validation loader."""
    for batch in val_loader:
        if isinstance(batch, (list, tuple)):
            x, _ = batch
        else:
            x = batch
        if x is None or x.ndim < 2:
            return None
        return int(x.shape[1])
    return None


def _node_count(node: Node) -> int:
    """Count nodes in an AST (simple structural size proxy)."""
    if isinstance(node, (ConstNode, AtomNode)):
        return 1
    if isinstance(node, (AddNode, MulNode)):
        return 1 + _node_count(node.left) + _node_count(node.right)
    if isinstance(node, PowNode):
        return 1 + _node_count(node.base)
    if isinstance(node, (LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode, ConjNode, RealNode, ImagNode, AbsNode, ArgNode)):
        return 1 + _node_count(node.arg)
    return 1


def _try_noiseless_sympy_simplify_state(
    state: StageBState,
    *,
    train_loader,
    val_loader,
    lm_hp,
    device: torch.device,
    dtype: torch.dtype,
    loss_scale: float,
    atom_factory=None,
    verbose: bool = True,
) -> Optional[StageBState]:
    """Try an exact/noiseless SymPy simplification and return updated state.

    Acceptance is strict:
    1) SymPy must certify symbolic equivalence exactly.
    2) Validation MSE must not regress beyond tiny tolerances.
    3) AST node count must strictly decrease.
    """
    try:
        import sympy as sp
        from nestynet_sr.sr_search.representation import pretty_print_state
    except Exception as exc:
        if verbose:
            print(f"[PruneSympy] skipped: import failed ({exc})")
        return None

    if len(collect_nn_atoms(state.root)) > 0:
        if verbose:
            print("[PruneSympy] skipped: NN atoms remain")
        return None

    nvars = _infer_nvars_from_loader(val_loader)
    if nvars is None:
        if verbose:
            print("[PruneSympy] skipped: could not infer input dimensionality")
        return None

    try:
        expr_str = pretty_print_state(state, sig=16)
        coefficient_nodes = coefficient_symbol_nodes_from_ast(state.root)
    except Exception as exc:
        if verbose:
            print(f"[PruneSympy] skipped: pretty print failed ({exc})")
        return None

    local_dict = {
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "arcsin": sp.asin,
        "arccos": sp.acos,
        "arctan": sp.atan,
        "exp": sp.exp,
        "log": sp.log,
        "pi": sp.pi,
        "E": sp.E,
    }
    try:
        expr_old = sp.sympify(expr_str.replace("^", "**"), locals=local_dict)
    except Exception as exc:
        if verbose:
            print(f"[PruneSympy] skipped: sympify failed ({exc})")
        return None

    old_size = _node_count(state.root)

    # Build exact-equivalent candidates and pick the smallest AST.
    try:
        base_simpl = sp.simplify(expr_old, ratio=1.5)
        candidate_exprs = [
            ("simplify", base_simpl),
            ("expand", sp.expand(expr_old)),
            ("expand_simplify", sp.expand(base_simpl)),
        ]
    except Exception as exc:
        if verbose:
            print(f"[PruneSympy] skipped: candidate generation failed ({exc})")
        return None

    seen: set[str] = set()
    best = None  # (name, expr, root, size, strlen)
    for name, cand in candidate_exprs:
        try:
            key = str(cand)
        except Exception:
            continue
        if key in seen:
            continue
        seen.add(key)
        if key == str(expr_old):
            continue

        # Noiseless symbolic guard: require exact symbolic identity.
        try:
            delta = sp.simplify(expr_old - cand)
            is_exact = bool(delta == 0) or bool(getattr(delta, "is_zero", False))
        except Exception:
            is_exact = False
        if not is_exact:
            continue

        try:
            cand_root = sympy_to_nestynet(
                cand,
                int(nvars),
                symbol_nodes=coefficient_nodes,
            )
        except Exception:
            continue

        cand_size = _node_count(cand_root)
        if cand_size >= old_size:
            continue

        cand_key = (int(cand_size), int(len(key)))
        if best is None or cand_key < (best[3], best[4]):
            best = (name, cand, cand_root, int(cand_size), int(len(key)))

    if best is None:
        if verbose:
            print("[PruneSympy] no-op: no exact strictly simpler reconstruction candidate")
        return None

    chosen_name, expr_new, simplified_root, new_size, _ = best
    if verbose:
        print(f"[PruneSympy] chose exact candidate: {chosen_name} (size {old_size} -> {new_size})")

    refit_epochs = int(
        getattr(
            lm_hp,
            "prune_sympy_refit_epochs",
            max(50, int(getattr(lm_hp, "prune_refit_epochs", 500) // 2)),
        )
    )
    if refit_epochs < 1:
        refit_epochs = 1

    trial_state = _fit_candidate_root(
        root=simplified_root,
        reuse={},
        train_loader=train_loader,
        val_loader=val_loader,
        lm_hp=lm_hp,
        device=device,
        dtype=dtype,
        epochs_stageB=refit_epochs,
        loss_scale=loss_scale,
        atom_factory=atom_factory,
    )

    old_mse = float(state.val_loss)
    new_mse = float(trial_state.val_loss)
    rel_tol = float(getattr(lm_hp, "prune_sympy_noiseless_rel_tol", 1.0e-12))
    abs_tol = float(getattr(lm_hp, "prune_sympy_noiseless_abs_tol", 1.0e-14))
    allowed = abs_tol + rel_tol * max(1.0, abs(old_mse))
    accepted = math.isfinite(new_mse) and (new_mse <= old_mse + allowed)

    if verbose:
        verdict = "ACCEPTED" if accepted else "REJECTED"
        print(
            f"[PruneSympy] MSE: {old_mse:.6e} -> {new_mse:.6e}, "
            f"size: {old_size} -> {new_size} -- {verdict}"
        )

    if not accepted:
        return None
    return _inherit_simplification_path(state, trial_state)


# ---------------------------------------------------------------------------
# Main pruning function
# ---------------------------------------------------------------------------

def prune_small_additive_terms(
    state: StageBState,
    train_loader,
    val_loader,
    lm_hp,
    device: torch.device,
    dtype: torch.dtype,
    loss_scale: float,
    atom_factory=None,
    verbose: bool = True,
) -> StageBState:
    """Try to prune small additive terms from a fully analytical expression.

    Parameters
    ----------
    state : StageBState
        Current Stage B state (AST + model + val_loss).
    train_loader : DataLoader
        Training data (single dataset).
    val_loader : DataLoader
        Validation data (single dataset).
    lm_hp : LMHyperparams
        Hyperparameters (includes pruning thresholds).
    device, dtype : torch device/dtype.
    loss_scale : float
        MAD-based loss scale.
    atom_factory : callable, optional
        Atom factory for building composites.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    StageBState
        Possibly pruned state.
    """
    # Gate: skip if disabled
    if not getattr(lm_hp, "prune_final_enable", True):
        return state

    # Gate: skip if any NN atoms remain
    num_nn = len(collect_nn_atoms(state.root))
    if num_nn > 0:
        if verbose:
            print(f"[Prune] Skipping: {num_nn} NN atoms remain")
        return state

    # Flatten additive terms
    terms = _flatten_additive_terms(state.root)
    n_terms = len(terms)
    if n_terms < 2:
        if verbose:
            print(f"[Prune] Skipping: only {n_terms} additive term(s)")
        return state

    if verbose:
        print(f"[Prune] {n_terms} additive terms, 0 NN atoms -- running pruning")

    # Estimate in-context contribution of each term (drop-one-term ablation RMS)
    rel_threshold = float(getattr(lm_hp, "prune_rel_threshold", 1e-3))
    loss_tolerance = float(getattr(lm_hp, "prune_loss_tolerance", 0.01))
    refit_epochs = int(getattr(lm_hp, "prune_refit_epochs", 500))

    reuse = state.reuse or {}
    contribs = []
    for i in range(n_terms):
        keep_indices = [j for j in range(n_terms) if j != i]
        kept_terms = [clone_ast(terms[j]) for j in keep_indices]
        ablated_root = _rebuild_additive_chain(kept_terms)
        rc_i = _estimate_ablation_delta_rms(
            base_model=state.model,
            ablated_root=ablated_root,
            reuse=reuse,
            device=device,
            dtype=dtype,
            val_loader=val_loader,
            atom_factory=atom_factory,
        )
        contribs.append(rc_i)

    # Normalise by total RMS of full prediction
    finite_vals = [c for c in contribs if math.isfinite(c) and c < 0.5 * _NONFINITE_TERM_RMS]
    if not finite_vals:
        if verbose:
            print("[Prune] Skipping: all term contribution estimates are non-finite/unstable")
        return state
    total_rms = math.sqrt(sum(c * c for c in finite_vals)) if finite_vals else 1.0
    if (not math.isfinite(total_rms)) or total_rms <= 0:
        total_rms = 1.0

    rel_contribs = [
        (c / total_rms) if (math.isfinite(c) and c < 0.5 * _NONFINITE_TERM_RMS) else float("nan")
        for c in contribs
    ]
    finite_rel_idxs = [i for i, rc in enumerate(rel_contribs) if math.isfinite(rc)]
    if not finite_rel_idxs:
        if verbose:
            print("[Prune] Skipping: no finite relative contribution estimates")
        return state

    # Find the largest-contribution term (never flag it)
    max_idx = max(finite_rel_idxs, key=lambda i: rel_contribs[i])

    # Flag small terms
    flagged = []
    from nestynet_sr.sr_search.representation import pretty_print_state as _pps
    for i, (term, rc) in enumerate(zip(terms, rel_contribs)):
        is_small = math.isfinite(rc) and (rc < rel_threshold) and (i != max_idx)
        label = " [SMALL]" if is_small else ""
        if verbose:
            # Quick label for the term
            try:
                from nestynet_sr.sr_search.stageB.engine import StageBState as _S
                atom_to_leaf_full = build_atom_to_leaf_map(state.root, state.model)
                term_atoms = _collect_all_atoms(term)
                term_leaves = [atom_to_leaf_full[id(atom)] for atom in term_atoms if id(atom) in atom_to_leaf_full]
                if len(term_leaves) != len(term_atoms):
                    raise RuntimeError("subtree leaf subset incomplete")
                mini_model = SimpleNamespace(leaf=term_leaves, ast_root=term)
                mini = _S(root=term, model=mini_model, reuse=reuse, val_loss=0.0)
                t_str = _pps(mini, sig=4)
            except Exception:
                t_str = str(term)
            rc_str = f"{rc:.3e}" if math.isfinite(rc) else "non-finite"
            print(f"[Prune]   term {i}: {t_str:50s} contrib={rc_str}{label}")
        if is_small:
            flagged.append(i)

    if not flagged:
        if verbose:
            print("[Prune] No small terms found -- nothing to prune")
        return state

    # Original metrics
    orig_mse = float(state.val_loss)
    orig_params = sum(p.numel() for p in state.model.parameters() if p.requires_grad)
    n_data = _count_n_data(val_loader)
    orig_aic = _compute_aic(orig_mse, n_data, orig_params)

    # --- Try dropping all flagged terms at once ---
    keep_indices = [i for i in range(n_terms) if i not in flagged]
    kept_terms = [clone_ast(terms[i]) for i in keep_indices]
    pruned_root = _rebuild_additive_chain(kept_terms)

    if verbose:
        print(f"[Prune] Dropping {len(flagged)} flagged term(s)...")

    pruned_state = _fit_candidate_root(
        root=pruned_root,
        reuse=_clone_reuse_safe(reuse, device, dtype),
        train_loader=train_loader,
        val_loader=val_loader,
        lm_hp=lm_hp,
        device=device,
        dtype=dtype,
        epochs_stageB=refit_epochs,
        loss_scale=loss_scale,
        atom_factory=atom_factory,
    )

    pruned_mse = float(pruned_state.val_loss)
    pruned_params = sum(p.numel() for p in pruned_state.model.parameters() if p.requires_grad)
    pruned_aic = _compute_aic(pruned_mse, n_data, pruned_params)

    accepted = pruned_mse < orig_mse * (1.0 + loss_tolerance)

    if verbose:
        print(f"[Prune]   Original:  MSE={orig_mse:.4e}  params={orig_params}  AIC={orig_aic:.1f}")
        print(f"[Prune]   Pruned:    MSE={pruned_mse:.4e}  params={pruned_params}  AIC={pruned_aic:.1f}")
        delta_mse_pct = (pruned_mse - orig_mse) / max(orig_mse, 1e-30) * 100
        delta_aic = pruned_aic - orig_aic
        if accepted:
            print(f"[Prune]   \033[32mACCEPTED\033[0m -- delta_MSE={delta_mse_pct:+.1f}%, delta_AIC={delta_aic:+.1f}")
        else:
            print(f"[Prune]   \033[31mREJECTED\033[0m -- delta_MSE={delta_mse_pct:+.1f}%, delta_AIC={delta_aic:+.1f}")

    if accepted:
        return _inherit_simplification_path(state, pruned_state)

    # --- Greedy one-at-a-time fallback (smallest contrib first) ---
    if len(flagged) > 1:
        if verbose:
            print("[Prune] Trying greedy one-at-a-time pruning (smallest first)...")

        # Sort flagged by contribution (ascending)
        flagged_sorted = sorted(flagged, key=lambda i: rel_contribs[i])

        best_state = state
        best_mse = orig_mse
        removed = set()

        for idx in flagged_sorted:
            trial_keep = [i for i in range(n_terms) if i not in removed and i != idx]
            trial_terms = [clone_ast(terms[i]) for i in trial_keep]
            trial_root = _rebuild_additive_chain(trial_terms)

            trial_state = _fit_candidate_root(
                root=trial_root,
                reuse=_clone_reuse_safe(reuse, device, dtype),
                train_loader=train_loader,
                val_loader=val_loader,
                lm_hp=lm_hp,
                device=device,
                dtype=dtype,
                epochs_stageB=refit_epochs,
                loss_scale=loss_scale,
                atom_factory=atom_factory,
            )

            trial_mse = float(trial_state.val_loss)
            if trial_mse < best_mse * (1.0 + loss_tolerance):
                removed.add(idx)
                best_state = _inherit_simplification_path(best_state, trial_state)
                best_mse = trial_mse
                if verbose:
                    print(f"[Prune]   Dropped term {idx}: MSE={trial_mse:.4e} -- accepted")
            else:
                if verbose:
                    print(f"[Prune]   Dropping term {idx}: MSE={trial_mse:.4e} -- rejected")

        if removed:
            pruned_params_g = sum(p.numel() for p in best_state.model.parameters() if p.requires_grad)
            pruned_aic_g = _compute_aic(best_mse, n_data, pruned_params_g)
            if verbose:
                delta_mse_pct = (best_mse - orig_mse) / max(orig_mse, 1e-30) * 100
                delta_aic = pruned_aic_g - orig_aic
                print(f"[Prune]   Greedy result: dropped {len(removed)} term(s), "
                      f"MSE={best_mse:.4e}, AIC={pruned_aic_g:.1f} "
                      f"(delta_MSE={delta_mse_pct:+.1f}%, delta_AIC={delta_aic:+.1f})")
            return best_state

    return state


# ===================================================================
# Nested additive-term pruning (descend into sub-expressions)
# ===================================================================

def prune_nested_additive_terms(
    state: StageBState,
    train_loader,
    val_loader,
    lm_hp,
    device: torch.device,
    dtype: torch.dtype,
    loss_scale: float,
    atom_factory=None,
    verbose: bool = True,
) -> StageBState:
    """Prune small additive terms nested inside sub-expressions.

    Unlike :func:`prune_small_additive_terms` which only looks at root-level
    AddNode chains, this function walks the *entire* AST to find AddNode
    chains nested inside MulNode, PowNode, unary ops, etc.

    For each chain with >= 2 terms it estimates per-term RMS contribution,
    proposes a trimmed full-tree AST with small terms removed, refits, and
    accepts if AIC improves (or stays within tolerance).

    Parameters
    ----------
    state : StageBState
        Current Stage B state (AST + model + val_loss).
    train_loader, val_loader : DataLoader
        Training / validation data (single dataset).
    lm_hp : LMHyperparams
        Hyperparameters.
    device, dtype : torch device/dtype.
    loss_scale : float
        MAD-based loss scale.
    atom_factory : callable, optional
        Atom factory for building composites.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    StageBState
        Possibly pruned state.
    """
    # Gate: skip if disabled
    if not getattr(lm_hp, "prune_final_enable", True):
        return state

    # Gate: skip if any NN atoms remain
    num_nn = len(collect_nn_atoms(state.root))
    if num_nn > 0:
        if verbose:
            print(f"[PruneNested] Skipping: {num_nn} NN atoms remain")
        return state

    # Find all additive sites with >= 2 terms
    sites = _find_additive_sites(state.root)
    if not sites:
        if verbose:
            print("[PruneNested] No nested additive sites with >= 2 terms")
        return state

    if verbose:
        print(f"[PruneNested] Found {len(sites)} additive site(s) with >= 2 terms")

    # Config
    rel_threshold = float(getattr(lm_hp, "prune_rel_threshold", 1e-3))
    refit_epochs = int(getattr(lm_hp, "prune_refit_epochs", 500))
    aic_tol = float(getattr(lm_hp, "prune_param_aic_tolerance", 2.0))

    # Baseline metrics
    n_data = _count_n_data(val_loader)
    current_state = state
    n_terms_pruned = 0

    # Iterative: drop one term per iteration, re-scan tree after each acceptance
    max_rounds = 20  # safety cap
    for _round in range(max_rounds):
        sites = _find_additive_sites(current_state.root)
        if not sites:
            break

        # Across all sites, find the single least-significant term
        reuse = current_state.reuse or {}
        best_candidate = None  # (top_id, terms, drop_idx, rel_contrib, rel_contribs)

        for top_id, terms in sites:
            n_terms = len(terms)

            # Estimate in-context contribution of each term (drop-one-term ablation RMS)
            contribs = []
            for i in range(n_terms):
                keep_indices = [j for j in range(n_terms) if j != i]
                kept_terms = [clone_ast(terms[j]) for j in keep_indices]
                replacement = _rebuild_additive_chain(kept_terms)
                ablated_root = _clone_ast_with_replacement(
                    current_state.root, top_id, replacement,
                )
                rc_i = _estimate_ablation_delta_rms(
                    base_model=current_state.model,
                    ablated_root=ablated_root,
                    reuse=reuse,
                    device=device,
                    dtype=dtype,
                    val_loader=val_loader,
                    atom_factory=atom_factory,
                )
                contribs.append(rc_i)

            finite_vals = [c for c in contribs if math.isfinite(c) and c < 0.5 * _NONFINITE_TERM_RMS]
            if not finite_vals:
                if verbose:
                    print("[PruneNested]   site skipped: all contribution estimates are non-finite/unstable")
                continue

            total_rms = math.sqrt(sum(c * c for c in finite_vals)) if finite_vals else 1.0
            if (not math.isfinite(total_rms)) or total_rms <= 0:
                total_rms = 1.0
            rel_contribs = [
                (c / total_rms) if (math.isfinite(c) and c < 0.5 * _NONFINITE_TERM_RMS) else float("nan")
                for c in contribs
            ]

            finite_rel_idxs = [i for i, rc in enumerate(rel_contribs) if math.isfinite(rc)]
            if not finite_rel_idxs:
                if verbose:
                    print("[PruneNested]   site skipped: all relative contributions are non-finite")
                continue

            # Never consider the largest-contribution term
            max_idx = max(finite_rel_idxs, key=lambda i: rel_contribs[i])

            for i, rc in enumerate(rel_contribs):
                if not math.isfinite(rc):
                    continue
                if i == max_idx:
                    continue
                if rc >= rel_threshold:
                    continue
                # This term is small — is it the least significant overall?
                if best_candidate is None or rc < best_candidate[3]:
                    best_candidate = (top_id, terms, i, rc, rel_contribs)

        if best_candidate is None:
            if verbose:
                print("[PruneNested] No small terms found across any site")
            break

        top_id, terms, drop_idx, drop_rc, drop_rel_contribs = best_candidate
        n_terms = len(terms)

        if verbose:
            for i in range(n_terms):
                label = " << DROP" if i == drop_idx else ""
                rc_i = float(drop_rel_contribs[i]) if i < len(drop_rel_contribs) else float("nan")
                rc_str = f"{rc_i:.3e}" if math.isfinite(rc_i) else "non-finite"
                print(f"[PruneNested]     term {i}: contrib={rc_str}{label}")

        # Build trimmed AST: drop just the one term
        keep_indices = [i for i in range(n_terms) if i != drop_idx]
        kept_terms = [terms[i] for i in keep_indices]
        replacement = _rebuild_additive_chain(kept_terms)
        trimmed_root = _clone_ast_with_replacement(
            current_state.root, top_id, replacement,
        )

        # Original metrics
        orig_mse = float(current_state.val_loss)
        orig_params = sum(
            p.numel() for p in current_state.model.parameters() if p.requires_grad
        )
        orig_aic = _compute_aic(orig_mse, n_data, orig_params)

        if verbose:
            print(
                f"[PruneNested]   Dropping term {drop_idx} (contrib={drop_rc:.3e}), "
                f"refit {refit_epochs} epochs..."
            )

        trial_state = _fit_candidate_root(
            root=trimmed_root,
            reuse=_clone_reuse_safe(reuse, device, dtype),
            train_loader=train_loader,
            val_loader=val_loader,
            lm_hp=lm_hp,
            device=device,
            dtype=dtype,
            epochs_stageB=refit_epochs,
            loss_scale=loss_scale,
            atom_factory=atom_factory,
        )

        trial_mse = float(trial_state.val_loss)
        trial_params = sum(
            p.numel() for p in trial_state.model.parameters() if p.requires_grad
        )
        trial_aic = _compute_aic(trial_mse, n_data, trial_params)

        accepted = trial_aic <= orig_aic + aic_tol

        if verbose:
            print(
                f"[PruneNested]   MSE: {orig_mse:.3e} -> {trial_mse:.3e}, "
                f"AIC: {orig_aic:.1f} -> {trial_aic:.1f} "
                f"-- {'ACCEPTED' if accepted else 'REJECTED (stop)'}"
            )

        if accepted:
            current_state = _inherit_simplification_path(current_state, trial_state)
            n_terms_pruned += 1
            # Loop: re-scan tree and try dropping the next least-significant
        else:
            break  # stop on first rejection

    if verbose:
        if n_terms_pruned > 0:
            print(f"[PruneNested] Pruned {n_terms_pruned} term(s)")
        else:
            print("[PruneNested] No nested terms pruned")

    return current_state


def _clone_reuse_safe(reuse: dict, device, dtype) -> dict:
    """Deep-copy a reuse map."""
    out = {}
    for k, v in (reuse or {}).items():
        mm = copy.deepcopy(v)
        mm.to(device=device, dtype=dtype)
        out[k] = mm
    return out


def run_stageb_pruning_pipeline(
    state: StageBState,
    *,
    train_loader,
    val_loader,
    lm_hp,
    device: torch.device,
    dtype: torch.dtype,
    loss_scale: float,
    atom_factory=None,
    verbose: bool = True,
) -> StageBState:
    """Run the canonical Stage B pruning pipeline.

    This is the exact pruning sequence used at the end of Stage B:
    1) per-parameter pruning,
    2) root-level additive-term pruning,
    3) nested additive-term pruning,
    with optional iterative noiseless SymPy simplification rounds.
    """
    def _run_prune_pass(cur: StageBState) -> StageBState:
        if getattr(lm_hp, "prune_param_enable", True):
            cur = prune_insignificant_parameters(
                state=cur,
                train_loader=train_loader,
                val_loader=val_loader,
                lm_hp=lm_hp,
                device=device,
                dtype=dtype,
                loss_scale=loss_scale,
                atom_factory=atom_factory,
                verbose=verbose,
            )

        if getattr(lm_hp, "prune_final_enable", True):
            cur = prune_small_additive_terms(
                state=cur,
                train_loader=train_loader,
                val_loader=val_loader,
                lm_hp=lm_hp,
                device=device,
                dtype=dtype,
                loss_scale=loss_scale,
                atom_factory=atom_factory,
                verbose=verbose,
            )

        if getattr(lm_hp, "prune_final_enable", True):
            cur = prune_nested_additive_terms(
                state=cur,
                train_loader=train_loader,
                val_loader=val_loader,
                lm_hp=lm_hp,
                device=device,
                dtype=dtype,
                loss_scale=loss_scale,
                atom_factory=atom_factory,
                verbose=verbose,
            )
        return cur

    state = _run_prune_pass(state)

    if not bool(getattr(lm_hp, "prune_sympy_iter_enable", False)):
        return state

    n_rounds = int(getattr(lm_hp, "prune_sympy_iter_max_rounds", 2))
    if n_rounds < 1:
        return state

    for i in range(n_rounds):
        if verbose:
            print(f"[PruneSympy] round {i + 1}/{n_rounds}")
        simplified_state = _try_noiseless_sympy_simplify_state(
            state,
            train_loader=train_loader,
            val_loader=val_loader,
            lm_hp=lm_hp,
            device=device,
            dtype=dtype,
            loss_scale=loss_scale,
            atom_factory=atom_factory,
            verbose=verbose,
        )
        if simplified_state is None:
            break
        state = _run_prune_pass(simplified_state)

    return state
