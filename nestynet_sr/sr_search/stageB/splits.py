# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Counterterm and counterfactor splitting algorithms for Stage B.

This module implements advanced splitting algorithms that detect and exploit
special factorization structures in multivariate NN atoms:
- Counterterm multiplicative splits: u(x) = P(x_A) + g(x_A) * h(x_B)
- Counterfactor additive splits: u(x) = P_A(x_A) * P_B(x_B) * (g(x_A) + h(x_B))
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from nestynet_sr.sr_core.atoms import PolyLeaf, RPolyLeaf, _enumerate_exponents, _eval_monomials
from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    MulNode,
    Node,
    PowNode,
    Scale,
    _collect_var_idxs_from_node,
    clone_ast,
    compound_input_expr,
    effective_arity,
    get_input_exprs,
    is_trivial_input,
    replace_atom_in_ast,
)
from nestynet_sr.sr_search.candidate_builders import _build_atom_input_tensor

from .atom_mapping import _collect_all_atoms, build_atom_to_leaf_map
from .leaf_utils import _leaf_coeff_param
from .subtree_utils import _infer_nn_hyperparams_from_root

_AFFINE_SPLIT_DEBUG = False
_COUNTERTERM_DEBUG = False

def _affine_split_log(*args, **kwargs) -> None:
    if _AFFINE_SPLIT_DEBUG:
        print(*args, **kwargs)

def _counterterm_debug_log(*args, **kwargs) -> None:
    if _COUNTERTERM_DEBUG:
        print(*args, **kwargs)

# ---------------------------------------------------------------------------
# Coordinate-space helper for compound vs simple atoms
# ---------------------------------------------------------------------------

def _leaf_coord_info(atom: AtomNode) -> List[Dict[str, Any]]:
    """Map local leaf coordinates to global variable info.

    Returns a list (one entry per local axis) of dicts:
        {"local": int, "global": int|None, "expr": Node}

    For trivial inputs (Var(i)), ``global`` is the raw variable index.
    For nontrivial input expressions, ``global`` is None.
    """
    inputs = get_input_exprs(atom)
    coords: List[Dict[str, Any]] = []
    for i, inp in enumerate(inputs):
        if is_trivial_input(inp):
            glob = int(inp.var_idxs[0])
        else:
            glob = None
        coords.append({"local": i, "global": glob, "expr": inp})
    return coords


def _fresh_atom_kwargs(atom_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return a fresh kwargs dict safe to attach to a new AtomNode.

    Compound info (input_expr/extra_var_idxs) is now propagated via
    ``inputs=`` on AtomNode, so this just returns a shallow copy of
    non-compound kwargs.
    """
    if not atom_kwargs:
        return {}
    return dict(atom_kwargs)


def _fresh_inputs(inputs: Tuple[Node, ...]) -> Tuple[Node, ...]:
    """Clone an inputs tuple so each atom gets its own copy (no shared DAGs)."""
    return tuple(clone_ast(inp) for inp in inputs)


def _partition_to_child_inputs(
    target: AtomNode, partition_local_indices: List[int]
) -> Tuple[Tuple[int, ...], Tuple[Node, ...]]:
    """Build (var_idxs, inputs) for a child atom from a partition of local indices.

    Works uniformly for simple and compound atoms: selects the input
    expressions at the given local indices, collects all raw variable
    indices they reference, and returns deep-copied input nodes.

    Parameters
    ----------
    target : AtomNode
        The parent atom being split.
    partition_local_indices : list of int
        Which local axes to include (indices into ``get_input_exprs(target)``).

    Returns
    -------
    var_idxs : tuple of int
        Sorted, deduplicated raw variable indices referenced by the selected inputs.
    inputs : tuple of Node
        Deep-copied input expression nodes for the child atom.
    """
    all_inputs = get_input_exprs(target)
    selected = [all_inputs[i] for i in partition_local_indices]
    var_idxs: list[int] = []
    for inp in selected:
        var_idxs.extend(_collect_var_idxs_from_node(inp))
    return tuple(sorted(set(var_idxs))), tuple(clone_ast(inp) for inp in selected)


def _enumerate_unique_partitions(m: int) -> List[Tuple[List[int], List[int]]]:
    """Enumerate unique non-trivial partitions A|B of local indices 0..m-1.

    Symmetry is broken by forcing 0 ∈ A.

    Returns
    -------
    List[(A,B)] with A,B sorted, disjoint, A∪B = {0..m-1}, A,B non-empty.
    """
    if m < 2:
        return []
    out: List[Tuple[List[int], List[int]]] = []
    # Subsets of {1..m-1}
    for mask in range(1 << (m - 1)):
        A = [0]
        for k in range(1, m):
            if (mask >> (k - 1)) & 1:
                A.append(k)
        if len(A) == m:
            continue
        B = [j for j in range(m) if j not in A]
        if not B:
            continue
        out.append((A, B))
    return out


def _cross_hessian_rank1_score(
    H: torch.Tensor,
    A: List[int],
    B: List[int],
    *,
    max_points: int = 1024,
    eps: float = 1e-12,
) -> float:
    """Heuristic rank-1 score for the cross-Hessian block H_AB.

    For a multiplicatively separable interaction u(x)=P_A(x_A)+P_B(x_B)+g(x_A)h(x_B),
    the cross-Hessian block H_AB(x)=∂²u/∂x_A∂x_B is rank-1 for all x.

    We measure median(σ2/(σ1+eps)) over a subsample of points, where σk are the
    singular values of H_AB at each sample.

    Lower is "more rank-1". Returns +inf if the block is not at least 2×2
    or if the SVD fails.
    """
    try:
        a = len(A)
        b = len(B)
        if min(a, b) < 2:
            return float("inf")
        N = int(H.shape[0])
        if N <= 0:
            return float("inf")
        # Subsample to keep this cheap (deterministic)
        if N > int(max_points):
            idx = torch.linspace(0, N - 1, int(max_points), device=H.device)
            idx = idx.round().to(torch.long)
            Hs = H.index_select(0, idx)
        else:
            Hs = H

        H_AB = Hs[:, A, :][:, :, B]  # [Ns, |A|, |B|]
        s = torch.linalg.svdvals(H_AB)  # [Ns, min(|A|,|B|)]
        if s.ndim != 2 or s.shape[1] < 2:
            return float("inf")
        s1 = s[:, 0]
        s2 = s[:, 1]
        ratio = s2 / (s1 + float(eps))

        # Be robust to points where the cross-block is ~0 (e.g. sin(Δ)≈0),
        # where numerical noise can dominate. Evaluate rank-1-ness on the
        # strongest quarter of samples (by σ1) and take a median there.
        Ns = int(ratio.shape[0])
        k = int(max(16, 0.25 * Ns))
        k = min(k, Ns)
        if k < Ns:
            _, idx = torch.topk(s1, k)
            ratio = ratio.index_select(0, idx)

        score = torch.median(ratio)
        if not torch.isfinite(score):
            return float("inf")
        return float(score.detach().cpu())
    except Exception:
        return float("inf")


def _prefilter_counterterm_partitions_by_rank1_cross_hessian(
    *,
    parts: List[Tuple[List[int], List[int]]],
    H: torch.Tensor,
    m: int,
    rank1_tol: float = 0.15,
    max_points: int = 1024,
) -> List[Tuple[List[int], List[int]]]:
    """Guard against degenerate 1|(m-1) splits.

    When m>=4 and there exists any *balanced* partition with min(|A|,|B|)>=2 whose
    cross-Hessian block is close to rank-1, we restrict counterterm fitting to those
    balanced partitions.

    Only if no balanced partition passes do we fall back to considering all partitions
    (including 1|(m-1) splits).
    """
    if int(m) < 4:
        return parts

    balanced: List[Tuple[List[int], List[int]]] = [
        (A, B) for (A, B) in parts if min(len(A), len(B)) >= 2
    ]
    if not balanced:
        return parts

    scored: List[Tuple[Tuple[List[int], List[int]], float]] = []
    for A, B in balanced:
        s = _cross_hessian_rank1_score(H, A, B, max_points=int(max_points))
        scored.append(((A, B), float(s)))

    passing = [p for (p, s) in scored if (s <= float(rank1_tol))]

    # Logging (keep it compact)
    scored_sorted = sorted(scored, key=lambda t: t[1])
    k_show = min(5, len(scored_sorted))
    print(
        f"[counterterm] Cross-Hessian rank-1 prefilter (balanced only): tol={float(rank1_tol):.3g}"
    )
    for (A, B), s in scored_sorted[:k_show]:
        print(f"[counterterm]   rank1_score={s:.3e} for A={A}, B={B}")

    if passing:
        print(
            f"[counterterm]   Keeping {len(passing)}/{len(balanced)} balanced partitions; excluding 1|(m-1) splits"
        )
        return passing

    print("[counterterm]   No balanced partition passed rank1_tol; falling back to all partitions")
    return parts


def _gather_nn_atom_value_grad_hess(
    *,
    root: Node,
    model: nn.Module,
    atom: AtomNode,
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    max_points: int = 4096,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Gather x_sub, u, du, H for an NN AtomNode from the current fitted model.

    Returns
    -------
    (X, X_raw, u, du, H) with shapes:
      X     : [N, m] - local atom input (for compound atoms, this is z)
      X_raw : [N, n_in] - raw global x values (only atom's base var columns)
      u     : [N]
      du    : [N, m]
      H     : [N, m, m]
    or None if unavailable.
    """
    if not isinstance(atom, AtomNode) or str(atom.kind).lower() != "nn":
        return None
    m = effective_arity(atom)  # Use effective arity for compound atoms
    if m < 2:
        return None

    try:
        atom_to_leaf = build_atom_to_leaf_map(root, model)
    except Exception:
        return None

    leaf = atom_to_leaf.get(id(atom), None)
    if leaf is None:
        return None

    if (not hasattr(leaf, "grad")) or (not hasattr(leaf, "grad_grad")):
        return None

    Xs: List[torch.Tensor] = []
    Xs_raw: List[torch.Tensor] = []  # Raw global x for compound atom initialization
    Us: List[torch.Tensor] = []
    Gs: List[torch.Tensor] = []
    Hs: List[torch.Tensor] = []

    # Get base var indices for raw x extraction (needed for compound atoms)
    kw = getattr(atom, 'kwargs', None) or {}
    extra_var_idxs = kw.get('extra_var_idxs', []) or []
    extra_set = set(int(v) for v in extra_var_idxs)
    base_vars = [int(v) for v in atom.var_idxs if int(v) not in extra_set]

    leaf.eval()
    n = 0
    with torch.no_grad():
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device=device, dtype=dtype)
            x_sub = _build_atom_input_tensor(atom, x)  # Handles compound atoms
            cache = {"x": x_sub}

            try:
                u = leaf(x_sub)  # [B,1]
                du = leaf.grad(cache)  # [B,1,m]
                H = leaf.grad_grad(cache)  # [B,1,m,m]
            except Exception:
                return None

            u = u.squeeze(-1)  # [B]
            du = du.squeeze(1)  # [B,m]
            H = H.squeeze(1)  # [B,m,m]

            # Filter non-finite rows (rare but can happen with bad initialisation)
            finite = torch.isfinite(u)
            finite = finite & torch.isfinite(du).all(dim=1)
            finite = finite & torch.isfinite(H.reshape(H.shape[0], -1)).all(dim=1)
            if not finite.any():
                continue
            x_sub = x_sub[finite]
            x_raw = x[finite][:, base_vars]  # Raw x values for base vars only
            u = u[finite]
            du = du[finite]
            H = H[finite]

            Xs.append(x_sub.detach())
            Xs_raw.append(x_raw.detach())  # Also save raw x
            Us.append(u.detach())
            Gs.append(du.detach())
            Hs.append(H.detach())
            n += int(x_sub.shape[0])
            if n >= max_points:
                break

    if n < max(64, 4 * m):
        return None

    X = torch.cat(Xs, dim=0)[:max_points]
    X_raw = torch.cat(Xs_raw, dim=0)[:max_points]  # Concatenate raw x
    u = torch.cat(Us, dim=0)[:max_points]
    du = torch.cat(Gs, dim=0)[:max_points]
    H = torch.cat(Hs, dim=0)[:max_points]
    return X, X_raw, u, du, H


# ---------------------------------------------------------------------------
# Overlap-prefactor peel helpers
# ---------------------------------------------------------------------------

def _collect_probe_x_from_loader(
    train_loader,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_points: int = 2048,
) -> Optional[torch.Tensor]:
    """Collect a moderate probe batch of x-values from a loader."""
    Xs: List[torch.Tensor] = []
    n = 0
    for batch in train_loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        x = x.to(device=device, dtype=dtype)
        Xs.append(x)
        n += int(x.shape[0])
        if n >= max_points:
            break
    if not Xs:
        return None
    return torch.cat(Xs, dim=0)[:max_points]


def _singleton_prefactor_peel_score(
    u: torch.Tensor,
    du: torch.Tensor,
    H: torch.Tensor,
    *,
    t_local: int,
    rest_locals: List[int],
    eps: float = 1.0e-10,
) -> tuple[float, float]:
    """Score whether one local axis acts as a multiplicative prefactor.

    For a perfect peel on axis ``t_local`` we expect

        u * d2u/dt dx_j  ~=  du/dt * du/dx_j

    for every remaining axis ``x_j``. The returned score is the worst
    normalised residual across those cross-pairs.
    """
    if not rest_locals:
        return float("inf"), 0.0

    y = u.view(-1)
    finite_y = torch.isfinite(y)
    worst = 0.0
    support = 1.0

    for j in rest_locals:
        lhs = y * H[:, t_local, j]
        rhs = du[:, t_local] * du[:, j]
        finite = (
            finite_y
            & torch.isfinite(lhs)
            & torch.isfinite(rhs)
            & torch.isfinite(du[:, t_local])
            & torch.isfinite(du[:, j])
        )
        if not finite.any():
            return float("inf"), 0.0
        lhs_f = lhs[finite]
        rhs_f = rhs[finite]
        resid = lhs_f - rhs_f
        scale = torch.median(torch.abs(lhs_f)) + torch.median(torch.abs(rhs_f))
        scale_val = float(scale.item())
        if (not math.isfinite(scale_val)) or scale_val < eps:
            scale_val = eps
        score = float((torch.median(torch.abs(resid)) / scale_val).item())
        worst = max(worst, score)
        support = min(support, float(finite.float().mean().item()))

    return worst, support


def _singleton_additive_counterterm_score(
    H: torch.Tensor,
    *,
    t_local: int,
    rest_locals: List[int],
    eps: float = 1.0e-10,
) -> tuple[float, float]:
    """Score whether one local axis behaves like an additive counterterm.

    For an exact decomposition

        u(x_t, x_rest) = C(x_t) + A(x_rest),

    all mixed second derivatives between ``x_t`` and the remaining variables
    vanish. The returned score is the worst mixed-Hessian magnitude,
    normalised by a robust overall Hessian scale.
    """
    if not rest_locals:
        return float("inf"), 0.0

    finite_all = torch.isfinite(H)
    if not finite_all.any():
        return float("inf"), 0.0
    H_scale = float(torch.median(torch.abs(H[finite_all])).item())
    if (not math.isfinite(H_scale)) or H_scale < eps:
        H_scale = eps

    worst = 0.0
    support = 1.0
    for j in rest_locals:
        hij = H[:, t_local, j]
        finite = torch.isfinite(hij)
        if not finite.any():
            return float("inf"), 0.0
        score = float((torch.median(torch.abs(hij[finite])) / H_scale).item())
        worst = max(worst, score)
        support = min(support, float(finite.float().mean().item()))
    return worst, support


def _fit_leaf_to_profile_adam(
    leaf: nn.Module,
    x_data: torch.Tensor,
    y_target: torch.Tensor,
    *,
    lr: float = 1.0e-2,
    n_iters: int = 160,
) -> float:
    """Warm-start a single leaf against fixed profile data."""
    params = [p for p in leaf.parameters() if p.requires_grad]
    if not params:
        return float("inf")

    dev = params[0].device
    dt = params[0].dtype
    x_in = x_data.to(device=dev, dtype=dt)
    y_ref = y_target.to(device=dev, dtype=dt)
    if y_ref.ndim == 1:
        y_ref = y_ref.unsqueeze(-1)

    opt = torch.optim.Adam(params, lr=lr)
    best = float("inf")
    best_state = None

    for _ in range(int(n_iters)):
        opt.zero_grad()
        y_pred = leaf(x_in)
        loss = torch.mean((y_pred - y_ref) ** 2)
        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()

        loss_val = float(loss.item())
        if loss_val < best:
            best = loss_val
            best_state = {k: v.detach().clone() for k, v in leaf.state_dict().items()}
        if loss_val < 1.0e-10:
            break

    if best_state is not None:
        leaf.load_state_dict(best_state)
    return best


@dataclass
class _OverlapPeelContext:
    left_atom: AtomNode
    right_atom: AtomNode
    left_leaf: nn.Module
    right_leaf: nn.Module
    x_full: torch.Tensor
    med: torch.Tensor
    y_left_full: torch.Tensor
    y_right_full: torch.Tensor
    num_segments: int
    dual_layer: bool
    inner_nn_kwargs: Dict[str, Any]


def _prepare_overlap_peel_context(
    *,
    root: Node,
    target: Node,
    expected_type: type,
    model: nn.Module,
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    max_points: int,
    log_fn: Optional[Callable[[str], None]],
    label: str,
    structure_desc: str,
) -> Optional[_OverlapPeelContext]:
    """Prepare shared probe context for overlap peel builders.

    This centralises the common direct-sibling validation and probe-data setup
    shared by overlap-prefactor and overlap-counterterm peels.
    """

    def _log(msg: str):
        if log_fn is None:
            return
        try:
            log_fn(msg)
        except Exception:
            pass

    if not isinstance(target, expected_type):
        return None
    if not isinstance(target.left, AtomNode) or not isinstance(target.right, AtomNode):
        return None
    if str(target.left.kind).lower() != "nn" or str(target.right.kind).lower() != "nn":
        return None

    left_atom = target.left
    right_atom = target.right
    if any(not is_trivial_input(inp) for inp in get_input_exprs(left_atom)):
        _log(f"[Stage B] {label}: skipping target with nontrivial left inputs")
        return None
    if any(not is_trivial_input(inp) for inp in get_input_exprs(right_atom)):
        _log(f"[Stage B] {label}: skipping target with nontrivial right inputs")
        return None

    try:
        atom_to_leaf = build_atom_to_leaf_map(root, model)
    except Exception:
        _log(f"[Stage B] {label}: failed to build atom-to-leaf map")
        return None

    left_leaf = atom_to_leaf.get(id(left_atom), None)
    right_leaf = atom_to_leaf.get(id(right_atom), None)
    if left_leaf is None or right_leaf is None:
        _log(f"[Stage B] {label}: missing leaf module for {structure_desc}")
        return None
    if not all(hasattr(left_leaf, name) for name in ("grad", "grad_grad")):
        _log(f"[Stage B] {label}: left leaf lacks grad/grad_grad support")
        return None
    if not all(hasattr(right_leaf, name) for name in ("grad", "grad_grad")):
        _log(f"[Stage B] {label}: right leaf lacks grad/grad_grad support")
        return None

    x_full = _collect_probe_x_from_loader(
        train_loader, device=device, dtype=dtype, max_points=max_points
    )
    if x_full is None or x_full.shape[0] < 128:
        _log(f"[Stage B] {label}: insufficient probe data")
        return None

    med = torch.median(x_full, dim=0).values

    with torch.no_grad():
        x_left_full = _build_atom_input_tensor(left_atom, x_full)
        x_right_full = _build_atom_input_tensor(right_atom, x_full)
        y_left_full = left_leaf(x_left_full).squeeze(-1)
        y_right_full = right_leaf(x_right_full).squeeze(-1)

    if not torch.isfinite(y_left_full).all() or not torch.isfinite(y_right_full).all():
        return None

    num_segments, dual_layer = _infer_nn_hyperparams_from_root(root)
    inner_nn_kwargs = {"num_segments": int(num_segments), "dual_layer": bool(dual_layer)}

    return _OverlapPeelContext(
        left_atom=left_atom,
        right_atom=right_atom,
        left_leaf=left_leaf,
        right_leaf=right_leaf,
        x_full=x_full,
        med=med,
        y_left_full=y_left_full,
        y_right_full=y_right_full,
        num_segments=int(num_segments),
        dual_layer=bool(dual_layer),
        inner_nn_kwargs=inner_nn_kwargs,
    )


def _build_overlap_counterterm_peel_candidates(
    *,
    root: Node,
    target: MulNode,
    model: nn.Module,
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    log_fn: Optional[Callable[[str], None]] = None,
    max_points: int = 2048,
    max_per_direction: int = 1,
    counter_score_tol: float = 0.12,
    min_unique_grad_ratio: float = 0.05,
    min_counter_support: float = 0.80,
    min_counter_variation: float = 0.05,
) -> List[Tuple[Node, Callable, Dict[str, Any]]]:
    """Propose branch-local additive counterterm peels on direct NN*NN products.

    v1 scope:
    - direct ``MulNode(nn(...), nn(...))`` only
    - simple inputs only (no compound coordinates)
    - singleton shared-variable counterterms only

    The rewrite family is:

        nn_L(u, r, t) * nn_R(v, r, t)
            -> (nn_C(t) + nn_A(u, r)) * nn_R(v, r, t)

    and the mirrored right-peel variant.
    """

    def _log(msg: str):
        if log_fn is None:
            return
        try:
            log_fn(msg)
        except Exception:
            pass

    prep = _prepare_overlap_peel_context(
        root=root,
        target=target,
        expected_type=MulNode,
        model=model,
        train_loader=train_loader,
        device=device,
        dtype=dtype,
        max_points=max_points,
        log_fn=log_fn,
        label="overlap_counterterm_peel",
        structure_desc="direct NN*NN target",
    )
    if prep is None:
        return []
    left_atom = prep.left_atom
    right_atom = prep.right_atom
    left_leaf = prep.left_leaf
    right_leaf = prep.right_leaf
    x_full = prep.x_full
    med = prep.med
    y_left_full = prep.y_left_full
    y_right_full = prep.y_right_full
    inner_nn_kwargs = dict(prep.inner_nn_kwargs)
    counter_segments = max(4, min(int(prep.num_segments), max(4, int(prep.num_segments) // 2)))
    counter_nn_kwargs = {
        "num_segments": int(counter_segments),
        "dual_layer": bool(prep.dual_layer),
    }

    proposals: List[Tuple[float, Node, Callable, Dict[str, Any]]] = []

    def _make_init_fn(
        *,
        counter_tag: str,
        reduced_tag: str,
        stay_tag: str,
        stay_state_cpu: Optional[Dict[str, torch.Tensor]],
        x_counter_cpu: torch.Tensor,
        y_counter_cpu: torch.Tensor,
        x_reduced_cpu: torch.Tensor,
        y_reduced_cpu: torch.Tensor,
        x_stay_cpu: torch.Tensor,
        y_stay_cpu: torch.Tensor,
        direction: str,
        peeled_global: int,
    ) -> Callable[[Node, nn.Module], None]:
        def _init_fn(
            root_new: Node,
            model_new: nn.Module,
            _ctag: str = counter_tag,
            _atag: str = reduced_tag,
            _stag: str = stay_tag,
            _stay_state: Optional[Dict[str, torch.Tensor]] = stay_state_cpu,
            _xc: torch.Tensor = x_counter_cpu,
            _yc: torch.Tensor = y_counter_cpu,
            _xa: torch.Tensor = x_reduced_cpu,
            _ya: torch.Tensor = y_reduced_cpu,
            _xs: torch.Tensor = x_stay_cpu,
            _ys: torch.Tensor = y_stay_cpu,
            _direction: str = direction,
            _peeled_global: int = peeled_global,
        ):
            try:
                atom_to_leaf_new = build_atom_to_leaf_map(root_new, model_new)
            except Exception:
                print(
                    "[Stage B] overlap_counterterm_peel init: atom-to-leaf map failed "
                    f"for {_direction}-peel x{int(_peeled_global)}"
                )
                return

            tag_to_leaf_new: Dict[str, nn.Module] = {}
            for atom in _collect_all_atoms(root_new):
                if isinstance(atom, AtomNode) and getattr(atom, "tag", None) is not None:
                    leaf_mod = atom_to_leaf_new.get(id(atom), None)
                    if leaf_mod is not None:
                        tag_to_leaf_new[str(atom.tag)] = leaf_mod

            leaf_counter = tag_to_leaf_new.get(_ctag)
            leaf_reduced = tag_to_leaf_new.get(_atag)
            leaf_stay = tag_to_leaf_new.get(_stag)
            if leaf_counter is None or leaf_reduced is None:
                print(
                    "[Stage B] overlap_counterterm_peel init: missing new leaves "
                    f"for {_direction}-peel x{int(_peeled_global)}"
                )
                return

            loss_counter = _fit_leaf_to_profile_adam(leaf_counter, _xc, _yc)
            loss_reduced = _fit_leaf_to_profile_adam(leaf_reduced, _xa, _ya)
            loss_stay = float("nan")
            if leaf_stay is not None:
                copied_stay = False
                if _stay_state:
                    try:
                        leaf_stay.load_state_dict(_stay_state, strict=True)
                        copied_stay = True
                    except Exception:
                        copied_stay = False
                # The stay branch is structurally unchanged. If we managed to
                # copy the original weights, a short refresh fit is enough;
                # otherwise fall back to a full warm-start fit.
                stay_iters = 80 if copied_stay else 160
                loss_stay = _fit_leaf_to_profile_adam(
                    leaf_stay, _xs, _ys, n_iters=stay_iters
                )
            elif _stag:
                print(
                    "[Stage B] overlap_counterterm_peel init: missing stay leaf "
                    f"for {_direction}-peel x{int(_peeled_global)} tag={_stag}"
                )
            print(
                "[Stage B] overlap_counterterm_peel init "
                f"({_direction}, x{int(_peeled_global)}): "
                f"C={loss_counter:.3e}, reduced={loss_reduced:.3e}, stay={loss_stay:.3e}"
            )

        return _init_fn

    def _direction_candidates(
        *,
        peel_atom: AtomNode,
        stay_atom: AtomNode,
        peel_leaf: nn.Module,
        stay_leaf: nn.Module,
        y_peel_full: torch.Tensor,
        y_stay_full: torch.Tensor,
        direction: str,
        place_left: bool,
    ) -> List[Tuple[float, Node, Callable, Dict[str, Any]]]:
        peel_vars = [int(v) for v in peel_atom.var_idxs]
        stay_vars = [int(v) for v in stay_atom.var_idxs]
        peel_set = set(peel_vars)
        stay_set = set(stay_vars)
        shared = sorted(peel_set & stay_set)
        unique = sorted(peel_set - stay_set)
        _log(
            f"[Stage B] overlap_counterterm_peel: probing {direction}-peel "
            f"peel_vars={peel_vars} stay_vars={stay_vars} shared={shared} unique={unique}"
        )
        if not shared or not unique:
            _log(
                f"[Stage B] overlap_counterterm_peel: {direction}-peel skipped "
                f"(shared={shared}, unique={unique})"
            )
            return []

        gathered = _gather_nn_atom_value_grad_hess(
            root=root,
            model=model,
            atom=peel_atom,
            train_loader=train_loader,
            device=device,
            dtype=dtype,
            max_points=max_points,
        )
        if gathered is None:
            _log(f"[Stage B] overlap_counterterm_peel: {direction}-peel gather failed")
            return []
        _X_loc, _X_raw, u, du, H = gathered

        local_map = {int(v): i for i, v in enumerate(peel_vars)}
        unique_locals = [local_map[v] for v in unique]
        if not unique_locals:
            _log(f"[Stage B] overlap_counterterm_peel: {direction}-peel has no unique locals")
            return []

        grad_meds = torch.median(torch.abs(du), dim=0).values
        all_grad = float(grad_meds.max().item()) if grad_meds.numel() else 0.0
        uniq_grad = (
            float(grad_meds[unique_locals].max().item()) if unique_locals else 0.0
        )
        if (not math.isfinite(all_grad)) or all_grad <= 1.0e-12:
            _log(
                f"[Stage B] overlap_counterterm_peel: {direction}-peel rejected "
                f"(all_grad={all_grad:.3e})"
            )
            return []
        grad_ratio = uniq_grad / max(all_grad, 1.0e-30)
        _log(
            f"[Stage B] overlap_counterterm_peel: {direction}-peel grad check "
            f"uniq_grad={uniq_grad:.3e}, all_grad={all_grad:.3e}, ratio={grad_ratio:.3e}"
        )
        if uniq_grad < float(min_unique_grad_ratio) * all_grad:
            _log(
                f"[Stage B] overlap_counterterm_peel: {direction}-peel rejected "
                f"(grad_ratio={grad_ratio:.3e} < {float(min_unique_grad_ratio):.3e})"
            )
            return []

        out: List[Tuple[float, Node, Callable, Dict[str, Any]]] = []
        for peeled_global in shared:
            peeled_local = local_map[peeled_global]
            rest_locals = [j for j in range(len(peel_vars)) if j != peeled_local]
            counter_score, counter_support = _singleton_additive_counterterm_score(
                H, t_local=peeled_local, rest_locals=rest_locals
            )
            _log(
                f"[Stage B] overlap_counterterm_peel: {direction}-peel test x{int(peeled_global)} "
                f"score={float(counter_score):.3e}, support={float(counter_support):.3f}"
            )
            if counter_score > float(counter_score_tol):
                _log(
                    f"[Stage B] overlap_counterterm_peel: reject x{int(peeled_global)} "
                    f"(score {float(counter_score):.3e} > tol {float(counter_score_tol):.3e})"
                )
                continue
            if counter_support < float(min_counter_support):
                _log(
                    f"[Stage B] overlap_counterterm_peel: reject x{int(peeled_global)} "
                    f"(support {float(counter_support):.3f} < {float(min_counter_support):.3f})"
                )
                continue

            x_counter = x_full.clone()
            for g in peel_vars:
                if int(g) != int(peeled_global):
                    x_counter[:, int(g)] = med[int(g)]
            with torch.no_grad():
                counter_target = peel_leaf(_build_atom_input_tensor(peel_atom, x_counter)).squeeze(-1)

            counter_rms = float(torch.sqrt(torch.mean(counter_target ** 2)).item())
            counter_centered = counter_target - torch.median(counter_target)
            counter_var = float(torch.sqrt(torch.mean(counter_centered ** 2)).item())
            if counter_rms <= 1.0e-12:
                _log(
                    f"[Stage B] overlap_counterterm_peel: reject x{int(peeled_global)} "
                    f"(counter_rms={counter_rms:.3e})"
                )
                continue
            counter_rel_var = counter_var / max(counter_rms, 1.0e-12)
            _log(
                f"[Stage B] overlap_counterterm_peel: x{int(peeled_global)} counter profile "
                f"rms={counter_rms:.3e}, rel_var={counter_rel_var:.3e}"
            )
            if counter_rel_var < float(min_counter_variation):
                _log(
                    f"[Stage B] overlap_counterterm_peel: reject x{int(peeled_global)} "
                    f"(counter rel_var {counter_rel_var:.3e} < {float(min_counter_variation):.3e})"
                )
                continue

            reduced_target = y_peel_full - counter_target
            reduced_vars = tuple(int(v) for v in peel_vars if int(v) != int(peeled_global))
            if not reduced_vars:
                _log(
                    f"[Stage B] overlap_counterterm_peel: reject x{int(peeled_global)} "
                    "(would remove all peeled-side variables)"
                )
                continue

            base_left_tag = str(getattr(target.left, "tag", None) or "L")
            base_right_tag = str(getattr(target.right, "tag", None) or "R")
            base_tag = (
                f"{base_left_tag}_{base_right_tag}_oct_{direction[0]}_x{int(peeled_global)}"
            )
            counter_tag = f"{base_tag}_C"
            reduced_tag = f"{base_tag}_A"

            counter_atom = AtomNode(
                "nn",
                (int(peeled_global),),
                kwargs=dict(counter_nn_kwargs),
                tag=counter_tag,
            )
            reduced_atom = AtomNode(
                "nn",
                reduced_vars,
                kwargs=dict(inner_nn_kwargs),
                tag=reduced_tag,
            )
            peeled_branch = AddNode(counter_atom, reduced_atom)
            stay_clone = clone_ast(stay_atom)
            stay_tag_raw = getattr(stay_clone, "tag", None)
            if stay_tag_raw is None or str(stay_tag_raw) == "":
                stay_tag = f"{base_tag}_B"
                setattr(stay_clone, "tag", stay_tag)
                _log(
                    f"[Stage B] overlap_counterterm_peel: assigned synthetic stay tag "
                    f"{stay_tag} for {direction}-peel x{int(peeled_global)}"
                )
            else:
                stay_tag = str(stay_tag_raw)
            if place_left:
                cand_subtree = MulNode(peeled_branch, stay_clone)
            else:
                cand_subtree = MulNode(stay_clone, peeled_branch)
            cand_root = replace_atom_in_ast(root, target, cand_subtree)

            x_counter_cpu = x_full[:, [int(peeled_global)]].detach().cpu()
            y_counter_cpu = counter_target.detach().cpu()
            x_reduced_cpu = x_full[:, list(reduced_vars)].detach().cpu()
            y_reduced_cpu = reduced_target.detach().cpu()
            x_stay_cpu = x_full[:, stay_vars].detach().cpu()
            y_stay_cpu = y_stay_full.detach().cpu()
            stay_state_cpu = {
                k: v.detach().cpu().clone() for k, v in stay_leaf.state_dict().items()
            }
            init_fn = _make_init_fn(
                counter_tag=counter_tag,
                reduced_tag=reduced_tag,
                stay_tag=stay_tag,
                stay_state_cpu=stay_state_cpu,
                x_counter_cpu=x_counter_cpu,
                y_counter_cpu=y_counter_cpu,
                x_reduced_cpu=x_reduced_cpu,
                y_reduced_cpu=y_reduced_cpu,
                x_stay_cpu=x_stay_cpu,
                y_stay_cpu=y_stay_cpu,
                direction=direction,
                peeled_global=int(peeled_global),
            )

            signature = (
                "overlap_counterterm_peel",
                tuple(int(v) for v in left_atom.var_idxs),
                tuple(int(v) for v in right_atom.var_idxs),
                str(direction),
                int(peeled_global),
            )
            metadata = {
                "structural": True,
                "partial_sep": True,
                "has_overlap": True,
                "overlap": True,
                "pattern_family": "overlap_counterterm_peel",
                "direction": str(direction),
                "peeled_var": int(peeled_global),
                "counter_score": float(counter_score),
                "log": (
                    f"[Stage B] overlap_counterterm_peel ({direction}) "
                    f"x{int(peeled_global)} score={float(counter_score):.3e}"
                ),
                "signature": signature,
            }
            _log(
                f"[Stage B] overlap_counterterm_peel: ACCEPT proposal "
                f"{direction}-peel x{int(peeled_global)} -> reduced_vars={list(reduced_vars)}, "
                f"stay_vars={stay_vars}"
            )
            out.append((float(counter_score), cand_root, init_fn, metadata))

        out.sort(key=lambda item: item[0])
        if not out:
            _log(f"[Stage B] overlap_counterterm_peel: no valid {direction}-peel proposals")
        return out[: max(1, int(max_per_direction))]

    proposals.extend(
        _direction_candidates(
            peel_atom=left_atom,
            stay_atom=right_atom,
            peel_leaf=left_leaf,
            stay_leaf=right_leaf,
            y_peel_full=y_left_full,
            y_stay_full=y_right_full,
            direction="left",
            place_left=True,
        )
    )
    proposals.extend(
        _direction_candidates(
            peel_atom=right_atom,
            stay_atom=left_atom,
            peel_leaf=right_leaf,
            stay_leaf=left_leaf,
            y_peel_full=y_right_full,
            y_stay_full=y_left_full,
            direction="right",
            place_left=False,
        )
    )
    proposals.sort(key=lambda item: item[0])
    return [(root_new, init_fn, meta) for _, root_new, init_fn, meta in proposals]


def _build_overlap_prefactor_peel_candidates(
    *,
    root: Node,
    target: AddNode,
    model: nn.Module,
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    log_fn: Optional[Callable[[str], None]] = None,
    max_points: int = 2048,
    max_per_direction: int = 1,
    peel_score_tol: float = 0.12,
    min_unique_grad_ratio: float = 0.05,
    min_peel_support: float = 0.80,
    min_outer_variation: float = 0.05,
    min_safe_frac: float = 0.60,
    min_safe_points: int = 96,
    min_anchor_abs_rel: float = 1.0e-6,
    min_anchor_abs: float = 1.0e-10,
) -> List[Tuple[Node, Callable, Dict[str, Any]]]:
    """Propose overlap-reducing shared-prefactor peels on direct NN+NN sums.

    v1 scope:
    - direct ``AddNode(nn(...), nn(...))`` only
    - simple inputs only (no compound coordinates)
    - singleton shared-variable peels only
    """
    def _log(msg: str):
        if log_fn is None:
            return
        try:
            log_fn(msg)
        except Exception:
            pass

    # Defensive: the Stage-B rule filters to direct AddNode(nn, nn) targets
    # already, but we keep the builder standalone for tests and future reuse.
    prep = _prepare_overlap_peel_context(
        root=root,
        target=target,
        expected_type=AddNode,
        model=model,
        train_loader=train_loader,
        device=device,
        dtype=dtype,
        max_points=max_points,
        log_fn=log_fn,
        label="overlap_prefactor_peel",
        structure_desc="direct NN+NN target",
    )
    if prep is None:
        return []

    left_atom = prep.left_atom
    right_atom = prep.right_atom
    left_leaf = prep.left_leaf
    right_leaf = prep.right_leaf
    x_full = prep.x_full
    med = prep.med
    y_left_full = prep.y_left_full
    y_right_full = prep.y_right_full
    inner_nn_kwargs = dict(prep.inner_nn_kwargs)
    factor_segments = max(4, min(int(prep.num_segments), max(4, int(prep.num_segments) // 2)))
    factor_nn_kwargs = {
        "num_segments": int(factor_segments),
        "dual_layer": bool(prep.dual_layer),
    }

    proposals: List[Tuple[float, Node, Callable, Dict[str, Any]]] = []

    def _make_init_fn(
        *,
        factor_tag: str,
        peeled_tag: str,
        stay_tag: str,
        x_factor_cpu: torch.Tensor,
        y_factor_cpu: torch.Tensor,
        x_peeled_cpu: torch.Tensor,
        y_peeled_cpu: torch.Tensor,
        x_stay_cpu: torch.Tensor,
        y_stay_cpu: torch.Tensor,
        direction: str,
        peeled_global: int,
        safe_frac: float,
    ) -> Callable[[Node, nn.Module], None]:
        """Bind init-time tensors/tags per candidate.

        This avoids late-binding closure bugs when multiple peel candidates are
        produced from the same target.
        """

        def _init_fn(
            root_new: Node,
            model_new: nn.Module,
            _ftag: str = factor_tag,
            _ptag: str = peeled_tag,
            _stag: str = stay_tag,
            _xf: torch.Tensor = x_factor_cpu,
            _yf: torch.Tensor = y_factor_cpu,
            _xp: torch.Tensor = x_peeled_cpu,
            _yp: torch.Tensor = y_peeled_cpu,
            _xs: torch.Tensor = x_stay_cpu,
            _ys: torch.Tensor = y_stay_cpu,
            _direction: str = direction,
            _peeled_global: int = peeled_global,
            _safe_frac: float = safe_frac,
        ):
            try:
                atom_to_leaf_new = build_atom_to_leaf_map(root_new, model_new)
            except Exception:
                print(
                    "[Stage B] overlap_prefactor_peel init: atom-to-leaf map failed "
                    f"for {_direction}-peel x{int(_peeled_global)}"
                )
                return

            tag_to_leaf_new: Dict[str, nn.Module] = {}
            for atom in _collect_all_atoms(root_new):
                if isinstance(atom, AtomNode) and getattr(atom, "tag", None) is not None:
                    leaf_mod = atom_to_leaf_new.get(id(atom), None)
                    if leaf_mod is not None:
                        tag_to_leaf_new[str(atom.tag)] = leaf_mod

            leaf_factor = tag_to_leaf_new.get(_ftag)
            leaf_peeled = tag_to_leaf_new.get(_ptag)
            leaf_stay = tag_to_leaf_new.get(_stag)
            if leaf_factor is None or leaf_peeled is None or leaf_stay is None:
                print(
                    "[Stage B] overlap_prefactor_peel init: missing new leaves "
                    f"for {_direction}-peel x{int(_peeled_global)}"
                )
                return

            loss_factor = _fit_leaf_to_profile_adam(leaf_factor, _xf, _yf)
            loss_peeled = _fit_leaf_to_profile_adam(leaf_peeled, _xp, _yp)
            loss_stay = _fit_leaf_to_profile_adam(leaf_stay, _xs, _ys)
            print(
                "[Stage B] overlap_prefactor_peel init "
                f"({_direction}, x{int(_peeled_global)}): "
                f"M={loss_factor:.3e}, peeled={loss_peeled:.3e}, stay={loss_stay:.3e}, "
                f"safe_frac={_safe_frac:.3f}"
            )

        return _init_fn

    def _direction_candidates(
        *,
        peel_atom: AtomNode,
        stay_atom: AtomNode,
        peel_leaf: nn.Module,
        stay_leaf: nn.Module,
        y_peel_full: torch.Tensor,
        y_stay_full: torch.Tensor,
        direction: str,
    ) -> List[Tuple[float, Node, Callable, Dict[str, Any]]]:
        peel_vars = [int(v) for v in peel_atom.var_idxs]
        stay_vars = [int(v) for v in stay_atom.var_idxs]
        peel_set = set(peel_vars)
        stay_set = set(stay_vars)
        shared = sorted(peel_set & stay_set)
        unique = sorted(peel_set - stay_set)
        _log(
            f"[Stage B] overlap_prefactor_peel: probing {direction}-peel "
            f"peel_vars={peel_vars} stay_vars={stay_vars} shared={shared} unique={unique}"
        )
        if not shared or not unique:
            _log(
                f"[Stage B] overlap_prefactor_peel: {direction}-peel skipped "
                f"(shared={shared}, unique={unique})"
            )
            return []

        gathered = _gather_nn_atom_value_grad_hess(
            root=root,
            model=model,
            atom=peel_atom,
            train_loader=train_loader,
            device=device,
            dtype=dtype,
            max_points=max_points,
        )
        if gathered is None:
            _log(f"[Stage B] overlap_prefactor_peel: {direction}-peel gather failed")
            return []
        _X_loc, _X_raw, u, du, H = gathered

        local_map = {int(v): i for i, v in enumerate(peel_vars)}
        unique_locals = [local_map[v] for v in unique]
        if not unique_locals:
            _log(f"[Stage B] overlap_prefactor_peel: {direction}-peel has no unique locals")
            return []

        grad_meds = torch.median(torch.abs(du), dim=0).values
        all_grad = float(grad_meds.max().item()) if grad_meds.numel() else 0.0
        uniq_grad = (
            float(grad_meds[unique_locals].max().item()) if unique_locals else 0.0
        )
        if (not math.isfinite(all_grad)) or all_grad <= 1.0e-12:
            _log(
                f"[Stage B] overlap_prefactor_peel: {direction}-peel rejected "
                f"(all_grad={all_grad:.3e})"
            )
            return []
        grad_ratio = uniq_grad / max(all_grad, 1.0e-30)
        _log(
            f"[Stage B] overlap_prefactor_peel: {direction}-peel grad check "
            f"uniq_grad={uniq_grad:.3e}, all_grad={all_grad:.3e}, ratio={grad_ratio:.3e}"
        )
        if uniq_grad < float(min_unique_grad_ratio) * all_grad:
            _log(
                f"[Stage B] overlap_prefactor_peel: {direction}-peel rejected "
                f"(grad_ratio={grad_ratio:.3e} < {float(min_unique_grad_ratio):.3e})"
            )
            return []

        out: List[Tuple[float, Node, Callable, Dict[str, Any]]] = []
        for peeled_global in shared:
            peeled_local = local_map[peeled_global]
            rest_locals = [j for j in range(len(peel_vars)) if j != peeled_local]
            peel_score, peel_support = _singleton_prefactor_peel_score(
                u, du, H, t_local=peeled_local, rest_locals=rest_locals
            )
            _log(
                f"[Stage B] overlap_prefactor_peel: {direction}-peel test x{int(peeled_global)} "
                f"score={float(peel_score):.3e}, support={float(peel_support):.3f}"
            )
            if peel_score > float(peel_score_tol):
                _log(
                    f"[Stage B] overlap_prefactor_peel: reject x{int(peeled_global)} "
                    f"(score {float(peel_score):.3e} > tol {float(peel_score_tol):.3e})"
                )
                continue
            if peel_support < float(min_peel_support):
                _log(
                    f"[Stage B] overlap_prefactor_peel: reject x{int(peeled_global)} "
                    f"(support {float(peel_support):.3f} < {float(min_peel_support):.3f})"
                )
                continue

            # Canonical outer factor: peel-side leaf with all non-peeled variables
            # fixed at their median values.
            x_outer = x_full.clone()
            for g in peel_vars:
                if int(g) != int(peeled_global):
                    x_outer[:, int(g)] = med[int(g)]
            with torch.no_grad():
                outer_target = peel_leaf(_build_atom_input_tensor(peel_atom, x_outer)).squeeze(-1)

            outer_rms = float(torch.sqrt(torch.mean(outer_target ** 2)).item())
            # Use median-centred RMS as a robust spread estimate. This is not a
            # classical variance, but it is much less sensitive to outliers.
            outer_centered = outer_target - torch.median(outer_target)
            outer_var = float(torch.sqrt(torch.mean(outer_centered ** 2)).item())
            if outer_rms <= 1.0e-12:
                _log(
                    f"[Stage B] overlap_prefactor_peel: reject x{int(peeled_global)} "
                    f"(outer_rms={outer_rms:.3e})"
                )
                continue
            outer_rel_var = outer_var / max(outer_rms, 1.0e-12)
            _log(
                f"[Stage B] overlap_prefactor_peel: x{int(peeled_global)} outer factor "
                f"rms={outer_rms:.3e}, rel_var={outer_rel_var:.3e}"
            )
            if outer_rel_var < float(min_outer_variation):
                _log(
                    f"[Stage B] overlap_prefactor_peel: reject x{int(peeled_global)} "
                    f"(outer rel_var {outer_rel_var:.3e} < {float(min_outer_variation):.3e})"
                )
                continue

            # Inner peeled branch target: fix only the peeled variable.
            x_inner = x_full.clone()
            x_inner[:, int(peeled_global)] = med[int(peeled_global)]
            with torch.no_grad():
                inner_raw = peel_leaf(_build_atom_input_tensor(peel_atom, x_inner)).squeeze(-1)

            # Anchor value to remove the remaining constant gauge.
            x_anchor = x_full[:1].clone()
            for g in peel_vars:
                x_anchor[:, int(g)] = med[int(g)]
            with torch.no_grad():
                anchor_val = float(
                    peel_leaf(_build_atom_input_tensor(peel_atom, x_anchor)).reshape(-1)[0].item()
                )
            anchor_floor = max(float(min_anchor_abs_rel) * outer_rms, float(min_anchor_abs))
            if (not math.isfinite(anchor_val)) or abs(anchor_val) < anchor_floor:
                _log(
                    f"[Stage B] overlap_prefactor_peel: reject x{int(peeled_global)} "
                    f"(anchor={anchor_val:.3e} < floor={anchor_floor:.3e})"
                )
                continue
            inner_target = inner_raw / anchor_val

            safe_eps = max(1.0e-4 * outer_rms, 1.0e-8)
            safe_mask = torch.isfinite(outer_target) & (torch.abs(outer_target) > safe_eps)
            safe_frac = float(safe_mask.float().mean().item())
            safe_n = int(safe_mask.sum().item())
            if safe_frac < float(min_safe_frac) or safe_n < int(min_safe_points):
                _log(
                    f"[Stage B] overlap_prefactor_peel: reject x{int(peeled_global)} "
                    f"(safe_frac={safe_frac:.3f}, safe_n={safe_n})"
                )
                continue
            stay_target = y_stay_full[safe_mask] / outer_target[safe_mask]

            reduced_vars = tuple(int(v) for v in peel_vars if int(v) != int(peeled_global))
            if not reduced_vars:
                _log(
                    f"[Stage B] overlap_prefactor_peel: reject x{int(peeled_global)} "
                    "(would remove all peeled-side variables)"
                )
                continue

            peel_kw = _fresh_atom_kwargs(getattr(peel_atom, "kwargs", None) or {})
            stay_kw = _fresh_atom_kwargs(getattr(stay_atom, "kwargs", None) or {})
            peel_kw.pop("input_expr", None)
            peel_kw.pop("extra_var_idxs", None)
            stay_kw.pop("input_expr", None)
            stay_kw.pop("extra_var_idxs", None)
            peel_kw.update(inner_nn_kwargs)
            stay_kw.update(inner_nn_kwargs)

            base_tag = (
                f"{getattr(target.left, 'tag', 'L')}_{getattr(target.right, 'tag', 'R')}"
                f"_opp_{direction[0]}_x{int(peeled_global)}"
            )
            factor_tag = f"{base_tag}_M"
            peeled_tag = f"{base_tag}_A"
            stay_tag = f"{base_tag}_B"

            factor_atom = AtomNode(
                "nn",
                (int(peeled_global),),
                kwargs=dict(factor_nn_kwargs),
                tag=factor_tag,
            )
            peeled_inner = AtomNode(
                "nn",
                reduced_vars,
                kwargs=dict(peel_kw),
                tag=peeled_tag,
            )
            stay_inner = AtomNode(
                "nn",
                tuple(int(v) for v in stay_vars),
                kwargs=dict(stay_kw),
                tag=stay_tag,
            )

            if direction == "left":
                inner_add = AddNode(peeled_inner, stay_inner)
            else:
                inner_add = AddNode(stay_inner, peeled_inner)
            cand_subtree = MulNode(factor_atom, inner_add)
            cand_root = replace_atom_in_ast(root, target, cand_subtree)

            x_factor_cpu = x_full[:, [int(peeled_global)]].detach().cpu()
            y_factor_cpu = outer_target.detach().cpu()
            x_peeled_cpu = x_full[:, list(reduced_vars)].detach().cpu()
            y_peeled_cpu = inner_target.detach().cpu()
            x_stay_cpu = x_full[safe_mask][:, stay_vars].detach().cpu()
            y_stay_cpu = stay_target.detach().cpu()
            _init_fn = _make_init_fn(
                factor_tag=factor_tag,
                peeled_tag=peeled_tag,
                stay_tag=stay_tag,
                x_factor_cpu=x_factor_cpu,
                y_factor_cpu=y_factor_cpu,
                x_peeled_cpu=x_peeled_cpu,
                y_peeled_cpu=y_peeled_cpu,
                x_stay_cpu=x_stay_cpu,
                y_stay_cpu=y_stay_cpu,
                direction=direction,
                peeled_global=int(peeled_global),
                safe_frac=float(safe_frac),
            )

            signature = (
                "overlap_prefactor_peel",
                tuple(int(v) for v in left_atom.var_idxs),
                tuple(int(v) for v in right_atom.var_idxs),
                str(direction),
                int(peeled_global),
            )
            metadata = {
                "structural": True,
                "partial_sep": True,
                "has_overlap": True,
                "overlap": True,
                "pattern_family": "overlap_prefactor_peel",
                "direction": str(direction),
                "peeled_var": int(peeled_global),
                "peel_score": float(peel_score),
                "log": (
                    f"[Stage B] overlap_prefactor_peel ({direction}) "
                    f"x{int(peeled_global)} score={float(peel_score):.3e}"
                ),
                "signature": signature,
            }
            _log(
                f"[Stage B] overlap_prefactor_peel: ACCEPT proposal "
                f"{direction}-peel x{int(peeled_global)} -> reduced_vars={list(reduced_vars)}, "
                f"stay_vars={stay_vars}, safe_frac={safe_frac:.3f}"
            )
            out.append((float(peel_score), cand_root, _init_fn, metadata))

        out.sort(key=lambda item: item[0])
        if not out:
            _log(f"[Stage B] overlap_prefactor_peel: no valid {direction}-peel proposals")
        return out[: max(1, int(max_per_direction))]

    proposals.extend(
        _direction_candidates(
            peel_atom=left_atom,
            stay_atom=right_atom,
            peel_leaf=left_leaf,
            stay_leaf=right_leaf,
            y_peel_full=y_left_full,
            y_stay_full=y_right_full,
            direction="left",
        )
    )
    proposals.extend(
        _direction_candidates(
            peel_atom=right_atom,
            stay_atom=left_atom,
            peel_leaf=right_leaf,
            stay_leaf=left_leaf,
            y_peel_full=y_right_full,
            y_stay_full=y_left_full,
            direction="right",
        )
    )
    proposals.sort(key=lambda item: item[0])
    return [(root_new, init_fn, meta) for _, root_new, init_fn, meta in proposals]


# ---------------------------------------------------------------------------
# Affine-in-variable splitting algorithms
# ---------------------------------------------------------------------------


def _detect_affine_variable(
    H: torch.Tensor,
    m: int,
    *,
    threshold: float = 0.05,
    eps: float = 1e-12,
    start_idx: int = 0,
) -> Optional[int]:
    """Detect if u is affine in one of its variables by checking if H[t,t] ≈ 0.

    If the second derivative w.r.t. variable t is approximately zero for all
    data points, then u is affine (linear) in t:
        u(z, t) = A(z) + t * B(z)

    Parameters
    ----------
    H : torch.Tensor
        Hessian tensor of shape [N, m, m] where N is number of points.
    m : int
        Number of variables.
    threshold : float
        Relative threshold for detecting affine dependence.
        If median(|H[t,t]|) / median(|H|) < threshold, variable t is affine.
    eps : float
        Small constant to avoid division by zero.
    start_idx : int
        Start checking from this index. For compound atoms, use start_idx=1
        to skip the compound variable slot at index 0.

    Returns
    -------
    int or None
        Index of the first affine variable found, or None if none detected.
    """
    if m < 2:
        return None

    # Compute median absolute value of full Hessian for scale reference
    H_flat = H.reshape(-1)
    H_scale = float(H_flat.abs().median().item())
    if not math.isfinite(H_scale) or H_scale < eps:
        H_scale = float(eps)

    # Check each variable for affine dependence (starting from start_idx)
    for t in range(start_idx, m):
        H_tt = H[:, t, t]
        H_tt_med = float(H_tt.abs().median().item())
        if not math.isfinite(H_tt_med):
            continue

        rel_H_tt = H_tt_med / H_scale
        if rel_H_tt < threshold:
            return t

    return None


def _fit_affine_split(
    X: torch.Tensor,
    u: torch.Tensor,
    du: torch.Tensor,
    H: torch.Tensor,
    t: int,
    *,
    eps: float = 1e-12,
    independence_tol: float = 0.10,
) -> Optional[Dict[str, Any]]:
    """Extract A(z) and B(z) from u(z, t) = A(z) + t * B(z).

    For a function affine in variable t:
        B(z) = ∂u/∂t  (the slope w.r.t. t)
        A(z) = u - t * B  (the intercept)

    Parameters
    ----------
    X : torch.Tensor
        Input data of shape [N, m].
    u : torch.Tensor
        Function values of shape [N].
    du : torch.Tensor
        Gradient of shape [N, m].
    H : torch.Tensor
        Hessian of shape [N, m, m].
    t : int
        Index of the affine variable.
    eps : float
        Small constant for numerical stability.
    independence_tol : float
        Tolerance for checking that dA/dt ≈ 0 and dB/dt ≈ 0.

    Returns
    -------
    dict or None
        Dictionary with keys: 'A_data', 'B_data', 't_idx', 'z_idxs', 'rel_err'
        or None if validation fails.
    """
    N, m = X.shape
    if t < 0 or t >= m:
        return None

    # Compute B = ∂u/∂t (the slope)
    B_data = du[:, t].clone()

    # Compute A = u - t * B (the intercept)
    x_t = X[:, t]
    A_data = u - x_t * B_data

    # Validate: check that dA/dt ≈ 0 and dB/dt ≈ 0
    # dA/dt = du/dt - B - t * dB/dt = B - B - t * H[t,t] = -t * H[t,t]
    # Since H[t,t] ≈ 0 (we detected affine), this should be small.
    # dB/dt = H[t,t] ≈ 0

    # Check H[t,t] is small relative to other Hessian entries
    H_tt = H[:, t, t]
    H_tt_med = float(H_tt.abs().median().item())

    # Compute scale from non-tt Hessian entries
    H_other = []
    for i in range(m):
        for j in range(m):
            if i != t or j != t:
                H_other.append(H[:, i, j].abs().median().item())
    if H_other:
        H_other_scale = float(max(H_other))
    else:
        H_other_scale = 1.0

    if H_other_scale < eps:
        H_other_scale = eps

    rel_err = H_tt_med / H_other_scale
    if rel_err > independence_tol:
        return None

    # Build indices for non-affine variables (z)
    z_idxs = [i for i in range(m) if i != t]

    return {
        "A_data": A_data.detach(),
        "B_data": B_data.detach(),
        "t_idx": int(t),
        "z_idxs": z_idxs,
        "rel_err": float(rel_err),
        "X": X.detach(),
    }


def _build_affine_split_candidate(
    *,
    root: Node,
    target: AtomNode,
    model: nn.Module,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    max_points: int = 4096,
    affine_threshold: float = 0.05,
    independence_tol: float = 0.10,
    units_spec=None,
    enforce_units: bool = False,
) -> Tuple[Optional[Node], Optional[Callable], Optional[Dict]]:
    """Try to rewrite a multivariate NN atom u(x) as:

        u(x) → A(z) + x_t * B(z)

    where x_t is detected as an affine variable (∂²u/∂x_t² ≈ 0) and z represents
    all other variables.

    This converts a 2D problem NN[z, x_t] into two 1D problems:
    - A(z): a univariate NN that can be solved by existing trig rules
    - B(z): another univariate NN that can also be solved

    Returns
    -------
    cand_root : Node or None
        New AST with affine split structure.
    init_fn : Callable or None
        Custom initialization function.
    metadata : dict or None
        Dictionary with 'signature', 'log', 'structural' keys.
    """
    if not isinstance(target, AtomNode) or str(target.kind).lower() != "nn":
        return None, None, None

    m = effective_arity(target)
    if m < 2:
        return None, None, None

    # Gather u, du, H from the fitted model
    data = _gather_nn_atom_value_grad_hess(
        root=root,
        model=model,
        atom=target,
        train_loader=train_loader,
        device=device,
        dtype=dtype,
        max_points=max_points,
    )
    if data is None:
        return None, None, None

    X, X_raw, u, du, H = data

    # Build coordinate map (unified for compound and simple atoms).
    coords = _leaf_coord_info(target)

    # Detect affine variable (including z at index 0 for compound atoms)
    t = _detect_affine_variable(H, m, threshold=affine_threshold, start_idx=0)
    if t is None:
        return None, None, None

    _affine_split_log(f"[affine_split] Detected affine variable t={t} for NN vars={target.var_idxs}")

    # Fit the affine split
    fit_result = _fit_affine_split(
        X, u, du, H, t, independence_tol=independence_tol
    )
    if fit_result is None:
        _affine_split_log(f"[affine_split] Fit validation failed for t={t}")
        return None, None, None

    A_data = fit_result["A_data"]
    B_data = fit_result["B_data"]
    z_idxs = fit_result["z_idxs"]
    rel_err = fit_result["rel_err"]

    _affine_split_log(f"[affine_split] Fit succeeded: t={t}, z_idxs={z_idxs}, rel_err={rel_err:.3e}")

    # Map local affine axis `t` to global via coord info.
    t_coord = coords[t]
    t_global = t_coord["global"]  # None when affine axis is nontrivial expr

    # Build child inputs uniformly from remaining coordinate expressions.
    A_var_idxs, A_inputs = _partition_to_child_inputs(target, z_idxs)
    B_var_idxs, B_inputs = _partition_to_child_inputs(target, z_idxs)
    A_glob = list(A_var_idxs)
    B_glob = list(B_var_idxs)
    A_atom_kwargs: Dict[str, Any] = {}
    B_atom_kwargs: Dict[str, Any] = {}

    # --- Dimensional reachability check for A atom ---
    skip_A = False
    if enforce_units and units_spec is not None:
        from nestynet_sr.sr_core.units import infer_atom_output_dim, _dim_in_rational_span
        target_dim = infer_atom_output_dim(root, target, units_spec)
        if target_dim is not None:
            A_input_dims = [units_spec.x_dims[i] for i in A_glob]
            if not _dim_in_rational_span(target_dim, A_input_dims):
                skip_A = True
                _affine_split_log(
                    f"[affine_split] A output dim {target_dim} unreachable "
                    f"from inputs {A_glob} — using zero-intercept split"
                )

    # Compute signature for deduplication
    from .engine import atom_content_hash

    signature = (
        atom_content_hash(target),
        "affine_split",
        int(t),
    )

    # Create deterministic tags
    parent_tag = target.tag if target.tag is not None else f"aff_{id(target)}"
    tag_A = f"{parent_tag}_A"
    tag_B = f"{parent_tag}_B"
    tag_t = f"{parent_tag}_xt"

    # Infer NN hyperparameters from root
    num_segments, dual_layer = _infer_nn_hyperparams_from_root(root)
    nn_kwargs = {"num_segments": int(num_segments), "dual_layer": bool(dual_layer)}

    # Build AST
    nn_B = AtomNode("nn", tuple(int(j) for j in B_glob), kwargs={**nn_kwargs, **B_atom_kwargs}, tag=tag_B, inputs=B_inputs)

    if t_global is None:
        z_expr = clone_ast(t_coord["expr"])
        product = MulNode(z_expr, nn_B)
    else:
        var_t = AtomNode("var", (int(t_global),), kwargs={}, tag=tag_t)
        product = MulNode(var_t, nn_B)

    if skip_A:
        new_subtree = product
    else:
        nn_A = AtomNode("nn", tuple(int(j) for j in A_glob), kwargs={**nn_kwargs, **A_atom_kwargs}, tag=tag_A, inputs=A_inputs)
        new_subtree = AddNode(nn_A, product)

    cand_root = replace_atom_in_ast(root, target, new_subtree)

    # Prepare initialization data: always use the *leaf input coordinates*.
    # For compound atoms, X already contains [z, extras...].
    X_init = X[:, z_idxs].detach().cpu()
    A_cpu = A_data.detach().cpu()
    B_cpu = B_data.detach().cpu()

    # DEBUG: Show z_idxs and X_init shape for affine_split verification
    _affine_split_log(f"[affine_split DEBUG] z_idxs={z_idxs}, X.shape={X.shape}, X_init.shape={X_init.shape}")
    _affine_split_log(f"[affine_split DEBUG] A_glob={A_glob}, B_glob={B_glob}, A_inputs={A_inputs}, B_inputs={B_inputs}")

    def _init_fn(root_new: Node, model_new: nn.Module):
        """Initialize A and B leaves to match the extracted data."""
        _affine_split_log(f"[affine_split] _init_fn called, looking for tags: {tag_A}, {tag_B}")
        try:
            atom_to_leaf_new = build_atom_to_leaf_map(root_new, model_new)
        except Exception as e:
            _affine_split_log(f"[affine_split] _init_fn: build_atom_to_leaf_map failed: {e}")
            return

        def _leaf_param_device_dtype(mod: nn.Module) -> Tuple[torch.device, torch.dtype]:
            for p in mod.parameters(recurse=True):
                if isinstance(p, torch.Tensor):
                    return p.device, p.dtype
            return device, dtype

        def _lm_fit_leaf_to_data(
            leaf_mod: nn.Module,
            x_data: torch.Tensor,
            y_data: torch.Tensor,
            *,
            epochs: int = 50,
            chisq_tol: float = 1e-12,
        ) -> bool:
            """Fit a leaf to match target data via LM solver (scale-aware success)."""
            try:
                import nestynet
                from torch.utils.data import DataLoader, TensorDataset

                dev, dt = _leaf_param_device_dtype(leaf_mod)
                x_all = x_data.to(dev, dt)
                y_all = y_data.to(dev, dt).reshape(-1, 1)

                # Subsample and split into train/val (80/20) to avoid overlap check
                n = x_all.shape[0]
                perm = torch.randperm(n, device=x_all.device)
                n_use = min(n, 1024)
                n_train = int(n_use * 0.8)
                n_val = n_use - n_train

                idx_train = perm[:n_train]
                idx_val = perm[n_train:n_train + n_val]

                x_train, y_train = x_all[idx_train], y_all[idx_train]
                x_val, y_val = x_all[idx_val], y_all[idx_val]

                dl_train = DataLoader(
                    TensorDataset(x_train, y_train),
                    batch_size=x_train.shape[0],
                    shuffle=False,
                )
                dl_val = DataLoader(
                    TensorDataset(x_val, y_val),
                    batch_size=x_val.shape[0],
                    shuffle=False,
                )

                def fac(dataloader):
                    def f(_):
                        return nestynet.optimizer.ResidualsModule(
                            providers=[leaf_mod],
                            dataloader=dataloader,
                            device=dev,
                        )
                    return f

                from nestynet_sr.sr_search.training import (
                    SR_LM_OVERRIDES,
                    _sr_latest_single_target_loss_metrics,
                )

                cfg = nestynet.optimizer.LMConfig(
                    verbose=False,
                    LM_strategy="direct_solve",
                    chisq_tol=chisq_tol,
                    log_to_console=False,
                    **SR_LM_OVERRIDES,
                )
                lm_opt = nestynet.optimizer.Predictive_LM_Optimizer(
                    list(leaf_mod.parameters()),
                    [fac(dl_train)],
                    residual_module_factories_val=[fac(dl_val)],
                    cfg=cfg,
                )

                best = float("inf")
                for _ in range(int(epochs)):
                    loss_obj, loss_val_obj = lm_opt.step()
                    loss_metrics = _sr_latest_single_target_loss_metrics(
                        lm_opt, label="[affine_split] "
                    )
                    raw_val = loss_metrics.get("val_data_mean_loss", loss_val_obj)
                    loss_val = None if raw_val is None else float(raw_val)
                    if loss_val is not None:
                        best = min(best, float(loss_val))
                    if lm_opt.state.get("halt"):
                        break

                # Compute scale-aware success metric on training data
                with torch.no_grad():
                    yp = leaf_mod(x_train)
                    yp = yp[:, 0] if yp.ndim == 2 else yp.reshape(-1)
                    yt = y_train[:, 0]
                    rmse = float((yp - yt).pow(2).mean().sqrt().item())
                    mad = float((yt - yt.median()).abs().median().item())
                    rel = rmse / (mad + 1e-12)

                _affine_split_log(f"[affine_split] LM fit: best={best:.3e} rmse={rmse:.3e} rmse/MAD={rel:.3e}")
                return rel < 0.05  # Success if RMSE < 5% of MAD

            except Exception as e:
                _affine_split_log(f"[affine_split] LM fit failed: {e}")
                if _AFFINE_SPLIT_DEBUG:
                    import traceback
                    traceback.print_exc()
                try:
                    leaf_mod.eval()
                except Exception:
                    pass
                return False

        # DEBUG: Show A_data and B_data scales to diagnose LM fit quality
        if not skip_A:
            _affine_split_log(f"[affine_split DEBUG] A_data: min={A_cpu.min():.3e}, max={A_cpu.max():.3e}, median={A_cpu.median():.3e}")
        _affine_split_log(f"[affine_split DEBUG] B_data: min={B_cpu.min():.3e}, max={B_cpu.max():.3e}, median={B_cpu.median():.3e}")

        # Find and initialize A and B leaves
        found_A = skip_A  # no A leaf to find when skipped
        found_B = False
        for a in _collect_all_atoms(root_new):
            if not isinstance(a, AtomNode) or str(a.kind).lower() != "nn":
                continue
            leaf_mod = atom_to_leaf_new.get(id(a))
            if leaf_mod is None:
                continue

            if not skip_A and getattr(a, "tag", None) == tag_A:
                found_A = True
                # DEBUG: Check leaf n_in vs X_init shape
                core = getattr(leaf_mod, 'model', leaf_mod)
                n_in = getattr(core, 'n_in', 'unknown')
                _affine_split_log(f"[affine_split DEBUG] A leaf n_in={n_in}, X_init.shape={X_init.shape}, var_idxs={a.var_idxs}")
                success = _lm_fit_leaf_to_data(leaf_mod, X_init, A_cpu)
                _affine_split_log(f"[affine_split] Initialized A leaf: success={success}")
            elif getattr(a, "tag", None) == tag_B:
                found_B = True
                # DEBUG: Check leaf n_in vs X_init shape
                core = getattr(leaf_mod, 'model', leaf_mod)
                n_in = getattr(core, 'n_in', 'unknown')
                _affine_split_log(f"[affine_split DEBUG] B leaf n_in={n_in}, X_init.shape={X_init.shape}, var_idxs={a.var_idxs}")
                success = _lm_fit_leaf_to_data(leaf_mod, X_init, B_cpu)
                _affine_split_log(f"[affine_split] Initialized B leaf: success={success}")

        if not found_A or not found_B:
            _affine_split_log(f"[affine_split] Warning: found_A={found_A}, found_B={found_B}")

    _init_fn._after_analytic_init = True

    # Build metadata
    from nestynet_sr.sr_core.bridges import ast_to_human_readable
    if t_global is None:
        t_str = ast_to_human_readable(compound_input_expr(target))
    else:
        t_str = f"x{t_global}"
    z_str = ", ".join(f"x{i}" for i in A_glob)
    if skip_A:
        log_message = (
            f"[Stage B]  Trying affine_split (zero-intercept): NN({', '.join(f'x{i}' for i in target.var_idxs)}) "
            f"→ {t_str} * B({z_str}) [rel_err={rel_err:.2e}]"
        )
    else:
        log_message = (
            f"[Stage B]  Trying affine_split: NN({', '.join(f'x{i}' for i in target.var_idxs)}) "
            f"→ A({z_str}) + {t_str} * B({z_str}) [rel_err={rel_err:.2e}]"
        )
    metadata = {"signature": signature, "log": log_message, "structural": True}

    return cand_root, _init_fn, metadata


def _eval_poly_design_and_grads(
    XA: torch.Tensor,
    degree: int,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """Return (Phi, dPhi_list, exps) for PolyLeaf-compatible basis on XA.

    Phi shape: [N, K]
    dPhi_list: list length d, each [N, K]
    exps shape: [K, d] (int64)
    """
    N, d = XA.shape
    exps_list = _enumerate_exponents(d, int(degree))
    exps = torch.tensor(exps_list, device=XA.device, dtype=torch.int64)
    Phi = _eval_monomials(XA, exps)
    dPhi_list: List[torch.Tensor] = []
    for i in range(d):
        exps_i = exps.clone()
        e_i = exps_i[:, i].to(XA.dtype)
        # reduce exponent along i (safe for e_i=0 because multiplier is 0)
        exps_i[:, i] = torch.clamp(exps_i[:, i] - 1, min=0)
        Phi_i = _eval_monomials(XA, exps_i)
        dPhi_list.append(Phi_i * e_i)
    return Phi, dPhi_list, exps


def _eval_counterterm_design_and_grads(
    XA: torch.Tensor,
    degree: int,
    *,
    basis: str = "poly",
    power: Optional[int] = None,
    eps: float = 1e-12,
) -> Optional[Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]]:
    """Return a counterterm basis and derivatives with respect to XA.

    ``poly`` is the historical full polynomial basis.  ``power`` is a sparse
    one-term basis c*z**p, and ``poly_inv`` is a full polynomial in 1/z.  The
    learned residual NN still receives z, not the transformed coordinate; only
    the explicit counterterm basis changes.
    """
    basis_s = str(basis or "poly").lower()
    if basis_s == "poly":
        return _eval_poly_design_and_grads(XA, int(degree))

    if int(XA.shape[1]) != 1:
        return None
    z = XA[:, 0]
    if not bool(torch.isfinite(z).all()):
        return None

    if basis_s == "power":
        if power is None:
            return None
        p = int(power)
        if p < 0 and not bool((z.abs() > float(eps)).all()):
            return None
        try:
            phi = z.pow(p).unsqueeze(1)
            dphi = (float(p) * z.pow(p - 1)).unsqueeze(1)
        except Exception:
            return None
        if not (bool(torch.isfinite(phi).all()) and bool(torch.isfinite(dphi).all())):
            return None
        exps = torch.tensor([[p]], device=XA.device, dtype=torch.int64)
        return phi, [dphi], exps

    if basis_s == "poly_inv":
        if not bool((z.abs() > float(eps)).all()):
            return None
        try:
            w = z.reciprocal()
            exps_list = _enumerate_exponents(1, int(degree))
            exps = torch.tensor(exps_list, device=XA.device, dtype=torch.int64)
            Phi = _eval_monomials(w.unsqueeze(1), exps)
            dPhi_cols: List[torch.Tensor] = []
            for e_row in exps:
                e = int(e_row[0].item())
                if e == 0:
                    dPhi_cols.append(torch.zeros_like(z))
                else:
                    # d/dz (1/z)^e = -e * (1/z)^(e+1)
                    dPhi_cols.append((-float(e)) * w.pow(e + 1))
            dPhi = torch.stack(dPhi_cols, dim=1) if dPhi_cols else Phi.new_zeros(Phi.shape)
        except Exception:
            return None
        if not (bool(torch.isfinite(Phi).all()) and bool(torch.isfinite(dPhi).all())):
            return None
        return Phi, [dPhi], exps

    return None


def _poly_D_from_exps(
    exps: torch.Tensor,
    *,
    w_cross: float = 25.0,
    deg_pow: float = 2.0,
    eps: float = 1e-18,
) -> torch.Tensor:
    """Build a diagonal simplicity penalty for PolyLeaf monomial bases.

    The basis is specified by integer exponent tuples `exps[k]`.

    We penalize high-degree and multivariate (cross) monomials:
        D_k ∝ (deg_k**deg_pow) * (1 + w_cross*(nnz_k-1))

    Constant term gets 0 penalty.

    Returns
    -------
    D: [K] tensor (float64) median-normalized so median(D[D>0])==1.
    """
    if exps is None or int(exps.numel()) == 0:
        return torch.zeros((0,), dtype=torch.float64)

    e = exps.to(torch.float64)
    deg = e.sum(dim=1)
    nnz = (e > 0).sum(dim=1)

    D = deg.clamp_min(0.0).pow(float(deg_pow))
    D = D * (1.0 + float(w_cross) * (nnz - 1.0).clamp_min(0.0))
    D = D.clone()
    D[deg == 0] = 0.0

    pos = D[D > 0]
    if int(pos.numel()) > 0:
        med = pos.median()
        if float(med.item()) > float(eps):
            D = D / med
    return D


def _poly_no_cross_mask(exps: torch.Tensor) -> torch.Tensor:
    """Mask of monomials that touch at most one variable (no cross terms)."""
    if exps is None or int(exps.numel()) == 0:
        return torch.zeros((0,), dtype=torch.bool)
    nnz = (exps > 0).sum(dim=1)
    return nnz <= 1


def _col_median_abs(Phi: torch.Tensor, *, eps: float = 1e-18) -> torch.Tensor:
    """Robust per-column scale for design matrices (median |Phi_k|)."""
    s = Phi.abs().median(dim=0).values
    return s.clamp_min(float(eps))


def _smallest_eigvec_reg(
    M: torch.Tensor,
    D: torch.Tensor,
    *,
    eta: float = 1e-2,
    eps: float = 1e-18,
) -> torch.Tensor:
    """Return smallest-eigenvector of (M^T M + lam*diag(D)).

    `eta` is a scale-free knob; lam is set from median(diag(M^T M)).
    """
    M_sq = M.T @ M
    diag_med = M_sq.diag().median()
    diag_med_v = float(diag_med.item()) if torch.isfinite(diag_med) else 1.0
    if not math.isfinite(diag_med_v) or diag_med_v <= float(eps):
        diag_med_v = 1.0
    lam = float(eta) * diag_med_v

    if int(D.numel()) != int(M_sq.shape[0]):
        # Defensive: fallback to unregularized solve
        try:
            _, eigvecs = torch.linalg.eigh(M_sq)
            c = eigvecs[:, 0]
            return c / c.norm().clamp_min(float(eps))
        except Exception:
            c = torch.randn((int(M_sq.shape[0]),), device=M_sq.device, dtype=M_sq.dtype)
            return c / c.norm().clamp_min(float(eps))

    A = M_sq + torch.diag((float(lam) * D.to(M_sq.dtype)))
    try:
        _, eigvecs = torch.linalg.eigh(A)
        c = eigvecs[:, 0]
    except Exception:
        # Rare: fall back to unregularized
        _, eigvecs = torch.linalg.eigh(M_sq)
        c = eigvecs[:, 0]
    return c / c.norm().clamp_min(float(eps))


def _sparsify_coeffs_v2(
    c: torch.Tensor,
    *,
    Phi_hat: torch.Tensor,
    exps_hat: torch.Tensor,
    relerr_fn,
    tau: float = 1e-2,
    Kmax: int = 4,
    keep_const: bool = True,
    allow_cross: bool = False,
    eps: float = 1e-18,
) -> Tuple[torch.Tensor, float]:
    """Sparsify a unit-norm eigenvector in a monomial basis.

    Operates in the *column-normalized* basis (Phi_hat). Contribution is |c_k|
    because Phi_hat columns have comparable scale.
    """
    if int(c.numel()) == 0:
        return c, float("inf")

    c = c / c.norm().clamp_min(float(eps))
    best_c = c
    best_e = float(relerr_fn(best_c))

    # Optional cross-term suppression (even if they exist in basis)
    if (not allow_cross) and (exps_hat is not None) and int(exps_hat.numel()) > 0:
        nnz = (exps_hat > 0).sum(dim=1)
        cross = nnz > 1
    else:
        cross = torch.zeros_like(c, dtype=torch.bool)

    contrib = c.abs().clone()
    contrib[cross] = 0.0
    idx = torch.argsort(contrib, descending=True)

    # Top-K search
    for K in range(1, int(Kmax) + 1):
        m = torch.zeros_like(c, dtype=torch.bool)
        m[idx[:K]] = True
        if keep_const and int(m.numel()) > 0:
            m[0] = True
        cK = c * m
        n = cK.norm()
        if float(n.item()) <= float(eps):
            continue
        cK = cK / n
        e = float(relerr_fn(cK))
        if e < best_e:
            best_c, best_e = cK, e

    # Simple thresholding
    thr = float(tau) * float(c.abs().max().clamp_min(float(eps)).item())
    m = c.abs() >= thr
    m = m & (~cross)
    if keep_const and int(m.numel()) > 0:
        m[0] = True
    cT = c * m
    n = cT.norm()
    if float(n.item()) > float(eps):
        cT = cT / n
        e = float(relerr_fn(cT))
        # Only keep if it doesn't materially worsen the identity
        if e <= best_e * 1.02:
            best_c, best_e = cT, e

    return best_c, float(best_e)


def _ridge_solve(
    M: torch.Tensor, bvec: torch.Tensor, ridge: float = 1e-8
) -> Optional[torch.Tensor]:
    """Ridge-stabilised least squares: solve (M.T @ M + ridge*I) c = M.T @ bvec."""
    K = int(M.shape[1])
    MtM = M.T @ M
    Mtb = M.T @ bvec
    MtM = MtM + float(ridge) * torch.eye(K, device=MtM.device, dtype=MtM.dtype)
    try:
        c = torch.linalg.solve(MtM, Mtb)
    except Exception:
        return None
    return c


def _fit_counterterm_polys_two_sided_for_mul_split(
    *,
    X: torch.Tensor,
    u: torch.Tensor,
    du: torch.Tensor,
    H: torch.Tensor,
    A: List[int],
    B: List[int],
    degree_A: int,
    degree_B: int,
    n_alt: int = 3,
    ridge: float = 1e-8,
    eps: float = 1e-12,
    variant: str = "both",  # "both", "A_only", or "B_only"
    basis_A: str = "poly",
    basis_B: str = "poly",
    power_A: Optional[int] = None,
    power_B: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Fit P_A(x_A) + P_B(x_B) such that r = u - P_A - P_B is approximately
    multiplicatively separable between A and B.

    Uses alternating minimisation:
      1. Fix P_B, solve for P_A via the counterterm identity on x_A.
      2. Fix P_A, solve for P_B via the counterterm identity on x_B.

    Returns the fitted polynomial coefficients and separability residual metric.
    """
    m = int(X.shape[1])
    if len(A) < 1 or len(B) < 1 or (len(A) + len(B) != m):
        return None

    XA = X[:, A]
    XB = X[:, B]
    basis_res_A = _eval_counterterm_design_and_grads(
        XA, int(degree_A), basis=str(basis_A), power=power_A, eps=float(eps)
    )
    basis_res_B = _eval_counterterm_design_and_grads(
        XB, int(degree_B), basis=str(basis_B), power=power_B, eps=float(eps)
    )
    if basis_res_A is None or basis_res_B is None:
        return None
    PhiA, dPhiA_list, expsA = basis_res_A
    PhiB, dPhiB_list, expsB = basis_res_B
    KA = int(PhiA.shape[1])
    KB = int(PhiB.shape[1])
    if KA == 0 or KB == 0:
        print(f"[counterterm fit] A={A} B={B}: Empty basis (KA={KA}, KB={KB})")
        return None

    # Map local A-index -> position inside XA (0..|A|-1)
    posA = {int(a): ia for ia, a in enumerate(A)}
    posB = {int(b): ib for ib, b in enumerate(B)}

    # Alternating minimisation
    cA = torch.zeros(KA, device=X.device, dtype=X.dtype)
    cB = torch.zeros(KB, device=X.device, dtype=X.dtype)

    # Track best solution across iterations (alternating min can oscillate)
    best_cA = cA.clone()
    best_cB = cB.clone()
    best_rel_err = float("inf")

    # Determine which sides to fit based on variant
    fit_A = variant in ("both", "A_only")
    fit_B = variant in ("both", "B_only")

    for iter_idx in range(n_alt):
        # Step 1: Fix P_B, solve for P_A
        if fit_A:
            PB = PhiB @ cB  # [N]
            PBx_by_b = {b: (dPhiB_list[posB[b]] @ cB) for b in B}  # each is [N]
            uA = u - PB  # effective target for A-side
            rows_A: List[torch.Tensor] = []
            rhs_A: List[torch.Tensor] = []
            for a in A:
                ia = posA[int(a)]
                dPhi_a = dPhiA_list[ia]  # [N,KA]
                u_a = du[:, a]
                for b in B:
                    u_ab = H[:, a, b]
                    rb = du[:, b] - PBx_by_b[b]
                    M_ab = (-u_ab).unsqueeze(1) * PhiA + rb.unsqueeze(1) * dPhi_a
                    b_ab = u_a * rb - u_ab * uA  # uA = u - PB
                    rows_A.append(M_ab)
                    rhs_A.append(b_ab)
            M_A = torch.cat(rows_A, dim=0)
            bvec_A = torch.cat(rhs_A, dim=0)
            if M_A.shape[0] < max(KA * 2, 64):
                print(
                    f"[counterterm fit] A={A} B={B}: Not enough rows for A (rows={M_A.shape[0]}, need>={max(KA * 2, 64)})"
                )
                return None
            cA_new = _ridge_solve(M_A, bvec_A, ridge)
            if cA_new is None:
                print(f"[counterterm fit] A={A} B={B}: Ridge solve failed for A-side")
                return None
            cA = cA_new

        # Step 2: Fix P_A, solve for P_B
        if fit_B:
            PA = PhiA @ cA  # [N]
            PAx_by_a = {a: (dPhiA_list[posA[a]] @ cA) for a in A}  # each is [N]
            uB = u - PA  # effective target for B-side
            rows_B: List[torch.Tensor] = []
            rhs_B: List[torch.Tensor] = []
            for b in B:
                ib = posB[int(b)]
                dPhi_b = dPhiB_list[ib]  # [N,KB]
                u_b = du[:, b]
                for a in A:
                    u_ab = H[:, a, b]
                    ra = du[:, a] - PAx_by_a[a]
                    M_ba = (-u_ab).unsqueeze(1) * PhiB + ra.unsqueeze(1) * dPhi_b
                    b_ba = ra * u_b - u_ab * uB  # uB = u - PA
                    rows_B.append(M_ba)
                    rhs_B.append(b_ba)
            M_B = torch.cat(rows_B, dim=0)
            bvec_B = torch.cat(rhs_B, dim=0)
            if M_B.shape[0] < max(KB * 2, 64):
                print(
                    f"[counterterm fit] A={A} B={B}: Not enough rows for B (rows={M_B.shape[0]}, need>={max(KB * 2, 64)})"
                )
                return None
            cB_new = _ridge_solve(M_B, bvec_B, ridge)
            if cB_new is None:
                print(f"[counterterm fit] A={A} B={B}: Ridge solve failed for B-side")
                return None
            cB = cB_new

        # Compute residual error after this iteration and track best
        PA_temp = PhiA @ cA
        PB_temp = PhiB @ cB
        r_temp = u - PA_temp - PB_temp
        PAx_temp = torch.stack([dPhiA_list[posA[a]] @ cA for a in A], dim=1)
        PBx_temp = torch.stack([dPhiB_list[posB[b]] @ cB for b in B], dim=1)
        errs_temp = []
        dens_temp = []
        for a in A:
            ia = posA[a]
            ra_temp = du[:, a] - PAx_temp[:, ia]
            for b in B:
                ib = posB[b]
                rb_temp = du[:, b] - PBx_temp[:, ib]
                u_ab = H[:, a, b]
                lhs = u_ab * r_temp
                rhs2 = ra_temp * rb_temp
                e = (lhs - rhs2).abs()
                d = (lhs.abs() + rhs2.abs()).clamp_min(eps)
                errs_temp.append(e)
                dens_temp.append(d)
        err_temp = torch.cat(errs_temp, dim=0)
        den_temp = torch.cat(dens_temp, dim=0)
        rel_temp = (err_temp / den_temp).median().item()

        # Track best solution
        if rel_temp < best_rel_err:
            best_rel_err = rel_temp
            best_cA = cA.clone()
            best_cB = cB.clone()

    # Use best solution found during alternating minimization (not final iterate)
    cA = best_cA
    cB = best_cB

    # Compute final separability residual metric using best solution
    PA = PhiA @ cA  # [N]
    PB = PhiB @ cB  # [N]
    PAx = torch.stack([dPhiA_list[posA[a]] @ cA for a in A], dim=1)  # [N,|A|]
    PBx = torch.stack([dPhiB_list[posB[b]] @ cB for b in B], dim=1)  # [N,|B|]
    r = u - PA - PB

    # Evaluate multiplicativity identity error over all (a,b)
    errs: List[torch.Tensor] = []
    dens: List[torch.Tensor] = []
    for a in A:
        ia = posA[a]
        ra = du[:, a] - PAx[:, ia]
        for b in B:
            ib = posB[b]
            rb = du[:, b] - PBx[:, ib]
            u_ab = H[:, a, b]
            lhs = u_ab * r
            rhs2 = ra * rb
            e = (lhs - rhs2).abs()
            d = (lhs.abs() + rhs2.abs()).clamp_min(eps)
            errs.append(e)
            dens.append(d)

    err = torch.cat(errs, dim=0)
    den = torch.cat(dens, dim=0)
    rel = (err / den).median().item()

    return {
        "coeffs_A": cA.detach(),
        "coeffs_B": cB.detach(),
        "degree_A": int(degree_A),
        "degree_B": int(degree_B),
        "A": list(map(int, A)),
        "B": list(map(int, B)),
        "rel_err": float(rel),
        "exps_A": expsA.detach().cpu(),
        "exps_B": expsB.detach().cpu(),
        "variant": variant,
        "basis_A": str(basis_A),
        "basis_B": str(basis_B),
        "power_A": None if power_A is None else int(power_A),
        "power_B": None if power_B is None else int(power_B),
    }


# -----------------------------------------------------------------------------
# Counterfactor-based additive splits (u(x) = P_A(x_A) * P_B(x_B) * (g(x_A) + h(x_B)))
# -----------------------------------------------------------------------------


def _fit_counterfactor_polys_two_sided_for_add_split(
    *,
    X: torch.Tensor,
    u: torch.Tensor,
    du: torch.Tensor,
    H: torch.Tensor,
    A: List[int],
    B: List[int],
    degree_A: int,
    degree_B: int,
    n_alt: int = 3,
    eps: float = 1e-12,
    renorm_median_abs: bool = True,
    # v2 knobs: bias hard toward simple, non-pathological counterfactors
    col_normalize: bool = True,
    reg_eta: float = 1e-2,
    w_cross: float = 25.0,
    deg_pow: float = 2.0,
    allow_cross_terms: bool = False,
    sparsify_tau: float = 1e-2,
    sparsify_Kmax: int = 4,
    min_nonzero_frac: float = 0.95,
    nonzero_eps_factor: float = 1e-3,
    sign_stability_frac: float = 0.0,
    min_gain_factor: float = 0.20,
) -> Optional[Dict[str, Any]]:
    """Fit P_A(x_A) * P_B(x_B) such that r = u / (P_A * P_B) is approximately
    additively separable between A and B.

    The additive-separability target is r(x_A, x_B) ≈ g(x_A) + h(x_B), for which
    the cross second derivatives vanish: r_{ab} = 0.

    We avoid explicit division by P_A P_B by enforcing an equivalent *homogeneous*
    identity derived from r_{ab}=0.

    Alternating minimisation:
      - Fix P_B, solve for P_A via a homogeneous least-squares (smallest-eigenvector)
      - Fix P_A, solve for P_B similarly

    Returns coeffs for both polynomials and a robust relative residual metric.
    """
    m = int(X.shape[1])
    if len(A) < 1 or len(B) < 1 or (len(A) + len(B) != m):
        return None

    XA = X[:, A]
    XB = X[:, B]
    PhiA, dPhiA_list, expsA = _eval_poly_design_and_grads(XA, int(degree_A))
    PhiB, dPhiB_list, expsB = _eval_poly_design_and_grads(XB, int(degree_B))
    KA = int(PhiA.shape[1])
    KB = int(PhiB.shape[1])
    if KA == 0 or KB == 0:
        return None

    posA = {int(a): ia for ia, a in enumerate(A)}
    posB = {int(b): ib for ib, b in enumerate(B)}

    # Column-normalize the monomial basis: Phi_hat = Phi / median(|Phi|).
    # This makes coefficients comparable across monomials and stabilizes eigen-solves.
    if bool(col_normalize):
        sA = _col_median_abs(PhiA, eps=float(eps))
        sB = _col_median_abs(PhiB, eps=float(eps))
        PhiA_hat = PhiA / sA
        PhiB_hat = PhiB / sB
        dPhiA_hat_list = [dPhi / sA for dPhi in dPhiA_list]
        dPhiB_hat_list = [dPhi / sB for dPhi in dPhiB_list]
    else:
        sA = torch.ones((KA,), device=PhiA.device, dtype=PhiA.dtype)
        sB = torch.ones((KB,), device=PhiB.device, dtype=PhiB.dtype)
        PhiA_hat = PhiA
        PhiB_hat = PhiB
        dPhiA_hat_list = list(dPhiA_list)
        dPhiB_hat_list = list(dPhiB_list)

    # Optionally exclude cross-terms entirely (default v2 behavior).
    if bool(allow_cross_terms):
        maskA = torch.ones((KA,), device=PhiA.device, dtype=torch.bool)
        maskB = torch.ones((KB,), device=PhiB.device, dtype=torch.bool)
    else:
        maskA = _poly_no_cross_mask(expsA).to(device=PhiA.device)
        maskB = _poly_no_cross_mask(expsB).to(device=PhiB.device)
        # Defensive: always keep constant term
        if int(maskA.numel()) > 0:
            maskA[0] = True
        if int(maskB.numel()) > 0:
            maskB[0] = True

    PhiA_sel = PhiA_hat[:, maskA]
    PhiB_sel = PhiB_hat[:, maskB]
    dPhiA_sel_list = [dPhi[:, maskA] for dPhi in dPhiA_hat_list]
    dPhiB_sel_list = [dPhi[:, maskB] for dPhi in dPhiB_hat_list]
    expsA_sel = expsA[maskA]
    expsB_sel = expsB[maskB]
    KA_sel = int(PhiA_sel.shape[1])
    KB_sel = int(PhiB_sel.shape[1])
    if KA_sel == 0 or KB_sel == 0:
        return None

    # Simplicity regularizer (diagonal) for eigen-solves.
    D_A = _poly_D_from_exps(
        expsA_sel, w_cross=float(w_cross), deg_pow=float(deg_pow), eps=float(eps)
    ).to(device=X.device)
    D_B = _poly_D_from_exps(
        expsB_sel, w_cross=float(w_cross), deg_pow=float(deg_pow), eps=float(eps)
    ).to(device=X.device)

    # Deterministic initialisation (stable, reproducible): start from constants.
    cA_sel = torch.zeros((KA_sel,), device=X.device, dtype=X.dtype)
    cB_sel = torch.zeros((KB_sel,), device=X.device, dtype=X.dtype)
    cA_sel[0] = 1.0
    cB_sel[0] = 1.0

    best_cA_sel = cA_sel.clone()
    best_cB_sel = cB_sel.clone()
    best_rel = float("inf")

    def _rel_err_sel(cA_s: torch.Tensor, cB_s: torch.Tensor) -> float:
        PA = PhiA_sel @ cA_s
        PB = PhiB_sel @ cB_s
        PAx_by_a = {a: (dPhiA_sel_list[posA[a]] @ cA_s) for a in A}
        PBx_by_b = {b: (dPhiB_sel_list[posB[b]] @ cB_s) for b in B}
        errs: List[torch.Tensor] = []
        dens: List[torch.Tensor] = []
        for a in A:
            u_a = du[:, a]
            PA_a = PAx_by_a[a]
            for b in B:
                u_b = du[:, b]
                u_ab = H[:, a, b]
                PB_b = PBx_by_b[b]
                t1 = PA * (u_ab * PB - u_a * PB_b)
                t2 = PA_a * (u * PB_b - u_b * PB)
                e = (t1 + t2).abs()
                d = (t1.abs() + t2.abs()).clamp_min(eps)
                errs.append(e)
                dens.append(d)
        rel = (torch.cat(errs, dim=0) / torch.cat(dens, dim=0)).median().item()
        return float(rel)

    # Null (no counterfactor) baseline: PA=1, PB=1.
    rel_null = _rel_err_sel(cA_sel, cB_sel)

    for _ in range(int(n_alt)):
        # Step 1: Fix B, solve A (regularized smallest-eigenvector)
        PB = PhiB_sel @ cB_sel
        PBx_by_b = {b: (dPhiB_sel_list[posB[b]] @ cB_sel) for b in B}

        rows_A: List[torch.Tensor] = []
        for b in B:
            PB_b = PBx_by_b[b]
            u_b = du[:, b]
            L_b = u * PB_b - u_b * PB
            for a in A:
                u_a = du[:, a]
                u_ab = H[:, a, b]
                K_ab = u_ab * PB - u_a * PB_b
                dPhiA_a = dPhiA_sel_list[posA[a]]
                rows_A.append(K_ab.unsqueeze(1) * PhiA_sel + L_b.unsqueeze(1) * dPhiA_a)

        if not rows_A:
            return None
        M_A = torch.cat(rows_A, dim=0)
        cA_sel = _smallest_eigvec_reg(M_A, D_A, eta=float(reg_eta), eps=float(eps))
        cA_sel, _ = _sparsify_coeffs_v2(
            cA_sel,
            Phi_hat=PhiA_sel,
            exps_hat=expsA_sel,
            relerr_fn=lambda c: _rel_err_sel(c, cB_sel),
            tau=float(sparsify_tau),
            Kmax=int(sparsify_Kmax),
            keep_const=True,
            allow_cross=bool(allow_cross_terms),
            eps=float(eps),
        )

        # Step 2: Fix A, solve B
        PA = PhiA_sel @ cA_sel
        PAx_by_a = {a: (dPhiA_sel_list[posA[a]] @ cA_sel) for a in A}

        rows_B: List[torch.Tensor] = []
        for a in A:
            u_a = du[:, a]
            PA_a = PAx_by_a[a]
            L_a = PA_a * u - PA * u_a
            for b in B:
                u_b = du[:, b]
                u_ab = H[:, a, b]
                K_ab = PA * u_ab - PA_a * u_b
                dPhiB_b = dPhiB_sel_list[posB[b]]
                rows_B.append(K_ab.unsqueeze(1) * PhiB_sel + L_a.unsqueeze(1) * dPhiB_b)

        if not rows_B:
            return None
        M_B = torch.cat(rows_B, dim=0)
        cB_sel = _smallest_eigvec_reg(M_B, D_B, eta=float(reg_eta), eps=float(eps))
        cB_sel, _ = _sparsify_coeffs_v2(
            cB_sel,
            Phi_hat=PhiB_sel,
            exps_hat=expsB_sel,
            relerr_fn=lambda c: _rel_err_sel(cA_sel, c),
            tau=float(sparsify_tau),
            Kmax=int(sparsify_Kmax),
            keep_const=True,
            allow_cross=bool(allow_cross_terms),
            eps=float(eps),
        )

        rel = _rel_err_sel(cA_sel, cB_sel)
        if rel < best_rel:
            best_rel = rel
            best_cA_sel = cA_sel.clone()
            best_cB_sel = cB_sel.clone()

    cA_sel = best_cA_sel
    cB_sel = best_cB_sel

    # Require a meaningful gain over the null counterfactor baseline.
    if (
        float(min_gain_factor) > 0.0
        and math.isfinite(float(rel_null))
        and float(rel_null) > float(eps)
    ):
        if not (float(best_rel) <= float(min_gain_factor) * float(rel_null)):
            return None

    # Expand from masked, normalized basis back to PolyLeaf coefficients.
    cA_hat_full = torch.zeros((KA,), device=X.device, dtype=X.dtype)
    cB_hat_full = torch.zeros((KB,), device=X.device, dtype=X.dtype)
    cA_hat_full[maskA] = cA_sel
    cB_hat_full[maskB] = cB_sel
    cA = cA_hat_full / sA.to(dtype=X.dtype, device=X.device)
    cB = cB_hat_full / sB.to(dtype=X.dtype, device=X.device)

    # Optional renormalisation: stabilise scale by forcing median |P| ≈ 1
    if bool(renorm_median_abs):
        PA_full = PhiA @ cA
        PB_full = PhiB @ cB
        sA_med = PA_full.abs().median().clamp_min(eps)
        sB_med = PB_full.abs().median().clamp_min(eps)
        cA = cA / sA_med
        cB = cB / sB_med

    # Guard: counterfactors should not vanish on a large fraction of points.
    PA_full = PhiA @ cA
    PB_full = PhiB @ cB
    epsA = float(nonzero_eps_factor) * float(PA_full.abs().median().clamp_min(eps).item())
    epsB = float(nonzero_eps_factor) * float(PB_full.abs().median().clamp_min(eps).item())
    fracA = float((PA_full.abs() > epsA).to(torch.float64).mean().item())
    fracB = float((PB_full.abs() > epsB).to(torch.float64).mean().item())
    if fracA < float(min_nonzero_frac) or fracB < float(min_nonzero_frac):
        return None

    # Optional sign-stability guard: reject wildly sign-flipping counterfactors.
    if float(sign_stability_frac) > 0.0:

        def _sign_majority_frac(v: torch.Tensor, epsv: float) -> float:
            m = v.abs() > float(epsv)
            if not bool(m.any()):
                return 0.0
            s = torch.sign(v[m])
            pos = float((s > 0).to(torch.float64).mean().item())
            neg = float((s < 0).to(torch.float64).mean().item())
            return max(pos, neg)

        if _sign_majority_frac(PA_full, epsA) < float(sign_stability_frac):
            return None
        if _sign_majority_frac(PB_full, epsB) < float(sign_stability_frac):
            return None

    # Final identity error on the actual PolyLeaf coefficients.
    rel = float("inf")
    try:
        # Reuse the selected-basis evaluator for speed by re-projecting to the masked basis.
        cA_sel_f = cA_hat_full[maskA]
        cB_sel_f = cB_hat_full[maskB]
        rel = float(_rel_err_sel(cA_sel_f, cB_sel_f))
    except Exception:
        rel = float("inf")

    # Complexity bookkeeping
    thrA = float(1e-12) * float(cA.abs().max().clamp_min(eps).item())
    thrB = float(1e-12) * float(cB.abs().max().clamp_min(eps).item())
    n_terms_A = int((cA.abs() > thrA).sum().item())
    n_terms_B = int((cB.abs() > thrB).sum().item())

    return {
        "coeffs_A": cA.detach(),
        "coeffs_B": cB.detach(),
        "degree_A": int(degree_A),
        "degree_B": int(degree_B),
        "A": list(map(int, A)),
        "B": list(map(int, B)),
        "rel_err": float(rel),
        "rel_err_null": float(rel_null),
        "n_terms_A": int(n_terms_A),
        "n_terms_B": int(n_terms_B),
        "allow_cross_terms": bool(allow_cross_terms),
        "exps_A": expsA.detach().cpu(),
        "exps_B": expsB.detach().cpu(),
    }


def _build_counterfactor_add_split_candidate(
    *,
    root: Node,
    target: AtomNode,
    model: nn.Module,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    degrees_A: Tuple[int, ...] = (1, 2),
    degrees_B: Tuple[int, ...] = (1, 2),
    n_alt: int = 3,
    max_points: int = 4096,
    # v2: this identity is only a *structure probe*; LM will ultimately decide.
    # Keep a moderately loose default to avoid false negatives.
    rel_err_tol: float = 5e-2,
) -> Tuple[Optional[Node], Optional[Callable], Optional[Dict]]:
    """Try to rewrite a multivariate NN atom u(x) as:

        u(x)  →  P_A(x_A) * P_B(x_B) * (nn(x_A) + nn(x_B))

    where P_A, P_B are low-degree polynomials ("counterfactors") chosen so that
    r = u / (P_A P_B) is approximately additively separable between A and B.

    Note
    ----
    The additive split is not unique: (g+h) is defined up to a constant shift.
    Earlier sketches used extra scalar (β, α) leaves; those are redundant
    because a constant offset and global scaling can be absorbed into the
    two NN leaves and/or the counterfactors.

    Returns:
        cand_root: New AST or None
        init_fn: Custom initialization or None
        metadata: Dict with "signature", "log", "structural" keys, or None
    """
    if not isinstance(target, AtomNode) or str(target.kind).lower() != "nn":
        return None, None, None
    if effective_arity(target) < 2:
        return None, None, None

    data = _gather_nn_atom_value_grad_hess(
        root=root,
        model=model,
        atom=target,
        train_loader=train_loader,
        device=device,
        dtype=dtype,
        max_points=max_points,
    )
    if data is None:
        return None, None, None
    X, _, u, du, H = data  # Ignore X_raw (not needed here)
    m = int(X.shape[1])
    if m > 8:
        return None, None, None

    parts = _enumerate_unique_partitions(m)
    if m >= 4:
        balanced = [p for p in parts if min(len(p[0]), len(p[1])) >= 2]
        parts = balanced if balanced else parts

    def _search_best(
        *,
        rel_tol: float,
        min_gain_factor: float,
        min_nonzero_frac: float,
        nonzero_eps_factor: float,
        reg_eta: float,
        sparsify_tau: float,
        sparsify_Kmax: int,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Tuple[float, int, int, int]]]:
        """Find best (A,B,PA,PB) split under a given probe configuration."""
        best: Optional[Dict[str, Any]] = None
        best_key: Optional[Tuple[float, int, int, int]] = None

        for A, B in parts:
            for dA in degrees_A:
                for dB in degrees_B:
                    # v2 search: prefer no-cross counterfactors; allow cross-terms only as fallback.
                    for allow_cross in (False, True):
                        # If degree is <=1 on both sides, there are no cross terms anyway.
                        if allow_cross and (int(dA) <= 1 and int(dB) <= 1):
                            continue
                        res = _fit_counterfactor_polys_two_sided_for_add_split(
                            X=X,
                            u=u,
                            du=du,
                            H=H,
                            A=A,
                            B=B,
                            degree_A=int(dA),
                            degree_B=int(dB),
                            n_alt=int(n_alt),
                            allow_cross_terms=bool(allow_cross),
                            # Probe knobs (relaxed in a second pass if needed).
                            min_gain_factor=float(min_gain_factor),
                            min_nonzero_frac=float(min_nonzero_frac),
                            nonzero_eps_factor=float(nonzero_eps_factor),
                            reg_eta=float(reg_eta),
                            sparsify_tau=float(sparsify_tau),
                            sparsify_Kmax=int(sparsify_Kmax),
                        )
                        if res is None:
                            continue

                        rel = float(res.get("rel_err", float("inf")))
                        if not math.isfinite(rel):
                            continue
                        if rel > float(rel_tol):
                            continue

                        n_terms = int(res.get("n_terms_A", 10**9)) + int(res.get("n_terms_B", 10**9))
                        allow_flag = int(bool(res.get("allow_cross_terms", False)))
                        deg_sum = int(res.get("degree_A", int(dA))) + int(res.get("degree_B", int(dB)))
                        key = (rel, n_terms, allow_flag, deg_sum)
                        if (best_key is None) or (key < best_key):
                            best_key = key
                            best = res

        return best, best_key

    # ------------------------------------------------------------------
    # Pass 1 (strict): keep the original v2 probe settings.
    #
    # Pass 2 (relaxed): for tough 2D leaves (common in AIF) we sometimes need
    # to allow counterfactors that are small on a non-trivial portion of the
    # sample (e.g. z**2 when z spans several decades) and/or accept a slightly
    # noisier identity probe due to imperfect NN Hessians.
    #
    # LM validation keeps the rewrite honest.
    # ------------------------------------------------------------------
    best, _best_key = _search_best(
        rel_tol=float(rel_err_tol),
        min_gain_factor=0.20,
        min_nonzero_frac=0.95,
        nonzero_eps_factor=1e-3,
        reg_eta=1e-2,
        sparsify_tau=1e-2,
        sparsify_Kmax=4,
    )

    probe_rel_tol = float(rel_err_tol)
    weak_probe = False
    rel_tol2: Optional[float] = None

    if best is None:
        # Use a slightly looser identity tolerance for binary leaves; this is
        # where we most often see the "counterfactor trick" (factor a 1D term
        # so that what's left is additively separable).
        rel_tol2 = float(max(float(rel_err_tol), 0.15 if m == 2 else 0.10))
        best, _best_key = _search_best(
            rel_tol=rel_tol2,
            min_gain_factor=0.0,
            min_nonzero_frac=0.80,
            nonzero_eps_factor=1e-4,
            reg_eta=5e-3,
            sparsify_tau=5e-3,
            sparsify_Kmax=6,
        )
        if best is not None:
            probe_rel_tol = rel_tol2
            weak_probe = probe_rel_tol > float(rel_err_tol)

    # Pass 3 (very weak): binary leaves with sharp trigs / 1/z terms can have
    # noisy NN Hessians, which makes the identity probe overly strict.
    # If we can find a counterfactor pair that *meaningfully* reduces the probe
    # residual relative to the null (PA=PB=1), propose it and let LM validation
    # decide.
    if best is None and m == 2:
        best_any, _best_key = _search_best(
            rel_tol=float("inf"),
            min_gain_factor=0.0,
            min_nonzero_frac=0.70,
            nonzero_eps_factor=1e-4,
            reg_eta=2e-3,
            sparsify_tau=2e-3,
            sparsify_Kmax=8,
        )
        if best_any is not None:
            rel = float(best_any.get("rel_err", float("inf")))
            rel_null = float(best_any.get("rel_err_null", float("inf")))
            rel_ratio = rel / max(rel_null, 1e-12)
            # Require a meaningful reduction in identity error (to avoid proposing
            # trivial PA=PB=1 solutions), and cap how weak we allow the probe to be.
            if math.isfinite(rel) and math.isfinite(rel_null) and (rel_null > 1e-12):
                if (rel_ratio <= 0.75) and (rel <= 0.85):
                    best = best_any
                    probe_rel_tol = float(max(rel_tol2 or float(rel_err_tol), 0.85))
                    weak_probe = True

    if best is None:
        return None, None, None

    # Annotate probe strength for downstream logging / debugging.
    best["_probe_rel_tol"] = float(probe_rel_tol)
    best["_weak_probe"] = bool(weak_probe)

    # Convert local indices (0..m-1) -> global var indices via input expressions.
    # Works uniformly for simple and compound atoms.
    A_loc = best["A"]
    B_loc = best["B"]

    A_var_idxs, A_inputs = _partition_to_child_inputs(target, A_loc)
    B_var_idxs, B_inputs = _partition_to_child_inputs(target, B_loc)
    A_glob = list(A_var_idxs)
    B_glob = list(B_var_idxs)

    # Compute signature for deduplication (before building AST)
    from .engine import atom_content_hash

    partition_key = tuple(sorted([tuple(map(int, A_loc)), tuple(map(int, B_loc))]))
    signature = (
        atom_content_hash(target),  # Content-based atom hash (not id())
        partition_key,
        int(best["degree_A"]),
        int(best["degree_B"]),
        int(bool(best.get("allow_cross_terms", False))),
    )

    parent_tag = target.tag if target.tag is not None else f"cf_{id(target)}"

    tag_polyA = f"{parent_tag}_PA"
    tag_polyB = f"{parent_tag}_PB"
    tag_NA = f"{parent_tag}_NA"
    tag_NB = f"{parent_tag}_NB"

    polyA = AtomNode(
        "poly",
        tuple(int(j) for j in A_glob),
        kwargs={"degree": int(best["degree_A"]), "min_total": 0},
        tag=tag_polyA,
        inputs=_fresh_inputs(A_inputs),
    )
    polyB = AtomNode(
        "poly",
        tuple(int(j) for j in B_glob),
        kwargs={"degree": int(best["degree_B"]), "min_total": 0},
        tag=tag_polyB,
        inputs=_fresh_inputs(B_inputs),
    )

    num_segments, dual_layer = _infer_nn_hyperparams_from_root(root)
    nn_kwargs = {"num_segments": int(num_segments), "dual_layer": bool(dual_layer)}

    nnA = AtomNode(
        "nn",
        tuple(int(j) for j in A_glob),
        kwargs=dict(nn_kwargs),
        tag=tag_NA,
        inputs=_fresh_inputs(A_inputs),
    )
    nnB = AtomNode(
        "nn",
        tuple(int(j) for j in B_glob),
        kwargs=dict(nn_kwargs),
        tag=tag_NB,
        inputs=_fresh_inputs(B_inputs),
    )

    inner_sum = AddNode(nnA, nnB)
    outer = MulNode(MulNode(polyA, polyB), inner_sum)

    cand_root = replace_atom_in_ast(root, target, outer)

    coeffsA_cpu = best["coeffs_A"].detach().cpu()
    coeffsB_cpu = best["coeffs_B"].detach().cpu()

    # Robust baseline constant for r = u/(PA*PB). We only need a good *scale* so
    # that the candidate model starts close to the current fitted one.
    with torch.no_grad():
        XA = X[:, best["A"]]
        XB = X[:, best["B"]]
        PhiA, _, _ = _eval_poly_design_and_grads(XA, int(best["degree_A"]))
        PhiB, _, _ = _eval_poly_design_and_grads(XB, int(best["degree_B"]))
        PA = (PhiA @ best["coeffs_A"]).detach()
        PB = (PhiB @ best["coeffs_B"]).detach()
        denom = PA * PB
        mask = denom.abs() > 1e-8
        if mask.any():
            rat = u[mask] / denom[mask]
            c0 = float(rat.median().item())
            # Median can be ~0 for symmetric targets; fall back to a robust scale.
            if (not math.isfinite(c0)) or (abs(c0) < 1e-12):
                mag = float(rat.abs().median().item())
                if (not math.isfinite(mag)) or (mag < 1e-12):
                    mag = float(
                        (u[mask].abs().median() / (denom[mask].abs().median() + 1e-12)).item()
                    )
                sgn = float(torch.sign((u[mask] * denom[mask]).median()).item())
                if (not math.isfinite(sgn)) or (sgn == 0.0):
                    sgn = 1.0
                c0 = sgn * mag
        else:
            c0 = 1.0

    # At init we set nnA ≈ c0 (constant) and nnB ≈ 0. This keeps the full model
    # close to the current fitted one while allowing LM to subsequently shape
    # nnA/nnB into the true additive components of r.
    cA0 = float(c0)
    cB0 = 0.0

    def _init_fn(root_new: Node, model_new: nn.Module):
        try:
            atom_to_leaf_new = build_atom_to_leaf_map(root_new, model_new)
        except Exception:
            return

        # Build a quick tag -> leaf lookup. (ASTCompositeAdaptor exposes a .leaf list
        # but does not provide a .leaves dict keyed by tag.)
        tag_to_leaf_new: Dict[str, nn.Module] = {}
        for _a in _collect_all_atoms(root_new):
            if isinstance(_a, AtomNode) and getattr(_a, 'tag', None) is not None:
                _leaf_mod = atom_to_leaf_new.get(id(_a))
                if _leaf_mod is not None:
                    tag_to_leaf_new[str(_a.tag)] = _leaf_mod

        def _leaf_param_device_dtype(mod: nn.Module) -> Tuple[torch.device, torch.dtype]:
            for p in mod.parameters(recurse=True):
                if isinstance(p, torch.Tensor):
                    return p.device, p.dtype
            return device, dtype

        def _try_init_segmented_leaf_constant(
            leaf_mod: nn.Module, value: float, x_ex: torch.Tensor
        ) -> bool:
            """Fast constant initialiser for NestyNet-style segmented leaves.

            If leaf_mod.base_model.get_parameters() is available, we set K=0 and b=0
            so the output is input-independent, then scale a single output weight
            to match the requested constant.
            """
            base = getattr(leaf_mod, "base_model", None)
            if base is None or (not hasattr(base, "get_parameters")):
                return False
            try:
                a_pieces, b_pieces, c_pieces, K_pieces = base.get_parameters()
            except Exception:
                return False
            if not a_pieces:
                return False
            try:
                with torch.no_grad():
                    # Make the model constant in x.
                    for t in K_pieces:
                        if hasattr(t, "zero_"):
                            t.zero_()
                    for t in b_pieces:
                        if hasattr(t, "zero_"):
                            t.zero_()
                    if c_pieces is not None:
                        for t in c_pieces:
                            if hasattr(t, "zero_"):
                                t.zero_()
                    for t in a_pieces:
                        if hasattr(t, "zero_"):
                            t.zero_()
                    a0 = a_pieces[0].view(-1)
                    a0[0] = torch.as_tensor(1.0, device=a0.device, dtype=a0.dtype)

                # Measure the resulting constant output and scale to target.
                with torch.no_grad():
                    y0 = leaf_mod(x_ex[:1])
                y0v = float(y0.reshape(-1)[0].item())
                if (not math.isfinite(y0v)) or (abs(y0v) < 1e-20):
                    return False
                scale = float(value) / y0v
                with torch.no_grad():
                    a0 = a_pieces[0].view(-1)
                    a0[0] = a0[0] * torch.as_tensor(scale, device=a0.device, dtype=a0.dtype)
                return True
            except Exception:
                return False

        def _gd_fit_leaf_constant(
            leaf_mod: nn.Module, value: float, x_ex: torch.Tensor, *, steps: int = 25
        ) -> bool:
            """Fallback: small autograd fit to a constant on a small sample."""
            try:
                import torch.optim

                dev, dt = _leaf_param_device_dtype(leaf_mod)
                x_ex = x_ex.to(device=dev, dtype=dt)
                y_tgt = torch.full((x_ex.shape[0],), float(value), device=dev, dtype=dt)

                leaf_mod.train()
                opt = torch.optim.Adam(leaf_mod.parameters(), lr=5e-2)
                for _ in range(int(steps)):
                    opt.zero_grad(set_to_none=True)
                    y = leaf_mod(x_ex)
                    if y.dim() == 2:
                        y = y[:, 0]
                    y = y.reshape(-1)
                    loss = (y - y_tgt).pow(2).mean()
                    if not torch.isfinite(loss):
                        break
                    loss.backward()
                    opt.step()
                    if float(loss.detach().item()) < 1e-10:
                        break
                # Clear grads to keep LM happy.
                for p in leaf_mod.parameters():
                    p.grad = None
                leaf_mod.eval()
                return True
            except Exception:
                try:
                    leaf_mod.eval()
                except Exception:
                    pass
                return False

        def _set_poly(tag: str, vec_full: torch.Tensor):
            """Set coefficients for a (r)poly leaf.

            Supports both PolyLeaf and RPolyLeaf. When setting an RPolyLeaf we
            interpret vec_full as the *full* coefficient vector in the same
            monomial ordering as PolyLeaf, and then:
              - extract the leading coefficient (of the fixed leading monomial)
              - push it into the associated multiplicative scale leaf (if present)
              - normalise remaining coefficients by that leading coefficient.

            The gauge-fix rewrite pass tags participating factors with
            kwargs['_mul_scale_tag'] so we can find the scale leaf deterministically.
            """
            for a in _collect_all_atoms(root_new):
                if not (isinstance(a, AtomNode) and getattr(a, "tag", None) == tag):
                    continue
                kind = str(getattr(a, "kind", "")).lower()
                if kind not in ("poly", "polynomial", "rpoly", "r_polynomial", "rpolynomial"):
                    continue

                leaf = atom_to_leaf_new.get(id(a))
                if leaf is None:
                    continue
                core = getattr(leaf, "core", getattr(leaf, "model", leaf))

                # Plain PolyLeaf: direct copy
                if isinstance(core, PolyLeaf):
                    p = _leaf_coeff_param(leaf)
                    if p is None:
                        continue
                    if p.numel() != int(vec_full.numel()):
                        continue
                    with torch.no_grad():
                        p.copy_(vec_full.to(device=p.device, dtype=p.dtype))
                    continue

                # Monic / reduced polynomial: convert full coeff vector -> reduced coeffs
                if isinstance(core, RPolyLeaf):
                    coeffs_param = getattr(core, "coeffs", None)
                    if coeffs_param is None:
                        continue
                    vec_full_t = vec_full.to(device=coeffs_param.device, dtype=coeffs_param.dtype).view(-1)

                    # If the vector already matches reduced parameterisation, just copy.
                    if int(vec_full_t.numel()) == int(coeffs_param.numel()):
                        with torch.no_grad():
                            coeffs_param.copy_(vec_full_t)
                        continue

                    # Otherwise treat it as full PolyLeaf coefficients.
                    exps_full = getattr(core, "exps_full", None)
                    if exps_full is None:
                        continue
                    n_full = int(exps_full.shape[0])
                    if int(vec_full_t.numel()) != n_full:
                        continue

                    lp = int(getattr(core, "lead_pos", 0))
                    a_lead = vec_full_t[lp].clone()
                    if float(a_lead.abs().detach().cpu()) < 1e-12:
                        a_lead = a_lead.new_tensor(1.0)

                    # Update scale leaf if present
                    scale_tag = (getattr(a, "kwargs", None) or {}).get("_mul_scale_tag")
                    if scale_tag is not None:
                        s_leaf = tag_to_leaf_new.get(str(scale_tag))
                        if s_leaf is not None:
                            inner = getattr(s_leaf, "base_model", getattr(s_leaf, "model", s_leaf))
                            s_core = getattr(inner, "core", inner)
                            if hasattr(s_core, "value"):
                                with torch.no_grad():
                                    s_core.value.mul_(
                                        a_lead.to(device=s_core.value.device, dtype=s_core.value.dtype)
                                    )

                    # Reduced coefficients: drop leading term and normalise
                    vec_free = torch.cat([vec_full_t[:lp], vec_full_t[lp + 1 :]]) / a_lead
                    if int(vec_free.numel()) != int(coeffs_param.numel()):
                        continue
                    with torch.no_grad():
                        coeffs_param.copy_(vec_free)
                    continue

                # Fallback: if we can find a parameter of matching size, copy
                p = _leaf_coeff_param(leaf)
                if p is not None and p.numel() == int(vec_full.numel()):
                    with torch.no_grad():
                        p.copy_(vec_full.to(device=p.device, dtype=p.dtype))


        _set_poly(tag_polyA, coeffsA_cpu)
        _set_poly(tag_polyB, coeffsB_cpu)

        # Initialise the new NN leaves to sane constants so the full model stays
        # close to the currently-fitted one.
        #
        # nnA ≈ cA0, nnB ≈ cB0 (usually 0)
        XAcpu = X[:, best["A"]].detach().cpu()
        XBcpu = X[:, best["B"]].detach().cpu()
        XAcpu = XAcpu[: min(256, XAcpu.shape[0])]
        XBcpu = XBcpu[: min(256, XBcpu.shape[0])]

        def _init_nn(tag: str, value: float, Xcpu: torch.Tensor):
            for a in _collect_all_atoms(root_new):
                if not (
                    isinstance(a, AtomNode)
                    and str(a.kind).lower() == "nn"
                    and getattr(a, "tag", None) == tag
                ):
                    continue
                leaf_mod = atom_to_leaf_new.get(id(a))
                if leaf_mod is None:
                    continue
                dev, dt = _leaf_param_device_dtype(leaf_mod)
                x_ex = Xcpu.to(device=dev, dtype=dt)
                # Save random init before destructive try_init
                orig_state = {k: v.clone() for k, v in leaf_mod.state_dict().items()}
                ok = _try_init_segmented_leaf_constant(leaf_mod, float(value), x_ex)
                if not ok:
                    # Restore random init (not zero) so GD has gradients
                    leaf_mod.load_state_dict(orig_state)
                    _gd_fit_leaf_constant(leaf_mod, float(value), x_ex, steps=500)

        _init_nn(tag_NA, cA0, XAcpu)
        _init_nn(tag_NB, cB0, XBcpu)

    _init_fn._after_analytic_init = True

    # Build metadata with signature for deduplication
    A_str = ", ".join(f"x{i}" for i in A_glob)
    B_str = ", ".join(f"x{i}" for i in B_glob)
    rel_probe = float(best.get("rel_err", float("inf")))
    probe_tol_used = float(best.get("_probe_rel_tol", float(rel_err_tol)))
    weak_probe_flag = bool(best.get("_weak_probe", False))
    nA = int(best.get("n_terms_A", -1))
    nB = int(best.get("n_terms_B", -1))
    cross_flag = "cross" if bool(best.get("allow_cross_terms", False)) else "no-cross"
    probe_tag = ""
    if weak_probe_flag:
        probe_tag = f", weak-probe@tol≈{probe_tol_used:.2e}"
    elif probe_tol_used != float(rel_err_tol):
        probe_tag = f", tol≈{probe_tol_used:.2e}"

    log_message = (
        f"[Stage B]  Trying counterfactor_add_split(v2): NN({', '.join(f'x{i}' for i in target.var_idxs)}) "
        f"→ PA({A_str},deg={int(best['degree_A'])},{nA}t) * PB({B_str},deg={int(best['degree_B'])},{nB}t) * (NN(A)+NN(B)) "
        f"[{cross_flag}, rel_id≈{rel_probe:.2e}{probe_tag}]"
    )
    metadata = {
        "signature": signature,
        "log": log_message,
        "structural": True,
        "probe_rel_tol": float(probe_tol_used),
        "weak_probe": bool(weak_probe_flag),
    }

    return cand_root, _init_fn, metadata


def _build_counterterm_mul_split_candidate(
    *,
    root: Node,
    target: AtomNode,
    model: nn.Module,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    degrees_A: Tuple[int, ...] = (2,),
    degrees_B: Tuple[int, ...] = (2,),
    n_alt: int = 10,
    max_points: int = 4096,
    rel_err_tol: float = 5e-3,
    ridge: float = 1e-8,
    # Guard against degenerate 1|(m-1) partitions:
    rank1_hess_tol: float = 0.05,  # Threshold for proposing multiplicative splits
    rank1_hess_max_points: int = 1024,
) -> Tuple[Optional[Node], Optional[Callable], Optional[Dict]]:
    """Try to rewrite a multivariate NN atom u(x) as:

        u(x)  ->  poly(x_A) + poly(x_B) + nn(x_A) * nn(x_B)

    where the polynomials are discovered by a two-sided counterterm-based LS solve
    using alternating minimisation.

    Returns:
        cand_root: New AST or None
        init_fn: Custom initialization or None
        metadata: Dict with "signature", "log", "structural" keys, or None
    """
    if not isinstance(target, AtomNode) or str(target.kind).lower() != "nn":
        return None, None, None
    if effective_arity(target) < 2:
        return None, None, None

    data = _gather_nn_atom_value_grad_hess(
        root=root,
        model=model,
        atom=target,
        train_loader=train_loader,
        device=device,
        dtype=dtype,
        max_points=max_points,
    )
    if data is None:
        print(f"[counterterm] Failed to gather u/du/H data for NN vars={target.var_idxs}")
        return None, None, None
    X, _, u, du, H = data  # Ignore X_raw (not needed here)
    m = int(X.shape[1])
    print(f"[counterterm] Gathered {X.shape[0]} points for NN vars={target.var_idxs} (m={m})")

    # Heuristic guard against combinatorial explosions
    if m > 8:
        print(f"[counterterm] Skipping: too many variables (m={m} > 8)")
        return None, None, None

    best: Optional[Dict[str, Any]] = None
    n_tried = 0
    n_succeeded = 0

    parts = _enumerate_unique_partitions(m)
    parts_all = parts
    parts = _prefilter_counterterm_partitions_by_rank1_cross_hessian(
        parts=parts,
        H=H,
        m=m,
        rank1_tol=float(rank1_hess_tol),
        max_points=int(rank1_hess_max_points),
    )
    # Progressive variant search: first try sparse one-term counterterms
    # c*z, c*z^2, c/z, c/z^2.  Only if none pass the probe tolerance do we
    # escalate to the historical full-polynomial counterterms.
    total_fits = len(parts) * (len(degrees_A) + len(degrees_B) + len(degrees_A) * len(degrees_B))
    print(
        f"[counterterm] Trying {len(parts)} partitions (of {len(parts_all)} total) with progressive variants (sparse first; A_only, B_only, both) = {total_fits} dense fits"
    )

    def _consider_result(res: Optional[Dict[str, Any]], label: str) -> None:
        nonlocal best, n_succeeded
        if res is None:
            return
        n_succeeded += 1
        if (best is None) or (res["rel_err"] < best["rel_err"]):
            best = res
            print(
                f"[counterterm]   New best ({label}): "
                f"A={res['A']}, B={res['B']}, rel_err={res['rel_err']:.4e}"
            )

    # Tier 1: sparse one-term counterterms.  This keeps easy physics-style
    # counterterms from being swallowed by a broader polynomial leaf.
    sparse_records: List[Tuple[str, List[int], List[int], Optional[float]]] = []
    for A, B in parts:
        if len(A) == 1:
            for p in (1, 2, -1, -2):
                n_tried += 1
                label = f"A_only_sparse_p{p:+d}"
                res = _fit_counterterm_polys_two_sided_for_mul_split(
                    X=X,
                    u=u,
                    du=du,
                    H=H,
                    A=A,
                    B=B,
                    degree_A=abs(int(p)),
                    degree_B=2,
                    n_alt=int(n_alt),
                    ridge=ridge,
                    variant="A_only",
                    basis_A="power",
                    power_A=int(p),
                )
                sparse_records.append(
                    (
                        label,
                        list(map(int, A)),
                        list(map(int, B)),
                        None if res is None else float(res["rel_err"]),
                    )
                )
                _consider_result(res, label)
        if len(B) == 1:
            for p in (1, 2, -1, -2):
                n_tried += 1
                label = f"B_only_sparse_p{p:+d}"
                res = _fit_counterterm_polys_two_sided_for_mul_split(
                    X=X,
                    u=u,
                    du=du,
                    H=H,
                    A=A,
                    B=B,
                    degree_A=2,
                    degree_B=abs(int(p)),
                    n_alt=int(n_alt),
                    ridge=ridge,
                    variant="B_only",
                    basis_B="power",
                    power_B=int(p),
                )
                sparse_records.append(
                    (
                        label,
                        list(map(int, A)),
                        list(map(int, B)),
                        None if res is None else float(res["rel_err"]),
                    )
                )
                _consider_result(res, label)

    if sparse_records:
        ok_sparse = [r for r in sparse_records if r[3] is not None]
        print(
            f"[counterterm] Sparse probe summary: tried={len(sparse_records)}, "
            f"finite={len(ok_sparse)}, tol={float(rel_err_tol):.3e}"
        )
        if ok_sparse:
            for label, A, B, rel in sorted(ok_sparse, key=lambda r: float(r[3]))[:5]:
                print(
                    f"[counterterm]   sparse {label}: A={A}, B={B}, "
                    f"rel_err={float(rel):.4e}"
                )
        else:
            print("[counterterm]   sparse probes produced no finite fits")

    if best is not None and best["rel_err"] <= float(rel_err_tol):
        print(
            f"[counterterm] Sparse counterterm accepted for proposal construction "
            f"(rel_err={best['rel_err']:.4e} <= tol={rel_err_tol:.4e}); skipping dense counterterm fits"
        )

    # Tier 2: dense counterterms.  This is the old full polynomial behavior plus
    # full polynomial-in-inverse for 1D sides.
    if best is None or best["rel_err"] > float(rel_err_tol):
        for A, B in parts:
            # Try variant 1: A_only (simpler - only one poly block)
            for deg_A in degrees_A:
                n_tried += 1
                res = _fit_counterterm_polys_two_sided_for_mul_split(
                    X=X,
                    u=u,
                    du=du,
                    H=H,
                    A=A,
                    B=B,
                    degree_A=int(deg_A),
                    degree_B=2,  # Dummy value, not used for A_only
                    n_alt=int(n_alt),
                    ridge=ridge,
                    variant="A_only",
                )
                _consider_result(res, f"A_only_poly_deg{int(deg_A)}")
                if len(A) == 1:
                    n_tried += 1
                    res = _fit_counterterm_polys_two_sided_for_mul_split(
                        X=X,
                        u=u,
                        du=du,
                        H=H,
                        A=A,
                        B=B,
                        degree_A=int(deg_A),
                        degree_B=2,
                        n_alt=int(n_alt),
                        ridge=ridge,
                        variant="A_only",
                        basis_A="poly_inv",
                    )
                    _consider_result(res, f"A_only_poly_inv_deg{int(deg_A)}")

            # Try variant 2: B_only (simpler - only one poly block)
            for deg_B in degrees_B:
                n_tried += 1
                res = _fit_counterterm_polys_two_sided_for_mul_split(
                    X=X,
                    u=u,
                    du=du,
                    H=H,
                    A=A,
                    B=B,
                    degree_A=2,  # Dummy value, not used for B_only
                    degree_B=int(deg_B),
                    n_alt=int(n_alt),
                    ridge=ridge,
                    variant="B_only",
                )
                _consider_result(res, f"B_only_poly_deg{int(deg_B)}")
                if len(B) == 1:
                    n_tried += 1
                    res = _fit_counterterm_polys_two_sided_for_mul_split(
                        X=X,
                        u=u,
                        du=du,
                        H=H,
                        A=A,
                        B=B,
                        degree_A=2,
                        degree_B=int(deg_B),
                        n_alt=int(n_alt),
                        ridge=ridge,
                        variant="B_only",
                        basis_B="poly_inv",
                    )
                    _consider_result(res, f"B_only_poly_inv_deg{int(deg_B)}")

            # Try variant 3: both (more complex - two poly blocks)
            for deg_A in degrees_A:
                for deg_B in degrees_B:
                    n_tried += 1
                    res = _fit_counterterm_polys_two_sided_for_mul_split(
                        X=X,
                        u=u,
                        du=du,
                        H=H,
                        A=A,
                        B=B,
                        degree_A=int(deg_A),
                        degree_B=int(deg_B),
                        n_alt=int(n_alt),
                        ridge=ridge,
                        variant="both",
                    )
                    _consider_result(res, f"both_poly_deg{int(deg_A)}_{int(deg_B)}")
                    if len(A) == 1:
                        n_tried += 1
                        res = _fit_counterterm_polys_two_sided_for_mul_split(
                            X=X,
                            u=u,
                            du=du,
                            H=H,
                            A=A,
                            B=B,
                            degree_A=int(deg_A),
                            degree_B=int(deg_B),
                            n_alt=int(n_alt),
                            ridge=ridge,
                            variant="both",
                            basis_A="poly_inv",
                        )
                        _consider_result(res, f"both_Ainv_deg{int(deg_A)}_{int(deg_B)}")
                    if len(B) == 1:
                        n_tried += 1
                        res = _fit_counterterm_polys_two_sided_for_mul_split(
                            X=X,
                            u=u,
                            du=du,
                            H=H,
                            A=A,
                            B=B,
                            degree_A=int(deg_A),
                            degree_B=int(deg_B),
                            n_alt=int(n_alt),
                            ridge=ridge,
                            variant="both",
                            basis_B="poly_inv",
                        )
                        _consider_result(res, f"both_Binv_deg{int(deg_A)}_{int(deg_B)}")

    print(f"[counterterm] Tried {n_tried} fits, {n_succeeded} succeeded")
    if best is None:
        print("[counterterm] No successful fits found")
        return None, None, None
    if best["rel_err"] > float(rel_err_tol):
        print(
            f"[counterterm] Best rel_err={best['rel_err']:.4e} > tol={rel_err_tol:.4e}, rejecting"
        )
        return None, None, None

    # Convert local indices (0..m-1) -> global var indices via input expressions.
    # Works uniformly for simple and compound atoms.
    A_loc = best["A"]
    B_loc = best["B"]

    A_var_idxs, A_inputs = _partition_to_child_inputs(target, A_loc)
    B_var_idxs, B_inputs = _partition_to_child_inputs(target, B_loc)
    A_glob = list(A_var_idxs)
    B_glob = list(B_var_idxs)
    basis_A = str(best.get("basis_A", "poly"))
    basis_B = str(best.get("basis_B", "poly"))
    power_A = best.get("power_A", None)
    power_B = best.get("power_B", None)

    # Compute signature for deduplication (before building AST)
    from .engine import atom_content_hash

    partition_key = tuple(sorted([tuple(map(int, A_loc)), tuple(map(int, B_loc))]))
    signature = (
        atom_content_hash(target),  # Content-based atom hash (not id())
        partition_key,
        int(best.get("degree_A", 2)),
        int(best.get("degree_B", 2)),
        str(best.get("variant", "both")),
        str(basis_A),
        None if power_A is None else int(power_A),
        str(basis_B),
        None if power_B is None else int(power_B),
    )

    num_segments, dual_layer = _infer_nn_hyperparams_from_root(root)
    nn_kwargs = {"num_segments": int(num_segments), "dual_layer": bool(dual_layer)}

    # Deterministic tag scheme for reuse across subsequent Stage-B rewrites
    parent_tag = target.tag if target.tag is not None else f"ct_{id(target)}"
    tag_polyA = f"{parent_tag}_PA"
    tag_polyB = f"{parent_tag}_PB"
    tag_scaleA = f"{parent_tag}_PA_scale"
    tag_scaleB = f"{parent_tag}_PB_scale"
    tag_L = f"{parent_tag}_L"
    tag_R = f"{parent_tag}_R"

    def _counterterm_node(
        *,
        side: str,
        basis: str,
        power: Optional[int],
        degree: int,
        var_idxs: List[int],
        inputs: Tuple[Node, ...],
        coeffs: torch.Tensor,
    ) -> Node:
        tag_poly = tag_polyA if side == "A" else tag_polyB
        tag_scale = tag_scaleA if side == "A" else tag_scaleB
        basis_s = str(basis or "poly").lower()
        if basis_s == "power":
            if len(inputs) != 1 or power is None:
                # Defensive fallback: should have been filtered by the fitter.
                return AtomNode(
                    kind="poly",
                    var_idxs=tuple(int(j) for j in var_idxs),
                    kwargs={"degree": int(degree), "min_total": 0},
                    tag=tag_poly,
                    inputs=_fresh_inputs(inputs),
                )
            init = 0.0
            try:
                init = float(coeffs.reshape(-1)[0].item())
            except Exception:
                init = 0.0
            scale = Scale(name=tag_scale, tag=tag_scale, init=init)
            base = clone_ast(inputs[0])
            feature = base if int(power) == 1 else PowNode(base, float(power))
            return MulNode(scale, feature)
        if basis_s == "poly_inv" and len(inputs) == 1:
            inv_input = (PowNode(clone_ast(inputs[0]), -1.0),)
            return AtomNode(
                kind="poly",
                var_idxs=tuple(int(j) for j in var_idxs),
                kwargs={"degree": int(degree), "min_total": 0},
                tag=tag_poly,
                inputs=inv_input,
            )
        return AtomNode(
            kind="poly",
            var_idxs=tuple(int(j) for j in var_idxs),
            kwargs={"degree": int(degree), "min_total": 0},
            tag=tag_poly,
            inputs=_fresh_inputs(inputs),
        )

    # Build AST based on variant
    variant = best.get("variant", "both")
    # Each AtomNode gets its own fresh copy of inputs (no shared DAGs).
    left = AtomNode(
        "nn",
        tuple(int(j) for j in A_glob),
        kwargs=dict(nn_kwargs),
        tag=tag_L,
        inputs=_fresh_inputs(A_inputs),
    )
    right = AtomNode(
        "nn",
        tuple(int(j) for j in B_glob),
        kwargs=dict(nn_kwargs),
        tag=tag_R,
        inputs=_fresh_inputs(B_inputs),
    )
    nn_product = MulNode(left, right)

    if variant == "A_only":
        # Build: PA(xA) + nn(xA)*nn(xB)
        polyA = _counterterm_node(
            side="A",
            basis=basis_A,
            power=None if power_A is None else int(power_A),
            degree=int(best["degree_A"]),
            var_idxs=A_glob,
            inputs=A_inputs,
            coeffs=best["coeffs_A"],
        )
        new_subtree = AddNode(polyA, nn_product)
        A_str = ", ".join(f"x{i}" for i in A_glob)
        B_str = ", ".join(f"x{i}" for i in B_glob)
        print(
            f"[counterterm] Building A_only variant: {basis_A}({A_str}) + (nn0({A_str}) * nn1({B_str}))"
        )

    elif variant == "B_only":
        # Build: PB(xB) + nn(xA)*nn(xB)
        polyB = _counterterm_node(
            side="B",
            basis=basis_B,
            power=None if power_B is None else int(power_B),
            degree=int(best["degree_B"]),
            var_idxs=B_glob,
            inputs=B_inputs,
            coeffs=best["coeffs_B"],
        )
        new_subtree = AddNode(polyB, nn_product)
        A_str = ", ".join(f"x{i}" for i in A_glob)
        B_str = ", ".join(f"x{i}" for i in B_glob)
        print(
            f"[counterterm] Building B_only variant: {basis_B}({B_str}) + (nn0({A_str}) * nn1({B_str}))"
        )

    else:  # variant == "both"
        # Build: PA(xA) + PB(xB) + nn(xA)*nn(xB)
        polyA = _counterterm_node(
            side="A",
            basis=basis_A,
            power=None if power_A is None else int(power_A),
            degree=int(best["degree_A"]),
            var_idxs=A_glob,
            inputs=A_inputs,
            coeffs=best["coeffs_A"],
        )
        polyB = _counterterm_node(
            side="B",
            basis=basis_B,
            power=None if power_B is None else int(power_B),
            degree=int(best["degree_B"]),
            var_idxs=B_glob,
            inputs=B_inputs,
            coeffs=best["coeffs_B"],
        )
        new_subtree = AddNode(AddNode(polyA, polyB), nn_product)
        A_str = ", ".join(f"x{i}" for i in A_glob)
        B_str = ", ".join(f"x{i}" for i in B_glob)
        print(
            f"[counterterm] Building both variant: {basis_A}({A_str}) + {basis_B}({B_str}) + (nn0({A_str}) * nn1({B_str}))"
        )

    cand_root = replace_atom_in_ast(root, target, new_subtree)

    coeffsA0 = best["coeffs_A"].detach().cpu()
    coeffsB0 = best["coeffs_B"].detach().cpu()

    def _copy_coeffs_into_param(param: Optional[torch.nn.Parameter], coeffs: torch.Tensor) -> bool:
        """Copy fitted coefficients into a leaf parameter, preserving target shape."""
        if param is None or param.numel() != coeffs.numel():
            return False
        with torch.no_grad():
            src = coeffs.to(device=param.device, dtype=param.dtype)
            param.copy_(src.reshape_as(param))
        return True

    def _init_fn(root_new: Node, model_new: nn.Module):
        try:
            atom_to_leaf_new = build_atom_to_leaf_map(root_new, model_new)
        except Exception:
            return
        # Find the explicit counterterm atoms by tag
        polyA_new = None
        polyB_new = None
        scaleA_new = None
        scaleB_new = None
        for a in _collect_all_atoms(root_new):
            if isinstance(a, AtomNode) and str(a.kind).lower() in ("poly", "polynomial", "rpoly", "rpolynomial", "r_polynomial"):
                if a.tag == tag_polyA:
                    polyA_new = a
                elif a.tag == tag_polyB:
                    polyB_new = a
            if isinstance(a, AtomNode) and str(a.kind).lower() in (
                "scale",
                "mul_scale",
                "free_const",
                "freeconst",
                "free_constant",
            ):
                if a.tag == tag_scaleA:
                    scaleA_new = a
                elif a.tag == tag_scaleB:
                    scaleB_new = a

        # Initialise polyA
        if polyA_new is not None:
            leafA = atom_to_leaf_new.get(id(polyA_new), None)
            if leafA is not None:
                pA = _leaf_coeff_param(leafA)
                _copy_coeffs_into_param(pA, coeffsA0)

        # Initialise polyB
        if polyB_new is not None:
            leafB = atom_to_leaf_new.get(id(polyB_new), None)
            if leafB is not None:
                pB = _leaf_coeff_param(leafB)
                _copy_coeffs_into_param(pB, coeffsB0)

        # Initialise sparse one-term counterterm scales.
        if scaleA_new is not None and coeffsA0.numel() == 1:
            leafA = atom_to_leaf_new.get(id(scaleA_new), None)
            if leafA is not None:
                pA = _leaf_coeff_param(leafA)
                _copy_coeffs_into_param(pA, coeffsA0)
        if scaleB_new is not None and coeffsB0.numel() == 1:
            leafB = atom_to_leaf_new.get(id(scaleB_new), None)
            if leafB is not None:
                pB = _leaf_coeff_param(leafB)
                _copy_coeffs_into_param(pB, coeffsB0)

        # --------------------------------------------------------------------------
        # Initialise NN leaves so the product term nn(A)*nn(B) starts near zero.
        # We set nn_L ≈ 1.0 and nn_R ≈ 0.0, so nn_L * nn_R ≈ 0 initially.
        # This means initial predictions ≈ poly_new(z), matching the probe result.
        # The optimizer can then learn the multiplicative residual correction.
        # --------------------------------------------------------------------------
        # Precompute X subsets (captured from outer scope)
        XAcpu = X[:, best["A"]].detach().cpu()
        XBcpu = X[:, best["B"]].detach().cpu()

        # Initialize nn_L to ~1.0, nn_R to ~0.0 (like counterfactor_add_split)
        cL0 = 1.0
        cR0 = 0.0

        # Helper to find device/dtype from leaf parameters
        def _leaf_param_device_dtype(mod: nn.Module) -> Tuple[torch.device, torch.dtype]:
            for p in mod.parameters(recurse=True):
                if isinstance(p, torch.Tensor):
                    return p.device, p.dtype
            return device, dtype

        # Helper: fast constant initialiser for NestyNet-style segmented leaves
        def _try_init_segmented_leaf_constant(
            leaf_mod: nn.Module, value: float, x_ex: torch.Tensor
        ) -> bool:
            base = getattr(leaf_mod, "base_model", None)
            if base is None or (not hasattr(base, "get_parameters")):
                _counterterm_debug_log(f"[counterterm DEBUG] _try_init_segmented: no base_model or get_parameters, type={type(leaf_mod).__name__}")
                return False
            try:
                a_pieces, b_pieces, c_pieces, K_pieces = base.get_parameters()
            except Exception:
                return False
            if not a_pieces:
                return False
            try:
                with torch.no_grad():
                    for t in K_pieces:
                        if hasattr(t, "zero_"):
                            t.zero_()
                    for t in b_pieces:
                        if hasattr(t, "zero_"):
                            t.zero_()
                    if c_pieces is not None:
                        for t in c_pieces:
                            if hasattr(t, "zero_"):
                                t.zero_()
                    for t in a_pieces:
                        if hasattr(t, "zero_"):
                            t.zero_()
                    a0 = a_pieces[0].view(-1)
                    a0[0] = torch.as_tensor(1.0, device=a0.device, dtype=a0.dtype)
                with torch.no_grad():
                    y0 = leaf_mod(x_ex[:1])
                y0v = float(y0.reshape(-1)[0].item())
                if (not math.isfinite(y0v)) or (abs(y0v) < 1e-20):
                    return False
                scale_factor = float(value) / y0v
                with torch.no_grad():
                    a0 = a_pieces[0].view(-1)
                    a0[0] = a0[0] * torch.as_tensor(scale_factor, device=a0.device, dtype=a0.dtype)
                return True
            except Exception:
                return False

        # Helper: fallback GD fit to a constant
        def _gd_fit_leaf_constant(
            leaf_mod: nn.Module, value: float, x_ex: torch.Tensor, *, steps: int = 200
        ) -> bool:
            try:
                import torch.optim
                dev, dt = _leaf_param_device_dtype(leaf_mod)
                x_ex = x_ex.to(device=dev, dtype=dt)
                y_tgt = torch.full((x_ex.shape[0],), float(value), device=dev, dtype=dt)
                leaf_mod.train()
                opt = torch.optim.Adam(leaf_mod.parameters(), lr=5e-2)
                for _ in range(int(steps)):
                    opt.zero_grad(set_to_none=True)
                    y = leaf_mod(x_ex)
                    if y.dim() == 2:
                        y = y[:, 0]
                    y = y.reshape(-1)
                    loss = (y - y_tgt).pow(2).mean()
                    if not torch.isfinite(loss):
                        break
                    loss.backward()
                    opt.step()
                    if float(loss.detach().item()) < 1e-10:
                        break
                for p in leaf_mod.parameters():
                    p.grad = None
                leaf_mod.eval()
                return True
            except Exception:
                try:
                    leaf_mod.eval()
                except Exception:
                    pass
                return False

        # Initialise the NN leaves
        def _init_nn(tag: str, value: float, Xcpu: torch.Tensor):
            found_count = 0
            all_atoms = _collect_all_atoms(root_new)
            for a in all_atoms:
                if not (
                    isinstance(a, AtomNode)
                    and str(a.kind).lower() == "nn"
                    and getattr(a, "tag", None) == tag
                ):
                    continue
                found_count += 1
                leaf_mod = atom_to_leaf_new.get(id(a))
                if leaf_mod is None:
                    _counterterm_debug_log(f"[counterterm DEBUG] Found NN atom tag={tag}, but leaf_mod is None (id={id(a)})")
                    continue
                dev, dt = _leaf_param_device_dtype(leaf_mod)
                x_ex = Xcpu[: min(256, Xcpu.shape[0])].to(device=dev, dtype=dt)
                # Save random init before destructive try_init
                orig_state = {k: v.clone() for k, v in leaf_mod.state_dict().items()}
                ok = _try_init_segmented_leaf_constant(leaf_mod, float(value), x_ex)
                if not ok:
                    # Restore random init (not zero) so GD has gradients
                    leaf_mod.load_state_dict(orig_state)
                    _gd_fit_leaf_constant(leaf_mod, float(value), x_ex, steps=500)
                # Debug: verify the output after initialization
                with torch.no_grad():
                    test_out = leaf_mod(x_ex[:10])
                    out_vals = test_out.reshape(-1)[:3].tolist()
                    _counterterm_debug_log(f"[counterterm DEBUG] NN tag={tag} init to {value}, output samples: {out_vals}")
            if found_count == 0:
                nn_tags = [getattr(a, "tag", None) for a in all_atoms if isinstance(a, AtomNode) and str(a.kind).lower() == "nn"]
                _counterterm_debug_log(f"[counterterm DEBUG] No NN atom found for tag={tag}; available NN tags: {nn_tags}")

        _init_nn(tag_L, cL0, XAcpu)
        _init_nn(tag_R, cR0, XBcpu)

    # Run after analytic init (no-op for multivariate polys, but safe)
    _init_fn._after_analytic_init = True

    # Build metadata with signature for deduplication
    A_str = ", ".join(f"x{i}" for i in A_glob)
    B_str = ", ".join(f"x{i}" for i in B_glob)
    basis_A_label = str(basis_A) if power_A is None else f"{basis_A}[p={int(power_A)}]"
    basis_B_label = str(basis_B) if power_B is None else f"{basis_B}[p={int(power_B)}]"
    log_message = (
        f"[Stage B]  Trying counterterm_mul_split: NN({', '.join(f'x{i}' for i in target.var_idxs)}) "
        f"→ {variant} (A={A_str}, B={B_str}; counterterms={basis_A_label}/{basis_B_label}; "
        f"rel_id≈{float(best['rel_err']):.2e})"
    )
    metadata = {
        "signature": signature,
        "log": log_message,
        "structural": True,
        "counterterm_basis_A": str(basis_A),
        "counterterm_basis_B": str(basis_B),
        "counterterm_power_A": None if power_A is None else int(power_A),
        "counterterm_power_B": None if power_B is None else int(power_B),
        "counterterm_rel_id": float(best["rel_err"]),
    }

    return cand_root, _init_fn, metadata
