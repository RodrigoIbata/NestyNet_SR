# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""
Class SR: Multi-dataset symbolic regression with shared (class) and
per-experiment constants.

After Stage B discovers a formula structure and independently fits each
dataset, this module:

1. **Auto-classifies** leaf atoms as "class" (shared physics) or
   "experiment" (per-dataset) by comparing fitted parameter values
   across datasets (coefficient of variation test).

2. **Jointly optimises** the formula with parameter sharing: class-atom
   leaf modules are shared across composites (same ``nn.Parameter``
   objects), experiment-atom leaves remain independent.

The joint optimiser uses L-BFGS (via ``torch.optim``) so that shared
parameters naturally receive accumulated gradients from all datasets.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from nestynet_sr.sr_core.bridges import (
    AtomNode,
    Node,
    build_composite_from_ast,
    clone_ast,
    collect_all_atoms,
    make_reuse_only_nn_factory,
)
from nestynet_sr.sr_core.fit_links import canonical_fit_link_name, fit_link_torch
from nestynet_sr.sr_search.param_sr import (
    ParamInvariant,
    ParamScalarRef,
    discover_param_invariants,
    evaluate_invariant_on_composite,
)
from nestynet_sr.sr_search.stageB.engine import StageBState

logger = logging.getLogger(__name__)


# ── result container ─────────────────────────────────────────────────────────


@dataclass
class ClassSRResult:
    """Result of class SR joint fitting."""

    root: Node
    composites: List[torch.nn.Module]
    val_losses: List[float]
    val_loss_agg: float
    class_tags: List[str]
    experiment_tags: List[str]
    class_params: Dict[str, torch.Tensor]       # tag -> param vector
    experiment_params: List[Dict[str, torch.Tensor]]  # per-dataset tag -> param vector
    cv_per_tag: Dict[str, float] = field(default_factory=dict)
    val_loss_agg_mode: str = "mean"
    derived_invariants: List[Dict[str, object]] = field(default_factory=list)


# ── Step 4: auto-classification ──────────────────────────────────────────────


def _extract_leaf_params(model: torch.nn.Module, leaf_idx: int) -> torch.Tensor:
    """Return the flat parameter vector for leaf *leaf_idx* of an ASTComposite."""
    leaf = model.leaf[leaf_idx]
    parts = [p.detach().flatten() for p in leaf.parameters()]
    if not parts:
        return torch.tensor([], dtype=torch.float64)
    return torch.cat(parts)


def _ordered_unique(tags: List[str]) -> List[str]:
    """Return tags with duplicates removed, preserving first occurrence order."""
    seen = set()
    out = []
    for t in tags:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _loader_n_samples(loader) -> Optional[int]:
    """Best-effort sample count for a DataLoader."""
    ds = getattr(loader, "dataset", None)
    if ds is None:
        return None
    try:
        return int(len(ds))
    except Exception:
        return None


def _aggregate_val_losses(
    val_losses: List[float],
    val_loaders: List,
) -> Tuple[float, str]:
    """Aggregate per-dataset validation losses, weighted by sample count when available."""
    counts = [_loader_n_samples(dl) for dl in val_loaders]
    if len(counts) == len(val_losses) and all((c is not None and c > 0) for c in counts):
        total_n = float(sum(int(c) for c in counts))
        agg = sum((float(c) / total_n) * float(vl) for c, vl in zip(counts, val_losses))
        return float(agg), "weighted_by_val_points"
    agg = sum(float(vl) for vl in val_losses) / max(len(val_losses), 1)
    return float(agg), "mean"


def _tag_to_leafidx(root: Node) -> Dict[str, int]:
    """Map each unique atom tag to its leaf index (first occurrence wins)."""
    atoms = collect_all_atoms(root)
    out: Dict[str, int] = {}
    for idx, atom in enumerate(atoms):
        tag = atom.tag or f"leaf{idx}"
        out.setdefault(tag, idx)
    return out


def _set_scopes_by_tags(root: Node, class_tags: List[str]) -> None:
    """Set atom.scope in-place from class tag set."""
    class_set = set(class_tags)
    atoms = collect_all_atoms(root)
    for idx, atom in enumerate(atoms):
        tag = atom.tag or f"leaf{idx}"
        atom.scope = "class" if tag in class_set else "experiment"


def _extract_params_for_tags(
    root: Node,
    models: List[torch.nn.Module],
    tags: List[str],
) -> List[Dict[str, torch.Tensor]]:
    """Extract per-dataset parameter vectors for the requested tags."""
    tag_map = _tag_to_leafidx(root)
    out: List[Dict[str, torch.Tensor]] = []
    tags_u = _ordered_unique(tags)
    for m in models:
        d: Dict[str, torch.Tensor] = {}
        for tag in tags_u:
            lidx = tag_map.get(tag, None)
            if lidx is None:
                continue
            d[tag] = _extract_leaf_params(m, lidx)
        out.append(d)
    return out


def _fit_link_from_states(states: List[StageBState]) -> Tuple[Optional[str], float]:
    """Pick a canonical fit-link configuration from states[0], warn on mismatches."""
    fit_link = canonical_fit_link_name(getattr(states[0].model, "fit_y_link", None))
    fit_link_scale = float(getattr(states[0].model, "fit_y_link_scale", 1.0))
    for i in range(1, len(states)):
        link_i = canonical_fit_link_name(getattr(states[i].model, "fit_y_link", None))
        scale_i = float(getattr(states[i].model, "fit_y_link_scale", 1.0))
        if link_i != fit_link or abs(scale_i - fit_link_scale) > 1e-12:
            logger.warning(
                "Class SR: dataset %d fit-link mismatch (link=%s, scale=%.6g) "
                "vs dataset 0 (link=%s, scale=%.6g); using dataset-0 fit-link.",
                i, str(link_i), scale_i, str(fit_link), fit_link_scale,
            )
    return fit_link, fit_link_scale


def _evaluate_models_on_loaders(
    models: List[torch.nn.Module],
    loaders: List,
    device: torch.device,
    *,
    fit_link: Optional[str],
    fit_link_scale: float,
    max_points_per_dataset: Optional[int] = None,
    seed_base: int = 3001,
    split_name: str = "val",
) -> Tuple[List[float], float, str]:
    """Evaluate models on loaders with optional deterministic subsampling."""
    if len(models) != len(loaders):
        raise ValueError(f"models/loaders length mismatch: {len(models)} vs {len(loaders)}")

    max_pts = int(max_points_per_dataset) if max_points_per_dataset is not None else None
    if max_pts is not None and max_pts <= 0:
        max_pts = None

    per_losses: List[float] = []
    counts: List[int] = []

    with torch.no_grad():
        for i, (model, loader) in enumerate(zip(models, loaders)):
            model.eval()
            xs, ys = [], []
            for batch in loader:
                x, y = batch[0].to(device), batch[1].to(device)
                xs.append(x)
                ys.append(y)
            x_all = torch.cat(xs, dim=0)
            y_all = torch.cat(ys, dim=0)

            if max_pts is not None:
                n = int(x_all.shape[0])
                if n > max_pts:
                    g = torch.Generator(device="cpu").manual_seed(int(seed_base + i))
                    idx_cpu = torch.randperm(n, generator=g)[:max_pts]
                    idx = idx_cpu.to(device=x_all.device)
                    x_all = x_all.index_select(0, idx)
                    y_all = y_all.index_select(0, idx)
                    logger.info(
                        "Class SR baseline: %s dataset %d subsample %d -> %d points",
                        split_name, i, n, max_pts,
                    )

            y_link = fit_link_torch(y_all, fit_link, fit_link_scale)
            y_pred = model(x_all)
            y_pred_link = fit_link_torch(y_pred, fit_link, fit_link_scale)
            per_losses.append(float(((y_pred_link - y_link) ** 2).mean()))
            counts.append(int(x_all.shape[0]))

    if len(counts) == len(per_losses) and all(c > 0 for c in counts):
        total_n = float(sum(counts))
        agg = sum((float(c) / total_n) * float(vl) for c, vl in zip(counts, per_losses))
        return per_losses, float(agg), "weighted_by_val_points"
    agg = sum(float(vl) for vl in per_losses) / max(len(per_losses), 1)
    return per_losses, float(agg), "mean"


def _is_unacceptable_worsening(
    new_loss: float,
    ref_loss: float,
    *,
    max_worsening_factor: float,
    abs_tol: float,
) -> bool:
    """True when new_loss is significantly worse than ref_loss."""
    if not math.isfinite(float(new_loss)):
        return True
    r = float(ref_loss)
    n = float(new_loss)
    return (n > r * float(max_worsening_factor)) and (n > r + float(abs_tol))


def _forced_free_const_tags(
    root: Node,
) -> Tuple[List[str], List[str]]:
    """Return (forced_class_tags, forced_experiment_tags) from free_const scopes."""
    free_const_kinds = {"free_const", "freeconst", "free_constant"}
    forced_class: List[str] = []
    forced_experiment: List[str] = []
    for idx, atom in enumerate(collect_all_atoms(root)):
        kind = str(getattr(atom, "kind", "")).lower()
        if kind not in free_const_kinds:
            continue
        tag = atom.tag or f"leaf{idx}"
        scope = str(getattr(atom, "scope", "experiment")).lower()
        if scope == "class":
            forced_class.append(tag)
        elif scope == "experiment":
            forced_experiment.append(tag)
    return _ordered_unique(forced_class), _ordered_unique(forced_experiment)


def auto_classify_atoms(
    root: Node,
    states: List[StageBState],
    cv_threshold: float = 0.15,
    *,
    exclude_scale_leaves: bool = True,
    focus_free_const_leaves: bool = True,
) -> Tuple[Node, List[str], List[str], Dict[str, float]]:
    """Classify AtomNode leaves as class or experiment based on parameter similarity.

    For each leaf atom (identified by DFS index), compute the coefficient of
    variation (CV = std / abs(mean)) of its parameter vector across the D fitted
    models.  Low CV → "class" (shared constant); high CV → "experiment".

    Parameters
    ----------
    root : Node
        AST with tagged atoms (after Stage B fitting).
    states : list[StageBState]
        Per-dataset fitting results.  Each must have ``state.model`` with the
        same leaf structure.
    cv_threshold : float
        CV below this marks a leaf as "class".
    focus_free_const_leaves : bool
        If True (default), only tags containing a free-constant leaf kind are
        considered for CV-based class/experiment discovery. All other tags are
        kept as experiment-scope unless explicitly forced by free-const scope.

    Returns
    -------
    root : Node
        Clone of the input AST with ``scope`` set on each AtomNode.
    class_tags : list[str]
        Tags of class-scope atoms.
    experiment_tags : list[str]
        Tags of experiment-scope atoms.
    cv_per_tag : dict[str, float]
        CV value for each tag (for diagnostics).
    """
    root = clone_ast(root)
    atoms = collect_all_atoms(root)

    models = [st.model for st in states]

    class_tags: List[str] = []
    experiment_tags: List[str] = []
    cv_per_tag: Dict[str, float] = {}

    # De-duplicate by tag. ASTs can legally contain repeated atom tags that map
    # to shared leaf modules; we classify each unique tag once.
    tag_to_leafidx: Dict[str, int] = {}
    tag_to_atoms: Dict[str, List[AtomNode]] = {}
    for leaf_idx, atom in enumerate(atoms):
        tag = atom.tag or f"leaf{leaf_idx}"
        tag_to_leafidx.setdefault(tag, leaf_idx)
        tag_to_atoms.setdefault(tag, []).append(atom)

    # Respect explicit scope constraints for free constants:
    # - free_const + scope=class      -> always class
    # - free_const + scope=experiment -> always experiment
    # This avoids CV-based over-sharing of explicitly local constants.
    free_const_kinds = {"free_const", "freeconst", "free_constant"}
    forced_scope_by_tag: Dict[str, str] = {}
    for tag, atoms_for_tag in tag_to_atoms.items():
        force_class = False
        force_experiment = False
        for atom in atoms_for_tag:
            kind = str(getattr(atom, "kind", "")).lower()
            scope = str(getattr(atom, "scope", "experiment")).lower()
            if kind in free_const_kinds:
                if scope == "class":
                    force_class = True
                elif scope == "experiment":
                    force_experiment = True
        if force_class:
            forced_scope_by_tag[tag] = "class"
        elif force_experiment:
            forced_scope_by_tag[tag] = "experiment"

    scale_kinds = {"scale", "mul_scale"}

    for tag, leaf_idx in tag_to_leafidx.items():
        atoms_for_tag = tag_to_atoms.get(tag, [])
        kinds_for_tag = {
            str(getattr(atom, "kind", "")).lower()
            for atom in atoms_for_tag
        }

        # Collect parameter vectors across datasets
        param_vecs = []
        for m in models:
            pv = _extract_leaf_params(m, leaf_idx)
            if pv.numel() == 0:
                continue
            param_vecs.append(pv)

        cv = float("nan")
        if len(param_vecs) >= 2:
            stacked = torch.stack(param_vecs)  # (D, P)
            mean = stacked.mean(dim=0)
            std = stacked.std(dim=0, unbiased=False)

            # Hybrid scale for near-zero means:
            # use max(|mean|, median(|values|)) to avoid CV blow-ups when
            # means cancel around zero across datasets.
            median_abs = stacked.abs().median(dim=0).values
            denom = torch.maximum(mean.abs(), median_abs).clamp(min=1e-12)
            cv_per_param = (std / denom)
            cv = float(cv_per_param.max())
        cv_per_tag[tag] = cv

        forced = forced_scope_by_tag.get(tag, None)
        if forced == "class":
            class_tags.append(tag)
            logger.info("  atom %s  CV=%.4f → class (forced by free_const scope)", tag, cv)
            continue
        if forced == "experiment":
            experiment_tags.append(tag)
            logger.info("  atom %s  CV=%.4f → experiment (forced by free_const scope)", tag, cv)
            continue

        has_free_const_kind = bool(kinds_for_tag & free_const_kinds)
        if focus_free_const_leaves and (not has_free_const_kind):
            experiment_tags.append(tag)
            logger.info("  atom %s  CV=%.4f → experiment (excluded non-free leaf)", tag, cv)
            continue

        if exclude_scale_leaves and kinds_for_tag and kinds_for_tag.issubset(scale_kinds):
            experiment_tags.append(tag)
            logger.info("  atom %s  CV=%.4f → experiment (excluded scale leaf)", tag, cv)
            continue

        if len(param_vecs) < 2:
            # Can't compute CV with fewer than 2 datasets; default to experiment
            experiment_tags.append(tag)
            continue

        if cv < cv_threshold:
            class_tags.append(tag)
            logger.info("  atom %s  CV=%.4f → class", tag, cv)
        else:
            experiment_tags.append(tag)
            logger.info("  atom %s  CV=%.4f → experiment", tag, cv)

    class_set = set(class_tags)
    for leaf_idx, atom in enumerate(atoms):
        tag = atom.tag or f"leaf{leaf_idx}"
        atom.scope = "class" if tag in class_set else "experiment"

    return root, class_tags, experiment_tags, cv_per_tag


# ── Step 5: joint fitting with shared class parameters ───────────────────────


def _build_shared_composites(
    root: Node,
    reuses: List[Dict[str, torch.nn.Module]],
    class_tags: List[str],
    dtype: torch.dtype,
    device: torch.device,
    fresh_nn_factory=None,
) -> List[torch.nn.Module]:
    """Build D composites from *root*, sharing leaf modules for *class_tags*.

    Strategy:
    - Composite 0 is built normally from its reuse map.
    - For composites 1..D-1, class-tag leaves are replaced by references to
      composite 0's leaf modules (same ``nn.Parameter`` objects).
    """
    nn_factory = make_reuse_only_nn_factory(
        device=device, dtype=dtype, fresh_nn_factory=fresh_nn_factory
    )

    atoms = collect_all_atoms(root)
    tag_to_leafidx: Dict[str, int] = {}
    for idx, atom in enumerate(atoms):
        tag = atom.tag or f"leaf{idx}"
        tag_to_leafidx.setdefault(tag, idx)

    class_tags = _ordered_unique(class_tags)
    class_set = set(class_tags)

    def _clone_leaf(m: torch.nn.Module) -> torch.nn.Module:
        return copy.deepcopy(m).to(device=device, dtype=dtype)

    if not reuses:
        raise ValueError("Class SR requires at least one reuse map")

    # Build composite 0 first; class-shared leaves are taken from this composite.
    reuse0 = reuses[0] or {}
    reuse0_build = {k: _clone_leaf(v) for k, v in reuse0.items()}
    comp0 = build_composite_from_ast(
        root,
        dtype=dtype,
        device=device,
        nn_factory=nn_factory,
        reuse=reuse0_build,
    )
    composites: List[torch.nn.Module] = [comp0]

    shared_by_tag: Dict[str, torch.nn.Module] = {}
    for tag in class_tags:
        lidx = tag_to_leafidx.get(tag, None)
        if lidx is None or lidx >= len(comp0.leaf):
            continue
        shared_by_tag[tag] = comp0.leaf[lidx]

    # Initialise shared leaves to the mean of independently-fitted values.
    for tag, shared_leaf in shared_by_tag.items():
        _init_shared_leaf_from_mean(
            shared_leaf=shared_leaf,
            all_reuses=reuses,
            tag=tag,
            device=device,
            dtype=dtype,
        )

    # Build remaining composites, injecting shared modules directly via reuse map.
    for i in range(1, len(reuses)):
        reuse_i = reuses[i] or {}
        reuse_build: Dict[str, torch.nn.Module] = {}

        for tag in class_set:
            if tag in shared_by_tag:
                reuse_build[tag] = shared_by_tag[tag]

        for tag, leaf in reuse_i.items():
            if tag in reuse_build:
                continue
            reuse_build[tag] = _clone_leaf(leaf)

        comp_i = build_composite_from_ast(
            root,
            dtype=dtype,
            device=device,
            nn_factory=nn_factory,
            reuse=reuse_build,
        )
        composites.append(comp_i)

    return composites


def _build_independent_composites(
    root: Node,
    reuses: List[Dict[str, torch.nn.Module]],
    dtype: torch.dtype,
    device: torch.device,
    fresh_nn_factory=None,
) -> List[torch.nn.Module]:
    """Build D composites from *root* with independent per-dataset leaves."""
    nn_factory = make_reuse_only_nn_factory(
        device=device, dtype=dtype, fresh_nn_factory=fresh_nn_factory
    )

    def _clone_leaf(m: torch.nn.Module) -> torch.nn.Module:
        return copy.deepcopy(m).to(device=device, dtype=dtype)

    composites: List[torch.nn.Module] = []
    for reuse in reuses:
        reuse_i = reuse or {}
        reuse_build = {k: _clone_leaf(v) for k, v in reuse_i.items()}
        comp_i = build_composite_from_ast(
            root,
            dtype=dtype,
            device=device,
            nn_factory=nn_factory,
            reuse=reuse_build,
        )
        composites.append(comp_i)
    return composites


def _init_shared_leaf_from_mean(
    shared_leaf: torch.nn.Module,
    all_reuses: List[Dict[str, torch.nn.Module]],
    tag: str,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Set shared_leaf parameters to the mean of the per-dataset values."""
    param_vecs = []
    for reuse in all_reuses:
        if tag in reuse:
            parts = [p.detach().flatten() for p in reuse[tag].parameters()]
            if parts:
                param_vecs.append(torch.cat(parts))

    if len(param_vecs) < 2:
        return

    mean_vec = torch.stack(param_vecs).mean(dim=0)
    offset = 0
    for p in shared_leaf.parameters():
        n = p.numel()
        p.data.copy_(mean_vec[offset:offset + n].view_as(p).to(device=device, dtype=dtype))
        offset += n


def _unique_params(composites: List[torch.nn.Module]) -> List[torch.nn.Parameter]:
    """Collect unique parameters across composites (shared ones appear once)."""
    seen_ids: set = set()
    params: List[torch.nn.Parameter] = []
    for comp in composites:
        for p in comp.parameters():
            pid = id(p)
            if p.requires_grad and pid not in seen_ids:
                seen_ids.add(pid)
                params.append(p)
    return params


def _global_param_starts(params: List[torch.nn.Parameter]) -> Dict[int, int]:
    """Map parameter object id -> flat start index in global LM vector."""
    out: Dict[int, int] = {}
    cursor = 0
    for p in params:
        out[id(p)] = int(cursor)
        cursor += int(p.numel())
    return out


def _leaf_global_indices(
    leaf: torch.nn.Module,
    param_starts: Dict[int, int],
) -> List[int]:
    """Return flattened global parameter indices for one leaf module."""
    idx: List[int] = []
    for p in leaf.parameters():
        if not p.requires_grad:
            continue
        start = param_starts.get(id(p), None)
        if start is None:
            continue
        n = int(p.numel())
        idx.extend(range(start, start + n))
    return idx


def _build_lm_tie_groups_for_tags(
    composites: List[torch.nn.Module],
    root: Node,
    class_tags: List[str],
    params: List[torch.nn.Parameter],
) -> List[List[int]]:
    """Build LM tie groups for class tags (per scalar element across datasets)."""
    tag_to_leafidx = _tag_to_leafidx(root)
    param_starts = _global_param_starts(params)
    groups: List[List[int]] = []

    for tag in _ordered_unique(class_tags):
        lidx = tag_to_leafidx.get(tag, None)
        if lidx is None:
            continue
        leaf_idx_by_ds: List[List[int]] = []
        for comp in composites:
            if lidx >= len(comp.leaf):
                leaf_idx_by_ds = []
                break
            leaf_idx_by_ds.append(_leaf_global_indices(comp.leaf[lidx], param_starts))

        if not leaf_idx_by_ds:
            continue
        n0 = len(leaf_idx_by_ds[0])
        if n0 == 0:
            continue
        if any(len(v) != n0 for v in leaf_idx_by_ds[1:]):
            logger.warning(
                "Class SR (LM ties): tag '%s' has inconsistent parameter lengths across datasets; skipping tie.",
                tag,
            )
            continue

        for j in range(n0):
            g = [v[j] for v in leaf_idx_by_ds]
            if len(set(g)) >= 2:
                groups.append(g)

    return groups


def _fit_class_sr_joint_lm_ties(
    root: Node,
    states: List[StageBState],
    class_tags: List[str],
    experiment_tags: List[str],
    train_loaders: List,
    val_loaders: List,
    device: torch.device,
    dtype: torch.dtype,
    fresh_nn_factory,
    max_epochs: int,
    max_points_per_dataset: Optional[int],
    derived_scalar_refs: Optional[List[ParamScalarRef]],
    derived_invariants: Optional[List[ParamInvariant]],
    param_sr_dataset_metadata: Optional[List[Dict[str, Any]]],
    cv_per_tag: Optional[Dict[str, float]],
    lm_verbose: bool = False,
) -> ClassSRResult:
    """Joint fit using Predictive LM with exact range-space tie constraints."""
    import nestynet

    D = len(states)
    reuses = [st.reuse for st in states]
    fit_link, fit_link_scale = _fit_link_from_states(states)

    composites = _build_independent_composites(
        root=root,
        reuses=reuses,
        dtype=dtype,
        device=device,
        fresh_nn_factory=fresh_nn_factory,
    )
    for comp in composites:
        setattr(comp, "fit_y_link", fit_link)
        setattr(comp, "fit_y_link_scale", fit_link_scale)

    # Optional deterministic subsampling for train/val to keep LM bounded.
    max_pts = int(max_points_per_dataset) if max_points_per_dataset is not None else None
    if max_pts is not None and max_pts <= 0:
        max_pts = None

    def _loader_to_tensor_loader(loader, *, seed: int):
        from torch.utils.data import DataLoader, TensorDataset

        xs, ys = [], []
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            xs.append(x)
            ys.append(y)
        x_all = torch.cat(xs, dim=0)
        y_all = torch.cat(ys, dim=0)
        if max_pts is not None:
            n = int(x_all.shape[0])
            if n > max_pts:
                g = torch.Generator(device="cpu").manual_seed(int(seed))
                idx_cpu = torch.randperm(n, generator=g)[:max_pts]
                idx = idx_cpu.to(device=x_all.device)
                x_all = x_all.index_select(0, idx)
                y_all = y_all.index_select(0, idx)
        ds = TensorDataset(x_all, y_all)
        return DataLoader(ds, batch_size=len(ds), shuffle=False, drop_last=False)

    train_full = [_loader_to_tensor_loader(dl, seed=1009 + i) for i, dl in enumerate(train_loaders)]
    val_full = [_loader_to_tensor_loader(dl, seed=3001 + i) for i, dl in enumerate(val_loaders)]

    params = _unique_params(composites)
    if not params:
        val_losses, val_loss_agg, agg_mode = _evaluate_models_on_loaders(
            models=composites,
            loaders=val_full,
            device=device,
            fit_link=fit_link,
            fit_link_scale=fit_link_scale,
            max_points_per_dataset=None,
            seed_base=3001,
            split_name="val",
        )
        return ClassSRResult(
            root=root,
            composites=composites,
            val_losses=val_losses,
            val_loss_agg=val_loss_agg,
            val_loss_agg_mode=agg_mode,
            class_tags=list(_ordered_unique(class_tags)),
            experiment_tags=list(_ordered_unique(experiment_tags)),
            class_params={},
            experiment_params=_extract_params_for_tags(root, composites, experiment_tags),
            cv_per_tag=cv_per_tag or {},
            derived_invariants=[inv.to_dict() for inv in (derived_invariants or [])],
        )

    def _seg_factory(model, dl):
        def factory(_):
            return nestynet.optimizer.ResidualsModule(
                providers=[model],
                dataloader=dl,
                device=device,
            )
        return factory

    residual_module_factories = [
        _seg_factory(composites[i], train_full[i]) for i in range(D)
    ]
    residual_module_factories_val = [
        _seg_factory(composites[i], val_full[i]) for i in range(D)
    ]

    from nestynet_sr.sr_search.training import (
        SR_LM_OVERRIDES,
        _sr_align_validation_patience,
        _sr_latest_joint_loss_metrics,
        _sr_lm_iter_check,
        _sr_maybe_trigger_prior_decay_from_stall,
        _sr_validation_fresh_after_step,
    )

    cfg = nestynet.optimizer.LMConfig(
        verbose=lm_verbose,
        LM_strategy="direct_solve",
        chisq_tol=1.0e-10,
        log_to_console=False,
        **SR_LM_OVERRIDES,
    )
    lm_opt = nestynet.optimizer.Predictive_LM_Optimizer(
        params,
        residual_module_factories,
        residual_module_factories_val=residual_module_factories_val,
        cfg=cfg,
    )
    # Some fully-analytic provider graphs report total_parameters=0 through
    # provider-offset packing, even when explicit params were passed in.
    # Tie constraints operate in flat-param coordinates, so use the explicit
    # parameter count in that case.
    p_flat = int(sum(int(p.numel()) for p in params))
    if int(getattr(lm_opt, "total_parameters", 0)) <= 0 and p_flat > 0:
        lm_opt.total_parameters = int(p_flat)
        logger.info(
            "Class SR (LM ties): overriding LM total_parameters to %d for constraint indexing.",
            p_flat,
        )

    tie_groups = _build_lm_tie_groups_for_tags(
        composites=composites,
        root=root,
        class_tags=class_tags,
        params=params,
    )
    if tie_groups:
        lm_opt.tie_parameters(
            groups=tie_groups,
            method="range_space",
            tol=1.0e-12,
        )
        logger.info(
            "Class SR (LM ties): applied %d equality tie rows from %d class tags",
            len(tie_groups),
            len(class_tags),
        )
    if derived_invariants:
        logger.info(
            "Class SR (LM ties): derived-invariant soft constraints are currently disabled for LM backend (%d candidates).",
            len(derived_invariants),
        )
    _sr_latest_joint_loss_metrics(
        lm_opt,
        target_count=D,
        label="Class SR (LM ties): ",
    )

    best_val_loss = float("inf")
    best_param_vec = None
    iter_check = _sr_lm_iter_check(lm_opt)
    patience = _sr_align_validation_patience(
        min(80, max(20, max_epochs // 6)),
        iter_check,
        label="Class SR (LM ties): ",
    )
    nval_worse = 0
    last_val_epoch = None
    last_report_train_selection_loss = None

    for epoch in range(max_epochs + 1):
        _loss_obj, loss_val_obj = lm_opt.step()
        loss_metrics = _sr_latest_joint_loss_metrics(
            lm_opt,
            target_count=D,
            label="Class SR (LM ties): ",
        )
        raw_val = loss_metrics.get("val_selection_loss", loss_metrics.get("val_data_mean_loss", loss_val_obj))
        loss_val = None if raw_val is None else float(raw_val)
        last_report_train_selection_loss, _ = _sr_maybe_trigger_prior_decay_from_stall(
            lm_opt,
            loss_metrics=loss_metrics,
            prev_report_train_selection_loss=last_report_train_selection_loss,
            epochs=max_epochs,
            label="Class SR (LM ties): ",
        )
        val_fresh_data = _sr_validation_fresh_after_step(
            lm_opt,
            require_metrics_ready=False,
        )
        val_fresh = _sr_validation_fresh_after_step(lm_opt)
        if val_fresh_data and loss_val is not None and loss_val < best_val_loss:
            best_val_loss = float(loss_val)
            best_param_vec = torch.cat([p.detach().view(-1) for p in params]).clone()
        if val_fresh and loss_val is not None and loss_val <= best_val_loss:
            nval_worse = 0
            last_val_epoch = epoch
        elif val_fresh and loss_val is not None:
            if last_val_epoch is None:
                nval_worse += iter_check
            else:
                nval_worse += max(1, epoch - last_val_epoch)
            last_val_epoch = epoch

        if lm_opt.state.get("halt", False):
            break
        if val_fresh and nval_worse >= patience:
            break

    if best_param_vec is not None:
        lm_opt._update_param_groups(best_param_vec)

    val_losses, val_loss_agg, agg_mode = _evaluate_models_on_loaders(
        models=composites,
        loaders=val_full,
        device=device,
        fit_link=fit_link,
        fit_link_scale=fit_link_scale,
        max_points_per_dataset=None,
        seed_base=3001,
        split_name="val",
    )

    tag_to_leafidx = _tag_to_leafidx(root)
    class_params: Dict[str, torch.Tensor] = {}
    for tag in class_tags:
        lidx = tag_to_leafidx.get(tag, None)
        if lidx is None:
            continue
        class_params[tag] = _extract_leaf_params(composites[0], lidx)

    experiment_params = _extract_params_for_tags(root, composites, experiment_tags)

    return ClassSRResult(
        root=root,
        composites=composites,
        val_losses=val_losses,
        val_loss_agg=val_loss_agg,
        val_loss_agg_mode=agg_mode,
        class_tags=list(_ordered_unique(class_tags)),
        experiment_tags=list(_ordered_unique(experiment_tags)),
        class_params=class_params,
        experiment_params=experiment_params,
        cv_per_tag=cv_per_tag or {},
        derived_invariants=[inv.to_dict() for inv in (derived_invariants or [])],
    )


def fit_class_sr_joint(
    root: Node,
    states: List[StageBState],
    class_tags: List[str],
    experiment_tags: List[str],
    train_loaders: List,
    val_loaders: List,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    fresh_nn_factory=None,
    max_epochs: int = 500,
    lr: float = 1.0,
    max_points_per_dataset: Optional[int] = None,
    optimizer_backend: str = "lbfgs",
    derived_scalar_refs: Optional[List[ParamScalarRef]] = None,
    derived_invariants: Optional[List[ParamInvariant]] = None,
    derived_penalty_weight: float = 1.0e-2,
    param_sr_dataset_metadata: Optional[List[Dict[str, Any]]] = None,
    cv_per_tag: Optional[Dict[str, float]] = None,
    lm_verbose: bool = False,
) -> ClassSRResult:
    """Jointly optimise a formula with shared class parameters.

    Uses L-BFGS starting from independently-fitted parameter values.
    Class-atom leaf modules are shared (same ``nn.Parameter`` objects)
    across all D composites, so their gradients accumulate naturally.

    Parameters
    ----------
    root : Node
        AST with scope-annotated atoms.
    states : list[StageBState]
        Per-dataset independent fitting results (source of reuse maps).
    class_tags, experiment_tags : list[str]
        Tags marking which atoms are shared vs independent.
    train_loaders, val_loaders : list
        Per-dataset DataLoaders.
    device, dtype : torch device/dtype
    fresh_nn_factory : callable or None
        Factory for NN atoms (typically not needed in Stage B).
    max_epochs : int
        Maximum L-BFGS epochs.
    lr : float
        L-BFGS learning rate (1.0 is standard for full-batch).

    Returns
    -------
    ClassSRResult
    """
    backend = str(optimizer_backend or "lbfgs").lower().strip()
    if backend in ("lm", "lm_tie", "lm_ties", "range_space"):
        return _fit_class_sr_joint_lm_ties(
            root=root,
            states=states,
            class_tags=class_tags,
            experiment_tags=experiment_tags,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            device=device,
            dtype=dtype,
            fresh_nn_factory=fresh_nn_factory,
            max_epochs=max_epochs,
            max_points_per_dataset=max_points_per_dataset,
            derived_scalar_refs=derived_scalar_refs,
            derived_invariants=derived_invariants,
            param_sr_dataset_metadata=param_sr_dataset_metadata,
            cv_per_tag=cv_per_tag,
            lm_verbose=lm_verbose,
        )
    if backend not in ("lbfgs",):
        logger.warning(
            "Class SR: unknown optimizer_backend=%r; falling back to lbfgs.",
            optimizer_backend,
        )

    D = len(states)
    class_tags = _ordered_unique(class_tags)
    experiment_tags = [t for t in _ordered_unique(experiment_tags) if t not in set(class_tags)]
    reuses = [st.reuse for st in states]
    max_pts = int(max_points_per_dataset) if max_points_per_dataset is not None else None
    if max_pts is not None and max_pts <= 0:
        max_pts = None
    fit_link, fit_link_scale = _fit_link_from_states(states)
    derived_scalar_refs = list(derived_scalar_refs or [])
    derived_invariants = list(derived_invariants or [])
    param_sr_dataset_metadata = (
        list(param_sr_dataset_metadata)
        if param_sr_dataset_metadata is not None
        else None
    )

    # ── build composites with shared class leaves ────────────────────────
    composites = _build_shared_composites(
        root=root,
        reuses=reuses,
        class_tags=class_tags,
        dtype=dtype,
        device=device,
        fresh_nn_factory=fresh_nn_factory,
    )

    # ── collect unique parameters and set up L-BFGS ─────────────────────
    unique_params = _unique_params(composites)
    n_total = sum(p.numel() for p in unique_params)
    logger.info(
        "Class SR joint fit: %d datasets, %d unique params (%d class tags, %d experiment tags)",
        D, n_total, len(class_tags), len(experiment_tags),
    )

    optimizer = torch.optim.LBFGS(
        unique_params, lr=lr, max_iter=20, line_search_fn="strong_wolfe",
    )

    best_val_loss = float("inf")
    best_param_vec = None

    # ── pre-load all data to device (full-batch for L-BFGS) ─────────────
    def _maybe_subsample(
        x_all: torch.Tensor,
        y_all: torch.Tensor,
        *,
        seed: int,
        ds_idx: int,
        split: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if max_pts is None:
            return x_all, y_all
        n = int(x_all.shape[0])
        if n <= max_pts:
            return x_all, y_all
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        idx_cpu = torch.randperm(n, generator=g)[:max_pts]
        idx = idx_cpu.to(device=x_all.device)
        logger.info("Class SR: %s dataset %d subsample %d -> %d points", split, ds_idx, n, max_pts)
        return x_all.index_select(0, idx), y_all.index_select(0, idx)

    train_data = []
    train_sizes = []
    for i, loader in enumerate(train_loaders):
        xs, ys = [], []
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            xs.append(x)
            ys.append(y)
        x_all = torch.cat(xs, dim=0)
        y_all = torch.cat(ys, dim=0)
        x_all, y_all = _maybe_subsample(
            x_all, y_all, seed=1009 + i, ds_idx=i, split="train"
        )
        y_link_all = fit_link_torch(y_all, fit_link, fit_link_scale)
        train_data.append((x_all, y_all, y_link_all))
        train_sizes.append(int(x_all.shape[0]))

    val_data = []
    val_sizes = []
    for i, loader in enumerate(val_loaders):
        xs, ys = [], []
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            xs.append(x)
            ys.append(y)
        x_all = torch.cat(xs, dim=0)
        y_all = torch.cat(ys, dim=0)
        x_all, y_all = _maybe_subsample(
            x_all, y_all, seed=3001 + i, ds_idx=i, split="val"
        )
        y_link_all = fit_link_torch(y_all, fit_link, fit_link_scale)
        val_data.append((x_all, y_all, y_link_all))
        val_sizes.append(int(x_all.shape[0]))

    train_total = max(sum(train_sizes), 1)
    val_total = max(sum(val_sizes), 1)
    train_weights = [float(n) / float(train_total) for n in train_sizes]
    val_weights = [float(n) / float(val_total) for n in val_sizes]
    tag_to_leafidx = _tag_to_leafidx(root)
    n_derived = len(derived_invariants)
    if n_derived > 0:
        logger.info(
            "Class SR: applying %d derived-invariant soft constraints (weight=%.3g)",
            n_derived,
            float(derived_penalty_weight),
        )

    # ── optimisation loop ────────────────────────────────────────────────
    patience = 50
    patience_counter = 0

    for epoch in range(max_epochs):
        def closure():
            optimizer.zero_grad()
            total = torch.tensor(0.0, dtype=dtype, device=device)
            for i in range(D):
                x, _, y_link = train_data[i]
                y_pred = composites[i](x)
                y_pred_link = fit_link_torch(y_pred, fit_link, fit_link_scale)
                total = total + float(train_weights[i]) * ((y_pred_link - y_link) ** 2).mean()
            if n_derived > 0:
                pen = torch.tensor(0.0, dtype=dtype, device=device)
                n_ok = 0
                for inv in derived_invariants:
                    vals = []
                    for i in range(D):
                        v = evaluate_invariant_on_composite(
                            comp=composites[i],
                            tag_to_leafidx=tag_to_leafidx,
                            refs=derived_scalar_refs,
                            invariant=inv,
                            dataset_metadata=(
                                param_sr_dataset_metadata[i]
                                if (
                                    param_sr_dataset_metadata is not None
                                    and i < len(param_sr_dataset_metadata)
                                )
                                else None
                            ),
                        )
                        if v is None:
                            vals = []
                            break
                        vals.append(v.to(device=device, dtype=dtype))
                    if len(vals) < 2:
                        continue
                    vv = torch.stack(vals)
                    # Keep the scale normalization stable and out of the gradient path.
                    denom = vv.abs().median().detach().clamp_min(1.0e-12)
                    mu = vv.mean()
                    pen = pen + (((vv - mu) / denom) ** 2).mean()
                    n_ok += 1
                if n_ok > 0:
                    total = total + float(derived_penalty_weight) * (pen / float(n_ok))
            total.backward()
            return total

        loss = optimizer.step(closure)

        # Validation
        with torch.no_grad():
            vl = []
            for i in range(D):
                x, _, y_link = val_data[i]
                y_pred = composites[i](x)
                y_pred_link = fit_link_torch(y_pred, fit_link, fit_link_scale)
                vl.append(float(((y_pred_link - y_link) ** 2).mean()))

        val_agg = sum(float(val_weights[i]) * float(vl[i]) for i in range(D))
        if val_agg < best_val_loss:
            best_val_loss = val_agg
            best_param_vec = torch.cat([p.detach().flatten() for p in unique_params]).clone()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info("Class SR: early stop at epoch %d (patience=%d)", epoch, patience)
            break

        if epoch % 50 == 0 or epoch == max_epochs - 1:
            logger.info(
                "Class SR epoch %d: train=%.4e, val=%.4e (best=%.4e)",
                epoch, float(loss), val_agg, best_val_loss,
            )

    # ── restore best parameters ──────────────────────────────────────────
    if best_param_vec is not None:
        offset = 0
        for p in unique_params:
            n = p.numel()
            p.data.copy_(best_param_vec[offset:offset + n].view_as(p))
            offset += n

    # ── extract final parameter values ───────────────────────────────────
    tag_to_leafidx = _tag_to_leafidx(root)

    class_params: Dict[str, torch.Tensor] = {}
    for tag in class_tags:
        lidx = tag_to_leafidx[tag]
        class_params[tag] = _extract_leaf_params(composites[0], lidx)

    experiment_params: List[Dict[str, torch.Tensor]] = []
    for i in range(D):
        ep: Dict[str, torch.Tensor] = {}
        for tag in experiment_tags:
            lidx = tag_to_leafidx[tag]
            ep[tag] = _extract_leaf_params(composites[i], lidx)
        experiment_params.append(ep)

    # ── final validation losses ──────────────────────────────────────────
    with torch.no_grad():
        final_val_losses = []
        for i in range(D):
            x, _, y_link = val_data[i]
            y_pred = composites[i](x)
            y_pred_link = fit_link_torch(y_pred, fit_link, fit_link_scale)
            final_val_losses.append(float(((y_pred_link - y_link) ** 2).mean()))

    return ClassSRResult(
        root=root,
        composites=composites,
        val_losses=final_val_losses,
        val_loss_agg=sum(float(val_weights[i]) * float(final_val_losses[i]) for i in range(D)),
        val_loss_agg_mode="weighted_by_val_points",
        class_tags=class_tags,
        experiment_tags=experiment_tags,
        class_params=class_params,
        experiment_params=experiment_params,
        cv_per_tag=cv_per_tag or {},
        derived_invariants=[inv.to_dict() for inv in derived_invariants],
    )


# ── convenience wrapper ──────────────────────────────────────────────────────


def _build_independent_result(
    root: Node,
    states: List[StageBState],
    val_loaders: List,
    *,
    device: torch.device,
    max_points_per_dataset: Optional[int],
    cv_per_tag: Dict[str, float],
    experiment_tags: List[str],
    derived_invariants: Optional[List[ParamInvariant]] = None,
) -> ClassSRResult:
    """Build a ClassSRResult from independent per-dataset models (no sharing)."""
    models = [st.model for st in states]
    fit_link, fit_link_scale = _fit_link_from_states(states)
    try:
        val_losses, val_loss_agg, agg_mode = _evaluate_models_on_loaders(
            models=models,
            loaders=val_loaders,
            device=device,
            fit_link=fit_link,
            fit_link_scale=fit_link_scale,
            max_points_per_dataset=max_points_per_dataset,
            seed_base=3001,
            split_name="val",
        )
    except Exception as e:
        logger.warning("Class SR: baseline re-evaluation failed (%s); using stored Stage-B losses.", e)
        val_losses = [float(st.val_loss) for st in states]
        val_loss_agg, agg_mode = _aggregate_val_losses(val_losses, val_loaders)

    _set_scopes_by_tags(root, [])
    return ClassSRResult(
        root=root,
        composites=models,
        val_losses=val_losses,
        val_loss_agg=val_loss_agg,
        val_loss_agg_mode=agg_mode,
        class_tags=[],
        experiment_tags=list(_ordered_unique(experiment_tags)),
        class_params={},
        experiment_params=_extract_params_for_tags(root, models, experiment_tags),
        cv_per_tag=cv_per_tag,
        derived_invariants=[inv.to_dict() for inv in (derived_invariants or [])],
    )


def run_class_sr(
    root: Node,
    states: List[StageBState],
    train_loaders: List,
    val_loaders: List,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    cv_threshold: float = 0.15,
    class_atom_tags: Optional[List[str]] = None,
    fresh_nn_factory=None,
    max_epochs: int = 500,
    lr: float = 1.0,
    max_points_per_dataset: Optional[int] = None,
    optimizer_backend: str = "lbfgs",
    param_sr_enable: bool = True,
    param_sr_max_invariants: int = 4,
    param_sr_score_threshold: float = 0.05,
    param_sr_penalty_weight: float = 1.0e-2,
    param_sr_max_scalars: int = 16,
    param_sr_dataset_metadata: Optional[List[Dict[str, Any]]] = None,
    auto_include_scale_leaves: bool = False,
    auto_focus_free_const_leaves: bool = True,
    lm_verbose: bool = False,
) -> ClassSRResult:
    """End-to-end class SR: auto-classify then jointly fit.

    Parameters
    ----------
    root : Node
        AST from Stage B.
    states : list[StageBState]
        Per-dataset Stage B results with fitted models.
    train_loaders, val_loaders : list
        Per-dataset DataLoaders.
    device, dtype : torch device/dtype.
    cv_threshold : float
        CV threshold for auto-classification.
    class_atom_tags : list[str] or None
        If provided, skip auto-classification and use these as class tags.
    fresh_nn_factory : callable or None
        NN factory (for NN atoms, usually None in Stage B).
    max_epochs : int
        Max L-BFGS epochs.
    lr : float
        L-BFGS learning rate.

    Returns
    -------
    ClassSRResult
    """
    # ── classify atoms ───────────────────────────────────────────────────
    auto_classification = class_atom_tags is None
    root_in = clone_ast(root)
    forced_class_tags, forced_experiment_tags = _forced_free_const_tags(root_in)

    if class_atom_tags is not None:
        # Manual specification
        root = clone_ast(root)
        atoms = collect_all_atoms(root)
        all_tags = _ordered_unique([a.tag or f"leaf{i}" for i, a in enumerate(atoms)])
        class_set = set(class_atom_tags)
        class_tags = [t for t in all_tags if t in class_set]
        experiment_tags = [t for t in all_tags if t not in class_set]
        _set_scopes_by_tags(root, class_tags)
        cv_per_tag = {}
        logger.info("Class SR: manual classification — class=%s, experiment=%s",
                    class_tags, experiment_tags)
    else:
        # Auto-classify from parameter variation.
        logger.info("Class SR: auto-classifying atoms (cv_threshold=%.3f)", cv_threshold)
        root, class_tags, experiment_tags, cv_per_tag = auto_classify_atoms(
            root,
            states,
            cv_threshold=cv_threshold,
            exclude_scale_leaves=(not bool(auto_include_scale_leaves)),
            focus_free_const_leaves=bool(auto_focus_free_const_leaves),
        )
        all_tags = _ordered_unique([a.tag or f"leaf{i}" for i, a in enumerate(collect_all_atoms(root))])

        # Enforce explicit free-constant scope hints from the original AST.
        class_tags = _ordered_unique(
            [t for t in class_tags if t not in set(forced_experiment_tags)]
            + [t for t in forced_class_tags if t in all_tags]
        )
        experiment_tags = [t for t in all_tags if t not in set(class_tags)]
        _set_scopes_by_tags(root, class_tags)

        # Robust fallback: strict filtering (free-const-only and/or no scales)
        # can miss shared factorized symbolic search parameters that live in non-free/scale leaves.
        # If strict mode found no class tags, retry once with relaxed inclusion.
        should_retry_relaxed = (
            (not class_tags)
            and (
                (not bool(auto_include_scale_leaves))
                or bool(auto_focus_free_const_leaves)
            )
        )
        if should_retry_relaxed:
            logger.info(
                "Class SR: strict auto-classification found no class tags; "
                "retrying with relaxed leaf inclusion (include scales + non-free leaves)."
            )
            relaxed_root, relaxed_class_tags, _, relaxed_cv_per_tag = auto_classify_atoms(
                root_in,
                states,
                cv_threshold=cv_threshold,
                exclude_scale_leaves=False,
                focus_free_const_leaves=False,
            )
            relaxed_all_tags = _ordered_unique(
                [a.tag or f"leaf{i}" for i, a in enumerate(collect_all_atoms(relaxed_root))]
            )
            relaxed_class_tags = _ordered_unique(
                [t for t in relaxed_class_tags if t not in set(forced_experiment_tags)]
                + [t for t in forced_class_tags if t in relaxed_all_tags]
            )
            if relaxed_class_tags:
                root = relaxed_root
                class_tags = relaxed_class_tags
                cv_per_tag = relaxed_cv_per_tag
                all_tags = relaxed_all_tags
                experiment_tags = [t for t in all_tags if t not in set(class_tags)]
                _set_scopes_by_tags(root, class_tags)
                logger.info(
                    "Class SR: relaxed auto-classification recovered class tags: %s",
                    class_tags,
                )
            else:
                logger.info(
                    "Class SR: relaxed auto-classification also found no class tags; "
                    "keeping strict classification."
                )

    all_tags = _ordered_unique([a.tag or f"leaf{i}" for i, a in enumerate(collect_all_atoms(root))])
    param_sr_dataset_metadata_norm: Optional[List[Dict[str, Any]]] = None
    if param_sr_dataset_metadata is not None:
        if len(param_sr_dataset_metadata) != len(states):
            logger.warning(
                "Class SR Param-SR metadata ignored: got %d dataset metadata rows for %d datasets.",
                len(param_sr_dataset_metadata),
                len(states),
            )
        else:
            param_sr_dataset_metadata_norm = []
            for row in param_sr_dataset_metadata:
                param_sr_dataset_metadata_norm.append(
                    dict(row) if isinstance(row, dict) else {}
                )
            logger.info(
                "Class SR Param-SR: dataset metadata enabled (%d rows).",
                len(param_sr_dataset_metadata_norm),
            )

    # Baseline: evaluate independent Stage-B models on the exact same val loaders
    # and fit-link space used by Class-SR.
    fit_link, fit_link_scale = _fit_link_from_states(states)
    try:
        baseline_val_losses, baseline_agg, baseline_mode = _evaluate_models_on_loaders(
            models=[st.model for st in states],
            loaders=val_loaders,
            device=device,
            fit_link=fit_link,
            fit_link_scale=fit_link_scale,
            max_points_per_dataset=max_points_per_dataset,
            seed_base=3001,
            split_name="val",
        )
    except Exception as e:
        logger.warning("Class SR: baseline re-evaluation failed (%s); using stored Stage-B losses.", e)
        baseline_val_losses = [float(st.val_loss) for st in states]
        baseline_agg, baseline_mode = _aggregate_val_losses(baseline_val_losses, val_loaders)

    derived_scalar_refs: List[ParamScalarRef] = []
    derived_invariants: List[ParamInvariant] = []
    if auto_classification and bool(param_sr_enable):
        try:
            derived_scalar_refs, derived_invariants = discover_param_invariants(
                root=root,
                models=[st.model for st in states],
                dataset_metadata=param_sr_dataset_metadata_norm,
                candidate_tags=experiment_tags if experiment_tags else all_tags,
                include_fixed_value_attr=True,
                max_scalars=max(2, int(param_sr_max_scalars)),
                max_invariants=max(0, int(param_sr_max_invariants)),
                score_threshold=float(param_sr_score_threshold),
            )
            if derived_invariants:
                logger.info(
                    "Class SR Param-SR: discovered %d derived invariants from %d scalar refs",
                    len(derived_invariants), len(derived_scalar_refs),
                )
                for inv in derived_invariants:
                    logger.info(
                        "  Param-SR invariant: %s (score=%.4g, cv=%.4g)",
                        inv.expr, float(inv.score), float(inv.cv),
                    )
        except Exception as e:
            logger.warning("Class SR Param-SR discovery failed: %s", e)
            derived_scalar_refs, derived_invariants = [], []

    if not class_tags and not derived_invariants:
        logger.warning(
            "Class SR: no direct class atoms and no derived invariants found — returning independent fit."
        )
        return _build_independent_result(
            root=root,
            states=states,
            val_loaders=val_loaders,
            device=device,
            max_points_per_dataset=max_points_per_dataset,
            cv_per_tag=cv_per_tag,
            experiment_tags=experiment_tags if experiment_tags else all_tags,
            derived_invariants=derived_invariants,
        )

    # ------------------------------------------------------------------
    # Auto mode robustness: greedy tag inclusion
    #
    # Rather than sharing all low-CV tags at once, add candidates one by one
    # (sorted by increasing CV) and keep only those that do not degrade
    # validation too much in a short screening fit.
    # ------------------------------------------------------------------
    if auto_classification:
        abs_tol = 1.0e-8
        step_max_worsening = 2.0
        screen_epochs = min(int(max_epochs), 120)

        forced_class_set = set(t for t in forced_class_tags if t in all_tags)
        selected_class_tags = [t for t in all_tags if t in forced_class_set]
        candidate_tags = [t for t in class_tags if t not in forced_class_set]
        candidate_tags.sort(
            key=lambda t: (
                1 if not math.isfinite(float(cv_per_tag.get(t, float("nan")))) else 0,
                float(cv_per_tag.get(t, float("inf"))),
                t,
            )
        )

        current_ref_agg = float(baseline_agg)
        if selected_class_tags:
            _set_scopes_by_tags(root, selected_class_tags)
            forced_result = fit_class_sr_joint(
                root=root,
                states=states,
                class_tags=selected_class_tags,
                experiment_tags=[t for t in all_tags if t not in set(selected_class_tags)],
                train_loaders=train_loaders,
                val_loaders=val_loaders,
                device=device,
                dtype=dtype,
                fresh_nn_factory=fresh_nn_factory,
                max_epochs=screen_epochs,
                lr=lr,
                max_points_per_dataset=max_points_per_dataset,
                optimizer_backend=optimizer_backend,
                derived_scalar_refs=derived_scalar_refs,
                derived_invariants=derived_invariants,
                derived_penalty_weight=param_sr_penalty_weight,
                param_sr_dataset_metadata=param_sr_dataset_metadata_norm,
                cv_per_tag=cv_per_tag,
                lm_verbose=lm_verbose,
            )
            current_ref_agg = float(forced_result.val_loss_agg)

        for tag in candidate_tags:
            trial_tags = selected_class_tags + [tag]
            _set_scopes_by_tags(root, trial_tags)
            trial_result = fit_class_sr_joint(
                root=root,
                states=states,
                class_tags=trial_tags,
                experiment_tags=[t for t in all_tags if t not in set(trial_tags)],
                train_loaders=train_loaders,
                val_loaders=val_loaders,
                device=device,
                dtype=dtype,
                fresh_nn_factory=fresh_nn_factory,
                max_epochs=screen_epochs,
                lr=lr,
                max_points_per_dataset=max_points_per_dataset,
                optimizer_backend=optimizer_backend,
                derived_scalar_refs=derived_scalar_refs,
                derived_invariants=derived_invariants,
                derived_penalty_weight=param_sr_penalty_weight,
                param_sr_dataset_metadata=param_sr_dataset_metadata_norm,
                cv_per_tag=cv_per_tag,
                lm_verbose=lm_verbose,
            )

            if _is_unacceptable_worsening(
                float(trial_result.val_loss_agg),
                float(current_ref_agg),
                max_worsening_factor=step_max_worsening,
                abs_tol=abs_tol,
            ):
                logger.info(
                    "Class SR greedy: reject tag '%s' (val %.4e -> %.4e)",
                    tag, float(current_ref_agg), float(trial_result.val_loss_agg),
                )
                continue

            prev_ref = float(current_ref_agg)
            selected_class_tags = trial_tags
            current_ref_agg = float(trial_result.val_loss_agg)
            logger.info(
                "Class SR greedy: accept tag '%s' (val %.4e -> %.4e)",
                tag, prev_ref, float(trial_result.val_loss_agg),
            )

        class_tags = _ordered_unique(selected_class_tags)
        experiment_tags = [t for t in all_tags if t not in set(class_tags)]
        _set_scopes_by_tags(root, class_tags)

        if not class_tags:
            if not derived_invariants:
                logger.warning("Class SR greedy selected no class tags; returning independent models.")
                return _build_independent_result(
                    root=root,
                    states=states,
                    val_loaders=val_loaders,
                    device=device,
                    max_points_per_dataset=max_points_per_dataset,
                    cv_per_tag=cv_per_tag,
                    experiment_tags=all_tags,
                    derived_invariants=derived_invariants,
                )
            logger.info(
                "Class SR greedy selected no direct class tags; continuing with %d derived invariants.",
                len(derived_invariants),
            )

    # ── joint fit ────────────────────────────────────────────────────────
    result = fit_class_sr_joint(
        root=root,
        states=states,
        class_tags=class_tags,
        experiment_tags=experiment_tags,
        train_loaders=train_loaders,
        val_loaders=val_loaders,
        device=device,
        dtype=dtype,
        fresh_nn_factory=fresh_nn_factory,
        max_epochs=max_epochs,
        lr=lr,
        max_points_per_dataset=max_points_per_dataset,
        optimizer_backend=optimizer_backend,
        derived_scalar_refs=derived_scalar_refs,
        derived_invariants=derived_invariants,
        derived_penalty_weight=param_sr_penalty_weight,
        param_sr_dataset_metadata=param_sr_dataset_metadata_norm,
        cv_per_tag=cv_per_tag,
        lm_verbose=lm_verbose,
    )

    # Safety guard for auto-classification: if sharing degrades fit badly,
    # apply a robust fallback (forced-only sharing when required, otherwise independent).
    if auto_classification:
        max_worsening_factor = 10.0
        abs_tol = 1.0e-8
        if _is_unacceptable_worsening(
            float(result.val_loss_agg),
            float(baseline_agg),
            max_worsening_factor=max_worsening_factor,
            abs_tol=abs_tol,
        ):
            logger.warning(
                "Class SR: auto-sharing worsened validation loss too much "
                "(baseline %.4e -> shared %.4e, factor %.3g > %.3g and abs_tol=%.1e); "
                "applying fallback.",
                float(baseline_agg),
                float(result.val_loss_agg),
                (float(result.val_loss_agg) / max(float(baseline_agg), 1e-300)),
                max_worsening_factor,
                abs_tol,
            )
            forced_class_set = set(t for t in forced_class_tags if t in all_tags)
            if forced_class_set:
                forced_only_tags = [t for t in all_tags if t in forced_class_set]
                logger.warning(
                    "Class SR: forced class constants are declared (%s); "
                    "retrying with forced class tags only instead of reverting to fully independent parameters.",
                    forced_only_tags,
                )
                _set_scopes_by_tags(root, forced_only_tags)
                try:
                    forced_only_result = fit_class_sr_joint(
                        root=root,
                        states=states,
                        class_tags=forced_only_tags,
                        experiment_tags=[t for t in all_tags if t not in set(forced_only_tags)],
                        train_loaders=train_loaders,
                        val_loaders=val_loaders,
                        device=device,
                        dtype=dtype,
                        fresh_nn_factory=fresh_nn_factory,
                        max_epochs=max_epochs,
                        lr=lr,
                        max_points_per_dataset=max_points_per_dataset,
                        optimizer_backend=optimizer_backend,
                        derived_scalar_refs=derived_scalar_refs,
                        derived_invariants=derived_invariants,
                        derived_penalty_weight=param_sr_penalty_weight,
                        param_sr_dataset_metadata=param_sr_dataset_metadata_norm,
                        cv_per_tag=cv_per_tag,
                        lm_verbose=lm_verbose,
                    )
                except Exception as e:
                    logger.warning(
                        "Class SR: forced-only retry failed (%s); keeping shared result "
                        "to preserve explicit free-constant scope constraints.",
                        e,
                    )
                    return result

                if _is_unacceptable_worsening(
                    float(forced_only_result.val_loss_agg),
                    float(baseline_agg),
                    max_worsening_factor=max_worsening_factor,
                    abs_tol=abs_tol,
                ):
                    logger.warning(
                        "Class SR: forced-only shared fit still exceeds auto safety threshold "
                        "(baseline %.4e -> forced %.4e), but preserving forced class constraints.",
                        float(baseline_agg),
                        float(forced_only_result.val_loss_agg),
                    )
                return forced_only_result

            return _build_independent_result(
                root=root,
                states=states,
                val_loaders=val_loaders,
                device=device,
                max_points_per_dataset=max_points_per_dataset,
                cv_per_tag=cv_per_tag,
                experiment_tags=all_tags,
                derived_invariants=derived_invariants,
            )

    return result
