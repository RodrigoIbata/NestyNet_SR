# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Candidate fitting utilities for Stage B.

This module provides functions for fitting candidate AST rewrites:
- Initializing new leaves safely
- Cloning reuse maps
- Fitting candidates with LM optimizer
- Summarizing global power-law structures
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from nestynet_sr.sr_core.bridges import (
    Node,
    build_composite_from_ast,
    collect_all_atoms,
    make_reuse_only_nn_factory,
)
from nestynet_sr.sr_core import ast_to_human_readable
from nestynet_sr.sr_search.config import LMHyperparams
from nestynet_sr.sr_search.features import TrigAxisSpec, _sample_values_from_model
from nestynet_sr.sr_search.gauge_fix import gauge_fix_multiplicative

# Import from parent sr_search module
from nestynet_sr.sr_search.training import train_initial_model
from nestynet_sr.sr_core.fit_links import canonical_fit_link_name, fit_link_torch

from .atom_mapping import _collect_all_atoms, _refresh_reuse_from_state
from .engine import StageBState
from .leaf_utils import _debug_report_leaf_cores


def _jitter_new_leaves(
    root: Node,
    atom_to_leaf: Dict[int, nn.Module],
    reuse: Dict[str, torch.nn.Module],
    *,
    scale: float,
) -> int:
    """Multiplicatively jitter the parameters of newly introduced leaves.

    Used on stall-retry restarts: deterministic custom inits would re-enter
    the same optimization basin on every start, so restarts perturb the new
    leaves' parameters by ``(1 + scale*randn)`` under the candidate-local RNG.
    Reused leaves (tag present in the reuse map) are left untouched, mirroring
    the selection logic of :func:`_safe_reinit_new_leaves`.  Returns the
    number of jittered leaves.
    """

    atoms = _collect_all_atoms(root)
    reused_atom_ids = {
        id(atom)
        for atom in atoms
        if atom.tag is not None and atom.tag in reuse
    }
    n_jittered = 0
    with torch.no_grad():
        for atom in atoms:
            leaf = atom_to_leaf.get(id(atom))
            if leaf is None or id(atom) in reused_atom_ids:
                continue
            for param in leaf.parameters():
                param.mul_(1.0 + float(scale) * torch.randn_like(param))
            n_jittered += 1
    return n_jittered


def _safe_reinit_new_leaves(
    root: Node,
    atom_to_leaf: Dict[int, nn.Module],
    reuse: Dict[str, torch.nn.Module],
):
    """
    Fallback initialisation for newly introduced analytic leaves when a
    custom initialiser produces NaNs/Infs or a crazy scale.

    Uses robust tag-based approach to identify reused vs new leaves:
      - Leaves with tags in reuse map are considered reused → leave untouched
      - Leaves without matching tags are new → reinitialize safely

    This replaces the fragile object-identity approach with deterministic
    tag-based identification (Phase 2 Task 3).

    Args:
        root: AST root node
        atom_to_leaf: Mapping from id(atom) to leaf module (from build_composite_from_ast)
        reuse: Mapping from tag to existing leaf module

    Example:
        >>> # Build model with reuse
        >>> root = AddNode(
        ...     AtomNode("poly", (0,), {"degree": 2}, tag="p0"),  # Has reuse
        ...     AtomNode("poly", (1,), {"degree": 2}, tag="p_new"),  # No reuse
        ... )
        >>> model, atom_map = build_composite_from_ast(root, reuse={"p0": old_leaf}, return_atom_map=True)
        >>> # If custom init produces NaNs:
        >>> _safe_reinit_new_leaves(root, atom_map, reuse)
        >>> # Only "p_new" leaf will be reinitialized; "p0" left untouched
    """
    # Import Node types

    # Collect all atoms from AST
    atoms = _collect_all_atoms(root)

    # Build set of atom IDs that have matching reuse tags
    reused_atom_ids = set()
    for atom in atoms:
        if atom.tag is not None and atom.tag in reuse:
            reused_atom_ids.add(id(atom))

    n_reused = 0
    n_reinit = 0

    with torch.no_grad():
        for atom in atoms:
            # Get leaf for this atom
            leaf = atom_to_leaf.get(id(atom))
            if leaf is None:
                continue

            # Check if this leaf was reused (tag-based)
            if id(atom) in reused_atom_ids:
                n_reused += 1
                continue  # Leave reused leaves untouched

            # This is a new leaf, reinitialize it
            n_reinit += 1

            if hasattr(leaf, "reset_parameters_safe"):
                leaf.reset_parameters_safe()
            elif hasattr(leaf, "reset_parameters"):
                leaf.reset_parameters()
            else:
                # Fallback: set to ~N(1, 0.01^2)
                for p in leaf.parameters():
                    p.copy_(torch.ones_like(p) + 0.01 * torch.randn_like(p))

    print(
        f"[Stage B]   Safe reinitialisation: {n_reinit} new leaves reinitialized, "
        f"{n_reused} reused leaves left untouched (≈1+0.01*randn)."
    )


def _clone_reuse(reuse, device, dtype):
    if not reuse:
        return {}
    out = {}
    for k, m in reuse.items():
        mm = copy.deepcopy(m)
        mm.to(device=device, dtype=dtype)
        out[k] = mm
    return out


def _pick_atom_factory(atom_factory, i: int):
    """Return the atom factory for dataset index ``i``.

    Accepts either:
    - a single callable (reused for all datasets), or
    - an indexable sequence of callables (one per dataset).
    """
    if atom_factory is None or callable(atom_factory):
        return atom_factory
    try:
        return atom_factory[i]
    except Exception:
        return None




def _eval_val_loss(
    model: torch.nn.Module,
    val_loader,
    *,
    device: torch.device,
    dtype: torch.dtype,
    fit_y_link: str | None = None,
    fit_y_link_scale: float = 1.0,
) -> float:
    """Compute mean MSE in either y-space or the configured fit-link space."""
    link = canonical_fit_link_name(fit_y_link)
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                x, y = batch, None
            if y is None:
                continue
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)
            try:
                y_pred = model(x)
            except Exception:
                return float("inf")
            if link is None:
                r = y_pred - y
            else:
                r = fit_link_torch(y_pred, link, fit_y_link_scale) - fit_link_torch(
                    y, link, fit_y_link_scale
                )
            losses.append((r * r).mean().item())
    return float(sum(losses) / len(losses)) if losses else float("inf")


def _fit_candidate_root_once(
    root: Node,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    val_loader,
    lm_hp: LMHyperparams,
    device: torch.device,
    dtype: torch.dtype,
    epochs_stageB: int,
    loss_scale: float,
    trig_by_axis: Optional[Dict[int, TrigAxisSpec]] = None,
    custom_init_fn=None,
    fresh_nn_factory=None,
    atom_factory=None,
) -> StageBState:
    root = gauge_fix_multiplicative(root)
    reuse_build = _clone_reuse(reuse, device, dtype)
    nn_factory = make_reuse_only_nn_factory(
        device=device, dtype=dtype, fresh_nn_factory=fresh_nn_factory
    )
    model, atom_to_leaf = build_composite_from_ast(
        root,
        dtype=dtype,
        device=device,
        nn_factory=nn_factory,
        atom_factory=atom_factory,
        reuse=reuse_build,
        return_atom_map=True,
    )

    # Optional fit-only output link (numeric conditioning)
    setattr(model, "fit_y_link", getattr(lm_hp, "fit_y_link", None))
    setattr(model, "fit_y_link_scale", getattr(lm_hp, "fit_y_link_scale", 1.0))

    def _apply_custom_init_and_validate() -> bool:
        try:
            custom_init_fn(root, model)
        except Exception as e:
            print("[Stage B] Warning: custom leaf initialisation failed with:", e)
        start_idx = int(getattr(custom_init_fn, "_candidate_start_idx", 0) or 0)
        if start_idx > 0:
            n_jittered = _jitter_new_leaves(
                root, atom_to_leaf, reuse, scale=0.03 * start_idx
            )
            print(
                f"[Stage B] Stall-retry start {start_idx}: jittered "
                f"{n_jittered} new leaf/leaves (scale {0.03 * start_idx:.3f})."
            )
        # Optional deep sanity printout (enable by setting custom_init_fn._debug_leaf_core = True)
        if getattr(custom_init_fn, "_debug_leaf_core", False):
            _debug_report_leaf_cores(
                root=root,
                model=model,
                only_tagged=True,
                raise_on_missing=False,
                header="leaf/core report after custom_init",
            )
        for attempt in range(2):
            try:
                model.eval()
                with torch.no_grad():
                    n_batches_checked = 0
                    y_preds, y_trues = [], []
                    bad_init = False
                    for batch in train_loader:
                        if isinstance(batch, (list, tuple)):
                            x, y = batch
                        else:
                            x, y = batch, None
                        x = x.to(device)
                        if y is not None:
                            y_trues.append(y)
                        try:
                            y_pred = model(x)
                        except Exception as e_eval:
                            print(
                                f"[Stage B] Warning: initialisation produces model that fails evaluation: {e_eval}"
                            )
                            bad_init = True
                            break
                        y_preds.append(y_pred.detach().cpu())
                        n_finite = torch.isfinite(y_pred).sum().item()
                        n_total = y_pred.numel()
                        y_min, y_max, y_mean = (
                            y_pred.min().item(),
                            y_pred.max().item(),
                            y_pred.mean().item(),
                        )
                        print(
                            f"[Stage B] Post-custom-init validation (attempt {attempt}), "
                            f"batch {n_batches_checked}, finite: {n_finite}/{n_total}, "
                            f"min: {y_min:.3e}, max: {y_max:.3e}, mean: {y_mean:.3e}"
                        )
                        if not torch.isfinite(y_pred).all():
                            print("[Stage B] Warning: initialisation produces non-finite outputs.")
                            bad_init = True
                            break
                        n_batches_checked += 1
                        if n_batches_checked >= 3:
                            break

                if bad_init:
                    if attempt == 0:
                        print(
                            "[Stage B]   Attempting safe reinitialisation of new leaves (1+0.01*randn fallback)."
                        )
                        _safe_reinit_new_leaves(root, atom_to_leaf, reuse)
                        continue
                    else:
                        print("[Stage B]   Safe reinitialisation failed; rejecting this candidate.")
                        return False

                if y_trues:
                    y_pred_all = torch.cat(y_preds, dim=0)
                    y_true_all = torch.cat(y_trues, dim=0).to(device)
                    pred_scale = float(y_pred_all.abs().median())
                    true_scale = float(y_true_all.abs().median())
                    if true_scale > 0:
                        scale_ratio = pred_scale / true_scale
                        print(
                            f"[Stage B] Prediction scale check (attempt {attempt}): "
                            f"pred_median={pred_scale:.3e}, true_median={true_scale:.3e}, "
                            f"ratio={scale_ratio:.3e}"
                        )
                        if (
                            (not math.isfinite(scale_ratio))
                            or (scale_ratio > 1e3)
                            or (scale_ratio < 1e-3)
                        ):
                            print(
                                "[Stage B] Warning: init scale_ratio extreme; rejecting candidate."
                            )
                            return False
                return True
            except Exception as e:
                print("[Stage B] Warning: validation after initialisation failed with:", e)
                if attempt == 0:
                    print(
                        "[Stage B]   Attempting safe reinitialisation of new leaves (1+0.01*randn fallback)."
                    )
                    _safe_reinit_new_leaves(root, atom_to_leaf, reuse)
                    continue
                else:
                    return False
        return False

    init_after = (
        bool(getattr(custom_init_fn, "_after_analytic_init", False))
        if custom_init_fn is not None
        else False
    )
    if (custom_init_fn is not None) and (not init_after):
        ok = _apply_custom_init_and_validate()
        if not ok:
            from .engine import _compute_nn_metrics

            num_nn, num_mv, max_ar = _compute_nn_metrics(root)
            return StageBState(
                root=root,
                model=model,
                reuse=reuse,
                val_loss=float("inf"),
                num_nn_atoms=num_nn,
                num_multivar_nn_atoms=num_mv,
                max_nn_arity=max_ar,
            )

    skip_analytic_init = (
        bool(getattr(custom_init_fn, "_skip_analytic_init", False))
        if custom_init_fn is not None
        else False
    )
    if not skip_analytic_init:
        try:
            # NOTE:
            # stageB.py also contains a small helper named
            # `_initialise_analytic_leaves_from_reuse` which only copies
            # compatible analytic leaf weights between two already-built
            # models. For Stage-B candidate fitting we want the *full*
            # initialiser that can also fit univariate analytic leaves
            # from teacher data (Stage-A NN or prior fitted analytic leaf).
            from nestynet_sr.sr_search.fitting_utils import (
                _initialise_analytic_leaves_from_reuse as _init_analytic_from_reuse,
            )

            _init_analytic_from_reuse(
                root=root,
                model=model,
                reuse=reuse,
                train_loader=train_loader,
                device=device,
                dtype=dtype,
                trig_by_axis=trig_by_axis,
            )
        except Exception as e:
            print("[Stage B] Warning: analytic-leaf initialisation failed with:", e)

    if (custom_init_fn is not None) and init_after:
        ok = _apply_custom_init_and_validate()
        if not ok:
            from .engine import _compute_nn_metrics

            num_nn, num_mv, max_ar = _compute_nn_metrics(root)
            return StageBState(
                root=root,
                model=model,
                reuse=reuse,
                val_loss=float("inf"),
                num_nn_atoms=num_nn,
                num_multivar_nn_atoms=num_mv,
                max_nn_arity=max_ar,
            )

    epochs = max(1, min(epochs_stageB, lm_hp.epochs))

    # --- Fast path for zero-parameter (fully analytic) candidates ---
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_params == 0:
        val_loss = _eval_val_loss(
            model,
            val_loader,
            device=device,
            dtype=dtype,
            fit_y_link=getattr(lm_hp, "fit_y_link", None),
            fit_y_link_scale=float(getattr(lm_hp, "fit_y_link_scale", 1.0)),
        )
        reuse_out = _refresh_reuse_from_state(root, model)
        from .engine import _compute_nn_metrics
        num_nn_atoms, num_multivar_nn_atoms, max_nn_arity = _compute_nn_metrics(root)
        return StageBState(
            root=root, model=model, reuse=reuse_out, val_loss=val_loss,
            num_nn_atoms=num_nn_atoms, num_multivar_nn_atoms=num_multivar_nn_atoms,
            max_nn_arity=max_nn_arity,
        )

    # Stage B also respects loss_in_MAD_units: scale the base target by
    # MAD(y)^2 if requested, then apply a small tightening factor.
    if getattr(lm_hp, "loss_in_MAD_units", False):
        loss_target_stageB = lm_hp.loss_target * loss_scale * 0.01
    else:
        loss_target_stageB = lm_hp.loss_target * 0.01

    # Optional lifted-space prefit for candidates that wrap a parametric core inside a
    # fragile outer transform (e.g. sqrt, log, reciprocal). The intent is to avoid
    # singular Jacobians during the main LM loop by first fitting in a stable lifted
    # space, then (optionally) refining in the original space.
    #
    # Protocol: candidate builder functions in candidate_builders.py may attach a
    # ``_fit_lift_link`` attribute (and optionally ``_fit_lift_scale``) to their
    # ``custom_init_fn`` callable. When present, this triggers a two-stage training:
    #   1. Prefit with model.fit_y_link set to _fit_lift_link (e.g. "square" for sqrt
    #      candidates, "exp" for log candidates, "recip" for reciprocal candidates).
    #   2. Short refine in the original baseline space.
    # See canonical_fit_link_name() in fit_links.py for supported link names.
    lift_link = None
    lift_scale = 1.0
    if getattr(lm_hp, "outer_transform_lift_enable", True) and (custom_init_fn is not None):
        lift_link = getattr(custom_init_fn, "_fit_lift_link", None)
        lift_scale = float(getattr(custom_init_fn, "_fit_lift_scale", 1.0))

    base_link = canonical_fit_link_name(getattr(model, "fit_y_link", None))
    base_scale = float(getattr(model, "fit_y_link_scale", 1.0))

    val_loss = float("inf")

    if lift_link is not None:
        lift_link = canonical_fit_link_name(lift_link)

        # --- Stage 1: lifted-space prefit ---
        pre_best_val_p = None
        try:
            model.fit_y_link = lift_link
            model.fit_y_link_scale = lift_scale

            pre_epochs = min(int(getattr(lm_hp, "outer_transform_lift_prefit_epochs", 200)), epochs)
            pre_epochs_min = min(
                int(getattr(lm_hp, "outer_transform_lift_prefit_epochs_min", 50)),
                max(0, pre_epochs - 1),
            )
            pre_patience = min(lm_hp.nval_patience, max(5, pre_epochs // 2))
            print(
                f"[Stage B] Lifted prefit enabled: link={lift_link}, "
                f"base_link={base_link or 'identity'}, prefit_epochs={pre_epochs}"
            )

            pre_best_val_loss, _, pre_best_val_p, pre_lm_opt = train_initial_model(
                model,
                train_loader,
                val_loader,
                epochs=pre_epochs,
                LM_strategy=lm_hp.strategy,
                nval_patience=pre_patience,
                # Loss is in lifted-space units; disable loss-target early stop.
                loss_target=None,
                epochs_min=pre_epochs_min,
                chisq_tol=lm_hp.chisq_tol,
                device=device,
                epochs_awful_check=lm_hp.epochs_awful_check,
                awful_threshold=lm_hp.awful_threshold,
                lm_verbose=lm_hp.LM_verbose,
                lm_hp=lm_hp,
            )
            if pre_best_val_p is not None:
                pre_lm_opt._update_param_groups(pre_best_val_p)
        except Exception as e_prefit:
            print(
                f"[Stage B] Warning: lifted prefit failed (link={lift_link}); "
                f"falling back to standard fit: {e_prefit}"
            )
        finally:
            # Restore baseline fit-link for evaluation / refine.
            model.fit_y_link = base_link
            model.fit_y_link_scale = base_scale

        # Baseline-space validation of the prefit solution
        val_loss = _eval_val_loss(
            model,
            val_loader,
            device=device,
            dtype=dtype,
            fit_y_link=base_link,
            fit_y_link_scale=base_scale,
        )

        # --- Stage 2: optional short refine in baseline space ---
        refine_epochs = min(
            int(getattr(lm_hp, "outer_transform_lift_refine_epochs", 200)), epochs
        )
        if refine_epochs > 0:
            refine_epochs_min = min(
                int(getattr(lm_hp, "outer_transform_lift_refine_epochs_min", 50)),
                max(0, refine_epochs - 1),
            )
            refine_patience = min(lm_hp.nval_patience, max(5, refine_epochs // 2))

            try:
                best_val_loss2, _, best_val_p2, lm_opt2 = train_initial_model(
                    model,
                    train_loader,
                    val_loader,
                    epochs=refine_epochs,
                    LM_strategy=lm_hp.strategy,
                    nval_patience=refine_patience,
                    loss_target=loss_target_stageB,
                    epochs_min=refine_epochs_min,
                    chisq_tol=lm_hp.chisq_tol,
                    device=device,
                    epochs_awful_check=lm_hp.epochs_awful_check,
                    awful_threshold=lm_hp.awful_threshold,
                    lm_verbose=lm_hp.LM_verbose,
                    lm_hp=lm_hp,
                )
                if best_val_p2 is not None:
                    lm_opt2._update_param_groups(best_val_p2)
                val_loss2 = float(best_val_loss2)

                # Keep the better of (prefit-evaluated) vs refined.
                if math.isfinite(val_loss2) and (val_loss2 <= val_loss):
                    val_loss = val_loss2
                else:
                    # If refine regressed, revert to the prefit params (if available).
                    if pre_best_val_p is not None:
                        lm_opt2._update_param_groups(pre_best_val_p)
            except Exception as e_refine:
                print(
                    f"[Stage B] Warning: refine after lifted prefit failed; "
                    f"keeping prefit solution: {e_refine}"
                )

    else:
        best_val_loss, _, best_val_p, lm_opt = train_initial_model(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            LM_strategy=lm_hp.strategy,
            nval_patience=lm_hp.nval_patience,
            loss_target=loss_target_stageB,
            epochs_min=lm_hp.epochs_min,
            chisq_tol=lm_hp.chisq_tol,
            device=device,
            epochs_awful_check=lm_hp.epochs_awful_check,
            awful_threshold=lm_hp.awful_threshold,
            lm_verbose=lm_hp.LM_verbose,
            lm_hp=lm_hp,
        )
        if best_val_p is not None:
            lm_opt._update_param_groups(best_val_p)
        val_loss = float(best_val_loss)

    reuse_out = _refresh_reuse_from_state(root, model)

    # Compute NN metrics for reporting
    from .engine import _compute_nn_metrics

    num_nn_atoms, num_multivar_nn_atoms, max_nn_arity = _compute_nn_metrics(root)

    return StageBState(
        root=root,
        model=model,
        reuse=reuse_out,
        val_loss=val_loss,
        num_nn_atoms=num_nn_atoms,
        num_multivar_nn_atoms=num_multivar_nn_atoms,
        max_nn_arity=max_nn_arity,
    )


def _stageB_candidate_local_seed(
    root: Node,
    custom_init_fn,
    *,
    start_idx: int = 0,
    base_seed: Optional[int] = None,
) -> int:
    """Derive a stable candidate seed without using Python's salted hash()."""
    if base_seed is None:
        base_seed = int(torch.initial_seed())
    try:
        root_key = ast_to_human_readable(root)
    except Exception:
        root_key = repr(root)
    init_key = getattr(custom_init_fn, "_candidate_seed_key", "")
    payload = f"{int(base_seed)}\0{root_key}\0{init_key}\0{int(start_idx)}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _run_with_candidate_local_rng(seed: int, fn):
    """Run one disposable candidate fit while preserving process RNG streams."""
    py_state = random.getstate()
    torch_cpu_state = torch.random.get_rng_state()
    numpy_state = None
    cuda_state = None
    try:
        import numpy as np

        numpy_state = np.random.get_state()
    except Exception:
        np = None
    try:
        if torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state_all()
    except Exception:
        cuda_state = None

    try:
        random.seed(int(seed))
        torch.manual_seed(int(seed))
        if np is not None:
            np.random.seed(int(seed) % (2**32))
        return fn()
    finally:
        random.setstate(py_state)
        torch.random.set_rng_state(torch_cpu_state)
        if np is not None and numpy_state is not None:
            np.random.set_state(numpy_state)
        if cuda_state is not None:
            try:
                torch.cuda.set_rng_state_all(cuda_state)
            except Exception:
                pass


def _fit_candidate_root(
    root: Node,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    val_loader,
    lm_hp: LMHyperparams,
    device: torch.device,
    dtype: torch.dtype,
    epochs_stageB: int,
    loss_scale: float,
    trig_by_axis: Optional[Dict[int, TrigAxisSpec]] = None,
    custom_init_fn=None,
    fresh_nn_factory=None,
    atom_factory=None,
) -> StageBState:
    """Fit with a semantic seed and bounded retries for fragile candidates."""
    try:
        max_starts = int(getattr(custom_init_fn, "_candidate_max_starts", 1))
    except Exception:
        max_starts = 1
    max_starts = max(1, min(4, max_starts))
    retry_nonfinite = bool(
        getattr(custom_init_fn, "_candidate_retry_nonfinite", False)
    )
    try:
        retry_stall_loss = float(
            getattr(custom_init_fn, "_candidate_retry_stall_loss", float("inf"))
        )
    except Exception:
        retry_stall_loss = float("inf")
    try:
        configured_seed = getattr(lm_hp, "candidate_seed_base", None)
        base_seed = (
            int(configured_seed)
            if configured_seed is not None
            else int(torch.initial_seed())
        )
    except Exception:
        base_seed = int(torch.initial_seed())

    best_state = None
    best_loss = float("inf")
    for start_idx in range(max_starts):
        if custom_init_fn is not None:
            # Consumed by _apply_custom_init_and_validate to jitter restart
            # inits (deterministic custom inits would otherwise re-enter the
            # same basin on every start).
            custom_init_fn._candidate_start_idx = start_idx
        seed = _stageB_candidate_local_seed(
            root,
            custom_init_fn,
            start_idx=start_idx,
            base_seed=base_seed,
        )

        def _fit_once():
            return _fit_candidate_root_once(
                root=root,
                reuse=reuse,
                train_loader=train_loader,
                val_loader=val_loader,
                lm_hp=lm_hp,
                device=device,
                dtype=dtype,
                epochs_stageB=epochs_stageB,
                loss_scale=loss_scale,
                trig_by_axis=trig_by_axis,
                custom_init_fn=custom_init_fn,
                fresh_nn_factory=fresh_nn_factory,
                atom_factory=atom_factory,
            )

        state = _run_with_candidate_local_rng(seed, _fit_once)
        try:
            loss = float(state.val_loss)
        except Exception:
            loss = float("inf")
        if best_state is None or (math.isfinite(loss) and loss < best_loss):
            best_state = state
            best_loss = loss
        stalled = math.isfinite(loss) and loss > retry_stall_loss
        nonfinite_retry = (not math.isfinite(loss)) and retry_nonfinite
        if not (stalled or nonfinite_retry):
            break
        if start_idx + 1 < max_starts:
            if stalled:
                print(
                    "[Stage B] Candidate fit stalled at "
                    f"{loss:.3e} > retry threshold {retry_stall_loss:.3e}; "
                    f"jittered restart {start_idx + 2}/{max_starts}."
                )
            else:
                print(
                    "[Stage B] Candidate fit was non-finite; retrying with stable "
                    f"start {start_idx + 2}/{max_starts}."
                )

    return best_state


def _aggregate_losses(
    losses: List[float], mode: str = "mean", weights: Optional[List[float]] = None
) -> float:
    mode = str(mode or "mean").lower().strip()
    if len(losses) == 0:
        return float("inf")
    if mode in ("mean", "avg", "average"):
        return float(sum(losses) / len(losses))
    if mode in ("sum",):
        return float(sum(losses))
    if mode in ("median",):
        xs = sorted(float(x) for x in losses)
        m = len(xs) // 2
        return float(xs[m]) if (len(xs) % 2 == 1) else 0.5 * float(xs[m - 1] + xs[m])
    if mode in ("weighted", "weighted_mean", "wmean"):
        if weights is None:
            return float(sum(losses) / len(losses))
        if len(weights) != len(losses):
            raise ValueError(f"weights length {len(weights)} != losses length {len(losses)}")
        wsum = float(sum(weights))
        if wsum <= 0:
            return float(sum(losses) / len(losses))
        return float(sum(w * float(l) for w, l in zip(weights, losses)) / wsum)
    raise ValueError(f"Unknown aggregation mode: {mode!r}")


def _build_joint_composites_for_class_sharing(
    root: Node,
    reuses: List[Dict[str, torch.nn.Module]],
    class_set: Set[str],
    device: torch.device,
    dtype: torch.dtype,
    fresh_nn_factory=None,
    atom_factory=None,
) -> Tuple[List[torch.nn.Module], Dict[str, torch.nn.Module]]:
    """Build composites, sharing class-tagged leaves via reuse-map injection.

    This mirrors Class-SR behavior: class-tag modules are injected directly into
    each dataset's reuse map before build. That preserves within-AST tag ties
    when a class tag appears multiple times.
    """
    nn_factory = make_reuse_only_nn_factory(
        device=device, dtype=dtype, fresh_nn_factory=fresh_nn_factory
    )

    atoms = collect_all_atoms(root)
    tag_to_leafidx: Dict[str, int] = {}
    for idx, atom in enumerate(atoms):
        tag = getattr(atom, "tag", None) or f"leaf{idx}"
        tag_to_leafidx.setdefault(tag, idx)

    composites: List[torch.nn.Module] = []
    shared_by_tag: Dict[str, torch.nn.Module] = {}

    for i, reuse_i in enumerate(reuses):
        reuse_build = _clone_reuse(reuse_i, device, dtype)
        if i > 0:
            for tag, shared_leaf in shared_by_tag.items():
                reuse_build[tag] = shared_leaf

        comp = build_composite_from_ast(
            root,
            dtype=dtype,
            device=device,
            nn_factory=nn_factory,
            atom_factory=_pick_atom_factory(atom_factory, i),
            reuse=reuse_build,
        )

        if i == 0:
            for tag in class_set:
                lidx = tag_to_leafidx.get(tag)
                if lidx is not None and lidx < len(comp.leaf):
                    shared_by_tag[tag] = comp.leaf[lidx]

        composites.append(comp)

    return composites, shared_by_tag


def _fit_joint_lm(
    root: Node,
    reuses: List[Dict[str, torch.nn.Module]],
    train_loaders: List[Any],
    val_loaders: List[Any],
    lm_hp: "LMHyperparams",
    device: torch.device,
    dtype: torch.dtype,
    epochs: int = 2000,
    fresh_nn_factory=None,
    atom_factory=None,
    custom_init_fn=None,
    dataset_ids: Optional[List[str]] = None,
) -> List["StageBState"]:
    """Joint LM solve: D composites sharing only global FreeConst leaves.

    Builds D composites from the same AST.  Leaves whose atoms have
    ``scope == "class"`` (global FreeConst) are **shared** (same
    ``nn.Parameter`` objects across composites).  Everything else —
    NN leaves, polynomial mapping, Scale, local FreeConst — is
    **independent** per dataset.

    The resulting Jacobian is block-diagonal with small off-diagonal
    blocks only for the shared global parameters.

    Returns a list of D ``StageBState`` objects (one per dataset).
    """
    import nestynet

    D = len(train_loaders)
    root = gauge_fix_multiplicative(root)

    # ── Identify shared (class) atoms ────────────────────────────────────
    from nestynet_sr.sr_core.bridges import AtomNode as _AtomNode

    all_atoms = collect_all_atoms(root)
    class_set: set = set()  # tags that should be shared
    for atom in all_atoms:
        if (isinstance(atom, _AtomNode)
                and getattr(atom, "scope", "experiment") == "class"):
            tag = getattr(atom, "tag", None)
            if tag is not None:
                class_set.add(tag)

    # ── Build D composites, sharing class-tagged leaves ──────────────────
    composites, shared_by_tag = _build_joint_composites_for_class_sharing(
        root=root,
        reuses=reuses,
        class_set=class_set,
        device=device,
        dtype=dtype,
        fresh_nn_factory=fresh_nn_factory,
        atom_factory=atom_factory,
    )

    if dataset_ids is None or len(dataset_ids) != D:
        dataset_ids = [f"ds{i}" for i in range(D)]

    def _call_custom_init(root_i, model_i, *, dataset_idx: int, dataset_id: str):
        if custom_init_fn is None:
            return
        try:
            import inspect

            sig = inspect.signature(custom_init_fn)
            accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            if accepts_kwargs or "dataset_idx" in sig.parameters or "dataset_id" in sig.parameters:
                custom_init_fn(
                    root_i,
                    model_i,
                    dataset_idx=int(dataset_idx),
                    dataset_id=str(dataset_id),
                )
            else:
                custom_init_fn(root_i, model_i)
        except Exception as e:
            print("[Stage B] Warning: custom leaf initialisation failed with:", e)

    for i, comp in enumerate(composites):
        setattr(comp, "fit_y_link", getattr(lm_hp, "fit_y_link", None))
        setattr(comp, "fit_y_link_scale", getattr(lm_hp, "fit_y_link_scale", 1.0))

        # Analytic leaf initialisation from this dataset's teacher data
        try:
            from nestynet_sr.sr_search.fitting_utils import (
                _initialise_analytic_leaves_from_reuse as _init_analytic,
            )
            _init_analytic(
                root=root, model=comp, reuse=reuses[i],
                train_loader=train_loaders[i], device=device, dtype=dtype,
            )
        except Exception:
            pass
        _call_custom_init(root, comp, dataset_idx=i, dataset_id=dataset_ids[i])

    # ── Collect unique parameters (shared ones appear once) ──────────────
    seen_ids: set = set()
    params: List[torch.nn.Parameter] = []
    for comp in composites:
        for p in comp.parameters():
            if p.requires_grad and id(p) not in seen_ids:
                seen_ids.add(id(p))
                params.append(p)

    if not params:
        # Zero-parameter model → just evaluate val losses
        from .engine import _compute_nn_metrics
        states = []
        for i in range(D):
            composites[i].eval()
            vl = 0.0
            n = 0
            with torch.no_grad():
                for batch in val_loaders[i]:
                    x, y = batch[0].to(device), batch[1].to(device)
                    vl += float(((composites[i](x) - y) ** 2).mean())
                    n += 1
            num_nn, num_mv, max_ar = _compute_nn_metrics(root)
            reuse_out = _refresh_reuse_from_state(root, composites[i])
            states.append(StageBState(
                root=root, model=composites[i], reuse=reuse_out,
                val_loss=vl / max(n, 1),
                num_nn_atoms=num_nn,
                num_multivar_nn_atoms=num_mv,
                max_nn_arity=max_ar,
            ))
        return states

    # ── Helper: build a residual-module factory ─────────────────────────
    def _seg_factory(composite, dataloader):
        def factory(_):
            return nestynet.optimizer.ResidualsModule(
                providers=[composite], dataloader=dataloader, device=device
            )
        return factory

    n_params = sum(p.numel() for p in params)
    shared_unique = {id(leaf): leaf for leaf in shared_by_tag.values()}
    n_shared = sum(
        p.numel() for leaf in shared_unique.values()
        for p in leaf.parameters() if p.requires_grad
    )
    print(
        f"[Stage B] Joint LM solve: {D} datasets, {n_params} params "
        f"({n_shared} shared), {len(class_set)} class tags"
    )

    # ── Per-composite warmup (shared params frozen) ───────────────────
    # Run a short per-dataset LBFGS warmup before joint LM. This stabilises
    # stiff candidate parameterisations (e.g. factorized symbolic search monomial wrappers),
    # especially in multi-dataset mode where direct joint LM can hit lam_max.

    def _has_distinct_dataset_reuses() -> bool:
        if D <= 1:
            return False
        common_tags = set(reuses[0].keys())
        for r in reuses[1:]:
            common_tags &= set(r.keys())
        for tag in common_tags:
            # Class-scoped tags are expected to be shared across datasets.
            if tag in class_set:
                continue
            leaf_ids = {id(r[tag]) for r in reuses if tag in r}
            if len(leaf_ids) > 1:
                return True
        return False

    has_distinct_dataset_reuses = _has_distinct_dataset_reuses()
    use_warmup = True

    shared_params_set = set()
    for leaf in shared_unique.values():
        for p in leaf.parameters():
            shared_params_set.add(id(p))

    def _freeze_shared():
        for pid in shared_params_set:
            for p in params:
                if id(p) == pid:
                    p.requires_grad_(False)

    def _unfreeze_shared():
        for pid in shared_params_set:
            for p in params:
                if id(p) == pid:
                    p.requires_grad_(True)

    def _lbfgs_warmup(composite, local_params, train_dl, val_dl, max_steps=300):
        """Quick LBFGS warmup for one composite against its own data.

        LBFGS handles the ill-conditioned parameterisation of factorized symbolic search
        (products like alpha*c1 create near-singular Jacobians) better
        than LM for coarse convergence.
        """
        # Collect train data into one batch
        xs, ys = [], []
        for batch in train_dl:
            x, y = batch[0].to(device), batch[1].to(device)
            xs.append(x)
            ys.append(y)
        X = torch.cat(xs, dim=0)
        Y = torch.cat(ys, dim=0)

        opt = torch.optim.LBFGS(local_params, lr=1.0, max_iter=20,
                                 line_search_fn="strong_wolfe")
        best_loss = float("inf")
        best_pvec = None

        for _step in range(max_steps // 20):
            def closure():
                opt.zero_grad()
                pred = composite(X)
                loss = ((pred - Y) ** 2).mean()
                loss.backward()
                return loss
            loss = opt.step(closure)
            lv = float(loss)
            if lv < best_loss:
                best_loss = lv
                best_pvec = torch.cat(
                    [p.detach().view(-1) for p in local_params]
                ).clone()
            if best_loss < 1e-12:
                break

        # Restore best and evaluate on val
        if best_pvec is not None:
            off = 0
            for p in local_params:
                n = p.numel()
                p.data.copy_(best_pvec[off:off + n].view_as(p))
                off += n

        # Evaluate val loss
        composite.eval()
        mses = []
        with torch.no_grad():
            for batch in val_dl:
                xv, yv = batch[0].to(device), batch[1].to(device)
                mses.append(float(((composite(xv) - yv) ** 2).mean()))
        return float(sum(mses) / max(len(mses), 1))

    if use_warmup:
        if has_distinct_dataset_reuses:
            print(
                "[Stage B]   warmup enabled with dataset-specific Stage-A reuses "
                "(candidate-stability mode)"
            )
        for i in range(D):
            local_params = [
                p for p in composites[i].parameters()
                if p.requires_grad and id(p) not in shared_params_set
            ]
            if not local_params:
                continue
            _freeze_shared()
            try:
                best_w = _lbfgs_warmup(composites[i], local_params,
                                        train_loaders[i], val_loaders[i])
                # If warmup didn't converge, try random restarts
                if best_w > 1e-2:
                    saved = torch.cat(
                        [p.detach().view(-1) for p in local_params]
                    ).clone()
                    for _restart in range(3):
                        with torch.no_grad():
                            for p in local_params:
                                p.copy_(torch.randn_like(p))
                        w2 = _lbfgs_warmup(composites[i], local_params,
                                            train_loaders[i], val_loaders[i])
                        if w2 < best_w:
                            best_w = w2
                            saved = torch.cat(
                                [p.detach().view(-1) for p in local_params]
                            ).clone()
                        if best_w < 1e-2:
                            break
                    # Restore overall best
                    off = 0
                    for p in local_params:
                        n = p.numel()
                        p.data.copy_(saved[off:off + n].view_as(p))
                        off += n
                print(f"[Stage B]   warmup ds={i}: val_loss={best_w:.4e}")
            except Exception as _we:
                print(f"[Stage B]   warmup ds={i} failed: {_we}")
            finally:
                _unfreeze_shared()
    else:
        print("[Stage B]   skipping LBFGS warmup")

    # ── Joint LM solve (all params, shared + local) ──────────────────
    residual_module_factories = [
        _seg_factory(composites[i], train_loaders[i]) for i in range(D)
    ]
    residual_module_factories_val = [
        _seg_factory(composites[i], val_loaders[i]) for i in range(D)
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
        verbose=lm_hp.LM_verbose,
        LM_strategy=lm_hp.strategy,
        chisq_tol=lm_hp.chisq_tol,
        **SR_LM_OVERRIDES,
    )
    lm_opt = nestynet.optimizer.Predictive_LM_Optimizer(
        params,
        residual_module_factories,
        residual_module_factories_val=residual_module_factories_val,
        cfg=cfg,
    )
    _sr_latest_joint_loss_metrics(lm_opt, target_count=D, label="[Stage B] ")

    # ── LM training loop ────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_param_vec = None
    nval_worse = 0
    iter_check = _sr_lm_iter_check(lm_opt)
    patience = _sr_align_validation_patience(
        lm_hp.nval_patience, iter_check, label="[Stage B] "
    )
    last_val_epoch = None
    last_report_train_selection_loss = None

    loss_target_stageB = lm_hp.loss_target * 0.01

    for epoch in range(epochs):
        loss_obj, loss_val_obj = lm_opt.step()
        loss_metrics = _sr_latest_joint_loss_metrics(
            lm_opt,
            target_count=D,
            label="[Stage B] ",
        )
        loss = float(loss_metrics.get("train_selection_loss", loss_metrics.get("train_data_mean_loss", loss_obj)))
        raw_val = loss_metrics.get("val_selection_loss", loss_metrics.get("val_data_mean_loss", loss_val_obj))
        loss_val = None if raw_val is None else float(raw_val)
        last_report_train_selection_loss, _ = _sr_maybe_trigger_prior_decay_from_stall(
            lm_opt,
            loss_metrics=loss_metrics,
            prev_report_train_selection_loss=last_report_train_selection_loss,
            epochs=epochs,
            label="[Stage B] ",
        )
        val_fresh_data = _sr_validation_fresh_after_step(
            lm_opt,
            require_metrics_ready=False,
        )
        val_fresh = _sr_validation_fresh_after_step(lm_opt)
        if val_fresh_data and loss_val is not None and loss_val < best_val_loss:
            best_val_loss = float(loss_val)
            best_param_vec = torch.cat(
                [p.detach().view(-1) for p in params]
            ).clone()
        if val_fresh and loss_val is not None and loss_val <= best_val_loss:
            nval_worse = 0
            last_val_epoch = epoch
        elif val_fresh and loss_val is not None:
            if last_val_epoch is None:
                nval_worse += iter_check
            else:
                nval_worse += max(1, epoch - last_val_epoch)
            last_val_epoch = epoch
        if lm_opt.state.get("halt"):
            break
        if val_fresh and nval_worse >= patience:
            break
        if loss is not None and loss_target_stageB and loss < loss_target_stageB:
            break

    # Restore best-validation parameters
    if best_param_vec is not None:
        offset = 0
        for p in params:
            n = p.numel()
            p.data.copy_(best_param_vec[offset : offset + n].view_as(p))
            offset += n

    # ── Build per-dataset StageBStates ───────────────────────────────────
    from .engine import _compute_nn_metrics

    states: List[StageBState] = []
    for i in range(D):
        composites[i].eval()
        mses = []
        with torch.no_grad():
            for batch in val_loaders[i]:
                x, y = batch[0].to(device), batch[1].to(device)
                mses.append(float(((composites[i](x) - y) ** 2).mean()))
        val_loss_i = float(sum(mses) / max(len(mses), 1))
        reuse_i = _refresh_reuse_from_state(root, composites[i])
        num_nn, num_mv, max_ar = _compute_nn_metrics(root)
        states.append(StageBState(
            root=root, model=composites[i], reuse=reuse_i,
            val_loss=val_loss_i,
            num_nn_atoms=num_nn,
            num_multivar_nn_atoms=num_mv,
            max_nn_arity=max_ar,
        ))

    per_ds = [f"{st.val_loss:.4e}" for st in states]
    print(
        f"[Stage B] Joint LM result: best_val={best_val_loss:.4e}, "
        f"per-dataset=[{', '.join(per_ds)}]"
    )

    return states


def _fit_candidate_root_multi(
    root: Node,
    reuses: List[Dict[str, torch.nn.Module]],
    train_loaders: List[Any],
    val_loaders: List[Any],
    lm_hp: LMHyperparams,
    device: torch.device,
    dtype: torch.dtype,
    epochs_stageB: int,
    loss_scales: Optional[List[float]] = None,
    trig_by_axis: Optional[Dict[int, TrigAxisSpec]] = None,
    custom_init_fn=None,
    fresh_nn_factory=None,
    dataset_ids: Optional[List[str]] = None,
    agg_mode: str = "mean",
    agg_weights: Optional[List[float]] = None,
    atom_factory=None,
) -> StageBState:
    """Fit a single AST structure to multiple datasets as one combined solve.

    Implementation:
    - Build D composites in one joint LM system.
    - If class-scoped atoms exist, those leaves are shared across datasets.
    - Otherwise all parameters remain dataset-specific (block-diagonal case).

    This keeps multi-dataset fitting mathematically "independent where possible"
    while still evaluating/optimising against the combined experiment.

    Returns StageBState with:
    - `state.model`: First dataset's model (primary model reference)
    - `state.models`: List of all dataset-specific models
    - `state.reuses/state.val_losses`: Per-dataset reuse maps and losses
    """
    root = gauge_fix_multiplicative(root)
    if len(train_loaders) != len(val_loaders):
        raise ValueError("train_loaders and val_loaders must have same length")
    if len(reuses) != len(train_loaders):
        raise ValueError("reuses must have same length as loaders")
    D = len(train_loaders)
    if loss_scales is None:
        loss_scales = [1.0 for _ in range(D)]
    if len(loss_scales) != D:
        raise ValueError("loss_scales must have same length as loaders")

    # Joint LM solve for multi-dataset mode (shared leaves only when
    # class-scoped atoms are present; otherwise params stay per-dataset).
    if D > 1:
        try:
            states = _fit_joint_lm(
                root=root,
                reuses=reuses,
                train_loaders=train_loaders,
                val_loaders=val_loaders,
                lm_hp=lm_hp,
                device=device,
                dtype=dtype,
                epochs=max(1, min(epochs_stageB, lm_hp.epochs)),
                fresh_nn_factory=fresh_nn_factory,
                atom_factory=atom_factory,
                custom_init_fn=custom_init_fn,
                dataset_ids=dataset_ids,
            )
        except Exception as _joint_exc:
            import traceback
            traceback.print_exc()
            print(f"[Stage B] Joint LM solve failed ({_joint_exc}); "
                  "falling back to independent fitting.")
            states = None

        if states is None:
            states = []
            for i in range(D):
                st = _fit_candidate_root(
                    root=root, reuse=reuses[i],
                    train_loader=train_loaders[i], val_loader=val_loaders[i],
                    lm_hp=lm_hp, device=device, dtype=dtype,
                    epochs_stageB=epochs_stageB, loss_scale=float(loss_scales[i]),
                    trig_by_axis=trig_by_axis, custom_init_fn=custom_init_fn,
                    fresh_nn_factory=fresh_nn_factory,
                    atom_factory=_pick_atom_factory(atom_factory, i),
                )
                states.append(st)
    else:
        states = []
        for i in range(D):
            st = _fit_candidate_root(
                root=root, reuse=reuses[i],
                train_loader=train_loaders[i], val_loader=val_loaders[i],
                lm_hp=lm_hp, device=device, dtype=dtype,
                epochs_stageB=epochs_stageB, loss_scale=float(loss_scales[i]),
                trig_by_axis=trig_by_axis, custom_init_fn=custom_init_fn,
                fresh_nn_factory=fresh_nn_factory,
                atom_factory=_pick_atom_factory(atom_factory, i),
            )
            states.append(st)

    val_losses = [float(st.val_loss) for st in states]
    agg = _aggregate_losses(val_losses, mode=agg_mode, weights=agg_weights)

    primary = states[0]
    return StageBState(
        root=root,
        model=primary.model,
        reuse=primary.reuse,
        val_loss=float(agg),
        models=[st.model for st in states],
        reuses=[st.reuse for st in states],
        val_losses=val_losses,
        dataset_ids=list(dataset_ids) if dataset_ids is not None else None,
        agg_mode=str(agg_mode),
        agg_weights=list(agg_weights) if agg_weights is not None else None,
        num_nn_atoms=primary.num_nn_atoms,
        num_multivar_nn_atoms=primary.num_multivar_nn_atoms,
        max_nn_arity=primary.max_nn_arity,
    )


def summarize_global_power_law(
    model,
    datagen,
    Nxvars: int,
    device: torch.device,
    max_points: int = 5000,
    min_points: int = 200,
    round_tol: float = 0.01,
    rel_rms_threshold: float = 1e-3,
):
    X, F = _sample_values_from_model(
        model, datagen, max_batches=8, max_points=max_points, device=device
    )
    if X is None:
        return None
    X = X[:, :Nxvars]
    m = torch.ones_like(F, dtype=torch.bool)
    for j in range(Nxvars):
        m &= X[:, j] > 0
    m &= F > 0
    X = X[m]
    F = F[m]
    N = X.size(0)
    if N < min_points:
        return None
    Z = torch.log(X)
    Y = torch.log(F)
    N, d = Z.shape
    Phi = torch.cat([torch.ones(N, 1, dtype=Z.dtype), Z], dim=1)
    beta = torch.linalg.lstsq(Phi, Y.unsqueeze(1)).solution.squeeze(1)
    ks_raw = beta[1:]
    ks_simpl = ks_raw.clone()
    for j in range(d):
        r = torch.round(ks_raw[j])
        if abs(ks_raw[j] - r) < round_tol:
            ks_simpl[j] = r
    logC = (Y - (Z @ ks_simpl)).mean()
    C = float(torch.exp(logC))
    F_simple = C * torch.exp(Z @ ks_simpl)
    resid = F_simple - F
    rms_abs = float(torch.sqrt(torch.mean(resid * resid)))
    std_target = float(F.std(unbiased=False))
    if std_target < 1e-12:
        return None
    rms_rel = rms_abs / std_target
    if rms_rel > rel_rms_threshold:
        return None
    return C, [float(k.item()) for k in ks_simpl], rms_abs, rms_rel


def _snap_exponent_to_half_integer(k: float, half_tol: float = 1e-3) -> Optional[float]:
    """
    Snap a real exponent to the nearest multiple of 0.5 if it is
    within 'half_tol'. Returns the snapped value or None.
    """
    m = round(2.0 * k)
    k_snap = 0.5 * m
    if abs(k - k_snap) <= half_tol:
        return k_snap
    return None


def _format_monomial_from_exponents(ks: List[float], zero_tol: float = 1e-9) -> str:
    """
    Build a human‑readable monomial string from exponents ks, with
    small exponents treated as zero and special‑cases for ±1 and ±1/2:

        x^1      -> x
        x^-1     -> 1/x
        x^0.5    -> sqrt(x)
        x^-0.5   -> 1/sqrt(x)
    """
    parts: List[str] = []
    for j, k in enumerate(ks):
        if abs(k) < zero_tol:
            continue
        if abs(k - 1.0) < zero_tol:
            parts.append(f"x{j}")
        elif abs(k + 1.0) < zero_tol:
            parts.append(f"1/x{j}")
        elif abs(k - 0.5) < zero_tol:
            parts.append(f"sqrt(x{j})")
        elif abs(k + 0.5) < zero_tol:
            parts.append(f"1/sqrt(x{j})")
        else:
            parts.append(f"x{j}^{k:g}")
    return "1" if not parts else " * ".join(parts)
