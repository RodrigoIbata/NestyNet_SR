# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Main Stage B orchestrator function.

This module contains run_stageB_from_model(), the main entry point that
orchestrates the entire Stage B refinement process.
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import replace
from typing import Any, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from nestynet_sr.sr_core.bridges import Node, collect_nn_atoms
from nestynet_sr.sr_core.coefficient_metadata import (
    CoefficientMetadataError,
    collect_coefficient_metadata,
    normalize_coefficient_metadata_by_dataset,
)
from nestynet_sr.sr_core.sympy_units import check_sympy_units

# Terminal colors
GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"
RESET = "\033[0m"


def _loss_str(loss: float, lm_hp) -> str:
    """Format loss value with asinh indicator if fit-link is active."""
    if getattr(lm_hp, "fit_y_link", None) == "asinh":
        return f"{loss:.4e} [asinh]"
    return f"{loss:.4e}"


def _with_stagec_unit_certificates(
    sympy_meta: Optional[dict],
    *,
    phi_expr_str: Optional[str],
    y_expr_str: Optional[str],
    Nxvars: int,
    units_spec,
    enforce_units: bool,
    coordinate_space: str = "internal",
) -> Optional[dict]:
    """Attach phi/raw-y unit certificates without changing numeric verdicts."""

    if not enforce_units or units_spec is None:
        return sympy_meta
    meta = dict(sympy_meta or {})
    prior_certificate = meta.get("unit_admissibility")
    if phi_expr_str is None or y_expr_str is None:
        reason = "phi-space or raw-y expression is unavailable for dimensional checking"
        unit_certificate = {
            "checked": False,
            "valid": None,
            "checker": "sympy_units_v1",
            "code": "expression_unavailable",
            "reason": reason,
            "expression_space": "phi_and_y",
            "coordinate_space": str(coordinate_space),
        }
        if (
            str(coordinate_space) == "raw"
            and isinstance(prior_certificate, dict)
            and prior_certificate.get("coordinate_space") != "raw"
        ):
            unit_certificate["internal_coordinates"] = prior_certificate
        meta.update(
            {
                "units_checked": False,
                "units_ok": None,
                "units_reason": reason,
                "unit_admissibility": unit_certificate,
            }
        )
        if meta.get("accepted") is True:
            meta.update(
                {
                    "raw_accepted_before_unit_check": True,
                    "accepted": False,
                    "kind": "unit_check_expression_unavailable",
                    "reason": reason,
                }
            )
        return meta
    variable_names = tuple(f"x{i}" for i in range(int(Nxvars)))
    phi_check = check_sympy_units(
        phi_expr_str,
        variable_names,
        units_spec,
        expression_space="phi",
    )
    y_check = check_sympy_units(
        y_expr_str,
        variable_names,
        units_spec,
        expression_space="y",
    )
    valid = bool(phi_check.ok and y_check.ok)
    first_failure = phi_check if not phi_check.ok else y_check
    unit_certificate = {
        "checked": bool(phi_check.checked and y_check.checked),
        "valid": valid,
        "checker": "sympy_units_v1",
        "code": "units_ok" if valid else first_failure.code,
        "reason": (
            "phi-space and raw-y expressions are dimensionally admissible"
            if valid
            else first_failure.reason
        ),
        "expression_space": "phi_and_y",
        "coordinate_space": str(coordinate_space),
        "phi": phi_check.to_dict(),
        "y": y_check.to_dict(),
    }
    if (
        str(coordinate_space) == "raw"
        and isinstance(prior_certificate, dict)
        and prior_certificate.get("coordinate_space") != "raw"
    ):
        unit_certificate["internal_coordinates"] = prior_certificate
    meta.update(
        {
            "units_checked": unit_certificate["checked"],
            "units_ok": unit_certificate["valid"],
            "units_reason": unit_certificate["reason"],
            "unit_admissibility": unit_certificate,
        }
    )
    if not valid:
        unit_check_invalidated_acceptance = meta.get("accepted") is True
        if unit_check_invalidated_acceptance:
            meta["raw_accepted_before_unit_check"] = True
        if unit_check_invalidated_acceptance or not sympy_meta:
            meta.update(
                {
                    "accepted": False,
                    "parse_success": meta.get("parse_success", True),
                    "kind": (
                        "phi_unit_check_failed"
                        if not phi_check.ok
                        else "raw_y_unit_check_failed"
                    ),
                    "reason": unit_certificate["reason"],
                }
            )
            proposal_budget = dict(meta.get("proposal_budget") or {})
            proposal_budget.update(
                {
                    "requested_count": int(
                        proposal_budget.get("requested_count", 1)
                    ),
                    "emitted": 0,
                    "exhausted": True,
                    "exhaustion_reason": meta["kind"],
                }
            )
            meta["proposal_budget"] = proposal_budget
    elif not sympy_meta:
        # Unit consistency alone is not a numerical equivalence certificate.
        meta.update(
            {
                "accepted": False,
                "parse_success": True,
                "kind": "unit_check_only",
                "reason": "Stage C produced no numerical-fidelity certificate",
            }
        )
    return meta

# Import from parent modules (sr_search).
# Prefer package-relative imports so representation.py is imported as
# nestynet_sr.sr_search.representation (fixes relative-import issues in Stage C).
# Keep a legacy fallback for non-package execution contexts.
try:
    from ..config import DataHyperparams, LMHyperparams
    from ..data_utils import build_datasets, build_datasets_multi
    from ..features import (
        PeriodicityStructureHint,
        TrigAxisSpec,
        TrigProbeTarget,
        _compound_to_probe_target,
        discover_scaling_features,
        discover_trig_argument_structure,
        discover_trig_axes,
        probe_oracle_scaling,
        probe_trig_scaling,
    )
    from ..gauge_fix import gauge_fix_additive_pairs
    from ..model_selection import (
        clamp_threshold_to_noise_floor,
        estimate_transform_noise_floor_raw,
        resolve_acceptance_noise_floor_raw,
    )
    from ..representation import (
        _HAVE_SYMPY,
        _sympy_simplify_expression,
        guarded_sympy_simplify_expression,
        pretty_print_state,
    )
except Exception:
    import os
    import sys
    _parent_dir = os.path.dirname(os.path.dirname(__file__))
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    from config import DataHyperparams, LMHyperparams
    from data_utils import build_datasets, build_datasets_multi
    from features import (
        PeriodicityStructureHint,
        TrigAxisSpec,
        TrigProbeTarget,
        _compound_to_probe_target,
        discover_scaling_features,
        discover_trig_argument_structure,
        discover_trig_axes,
        probe_oracle_scaling,
        probe_trig_scaling,
    )
    from gauge_fix import gauge_fix_additive_pairs
    from model_selection import (
        clamp_threshold_to_noise_floor,
        estimate_transform_noise_floor_raw,
        resolve_acceptance_noise_floor_raw,
    )
    from representation import (
        _HAVE_SYMPY,
        _sympy_simplify_expression,
        guarded_sympy_simplify_expression,
        pretty_print_state,
    )

from .atom_mapping import _refresh_reuse_from_state
from .engine import (
    StageBContext,
    StageBEngine,
    StageBState,
    _annotate_nonsense_units_leaves,
    _asinh_yspace_scale_from_loader,
)

# Import from stageB package
from .evaluation import (
    _compute_original_y_mad2_with_inverse,
    _compute_y_med_mad_from_loader,
    _eval_original_y_mse_with_inverse,
    _eval_mse_and_rms,
    _eval_val_mse,
    _phi_pred_error_from_loader,
    _print_val_batch_stats,
    _shuffle_axis_sensitivity,
)
from .feature_utils import _scaling_index, _trig_index
from .fitting import (
    _fit_candidate_root,
    _fit_candidate_root_multi,
    _format_monomial_from_exponents,
    _snap_exponent_to_half_integer,
    summarize_global_power_law,
)


def _wrap_with_inverse_transform(phi_expr_str: str, y_op_inv, simplify: bool = True) -> Optional[str]:
    """Wrap a φ-space expression string with the inverse y-transform to produce y-space.

    This is a small compatibility wrapper; the single source of truth for the
    mapping from inverse transforms to symbolic wrappers lives in
    :mod:`nestynet_sr.sr_search.transform_render`.
    """
    from ..transform_render import wrap_phi_expr_str

    return wrap_phi_expr_str(phi_expr_str, y_op_inv, simplify=simplify)


def _trig_scale_to_axis_specs(trig_scale_specs):
    """Convert oracle TrigScaleSpecs to TrigAxisSpecs for Stage B."""
    result = []
    for ts in (trig_scale_specs or []):
        # Map trig_fn to phase: cos → 0.0, sin → π/2
        phase = 0.0 if ts.trig_fn == "cos" else math.pi / 2
        result.append(TrigAxisSpec(
            axis=int(ts.axis),
            omega=float(ts.omega),
            strength=100.0,  # synthetic; oracle-verified
            n_points=int(ts.n_points),
            tmin=0.0,
            tmax=0.0,
            phase=phase,
            rel_std=float(ts.rel_std),
            basis_fn=str(getattr(ts, "basis_fn", "") or getattr(ts, "trig_fn", "")),
        ))
    return result


def _unresolved_nn_leaf_records(root: Node) -> List[dict]:
    """Summarize trainable NN atoms that remain in the final symbolic AST."""
    records: List[dict] = []
    try:
        from nestynet_sr.sr_core.bridges import compound_input_expr
        from nestynet_sr.sr_core import ast_to_human_readable

        for atom in collect_nn_atoms(root):
            try:
                expr = compound_input_expr(atom)
                input_s = ast_to_human_readable(expr)
            except Exception:
                input_s = None
            records.append(
                {
                    "tag": getattr(atom, "tag", None),
                    "var_idxs": list(getattr(atom, "var_idxs", ()) or ()),
                    "input": input_s,
                }
            )
    except Exception:
        return records
    return records


def _extract_nontrivial_targets_from_model(root: Node) -> List[TrigProbeTarget]:
    """Extract nontrivial input-expression targets from model's NN atoms for trig probing.

    Iterates over all NN atoms in the AST and extracts any nontrivial input
    expressions that can be used as trig probe targets.
    """
    from nestynet_sr.sr_core.bridges import collect_nn_atoms, get_input_exprs, is_trivial_input

    targets = []
    seen_names: set = set()

    for atom in collect_nn_atoms(root):
        for z_expr in get_input_exprs(atom):
            # Skip trivial Var(i) inputs — only nontrivial expressions are probed
            if is_trivial_input(z_expr):
                continue

            # Try to convert expression to a TrigProbeTarget
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


# Backward-compatible alias
_extract_compound_targets_from_model = _extract_nontrivial_targets_from_model


def _apply_additive_gauge_fix_to_state_reuse(
    *,
    root: Node,
    state: StageBState,
    train_loader_probe,
    device: torch.device,
    dtype: torch.dtype,
    cancel_ratio_thresh: float = 8.0,
    log_fn=None,
) -> StageBState:
    """Apply additive gauge-fix to the reuse maps Stage B actually consults.

    Stage B proposal builders and candidate fits consume ``state.reuse`` /
    ``state.reuses`` rather than the original local ``reuse`` dict assembled
    during startup. Mutating that stale local dict does not affect subsequent
    rule probes, which leaves large cancelling offsets in additive leaf pairs.
    """
    if getattr(state, "reuses", None):
        fixed_reuses = []
        loaders = train_loader_probe
        for i, reuse_i in enumerate(state.reuses):
            loader_i = loaders[i] if isinstance(loaders, (list, tuple)) else loaders
            fixed_reuses.append(
                gauge_fix_additive_pairs(
                    root,
                    reuse_i,
                    loader_i,
                    device=device,
                    dtype=dtype,
                    cancel_ratio_thresh=cancel_ratio_thresh,
                    log_fn=log_fn,
                )
            )
        state.reuses = fixed_reuses
        state.reuse = fixed_reuses[0] if fixed_reuses else {}
        return state

    state.reuse = gauge_fix_additive_pairs(
        root,
        state.reuse,
        train_loader_probe,
        device=device,
        dtype=dtype,
        cancel_ratio_thresh=cancel_ratio_thresh,
        log_fn=log_fn,
    )
    return state


class _XTransformDataset(Dataset):
    """Wrap a dataset and apply an x-coordinate transform to the first tensor."""

    def __init__(self, base_ds, x_op):
        self.base_ds = base_ds
        self.x_op = x_op

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        item = self.base_ds[idx]
        # Most SR datasets yield (x, y) but preserve any extra fields.
        x = item[0]
        rest = item[1:]
        return (self.x_op(x),) + rest


def run_stageB_from_model(
    stageA_model: torch.nn.Module,
    stageA_ast: Node,
    filepath: Any,
    Nxvars: int,
    data_hp: DataHyperparams,
    lm_hp: LMHyperparams,
    device: torch.device,
    dtype: torch.dtype,
    np_dtype,
    y_op=None,
    y_op_inv=None,
    y_transform_name: str = "identity",
    y_raw_full=None,
    noise_sigma_y: Optional[float] = None,
    noise_floor_mc_samples: int = 8,
    max_outer_iters: int = 30,
    epochs_stageB: int = 2000,
    score_tol: float = 0.0,
    verbose: bool = True,
    fresh_nn_factory=None,
    disabled_patterns: list = None,
    use_stageA_reuse: bool = True,
    stageA_models: Optional[List[torch.nn.Module]] = None,
    # Optional overrides for non-CSV workflows (e.g. DE residual fitting)
    # Provide pre-built datasets/loaders to bypass build_datasets(_multi).
    datasets_override: Any = None,
    # Optional explicit reuse map(s) keyed by atom tags.
    # - single-dataset: dict[tag -> module]
    # - multi-dataset  : list[dict[tag -> module]]
    reuse_override: Any = None,
    units_spec=None,
    enforce_units: bool = False,
    verbose_separabilities: bool = False,
    max_backtracks: int = 3,
    extra_rules: list = None,
    use_factorized_search: bool = True,
    factorized_search_hp=None,
    atom_factory=None,
    phase_hints=None,
    phase_context_hints=None,
    outer_link_hints=None,
) -> StageBState:
    """
    Entry point: run a few Stage-B refinement steps starting from a
    trained Stage-A ASTCompositeAdaptor.

    Parameters
    ----------
    stageA_model : ASTCompositeAdaptor
        Trained Stage-A model (notation='prefix').
    stageA_ast   : Node
        AST representing the Stage-A expression structure.
    stageA_models : list[ASTCompositeAdaptor] | None
        Optional per-dataset Stage-A models (multi-dataset mode). When provided,
        Stage B initialises each dataset from its own teacher reuse map instead of
        cloning dataset-0 leaves across all datasets.
    filepath     : str
        Path to the CSV data, used to rebuild datasets.
    Nxvars       : int
        Number of input variables.
    data_hp      : DataHyperparams
    lm_hp        : LMHyperparams
    device       : torch.device
    dtype        : torch.dtype
        np_dtype     : numpy dtype for build_datasets
        y_op         : callable or None
            Optional numpy-level y-transform. If not None, Stage B
            optimises the model in the same φ(y)-space that Stage A
            used (e.g. log(y)).
        y_op_inv     : callable or None
            Optional inverse y-transform φ^{-1}. If provided, Stage B
            will *report* the final expression both in φ(y)-space and
            in the original y-space by wrapping the pretty-printed
            φ-expression in y_op_inv(·). The underlying AST and model
            remain in φ(y)-space.
    max_outer_iters : int
        Maximum number of accepted rewrite steps.
    epochs_stageB   : int
        Maximum number of LM epochs for each candidate.
    score_tol       : float
        Minimum required improvement in score to accept a rewrite.

    Returns
    -------
    StageBState with the final AST, fitted model, and score.
    """
    raw_units_spec = units_spec
    # Rebuild datasets in the same y-space as Stage A. If y_op is None this
    # reduces to the original identity case.
    dataset_ids = None
    loss_scales = None
    agg_mode = "mean"
    agg_weights = None

    # Optional non-CSV path: caller supplies datasets/loaders.
    if datasets_override is not None:
        try:
            dataset_train = datasets_override.get("dataset_train", None)
            dataset_val = datasets_override.get("dataset_val", None)
            train_loader = datasets_override.get("train_loader", None)
            val_loader = datasets_override.get("val_loader", None)
            dataset_ids = datasets_override.get("dataset_ids", None)
            loss_scales = datasets_override.get("loss_scales", None)
            agg_mode = datasets_override.get("agg_mode", agg_mode)
            agg_weights = datasets_override.get("agg_weights", agg_weights)
        except Exception as e:
            raise ValueError("datasets_override must be a dict-like object") from e

        # Fall back to extracting datasets from loaders when not supplied.
        if dataset_train is None and train_loader is not None:
            if isinstance(train_loader, (list, tuple)):
                dataset_train = [dl.dataset for dl in train_loader]
            else:
                dataset_train = train_loader.dataset
        if dataset_val is None and val_loader is not None:
            if isinstance(val_loader, (list, tuple)):
                dataset_val = [dl.dataset for dl in val_loader]
            else:
                dataset_val = val_loader.dataset

        # Infer multi-dataset mode from loader structure.
        is_multi = isinstance(train_loader, (list, tuple))

        # Default dataset ids if not provided.
        if dataset_ids is None:
            if isinstance(filepath, (list, tuple)):
                dataset_ids = [pathlib.Path(str(p)).stem for p in filepath]
            else:
                dataset_ids = [pathlib.Path(str(filepath)).stem]

        # Aggregate weights default to val set sizes in multi mode.
        if is_multi:
            try:
                _n_val = sum(len(ds) for ds in dataset_val)
            except Exception:
                _n_val = 0
            if agg_mode is None:
                agg_mode = "weighted"
            if agg_weights is None:
                try:
                    agg_weights = [float(len(ds)) for ds in dataset_val]
                except Exception:
                    agg_weights = None
        else:
            try:
                _n_val = len(dataset_val)
            except Exception:
                _n_val = 0

        if train_loader is None or val_loader is None:
            raise RuntimeError("Stage B: datasets_override missing train/val loaders")

    else:
        is_multi = isinstance(filepath, (list, tuple))
        if is_multi:
            filepaths = [str(p) for p in filepath]
            dataset_ids = [pathlib.Path(p).stem for p in filepaths]
            dataset_train, dataset_val, train_loader, val_loader = build_datasets_multi(
                filepaths=filepaths,
                Nxvars=Nxvars,
                np_dtype=np_dtype,
                data_hp=data_hp,
                y_op=y_op,
            )
            if train_loader is None or val_loader is None:
                raise RuntimeError("Stage B: failed to build datasets for one or more CSVs.")
            _n_val = sum(len(ds) for ds in dataset_val)
            agg_mode = "weighted"
            agg_weights = [float(len(ds)) for ds in dataset_val]
        else:
            dataset_ids = [pathlib.Path(str(filepath)).stem]
            dataset_train, dataset_val, train_loader, val_loader = build_datasets(
                filepath=str(filepath),
                Nxvars=Nxvars,
                np_dtype=np_dtype,
                data_hp=data_hp,
                y_op=y_op,
            )
            if train_loader is None or val_loader is None:
                raise RuntimeError("Stage B: failed to build datasets.")
            _n_val = len(dataset_val)

    # Derive MAD-based scaling for Stage B if requested.
    if getattr(lm_hp, "loss_in_MAD_units", False):
        fit_link_name = getattr(lm_hp, "fit_y_link", None)
        fit_link_scale = float(getattr(lm_hp, "fit_y_link_scale", 1.0))
        if is_multi:
            loss_scales = []
            for dl in train_loader:
                _, y_mad = _compute_y_med_mad_from_loader(
                    dl,
                    device,
                    fit_link=fit_link_name,
                    fit_link_scale=fit_link_scale,
                )
                if (y_mad is None) or (y_mad <= 0.0):
                    loss_scales.append(1.0)
                else:
                    loss_scales.append(float(y_mad) ** 2)
            loss_scale = float(sum(loss_scales) / max(1, len(loss_scales)))
            if verbose:
                labels = dataset_ids if dataset_ids is not None else [f"dataset_{i}" for i in range(len(loss_scales))]
                msg = ", ".join(
                    [f"{str(lbl)}: {s:.3g}" for lbl, s in zip(labels, loss_scales)]
                )
                print(f"[Stage B] Using per-dataset MAD scales: {msg}")
        else:
            y_med_stageB, y_mad_stageB = _compute_y_med_mad_from_loader(
                train_loader,
                device,
                fit_link=fit_link_name,
                fit_link_scale=fit_link_scale,
            )
            if (y_mad_stageB is None) or (y_mad_stageB <= 0.0):
                loss_scale = 1.0
                if verbose:
                    print(
                        "[Stage B] Warning: MAD(y) non-positive or undefined; using unscaled LM thresholds."
                    )
            else:
                loss_scale = float(y_mad_stageB**2)
                if verbose:
                    print(
                        "[Stage B] Using MAD-normalised thresholds: MAD(y)≈{:.3g}, scale={:.3g}".format(
                            y_mad_stageB, loss_scale
                        )
                    )
    else:
        loss_scale = 1.0

    def _has_explicit_acceptance_noise_floor() -> bool:
        try:
            raw = float(getattr(lm_hp, "acceptance_noise_floor_raw", None))
            if math.isfinite(raw) and raw >= 0.0:
                return True
        except Exception:
            pass
        try:
            base = float(getattr(lm_hp, "acceptance_noise_floor", None))
            if math.isfinite(base) and base >= 0.0:
                return True
        except Exception:
            pass
        return False

    acceptance_noise_floor_raw = 0.0
    original_y_noise_floor_raw = 0.0
    try:
        _sigma_y_for_original = float(noise_sigma_y)
        if math.isfinite(_sigma_y_for_original) and _sigma_y_for_original > 0.0:
            original_y_noise_floor_raw = float(_sigma_y_for_original * _sigma_y_for_original)
    except Exception:
        original_y_noise_floor_raw = 0.0
    if (
        (not _has_explicit_acceptance_noise_floor())
        and (y_raw_full is not None)
        and (noise_sigma_y is not None)
        and (float(noise_sigma_y) > 0.0)
    ):
        acceptance_noise_floor_raw = float(
            estimate_transform_noise_floor_raw(
                y_raw_full,
                y_op,
                noise_sigma_y,
                fit_link=getattr(lm_hp, "fit_y_link", None),
                fit_link_scale=float(getattr(lm_hp, "fit_y_link_scale", 1.0)),
                n_mc=noise_floor_mc_samples,
            )
            or 0.0
        )
        lm_hp.acceptance_noise_floor_raw = (
            float(acceptance_noise_floor_raw)
            if acceptance_noise_floor_raw > 0.0
            else None
        )
        if verbose:
            label = "identity" if y_op is None else getattr(y_op, "__name__", str(y_op))
            if acceptance_noise_floor_raw > 0.0:
                print(
                    f"[Stage B][Noise] y-space={label}: sigma_y={float(noise_sigma_y):.3g}, "
                    f"floor_raw={acceptance_noise_floor_raw:.3g}"
                )
            else:
                print(
                    f"[Stage B][Noise] y-space={label}: could not estimate a stable loss floor; "
                    "leaving thresholds unchanged."
                )

    if getattr(lm_hp, "loss_in_MAD_units", False):
        loss_good_enough_raw = lm_hp.loss_target * loss_scale
    else:
        loss_good_enough_raw = lm_hp.loss_target

    # Initialize disabled patterns and tracking
    if disabled_patterns is None:
        disabled_patterns = []
    enabled_patterns = []  # Track which patterns were actually tried/accepted

    # Use the AST passed directly from Stage A
    root = stageA_ast

    # Build reuse map from the Stage-A model's leaves (if use_stageA_reuse=True),
    # unless an explicit reuse_override is provided.
    reuse = {}
    reuses = None
    if reuse_override is not None:
        if is_multi:
            if not isinstance(reuse_override, (list, tuple)):
                raise ValueError("reuse_override must be a list of reuse maps in multi-dataset mode")
            if len(reuse_override) != len(train_loader):
                raise ValueError(
                    f"reuse_override length {len(reuse_override)} must match number of datasets {len(train_loader)}"
                )
            reuses = [dict(r) if r is not None else {} for r in reuse_override]
            reuse = reuses[0] if len(reuses) > 0 else {}
        else:
            if isinstance(reuse_override, (list, tuple)):
                reuse = dict(reuse_override[0]) if len(reuse_override) > 0 else {}
            else:
                reuse = dict(reuse_override)
    elif use_stageA_reuse:
        if is_multi and stageA_models is not None:
            if len(stageA_models) != len(train_loader):
                raise ValueError(
                    f"stageA_models length {len(stageA_models)} must match number of datasets {len(train_loader)}"
                )
            reuses = []
            for m in stageA_models:
                reuse_i = {}
                for i, leaf in enumerate(m.leaf):
                    tag = f"leaf{i}"
                    reuse_i[tag] = leaf
                reuses.append(reuse_i)
            reuse = reuses[0] if len(reuses) > 0 else {}
        else:
            for i, leaf in enumerate(stageA_model.leaf):
                tag = f"leaf{i}"
                reuse[tag] = leaf
            if is_multi:
                # Keep dict objects separate even if leaves are shared.
                reuses = [dict(reuse) for _ in range(len(train_loader))]
    else:
        # If we're not reusing Stage A weights (e.g., because we switched φ(y)-space),
        # fresh_nn_factory must be provided for any NN atoms in the AST.
        if fresh_nn_factory is None:
            raise ValueError(
                "Stage B requested fresh NN leaves (use_stageA_reuse=False) but no "
                "fresh_nn_factory was provided."
            )

    # -------------------------------------------------------------------------
    # Stage-A x-coordinate transforms
    #
    # If Stage A applied an x-coordinate transformation (e.g. trig substitution or
    # x -> x^2), we operate Stage B in those *internal* coordinates by transforming
    # the dataloaders. This keeps refitting and rule checks consistent with how the
    # Stage-A model was trained, while still allowing us to rewrite expressions back
    # to raw x later.
    stageA_x_transforms = getattr(stageA_model, "_x_transform", None) or {}
    xcoords = None
    xcoords_applied = False
    try:
        from nestynet_sr.sr_search.xcoord import XCoordSystem

        if stageA_x_transforms:
            xcoords = XCoordSystem.from_map(stageA_x_transforms, Nx_raw=Nxvars)
            if xcoords is not None and (not xcoords.is_identity()):
                def x_op(x, _xc=xcoords):
                    return _xc.apply_torch(x)
                if is_multi:
                    train_loader = [
                        DataLoader(
                            _XTransformDataset(ds, x_op),
                            batch_size=getattr(dl, "batch_size", None) or data_hp.batch_size,
                            shuffle=False,
                        )
                        for ds, dl in zip(dataset_train, train_loader)
                    ]
                    val_loader = [
                        DataLoader(
                            _XTransformDataset(ds, x_op),
                            batch_size=getattr(dl, "batch_size", None) or data_hp.batch_size,
                            shuffle=False,
                        )
                        for ds, dl in zip(dataset_val, val_loader)
                    ]
                else:
                    train_loader = DataLoader(
                        _XTransformDataset(dataset_train, x_op),
                        batch_size=getattr(train_loader, "batch_size", None) or data_hp.batch_size,
                        shuffle=False,
                    )
                    val_loader = DataLoader(
                        _XTransformDataset(dataset_val, x_op),
                        batch_size=getattr(val_loader, "batch_size", None) or data_hp.batch_size,
                        shuffle=False,
                    )
                xcoords_applied = True

                # If we're enforcing units in internal coords, update the x-dim spec.
                if units_spec is not None:
                    try:
                        internal_x_dims = xcoords.internal_x_dims(units_spec.unit_system, units_spec.x_dims)
                        units_spec = replace(units_spec, x_dims=internal_x_dims)
                    except Exception as _e:
                        if verbose:
                            print(
                                "[Stage B] Warning: failed to update units_spec for internal x-coords:",
                                str(_e),
                            )
    except Exception as _e:
        if verbose and stageA_x_transforms:
            print("[Stage B] Warning: failed to apply Stage-A x-transforms:", str(_e))

    # Baseline: re-fit the AST version of the Stage-A model with a short LM run,
    # so that the validation loss is in the same units as lm_hp.loss_target.
    if is_multi:
        if reuses is None:
            reuses = [dict(reuse) for _ in range(len(train_loader))]
        state = _fit_candidate_root_multi(
            root=root,
            reuses=reuses,
            train_loaders=list(train_loader),
            val_loaders=list(val_loader),
            lm_hp=lm_hp,
            device=device,
            dtype=dtype,
            epochs_stageB=epochs_stageB,
            loss_scales=loss_scales
            if loss_scales is not None
            else [loss_scale for _ in range(len(train_loader))],
            fresh_nn_factory=fresh_nn_factory,
            dataset_ids=dataset_ids,
            agg_mode=agg_mode,
            agg_weights=agg_weights,
            atom_factory=atom_factory,
        )
    else:
        state = _fit_candidate_root(
            root=root,
            reuse=reuse,
            train_loader=train_loader,
            val_loader=val_loader,
            lm_hp=lm_hp,
            device=device,
            dtype=dtype,
            epochs_stageB=epochs_stageB,
            loss_scale=loss_scale,
            fresh_nn_factory=fresh_nn_factory,
            atom_factory=atom_factory,
        )
    # Ensure reuse map is initialised (and consistent with multi-dataset mode)
    if getattr(state, "models", None) is not None:
        state.reuses = [_refresh_reuse_from_state(state.root, m) for m in state.models]
        state.reuse = state.reuses[0] if len(state.reuses) > 0 else {}
        state.model = state.models[0] if len(state.models) > 0 else state.model
    else:
        state.reuse = _refresh_reuse_from_state(state.root, state.model)
    try:
        acceptance_noise_n_eff = float(_n_val) if int(_n_val) > 0 else None
    except Exception:
        acceptance_noise_n_eff = None
    state.acceptance_noise_n_eff = acceptance_noise_n_eff
    try:
        lm_hp.acceptance_noise_n_eff = acceptance_noise_n_eff
    except Exception:
        pass
    n_params0 = int(state.model.num_parameters())

    # In multi-dataset mode we use the first dataset for feature probes
    train_loader_probe = train_loader[0] if is_multi else train_loader
    val_loader_probe = val_loader[0] if is_multi else val_loader

    if verbose:
        print(
            "\n[Stage B] Initial model: params={}, val-loss={}".format(
                n_params0, _loss_str(state.val_loss, lm_hp)
            )
        )
        if (
            state.num_nn_atoms is not None
            and state.num_multivar_nn_atoms is not None
            and state.max_nn_arity is not None
        ):
            print(
                "[Stage B] Initial NN metrics: total={}, multivar={}, max_arity={}".format(
                    state.num_nn_atoms, state.num_multivar_nn_atoms, state.max_nn_arity
                )
            )

    # Early exit if initial fit is unacceptable - no point doing separability analysis
    # on garbage derivatives
    if _has_explicit_acceptance_noise_floor():
        acceptance_noise_floor_raw = float(
            resolve_acceptance_noise_floor_raw(lm_hp, loss_scale)
        )
    loss_acceptable_eff = (
        lm_hp.loss_acceptable * loss_scale
        if getattr(lm_hp, "loss_in_MAD_units", False)
        else lm_hp.loss_acceptable
    )
    loss_acceptable_eff = clamp_threshold_to_noise_floor(
        loss_acceptable_eff,
        acceptance_noise_floor_raw,
        min_factor=3.0,
    )
    state.loss_scale = float(loss_scale)
    state.loss_good_enough_eff = float(loss_good_enough_raw)
    state.loss_acceptable_eff = float(loss_acceptable_eff)
    state.acceptance_noise_floor_raw = float(acceptance_noise_floor_raw)
    state.acceptance_noise_n_eff = acceptance_noise_n_eff
    state.original_y_noise_floor_raw = float(original_y_noise_floor_raw)
    if state.val_loss > loss_acceptable_eff:
        if verbose:
            norm = state.val_loss / max(loss_scale, 1e-30) if getattr(lm_hp, "loss_in_MAD_units", False) else state.val_loss
            print(
                f"{RED}[Stage B] Y-transform REJECTED: initial fit unacceptable "
                f"(val_loss={_loss_str(state.val_loss, lm_hp)} (norm={norm:.4e}) > loss_acceptable_eff={loss_acceptable_eff:.4e}). "
                f"Skipping separability analysis.{RESET}"
            )
        # Return early with the awful state - caller can decide to retry with different transform
        state.phi_expr_str = None
        state.y_expr_str = None
        state.sympy_meta = {"accepted": False, "reason": "initial_fit_unacceptable"}
        return state

    # Discover approximate homogeneity / scaling features once, using the
    # fitted baseline model as the probe. These are used as *hints* for rewrites.
    try:
        scale_specs = discover_scaling_features(
            model=state.model,
            datagen=train_loader_probe,
            Nxvars=Nxvars,
            device=device,
            max_group_size=Nxvars,
        )
        scaling_by_axis = _scaling_index(scale_specs)
        if verbose and scale_specs:
            print("[Stage B] Found {} scaling candidates:".format(len(scale_specs)))
            for sp in scale_specs:
                if sp.compound_name:
                    print(
                        "  compound {}: k≈{:.3f}, rel_std={:.3f}, n={}".format(
                            sp.compound_name, sp.k_hat, sp.rel_std, sp.n_points
                        )
                    )
                elif len(sp.indices) == 1:
                    print(
                        "  axis {}: k≈{:.3f}, rel_std={:.3f}, n={}".format(
                            sp.indices[0], sp.k_hat, sp.rel_std, sp.n_points
                        )
                    )
                else:
                    idx_str = ",".join(str(i) for i in sp.indices)
                    print(
                        "  joint [{}]: k≈{:.3f}, rel_std={:.3f}, n={}".format(
                            idx_str, sp.k_hat, sp.rel_std, sp.n_points
                        )
                    )
    except Exception as e:
        if verbose:
            print("[Stage B] Scaling feature discovery failed:", e)
        scale_specs = []
        scaling_by_axis = {}

    # --- Discover trig-like axes ---
    # FFT-based detection (runs first, informational)
    trig_specs_fft = []
    try:
        trig_specs_fft = discover_trig_axes(
            model=state.model,
            datagen=train_loader_probe,
            Nxvars=Nxvars,
            device=device,
            strength_threshold=5.0,
        )
        if verbose and trig_specs_fft:
            print("[Stage B] FFT trig-axis candidates (informational):")
            for sp in trig_specs_fft:
                print(
                    f"[Stage B]   axis {sp.axis}: omega≈{sp.omega:.3f}, strength={sp.strength:.1f}, "
                    f"span=[{sp.tmin:.3g},{sp.tmax:.3g}]"
                )
    except Exception as e:
        if verbose:
            print(f"[Stage B] Trig-axis FFT discovery failed: {e}")

    # Oracle-based trig scaling probe (primary source)
    # Extract compound targets from the model's AST for trig probing
    compound_targets: List[TrigProbeTarget] = []
    try:
        compound_targets = _extract_nontrivial_targets_from_model(root)
        if verbose and compound_targets:
            print(f"[Stage B] Found {len(compound_targets)} nontrivial target(s) for trig probing:")
            for ct in compound_targets:
                print(f"[Stage B]   {ct.name} (kind={ct.kind}, indices={ct.indices})")
    except Exception as e:
        if verbose:
            print(f"[Stage B] Compound target extraction failed: {e}")

    trig_scale_specs = []
    oracle_trig_ran = False
    try:
        # Run oracle scaling probe on compounds first so the trig probe
        # can skip clean-polynomial compounds (prevents Taylor aliasing).
        oracle_specs_merged = list(scale_specs or [])
        if compound_targets:
            oracle_compound = probe_oracle_scaling(
                model=state.model,
                datagen=train_loader_probe,
                Nxvars=Nxvars,
                device=device,
                compound_targets=compound_targets,
            )
            oracle_specs_merged = oracle_specs_merged + oracle_compound

        trig_scale_specs = probe_trig_scaling(
            model=state.model,
            datagen=train_loader_probe,
            Nxvars=Nxvars,
            device=device,
            oracle_specs=oracle_specs_merged,  # includes compound polynomial info
            compound_targets=compound_targets,
        )
        oracle_trig_ran = True
        if verbose and trig_scale_specs:
            print(f"[Stage B] Oracle trig probe found {len(trig_scale_specs)} spec(s):")
            for ts in trig_scale_specs:
                display = ts.compound_name if ts.compound_name else f"x{ts.axis}"
                basis = str(getattr(ts, "basis_fn", "") or getattr(ts, "trig_fn", ""))
                print(
                    f"[Stage B]   {display}: {basis}(ω={ts.omega:.4g})^{ts.k_hat:.2f}, "
                    f"rel_std={ts.rel_std:.3f}"
                )
    except Exception as e:
        if verbose:
            print(f"[Stage B] Oracle trig probe failed: {e}")

    # Oracle-first: use oracle specs if available, else fall back to FFT
    if trig_scale_specs:
        trig_specs = _trig_scale_to_axis_specs(trig_scale_specs)
        if verbose:
            print(f"[Stage B] Using {len(trig_specs)} oracle trig spec(s)")
    elif oracle_trig_ran:
        # Oracle ran successfully but found no trig — trust it.
        trig_specs = []
        if verbose:
            print("[Stage B] Oracle trig probe found no trig")
    else:
        trig_specs = []
        if verbose:
            print("[Stage B] Oracle trig probe did not run; no trig specs (FFT not used in logic)")

    trig_by_axis = _trig_index(trig_specs)
    if verbose and trig_by_axis:
        print(f"[Stage B] trig_by_axis: {list(trig_by_axis.keys())}")

    # Second-stage trig structure probe: try to infer whether the dominant
    # periodicity depends on a product (x_i*x_j) or difference (x_i-x_j).
    try:
        trig_structure: List[PeriodicityStructureHint] = discover_trig_argument_structure(
            model=state.model,
            datagen=train_loader_probe,
            Nxvars=Nxvars,
            trig_specs=trig_specs,
            device=device,
        )
        trig_structure_by_axis = {int(h.axis): h for h in trig_structure}
        if verbose and trig_structure:
            print("[Stage B] Trig argument-structure hints:")
            for h in trig_structure:
                if h.kind == "none" or h.partner is None:
                    print(f"  axis {h.axis}: none")
                else:
                    print(
                        f"  axis {h.axis}: {h.kind} with partner {h.partner} "
                        f"(score={h.score:.2f}, omega_r2={h.omega_r2:.2f}, phase_r2={h.phase_r2:.2f})"
                    )
    except Exception as e:
        if verbose:
            print("[Stage B] Trig argument-structure discovery failed:", e)
        trig_structure_by_axis = {}

    # Stage-A x-preprocessing map (already applied to loaders above when possible).
    try:
        if isinstance(stageA_x_transforms, dict):
            stageA_x_transforms = {int(k): v for k, v in stageA_x_transforms.items()}
        else:
            stageA_x_transforms = {}
    except Exception:
        stageA_x_transforms = {}

    # Import rules locally to avoid circular import
    from .rule_factorized_search import RuleFactorizedSearchFallback
    from .rules import (
        RuleAdditiveLogRatio,
        RuleAdditiveGaugeTransfer,
        RuleAffineDecomposition,
        RuleBarycentricCompound,
        RuleCommonPrefactor,
        RuleCompoundFunctionMacros,
        RuleCompoundPlanck,
        RuleCounterfactorAddSplitNN,
        RuleCountertermMulSplitNN,
        RuleCoupledLeafRatio,
        RuleHomogeneityPeel,
        RuleJointProductMonomialClosure,
        RuleLastHardAtomRescue,
        RuleLastHardTrigSquare1D,
        RuleLastHardTrigPower1D,
        RuleLogExpCompound,
        RuleMonomialPeelPriority,
        RuleMonomialPrefactorCompound,
        RuleMetricDistance,
        RuleMultiplicativeHomogeneityTransfer,
        RuleMultiDNN,
        RuleNNLeafSeparability,
        RuleNonsenseUnitsZeroPrune,
        RuleNonlinearSubstitution,
        RuleOverlapCountertermPeelNN,
        RuleOverlapPrefactorPeelNN,
        RuleOuterTransformSplitNN,
        RuleInverseTrigOuterClosure,
        RuleInverseTrigRationalOuterClosure,
        RulePhaseContextTrigClosure,
        RulePhaseHintTrigClosure,
        RulePhaseHintReciprocalTrigPower,
        RulePolySplit,
        RulePowerProduct,
        RulePreconditionerFallbackNN,
        RuleProductHomogeneity,
        RuleR1OperatorCertificate,
        RuleRatioInvariance,
        RuleSubtreeSeparability,
        RuleUniNN,
        RuleUnivariateMulPeel,
        RuleUnivariateOracleInvariants,
    )

    # Create context for engine-based Stage B rewrites
    ctx = StageBContext(
        state=state,
        train_loader=train_loader,
        val_loader=val_loader,
        lm_hp=lm_hp,
        device=device,
        dtype=dtype,
        epochs_stageB=epochs_stageB,
        loss_scale=loss_scale,
        loss_scales=loss_scales,
        dataset_ids=dataset_ids,
        agg_mode=agg_mode,
        agg_weights=agg_weights,
        loss_good_enough_raw=loss_good_enough_raw,
        score_tol=score_tol,
        scale_specs=scale_specs,
        scaling_by_axis=scaling_by_axis,
        trig_by_axis=trig_by_axis,
        phase_hints=list(phase_hints or []),
        phase_context_hints=list(phase_context_hints or []),
        outer_link_hints=list(outer_link_hints or []),
        stageA_x_transforms=stageA_x_transforms,
        xcoords=xcoords,
        xcoords_applied=xcoords_applied,
        trig_structure_by_axis=trig_structure_by_axis,
        verbose=verbose,
        fresh_nn_factory=fresh_nn_factory,
        atom_factory=atom_factory,
        disabled_patterns=set(disabled_patterns),
        enabled_patterns=[],
        y_op=y_op,
        y_op_inv=y_op_inv,
        y_transform_name=str(y_transform_name or "identity"),
        units_spec=units_spec,
        enforce_units=enforce_units,
        verbose_separabilities=verbose_separabilities,
        acceptance_noise_n_eff=acceptance_noise_n_eff,
    )

    state = _apply_additive_gauge_fix_to_state_reuse(
        root=root,
        state=state,
        train_loader_probe=train_loader_probe,
        device=device,
        dtype=dtype,
        cancel_ratio_thresh=8.0,
        log_fn=ctx.log if hasattr(ctx, "log") else print,
    )

    # Create engine with rule pipeline.  The list below is the registration
    # list; StageBEngine partitions and reranks it before execution.  In
    # particular, overlap/addition-aware separability rules run before
    # leaf-local coordinate-compression rules such as homogeneity_peel, so a
    # shared-variable additive gauge gets a chance to resolve first.
    _factorized_search_rule = RuleFactorizedSearchFallback(factorized_search_hp=factorized_search_hp) if use_factorized_search else None

    engine = StageBEngine(
        [
            RuleNonsenseUnitsZeroPrune(),
            RuleMonomialPeelPriority(),
            # Cleanup pass: prune tiny additive poly blocks first (especially after counterterm splits)
            RuleMultiDNN(factorized_search_rule=_factorized_search_rule),
            # Homogeneity peel: extract x^k factor from homogeneous degree-k functions
            # NN(xi, xj) → xi^k * NN(xj/xi) when xi·∂f/∂xi + xj·∂f/∂xj = k·f
            RuleHomogeneityPeel(),
            # Product-homogeneity: extract x^k factor when residual depends on product
            # NN(xi, xj) → xi^k * NN(xi*xj) when xi·∂f/∂xi - xj·∂f/∂xj = k·f
            RuleProductHomogeneity(),
            # Ratio-invariance: collapse bivariate NN(xi,xj) to univariate (poly(xj/xi))^exponent
            # when the NN depends only on the ratio (homogeneous degree-0)
            RuleRatioInvariance(),
            # Coupled-leaf ratio: factor out common structure when F/G is a simple polynomial
            RuleCoupledLeafRatio(),
            # Compound-function macros: high-payoff motifs (sinc, sin-ratio, sqrt1±, etc.)
            # screened cheaply before LM.
            RuleCompoundFunctionMacros(),
            # Metric-distance closures: law-of-cosines and coordinate-difference
            # distance forms, kept visible and validation-gated.
            RuleMetricDistance(),
            # Direct visible periodic closures from Stage-0 phase hints.
            # This does not require accepting an intermediate NN[z] surrogate.
            RuleInverseTrigOuterClosure(),
            RuleInverseTrigRationalOuterClosure(),
            RulePhaseHintTrigClosure(),
            RulePhaseHintReciprocalTrigPower(),
            RulePhaseContextTrigClosure(),
            # Weighted-average / barycentric closures, kept visible and
            # validation-gated.
            RuleBarycentricCompound(),
            # Dimensionless log/exp compound closures, kept visible and
            # validation-gated.
            RuleLogExpCompound(),
            # Monomial compound (log-deriv) + monomial prefactor peel → analytic Planck leaf.
            # This targets AIFeynman-style expressions like x0^3*x2/(exp(c*x0*x2/(x1*x3)) - 1).
            RuleMonomialPrefactorCompound(),
            # Planck templates for compound atoms with extra vars (e.g., NN[z, x0, x1]).
            # Fits y ≈ z * A * w^p / (exp(α*w) - 1) where w = (x0*x1)/z.
            RuleCompoundPlanck(),
            # Nonlinear substitution: detect if T(v) for T in {cos,sin,exp,log}
            # renders an NN leaf rational. Cheap parity+SVD screening before LM.
            RuleNonlinearSubstitution(),
            # Affine decomposition: g(f(z,w)) = a(z) + b(z)*h(w)
            # reduces 2D atom to two 1D atoms
            RuleAffineDecomposition(),
            RulePolySplit(),
            RuleSubtreeSeparability(),
            RuleUnivariateOracleInvariants(),
            RuleR1OperatorCertificate(),
            # Final low-arity atom rescue: only activates when exactly one NN remains,
            # units are enforced, and the atom output dimension is known.
            RuleLastHardTrigSquare1D(),
            RuleLastHardTrigPower1D(),
            RuleLastHardAtomRescue(),
            RuleUniNN(factorized_search_rule=_factorized_search_rule),
            # Univariate multiplicative decomposition: detect exp*trig products via log-derivative analysis
            RuleUnivariateMulPeel(),
            # Log-ratio: detect additive pairs of univariate NN atoms forming log(x_i) - log(x_j) patterns
            RuleAdditiveLogRatio(),
            # Power-product probe: detect c * x_i^a * x_j^b via log-linear regression
            # (must run before nn_leaf_separability to catch clean power laws directly)
            RulePowerProduct(),
            # Joint product monomial closure: resolve retained-axis gauges where
            # a product of NN leaves is already a visible raw monomial.
            RuleJointProductMonomialClosure(),
            # Overlap-prefactor peel: factor a shared singleton variable out of one
            # side of a direct NN+NN sum when that side is multiplicatively separable.
            RuleOverlapPrefactorPeelNN(),
            # Overlap-counterterm peel: peel an additive singleton counterterm out of
            # one side of a direct NN*NN product when that side is additively separable.
            RuleOverlapCountertermPeelNN(),
            # Common prefactor: factor shared multiplicative structure from AddNode siblings
            # e.g. P*(1 + Q*cos(θ)) patterns in orbital/scattering physics
            RuleCommonPrefactor(),
            # Additive gauge transfer: expose NN(u,s) as NN(u)+h(s) only when
            # the resulting AST visibly improves the unresolved additive scope.
            RuleAdditiveGaugeTransfer(),
            # Multiplicative homogeneous-gauge transfer: try the reciprocal
            # ratio representative and require an immediate visible analytic
            # closure before generic 1D approximants get a chance to commit.
            RuleMultiplicativeHomogeneityTransfer(),
            # Fallback target-local monomial pass. The global priority pass
            # above handles obvious peels first; this keeps compound-only
            # monomial cases reachable after gauge/overlap rules have spoken.
            RuleUniNN(monomial_only=True),
            # Separability check on NN leaves: try additive/multiplicative separability BEFORE counterterm splits
            RuleNNLeafSeparability(),
            # Counterterm multiplicative split: fallback for NN leaves that aren't fully separable
            RuleCountertermMulSplitNN(),
            # Counterfactor additive split (v2): peel multiplicative prefactors so the quotient becomes additively separable
            RuleCounterfactorAddSplitNN(),
            RuleOuterTransformSplitNN(),
            RulePreconditionerFallbackNN(),
        ] + (extra_rules or [])
    )

    # Capture baseline y-MSE before engine runs for post-run asinh sanity check (Strategy B)
    is_asinh = getattr(lm_hp, "fit_y_link", None) == "asinh"
    base_y_mse_for_postcheck = None
    if is_asinh:
        try:
            base_y_mse_for_postcheck = float(_eval_val_mse(state.model, val_loader_probe, device))
        except Exception:
            pass

    # Run engine: iteratively apply rules until no improvement or max iterations reached
    state = engine.run(ctx, max_outer_iters=max_outer_iters, max_backtracks=max_backtracks)
    state.loss_scale = float(loss_scale)
    state.loss_good_enough_eff = float(loss_good_enough_raw)
    state.loss_acceptable_eff = float(loss_acceptable_eff)
    state.acceptance_noise_floor_raw = float(acceptance_noise_floor_raw)
    state.acceptance_noise_n_eff = acceptance_noise_n_eff

    # Hard sanity check: recompute validation MSE and ensure consistency.
    if getattr(state, "models", None) is not None and isinstance(val_loader, (list, tuple)):
        mses = [float(_eval_val_mse(m, dl, device)) for m, dl in zip(state.models, val_loader)]
        mse = float(sum(mses) / max(1, len(mses)))
        state.val_losses = mses
    else:
        mse = float(_eval_val_mse(state.model, val_loader_probe, device))
        mses = None

    if not math.isfinite(mse):
        raise RuntimeError("Stage B val MSE is not finite")

    if is_asinh:
        # Under asinh fit-link, the stored val_loss is in asinh-space while mse is y-space.
        # The mismatch is expected; don't treat it as a consistency bug.
        # Instead, apply the same two-gate sanity check as engine.py's should_accept().
        asinh_loss = float(state.val_loss)
        y_mse = float(mse)

        s = float(getattr(lm_hp, "fit_y_link_scale", 1.0))
        q = float(getattr(lm_hp, "asinh_yspace_sanity_quantile", 0.90))
        alpha = float(getattr(lm_hp, "asinh_yspace_sanity_factor", 20.0))
        beta = float(getattr(lm_hp, "asinh_yspace_regress_factor", 5.0))

        # Strategy A: Correct Jacobian scaling via D_ref = quantile(s² + y²)
        D_ref = _asinh_yspace_scale_from_loader(val_loader_probe, device, s, q)
        y_mse_allowed_A = alpha * max(asinh_loss, 1e-30) * max(D_ref, 1e-30)

        # Strategy B: Baseline-relative guard (don't regress too far from initial)
        y_mse_allowed_B = beta * max(base_y_mse_for_postcheck or 1e-30, 1e-30)

        # Combined: pass if either strategy allows it
        y_mse_allowed = max(y_mse_allowed_A, y_mse_allowed_B)

        if math.isfinite(y_mse) and math.isfinite(y_mse_allowed) and y_mse > y_mse_allowed:
            print(
                f"{RED}[Stage B] REJECTING: y-space MSE {y_mse:.6g} > allowed {y_mse_allowed:.6g} "
                f"(asinh={asinh_loss:.6g}, D_ref={D_ref:.3g}, base_y_mse={base_y_mse_for_postcheck:.6g}){RESET}"
            )
            state.sympy_meta = {"accepted": False, "reason": "asinh_yspace_mse_sanity_failed"}
            state.phi_expr_str = None
            state.y_expr_str = None
            state.val_loss = y_mse  # Use y-space MSE for reporting
            return state

        # Store both metrics for reporting/debug
        if state.sympy_meta is None:
            state.sympy_meta = {}
        state.sympy_meta.update({
            "asinh_val_loss": asinh_loss,
            "y_mse": y_mse,
            "D_ref": D_ref,
            "y_over_asinhD": y_mse / max(asinh_loss * D_ref, 1e-30),
        })

        # Use y-space MSE as the final val_loss for downstream reporting
        state.val_loss = y_mse
    else:
        # Non-asinh: the old consistency check can remain
        if abs(mse - float(state.val_loss)) > 1e-6 * max(1.0, abs(mse), abs(float(state.val_loss))):
            print(f"[Stage B] WARNING: stored val_loss={state.val_loss:.6g} but recomputed MSE={mse:.6g}")
        state.val_loss = mse

    # Transfer enabled patterns tracking to state for report generation
    enabled_patterns = ctx.enabled_patterns
    state.enabled_patterns = enabled_patterns

    from .pruning import run_stageb_pruning_pipeline

    pre_prune_val_loss = float(state.val_loss)
    pre_prune_params = int(state.model.num_parameters())

    state = run_stageb_pruning_pipeline(
        state=state,
        train_loader=train_loader_probe,
        val_loader=val_loader_probe,
        lm_hp=lm_hp,
        device=device,
        dtype=dtype,
        loss_scale=loss_scale,
        atom_factory=atom_factory,
        verbose=verbose,
    )

    if is_asinh and y_op_inv is None:
        try:
            if getattr(state, "models", None) is not None and isinstance(val_loader, (list, tuple)):
                y_mses = [float(_eval_val_mse(m, dl, device)) for m, dl in zip(state.models, val_loader)]
                original_y_mse = float(sum(y_mses) / max(1, len(y_mses)))
            else:
                original_y_mse = float(_eval_val_mse(state.model, val_loader_probe, device))
            raw_loss_scale = 1.0
            if getattr(lm_hp, "loss_in_MAD_units", False):
                _, raw_y_mad = _compute_y_med_mad_from_loader(
                    val_loader_probe,
                    device,
                    fit_link=None,
                    fit_link_scale=1.0,
                )
                if raw_y_mad is not None and math.isfinite(raw_y_mad) and raw_y_mad > 0.0:
                    raw_loss_scale = float(raw_y_mad) ** 2
            original_good_eff = float(lm_hp.loss_target) * float(raw_loss_scale)
            original_acceptable_eff = float(lm_hp.loss_acceptable) * float(raw_loss_scale)
            original_acceptable_eff = clamp_threshold_to_noise_floor(
                original_acceptable_eff,
                original_y_noise_floor_raw,
                min_factor=3.0,
            )
            state.original_y_val_loss = float(original_y_mse)
            state.original_y_loss_good_enough_eff = float(original_good_eff)
            state.original_y_loss_acceptable_eff = float(original_acceptable_eff)
            state.original_y_noise_floor_raw = float(original_y_noise_floor_raw)
            state.val_loss = float(original_y_mse)
            if state.sympy_meta is None:
                state.sympy_meta = {}
            state.sympy_meta.update({
                "original_y_val_loss": float(original_y_mse),
                "original_y_loss_good_enough_eff": float(original_good_eff),
                "original_y_loss_acceptable_eff": float(original_acceptable_eff),
                "original_y_noise_floor_raw": float(original_y_noise_floor_raw),
            })
            if verbose:
                print(
                    "[Stage B] Original-y validation (fit-link): "
                    f"mse={original_y_mse:.4e}, "
                    f"good_enough={original_good_eff:.4e}, "
                    f"acceptable={original_acceptable_eff:.4e}"
                )
        except Exception as e:
            if verbose:
                print(f"[Stage B] Original-y validation (fit-link) failed: {e}")

    if y_op_inv is not None:
        try:
            original_y_mse = _eval_original_y_mse_with_inverse(
                state.model,
                val_loader_probe,
                device,
                y_op_inv,
            )
            raw_loss_scale = 1.0
            if getattr(lm_hp, "loss_in_MAD_units", False):
                raw_mad2 = _compute_original_y_mad2_with_inverse(
                    val_loader_probe,
                    device,
                    y_op_inv,
                )
                if math.isfinite(raw_mad2) and raw_mad2 > 0.0:
                    raw_loss_scale = float(raw_mad2)
            original_good_eff = float(lm_hp.loss_target) * float(raw_loss_scale)
            original_acceptable_eff = float(lm_hp.loss_acceptable) * float(raw_loss_scale)
            original_acceptable_eff = clamp_threshold_to_noise_floor(
                original_acceptable_eff,
                original_y_noise_floor_raw,
                min_factor=3.0,
            )
            state.original_y_val_loss = float(original_y_mse)
            state.original_y_loss_good_enough_eff = float(original_good_eff)
            state.original_y_loss_acceptable_eff = float(original_acceptable_eff)
            state.original_y_noise_floor_raw = float(original_y_noise_floor_raw)
            if verbose:
                print(
                    "[Stage B] Original-y validation: "
                    f"mse={original_y_mse:.4e}, "
                    f"good_enough={original_good_eff:.4e}, "
                    f"acceptable={original_acceptable_eff:.4e}"
                )
        except Exception as e:
            if verbose:
                print(f"[Stage B] Original-y validation check failed: {e}")

    # Record pruning step in simplification path if state changed
    post_prune_params = int(state.model.num_parameters())
    if pre_prune_val_loss != float(state.val_loss) or pre_prune_params != post_prune_params:
        from nestynet_sr.sr_core.bridges import ast_to_human_readable
        from ..model_selection import ast_cost_physics_prior

        try:
            _prune_ast_cost = float(ast_cost_physics_prior(state.root))
        except Exception:
            _prune_ast_cost = None
        if not hasattr(state, 'simplification_path'):
            state.simplification_path = []
        state.simplification_path.append({
            "step": len(state.simplification_path),
            "stage": "prune",
            "action": "parameter pruning",
            "expression": ast_to_human_readable(state.root),
            "val_loss": float(state.val_loss),
            "mse_raw": float(state.val_loss),
            "mse_eff": None,
            "base_loss": pre_prune_val_loss,
            "threshold": None,
            "n_params": post_prune_params,
            "ast_cost": _prune_ast_cost,
            "detail": f"params {pre_prune_params} -> {post_prune_params}",
        })

    # Diagnostic: sensitivity to shuffling selected axes (runs before pretty-print).
    shuffle_diag = _shuffle_axis_sensitivity(
        model=state.model,
        val_loader=val_loader_probe,
        device=device,
        axes=(2, 3),
        verbose=verbose,
    )
    if shuffle_diag is not None:
        state.shuffle_diag = shuffle_diag

    phi_diag = _phi_pred_error_from_loader(
        model=state.model,
        val_loader=val_loader_probe,
        device=device,
        y_is_phi=(y_op is not None),
        verbose=verbose,
    )
    if phi_diag is not None:
        state.phi_diag = phi_diag

    batch_stats = _print_val_batch_stats(
        model=state.model,
        val_loader=val_loader_probe,
        device=device,
        verbose=verbose,
    )
    if batch_stats is not None:
        state.batch_stats = batch_stats

    problem_leaves = _annotate_nonsense_units_leaves(
        state,
        units_spec=units_spec,
        enforce_units=bool(enforce_units),
        log_fn=print,
    )
    resolved_problem_prunes = {
        str(rec.get("target_uid"))
        for rec in getattr(ctx, "decision_log", [])
        if rec.get("outcome") == "accept"
        and rec.get("label") == "nonsense_units_zero_prune"
        and rec.get("target_uid") is not None
    }
    if problem_leaves:
        print(
            f"{RED}[Stage B] Problem summary: {len(problem_leaves)} unresolved "
            f"'nonsense_units' leaf/leaves remain. Continuing to report the final AST, "
            f"but skipping Stage C symbolic simplification.{RESET}"
        )
        for rec in problem_leaves:
            tag_s = f"#{rec['tag']}" if rec.get("tag") else ""
            inputs_s = ", ".join(rec.get("inputs", [])) if rec.get("inputs") else "no inputs"
            basis_s = ", ".join(rec.get("basis_dims", [])) if rec.get("basis_dims") else "dimless-only"
            print(
                f"{RED}[Stage B]   nonsense_units{tag_s}: inputs=({inputs_s}) "
                f"target={rec.get('target_dim')} basis=[{basis_s}]{RESET}"
            )
    elif resolved_problem_prunes:
        print(
            f"{GREEN}[Stage B] Problem summary: resolved "
            f"{len(resolved_problem_prunes)} 'nonsense_units' leaf/leaves via zero-prune; "
            f"final AST has no unresolved problem leaves.{RESET}"
        )

    unresolved_nn_leaves = _unresolved_nn_leaf_records(state.root)
    if unresolved_nn_leaves:
        print(
            f"{RED}[Stage B] Final AST still contains {len(unresolved_nn_leaves)} "
            f"trainable NN leaf/leaves; Stage C expression will be marked unresolved "
            f"and skipped by truth-eval/final-polish.{RESET}"
        )

    # Compute expression strings (always, regardless of verbose)
    phi_expr_str: Optional[str] = None
    y_expr_str: Optional[str] = None
    sympy_meta: Optional[dict] = None
    coefficient_metadata_by_dataset = None
    coefficient_metadata_group_error = None

    metadata_models = list(getattr(state, "models", None) or [])
    dataset_labels = list(getattr(state, "dataset_ids", None) or [])
    if metadata_models:
        coefficient_metadata_by_dataset = []
        metadata_dataset_ids = []
        for dataset_index, metadata_model in enumerate(metadata_models):
            dataset_id = (
                dataset_labels[dataset_index]
                if dataset_index < len(dataset_labels)
                else f"dataset_{dataset_index}"
            )
            metadata_dataset_ids.append(str(dataset_id))
            coefficient_metadata_by_dataset.append(
                collect_coefficient_metadata(
                    state.root,
                    metadata_model,
                    units_spec,
                    dataset_id=dataset_id,
                    dataset_index=dataset_index,
                )
            )
        try:
            coefficient_metadata_by_dataset = normalize_coefficient_metadata_by_dataset(
                coefficient_metadata_by_dataset,
                units_spec=units_spec,
                expected_count=len(metadata_models),
                expected_dataset_ids=metadata_dataset_ids,
            )
            coefficient_metadata = coefficient_metadata_by_dataset[0]
        except CoefficientMetadataError as exc:
            coefficient_metadata_group_error = {
                "code": exc.code,
                "reason": exc.reason,
            }
            coefficient_metadata = dict(coefficient_metadata_by_dataset[0])
            coefficient_metadata.update(
                {
                    "valid": False,
                    "code": exc.code,
                    "reason": exc.reason,
                }
            )
    else:
        coefficient_metadata = collect_coefficient_metadata(
            state.root,
            state.model,
            units_spec,
            dataset_id=dataset_labels[0] if dataset_labels else None,
            dataset_index=0 if dataset_labels else None,
        )
    if verbose and coefficient_metadata.get("valid") is not True:
        print(
            "[Stage C] Coefficient metadata is invalid: "
            f"{coefficient_metadata.get('code')}: "
            f"{coefficient_metadata.get('reason')}"
        )

    try:
        # Use higher precision here so the string->SymPy->lambdify roundtrip
        # stays numerically faithful to the trained model. Low-precision
        # printing can inflate Stage-C's baseline error, forcing large
        # acceptance tolerances and yielding "peculiar" simplifications.
        expr_str = pretty_print_state(state, sig=16)
        phi_expr_str = expr_str
        if verbose:
            print(f"[Stage C] pretty_print_state FULL expr: {expr_str}")

        # Optional SymPy-based post-simplification, gated on SymPy availability.
        if unresolved_nn_leaves:
            sympy_meta = {
                "accepted": False,
                "parse_success": True,
                "reason": "unresolved_nn_atoms_present",
                "kind": "unresolved_symbolic_leaves",
                "num_nn_atoms": len(unresolved_nn_leaves),
                "unresolved_nn_leaves": unresolved_nn_leaves,
                "coefficient_metadata": coefficient_metadata,
            }
            if problem_leaves:
                sympy_meta["problem_leaves"] = problem_leaves
        elif problem_leaves:
            sympy_meta = {
                "accepted": False,
                "reason": "problem_leaves_present",
                "problem_leaves": problem_leaves,
                "coefficient_metadata": coefficient_metadata,
            }
        elif _HAVE_SYMPY:
            try:
                use_guard = bool(getattr(lm_hp, "stagec_sympy_subprocess", True))
                sympy_fn = (
                    guarded_sympy_simplify_expression
                    if use_guard
                    else _sympy_simplify_expression
                )
                if use_guard and verbose:
                    print(
                        "[Stage C] Running SymPy simplification in guarded worker "
                        f"(max_seconds={float(getattr(lm_hp, 'stagec_sympy_max_seconds', 300.0)):.1f}, "
                        f"mem_fraction={float(getattr(lm_hp, 'stagec_sympy_mem_fraction', 0.20)):.3g})."
                    )
                if use_guard:
                    phi_sym, y_sym, meta = sympy_fn(
                        expr_str,
                        model=state.model,
                        val_loader=val_loader_probe,
                        device=device,
                        Nxvars=Nxvars,
                        y_op_inv=y_op_inv,
                        noise_floor_raw=acceptance_noise_floor_raw,
                        units_spec=units_spec if enforce_units else None,
                        coefficient_metadata=coefficient_metadata,
                        verbose=verbose,
                        max_seconds=float(
                            getattr(lm_hp, "stagec_sympy_max_seconds", 300.0)
                        ),
                        mem_fraction=float(
                            getattr(lm_hp, "stagec_sympy_mem_fraction", 0.20)
                        ),
                    )
                else:
                    phi_sym, y_sym, meta = sympy_fn(
                        expr_str,
                        model=state.model,
                        val_loader=val_loader_probe,
                        device=device,
                        Nxvars=Nxvars,
                        y_op_inv=y_op_inv,
                        noise_floor_raw=acceptance_noise_floor_raw,
                        units_spec=units_spec if enforce_units else None,
                        coefficient_metadata=coefficient_metadata,
                        verbose=verbose,
                    )
                if phi_sym is not None:
                    phi_expr_str = phi_sym
                    if y_sym is not None:
                        y_expr_str = y_sym
                if meta is not None:
                    sympy_meta = meta
            except Exception as e_sym:
                if verbose:
                    print("[Stage C] SymPy simplification failed with error:", e_sym)
                sympy_meta = {
                    "accepted": False,
                    "parse_success": False,
                    "error": str(e_sym),
                    "coefficient_metadata": coefficient_metadata,
                }
    except Exception as e:
        if verbose:
            print("[Stage B] Pretty-print failed with error:", e)
        phi_expr_str = None
        y_expr_str = None
        sympy_meta = {
            "accepted": False,
            "parse_success": False,
            "error": str(e),
            "coefficient_metadata": coefficient_metadata,
        }

    if sympy_meta is None:
        sympy_meta = {"coefficient_metadata": coefficient_metadata}
    else:
        sympy_meta = dict(sympy_meta)
        sympy_meta["coefficient_metadata"] = coefficient_metadata
    if coefficient_metadata_by_dataset is not None:
        sympy_meta["coefficient_metadata_by_dataset"] = (
            coefficient_metadata_by_dataset
        )
    if coefficient_metadata_group_error is not None:
        sympy_meta["coefficient_metadata_group_error"] = (
            coefficient_metadata_group_error
        )

    # Fallback: if SymPy didn't produce y_expr_str, compute it from phi_expr_str
    # by wrapping with the inverse y-transform
    if y_expr_str is None and phi_expr_str is not None:
        if y_op_inv is not None:
            y_expr_str = _wrap_with_inverse_transform(phi_expr_str, y_op_inv)
        else:
            y_expr_str = phi_expr_str

    sympy_meta = _with_stagec_unit_certificates(
        sympy_meta,
        phi_expr_str=phi_expr_str,
        y_expr_str=y_expr_str,
        Nxvars=Nxvars,
        units_spec=units_spec,
        enforce_units=bool(enforce_units),
        coordinate_space=(
            "raw"
            if xcoords is None
            or not hasattr(xcoords, "is_identity")
            or xcoords.is_identity()
            else "internal"
        ),
    )

    # Store in state
    state.phi_expr_str = phi_expr_str
    state.y_expr_str = y_expr_str

    # If Stage B was run in internal coordinates (xcoords applied to the loaders),
    # rewrite the final expression back into raw-coordinate form for reporting.
    phi_expr_raw_str = None
    y_expr_raw_str = None
    raw_coordinate_rewrite_required = bool(
        xcoords is not None
        and hasattr(xcoords, "is_identity")
        and not xcoords.is_identity()
    )
    if (
        phi_expr_str is not None
        and raw_coordinate_rewrite_required
        and _HAVE_SYMPY
        and not (
            isinstance(sympy_meta, dict)
            and bool(sympy_meta.get("guarded_subprocess"))
            and not bool(sympy_meta.get("accepted", False))
        )
    ):
        try:
            import sympy as sp

            phi_sym = sp.sympify(phi_expr_str, locals={"pi": sp.pi, "E": sp.E})
            phi_raw_sym = xcoords.sympy_rewrite_internal_expr_to_raw(
                phi_sym, const_mode="number"
            )
            phi_expr_raw_str = str(phi_raw_sym)
            if y_op_inv is not None:
                y_expr_raw_str = _wrap_with_inverse_transform(
                    phi_expr_raw_str, y_op_inv, simplify=False
                )
            else:
                y_expr_raw_str = phi_expr_raw_str
        except Exception as e:
            if verbose:
                print("[Stage B] Raw-x rewrite failed with error:", e)
    if y_expr_raw_str is None and phi_expr_raw_str is not None and y_op_inv is None:
        y_expr_raw_str = phi_expr_raw_str

    if raw_coordinate_rewrite_required:
        sympy_meta = _with_stagec_unit_certificates(
            sympy_meta,
            phi_expr_str=phi_expr_raw_str,
            y_expr_str=y_expr_raw_str,
            Nxvars=Nxvars,
            units_spec=raw_units_spec,
            enforce_units=bool(enforce_units),
            coordinate_space="raw",
        )

    # Record Stage C only after the final reported coordinate form is certified.
    if sympy_meta and sympy_meta.get("accepted") and phi_expr_str is not None:
        if not hasattr(state, "simplification_path"):
            state.simplification_path = []
        state.simplification_path.append(
            {
                "step": len(state.simplification_path),
                "stage": "C",
                "action": "SymPy simplification",
                "expression": phi_expr_raw_str or phi_expr_str,
                "val_loss": float(state.val_loss),
                "mse_raw": float(state.val_loss),
                "mse_eff": None,
                "base_loss": float(state.val_loss),
                "threshold": None,
                "n_params": int(state.model.num_parameters()),
                "ast_cost": None,
                "detail": (
                    f"ops={sympy_meta.get('ops')}, "
                    f"max_err={sympy_meta.get('max_err', '?')}, "
                    f"units_ok={sympy_meta.get('units_ok')}"
                ),
            }
        )

    state.x_transform_map = stageA_x_transforms
    state.phi_expr_raw_str = phi_expr_raw_str
    state.y_expr_raw_str = y_expr_raw_str
    state.coefficient_metadata = coefficient_metadata
    state.coefficient_metadata_by_dataset = coefficient_metadata_by_dataset
    state.sympy_meta = sympy_meta
    state.enabled_patterns = enabled_patterns
    state.decision_log = ctx.decision_log
    state.coe_stageB_dry_run_log = list(getattr(ctx, "coe_stageB_dry_run_log", []) or [])
    state.coe_stageB_gate_log = list(getattr(ctx, "coe_stageB_gate_log", []) or [])
    state.loss_scale = float(loss_scale)
    state.loss_good_enough_eff = float(loss_good_enough_raw)
    state.loss_acceptable_eff = float(loss_acceptable_eff)
    state.acceptance_noise_floor_raw = float(acceptance_noise_floor_raw)
    state.acceptance_noise_n_eff = acceptance_noise_n_eff
    state.original_y_noise_floor_raw = float(original_y_noise_floor_raw)

    if verbose:
        print(
            "[Stage B] Final model: params={}, val-loss={}".format(
                state.model.num_parameters(), _loss_str(state.val_loss, lm_hp)
            )
        )
        try:
            mse, rms = _eval_mse_and_rms(state.model, val_loader_probe, device)
            print(f"[Stage B] recomputed final MSE: {mse:.6g} RMS: {rms:.6g}")
            print(f"[Stage B] stored val_loss: {_loss_str(state.val_loss, lm_hp)}")
        except Exception as e:
            print(f"[Stage B] recomputed MSE failed: {e}")
        if (
            state.num_nn_atoms is not None
            and state.num_multivar_nn_atoms is not None
            and state.max_nn_arity is not None
        ):
            print(
                "[Stage B] Final NN metrics: total={}, multivar={}, max_arity={}".format(
                    state.num_nn_atoms, state.num_multivar_nn_atoms, state.max_nn_arity
                )
            )
        if enabled_patterns:
            print(f"[Stage B] {GREEN}Accepted{RESET} patterns: {', '.join(enabled_patterns)}")
            if phi_expr_str is not None:
                print(f"[Stage B] After pruning/simplification: {phi_expr_str}")
        print(f"{CYAN}[Stage B] Final AST:{RESET}", state.root)

        # This is the expression for the model as it is actually
        # trained in Stage B, i.e. in φ(y)-space.
        if phi_expr_str is not None:
            print(f"{CYAN}[Stage B] Final expression (in φ(y)-space):{RESET}", phi_expr_str)

            if phi_expr_raw_str is not None:
                print(
                    f"{CYAN}[Stage B] Final expression (in φ(y)-space, raw x):{RESET}",
                    phi_expr_raw_str,
                )

            # If we know the inverse y-transform, also report the expression in the
            # original y-space. Prefer the SymPy-simplified y-expression when available.
            if y_op_inv is not None:
                if y_expr_str is not None:
                    print(
                        f"{CYAN}[Stage B] Final expression (in original y-space):{RESET}",
                        y_expr_str,
                    )

                    if y_expr_raw_str is not None:
                        print(
                            f"{CYAN}[Stage B] Final expression (in original y-space, raw x):{RESET}",
                            y_expr_raw_str,
                        )
                else:
                    inv_name = getattr(y_op_inv, "__name__", str(y_op_inv))
                    print(
                        f"{CYAN}[Stage B] Final expression (in original y-space):{RESET} {inv_name}({phi_expr_str})"
                    )
            else:
                print(
                    f"{CYAN}[Stage B] Final expression (in original y-space):{RESET}", phi_expr_str
                )
        try:
            pl = summarize_global_power_law(
                state.model,
                val_loader_probe,
                Nxvars,
                device=device,
            )
            if pl is not None:
                C, ks, rms_abs, rms_rel = pl
                # Basic power‑law summary using the raw fitted exponents.
                core = _format_monomial_from_exponents(ks, zero_tol=1e-9)
                lhs = "φ(y)" if y_op_inv is not None else "y"
                print(f"{CYAN}[Stage C] Power-law summary: {lhs} ≈ {C:.6g} * {core}{RESET}")
                print(f"{CYAN}[Stage C]   rms_abs={rms_abs:.3e}, rms_rel={rms_rel:.3e}{RESET}")
                if y_op_inv is not None:
                    phi_str = f"{C:.6g} * {core}"
                    y_str = _wrap_with_inverse_transform(phi_str, y_op_inv, simplify=False)
                    if y_str:
                        print(f"{CYAN}[Stage C] Power-law (in original y-space): y ≈ {y_str}{RESET}")

                # If the fit is extremely good and all exponents are close to
                # half‑integers (including integers), emit a snapped monomial
                # candidate as a potential closed‑form.
                rms_rel_tol = 1e-4
                half_tol = 5e-3
                zero_tol = 1e-6

                snapped: List[float] = []
                monomial_ok = True
                for k in ks:
                    if abs(k) < zero_tol:
                        snapped.append(0.0)
                        continue
                    k_snap = _snap_exponent_to_half_integer(k, half_tol=half_tol)
                    if k_snap is None:
                        monomial_ok = False
                        break
                    snapped.append(k_snap)

                if monomial_ok:
                    # Require at least one genuinely non‑zero exponent.
                    if all(abs(k) < zero_tol for k in snapped):
                        monomial_ok = False
                    # And require a very clean global fit.
                    if rms_rel > rms_rel_tol:
                        monomial_ok = False

                if monomial_ok:
                    core_snapped = _format_monomial_from_exponents(snapped, zero_tol=zero_tol)
                    lhs = "φ(y)" if y_op_inv is not None else "y"
                    print(
                        f"{CYAN}[Stage C] Monomial candidate (snapped exponents): "
                        f"{lhs} ≈ {C:.6g} * {core_snapped}{RESET}"
                    )
                    if y_op_inv is not None:
                        phi_str = f"{C:.6g} * {core_snapped}"
                        y_str = _wrap_with_inverse_transform(phi_str, y_op_inv, simplify=False)
                        if y_str:
                            print(f"{CYAN}[Stage C] Monomial (in original y-space): y ≈ {y_str}{RESET}")
        except Exception as e:
            print("[Stage C] Power-law summary failed with error:", e)

    return state


# Subtrees for local Stage A separability (composite NN+analytic blobs)
