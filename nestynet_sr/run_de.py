#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
DE Discovery from Surrogate Neural Networks

This script discovers differential equations from data by:
1. Training a surrogate neural network u(x)
2. Using STLSQ to find sparse DE coefficients from the surrogate's derivatives
3. Supporting both 1st-order and 2nd-order DEs

Usage:
    python run_de.py --filepath data/pb000_I_6_2a_data.csv
    python run_de.py --filepath data/pb001_I_6_2_data.csv --order_candidates 2 --include_xdu
"""

import argparse
import ast
import copy
import json
import math
import os
import pathlib
import re
import sys
import timeit
from dataclasses import replace
from itertools import combinations
from typing import Any, Mapping

import numpy as np
import torch
import nestynet_sr.sr_de.de_search as de_search_mod
from nestynet_sr.sr_expr_ir.config import add_expr_ir_cli_args, apply_expr_ir_args_to_obj, expr_ir_arg_items

# ------------------------------------------------------------------
# Ensure the local checkout takes precedence when running this file as
# a script (e.g. `python nestynet_sr/run_de.py`). In that invocation,
# `sys.path[0]` is the *package directory* itself, so `import nestynet_sr`
# would otherwise resolve to a site-packages installation if present.
# ------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_this_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from nestynet_sr.sr_core import build_initial_ast
from nestynet_sr.sr_de import (
    FactorizedSearchDERescueConfig,
    FactorizedSearchDEResult,
    DESearchConfig,
    DESearchResult,
    DESearchResultMulti,
    FactorizedDERescueConfig,
    FactorizedDEResult,
    DELadderPolicy,
    LegacyDEResultPayloads,
    build_de_candidate_eval_report,
    run_legacy_de_ladder,
    factorized_search_report_shortlist,
    build_factorized_search_de_feature_groups_from_surrogate,
    build_factorized_search_de_feature_groups_from_surrogates,
    default_physics_rescue_hp,
    run_direct_residual_fss_from_feature_groups,
    run_factorized_coeff_rescue_from_feature_groups,
    run_regularized_implicit_residual_fss_from_feature_groups,
    run_factorized_search_de_from_feature_groups,
    run_factorized_search_de_from_surrogate,
    run_factorized_search_de_from_surrogates,
    discover_de_from_surrogate,
    discover_de_from_surrogates,
)
from nestynet_sr.sr_de.varpro_de import (
    varpro_refine_linear,
    varpro_refine_linear_multi,
    varpro_template_search,
    varpro_template_search_multi,
)
from nestynet_sr.sr_search.config import DataHyperparams, LMHyperparams, ModelHyperparams
from nestynet_sr.sr_search.data_utils import build_datasets
from nestynet_sr.sr_search.factorized_search.config import apply_refine_mode_placement_defaults, apply_refine_profile
from nestynet_sr.sr_search.model_builders import LeafBuilder, build_composite_ast
from nestynet_sr.sr_search.training import train_initial_model


def _unwrap_leaf_core(module):
    """Best-effort unwrap of adaptor wrappers to access leaf core parameters."""
    m = module
    for _ in range(6):
        nxt = None
        if hasattr(m, "module"):
            nxt = getattr(m, "module", None)
        if nxt is None and hasattr(m, "model"):
            nxt = getattr(m, "model", None)
        if nxt is None and hasattr(m, "base_model"):
            nxt = getattr(m, "base_model", None)
        if nxt is None or nxt is m:
            break
        m = nxt
    return m


def _leaf_scalar_value(leaf) -> float:
    """Extract a scalar value from a constant-like leaf module."""
    core = _unwrap_leaf_core(leaf)
    if hasattr(core, "value"):
        v = getattr(core, "value")
        if isinstance(v, torch.Tensor):
            return float(v.detach().reshape(-1)[0].cpu().item())
        return float(v)
    raise ValueError(f"Leaf core does not expose scalar 'value': {type(core).__name__}")


def _flatten_add_terms(node):
    from nestynet_sr.sr_core.bridges import AddNode

    if isinstance(node, AddNode):
        return _flatten_add_terms(node.left) + _flatten_add_terms(node.right)
    return [node]


def _is_scalar_coeff_atom(node) -> bool:
    from nestynet_sr.sr_core.bridges import AtomNode

    if not isinstance(node, AtomNode):
        return False
    kind = str(getattr(node, "kind", "")).lower()
    return kind in ("free_const", "freeconst", "free_constant", "scale", "mul_scale", "fixed_const", "fixedconstant", "fixed_constant")


def _coeff_atom_from_add_term(term):
    from nestynet_sr.sr_core.bridges import MulNode

    if _is_scalar_coeff_atom(term):
        return term
    if isinstance(term, MulNode):
        if _is_scalar_coeff_atom(term.left):
            return term.left
        if _is_scalar_coeff_atom(term.right):
            return term.right
    return None


def _extract_linear_coeffs_from_residual_state(root, model, num_terms: int) -> torch.Tensor:
    """Read fitted DE linear coefficients from a fitted residual model/state."""
    from nestynet_sr.sr_search.stageB.atom_mapping import build_atom_to_leaf_map
    from nestynet_sr.sr_core.bridges import collect_all_atoms

    add_terms = _flatten_add_terms(root)
    coeff_atoms = []
    for term in add_terms:
        a = _coeff_atom_from_add_term(term)
        if a is not None:
            coeff_atoms.append(a)

    if len(coeff_atoms) < int(num_terms):
        # Fallback: collect scalar atoms globally and pick a deterministic subset.
        scalar_atoms = [a for a in collect_all_atoms(root) if _is_scalar_coeff_atom(a)]

        def _scalar_key(atom):
            tag = str(getattr(atom, "tag", "") or "")
            m = re.fullmatch(r"c(\d+)", tag)
            if m is not None:
                return (0, int(m.group(1)), tag)
            return (1, 0, tag)

        scalar_atoms = sorted(scalar_atoms, key=_scalar_key)
        coeff_atoms = scalar_atoms[: int(num_terms)]

    if len(coeff_atoms) < int(num_terms):
        raise ValueError(
            f"Could not find enough scalar coefficient atoms in residual AST: "
            f"found {len(coeff_atoms)}, expected {int(num_terms)}"
        )
    coeff_atoms = coeff_atoms[: int(num_terms)]

    atom_to_leaf = build_atom_to_leaf_map(root, model)
    vals = []
    for a in coeff_atoms:
        leaf = atom_to_leaf.get(id(a), None)
        if leaf is None:
            raise ValueError(f"No leaf mapping for coefficient atom tag={getattr(a, 'tag', None)!r}")
        vals.append(_leaf_scalar_value(leaf))
    return torch.tensor(vals, dtype=torch.float64)


class ZeroTargetDataset(torch.utils.data.Dataset):
    """Wrap a dataset so targets become zeros with identical shape/dtype."""

    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        item = self.base_ds[idx]
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise ValueError("ZeroTargetDataset expects dataset items like (x, y, ...)")
        x = item[0]
        y = item[1]
        if torch.is_tensor(y):
            z = torch.zeros_like(y)
        else:
            z = np.zeros_like(y)
        return (x, z) + tuple(item[2:])


def _make_zero_target_loader(base_ds, ref_loader, fallback_batch_size: int):
    """Mirror loader batch/drop settings while replacing targets with zeros."""
    bs = getattr(ref_loader, "batch_size", None) or int(fallback_batch_size)
    return torch.utils.data.DataLoader(
        ZeroTargetDataset(base_ds),
        batch_size=int(bs),
        shuffle=False,
        drop_last=bool(getattr(ref_loader, "drop_last", False)),
    )


def _ensure_single_residual_ast(res, cfg: DESearchConfig):
    ast0 = getattr(res, "residual_ast", None)
    if ast0 is not None:
        return ast0
    from nestynet_sr.sr_de.de_search import build_de_residual_ast

    return build_de_residual_ast(
        res,
        units_spec=cfg.units_spec,
        enforce_units=bool(cfg.enforce_units),
    )


def _ensure_multi_residual_asts(res, cfg: DESearchConfig):
    residual_asts = getattr(res, "residual_asts", None)
    if residual_asts is not None and len(residual_asts) > 0:
        return list(residual_asts)

    from nestynet_sr.sr_de.de_search import DESearchResult, build_de_residual_ast

    coeffs = getattr(res, "coeffs", None)
    if not isinstance(coeffs, torch.Tensor) or coeffs.ndim != 2:
        raise ValueError("Expected multi-dataset coeff matrix with shape (D, K)")

    roots = []
    rms_train = getattr(res, "rms_train", None)
    rms_val = getattr(res, "rms_val", None)
    for d in range(int(coeffs.shape[0])):
        tmp = DESearchResult(
            order=int(res.order),
            x_axis=int(res.x_axis),
            term_asts=list(res.term_asts),
            coeffs=coeffs[d].detach().cpu(),
            rms_train=float(rms_train[d]) if isinstance(rms_train, (list, tuple)) else float("nan"),
            rms_val=float(rms_val[d]) if isinstance(rms_val, (list, tuple)) else None,
        )
        roots.append(
            build_de_residual_ast(
                tmp,
                units_spec=cfg.units_spec,
                enforce_units=bool(cfg.enforce_units),
            )
        )
    return roots


def _run_stageb_residual_refine_single(
    *,
    res,
    surrogate,
    ds_tr,
    ds_va,
    dl_tr,
    dl_va,
    lm_hp: LMHyperparams,
    cfg: DESearchConfig,
    device: torch.device,
    dtype: torch.dtype,
    epochs_stageB: int,
):
    """Fit discovered residual AST against zero targets with Stage-B LM."""
    from nestynet_sr.sr_de.de_search import make_u_feature_atom_factory
    from nestynet_sr.sr_search.stageB.fitting import _fit_candidate_root

    residual_ast = _ensure_single_residual_ast(res, cfg)
    train0 = _make_zero_target_loader(
        ds_tr,
        dl_tr,
        fallback_batch_size=getattr(dl_tr, "batch_size", None) or max(1, len(ds_tr)),
    )
    val0 = _make_zero_target_loader(
        ds_va,
        dl_va,
        fallback_batch_size=getattr(dl_va, "batch_size", None) or max(1, len(ds_va)),
    )
    atom_factory = make_u_feature_atom_factory(surrogate)

    st = _fit_candidate_root(
        root=residual_ast,
        reuse={},
        train_loader=train0,
        val_loader=val0,
        lm_hp=lm_hp,
        device=device,
        dtype=dtype,
        epochs_stageB=max(1, int(epochs_stageB)),
        loss_scale=1.0,
        atom_factory=atom_factory,
    )

    val_mse = float(st.val_loss)
    val_rms = math.sqrt(max(0.0, val_mse)) if math.isfinite(val_mse) else float("inf")
    meta = {
        "mode": "single",
        "epochs": int(max(1, epochs_stageB)),
        "val_mse": val_mse,
        "val_rms": float(val_rms),
    }
    try:
        coeffs_new = _extract_linear_coeffs_from_residual_state(
            st.root,
            st.model,
            num_terms=len(getattr(res, "term_asts", []) or []),
        )
        coeffs_old = getattr(res, "coeffs", None)
        if isinstance(coeffs_old, torch.Tensor):
            coeffs_new = coeffs_new.to(dtype=coeffs_old.dtype, device=coeffs_old.device)
        setattr(res, "coeffs", coeffs_new)
        setattr(res, "residual_ast", st.root)
        meta["coefficients_updated"] = True
        meta["coefficients"] = [float(v) for v in coeffs_new.detach().cpu().tolist()]
    except Exception as e:
        meta["coefficients_updated"] = False
        meta["coeff_update_error"] = str(e)
    return st, meta


def _run_stageb_residual_refine_multi(
    *,
    res,
    surrogates: list,
    ds_tr_list: list,
    ds_va_list: list,
    dl_tr_list: list,
    dl_va_list: list,
    lm_hp: LMHyperparams,
    cfg: DESearchConfig,
    device: torch.device,
    dtype: torch.dtype,
    epochs_stageB: int,
):
    """Joint Stage-B LM refinement for multi-dataset residual fitting."""
    from nestynet_sr.sr_de.de_search import make_u_feature_atom_factory
    from nestynet_sr.sr_search.stageB.fitting import _fit_candidate_root_multi

    residual_asts = _ensure_multi_residual_asts(res, cfg)
    D = len(surrogates)
    if len(residual_asts) != D:
        raise ValueError(f"Residual AST count {len(residual_asts)} does not match datasets {D}")
    if len(ds_tr_list) != D or len(ds_va_list) != D or len(dl_tr_list) != D or len(dl_va_list) != D:
        raise ValueError("Mismatch between dataset/loader list lengths for multi-dataset refinement")

    train0 = [
        _make_zero_target_loader(ds, dl, fallback_batch_size=getattr(dl, "batch_size", None) or max(1, len(ds)))
        for ds, dl in zip(ds_tr_list, dl_tr_list)
    ]
    val0 = [
        _make_zero_target_loader(ds, dl, fallback_batch_size=getattr(dl, "batch_size", None) or max(1, len(ds)))
        for ds, dl in zip(ds_va_list, dl_va_list)
    ]
    atom_factories = [make_u_feature_atom_factory(s) for s in surrogates]

    st = _fit_candidate_root_multi(
        root=residual_asts[0],
        reuses=[{} for _ in range(D)],
        train_loaders=train0,
        val_loaders=val0,
        lm_hp=lm_hp,
        device=device,
        dtype=dtype,
        epochs_stageB=max(1, int(epochs_stageB)),
        loss_scales=[1.0 for _ in range(D)],
        dataset_ids=getattr(res, "dataset_ids", None),
        agg_mode="mean",
        atom_factory=atom_factories,
    )

    val_losses = [float(v) for v in (getattr(st, "val_losses", None) or [st.val_loss])]
    val_rms = [math.sqrt(max(0.0, v)) if math.isfinite(v) else float("inf") for v in val_losses]
    meta = {
        "mode": "multi",
        "epochs": int(max(1, epochs_stageB)),
        "val_mse": float(st.val_loss),
        "val_rms": float(math.sqrt(max(0.0, float(st.val_loss)))),
        "val_mse_per_dataset": val_losses,
        "val_rms_per_dataset": [float(v) for v in val_rms],
    }
    try:
        models = list(getattr(st, "models", None) or [])
        if not models:
            models = [st.model]
        coeff_rows = [
            _extract_linear_coeffs_from_residual_state(
                st.root,
                m,
                num_terms=len(getattr(res, "term_asts", []) or []),
            )
            for m in models
        ]
        coeffs_new = torch.stack(coeff_rows, dim=0)
        coeffs_old = getattr(res, "coeffs", None)
        if isinstance(coeffs_old, torch.Tensor):
            coeffs_new = coeffs_new.to(dtype=coeffs_old.dtype, device=coeffs_old.device)
        setattr(res, "coeffs", coeffs_new)
        setattr(res, "residual_asts", [st.root for _ in range(int(coeffs_new.shape[0]))])
        meta["coefficients_updated"] = True
        meta["coefficients"] = [[float(v) for v in row] for row in coeffs_new.detach().cpu().tolist()]
    except Exception as e:
        meta["coefficients_updated"] = False
        meta["coeff_update_error"] = str(e)
    return st, meta


def _merge_results_to_multi(
    results: list,  # List[DESearchResult]
    dataset_ids: list[str] = None,
):
    """Merge list of per-dataset DESearchResult into DESearchResultMulti.

    Used after varpro_refine_linear_multi() which returns List[DESearchResult].

    Parameters
    ----------
    results : list of DESearchResult
        Per-dataset results with shared term support
    dataset_ids : list of str, optional
        Dataset identifiers

    Returns
    -------
    DESearchResultMulti
        Merged result with (D, K) coefficient matrix
    """
    if len(results) == 0:
        raise ValueError("Empty results list")

    # Validate shared structure
    term_asts_0 = results[0].term_asts
    for i, res in enumerate(results[1:], start=1):
        if len(res.term_asts) != len(term_asts_0):
            raise ValueError(f"Result {i} has different number of terms")

    # Build residual ASTs
    from nestynet_sr.sr_de.de_search import build_de_residual_ast

    residual_asts = [
        res.residual_ast
        if hasattr(res, "residual_ast") and res.residual_ast is not None
        else build_de_residual_ast(res)
        for res in results
    ]

    merged = DESearchResultMulti(
        order=results[0].order,
        x_axis=results[0].x_axis,
        term_asts=term_asts_0,
        coeffs=torch.stack([r.coeffs for r in results], dim=0),
        rms_train=[float(r.rms_train) for r in results],
        rms_val=[float(r.rms_val) for r in results] if results[0].rms_val is not None else None,
        dataset_ids=dataset_ids,
        residual_asts=residual_asts,
        term_sources=getattr(results[0], "term_sources", None),
        prolongation_metadata=getattr(results[0], "prolongation_metadata", None),
        determining_certificate=getattr(results[0], "determining_certificate", None),
    )
    for _attr in ("expr_ir_report", "expr_ir_reports_by_order"):
        if hasattr(results[0], _attr):
            setattr(merged, _attr, getattr(results[0], _attr))
    return merged


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Discover DEs from data using neural network surrogates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input data: single CSV or a list of CSVs
    parser.add_argument(
        "--filepath", type=str, default=None, help="Path to input CSV file (single-dataset mode)"
    )
    parser.add_argument(
        "--filepaths",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of input CSV files (multi-dataset mode). "
        "If provided, a separate surrogate is trained per dataset and a shared-support DE is discovered.",
    )
    parser.add_argument("--stat-selection", action="store_true", help="Freeze the DE candidate archive and certify it on untouched whole trajectories")
    parser.add_argument("--stat-audit-filepaths", nargs="+", default=None, help="Independent audit trajectory CSVs; never opened by search")
    parser.add_argument("--stat-audit-trajectories", type=int, default=2, help="Without external audit files, reserve this many final input trajectories")
    parser.add_argument("--stat-alpha", type=float, default=0.05, help="Familywise error level for simultaneous Pareto bounds")
    parser.add_argument("--stat-delta", type=float, default=0.01, help="Practical noninferiority margin in squared rollout-NRMSE units")
    parser.add_argument("--stat-resamples", type=int, default=4000, help="Multiplier-bootstrap draws")
    parser.add_argument("--stat-seed", type=int, default=12345, help="Statistical-selection resampling seed")
    parser.add_argument("--stat-multiplier", choices=["normal", "rademacher"], default="normal")
    parser.add_argument("--stat-failure-loss", type=float, default=100.0, help="Common whole-trajectory loss for compile/domain/integration failure")
    parser.add_argument("--stat-max-candidates", type=int, default=256, help="Deterministic cap on the frozen DE archive")
    parser.add_argument("--stat-rollout-window-fraction", type=float, default=1.0, help="Fraction of each audit trajectory used for rollout scoring")
    parser.add_argument("--stat-rollout-max-span", type=float, default=None, help="Optional maximum rollout span; default evaluates the full trajectory")
    parser.add_argument("--stat-traj-time-budget-s", type=float, default=20.0, help="Per-candidate, per-trajectory integration budget")
    parser.add_argument("--stat-certificate-json", type=str, default=None, help="Optional separate Pareto-certificate JSON path")
    parser.add_argument("--stat-coherent-loss-draws", type=str, default=None, help="NPZ bundle of coherent surrogate-draw rollout losses with shape (draw, trajectory, candidate)")
    parser.add_argument("--stat-rediscovery-reports", nargs="+", default=None, help="Independent full-pipeline DE report JSONs used to estimate structural rediscovery frequencies")
    parser.add_argument("--stat-calibration-repetitions", type=int, default=0, help="Run a deterministic paired-comparison calibration smoke test and record it in the certificate")

    # DE discovery parameters
    parser.add_argument(
        "--x_axis",
        type=int,
        default=None,
        help="Index of the independent variable (x-axis) for DE. If not specified, will auto-detect from dataset coordinate metadata (looks for time-like coordinates).",
    )
    parser.add_argument(
        "--order_candidates",
        type=str,
        default="1,2",
        help='Comma-separated list of DE orders to try (e.g., "1" or "1,2")',
    )

    # Term library controls
    parser.add_argument(
        "--max_x_power", type=int, default=1, help="Maximum power of x in library terms"
    )
    parser.add_argument(
        "--max_u_power", type=int, default=1, help="Maximum power of u in library terms"
    )
    parser.add_argument(
        "--max_xu_total_degree", type=int, default=0,
        help="If >0, cap total degree p+q for x^p*u^q cross terms (0=unlimited)"
    )
    parser.add_argument(
        "--include_xdu",
        action="store_true",
        help="Include x*du terms (critical for 2nd-order DEs like u_xx + x*u_x + u = 0)",
    )
    parser.add_argument(
        "--include_inv_xdu",
        action="store_true",
        help="Include x^-1*du terms (useful for singular equations like Lane-Emden)",
    )
    parser.add_argument(
        "--include_inv_xu",
        action="store_true",
        help="Include x^-1*u terms (useful for radial inflow dv/dr = -v/r)",
    )
    parser.add_argument(
        "--include_inv_x2u",
        action="store_true",
        help="Include x^-2*u terms (useful for Bessel-nu equations)",
    )
    parser.add_argument(
        "--include_du", action="store_true", help="Include du terms in library (if not the anchor)"
    )
    parser.add_argument(
        "--include_d2u",
        action="store_true",
        help="Include d2u terms in library (if not the anchor)",
    )
    parser.add_argument(
        "--include_udu",
        action="store_true",
        help="Include u*du cross terms (useful for logistic-like dynamics)",
    )
    parser.add_argument("--de-hard-tail-templates", action="store_true", default=False, help="Append explicit hard-tail structural-prior DE templates without enabling GS")
    parser.add_argument("--de-hard-tail-no-radial-templates", dest="de_hard_tail_radial_templates", action="store_false", default=True, help="Disable neutral hard-tail radial/singular DE templates")
    parser.add_argument("--de-hard-tail-velocity-templates", action="store_true", default=False, help="Enable neutral hard-tail velocity-dependent templates")

    # Generalized-symmetry switches. Disabled by default.
    parser.add_argument("--gs-enable", action="store_true", help="Enable generalized-symmetry diagnostics/templates")
    parser.add_argument("--gs-mode", type=str, choices=["off", "audit", "propose", "auto"], default="propose", help="GS mode for DE diagnostics; DE templates are controlled separately")
    parser.add_argument("--gs-policy", type=str, choices=["augment", "replace-shadowed", "gs-only-affine"], default="augment", help="GS DE policy: augment baseline library, replace shadowed radial templates, or use GS-only for shadowed templates")
    parser.add_argument("--gs-auto", action="store_true", help="Alias for --gs-enable --gs-mode auto; writes a GS DE report")
    parser.add_argument("--gs-known-generators", "--gs-known-lie", dest="gs_known_generators", action="store_true", default=True, help="Enable named affine GS generator diagnostics in DE reports")
    parser.add_argument("--gs-no-known-generators", "--gs-no-known-lie", dest="gs_known_generators", action="store_false", help="Disable named affine GS generator diagnostics")
    parser.add_argument("--gs-general-affine", dest="gs_general_affine", action="store_true", default=False, help="Enable learned sparse affine generator diagnostics on fitted DE surrogates")
    parser.add_argument("--gs-no-general-affine", dest="gs_general_affine", action="store_false", help="Disable learned sparse affine generator diagnostics")
    parser.add_argument("--gs-jet-enable", dest="gs_jet_enable", action="store_true", default=True, help="Enable GS jet diagnostics where a path supports them")
    parser.add_argument("--gs-no-jet", dest="gs_jet_enable", action="store_false", help="Disable GS jet diagnostics")
    parser.add_argument("--gs-no-jet-separability", dest="gs_jet_separability", action="store_false", default=True, help="Disable additive Hessian-block GS jet witnesses")
    parser.add_argument("--gs-no-jet-multiplicative", dest="gs_jet_multiplicative", action="store_false", default=True, help="Disable multiplicative f*f_ij-f_i*f_j GS jet witnesses")
    parser.add_argument("--gs-no-translations", dest="gs_translations", action="store_false", default=True, help="Disable named translation generator diagnostics")
    parser.add_argument("--gs-no-diagonal-translations", dest="gs_diagonal_translations", action="store_false", default=True, help="Disable named diagonal-translation generator diagnostics")
    parser.add_argument("--gs-no-scalings", dest="gs_scalings", action="store_false", default=True, help="Disable named scaling generator diagnostics")
    parser.add_argument("--gs-no-rotations", dest="gs_rotations", action="store_false", default=True, help="Disable named rotation generator diagnostics")
    parser.add_argument("--gs-lorentz-boosts", dest="gs_lorentz_boosts", action="store_true", default=False, help="Enable named Lorentz/hyperbolic generator diagnostics")
    parser.add_argument("--gs-no-output-equivariance", dest="gs_output_equivariance", action="store_false", default=True, help="Require strict Xf≈0 for affine diagnostics")
    parser.add_argument("--gs-residual-tol", type=float, default=0.03, help="GS clean residual threshold")
    parser.add_argument("--gs-audit-residual-tol", type=float, default=0.10, help="GS audit residual threshold")
    parser.add_argument("--gs-min-confidence", type=float, default=0.65, help="Minimum heuristic score for affine diagnostics")
    parser.add_argument("--gs-affine-max-terms", type=int, default=4, help="Maximum nonzero affine-basis terms in learned sparse affine generators")
    parser.add_argument("--gs-affine-num-candidates", type=int, default=4, help="Number of affine generator candidates to inspect")
    parser.add_argument("--gs-de-templates", action="store_true", help="Deprecated alias for --de-hard-tail-templates")
    parser.add_argument("--gs-de-no-radial-templates", dest="gs_de_radial_templates", action="store_false", default=True, help="Deprecated alias for --de-hard-tail-no-radial-templates")
    parser.add_argument("--gs-de-velocity-templates", action="store_true", help="Deprecated alias for --de-hard-tail-velocity-templates")
    parser.add_argument("--gs-de-all-upgrades", dest="gs_de_all_upgrades", action="store_true", default=False, help="Enable all bounded DE probe/prior families")
    parser.add_argument("--gs-de-determining-equations", dest="gs_de_determining_equations", action="store_true", default=False, help="Enable the coupled degree-bounded point-symmetry nullspace solve for DE candidates")
    parser.add_argument("--gs-de-auto-nonlinear", dest="gs_de_auto_nonlinear", action="store_true", default=True, help="When GS is enabled, automatically compare affine and bounded-quadratic scalar-ODE determining solves")
    parser.add_argument("--gs-de-no-auto-nonlinear", dest="gs_de_auto_nonlinear", action="store_false", help="Disable automatic quadratic scalar-ODE escalation while retaining explicitly requested GS lanes")
    parser.add_argument("--gs-de-auto-fss", dest="gs_de_auto_fss", action="store_true", default=True, help="Run one bounded FSS challenger when automatic GS produces a certified nontrivial carrier")
    parser.add_argument("--gs-de-no-auto-fss", dest="gs_de_auto_fss", action="store_false", help="Do not launch the automatic carrier-seeded FSS challenger")
    parser.add_argument("--gs-de-auto-fss-max-attempts", type=int, default=1, help="Maximum automatic FSS attempts launched by certified nonlinear GS carriers")
    parser.add_argument("--gs-de-auto-fss-n-iter", type=int, default=1500, help="Automatic carrier-seeded FSS iteration cap")
    parser.add_argument("--gs-de-auto-fss-n-fit", type=int, default=1024, help="Automatic carrier-seeded FSS fit-sample cap")
    parser.add_argument("--gs-de-auto-fss-n-probe", type=int, default=1024, help="Automatic carrier-seeded FSS probe-sample cap")
    parser.add_argument("--gs-de-auto-fss-max-depth", type=int, default=4, help="Automatic carrier-seeded FSS expression-depth cap")
    parser.add_argument("--gs-de-auto-fss-return-topk", type=int, default=8, help="Automatic carrier-seeded FSS shortlist cap")
    parser.add_argument("--gs-de-certificate", dest="gs_de_determining_certificate", action="store_true", default=False, help="Attach a point-symmetry determining certificate (on-shell recovery + relative-invariance test, incl. nullspace combinations) to selected DE candidates")
    parser.add_argument("--gs-de-reduction", dest="gs_de_reduction", action="store_true", default=False, help="Discover ensemble symmetries, rectify to canonical coordinates, fit the reduced univariate law, and inject the pulled-back compositional terms into the DE library (needs >=2 trajectory files; scalar first-order)")
    parser.add_argument("--gs-de-certificate-tol", dest="gs_de_certificate_tol", type=float, default=1.0e-6, help="On/off-shell relative residual tolerance for the determining certificate")
    parser.add_argument("--gs-de-certificate-coeff-prune-tol", dest="gs_de_certificate_coeff_prune_tol", type=float, default=0.0, help="Drop candidate terms with |coefficient| below this before certification")
    parser.add_argument("--gs-de-contact-templates", dest="gs_de_contact_templates", action="store_true", default=False, help="Enable velocity-monomial structural-prior templates")
    parser.add_argument("--gs-de-noether-templates", dest="gs_de_noether_templates", action="store_true", default=False, help="Enable autonomous/even-velocity structural-prior templates and diagnostics")
    parser.add_argument("--gs-de-discrete-symmetry-templates", dest="gs_de_discrete_symmetry_templates", action="store_true", default=False, help="Enable parity/time-reversal structural-prior templates and diagnostics")
    parser.add_argument("--gs-de-weighted-scaling-templates", dest="gs_de_weighted_scaling_templates", action="store_true", default=False, help="Enable assumed weight-profile structural-prior templates")
    parser.add_argument("--gs-de-radial-reduction-templates", dest="gs_de_radial_reduction_templates", action="store_true", default=False, help="Enable radial-shaped structural-prior templates")
    parser.add_argument("--gs-de-invariant-library", dest="gs_de_invariant_library", action="store_true", default=False, help="Enable conservative scalar differential-invariant library rows")
    parser.add_argument("--gs-de-upgrade-max-terms", type=int, default=64, help="Maximum source-aware rows emitted by the bounded GS DE upgrade bridge")
    parser.add_argument("--gs-de-determining-max-degree", type=int, choices=[1, 2], default=2, help="Polynomial generator degree: affine regression (1) or bounded quadratic lane (2)")
    parser.add_argument("--gs-de-determining-max-generators", type=int, default=4, help="Maximum accepted generator-basis probe rows reported per candidate")
    parser.add_argument("--gs-de-determining-multiplier-degree", type=int, default=2, help="Maximum jet-monomial degree for the functional relative-invariance multiplier")
    parser.add_argument("--gs-de-determining-bootstraps", type=int, default=8, help="Bootstrap resamples used to audit the determining nullspace projector")
    parser.add_argument("--gs-de-no-sparse-rotation", dest="gs_de_determining_sparse_rotation", action="store_false", default=True, help="Disable sparse rotation inside the recovered nonlinear-generator subspace")
    parser.add_argument("--gs-de-no-bracket-certificate", dest="gs_de_determining_bracket_certificate", action="store_false", default=True, help="Disable evaluated Lie-bracket closure checks")
    parser.add_argument("--gs-de-nonlinear-invariants", action="store_true", default=False, help="Compile low-complexity invariants and orbit coordinates from accepted nonlinear generators")
    parser.add_argument("--gs-de-nonlinear-invariant-max-degree", type=int, default=3, help="Maximum polynomial carrier degree for nonlinear invariant compilation")
    parser.add_argument("--gs-de-nonlinear-invariant-max-candidates", type=int, default=8, help="Maximum certified nonlinear invariant/orbit carriers retained")
    parser.add_argument("--gs-de-nonlinear-invariant-tol", type=float, default=0.03, help="Held-out normalized action tolerance for nonlinear invariant carriers")
    parser.add_argument("--gs-de-no-orbit-coordinate", dest="gs_de_nonlinear_orbit_coordinate", action="store_false", default=True, help="Do not search for Xs=1 orbit coordinates for one-generator algebras")
    parser.add_argument("--gs-de-weighted-max-abs-x-power", type=int, default=2, help="Maximum absolute x power for weighted-scaling templates")
    parser.add_argument("--gs-de-weighted-max-u-power", type=int, default=5, help="Maximum u power for weighted-scaling templates")
    parser.add_argument("--gs-de-weighted-max-du-power", type=int, default=4, help="Maximum u_x power for weighted-scaling templates")
    parser.add_argument("--gs-de-lie-prolongation", action="store_true", help="Audit sparse DE candidates with finite point-Lie prolongation invariance tests")
    parser.add_argument("--gs-de-lie-use-for-selection", action="store_true", default=False, help="Allow Lie-prolongation diagnostics to affect DE model selection; default is audit-only")
    parser.add_argument("--gs-de-lie-audit-only", dest="gs_de_lie_use_for_selection", action="store_false", help="Keep Lie-prolongation diagnostics out of DE model selection")
    parser.add_argument("--gs-de-lie-prolongation-weight", type=float, default=0.05, help="Selection penalty weight used only with --gs-de-lie-use-for-selection")
    parser.add_argument("--gs-de-lie-prolongation-tol", type=float, default=0.05, help="Acceptance tolerance for Lie-prolongation on-shell residual metric")
    parser.add_argument("--gs-de-lie-prolongation-max-samples", type=int, default=2048, help="Maximum surrogate jet samples used by each Lie-prolongation candidate score")
    parser.add_argument("--gs-de-lie-prolongation-min-coverage", type=float, default=0.90, help="Minimum finite-sample coverage required for a Lie-prolongation metric to be eligible")
    parser.add_argument("--gs-unit-torus", dest="gs_unit_torus", action="store_true", default=False, help="Enable unit-torus/Buckingham-pi dimensional GS audit or proposals")
    parser.add_argument("--gs-no-unit-torus", dest="gs_unit_torus", action="store_false", help="Disable unit-torus dimensional GS")
    parser.add_argument("--gs-pi-invariants", dest="gs_pi_invariants", action="store_true", default=False, help="Enable Buckingham-pi invariant proposals where units are available")
    parser.add_argument("--gs-no-pi-invariants", dest="gs_pi_invariants", action="store_false", help="Disable Buckingham-pi invariant proposals")
    parser.add_argument("--gs-dim-policy", type=str, choices=["baseline", "audit", "augment", "both", "replace-rref", "gs-only"], default="audit", help="Dimensional GS policy; --gs-unit-torus defaults to audit")
    parser.add_argument("--gs-dim-both-rule", type=str, choices=["rref-dominates", "require-both", "either", "gs-dominates"], default="rref-dominates", help="Arbitration rule for --gs-dim-policy both")
    parser.add_argument("--gs-dim-validator", type=str, choices=["local", "nullspace", "linear"], default="nullspace", help="Unit-torus dimensional validator")
    parser.add_argument("--gs-dim-keep-local-gates", dest="gs_dim_keep_local_gates", action="store_true", default=True, help="Keep local dimensional safety gates when replacing global dimensional checks")
    parser.add_argument("--gs-dim-no-local-gates", dest="gs_dim_keep_local_gates", action="store_false", help="Disable local dimensional safety gates in GS dimensional replacement modes")
    parser.add_argument("--gs-pi-max-exponent", type=int, default=3, help="Maximum absolute exponent in bounded Buckingham-pi enumeration")
    parser.add_argument("--gs-pi-max-l1", type=int, default=6, help="Maximum L1 exponent norm in bounded Buckingham-pi enumeration")
    parser.add_argument("--gs-pi-max-proposals", type=int, default=24, help="Maximum unit-torus pi/prefactor proposals")
    parser.add_argument("--gs-pi-max-basis", type=int, default=8, help="Maximum support size for bounded pi/prefactor enumeration")
    parser.add_argument("--gs-pi-rational-denom", type=int, default=1, help="Maximum rational denominator for pi exponents")
    parser.add_argument("--gs-pi-include-free-consts", dest="gs_pi_include_free_consts", action="store_true", default=True, help="Allow declared free constants in GS dimensional span checks")
    parser.add_argument("--gs-pi-no-free-consts", dest="gs_pi_include_free_consts", action="store_false", help="Ignore free constants in GS dimensional span checks")
    parser.add_argument("--gs-unit-report-json", type=str, default=None, help="Optional path for the GS report JSON when unit-torus reporting is enabled")
    parser.add_argument("--gs-unit-report-md", type=str, default=None, help="Optional path for the GS report markdown when unit-torus reporting is enabled")
    parser.add_argument("--gs-report-dim-disagreements", dest="gs_report_dim_disagreements", action="store_true", default=True, help="Report baseline/GS dimensional disagreements")
    parser.add_argument("--gs-no-report-dim-disagreements", dest="gs_report_dim_disagreements", action="store_false", help="Suppress baseline/GS dimensional disagreement rows")
    parser.add_argument("--gs-report-pi-rejected", dest="gs_report_pi_rejected", action="store_true", default=False, help="Report rejected pi proposals where available")
    parser.add_argument("--gs-report-rejected", dest="gs_report_rejected", action="store_true", default=True, help="Compatibility flag for SR/DE ablation scripts")
    parser.add_argument("--gs-no-report-rejected", dest="gs_report_rejected", action="store_false", help="Compatibility flag for SR/DE ablation scripts")
    parser.add_argument("--gs-report-top-k-rejected", type=int, default=40, help="Compatibility flag for SR/DE ablation scripts")
    parser.add_argument("--ast-simplify", dest="ast_simplify", action="store_true", default=False, help="Enable conservative AST canonicalisation/deduplication in DE discovery")
    parser.add_argument("--no-ast-simplify", dest="ast_simplify", action="store_false", help="Disable AST canonicalisation/deduplication")
    parser.add_argument("--ast-simplify-level", type=str, choices=["safe", "symmetry"], default="safe", help="AST simplification level")
    parser.add_argument("--ast-simplify-domain-policy", type=str, choices=["strict", "common-domain"], default="strict", help="Domain policy for AST simplification")
    parser.add_argument("--ast-simplify-max-passes", type=int, default=12, help="Maximum AST simplification passes")
    parser.add_argument("--ast-simplify-validate", dest="ast_simplify_validate", action="store_true", default=False, help="Enable integration-site numeric validation for AST simplification where available")
    parser.add_argument("--ast-simplify-trace", dest="ast_simplify_trace", action="store_true", default=False, help="Record detailed AST simplification diagnostics")
    add_expr_ir_cli_args(parser)

    # Library exclusion flags (mirror run_SR.py --de_no_* pattern)
    parser.add_argument(
        "--no_const", dest="include_const", action="store_false", default=True,
        help="Exclude constant offset term from library",
    )
    parser.add_argument(
        "--no_x", dest="include_x_flag", action="store_false", default=True,
        help="Exclude x (and x^p) terms from library",
    )
    parser.add_argument(
        "--no_u", dest="include_u_flag", action="store_false", default=True,
        help="Exclude u (and u^q) terms from library",
    )
    parser.add_argument(
        "--no_xu", dest="include_xu_flag", action="store_false", default=True,
        help="Exclude x*u cross terms from library",
    )

    # STLSQ parameters
    parser.add_argument(
        "--stlsq_lambda", type=float, default=1e-3, help="STLSQ sparsification threshold"
    )
    parser.add_argument("--stlsq_max_iter", type=int, default=10, help="Maximum STLSQ iterations")
    parser.add_argument("--ridge", type=float, default=1e-10, help="Ridge regularization for STLSQ")
    parser.add_argument(
        "--sparsity_penalty", type=float, default=1e-3,
        help="Per-term penalty for model selection between candidate orders"
    )
    parser.add_argument(
        "--factorized-search-rescue",
        type=str,
        choices=["never", "auto", "always"],
        default="never",
        help="Run factorized symbolic search rescue after first-line DE discovery.",
    )
    parser.add_argument(
        "--factorized-rescue",
        type=str,
        choices=["never", "auto", "always"],
        default="never",
        help="Run typed coefficient-on-carrier rescue before whole-RHS factorized symbolic search.",
    )
    parser.add_argument(
        "--factorized-two-block-shared-coord",
        type=str,
        choices=["never", "auto", "always"],
        default="never",
        help="Enable the guarded shared-coordinate two-block factorized rescue lane.",
    )
    parser.add_argument(
        "--factorized-search-trigger-val-rms",
        type=float,
        default=1.0e-3,
        help="Auto-rescue threshold on first-line validation RMS.",
    )
    parser.add_argument(
        "--factorized-search-trigger-rel-rms",
        type=float,
        default=1.0e-3,
        help="Scale-relative RMS threshold for stopping DE-facing factorized symbolic search attempts.",
    )
    parser.add_argument(
        "--factorized-search-trigger-cond",
        type=float,
        default=1.0e8,
        help="Auto-rescue threshold on first-line condition number.",
    )
    parser.add_argument(
        "--factorized-implicit-clean-score",
        type=float,
        default=5.0e-3,
        help=(
            "Normalized probe-score threshold below which a regularized implicit "
            "linear candidate is certified clean, skipping the full-direct and "
            "curated typed challenger lanes."
        ),
    )
    parser.add_argument(
        "--factorized-clean-gate-val-rms",
        type=float,
        default=None,
        help=(
            "Optional cleanliness gate on first-line probe RMS for skipping "
            "challenger lanes. Defaults to --factorized-search-trigger-val-rms."
        ),
    )
    parser.add_argument(
        "--factorized-clean-gate-rel-rms",
        type=float,
        default=None,
        help=(
            "Optional cleanliness gate on first-line relative probe RMS for "
            "skipping challenger lanes. Defaults to --factorized-search-trigger-rel-rms."
        ),
    )
    parser.add_argument(
        "--factorized-search-replace-rel-factor",
        type=float,
        default=0.98,
        help="Require rescue RMS < factor * first-line RMS before selecting it.",
    )
    parser.add_argument(
        "--factorized-search-preset",
        type=str,
        choices=["fast", "default", "paper", "compositional", "compositional_fast"],
        default="default",
        help="factorized symbolic search rescue hyperparameter preset.",
    )
    parser.add_argument(
        "--factorized-search-only",
        action="store_true",
        help="Skip first-line sparse/VarPro DE discovery and run factorized symbolic search alone on the surrogate features.",
    )
    parser.add_argument(
        "--factorized-de",
        action="store_true",
        help=(
            "Skip STLSQ and run zero-base operator-factorized DE plus whole-RHS "
            "factorized symbolic search, selecting between the two by held-out diagnostics."
        ),
    )
    parser.add_argument(
        "--factorized-de-whole-rhs",
        type=str,
        choices=["never", "auto", "always"],
        default="auto",
        help=(
            "Policy for launching the broad whole-RHS factorized symbolic search lane in "
            "--factorized-de mode. 'always' preserves the old behavior; 'auto' only "
            "runs it when typed factorized proposals look absent, fragile, or ambiguous."
        ),
    )
    parser.add_argument(
        "--factorized-de-typed-lanes",
        type=str,
        choices=["never", "auto", "always", "force"],
        default="never",
        help=(
            "Policy for launching curated typed operator-factorized lanes in --factorized-de mode. "
            "'auto' runs them only when the first-line law is not certified clean; "
            "'always' launches them unconditionally ('force' is a deprecated alias)."
        ),
    )
    parser.add_argument(
        "--factorized-de-typed-lane-workers",
        type=int,
        default=1,
        help=(
            "Maximum worker threads for independent curated typed-lane explorer launches "
            "inside --factorized-de. Default 1 preserves serial behavior."
        ),
    )
    parser.add_argument(
        "--factorized-search-de-refine-mode",
        type=str,
        choices=["off", "rare_final_polish", "rare_slate"],
        default="rare_final_polish",
        help=(
            "Continuous-refinement profile for DE-facing factorized symbolic search. "
            "Use rare_slate to restore broad scheduled-slate CSR."
        ),
    )
    parser.add_argument("--factorized-search-n-iter", type=int, default=None, help="Optional factorized symbolic search iteration override.")
    parser.add_argument(
        "--factorized-search-max-depth",
        type=int,
        default=None,
        help="Optional factorized symbolic search max-depth override.",
    )
    parser.add_argument("--factorized-search-n-fit", type=int, default=None, help="Optional factorized symbolic search fit-budget override.")
    parser.add_argument(
        "--factorized-search-n-probe",
        type=int,
        default=None,
        help="Optional factorized symbolic search probe-budget override.",
    )
    parser.add_argument(
        "--factorized-search-return-topk",
        type=int,
        default=None,
        help="Optional factorized symbolic search top-k return override.",
    )
    parser.add_argument(
        "--factorized-search-max-attempts",
        type=int,
        default=None,
        help=(
            "Maximum broad whole-RHS factorized symbolic search heuristic attempts. "
            "Unset preserves legacy standalone behavior; factorized-de auto defaults to a bounded reservoir."
        ),
    )
    parser.add_argument(
        "--factorized-search-integrate-topk",
        type=int,
        default=None,
        help=(
            "Internal integration-validation top-k for broad whole-RHS factorized symbolic search. "
            "Default follows legacy standalone rerank policy; use 0 when an outer benchmark/committee rollout "
            "already arbitrates candidates."
        ),
    )
    parser.add_argument(
        "--factorized-search-direct-generator-witness-topk",
        type=int,
        default=1,
        help=(
            "Top-k direct residual FSS candidates to validate with the order-2 generator witness. "
            "This is independent of broad whole-RHS integration validation."
        ),
    )
    parser.add_argument(
        "--factorized-search-budget-scope",
        type=str,
        choices=["per_group", "global"],
        default=None,
        help=(
            "Broad whole-RHS factorized symbolic search row-budget scope. "
            "per_group preserves legacy behavior; global divides n_fit/n_probe across feature groups."
        ),
    )

    # Surrogate training parameters
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, auto-detected if not specified)",
    )
    parser.add_argument(
        "--num_segments",
        type=int,
        default=32,
        help="Number of segments for neural network surrogate",
    )
    parser.add_argument(
        "--surrogate_max_periods_per_window",
        type=float,
        default=4.0,
        help=(
            "Split oscillation-dense 1-D datasets into contiguous windows of at most "
            "this many oscillation periods and train one surrogate per window "
            "(restores segments-per-wavelength for long oscillatory records; 0 disables)"
        ),
    )
    parser.add_argument(
        "--surrogate_retrain_attempts",
        type=int,
        default=1,
        help=(
            "Extra surrogate training attempts with a fresh initialization when "
            "val_loss stalls far above the loss target (0 disables)"
        ),
    )
    parser.add_argument(
        "--single_layer",
        action="store_true",
        help="Use single-layer architecture for NN atoms (default is dual-layer)",
    )
    parser.add_argument(
        "--epochs", type=int, default=8000, help="Maximum training epochs for surrogate"
    )
    parser.add_argument(
        "--epochs_min", type=int, default=500, help="Minimum training epochs before early stopping"
    )
    parser.add_argument(
        "--nval_patience", type=int, default=400, help="Validation patience for early stopping"
    )
    parser.add_argument("--batch_size", type=int, default=2000, help="Batch size for training")
    parser.add_argument(
        "--ndata_train", type=int, default=2000, help="Number of training data points"
    )
    parser.add_argument(
        "--ndata_val", type=int, default=2000, help="Number of validation data points"
    )
    parser.add_argument(
        "--data_split",
        type=str,
        choices=["contiguous", "interleaved"],
        default="contiguous",
        help=(
            "Train/validation row split strategy for surrogate fitting. "
            "'contiguous' preserves historical first-block/next-block behavior; "
            "'interleaved' spreads both subsets across the selected domain."
        ),
    )
    parser.add_argument(
        "--loss_target", type=float, default=1e-7, help="Target loss for surrogate training"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="direct_solve",
        choices=["direct_solve", "explicit", "matfree"],
        help="LM optimization strategy",
    )
    parser.add_argument(
        "--evidence",
        action="store_true",
        help="Train the surrogate with NestyNet segment-prior evidence guidance.",
    )
    parser.add_argument(
        "--evidence_disable_residual_whitening",
        action="store_true",
        help="Disable evidence residual-whitening / patch terms.",
    )
    parser.add_argument(
        "--evidence_disable_segment_priors",
        action="store_true",
        help="Disable evidence segment priors, making --evidence a no-op unless other terms are active.",
    )
    parser.add_argument(
        "--evidence_lambda_patch",
        type=float,
        default=None,
        help="Residual-whitening weight. Positive values are rejected by the SR/DE surrogate path.",
    )
    parser.add_argument(
        "--evidence_prior_decay_start",
        type=int,
        default=None,
        help="LM iteration where segment-prior decay starts; automatic default is 800, moved earlier if needed so the default interval still fits into the LM budget.",
    )
    parser.add_argument(
        "--evidence_prior_decay_interval",
        type=int,
        default=None,
        help="Number of LM iterations used to decay the segment prior once decay starts. Default: 200.",
    )
    parser.add_argument(
        "--evidence_prior_decay_shape",
        type=str,
        choices=["linear", "smoothstep", "cosine"],
        default=None,
        help="Shape of the segment-prior decay ramp.",
    )
    parser.add_argument(
        "--evidence_prior_decay_final_scale",
        type=float,
        default=None,
        help="Final global segment-prior multiplier after decay (default: 0).",
    )
    parser.add_argument(
        "--evidence_prior_cutoff_tol",
        type=float,
        default=None,
        help=(
            "Early-start trigger for segment-prior decay. If plain training selection-loss "
            "loss improvement over an LM report period falls below this threshold before "
            "the scheduled decay start, DE starts the decay immediately. Interpreted on "
            "the plain DE selection-loss scale, so it stays aligned with the visible DE "
            "loss thresholds rather than the augmented evidence objective. "
            "Default: 1e-9."
        ),
    )
    parser.add_argument(
        "--no_evidence_prior_decay_auto",
        dest="evidence_prior_decay_auto",
        action="store_false",
        default=True,
        help="Disable automatic 800-to-1000 segment-prior decay when --evidence is active.",
    )
    parser.add_argument(
        "--no_evidence_metric_gate",
        dest="evidence_metric_gate",
        action="store_false",
        default=True,
        help="Allow surrogate validation / stopping metrics before prior decay completes.",
    )

    # Units / dimensional analysis (on by default; pass --ignore_units to skip)
    parser.add_argument(
        "--ignore_units",
        action="store_true",
        help="Disable dimensional consistency filtering (units are enforced by default when a units spec is provided).",
    )
    parser.add_argument(
        "--units",
        type=str,
        default=None,
        help='Units spec. Either JSON {"y":[...],"x":[[...],...]} or two bracket-lists "[...]" "[[...],...]".',
    )
    parser.add_argument(
        "--y_units", type=str, default=None, help="Units for y as a Python/JSON list of exponents."
    )
    parser.add_argument(
        "--x_units",
        type=str,
        default=None,
        help="Units for x variables as a Python/JSON list-of-lists of exponents.",
    )
    parser.add_argument(
        "--units_basis",
        type=str,
        default=None,
        help='Comma-separated basis names for unit exponents (e.g. "L,T,M,I,Θ"). If omitted, inferred.',
    )
    parser.add_argument(
        "--equations_txt",
        type=str,
        default=None,
        help="Path to equations.txt with y_units/x_units columns; if set and no units are passed explicitly, load units for this dataset id (CSV stem).",
    )
    parser.add_argument(
        "--free_consts",
        type=str,
        default=None,
        help="Optional mapping of trainable free constant names to unit vectors, as JSON or Python dict. Example: {\"c\":[1,-1,0]}.",
    )
    parser.add_argument(
        "--local_consts",
        type=str,
        default=None,
        help="Per-dataset trainable constants with unit vectors (scope='experiment'). JSON dict: {\"c0\":[1,0], \"c1\":[0,-1]}.",
    )
    parser.add_argument(
        "--global_consts",
        type=str,
        default=None,
        help="Shared trainable constants with unit vectors (scope='class'). JSON dict: {\"g0\":[0,-1]}.",
    )
    parser.add_argument(
        "--fixed_consts",
        type=str,
        default=None,
        help="Optional mapping of fixed physical constant names to (value, unit_vec). JSON/Python dict accepted.",
    )
    parser.add_argument(
        "--fixed_consts_mode",
        type=str,
        default="strict",
        choices=["strict", "minimal", "off"],
        help="Declared fixed-constant policy for candidate construction.",
    )
    parser.add_argument(
        "--units_policy",
        type=str,
        default="free_const_only",
        help='Units policy for checker. "free_const_only" (default) enforces only declared free constants may be unitful.',
    )
    parser.add_argument(
        "--nn_units_semantics",
        type=str,
        default="unknown",
        help='How to treat NN leaves under unit checking: "unknown" (default), "dimless", or "span".',
    )

    # Output control
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory for output files"
    )
    parser.add_argument("--save_json", action="store_true", help="Save results as JSON report")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument(
        "--de-coe-mode",
        type=str,
        choices=["off", "audit", "adjudicate", "reservoir"],
        default="off",
        help=(
            "DE Committee-of-Experts mode. 'audit' reports what the committee would "
            "select; 'adjudicate' lets the committee select the serialized DE result; "
            "'reservoir' reports support-aware committee state for benchmark-level scouts."
        ),
    )
    parser.add_argument(
        "--de-coe-reservoir-scouts",
        type=int,
        default=0,
        help=(
            "Requested number of bounded DE reservoir scouts. The benchmark runner "
            "executes trajectory-subset scouts; run_de.py records the request."
        ),
    )
    parser.add_argument(
        "--de-coe-csr-on-ties",
        action="store_true",
        help=(
            "Request late bounded continuous skeleton refinement for committee-tied "
            "DE candidates. The benchmark runner performs the rollout-gated pass."
        ),
    )

    # VarPro refinement (Phase 1)
    parser.add_argument(
        "--varpro",
        action="store_true",
        help="Refine STLSQ coefficients with Variable Projection + LM",
    )
    parser.add_argument(
        "--varpro_epochs", type=int, default=500, help="Maximum epochs for VarPro refinement"
    )
    parser.add_argument(
        "--stageb_refine_residual",
        action="store_true",
        help="Run Stage-B LM refinement on discovered residual AST(s) against zero targets.",
    )
    parser.add_argument(
        "--stageb_epochs",
        type=int,
        default=500,
        help="Maximum LM epochs for Stage-B residual refinement.",
    )

    # VarPro template search (Phase 2)
    parser.add_argument(
        "--varpro_templates",
        type=str,
        default=None,
        help="Comma-separated template families to try: power,exp,sin,saturation",
    )
    parser.add_argument(
        "--max_templates", type=int, default=3, help="Maximum number of template terms to add"
    )
    parser.add_argument(
        "--complexity_penalty", type=float, default=1e-3, help="Penalty per term in template search"
    )
    parser.add_argument(
        "--prefer_autonomous",
        action="store_true",
        help="Prefer autonomous (state-only) DE forms like u_x + f(u) = 0",
    )
    parser.add_argument(
        "--prefer_forced",
        action="store_true",
        help="Prefer forced (x-only) DE forms like u_x + g(x) = 0",
    )
    parser.add_argument(
        "--support_minimization",
        action="store_true",
        help="Enable greedy support minimization after template insertion",
    )
    parser.add_argument(
        "--rms_tol_factor",
        type=float,
        default=1.05,
        help="RMS tolerance factor for support minimization (e.g., 1.05 = 5%% worse allowed)",
    )

    # Template parameter LM (Phase 2: LM over ψ)
    parser.add_argument(
        "--template_lm",
        action="store_true",
        help="Optimize nonlinear template parameters ψ using LM (VarPro) in each Phase 2 trial",
    )
    parser.add_argument(
        "--template_lm_epochs",
        type=int,
        default=200,
        help="Max LM epochs for ψ optimization per template trial",
    )
    parser.add_argument(
        "--template_lm_epochs_min",
        type=int,
        default=20,
        help="Minimum LM epochs for ψ optimization per template trial",
    )
    parser.add_argument(
        "--template_lm_nval_patience",
        type=int,
        default=50,
        help="Patience for ψ optimization (train-as-val inside LM)",
    )
    parser.add_argument(
        "--template_lm_loss_target",
        type=float,
        default=None,
        help="Optional loss target for ψ optimization per template trial",
    )
    args = parser.parse_args()
    argv_flags = set(sys.argv[1:])
    args.gs_legacy_alias_provenance = []

    def _record_gs_legacy_alias(alias: str, effect: str) -> None:
        args.gs_legacy_alias_provenance.append(
            {"alias": str(alias), "effect": str(effect), "deprecated": True}
        )

    # Legacy GS-named hard-tail flags are neutral DE-prior aliases. Normalize
    # them once, then keep the GS machinery itself inactive unless another
    # explicit GS flag enables it.
    if "--gs-de-templates" in argv_flags and bool(getattr(args, "gs_de_templates", False)):
        args.de_hard_tail_templates = True
        args.gs_de_templates = False
        _record_gs_legacy_alias("--gs-de-templates", "--de-hard-tail-templates")
    if "--gs-de-no-radial-templates" in argv_flags:
        args.de_hard_tail_radial_templates = False
        args.gs_de_radial_templates = True
        _record_gs_legacy_alias(
            "--gs-de-no-radial-templates",
            "--de-hard-tail-no-radial-templates",
        )
    if "--gs-de-velocity-templates" in argv_flags and bool(
        getattr(args, "gs_de_velocity_templates", False)
    ):
        args.de_hard_tail_velocity_templates = True
        args.gs_de_velocity_templates = False
        _record_gs_legacy_alias(
            "--gs-de-velocity-templates",
            "--de-hard-tail-velocity-templates",
        )
    if bool(getattr(args, "gs_auto", False)):
        args.gs_enable = True
        args.gs_mode = "auto"

    def _env_truthy(name: str) -> bool:
        val = os.environ.get(name)
        if val is None:
            return False
        return str(val).strip().lower() in {"1", "true", "yes", "on", "y"}

    # Environment defaults used by examples/gs_ablation and benchmark wrappers.
    if _env_truthy("NESTYNET_GS_ENABLE") or _env_truthy("NESTYNET_SR_GS_ENABLE"):
        args.gs_enable = True
    env_mode = os.environ.get("NESTYNET_GS_MODE") or os.environ.get("NESTYNET_SR_GS_MODE")
    if env_mode:
        env_mode = str(env_mode).strip().lower()
        if env_mode in {"off", "audit", "propose", "auto"}:
            args.gs_mode = env_mode
            if env_mode != "off":
                args.gs_enable = True
            if env_mode == "auto":
                args.gs_auto = True
    if _env_truthy("NESTYNET_GS_AUTO") or _env_truthy("NESTYNET_SR_GS_AUTO"):
        args.gs_auto = True
        args.gs_enable = True
        args.gs_mode = "auto"
    if _env_truthy("NESTYNET_DE_HARD_TAIL_TEMPLATES") or _env_truthy("NESTYNET_SR_DE_HARD_TAIL_TEMPLATES"):
        args.de_hard_tail_templates = True
    if _env_truthy("NESTYNET_DE_HARD_TAIL_NO_RADIAL_TEMPLATES") or _env_truthy("NESTYNET_SR_DE_HARD_TAIL_NO_RADIAL_TEMPLATES"):
        args.de_hard_tail_radial_templates = False
    if _env_truthy("NESTYNET_DE_HARD_TAIL_VELOCITY_TEMPLATES") or _env_truthy("NESTYNET_SR_DE_HARD_TAIL_VELOCITY_TEMPLATES"):
        args.de_hard_tail_velocity_templates = True
    if _env_truthy("NESTYNET_GS_DE_TEMPLATES") or _env_truthy("NESTYNET_SR_GS_DE_TEMPLATES"):
        args.de_hard_tail_templates = True
        _record_gs_legacy_alias(
            "NESTYNET[_SR]_GS_DE_TEMPLATES",
            "NESTYNET[_SR]_DE_HARD_TAIL_TEMPLATES",
        )
    if _env_truthy("NESTYNET_GS_DE_VELOCITY_TEMPLATES") or _env_truthy("NESTYNET_SR_GS_DE_VELOCITY_TEMPLATES"):
        args.de_hard_tail_velocity_templates = True
        _record_gs_legacy_alias(
            "NESTYNET[_SR]_GS_DE_VELOCITY_TEMPLATES",
            "NESTYNET[_SR]_DE_HARD_TAIL_VELOCITY_TEMPLATES",
        )
    for env_name, attr in (
        ("NESTYNET_GS_DE_ALL_UPGRADES", "gs_de_all_upgrades"),
        ("NESTYNET_GS_DE_DETERMINING_EQUATIONS", "gs_de_determining_equations"),
        ("NESTYNET_GS_DE_CERTIFICATE", "gs_de_determining_certificate"),
        ("NESTYNET_GS_DE_CONTACT_TEMPLATES", "gs_de_contact_templates"),
        ("NESTYNET_GS_DE_NOETHER_TEMPLATES", "gs_de_noether_templates"),
        ("NESTYNET_GS_DE_DISCRETE_SYMMETRY_TEMPLATES", "gs_de_discrete_symmetry_templates"),
        ("NESTYNET_GS_DE_WEIGHTED_SCALING_TEMPLATES", "gs_de_weighted_scaling_templates"),
        ("NESTYNET_GS_DE_RADIAL_REDUCTION_TEMPLATES", "gs_de_radial_reduction_templates"),
        ("NESTYNET_GS_DE_INVARIANT_LIBRARY", "gs_de_invariant_library"),
        ("NESTYNET_GS_DE_NONLINEAR_INVARIANTS", "gs_de_nonlinear_invariants"),
    ):
        if _env_truthy(env_name) or _env_truthy(env_name.replace("NESTYNET_GS_", "NESTYNET_SR_GS_")):
            args.gs_enable = True
            setattr(args, attr, True)
    if _env_truthy("NESTYNET_GS_DE_LIE_PROLONGATION") or _env_truthy("NESTYNET_SR_GS_DE_LIE_PROLONGATION"):
        args.gs_enable = True
        args.gs_de_lie_prolongation = True
    if _env_truthy("NESTYNET_GS_DE_LIE_USE_FOR_SELECTION") or _env_truthy("NESTYNET_SR_GS_DE_LIE_USE_FOR_SELECTION"):
        args.gs_enable = True
        args.gs_de_lie_use_for_selection = True
    env_lie_weight = os.environ.get("NESTYNET_GS_DE_LIE_PROLONGATION_WEIGHT") or os.environ.get("NESTYNET_SR_GS_DE_LIE_PROLONGATION_WEIGHT")
    if env_lie_weight:
        try:
            args.gs_de_lie_prolongation_weight = float(env_lie_weight)
        except Exception:
            pass
    env_lie_tol = os.environ.get("NESTYNET_GS_DE_LIE_PROLONGATION_TOL") or os.environ.get("NESTYNET_SR_GS_DE_LIE_PROLONGATION_TOL")
    if env_lie_tol:
        try:
            args.gs_de_lie_prolongation_tol = float(env_lie_tol)
        except Exception:
            pass
    env_lie_samples = os.environ.get("NESTYNET_GS_DE_LIE_PROLONGATION_MAX_SAMPLES") or os.environ.get("NESTYNET_SR_GS_DE_LIE_PROLONGATION_MAX_SAMPLES")
    if env_lie_samples:
        try:
            args.gs_de_lie_prolongation_max_samples = int(env_lie_samples)
        except Exception:
            pass
    env_lie_coverage = os.environ.get("NESTYNET_GS_DE_LIE_PROLONGATION_MIN_COVERAGE") or os.environ.get("NESTYNET_SR_GS_DE_LIE_PROLONGATION_MIN_COVERAGE")
    if env_lie_coverage:
        try:
            args.gs_de_lie_prolongation_min_coverage = float(env_lie_coverage)
        except Exception:
            pass
    if _env_truthy("NESTYNET_GS_UNIT_TORUS") or _env_truthy("NESTYNET_SR_GS_UNIT_TORUS"):
        args.gs_enable = True
        args.gs_unit_torus = True
    if _env_truthy("NESTYNET_GS_NO_UNIT_TORUS") or _env_truthy("NESTYNET_SR_GS_NO_UNIT_TORUS"):
        args.gs_unit_torus = False
    if _env_truthy("NESTYNET_GS_PI_INVARIANTS") or _env_truthy("NESTYNET_SR_GS_PI_INVARIANTS"):
        args.gs_enable = True
        args.gs_unit_torus = True
        args.gs_pi_invariants = True
    if _env_truthy("NESTYNET_GS_NO_PI_INVARIANTS") or _env_truthy("NESTYNET_SR_GS_NO_PI_INVARIANTS"):
        args.gs_pi_invariants = False
    env_dim_policy = os.environ.get("NESTYNET_GS_DIM_POLICY") or os.environ.get("NESTYNET_SR_GS_DIM_POLICY")
    if env_dim_policy:
        env_dim_policy = str(env_dim_policy).strip().lower().replace("_", "-")
        if env_dim_policy in {"baseline", "audit", "augment", "both", "replace-rref", "gs-only"}:
            args.gs_dim_policy = env_dim_policy
            if env_dim_policy not in {"baseline", "audit"}:
                args.gs_enable = True
                args.gs_unit_torus = True
    env_dim_both_rule = os.environ.get("NESTYNET_GS_DIM_BOTH_RULE") or os.environ.get("NESTYNET_SR_GS_DIM_BOTH_RULE")
    if env_dim_both_rule:
        env_dim_both_rule = str(env_dim_both_rule).strip().lower().replace("_", "-")
        if env_dim_both_rule in {"rref-dominates", "require-both", "either", "gs-dominates"}:
            args.gs_dim_both_rule = env_dim_both_rule
    env_dim_validator = os.environ.get("NESTYNET_GS_DIM_VALIDATOR") or os.environ.get("NESTYNET_SR_GS_DIM_VALIDATOR")
    if env_dim_validator:
        env_dim_validator = str(env_dim_validator).strip().lower().replace("_", "-")
        if env_dim_validator in {"local", "nullspace", "linear"}:
            args.gs_dim_validator = env_dim_validator
    for env_name, attr, cast in (
        ("NESTYNET_GS_PI_MAX_EXPONENT", "gs_pi_max_exponent", int),
        ("NESTYNET_GS_PI_MAX_L1", "gs_pi_max_l1", int),
        ("NESTYNET_GS_PI_MAX_PROPOSALS", "gs_pi_max_proposals", int),
        ("NESTYNET_GS_PI_MAX_BASIS", "gs_pi_max_basis", int),
        ("NESTYNET_GS_PI_RATIONAL_DENOM", "gs_pi_rational_denom", int),
        ("NESTYNET_GS_DE_UPGRADE_MAX_TERMS", "gs_de_upgrade_max_terms", int),
        ("NESTYNET_GS_DE_DETERMINING_MAX_DEGREE", "gs_de_determining_max_degree", int),
        ("NESTYNET_GS_DE_DETERMINING_MAX_GENERATORS", "gs_de_determining_max_generators", int),
        ("NESTYNET_GS_DE_DETERMINING_MULTIPLIER_DEGREE", "gs_de_determining_multiplier_degree", int),
        ("NESTYNET_GS_DE_DETERMINING_BOOTSTRAPS", "gs_de_determining_bootstraps", int),
        ("NESTYNET_GS_DE_NONLINEAR_INVARIANT_MAX_DEGREE", "gs_de_nonlinear_invariant_max_degree", int),
        ("NESTYNET_GS_DE_NONLINEAR_INVARIANT_MAX_CANDIDATES", "gs_de_nonlinear_invariant_max_candidates", int),
        ("NESTYNET_GS_DE_WEIGHTED_MAX_ABS_X_POWER", "gs_de_weighted_max_abs_x_power", int),
        ("NESTYNET_GS_DE_WEIGHTED_MAX_U_POWER", "gs_de_weighted_max_u_power", int),
        ("NESTYNET_GS_DE_WEIGHTED_MAX_DU_POWER", "gs_de_weighted_max_du_power", int),
    ):
        value = os.environ.get(env_name) or os.environ.get(env_name.replace("NESTYNET_GS_", "NESTYNET_SR_GS_"))
        if value not in (None, ""):
            try:
                setattr(args, attr, cast(value))
            except Exception:
                pass
    if _env_truthy("NESTYNET_GS_GENERAL_AFFINE") or _env_truthy("NESTYNET_SR_GS_GENERAL_AFFINE"):
        args.gs_enable = True
        args.gs_general_affine = True
    if _env_truthy("NESTYNET_GS_NO_KNOWN_GENERATORS") or _env_truthy("NESTYNET_SR_GS_NO_KNOWN_GENERATORS"):
        args.gs_known_generators = False
    if _env_truthy("NESTYNET_GS_KNOWN_GENERATORS") or _env_truthy("NESTYNET_SR_GS_KNOWN_GENERATORS"):
        args.gs_known_generators = True
    if _env_truthy("NESTYNET_GS_NO_JET") or _env_truthy("NESTYNET_SR_GS_NO_JET"):
        args.gs_jet_enable = False
    if _env_truthy("NESTYNET_GS_LORENTZ_BOOSTS"):
        args.gs_lorentz_boosts = True
    env_policy = os.environ.get("NESTYNET_GS_POLICY") or os.environ.get("NESTYNET_SR_GS_POLICY")
    if env_policy:
        env_policy = str(env_policy).strip().lower().replace("_", "-")
        if env_policy in {"augment", "replace-shadowed", "gs-only-affine"}:
            args.gs_policy = env_policy
            if env_policy != "augment":
                args.gs_enable = True
    if bool(getattr(args, "gs_auto", False)):
        args.gs_enable = True
        args.gs_mode = "auto"
    if bool(getattr(args, "gs_de_all_upgrades", False)):
        args.gs_enable = True
        args.gs_de_determining_equations = True
        args.gs_de_contact_templates = True
        args.gs_de_noether_templates = True
        args.gs_de_discrete_symmetry_templates = True
        args.gs_de_weighted_scaling_templates = True
        args.gs_de_radial_reduction_templates = True
        args.gs_de_invariant_library = True
        args.gs_de_nonlinear_invariants = True
        args.gs_de_lie_prolongation = True
    if bool(getattr(args, "gs_de_determining_equations", False)):
        args.gs_enable = True
        args.gs_de_lie_prolongation = True
    if bool(getattr(args, "gs_de_nonlinear_invariants", False)):
        args.gs_enable = True
        args.gs_de_determining_equations = True
        args.gs_de_determining_certificate = True
        args.gs_de_lie_prolongation = True
    if bool(getattr(args, "gs_pi_invariants", False)):
        args.gs_unit_torus = True
    if bool(getattr(args, "gs_unit_torus", False)):
        args.gs_enable = True
    if bool(getattr(args, "gs_de_lie_prolongation", False)):
        args.gs_enable = True
    if str(getattr(args, "gs_mode", "propose") or "propose").lower() == "off":
        args.gs_enable = False
    if (
        bool(getattr(args, "gs_enable", False))
        and bool(getattr(args, "gs_de_lie_use_for_selection", False))
        and not bool(getattr(args, "gs_de_lie_prolongation", False))
    ):
        parser.error(
            "--gs-de-lie-use-for-selection requires --gs-de-lie-prolongation; "
            "selection without an active scorer is invalid"
        )

    gs_env_names = sorted(
        name
        for name, value in os.environ.items()
        if (
            name.startswith("NESTYNET_GS_")
            or name.startswith("NESTYNET_SR_GS_")
        )
        and value not in (None, "")
    )
    args.gs_env_activation_provenance = [
        {"name": name, "value": str(os.environ.get(name, ""))[:120]}
        for name in gs_env_names
    ]
    args.gs_activation_provenance = list(getattr(args, "gs_env_activation_provenance", []) or [])
    if args.gs_legacy_alias_provenance:
        args.gs_activation_provenance.extend(
            {"legacy_alias": row["alias"], "effect": row["effect"], "deprecated": True}
            for row in args.gs_legacy_alias_provenance
        )

    # Translate --ignore_units → enforce_units for all internal code
    args.enforce_units = not args.ignore_units
    return args


def _parse_py_or_json_literal(s):
    """Parse a Python literal (via ast.literal_eval) or JSON string."""
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception as e:
        raise ValueError(f"Failed to parse literal: {s!r} ({e})")


def _extract_bracketed_from_start(s: str):
    """Extract the first balanced [...] substring from the start of s."""
    s = str(s).lstrip()
    if not s.startswith("["):
        raise ValueError("Expected '[' at start of bracketed literal")
    level = 0
    for i, ch in enumerate(s):
        if ch == "[":
            level += 1
        elif ch == "]":
            level -= 1
            if level == 0:
                return s[: i + 1], s[i + 1 :]
    raise ValueError("Unbalanced brackets in literal")


def _extract_last_bracketed(s: str):
    """Extract the last balanced [...] substring from s (supports nested lists)."""
    s = str(s)
    i = s.rfind("]")
    if i < 0:
        raise ValueError("No closing ']' found")
    level = 0
    for j in range(i, -1, -1):
        ch = s[j]
        if ch == "]":
            level += 1
        elif ch == "[":
            level -= 1
            if level == 0:
                return s[j : i + 1], s[:j].rstrip()
    raise ValueError("Unbalanced brackets when scanning from end")


def _infer_units_basis(n: int, basis_arg=None):
    """Infer basis labels for unit vectors of length n."""
    if basis_arg is not None:
        if isinstance(basis_arg, (list, tuple)):
            parts = [str(p) for p in basis_arg]
        else:
            parts = [p.strip() for p in str(basis_arg).split(",") if p.strip()]
        if len(parts) != int(n):
            raise ValueError(f"units_basis has length {len(parts)} but expected {n}")
        return tuple(parts)
    if int(n) == 5:
        return ("L", "T", "M", "I", "Θ")
    if int(n) == 7:
        return ("L", "M", "T", "I", "Θ", "N", "J")
    return tuple(f"d{i}" for i in range(int(n)))


def _parse_units_arg(units_str: str):
    """Parse --units into (y_units_vec, x_units_mat, basis or None)."""
    if units_str is None:
        return None, None, None
    parsed = _parse_py_or_json_literal(units_str)
    if isinstance(parsed, dict):
        y = parsed.get("y", parsed.get("y_units"))
        x = parsed.get("x", parsed.get("x_units"))
        basis = parsed.get("basis", parsed.get("units_basis"))
        return y, x, basis
    if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
        return parsed[0], parsed[1], None
    y_s, rest = _extract_bracketed_from_start(units_str)
    x_s, _ = _extract_bracketed_from_start(rest)
    y = _parse_py_or_json_literal(y_s)
    x = _parse_py_or_json_literal(x_s)
    return y, x, None


def _load_units_from_equations(path: str, eq_id: str):
    """Load (y_units, x_units) for an equation id from equations.txt."""
    eq_id = str(eq_id).strip()
    match = re.match(r"pb(\d+)", eq_id)
    numeric_id = match.group(1) if match else eq_id

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            tok = line.split(None, 1)[0]
            if tok == eq_id or tok == numeric_id:
                x_s, rem = _extract_last_bracketed(line)
                y_s, _ = _extract_last_bracketed(rem)
                y = _parse_py_or_json_literal(y_s)
                x = _parse_py_or_json_literal(x_s)
                return y, x

    raise ValueError(f"Could not find units for id '{eq_id}' (tried '{numeric_id}') in {path}")


def _apply_de_refine_mode(hp, mode: str):
    mode_l = str(mode or "rare_final_polish").strip().lower().replace("-", "_")
    if mode_l == "off":
        hp.refine_profile = "off"
        hp.refine_enable = False
        return apply_refine_mode_placement_defaults(hp, "off")
    return apply_refine_profile(hp, mode_l)


def build_factorized_search_rescue_config_from_args(args) -> FactorizedSearchDERescueConfig:
    """Construct the factorized symbolic search rescue config from CLI arguments."""
    hp = default_physics_rescue_hp(preset=str(getattr(args, "factorized_search_preset", "default")))
    refine_mode = str(getattr(args, "factorized_search_de_refine_mode", "rare_final_polish") or "rare_final_polish")
    hp = _apply_de_refine_mode(hp, refine_mode)

    overrides = {
        "n_iter": getattr(args, "factorized_search_n_iter", None),
        "max_depth": getattr(args, "factorized_search_max_depth", None),
        "n_fit": getattr(args, "factorized_search_n_fit", None),
        "n_probe": getattr(args, "factorized_search_n_probe", None),
        "return_topk": getattr(args, "factorized_search_return_topk", None),
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(hp, name, int(value))

    integrate_topk_arg = getattr(args, "factorized_search_integrate_topk", None)
    legacy_integrate_topk = max(0, min(8, int(getattr(hp, "return_topk", 0))))
    validate_integrate_topk = legacy_integrate_topk
    if integrate_topk_arg is not None:
        validate_integrate_topk = max(0, int(integrate_topk_arg))
    max_attempts_arg = getattr(args, "factorized_search_max_attempts", None)
    max_attempts = None if max_attempts_arg is None else max(0, int(max_attempts_arg))

    return FactorizedSearchDERescueConfig(
        mode=str(getattr(args, "factorized_search_rescue", "never")),
        trigger_val_rms=float(getattr(args, "factorized_search_trigger_val_rms", 1.0e-3)),
        trigger_rel_rms=float(getattr(args, "factorized_search_trigger_rel_rms", 1.0e-3)),
        trigger_cond=float(getattr(args, "factorized_search_trigger_cond", 1.0e8)),
        replace_rel_factor=float(getattr(args, "factorized_search_replace_rel_factor", 0.98)),
        strict_shared_rhs=True,
        validate_integrate_topk=validate_integrate_topk,
        budget_scope=str(getattr(args, "factorized_search_budget_scope", None) or "per_group"),
        max_attempts=max_attempts,
        coefficient_dim_mode=str(getattr(args, "factorized_search_coefficient_dim_mode", "strict_expression")),
        direct_generator_witness_topk=max(
            0,
            int(getattr(args, "factorized_search_direct_generator_witness_topk", 1) or 0),
        ),
        regularized_implicit_clean_score=float(
            getattr(args, "factorized_implicit_clean_score", 5.0e-3)
        ),
        clean_gate_val_rms=getattr(args, "factorized_clean_gate_val_rms", None),
        clean_gate_rel_rms=getattr(args, "factorized_clean_gate_rel_rms", None),
        hp=hp,
    )


def build_factorized_rescue_config_from_args(args) -> FactorizedDERescueConfig:
    """Construct the typed factorized rescue config from CLI arguments."""
    hp = default_physics_rescue_hp(preset=str(getattr(args, "factorized_search_preset", "default")))
    refine_mode = str(getattr(args, "factorized_search_de_refine_mode", "rare_final_polish") or "rare_final_polish")
    hp = _apply_de_refine_mode(hp, refine_mode)
    overrides = {
        "n_iter": getattr(args, "factorized_search_n_iter", None),
        "max_depth": getattr(args, "factorized_search_max_depth", None),
        "n_fit": getattr(args, "factorized_search_n_fit", None),
        "n_probe": getattr(args, "factorized_search_n_probe", None),
        "return_topk": getattr(args, "factorized_search_return_topk", None),
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(hp, name, int(value))
    return FactorizedDERescueConfig(
        mode=str(getattr(args, "factorized_rescue", "never")),
        trigger_val_rms=float(getattr(args, "factorized_search_trigger_val_rms", 1.0e-3)),
        trigger_cond=float(getattr(args, "factorized_search_trigger_cond", 1.0e8)),
        replace_rel_factor=float(getattr(args, "factorized_search_replace_rel_factor", 0.98)),
        two_block_shared_coord_mode=str(getattr(args, "factorized_two_block_shared_coord", "never")),
        typed_lane_workers=max(1, int(getattr(args, "factorized_de_typed_lane_workers", 1) or 1)),
        hp=hp,
    )


def _feature_group_rescue_cfg_from_factorized(
    factorized_cfg: FactorizedDERescueConfig,
) -> FactorizedSearchDERescueConfig:
    """Mirror factorized rescue sampling settings into feature-group preparation."""
    return FactorizedSearchDERescueConfig(
        mode="never",
        trigger_val_rms=float(getattr(factorized_cfg, "trigger_val_rms", 1.0e-3)),
        trigger_rel_rms=float(getattr(factorized_cfg, "trigger_rel_rms", 1.0e-3)),
        trigger_cond=float(getattr(factorized_cfg, "trigger_cond", 1.0e8)),
        replace_rel_factor=float(getattr(factorized_cfg, "replace_rel_factor", 0.98)),
        strict_shared_rhs=True,
        validate_integrate_topk=0,
        budget_scope="per_group",
        max_attempts=None,
        hp=copy.deepcopy(getattr(factorized_cfg, "hp", None)),
    )


def _as_factorized_de_factorized_config(
    factorized_cfg: FactorizedDERescueConfig,
) -> FactorizedDERescueConfig:
    """Force the factorized lane into first-line, zero-base operator-factorized DE mode."""
    out = copy.deepcopy(factorized_cfg)
    out.mode = "always"
    out.base_modes = ("zero",)
    return out


def _as_factorized_de_rescue_config(
    rescue_cfg: FactorizedSearchDERescueConfig,
) -> FactorizedSearchDERescueConfig:
    """Force the whole-RHS factorized symbolic search lane into first-line mode."""
    out = copy.deepcopy(rescue_cfg)
    out.mode = "always"
    out.coefficient_dim_mode = "inferred_outer"
    return out


def _factorized_de_whole_rhs_decision(
    factorized_res,
    rescue_cfg: FactorizedSearchDERescueConfig,
    *,
    policy: str,
) -> tuple[bool, dict[str, Any]]:
    """Decide whether the broad whole-RHS FSS lane should run in factorized-de."""

    policy_l = str(policy or "auto").strip().lower()
    if policy_l not in {"never", "auto", "always"}:
        raise ValueError(f"unknown factorized-de whole-RHS policy: {policy!r}")

    rms = candidate_probe_rms(factorized_res)
    diagnostics = getattr(factorized_res, "diagnostics", {}) or {}
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    threshold = float(getattr(rescue_cfg, "trigger_val_rms", 1.0e-3))
    if not math.isfinite(threshold) or threshold <= 0.0:
        threshold = 1.0e-3
    relaxed_threshold = 10.0 * threshold
    rel_threshold = float(getattr(rescue_cfg, "trigger_rel_rms", 1.0e-3))
    if not math.isfinite(rel_threshold) or rel_threshold <= 0.0:
        rel_threshold = 1.0e-3
    relaxed_rel_threshold = 10.0 * rel_threshold
    try:
        rel_rms = float(diagnostics.get("probe_rel_rms", float("inf")))
    except Exception:
        rel_rms = float("inf")

    decision = {
        "policy": policy_l,
        "run": False,
        "reason": "",
        "typed_probe_rms": float(rms) if math.isfinite(rms) else None,
        "typed_probe_rel_rms": float(rel_rms) if math.isfinite(rel_rms) else None,
        "trigger_val_rms": float(threshold),
        "trigger_rel_rms": float(rel_threshold),
        "relaxed_trigger_val_rms": float(relaxed_threshold),
        "relaxed_trigger_rel_rms": float(relaxed_rel_threshold),
        "budget_scope": str(getattr(rescue_cfg, "budget_scope", "per_group")),
        "max_attempts": _factorized_search_max_attempts(rescue_cfg),
        "typed_domain_ok": diagnostics.get("domain_ok", None),
        "typed_integrate_ok": diagnostics.get("integrate_ok", None),
        "typed_evidence_tier": diagnostics.get("evidence_tier", None),
        "typed_generator_status": diagnostics.get("generator_status", None),
        "typed_consistency_score": diagnostics.get("consistency_score", None),
    }

    if policy_l == "never":
        decision["run"] = False
        decision["reason"] = "policy_never"
        return False, decision
    if _factorized_search_max_attempts(rescue_cfg) == 0:
        decision["run"] = False
        decision["reason"] = "max_attempts_zero"
        return False, decision
    if policy_l == "always":
        decision["run"] = True
        decision["reason"] = "policy_always"
        return True, decision

    if factorized_res is None:
        decision["run"] = True
        decision["reason"] = "typed_lane_empty"
        return True, decision
    generator_status = str(diagnostics.get("generator_status", "") or "").strip().upper()
    if not generator_status:
        generator_witness = diagnostics.get("generator_witness", None)
        if isinstance(generator_witness, dict):
            generator_status = str(generator_witness.get("generator_status", "") or "").strip().upper()
    if _is_exact_structural_generator_status(generator_status):
        decision["run"] = False
        decision["reason"] = "generator_witness_pass"
        decision["typed_generator_status"] = generator_status
        return False, decision
    if not math.isfinite(rms):
        decision["run"] = True
        decision["reason"] = "typed_probe_rms_nonfinite"
        return True, decision
    if diagnostics.get("domain_ok", None) is False:
        decision["run"] = True
        decision["reason"] = "typed_domain_failed"
        return True, decision
    if diagnostics.get("integrate_ok", None) is False:
        decision["run"] = True
        decision["reason"] = "typed_integrate_failed"
        return True, decision

    if rms <= threshold:
        decision["run"] = False
        decision["reason"] = "typed_probe_rms_pass"
        return False, decision
    if math.isfinite(rel_rms) and rel_rms <= rel_threshold:
        decision["run"] = False
        decision["reason"] = "typed_probe_rel_rms_pass"
        return False, decision

    evidence_tier = str(diagnostics.get("evidence_tier", "") or "").strip().lower()
    evidence_verified = evidence_tier == "verified" or "witness" in evidence_tier
    consistency_good = False
    try:
        consistency_score = float(diagnostics.get("consistency_score", float("inf")))
        consistency_good = math.isfinite(consistency_score) and consistency_score <= max(1.0e-6, threshold * threshold)
    except Exception:
        consistency_good = False
    relaxed_quality_ok = bool(
        rms <= relaxed_threshold
        or (math.isfinite(rel_rms) and rel_rms <= relaxed_rel_threshold)
    )
    if relaxed_quality_ok and (evidence_verified or consistency_good):
        decision["run"] = False
        decision["reason"] = "typed_witness_consistent"
        return False, decision

    decision["run"] = True
    decision["reason"] = "typed_candidate_ambiguous"
    return True, decision


def _mapping_domain_projection_ok(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return True
    if value.get("ok", None) is False:
        return False
    for key in ("domain_projection", "domain_projection_eval", "_domain_projection"):
        child = value.get(key, None)
        if isinstance(child, Mapping) and child.get("ok", None) is False:
            return False
    return True


def _mapping_structural_ok(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return True
    if value.get("structural_ok", None) is False:
        return False
    if value.get("structural_hard_reject", None) is True:
        return False
    if value.get("hidden_score_head", None) is True:
        return False
    return True


def _candidate_domain_safe(obj: Any) -> bool:
    if obj is None:
        return False
    if getattr(obj, "structural_ok", None) is False:
        return False
    diagnostics = getattr(obj, "diagnostics", {}) or {}
    if isinstance(diagnostics, Mapping):
        if not _mapping_structural_ok(diagnostics):
            return False
        if diagnostics.get("domain_ok", None) is False:
            return False
        if not _mapping_domain_projection_ok(diagnostics):
            return False
        report = diagnostics.get("report", None)
        if isinstance(report, Mapping):
            best = report.get("best", None)
            if isinstance(best, Mapping) and not _mapping_structural_ok(best):
                return False
    mapping = getattr(obj, "mapping", {}) or {}
    if isinstance(mapping, Mapping) and not _mapping_domain_projection_ok(mapping):
        return False
    return True


def _result_generator_status(result: Any) -> str:
    diagnostics = getattr(result, "diagnostics", {}) or {}
    if not isinstance(diagnostics, Mapping):
        return ""
    status = str(diagnostics.get("generator_status", "") or "").strip().upper()
    if status:
        return status
    witness = diagnostics.get("generator_witness", None)
    if isinstance(witness, Mapping):
        return str(witness.get("generator_status", "") or "").strip().upper()
    direct = diagnostics.get("direct_residual_attempt", None)
    if isinstance(direct, Mapping):
        status = str(direct.get("generator_status", "") or "").strip().upper()
        if status:
            return status
        witness = direct.get("generator_witness", None)
        if isinstance(witness, Mapping):
            return str(witness.get("generator_status", "") or "").strip().upper()
    return ""


def _is_exact_structural_generator_status(status: str) -> bool:
    return str(status or "").strip().upper() in {"EXACT_STRUCTURAL_GENERATOR"}


def _is_dynamic_or_ambiguous_generator_status(status: str) -> bool:
    return str(status or "").strip().upper() in {
        "DYNAMICALLY_COMPATIBLE",
        "AMBIGUOUS_ROLE",
        "VIABLE_WITH_MODELLING_ERROR",
        "AMBIGUOUS_UNDER_ERROR_MODEL",
    }


def _is_rollout_promotable_generator_status(status: str) -> bool:
    return str(status or "").strip().upper() in {
        "EXACT_STRUCTURAL_GENERATOR",
        "DYNAMICALLY_COMPATIBLE",
        "VIABLE_WITH_MODELLING_ERROR",
    }


def _result_probe_rel_rms(result: Any) -> float:
    diagnostics = getattr(result, "diagnostics", {}) or {}
    if not isinstance(diagnostics, Mapping):
        return float("inf")
    for container in (diagnostics, diagnostics.get("direct_residual_attempt", None)):
        if not isinstance(container, Mapping):
            continue
        try:
            rel = float(container.get("probe_rel_rms", float("inf")))
        except Exception:
            rel = float("inf")
        if math.isfinite(rel):
            return float(rel)
    return float("inf")


def _clean_regularized_implicit_invariant_result(
    result: Any,
    rescue_cfg: FactorizedSearchDERescueConfig,
) -> bool:
    diagnostics = getattr(result, "diagnostics", {}) or {}
    if not isinstance(diagnostics, Mapping):
        return False
    if str(diagnostics.get("candidate_source", "") or "") != "regularized_implicit_residual":
        return False
    implicit = diagnostics.get("implicit_residual", None)
    if not isinstance(implicit, Mapping):
        return False
    if str(implicit.get("b_coeff_source", "") or "") != "separable_invariant_refit":
        return False

    a_expr = str(implicit.get("a_expr", "") or "")
    b_exprs = list(implicit.get("b_exprs", []) or [])
    if a_expr not in {"1", "x", "x0"}:
        return False
    if len(b_exprs) != 1 or str(b_exprs[0]) not in {"1", "u"}:
        return False

    inv = implicit.get("invariant_refit", None)
    if not isinstance(inv, Mapping):
        return False
    if str(inv.get("kind", "") or "") != "separable_invariant":
        return False

    def _float(value: Any, default: float = float("inf")) -> float:
        try:
            out = float(value)
        except Exception:
            return float(default)
        return float(out) if math.isfinite(out) else float(default)

    accept_score = _float(
        getattr(rescue_cfg, "regularized_implicit_invariant_accept_score", 1.0e-2),
        1.0e-2,
    )
    max_traj_score = _float(
        getattr(rescue_cfg, "regularized_implicit_invariant_max_traj_score", 5.0e-2),
        5.0e-2,
    )
    max_coeff_spread = _float(
        getattr(rescue_cfg, "regularized_implicit_invariant_max_coeff_spread_rel", 1.0e-1),
        1.0e-1,
    )
    min_abs_coeff = _float(
        getattr(rescue_cfg, "regularized_implicit_invariant_min_abs_coeff", 1.0e-10),
        1.0e-10,
    )
    max_abs_coeff = _float(
        getattr(rescue_cfg, "regularized_implicit_invariant_max_abs_coeff", 1.0e6),
        1.0e6,
    )

    probe_score = _float(inv.get("probe_score", implicit.get("normalized_probe_score", float("inf"))))
    if not math.isfinite(probe_score) or probe_score > accept_score:
        return False
    coeffs = list(inv.get("coeffs", implicit.get("b_coeffs", [])) or [])
    if len(coeffs) != 1:
        return False
    coeff = _float(coeffs[0])
    if not math.isfinite(coeff):
        return False
    if abs(coeff) < min_abs_coeff or abs(coeff) > max_abs_coeff:
        return False

    for spread_key in ("fit_coeff_spread_rel", "probe_coeff_spread_rel"):
        spread = _float(inv.get(spread_key, float("inf")))
        if not math.isfinite(spread) or spread > max_coeff_spread:
            return False

    probe_traj = list(inv.get("probe_traj", []) or [])
    if not probe_traj:
        return False
    for row in probe_traj:
        if not isinstance(row, Mapping):
            return False
        score_i = _float(row.get("score", float("inf")))
        if not math.isfinite(score_i) or score_i > max_traj_score:
            return False

    multiplier = implicit.get("multiplier", None)
    if isinstance(multiplier, Mapping):
        if multiplier.get("ok", None) is False:
            return False
        if a_expr in {"x", "x0"}:
            if multiplier.get("sign_ok", None) is False:
                return False
            nonzero = _float(multiplier.get("nonzero_frac", 0.0), 0.0)
            if nonzero < 0.995:
                return False
    return True


def _clean_regularized_implicit_linear_result(
    result: Any,
    rescue_cfg: FactorizedSearchDERescueConfig,
) -> bool:
    """Certify a regularized implicit linear candidate as clean.

    Unlike the single-term invariant certificate, this accepts multi-term ``B``
    libraries (damped oscillators, Lane-Emden n=1, Bessel J0, ...) when the
    normalized derivative-residual probe score beats
    ``regularized_implicit_clean_score`` without overfit or degenerate
    coefficients.
    """
    diagnostics = getattr(result, "diagnostics", {}) or {}
    if not isinstance(diagnostics, Mapping):
        return False
    if str(diagnostics.get("candidate_source", "") or "") != "regularized_implicit_residual":
        return False
    implicit = diagnostics.get("implicit_residual", None)
    if not isinstance(implicit, Mapping):
        return False

    def _float(value: Any, default: float = float("inf")) -> float:
        try:
            out = float(value)
        except Exception:
            return float(default)
        return float(out) if math.isfinite(out) else float(default)

    clean_score = _float(getattr(rescue_cfg, "regularized_implicit_clean_score", 5.0e-3), 0.0)
    if not (clean_score > 0.0):
        return False
    probe_score = _float(implicit.get("normalized_probe_score", float("inf")))
    fit_score = _float(implicit.get("normalized_fit_score", float("inf")))
    if probe_score > clean_score or not math.isfinite(fit_score):
        return False
    overfit_ratio = _float(
        getattr(rescue_cfg, "regularized_implicit_clean_max_overfit_ratio", 5.0), 5.0
    )
    if probe_score > overfit_ratio * max(fit_score, 1.0e-12):
        return False

    b_exprs = list(implicit.get("b_exprs", []) or [])
    coeffs = list(implicit.get("b_coeffs", []) or [])
    if not b_exprs or len(coeffs) != len(b_exprs):
        return False
    min_abs_coeff = _float(
        getattr(rescue_cfg, "regularized_implicit_invariant_min_abs_coeff", 1.0e-10), 1.0e-10
    )
    max_abs_coeff = _float(
        getattr(rescue_cfg, "regularized_implicit_invariant_max_abs_coeff", 1.0e6), 1.0e6
    )
    for coeff in coeffs:
        c = _float(coeff)
        if not math.isfinite(c) or abs(c) < min_abs_coeff or abs(c) > max_abs_coeff:
            return False

    multiplier = implicit.get("multiplier", None)
    if isinstance(multiplier, Mapping):
        if multiplier.get("ok", None) is False:
            return False
        a_expr = str(implicit.get("a_expr", "") or "")
        if a_expr in {"x", "x0"}:
            if multiplier.get("sign_ok", None) is False:
                return False
            nonzero = _float(multiplier.get("nonzero_frac", 0.0), 0.0)
            if nonzero < 0.995:
                return False
    return True


def _first_line_certified(result: Any, rescue_cfg: FactorizedSearchDERescueConfig) -> bool:
    """True when a first-line candidate carries structural evidence of exactness.

    Certified candidates (exact structural generator witness, or a certified
    regularized implicit linear law) should only be displaced by challenger
    lanes that are strictly better, never on the cleaner-lane rank bonus.
    """
    if result is None:
        return False
    if _is_exact_structural_generator_status(_result_generator_status(result)):
        return True
    if _clean_regularized_implicit_invariant_result(result, rescue_cfg):
        return True
    if _clean_regularized_implicit_linear_result(result, rescue_cfg):
        return True
    return False


def _clean_gate_threshold(override: Any, base: float) -> float:
    try:
        value = float(override)
    except Exception:
        return base
    if math.isfinite(value) and value > 0.0:
        return value
    return base


def _direct_result_needs_typed_lane(
    direct_res: Any,
    rescue_cfg: FactorizedSearchDERescueConfig,
) -> bool:
    if direct_res is None:
        return True
    if not _candidate_domain_safe(direct_res):
        return True
    rms = candidate_probe_rms(direct_res)
    if not math.isfinite(rms):
        return True
    rel = _result_probe_rel_rms(direct_res)
    generator_status = _result_generator_status(direct_res)
    if _is_exact_structural_generator_status(generator_status):
        return False
    if _clean_regularized_implicit_invariant_result(direct_res, rescue_cfg):
        return False
    if _clean_regularized_implicit_linear_result(direct_res, rescue_cfg):
        return False
    if _is_dynamic_or_ambiguous_generator_status(generator_status):
        return True
    trigger = float(getattr(rescue_cfg, "trigger_val_rms", 1.0e-3))
    if not math.isfinite(trigger) or trigger <= 0.0:
        trigger = 1.0e-3
    trigger = _clean_gate_threshold(getattr(rescue_cfg, "clean_gate_val_rms", None), trigger)
    rel_trigger = float(getattr(rescue_cfg, "trigger_rel_rms", 1.0e-3))
    if not math.isfinite(rel_trigger) or rel_trigger <= 0.0:
        rel_trigger = 1.0e-3
    rel_trigger = _clean_gate_threshold(getattr(rescue_cfg, "clean_gate_rel_rms", None), rel_trigger)
    if rms <= trigger:
        return False
    if math.isfinite(rel) and rel <= rel_trigger:
        return False
    return True


def _factorized_de_lane_rank(result: Any, lane_name: str) -> int:
    if not _candidate_domain_safe(result):
        return 99
    generator_status = _result_generator_status(result)
    if _is_exact_structural_generator_status(generator_status):
        return 0
    lane = str(lane_name or "").strip()
    if lane == "factorized":
        return 0
    if lane == "regularized_implicit_residual":
        return 1
    if lane == "direct_residual_fss":
        return 1
    if lane == "factorized_search":
        return 2
    return 3


def _factorized_de_preferred(
    lhs: Any,
    lhs_lane: str,
    rhs: Any,
    rhs_lane: str,
    *,
    same_lane_rel_factor: float = 1.0,
    cleaner_lane_factor: float = 2.0,
    dirtier_lane_factor: float = 0.5,
) -> bool:
    if lhs is None or not _candidate_domain_safe(lhs):
        return False
    if rhs is None or not _candidate_domain_safe(rhs):
        return True
    lhs_rms = candidate_probe_rms(lhs)
    rhs_rms = candidate_probe_rms(rhs)
    if not math.isfinite(lhs_rms):
        return False
    if not math.isfinite(rhs_rms):
        return True
    lhs_rank = _factorized_de_lane_rank(lhs, lhs_lane)
    rhs_rank = _factorized_de_lane_rank(rhs, rhs_lane)
    if lhs_rank < rhs_rank:
        return bool(lhs_rms <= float(cleaner_lane_factor) * rhs_rms)
    if lhs_rank > rhs_rank:
        return bool(lhs_rms < float(dirtier_lane_factor) * rhs_rms)
    return bool(lhs_rms < float(same_lane_rel_factor) * rhs_rms)


def _jsonable_report_value(value):
    if torch.is_tensor(value):
        if value.ndim == 0:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable_report_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_report_value(v) for v in value]
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _serialize_de_ast(node):
    from nestynet_sr.sr_core.bridges import (
        AbsNode,
        AddNode,
        ArgNode,
        AtomNode,
        ConjNode,
        ConstNode,
        CosNode,
        ExpNode,
        ImagNode,
        LogNode,
        MulNode,
        PowNode,
        RealNode,
        SinNode,
    )

    if node is None:
        return None
    if isinstance(node, AtomNode):
        payload = {
            "type": "atom",
            "kind": str(node.kind),
            "var_idxs": [int(i) for i in getattr(node, "var_idxs", ())],
            "kwargs": _jsonable_report_value(getattr(node, "kwargs", {}) or {}),
            "tag": getattr(node, "tag", None),
            "scope": getattr(node, "scope", None),
        }
        inputs = getattr(node, "inputs", None)
        if inputs:
            payload["inputs"] = [_serialize_de_ast(inp) for inp in inputs]
        return payload
    if isinstance(node, AddNode):
        return {"type": "add", "left": _serialize_de_ast(node.left), "right": _serialize_de_ast(node.right)}
    if isinstance(node, MulNode):
        return {"type": "mul", "left": _serialize_de_ast(node.left), "right": _serialize_de_ast(node.right)}
    if isinstance(node, PowNode):
        exponent = node.exponent
        if not isinstance(exponent, (int, float, str, bool)) and exponent is not None:
            exponent = _serialize_de_ast(exponent)
        return {
            "type": "pow",
            "base": _serialize_de_ast(node.base),
            "exponent": _jsonable_report_value(exponent),
        }
    if isinstance(node, LogNode):
        return {"type": "log", "arg": _serialize_de_ast(node.arg)}
    if isinstance(node, ExpNode):
        return {"type": "exp", "arg": _serialize_de_ast(node.arg)}
    if isinstance(node, SinNode):
        return {"type": "sin", "arg": _serialize_de_ast(node.arg)}
    if isinstance(node, CosNode):
        return {"type": "cos", "arg": _serialize_de_ast(node.arg)}
    if isinstance(node, ConjNode):
        return {"type": "conj", "arg": _serialize_de_ast(node.arg)}
    if isinstance(node, RealNode):
        return {"type": "real", "arg": _serialize_de_ast(node.arg)}
    if isinstance(node, ImagNode):
        return {"type": "imag", "arg": _serialize_de_ast(node.arg)}
    if isinstance(node, AbsNode):
        return {"type": "abs", "arg": _serialize_de_ast(node.arg)}
    if isinstance(node, ArgNode):
        return {"type": "arg", "arg": _serialize_de_ast(node.arg)}
    if isinstance(node, ConstNode):
        return {"type": "const", "value": _jsonable_report_value(node.value)}
    raise TypeError(f"Unsupported AST node type for JSON serialization: {type(node).__name__}")


def _build_library_design_matrix(
    surrogate,
    dataloader,
    *,
    order: int,
    term_asts: list,
    cfg: DESearchConfig,
    device,
):
    X = de_search_mod._gather_x(
        dataloader,
        max_batches=cfg.max_batches,
        max_points=cfg.max_points,
        device=device,
    )
    cache = de_search_mod.UFeatureCache(surrogate)
    cache.reset()
    if int(order) == 1:
        cache.ensure(X, need_grad=True, need_hess=False)
        anchor = cache.g[:, 0, int(cfg.x_axis)]
    elif int(order) == 2:
        cache.ensure(X, need_grad=False, need_hess=True)
        anchor = cache.H[:, 0, int(cfg.x_axis), int(cfg.x_axis)]
    else:
        raise ValueError(f"Unsupported DE order for validation candidate: {order}")

    cols = []
    for term_ast in term_asts:
        if term_ast is None:
            cols.append(torch.ones(X.shape[0], device=X.device, dtype=X.dtype))
        else:
            cols.append(de_search_mod._as_N(de_search_mod._eval_ast(term_ast, X, cache)))
    if cols:
        Phi = torch.stack(cols, dim=1)
    else:
        Phi = torch.zeros((int(X.shape[0]), 0), device=X.device, dtype=X.dtype)
    y = -anchor
    mask = torch.isfinite(y)
    if int(Phi.numel()) > 0:
        mask &= torch.isfinite(Phi).all(dim=1)
    if int(mask.sum()) < 10:
        raise RuntimeError(
            f"Too few finite rows for validation candidate (order={order}, rows={int(mask.sum())})"
        )
    return Phi[mask], y[mask]


def _residual_rms(Phi: torch.Tensor, y: torch.Tensor, coeffs: torch.Tensor) -> float:
    coeffs = coeffs.to(device=Phi.device, dtype=Phi.dtype)
    resid = Phi @ coeffs - y
    return float(resid.square().mean().sqrt().detach().cpu())


def _build_library_validation_candidate(
    res,
    *,
    surrogates: list,
    train_dataloaders: list,
    val_dataloaders: list,
    cfg: DESearchConfig,
    device,
):
    if isinstance(res, (FactorizedSearchDEResult, FactorizedDEResult)):
        return None
    if not hasattr(res, "term_asts") or not hasattr(res, "coeffs"):
        return None

    term_asts = list(getattr(res, "term_asts", []) or [])
    term_asts_json = [_serialize_de_ast(term_ast) for term_ast in term_asts]
    coeffs_raw = getattr(res, "coeffs", None)
    coeffs_tensor = (
        coeffs_raw.detach().cpu()
        if torch.is_tensor(coeffs_raw)
        else torch.as_tensor(coeffs_raw, dtype=torch.float64)
    )
    dataset_ids = list(getattr(res, "dataset_ids", None) or [])

    coeff_mode = "selected_result"
    shared_coeffs = coeffs_tensor
    residual_ast = getattr(res, "residual_ast", None)
    canonical_equation = res.format_equation() if hasattr(res, "format_equation") else None
    shared_cond = getattr(res, "condition_number", None)
    rms_train = getattr(res, "rms_train", None)
    rms_val = getattr(res, "rms_val", None)

    if coeffs_tensor.ndim == 2:
        coeff_mode = "shared_pooled_refit"
        Phis_tr = []
        ys_tr = []
        for surrogate, dl_tr in zip(surrogates, train_dataloaders):
            Phi_tr, y_tr = _build_library_design_matrix(
                surrogate,
                dl_tr,
                order=int(res.order),
                term_asts=term_asts,
                cfg=cfg,
                device=device,
            )
            Phis_tr.append(Phi_tr)
            ys_tr.append(y_tr)

        if Phis_tr:
            if int(Phis_tr[0].shape[1]) > 0:
                Phi_cat = torch.cat(Phis_tr, dim=0)
                y_cat = torch.cat(ys_tr, dim=0)
                shared_coeffs = de_search_mod.ridge_lstsq(Phi_cat, y_cat, ridge=0.0).detach().cpu()
                shared_coeffs_refit, used_scale_refit = de_search_mod._maybe_scale_normalized_refit_matrix(
                    Phi_cat,
                    y_cat,
                    shared_coeffs.to(device=Phi_cat.device, dtype=Phi_cat.dtype),
                )
                if used_scale_refit:
                    shared_coeffs = shared_coeffs_refit.detach().cpu()
                    coeff_mode = "shared_scale_normalized_refit"
                shared_cond = de_search_mod._compute_condition_number(Phi_cat)
            else:
                shared_coeffs = torch.zeros((0,), dtype=torch.float64)
                shared_cond = None
        else:
            shared_coeffs = torch.zeros((0,), dtype=torch.float64)
            shared_cond = None

        rms_train = [
            _residual_rms(Phi_tr, y_tr, shared_coeffs)
            for Phi_tr, y_tr in zip(Phis_tr, ys_tr)
        ]
        if val_dataloaders is not None:
            rms_val_list = []
            for surrogate, dl_va in zip(surrogates, val_dataloaders):
                Phi_va, y_va = _build_library_design_matrix(
                    surrogate,
                    dl_va,
                    order=int(res.order),
                    term_asts=term_asts,
                    cfg=cfg,
                    device=device,
                )
                rms_val_list.append(_residual_rms(Phi_va, y_va, shared_coeffs))
            rms_val = rms_val_list
        else:
            rms_val = None

        try:
            shared_res = DESearchResult(
                order=int(res.order),
                x_axis=int(res.x_axis),
                term_asts=term_asts,
                coeffs=shared_coeffs,
                rms_train=float(max(rms_train) if rms_train else float("nan")),
                rms_val=float(max(rms_val) if rms_val else float("nan")) if rms_val is not None else None,
            )
            residual_ast = de_search_mod.build_de_residual_ast(
                shared_res,
                units_spec=cfg.units_spec,
                enforce_units=bool(cfg.enforce_units),
            )
            canonical_equation = shared_res.format_equation()
        except Exception:
            residual_ast = None

    coeff_vec = shared_coeffs.reshape(-1)
    try:
        shared_cond = float(shared_cond) if shared_cond is not None else None
    except Exception:
        shared_cond = None

    return {
        "kind": "library_rhs",
        "order": int(res.order),
        "x_axis": int(res.x_axis),
        "coefficient_mode": str(coeff_mode),
        "coefficients": _jsonable_report_value(coeff_vec),
        "dataset_coefficients": _jsonable_report_value(coeffs_tensor) if coeffs_tensor.ndim == 2 else None,
        "term_asts_json": term_asts_json,
        "term_strings": [repr(t) if t is not None else "1" for t in term_asts],
        "rms_train": _jsonable_report_value(rms_train),
        "rms_val": _jsonable_report_value(rms_val),
        "condition_number": shared_cond,
        "canonical_equation": canonical_equation,
        "residual_ast": None if residual_ast is None else repr(residual_ast),
        "dataset_ids": dataset_ids or None,
    }


def _factorized_validation_candidate(obj: FactorizedDEResult) -> dict[str, Any]:
    return {
        "order": int(obj.order),
        "x_axis": int(obj.x_axis),
        "coefficients": [1.0],
        "term_asts_json": [_serialize_de_ast(obj.nonanchor_ast)],
        "residual_ast": None if obj.residual_ast is None else repr(obj.residual_ast),
    }


_FACTORIZED_TYPED_AST_KEYS = (
    "carrier_ast",
    "coord_ast",
    "coeff_ast",
)

_FACTORIZED_TYPED_VALUE_KEYS = (
    "lane",
    "family",
    "base_mode",
    "evidence_tier",
    "witness_kind",
    "coeff_expr",
    "consistency_score",
    "consistency_pairs",
    "consistency_total_pairs",
    "shape_score",
    "sign_changes",
    "curvature_ratio",
    "tv_ratio",
    "collapse_score",
    "collapse_coverage",
    "collapse_group_coverage",
    "collapse_pairs",
    "collapse_total_pairs",
    "collapse_reason",
    "collapse_confidence",
    "collapse_safe_rows",
    "collapse_total_rows",
    "collapse_domain_safe_fraction",
    "collapse_within_bin_variance",
    "collapse_cross_trajectory_variance",
    "collapse_monotonic_support",
    "collapse_sign_changes_mean",
    "collapse_mixed_sign_fraction",
    "assembly_lanes",
    "assembly_families",
    "assembly_pruned",
    "symbolic_size_raw",
    "symbolic_size_simplified",
    "projection_kind",
    "projection_support",
    "projection_coeffs",
    "projection_full_basis_size",
    "projection_signature",
    "projection_snap_report",
    "projection_snap_cost",
)

_FACTORIZED_TYPED_FLAT_KEYS = (
    "collapse_score",
    "collapse_coverage",
    "collapse_group_coverage",
    "collapse_pairs",
    "collapse_total_pairs",
    "collapse_reason",
    "collapse_confidence",
    "collapse_safe_rows",
    "collapse_total_rows",
    "collapse_domain_safe_fraction",
    "collapse_within_bin_variance",
    "collapse_cross_trajectory_variance",
    "collapse_monotonic_support",
    "collapse_sign_changes_mean",
    "collapse_mixed_sign_fraction",
    "assembly_lanes",
    "assembly_families",
    "assembly_pruned",
    "symbolic_size_raw",
    "symbolic_size_simplified",
    "projection_kind",
    "projection_support",
    "projection_coeffs",
    "projection_full_basis_size",
    "projection_signature",
    "projection_snap_report",
    "projection_snap_cost",
)


def _factorized_typed_metadata(source: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    out: dict[str, Any] = {}
    for key in _FACTORIZED_TYPED_VALUE_KEYS:
        if key not in source:
            continue
        value = source.get(key, None)
        if value is None or (isinstance(value, str) and not value):
            continue
        out[key] = _jsonable_report_value(value)

    for key in _FACTORIZED_TYPED_AST_KEYS:
        value = source.get(key, None)
        if value is None:
            continue
        out[key] = repr(value)

    coeff_asts = [repr(ast) for ast in list(source.get("coeff_asts", []) or []) if ast is not None]
    if coeff_asts:
        out["coeff_asts"] = coeff_asts
    return out


def _factorized_typed_flat_fields(source: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    out: dict[str, Any] = {}
    for key in _FACTORIZED_TYPED_FLAT_KEYS:
        if key in source:
            out[key] = _jsonable_report_value(source.get(key, None))
    return out


def _selected_factorized_shortlist_row(
    diagnostics: dict[str, Any],
    shortlist_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not shortlist_rows:
        return None
    try:
        rank = int(diagnostics.get("selected_shortlist_rank", 0))
    except Exception:
        rank = 0
    if 0 <= rank < len(shortlist_rows):
        row = shortlist_rows[rank]
        return row if isinstance(row, dict) else None
    return None


def _aggregate_metric(metric) -> float:
    if metric is None:
        return float("inf")
    if torch.is_tensor(metric):
        metric = metric.detach().cpu().tolist()
    elif isinstance(metric, np.ndarray):
        metric = metric.tolist()
    if isinstance(metric, (list, tuple)):
        vals = []
        for v in metric:
            try:
                f = float(v)
            except Exception:
                continue
            if math.isfinite(f):
                vals.append(f)
        return max(vals) if vals else float("inf")
    try:
        return float(metric)
    except Exception:
        return float("inf")


def candidate_probe_rms(obj) -> float:
    """Return the comparable held-out RMS used for rescue decisions."""
    if isinstance(obj, FactorizedSearchDEResult):
        return float(obj.probe_rms)
    if isinstance(obj, FactorizedDEResult):
        return float(obj.probe_rms)
    rms_val = getattr(obj, "rms_val", None)
    rms_train = getattr(obj, "rms_train", None)
    return _aggregate_metric(rms_val if rms_val is not None else rms_train)


def _node_uses_x(node, *, x_axis: int) -> bool:
    from nestynet_sr.sr_core.bridges import (
        AbsNode,
        AddNode,
        ArgNode,
        AtomNode,
        ConjNode,
        CosNode,
        ExpNode,
        ImagNode,
        LogNode,
        MulNode,
        PowNode,
        RealNode,
        SinNode,
    )

    if node is None:
        return False
    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        if kind in ("var", "x", "input"):
            var_idxs = list(getattr(node, "var_idxs", ()) or ())
            return len(var_idxs) == 1 and int(var_idxs[0]) == int(x_axis)
        inputs = getattr(node, "inputs", None)
        return any(_node_uses_x(inp, x_axis=x_axis) for inp in list(inputs or []))
    if isinstance(node, AddNode):
        return _node_uses_x(node.left, x_axis=x_axis) or _node_uses_x(node.right, x_axis=x_axis)
    if isinstance(node, MulNode):
        return _node_uses_x(node.left, x_axis=x_axis) or _node_uses_x(node.right, x_axis=x_axis)
    if isinstance(node, PowNode):
        if _node_uses_x(node.base, x_axis=x_axis):
            return True
        exponent = getattr(node, "exponent", None)
        if isinstance(exponent, (int, float, str, bool)) or exponent is None:
            return False
        return _node_uses_x(exponent, x_axis=x_axis)
    if isinstance(node, (LogNode, ExpNode, SinNode, CosNode, ConjNode, RealNode, ImagNode, AbsNode, ArgNode)):
        return _node_uses_x(node.arg, x_axis=x_axis)
    return False


def _node_uses_state(node) -> bool:
    from nestynet_sr.sr_core.bridges import (
        AbsNode,
        AddNode,
        ArgNode,
        AtomNode,
        ConjNode,
        CosNode,
        ExpNode,
        ImagNode,
        LogNode,
        MulNode,
        PowNode,
        RealNode,
        SinNode,
    )

    if node is None:
        return False
    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        if kind in ("u", "du", "d2u"):
            return True
        inputs = getattr(node, "inputs", None)
        return any(_node_uses_state(inp) for inp in list(inputs or []))
    if isinstance(node, AddNode):
        return _node_uses_state(node.left) or _node_uses_state(node.right)
    if isinstance(node, MulNode):
        return _node_uses_state(node.left) or _node_uses_state(node.right)
    if isinstance(node, PowNode):
        if _node_uses_state(node.base):
            return True
        exponent = getattr(node, "exponent", None)
        if isinstance(exponent, (int, float, str, bool)) or exponent is None:
            return False
        return _node_uses_state(exponent)
    if isinstance(node, (LogNode, ExpNode, SinNode, CosNode, ConjNode, RealNode, ImagNode, AbsNode, ArgNode)):
        return _node_uses_state(node.arg)
    return False


def _result_is_autonomous(primary) -> bool | None:
    term_asts = getattr(primary, "term_asts", None)
    if term_asts is None:
        return None
    x_axis = int(getattr(primary, "x_axis", 0))
    return not any(_node_uses_x(term, x_axis=x_axis) for term in list(term_asts or []) if term is not None)


def _result_is_forced_only(primary) -> bool | None:
    term_asts = getattr(primary, "term_asts", None)
    if term_asts is None:
        return None
    return not any(_node_uses_state(term) for term in list(term_asts or []) if term is not None)


def _rescue_cfg_signature(cfg: DESearchConfig) -> tuple[tuple[int, ...], bool, bool, bool]:
    return (
        tuple(int(o) for o in getattr(cfg, "order_candidates", (1, 2))),
        bool(getattr(cfg, "include_x", True)),
        bool(getattr(cfg, "include_u", True)),
        bool(getattr(cfg, "include_du", True)),
    )


def _build_factorized_search_rescue_attempts(
    primary,
    cfg: DESearchConfig,
    *,
    use_primary_constraints: bool = False,
) -> list[dict]:
    if not bool(use_primary_constraints):
        attempts = _build_factorized_search_only_attempts(cfg)
        for attempt in attempts:
            attempt["conditioned_on_primary"] = False
        return attempts

    attempts: list[dict] = []
    restricted_cfg = cfg
    constraints: list[str] = []

    order = int(getattr(primary, "order", -1))
    if order in (1, 2):
        ords = tuple(int(o) for o in getattr(cfg, "order_candidates", (1, 2)))
        if ords != (order,):
            restricted_cfg = replace(restricted_cfg, order_candidates=(order,))
            constraints.append(f"order={order}")

    autonomous = _result_is_autonomous(primary)
    if autonomous is True and bool(getattr(restricted_cfg, "include_x", True)):
        restricted_cfg = replace(restricted_cfg, include_x=False)
        constraints.append("no_x")

    forced_only = _result_is_forced_only(primary)
    if forced_only is True:
        if bool(getattr(restricted_cfg, "include_u", True)):
            restricted_cfg = replace(restricted_cfg, include_u=False)
            constraints.append("no_u")
        if int(order) == 2 and bool(getattr(restricted_cfg, "include_du", True)):
            restricted_cfg = replace(restricted_cfg, include_du=False)
            constraints.append("no_du")

    if constraints:
        attempts.append(
            {
                "name": "restricted",
                "cfg": restricted_cfg,
                "constraints": list(constraints),
                "restricted": True,
                "conditioned_on_primary": True,
                "autonomous_first_line": bool(autonomous),
                "forced_only_first_line": bool(forced_only),
            }
        )

    if not attempts or _rescue_cfg_signature(attempts[-1]["cfg"]) != _rescue_cfg_signature(cfg):
        attempts.append(
            {
                "name": "full",
                "cfg": cfg,
                "constraints": [],
                "restricted": False,
                "conditioned_on_primary": True,
                "autonomous_first_line": bool(autonomous) if autonomous is not None else None,
                "forced_only_first_line": bool(forced_only) if forced_only is not None else None,
            }
        )
    return attempts


def _factorized_search_result_selection_key(rescue_res) -> tuple[int, int, float, float, int, float]:
    diagnostics = getattr(rescue_res, "diagnostics", {}) or {}
    domain_bucket = 0 if _candidate_domain_safe(rescue_res) else 1

    integrate_ok = diagnostics.get("integrate_ok", None)
    integrate_mse = diagnostics.get("integrate_mse", None)
    try:
        integrate_mse_f = float(integrate_mse)
    except Exception:
        integrate_mse_f = float("inf")
    if integrate_ok is True and math.isfinite(integrate_mse_f):
        integrate_bucket = 0
    else:
        integrate_bucket = 1

    try:
        fragility = float(diagnostics.get("domain_fragility_penalty", 0.0))
    except Exception:
        fragility = float("inf")
    try:
        size = int(diagnostics.get("size", 10**9))
    except Exception:
        size = 10**9
    return (
        int(domain_bucket),
        int(integrate_bucket),
        float(integrate_mse_f) if int(integrate_bucket) == 0 else float("inf"),
        float(candidate_probe_rms(rescue_res)),
        int(size),
        float(fragility),
    )


def _estimate_oscillation_periods(x, y, *, hysteresis_frac: float = 0.05) -> float:
    """Estimate the number of oscillation periods in y(x) via direction reversals.

    Counts direction reversals of y along sorted x, ignoring excursions smaller
    than ``hysteresis_frac`` of the signal range (robust to mild noise). A full
    period has two reversals.
    """
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if x_arr.size < 8 or y_arr.size != x_arr.size:
        return 0.0
    ys = y_arr[np.argsort(x_arr)]
    rng = float(ys.max() - ys.min())
    if not math.isfinite(rng) or rng <= 0.0:
        return 0.0
    h = float(hysteresis_frac) * rng
    reversals = 0
    direction = 0
    ext = float(ys[0])
    for v in ys[1:]:
        v = float(v)
        if direction == 0:
            if abs(v - ext) >= h:
                direction = 1 if v > ext else -1
                ext = v
            continue
        if direction == 1:
            if v > ext:
                ext = v
            elif ext - v >= h:
                reversals += 1
                direction = -1
                ext = v
        else:
            if v < ext:
                ext = v
            elif v - ext >= h:
                reversals += 1
                direction = 1
                ext = v
    return reversals / 2.0


def _plan_surrogate_windows(
    filepath: str,
    *,
    x_axis: int,
    max_periods: float,
    output_dir: str,
    min_rows: int = 400,
    max_windows: int = 8,
) -> list[str]:
    """Split an oscillation-dense 1-D dataset into contiguous x-windows.

    Long oscillatory records exceed what a fixed-segment surrogate can resolve
    (de122: the Duffing third harmonic needs ~4x more segments than the full
    span allows, and LM cannot train that many). One surrogate per few-period
    window restores the segments-per-wavelength ratio without growing the
    model. Returns the list of CSV paths to train on (just ``[filepath]``
    when no split is needed).
    """
    try:
        data = np.genfromtxt(filepath, delimiter=",", names=True)
    except Exception:
        return [filepath]
    names = list(data.dtype.names or [])
    x_name = f"x{int(x_axis)}"
    if "y" not in names or x_name not in names or len(data.shape) != 1:
        return [filepath]
    n_rows = int(data.shape[0])
    periods = _estimate_oscillation_periods(data[x_name], data["y"])
    if not (float(max_periods) > 0.0) or periods <= float(max_periods):
        return [filepath]
    n_windows = int(math.ceil(periods / float(max_periods)))
    n_windows = max(1, min(int(max_windows), n_windows, n_rows // max(1, int(min_rows))))
    if n_windows <= 1:
        return [filepath]

    order = np.argsort(np.asarray(data[x_name], dtype=np.float64))
    window_dir = pathlib.Path(output_dir) / "surrogate_windows"
    window_dir.mkdir(parents=True, exist_ok=True)
    stem = pathlib.Path(filepath).stem
    bounds = np.linspace(0, n_rows, n_windows + 1).astype(int)
    out_paths: list[str] = []
    for k in range(n_windows):
        rows = data[order[bounds[k]:bounds[k + 1]]]
        path = window_dir / f"{stem}_w{k}.csv"
        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join(names) + "\n")
            for row in rows:
                f.write(",".join(f"{float(row[name]):.18e}" for name in names) + "\n")
        out_paths.append(str(path))
    print(
        f"[Surrogate] Oscillation-dense dataset (~{periods:.1f} periods): "
        f"splitting into {n_windows} windows for surrogate training."
    )
    return out_paths


def _window_data_hp(data_hp, n_rows: int):
    """Scale ndata/batch settings to a window's row count."""
    cap = max(1, int(n_rows * 0.45))
    ndata = min(int(data_hp.ndata_select), cap)
    ndata_val = min(int(data_hp.ndata_select_val), cap)
    hp = DataHyperparams(
        batch_size=min(int(data_hp.batch_size), ndata),
        ndata_select=ndata,
        ndata_select_val=ndata_val,
    )
    hp.data_split_strategy = str(data_hp.data_split_strategy)
    return hp


def _make_factorized_search_attempt(
    name: str,
    cfg_attempt: DESearchConfig,
    constraints: list[str],
) -> dict:
    return {
        "name": str(name),
        "cfg": cfg_attempt,
        "constraints": list(constraints),
    }


def _factorized_search_max_attempts(rescue_cfg: FactorizedSearchDERescueConfig) -> int | None:
    value = getattr(rescue_cfg, "max_attempts", None)
    if value is None:
        return None
    try:
        return max(0, int(value))
    except Exception:
        return None


def _prepare_factorized_search_feature_groups(
    *,
    cfg: DESearchConfig,
    rescue_cfg: FactorizedSearchDERescueConfig,
    surrogates: list,
    dl_tr_list: list,
    dl_va_list: list,
    dataset_ids: list[str],
    surrogate_val_losses: list[float] | None = None,
    device,
    dtype: torch.dtype,
):
    if len(surrogates) == 1:
        groups = build_factorized_search_de_feature_groups_from_surrogate(
            surrogates[0],
            dl_tr_list[0],
            dl_va_list[0],
            cfg=cfg,
            rescue_cfg=rescue_cfg,
            device=device,
            dtype=dtype,
            group_id=str(dataset_ids[0]) if dataset_ids else "dataset0",
        )
    else:
        groups = build_factorized_search_de_feature_groups_from_surrogates(
            surrogates,
            dl_tr_list,
            dl_va_list,
            cfg=cfg,
            rescue_cfg=rescue_cfg,
            device=device,
            dataset_ids=dataset_ids,
            dtype=dtype,
        )

    losses = list(surrogate_val_losses) if surrogate_val_losses is not None else []
    if losses:
        if len(losses) != len(groups):
            raise ValueError("surrogate_val_losses must match the number of DE feature groups")
        groups = [
            replace(group, surrogate_val_loss=float(losses[i]))
            for i, group in enumerate(groups)
        ]
    return groups


def _build_factorized_search_only_attempts(cfg: DESearchConfig) -> list[dict]:
    attempts: list[dict] = []
    seen: set[tuple[tuple[int, ...], bool, bool, bool]] = set()
    base_orders = tuple(int(o) for o in getattr(cfg, "order_candidates", (1, 2)))

    def _append(name: str, cfg_attempt: DESearchConfig, constraints: list[str]) -> None:
        sig = _rescue_cfg_signature(cfg_attempt)
        if sig in seen:
            return
        seen.add(sig)
        attempts.append(_make_factorized_search_attempt(name, cfg_attempt, constraints))

    _append("full", cfg, [])

    for order in base_orders:
        cfg_order = replace(cfg, order_candidates=(int(order),))
        available = []
        if bool(getattr(cfg_order, "include_x", True)):
            available.append(("x", "include_x"))
        if bool(getattr(cfg_order, "include_u", True)):
            available.append(("u", "include_u"))
        if int(order) == 2 and bool(getattr(cfg_order, "include_du", True)):
            available.append(("du", "include_du"))

        keep_full = [name for name, _ in available]
        _append(
            name=f"order{order}_full",
            cfg_attempt=cfg_order,
            constraints=[f"order={order}", f"features={'+'.join(keep_full) if keep_full else 'none'}"],
        )

        for subset_size in (1,):
            for subset in combinations(available, subset_size):
                keep = {name for name, _ in subset}
                cfg_attempt = replace(
                    cfg_order,
                    include_x=("x" in keep),
                    include_u=("u" in keep),
                    include_du=("du" in keep) if int(order) == 2 else bool(getattr(cfg_order, "include_du", True)),
                )
                _append(
                    name=f"order{order}_{'+'.join(sorted(keep))}",
                    cfg_attempt=cfg_attempt,
                    constraints=[f"order={order}", f"features={'+'.join(sorted(keep))}"],
                )

    return attempts


def _should_fallback_from_full_factorized_search_only(
    full_res,
    rescue_cfg: FactorizedSearchDERescueConfig,
) -> bool:
    if full_res is None:
        return True

    score = candidate_probe_rms(full_res)
    if not math.isfinite(score):
        return True
    if not _candidate_domain_safe(full_res):
        return True

    diagnostics = getattr(full_res, "diagnostics", {}) or {}
    if isinstance(diagnostics, dict):
        if diagnostics.get("integrate_ok", None) is False:
            return True
        try:
            integrate_mse = float(diagnostics.get("integrate_mse", float("inf")))
        except Exception:
            integrate_mse = float("inf")
        if math.isfinite(integrate_mse):
            mse_cap = max(1.0e-4, (10.0 * float(getattr(rescue_cfg, "trigger_val_rms", 1.0e-3))) ** 2)
            if integrate_mse > mse_cap:
                return True

    score_cap = max(0.1, 10.0 * float(getattr(rescue_cfg, "trigger_val_rms", 1.0e-3)))
    return bool(score > score_cap)


def _run_factorized_search_only_with_heuristics(
    *,
    cfg: DESearchConfig,
    rescue_cfg: FactorizedSearchDERescueConfig,
    filepaths: list[str],
    surrogates: list,
    dl_tr_list: list,
    dl_va_list: list,
    dataset_ids: list[str],
    device,
    dtype: torch.dtype,
    verbose: bool = True,
    feature_groups=None,
    surrogate_val_losses: list[float] | None = None,
):
    attempts_available = _build_factorized_search_only_attempts(cfg)
    max_attempts = _factorized_search_max_attempts(rescue_cfg)
    attempts = (
        attempts_available
        if max_attempts is None
        else attempts_available[: max(0, int(max_attempts))]
    )
    if not attempts:
        return None
    if feature_groups is None:
        feature_groups = _prepare_factorized_search_feature_groups(
            cfg=cfg,
            rescue_cfg=rescue_cfg,
            surrogates=surrogates,
            dl_tr_list=dl_tr_list,
            dl_va_list=dl_va_list,
            dataset_ids=dataset_ids,
            surrogate_val_losses=surrogate_val_losses,
            device=device,
            dtype=dtype,
        )
    attempt_logs: list[dict] = []
    attempt_shortlists: list[tuple[str, list[dict[str, Any]]]] = []
    best_res = None
    best_key = None

    for i, attempt in enumerate(attempts):
        cfg_attempt = attempt["cfg"]
        if verbose:
            print(
                "[factorized symbolic search] Attempt {}/{}: {} order_candidates={} include_x={} include_u={} include_du={}".format(
                    i + 1,
                    len(attempts),
                    attempt["name"],
                    tuple(int(o) for o in getattr(cfg_attempt, "order_candidates", ())),
                    bool(getattr(cfg_attempt, "include_x", True)),
                    bool(getattr(cfg_attempt, "include_u", True)),
                    bool(getattr(cfg_attempt, "include_du", True)),
                )
            )
        rescue_candidate = _run_factorized_search_rescue_attempt(
            cfg_attempt=cfg_attempt,
            rescue_cfg=rescue_cfg,
            filepaths=filepaths,
            surrogates=surrogates,
            dl_tr_list=dl_tr_list,
            dl_va_list=dl_va_list,
            dataset_ids=dataset_ids,
            feature_groups=feature_groups,
            device=device,
            dtype=dtype,
        )
        attempt_shortlist = _extract_factorized_search_shortlist_from_result(rescue_candidate)
        attempt_shortlists.append((str(attempt["name"]), attempt_shortlist))
        key = _factorized_search_result_selection_key(rescue_candidate)
        attempt_log = {
            "name": str(attempt["name"]),
            "constraints": list(attempt.get("constraints", []) or []),
            "probe_rms": float(candidate_probe_rms(rescue_candidate)),
            "selection_key": [float(v) if isinstance(v, (int, float)) else str(v) for v in key],
            "order_candidates": [int(o) for o in getattr(cfg_attempt, "order_candidates", ())],
            "include_x": bool(getattr(cfg_attempt, "include_x", True)),
            "include_u": bool(getattr(cfg_attempt, "include_u", True)),
            "include_du": bool(getattr(cfg_attempt, "include_du", True)),
            "domain_ok": getattr(rescue_candidate, "diagnostics", {}).get("domain_ok", None),
            "integrate_ok": getattr(rescue_candidate, "diagnostics", {}).get("integrate_ok", None),
            "shortlist_size": int(len(attempt_shortlist)),
        }
        attempt_logs.append(attempt_log)
        if best_key is None or key < best_key:
            best_key = key
            best_res = rescue_candidate
            if isinstance(getattr(best_res, "diagnostics", None), dict):
                best_res.diagnostics["selected_attempt"] = str(attempt["name"])

        if int(i) == 0 and str(attempt.get("name", "")) == "full":
            attempt_log["fallback_triggered"] = bool(
                _should_fallback_from_full_factorized_search_only(rescue_candidate, rescue_cfg)
            )
            if not bool(attempt_log["fallback_triggered"]):
                break

    if best_res is not None and isinstance(getattr(best_res, "diagnostics", None), dict):
        best_res.diagnostics["rescue_attempts"] = attempt_logs
        best_res.diagnostics["rescue_attempts_available"] = int(len(attempts_available))
        best_res.diagnostics["rescue_attempts_run"] = int(len(attempt_logs))
        best_res.diagnostics["rescue_max_attempts"] = None if max_attempts is None else int(max_attempts)
        best_res.diagnostics["rescue_attempts_capped"] = bool(
            max_attempts is not None and int(len(attempts)) < int(len(attempts_available))
        )
        best_res.diagnostics["shortlist_union"] = _merge_factorized_search_attempt_shortlists(attempt_shortlists)
    return best_res


def _should_widen_from_restricted_rescue(
    restricted_res,
    primary,
    rescue_cfg: FactorizedSearchDERescueConfig,
) -> bool:
    s1 = candidate_probe_rms(restricted_res)
    if not math.isfinite(s1):
        return True

    if not _candidate_domain_safe(restricted_res):
        return True

    s0 = candidate_probe_rms(primary)
    replace_factor = float(getattr(rescue_cfg, "replace_rel_factor", 0.98))
    if math.isfinite(s0) and s1 <= replace_factor * s0:
        return False

    soft_cap = 10.0 * float(getattr(rescue_cfg, "trigger_val_rms", 1.0e-3))
    if math.isfinite(s0):
        soft_cap = max(soft_cap, 2.0 * s0)
    return bool(s1 > soft_cap)


def _run_factorized_de(
    *,
    cfg: DESearchConfig,
    factorized_cfg: FactorizedDERescueConfig,
    rescue_cfg: FactorizedSearchDERescueConfig,
    filepaths: list[str],
    surrogates: list,
    dl_tr_list: list,
    dl_va_list: list,
    dataset_ids: list[str],
    surrogate_val_losses: list[float] | None,
    device,
    dtype: torch.dtype,
    verbose: bool = True,
    whole_rhs_policy: str = "auto",
    typed_lanes_policy: str = "never",
    feature_groups=None,
):
    """Run STLSQ-free DE-FSS with optional typed and broad fallback lanes."""

    def _candidate_search_diagnostics_summary(candidate):
        diag = getattr(candidate, "diagnostics", None)
        if not isinstance(diag, dict):
            return None
        fdf_diag = diag.get("factorized_de_diagnostics", None)
        if not isinstance(fdf_diag, dict):
            return None
        summary = fdf_diag.get("search_diagnostics_summary", None)
        if isinstance(summary, dict):
            return _jsonable_report_value(summary)
        return None

    def _candidate_search_diagnostics(candidate):
        diag = getattr(candidate, "diagnostics", None)
        if not isinstance(diag, dict):
            return None
        fdf_diag = diag.get("factorized_de_diagnostics", None)
        if isinstance(fdf_diag, dict):
            return _jsonable_report_value(fdf_diag)
        return None

    whole_rhs_policy_l = str(whole_rhs_policy or "auto").strip().lower()
    typed_lanes_policy_l = str(typed_lanes_policy or "never").strip().lower()
    if typed_lanes_policy_l not in {"never", "auto", "always", "force"}:
        raise ValueError(f"unknown factorized-de typed-lanes policy: {typed_lanes_policy!r}")
    if whole_rhs_policy_l != "always" and getattr(rescue_cfg, "max_attempts", None) is None:
        rescue_cfg = copy.deepcopy(rescue_cfg)
        rescue_cfg.max_attempts = 1

    if feature_groups is None:
        feature_cfg = _feature_group_rescue_cfg_from_factorized(factorized_cfg)
        feature_groups = _prepare_factorized_search_feature_groups(
            cfg=cfg,
            rescue_cfg=feature_cfg,
            surrogates=surrogates,
            dl_tr_list=dl_tr_list,
            dl_va_list=dl_va_list,
            dataset_ids=dataset_ids,
            surrogate_val_losses=surrogate_val_losses,
            device=device,
            dtype=dtype,
        )

    if verbose:
        print("[factorized DE] Direct anchored residual/RHS FSS lane (autonomous phase)...")
    direct_res = run_direct_residual_fss_from_feature_groups(
        feature_groups,
        cfg=cfg,
        rescue_cfg=rescue_cfg,
        dtype=dtype,
        verbose=bool(verbose),
        attempt_phase="autonomous",
    )

    implicit_res = None
    if bool(getattr(rescue_cfg, "regularized_implicit_enable", True)):
        if verbose:
            print("[factorized DE] Regularized implicit residual FSS lane...")
        implicit_res = run_regularized_implicit_residual_fss_from_feature_groups(
            feature_groups,
            cfg=cfg,
            rescue_cfg=rescue_cfg,
            dtype=dtype,
            verbose=bool(verbose),
        )
    elif verbose:
        print("[factorized DE] Skipping regularized implicit residual lane (disabled).")

    direct_needs_typed = _direct_result_needs_typed_lane(direct_res, rescue_cfg)
    best_pretyped = direct_res if _candidate_domain_safe(direct_res) else None
    pretyped_lane_name = "direct_residual_fss"
    if _candidate_domain_safe(implicit_res) and (
        best_pretyped is None or direct_needs_typed
    ):
        if _factorized_de_preferred(
            implicit_res,
            "regularized_implicit_residual",
            best_pretyped,
            pretyped_lane_name,
            same_lane_rel_factor=1.0,
            cleaner_lane_factor=2.0,
            dirtier_lane_factor=0.5,
        ):
            best_pretyped = implicit_res
            pretyped_lane_name = "regularized_implicit_residual"

    full_direct_attempted = False
    if _direct_result_needs_typed_lane(best_pretyped, rescue_cfg):
        full_direct_attempted = True
        if verbose:
            print("[factorized DE] Direct anchored residual/RHS FSS lane (full phase)...")
        direct_full_res = run_direct_residual_fss_from_feature_groups(
            feature_groups,
            cfg=cfg,
            rescue_cfg=rescue_cfg,
            dtype=dtype,
            verbose=bool(verbose),
            attempt_phase="full",
        )
        if _factorized_de_preferred(
            direct_full_res,
            "direct_residual_fss",
            direct_res,
            "direct_residual_fss",
            same_lane_rel_factor=1.0,
            cleaner_lane_factor=2.0,
            dirtier_lane_factor=0.5,
        ):
            direct_res = direct_full_res
        if _factorized_de_preferred(
            direct_full_res,
            "direct_residual_fss",
            best_pretyped,
            pretyped_lane_name,
            same_lane_rel_factor=1.0,
            cleaner_lane_factor=2.0,
            dirtier_lane_factor=0.5,
        ):
            best_pretyped = direct_full_res
            pretyped_lane_name = "direct_residual_fss"
    elif verbose:
        print("[factorized DE] Skipping full direct residual phase; first-line candidate is clean.")

    factorized_res = None
    typed_attempted = False
    direct_needs_typed = _direct_result_needs_typed_lane(direct_res, rescue_cfg)
    first_line_needs_typed = _direct_result_needs_typed_lane(best_pretyped, rescue_cfg)
    first_line_certified = _first_line_certified(best_pretyped, rescue_cfg)
    # A certified first-line law may only be displaced by a strictly better
    # challenger; the typed lane's cleaner-rank bonus would otherwise let a
    # candidate within 2x RMS replace an exact equation.
    challenger_cleaner_factor = (
        float(getattr(rescue_cfg, "replace_rel_factor", 0.98)) if first_line_certified else 2.0
    )
    typed_decision = {
        "policy": str(typed_lanes_policy_l),
        "run": False,
        "reason": "",
        "direct_needs_typed": bool(direct_needs_typed),
        "first_line_needs_typed": bool(first_line_needs_typed),
        "first_line_certified": bool(first_line_certified),
        "pretyped_lane": str(pretyped_lane_name),
        "full_direct_attempted": bool(full_direct_attempted),
        "direct_generator_status": _result_generator_status(direct_res),
        "direct_probe_rms": candidate_probe_rms(direct_res),
        "direct_probe_rel_rms": _result_probe_rel_rms(direct_res),
        "regularized_implicit_probe_rms": candidate_probe_rms(implicit_res),
        "regularized_implicit_probe_rel_rms": _result_probe_rel_rms(implicit_res),
    }
    if typed_lanes_policy_l == "never":
        typed_decision["reason"] = "policy_never"
    elif typed_lanes_policy_l in {"always", "force"}:
        # 'always' means always: the cleanliness gate only applies to 'auto'.
        # 'force' is kept as a deprecated alias.
        typed_decision["run"] = True
        typed_decision["reason"] = f"policy_{typed_lanes_policy_l}"
    elif first_line_needs_typed:
        typed_decision["run"] = True
        typed_decision["reason"] = f"policy_{typed_lanes_policy_l}_first_line_needs_typed"
    else:
        typed_decision["reason"] = f"policy_{typed_lanes_policy_l}_first_line_clean"

    if bool(typed_decision["run"]):
        typed_attempted = True
        if verbose:
            print("[operator-factorized DE] Curated typed zero-base proposal lane...")
        factorized_res = run_factorized_coeff_rescue_from_feature_groups(
            feature_groups,
            cfg=cfg,
            rescue_cfg=factorized_cfg,
            primary=None,
            dtype=dtype,
        )
    elif verbose:
        print(
            "[operator-factorized DE] Skipping curated typed lanes "
            f"(policy={typed_lanes_policy_l}, reason={typed_decision['reason']})."
        )

    auxiliary_rollout_candidates: list[dict[str, Any]] = []
    direct_generator_status = _result_generator_status(direct_res)
    if (
        isinstance(direct_res, FactorizedSearchDEResult)
        and _candidate_domain_safe(direct_res)
        and _is_rollout_promotable_generator_status(direct_generator_status)
    ):
        direct_payload = serialize_de_candidate(direct_res)
        if isinstance(direct_payload, dict):
            direct_payload["source_lane"] = "direct_residual_fss"
            direct_payload["auxiliary_first_line"] = True
            direct_payload["first_line_certified"] = bool(_first_line_certified(direct_res, rescue_cfg))
            direct_payload["generator_status"] = str(direct_generator_status)
            direct_payload["generator_witness_promoted"] = True
            auxiliary_rollout_candidates.append(direct_payload)
    if isinstance(implicit_res, FactorizedSearchDEResult) and _candidate_domain_safe(implicit_res):
        implicit_payload = serialize_de_candidate(implicit_res)
        if isinstance(implicit_payload, dict):
            implicit_payload["source_lane"] = "regularized_implicit_residual"
            implicit_payload["auxiliary_first_line"] = True
            implicit_payload["first_line_certified"] = bool(_first_line_certified(implicit_res, rescue_cfg))
            auxiliary_rollout_candidates.append(implicit_payload)
    if auxiliary_rollout_candidates and isinstance(factorized_res, FactorizedDEResult):
        factorized_res.diagnostics.setdefault("auxiliary_rollout_candidates", [])
        factorized_res.diagnostics["auxiliary_rollout_candidates"].extend(auxiliary_rollout_candidates)

    best_first_line = best_pretyped
    first_line_name = str(pretyped_lane_name)
    if _factorized_de_preferred(
        factorized_res,
        "factorized",
        best_first_line,
        first_line_name,
        same_lane_rel_factor=1.0,
        cleaner_lane_factor=challenger_cleaner_factor,
        dirtier_lane_factor=0.5,
    ):
        best_first_line = factorized_res
        first_line_name = "factorized"

    if verbose:
        rms = candidate_probe_rms(best_first_line)
        rms_text = f"{rms:.6e}" if math.isfinite(rms) else "inf"
        print(f"[factorized DE] Best first-line lane: {first_line_name} probe RMS={rms_text}")

    run_whole_rhs, whole_rhs_diag = _factorized_de_whole_rhs_decision(
        best_first_line,
        rescue_cfg,
        policy=whole_rhs_policy_l,
    )

    rescue_res = None
    if run_whole_rhs:
        if verbose:
            print(
                "[operator-factorized DE] Whole-RHS / basis-state factorized symbolic search proposal lane "
                f"(policy={whole_rhs_diag['policy']}, reason={whole_rhs_diag['reason']})..."
            )
        rescue_res = _run_factorized_search_only_with_heuristics(
            cfg=cfg,
            rescue_cfg=rescue_cfg,
            filepaths=filepaths,
            surrogates=surrogates,
            dl_tr_list=dl_tr_list,
            dl_va_list=dl_va_list,
            dataset_ids=dataset_ids,
            device=device,
            dtype=dtype,
            verbose=verbose,
            feature_groups=feature_groups,
            surrogate_val_losses=surrogate_val_losses,
        )
    elif verbose:
        print(
            "[operator-factorized DE] Skipping whole-RHS factorized symbolic search "
            f"(policy={whole_rhs_diag['policy']}, reason={whole_rhs_diag['reason']})."
        )

    selected = best_first_line if _candidate_domain_safe(best_first_line) else None
    selected_lane_name = first_line_name
    selected_engine = (
        "factorized_search"
        if first_line_name in {"direct_residual_fss", "regularized_implicit_residual"}
        else first_line_name
    )
    if _factorized_de_preferred(
        rescue_res,
        "factorized_search",
        selected,
        selected_lane_name,
        same_lane_rel_factor=1.0,
        cleaner_lane_factor=2.0,
        dirtier_lane_factor=0.5,
    ):
        selected = rescue_res
        selected_lane_name = "factorized_search"
        selected_engine = "factorized_search"
    if selected is None:
        raise RuntimeError("factorized DE first-line search produced no domain-valid candidate")

    diag = getattr(selected, "diagnostics", None)
    if isinstance(diag, dict):
        selected_coefficient_dim_mode = (
            "regularized_implicit"
            if str(selected_lane_name) == "regularized_implicit_residual"
            else "inferred_outer"
            if str(selected_lane_name) == "direct_residual_fss"
            else str(getattr(rescue_cfg, "coefficient_dim_mode", "strict_expression"))
        )
        diag["factorized_de"] = {
            "enabled": True,
            "selected_lane": selected_lane_name,
            "selected_engine": selected_engine,
            "direct_residual_probe_rms": candidate_probe_rms(direct_res),
            "regularized_implicit_probe_rms": candidate_probe_rms(implicit_res),
            "factorized_probe_rms": candidate_probe_rms(factorized_res),
            "factorized_search_probe_rms": candidate_probe_rms(rescue_res),
            "factorized_base_modes": list(getattr(factorized_cfg, "base_modes", ())),
            "factorized_search_attempted": rescue_res is not None,
            "direct_residual_attempted": direct_res is not None,
            "full_direct_residual_attempted": bool(full_direct_attempted),
            "regularized_implicit_attempted": implicit_res is not None,
            "typed_lanes_policy": str(typed_lanes_policy_l),
            "typed_lanes_attempted": bool(typed_attempted),
            "typed_lanes_decision": typed_decision,
            "factorized_attempted": factorized_res is not None,
            "coefficient_dim_mode": str(selected_coefficient_dim_mode),
            "whole_rhs_policy": whole_rhs_diag,
            "selected_search_diagnostics_summary": _candidate_search_diagnostics_summary(selected),
            "selected_search_diagnostics": _candidate_search_diagnostics(selected),
            "direct_residual_search_diagnostics_summary": _candidate_search_diagnostics_summary(direct_res),
            "direct_residual_search_diagnostics": _candidate_search_diagnostics(direct_res),
            "regularized_implicit_search_diagnostics_summary": _candidate_search_diagnostics_summary(implicit_res),
            "regularized_implicit_search_diagnostics": _candidate_search_diagnostics(implicit_res),
            "factorized_search_diagnostics_summary": _candidate_search_diagnostics_summary(rescue_res),
            "factorized_search_diagnostics": _candidate_search_diagnostics(rescue_res),
        }
    return selected, factorized_res, rescue_res, selected_engine, whole_rhs_diag


def _run_factorized_search_rescue_attempt(
    *,
    cfg_attempt: DESearchConfig,
    rescue_cfg: FactorizedSearchDERescueConfig,
    filepaths: list[str],
    surrogates: list,
    dl_tr_list: list,
    dl_va_list: list,
    dataset_ids: list[str],
    feature_groups=None,
    device,
    dtype: torch.dtype,
):
    if feature_groups is not None:
        return run_factorized_search_de_from_feature_groups(
            feature_groups,
            cfg=cfg_attempt,
            rescue_cfg=rescue_cfg,
            dtype=dtype,
        )
    if len(surrogates) == 1:
        return run_factorized_search_de_from_surrogate(
            surrogates[0],
            dl_tr_list[0],
            dl_va_list[0],
            cfg=cfg_attempt,
            rescue_cfg=rescue_cfg,
            device=device,
            dtype=dtype,
        )
    return run_factorized_search_de_from_surrogates(
        surrogates,
        dl_tr_list,
        dl_va_list,
        cfg=cfg_attempt,
        rescue_cfg=rescue_cfg,
        device=device,
        dataset_ids=dataset_ids,
        dtype=dtype,
    )


def _run_factorized_search_rescue_with_heuristics(
    *,
    primary_res,
    cfg: DESearchConfig,
    rescue_cfg: FactorizedSearchDERescueConfig,
    filepaths: list[str],
    surrogates: list,
    dl_tr_list: list,
    dl_va_list: list,
    dataset_ids: list[str],
    device,
    dtype: torch.dtype,
    verbose: bool = True,
    feature_groups=None,
    surrogate_val_losses: list[float] | None = None,
):
    attempts_available = _build_factorized_search_rescue_attempts(primary_res, cfg)
    max_attempts = _factorized_search_max_attempts(rescue_cfg)
    attempts = (
        attempts_available
        if max_attempts is None
        else attempts_available[: max(0, int(max_attempts))]
    )
    if not attempts:
        return None
    if feature_groups is None:
        feature_groups = _prepare_factorized_search_feature_groups(
            cfg=cfg,
            rescue_cfg=rescue_cfg,
            surrogates=surrogates,
            dl_tr_list=dl_tr_list,
            dl_va_list=dl_va_list,
            dataset_ids=dataset_ids,
            surrogate_val_losses=surrogate_val_losses,
            device=device,
            dtype=dtype,
        )
    attempt_logs: list[dict] = []
    attempt_shortlists: list[tuple[str, list[dict[str, Any]]]] = []
    best_res = None
    best_key = None

    for i, attempt in enumerate(attempts):
        cfg_attempt = attempt["cfg"]
        if verbose:
            print(
                "[factorized symbolic search] Attempt {}/{}: {} order_candidates={} include_x={} include_u={} include_du={}".format(
                    i + 1,
                    len(attempts),
                    attempt["name"],
                    tuple(int(o) for o in getattr(cfg_attempt, "order_candidates", ())),
                    bool(getattr(cfg_attempt, "include_x", True)),
                    bool(getattr(cfg_attempt, "include_u", True)),
                    bool(getattr(cfg_attempt, "include_du", True)),
                )
            )
        rescue_candidate = _run_factorized_search_rescue_attempt(
            cfg_attempt=cfg_attempt,
            rescue_cfg=rescue_cfg,
            filepaths=filepaths,
            surrogates=surrogates,
            dl_tr_list=dl_tr_list,
            dl_va_list=dl_va_list,
            dataset_ids=dataset_ids,
            feature_groups=feature_groups,
            device=device,
            dtype=dtype,
        )
        attempt_shortlist = _extract_factorized_search_shortlist_from_result(rescue_candidate)
        attempt_shortlists.append((str(attempt["name"]), attempt_shortlist))
        key = _factorized_search_result_selection_key(rescue_candidate)
        probe_rms = candidate_probe_rms(rescue_candidate)
        attempt_log = {
            "name": str(attempt["name"]),
            "order_candidates": [int(o) for o in getattr(cfg_attempt, "order_candidates", ())],
            "include_x": bool(getattr(cfg_attempt, "include_x", True)),
            "include_u": bool(getattr(cfg_attempt, "include_u", True)),
            "include_du": bool(getattr(cfg_attempt, "include_du", True)),
            "constraints": list(attempt.get("constraints", []) or []),
            "conditioned_on_primary": bool(attempt.get("conditioned_on_primary", False)),
            "probe_rms": float(probe_rms) if math.isfinite(probe_rms) else None,
            "selection_key": [float(v) if isinstance(v, (int, float)) else str(v) for v in key],
            "domain_ok": getattr(rescue_candidate, "diagnostics", {}).get("domain_ok", None),
            "integrate_ok": getattr(rescue_candidate, "diagnostics", {}).get("integrate_ok", None),
            "shortlist_size": int(len(attempt_shortlist)),
        }
        attempt_logs.append(attempt_log)
        if best_key is None or key < best_key:
            best_key = key
            best_res = rescue_candidate
            if isinstance(getattr(best_res, "diagnostics", None), dict):
                best_res.diagnostics["selected_attempt"] = str(attempt["name"])
        if verbose:
            print("[factorized symbolic search] Attempt probe RMS: {:.6e}".format(probe_rms))

        if not bool(attempt.get("restricted", False)):
            fallback = False
            if int(i) == 0 and str(attempt.get("name", "")) == "full":
                fallback = bool(_should_fallback_from_full_factorized_search_only(rescue_candidate, rescue_cfg))
                attempt_log["fallback_triggered"] = bool(fallback)
                if verbose:
                    print(
                        "[factorized symbolic search] {}".format(
                            "Full pass failed badly; trying generic restricted feature searches."
                            if fallback
                            else "Full pass kept; not trying additional feature searches."
                        )
                    )
            if fallback and i < len(attempts) - 1:
                continue
            break

        widen = _should_widen_from_restricted_rescue(rescue_candidate, primary_res, rescue_cfg)
        attempt_log["widened_to_full"] = bool(widen)
        if verbose:
            print(
                "[factorized symbolic search] {}".format(
                    "Restricted pass failed badly; widening to full search."
                    if widen
                    else "Restricted pass kept; not widening to full search."
                )
            )
        if not widen:
            break

    if isinstance(getattr(best_res, "diagnostics", None), dict):
        best_res.diagnostics["rescue_attempts"] = attempt_logs
        best_res.diagnostics["rescue_attempts_available"] = int(len(attempts_available))
        best_res.diagnostics["rescue_attempts_run"] = int(len(attempt_logs))
        best_res.diagnostics["rescue_max_attempts"] = None if max_attempts is None else int(max_attempts)
        best_res.diagnostics["rescue_attempts_capped"] = bool(
            max_attempts is not None and int(len(attempts)) < int(len(attempts_available))
        )
        best_res.diagnostics["shortlist_union"] = _merge_factorized_search_attempt_shortlists(attempt_shortlists)
    return best_res


def _factorized_search_trigger_reason(primary, rescue_cfg: FactorizedSearchDERescueConfig) -> str | None:
    mode = str(getattr(rescue_cfg, "mode", "never")).strip().lower()
    if mode == "never":
        return None
    if mode == "always":
        return "mode_always"

    rms = candidate_probe_rms(primary)
    cond = getattr(primary, "condition_number", None)
    try:
        cond = float(cond) if cond is not None else None
    except Exception:
        cond = None

    if not math.isfinite(rms):
        return "nonfinite_rms"
    if rms > float(rescue_cfg.trigger_val_rms):
        return "high_val_rms"
    if cond is not None and math.isfinite(cond) and cond > float(rescue_cfg.trigger_cond):
        return "ill_conditioned"
    return None


def _automatic_gs_carrier_is_nontrivial(ast_node: Any) -> bool:
    """Return whether a carrier adds structure beyond one affine coordinate."""

    from nestynet_sr.sr_core.bridges import AddNode, AtomNode, ConstNode, MulNode

    if ast_node is None or isinstance(ast_node, (AtomNode, ConstNode)):
        return False
    if isinstance(ast_node, AddNode):
        return _automatic_gs_carrier_is_nontrivial(
            ast_node.left
        ) or _automatic_gs_carrier_is_nontrivial(ast_node.right)
    if isinstance(ast_node, MulNode):
        left_const = isinstance(ast_node.left, ConstNode)
        right_const = isinstance(ast_node.right, ConstNode)
        if left_const:
            return _automatic_gs_carrier_is_nontrivial(ast_node.right)
        if right_const:
            return _automatic_gs_carrier_is_nontrivial(ast_node.left)
        return True
    return True


def _attach_automatic_gs_carriers(primary: Any, cfg: DESearchConfig) -> dict[str, Any]:
    """Expose certified nonlinear carriers to the bounded FSS challenger."""

    diagnostic: dict[str, Any] = {
        "enabled": bool(
            getattr(cfg, "gs_enable", False)
            and getattr(cfg, "gs_de_auto_nonlinear", True)
        ),
        "attached": False,
        "nontrivial_carrier_count": 0,
        "trigger_fss": False,
        "reason": "automatic_nonlinear_gs_disabled",
    }
    if not diagnostic["enabled"] or primary is None:
        return diagnostic
    certificate = getattr(primary, "determining_certificate", None)
    compilation = getattr(certificate, "compiled_invariants", None)
    if compilation is None:
        diagnostic["reason"] = "no_certified_carrier_compilation"
        if isinstance(certificate, dict):
            certificate["automatic_fss_handoff"] = diagnostic
        return diagnostic
    try:
        from nestynet_sr.sr_gs.de_bridge import nonlinear_invariant_de_term_rows

        rows = nonlinear_invariant_de_term_rows(
            compilation,
            x_axis=int(getattr(cfg, "x_axis", 0)),
        )
    except Exception as exc:
        diagnostic["reason"] = f"carrier_bridge_failed:{type(exc).__name__}"
        if isinstance(certificate, dict):
            certificate["automatic_fss_handoff"] = diagnostic
        return diagnostic

    nontrivial = [row for row in rows if _automatic_gs_carrier_is_nontrivial(row[0])]
    diagnostic.update(
        {
            "attached": bool(rows),
            "carrier_count": int(len(rows)),
            "nontrivial_carrier_count": int(len(nontrivial)),
            "sources": [str(source) for _, source, _ in rows],
            "trigger_fss": bool(
                nontrivial
                and getattr(cfg, "gs_de_auto_fss", True)
                and int(getattr(cfg, "gs_de_auto_fss_max_attempts", 1)) > 0
            ),
            "reason": (
                "certified_nontrivial_carriers"
                if nontrivial
                else "only_affine_coordinate_carriers"
            ),
        }
    )
    cfg.gs_de_compiled_nonlinear_invariants = compilation
    if isinstance(certificate, dict):
        certificate["automatic_fss_handoff"] = diagnostic
    return diagnostic


def _bound_automatic_gs_fss(
    rescue_cfg: FactorizedSearchDERescueConfig,
    cfg: DESearchConfig,
) -> dict[str, int]:
    """Apply automatic-only caps without increasing explicit smaller budgets."""

    hp = getattr(rescue_cfg, "hp", None)
    caps = {
        "n_iter": max(1, int(getattr(cfg, "gs_de_auto_fss_n_iter", 1500))),
        "n_fit": max(16, int(getattr(cfg, "gs_de_auto_fss_n_fit", 1024))),
        "n_probe": max(16, int(getattr(cfg, "gs_de_auto_fss_n_probe", 1024))),
        "max_depth": max(1, int(getattr(cfg, "gs_de_auto_fss_max_depth", 4))),
        "return_topk": max(
            1, int(getattr(cfg, "gs_de_auto_fss_return_topk", 8))
        ),
    }
    applied: dict[str, int] = {}
    if hp is not None:
        for name, cap in caps.items():
            current = getattr(hp, name, None)
            bounded = cap if current is None else min(int(current), cap)
            setattr(hp, name, bounded)
            applied[name] = bounded
    rescue_cfg.validate_integrate_topk = min(
        int(getattr(rescue_cfg, "validate_integrate_topk", 0)),
        int(caps["return_topk"]),
    )
    rescue_cfg.budget_scope = "global"
    return applied


def _factorized_trigger_reason(primary, rescue_cfg: FactorizedDERescueConfig) -> str | None:
    mode = str(getattr(rescue_cfg, "mode", "never")).strip().lower()
    if mode == "never":
        return None
    if mode == "always":
        return "mode_always"

    rms = candidate_probe_rms(primary)
    cond = getattr(primary, "condition_number", None)
    try:
        cond = float(cond) if cond is not None else None
    except Exception:
        cond = None

    if not math.isfinite(rms):
        return "nonfinite_rms"
    if rms > float(rescue_cfg.trigger_val_rms):
        return "high_val_rms"
    if cond is not None and math.isfinite(cond) and cond > float(rescue_cfg.trigger_cond):
        return "ill_conditioned"
    return None


def _better_than(lhs, rhs, *, rel_factor: float) -> bool:
    if lhs is None:
        return False
    if not _candidate_domain_safe(lhs):
        return False
    if rhs is None:
        return True
    if not _candidate_domain_safe(rhs):
        return True
    s1 = candidate_probe_rms(lhs)
    s0 = candidate_probe_rms(rhs)
    return bool(math.isfinite(s1) and (not math.isfinite(s0) or s1 < float(rel_factor) * s0))


def should_escalate_to_factorized_search(primary, rescue_cfg: FactorizedSearchDERescueConfig) -> bool:
    """Decide whether factorized symbolic search rescue should run for the first-line result."""
    return _factorized_search_trigger_reason(primary, rescue_cfg) is not None


def choose_best_de_candidate(primary, rescue, rescue_cfg: FactorizedSearchDERescueConfig):
    """Select between the first-line and rescue candidate."""
    if rescue is None:
        return primary, "stlsq"
    if not _candidate_domain_safe(rescue):
        return primary, "stlsq"
    if primary is None or not _candidate_domain_safe(primary):
        return rescue, "factorized_search"

    s0 = candidate_probe_rms(primary)
    s1 = candidate_probe_rms(rescue)
    if math.isfinite(s1) and (not math.isfinite(s0) or s1 < float(rescue_cfg.replace_rel_factor) * s0):
        return rescue, "factorized_search"
    return primary, "stlsq"


def _factorized_search_shortlist_key(row: dict[str, Any]) -> str:
    payload = {
        "engine": row.get("engine", "factorized_search"),
        "kind": row.get("kind", "factorized"),
        "order": row.get("order", None),
        "x_axis": row.get("x_axis", None),
        "include_x": row.get("include_x", None),
        "include_u": row.get("include_u", None),
        "include_du": row.get("include_du", None),
        "constants_ordered": row.get("constants_ordered", None),
        "expr_ast": row.get("expr_ast", None),
        "mapping": row.get("mapping", None),
        "mapping_kind": row.get("mapping_kind", None),
        "canonical_equation": row.get("canonical_equation", None),
    }
    return json.dumps(_jsonable_report_value(payload), sort_keys=True)


def _factorized_search_selected_match_key(row: dict[str, Any]) -> str:
    payload = {
        "engine": row.get("engine", "factorized_search"),
        "kind": row.get("kind", "factorized"),
        "order": row.get("order", None),
        "x_axis": row.get("x_axis", None),
        "expr_ast": row.get("expr_ast", None),
        "mapping": row.get("mapping", None),
        "mapping_kind": row.get("mapping_kind", None),
        "canonical_equation": row.get("canonical_equation", None),
    }
    return json.dumps(_jsonable_report_value(payload), sort_keys=True)


def _extract_factorized_search_shortlist_from_result(result) -> list[dict[str, Any]]:
    if not isinstance(result, FactorizedSearchDEResult):
        return []
    diagnostics = getattr(result, "diagnostics", {}) or {}
    if not isinstance(diagnostics, dict):
        return []
    union = diagnostics.get("shortlist_union", None)
    if isinstance(union, list):
        return [dict(row) for row in union if isinstance(row, dict)]

    report = diagnostics.get("report", None)
    if not isinstance(report, dict):
        return []
    hp = report.get("hp", {}) if isinstance(report.get("hp", None), dict) else {}
    limit = hp.get("return_topk", None)
    shortlist = factorized_search_report_shortlist(report, limit=limit)
    return [dict(row) for row in shortlist if isinstance(row, dict)]


def _merge_factorized_search_attempt_shortlists(
    shortlists: list[tuple[str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for attempt_name, rows in shortlists:
        for row in list(rows or []):
            if not isinstance(row, dict):
                continue
            payload = dict(row)
            payload.setdefault("attempt_name", str(attempt_name))
            key = _factorized_search_shortlist_key(payload)
            if key in seen:
                continue
            seen.add(key)
            rank = int(len(out))
            payload["candidate_rank"] = rank
            payload["shortlist_rank"] = rank
            out.append(payload)
    return out


def _find_internal_selected_shortlist_rank(
    result: FactorizedSearchDEResult,
    shortlist: list[dict[str, Any]],
) -> int | None:
    target = {
        "engine": str(getattr(result, "engine", "factorized_search") or "factorized_search"),
        "kind": "factorized",
        "order": int(result.order),
        "x_axis": int(result.x_axis),
        "expr_ast": _jsonable_report_value(result.expr_ast),
        "mapping": _jsonable_report_value(result.mapping),
        "mapping_kind": str(result.mapping_kind),
        "canonical_equation": str(result.canonical_equation),
    }
    target_key = _factorized_search_selected_match_key(target)
    for row in list(shortlist or []):
        if not isinstance(row, dict):
            continue
        if _factorized_search_selected_match_key(row) == target_key:
            try:
                return int(row.get("candidate_rank", row.get("shortlist_rank", 0)))
            except Exception:
                return 0
    return None


def serialize_de_candidate(obj, *, validation_candidate=None) -> dict | None:
    """Normalize first-line and factorized symbolic search DE results into one report shape."""
    if obj is None:
        return None

    if isinstance(obj, FactorizedSearchDEResult):
        diagnostics = getattr(obj, "diagnostics", {}) or {}
        shortlist = _extract_factorized_search_shortlist_from_result(obj)
        internal_selected_rank = _find_internal_selected_shortlist_rank(obj, shortlist) if shortlist else None
        return {
            "engine": str(getattr(obj, "engine", "factorized_search") or "factorized_search"),
            "kind": "factorized",
            "order": int(obj.order),
            "x_axis": int(obj.x_axis),
            "rms_train": None,
            "rms_val": None,
            "probe_mse": float(obj.probe_mse),
            "probe_rms": float(obj.probe_rms),
            "condition_number": None,
            "num_terms": None,
            "coefficients": None,
            "terms": None,
            "feature_names": list(obj.feature_names),
            "expr_ast": _jsonable_report_value(obj.expr_ast),
            "mapping": _jsonable_report_value(obj.mapping),
            "mapping_kind": str(obj.mapping_kind),
            "rhs_ast": None if obj.rhs_ast is None else repr(obj.rhs_ast),
            "residual_ast": None if obj.residual_ast is None else repr(obj.residual_ast),
            "rhs_ast_raw": None if getattr(obj, "rhs_ast_raw", None) is None else repr(obj.rhs_ast_raw),
            "residual_ast_raw": None
            if getattr(obj, "residual_ast_raw", None) is None
            else repr(obj.residual_ast_raw),
            "rhs_ast_simplified": None
            if getattr(obj, "rhs_ast_simplified", None) is None
            else repr(obj.rhs_ast_simplified),
            "residual_ast_simplified": None
            if getattr(obj, "residual_ast_simplified", None) is None
            else repr(obj.residual_ast_simplified),
            "residual_asts": None,
            "canonical_equation": str(obj.canonical_equation),
            "canonical_equation_raw": getattr(obj, "canonical_equation_raw", None),
            "canonical_equation_simplified": getattr(obj, "canonical_equation_simplified", None),
            "canonical_equations": None,
            "varpro_metadata": None,
            "stageb_residual": _jsonable_report_value(getattr(obj, "stageb_residual_metadata", None)),
            "term_asts_json": None,
            "validation_candidate": None,
            "shortlist": _jsonable_report_value(shortlist),
            "internal_selected_shortlist_rank": _jsonable_report_value(internal_selected_rank),
            "diagnostics": _jsonable_report_value(diagnostics),
        }

    if isinstance(obj, FactorizedDEResult):
        diagnostics = getattr(obj, "diagnostics", {}) or {}
        shortlist_rows = list(diagnostics.get("shortlist_rows", []) or []) if isinstance(diagnostics, dict) else []
        shortlist = []
        for rank, row in enumerate(shortlist_rows):
            if not isinstance(row, dict):
                continue
            order = int(row.get("order", obj.order))
            row_nonanchor = row.get("nonanchor_ast", None)
            row_residual = row.get("residual_ast", None)
            row_payload = {
                "engine": "factorized",
                "kind": "factorized_blocks",
                "order": int(order),
                "x_axis": int(obj.x_axis),
                "rms_train": None,
                "rms_val": None,
                "probe_mse": float(row.get("probe_mse", float("inf"))),
                "probe_rms": float(row.get("probe_rms", float("inf"))),
                "symbolic_size_raw": _jsonable_report_value(row.get("symbolic_size_raw", None)),
                "symbolic_size_simplified": _jsonable_report_value(row.get("symbolic_size_simplified", None)),
                "condition_number": None,
                "num_terms": None,
                "coefficients": None,
                "terms": None,
                "feature_names": None,
                "expr_ast": None,
                "mapping": None,
                "mapping_kind": None,
                "rhs_ast": None,
                "residual_ast": None if row_residual is None else repr(row_residual),
                "residual_ast_raw": None
                if row.get("residual_ast_raw", None) is None
                else repr(row.get("residual_ast_raw", None)),
                "residual_ast_simplified": None
                if row.get("residual_ast_simplified", None) is None
                else repr(row.get("residual_ast_simplified", None)),
                "residual_asts": None,
                "canonical_equation": str(row.get("canonical_equation", "")),
                "canonical_equation_raw": str(row.get("canonical_equation_raw", row.get("canonical_equation", ""))),
                "canonical_equation_simplified": str(
                    row.get("canonical_equation_simplified", row.get("canonical_equation", ""))
                ),
                "canonical_equations": None,
                "varpro_metadata": None,
                "stageb_residual": None,
                "term_asts_json": None,
                "validation_candidate": {
                    "order": int(order),
                    "x_axis": int(obj.x_axis),
                    "coefficients": [1.0],
                    "term_asts_json": [_serialize_de_ast(row_nonanchor)],
                }
                if row_nonanchor is not None
                else None,
                "candidate_rank": int(rank),
                "shortlist_rank": int(rank),
                "lane": str(row.get("lane", "")),
                "family": str(row.get("family", "")),
                "base_mode": str(row.get("base_mode", "")),
                "evidence_tier": str(row.get("evidence_tier", "")),
                "witness_kind": str(row.get("witness_kind", "")),
                "consistency_score": _jsonable_report_value(row.get("consistency_score", None)),
                "consistency_pairs": _jsonable_report_value(row.get("consistency_pairs", None)),
                "consistency_total_pairs": _jsonable_report_value(row.get("consistency_total_pairs", None)),
                "shape_score": _jsonable_report_value(row.get("shape_score", None)),
                "sign_changes": _jsonable_report_value(row.get("sign_changes", None)),
                "curvature_ratio": _jsonable_report_value(row.get("curvature_ratio", None)),
                "tv_ratio": _jsonable_report_value(row.get("tv_ratio", None)),
                "carrier_ast": None if row.get("carrier_ast", None) is None else repr(row.get("carrier_ast", None)),
                "coord_ast": None if row.get("coord_ast", None) is None else repr(row.get("coord_ast", None)),
                "coeff_ast": None if row.get("coeff_ast", None) is None else repr(row.get("coeff_ast", None)),
                "coeff_asts": [
                    repr(coeff_ast) for coeff_ast in list(row.get("coeff_asts", []) or []) if coeff_ast is not None
                ] or None,
                "coeff_expr": str(row.get("coeff_expr", "")),
                "projection_kind": str(row.get("projection_kind", "")),
                "projection_support": _jsonable_report_value(row.get("projection_support", None)),
                "projection_coeffs": _jsonable_report_value(row.get("projection_coeffs", None)),
                "projection_full_basis_size": _jsonable_report_value(row.get("projection_full_basis_size", None)),
                "projection_signature": str(row.get("projection_signature", "")),
                "projection_snap_report": _jsonable_report_value(row.get("projection_snap_report", None)),
                "projection_snap_cost": _jsonable_report_value(row.get("projection_snap_cost", None)),
                "typed_metadata": _factorized_typed_metadata(row),
            }
            row_payload.update(_factorized_typed_flat_fields(row))
            shortlist.append(row_payload)

        internal_selected_rank = None
        if shortlist:
            try:
                internal_selected_rank = int(diagnostics.get("selected_shortlist_rank", 0))
            except Exception:
                internal_selected_rank = 0
        selected_row = _selected_factorized_shortlist_row(diagnostics, shortlist_rows)
        selected_metadata_source = dict(diagnostics)
        if isinstance(selected_row, dict):
            selected_metadata_source.update(selected_row)
        selected_typed_metadata = _factorized_typed_metadata(selected_metadata_source)
        selected_typed_flat = _factorized_typed_flat_fields(selected_metadata_source)

        payload = {
            "engine": str(getattr(obj, "engine", "factorized") or "factorized"),
            "kind": "factorized_blocks",
            "order": int(obj.order),
            "x_axis": int(obj.x_axis),
            "rms_train": None,
            "rms_val": None,
            "probe_mse": float(obj.probe_mse),
            "probe_rms": float(obj.probe_rms),
            "condition_number": None,
            "num_terms": None,
            "coefficients": None,
            "terms": None,
            "feature_names": None,
            "expr_ast": None,
            "mapping": None,
            "mapping_kind": None,
            "rhs_ast": None,
            "residual_ast": None if obj.residual_ast is None else repr(obj.residual_ast),
            "residual_ast_raw": None
            if getattr(obj, "residual_ast_raw", None) is None
            else repr(obj.residual_ast_raw),
            "residual_ast_simplified": None
            if getattr(obj, "residual_ast_simplified", None) is None
            else repr(obj.residual_ast_simplified),
            "residual_asts": None,
            "canonical_equation": str(obj.canonical_equation),
            "canonical_equation_raw": getattr(obj, "canonical_equation_raw", None),
            "canonical_equation_simplified": getattr(obj, "canonical_equation_simplified", None),
            "canonical_equations": None,
            "varpro_metadata": None,
            "stageb_residual": None,
            "term_asts_json": None,
            "validation_candidate": _factorized_validation_candidate(obj),
            "shortlist": shortlist,
            "auxiliary_rollout_candidates": _jsonable_report_value(
                diagnostics.get("auxiliary_rollout_candidates", [])
            )
            if isinstance(diagnostics, dict)
            else [],
            "internal_selected_shortlist_rank": _jsonable_report_value(internal_selected_rank),
            "lane": str(diagnostics.get("lane", "")) if isinstance(diagnostics, dict) else None,
            "family": str(diagnostics.get("family", "")) if isinstance(diagnostics, dict) else None,
            "base_mode": str(diagnostics.get("base_mode", "")) if isinstance(diagnostics, dict) else None,
            "evidence_tier": str(diagnostics.get("evidence_tier", "")) if isinstance(diagnostics, dict) else None,
            "witness_kind": str(diagnostics.get("witness_kind", "")) if isinstance(diagnostics, dict) else None,
            "consistency_score": _jsonable_report_value(diagnostics.get("consistency_score", None))
            if isinstance(diagnostics, dict)
            else None,
            "consistency_pairs": _jsonable_report_value(diagnostics.get("consistency_pairs", None))
            if isinstance(diagnostics, dict)
            else None,
            "consistency_total_pairs": _jsonable_report_value(diagnostics.get("consistency_total_pairs", None))
            if isinstance(diagnostics, dict)
            else None,
            "shape_score": _jsonable_report_value(diagnostics.get("shape_score", None))
            if isinstance(diagnostics, dict)
            else None,
            "sign_changes": _jsonable_report_value(diagnostics.get("sign_changes", None))
            if isinstance(diagnostics, dict)
            else None,
            "curvature_ratio": _jsonable_report_value(diagnostics.get("curvature_ratio", None))
            if isinstance(diagnostics, dict)
            else None,
            "tv_ratio": _jsonable_report_value(diagnostics.get("tv_ratio", None))
            if isinstance(diagnostics, dict)
            else None,
            "carrier_ast": None
            if selected_metadata_source.get("carrier_ast", None) is None
            else repr(selected_metadata_source.get("carrier_ast", None)),
            "coord_ast": None
            if selected_metadata_source.get("coord_ast", None) is None
            else repr(selected_metadata_source.get("coord_ast", None)),
            "coeff_ast": None
            if selected_metadata_source.get("coeff_ast", None) is None
            else repr(selected_metadata_source.get("coeff_ast", None)),
            "coeff_asts": [
                repr(coeff_ast)
                for coeff_ast in list(selected_metadata_source.get("coeff_asts", []) or [])
                if coeff_ast is not None
            ]
            or None,
            "coeff_expr": str(selected_metadata_source.get("coeff_expr", "")),
            "typed_metadata": selected_typed_metadata,
            "diagnostics": _jsonable_report_value(diagnostics),
        }
        payload.update(selected_typed_flat)
        return payload

    coeffs = getattr(obj, "coeffs", None)
    coeffs_json = _jsonable_report_value(coeffs) if coeffs is not None else None
    term_asts = list(getattr(obj, "term_asts", []) or [])
    term_asts_json = [_serialize_de_ast(term_ast) for term_ast in term_asts]
    residual_ast = getattr(obj, "residual_ast", None)
    residual_asts = getattr(obj, "residual_asts", None)
    cond_num = getattr(obj, "condition_number", None)
    try:
        cond_num = float(cond_num) if cond_num is not None else None
    except Exception:
        cond_num = None

    canonical_equation = obj.format_equation() if hasattr(obj, "format_equation") else None
    canonical_equations = None
    dataset_ids = getattr(obj, "dataset_ids", None)
    if hasattr(obj, "format_equation_for_dataset") and isinstance(coeffs, torch.Tensor) and coeffs.ndim == 2:
        canonical_equations = [
            obj.format_equation_for_dataset(i)
            for i in range(int(coeffs.shape[0]))
        ]

    return {
        "engine": "stlsq",
        "kind": "library",
        "order": int(obj.order),
        "x_axis": int(obj.x_axis),
        "rms_train": _jsonable_report_value(getattr(obj, "rms_train", None)),
        "rms_val": _jsonable_report_value(getattr(obj, "rms_val", None)),
        "probe_mse": None,
        "probe_rms": candidate_probe_rms(obj),
        "condition_number": cond_num,
        "num_terms": len(term_asts),
        "coefficients": coeffs_json,
        "terms": [repr(t) if t is not None else "1" for t in term_asts],
        "term_asts_json": term_asts_json,
        "feature_names": None,
        "expr_ast": None,
        "mapping": None,
        "mapping_kind": None,
        "rhs_ast": None,
        "residual_ast": None if residual_ast is None else repr(residual_ast),
        "residual_asts": _jsonable_report_value([repr(t) for t in residual_asts]) if residual_asts is not None else None,
        "canonical_equation": canonical_equation,
        "canonical_equations": canonical_equations,
        "varpro_metadata": _jsonable_report_value(getattr(obj, "varpro_metadata", None)),
        "stageb_residual": _jsonable_report_value(getattr(obj, "stageb_residual_metadata", None)),
        "term_sources": _jsonable_report_value(getattr(obj, "term_sources", None)),
        "prolongation_metadata": _jsonable_report_value(getattr(obj, "prolongation_metadata", None)),
        "determining_certificate": _jsonable_report_value(getattr(obj, "determining_certificate", None)),
        "expr_ir_report": _jsonable_report_value(getattr(obj, "expr_ir_report", None)),
        "expr_ir_reports_by_order": _jsonable_report_value(
            getattr(obj, "expr_ir_reports_by_order", None)
        ),
        "validation_candidate": _jsonable_report_value(validation_candidate),
        "shortlist": None,
        "diagnostics": {
            "dataset_ids": list(dataset_ids) if dataset_ids is not None else None,
        },
    }


def _serialize_rescue_reason(
    primary,
    rescue_cfg: FactorizedSearchDERescueConfig,
    *,
    triggered: bool,
    trigger_reason: str | None,
) -> dict:
    if primary is None:
        return {
            "mode": str(getattr(rescue_cfg, "mode", "never")),
            "triggered": bool(triggered),
            "trigger": trigger_reason,
            "trigger_val_rms": float(getattr(rescue_cfg, "trigger_val_rms", 1.0e-3)),
            "trigger_rel_rms": float(getattr(rescue_cfg, "trigger_rel_rms", 1.0e-3)),
            "trigger_cond": float(getattr(rescue_cfg, "trigger_cond", 1.0e8)),
            "replace_rel_factor": float(getattr(rescue_cfg, "replace_rel_factor", 0.98)),
            "primary_rms": None,
            "primary_condition_number": None,
        }
    cond = getattr(primary, "condition_number", None)
    try:
        cond = float(cond) if cond is not None else None
    except Exception:
        cond = None
    return {
        "mode": str(getattr(rescue_cfg, "mode", "never")),
        "triggered": bool(triggered),
        "trigger": trigger_reason,
        "trigger_val_rms": float(getattr(rescue_cfg, "trigger_val_rms", 1.0e-3)),
        "trigger_rel_rms": float(getattr(rescue_cfg, "trigger_rel_rms", 1.0e-3)),
        "trigger_cond": float(getattr(rescue_cfg, "trigger_cond", 1.0e8)),
        "replace_rel_factor": float(getattr(rescue_cfg, "replace_rel_factor", 0.98)),
        "primary_rms": candidate_probe_rms(primary),
        "primary_condition_number": cond,
    }


def _serialize_factorized_reason(
    primary,
    rescue_cfg: FactorizedDERescueConfig,
    *,
    triggered: bool,
    trigger_reason: str | None,
) -> dict:
    if primary is None:
        return {
            "mode": str(getattr(rescue_cfg, "mode", "never")),
            "triggered": bool(triggered),
            "trigger": trigger_reason,
            "trigger_val_rms": float(getattr(rescue_cfg, "trigger_val_rms", 1.0e-3)),
            "trigger_cond": float(getattr(rescue_cfg, "trigger_cond", 1.0e8)),
            "replace_rel_factor": float(getattr(rescue_cfg, "replace_rel_factor", 0.98)),
            "primary_rms": None,
            "primary_condition_number": None,
        }
    cond = getattr(primary, "condition_number", None)
    try:
        cond = float(cond) if cond is not None else None
    except Exception:
        cond = None
    return {
        "mode": str(getattr(rescue_cfg, "mode", "never")),
        "triggered": bool(triggered),
        "trigger": trigger_reason,
        "trigger_val_rms": float(getattr(rescue_cfg, "trigger_val_rms", 1.0e-3)),
        "trigger_cond": float(getattr(rescue_cfg, "trigger_cond", 1.0e8)),
        "replace_rel_factor": float(getattr(rescue_cfg, "replace_rel_factor", 0.98)),
        "primary_rms": candidate_probe_rms(primary),
        "primary_condition_number": cond,
    }


def write_de_json_report(
    filepaths: list,
    report_path: str,
    surrogate_losses: list,
    de_result,
    args: argparse.Namespace,
    walltime: float,
    *,
    primary_result=None,
    factorized_result=None,
    rescue_result=None,
    selected_engine: str | None = None,
    factorized_cfg: FactorizedDERescueConfig | None = None,
    rescue_cfg: FactorizedSearchDERescueConfig | None = None,
    factorized_triggered: bool = False,
    factorized_trigger_reason: str | None = None,
    rescue_triggered: bool = False,
    rescue_trigger_reason: str | None = None,
    primary_validation_candidate=None,
    selected_validation_candidate=None,
    de_candidate_eval_report=None,
):
    """Write a JSON report for the DE discovery run."""
    if primary_result is None and not isinstance(de_result, (FactorizedSearchDEResult, FactorizedDEResult)):
        primary_result = de_result
    if factorized_cfg is None:
        factorized_cfg = build_factorized_rescue_config_from_args(args)
    if rescue_cfg is None:
        rescue_cfg = build_factorized_search_rescue_config_from_args(args)

    selected_payload = serialize_de_candidate(
        de_result,
        validation_candidate=selected_validation_candidate,
    )
    first_line_payload = serialize_de_candidate(
        primary_result,
        validation_candidate=primary_validation_candidate,
    )
    factorized_payload = serialize_de_candidate(factorized_result)
    rescue_payload = serialize_de_candidate(rescue_result)
    selected_engine = str(selected_engine or (selected_payload or {}).get("engine", "stlsq"))
    de_coe_mode = str(getattr(args, "de_coe_mode", "off") or "off").strip().lower()
    ladder_report = run_legacy_de_ladder(
        LegacyDEResultPayloads(
            first_line=first_line_payload,
            factorized=factorized_payload,
            factorized_search=rescue_payload,
            selected=selected_payload,
            selected_engine=selected_engine,
        ),
        policy=DELadderPolicy(
            coe_mode=de_coe_mode,
            source="run_de",
            reservoir_scouts_requested=int(getattr(args, "de_coe_reservoir_scouts", 0) or 0),
        ),
    )
    proposal_slate = ladder_report.proposal_slate
    selected_payload = ladder_report.selected_payload
    selected_engine = ladder_report.selected_engine
    internal_selected_engine = ladder_report.internal_selected_engine
    internal_selected_payload = ladder_report.internal_selected_payload
    committee_decision = ladder_report.committee_decision
    committee_adjudicated = ladder_report.committee_adjudicated
    committee_adjudication_fallback = ladder_report.committee_adjudication_fallback
    factorized_de_diag = None
    for payload in (selected_payload, factorized_payload, rescue_payload):
        if not isinstance(payload, dict):
            continue
        diagnostics = payload.get("diagnostics", None)
        if not isinstance(diagnostics, dict):
            continue
        diag = diagnostics.get("factorized_de", None)
        if isinstance(diag, dict):
            factorized_de_diag = diag
            break
    factorized_attempted = factorized_result is not None
    rescue_attempted = rescue_result is not None
    factorized_reason = _serialize_factorized_reason(
        primary_result,
        factorized_cfg,
        triggered=bool(factorized_triggered),
        trigger_reason=factorized_trigger_reason,
    )
    rescue_reason = _serialize_rescue_reason(
        primary_result,
        rescue_cfg,
        triggered=bool(rescue_triggered),
        trigger_reason=rescue_trigger_reason,
    )

    de_discovery = {
        "selected_engine": selected_engine,
        "internal_selected_engine": internal_selected_engine,
        "internal_selected": internal_selected_payload,
        "committee_adjudicated": bool(committee_adjudicated),
        "committee_adjudication_fallback": bool(committee_adjudication_fallback),
        "committee_csr_requested": bool(getattr(args, "de_coe_csr_on_ties", False)),
        "committee_reservoir_requested": bool(
            de_coe_mode == "reservoir" or int(getattr(args, "de_coe_reservoir_scouts", 0) or 0) > 0
        ),
        "factorized_search_only": bool(getattr(args, "factorized_search_only", False)),
        "factorized_de": bool(getattr(args, "factorized_de", False)),
        "factorized_attempted": bool(factorized_attempted),
        "factorized_triggered": bool(factorized_triggered),
        "factorized_reason": factorized_reason,
        "factorized_de_diagnostics": factorized_de_diag,
        "rescue_attempted": bool(rescue_attempted),
        "rescue_triggered": bool(rescue_triggered),
        "rescue_reason": rescue_reason,
        "first_line": first_line_payload,
        "factorized_rescue": factorized_payload,
        "factorized_search_rescue": rescue_payload,
        "proposal_slate": proposal_slate,
        "committee_decision": committee_decision,
        "de_candidate_eval": _jsonable_report_value(de_candidate_eval_report),
        "selected": selected_payload,
    }
    for key in (
        "order",
        "x_axis",
        "rms_train",
        "rms_val",
        "probe_mse",
        "probe_rms",
        "condition_number",
        "num_terms",
        "coefficients",
        "terms",
        "feature_names",
        "expr_ast",
        "mapping",
        "mapping_kind",
        "rhs_ast",
        "residual_ast",
        "residual_asts",
        "canonical_equation",
        "canonical_equations",
        "varpro_metadata",
        "stageb_residual",
        "term_sources",
        "prolongation_metadata",
        "determining_certificate",
        "expr_ir_report",
        "expr_ir_reports_by_order",
        "term_asts_json",
        "validation_candidate",
        "shortlist",
    ):
        de_discovery[key] = None if selected_payload is None else selected_payload.get(key)

    report = {
        "metadata": {
            "dataset": filepaths[0]
            if isinstance(filepaths, (list, tuple)) and len(filepaths) > 0
            else str(filepaths),
            "datasets": list(filepaths)
            if isinstance(filepaths, (list, tuple))
            else [str(filepaths)],
            "x_axis": de_result.x_axis,
            "device": str(args.device) if args.device else "auto",
            "walltime_hours": walltime,
        },
        "surrogate": {
            "num_segments": args.num_segments,
            "epochs": args.epochs,
            "val_losses": [
                float(x)
                for x in (
                    surrogate_losses
                    if isinstance(surrogate_losses, (list, tuple))
                    else [surrogate_losses]
                )
            ],
            "loss_target": args.loss_target,
        },
        "de_discovery": de_discovery,
        "config": {
            "order_candidates": args.order_candidates,
            "max_x_power": args.max_x_power,
            "max_u_power": args.max_u_power,
            "max_xu_total_degree": args.max_xu_total_degree,
            "include_xdu": args.include_xdu,
            "include_inv_xdu": args.include_inv_xdu,
            "include_inv_xu": args.include_inv_xu,
            "include_inv_x2u": args.include_inv_x2u,
            "include_du": args.include_du,
            "include_d2u": args.include_d2u,
            "include_udu": args.include_udu,
            "stlsq_lambda": args.stlsq_lambda,
            "sparsity_penalty": args.sparsity_penalty,
            "enforce_units": bool(getattr(args, "enforce_units", False)),
            "units_policy": getattr(args, "units_policy", None),
            "nn_units_semantics": getattr(args, "nn_units_semantics", None),
            "factorized_rescue": str(getattr(args, "factorized_rescue", "never")),
            "factorized_two_block_shared_coord": str(
                getattr(args, "factorized_two_block_shared_coord", "never")
            ),
            "factorized_de_typed_lane_workers": int(
                getattr(args, "factorized_de_typed_lane_workers", 1) or 1
            ),
            "factorized_search_rescue": str(getattr(args, "factorized_search_rescue", "never")),
            "factorized_search_preset": str(getattr(args, "factorized_search_preset", "default")),
            "factorized_search_trigger_val_rms": float(
                getattr(args, "factorized_search_trigger_val_rms", 1.0e-3)
            ),
            "factorized_search_trigger_rel_rms": float(
                getattr(args, "factorized_search_trigger_rel_rms", 1.0e-3)
            ),
            "factorized_search_trigger_cond": float(
                getattr(args, "factorized_search_trigger_cond", 1.0e8)
            ),
            "factorized_search_replace_rel_factor": float(
                getattr(args, "factorized_search_replace_rel_factor", 0.98)
            ),
            "factorized_search_n_iter": getattr(args, "factorized_search_n_iter", None),
            "factorized_search_max_depth": getattr(args, "factorized_search_max_depth", None),
            "factorized_search_n_fit": getattr(args, "factorized_search_n_fit", None),
            "factorized_search_n_probe": getattr(args, "factorized_search_n_probe", None),
            "factorized_search_return_topk": getattr(args, "factorized_search_return_topk", None),
            "factorized_search_max_attempts": getattr(args, "factorized_search_max_attempts", None),
            "factorized_search_effective_max_attempts": _factorized_search_max_attempts(rescue_cfg),
            "factorized_search_integrate_topk": getattr(args, "factorized_search_integrate_topk", None),
            "factorized_search_validate_integrate_topk": int(
                getattr(rescue_cfg, "validate_integrate_topk", 0)
            ),
            "factorized_search_direct_generator_witness_topk": int(
                getattr(rescue_cfg, "direct_generator_witness_topk", 0)
            ),
            "factorized_search_budget_scope": str(getattr(rescue_cfg, "budget_scope", "per_group")),
            "factorized_de_whole_rhs": str(
                getattr(args, "factorized_de_whole_rhs", "auto")
            ),
            "factorized_search_de_refine_mode": str(
                getattr(args, "factorized_search_de_refine_mode", "rare_final_polish")
            ),
            "de_coe_mode": de_coe_mode,
            "de_coe_csr_on_ties": bool(getattr(args, "de_coe_csr_on_ties", False)),
            "de_coe_reservoir_scouts": int(getattr(args, "de_coe_reservoir_scouts", 0) or 0),
            "stageb_refine_residual": bool(getattr(args, "stageb_refine_residual", False)),
            "stageb_epochs": int(getattr(args, "stageb_epochs", 0) or 0),
            "de_structural_priors": {
                "hard_tail_templates": bool(getattr(args, "de_hard_tail_templates", False)),
                "hard_tail_radial_templates": bool(getattr(args, "de_hard_tail_radial_templates", True)),
                "hard_tail_velocity_templates": bool(getattr(args, "de_hard_tail_velocity_templates", False)),
            },
            "ast_simplify": {
                "enabled": bool(getattr(args, "ast_simplify", False)),
                "level": str(getattr(args, "ast_simplify_level", "safe") or "safe"),
                "domain_policy": str(
                    getattr(args, "ast_simplify_domain_policy", "strict") or "strict"
                ),
                "max_passes": int(getattr(args, "ast_simplify_max_passes", 12) or 12),
                "validate": bool(getattr(args, "ast_simplify_validate", False)),
                "trace": bool(getattr(args, "ast_simplify_trace", False)),
            },
            "expr_ir": expr_ir_arg_items(args),
            "generalized_symmetries": {
                "enabled": bool(getattr(args, "gs_enable", False)),
                "mode": str(getattr(args, "gs_mode", "off") or "off"),
                "policy": str(getattr(args, "gs_policy", "augment") or "augment"),
                "known_generators": bool(getattr(args, "gs_known_generators", True)),
                "general_affine": bool(getattr(args, "gs_general_affine", False)),
                "de_templates": bool(getattr(args, "gs_de_templates", False)),
                "de_all_upgrades": bool(getattr(args, "gs_de_all_upgrades", False)),
                "de_determining_equations": bool(
                    getattr(args, "gs_de_determining_equations", False)
                ),
                "de_auto_nonlinear": bool(
                    getattr(args, "gs_de_auto_nonlinear", True)
                ),
                "de_auto_fss": bool(getattr(args, "gs_de_auto_fss", True)),
                "de_auto_fss_max_attempts": int(
                    getattr(args, "gs_de_auto_fss_max_attempts", 1)
                ),
                "de_auto_fss_budgets": {
                    "n_iter": int(getattr(args, "gs_de_auto_fss_n_iter", 1500)),
                    "n_fit": int(getattr(args, "gs_de_auto_fss_n_fit", 1024)),
                    "n_probe": int(getattr(args, "gs_de_auto_fss_n_probe", 1024)),
                    "max_depth": int(getattr(args, "gs_de_auto_fss_max_depth", 4)),
                    "return_topk": int(
                        getattr(args, "gs_de_auto_fss_return_topk", 8)
                    ),
                },
                "de_determining_max_degree": int(
                    getattr(args, "gs_de_determining_max_degree", 2)
                ),
                "de_determining_multiplier_degree": int(
                    getattr(args, "gs_de_determining_multiplier_degree", 2)
                ),
                "de_determining_bootstraps": int(
                    getattr(args, "gs_de_determining_bootstraps", 8)
                ),
                "de_determining_sparse_rotation": bool(
                    getattr(args, "gs_de_determining_sparse_rotation", True)
                ),
                "de_determining_bracket_certificate": bool(
                    getattr(args, "gs_de_determining_bracket_certificate", True)
                ),
                "de_nonlinear_invariants": bool(
                    getattr(args, "gs_de_nonlinear_invariants", False)
                ),
                "de_nonlinear_invariant_max_degree": int(
                    getattr(args, "gs_de_nonlinear_invariant_max_degree", 3)
                ),
                "de_lie_prolongation": bool(
                    getattr(args, "gs_de_lie_prolongation", False)
                ),
                "de_lie_use_for_selection": bool(
                    getattr(args, "gs_de_lie_use_for_selection", False)
                ),
                "de_lie_prolongation_min_coverage": float(
                    getattr(args, "gs_de_lie_prolongation_min_coverage", 0.90)
                ),
                "unit_torus": bool(getattr(args, "gs_unit_torus", False)),
                "pi_invariants": bool(getattr(args, "gs_pi_invariants", False)),
                "dim_policy": str(getattr(args, "gs_dim_policy", "audit") or "audit"),
                "env_activation_provenance": list(
                    getattr(args, "gs_env_activation_provenance", []) or []
                ),
                "activation_provenance": list(
                    getattr(args, "gs_activation_provenance", []) or []
                ),
                "legacy_alias_provenance": list(
                    getattr(args, "gs_legacy_alias_provenance", []) or []
                ),
            },
        },
    }

    if bool(getattr(args, "stat_selection", False)):
        plan = getattr(args, "_stat_de_audit_plan", None)
        if plan is None:
            raise RuntimeError("DE statistical selection requested without a sealed audit plan")
        from nestynet_sr.stat_selection.de_pipeline import run_de_statistical_selection
        report["statistical_selection"] = run_de_statistical_selection(
            report, plan,
            alpha=float(getattr(args, "stat_alpha", 0.05)),
            delta=float(getattr(args, "stat_delta", 0.01)),
            n_resamples=int(getattr(args, "stat_resamples", 4000)),
            seed=int(getattr(args, "stat_seed", 12345)),
            multiplier=str(getattr(args, "stat_multiplier", "normal")),
            failure_loss=float(getattr(args, "stat_failure_loss", 100.0)),
            max_candidates=int(getattr(args, "stat_max_candidates", 256)),
            rollout_window_fraction=float(getattr(args, "stat_rollout_window_fraction", 1.0)),
            rollout_max_span=getattr(args, "stat_rollout_max_span", None),
            traj_time_budget_s=float(getattr(args, "stat_traj_time_budget_s", 20.0)),
            coherent_loss_draws_path=getattr(args, "stat_coherent_loss_draws", None),
            rediscovery_report_paths=getattr(args, "stat_rediscovery_reports", None),
            calibration_repetitions=int(getattr(args, "stat_calibration_repetitions", 0)),
        )
        certificate_path = getattr(args, "stat_certificate_json", None)
        if certificate_path:
            pathlib.Path(certificate_path).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(certificate_path).write_text(json.dumps(report["statistical_selection"], indent=2) + "\n")

    try:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote JSON report to {report_path}")
    except Exception as e:
        print(f"\nWarning: failed to write JSON report: {e}")


def _run_de_affine_gs_probe(surrogates, train_dataloaders, *, cfg, device, dataset_ids=None) -> None:
    """Audit affine GS generators on fitted DE surrogates and record them."""
    if not bool(getattr(cfg, "gs_enable", False)):
        return
    mode = str(getattr(cfg, "gs_mode", "propose") or "propose").lower()
    if mode == "off":
        return
    if not (bool(getattr(cfg, "gs_known_generators", True)) or bool(getattr(cfg, "gs_general_affine", False))):
        return
    try:
        from nestynet_sr.sr_gs.config import GeneralizedSymmetryConfig
        from nestynet_sr.sr_gs.generators import discover_generator_specs, summarize_specs
        from nestynet_sr.sr_gs.reporting import record_policy_event, record_stagea_event
    except Exception as exc:
        print(f"[GS-DE] Affine audit unavailable: {type(exc).__name__}: {exc}")
        return

    gs_cfg = GeneralizedSymmetryConfig(
        enabled=True,
        mode=mode,
        policy=str(getattr(cfg, "gs_policy", "augment") or "augment"),
        known_generators=bool(getattr(cfg, "gs_known_generators", True)),
        known_lie=bool(getattr(cfg, "gs_known_generators", True)),
        general_affine=bool(getattr(cfg, "gs_general_affine", False)),
        translations=bool(getattr(cfg, "gs_translations", True)),
        diagonal_translations=bool(getattr(cfg, "gs_diagonal_translations", True)),
        scalings=bool(getattr(cfg, "gs_scalings", True)),
        rotations=bool(getattr(cfg, "gs_rotations", True)),
        lorentz_boosts=bool(getattr(cfg, "gs_lorentz_boosts", False)),
        output_equivariance=bool(getattr(cfg, "gs_output_equivariance", True)),
        residual_tol=float(getattr(cfg, "gs_residual_tol", 0.03)),
        audit_residual_tol=float(getattr(cfg, "gs_audit_residual_tol", 0.10)),
        min_confidence=float(getattr(cfg, "gs_min_confidence", 0.65)),
        affine_max_nonzero=int(getattr(cfg, "gs_affine_max_terms", 4)),
        affine_max_generators=int(getattr(cfg, "gs_affine_num_candidates", 4)),
        max_pair_generators=max(1, int(getattr(cfg, "gs_affine_num_candidates", 4))),
        report_rejected=True,
        report_top_k_rejected=40,
        ast_simplify=bool(getattr(cfg, "ast_simplify", False)),
        ast_simplify_level=str(getattr(cfg, "ast_simplify_level", "safe") or "safe"),
        ast_simplify_domain_policy=str(getattr(cfg, "ast_simplify_domain_policy", "strict") or "strict"),
        ast_simplify_max_passes=int(getattr(cfg, "ast_simplify_max_passes", 12)),
        ast_simplify_trace=bool(getattr(cfg, "ast_simplify_trace", False)),
    )

    dataset_ids = list(dataset_ids or [])
    for idx, (surrogate, dl) in enumerate(zip(list(surrogates or []), list(train_dataloaders or []))):
        try:
            X = de_search_mod._gather_x(dl, max_batches=2, max_points=2048, device=device)
            cache = de_search_mod.UFeatureCache(surrogate)
            cache.ensure(X, need_grad=True, need_hess=False)
            y = cache.u[:, 0].detach().cpu().numpy()
            G = cache.g[:, 0, :].detach().cpu().numpy()
            X_np = X.detach().cpu().numpy()
            cols = tuple(range(int(X_np.shape[1])))
            if bool(getattr(cfg, "gs_general_affine", False)) and int(X_np.shape[1]) < 2:
                record_policy_event(
                    policy=str(getattr(cfg, "gs_policy", "augment")),
                    action="de_general_affine_skipped",
                    details={
                        "dataset_index": int(idx),
                        "dataset_id": str(dataset_ids[idx]) if idx < len(dataset_ids) else str(idx),
                        "reason": "general_affine_pair_probe_requires_at_least_two_independent_variables",
                        "n_input_variables": int(X_np.shape[1]),
                    },
                )
            specs = discover_generator_specs(
                X_np,
                y,
                G,
                cols=cols,
                cfg=gs_cfg,
                include_rejected=True,
            )
            diagnostics = summarize_specs(specs)
            if diagnostics:
                record_stagea_event(
                    cols=cols,
                    diagnostics=diagnostics,
                    proposals=[],
                    context={
                        "entrypoint": "run_de",
                        "probe": "surrogate_field_affine_audit",
                        "dataset_index": int(idx),
                        "dataset_id": str(dataset_ids[idx]) if idx < len(dataset_ids) else str(idx),
                        "policy": str(getattr(cfg, "gs_policy", "augment")),
                        "mode": mode,
                        "known_generators": bool(getattr(cfg, "gs_known_generators", True)),
                        "general_affine": bool(getattr(cfg, "gs_general_affine", False)),
                        "note": "Audits affine witnesses on u(x); DE selection still uses DE template/library machinery.",
                    },
                )
            print(
                f"[GS-DE] Affine audit dataset {idx}: tested {len(diagnostics)} generator diagnostic(s) "
                f"on {int(X_np.shape[1])} input variable(s)."
            )
        except Exception as exc:
            try:
                record_policy_event(
                    policy=str(getattr(cfg, "gs_policy", "augment")),
                    action="de_affine_audit_failed",
                    details={"dataset_index": int(idx), "error": f"{type(exc).__name__}: {exc}"},
                )
            except Exception:
                pass
            print(f"[GS-DE] Affine audit failed for dataset {idx}: {type(exc).__name__}: {exc}")


def _attach_gs_reduction_rows(cfg, filepaths, *, x_axis: int) -> None:
    """Pre-search symmetry reduction: ensemble generators -> chart -> pullback rows.

    Loads the raw per-file trajectories (each CSV is one trajectory), discovers
    data-supported generators with the exp(eps*V) flow test, rectifies them,
    fits the reduced univariate law, and stashes the pulled-back library rows
    on the config for the GS DE-library bridge.  Best-effort: failures leave
    the search unchanged.
    """

    try:
        import numpy as _np

        from nestynet_sr.sr_gs.de_reduction import symmetry_reduction_proposals

        if not filepaths or len(filepaths) < 2:
            print("[gs-de-reduction] skipped: needs >=2 trajectory files")
            return
        trajectories = []
        for path in filepaths:
            arr = _np.loadtxt(str(path), delimiter=",", skiprows=1)
            if arr.ndim != 2 or arr.shape[1] < 2:
                print(f"[gs-de-reduction] skipped: {path} is not a 2-column trajectory CSV")
                return
            # convention of the DE CSVs: inputs first, target last
            trajectories.append((arr[:, int(x_axis)], arr[:, -1]))
        orders = [
            int(o) for o in (getattr(cfg, "order_candidates", None) or (1,)) if int(o) in (1, 2)
        ] or [1]
        rows: list = []
        seed_asts: list = []
        report_all: list = []
        n_generators = 0
        for order in sorted(set(orders)):
            result = symmetry_reduction_proposals(
                trajectories, x_axis=int(x_axis), order=int(order)
            )
            n_generators = max(n_generators, int(result.get("n_generators", 0)))
            report_all.extend(result.get("reports") or [])
            for row in result.get("library_rows", []):
                rep = repr(row[0])
                if not any(repr(r0) == rep for r0, _s, _f in rows):
                    rows.append(row)
                    seed_asts.append({"node": row[0], "label": f"gs_row:{row[2]}"})
            for prop in result.get("proposals", []):
                rhs = prop.get("rhs_ast")
                if rhs is not None and not any(
                    repr(rhs) == repr(entry["node"]) for entry in seed_asts
                ):
                    seed_asts.append(
                        {"node": rhs, "label": f"gs_law:order{order}:{prop.get('fit_family', prop.get('chart_kind', ''))}"}
                    )
        if rows:
            setattr(cfg, "gs_de_reduction_rows", rows)
        if seed_asts:
            # whole-law proposals + pulled-back terms seed the factorized
            # search additive-combo pool (engine-level complement)
            setattr(cfg, "gs_de_reduction_seed_asts", seed_asts)
        print(
            f"[gs-de-reduction] generators tested: {n_generators}; orders {sorted(set(orders))}; "
            f"injected {len(rows)} pulled-back library rows"
        )
        for term, _source, family in rows:
            print(f"[gs-de-reduction]   row [{family}]: {term!r}")
        setattr(cfg, "_gs_de_reduction_report", report_all)
    except Exception as exc:
        print(f"[gs-de-reduction] failed (search unchanged): {str(exc)[:200]}")


def main():
    """Main entry point for DE discovery."""
    start_time = timeit.default_timer()

    args = parse_args()

    filepaths = None
    if args.filepaths is not None and len(args.filepaths) > 0:
        filepaths = [str(p) for p in args.filepaths]
    elif args.filepath is not None:
        filepaths = [str(args.filepath)]
    if filepaths is None or len(filepaths) == 0:
        raise ValueError("Provide either --filepath <csv> or --filepaths <csv1> <csv2> ...")
    for fp in filepaths:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Data file not found: {fp}")

    stat_de_plan = None
    if bool(getattr(args, "stat_selection", False)):
        from nestynet_sr.stat_selection.de_pipeline import prepare_de_audit_plan
        stat_de_plan = prepare_de_audit_plan(
            filepaths,
            external_audit_paths=getattr(args, "stat_audit_filepaths", None),
            reserve_trajectories=int(getattr(args, "stat_audit_trajectories", 2)),
        )
        filepaths = list(stat_de_plan.search_paths)
        args._stat_de_audit_plan = stat_de_plan
        print(f"[stat-selection:DE] sealed {len(stat_de_plan.audit_paths)} whole audit trajectories; search sees {len(filepaths)}")

    # Extract base filename for output paths
    def _derive_base_filename(paths):
        if len(paths) == 1:
            return pathlib.Path(paths[0]).stem
        stems = [pathlib.Path(p).stem for p in paths]
        common = os.path.commonprefix(stems).rstrip("_-.")
        if common and len(common) >= 3:
            return f"{common}_multi{len(paths)}"
        return f"multi{len(paths)}_{stems[0]}"

    base_filename = _derive_base_filename(filepaths)
    filepath0 = filepaths[0]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    if args.device is not None:
        dev = torch.device(args.device)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.float64
    np_dtype = np.float64

    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Auto-detect number of x variables from data (and ensure consistency)
    import nestynet

    _, y_data, Nxvars = nestynet.dataloader.get_csv_data_as_pandas(filepath0)
    if len(filepaths) > 1:
        for fp in filepaths[1:]:
            _, _, Nx2 = nestynet.dataloader.get_csv_data_as_pandas(fp)
            if int(Nx2) != int(Nxvars):
                raise ValueError(
                    f"Nxvars mismatch: {pathlib.Path(filepath0).name} has Nxvars={Nxvars} but {pathlib.Path(fp).name} has Nxvars={Nx2}"
                )

    # Optional units context (for dimensional filtering of DE terms/templates)
    units_payload = None
    if (
        args.units
        or args.y_units
        or args.x_units
        or args.equations_txt
        or args.free_consts
        or args.local_consts
        or args.global_consts
        or args.fixed_consts
    ):
        y_vec = None
        x_mat = None
        basis_hint = None

        if args.units:
            y_vec, x_mat, basis_hint = _parse_units_arg(args.units)
        if args.y_units:
            y_vec = _parse_py_or_json_literal(args.y_units)
        if args.x_units:
            x_mat = _parse_py_or_json_literal(args.x_units)
        if (y_vec is None or x_mat is None) and args.equations_txt:
            y2, x2 = _load_units_from_equations(args.equations_txt, base_filename)
            if y_vec is None:
                y_vec = y2
            if x_mat is None:
                x_mat = x2

        if (y_vec is not None) and (x_mat is not None):
            try:
                y_vec = list(y_vec)
            except Exception:
                raise ValueError(f"y_units must be a sequence, got: {type(y_vec)}")
            try:
                x_mat = [list(u) for u in x_mat]
            except Exception:
                raise ValueError(f"x_units must be a sequence of sequences, got: {type(x_mat)}")
            if len(x_mat) != int(Nxvars):
                raise ValueError(f"x_units has {len(x_mat)} entries but Nxvars={Nxvars}")
            n_basis = len(y_vec)
            if any(len(u) != n_basis for u in x_mat):
                raise ValueError(f"All x_units vectors must have length {n_basis}")

            from nestynet_sr.sr_core.problem_dims import units_spec_from_dim_vectors
            from nestynet_sr.sr_core.units import UnitSystem

            basis = _infer_units_basis(
                n_basis, args.units_basis if basis_hint is None else basis_hint
            )
            us = UnitSystem(base=basis)
            y_dim = us.dim(y_vec)
            x_dims = tuple(us.dim(u) for u in x_mat)

            free_const_dims = {}
            free_const_scope = {}
            if args.free_consts:
                fc_map = _parse_py_or_json_literal(args.free_consts)
                if not isinstance(fc_map, dict):
                    raise ValueError("--free_consts must parse to a dict {name: unit_vec}")
                for name, vec in fc_map.items():
                    free_const_dims[str(name)] = us.dim(vec)
            if args.local_consts:
                lc_map = _parse_py_or_json_literal(args.local_consts)
                if not isinstance(lc_map, dict):
                    raise ValueError("--local_consts must parse to a dict {name: unit_vec}")
                for name, vec in lc_map.items():
                    nm = str(name)
                    free_const_dims[nm] = us.dim(vec)
                    free_const_scope[nm] = "experiment"
            if args.global_consts:
                gc_map = _parse_py_or_json_literal(args.global_consts)
                if not isinstance(gc_map, dict):
                    raise ValueError("--global_consts must parse to a dict {name: unit_vec}")
                for name, vec in gc_map.items():
                    nm = str(name)
                    free_const_dims[nm] = us.dim(vec)
                    free_const_scope[nm] = "class"

            fixed_const_dims = {}
            fixed_const_values = {}
            if args.fixed_consts:
                fx_map = _parse_py_or_json_literal(args.fixed_consts)
                if not isinstance(fx_map, dict):
                    raise ValueError("--fixed_consts must parse to a dict {name: (value, unit_vec)}")
                for name, payload in fx_map.items():
                    nm = str(name)
                    val = None
                    vec = None
                    if isinstance(payload, dict):
                        val = payload.get("value", payload.get("val", None))
                        vec = payload.get("units", payload.get("unit_vec", payload.get("unit", None)))
                    elif isinstance(payload, (list, tuple)) and len(payload) == 2:
                        val, vec = payload[0], payload[1]
                    else:
                        raise ValueError(
                            "--fixed_consts entries must be either [value, unit_vec] or {value:..., units:[...]}"
                        )
                    if val is None or vec is None:
                        raise ValueError(
                            f"--fixed_consts missing value/units for {nm!r}: got {payload}"
                        )
                    fixed_const_values[nm] = float(val)
                    fixed_const_dims[nm] = us.dim(vec)

            units_payload = dict(
                unit_system=us,
                y_dim=y_dim,
                x_dims=x_dims,
                free_const_dims=free_const_dims,
                free_const_scope=free_const_scope,
                fixed_const_dims=fixed_const_dims,
                fixed_const_values=fixed_const_values,
                fixed_const_mode=str(args.fixed_consts_mode),
                basis=basis,
                raw_y_units=y_vec,
                raw_x_units=x_mat,
            )
            print(f"[Units] Loaded basis={basis}; y_units={y_vec}; x_units={x_mat}")
        elif args.enforce_units:
            raise ValueError(
                "Units are enforced by default but y_units/x_units could not be loaded. "
                "Provide --units or --y_units/--x_units, or --equations_txt, "
                "or pass --ignore_units to skip dimensional analysis."
            )

    if args.enforce_units and units_payload is None:
        raise ValueError(
            "Units are enforced by default but no units spec was found. "
            "Provide --units or --y_units/--x_units, or --equations_txt, "
            "or pass --ignore_units to skip dimensional analysis."
        )

    units_spec = None
    if units_payload is not None:
        from nestynet_sr.sr_core.problem_dims import units_spec_from_dim_vectors

        units_spec = units_spec_from_dim_vectors(
            basis=units_payload["basis"],
            x_dims=units_payload["x_dims"],
            y_dim=units_payload["y_dim"],
            y_transform_name="identity",
            free_const_dims=units_payload.get("free_const_dims", {}),
            free_const_scope=units_payload.get("free_const_scope", {}),
            fixed_const_dims=units_payload.get("fixed_const_dims", {}),
            fixed_const_values=units_payload.get("fixed_const_values", {}),
            fixed_const_mode=units_payload.get("fixed_const_mode", "strict"),
            policy=args.units_policy,
            nn_semantics=args.nn_units_semantics,
        )
        if args.enforce_units:
            print(
                f"[Units] Dimensional filtering ENABLED (policy={args.units_policy}, nn_semantics={args.nn_units_semantics})."
            )

    y_op = None
    dual_layer = not args.single_layer

    if len(filepaths) == 1:
        print(f"Dataset: {filepath0}")
    else:
        print(f"Datasets: {filepaths}")
    print(f"Base name: {base_filename}")
    print(f"Device: {dev}")
    print(f"Number of x variables: {Nxvars}")
    print(f"Output directory: {args.output_dir}")

    # Configure hyperparameters
    data_hp = DataHyperparams(
        batch_size=args.batch_size, ndata_select=args.ndata_train, ndata_select_val=args.ndata_val
    )
    data_hp.data_split_strategy = str(args.data_split)

    model_hp = ModelHyperparams(
        double_precision=True,
        repeatable_runs=True,
        model_base_name="G_Model",
        num_segments_min=args.num_segments,
        num_segments_max=args.num_segments,
    )

    lm_hp = LMHyperparams(
        epochs=args.epochs,
        epochs_min=args.epochs_min,
        nval_patience=args.nval_patience,
        loss_target=args.loss_target,
        chisq_tol=1e-10,
        strategy=args.strategy,
    )
    lm_hp.evidence_enable = bool(getattr(args, "evidence", False))
    lm_hp.evidence_disable_residual_whitening = bool(
        getattr(args, "evidence_disable_residual_whitening", False)
    )
    lm_hp.evidence_disable_segment_priors = bool(
        getattr(args, "evidence_disable_segment_priors", False)
    )
    if getattr(args, "evidence_lambda_patch", None) is not None:
        lm_hp.evidence_lambda_patch = float(args.evidence_lambda_patch)
    lm_hp.evidence_prior_decay_auto = bool(getattr(args, "evidence_prior_decay_auto", True))
    if getattr(args, "evidence_prior_decay_start", None) is not None:
        lm_hp.evidence_prior_decay_start = int(args.evidence_prior_decay_start)
    if getattr(args, "evidence_prior_decay_interval", None) is not None:
        lm_hp.evidence_prior_decay_interval = int(args.evidence_prior_decay_interval)
    if getattr(args, "evidence_prior_decay_shape", None) is not None:
        lm_hp.evidence_prior_decay_shape = str(args.evidence_prior_decay_shape)
    if getattr(args, "evidence_prior_decay_final_scale", None) is not None:
        lm_hp.evidence_prior_decay_final_scale = float(args.evidence_prior_decay_final_scale)
    if getattr(args, "evidence_prior_cutoff_tol", None) is not None:
        lm_hp.evidence_prior_cutoff_tol = float(args.evidence_prior_cutoff_tol)
    lm_hp.evidence_gate_metrics_until_prior_decay = bool(getattr(args, "evidence_metric_gate", True))

    if lm_hp.evidence_enable:
        from nestynet_sr.sr_search.training import build_sr_evidence_config

        cfg_preview = build_sr_evidence_config(lm_hp, epochs=lm_hp.epochs)
        if cfg_preview is None:
            print("[Evidence] Requested, but auxiliary evidence terms are fully disabled; using legacy LM path.")
        else:
            decay_bits = []
            if getattr(cfg_preview, "prior_decay_start_iter", None) is not None:
                decay_interval = max(
                    0,
                    int(cfg_preview.prior_decay_end_iter - cfg_preview.prior_decay_start_iter),
                )
                decay_bits.append(
                    f"prior decay start={cfg_preview.prior_decay_start_iter}, interval={decay_interval} "
                    f"(end={cfg_preview.prior_decay_end_iter}) "
                    f"({cfg_preview.prior_decay_shape}, final={cfg_preview.prior_decay_final_scale:g})"
                )
            else:
                decay_bits.append("no prior decay schedule")
            print("[Evidence] Segment-prior guidance active; " + ", ".join(decay_bits) + ".")

    # Build datasets and train surrogate(s)
    print("\nLoading data and building datasets...")
    leaf_builder = LeafBuilder(model_hp, dev, dtype)

    surrogates = []
    ds_tr_list = []
    ds_va_list = []
    dl_tr_list = []
    dl_va_list = []
    val_losses = []
    surrogate_dataset_ids = []

    # Oscillation-dense 1-D datasets are split into few-period windows, with
    # one surrogate (and one downstream DE feature group) per window.
    max_periods_per_window = float(getattr(args, "surrogate_max_periods_per_window", 0.0) or 0.0)
    surrogate_inputs: list[tuple[str, str]] = []
    for fp in filepaths:
        stem = pathlib.Path(fp).stem
        window_fps = [fp]
        if max_periods_per_window > 0.0 and Nxvars == 1:
            window_fps = _plan_surrogate_windows(
                fp,
                x_axis=int(args.x_axis if args.x_axis is not None else 0),
                max_periods=max_periods_per_window,
                output_dir=args.output_dir,
            )
        if len(window_fps) == 1:
            surrogate_inputs.append((fp, stem))
        else:
            surrogate_inputs.extend(
                (wfp, f"{stem}#w{k}") for k, wfp in enumerate(window_fps)
            )

    for i, (fp, dataset_id) in enumerate(surrogate_inputs):
        print(f"\n[{i + 1}/{len(surrogate_inputs)}] Dataset: {fp}")
        n_rows = max(0, sum(1 for _ in open(fp, encoding="utf-8")) - 1)
        fp_data_hp = _window_data_hp(data_hp, n_rows) if "#w" in dataset_id else data_hp
        ds_tr, ds_va, dl_tr, dl_va = build_datasets(fp, Nxvars, np_dtype, fp_data_hp, y_op)
        ds_tr_list.append(ds_tr)
        ds_va_list.append(ds_va)
        dl_tr_list.append(dl_tr)
        dl_va_list.append(dl_va)
        surrogate_dataset_ids.append(dataset_id)

        # LM occasionally stalls in a poor basin on oscillatory data; a fresh
        # init usually escapes, so retry when far above the loss target.
        retrain_threshold = max(100.0 * float(lm_hp.loss_target), 1.0e-7)
        max_attempts = 1 + max(0, int(getattr(args, "surrogate_retrain_attempts", 1)))
        best_run = None
        for attempt in range(max_attempts):
            print(f"Building surrogate model with {args.num_segments} segments...")
            ast0 = build_initial_ast(
                Nxvars, num_segments=args.num_segments, dual_layer=dual_layer, tag=f"A{i}"
            )
            surrogate, nparam, _ = build_composite_ast(
                ast0,
                args.num_segments,
                dual_layer=dual_layer,
                leaf_builder=leaf_builder,
                device=dev,
                dtype=dtype,
            )
            print(f"Model has {nparam} parameters")

            print(f"Training surrogate (max epochs: {args.epochs})...")
            best_val_loss, _, best_p, lm_opt = train_initial_model(
                surrogate,
                dl_tr,
                dl_va,
                epochs=lm_hp.epochs,
                LM_strategy=lm_hp.strategy,
                nval_patience=lm_hp.nval_patience,
                loss_target=lm_hp.loss_target,
                epochs_min=lm_hp.epochs_min,
                chisq_tol=lm_hp.chisq_tol,
                device=dev,
                epochs_awful_check=lm_hp.epochs_awful_check,
                awful_threshold=lm_hp.awful_threshold,
                lm_hp=lm_hp,
            )
            if best_run is None or float(best_val_loss) < float(best_run[0]):
                best_run = (float(best_val_loss), surrogate, best_p, lm_opt)
            if float(best_run[0]) <= retrain_threshold:
                break
            if attempt + 1 < max_attempts:
                print(
                    f"[Surrogate] val_loss {best_val_loss:.3e} above retrain threshold "
                    f"{retrain_threshold:.1e}; retrying with a fresh initialization..."
                )
        best_val_loss, surrogate, best_p, lm_opt = best_run
        lm_opt._update_param_groups(best_p)
        surrogate.eval()
        surrogates.append(surrogate)
        val_losses.append(float(best_val_loss))
        print(f"Surrogate complete! val_loss={best_val_loss:.6e}")

    # Configure DE discovery
    order_candidates = tuple(int(s) for s in args.order_candidates.split(",") if s.strip() != "")

    # Use explicit x_axis if provided, otherwise will auto-detect from dataset metadata
    x_axis_value = args.x_axis if args.x_axis is not None else 0

    if bool(getattr(args, "gs_de_reduction", False)) and not bool(getattr(args, "gs_enable", False)):
        print("[gs-de-reduction] enabling --gs-enable (required for GS library rows)")
        args.gs_enable = True
    cfg = DESearchConfig(
        x_axis=x_axis_value,
        order_candidates=order_candidates,
        max_x_power=args.max_x_power,
        max_u_power=args.max_u_power,
        max_xu_total_degree=args.max_xu_total_degree,
        include_const=args.include_const,
        include_x=args.include_x_flag,
        include_u=args.include_u_flag,
        include_xu=args.include_xu_flag,
        include_xdu=args.include_xdu,
        include_inv_xdu=args.include_inv_xdu,
        include_inv_xu=args.include_inv_xu,
        include_inv_x2u=args.include_inv_x2u,
        include_du=args.include_du,
        include_d2u=args.include_d2u,
        include_udu=args.include_udu,
        de_hard_tail_templates=bool(getattr(args, "de_hard_tail_templates", False)),
        de_hard_tail_radial_templates=bool(getattr(args, "de_hard_tail_radial_templates", True)),
        de_hard_tail_velocity_templates=bool(getattr(args, "de_hard_tail_velocity_templates", False)),
        stlsq_lambda=args.stlsq_lambda,
        stlsq_max_iter=args.stlsq_max_iter,
        ridge=args.ridge,
        sparsity_penalty=args.sparsity_penalty,
        units_spec=units_spec,
        enforce_units=bool(args.enforce_units),
        ast_simplify=bool(getattr(args, "ast_simplify", False)),
        ast_simplify_level=str(getattr(args, "ast_simplify_level", "safe") or "safe"),
        ast_simplify_domain_policy=str(getattr(args, "ast_simplify_domain_policy", "strict") or "strict"),
        ast_simplify_max_passes=int(getattr(args, "ast_simplify_max_passes", 12)),
        ast_simplify_validate=bool(getattr(args, "ast_simplify_validate", False)),
        ast_simplify_trace=bool(getattr(args, "ast_simplify_trace", False)),
        gs_enable=bool(getattr(args, "gs_enable", False)),
        gs_mode=str(getattr(args, "gs_mode", "propose") or "propose"),
        gs_policy=str(getattr(args, "gs_policy", "augment") or "augment"),
        gs_known_generators=bool(getattr(args, "gs_known_generators", True)),
        gs_general_affine=bool(getattr(args, "gs_general_affine", False)),
        gs_jet_enable=bool(getattr(args, "gs_jet_enable", True)),
        gs_jet_separability=bool(getattr(args, "gs_jet_separability", True)),
        gs_jet_multiplicative=bool(getattr(args, "gs_jet_multiplicative", True)),
        gs_translations=bool(getattr(args, "gs_translations", True)),
        gs_diagonal_translations=bool(getattr(args, "gs_diagonal_translations", True)),
        gs_scalings=bool(getattr(args, "gs_scalings", True)),
        gs_rotations=bool(getattr(args, "gs_rotations", True)),
        gs_lorentz_boosts=bool(getattr(args, "gs_lorentz_boosts", False)),
        gs_output_equivariance=bool(getattr(args, "gs_output_equivariance", True)),
        gs_residual_tol=float(getattr(args, "gs_residual_tol", 0.03)),
        gs_audit_residual_tol=float(getattr(args, "gs_audit_residual_tol", 0.10)),
        gs_min_confidence=float(getattr(args, "gs_min_confidence", 0.65)),
        gs_affine_max_terms=int(getattr(args, "gs_affine_max_terms", 4)),
        gs_affine_num_candidates=int(getattr(args, "gs_affine_num_candidates", 4)),
        gs_de_templates=bool(getattr(args, "gs_de_templates", False)),
        gs_de_radial_templates=bool(getattr(args, "gs_de_radial_templates", True)),
        gs_de_velocity_templates=bool(getattr(args, "gs_de_velocity_templates", False)),
        gs_de_all_upgrades=bool(getattr(args, "gs_de_all_upgrades", False)),
        gs_de_determining_equations=bool(getattr(args, "gs_de_determining_equations", False)),
        gs_de_auto_nonlinear=bool(getattr(args, "gs_de_auto_nonlinear", True)),
        gs_de_auto_fss=bool(getattr(args, "gs_de_auto_fss", True)),
        gs_de_auto_fss_max_attempts=max(
            0, int(getattr(args, "gs_de_auto_fss_max_attempts", 1))
        ),
        gs_de_auto_fss_n_iter=max(1, int(getattr(args, "gs_de_auto_fss_n_iter", 1500))),
        gs_de_auto_fss_n_fit=max(16, int(getattr(args, "gs_de_auto_fss_n_fit", 1024))),
        gs_de_auto_fss_n_probe=max(16, int(getattr(args, "gs_de_auto_fss_n_probe", 1024))),
        gs_de_auto_fss_max_depth=max(1, int(getattr(args, "gs_de_auto_fss_max_depth", 4))),
        gs_de_auto_fss_return_topk=max(1, int(getattr(args, "gs_de_auto_fss_return_topk", 8))),
        gs_de_contact_templates=bool(getattr(args, "gs_de_contact_templates", False)),
        gs_de_noether_templates=bool(getattr(args, "gs_de_noether_templates", False)),
        gs_de_discrete_symmetry_templates=bool(getattr(args, "gs_de_discrete_symmetry_templates", False)),
        gs_de_weighted_scaling_templates=bool(getattr(args, "gs_de_weighted_scaling_templates", False)),
        gs_de_radial_reduction_templates=bool(getattr(args, "gs_de_radial_reduction_templates", False)),
        gs_de_invariant_library=bool(getattr(args, "gs_de_invariant_library", False)),
        gs_de_upgrade_max_terms=int(getattr(args, "gs_de_upgrade_max_terms", 64)),
        gs_de_determining_max_degree=int(getattr(args, "gs_de_determining_max_degree", 2)),
        gs_de_determining_max_generators=int(getattr(args, "gs_de_determining_max_generators", 4)),
        gs_de_determining_multiplier_degree=int(getattr(args, "gs_de_determining_multiplier_degree", 2)),
        gs_de_determining_bootstraps=int(getattr(args, "gs_de_determining_bootstraps", 8)),
        gs_de_determining_sparse_rotation=bool(getattr(args, "gs_de_determining_sparse_rotation", True)),
        gs_de_determining_bracket_certificate=bool(getattr(args, "gs_de_determining_bracket_certificate", True)),
        gs_de_nonlinear_invariants=bool(getattr(args, "gs_de_nonlinear_invariants", False)),
        gs_de_nonlinear_invariant_max_degree=int(getattr(args, "gs_de_nonlinear_invariant_max_degree", 3)),
        gs_de_nonlinear_invariant_max_candidates=int(getattr(args, "gs_de_nonlinear_invariant_max_candidates", 8)),
        gs_de_nonlinear_invariant_tol=float(getattr(args, "gs_de_nonlinear_invariant_tol", 0.03)),
        gs_de_nonlinear_orbit_coordinate=bool(getattr(args, "gs_de_nonlinear_orbit_coordinate", True)),
        gs_de_weighted_max_abs_x_power=int(getattr(args, "gs_de_weighted_max_abs_x_power", 2)),
        gs_de_weighted_max_u_power=int(getattr(args, "gs_de_weighted_max_u_power", 5)),
        gs_de_weighted_max_du_power=int(getattr(args, "gs_de_weighted_max_du_power", 4)),
        gs_de_lie_prolongation=bool(getattr(args, "gs_de_lie_prolongation", False)),
        gs_de_lie_use_for_selection=bool(getattr(args, "gs_de_lie_use_for_selection", False)),
        gs_de_determining_certificate=bool(getattr(args, "gs_de_determining_certificate", False)),
        gs_de_certificate_tol=float(getattr(args, "gs_de_certificate_tol", 1.0e-6)),
        gs_de_certificate_coeff_prune_tol=float(getattr(args, "gs_de_certificate_coeff_prune_tol", 0.0)),
        gs_de_lie_prolongation_weight=float(getattr(args, "gs_de_lie_prolongation_weight", 0.05)),
        gs_de_lie_prolongation_tol=float(getattr(args, "gs_de_lie_prolongation_tol", 0.05)),
        gs_de_lie_prolongation_max_samples=int(getattr(args, "gs_de_lie_prolongation_max_samples", 2048)),
        gs_de_lie_prolongation_min_coverage=float(getattr(args, "gs_de_lie_prolongation_min_coverage", 0.90)),
        gs_unit_torus=bool(getattr(args, "gs_unit_torus", False)),
        gs_pi_invariants=bool(getattr(args, "gs_pi_invariants", False)),
        gs_dim_policy=str(getattr(args, "gs_dim_policy", "audit") or "audit"),
        gs_dim_both_rule=str(getattr(args, "gs_dim_both_rule", "rref-dominates") or "rref-dominates"),
        gs_dim_validator=str(getattr(args, "gs_dim_validator", "nullspace") or "nullspace"),
        gs_dim_keep_local_gates=bool(getattr(args, "gs_dim_keep_local_gates", True)),
        gs_pi_max_exponent=int(getattr(args, "gs_pi_max_exponent", 3)),
        gs_pi_max_l1=int(getattr(args, "gs_pi_max_l1", 6)),
        gs_pi_max_proposals=int(getattr(args, "gs_pi_max_proposals", 24)),
        gs_pi_max_basis=int(getattr(args, "gs_pi_max_basis", 8)),
        gs_pi_rational_denom=int(getattr(args, "gs_pi_rational_denom", 1)),
        gs_pi_include_free_consts=bool(getattr(args, "gs_pi_include_free_consts", True)),
        gs_report_dim_disagreements=bool(getattr(args, "gs_report_dim_disagreements", True)),
    )
    apply_expr_ir_args_to_obj(args, cfg)
    if bool(getattr(args, "gs_de_reduction", False)):
        _attach_gs_reduction_rows(cfg, filepaths, x_axis=int(cfg.x_axis))
    # One id per trained surrogate (windowed datasets contribute several).
    dataset_ids = list(surrogate_dataset_ids)
    if bool(getattr(args, "gs_enable", False)):
        try:
            from nestynet_sr.sr_gs.reporting import configure_gs_reporter, reset_gs_reporter

            reset_gs_reporter(
                {
                    "entrypoint": "run_de",
                    "base_filename": str(base_filename),
                    "mode": str(getattr(args, "gs_mode", "propose")),
                    "policy": str(getattr(args, "gs_policy", "augment")),
                    "unit_torus": bool(getattr(args, "gs_unit_torus", False)),
                    "pi_invariants": bool(getattr(args, "gs_pi_invariants", False)),
                    "dim_policy": str(getattr(args, "gs_dim_policy", "audit")),
                    "de_templates": bool(getattr(args, "gs_de_templates", False)),
                    "de_lie_prolongation": bool(getattr(args, "gs_de_lie_prolongation", False)),
                    "de_lie_use_for_selection": bool(getattr(args, "gs_de_lie_use_for_selection", False)),
                    "de_lie_prolongation_weight": float(getattr(args, "gs_de_lie_prolongation_weight", 0.05)),
                    "de_lie_prolongation_tol": float(getattr(args, "gs_de_lie_prolongation_tol", 0.05)),
                    "de_lie_prolongation_max_samples": int(getattr(args, "gs_de_lie_prolongation_max_samples", 2048)),
                    "de_lie_prolongation_min_coverage": float(getattr(args, "gs_de_lie_prolongation_min_coverage", 0.90)),
                    "ast_simplify": bool(getattr(args, "ast_simplify", False)),
                    "env_activation_provenance": list(getattr(args, "gs_env_activation_provenance", []) or []),
                    "activation_provenance": list(getattr(args, "gs_activation_provenance", []) or []),
                    "legacy_alias_provenance": list(getattr(args, "gs_legacy_alias_provenance", []) or []),
                }
            )
            configure_gs_reporter(
                gs_enabled=True,
                mode=str(getattr(args, "gs_mode", "propose")),
                policy=str(getattr(args, "gs_policy", "augment")),
                report_rejected=bool(getattr(args, "gs_report_rejected", True)),
                top_k_rejected=int(getattr(args, "gs_report_top_k_rejected", 40)),
            )
        except Exception as exc:
            print(f"[GS-DE] Warning: failed to initialize GS reporter: {type(exc).__name__}: {exc}")
        print(
            "[GS-DE] Generalized-symmetry DE layer enabled "
            f"(mode={str(getattr(args, 'gs_mode', 'propose'))}, "
            f"policy={str(getattr(args, 'gs_policy', 'augment'))}, "
            f"known={bool(getattr(args, 'gs_known_generators', True))}, "
            f"general_affine={bool(getattr(args, 'gs_general_affine', False))}, "
            f"jet={bool(getattr(args, 'gs_jet_enable', True))}, "
            f"hard_tail_priors={bool(getattr(args, 'de_hard_tail_templates', False))}, "
            f"gs_upgrades={bool(getattr(args, 'gs_de_all_upgrades', False))}, "
            f"lie_prolongation={bool(getattr(args, 'gs_de_lie_prolongation', False))}, "
            f"lie_selection={bool(getattr(args, 'gs_de_lie_use_for_selection', False))}, "
            f"unit_torus={bool(getattr(args, 'gs_unit_torus', False))})"
        )
        legacy_prov = list(getattr(args, "gs_legacy_alias_provenance", []) or [])
        if legacy_prov:
            print("[GS-DE] Deprecated GS hard-tail aliases normalized to neutral priors:")
            for row in legacy_prov:
                print(f"  - {row.get('alias')} -> {row.get('effect')}")
        env_prov = list(getattr(args, "gs_env_activation_provenance", []) or [])
        if env_prov:
            print("[GS-DE] Environment GS activation/provenance:")
            for row in env_prov:
                print(f"  - {row.get('name')}={row.get('value')}")

    # Discover DE
    print("\n" + "=" * 70)
    print("DE DISCOVERY")
    print("=" * 70)

    factorized_cfg = build_factorized_rescue_config_from_args(args)
    rescue_cfg = build_factorized_search_rescue_config_from_args(args)
    integrate_topk_explicit = getattr(args, "factorized_search_integrate_topk", None) is not None
    max_attempts_explicit = getattr(args, "factorized_search_max_attempts", None) is not None
    de_coe_active = str(getattr(args, "de_coe_mode", "off") or "off").strip().lower() in {
        "audit",
        "adjudicate",
        "reservoir",
    } or int(getattr(args, "de_coe_reservoir_scouts", 0) or 0) > 0
    if de_coe_active and not integrate_topk_explicit:
        rescue_cfg.validate_integrate_topk = 0
    if bool(getattr(args, "factorized_search_only", False)) and bool(getattr(args, "factorized_de", False)):
        raise ValueError("--factorized-search-only and --factorized-de are mutually exclusive")
    _run_de_affine_gs_probe(
        surrogates,
        dl_tr_list,
        cfg=cfg,
        device=dev,
        dataset_ids=dataset_ids,
    )
    if bool(getattr(args, "factorized_de", False)):
        factorized_cfg = _as_factorized_de_factorized_config(factorized_cfg)
        rescue_cfg = _as_factorized_de_rescue_config(rescue_cfg)
        whole_rhs_policy = str(getattr(args, "factorized_de_whole_rhs", "auto") or "auto").strip().lower()
        if getattr(args, "factorized_search_budget_scope", None) is None:
            rescue_cfg.budget_scope = "global"
        if (not max_attempts_explicit) and whole_rhs_policy != "always":
            rescue_cfg.max_attempts = 1
        if (
            not integrate_topk_explicit
            and whole_rhs_policy != "always"
        ):
            rescue_cfg.validate_integrate_topk = max(4, int(getattr(rescue_cfg, "validate_integrate_topk", 0) or 0))
    primary_res = None
    factorized_res = None
    factorized_trigger_reason = None
    factorized_triggered = False
    rescue_res = None
    rescue_trigger_reason = None
    rescue_triggered = False
    whole_rhs_diag = None
    feature_groups_for_eval = None
    gs_auto_handoff = None

    if bool(getattr(args, "factorized_de", False)):
        print("\n" + "=" * 70)
        print("FACTORIZED DE FIRST-LINE SEARCH")
        print("=" * 70)
        if args.varpro or args.varpro_templates:
            print("[factorized DE] Warning: ignoring first-line VarPro/template options in --factorized-de mode.")
        factorized_triggered = True
        factorized_trigger_reason = "mode_factorized_de"
        whole_rhs_policy = str(getattr(args, "factorized_de_whole_rhs", "auto") or "auto")
        typed_lanes_policy = str(getattr(args, "factorized_de_typed_lanes", "never") or "never")
        feature_groups_for_eval = _prepare_factorized_search_feature_groups(
            cfg=cfg,
            rescue_cfg=_feature_group_rescue_cfg_from_factorized(factorized_cfg),
            surrogates=surrogates,
            dl_tr_list=dl_tr_list,
            dl_va_list=dl_va_list,
            dataset_ids=dataset_ids,
            surrogate_val_losses=val_losses,
            device=dev,
            dtype=dtype,
        )
        res, factorized_res, rescue_res, selected_engine, whole_rhs_diag = _run_factorized_de(
            cfg=cfg,
            factorized_cfg=factorized_cfg,
            rescue_cfg=rescue_cfg,
            filepaths=filepaths,
            surrogates=surrogates,
            dl_tr_list=dl_tr_list,
            dl_va_list=dl_va_list,
            dataset_ids=dataset_ids,
            surrogate_val_losses=val_losses,
            device=dev,
            dtype=dtype,
            verbose=True,
            whole_rhs_policy=whole_rhs_policy,
            typed_lanes_policy=typed_lanes_policy,
            feature_groups=feature_groups_for_eval,
        )
        rescue_triggered = bool(rescue_res is not None)
        if isinstance(whole_rhs_diag, dict):
            rescue_trigger_reason = str(whole_rhs_diag.get("reason", "mode_factorized_de"))
        else:
            rescue_trigger_reason = "mode_factorized_de" if rescue_triggered else "whole_rhs_not_attempted"
        if factorized_res is not None:
            print(f"[factorized DE] Typed-lane probe RMS: {candidate_probe_rms(factorized_res):.6e}")
        if rescue_res is not None:
            print(f"[factorized DE] Whole-RHS FSS probe RMS:   {candidate_probe_rms(rescue_res):.6e}")
    elif bool(getattr(args, "factorized_search_only", False)):
        print("\n" + "=" * 70)
        print("FACTORIZED SYMBOLIC SEARCH ONLY")
        print("=" * 70)
        if args.varpro or args.varpro_templates:
            print("[factorized symbolic search] Warning: ignoring first-line VarPro/template options in --factorized-search-only mode.")
        rescue_triggered = True
        rescue_trigger_reason = "mode_factorized_search_only"
        feature_groups_for_eval = _prepare_factorized_search_feature_groups(
            cfg=cfg,
            rescue_cfg=rescue_cfg,
            surrogates=surrogates,
            dl_tr_list=dl_tr_list,
            dl_va_list=dl_va_list,
            dataset_ids=dataset_ids,
            surrogate_val_losses=val_losses,
            device=dev,
            dtype=dtype,
        )
        rescue_res = _run_factorized_search_only_with_heuristics(
            cfg=cfg,
            rescue_cfg=rescue_cfg,
            filepaths=filepaths,
            surrogates=surrogates,
            dl_tr_list=dl_tr_list,
            dl_va_list=dl_va_list,
            dataset_ids=dataset_ids,
            device=dev,
            dtype=dtype,
            verbose=True,
            surrogate_val_losses=val_losses,
            feature_groups=feature_groups_for_eval,
        )
        print(
            "[factorized symbolic search] Probe RMS: {:.6e}".format(
                candidate_probe_rms(rescue_res),
            )
        )
        res = rescue_res
        selected_engine = "factorized_search"
    else:
        # Pass dataset for automatic x_axis detection if not explicitly set
        # (windowed datasets train several surrogates even for one filepath)
        if len(surrogates) == 1:
            surrogate0 = surrogates[0]
            dl_tr0, dl_va0 = dl_tr_list[0], dl_va_list[0]
            ds_tr0 = ds_tr_list[0]
            res = discover_de_from_surrogate(
                surrogate0,
                dl_tr0,
                dl_va0,
                cfg=cfg,
                device=dev,
                dataset=ds_tr0 if args.x_axis is None else None,
            )

            # VarPro refinement (Phase 1)
            if args.varpro:
                res = varpro_refine_linear(
                    res,
                    surrogate0,
                    dl_tr0,
                    dl_va0,
                    cfg=cfg,
                    device=dev,
                    epochs=args.varpro_epochs,
                    strategy=args.strategy,
                    verbose=True,
                )

            # VarPro template search (Phase 2)
            if args.varpro_templates:
                template_families = [s.strip() for s in args.varpro_templates.split(",") if s.strip()]
                res = varpro_template_search(
                    res,
                    surrogate0,
                    dl_tr0,
                    dl_va0,
                    template_families=template_families,
                    cfg=cfg,
                    device=dev,
                    max_templates=args.max_templates,
                    complexity_penalty=args.complexity_penalty,
                    prefer_autonomous=args.prefer_autonomous,
                    prefer_forced=args.prefer_forced,
                    enable_support_minimization=args.support_minimization,
                    rms_tol_factor=args.rms_tol_factor,
                    optimize_psi=args.template_lm,
                    psi_lm_epochs=args.template_lm_epochs,
                    psi_lm_epochs_min=args.template_lm_epochs_min,
                    psi_lm_nval_patience=args.template_lm_nval_patience,
                    psi_lm_loss_target=args.template_lm_loss_target,
                    psi_lm_strategy=args.strategy,
                    verbose=True,
                )
        else:
            dataset_ids = list(surrogate_dataset_ids)
            res = discover_de_from_surrogates(
                surrogates,
                dl_tr_list,
                dl_va_list,
                cfg=cfg,
                device=dev,
                datasets=ds_tr_list if args.x_axis is None else None,
                dataset_ids=dataset_ids,
            )

            # Multi-dataset VarPro refinement
            if args.varpro:
                print("\n[VarPro] Refining linear coefficients (multi-dataset)...")
                results_per_dataset = varpro_refine_linear_multi(
                    de_result=res,
                    surrogates=surrogates,
                    train_dataloaders=dl_tr_list,
                    val_dataloaders=dl_va_list,
                    cfg=cfg,
                    device=dev,
                    epochs=args.varpro_epochs,
                    strategy=args.strategy,
                    verbose=True,
                )
                # Merge back to DESearchResultMulti
                res = _merge_results_to_multi(results_per_dataset, dataset_ids)

            # Multi-dataset template search
            if args.varpro_templates:
                print("\n[VarPro] Searching templates (multi-dataset)...")
                template_families = [s.strip() for s in args.varpro_templates.split(",") if s.strip()]
                res = varpro_template_search_multi(
                    de_result=res,
                    surrogates=surrogates,
                    train_dataloaders=dl_tr_list,
                    val_dataloaders=dl_va_list,
                    template_families=template_families,
                    cfg=cfg,
                    device=dev,
                    max_templates=args.max_templates,
                    complexity_penalty=args.complexity_penalty,
                    prefer_autonomous=args.prefer_autonomous,
                    prefer_forced=args.prefer_forced,
                    enable_support_minimization=args.support_minimization,
                    rms_tol_factor=args.rms_tol_factor,
                    optimize_psi=args.template_lm,
                    psi_lm_epochs=args.template_lm_epochs,
                    psi_lm_epochs_min=args.template_lm_epochs_min,
                    psi_lm_nval_patience=args.template_lm_nval_patience,
                    psi_lm_loss_target=args.template_lm_loss_target,
                    psi_lm_strategy=args.strategy,
                    verbose=True,
                )

        primary_res = res
        gs_auto_handoff = _attach_automatic_gs_carriers(primary_res, cfg)
        best_so_far = primary_res
        selected_engine = "stlsq"
        feature_groups = None
        factorized_trigger_reason = _factorized_trigger_reason(primary_res, factorized_cfg)
        factorized_triggered = factorized_trigger_reason is not None

        if factorized_triggered:
            print("\n" + "=" * 70)
            print("FACTORIZED COEFF-ON-CARRIER RESCUE")
            print("=" * 70)
            print(
                "[Factorized] Triggered (mode={}, reason={})".format(
                    factorized_cfg.mode,
                    factorized_trigger_reason,
                )
            )
            feature_groups = _prepare_factorized_search_feature_groups(
                cfg=cfg,
                rescue_cfg=_feature_group_rescue_cfg_from_factorized(factorized_cfg),
                surrogates=surrogates,
                dl_tr_list=dl_tr_list,
                dl_va_list=dl_va_list,
                dataset_ids=dataset_ids,
                surrogate_val_losses=val_losses,
                device=dev,
                dtype=dtype,
            )
            feature_groups_for_eval = feature_groups
            factorized_res = run_factorized_coeff_rescue_from_feature_groups(
                feature_groups,
                cfg=cfg,
                rescue_cfg=factorized_cfg,
                primary=primary_res,
                dtype=dtype,
            )
            if factorized_res is not None:
                print(
                    "[Factorized] Probe RMS: {:.6e}".format(
                        candidate_probe_rms(factorized_res),
                    )
                )
            if _better_than(
                factorized_res,
                best_so_far,
                rel_factor=float(getattr(factorized_cfg, "replace_rel_factor", 0.98)),
            ):
                best_so_far = factorized_res
                selected_engine = "factorized"

        rescue_trigger_reason = _factorized_search_trigger_reason(best_so_far, rescue_cfg)
        if bool((gs_auto_handoff or {}).get("trigger_fss", False)):
            rescue_trigger_reason = "gs_certified_nonlinear_carriers"
            automatic_budgets = _bound_automatic_gs_fss(rescue_cfg, cfg)
            gs_auto_handoff["fss_budgets"] = automatic_budgets
            automatic_limit = max(
                0, int(getattr(cfg, "gs_de_auto_fss_max_attempts", 1))
            )
            if rescue_cfg.max_attempts is None:
                rescue_cfg.max_attempts = automatic_limit
            else:
                rescue_cfg.max_attempts = min(
                    int(rescue_cfg.max_attempts), automatic_limit
                )
        rescue_triggered = rescue_trigger_reason is not None

        if rescue_triggered:
            print("\n" + "=" * 70)
            print("FACTORIZED SYMBOLIC SEARCH RESCUE")
            print("=" * 70)
            print(
                "[factorized symbolic search] Triggered (mode={}, reason={})".format(
                    rescue_cfg.mode,
                    rescue_trigger_reason,
                )
            )
            if feature_groups is None:
                feature_groups = _prepare_factorized_search_feature_groups(
                    cfg=cfg,
                    rescue_cfg=rescue_cfg,
                    surrogates=surrogates,
                    dl_tr_list=dl_tr_list,
                    dl_va_list=dl_va_list,
                    dataset_ids=dataset_ids,
                    surrogate_val_losses=val_losses,
                    device=dev,
                    dtype=dtype,
                )
            feature_groups_for_eval = feature_groups
            rescue_res = _run_factorized_search_rescue_with_heuristics(
                primary_res=primary_res,
                cfg=cfg,
                rescue_cfg=rescue_cfg,
                filepaths=filepaths,
                surrogates=surrogates,
                dl_tr_list=dl_tr_list,
                dl_va_list=dl_va_list,
                dataset_ids=dataset_ids,
                device=dev,
                dtype=dtype,
                verbose=True,
                feature_groups=feature_groups,
                surrogate_val_losses=val_losses,
            )
            if isinstance(getattr(rescue_res, "diagnostics", None), dict):
                rescue_res.diagnostics["automatic_gs_handoff"] = dict(
                    gs_auto_handoff or {}
                )
            print(
                "[factorized symbolic search] Probe RMS: {:.6e}".format(
                    candidate_probe_rms(rescue_res),
                )
            )

        if _better_than(
            rescue_res,
            best_so_far,
            rel_factor=float(getattr(rescue_cfg, "replace_rel_factor", 0.98)),
        ):
            res = rescue_res
            selected_engine = "factorized_search"
        else:
            res = best_so_far

    if args.stageb_refine_residual:
        print("\n" + "=" * 70)
        print("STAGE-B RESIDUAL REFINEMENT")
        print("=" * 70)
        if isinstance(res, FactorizedSearchDEResult):
            print("[Stage B] Warning: residual refinement is not yet wired for factorized symbolic search-selected candidates; skipping.")
        elif isinstance(res, FactorizedDEResult):
            print("[Stage B] Warning: residual refinement is not yet wired for factorized-selected candidates; skipping.")
        else:
            try:
                if len(surrogates) == 1:
                    _st, meta = _run_stageb_residual_refine_single(
                        res=res,
                        surrogate=surrogates[0],
                        ds_tr=ds_tr_list[0],
                        ds_va=ds_va_list[0],
                        dl_tr=dl_tr_list[0],
                        dl_va=dl_va_list[0],
                        lm_hp=lm_hp,
                        cfg=cfg,
                        device=dev,
                        dtype=dtype,
                        epochs_stageB=int(args.stageb_epochs),
                    )
                    setattr(res, "stageb_residual_metadata", meta)
                    print(
                        "[Stage B] Residual fit complete: val_mse={:.6e}, val_rms={:.6e}".format(
                            float(meta["val_mse"]),
                            float(meta["val_rms"]),
                        )
                    )
                else:
                    _st, meta = _run_stageb_residual_refine_multi(
                        res=res,
                        surrogates=surrogates,
                        ds_tr_list=ds_tr_list,
                        ds_va_list=ds_va_list,
                        dl_tr_list=dl_tr_list,
                        dl_va_list=dl_va_list,
                        lm_hp=lm_hp,
                        cfg=cfg,
                        device=dev,
                        dtype=dtype,
                        epochs_stageB=int(args.stageb_epochs),
                    )
                    setattr(res, "stageb_residual_metadata", meta)
                    print(
                        "[Stage B] Residual joint fit complete: val_mse={:.6e}, val_rms={:.6e}".format(
                            float(meta["val_mse"]),
                            float(meta["val_rms"]),
                        )
                    )
                    if "val_mse_per_dataset" in meta:
                        vals = ", ".join(f"{float(v):.6e}" for v in meta["val_mse_per_dataset"])
                        print(f"[Stage B] Per-dataset val_mse: [{vals}]")
            except Exception as e:
                print(f"[Stage B] Warning: residual refinement failed ({e})")
                if args.verbose:
                    import traceback

                    traceback.print_exc()

    # Print / save results
    selected_is_factorized_search = isinstance(res, FactorizedSearchDEResult)
    selected_is_factorized = isinstance(res, FactorizedDEResult)
    is_multi_res = (
        (not selected_is_factorized_search)
        and (not selected_is_factorized)
        and hasattr(res, "coeffs")
        and isinstance(res.coeffs, torch.Tensor)
        and res.coeffs.ndim == 2
    )
    dataset_ids_out = list(dataset_ids)

    print(f"\nSelected engine: {selected_engine}")
    if primary_res is not None and (factorized_res is not None or rescue_res is not None):
        print("Engine comparison:")
        print(f"  first-line probe RMS: {candidate_probe_rms(primary_res):.6e}")
        if factorized_res is not None:
            print(f"  factorized probe RMS: {candidate_probe_rms(factorized_res):.6e}")
        if rescue_res is not None:
            print(f"  factorized symbolic search probe RMS:   {candidate_probe_rms(rescue_res):.6e}")
    elif primary_res is None and (factorized_res is not None or rescue_res is not None):
        print("STLSQ-free operator-factorized DE mode:")
        print("  first-line candidate: none")
        if factorized_res is not None:
            print(f"  factorized probe RMS: {candidate_probe_rms(factorized_res):.6e}")
        if rescue_res is not None:
            print(f"  factorized symbolic search probe RMS:   {candidate_probe_rms(rescue_res):.6e}")
    else:
        if str(factorized_cfg.mode) != "never":
            print("Factorized rescue check:")
            print(
                "  triggered: {} ({})".format(
                    bool(factorized_triggered),
                    factorized_trigger_reason or "no_trigger",
                )
            )
        if str(rescue_cfg.mode) != "never":
            print("factorized symbolic search rescue check:")
            print(
                "  triggered: {} ({})".format(
                    bool(rescue_triggered),
                    rescue_trigger_reason or "no_trigger",
                )
            )

    print(f"\nDiscovered DE (order {res.order}):")
    print("-" * 70)

    human_output = os.path.join(args.output_dir, f"{base_filename}_de.human")
    with open(human_output, "w") as f:
        f.write(f"Datasets: {filepaths}\n")
        f.write(f"Surrogate val_losses: {val_losses}\n")
        f.write(f"Selected engine: {selected_engine}\n")
        if primary_res is not None and (factorized_res is not None or rescue_res is not None):
            f.write("Engine comparison:\n")
            f.write(f"  first-line probe RMS: {candidate_probe_rms(primary_res):.6e}\n")
            if factorized_res is not None:
                f.write(f"  factorized probe RMS: {candidate_probe_rms(factorized_res):.6e}\n")
            if rescue_res is not None:
                f.write(f"  factorized symbolic search probe RMS:   {candidate_probe_rms(rescue_res):.6e}\n")
        elif primary_res is None and (factorized_res is not None or rescue_res is not None):
            f.write("STLSQ-free operator-factorized DE mode:\n")
            f.write("  first-line candidate: none\n")
            if factorized_res is not None:
                f.write(f"  factorized probe RMS: {candidate_probe_rms(factorized_res):.6e}\n")
            if rescue_res is not None:
                f.write(f"  factorized symbolic search probe RMS:   {candidate_probe_rms(rescue_res):.6e}\n")
        else:
            if str(factorized_cfg.mode) != "never":
                f.write(
                    "Factorized rescue check:\n"
                    f"  triggered: {bool(factorized_triggered)} ({factorized_trigger_reason or 'no_trigger'})\n"
                )
            if str(rescue_cfg.mode) != "never":
                f.write(
                    "factorized symbolic search rescue check:\n"
                    f"  triggered: {bool(rescue_triggered)} ({rescue_trigger_reason or 'no_trigger'})\n"
                )
        f.write(f"\nDiscovered DE (order {res.order}):\n")

        if selected_is_factorized_search:
            print("\nCanonical equation:")
            print(f"  {res.format_equation()}")

            print("\nFactorized search:")
            print(f"  Mapping kind: {res.mapping_kind}")
            print(f"  Feature names: {res.feature_names}")

            if res.rhs_ast is not None:
                print("\nRHS AST:")
                print(f"  {repr(res.rhs_ast)}")

            print("\nResidual AST:")
            print(f"  {repr(res.residual_ast)}")

            print("\nFit quality:")
            print(f"  Probe MSE: {float(res.probe_mse):.6e}")
            print(f"  Probe RMS: {float(res.probe_rms):.6e}")

            fit_ids = list((res.diagnostics or {}).get("fit_traj_ids", []) or [])
            probe_ids = list((res.diagnostics or {}).get("probe_traj_ids", []) or [])
            if fit_ids:
                print(f"  Fit groups:   {fit_ids}")
            if probe_ids:
                print(f"  Probe groups: {probe_ids}")

            f.write(f"  {res.format_equation()}\n")
            f.write("\nFactorized search:\n")
            f.write(f"  Mapping kind: {res.mapping_kind}\n")
            f.write(f"  Feature names: {res.feature_names}\n")
            if res.rhs_ast is not None:
                f.write(f"\nRHS AST:\n  {repr(res.rhs_ast)}\n")
            f.write(f"\nResidual AST:\n  {repr(res.residual_ast)}\n")
            f.write(f"\nProbe MSE: {float(res.probe_mse):.6e}\n")
            f.write(f"Probe RMS: {float(res.probe_rms):.6e}\n")
            if fit_ids:
                f.write(f"Fit groups:   {fit_ids}\n")
            if probe_ids:
                f.write(f"Probe groups: {probe_ids}\n")
        elif selected_is_factorized:
            print("\nCanonical equation:")
            print(f"  {res.format_equation()}")

            print("\nBlocks:")
            for i, block in enumerate(list(getattr(res, "blocks", []) or [])):
                print(f"  block[{i}] role={getattr(block, 'role', '?')}")
                print(f"    carrier: {repr(getattr(block, 'carrier_ast', None))}")
                print(f"    coord:   {repr(getattr(block, 'coord_ast', None))}")
                print(f"    coeff:   {repr(getattr(block, 'coeff_ast', None))}")

            print("\nResidual AST:")
            print(f"  {repr(res.residual_ast)}")

            print("\nFit quality:")
            print(f"  Probe MSE: {float(res.probe_mse):.6e}")
            print(f"  Probe RMS: {float(res.probe_rms):.6e}")

            f.write(f"  {res.format_equation()}\n")
            f.write("\nBlocks:\n")
            for i, block in enumerate(list(getattr(res, "blocks", []) or [])):
                f.write(f"  block[{i}] role={getattr(block, 'role', '?')}\n")
                f.write(f"    carrier: {repr(getattr(block, 'carrier_ast', None))}\n")
                f.write(f"    coord:   {repr(getattr(block, 'coord_ast', None))}\n")
                f.write(f"    coeff:   {repr(getattr(block, 'coeff_ast', None))}\n")
            f.write(f"\nResidual AST:\n  {repr(res.residual_ast)}\n")
            f.write(f"\nProbe MSE: {float(res.probe_mse):.6e}\n")
            f.write(f"Probe RMS: {float(res.probe_rms):.6e}\n")
        elif not is_multi_res:
            print("\nDiscovered terms (shared support):")
            for j, t in enumerate(res.term_asts):
                if t is None:
                    term_str = "1"
                else:
                    try:
                        term_str = repr(t)
                    except Exception:
                        term_str = str(type(t).__name__)
                print(f"  term[{j}]: {term_str}")

            print("\nCanonical equation:")
            print(f"  {res.format_equation()}")

            print("\nCoefficients:")
            for t, c in zip(res.term_asts, res.coeffs.tolist()):
                term_str = "1" if t is None else (repr(t) if hasattr(t, "__repr__") else str(t))
                print(f"  {c:12.6g}   {term_str}")

            print("\nResidual AST:")
            print(f"  {repr(res.residual_ast)}")

            rms_val_str = f"{res.rms_val:.6e}" if res.rms_val is not None else "N/A"
            print("\nFit quality:")
            print(f"  RMS (train): {res.rms_train:.6e}")
            print(f"  RMS (val):   {rms_val_str}")
            if getattr(res, "condition_number", None) is not None:
                print(f"  Condition number: {float(res.condition_number):.2e}")

            f.write(f"  {res.format_equation()}\n")
            f.write("\nCoefficients:\n")
            for t, c in zip(res.term_asts, res.coeffs.tolist()):
                term_str = "1" if t is None else repr(t)
                f.write(f"  {c:12.6g} * {term_str}\n")
            f.write(f"\nRMS (train): {res.rms_train:.6e}\n")
            f.write(f"RMS (val):   {rms_val_str}\n")
            if getattr(res, "condition_number", None) is not None:
                f.write(f"Condition number: {float(res.condition_number):.2e}\n")
            sb_meta = getattr(res, "stageb_residual_metadata", None)
            if isinstance(sb_meta, dict):
                print("\nStage-B residual refinement:")
                print(
                    "  val_mse={:.6e}  val_rms={:.6e}  coeffs_updated={}".format(
                        float(sb_meta.get("val_mse", float("nan"))),
                        float(sb_meta.get("val_rms", float("nan"))),
                        bool(sb_meta.get("coefficients_updated", False)),
                    )
                )
                f.write("\nStage-B residual refinement:\n")
                f.write(
                    "  val_mse={:.6e}  val_rms={:.6e}  coeffs_updated={}\n".format(
                        float(sb_meta.get("val_mse", float("nan"))),
                        float(sb_meta.get("val_rms", float("nan"))),
                        bool(sb_meta.get("coefficients_updated", False)),
                    )
                )

        else:
            print("\nDiscovered terms (shared support):")
            for j, t in enumerate(res.term_asts):
                if t is None:
                    term_str = "1"
                else:
                    try:
                        term_str = repr(t)
                    except Exception:
                        term_str = str(type(t).__name__)
                print(f"  term[{j}]: {term_str}")

            # Multi-dataset coefficients: one equation per dataset
            for d, dsid in enumerate(dataset_ids_out):
                eq = res.format_equation_for_dataset(d)
                print(f"\nDataset {d}: {dsid}")
                print(f"  {eq}")
                rt = res.rms_train[d]
                rv = res.rms_val[d] if res.rms_val is not None else None
                rv_str = f"{rv:.6e}" if rv is not None else "N/A"
                print(f"  RMS(train)={rt:.6e}  RMS(val)={rv_str}")

                f.write(f"\nDataset {d}: {dsid}\n")
                f.write(f"  {eq}\n")
                f.write(f"  RMS(train): {rt:.6e}\n")
                f.write(f"  RMS(val):   {rv_str}\n")

            # Show one residual AST if available
            if getattr(res, "residual_asts", None) is not None and len(res.residual_asts) > 0:
                print("\nResidual AST (dataset 0):")
                print(f"  {repr(res.residual_asts[0])}")
                f.write(f"\nResidual AST (dataset 0):\n  {repr(res.residual_asts[0])}\n")
            sb_meta = getattr(res, "stageb_residual_metadata", None)
            if isinstance(sb_meta, dict):
                print("\nStage-B residual refinement (joint):")
                print(
                    "  val_mse={:.6e}  val_rms={:.6e}  coeffs_updated={}".format(
                        float(sb_meta.get("val_mse", float("nan"))),
                        float(sb_meta.get("val_rms", float("nan"))),
                        bool(sb_meta.get("coefficients_updated", False)),
                    )
                )
                if "val_mse_per_dataset" in sb_meta:
                    vals = ", ".join(f"{float(v):.6e}" for v in sb_meta.get("val_mse_per_dataset", []))
                    print(f"  per-dataset val_mse=[{vals}]")
                f.write("\nStage-B residual refinement (joint):\n")
                f.write(
                    "  val_mse={:.6e}  val_rms={:.6e}  coeffs_updated={}\n".format(
                        float(sb_meta.get("val_mse", float("nan"))),
                        float(sb_meta.get("val_rms", float("nan"))),
                        bool(sb_meta.get("coefficients_updated", False)),
                    )
                )

    print(f"\nSaved human-readable output to {human_output}")

    # Save JSON report if requested
    if args.save_json:
        elapsed_time = (timeit.default_timer() - start_time) / 3600.0
        report_path = os.path.join(args.output_dir, f"{base_filename}_de.json")
        primary_validation_candidate = None
        if primary_res is not None and not isinstance(primary_res, (FactorizedSearchDEResult, FactorizedDEResult)):
            primary_validation_candidate = _build_library_validation_candidate(
                primary_res,
                surrogates=surrogates,
                train_dataloaders=dl_tr_list,
                val_dataloaders=dl_va_list,
                cfg=cfg,
                device=dev,
            )
        selected_validation_candidate = None
        if not isinstance(res, (FactorizedSearchDEResult, FactorizedDEResult)):
            selected_validation_candidate = _build_library_validation_candidate(
                res,
                surrogates=surrogates,
                train_dataloaders=dl_tr_list,
                val_dataloaders=dl_va_list,
                cfg=cfg,
                device=dev,
            )
        de_candidate_eval_report = None
        if feature_groups_for_eval is not None:
            try:
                de_candidate_eval_report = build_de_candidate_eval_report(
                    feature_groups_for_eval,
                    cfg=cfg,
                    order=int(getattr(res, "order", tuple(cfg.order_candidates)[0])),
                    x_axis=int(getattr(res, "x_axis", cfg.x_axis)),
                    primary_result=primary_res,
                    factorized_result=factorized_res,
                    factorized_search_result=rescue_res,
                    ast_serializer=_serialize_de_ast,
                    dtype=dtype,
                )
            except Exception as exc:
                de_candidate_eval_report = {
                    "enabled": True,
                    "mode": "diagnostics_only",
                    "status": "ERROR",
                    "message": str(exc),
                }
        write_de_json_report(
            filepaths,
            report_path,
            val_losses,
            res,
            args,
            elapsed_time,
            primary_result=primary_res,
            factorized_result=factorized_res,
            rescue_result=rescue_res,
            selected_engine=selected_engine,
            factorized_cfg=factorized_cfg,
            rescue_cfg=rescue_cfg,
            factorized_triggered=factorized_triggered,
            factorized_trigger_reason=factorized_trigger_reason,
            rescue_triggered=rescue_triggered,
            rescue_trigger_reason=rescue_trigger_reason,
            primary_validation_candidate=primary_validation_candidate,
            selected_validation_candidate=selected_validation_candidate,
            de_candidate_eval_report=de_candidate_eval_report,
        )

    if bool(getattr(args, "gs_enable", False)):
        try:
            from nestynet_sr.sr_gs.reporting import record_policy_event, write_gs_reports

            final_expr = None
            if hasattr(res, "format_equation_for_dataset") and dataset_ids_out:
                try:
                    final_expr = res.format_equation_for_dataset(0)
                except Exception:
                    final_expr = None
            if final_expr is None and hasattr(res, "format_equation"):
                try:
                    final_expr = res.format_equation()
                except Exception:
                    final_expr = None
            prolong_meta = getattr(res, "prolongation_metadata", None)
            if isinstance(prolong_meta, dict):
                record_policy_event(
                    policy=str(getattr(args, "gs_policy", "augment")),
                    action="de_lie_prolongation_selected",
                    details={"metadata": prolong_meta},
                )
            gs_json_path = str(
                getattr(args, "gs_unit_report_json", None)
                or os.path.join(args.output_dir, f"{base_filename}.gs_report.json")
            )
            gs_md_path = str(
                getattr(args, "gs_unit_report_md", None)
                or os.path.join(args.output_dir, f"{base_filename}.gs_report.md")
            )
            write_gs_reports(
                json_path=gs_json_path,
                markdown_path=gs_md_path,
                final_expression=str(final_expr) if final_expr is not None else None,
                mode=f"de_{str(getattr(args, 'gs_mode', 'propose') or 'propose')}",
                include_rejected=bool(getattr(args, "gs_report_rejected", True)),
                top_k_rejected=int(getattr(args, "gs_report_top_k_rejected", 40)),
            )
            print(f"[GS-DE] GS report written: {gs_json_path} ; {gs_md_path}")
        except Exception as e_gs_report:
            print(f"[GS-DE] Warning: failed to write GS report: {e_gs_report}")

    elapsed_time = (timeit.default_timer() - start_time) / 3600.0
    print(f"\nTotal time: {elapsed_time:.4f} hours")
    print("=" * 70)


if __name__ == "__main__":
    main()
