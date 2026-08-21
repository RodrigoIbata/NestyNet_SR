# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""DE-discovery helpers used by the symbolic-regression CLI."""

from __future__ import annotations

import pathlib

import torch

from nestynet_sr.run_sr_reports import _make_json_serializable
from nestynet_sr.sr_search.model_builders import build_composite_ast


def _de_stageB_report_payload(state, *, phi_expr_strs=None):
    """Return the persistent Stage-B portion of a DE result payload."""

    if state is None:
        return None
    return {
        "val_loss": float(getattr(state, "val_loss", 0.0)),
        "val_losses": getattr(state, "val_losses", None),
        "agg_mode": getattr(state, "agg_mode", None),
        "agg_weights": getattr(state, "agg_weights", None),
        "phi_expr_str": getattr(state, "phi_expr_str", None),
        "phi_expr_strs": phi_expr_strs,
        "enabled_patterns": getattr(state, "enabled_patterns", None),
        "coefficient_metadata": getattr(state, "coefficient_metadata", None),
        "coefficient_metadata_by_dataset": getattr(
            state, "coefficient_metadata_by_dataset", None
        ),
    }


def _coefficient_definitions_for_human(metadata):
    if not isinstance(metadata, dict) or metadata.get("valid") is not True:
        return []
    definitions = []
    for record in list(metadata.get("records") or []):
        if not isinstance(record, dict) or record.get("symbol") is None:
            continue
        dimension = record.get("dimension")
        dimension_text = (
            "unavailable" if dimension is None else "[" + ", ".join(dimension) + "]"
        )
        definitions.append(
            f"{record['symbol']} = {record.get('value')} "
            f"(name={record.get('name')}, dim={dimension_text}, scope={record.get('scope')})"
        )
    return definitions


class _StageAXTransformDataset(torch.utils.data.Dataset):
    """Dataset wrapper that applies x-coordinate transform to input tensor."""

    def __init__(self, base_ds, x_op):
        self.base_ds = base_ds
        self.x_op = x_op

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        item = self.base_ds[idx]
        if isinstance(item, tuple):
            x = item[0]
            return (self.x_op(x), *item[1:])
        if isinstance(item, list):
            x = item[0]
            return [self.x_op(x), *item[1:]]
        if isinstance(item, dict):
            out = dict(item)
            if "x" in out:
                out["x"] = self.x_op(out["x"])
            else:
                for k, v in out.items():
                    out[k] = self.x_op(v)
                    break
            return out
        return self.x_op(item)


def _train_stageA_models_multi_for_stageB(
    *,
    base_model: torch.nn.Module,
    ast_template,
    filepaths: list,
    Nxvars: int,
    np_dtype,
    data_hp,
    y_op,
    leaf_builder,
    lm_hp,
    device: torch.device,
    dtype: torch.dtype,
):
    """Build per-dataset Stage-A teacher models (same AST structure) for Stage B."""
    if filepaths is None or len(filepaths) <= 1:
        return [base_model]

    from nestynet_sr.sr_core.bridges import clone_ast
    from nestynet_sr.sr_search.data_utils import build_datasets_multi
    from nestynet_sr.sr_search.training import train_initial_model

    _, _, train_loaders, val_loaders = build_datasets_multi(
        filepaths=list(filepaths),
        Nxvars=Nxvars,
        np_dtype=np_dtype,
        data_hp=data_hp,
        y_op=y_op,
    )
    if train_loaders is None or val_loaders is None:
        raise RuntimeError("Failed to build multi-dataset loaders for Stage-A teacher bootstrap")

    # Preserve Stage-A x-coordinate map, if any.
    x_transform_map = getattr(base_model, "_x_transform", None) or {}
    if x_transform_map:
        try:
            from nestynet_sr.sr_search.xcoord import XCoordSystem

            xcoords = XCoordSystem.from_map(x_transform_map, Nx_raw=Nxvars)
            if xcoords is not None and (not xcoords.is_identity()):
                def _x_op(x, _xc=xcoords):
                    return _xc.apply_torch(x)

                train_loaders = [
                    torch.utils.data.DataLoader(
                        _StageAXTransformDataset(dl.dataset, _x_op),
                        batch_size=getattr(dl, "batch_size", None) or data_hp.batch_size,
                        shuffle=False,
                        drop_last=bool(getattr(dl, "drop_last", False)),
                    )
                    for dl in train_loaders
                ]
                val_loaders = [
                    torch.utils.data.DataLoader(
                        _StageAXTransformDataset(dl.dataset, _x_op),
                        batch_size=getattr(dl, "batch_size", None) or data_hp.batch_size,
                        shuffle=False,
                        drop_last=bool(getattr(dl, "drop_last", False)),
                    )
                    for dl in val_loaders
                ]
        except Exception as e:
            print(f"[Stage A multi] Warning: failed to apply x-transform map to bootstrap loaders: {e}")

    models = [base_model]
    print(
        f"[Stage A multi] Bootstrapping per-dataset teachers for Stage B "
        f"({len(filepaths)} datasets, shared AST)."
    )

    for i in range(1, len(filepaths)):
        ast_i = clone_ast(ast_template)
        model_i, nparam_i, _ = build_composite_ast(
            ast_i,
            num_segments=None,
            dual_layer=None,
            leaf_builder=leaf_builder,
            device=device,
            dtype=dtype,
        )
        setattr(model_i, "fit_y_link", getattr(lm_hp, "fit_y_link", None))
        setattr(model_i, "fit_y_link_scale", float(getattr(lm_hp, "fit_y_link_scale", 1.0)))

        # Warm-start from dataset-0 Stage-A state when compatible.
        try:
            model_i.load_state_dict(base_model.state_dict(), strict=True)
        except Exception:
            pass

        best_val_loss, _, best_val_p, lm_opt = train_initial_model(
            model=model_i,
            train_dl=train_loaders[i],
            val_dl=val_loaders[i],
            epochs=lm_hp.epochs,
            LM_strategy=lm_hp.strategy,
            nval_patience=lm_hp.nval_patience,
            loss_target=lm_hp.loss_target,
            epochs_min=lm_hp.epochs_min,
            chisq_tol=lm_hp.chisq_tol,
            device=device,
            epochs_awful_check=lm_hp.epochs_awful_check,
            awful_threshold=lm_hp.awful_threshold,
            log_file=getattr(lm_hp, "log_file", None),
            log_to_console=getattr(lm_hp, "log_to_console", True),
            log_level=getattr(lm_hp, "log_level", None),
            lm_hp=lm_hp,
        )
        if best_val_p is not None:
            lm_opt._update_param_groups(best_val_p)

        if x_transform_map:
            setattr(model_i, "_x_transform", dict(x_transform_map))

        print(
            f"[Stage A multi] dataset[{i}] {pathlib.Path(filepaths[i]).name}: "
            f"params={nparam_i}, val_loss={best_val_loss:.6e}"
        )
        models.append(model_i)

    return models


def _parse_int_tuple(csv: str, *, default: tuple = (1, 2)) -> tuple:
    if csv is None:
        return tuple(default)
    try:
        parts = [p.strip() for p in str(csv).split(",") if p.strip()]
        out = tuple(int(p) for p in parts)
        return out if len(out) > 0 else tuple(default)
    except Exception:
        return tuple(default)


def _run_firstclass_de_for_sr(
    *,
    args,
    filepaths: list,
    base_filename: str,
    results_dir: str,
    Nxvars: int,
    np_dtype,
    data_hp,
    lm_hp,
    leaf_builder,
    device: torch.device,
    dtype: torch.dtype,
    final_model,
    final_ast,
    final_y_op,
    final_y_op_inv,
    final_y_op_name: str,
    model_output_identity: str,
    model_sep_output_identity: str,
    units_payload=None,
    factorized_search_hp=None,
    disabled_patterns: list = None,
):
    """Optional DE discovery pass that behaves like a first-class SR output.

    The workflow is:
      1) pick a surrogate family (identity y-space by default)
      2) discover an implicit residual DE across datasets
      3) (optional) run Stage B on the residual with y≡0 targets
      4) write results/<stem>_de.{pkl,human} and return a JSON-serializable dict
    """

    if not getattr(args, "discover_de", False):
        return None

    import os
    import pickle
    import pathlib
    from dataclasses import asdict

    import torch
    from torch.utils.data import DataLoader, Dataset

    from nestynet_sr.sr_core.bridges import AtomNode, build_composite_from_ast, collect_all_atoms
    from nestynet_sr.sr_search.data_utils import build_datasets_multi
    from nestynet_sr.sr_search.representation import pretty_print_state
    from nestynet_sr.sr_search.stageB import run_stageB_from_model
    from nestynet_sr.sr_search.stageB.leaf_utils import _set_constant_leaf_value
    from nestynet_sr.sr_expr_ir.config import apply_expr_ir_args_to_obj

    from nestynet_sr.sr_de.de_search import (
        DESearchConfig,
        DESearchResult,
        build_de_residual_ast,
        discover_de_from_surrogates,
        make_u_feature_atom_factory,
    )

    # -------------------------
    # Choose y-space + surrogate
    # -------------------------
    de_y_space = getattr(args, "de_y_space", "identity")
    de_y_name = "identity" if de_y_space == "identity" else (final_y_op_name or "final")

    # Units straightjacket context (optional): build a UnitsSpec consistent with the
    # DE y-space (identity vs. final φ(y)).
    de_units_spec = None
    if units_payload is not None:
        try:
            from nestynet_sr.sr_core.units import UnitsSpec

            de_units_spec = UnitsSpec(
                unit_system=units_payload["unit_system"],
                x_dims=units_payload["x_dims"],
                y_dim=units_payload["y_dim"],
                y_transform_name=de_y_name,
                policy=getattr(args, "units_policy", "strict"),
                nn_semantics=getattr(args, "nn_units_semantics", "dimless"),
                free_const_dims=units_payload.get("free_const_dims", {}),
                free_const_scope=units_payload.get("free_const_scope", {}),
                fixed_const_dims=units_payload.get("fixed_const_dims", {}),
                fixed_const_values=units_payload.get("fixed_const_values", {}),
                fixed_const_mode=units_payload.get("fixed_const_mode", "strict"),
            )
        except Exception:
            de_units_spec = None
    de_y_op = None
    base_surrogate = None
    ast_template = None

    def _load_stageA_model_from_file(path: str):
        if path is None or (not os.path.exists(path)):
            return None
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            return None
        ast = payload.get("ast", None)
        if ast is None:
            return None
        num_segments = payload.get("num_segments", None)
        dual_layer = payload.get("dual_layer", None)
        model, _, _ = build_composite_ast(
            ast,
            num_segments,
            dual_layer=dual_layer,
            leaf_builder=leaf_builder,
            device=device,
            dtype=dtype,
        )
        try:
            model.load_state_dict(payload.get("model_state_dict", {}), strict=False)
        except Exception:
            pass
        setattr(model, "fit_y_link", payload.get("fit_y_link", None))
        setattr(model, "fit_y_link_scale", float(payload.get("fit_y_link_scale", 1.0)))
        try:
            model.eval()
        except Exception:
            pass
        return model

    if de_y_space == "final":
        base_surrogate = final_model
        de_y_op = final_y_op
        ast_template = final_ast
    else:
        base_surrogate = _load_stageA_model_from_file(model_sep_output_identity)
        if base_surrogate is None:
            base_surrogate = _load_stageA_model_from_file(model_output_identity)
        if base_surrogate is None:
            # Fall back to whatever we have; warn via payload.
            base_surrogate = final_model
            de_y_name = final_y_op_name or "final"
            de_y_op = final_y_op
            ast_template = final_ast
        else:
            de_y_op = None
            ast_template = final_ast

    # Prefer an AST template extracted from the chosen surrogate (matches leaf shapes).
    try:
        from nestynet_sr.sr_core import ast_from_composite as _ast_from_comp

        _tmp_ast, _ = _ast_from_comp(base_surrogate)
        if _tmp_ast is not None:
            ast_template = _tmp_ast
    except Exception:
        pass

    # Build per-dataset surrogate family when needed.
    surrogates = [base_surrogate]
    if len(filepaths) > 1:
        try:
            surrogates = _train_stageA_models_multi_for_stageB(
                base_model=base_surrogate,
                ast_template=ast_template,
                filepaths=filepaths,
                Nxvars=Nxvars,
                np_dtype=np_dtype,
                data_hp=data_hp,
                y_op=de_y_op,
                leaf_builder=leaf_builder,
                lm_hp=lm_hp,
                device=device,
                dtype=dtype,
            )
        except Exception as e:
            print(f"[DE/SR] Warning: failed to bootstrap per-dataset surrogates; using dataset-0 only. ({e})")
            surrogates = [base_surrogate]

    D = len(surrogates)
    dataset_ids = [pathlib.Path(p).stem for p in filepaths[:D]]

    # -------------------------
    # Build loaders in y-space
    # -------------------------
    ds_tr_list, ds_va_list, dl_tr_list, dl_va_list = build_datasets_multi(
        filepaths=filepaths[:D],
        Nxvars=Nxvars,
        np_dtype=np_dtype,
        data_hp=data_hp,
        y_op=de_y_op,
    )
    if dl_tr_list is None or dl_va_list is None:
        raise RuntimeError("[DE/SR] Failed to build datasets for DE discovery")

    class _ZeroTargetDataset(Dataset):
        def __init__(self, base: Dataset):
            self.base = base

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            item = self.base[idx]
            if isinstance(item, (tuple, list)):
                x = item[0]
                y_ref = item[1]
                if isinstance(y_ref, torch.Tensor):
                    y0 = torch.zeros_like(y_ref)
                else:
                    # Some datasets return NumPy targets; normalize to Tensor first.
                    y0 = torch.zeros_like(torch.as_tensor(y_ref))
                return (x, y0, *item[2:])
            raise ValueError("Dataset must return (x,y,...) tuples")

    def _zero_loader_from(dl: DataLoader):
        bs = getattr(dl, "batch_size", None) or getattr(data_hp, "batch_size", 1024)
        drop_last = bool(getattr(dl, "drop_last", False))
        return DataLoader(_ZeroTargetDataset(dl.dataset), batch_size=bs, shuffle=False, drop_last=drop_last)

    zero_train_loaders = [_zero_loader_from(dl) for dl in dl_tr_list]
    zero_val_loaders = [_zero_loader_from(dl) for dl in dl_va_list]
    zero_train_datasets = [_ZeroTargetDataset(ds) for ds in ds_tr_list]
    zero_val_datasets = [_ZeroTargetDataset(ds) for ds in ds_va_list]

    # -------------------------
    # DE discovery
    # -------------------------
    order_candidates = _parse_int_tuple(getattr(args, "de_order_candidates", "1,2"), default=(1, 2))
    x_axis = int(getattr(args, "de_x_axis", 0) if getattr(args, "de_x_axis", None) is not None else 0)
    cfg = DESearchConfig(
        x_axis=x_axis,
        order_candidates=order_candidates,
        max_x_power=int(getattr(args, "de_max_x_power", 1)),
        max_u_power=int(getattr(args, "de_max_u_power", 2)),
        include_const=bool(getattr(args, "de_include_const", True)),
        include_x=bool(getattr(args, "de_include_x", True)),
        include_u=bool(getattr(args, "de_include_u", True)),
        include_du=bool(getattr(args, "de_include_du", False)),
        include_d2u=bool(getattr(args, "de_include_d2u", False)),
        include_xu=bool(getattr(args, "de_include_xu", True)),
        include_xdu=bool(getattr(args, "de_include_xdu", True)),
        include_udu=bool(getattr(args, "de_include_udu", False)),
        ridge=float(getattr(args, "de_ridge", 1e-10)),
        stlsq_lambda=float(getattr(args, "de_stlsq_lambda", 1e-3)),
        stlsq_max_iter=int(getattr(args, "de_stlsq_max_iter", 10)),
        max_batches=int(getattr(args, "de_max_batches", 32)),
        max_points=int(getattr(args, "de_max_points", 20000)),
        sparsity_penalty=float(getattr(args, "de_sparsity_penalty", 1e-3)),
        units_spec=de_units_spec,
        enforce_units=bool(getattr(args, "enforce_units", False)),
        ast_simplify=bool(getattr(args, "de_ast_simplify", False)),
        ast_simplify_level=str(getattr(args, "de_ast_simplify_level", "safe") or "safe"),
        ast_simplify_domain_policy=str(getattr(args, "de_ast_simplify_domain_policy", "strict") or "strict"),
        ast_simplify_max_passes=int(getattr(args, "de_ast_simplify_max_passes", 12)),
        ast_simplify_validate=bool(getattr(args, "de_ast_simplify_validate", False)),
        ast_simplify_trace=bool(getattr(args, "de_ast_simplify_trace", False)),
        gs_enable=bool(getattr(args, "gs_unit_torus", False) or getattr(args, "gs_pi_invariants", False)),
        gs_mode="propose",
        gs_unit_torus=bool(getattr(args, "gs_unit_torus", False) or getattr(args, "gs_pi_invariants", False)),
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
    de_res_multi = discover_de_from_surrogates(
        surrogates=surrogates,
        train_dataloaders=dl_tr_list,
        val_dataloaders=dl_va_list,
        cfg=cfg,
        device=device,
        datasets=ds_tr_list,
        dataset_ids=dataset_ids,
    )

    residual_asts = []
    eqn_raw = []
    for d in range(D):
        tmp = DESearchResult(
            order=de_res_multi.order,
            x_axis=de_res_multi.x_axis,
            term_asts=de_res_multi.term_asts,
            coeffs=de_res_multi.coeffs[d].detach().cpu(),
            rms_train=float(de_res_multi.rms_train[d]),
            rms_val=float(de_res_multi.rms_val[d]) if de_res_multi.rms_val is not None else None,
        )
        residual_asts.append(
            build_de_residual_ast(
                tmp,
                coeff_prefix=str(getattr(args, "de_coeff_prefix", "c")),
                units_spec=de_units_spec,
                enforce_units=bool(getattr(args, "enforce_units", False)),
            )
        )
        try:
            eqn_raw.append(tmp.format_equation(tol=1e-3, var_name=f"x{de_res_multi.x_axis}"))
        except Exception:
            eqn_raw.append(None)
    de_res_multi.residual_asts = residual_asts

    # -------------------------
    # Class-DE promotion (optional)
    # -------------------------
    class_de_info = None
    if bool(getattr(args, "de_class_de", False)) and D > 1:
        try:
            coeffs = de_res_multi.coeffs.detach().cpu()  # (D,K)
            mu = coeffs.mean(dim=0)
            sd = coeffs.std(dim=0, unbiased=False)
            cv = sd / (mu.abs() + 1e-12)
            thresh = float(getattr(args, "de_class_de_cv", 0.05))
            stable = (cv <= thresh).tolist()
            stable_idx = [k for k, ok in enumerate(stable) if ok]

            # Promote stable coefficient atoms on the *root* residual (dataset 0)
            prefix = str(getattr(args, "de_coeff_prefix", "c"))
            tag_to_mu = {f"{prefix}{k}": float(mu[k]) for k in stable_idx}

            atoms0 = [a for a in collect_all_atoms(residual_asts[0]) if isinstance(a, AtomNode)]
            promoted_tags = []
            for a in atoms0:
                if getattr(a, "tag", None) in tag_to_mu:
                    setattr(a, "scope", "class")
                    promoted_tags.append(str(a.tag))

            class_de_info = {
                "cv_threshold": thresh,
                "promoted_tags": promoted_tags,
            }
        except Exception as e:
            print(f"[DE/SR] Class-DE promotion failed: {e}")
            class_de_info = None

    # -------------------------
    # Optional Stage B refinement on residual (DE as SR)
    # -------------------------
    de_stageB_state = None
    de_phi_expr_strs = None

    if bool(getattr(args, "de_stageB", True)):
        atom_factories = [make_u_feature_atom_factory(s) for s in surrogates]

        # Seed per-dataset reuse maps with the discovered coefficient values.
        def _is_scalar_coeff_atom(atom: AtomNode) -> bool:
            kind = str(getattr(atom, "kind", "")).lower()
            if kind in ("u", "du", "d2u", "field", "state", "var", "x", "input", "nn"):
                return False
            if len(getattr(atom, "var_idxs", ())) != 0:
                return False
            return kind in ("scale", "free_const", "fixed_const")

        reuses = []
        for i in range(D):
            comp_i, atom_map_i = build_composite_from_ast(
                residual_asts[i],
                device=device,
                dtype=dtype,
                atom_factory=atom_factories[i],
                reuse={},
                return_atom_map=True,
            )
            reuse_i = {}
            for atom in collect_all_atoms(residual_asts[i]):
                if not isinstance(atom, AtomNode):
                    continue
                if not _is_scalar_coeff_atom(atom):
                    continue
                tag = getattr(atom, "tag", None)
                if tag is None:
                    continue
                leaf = atom_map_i.get(id(atom), None)
                if leaf is None:
                    continue
                reuse_i[str(tag)] = leaf
            reuses.append(reuse_i)

        # If Class-DE promoted tags exist, initialise shared leaf (dataset 0) to mean coeff.
        if class_de_info is not None:
            try:
                prefix = str(getattr(args, "de_coeff_prefix", "c"))
                mu = de_res_multi.coeffs.detach().cpu().mean(dim=0)
                for tag in class_de_info.get("promoted_tags", []):
                    try:
                        k = int(str(tag).replace(prefix, ""))
                    except Exception:
                        continue
                    leaf0 = reuses[0].get(tag)
                    if leaf0 is not None:
                        _set_constant_leaf_value(leaf0, float(mu[k]))
            except Exception:
                pass

        # Minimal stageA_model placeholder: only used for x-transform metadata.
        stageA_model0 = build_composite_from_ast(
            residual_asts[0],
            device=device,
            dtype=dtype,
            atom_factory=atom_factories[0],
            reuse={},
        )
        setattr(stageA_model0, "_x_transform", getattr(surrogates[0], "_x_transform", None) or {})

        de_max_outer = (
            int(getattr(args, "de_stageB_max_outer_iters", 0) or 0)
            or int(getattr(args, "stageB_max_outer_iters", 0) or 0)
            or 30
        )
        de_epochs = (
            int(getattr(args, "de_stageB_epochs", 0) or 0)
            or int(getattr(args, "stageB_epochs", 0) or 0)
            or 2000
        )
        de_score_tol = float(getattr(args, "stageB_score_tol", 0.0) or 0.0)

        if D > 1:
            datasets_override = {
                "dataset_train": zero_train_datasets,
                "dataset_val": zero_val_datasets,
                "train_loader": zero_train_loaders,
                "val_loader": zero_val_loaders,
                "dataset_ids": dataset_ids,
                "agg_mode": "weighted",
                "agg_weights": [float(len(ds)) for ds in zero_val_datasets],
            }
        else:
            datasets_override = {
                "dataset_train": zero_train_datasets[0],
                "dataset_val": zero_val_datasets[0],
                "train_loader": zero_train_loaders[0],
                "val_loader": zero_val_loaders[0],
                "dataset_ids": dataset_ids,
                "agg_mode": "mean",
                "agg_weights": None,
            }

        de_stageB_state = run_stageB_from_model(
            stageA_model=stageA_model0,
            stageA_ast=residual_asts[0],
            filepath=filepaths[:D] if D > 1 else filepaths[0],
            Nxvars=Nxvars,
            data_hp=data_hp,
            lm_hp=lm_hp,
            device=device,
            dtype=dtype,
            np_dtype=np_dtype,
            y_op=None,
            y_op_inv=None,
            max_outer_iters=de_max_outer,
            epochs_stageB=de_epochs,
            score_tol=de_score_tol,
            verbose=bool(getattr(args, "verbose", True)),
            disabled_patterns=disabled_patterns or [],
            use_stageA_reuse=True,
            datasets_override=datasets_override,
            reuse_override=reuses if D > 1 else reuses[0],
            units_spec=de_units_spec,
            enforce_units=bool(getattr(args, "enforce_units", False)),
            use_factorized_search=bool(getattr(args, "use_factorized_search", False)),
            factorized_search_hp=factorized_search_hp,
            atom_factory=atom_factories if D > 1 else atom_factories[0],
        )

        # Robust printing for DE feature atoms (u, du, d2u).
        if D > 1 and getattr(de_stageB_state, "models", None) is not None:
            de_phi_expr_strs = []
            try:
                from nestynet_sr.sr_search.stageB.engine import StageBState

                for i in range(D):
                    st_i = StageBState(
                        root=de_stageB_state.root,
                        model=de_stageB_state.models[i],
                        reuse=de_stageB_state.reuses[i],
                        val_loss=float(de_stageB_state.val_losses[i]),
                    )
                    de_phi_expr_strs.append(pretty_print_state(st_i, sig=16))
            except Exception:
                de_phi_expr_strs = None

    # -------------------------
    # Write outputs
    # -------------------------
    de_pkl = os.path.join(results_dir, f"{base_filename}_de.pkl")
    de_human = os.path.join(results_dir, f"{base_filename}_de.human")

    try:
        with open(de_pkl, "wb") as f:
            pickle.dump(
                {
                    "y_space": de_y_name,
                    "de_cfg": asdict(cfg),
                    "dataset_ids": dataset_ids,
                    "order": int(de_res_multi.order),
                    "x_axis": int(de_res_multi.x_axis),
                    "term_asts": de_res_multi.term_asts,
                    "coeffs": de_res_multi.coeffs.detach().cpu(),
                    "rms_train": de_res_multi.rms_train,
                    "rms_val": de_res_multi.rms_val,
                    "residual_asts": residual_asts,
                    "eqn_raw": eqn_raw,
                    "class_de": class_de_info,
                    "units": _make_json_serializable(de_units_spec) if de_units_spec is not None else None,
                    "stageB": _de_stageB_report_payload(
                        de_stageB_state,
                        phi_expr_strs=de_phi_expr_strs,
                    ),
                },
                f,
            )
    except Exception as e:
        print(f"[DE/SR] Warning: failed to write {de_pkl}: {e}")

    try:
        with open(de_human, "w") as f:
            f.write("Implicit DE (residual form)\n")
            f.write(f"y_space = {de_y_name}\n")
            f.write(f"order = {int(de_res_multi.order)}, x_axis = {int(de_res_multi.x_axis)}\n")
            f.write("\n")
            for i in range(D):
                f.write(f"Dataset {i}: {dataset_ids[i]}\n")
                if eqn_raw[i] is not None:
                    f.write(f"  Raw: {eqn_raw[i]}\n")
                if de_stageB_state is not None and de_phi_expr_strs is not None:
                    f.write(f"  StageB residual: ({de_phi_expr_strs[i]}) = 0\n")
                elif de_stageB_state is not None and i == 0:
                    f.write(f"  StageB residual: ({de_stageB_state.phi_expr_str}) = 0\n")
                if de_stageB_state is not None:
                    metadata_by_dataset = getattr(
                        de_stageB_state,
                        "coefficient_metadata_by_dataset",
                        None,
                    )
                    metadata_i = (
                        metadata_by_dataset[i]
                        if isinstance(metadata_by_dataset, (list, tuple))
                        and i < len(metadata_by_dataset)
                        else getattr(
                            de_stageB_state,
                            "coefficient_metadata",
                            None,
                        )
                    )
                    for definition in _coefficient_definitions_for_human(
                        metadata_i
                    ):
                        f.write(f"  Coefficient: {definition}\n")
                f.write("\n")
            if class_de_info is not None:
                f.write(f"Class-DE promoted tags (CV≤{class_de_info['cv_threshold']}): {class_de_info['promoted_tags']}\n")
    except Exception as e:
        print(f"[DE/SR] Warning: failed to write {de_human}: {e}")

    # Return JSON-ish payload for the main report.
    out = {
        "enabled": True,
        "y_space": de_y_name,
        "order": int(de_res_multi.order),
        "x_axis": int(de_res_multi.x_axis),
        "dataset_ids": dataset_ids,
        "cfg": asdict(cfg),
        "rms_train": [float(x) for x in de_res_multi.rms_train],
        "rms_val": [float(x) for x in de_res_multi.rms_val] if de_res_multi.rms_val is not None else None,
        "eqn_raw": eqn_raw,
        "class_de": class_de_info,
        "stageB": None,
        "artifacts": {
            "pkl": de_pkl,
            "human": de_human,
        },
    }
    out["stageB"] = _de_stageB_report_payload(
        de_stageB_state,
        phi_expr_strs=de_phi_expr_strs,
    )
    return out
