# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Guarded final Pareto-polish helpers for ``nestynet_sr.run_SR``."""

from __future__ import annotations

import argparse
import math
import os
import pathlib

import numpy as np

from nestynet_sr.sr_core.coefficient_metadata import (
    CoefficientMetadataError,
    coefficient_symbol_values,
    normalize_coefficient_metadata,
    normalize_coefficient_metadata_by_dataset,
)
from nestynet_sr.run_sr_reports import (
    _make_json_serializable,
    _polish_record_for_report,
    _protect_exact_stageB_seed_in_final_polish,
    _select_final_polish_seed,
)
from nestynet_sr.sr_core.sympy_units import check_sympy_units


def _certify_final_polish_recommendation_units(
    result,
    *,
    variable_names,
    units_spec,
    require_valid: bool,
):
    """Certify the post-snap recommendation and fail closed when required."""

    rec = getattr(result, "recommended", None)
    if rec is None:
        return None
    certificate = check_sympy_units(
        rec.expr,
        variable_names,
        units_spec,
        expression_space="y",
    ).to_dict()
    valid = bool(
        certificate.get("checked") is True
        and certificate.get("valid") is True
    )
    if require_valid and not valid:
        reason = str(
            certificate.get("reason")
            or "final-polish recommendation lacks a checked-valid unit certificate"
        )
        try:
            rec.is_recommended = False
        except Exception:
            pass
        result.recommended = None
        result.selection_status = "no_safe_unit_valid_replacement"
        result.selection_reason = reason
        warning = f"final recommendation rejected by unit certification: {reason}"
        try:
            if warning not in result.warnings:
                result.warnings.append(warning)
        except Exception:
            pass
    return certificate


def _stageB_data_for_final_polish_worker(stageB_data):
    """Return the small Stage-B payload needed by the final-polish worker."""
    if stageB_data is None:
        return None
    keep = (
        "phi_expr_str",
        "phi_expr_raw_str",
        "y_expr_str",
        "y_expr_raw_str",
        "y_selected",
        "sympy_meta",
        "coefficient_metadata",
        "coefficient_metadata_by_dataset",
        "dataset_ids",
        "candidate_metrics",
        "num_nn_atoms",
    )
    payload = {key: stageB_data.get(key) for key in keep if key in stageB_data}
    return _make_json_serializable(payload)


def _run_final_pareto_polish_impl(
    *,
    args,
    filepath,
    filepaths,
    report_path,
    results_dir,
    base_filename,
    stageB_data,
    seed,
    units_payload=None,
    noise_sigma_y=None,
):
    if isinstance(args, dict):
        args = argparse.Namespace(**args)
    if not bool(getattr(args, "final_polish", True)):
        return {"enabled": False, "status": "disabled"}
    if stageB_data is None:
        return {"enabled": True, "status": "skipped", "reason": "no Stage B result"}
    if filepaths is not None and len(list(filepaths)) > 1:
        bundles = stageB_data.get("coefficient_metadata_by_dataset")
        if bundles is not None:
            try:
                normalize_coefficient_metadata_by_dataset(
                    bundles,
                    primary_payload=stageB_data.get("coefficient_metadata"),
                    expected_count=len(list(filepaths)),
                    expected_dataset_ids=stageB_data.get("dataset_ids"),
                )
            except CoefficientMetadataError as exc:
                return {
                    "enabled": True,
                    "status": "skipped",
                    "reason": f"invalid per-dataset coefficient metadata: {exc.reason}",
                    "coefficient_metadata_error": {
                        "code": exc.code,
                        "reason": exc.reason,
                    },
                }
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "multi-dataset polishing is not supported yet",
        }

    seed_expr, seed_space, reason = _select_final_polish_seed(stageB_data)
    if seed_expr is None:
        return {"enabled": True, "status": "skipped", "reason": reason or "no seed expression"}

    out_dir = (
        str(getattr(args, "final_polish_out_dir", None))
        if getattr(args, "final_polish_out_dir", None)
        else os.path.join(results_dir, f"{base_filename}_polish")
    )
    max_rows = int(getattr(args, "final_polish_max_rows", 10000) or 0)
    if max_rows <= 0:
        max_rows = None
    val_fraction = float(getattr(args, "final_polish_val_fraction", 0.2) or 0.2)
    max_candidates = int(getattr(args, "final_polish_max_candidates", 256) or 256)
    max_seconds = float(getattr(args, "final_polish_max_seconds", 30.0) or 30.0)
    seed_i = int(seed if seed is not None else 1234)

    summary = {
        "enabled": True,
        "status": "running",
        "seed_expr": seed_expr,
        "seed_space": seed_space,
        "out_dir": out_dir,
        "max_rows": max_rows,
        "val_fraction": val_fraction,
        "max_candidates": max_candidates,
    }
    try:
        from nestynet_sr.equation_polisher import (
            PolishConfig,
            apply_full_dataset_snap_adjudication,
            load_artifact_hints,
            load_csv_data,
            polish_expression,
            write_outputs,
        )

        decisions_path = os.path.join(results_dir, f"{base_filename}.decisions.json")
        path_file = os.path.join(results_dir, f"{base_filename}.path")
        final_human = os.path.join(results_dir, f"{base_filename}_final.human")
        hints = load_artifact_hints(
            report_json=report_path if os.path.exists(str(report_path)) else None,
            decisions_json=decisions_path if os.path.exists(decisions_path) else None,
            path_file=path_file if os.path.exists(path_file) else None,
            final_human=final_human if os.path.exists(final_human) else None,
        )
        Xtr, ytr, Xva, yva, names = load_csv_data(
            filepath,
            val_fraction=val_fraction,
            seed=seed_i,
            max_rows=max_rows,
        )
        units_spec = None
        if units_payload is not None and bool(getattr(args, "enforce_units", False)):
            from nestynet_sr.sr_core.units import UnitsSpec

            units_spec = UnitsSpec(
                unit_system=units_payload["unit_system"],
                x_dims=units_payload["x_dims"],
                y_dim=units_payload["y_dim"],
                y_transform_name="identity",
                policy=str(getattr(args, "units_policy", "free_const_only")),
                nn_semantics=str(getattr(args, "nn_units_semantics", "unknown")),
                free_const_dims=units_payload.get("free_const_dims", {}),
                free_const_scope=units_payload.get("free_const_scope", {}),
                fixed_const_dims=units_payload.get("fixed_const_dims", {}),
                fixed_const_values=units_payload.get("fixed_const_values", {}),
                fixed_const_mode=units_payload.get("fixed_const_mode", "strict"),
            )
            summary["units_enforced"] = True
        else:
            summary["units_enforced"] = False
        coefficient_metadata_payload = stageB_data.get("coefficient_metadata")
        if coefficient_metadata_payload is None:
            coefficient_metadata_payload = getattr(
                hints, "coefficient_metadata", None
            )
        coefficient_metadata = normalize_coefficient_metadata(
            coefficient_metadata_payload,
            variable_names=names,
            require_values=True,
            units_spec=units_spec,
        )
        coefficient_values = coefficient_symbol_values(
            coefficient_metadata,
            variable_names=names,
            units_spec=units_spec,
        )
        summary["coefficient_metadata"] = _make_json_serializable(
            coefficient_metadata
        )
        noise_loss_equiv_abs_floor = 0.0
        noise_floor_raw_polish = 0.0
        try:
            sigma = float(noise_sigma_y)
            n_eff = int(np.asarray(yva).reshape(-1).size)
            if math.isfinite(sigma) and sigma > 0.0 and n_eff > 0:
                noise_floor_raw_polish = float(sigma * sigma)
                noise_loss_equiv_abs_floor = float(
                    noise_floor_raw_polish * math.sqrt(2.0 / float(n_eff))
                )
        except Exception:
            noise_loss_equiv_abs_floor = 0.0
            noise_floor_raw_polish = 0.0
        cfg = PolishConfig(
            max_candidates=max_candidates,
            max_seconds=max_seconds,
            val_fraction=val_fraction,
            seed=seed_i,
            noise_floor_raw=float(noise_floor_raw_polish),
            loss_equiv_abs_floor=max(
                float(getattr(PolishConfig(), "loss_equiv_abs_floor", 1.0e-24)),
                float(noise_loss_equiv_abs_floor),
            ),
            symbol_values=coefficient_values,
            enable_drop_addend_refit=bool(
                getattr(args, "final_polish_drop_addend_refit", True)
            ),
            drop_refit_site_rel_ratio_max=float(
                getattr(args, "final_polish_drop_refit_rel_ratio", 5.0e-2)
            ),
        )
        if noise_loss_equiv_abs_floor > 0.0:
            summary["noise_loss_equiv_abs_floor"] = float(noise_loss_equiv_abs_floor)
        result = polish_expression(
            seed_expr,
            Xtr,
            ytr,
            Xva,
            yva,
            variable_names=names,
            artifact_hints=hints,
            units_spec=units_spec,
            config=cfg,
        )
        result, protect_reason = _protect_exact_stageB_seed_in_final_polish(
            result,
            cfg,
            stageB_data,
        )
        if protect_reason:
            result.warnings.append(protect_reason)
        statistical_proposal_only = bool(getattr(args, "stat_selection", False))
        summary["proposal_only"] = statistical_proposal_only
        summary["legacy_recommendation_authoritative"] = not statistical_proposal_only
        full_snap_enabled = bool(getattr(args, "final_polish_full_dataset_snap", True)) and not statistical_proposal_only
        if full_snap_enabled and noise_floor_raw_polish > 0.0:
            try:
                Xtr_full, ytr_full, Xva_full, yva_full, names_full = load_csv_data(
                    filepath,
                    val_fraction=val_fraction,
                    seed=seed_i,
                    max_rows=None,
                )
                X_full = np.vstack([Xtr_full, Xva_full])
                y_full = np.concatenate([ytr_full, yva_full])
                result, full_snap_summary = apply_full_dataset_snap_adjudication(
                    result,
                    X_full,
                    y_full,
                    variable_names=names_full or names,
                    config=cfg,
                    units_spec=units_spec,
                )
                summary["full_dataset_snap"] = _make_json_serializable(full_snap_summary)
            except Exception as e_full_snap:
                summary["full_dataset_snap"] = {
                    "enabled": True,
                    "status": "error",
                    "error": str(e_full_snap),
                }
                try:
                    result.warnings.append(
                        f"full-dataset snap adjudication failed: {e_full_snap}"
                    )
                except Exception:
                    pass
        elif full_snap_enabled:
            summary["full_dataset_snap"] = {
                "enabled": True,
                "status": "skipped",
                "reason": "noise floor not active",
            }
        else:
            summary["full_dataset_snap"] = {"enabled": False, "status": "disabled"}
        recommendation_unit_certificate = _certify_final_polish_recommendation_units(
            result,
            variable_names=names,
            units_spec=units_spec,
            require_valid=bool(getattr(args, "enforce_units", False))
            or str(seed_space).endswith("_diagnostic_unit_invalid"),
        )
        write_outputs(result, out_dir)
        selection_status = str(
            getattr(result, "selection_status", "selected") or "selected"
        )
        selection_reason = getattr(result, "selection_reason", None)
        selection_succeeded = bool(
            result.recommended is not None and selection_status == "selected"
        )
        if not selection_succeeded and selection_status == "selected":
            selection_status = "no_safe_unit_valid_replacement"
            selection_reason = selection_reason or (
                "final polisher produced no safely promotable recommendation"
            )
        summary.update(
            {
                "status": "success" if selection_succeeded else selection_status,
                "reason": None if selection_succeeded else selection_reason,
                "selection_status": selection_status,
                "selection_reason": selection_reason,
                "needs_escalation": not selection_succeeded,
                "escalation_reason": (
                    None
                    if selection_succeeded
                    else "final_polish_no_safe_unit_valid_replacement"
                ),
                "n_candidates": len(result.all_candidates),
                "all_candidates": [
                    _polish_record_for_report(r) for r in result.all_candidates
                ],
                "n_strict_pareto": len(result.strict_pareto),
                "n_epsilon_pareto": len(result.epsilon_pareto),
                "recommended": _polish_record_for_report(result.recommended),
                "seed_baseline": _polish_record_for_report(result.seed_baseline),
                "seed_units_ok": result.seed_units_ok,
                "seed_units_reason": result.seed_units_reason,
                "strict_pareto": [
                    _polish_record_for_report(r) for r in result.strict_pareto
                ],
                "epsilon_pareto": [
                    _polish_record_for_report(r) for r in result.epsilon_pareto
                ],
                "warnings": list(result.warnings),
                "frontier_json": os.path.join(out_dir, "frontier.json"),
                "frontier_csv": os.path.join(out_dir, "frontier.csv"),
                "rewrite_trace": os.path.join(out_dir, "rewrite_trace.txt"),
            }
        )
        summary["unit_admissibility"] = _make_json_serializable(
            recommendation_unit_certificate
        )
        if isinstance(summary.get("recommended"), dict):
            summary["recommended"]["unit_admissibility"] = (
                _make_json_serializable(recommendation_unit_certificate)
            )
        seed_unit_certificate = check_sympy_units(
            seed_expr,
            names,
            units_spec,
            expression_space="y",
        ).to_dict()
        summary["seed_unit_admissibility"] = _make_json_serializable(
            seed_unit_certificate
        )
        if isinstance(summary.get("seed_baseline"), dict):
            summary["seed_baseline"]["unit_admissibility"] = (
                _make_json_serializable(seed_unit_certificate)
            )
        for group_name in ("all_candidates", "strict_pareto", "epsilon_pareto"):
            for record in summary.get(group_name) or []:
                if not isinstance(record, dict) or not record.get("expr"):
                    continue
                record["unit_admissibility"] = _make_json_serializable(
                    check_sympy_units(
                        record["expr"],
                        names,
                        units_spec,
                        expression_space="y",
                    ).to_dict()
                )

        rec = result.recommended
        if rec is not None:
            try:
                from nestynet_sr.sr_search.truth_eval import evaluate_canary

                truth_kwargs = {
                    "dataset_stem": str(base_filename),
                    "discovered_expr_str": rec.expr,
                    "verbose": False,
                }
                if coefficient_values:
                    truth_kwargs["symbol_values"] = coefficient_values
                truth = evaluate_canary(**truth_kwargs)
                if truth is not None:
                    summary["truth_eval"] = _make_json_serializable(truth)
            except Exception as e_truth:
                summary["truth_eval_error"] = str(e_truth)
    except Exception as e:
        import traceback

        summary.update(
            {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(limit=8),
            }
        )
        print(f"[FinalPolish] Warning: final Pareto polish failed: {e}")
    return summary


def _run_final_pareto_polish(
    *,
    args,
    filepath,
    filepaths,
    report_path,
    results_dir,
    base_filename,
    stageB_data,
    seed,
    units_payload=None,
    noise_sigma_y=None,
):
    if not bool(getattr(args, "final_polish", True)):
        return {"enabled": False, "status": "disabled"}
    if stageB_data is None:
        return {"enabled": True, "status": "skipped", "reason": "no Stage B result"}
    if filepaths is not None and len(list(filepaths)) > 1:
        bundles = stageB_data.get("coefficient_metadata_by_dataset")
        if bundles is not None:
            try:
                normalize_coefficient_metadata_by_dataset(
                    bundles,
                    primary_payload=stageB_data.get("coefficient_metadata"),
                    expected_count=len(list(filepaths)),
                    expected_dataset_ids=stageB_data.get("dataset_ids"),
                )
            except CoefficientMetadataError as exc:
                return {
                    "enabled": True,
                    "status": "skipped",
                    "reason": f"invalid per-dataset coefficient metadata: {exc.reason}",
                    "coefficient_metadata_error": {
                        "code": exc.code,
                        "reason": exc.reason,
                    },
                }
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "multi-dataset polishing is not supported yet",
        }

    worker_stageB_data = _stageB_data_for_final_polish_worker(stageB_data)
    seed_expr, _, reason = _select_final_polish_seed(worker_stageB_data)
    if seed_expr is None:
        return {"enabled": True, "status": "skipped", "reason": reason or "no seed expression"}

    if not bool(getattr(args, "final_polish_subprocess", True)):
        return _run_final_pareto_polish_impl(
            args=args,
            filepath=filepath,
            filepaths=filepaths,
            report_path=report_path,
            results_dir=results_dir,
            base_filename=base_filename,
            stageB_data=worker_stageB_data,
            seed=seed,
            units_payload=units_payload,
            noise_sigma_y=noise_sigma_y,
        )

    max_seconds = float(getattr(args, "final_polish_worker_max_seconds", 300.0) or 300.0)
    mem_fraction = float(getattr(args, "final_polish_worker_mem_fraction", 0.20) or 0.20)
    print(
        "[FinalPolish] Running in guarded worker "
        f"(max_seconds={max_seconds:.1f}, mem_fraction={mem_fraction:.3g})."
    )
    try:
        from nestynet_sr.sr_search.postprocess_guard import run_guarded_function

        outcome = run_guarded_function(
            "nestynet_sr.run_sr_final_polish:_run_final_pareto_polish_impl",
            kwargs={
                "args": dict(vars(args)),
                "filepath": filepath,
                "filepaths": filepaths,
                "report_path": report_path,
                "results_dir": results_dir,
                "base_filename": base_filename,
                "stageB_data": worker_stageB_data,
                "seed": seed,
                # ``run_guarded_function`` uses multiprocessing's spawn/pickle
                # transport, so preserve UnitSystem and exact Fraction-valued
                # dimensions.  The report JSON serializer stringifies those
                # objects and loses the active dimensional basis in the worker.
                "units_payload": units_payload,
                "noise_sigma_y": noise_sigma_y,
            },
            max_seconds=max_seconds,
            mem_fraction=mem_fraction,
            label="final_polish",
        )
    except Exception as exc:
        return {
            "enabled": True,
            "status": "safe_failed",
            "reason": "guarded final-polish setup failed",
            "error": str(exc),
            "guarded_subprocess": True,
        }

    if outcome.get("ok") and isinstance(outcome.get("result"), dict):
        summary = dict(outcome["result"])
        summary["guarded_subprocess"] = True
        summary["guard_memory_limit_bytes"] = outcome.get("memory_limit_bytes")
        summary["guard_max_seconds"] = outcome.get("max_seconds")
        return summary

    reason = outcome.get("reason") or outcome.get("error") or outcome.get("status")
    print(f"[FinalPolish] Guarded worker failed safely: {reason}")
    return {
        "enabled": True,
        "status": "safe_failed",
        "reason": reason,
        "error": outcome.get("error"),
        "error_type": outcome.get("error_type"),
        "returncode": outcome.get("returncode"),
        "guarded_subprocess": True,
        "guard_memory_limit_bytes": outcome.get("memory_limit_bytes"),
        "guard_max_seconds": outcome.get("max_seconds"),
    }
