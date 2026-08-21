# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Reporting and JSON serialization helpers for ``nestynet_sr.run_SR``."""

from __future__ import annotations

import json
import math
import os
import platform
import re
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

import torch

from nestynet_sr.campaign_escalation import report_campaign_outcome
from nestynet_sr.sr_core.coefficient_metadata import (
    CoefficientMetadataError,
    coefficient_symbol_values,
    coefficient_symbol_values_for_expression,
    normalize_coefficient_metadata,
    normalize_coefficient_metadata_by_dataset,
)


def _git_repo_provenance(path) -> dict:
    """Return commit and dirty state for the repository containing ``path``."""
    cwd = Path(path).resolve()
    if cwd.is_file():
        cwd = cwd.parent
    out = {"git_hash": None, "git_dirty": None, "repo_root": None}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return out
        repo_root = result.stdout.strip()
        out["repo_root"] = repo_root
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if head.returncode == 0:
            out["git_hash"] = head.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if status.returncode == 0:
            out["git_dirty"] = bool(status.stdout.strip())
    except Exception:
        pass
    return out


def get_git_hash():
    """Get the NestyNet_SR commit hash independently of the process cwd."""
    repo = _git_repo_provenance(Path(__file__).resolve().parents[1])
    return repo.get("git_hash")


def _distribution_version(name: str):
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


def _module_source_provenance(module, distribution_name: str) -> dict:
    module_file = getattr(module, "__file__", None)
    repo = _git_repo_provenance(module_file) if module_file else {
        "git_hash": None,
        "git_dirty": None,
        "repo_root": None,
    }
    return {
        "version": (
            getattr(module, "__version__", None)
            or _distribution_version(distribution_name)
        ),
        "module_file": str(Path(module_file).resolve()) if module_file else None,
        **repo,
    }


def _build_run_provenance(*, seed: int, device, dtype) -> dict:
    """Collect the small, allow-listed runtime fingerprint needed to replay a run."""
    import nestynet
    import nestynet_sr

    env_keys = (
        "PYTHONHASHSEED",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "CUBLAS_WORKSPACE_CONFIG",
    )
    mps_backend = getattr(torch.backends, "mps", None)
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    cuda_available = bool(torch.cuda.is_available())
    return {
        "schema_version": 1,
        "source": {
            "nestynet_sr": _module_source_provenance(nestynet_sr, "nestynet-sr"),
            "nestynet": _module_source_provenance(nestynet, "nestynet"),
        },
        "dependencies": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": _distribution_version("numpy"),
            "scipy": _distribution_version("scipy"),
            "pandas": _distribution_version("pandas"),
            "sympy": _distribution_version("sympy"),
        },
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "device": str(device),
            "dtype": str(dtype),
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        },
        "backend": {
            "cuda_available": cuda_available,
            "cuda_version": getattr(torch.version, "cuda", None),
            "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
            "cudnn_version": (
                torch.backends.cudnn.version() if cudnn_backend is not None else None
            ),
            "cudnn_deterministic": (
                bool(cudnn_backend.deterministic) if cudnn_backend is not None else None
            ),
            "cudnn_benchmark": (
                bool(cudnn_backend.benchmark) if cudnn_backend is not None else None
            ),
            "mps_available": (
                bool(mps_backend.is_available()) if mps_backend is not None else False
            ),
            "mps_built": (
                bool(mps_backend.is_built()) if mps_backend is not None else False
            ),
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
        },
        "rng": {
            "reported_seed": int(seed),
            "torch_initial_seed": int(torch.initial_seed()),
            "repeatable_seed_enabled": bool(int(seed) >= 0),
            "seeded_streams": (
                ["python_random", "numpy_legacy", "numpy_generator", "torch"]
                if int(seed) >= 0
                else []
            ),
            "stageB_candidate_seed_policy": "sha256(run_seed, candidate_ast, init_key, start)",
        },
        "environment": {key: os.environ.get(key) for key in env_keys},
    }


def _make_json_serializable(obj):
    """
    Convert an object to a JSON-serializable format.

    Handles common non-serializable types like AST nodes, numpy types, etc.
    """
    if obj is None:
        return None

    # Handle numpy types
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()

    # Handle torch tensors
    if hasattr(obj, "detach"):
        return obj.detach().cpu().numpy().tolist()

    # Handle lists and tuples recursively
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]

    # Handle dicts recursively
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}

    # Handle basic JSON-serializable types
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # For anything else, convert to string
    return str(obj)


def _refresh_final_selection_truth_eval(
    report_path: str | Path,
    *,
    source: str,
    preserve_as: str,
    verbose: bool = False,
) -> dict | None:
    """Evaluate the authoritative final expression and update its provenance."""
    path = Path(report_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    final_selection = report.get("final_selection")
    if not isinstance(final_selection, dict):
        return None
    if (
        final_selection.get("applied") is False
        or final_selection.get("eligible_for_success") is False
    ):
        return None
    expr = final_selection.get("expr")
    if not isinstance(expr, str) or not expr.strip():
        return None

    from nestynet_sr.sr_search.truth_eval import evaluate_canary

    metadata = report.get("metadata") or {}
    dataset = metadata.get("dataset")
    if not dataset:
        return None
    final_polish = report.get("final_polish") or {}
    stagec = report.get("stageC") or {}
    stageb = report.get("stageB") or {}
    coefficient_metadata = next(
        (
            payload
            for payload in (
                final_selection.get("coefficient_metadata"),
                final_polish.get("coefficient_metadata"),
                stagec.get("coefficient_metadata"),
                stageb.get("coefficient_metadata"),
            )
            if payload is not None
        ),
        None,
    )
    try:
        coefficient_values = coefficient_symbol_values(coefficient_metadata)
        truth_kwargs = {
            "dataset_stem": Path(str(dataset)).stem,
            "discovered_expr_str": expr,
            "verbose": verbose,
        }
        if coefficient_values:
            truth_kwargs["symbol_values"] = coefficient_values
        truth_result = evaluate_canary(**truth_kwargs)
    except Exception as exc:
        truth_result = {
            "success": False,
            "skipped": True,
            "reason": f"{source}_truth_eval_error",
            "error_message": str(exc),
        }
    if truth_result is None:
        return None

    final_truth = _make_json_serializable(truth_result)
    final_truth = dict(final_truth)
    final_truth["source"] = source
    final_truth["expr"] = expr
    old_truth = report.get("truth_eval")
    if isinstance(old_truth, dict) and preserve_as not in report:
        report[preserve_as] = old_truth
    final_selection = dict(final_selection)
    final_selection["truth_eval"] = final_truth
    report["final_selection"] = final_selection
    report["truth_eval"] = final_truth
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return final_truth


def write_json_report(
    filepath: str,
    report_path: str,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    walltime: float,
    filepaths: list = None,
    stageA_data: dict = None,
    stageB_data: dict = None,
    de_data: dict = None,
    class_sr_summary: dict = None,
    discovery_summary: dict = None,
    enable_truth_eval: bool = True,
    verbose: bool = False,
    fast_mode: bool = False,
    disabled_patterns: list = None,
    factorized_search_enabled: bool | None = None,
    factorized_search_config: dict | None = None,
):
    """
    Write a structured JSON report for the SR run.

    Parameters
    ----------
    filepath : str
        Path to the input CSV file
    report_path : str
        Path to write the JSON report
    device : torch.device
        Device used for computation
    dtype : torch.dtype
        Data type used for computation
    seed : int
        Random seed used
    walltime : float
        Total walltime in hours
    stageA_data : dict, optional
        Dictionary with Stage A results (y_op_name, ast, val_loss, rest_add, rest_mult)
    stageB_data : dict, optional
        Dictionary with Stage B results (phi_expr_str, y_expr_str, sympy_meta, val_loss, ast)
    enable_truth_eval : bool
        Whether to attempt ground-truth evaluation for known canaries
    verbose : bool
        Print diagnostic information during truth evaluation
    """
    report = {
        "metadata": {
            "dataset": filepath,
            "datasets": list(filepaths) if filepaths is not None else [filepath],
            "git_hash": get_git_hash(),
            "device": str(device),
            "dtype": str(dtype),
            "seed": seed,
            "walltime_hours": walltime,
            "fast_mode": fast_mode,
            "disabled_patterns": disabled_patterns if disabled_patterns else [],
            "provenance": _build_run_provenance(
                seed=seed,
                device=device,
                dtype=dtype,
            ),
        }
    }
    if factorized_search_enabled is not None:
        report["metadata"]["factorized_search_enabled"] = bool(factorized_search_enabled)
    if factorized_search_config is not None:
        report["metadata"]["factorized_search_config"] = _make_json_serializable(
            factorized_search_config
        )

    if stageA_data is not None:
        from nestynet_sr.sr_core import ast_to_human_readable

        ast = stageA_data.get("ast")
        x_transform_map = stageA_data.get("x_transform_map")

        # Convert AST to JSON-serializable format (include x-transforms in display)
        ast_human = ast_to_human_readable(ast, x_transform_map) if ast is not None else None

        report["stageA"] = {
            "y_transform": stageA_data.get("y_op_name"),
            "status": stageA_data.get("stageA_status"),
            "ast_human": ast_human,
            "x_transform": _make_json_serializable(x_transform_map),
            "val_loss": _make_json_serializable(stageA_data.get("val_loss")),
            "val_losses": _make_json_serializable(stageA_data.get("val_losses")),
            "val_loss_agg_mode": _make_json_serializable(stageA_data.get("val_loss_agg_mode")),
            "val_loss_agg_weights": _make_json_serializable(stageA_data.get("val_loss_agg_weights")),
            "dataset_ids": _make_json_serializable(stageA_data.get("dataset_ids")),
            "rest_add": _make_json_serializable(stageA_data.get("rest_add")),
            "rest_mult": _make_json_serializable(stageA_data.get("rest_mult")),
            "outer_peel_square": _make_json_serializable(stageA_data.get("outer_peel_square")),
            "outer_peel_ranked": _make_json_serializable(stageA_data.get("outer_peel_ranked")),
            "stageB_virtual_top_names": _make_json_serializable(
                stageA_data.get("stageB_virtual_top_names")
            ),
            "stageB_virtual_portfolio": _make_json_serializable(
                stageA_data.get("stageB_virtual_portfolio")
            ),
            "stageB_y_shortlist_sources": _make_json_serializable(
                stageA_data.get("stageB_y_shortlist_sources")
            ),
            "move_records": _make_json_serializable(
                stageA_data.get("stageA_move_records")
            ),
        }
        if stageA_data.get("coe_stageA_dry_run_summary"):
            report["stageA"]["coe_stageA_dry_run_summary"] = _make_json_serializable(
                stageA_data.get("coe_stageA_dry_run_summary")
            )
        if stageA_data.get("coe_stageA_dry_run_log"):
            report["stageA"]["coe_stageA_dry_run_log"] = _make_json_serializable(
                stageA_data.get("coe_stageA_dry_run_log")
            )
        if stageA_data.get("coe_stageA_exit_audit"):
            report["stageA"]["coe_stageA_exit_audit"] = _make_json_serializable(
                stageA_data.get("coe_stageA_exit_audit")
            )
        if stageA_data.get("coe_stageA_ybranch_committee"):
            report["stageA"]["coe_stageA_ybranch_committee"] = _make_json_serializable(
                stageA_data.get("coe_stageA_ybranch_committee")
            )
        if stageA_data.get("coe_stageA_compound_shortlist"):
            report["stageA"]["coe_stageA_compound_shortlist"] = _make_json_serializable(
                stageA_data.get("coe_stageA_compound_shortlist")
            )
        if stageA_data.get("coe_stageA_fit_tournament_records"):
            report["stageA"]["coe_stageA_fit_tournament_records"] = (
                _make_json_serializable(
                    stageA_data.get("coe_stageA_fit_tournament_records")
                )
            )
        if stageA_data.get("coe_stageA_proposal_reservoir"):
            report["stageA"]["coe_stageA_proposal_reservoir"] = _make_json_serializable(
                stageA_data.get("coe_stageA_proposal_reservoir")
            )
        if stageA_data.get("coe_stageA_materialization"):
            report["stageA"]["coe_stageA_materialization"] = _make_json_serializable(
                stageA_data.get("coe_stageA_materialization")
            )
        if stageA_data.get("coe_stageA_replay_log"):
            report["stageA"]["coe_stageA_replay_log"] = _make_json_serializable(
                stageA_data.get("coe_stageA_replay_log")
            )
        if stageA_data.get("coe_stageA_scout_proposers"):
            report["stageA"]["coe_stageA_scout_proposers"] = _make_json_serializable(
                stageA_data.get("coe_stageA_scout_proposers")
            )
        if stageA_data.get("stageA_provisional_commits"):
            report["stageA"]["provisional_commits"] = _make_json_serializable(
                stageA_data.get("stageA_provisional_commits")
            )
        if stageA_data.get("stageA_rejection_records"):
            report["stageA"]["rejected_transactions"] = _make_json_serializable(
                stageA_data.get("stageA_rejection_records")
            )
        if stageA_data.get("stageA_provisional_confirmation"):
            report["stageA"]["provisional_confirmation"] = _make_json_serializable(
                stageA_data.get("stageA_provisional_confirmation")
            )
        if stageA_data.get("stageA_provisional_guard"):
            report["stageA"]["provisional_guard"] = _make_json_serializable(
                stageA_data.get("stageA_provisional_guard")
            )

    if stageB_data is not None:
        from nestynet_sr.sr_core import ast_to_human_readable

        ast = stageB_data.get("ast")
        # Use stageB x_transform if available, else fall back to stageA
        stageB_x_transform = stageB_data.get("x_transform_map")
        if stageB_x_transform is None and stageA_data is not None:
            stageB_x_transform = stageA_data.get("x_transform_map")

        # Convert AST to JSON-serializable format
        ast_human = ast_to_human_readable(ast, stageB_x_transform) if ast is not None else None

        report["stageB"] = {
            "ast_human": ast_human,
            "val_loss": _make_json_serializable(stageB_data.get("val_loss")),
            "val_losses": _make_json_serializable(stageB_data.get("val_losses")),
            "dataset_ids": _make_json_serializable(stageB_data.get("dataset_ids")),
            "agg_mode": _make_json_serializable(stageB_data.get("agg_mode")),
            "params": _make_json_serializable(stageB_data.get("params")),
            "num_nn_atoms": _make_json_serializable(stageB_data.get("num_nn_atoms")),
            "enabled_patterns": _make_json_serializable(stageB_data.get("enabled_patterns", [])),
            "y_selected": _make_json_serializable(stageB_data.get("y_selected")),
            "y_shortlist_names": _make_json_serializable(stageB_data.get("y_shortlist_names")),
            "y_shortlist_sources": _make_json_serializable(stageB_data.get("y_shortlist_sources")),
            "y_branch_artifacts": _make_json_serializable(stageB_data.get("y_branch_artifacts")),
            "candidate_metrics": _make_json_serializable(
                stageB_data.get("candidate_metrics")
            ),
            "coefficient_metadata": _make_json_serializable(
                stageB_data.get("coefficient_metadata")
            ),
            "coefficient_metadata_by_dataset": _make_json_serializable(
                stageB_data.get("coefficient_metadata_by_dataset")
            ),
            "decision_log_summary": _make_json_serializable(
                stageB_data.get("decision_log_summary")
            ),
        }
        if stageB_data.get("coe_stageB_dry_run_summary"):
            report["stageB"]["coe_stageB_dry_run_summary"] = _make_json_serializable(
                stageB_data.get("coe_stageB_dry_run_summary")
            )
        if stageB_data.get("coe_stageB_dry_run_log"):
            report["stageB"]["coe_stageB_dry_run_log"] = _make_json_serializable(
                stageB_data.get("coe_stageB_dry_run_log")
            )
        if stageB_data.get("coe_stageB_gate_summary"):
            report["stageB"]["coe_stageB_gate_summary"] = _make_json_serializable(
                stageB_data.get("coe_stageB_gate_summary")
            )
        if stageB_data.get("coe_stageB_gate_log"):
            report["stageB"]["coe_stageB_gate_log"] = _make_json_serializable(
                stageB_data.get("coe_stageB_gate_log")
            )
        if stageB_data.get("coe_proposal_reservoir"):
            report["stageB"]["coe_proposal_reservoir"] = _make_json_serializable(
                stageB_data.get("coe_proposal_reservoir")
            )
        if stageB_data.get("coe_scout_proposers"):
            report["stageB"]["coe_scout_proposers"] = _make_json_serializable(
                stageB_data.get("coe_scout_proposers")
            )
        if stageB_data.get("stageA_provisional_guard"):
            report["stageB"]["stageA_provisional_guard"] = _make_json_serializable(
                stageB_data.get("stageA_provisional_guard")
            )

        # Stage C (SymPy simplification)
        # Prefer raw-x expressions when available (these have x-transforms applied)
        phi_expr = stageB_data.get("phi_expr_raw_str") or stageB_data.get("phi_expr_str")
        y_expr = stageB_data.get("y_expr_raw_str") or stageB_data.get("y_expr_str")
        sympy_meta = stageB_data.get("sympy_meta")
        unresolved_symbolic = _stageB_unresolved_symbolic_info(stageB_data)
        if unresolved_symbolic.get("unresolved"):
            sympy_meta = _with_unresolved_stagec_meta(sympy_meta, unresolved_symbolic)
        stagec_verified, stagec_reason = _stagec_expression_is_verified(stageB_data)
        unit_admissibility = (
            sympy_meta.get("unit_admissibility")
            if isinstance(sympy_meta, dict)
            else None
        )
        unit_invalid = bool(
            isinstance(unit_admissibility, dict)
            and unit_admissibility.get("checked") is True
            and unit_admissibility.get("valid") is False
        )

        report["stageC"] = {
            "phi_expr_str": phi_expr,
            "y_expr_str": y_expr,
            "sympy_meta": _make_json_serializable(sympy_meta),
            "symbolic_status": (
                "unresolved_nn"
                if unresolved_symbolic.get("unresolved")
                else "unit_invalid"
                if unit_invalid
                else "uncertified_expression"
                if not stagec_verified
                else "fully_analytic"
            ),
            "certified": bool(stagec_verified),
            "certification_reason": _make_json_serializable(stagec_reason),
            "units_checked": (
                sympy_meta.get("units_checked")
                if isinstance(sympy_meta, dict)
                else None
            ),
            "units_ok": (
                sympy_meta.get("units_ok")
                if isinstance(sympy_meta, dict)
                else None
            ),
            "units_reason": (
                sympy_meta.get("units_reason")
                if isinstance(sympy_meta, dict)
                else None
            ),
            "unit_admissibility": _make_json_serializable(unit_admissibility),
            "coefficient_metadata": _make_json_serializable(
                stageB_data.get("coefficient_metadata")
            ),
            "coefficient_metadata_by_dataset": _make_json_serializable(
                stageB_data.get("coefficient_metadata_by_dataset")
            ),
            "unresolved": _make_json_serializable(unresolved_symbolic),
        }

        # Simplification path (curated user-facing timeline)
        _sp = stageB_data.get("simplification_path", [])
        if _sp:
            report["simplification_path"] = _make_json_serializable(_sp)

    if de_data is not None:
        # DE payload is already mostly JSON-friendly; run it through the
        # serializer to handle tensors/paths.
        report["de"] = _make_json_serializable(de_data)

    if class_sr_summary is not None:
        report["class_sr"] = _make_json_serializable(class_sr_summary)

    if discovery_summary is not None:
        report["discovery"] = _make_json_serializable(discovery_summary)

    # Ground-truth evaluation for known canaries
    if enable_truth_eval and stageB_data is not None:
        import pathlib

        from nestynet_sr.sr_search.truth_eval import evaluate_canary

        # Extract dataset stem from filepath
        dataset_stem = pathlib.Path(filepath).stem

        # Try to evaluate against ground truth
        # Prefer raw-x expressions (with x-transforms applied), then y-space, then phi-space
        discovered_expr = stageB_data.get("y_expr_raw_str")
        if discovered_expr is None:
            discovered_expr = stageB_data.get("y_expr_str")
        if discovered_expr is None:
            discovered_expr = stageB_data.get("phi_expr_raw_str")
        if discovered_expr is None:
            discovered_expr = stageB_data.get("phi_expr_str")

        unresolved_symbolic = _stageB_unresolved_symbolic_info(stageB_data)
        if unresolved_symbolic.get("unresolved"):
            report["truth_eval"] = {
                "success": False,
                "skipped": True,
                "reason": "unresolved_symbolic_leaves",
                "error_message": (
                    "Skipped truth evaluation because the final expression still "
                    "contains NN atoms or leafN(...) placeholders."
                ),
                "unresolved": _make_json_serializable(unresolved_symbolic),
            }
        else:
            stagec_verified, stagec_reason = _stagec_expression_is_verified(stageB_data)
            if not stagec_verified:
                report["truth_eval"] = {
                    "success": False,
                    "skipped": True,
                    "reason": "stagec_expression_uncertified",
                    "error_message": (
                        stagec_reason
                        or "Skipped truth evaluation because Stage C did not certify the final expression."
                    ),
                    "sympy_meta": _make_json_serializable(stageB_data.get("sympy_meta")),
                }
            elif discovered_expr is not None:
                truth_kwargs = {
                    "dataset_stem": dataset_stem,
                    "discovered_expr_str": discovered_expr,
                    "verbose": verbose,
                }
                coefficient_metadata = stageB_data.get("coefficient_metadata")
                if coefficient_metadata is None:
                    sympy_meta = stageB_data.get("sympy_meta")
                    if isinstance(sympy_meta, dict):
                        coefficient_metadata = sympy_meta.get(
                            "coefficient_metadata"
                        )
                coefficient_values = coefficient_symbol_values(coefficient_metadata)
                if coefficient_values:
                    truth_kwargs["symbol_values"] = coefficient_values
                truth_result = evaluate_canary(
                    **truth_kwargs,
                )

                if truth_result is not None:
                    report["truth_eval"] = _make_json_serializable(truth_result)
                    if verbose and truth_result.get("success"):
                        print("\n[Truth Eval] Ground truth comparison:")
                        print(f"  RMSE (abs): {truth_result['rmse_abs']:.3e}")
                        print(f"  RMSE (rel): {truth_result['rmse_rel']:.3e}")
                        print(f"  Max error: {truth_result['max_abs_err']:.3e}")
                        print(f"  Valid points: {truth_result['frac_valid']:.1%}")

        truth_payload = report.get("truth_eval")
        if isinstance(truth_payload, dict):
            truth_payload["source"] = "stageB"
            if discovered_expr is not None:
                truth_payload["expr"] = discovered_expr

    try:
        from nestynet_sr.sr_search.gate_telemetry import drain as _drain_gates
        from nestynet_sr.sr_search.gate_telemetry import summarize as _summarize_gates

        _gate_records = _drain_gates()
        if _gate_records:
            report["gate_telemetry"] = {
                "records": _gate_records,
                "summary": _summarize_gates(_gate_records),
            }
    except Exception:
        pass  # telemetry must never block the report

    try:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote JSON report to {report_path}")
    except Exception as e:
        print(f"\nWarning: failed to write JSON report to {report_path}: {e}")


def has_nn_atoms(ast):
    """Check if AST contains any NN atoms."""
    from nestynet_sr.sr_core import collect_nn_atoms

    nn_atoms = collect_nn_atoms(ast)
    return len(nn_atoms) > 0


_UNRESOLVED_LEAF_CALL_RE = re.compile(r"\bleaf\d+\s*\(")


def _expr_has_unresolved_leaf_call(expr) -> bool:
    """Return True when a user-facing expression still calls leafN(...)."""
    if expr is None:
        return False
    return bool(_UNRESOLVED_LEAF_CALL_RE.search(str(expr)))


def _stageB_unresolved_symbolic_info(stageB_data) -> dict:
    """Detect Stage-B outputs that are still neural/leaf placeholders.

    Stage C may produce diagnostic strings for partially symbolic models, but
    those are not final equations.  This helper is deliberately shared by JSON
    reporting, truth-eval, and final polish seed selection so unresolved
    expressions cannot be accidentally treated as solved downstream.
    """
    if not stageB_data:
        return {"unresolved": False}

    num_nn = 0
    tags = []
    ast_obj = stageB_data.get("ast")
    if ast_obj is not None:
        try:
            from nestynet_sr.sr_core import collect_nn_atoms

            atoms = collect_nn_atoms(ast_obj)
            num_nn = len(atoms)
            tags = [getattr(a, "tag", None) for a in atoms]
        except Exception:
            num_nn = int(stageB_data.get("num_nn_atoms") or 0)
    else:
        try:
            num_nn = int(stageB_data.get("num_nn_atoms") or 0)
        except Exception:
            num_nn = 0

    expr_keys = (
        "y_expr_raw_str",
        "y_expr_str",
        "phi_expr_raw_str",
        "phi_expr_str",
    )
    leaf_expr_keys = [
        key for key in expr_keys if _expr_has_unresolved_leaf_call(stageB_data.get(key))
    ]

    meta = stageB_data.get("sympy_meta")
    meta_reason = None
    meta_unresolved = False
    if isinstance(meta, dict):
        meta_reason = meta.get("reason") or meta.get("kind")
        if (
            meta.get("reason") in {"unresolved_nn_atoms_present", "problem_leaves_present"}
            or meta.get("kind") in {"unresolved_symbolic_leaves", "undefined_functions"}
        ):
            meta_unresolved = True
            num_nn = max(num_nn, int(meta.get("num_nn_atoms") or 0))

    unresolved = bool(num_nn > 0 or leaf_expr_keys or meta_unresolved)
    return {
        "unresolved": unresolved,
        "num_nn_atoms": int(num_nn),
        "nn_tags": [t for t in tags if t is not None],
        "leaf_expr_keys": leaf_expr_keys,
        "meta_reason": meta_reason,
    }


def _with_unresolved_stagec_meta(sympy_meta, unresolved_info: dict) -> dict:
    """Return Stage-C metadata that explicitly marks unresolved symbolic output."""
    meta = dict(sympy_meta) if isinstance(sympy_meta, dict) else {}
    if meta.get("accepted"):
        meta["raw_accepted_before_unresolved_guard"] = True
    reason = "unresolved_nn_atoms_present"
    if (
        int(unresolved_info.get("num_nn_atoms") or 0) <= 0
        and not unresolved_info.get("leaf_expr_keys")
        and unresolved_info.get("meta_reason") == "problem_leaves_present"
    ):
        reason = "problem_leaves_present"
    meta.update(
        {
            "accepted": False,
            "parse_success": meta.get("parse_success", True),
            "reason": reason,
            "kind": "unresolved_symbolic_leaves",
            "num_nn_atoms": int(unresolved_info.get("num_nn_atoms") or 0),
            "leaf_expr_keys": list(unresolved_info.get("leaf_expr_keys") or []),
        }
    )
    if unresolved_info.get("nn_tags"):
        meta["nn_tags"] = list(unresolved_info.get("nn_tags") or [])
    return meta


def _stageA_status_message(stageA_status, separability_success):
    """User-facing Stage A summary without changing controller semantics."""
    if separability_success:
        return None
    status = str(stageA_status or "unresolved")
    if status == "compound_outer_confirmed":
        return "Stage A found full-variable compound compression; outer map confirmed."
    if status == "compound_unresolved":
        return "Stage A found full-variable compound compression; outer map unresolved."
    if status == "phase_hint_confirmed":
        return "Stage-0 phase-coordinate closure confirmed; Stage A surrogate training skipped."
    return "No Stage A separability found."


def _simplification_path_loss_metric(entry: dict):
    """Return (label, value, space) for the loss shown in path reports."""
    if not isinstance(entry, dict):
        return None, None, None
    space = str(entry.get("loss_space") or "").strip().lower()
    y_transform = entry.get("y_transform")
    stage = str(entry.get("stage", "") or "")
    if (
        not space
        and stage in {"B", "C"}
        and y_transform is not None
        and not _is_identity_y_transform(y_transform)
    ):
        space = "phi"
    try:
        if entry.get("mse_raw") is not None and space != "phi":
            return "mse_raw", float(entry.get("mse_raw")), "raw"
        if entry.get("mse_phi") is not None:
            return "mse_phi", float(entry.get("mse_phi")), "phi"
        if entry.get("val_loss") is not None:
            label = "mse_phi" if space == "phi" else "mse_raw"
            return label, float(entry.get("val_loss")), "phi" if space == "phi" else "raw"
    except Exception:
        return None, None, None
    return None, None, None


def _format_simplification_path(path: list) -> str:
    """Render simplification path as a human-readable text block."""
    if not path:
        return ""
    lines = ["=== Simplification Path ===", ""]
    for entry in path:
        step = entry.get("step", "?")
        stage = entry.get("stage", "?")
        action = entry.get("action", "?")
        detail = entry.get("detail")
        detail_str = f" ({detail})" if detail else ""
        lines.append(f"Step {step} [Stage {stage}] {action}{detail_str}")
        expr_phi = entry.get("phi_expression")
        expr_y = entry.get("y_expression")
        if expr_phi is not None and expr_y is not None:
            lines.append(f"  phi(y): {expr_phi}")
            lines.append(f"  y: {expr_y}")
        elif expr_phi is not None:
            lines.append(f"  phi(y): {expr_phi}")
        elif expr_y is not None:
            lines.append(f"  y: {expr_y}")
        else:
            expr = entry.get("expression", "?")
            lines.append(f"  {expr}")
        recipe = entry.get("recipe")
        if recipe:
            lines.append(f"  recipe: {recipe}")
        # Primary metrics line
        metric_label, metric_value, metric_space = _simplification_path_loss_metric(entry)
        mse_eff = entry.get("mse_eff")
        n_params = entry.get("n_params")
        ast_cost = entry.get("ast_cost")
        metrics = []
        if metric_label is not None and metric_value is not None:
            metrics.append(f"{metric_label}={metric_value:.4e}")
        if metric_label != "mse_phi" and entry.get("mse_phi") is not None:
            try:
                phi_val = float(entry.get("mse_phi"))
                if math.isfinite(phi_val):
                    metrics.append(f"mse_phi={phi_val:.4e}")
            except Exception:
                pass
        if mse_eff is not None:
            metrics.append(f"mse_eff={mse_eff:.4e}")
        complexity_total = entry.get("complexity_total")
        if complexity_total is not None:
            try:
                metrics.append(f"complexity_total={float(complexity_total):.3g}")
            except Exception:
                pass
        if n_params is not None:
            metrics.append(f"params={n_params}")
        if ast_cost is not None:
            metrics.append(f"ast_cost={ast_cost:.1f}")
        if metrics:
            lines.append(f"  {' '.join(metrics)}")
        # Threshold line (Stage B only)
        base_loss = entry.get("base_loss")
        threshold = entry.get("threshold")
        if base_loss is not None or threshold is not None:
            thr_parts = []
            base_space = str(entry.get("base_loss_space") or metric_space or "").lower()
            suffix = "_phi" if base_space == "phi" else ""
            if base_loss is not None:
                thr_parts.append(f"base_loss{suffix}={base_loss:.4e}")
            if threshold is not None:
                thr_parts.append(f"threshold{suffix}={threshold:.4e}")
            lines.append(f"  {' '.join(thr_parts)}")
        lines.append("")
    # Final expression
    final_expr = path[-1].get("y_expression") or path[-1].get("expression", "?")
    lines.append(f"=== Final: {final_expr} ===")

    # Accuracy progression summary
    lines.append("")
    lines.append("--- Accuracy Progression ---")
    prev_loss = None
    prev_space = None
    for entry in path:
        stage = entry.get("stage", "?")
        metric_label, loss_value, metric_space = _simplification_path_loss_metric(entry)
        n_params = entry.get("n_params")
        threshold = entry.get("threshold")
        display_label = metric_label or "mse"
        loss_str = f"{loss_value:.4e}" if loss_value is not None else "?"
        params_str = f"{n_params}" if n_params is not None else "?"
        base_space = str(entry.get("base_loss_space") or metric_space or "").lower()
        thr_name = "thr_phi" if base_space == "phi" else "thr"
        thr_str = f"  {thr_name}={threshold:.4e}" if threshold is not None else ""
        improvement = ""
        if prev_loss is not None and loss_value is not None and prev_loss > 0 and loss_value > 0:
            if prev_space is not None and metric_space is not None and prev_space != metric_space:
                improvement = "  (metric changed)"
            else:
                ratio = prev_loss / loss_value
                if ratio >= 1.01:
                    improvement = f"  ({ratio:.1e}x better)"
                elif ratio <= 0.99:
                    improvement = f"  ({1.0/ratio:.1e}x worse)"
                else:
                    improvement = "  (unchanged)"
        prev_loss = loss_value
        prev_space = metric_space
        lines.append(f"  Stage {stage:5s}  {display_label}={loss_str:>12s}  params={params_str:>5s}{thr_str}{improvement}")

    return "\n".join(lines)


def _simplification_path_entry_expression(entry: dict) -> str | None:
    """Return the user-facing expression carried by a simplification-path row."""
    if not isinstance(entry, dict):
        return None
    for key in ("y_expression", "expression", "phi_expression"):
        value = entry.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and text != "?":
            return text
    return None


def _simplification_path_expr_key(expr) -> str:
    if expr is None:
        return ""
    return " ".join(str(expr).strip().split())


def _append_final_simplification_path_state(
    path: list,
    *,
    final_expr,
    val_loss=None,
    n_params=None,
    num_nn_atoms=None,
    detail: str | None = None,
) -> list:
    """Ensure the path's final row agrees with the actual reported final state."""
    final_text = str(final_expr).strip() if final_expr is not None else ""
    if not final_text or final_text == "?":
        return list(path or [])

    out = [dict(row) if isinstance(row, dict) else row for row in (path or [])]
    if out:
        last_expr = _simplification_path_entry_expression(out[-1])
        if _simplification_path_expr_key(last_expr) == _simplification_path_expr_key(final_text):
            return out

    try:
        nn_count = int(num_nn_atoms) if num_nn_atoms is not None else None
    except Exception:
        nn_count = None
    final_detail = detail
    if final_detail is None:
        final_detail = "unresolved Stage-B scaffold" if nn_count and nn_count > 0 else "final Stage-B state"
    if nn_count and nn_count > 0 and "nn_atoms" not in final_detail:
        final_detail = f"{final_detail}, nn_atoms={nn_count}"

    try:
        loss_value = float(val_loss) if val_loss is not None else None
    except Exception:
        loss_value = None

    try:
        param_value = int(n_params) if n_params is not None else None
    except Exception:
        param_value = n_params

    out.append(
        {
            "step": len(out),
            "stage": "Final",
            "action": "reported final state",
            "expression": final_text,
            "val_loss": loss_value,
            "mse_raw": loss_value,
            "mse_eff": None,
            "base_loss": None,
            "threshold": None,
            "n_params": param_value,
            "ast_cost": None,
            "detail": final_detail,
            "num_nn_atoms": nn_count,
        }
    )
    return out


def _format_truth_eval_summary(
    truth_eval: dict | None,
    *,
    title: str = "=== Noiseless Ground Truth Check ===",
) -> str:
    """Render final-expression-vs-truth diagnostics for logs and human reports."""
    if not isinstance(truth_eval, dict):
        return ""

    lines = [title]
    if truth_eval.get("skipped"):
        reason = truth_eval.get("reason") or truth_eval.get("error_message") or "unknown"
        lines.append(f"status: skipped ({reason})")
        return "\n".join(lines)

    if truth_eval.get("success"):
        lines.append("status: success")
        rmse_abs = truth_eval.get("rmse_abs")
        rmse_rel = truth_eval.get("rmse_rel")
        max_abs = truth_eval.get("max_abs_err")
        max_rel = truth_eval.get("max_rel_err")
        if rmse_abs is not None or rmse_rel is not None:
            parts = []
            if rmse_abs is not None:
                parts.append(f"rmse_abs={float(rmse_abs):.4e}")
            if rmse_rel is not None:
                parts.append(f"rmse_rel={float(rmse_rel):.4e}")
            lines.append(" ".join(parts))
        if max_abs is not None or max_rel is not None:
            parts = []
            if max_abs is not None:
                parts.append(f"max_abs_err={float(max_abs):.4e}")
            if max_rel is not None:
                parts.append(f"max_rel_err={float(max_rel):.4e}")
            lines.append(" ".join(parts))
        frac_valid = truth_eval.get("frac_valid")
        n_valid = truth_eval.get("n_valid")
        n_total = truth_eval.get("n_total")
        if frac_valid is not None:
            valid = f"{float(frac_valid):.1%}"
            if n_valid is not None and n_total is not None:
                valid += f" ({int(n_valid)}/{int(n_total)})"
            lines.append(f"valid_points={valid}")
        return "\n".join(lines)

    err = truth_eval.get("error_message") or truth_eval.get("reason") or "unknown error"
    lines.append(f"status: failed ({err})")
    return "\n".join(lines)


def _append_truth_eval_summary_to_file(
    *,
    report_path: str,
    output_path: str,
    title: str = "=== Noiseless Ground Truth Check ===",
) -> None:
    """Append the stored truth-eval summary to a human report, when available."""
    try:
        with open(report_path, "r") as f_report:
            report = json.load(f_report)
        summary = _format_truth_eval_summary(report.get("truth_eval"), title=title)
        if not summary:
            return
        print("\n" + summary)
        if os.path.exists(output_path):
            with open(output_path, "a") as f_out:
                f_out.write("\n" + summary + "\n")
    except Exception as e:
        print(f"Warning: could not append ground-truth check to final.human: {e}")


def _decorate_simplification_path_y_space(
    path: list,
    *,
    y_transform_name,
    phi_expr_str,
    y_expr_str,
    original_y_val_loss=None,
) -> list:
    """Annotate the path with the winning y-transform recipe.

    Stage B works in φ(y)-space for non-identity y-transforms.  The user-facing
    final expression should be in original y-space, while still showing the
    transformed equation that Stage B actually simplified.
    """
    if not path:
        return path
    y_name = (
        "identity"
        if y_transform_name in {None, "", "None", "none", "null"}
        else str(y_transform_name)
    )
    if y_name == "identity" or y_expr_str is None:
        return path

    out = []
    for entry in path:
        e = dict(entry)
        if e.get("stage") in {"B", "C"}:
            e.setdefault("y_transform", y_name)
            if e.get("mse_phi") is None:
                if e.get("mse_raw") is not None:
                    e["mse_phi"] = e.get("mse_raw")
                elif e.get("val_loss") is not None:
                    e["mse_phi"] = e.get("val_loss")
            e.setdefault("loss_space", "phi")
            e.setdefault("base_loss_space", "phi")
            detail = e.get("detail")
            note = f"y_op={y_name}"
            if detail:
                if note not in str(detail):
                    e["detail"] = f"{note}; {detail}"
            else:
                e["detail"] = note
        out.append(e)

    final = out[-1]
    phi_final = phi_expr_str or final.get("expression")
    final["phi_expression"] = phi_final
    final["y_expression"] = y_expr_str
    final["expression"] = y_expr_str
    try:
        raw_loss = float(original_y_val_loss) if original_y_val_loss is not None else None
        if raw_loss is not None and math.isfinite(raw_loss):
            final["mse_raw"] = raw_loss
            final["loss_space"] = "raw"
    except Exception:
        pass
    if phi_final:
        final["recipe"] = f"{y_name}(y) = {phi_final}; y = {y_expr_str}"
    return out


def _clean_final_polish_expr(value):
    """Normalize expression strings and discard textual null sentinels."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"none", "null", "nan", "<none>", "n/a"}:
        return None
    return text


def _is_identity_y_transform(name) -> bool:
    text = str(name or "identity").strip().lower()
    return text in {"", "identity", "none", "null"}


def _stagec_expression_is_verified(stageB_data):
    """Return whether Stage-C pretty/SymPy strings are safe as raw polish seeds."""
    if not stageB_data:
        return False, "no Stage B result"
    coefficient_metadata_by_dataset = stageB_data.get(
        "coefficient_metadata_by_dataset"
    )
    if coefficient_metadata_by_dataset is not None:
        try:
            dataset_ids = stageB_data.get("dataset_ids")
            expected_count = (
                len(dataset_ids) if isinstance(dataset_ids, (list, tuple)) else None
            )
            normalized_by_dataset = normalize_coefficient_metadata_by_dataset(
                coefficient_metadata_by_dataset,
                primary_payload=stageB_data.get("coefficient_metadata"),
                expected_count=expected_count,
                expected_dataset_ids=(
                    dataset_ids
                    if isinstance(dataset_ids, (list, tuple))
                    else None
                ),
            )
        except CoefficientMetadataError as exc:
            return False, f"invalid per-dataset coefficient metadata: {exc.reason}"
        coefficient_metadata = (
            normalized_by_dataset[0]
            if normalized_by_dataset
            else stageB_data.get("coefficient_metadata")
        )
    else:
        coefficient_metadata = stageB_data.get("coefficient_metadata")
    if coefficient_metadata is None:
        sympy_meta = stageB_data.get("sympy_meta")
        if isinstance(sympy_meta, dict):
            coefficient_metadata = sympy_meta.get("coefficient_metadata")
    if coefficient_metadata is not None:
        try:
            normalize_coefficient_metadata(
                coefficient_metadata,
                require_values=True,
            )
        except CoefficientMetadataError as exc:
            return False, f"invalid coefficient metadata: {exc.reason}"
    expression = (
        stageB_data.get("y_expr_raw_str")
        or stageB_data.get("y_expr_str")
        or stageB_data.get("phi_expr_raw_str")
        or stageB_data.get("phi_expr_str")
    )

    def coefficient_coverage_error():
        if expression is None:
            return None
        try:
            coefficient_symbol_values_for_expression(
                coefficient_metadata,
                expression,
            )
        except CoefficientMetadataError as exc:
            return f"coefficient metadata does not cover Stage C: {exc.reason}"
        return None

    unresolved_symbolic = _stageB_unresolved_symbolic_info(stageB_data)
    if unresolved_symbolic.get("unresolved"):
        return False, "Stage C expression contains unresolved NN atoms or leaf functions"
    meta = stageB_data.get("sympy_meta")
    if meta is None:
        coverage_error = coefficient_coverage_error()
        if coverage_error is not None:
            return False, coverage_error
        return True, None
    if not isinstance(meta, dict):
        coverage_error = coefficient_coverage_error()
        if coverage_error is not None:
            return False, coverage_error
        return True, None
    unit_certificate = meta.get("unit_admissibility")
    if isinstance(unit_certificate, dict):
        if str(unit_certificate.get("code") or "") == "expression_unavailable":
            return False, str(
                unit_certificate.get("reason")
                or "Stage C expression is unavailable for dimensional checking"
            )
        if (
            unit_certificate.get("checked") is True
            and unit_certificate.get("valid") is False
        ):
            return False, str(
                unit_certificate.get("reason")
                or "Stage C expression failed dimensional admissibility"
            )
        if (
            bool(meta.get("accepted", False))
            and unit_certificate.get("checked") is True
            and unit_certificate.get("valid") is not True
        ):
            return False, "Stage C unit certificate is incomplete"
    if meta.get("units_checked") is True and meta.get("units_ok") is False:
        return False, str(
            meta.get("units_reason")
            or meta.get("reason")
            or "Stage C expression failed dimensional admissibility"
        )
    if (
        bool(meta.get("accepted", False))
        and meta.get("units_checked") is True
        and meta.get("units_ok") is not True
    ):
        return False, "Stage C unit certificate is incomplete"
    if bool(meta.get("accepted", False)):
        coverage_error = coefficient_coverage_error()
        if coverage_error is not None:
            return False, coverage_error
        return True, None
    kind = meta.get("kind")
    reason = meta.get("reason")
    if kind == "unit_check_expression_unavailable":
        return False, str(
            reason or "Stage C expression is unavailable for dimensional checking"
        )
    if kind == "bad_pretty_print":
        return False, "Stage C pretty-print failed numeric verification"
    if kind == "undefined_functions":
        return False, "Stage C expression contains unresolved leaf functions"
    if reason == "problem_leaves_present":
        return False, "Stage C expression has unresolved problem leaves"
    if meta.get("parse_success") is False:
        return False, "Stage C expression failed to parse"
    return False, "Stage C expression was not accepted"


def _select_final_polish_seed(stageB_data):
    """Choose a raw-y expression for final polishing.

    The equation polisher scores candidates against the original dataset y.
    Therefore phi-space is only safe as a seed when the selected y-transform is
    identity.  Non-identity branches must provide an original-y expression.
    """
    if not stageB_data:
        return None, None, "no Stage B result"
    unresolved_symbolic = _stageB_unresolved_symbolic_info(stageB_data)
    if unresolved_symbolic.get("unresolved"):
        return (
            None,
            None,
            "final expression contains unresolved NN atoms or leaf functions",
        )
    stagec_verified, stagec_reason = _stagec_expression_is_verified(stageB_data)
    sympy_meta = stageB_data.get("sympy_meta")
    sympy_meta = sympy_meta if isinstance(sympy_meta, dict) else {}
    unit_certificate = sympy_meta.get("unit_admissibility")
    diagnostic_unit_invalid_seed = bool(
        sympy_meta.get("numeric_fidelity_ok") is True
        and sympy_meta.get("parse_success") is not False
        and isinstance(unit_certificate, dict)
        and unit_certificate.get("checked") is True
        and unit_certificate.get("valid") is False
    )
    y_expr = (
        _clean_final_polish_expr(stageB_data.get("y_expr_raw_str"))
        or _clean_final_polish_expr(stageB_data.get("y_expr_str"))
    )
    if y_expr is not None:
        if not stagec_verified:
            if diagnostic_unit_invalid_seed:
                return y_expr, "y_diagnostic_unit_invalid", None
            return None, None, stagec_reason or "Stage C expression was not accepted"
        return y_expr, "y", None

    y_name = stageB_data.get("y_selected")
    if _is_identity_y_transform(y_name):
        phi_expr = (
            _clean_final_polish_expr(stageB_data.get("phi_expr_raw_str"))
            or _clean_final_polish_expr(stageB_data.get("phi_expr_str"))
        )
        if phi_expr is not None:
            if not stagec_verified:
                if diagnostic_unit_invalid_seed:
                    return phi_expr, "phi_identity_diagnostic_unit_invalid", None
                return None, None, stagec_reason or "Stage C expression was not accepted"
            return phi_expr, "phi_identity", None
        return None, None, "identity branch has no phi-space expression"

    return (
        None,
        None,
        f"selected y-transform {y_name!r} has no original-y expression",
    )


def _polish_record_for_report(rec):
    if rec is None:
        return None
    data = rec.to_dict() if hasattr(rec, "to_dict") else dict(rec)
    keep = [
        "expr",
        "display_expr",
        "label",
        "train_mse",
        "val_mse",
        "val_mse_se",
        "complexity",
        "n_free_params",
        "n_snapped_consts",
        "frac_valid",
        "full_dataset_mse",
        "full_dataset_mse_se",
        "full_dataset_frac_valid",
        "full_dataset_snap_selected",
        "seed_nrmse",
        "distance_from_seed",
        "is_strict_pareto",
        "is_epsilon_pareto",
        "is_recommended",
        "assumptions",
        "source_hints",
        "rewrite_trace",
        "selection_n_free_params",
    ]
    return {k: data.get(k) for k in keep if k in data}


def _protect_exact_stageB_seed_in_final_polish(result, config, stageB_data):
    """Keep an exact non-generic Stage-B closure if polish finds only a worse form."""
    metrics = (stageB_data or {}).get("candidate_metrics")
    if not isinstance(metrics, dict):
        return result, None
    protected = bool(
        metrics.get("exact_loss", False)
        and metrics.get("full_rewrite", False)
        and not metrics.get("generic_approximant", False)
        and int(metrics.get("accepted_patterns", 0) or 0) > 0
    )
    if not protected or result is None or getattr(result, "recommended", None) is None:
        return result, None
    seed_rec = None
    for rec in getattr(result, "all_candidates", []) or []:
        if str(getattr(rec, "label", "")) == "seed":
            seed_rec = rec
            break
    if seed_rec is None:
        return result, None
    rec = result.recommended
    if rec is seed_rec or str(getattr(rec, "label", "")) == "seed":
        return result, None
    try:
        tol = max(
            float(getattr(config, "epsilon_pareto_k", 0.0))
            * max(
                float(getattr(seed_rec, "val_mse_se", 0.0)),
                float(getattr(rec, "val_mse_se", 0.0)),
            ),
            float(getattr(config, "loss_equiv_abs_floor", 0.0)),
        )
        if float(rec.val_mse) <= float(seed_rec.val_mse) + tol:
            return result, None
    except Exception:
        return result, None
    try:
        rec.is_recommended = False
        seed_rec.is_recommended = True
        result.recommended = seed_rec
    except Exception:
        pass
    return (
        result,
        "kept exact non-generic Stage-B seed because final-polish recommendation "
        "had worse original-y validation loss",
    )


def _format_final_polish_report(summary: dict) -> str:
    lines = ["=== Final Pareto Polish ==="]
    status = str(summary.get("status", "unknown"))
    lines.append(f"status: {status}")
    if summary.get("reason"):
        lines.append(f"reason: {summary.get('reason')}")
    if summary.get("error"):
        lines.append(f"error: {summary.get('error')}")
    if summary.get("needs_escalation"):
        lines.append(
            "needs_escalation: "
            + str(summary.get("escalation_reason") or "unspecified")
        )
    seed_baseline = summary.get("seed_baseline")
    if isinstance(seed_baseline, dict) and seed_baseline.get("val_mse") is not None:
        lines.append(
            "seed_baseline: "
            f"val_mse={float(seed_baseline.get('val_mse')):.4e} "
            f"units_ok={summary.get('seed_units_ok')}"
        )
    if summary.get("seed_units_reason") and summary.get("seed_units_ok") is False:
        lines.append(f"seed_units_reason: {summary.get('seed_units_reason')}")
    rec = summary.get("recommended")
    if isinstance(rec, dict):
        lines.append(f"recommended: {rec.get('expr')}")
        if rec.get("val_mse") is not None:
            lines.append(
                f"val_mse={float(rec.get('val_mse')):.4e} "
                f"complexity={float(rec.get('complexity', float('nan'))):.3g}"
            )
        if rec.get("seed_nrmse") is not None:
            lines.append(f"seed_nrmse={float(rec.get('seed_nrmse')):.4e}")
    full_snap = summary.get("full_dataset_snap")
    if isinstance(full_snap, dict) and full_snap.get("status") in {"selected", "unchanged"}:
        lines.append(
            "full_dataset_snap: "
            f"{full_snap.get('status')} "
            f"full_mse={float(full_snap.get('selected_full_mse', float('nan'))):.4e} "
            f"n={int(full_snap.get('n_full', 0) or 0)}"
        )
    truth_summary = _format_truth_eval_summary(
        summary.get("truth_eval"),
        title="=== Final Polish Ground Truth Check ===",
    )
    if truth_summary:
        lines.append("")
        lines.append(truth_summary)
    if summary.get("out_dir"):
        lines.append(f"frontier_dir: {summary.get('out_dir')}")
    return "\n".join(lines)


def _format_final_selection_report(final_selection: dict | None) -> str:
    """Render the post-adjudication final result as the last user-facing block."""
    if not isinstance(final_selection, dict):
        return ""
    expr = final_selection.get("expr")
    if not expr:
        return ""

    source = str(final_selection.get("source") or "unknown")
    diagnostic_only = final_selection.get("eligible_for_success") is False
    lines = [
        "=== Diagnostic Incumbent (Not Eligible for Success) ==="
        if diagnostic_only
        else "=== Final Selected Result ==="
    ]
    lines.append(f"source: {source}")
    if final_selection.get("status"):
        lines.append(f"status: {final_selection.get('status')}")
    if final_selection.get("reason"):
        lines.append(f"reason: {final_selection.get('reason')}")
    if "applied" in final_selection:
        lines.append(f"applied: {bool(final_selection.get('applied'))}")
    mode = final_selection.get("mode")
    if mode:
        lines.append(f"mode: {mode}")
    candidate_id = final_selection.get("candidate_id")
    if candidate_id is not None:
        lines.append(f"candidate_id: {candidate_id}")
    candidate_source = final_selection.get("candidate_source")
    if candidate_source:
        lines.append(f"candidate_source: {candidate_source}")
    selection_basis = final_selection.get("selection_basis")
    if selection_basis:
        lines.append(f"selection_basis: {selection_basis}")
    lines.append(f"expr: {expr}")

    truth_summary = _format_truth_eval_summary(
        final_selection.get("truth_eval"),
        title="=== Final Selected Ground Truth Check ===",
    )
    if truth_summary:
        lines.append("")
        lines.append(truth_summary)
    return "\n".join(lines)


def _final_selection_from_report(report: dict) -> dict | None:
    """Return the effective final selection, with a Stage-B fallback for older reports."""
    final_selection = report.get("final_selection")
    if isinstance(final_selection, dict) and final_selection.get("expr"):
        return final_selection

    stagec = report.get("stageC")
    expr = None
    if isinstance(stagec, dict):
        expr = stagec.get("y_expr_str") or stagec.get("phi_expr_str")
    if not expr:
        return None
    if isinstance(stagec, dict) and stagec.get("certified") is False:
        out = {
            "source": "stageB",
            "applied": False,
            "eligible_for_success": False,
            "status": "stagec_expression_uncertified",
            "reason": stagec.get("certification_reason"),
            "expr": expr,
        }
        if isinstance(stagec.get("unit_admissibility"), dict):
            out["unit_admissibility"] = stagec["unit_admissibility"]
        if isinstance(stagec.get("coefficient_metadata"), dict):
            out["coefficient_metadata"] = stagec["coefficient_metadata"]
        return out
    out = {
        "source": "stageB",
        "applied": True,
        "expr": expr,
    }
    if isinstance(stagec, dict) and isinstance(stagec.get("unit_admissibility"), dict):
        out["unit_admissibility"] = stagec["unit_admissibility"]
    if isinstance(stagec, dict) and isinstance(stagec.get("coefficient_metadata"), dict):
        out["coefficient_metadata"] = stagec["coefficient_metadata"]
    truth_eval = report.get("truth_eval")
    if isinstance(truth_eval, dict):
        out["truth_eval"] = truth_eval
    return out


def _unit_certificate_has_risk(certificate) -> bool:
    """Distinguish a required-but-unavailable check from legacy unchecked mode."""

    return bool(
        isinstance(certificate, dict)
        and (
            (
                certificate.get("checked") is True
                and certificate.get("valid") is not True
            )
            or str(certificate.get("code") or "") == "expression_unavailable"
        )
    )


def _stagec_has_unit_risk(stagec: dict | None) -> bool:
    """Return whether Stage C requires a checked-valid downstream override."""

    stagec = stagec if isinstance(stagec, dict) else {}
    certificate = stagec.get("unit_admissibility")
    sympy_meta = stagec.get("sympy_meta")
    sympy_meta = sympy_meta if isinstance(sympy_meta, dict) else {}
    nested_certificate = sympy_meta.get("unit_admissibility")
    return bool(
        _unit_certificate_has_risk(certificate)
        or _unit_certificate_has_risk(nested_certificate)
        or str(sympy_meta.get("kind") or "")
        == "unit_check_expression_unavailable"
        or (
            stagec.get("units_checked") is True
            and stagec.get("units_ok") is not True
        )
        or str(stagec.get("symbolic_status") or "") == "unit_invalid"
    )


def _report_final_selection_eligibility(report: dict) -> tuple[bool, str | None]:
    """Return whether a report has a result eligible to count as a success.

    Explicit final selections take precedence so a later successful CoE
    adjudication can supersede a failed final-polish attempt.  Legacy reports
    without either marker remain eligible for backward compatibility.
    """

    final_polish = report.get("final_polish")
    no_safe_polish = bool(
        isinstance(final_polish, dict)
        and str(final_polish.get("status") or "")
        == "no_safe_unit_valid_replacement"
    )
    stagec = report.get("stageC")
    stagec = stagec if isinstance(stagec, dict) else {}
    stagec_unit_risk = _stagec_has_unit_risk(stagec)

    final_selection = report.get("final_selection")
    if isinstance(final_selection, dict):
        coefficient_metadata = final_selection.get("coefficient_metadata")
        expression = final_selection.get("expr")
        if expression:
            try:
                coefficient_symbol_values_for_expression(
                    coefficient_metadata,
                    expression,
                )
            except CoefficientMetadataError as exc:
                return False, f"invalid final coefficient metadata: {exc.reason}"
        unit_certificate = final_selection.get("unit_admissibility")
        if final_selection.get("eligible_for_success") is False:
            return False, str(
                final_selection.get("reason")
                or final_selection.get("status")
                or "final selection is diagnostic only"
            )
        if final_selection.get("applied") is False:
            return False, str(
                final_selection.get("reason")
                or final_selection.get("status")
                or "final selection was not applied"
            )
        if final_selection.get("expr"):
            if _unit_certificate_has_risk(unit_certificate):
                return False, str(
                    unit_certificate.get("reason")
                    or "final selection lacks a valid unit certificate"
                )
            if no_safe_polish or stagec_unit_risk:
                if not (
                    isinstance(unit_certificate, dict)
                    and unit_certificate.get("checked") is True
                    and unit_certificate.get("valid") is True
                ):
                    return False, "unit-risk final selection lacks a valid unit certificate"
            return True, None

    if isinstance(final_polish, dict):
        status = str(final_polish.get("status") or "")
        if status == "no_safe_unit_valid_replacement":
            return False, str(
                final_polish.get("reason")
                or "no safe unit-valid final-polish replacement"
            )
    if isinstance(stagec, dict) and stagec.get("certified") is False:
        return False, str(
            stagec.get("certification_reason")
            or "Stage C expression is not certified"
        )
    return True, None


def _update_report_with_campaign_outcome(report_path: str) -> dict | None:
    """Persist the settled truth-blind cheap/CoE campaign decision."""

    try:
        with open(report_path, "r") as f_report:
            report = json.load(f_report)
        if not isinstance(report, dict):
            raise ValueError("report root is not a JSON object")
        outcome = report_campaign_outcome(report)
        report["campaign_outcome"] = outcome
        with open(report_path, "w") as f_report:
            json.dump(report, f_report, indent=2)
        return outcome
    except Exception as exc:
        print(f"[Campaign] Warning: could not persist campaign outcome: {exc}")
        return None


def _append_final_selection_report(
    *,
    report_path: str,
    path_output: str | None,
    final_human_output: str | None,
) -> None:
    """Print and append the actual final result after all optional adjudication."""
    try:
        with open(report_path, "r") as f_report:
            report = json.load(f_report)
        final_selection = _final_selection_from_report(report)
        final_text = _format_final_selection_report(final_selection)
        if not final_text:
            return
        print("\n" + final_text)
        if path_output is not None:
            with open(path_output, "a") as f_path:
                f_path.write("\n" + final_text + "\n")
        if final_human_output and os.path.exists(final_human_output):
            with open(final_human_output, "a") as f_final:
                f_final.write("\n" + final_text + "\n")
    except Exception as e:
        print(f"Warning: could not append final selected result: {e}")


def _update_report_with_final_polish(report_path: str, summary: dict) -> None:
    try:
        with open(report_path, "r") as f:
            report = json.load(f)
    except Exception as e:
        print(f"[FinalPolish] Warning: could not read report for update: {e}")
        return
    report["final_polish"] = _make_json_serializable(summary)
    if bool(summary.get("proposal_only", False)):
        try:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            print(
                "[FinalPolish] Stored proposal-only frontier; "
                "statistical selection retains final authority"
            )
        except Exception as e:
            print(f"[FinalPolish] Warning: could not update report: {e}")
        return
    truth = summary.get("truth_eval")
    rec = summary.get("recommended")
    recommendation_unit_certificate = None
    if isinstance(rec, dict) and isinstance(rec.get("unit_admissibility"), dict):
        recommendation_unit_certificate = rec.get("unit_admissibility")
    elif isinstance(summary.get("unit_admissibility"), dict):
        recommendation_unit_certificate = summary.get("unit_admissibility")
    status = str(summary.get("status") or "")
    if (
        status == "success"
        and isinstance(truth, dict)
        and isinstance(rec, dict)
        and rec.get("expr")
    ):
        old_truth = report.get("truth_eval")
        if isinstance(old_truth, dict) and "truth_eval_pre_final_polish" not in report:
            report["truth_eval_pre_final_polish"] = old_truth
        final_truth = _make_json_serializable(truth)
        if isinstance(final_truth, dict):
            final_truth = dict(final_truth)
            final_truth["source"] = "final_polish"
            final_truth["expr"] = rec.get("expr")
            report["truth_eval"] = final_truth
        report["final_selection"] = {
            "source": "final_polish",
            "applied": True,
            "eligible_for_success": True,
            "expr": rec.get("expr"),
            "truth_eval": final_truth,
        }
        if isinstance(summary.get("coefficient_metadata"), dict):
            report["final_selection"]["coefficient_metadata"] = (
                _make_json_serializable(summary.get("coefficient_metadata"))
            )
        if isinstance(recommendation_unit_certificate, dict):
            report["final_selection"]["unit_admissibility"] = (
                _make_json_serializable(recommendation_unit_certificate)
            )
    elif status == "no_safe_unit_valid_replacement":
        stagec = report.get("stageC")
        incumbent_expr = None
        if isinstance(stagec, dict):
            incumbent_expr = stagec.get("y_expr_str") or stagec.get("phi_expr_str")
        if incumbent_expr:
            diagnostic = {
                "source": "stageB",
                "applied": False,
                "eligible_for_success": False,
                "status": status,
                "reason": summary.get("reason"),
                "expr": incumbent_expr,
            }
            if isinstance(stagec.get("unit_admissibility"), dict):
                diagnostic["unit_admissibility"] = _make_json_serializable(
                    stagec.get("unit_admissibility")
                )
            if isinstance(stagec.get("coefficient_metadata"), dict):
                diagnostic["coefficient_metadata"] = _make_json_serializable(
                    stagec.get("coefficient_metadata")
                )
            old_truth = report.get("truth_eval")
            if isinstance(old_truth, dict):
                diagnostic_truth = _make_json_serializable(old_truth)
                diagnostic["truth_eval"] = diagnostic_truth
                report["truth_eval_diagnostic_incumbent"] = diagnostic_truth
            report["final_selection"] = diagnostic
            failure_truth = {
                "success": False,
                "skipped": True,
                "reason": "final_selection_ineligible",
                "selection_status": status,
            }
            if isinstance(old_truth, dict):
                failure_truth["diagnostic_truth_eval_field"] = (
                    "truth_eval_diagnostic_incumbent"
                )
            report["truth_eval"] = failure_truth
    try:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[FinalPolish] Updated JSON report with final_polish: {report_path}")
    except Exception as e:
        print(f"[FinalPolish] Warning: could not update report: {e}")
