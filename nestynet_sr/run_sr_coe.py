# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""CoE reporting helpers for ``nestynet_sr.run_SR``."""

from __future__ import annotations

import json
import math
import os
import pathlib
import pickle
import re
import subprocess
import sys
import time
import timeit
from typing import Any, Optional

import numpy as np
import torch

from nestynet_sr.sr_core import ast_to_human_readable
from nestynet_sr.sr_core.coefficient_metadata import (
    coefficient_symbol_values_for_expression,
    collect_coefficient_metadata,
)
from nestynet_sr.sr_core.fit_links import canonical_fit_link_name
from nestynet_sr.sr_core.problem_identity import canonical_problem_id
from nestynet_sr.sr_search.coe_witness import (
    CoEWitnessExecutor,
    coe_witness_execution_metadata,
    coe_witness_jobs_from_specs,
    run_fixed_expression_candidate_witnesses,
    run_fixed_expression_pair_witnesses,
    run_threaded_witnesses,
)
from nestynet_sr.run_sr_reports import (
    _make_json_serializable,
    _stagec_has_unit_risk,
    _unit_certificate_has_risk,
)


def _run_coe_stageA_exit_audit(
    *,
    args,
    filepath,
    results_dir: str,
    base_filename: str,
    stageA_data: Optional[dict],
    noise_sigma_y=None,
    y_op_inv=None,
    initial_model=None,
    final_model=None,
    units_spec=None,
) -> Optional[dict]:
    """Observe-only Stage-A exit audit for CoE PR-A1.

    This deliberately does not change Stage-A behavior.  Portable fixed
    expressions are evaluated on independent slices; NN-containing exits are
    recorded as requiring the later same-history refit ladder.
    """
    if not isinstance(stageA_data, dict):
        return None
    mode = str(getattr(args, "coe_mode", "off") or "off")
    enabled = bool(
        getattr(args, "coe_stageA_dry_run", False)
        or mode in {"committee_gated", "reservoir_discovery"}
    )
    if not enabled:
        return None

    def _noise_floor() -> float:
        try:
            sigma = float(noise_sigma_y)
            if math.isfinite(sigma) and sigma > 0.0:
                return float(sigma * sigma)
        except Exception:
            pass
        return 0.0

    def _nn_count(node) -> Optional[int]:
        if node is None:
            return None
        try:
            from nestynet_sr.sr_core.bridges import collect_nn_atoms

            return int(len(collect_nn_atoms(node)))
        except Exception:
            return None

    def _fixed_expr(node) -> str:
        expr = ast_to_human_readable(node, stageA_data.get("x_transform_map"))
        y_name = str(stageA_data.get("y_op_name", "identity") or "identity")
        if y_name != "identity":
            if y_op_inv is None:
                raise ValueError("non-identity y-transform has no inverse renderer")
            from nestynet_sr.sr_search.transform_render import wrap_phi_expr_str

            expr_wrapped = wrap_phi_expr_str(str(expr), y_op_inv, simplify=False)
            if not expr_wrapped:
                raise ValueError("inverse y-transform renderer returned an empty expression")
            expr = str(expr_wrapped)
        return str(expr)

    initial_ast = stageA_data.get("initial_ast")
    final_ast = stageA_data.get("ast")
    n_initial_nn = _nn_count(initial_ast)
    n_candidate_nn = _nn_count(final_ast)
    reference_slice = max(0, int(getattr(args, "data_slice", 0) or 0))
    summary = {
        "enabled": True,
        "mode": "stageA_exit_audit",
        "status": "unsupported",
        "committee_status": "unsupported-refit-ladder-required",
        "stageA_status": str(stageA_data.get("stageA_status", "") or ""),
        "y_transform": str(stageA_data.get("y_op_name", "identity") or "identity"),
        "fit_y_link": stageA_data.get("fit_y_link"),
        "n_initial_nn": n_initial_nn,
        "n_candidate_nn": n_candidate_nn,
        "reference_slice": reference_slice,
        "noise_floor_raw": _noise_floor(),
        "would_change_decision": False,
        "real_gate_supported": False,
        "warnings": [],
    }
    if final_ast is None:
        summary["status"] = "skipped"
        summary["committee_status"] = "skipped-no-stageA-final"
        summary["reason"] = "Stage-A final AST is unavailable."
        return _make_json_serializable(summary)

    final_is_fixed = n_candidate_nn == 0
    initial_is_fixed = n_initial_nn == 0
    if not final_is_fixed:
        summary["reason"] = (
            "Final Stage-A state still contains NN leaves; PR-A1 records this "
            "as requiring the later same-history committee refit ladder."
        )
        try:
            summary["initial_snapshot"] = ast_to_human_readable(
                initial_ast, stageA_data.get("x_transform_map")
            )
        except Exception:
            summary["initial_snapshot"] = str(initial_ast)
        try:
            summary["candidate_snapshot"] = ast_to_human_readable(
                final_ast, stageA_data.get("x_transform_map")
            )
        except Exception:
            summary["candidate_snapshot"] = str(final_ast)
        return _make_json_serializable(summary)

    try:
        candidate_expr_raw = _fixed_expr(final_ast)
        incumbent_expr_raw = _fixed_expr(initial_ast) if initial_is_fixed and initial_ast is not None else None
    except Exception as exc:
        summary["status"] = "unsupported"
        summary["committee_status"] = "unsupported-expression-rendering"
        summary["reason"] = f"Stage-A expression rendering failed: {type(exc).__name__}: {exc}"
        return _make_json_serializable(summary)

    try:
        from nestynet_sr.sr_search.coe_committee import (
            CandidateArtifact,
            CommitteeEvalCache,
            _committee_tolerance,
            _clean_expr,
            _load_dataset_arrays,
            build_slice_specs,
        )
    except Exception as exc:
        summary["status"] = "error"
        summary["committee_status"] = "error"
        summary["reason"] = f"CoE committee evaluator unavailable: {type(exc).__name__}: {exc}"
        return _make_json_serializable(summary)

    candidate_expr = _clean_expr(candidate_expr_raw)
    incumbent_expr = _clean_expr(incumbent_expr_raw) if incumbent_expr_raw is not None else None
    if candidate_expr is None:
        summary.update(
            {
                "status": "unsupported",
                "committee_status": "unsupported-nonportable-fixed-expression",
                "reason": (
                    "Stage-A final expression contains local learned wrappers "
                    "or placeholders that cannot be scored by the fixed-expression committee."
                ),
                "candidate_expr": str(candidate_expr_raw),
            }
        )
        return _make_json_serializable(summary)
    if incumbent_expr_raw is not None and incumbent_expr is None:
        summary.setdefault("warnings", []).append(
            "Stage-A initial expression is non-portable for fixed-expression committee scoring."
        )

    candidate_coefficient_metadata = None
    if final_model is not None:
        candidate_coefficient_metadata = collect_coefficient_metadata(
            final_ast,
            final_model,
            units_spec,
        )
    incumbent_coefficient_metadata = None
    if initial_model is None and final_model is not None:
        try:
            from nestynet_sr.sr_core.bridges import ast_equals

            if ast_equals(initial_ast, final_ast):
                initial_model = final_model
        except Exception:
            pass
    if incumbent_expr is not None and initial_model is not None:
        incumbent_coefficient_metadata = collect_coefficient_metadata(
            initial_ast,
            initial_model,
            units_spec,
        )

    n_slices = min(
        max(0, int(getattr(args, "coe_num_slices", 0) or 0)),
        max(0, int(getattr(args, "coe_stageB_gate_slices", 0) or 0)),
    )
    if n_slices <= 0:
        summary["status"] = "skipped"
        summary["committee_status"] = "skipped-no-slices-configured"
        summary["reason"] = "No CoE Stage-A exit audit slices are configured."
        return _make_json_serializable(summary)
    n_train = max(1, int(getattr(args, "ndata_train", None) or 2000))
    n_val = max(1, int(getattr(args, "ndata_val", None) or 2000))
    max_rows = None
    try:
        _X_all, _y_all, _cols = _load_dataset_arrays(str(filepath))
        max_rows = int(_y_all.shape[0])
    except Exception as exc:
        summary.setdefault("warnings", []).append(
            f"could not inspect data row count: {type(exc).__name__}: {exc}"
        )
    specs = build_slice_specs(
        n_slices=n_slices,
        ndata_train=n_train,
        ndata_val=n_val,
        start_slice=max(0, int(getattr(args, "coe_start_slice", 0) or 0)),
        skip_slice_ids=(reference_slice,),
        max_rows=max_rows,
    )
    summary["slice_specs"] = [s.to_dict() for s in specs]
    if not specs:
        summary["status"] = "skipped"
        summary["committee_status"] = "skipped-no-valid-slices"
        summary["reason"] = "No independent CoE validation slices fit inside the dataset."
        return _make_json_serializable(summary)

    candidate = CandidateArtifact(
        candidate_id="stageA_final",
        expr=str(candidate_expr),
        source="stageA:exit",
        label="Stage-A final",
        n_free_params=0,
        metadata={
            "stageA_status": summary.get("stageA_status"),
            "coefficient_metadata": candidate_coefficient_metadata,
        },
    )
    incumbent = None
    if incumbent_expr is not None:
        incumbent = CandidateArtifact(
            candidate_id="stageA_initial",
            expr=str(incumbent_expr),
            source="stageA:initial",
            label="Stage-A initial",
            n_free_params=0,
            metadata={
                "coefficient_metadata": incumbent_coefficient_metadata,
            },
        )

    cache = CommitteeEvalCache(enabled=True)
    rows = []
    wins = losses = ties = 0
    cand_losses = []
    inc_losses = []
    min_valid_fraction = float(getattr(args, "coe_min_valid_fraction", 0.80) or 0.80)
    executor = CoEWitnessExecutor(parallelism=max(1, int(getattr(args, "coe_witness_parallelism", 1) or 1)))
    if incumbent is not None:
        witness_rows = run_fixed_expression_pair_witnesses(
            specs=specs,
            incumbent=incumbent,
            candidate=candidate,
            filepath=str(filepath),
            min_valid_fraction=min_valid_fraction,
            executor=executor,
            prefix="stageA_exit",
        )
        for witness in witness_rows:
            cand_row = dict(witness.get("candidate_result") or {})
            inc_row = dict(witness.get("incumbent_result") or {})
            row = {"slice_id": int(witness.get("slice_id", -1)), "candidate": cand_row, "incumbent": inc_row}
            if cand_row.get("status") == "success" and math.isfinite(float(cand_row.get("val_mse", float("inf")))):
                cand_losses.append(float(cand_row["val_mse"]))
            if inc_row.get("status") == "success" and math.isfinite(float(inc_row.get("val_mse", float("inf")))):
                inc_losses.append(float(inc_row["val_mse"]))
            if (
                cand_row.get("status") == "success"
                and inc_row.get("status") == "success"
                and math.isfinite(float(cand_row.get("val_mse", float("inf"))))
                and math.isfinite(float(inc_row.get("val_mse", float("inf"))))
            ):
                tol = _committee_tolerance(
                    loss_a=float(inc_row["val_mse"]),
                    loss_b=float(cand_row["val_mse"]),
                    noise_floor_raw=float(summary["noise_floor_raw"]),
                    n_eff=max(1, int(cand_row.get("n_val", 1) or 1)),
                    noise_mult=float(getattr(args, "coe_noise_mult", 3.0) or 3.0),
                    rel_tol=float(getattr(args, "coe_rel_tol", 1.0e-3) or 1.0e-3),
                )
                delta = float(cand_row["val_mse"]) - float(inc_row["val_mse"])
                if delta < -tol:
                    vote = "win"
                    wins += 1
                elif delta > tol:
                    vote = "loss"
                    losses += 1
                else:
                    vote = "tie"
                    ties += 1
                row.update({"delta": delta, "tolerance": float(tol), "vote": vote})
            rows.append(row)
    else:
        witness_rows = run_fixed_expression_candidate_witnesses(
            specs=specs,
            candidates=[candidate],
            filepath=str(filepath),
            min_valid_fraction=min_valid_fraction,
            executor=executor,
            prefix="stageA_exit",
        )
        for cand_row in witness_rows:
            row = {"slice_id": int(cand_row.get("slice_id", -1)), "candidate": dict(cand_row)}
            if cand_row.get("status") == "success" and math.isfinite(float(cand_row.get("val_mse", float("inf")))):
                cand_losses.append(float(cand_row["val_mse"]))
            rows.append(row)

    def _median(vals) -> float:
        arr = np.asarray([float(v) for v in vals if math.isfinite(float(v))], dtype=np.float64)
        if arr.size == 0:
            return float("inf")
        return float(np.median(arr))

    paired = incumbent is not None
    summary.update(
        {
            "status": "evaluated",
            "committee_status": "evaluated-paired" if paired else "evaluated-candidate-only",
            "reason": (
                "Observe-only Stage-A exit audit; no Stage-A decision is changed."
                if paired
                else "Observe-only Stage-A exit audit of a fixed final expression; "
                "the initial NN baseline requires the later refit ladder for a paired comparison."
            ),
            "candidate_expr": str(candidate_expr),
            "incumbent_expr": str(incumbent_expr) if incumbent_expr is not None else None,
            "n_slices": len(specs),
            "candidate_median_val_mse": _median(cand_losses),
            "candidate_n_success": int(len(cand_losses)),
            "incumbent_median_val_mse": _median(inc_losses) if paired else None,
            "incumbent_n_success": int(len(inc_losses)) if paired else 0,
            "wins": int(wins),
            "ties": int(ties),
            "losses": int(losses),
            "results": rows,
            "cache": cache.stats(),
            "witness_executor": coe_witness_execution_metadata(executor, witness_rows),
        }
    )
    try:
        jsonl_path = os.path.join(results_dir, f"{base_filename}.coe_stageA_exit_audit.jsonl")
        payload_rows = [
            {
                "mode": "stageA_exit_audit",
                "kind": "summary",
                "status": summary.get("status"),
                "committee_status": summary.get("committee_status"),
                "candidate_median_val_mse": summary.get("candidate_median_val_mse"),
                "incumbent_median_val_mse": summary.get("incumbent_median_val_mse"),
                "wins": summary.get("wins"),
                "ties": summary.get("ties"),
                "losses": summary.get("losses"),
            },
            *[
                {
                    "mode": "stageA_exit_audit",
                    "kind": "slice",
                    **row,
                }
                for row in rows
            ],
        ]
        written = _write_coe_stageA_dry_run_jsonl(jsonl_path, payload_rows)
        if written:
            summary["jsonl_path"] = written
    except Exception as exc:
        summary.setdefault("warnings", []).append(
            f"could not write exit-audit JSONL: {type(exc).__name__}: {exc}"
        )
    return _make_json_serializable(summary)


def _format_coe_stageA_exit_audit_report(summary: Optional[dict]) -> str:
    if not isinstance(summary, dict):
        return "=== CoE Stage A Exit Audit ===\nenabled=False"
    lines = ["=== CoE Stage A Exit Audit ==="]
    lines.append(
        f"enabled={bool(summary.get('enabled', False))} "
        f"status={summary.get('status', 'unknown')} "
        f"committee_status={summary.get('committee_status', 'unknown')}"
    )
    if summary.get("n_slices") is not None:
        lines.append(
            f"slices={int(summary.get('n_slices', 0) or 0)} "
            f"candidate_success={int(summary.get('candidate_n_success', 0) or 0)}"
        )
    if summary.get("candidate_median_val_mse") is not None:
        try:
            lines.append(
                f"candidate_median_val_mse={float(summary.get('candidate_median_val_mse')):.6e}"
            )
        except Exception:
            lines.append(f"candidate_median_val_mse={summary.get('candidate_median_val_mse')}")
    if summary.get("incumbent_median_val_mse") is not None:
        try:
            lines.append(
                f"incumbent_median_val_mse={float(summary.get('incumbent_median_val_mse')):.6e}"
            )
        except Exception:
            lines.append(f"incumbent_median_val_mse={summary.get('incumbent_median_val_mse')}")
        lines.append(
            f"votes: wins={int(summary.get('wins', 0) or 0)} "
            f"ties={int(summary.get('ties', 0) or 0)} "
            f"losses={int(summary.get('losses', 0) or 0)}"
        )
    if summary.get("reason"):
        lines.append(str(summary.get("reason")))
    if summary.get("jsonl_path"):
        lines.append(f"jsonl: {summary.get('jsonl_path')}")
    return "\n".join(lines)


def _coe_stageA_ybranch_committee_rank(
    *,
    lm_hp,
    filepath,
    identity_branch: dict,
    candidate_branches,
    legacy_selected_branch: Optional[dict] = None,
    dtype=torch.float64,
    device: Optional[torch.device] = None,
) -> tuple[Optional[dict], str, dict]:
    """Rank confirmed y-search branches against identity on independent slices.

    This is the PR-A4 real gate/ranker.  It deliberately evaluates the same
    reference-trained branch models, rather than letting witness slices run their
    own Stage-A searches.  Non-identity branches may train in phi(y), but votes
    are always cast in original-y space when an inverse is available.
    """
    mode = str(getattr(lm_hp, "coe_mode", "off") or "off")
    legacy_id = None
    if isinstance(legacy_selected_branch, dict):
        legacy_id = str(legacy_selected_branch.get("branch_id") or legacy_selected_branch.get("name") or "")
    summary: dict[str, Any] = {
        "enabled": mode in {"committee_gated", "reservoir_discovery"},
        "mode": "stageA_ybranch_committee",
        "status": "skipped",
        "decision": "legacy",
        "legacy_selected_branch": legacy_id,
        "selected_branch": legacy_id,
        "reason": "",
        "excluded_slice_ids": [],
        "branches": [],
    }
    if mode not in {"committee_gated", "reservoir_discovery"}:
        summary["reason"] = "CoE y-branch committee is inactive in normal mode."
        return legacy_selected_branch, "legacy-coe-off", _make_json_serializable(summary)
    filepath_s = str(filepath or getattr(lm_hp, "coe_filepath", "") or "")
    if not filepath_s:
        summary.update(
            {
                "status": "unsupported",
                "decision": "legacy",
                "reason": "No single raw data filepath is available for CoE y-branch witnesses.",
            }
        )
        return legacy_selected_branch, "legacy-coe-stageA-ybranch-no-filepath", _make_json_serializable(summary)
    if not isinstance(identity_branch, dict) or identity_branch.get("model") is None:
        summary.update(
            {
                "status": "unsupported",
                "decision": "legacy",
                "reason": "Identity branch model is unavailable for paired y-branch comparison.",
            }
        )
        return legacy_selected_branch, "legacy-coe-stageA-ybranch-no-identity", _make_json_serializable(summary)

    branches = [b for b in list(candidate_branches or []) if isinstance(b, dict) and b.get("model") is not None]
    if not branches:
        summary.update(
            {
                "status": "skipped",
                "decision": "identity",
                "selected_branch": None,
                "reason": "No confirmed y-branches with models were available for CoE ranking.",
            }
        )
        return None, "identity-no-confirmed-ybranches", _make_json_serializable(summary)

    try:
        from nestynet_sr.sr_search.coe_committee import (
            _committee_tolerance,
            _load_dataset_arrays,
            build_slice_specs,
        )
        from nestynet_sr.sr_search.search import _eval_yspace_mse
        from nestynet_sr.sr_search.stageB.evaluation import _eval_original_y_mse_with_inverse
    except Exception as exc:
        summary.update(
            {
                "status": "unsupported",
                "decision": "legacy",
                "reason": f"CoE y-branch evaluator unavailable: {type(exc).__name__}: {exc}",
            }
        )
        return legacy_selected_branch, "legacy-coe-stageA-ybranch-import-error", _make_json_serializable(summary)

    try:
        X_all, y_all, _cols = _load_dataset_arrays(filepath_s)
        max_rows = int(y_all.shape[0])
    except Exception as exc:
        summary.update(
            {
                "status": "unsupported",
                "decision": "legacy",
                "reason": f"Could not load CoE witness data: {type(exc).__name__}: {exc}",
            }
        )
        return legacy_selected_branch, "legacy-coe-stageA-ybranch-data-error", _make_json_serializable(summary)

    n_slices = min(
        max(0, int(getattr(lm_hp, "coe_num_slices", 0) or 0)),
        max(0, int(getattr(lm_hp, "coe_stageB_gate_slices", 0) or 0)),
    )
    if n_slices <= 0:
        summary.update(
            {
                "status": "skipped",
                "decision": "legacy",
                "reason": "No CoE y-branch witness slices are configured.",
            }
        )
        return legacy_selected_branch, "legacy-coe-stageA-ybranch-no-slices", _make_json_serializable(summary)
    reference_slice = max(0, int(getattr(lm_hp, "coe_reference_slice", 0) or 0))
    specs = build_slice_specs(
        n_slices=n_slices,
        ndata_train=max(1, int(getattr(lm_hp, "coe_ndata_train", 2000) or 2000)),
        ndata_val=max(1, int(getattr(lm_hp, "coe_ndata_val", 2000) or 2000)),
        start_slice=max(0, int(getattr(lm_hp, "coe_start_slice", 0) or 0)),
        skip_slice_ids=(reference_slice,),
        max_rows=max_rows,
    )
    summary["excluded_slice_ids"] = [int(reference_slice)]
    summary["slice_specs"] = [s.to_dict() for s in specs]
    if not specs:
        summary.update(
            {
                "status": "skipped",
                "decision": "legacy",
                "reason": "No independent CoE y-branch witness slices fit inside the dataset.",
            }
        )
        return legacy_selected_branch, "legacy-coe-stageA-ybranch-no-valid-slices", _make_json_serializable(summary)

    dev = device if device is not None else torch.device("cpu")
    torch_dtype = dtype if dtype is not None else torch.float64

    def _branch_label(branch: dict) -> str:
        return str(branch.get("branch_id") or branch.get("name") or "branch")

    def _transform_targets(y_raw, y_op):
        y_arr = np.asarray(y_raw, dtype=np.float64).reshape(-1, 1)
        if y_op is None:
            return y_arr
        out = y_op(y_arr)
        return np.asarray(out, dtype=np.float64).reshape(-1, 1)

    def _make_val_loader(spec, branch: dict):
        y_op = branch.get("y_op", None)
        x_np = np.asarray(X_all[int(spec.val_start) : int(spec.val_stop)], dtype=np.float64)
        y_np = _transform_targets(y_all[int(spec.val_start) : int(spec.val_stop)], y_op)
        ds = torch.utils.data.TensorDataset(
            torch.as_tensor(x_np, dtype=torch_dtype),
            torch.as_tensor(y_np, dtype=torch_dtype),
        )
        return torch.utils.data.DataLoader(
            ds,
            batch_size=max(1, min(len(ds), int(getattr(lm_hp, "coe_ndata_val", len(ds)) or len(ds)))),
            shuffle=False,
        )

    def _eval_branch(branch: dict, spec) -> tuple[float, Optional[str]]:
        model_i = branch.get("model", None)
        if model_i is None:
            return float("inf"), "missing-model"
        y_op = branch.get("y_op", None)
        y_op_inv = branch.get("y_op_inv", None)
        if y_op is not None and y_op_inv is None:
            return float("inf"), "missing-y-inverse"
        try:
            val_loader = _make_val_loader(spec, branch)
            if y_op is not None:
                loss = float(_eval_original_y_mse_with_inverse(model_i, val_loader, dev, y_op_inv))
            else:
                loss = float(_eval_yspace_mse(model_i, val_loader, dev))
            if not math.isfinite(loss):
                return float("inf"), "nonfinite-loss"
            return loss, None
        except Exception as exc:
            return float("inf"), f"{type(exc).__name__}: {exc}"

    def _eval_branch_rows(branch: dict, spec) -> tuple[Optional[np.ndarray], Optional[str]]:
        """Per-row squared errors in raw-y space, NaN where non-finite.

        Mirrors ``_eval_branch``'s comparison space (inverse-transforming both
        prediction and target when the branch carries a y-transform) but keeps
        the rows, so the paired max-T observer can treat rows as units.
        """
        model_i = branch.get("model", None)
        if model_i is None:
            return None, "missing-model"
        y_op = branch.get("y_op", None)
        y_op_inv = branch.get("y_op_inv", None)
        if y_op is not None and y_op_inv is None:
            return None, "missing-y-inverse"
        try:
            val_loader = _make_val_loader(spec, branch)
            model_i.eval()
            parts: list[np.ndarray] = []
            with torch.no_grad():
                for batch in val_loader:
                    x, target = batch[0].to(dev), batch[1].to(dev)
                    pred = model_i(x)
                    pred = pred[:, 0] if pred.dim() == 2 and pred.shape[1] == 1 else pred.view(-1)
                    target = (
                        target[:, 0]
                        if target.dim() == 2 and target.shape[1] == 1
                        else target.view(-1)
                    )
                    if y_op is not None:
                        pred = y_op_inv(pred).view(-1)
                        target = y_op_inv(target).view(-1)
                    finite = (torch.isfinite(pred) & torch.isfinite(target)).cpu().numpy()
                    diff = (pred - target).double().cpu().numpy()
                    parts.append(np.where(finite, diff * diff, np.nan))
            if not parts:
                return None, "empty-loader"
            return np.concatenate(parts), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def _median(vals) -> float:
        arr = np.asarray([float(v) for v in vals if math.isfinite(float(v))], dtype=np.float64)
        if arr.size == 0:
            return float("inf")
        return float(np.median(arr))

    executor = CoEWitnessExecutor.from_config(lm_hp)
    witness_jobs = coe_witness_jobs_from_specs(specs, prefix="stageA_ybranch_identity")

    def _identity_worker(job) -> dict:
        spec = job.payload
        loss_i, err_i = _eval_branch(identity_branch, spec)
        row = {"slice_id": int(spec.slice_id), "loss": float(loss_i)}
        if err_i:
            row["error"] = str(err_i)
        return row

    identity_rows = run_threaded_witnesses(witness_jobs, _identity_worker, executor=executor)
    identity_losses: dict[int, float] = {
        int(row["slice_id"]): float(row["loss"])
        for row in identity_rows
        if not row.get("error") and math.isfinite(float(row.get("loss", float("inf"))))
    }
    if not identity_losses:
        summary.update(
            {
                "status": "unsupported",
                "decision": "legacy",
                "reason": "Identity branch had no successful independent witness evaluations.",
                "identity_rows": identity_rows,
            }
        )
        return legacy_selected_branch, "legacy-coe-stageA-ybranch-no-identity-witness", _make_json_serializable(summary)

    summary["identity_median_raw_y_mse"] = _median(identity_losses.values())
    summary["identity_rows"] = identity_rows
    summary["witness_executor"] = coe_witness_execution_metadata(executor, identity_rows)

    branch_summaries = []
    allowed = []
    noise_floor_raw = float(getattr(lm_hp, "coe_noise_floor_raw", 0.0) or 0.0)
    noise_mult = float(getattr(lm_hp, "coe_noise_mult", 3.0) or 3.0)
    rel_tol = float(getattr(lm_hp, "coe_rel_tol", 1.0e-3) or 1.0e-3)
    for branch in branches:
        def _branch_worker(job) -> dict:
            spec = job.payload
            sid = int(spec.slice_id)
            cand_loss, err_c = _eval_branch(branch, spec)
            row = {"slice_id": sid, "candidate_loss": float(cand_loss)}
            if err_c:
                row["error"] = str(err_c)
                return row
            if sid not in identity_losses:
                row["error"] = "identity-loss-unavailable"
                return row
            base_loss = float(identity_losses[sid])
            tol = _committee_tolerance(
                loss_a=base_loss,
                loss_b=float(cand_loss),
                noise_floor_raw=noise_floor_raw,
                n_eff=max(1, int(getattr(spec, "val_stop", 0) - getattr(spec, "val_start", 0))),
                noise_mult=noise_mult,
                rel_tol=rel_tol,
            )
            delta = float(cand_loss) - base_loss
            if delta < -float(tol):
                vote = "win"
            elif delta > float(tol):
                vote = "loss"
            else:
                vote = "tie"
            row.update(
                {
                    "identity_loss": base_loss,
                    "delta": float(delta),
                    "tolerance": float(tol),
                    "vote": vote,
                }
            )
            return row

        rows = run_threaded_witnesses(
            coe_witness_jobs_from_specs(specs, prefix=f"stageA_ybranch_{_branch_label(branch)}"),
            _branch_worker,
            executor=executor,
        )
        wins = sum(1 for row in rows if row.get("vote") == "win")
        ties = sum(1 for row in rows if row.get("vote") == "tie")
        losses = sum(1 for row in rows if row.get("vote") == "loss")
        invalid = sum(1 for row in rows if row.get("error"))
        deltas = [
            float(row["delta"])
            for row in rows
            if row.get("delta") is not None and math.isfinite(float(row.get("delta")))
        ]
        branch_losses = [
            float(row["candidate_loss"])
            for row in rows
            if not row.get("error")
            and row.get("candidate_loss") is not None
            and math.isfinite(float(row.get("candidate_loss")))
        ]
        n_paired = int(wins + ties + losses)
        median_delta = _median(deltas)
        branch_ok = bool(
            n_paired > 0
            and invalid == 0
            and losses == 0
            and math.isfinite(median_delta)
            and median_delta
            <= max(
                0.0,
                abs(
                    _median(
                        [
                            r.get("tolerance", 0.0)
                            for r in rows
                            if isinstance(r, dict) and r.get("tolerance") is not None
                        ]
                    )
                ),
            )
        )
        branch_summary = {
            "branch_id": _branch_label(branch),
            "name": str(branch.get("name") or _branch_label(branch)),
            "confirmation": str(branch.get("confirmation", "")),
            "wins": int(wins),
            "ties": int(ties),
            "losses": int(losses),
            "invalid": int(invalid),
            "n_paired": int(n_paired),
            "median_raw_y_mse": _median(branch_losses),
            "median_delta_vs_identity": float(median_delta),
            "allowed": branch_ok,
            "rows": rows,
        }
        branch_summaries.append(branch_summary)
        if branch_ok:
            rank_key = branch.get("rank_key")
            if rank_key is None:
                rank_key = (10**9, 10**9, branch_summary["median_raw_y_mse"], branch_summary["branch_id"])
            allowed.append(
                (
                    0 if wins > 0 else 1,
                    float(branch_summary["median_raw_y_mse"]),
                    float(median_delta),
                    rank_key,
                    branch,
                    branch_summary,
                )
            )

    summary["branches"] = branch_summaries
    if str(getattr(lm_hp, "coe_inference", "legacy") or "legacy") == "maxt_observe":
        # Observe-only: calibrated paired max-T over the witness rows (rows as
        # units, identity as baseline, branches as the comparison family),
        # recorded next to the legacy votes.  Never changes the decision.
        try:
            from nestynet_sr.stat_selection.committee_inference import (
                maxt_decision_from_slice_rows,
            )

            identity_rows_map: dict[int, tuple[int, np.ndarray]] = {}
            for spec in specs:
                rows_arr, err_rows = _eval_branch_rows(identity_branch, spec)
                if rows_arr is not None and err_rows is None:
                    identity_rows_map[int(spec.slice_id)] = (int(spec.val_start), rows_arr)
            member_rows_map: dict[str, dict[int, tuple[int, np.ndarray]]] = {}
            for branch in branches:
                branch_map: dict[int, tuple[int, np.ndarray]] = {}
                for spec in specs:
                    rows_arr, err_rows = _eval_branch_rows(branch, spec)
                    if rows_arr is not None and err_rows is None:
                        branch_map[int(spec.slice_id)] = (int(spec.val_start), rows_arr)
                if branch_map:
                    member_rows_map[str(_branch_label(branch))] = branch_map
            if identity_rows_map and member_rows_map:
                maxt = maxt_decision_from_slice_rows(
                    baseline_rows=identity_rows_map,
                    member_rows=member_rows_map,
                    seed=int(getattr(lm_hp, "coe_maxt_seed", 0) or 0),
                )
                per_branch: dict[str, dict] = {}
                for row in branch_summaries:
                    bid = str(row.get("branch_id"))
                    if bid not in member_rows_map:
                        continue
                    verdict = maxt.verdict_for(bid)
                    maxt_ok = verdict != "worse"
                    legacy_ok = bool(row.get("allowed", False))
                    per_branch[bid] = {
                        "verdict": verdict,
                        "maxt_allowed": maxt_ok,
                        "legacy_allowed": legacy_ok,
                        "agrees_with_legacy": bool(maxt_ok == legacy_ok),
                    }
                summary["maxt_observe"] = {
                    **maxt.to_dict(),
                    "per_branch": per_branch,
                }
                disagreeing = sorted(
                    bid for bid, v in per_branch.items() if not v["agrees_with_legacy"]
                )
                if disagreeing:
                    print(
                        "[CoE maxt-observe] y-branch verdicts DISAGREE with legacy "
                        f"votes for: {', '.join(disagreeing)} "
                        f"(G={maxt.n_units}, critical={maxt.critical_value:.3f})"
                    )
            else:
                summary["maxt_observe"] = {
                    "status": "unavailable",
                    "reason": "no per-row branch evaluations succeeded",
                }
        except Exception as exc:
            summary["maxt_observe"] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
    if not allowed:
        summary.update(
            {
                "status": "evaluated",
                "decision": "identity",
                "selected_branch": None,
                "reason": "All confirmed y-branches lost or failed on independent raw-y witnesses; keeping identity.",
            }
        )
        return None, "reject-coe-stageA-ybranch-committee", _make_json_serializable(summary)

    allowed.sort(key=lambda item: item[:4])
    selected = allowed[0][4]
    selected_summary = allowed[0][5]
    summary.update(
        {
            "status": "evaluated",
            "decision": "select_branch",
            "selected_branch": str(selected_summary.get("branch_id")),
            "selected_branch_median_raw_y_mse": selected_summary.get("median_raw_y_mse"),
            "selected_branch_median_delta_vs_identity": selected_summary.get("median_delta_vs_identity"),
            "reason": (
                "Selected confirmed y-branch by independent raw-y committee ranking "
                f"against identity ({selected_summary.get('wins', 0)} wins, "
                f"{selected_summary.get('ties', 0)} ties, {selected_summary.get('losses', 0)} losses)."
            ),
        }
    )
    return selected, "accepted-coe-stageA-ybranch-committee", _make_json_serializable(summary)


def _format_coe_stageA_ybranch_committee_report(summary: Optional[dict]) -> str:
    if not isinstance(summary, dict):
        return "=== CoE Stage A y-Branch Committee ===\nenabled=False"
    lines = ["=== CoE Stage A y-Branch Committee ==="]
    lines.append(
        f"enabled={bool(summary.get('enabled', False))} "
        f"status={summary.get('status', 'unknown')} "
        f"decision={summary.get('decision', 'unknown')} "
        f"selected={summary.get('selected_branch')}"
    )
    if summary.get("identity_median_raw_y_mse") is not None:
        try:
            lines.append(
                f"identity_median_raw_y_mse={float(summary.get('identity_median_raw_y_mse')):.6e}"
            )
        except Exception:
            lines.append(f"identity_median_raw_y_mse={summary.get('identity_median_raw_y_mse')}")
    for row in list(summary.get("branches") or [])[:6]:
        try:
            lines.append(
                f"branch {row.get('branch_id')}: median={float(row.get('median_raw_y_mse')):.6e} "
                f"delta={float(row.get('median_delta_vs_identity')):.6e} "
                f"votes W/T/L/I={int(row.get('wins', 0))}/{int(row.get('ties', 0))}/"
                f"{int(row.get('losses', 0))}/{int(row.get('invalid', 0))} "
                f"allowed={bool(row.get('allowed', False))}"
            )
        except Exception:
            lines.append(f"branch {row.get('branch_id')}: {row}")
    if summary.get("reason"):
        lines.append(str(summary.get("reason")))
    return "\n".join(lines)


def _build_coe_stageA_dry_run_records(stageA_data: Optional[dict], *, noise_sigma_y=None) -> list[dict]:
    if not isinstance(stageA_data, dict):
        return []
    records: list[dict] = []
    exit_audit = stageA_data.get("coe_stageA_exit_audit")
    if isinstance(exit_audit, dict):
        records.append(
            {
                "time": timeit.default_timer(),
                "mode": "stageA_exit_audit",
                "outcome": "observe",
                "stageA_status": str(stageA_data.get("stageA_status", "") or ""),
                "risk_tags": ["stageA_exit"],
                "committee_status": str(exit_audit.get("committee_status", "unknown")),
                "would_change_decision": False,
                "real_gate_supported": False,
                "reason": str(exit_audit.get("reason", "")),
                "y_transform": exit_audit.get("y_transform", stageA_data.get("y_op_name")),
                "fit_y_link": exit_audit.get("fit_y_link", stageA_data.get("fit_y_link")),
                "n_initial_nn": exit_audit.get("n_initial_nn"),
                "n_candidate_nn": exit_audit.get("n_candidate_nn"),
                "candidate_median_val_mse": exit_audit.get("candidate_median_val_mse"),
                "incumbent_median_val_mse": exit_audit.get("incumbent_median_val_mse"),
                "wins": exit_audit.get("wins"),
                "ties": exit_audit.get("ties"),
                "losses": exit_audit.get("losses"),
                "jsonl_path": exit_audit.get("jsonl_path"),
            }
        )
    ybranch = stageA_data.get("coe_stageA_ybranch_committee")
    if isinstance(ybranch, dict):
        records.append(
            {
                "time": timeit.default_timer(),
                "mode": "stageA_ybranch_committee",
                "outcome": str(ybranch.get("decision", "unknown")),
                "stageA_status": str(stageA_data.get("stageA_status", "") or ""),
                "risk_tags": ["y_transform_branch"],
                "committee_status": str(ybranch.get("status", "unknown")),
                "would_change_decision": bool(
                    ybranch.get("decision") == "identity"
                    and ybranch.get("legacy_selected_branch")
                ),
                "real_gate_supported": bool(ybranch.get("status") == "evaluated"),
                "reason": str(ybranch.get("reason", "")),
                "y_transform": stageA_data.get("y_op_name"),
                "selected_branch": ybranch.get("selected_branch"),
                "legacy_selected_branch": ybranch.get("legacy_selected_branch"),
                "identity_median_raw_y_mse": ybranch.get("identity_median_raw_y_mse"),
                "branches": [
                    {
                        "branch_id": row.get("branch_id"),
                        "median_raw_y_mse": row.get("median_raw_y_mse"),
                        "median_delta_vs_identity": row.get("median_delta_vs_identity"),
                        "wins": row.get("wins"),
                        "ties": row.get("ties"),
                        "losses": row.get("losses"),
                        "invalid": row.get("invalid"),
                        "allowed": row.get("allowed"),
                    }
                    for row in list(ybranch.get("branches") or [])[:8]
                    if isinstance(row, dict)
                ],
            }
        )
    compound_shortlist = stageA_data.get("coe_stageA_compound_shortlist")
    if isinstance(compound_shortlist, dict):
        records.append(
            {
                "time": timeit.default_timer(),
                "mode": "stageA_compound_shortlist_rank",
                "outcome": str(compound_shortlist.get("decision", "unknown")),
                "stageA_status": str(stageA_data.get("stageA_status", "") or ""),
                "risk_tags": ["compound_coordinate", "compound_shortlist_rank"],
                "committee_status": str(compound_shortlist.get("gate_status", "unknown")),
                "would_change_decision": bool(
                    compound_shortlist.get("decision")
                    in {"select_candidate", "select_provisional_candidate", "veto_all"}
                    and compound_shortlist.get("legacy_selected") != compound_shortlist.get("selected")
                ),
                "real_gate_supported": bool(
                    compound_shortlist.get("gate_status")
                    in {"evaluated", "accepted", "accepted_provisional", "veto"}
                ),
                "reason": str(compound_shortlist.get("reason", "")),
                "y_transform": stageA_data.get("y_op_name"),
                "selected": compound_shortlist.get("selected"),
                "legacy_selected": compound_shortlist.get("legacy_selected"),
                "incumbent_median_mse": compound_shortlist.get("incumbent_median_mse"),
                "candidates": [
                    {
                        "z_name": row.get("z_name"),
                        "kind": row.get("kind"),
                        "old_arity": row.get("old_arity"),
                        "new_arity": row.get("new_arity"),
                        "median_mse": row.get("median_mse"),
                        "median_delta": row.get("median_delta"),
                        "wins": row.get("wins"),
                        "ties": row.get("ties"),
                        "losses": row.get("losses"),
                        "invalid": row.get("invalid"),
                        "allowed": row.get("allowed"),
                    }
                    for row in list(compound_shortlist.get("results") or [])[:8]
                    if isinstance(row, dict)
                ],
            }
        )
    provisional_summary = stageA_data.get("stageA_provisional_confirmation")
    if isinstance(provisional_summary, dict) and provisional_summary.get("enabled"):
        records.append(
            {
                "time": timeit.default_timer(),
                "mode": "stageA_provisional_confirmation",
                "outcome": (
                    "confirmed"
                    if int(provisional_summary.get("unconfirmed", 0) or 0) == 0
                    else "pending"
                ),
                "stageA_status": str(stageA_data.get("stageA_status", "") or ""),
                "risk_tags": ["provisional_commit", "stageB_confirmation_required"],
                "committee_status": str(provisional_summary.get("status", "unknown")),
                "would_change_decision": False,
                "real_gate_supported": True,
                "reason": str(provisional_summary.get("reason", "")),
                "total": provisional_summary.get("total"),
                "confirmed": provisional_summary.get("confirmed"),
                "unconfirmed": provisional_summary.get("unconfirmed"),
                "stageA_final_burden": provisional_summary.get("stageA_final_burden"),
                "stageB_final_burden": provisional_summary.get("stageB_final_burden"),
            }
        )
    raw_moves = stageA_data.get("stageA_move_records") or []
    if isinstance(raw_moves, list):
        for move in raw_moves:
            if not isinstance(move, dict):
                continue
            cand_burden = move.get("candidate_burden") if isinstance(move.get("candidate_burden"), dict) else {}
            try:
                n_candidate_nn_move = int(cand_burden.get("nn_total", 0) or 0)
            except Exception:
                n_candidate_nn_move = 0
            committee_status = (
                "candidate-fixed-expression-only"
                if n_candidate_nn_move == 0
                else "unsupported-refit-ladder-required"
            )
            move_details = move.get("details", {})
            coe_split_gate = (
                move_details.get("coe_stageA_overlap_split_gate")
                if isinstance(move_details, dict)
                else None
            )
            real_gate_supported = False
            outcome = "observe"
            reason = (
                "Real Stage-A accept record; current CoE mode observes "
                "the transaction without changing Stage-A behavior."
            )
            if isinstance(coe_split_gate, dict):
                committee_status = str(coe_split_gate.get("gate_status", committee_status))
                outcome = str(coe_split_gate.get("decision", "observe"))
                real_gate_supported = committee_status in {
                    "evaluated",
                    "accepted",
                    "accepted_provisional",
                    "veto",
                }
                reason = str(
                    coe_split_gate.get(
                        "reason",
                        "Real Stage-A split record includes a CoE overlap split gate result.",
                    )
                )
            records.append(
                {
                    "time": timeit.default_timer(),
                    "mode": "stageA_move_record",
                    "outcome": outcome,
                    "stageA_status": str(stageA_data.get("stageA_status", "") or ""),
                    "move_kind": str(move.get("move_kind", "unknown")),
                    "move_seq": move.get("seq"),
                    "provisional": bool(move.get("provisional", False)),
                    "provisional_reason": move.get("provisional_reason"),
                    "requires_stageB_confirmation": bool(
                        move.get("requires_stageB_confirmation", False)
                    ),
                    "confirmation_status": move.get("confirmation_status"),
                    "risk_tags": list(move.get("risk_tags") or []),
                    "committee_status": committee_status,
                    "would_change_decision": False,
                    "real_gate_supported": bool(real_gate_supported),
                    "reason": reason,
                    "y_transform": move.get("y_transform", stageA_data.get("y_op_name")),
                    "fit_y_link": move.get("fit_y_link", stageA_data.get("fit_y_link")),
                    "parent_loss": move.get("parent_loss"),
                    "candidate_loss": move.get("candidate_loss"),
                    "parent_burden": move.get("parent_burden"),
                    "candidate_burden": move.get("candidate_burden"),
                    "nn_burden_delta": move.get("nn_burden_delta"),
                    "raw_support_removed": move.get("raw_support_removed"),
                    "raw_support_added": move.get("raw_support_added"),
                    "parent_snapshot": move.get("parent_ast_human"),
                    "candidate_snapshot": move.get("candidate_ast_human"),
                    "details": move_details,
                }
            )
    tags = set()
    status = str(stageA_data.get("stageA_status", "") or "")
    y_op = str(stageA_data.get("y_op_name", "identity") or "identity")
    fit_link = stageA_data.get("fit_y_link")
    if y_op != "identity":
        tags.add("y_transform_branch")
    if fit_link:
        tags.add("transformed_link")
        if not bool(stageA_data.get("fit_link_original_y_certified", False)):
            tags.add("transformed_link_uncertified")
    if bool(stageA_data.get("full_compound_compressed", False)):
        tags.add("compound_coordinate_unresolved")
    if bool(stageA_data.get("has_remaining_nns", False)):
        tags.add("remaining_nn")
    if "compound" in status:
        tags.add("compound_accept")
    if "split" in status:
        tags.add("split_accept")
    if "phase" in status:
        tags.add("phase_direct")
    try:
        ast_s = ast_to_human_readable(stageA_data.get("ast"))
        init_s = ast_to_human_readable(stageA_data.get("initial_ast"))
        if ast_s != init_s:
            tags.add("stageA_changed_ast")
    except Exception:
        ast_s = str(stageA_data.get("ast"))
        init_s = str(stageA_data.get("initial_ast"))
    try:
        from nestynet_sr.sr_core.bridges import collect_nn_atoms

        n_candidate_nn = len(collect_nn_atoms(stageA_data.get("ast")))
        n_initial_nn = len(collect_nn_atoms(stageA_data.get("initial_ast")))
    except Exception:
        n_candidate_nn = None
        n_initial_nn = None
    try:
        val_loss = float(stageA_data.get("val_loss"))
        noise_floor = 0.0
        sigma = float(noise_sigma_y)
        if math.isfinite(sigma) and sigma > 0.0:
            noise_floor = float(sigma * sigma)
        if noise_floor > 0.0 and math.isfinite(val_loss) and val_loss <= 10.0 * noise_floor:
            tags.add("near_noise_floor")
    except Exception:
        val_loss = None
        noise_floor = 0.0
    if not tags:
        return records
    committee_status = "unsupported-refit-ladder-required"
    if n_candidate_nn == 0:
        committee_status = "candidate-fixed-expression-only"
    records.append(
        {
            "time": timeit.default_timer(),
            "mode": "stageA_dry_run",
            "outcome": "observe",
            "stageA_status": status,
            "risk_tags": sorted(tags),
            "committee_status": committee_status,
            "would_change_decision": False,
            "real_gate_supported": False,
            "reason": (
                "Stage-A committee gating requires per-slice refits for NN-containing "
                "incumbents/candidates; Wave 2 records this as dry-run evidence."
            ),
            "y_transform": y_op,
            "fit_y_link": fit_link,
            "fit_link_original_y_certified": bool(
                stageA_data.get("fit_link_original_y_certified", False)
            ),
            "val_loss": val_loss,
            "nn_val_loss": stageA_data.get("nn_val_loss"),
            "noise_floor_raw": noise_floor,
            "n_initial_nn": n_initial_nn,
            "n_candidate_nn": n_candidate_nn,
            "initial_snapshot": init_s,
            "candidate_snapshot": ast_s,
        }
    )
    return records


def _summarize_coe_stageA_dry_run(records, *, enabled: bool) -> dict:
    from collections import Counter

    rows = list(records or [])
    by_tag = Counter()
    by_status = Counter()
    for rec in rows:
        for tag in list(rec.get("risk_tags") or []):
            by_tag[str(tag)] += 1
        by_status[str(rec.get("committee_status", ""))] += 1
    return {
        "enabled": bool(enabled),
        "mode": "stageA_dry_run",
        "total": len(rows),
        "by_tag": dict(sorted(by_tag.items())),
        "by_status": dict(sorted(by_status.items())),
        "would_change_decisions": 0,
        "sample": rows[:5],
    }


def _write_coe_stageA_dry_run_jsonl(path: str, records) -> Optional[str]:
    rows = list(records or [])
    if not rows:
        return None
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for rec in rows:
                f.write(json.dumps(_make_json_serializable(rec), sort_keys=True) + "\n")
        return path
    except Exception as e:
        print(f"[CoE StageA dry-run] Warning: could not write JSONL: {e}")
        return None


def _format_coe_stageA_dry_run_report(summary: dict) -> str:
    lines = ["=== CoE Stage A Dry-Run ==="]
    lines.append(
        f"enabled={bool(summary.get('enabled', False))} "
        f"mode={summary.get('mode', 'stageA_dry_run')} "
        f"events={int(summary.get('total', 0) or 0)}"
    )
    by_tag = summary.get("by_tag") or {}
    if by_tag:
        lines.append(
            "risk_tags: "
            + ", ".join(f"{k}={v}" for k, v in sorted(by_tag.items()))
        )
    by_status = summary.get("by_status") or {}
    if by_status:
        lines.append(
            "committee_status: "
            + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items()))
        )
    if summary.get("jsonl_path"):
        lines.append(f"jsonl: {summary.get('jsonl_path')}")
    return "\n".join(lines)


def _stageA_nn_burden_summary_for_ast(ast_obj) -> dict:
    try:
        from nestynet_sr.sr_core.bridges import collect_nn_atoms, effective_arity

        atoms = list(collect_nn_atoms(ast_obj))
    except Exception:
        atoms = []
    arities = []
    for atom in atoms:
        try:
            arities.append(max(0, int(effective_arity(atom))))
        except Exception:
            arities.append(0)
    return {
        "nn_total": int(len(atoms)),
        "nn_multivar": int(sum(1 for a in arities if a > 1)),
        "nn_max_arity": int(max(arities) if arities else 0),
        "nn_arities": list(arities),
    }


def _stageA_provisional_confirmation_summary(stageA_data: Optional[dict], stageB_data: Optional[dict]) -> dict:
    commits = []
    if isinstance(stageA_data, dict):
        commits = [
            dict(row)
            for row in list(stageA_data.get("stageA_provisional_commits") or [])
            if isinstance(row, dict) and bool(row.get("active", True))
        ]
    if not commits:
        return {"enabled": False, "total": 0, "confirmed": 0, "unconfirmed": 0}

    stageA_burden = _stageA_nn_burden_summary_for_ast(stageA_data.get("ast")) if isinstance(stageA_data, dict) else {}
    stageB_burden = None
    stageB_loss = None
    stageA_loss = None
    accept_count = 0
    if isinstance(stageB_data, dict):
        try:
            stageB_loss = float(stageB_data.get("val_loss"))
            if not math.isfinite(stageB_loss):
                stageB_loss = None
        except Exception:
            stageB_loss = None
        try:
            stageA_loss = float(stageA_data.get("val_loss")) if isinstance(stageA_data, dict) else None
            if stageA_loss is not None and not math.isfinite(stageA_loss):
                stageA_loss = None
        except Exception:
            stageA_loss = None
        stageB_burden = {
            "nn_total": int(stageB_data.get("num_nn_atoms") or 0),
            "nn_multivar": int(stageB_data.get("num_multivar_nn_atoms") or 0),
            "nn_max_arity": int(stageB_data.get("max_nn_arity") or 0),
        }
        if stageB_data.get("num_nn_atoms") is None:
            stageB_burden = _stageA_nn_burden_summary_for_ast(stageB_data.get("ast"))
        dlog_summary = stageB_data.get("decision_log_summary")
        if isinstance(dlog_summary, dict):
            try:
                accept_count = int(dlog_summary.get("accept", 0) or 0)
            except Exception:
                accept_count = 0

    if not isinstance(stageB_data, dict):
        status = "pending_no_stageB"
        reason = "Stage B was not run; provisional Stage-A commits remain unconfirmed."
        confirmed = False
    else:
        a_sig = (
            int(stageA_burden.get("nn_multivar", 0)),
            int(stageA_burden.get("nn_max_arity", 0)),
            int(stageA_burden.get("nn_total", 0)),
        )
        b_sig = (
            int(stageB_burden.get("nn_multivar", 0)),
            int(stageB_burden.get("nn_max_arity", 0)),
            int(stageB_burden.get("nn_total", 0)),
        )
        loss_ok = False
        if stageA_loss is not None and stageB_loss is not None:
            tol = max(1.0e-14 * max(1.0, abs(stageA_loss)), 1.0e-8 * max(abs(stageA_loss), 1.0e-30))
            loss_ok = bool(stageB_loss <= stageA_loss + tol)
        if int(stageB_burden.get("nn_total", 0)) == 0:
            status = "confirmed_terminal_stageB"
            reason = "Stage B removed all NN atoms after the provisional Stage-A move."
            confirmed = True
        elif b_sig < a_sig:
            status = "confirmed_stageB_burden_reduction"
            reason = "Stage B reduced unresolved NN burden after the provisional Stage-A move."
            confirmed = True
        elif accept_count > 0 and loss_ok:
            status = "confirmed_stageB_rewrite_nonregression"
            reason = "Stage B accepted downstream rewrite(s) without regressing validation loss."
            confirmed = True
        else:
            status = "pending_unconfirmed"
            reason = "No downstream Stage-B simplification has yet confirmed the provisional Stage-A move."
            confirmed = False

    annotated = []
    for row in commits:
        row["confirmation_status"] = status
        row["confirmation_reason"] = reason
        row["confirmed_by_downstream"] = bool(confirmed)
        row["stageA_final_burden"] = dict(stageA_burden)
        if stageB_burden is not None:
            row["stageB_final_burden"] = dict(stageB_burden)
        if stageB_loss is not None:
            row["stageB_val_loss"] = float(stageB_loss)
        annotated.append(row)
    return {
        "enabled": True,
        "total": len(annotated),
        "confirmed": int(sum(1 for row in annotated if bool(row.get("confirmed_by_downstream", False)))),
        "unconfirmed": int(sum(1 for row in annotated if not bool(row.get("confirmed_by_downstream", False)))),
        "status": status,
        "reason": reason,
        "stageA_final_burden": dict(stageA_burden),
        "stageB_final_burden": dict(stageB_burden) if stageB_burden is not None else None,
        "commits": annotated,
    }


def _write_stageA_provisional_confirmation_jsonl(path: str, summary: Optional[dict]) -> Optional[str]:
    if not isinstance(summary, dict) or not summary.get("enabled"):
        return None
    rows = [
        {
            "mode": "stageA_provisional_confirmation",
            "kind": "summary",
            "status": summary.get("status"),
            "total": summary.get("total"),
            "confirmed": summary.get("confirmed"),
            "unconfirmed": summary.get("unconfirmed"),
            "reason": summary.get("reason"),
        }
    ]
    for row in list(summary.get("commits") or []):
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "mode": "stageA_provisional_confirmation",
                "kind": "commit",
                "seq": row.get("seq"),
                "move_kind": row.get("move_kind"),
                "provisional_reason": row.get("provisional_reason"),
                "confirmation_status": row.get("confirmation_status"),
                "confirmed_by_downstream": row.get("confirmed_by_downstream"),
                "candidate_loss": row.get("candidate_loss"),
                "stageB_val_loss": row.get("stageB_val_loss"),
                "parent_ast_human": row.get("parent_ast_human"),
                "candidate_ast_human": row.get("candidate_ast_human"),
            }
        )
    return _write_coe_stageA_dry_run_jsonl(path, rows)


def _apply_stageA_provisional_guard(
    *,
    args,
    stageA_data: Optional[dict],
    stageB_data: Optional[dict],
) -> dict:
    """In CoE gate modes, do not certify finals built on unconfirmed provisional Stage-A commits."""

    mode = str(getattr(args, "coe_mode", "off") or "off")
    enabled = mode in {"committee_gated", "reservoir_discovery"}
    summary = (
        stageA_data.get("stageA_provisional_confirmation")
        if isinstance(stageA_data, dict)
        else None
    )
    unconfirmed = 0
    total = 0
    if isinstance(summary, dict):
        try:
            unconfirmed = int(summary.get("unconfirmed", 0) or 0)
        except Exception:
            unconfirmed = 0
        try:
            total = int(summary.get("total", 0) or 0)
        except Exception:
            total = 0
    guard = {
        "enabled": bool(enabled and total > 0),
        "mode": mode,
        "total": int(total),
        "unconfirmed": int(unconfirmed),
        "decision": "allow",
        "status": "not_applicable" if total <= 0 else "confirmed",
        "reason": "No provisional Stage-A commits require downstream confirmation.",
    }
    if not guard["enabled"]:
        return guard
    if unconfirmed <= 0:
        guard["reason"] = "All provisional Stage-A commits were confirmed downstream."
        if isinstance(stageA_data, dict):
            stageA_data["stageA_provisional_guard"] = _make_json_serializable(guard)
        if isinstance(stageB_data, dict):
            stageB_data["stageA_provisional_guard"] = _make_json_serializable(guard)
        return guard

    guard.update(
        {
            "decision": "mark_uncertified",
            "status": "unconfirmed",
            "reason": (
                "CoE requires downstream confirmation for exploratory Stage-A commits; "
                f"{unconfirmed} provisional commit(s) remain unconfirmed."
            ),
        }
    )
    if isinstance(stageA_data, dict):
        stageA_data["stageA_provisional_guard"] = _make_json_serializable(guard)
    if isinstance(stageB_data, dict):
        stageB_data["stageA_provisional_guard"] = _make_json_serializable(guard)
        meta = dict(stageB_data.get("sympy_meta") or {})
        if meta.get("accepted"):
            meta["raw_accepted_before_stageA_provisional_guard"] = True
        meta.update(
            {
                "accepted": False,
                "parse_success": meta.get("parse_success", True),
                "reason": "stageA_provisional_unconfirmed",
                "kind": "stageA_provisional_unconfirmed",
                "unconfirmed_stageA_provisional_count": int(unconfirmed),
                "stageA_provisional_guard": _make_json_serializable(guard),
            }
        )
        stageB_data["sympy_meta"] = meta
    return guard


def _enforce_stageA_provisional_guard_on_report(
    report_path: str,
    guard: Optional[dict],
) -> Optional[dict]:
    """Keep a later terminal selector from erasing an unconfirmed Stage-A debt."""

    if not isinstance(guard, dict) or guard.get("decision") != "mark_uncertified":
        return None
    path = pathlib.Path(report_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("report root is not a JSON object")
    selected = report.get("final_selection")
    if not isinstance(selected, dict):
        raise ValueError("provisional guard found no final selection to protect")

    blocked = dict(selected)
    report["blocked_final_selection"] = blocked
    final_selection = dict(selected)
    final_selection.update(
        {
            "applied": False,
            "eligible_for_success": False,
            "status": "stageA_provisional_unconfirmed",
            "reason": guard.get("reason"),
            "stageA_provisional_guard": _make_json_serializable(guard),
            "certification_status_before_guard": blocked.get("status"),
        }
    )
    report["final_selection"] = final_selection
    statistical = report.get("statistical_selection")
    if isinstance(statistical, dict):
        statistical["selection_blocked_by_stageA_provisional_guard"] = True
        statistical["eligible_for_success"] = False
        statistical["stageA_provisional_guard"] = _make_json_serializable(guard)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return final_selection


def _coe_selected_candidate_row(summary: dict) -> Optional[dict]:
    rec_id = summary.get("recommended_id")
    if rec_id is None:
        return None
    for row in list(summary.get("candidate_summary") or []):
        cand = row.get("candidate") or {}
        if cand.get("candidate_id") == rec_id:
            return row
    return None


def _coe_unit_admissibility_certificate(row: dict) -> Optional[dict]:
    """Return the strongest explicit unit certificate attached to a CoE row."""

    candidate = row.get("candidate") if isinstance(row, dict) else None
    candidate = candidate if isinstance(candidate, dict) else {}
    metadata = candidate.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    certificates = (
        row.get("unit_admissibility") if isinstance(row, dict) else None,
        candidate.get("unit_admissibility"),
        metadata.get("unit_admissibility"),
    )
    fallback = None
    checked_valid = None
    for certificate in certificates:
        if not isinstance(certificate, dict):
            continue
        if fallback is None:
            fallback = certificate
        if certificate.get("checked") is True:
            if certificate.get("valid") is not True:
                return dict(certificate)
            if checked_valid is None:
                checked_valid = certificate
    if isinstance(checked_valid, dict):
        return dict(checked_valid)
    return dict(fallback) if isinstance(fallback, dict) else None


def _coe_unit_admissibility_certified(row: dict) -> bool:
    """Require an explicit checked-and-valid certificate for unit-risk rescues."""

    certificate = _coe_unit_admissibility_certificate(row)
    return bool(
        isinstance(certificate, dict)
        and certificate.get("checked") is True
        and certificate.get("valid") is True
    )


def _apply_coe_final_adjudication(report_path: str, summary: dict) -> Optional[dict]:
    """Apply final_adjudicate as a final-selection override only.

    This deliberately does not mutate Stage A/B state.  It only records the
    CoE-selected final expression in the report and, when a final-polish block
    exists, replaces its displayed recommendation with a committee-backed
    record while preserving the pre-CoE recommendation.
    """
    if str(summary.get("mode", "")) not in {"final_adjudicate", "committee_gated", "reservoir_discovery"}:
        return None
    if str(summary.get("status", "")) != "success":
        return None
    row = _coe_selected_candidate_row(summary)
    if not row:
        return None
    cand = row.get("candidate") or {}
    expr = cand.get("expr") or summary.get("recommended_expr")
    if not expr:
        return None
    unit_certificate = _coe_unit_admissibility_certificate(row)
    selected = {
        "expr": expr,
        "display_expr": expr,
        "label": "coe_final_adjudication",
        "val_mse": row.get("median_val_mse"),
        "val_mse_se": None,
        "complexity": cand.get("complexity"),
        "n_free_params": cand.get("n_free_params", 0),
        "frac_valid": 1.0 if int(row.get("n_success", 0) or 0) > 0 else 0.0,
        "is_recommended": True,
        "source_hints": [
            str(cand.get("source", "")),
            "coe_committee",
            f"coe_candidate_id={cand.get('candidate_id')}",
        ],
        "coe_candidate_id": cand.get("candidate_id"),
        "coe_source": cand.get("source"),
        "coe_n_success": row.get("n_success"),
        "coe_median_val_mse": row.get("median_val_mse"),
        "coe_mean_val_mse": row.get("mean_val_mse"),
        "coe_noise_tied_with_best": row.get("noise_tied_with_best"),
    }
    if isinstance(unit_certificate, dict):
        selected["unit_admissibility"] = unit_certificate
    final_selection = {
        "source": "coe_committee",
        "mode": str(summary.get("mode", "final_adjudicate") or "final_adjudicate"),
        "applied": True,
        "eligible_for_success": True,
        "candidate_id": cand.get("candidate_id"),
        "expr": expr,
        "candidate_source": cand.get("source"),
        "selection_basis": (summary.get("config") or {}).get("selection_basis"),
        "median_val_mse": row.get("median_val_mse"),
        "mean_val_mse": row.get("mean_val_mse"),
        "n_success": row.get("n_success"),
        "n_slices": summary.get("n_slices"),
    }
    if isinstance(unit_certificate, dict):
        final_selection["unit_admissibility"] = unit_certificate
    try:
        with open(report_path, "r") as f:
            report = json.load(f)
    except Exception as e:
        print(f"[CoE] Warning: could not read report for final adjudication: {e}")
        return None
    final_polish = report.get("final_polish")
    no_safe_polish = bool(
        isinstance(final_polish, dict)
        and str(final_polish.get("status") or "")
        == "no_safe_unit_valid_replacement"
    )
    stagec = report.get("stageC")
    stagec = stagec if isinstance(stagec, dict) else {}
    candidate_metadata = cand.get("metadata")
    candidate_metadata = (
        candidate_metadata if isinstance(candidate_metadata, dict) else {}
    )
    coefficient_metadata = cand.get("coefficient_metadata")
    if coefficient_metadata is None:
        coefficient_metadata = candidate_metadata.get("coefficient_metadata")
    if isinstance(coefficient_metadata, dict):
        selected["coefficient_metadata"] = coefficient_metadata
        final_selection["coefficient_metadata"] = coefficient_metadata
    coefficient_metadata_error = None
    try:
        coefficient_values = coefficient_symbol_values_for_expression(
            coefficient_metadata,
            expr,
        )
    except Exception as exc:
        coefficient_values = {}
        coefficient_metadata_error = str(exc)
    stagec_unit_risk = _stagec_has_unit_risk(stagec)
    selected_certificate_invalid = _unit_certificate_has_risk(unit_certificate)
    if (
        coefficient_metadata_error is not None
        or selected_certificate_invalid
        or (
            (no_safe_polish or stagec_unit_risk)
            and not _coe_unit_admissibility_certified(row)
        )
    ):
        if coefficient_metadata_error is not None:
            block_reason = "coe_selection_has_invalid_coefficient_metadata"
        elif selected_certificate_invalid:
            block_reason = "coe_selection_has_invalid_unit_admissibility_certificate"
        else:
            block_reason = "coe_selection_lacks_unit_admissibility_certificate"
        block = {
            "status": "blocked",
            "reason": block_reason,
            "candidate_id": cand.get("candidate_id"),
            "expr": expr,
            "required_certificate": {
                "unit_admissibility": {"checked": True, "valid": True}
            },
        }
        if coefficient_metadata_error is not None:
            block["coefficient_metadata_error"] = coefficient_metadata_error
        summary["final_adjudication"] = block
        report["coe_committee"] = _make_json_serializable(summary)
        report["coe_final_adjudication"] = block
        try:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            print(
                "[CoE] Final selection blocked after a certification failure: "
                f"{block_reason}"
            )
        except Exception as e:
            print(f"[CoE] Warning: could not record blocked adjudication: {e}")
        return None
    truth_result = None
    try:
        from nestynet_sr.sr_search.truth_eval import evaluate_canary

        dataset_s = str(((report.get("metadata") or {}).get("dataset")) or "")
        dataset_stem = pathlib.Path(dataset_s).stem if dataset_s else None
        if dataset_stem:
            truth_kwargs = {
                "dataset_stem": dataset_stem,
                "discovered_expr_str": str(expr),
                "verbose": False,
            }
            if coefficient_values:
                truth_kwargs["symbol_values"] = coefficient_values
            truth_result = evaluate_canary(**truth_kwargs)
    except Exception as e_truth:
        truth_result = {
            "success": False,
            "skipped": True,
            "reason": "coe_final_truth_eval_error",
            "error_message": str(e_truth),
        }
    if truth_result is not None:
        truth_result = _make_json_serializable(truth_result)
        truth_result = dict(truth_result)
        truth_result["source"] = "coe_final_adjudication"
        truth_result["expr"] = str(expr)
        selected["truth_eval"] = truth_result
        final_selection["truth_eval"] = truth_result
    fp = report.get("final_polish")
    replace_polish_recommendation = cand.get("candidate_id") != summary.get("incumbent_id")
    if isinstance(fp, dict) and replace_polish_recommendation:
        old = fp.get("recommended")
        if isinstance(old, dict):
            old = dict(old)
            old["is_recommended"] = False
            fp["pre_coe_recommended"] = old
        fp["recommended"] = selected
        fp["coe_adjudicated"] = True
        fp["coe_committee_id"] = cand.get("candidate_id")
        fp["coe_selection_basis"] = final_selection.get("selection_basis")
        if truth_result is not None:
            fp["truth_eval"] = truth_result
        report["final_polish"] = fp
    if truth_result is not None:
        old_truth = report.get("truth_eval")
        if isinstance(old_truth, dict):
            report["truth_eval_pre_coe"] = old_truth
        report["truth_eval"] = truth_result
    report["final_selection"] = _make_json_serializable(final_selection)
    try:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[CoE] Applied final committee selection: {expr}")
    except Exception as e:
        print(f"[CoE] Warning: could not update report with final adjudication: {e}")
        return None
    return final_selection


def _run_coe_final_committee(
    *,
    args,
    filepath,
    report_path,
    results_dir,
    base_filename,
    stageB_data,
    final_polish_summary,
    noise_sigma_y=None,
):
    mode = str(getattr(args, "coe_mode", "off") or "off")
    if mode == "off":
        return None
    if stageB_data is None:
        return {
            "mode": mode,
            "status": "skipped",
            "reason": "no Stage B result",
            "enabled": True,
        }
    try:
        from nestynet_sr.sr_search.coe_committee import (
            CommitteeEvalCache,
            run_final_committee_audit,
            write_committee_jsonl,
        )

        n_train = int(getattr(args, "ndata_train", None) or 2000)
        n_val = int(getattr(args, "ndata_val", None) or 2000)
        noise_floor_raw = 0.0
        try:
            sigma = float(noise_sigma_y)
            if math.isfinite(sigma) and sigma > 0.0:
                noise_floor_raw = float(sigma * sigma)
        except Exception:
            noise_floor_raw = 0.0
        cache = CommitteeEvalCache(enabled=True)
        _support_bonus = getattr(args, "coe_reservoir_support_bonus", None)
        if _support_bonus is None:
            _support_bonus = 0.5 if mode == "reservoir_discovery" else 0.0
        _explicit_reservoir_paths = bool(
            str(getattr(args, "coe_reservoir_paths", "") or "").strip()
        )
        decision = run_final_committee_audit(
            filepath=str(filepath),
            stageB_data=stageB_data,
            final_polish_summary=final_polish_summary,
            mode=mode,
            n_slices=max(1, int(getattr(args, "coe_num_slices", 25) or 25)),
            start_slice=max(0, int(getattr(args, "coe_start_slice", 0) or 0)),
            ndata_train=max(1, n_train),
            ndata_val=max(1, n_val),
            max_candidates=max(1, int(getattr(args, "coe_max_candidates", 16) or 16)),
            noise_floor_raw=float(noise_floor_raw),
            noise_mult=float(getattr(args, "coe_noise_mult", 3.0) or 3.0),
            rel_tol=float(getattr(args, "coe_rel_tol", 1.0e-3) or 1.0e-3),
            min_valid_fraction=float(getattr(args, "coe_min_valid_fraction", 0.80) or 0.80),
            reservoir_support_bonus=float(_support_bonus or 0.0),
            cache=cache,
            reference_slice=max(0, int(getattr(args, "data_slice", 0) or 0)),
            include_reservoir=bool(mode == "reservoir_discovery" or _explicit_reservoir_paths),
            witness_parallelism=max(1, int(getattr(args, "coe_witness_parallelism", 1) or 1)),
        )
        summary = decision.to_dict()
        summary["enabled"] = True
        jsonl_path = os.path.join(results_dir, f"{base_filename}.coe_committee.jsonl")
        try:
            write_committee_jsonl(jsonl_path, decision)
            summary["jsonl_path"] = jsonl_path
        except Exception as e_jsonl:
            summary.setdefault("warnings", []).append(f"could not write JSONL: {e_jsonl}")
        return _make_json_serializable(summary)
    except Exception as e:
        import traceback

        return {
            "mode": mode,
            "status": "error",
            "enabled": True,
            "error": str(e),
            "traceback": traceback.format_exc(limit=8),
        }


def _coe_problem_stem(filepath) -> str:
    try:
        return canonical_problem_id(filepath)
    except Exception:
        return str(filepath)


def _coe_stageA_materialization_mode_enabled(args) -> bool:
    mode = str(getattr(args, "coe_mode", "off") or "off")
    return mode in {"committee_gated", "reservoir_discovery"}


def _load_coe_external_stageA_proposal_reservoir(*, args, filepath) -> tuple[Optional[dict], Optional[dict]]:
    mode = str(getattr(args, "coe_mode", "off") or "off")
    if mode == "off":
        return None, None
    raw_paths = str(getattr(args, "coe_reservoir_paths", "") or "").strip()
    if not raw_paths:
        return None, None
    if not _coe_stageA_materialization_mode_enabled(args):
        return None, {
            "enabled": True,
            "source": "coe_reservoir_paths",
            "mode": mode,
            "status": "skipped",
            "reason": "stageA_materialization_disabled_for_mode",
        }
    try:
        from nestynet_sr.sr_search.coe_committee import (
            load_stageA_proposal_reservoir_payloads,
            merge_stageA_proposal_reservoir_payloads,
            split_reservoir_path_string,
        )

        paths = split_reservoir_path_string(raw_paths)
        problem_stem = _coe_problem_stem(filepath)
        payloads, warnings = load_stageA_proposal_reservoir_payloads(
            paths,
            problem_stem=problem_stem,
        )
        summary = {
            "enabled": True,
            "source": "coe_reservoir_paths",
            "mode": mode,
            "external_sources": list(paths),
            "external_payload_count": int(len(payloads)),
        }
        if warnings:
            summary["warnings"] = list(warnings)
        if not payloads:
            summary["status"] = "empty"
            return None, summary
        merged = merge_stageA_proposal_reservoir_payloads(
            payloads,
            max_candidates=max(1, int(getattr(args, "coe_max_candidates", 16) or 16) * 4),
        )
        merged["source"] = "external_stageA_proposal_reservoir"
        merged["external_sources"] = list(paths)
        merged["external_payload_count"] = int(len(payloads))
        summary["status"] = "loaded"
        summary["total_unique"] = int(merged.get("total_unique", 0) or 0)
        summary["candidate_count"] = len(list(merged.get("candidates") or []))
        return merged, summary
    except Exception as exc:
        return None, {
            "enabled": True,
            "source": "coe_reservoir_paths",
            "mode": mode,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _materialize_stageA_y_branch_proposals(
    *,
    reservoir_payload: Optional[dict],
    materialization_summary: Optional[dict],
    y_transform_names: list[str],
    available_y_transform_names: list[str],
) -> tuple[list[str], Optional[dict]]:
    if not isinstance(reservoir_payload, dict):
        return y_transform_names, materialization_summary
    summary = dict(materialization_summary or {})
    try:
        from nestynet_sr.sr_search.coe_committee import (
            stageA_y_branch_names_from_proposal_reservoir,
        )

        requested = stageA_y_branch_names_from_proposal_reservoir(reservoir_payload)
    except Exception as exc:
        summary.setdefault("warnings", []).append(
            f"could not extract Stage-A y-branch proposals: {type(exc).__name__}: {exc}"
        )
        return y_transform_names, summary

    available = set(str(x) for x in available_y_transform_names)
    existing = set(str(x) for x in y_transform_names)
    added: list[str] = []
    skipped: list[dict] = []
    for name in requested:
        name_s = str(name or "").strip()
        if not name_s or name_s == "identity":
            continue
        if name_s not in available:
            skipped.append({"name": name_s, "reason": "not in base y-transform registry"})
            continue
        if name_s in existing:
            continue
        y_transform_names.append(name_s)
        existing.add(name_s)
        added.append(name_s)

    summary["y_branch_requested"] = list(requested)
    summary["y_branch_materialized"] = list(added)
    if skipped:
        summary["y_branch_skipped"] = skipped
    if added:
        print(
            "[CoE Stage-A materialization] Added y-transform proposal(s) from reservoir: "
            + ", ".join(added)
        )
    elif requested:
        print("[CoE Stage-A materialization] No new y-transform proposals added from reservoir.")
    return y_transform_names, summary


def _parse_coe_scout_slice_ids(args) -> list[int]:
    mode = str(getattr(args, "coe_mode", "off") or "off")
    if mode != "reservoir_discovery":
        return []
    current = max(0, int(getattr(args, "data_slice", 0) or 0))
    raw = getattr(args, "coe_scout_slices", None)
    out: list[int] = []
    if raw is not None and str(raw).strip():
        for tok in re.split(r"[,\s]+", str(raw).strip()):
            if not tok:
                continue
            try:
                out.append(int(tok))
            except Exception:
                print(f"[CoE scouts] Ignoring invalid scout slice id: {tok!r}")
    else:
        count = max(0, int(getattr(args, "coe_scout_count", 0) or 0))
        out.extend(range(current + 1, current + 1 + count))
    seen: set[int] = set()
    clean: list[int] = []
    for sid in out:
        if sid < 0 or sid == current or sid in seen:
            continue
        seen.add(int(sid))
        clean.append(int(sid))
    return clean


def _maybe_add_flag(cmd: list[str], args, attr: str, flag: str) -> None:
    if bool(getattr(args, attr, False)):
        cmd.append(flag)


def _run_coe_scout_proposers(
    *,
    args,
    filepath,
    results_dir,
    base_filename,
    current_stageA_reservoir,
    current_reservoir,
    phase: str = "pre_stageA",
    continuation_ast=None,
    continuation_y_op_name: Optional[str] = None,
    continuation_fit_link_name: Optional[str] = None,
    continuation_fit_link_scale: float = 1.0,
) -> Optional[dict]:
    scout_slices = _parse_coe_scout_slice_ids(args)
    if continuation_ast is not None:
        continuation_count_arg = getattr(args, "coe_continuation_scout_count", None)
        if continuation_count_arg is None:
            continuation_count_arg = getattr(args, "coe_scout_count", 0)
        continuation_count = max(
            0,
            int(continuation_count_arg or 0),
        )
        scout_slices = list(scout_slices)[:continuation_count]
    if not scout_slices:
        return None
    try:
        from nestynet_sr.sr_search.coe_committee import (
            load_stageA_proposal_reservoir_payloads,
            load_proposal_reservoir_payloads,
            merge_stageA_proposal_reservoir_payloads,
            merge_proposal_reservoir_payloads,
            stageA_terminal_proposals_as_expression_reservoir,
        )
    except Exception as exc:
        return {"summary": {"enabled": True, "status": "error", "error": f"could not import reservoir tools: {exc}"}}

    phase_label = str(phase or "pre_stageA")
    phase_safe = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in phase_label).strip("_")
    if not phase_safe:
        phase_safe = "pre_stageA"
    if continuation_ast is None and phase_safe == "pre_stageA":
        scout_root = os.path.join(results_dir, "_coe_scouts", str(base_filename))
    else:
        scout_root = os.path.join(results_dir, "_coe_scouts", str(base_filename), phase_safe)
    os.makedirs(scout_root, exist_ok=True)
    continuation_seed_path = None
    if continuation_ast is not None:
        continuation_seed_path = os.path.join(scout_root, f"{base_filename}.{phase_safe}.seed_ast.pkl")
        try:
            with open(continuation_seed_path, "wb") as f_seed:
                pickle.dump(continuation_ast, f_seed)
        except Exception as exc:
            return {
                "summary": {
                    "enabled": True,
                    "phase": phase_label,
                    "status": "error",
                    "error": f"could not write continuation seed: {type(exc).__name__}: {exc}",
                }
            }
    rows: list[dict] = []
    jobs: list[dict] = []
    loaded_payloads: list[dict] = []
    loaded_stageA_payloads: list[dict] = []
    timeout = float(getattr(args, "coe_scout_timeout_seconds", 0.0) or 0.0)
    problem_stem = _coe_problem_stem(filepath)
    run_sr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_SR.py")
    stageA_only = not bool(getattr(args, "coe_scout_with_stageB", False))

    for sid in scout_slices:
        scout_dir = os.path.join(scout_root, f"slice_{sid:03d}")
        os.makedirs(scout_dir, exist_ok=True)
        scout_log = os.path.join(scout_dir, f"{base_filename}.scout.log")
        cmd = [
            sys.executable,
            "-u",
            run_sr_path,
            "--filepath",
            str(filepath),
            "--results_dir",
            scout_dir,
            "--data_slice",
            str(int(sid)),
            "--coe_mode",
            "audit_final",
            "--coe_num_slices",
            "1",
            "--coe_start_slice",
            str(int(sid)),
            "--coe_max_candidates",
            str(max(1, int(getattr(args, "coe_max_candidates", 16) or 16))),
            "--coe_scout_count",
            "0",
            "--coe_scout_slices",
            "",
        ]
        if stageA_only:
            cmd.append("--no_stageB")
        if continuation_seed_path is not None:
            cmd.extend(
                [
                    "--load_expressions",
                    continuation_seed_path,
                    "--stageA_continuation_seed",
                    "--stageA_continuation_y_op",
                    str(continuation_y_op_name or "identity"),
                    "--no_stageA_auto_fit_link",
                ]
            )
            try:
                cont_fit_link = canonical_fit_link_name(continuation_fit_link_name)
            except Exception as exc:
                return {
                    "summary": {
                        "enabled": True,
                        "phase": phase_label,
                        "status": "error",
                        "error": f"invalid continuation fit-link: {type(exc).__name__}: {exc}",
                    }
                }
            if cont_fit_link is not None:
                try:
                    cont_fit_scale = float(continuation_fit_link_scale)
                except Exception:
                    cont_fit_scale = 1.0
                cmd.extend(
                    [
                        "--stageA_continuation_fit_link",
                        str(cont_fit_link),
                        "--stageA_continuation_fit_link_scale",
                        f"{cont_fit_scale:.17g}",
                    ]
                )
        if getattr(args, "log_level", None) is not None:
            cmd.extend(["--log_level", str(args.log_level)])
        if getattr(args, "noise_sigma_frac_y_rms", None) is not None:
            cmd.extend(["--noise_sigma_frac_y_rms", str(args.noise_sigma_frac_y_rms)])
        if getattr(args, "ndata_train", None) is not None:
            cmd.extend(["--ndata_train", str(args.ndata_train)])
        if getattr(args, "ndata_val", None) is not None:
            cmd.extend(["--ndata_val", str(args.ndata_val)])
        if getattr(args, "batch_size", None) is not None:
            cmd.extend(["--batch_size", str(args.batch_size)])
        if getattr(args, "equations_txt", None) is not None:
            cmd.extend(["--equations_txt", str(args.equations_txt)])
        if continuation_seed_path is not None:
            cmd.extend(["--force_y_ops", str(continuation_y_op_name or "identity")])
        elif getattr(args, "force_y_ops", None) is not None:
            cmd.extend(["--force_y_ops", str(args.force_y_ops)])
        if getattr(args, "disable_stageB_patterns", None) is not None:
            cmd.extend(["--disable_stageB_patterns", str(args.disable_stageB_patterns)])
        if not stageA_only:
            cmd.extend(["--stageB_epochs", str(max(1, int(getattr(args, "coe_scout_stageB_epochs", 800) or 800)))])
            cmd.extend(["--stageB_max_outer_iters", str(max(1, int(getattr(args, "coe_scout_stageB_max_outer_iters", 12) or 12)))])
        cmd.extend(["--max_ab_iters", str(max(1, int(getattr(args, "coe_scout_max_ab_iters", 1) or 1)))])
        scout_stageA_max_passes = max(
            0,
            int(getattr(args, "coe_scout_stageA_max_passes", 1) or 0),
        )
        if scout_stageA_max_passes > 0:
            cmd.extend(["--stageA_max_passes", str(scout_stageA_max_passes)])
        if not bool(getattr(args, "coe_scout_final_polish", False)):
            cmd.append("--no_final_polish")
        _maybe_add_flag(cmd, args, "ignore_units", "--ignore_units")
        _maybe_add_flag(cmd, args, "single_layer", "--single_layer")
        if not bool(getattr(args, "stageA_separabilities", True)):
            cmd.append("--no_stageA_separabilities")
        if bool(getattr(args, "stageB_overcap_fallback", False)):
            cmd.append("--stageB_overcap_fallback")
        if bool(getattr(args, "canonical_init", False)):
            cmd.append("--canonical_init")
        if bool(getattr(args, "evidence", False)):
            cmd.append("--evidence")
        if bool(getattr(args, "evidence_disable_residual_whitening", False)):
            cmd.append("--evidence_disable_residual_whitening")
        if bool(getattr(args, "evidence_disable_segment_priors", False)):
            cmd.append("--evidence_disable_segment_priors")
        if not bool(getattr(args, "evidence_prior_decay_auto", True)):
            cmd.append("--no_evidence_prior_decay_auto")
        if not bool(getattr(args, "evidence_metric_gate", True)):
            cmd.append("--no_evidence_metric_gate")
        if getattr(args, "use_factorized_search", None) is True:
            cmd.append("--factorized-search")
        elif getattr(args, "use_factorized_search", None) is False:
            cmd.append("--no-factorized-search")
        if getattr(args, "use_refine_skeleton", None) is True:
            cmd.append("--refine-skeleton")
        elif getattr(args, "use_refine_skeleton", None) is False:
            cmd.append("--no-refine-skeleton")

        print(f"[CoE scouts] Queued scout slice {sid}: log={scout_log}")
        row = {
            "slice_id": int(sid),
            "results_dir": scout_dir,
            "log_path": scout_log,
            "phase": phase_label,
            "continuation_seed": bool(continuation_seed_path is not None),
            "status": "unknown",
        }
        if continuation_seed_path is not None:
            row["continuation_seed_path"] = continuation_seed_path
            row["continuation_y_op_name"] = str(continuation_y_op_name or "identity")
            row["continuation_fit_link"] = canonical_fit_link_name(continuation_fit_link_name)
            row["continuation_fit_link_scale"] = (
                float(continuation_fit_link_scale)
                if canonical_fit_link_name(continuation_fit_link_name) is not None
                else None
            )
        jobs.append(
            {
                "slice_id": int(sid),
                "cmd": cmd,
                "row": row,
                "scout_dir": scout_dir,
                "scout_log": scout_log,
            }
        )

    scout_parallelism = max(1, int(getattr(args, "coe_scout_parallelism", 1) or 1))
    scout_parallelism = min(scout_parallelism, max(1, len(jobs)))
    try:
        scout_threads = max(1, int(os.environ.get("COE_WORKER_THREADS", "1")))
    except (TypeError, ValueError):
        scout_threads = 1
    scout_thread_env = {
        "OMP_NUM_THREADS": str(scout_threads),
        "MKL_NUM_THREADS": str(scout_threads),
        "OPENBLAS_NUM_THREADS": str(scout_threads),
        "VECLIB_MAXIMUM_THREADS": str(scout_threads),
        "NUMEXPR_NUM_THREADS": str(scout_threads),
    }
    scout_env = os.environ.copy()
    scout_env.update(scout_thread_env)
    if jobs:
        print(
            "[CoE scouts] Running "
            f"{len(jobs)} scout subprocess(es) with parallelism={scout_parallelism} "
            f"and BLAS/OpenMP thread caps={scout_threads}"
        )

    def _finish_scout_job(job: dict, *, status: str, returncode=None, error: Optional[str] = None) -> None:
        f_log = job.pop("log_file", None)
        if f_log is not None:
            try:
                f_log.close()
            except Exception:
                pass
        row = job["row"]
        row["status"] = str(status)
        if returncode is not None:
            try:
                row["returncode"] = int(returncode)
            except Exception:
                row["returncode"] = returncode
        elif status == "timeout":
            row["returncode"] = None
        if error:
            row["error"] = str(error)
        t0 = float(job.get("t0", timeit.default_timer()))
        row["walltime_seconds"] = float(timeit.default_timer() - t0)
        payloads, warnings = load_proposal_reservoir_payloads(
            [job["scout_dir"]],
            problem_stem=problem_stem,
        )
        stageA_payloads, stageA_warnings = load_stageA_proposal_reservoir_payloads(
            [job["scout_dir"]],
            problem_stem=problem_stem,
        )
        row["reservoir_payloads"] = int(len(payloads))
        row["stageA_reservoir_payloads"] = int(len(stageA_payloads))
        if warnings:
            row["warnings"] = list(warnings)
        if stageA_warnings:
            row.setdefault("warnings", [])
            row["warnings"].extend(stageA_warnings)
        loaded_payloads.extend(payloads)
        loaded_stageA_payloads.extend(stageA_payloads)
        rows.append(row)

    pending = list(jobs)
    active: list[dict] = []
    while pending or active:
        while pending and len(active) < scout_parallelism:
            job = pending.pop(0)
            job["t0"] = timeit.default_timer()
            try:
                f_log = open(job["scout_log"], "w")
                job["log_file"] = f_log
                job["proc"] = subprocess.Popen(
                    job["cmd"],
                    stdout=f_log,
                    stderr=subprocess.STDOUT,
                    env=scout_env,
                )
                active.append(job)
            except Exception as exc:
                _finish_scout_job(
                    job,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
        still_active: list[dict] = []
        now = timeit.default_timer()
        for job in active:
            proc = job.get("proc")
            if proc is None:
                _finish_scout_job(job, status="error", error="missing subprocess handle")
                continue
            returncode = proc.poll()
            if returncode is not None:
                _finish_scout_job(
                    job,
                    status="success" if returncode == 0 else "failed",
                    returncode=returncode,
                )
                continue
            if timeout > 0.0 and (now - float(job.get("t0", now))) >= timeout:
                try:
                    proc.kill()
                    returncode = proc.wait(timeout=10.0)
                except Exception:
                    returncode = None
                _finish_scout_job(job, status="timeout", returncode=returncode)
                continue
            still_active.append(job)
        active = still_active
        if pending or active:
            time.sleep(0.25)

    order = {int(sid): idx for idx, sid in enumerate(scout_slices)}
    rows.sort(key=lambda row: order.get(int(row.get("slice_id", -1)), len(order)))

    merged = None
    merged_stageA = None
    if loaded_payloads:
        payload_inputs = []
        if isinstance(current_reservoir, dict):
            payload_inputs.append(current_reservoir)
        payload_inputs.extend(loaded_payloads)
        merged = merge_proposal_reservoir_payloads(
            payload_inputs,
            max_candidates=max(1, int(getattr(args, "coe_max_candidates", 16) or 16) * 2),
        )
        merged["source"] = "reference_refine_scout_reservoir"
        merged["scout_slice_ids"] = [int(x) for x in scout_slices]
    if loaded_stageA_payloads:
        stageA_inputs = []
        if isinstance(current_stageA_reservoir, dict):
            stageA_inputs.append(current_stageA_reservoir)
        stageA_inputs.extend(loaded_stageA_payloads)
        merged_stageA = merge_stageA_proposal_reservoir_payloads(
            stageA_inputs,
            max_candidates=max(1, int(getattr(args, "coe_max_candidates", 16) or 16) * 4),
        )
        merged_stageA["source"] = "reference_refine_scout_stageA_reservoir"
        merged_stageA["scout_slice_ids"] = [int(x) for x in scout_slices]
        stageA_terminal_exprs = stageA_terminal_proposals_as_expression_reservoir(
            merged_stageA,
            max_candidates=max(1, int(getattr(args, "coe_max_candidates", 16) or 16) * 2),
        )
        if stageA_terminal_exprs.get("candidates"):
            expr_inputs = []
            if isinstance(merged, dict):
                expr_inputs.append(merged)
            elif isinstance(current_reservoir, dict):
                expr_inputs.append(current_reservoir)
            expr_inputs.append(stageA_terminal_exprs)
            merged = merge_proposal_reservoir_payloads(
                expr_inputs,
                max_candidates=max(1, int(getattr(args, "coe_max_candidates", 16) or 16) * 2),
            )
            merged["source"] = "reference_refine_scout_reservoir"
            merged["scout_slice_ids"] = [int(x) for x in scout_slices]
            merged["stageA_terminal_payload_count"] = len(
                list(stageA_terminal_exprs.get("candidates") or [])
            )

    summary = {
        "enabled": True,
        "mode": "reservoir_discovery",
        "phase": phase_label,
        "stageA_only": bool(stageA_only),
        "parallelism": int(scout_parallelism),
        "thread_env": dict(scout_thread_env),
        "continuation_seed": bool(continuation_seed_path is not None),
        "continuation_y_op_name": (
            str(continuation_y_op_name or "identity")
            if continuation_seed_path is not None
            else None
        ),
        "continuation_fit_link": (
            canonical_fit_link_name(continuation_fit_link_name)
            if continuation_seed_path is not None
            else None
        ),
        "continuation_fit_link_scale": (
            float(continuation_fit_link_scale)
            if (
                continuation_seed_path is not None
                and canonical_fit_link_name(continuation_fit_link_name) is not None
            )
            else None
        ),
        "requested_slices": [int(x) for x in scout_slices],
        "completed": sum(1 for r in rows if r.get("status") == "success"),
        "failed": sum(1 for r in rows if r.get("status") not in {"success"}),
        "loaded_payloads": int(len(loaded_payloads)),
        "loaded_stageA_payloads": int(len(loaded_stageA_payloads)),
        "scout_root": scout_root,
        "rows": rows,
    }
    if merged is not None:
        summary["merged_total_unique"] = merged.get("total_unique")
        summary["merged_candidate_count"] = len(list(merged.get("candidates") or []))
        summary["stageA_terminal_payload_count"] = merged.get("stageA_terminal_payload_count")
    if merged_stageA is not None:
        summary["merged_stageA_total_unique"] = merged_stageA.get("total_unique")
        summary["merged_stageA_candidate_count"] = len(list(merged_stageA.get("candidates") or []))
    return {"summary": summary, "merged_reservoir": merged, "merged_stageA_reservoir": merged_stageA}


def _merge_coe_expression_reservoir_payload(
    current: Optional[dict],
    payload: Optional[dict],
    *,
    max_candidates: int,
    source: str,
) -> Optional[dict]:
    """Merge fixed-expression scout reservoirs without dropping earlier scout artifacts."""

    if not isinstance(payload, dict):
        return current if isinstance(current, dict) else None
    try:
        candidates = list(payload.get("candidates") or [])
    except Exception:
        candidates = []
    if not candidates:
        return current if isinstance(current, dict) else None
    try:
        from nestynet_sr.sr_search.coe_committee import merge_proposal_reservoir_payloads

        inputs = []
        if isinstance(current, dict):
            inputs.append(current)
        inputs.append(payload)
        merged = merge_proposal_reservoir_payloads(
            inputs,
            max_candidates=max(1, int(max_candidates)),
        )
        merged["source"] = str(source)
        return _make_json_serializable(merged)
    except Exception as exc:
        out = dict(current) if isinstance(current, dict) else dict(payload)
        out.setdefault("warnings", []).append(
            f"could not merge scout expression reservoir: {type(exc).__name__}: {exc}"
        )
        return _make_json_serializable(out)


def _format_coe_committee_report(summary: dict) -> str:
    lines = ["=== CoE Final Committee ==="]
    status = str(summary.get("status", "unknown"))
    mode = str(summary.get("mode", "off"))
    lines.append(f"mode={mode} status={status}")
    if summary.get("reason"):
        lines.append(f"reason: {summary.get('reason')}")
    if summary.get("error"):
        lines.append(f"error: {summary.get('error')}")
    lines.append(
        f"candidates={int(summary.get('n_candidates', 0) or 0)} "
        f"slices={int(summary.get('n_slices', 0) or 0)}"
    )
    inc = summary.get("incumbent_id")
    rec = summary.get("recommended_id")
    if inc is not None or rec is not None:
        lines.append(f"incumbent={inc} recommended={rec}")
    cfg = summary.get("config") or {}
    if cfg.get("selection_basis"):
        lines.append(f"selection_basis={cfg.get('selection_basis')}")
    if summary.get("recommended_expr"):
        lines.append(f"recommended_expr={summary.get('recommended_expr')}")
    cand_summary = list(summary.get("candidate_summary") or [])
    cand_summary.sort(
        key=lambda row: (
            float(row.get("median_val_mse", float("inf"))),
            float(((row.get("candidate") or {}).get("complexity", float("inf")) or float("inf"))),
        )
    )
    for row in cand_summary[:5]:
        cand = row.get("candidate") or {}
        val = row.get("median_val_mse")
        cx = cand.get("complexity")
        lines.append(
            "  "
            f"{cand.get('candidate_id')}: "
            f"median_val_mse={float(val):.4e} "
            f"complexity={float(cx):.3g} "
            f"support={int(row.get('reservoir_support_count', 1) or 1)} "
            f"source={cand.get('source')} "
            f"noise_tied={bool(row.get('noise_tied_with_best', False))}"
        )
    for warn in list(summary.get("warnings") or [])[:4]:
        lines.append(f"warning: {warn}")
    if summary.get("jsonl_path"):
        lines.append(f"jsonl: {summary.get('jsonl_path')}")
    return "\n".join(lines)


def _update_report_with_coe_committee(report_path: str, summary: dict) -> None:
    try:
        with open(report_path, "r") as f:
            report = json.load(f)
    except Exception as e:
        print(f"[CoE] Warning: could not read report for update: {e}")
        return
    report["coe_committee"] = _make_json_serializable(summary)
    try:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[CoE] Updated JSON report with final committee audit: {report_path}")
    except Exception as e:
        print(f"[CoE] Warning: could not update report: {e}")


def _summarize_coe_stageB_dry_run(records, *, enabled: bool) -> dict:
    from collections import Counter

    rows = list(records or [])
    by_tag = Counter()
    by_rule = Counter()
    by_label = Counter()
    for rec in rows:
        for tag in list(rec.get("risk_tags") or []):
            by_tag[str(tag)] += 1
        by_rule[str(rec.get("rule", ""))] += 1
        by_label[str(rec.get("label", ""))] += 1
    return {
        "enabled": bool(enabled),
        "mode": "dry_run",
        "total": len(rows),
        "by_tag": dict(sorted(by_tag.items())),
        "by_rule": dict(sorted(by_rule.items())),
        "by_label": dict(sorted(by_label.items())),
        "would_change_decisions": 0,
        "sample": rows[:5],
    }


def _write_coe_stageB_dry_run_jsonl(path: str, records) -> Optional[str]:
    rows = list(records or [])
    if not rows:
        return None
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for rec in rows:
                f.write(json.dumps(_make_json_serializable(rec), sort_keys=True) + "\n")
        return path
    except Exception as e:
        print(f"[CoE StageB dry-run] Warning: could not write JSONL: {e}")
        return None


def _format_coe_stageB_dry_run_report(summary: dict) -> str:
    lines = ["=== CoE Stage B Dry-Run ==="]
    lines.append(
        f"enabled={bool(summary.get('enabled', False))} "
        f"mode={summary.get('mode', 'dry_run')} "
        f"observed_risky_accepts={int(summary.get('total', 0) or 0)}"
    )
    by_tag = summary.get("by_tag") or {}
    if by_tag:
        lines.append(
            "risk_tags: "
            + ", ".join(f"{k}={v}" for k, v in sorted(by_tag.items()))
        )
    by_rule = summary.get("by_rule") or {}
    if by_rule:
        top_rules = sorted(by_rule.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:5]
        lines.append(
            "top_rules: "
            + ", ".join(f"{k}={v}" for k, v in top_rules)
        )
    if summary.get("jsonl_path"):
        lines.append(f"jsonl: {summary.get('jsonl_path')}")
    return "\n".join(lines)


def _summarize_coe_stageB_gate(records, *, enabled: bool) -> dict:
    from collections import Counter

    rows = list(records or [])
    by_tag = Counter()
    by_rule = Counter()
    by_status = Counter()
    by_outcome = Counter()
    for rec in rows:
        for tag in list(rec.get("risk_tags") or []):
            by_tag[str(tag)] += 1
        by_rule[str(rec.get("rule", ""))] += 1
        by_status[str(rec.get("gate_status", ""))] += 1
        by_outcome[str(rec.get("outcome", ""))] += 1
    return {
        "enabled": bool(enabled),
        "mode": "committee_gated",
        "total": len(rows),
        "vetoes": int(by_outcome.get("veto", 0)),
        "allows": int(by_outcome.get("allow", 0)),
        "unsupported": int(
            sum(v for k, v in by_status.items() if "unsupported" in str(k))
        ),
        "by_tag": dict(sorted(by_tag.items())),
        "by_rule": dict(sorted(by_rule.items())),
        "by_status": dict(sorted(by_status.items())),
        "by_outcome": dict(sorted(by_outcome.items())),
        "sample": rows[:5],
    }


def _write_coe_stageB_gate_jsonl(path: str, records) -> Optional[str]:
    rows = list(records or [])
    if not rows:
        return None
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for rec in rows:
                f.write(json.dumps(_make_json_serializable(rec), sort_keys=True) + "\n")
        return path
    except Exception as e:
        print(f"[CoE StageB gate] Warning: could not write JSONL: {e}")
        return None


def _format_coe_stageB_gate_report(summary: dict) -> str:
    lines = ["=== CoE Stage B Gate ==="]
    lines.append(
        f"enabled={bool(summary.get('enabled', False))} "
        f"mode={summary.get('mode', 'committee_gated')} "
        f"events={int(summary.get('total', 0) or 0)} "
        f"allows={int(summary.get('allows', 0) or 0)} "
        f"vetoes={int(summary.get('vetoes', 0) or 0)} "
        f"unsupported={int(summary.get('unsupported', 0) or 0)}"
    )
    by_tag = summary.get("by_tag") or {}
    if by_tag:
        lines.append(
            "risk_tags: "
            + ", ".join(f"{k}={v}" for k, v in sorted(by_tag.items()))
        )
    by_status = summary.get("by_status") or {}
    if by_status:
        lines.append(
            "gate_status: "
            + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items()))
        )
    if summary.get("jsonl_path"):
        lines.append(f"jsonl: {summary.get('jsonl_path')}")
    return "\n".join(lines)
