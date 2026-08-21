# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Domain-tube projection for scoring domain-restricted symbolic candidates.

The strict tuple-AST evaluator intentionally remains unchanged.  This module is
an opt-in scoring/validation policy for DE discovery, where a surrogate may put
a physically nonnegative state a tiny distance below zero near a boundary.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from .expr_ast import eval_node


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_bool(cfg: Any, key: str, default: bool = False) -> bool:
    try:
        return bool(_cfg_get(cfg, key, default))
    except Exception:
        return bool(default)


def _cfg_float(cfg: Any, key: str, default: float) -> float:
    try:
        out = float(_cfg_get(cfg, key, default))
    except Exception:
        out = float(default)
    if not math.isfinite(out):
        out = float(default)
    return float(out)


def domain_projection_enabled(cfg: Any) -> bool:
    return _cfg_bool(cfg, "score_domain_projection_enable", False)


def _tensor_scale(value: torch.Tensor) -> float:
    try:
        finite = value.detach()[torch.isfinite(value.detach())]
        if int(finite.numel()) <= 0:
            return 1.0
        abs_v = torch.abs(finite.reshape(-1))
        mean_abs = float(torch.mean(abs_v).detach().cpu().item())
        max_abs = float(torch.max(abs_v).detach().cpu().item())
        mean = torch.mean(finite)
        std = float(torch.sqrt(torch.mean((finite - mean) ** 2)).detach().cpu().item())
        return max(1.0e-12, mean_abs, 0.1 * max_abs, std)
    except Exception:
        return 1.0


def _empty_diag(cfg: Any, *, enabled: bool, rows: int) -> dict[str, Any]:
    abs_tol = max(0.0, _cfg_float(cfg, "score_domain_projection_abs_tol", 1.0e-8))
    rel_tol = max(0.0, _cfg_float(cfg, "score_domain_projection_rel_tol", 1.0e-8))
    reference_scale = _cfg_float(cfg, "score_domain_projection_reference_scale", float("nan"))
    if (not math.isfinite(reference_scale)) or reference_scale <= 0.0:
        reference_scale = None
    max_frac = _cfg_float(cfg, "score_domain_projection_max_frac", 1.0)
    if max_frac < 0.0 or not math.isfinite(max_frac):
        max_frac = 1.0
    max_frac = min(1.0, max(0.0, max_frac))
    return {
        "enabled": bool(enabled),
        "applied": False,
        "ok": True,
        "status": "disabled" if not enabled else "no_projection_needed",
        "projected_rows": 0,
        "total_rows": int(max(0, rows)),
        "projected_frac": 0.0,
        "max_violation": 0.0,
        "mean_violation": 0.0,
        "abs_tol": float(abs_tol),
        "rel_tol": float(rel_tol),
        "reference_scale": None if reference_scale is None else float(reference_scale),
        "max_frac": float(max_frac),
        "ops": [],
    }


def _record_projection(
    diag: dict[str, Any],
    *,
    op: str,
    raw: torch.Tensor,
    bad_mask: torch.Tensor,
    violation: torch.Tensor,
) -> None:
    try:
        bad_flat = bad_mask.reshape(-1)
        n_bad = int(bad_flat.sum().detach().cpu().item())
        n_total = int(bad_flat.numel())
    except Exception:
        n_bad = 0
        n_total = 0
    if n_bad <= 0 or n_total <= 0:
        return

    try:
        v = violation.reshape(-1)[bad_flat]
        max_v = float(torch.max(v).detach().cpu().item())
        mean_v = float(torch.mean(v).detach().cpu().item())
    except Exception:
        max_v = float("inf")
        mean_v = float("inf")
    try:
        reference_scale = float(diag.get("reference_scale", float("nan")))
    except Exception:
        reference_scale = float("nan")
    if math.isfinite(reference_scale) and reference_scale > 0.0:
        scale = float(reference_scale)
    else:
        scale = _tensor_scale(raw)
    tube_tol = float(diag.get("abs_tol", 0.0)) + float(diag.get("rel_tol", 0.0)) * float(scale)
    frac = float(n_bad) / float(n_total)
    max_frac = float(diag.get("max_frac", 1.0))
    op_ok = math.isfinite(max_v) and max_v <= tube_tol and frac <= max_frac

    diag["applied"] = True
    diag["projected_rows"] = int(diag.get("projected_rows", 0)) + int(n_bad)
    diag["total_rows"] = int(diag.get("total_rows", 0)) + 0
    diag["projected_frac"] = (
        float(diag["projected_rows"]) / float(max(1, int(diag.get("total_rows", n_total))))
    )
    diag["max_violation"] = max(float(diag.get("max_violation", 0.0)), float(max_v))
    prev_rows = max(0, int(diag.get("_mean_rows", 0)))
    prev_mean = float(diag.get("mean_violation", 0.0))
    denom = max(1, prev_rows + n_bad)
    diag["mean_violation"] = float((prev_mean * prev_rows + mean_v * n_bad) / float(denom))
    diag["_mean_rows"] = int(denom)
    if not op_ok:
        diag["ok"] = False
        diag["status"] = "rejected_outside_tube"
    elif bool(diag.get("ok", True)):
        diag["status"] = "projected_within_tube"
    diag.setdefault("ops", []).append(
        {
            "op": str(op),
            "projected_rows": int(n_bad),
            "total_rows": int(n_total),
            "projected_frac": float(frac),
            "max_violation": float(max_v),
            "mean_violation": float(mean_v),
            "argument_scale": float(scale),
            "tube_tol": float(tube_tol),
            "ok": bool(op_ok),
        }
    )


def _finalize_diag(diag: dict[str, Any]) -> dict[str, Any]:
    diag.pop("_mean_rows", None)
    total = max(1, int(diag.get("total_rows", 0)))
    diag["projected_frac"] = float(int(diag.get("projected_rows", 0))) / float(total)
    if bool(diag.get("enabled", False)) and not bool(diag.get("applied", False)):
        diag["status"] = "no_projection_needed"
    return diag


def domain_projection_is_acceptable(diag: Mapping[str, Any] | None) -> bool:
    if not isinstance(diag, Mapping):
        return True
    if not bool(diag.get("enabled", False)):
        return True
    return bool(diag.get("ok", True))


def merge_domain_projection_diagnostics(
    *diags: Mapping[str, Any] | None,
    labels: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    valid = [d for d in diags if isinstance(d, Mapping)]
    if not valid:
        return {"enabled": False, "applied": False, "ok": True, "status": "disabled"}
    enabled = any(bool(d.get("enabled", False)) for d in valid)
    out = {
        "enabled": bool(enabled),
        "applied": any(bool(d.get("applied", False)) for d in valid),
        "ok": all(domain_projection_is_acceptable(d) for d in valid),
        "status": "disabled" if not enabled else "no_projection_needed",
        "projected_rows": 0,
        "total_rows": 0,
        "projected_frac": 0.0,
        "max_violation": 0.0,
        "mean_violation": 0.0,
        "ops": [],
    }
    mean_num = 0.0
    mean_den = 0
    for i, d in enumerate(valid):
        projected_rows = int(d.get("projected_rows", 0) or 0)
        total_rows = int(d.get("total_rows", 0) or 0)
        out["projected_rows"] = int(out["projected_rows"]) + projected_rows
        out["total_rows"] = int(out["total_rows"]) + total_rows
        try:
            out["max_violation"] = max(float(out["max_violation"]), float(d.get("max_violation", 0.0) or 0.0))
        except Exception:
            out["max_violation"] = float("inf")
        if projected_rows > 0:
            try:
                mean_num += float(d.get("mean_violation", 0.0) or 0.0) * projected_rows
                mean_den += projected_rows
            except Exception:
                pass
        split = None
        if labels is not None and i < len(labels):
            split = str(labels[i])
        for op_row in list(d.get("ops", []) or []):
            if not isinstance(op_row, Mapping):
                continue
            row = dict(op_row)
            if split:
                row["split"] = split
            out["ops"].append(row)
    out["projected_frac"] = float(out["projected_rows"]) / float(max(1, int(out["total_rows"])))
    out["mean_violation"] = float(mean_num / float(max(1, mean_den)))
    if not bool(out["ok"]):
        out["status"] = "rejected_outside_tube"
    elif bool(out["applied"]):
        out["status"] = "projected_within_tube"
    return out


def eval_node_with_domain_projection(
    node: Any,
    x: torch.Tensor,
    cfg: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Evaluate ``node`` with opt-in projection for restricted-domain ops."""

    enabled = domain_projection_enabled(cfg)
    if not enabled:
        return eval_node(node, x), _empty_diag(cfg, enabled=False, rows=int(x.shape[0]))

    diag = _empty_diag(cfg, enabled=True, rows=int(x.shape[0]))
    pos_floor = max(0.0, _cfg_float(cfg, "score_domain_projection_positive_floor", 1.0e-12))
    if pos_floor <= 0.0:
        pos_floor = 1.0e-12

    def _eval(cur: Any) -> torch.Tensor:
        op = cur[0]
        if op == "var":
            i = int(cur[1])
            return x[:, i : i + 1]
        if op == "const":
            return torch.full((x.shape[0], 1), float(cur[1]), dtype=x.dtype, device=x.device)
        if op == "sin":
            return torch.sin(_eval(cur[1]))
        if op == "cos":
            return torch.cos(_eval(cur[1]))
        if op == "exp":
            return torch.exp(_eval(cur[1]))
        if op == "sqr":
            c = _eval(cur[1])
            return c * c
        if op == "neg":
            return -_eval(cur[1])
        if op == "sqrt":
            raw = _eval(cur[1])
            finite = torch.isfinite(raw)
            bad = finite & (raw < 0.0)
            if bool(bad.any().detach().cpu().item()):
                violation = torch.clamp(-raw, min=0.0)
                _record_projection(diag, op="sqrt", raw=raw, bad_mask=bad, violation=violation)
            return torch.sqrt(torch.where(bad, torch.zeros_like(raw), raw))
        if op == "log":
            raw = _eval(cur[1])
            finite = torch.isfinite(raw)
            floor = torch.full_like(raw, float(pos_floor))
            bad = finite & (raw < floor)
            if bool(bad.any().detach().cpu().item()):
                violation = torch.clamp(floor - raw, min=0.0)
                _record_projection(diag, op="log", raw=raw, bad_mask=bad, violation=violation)
            return torch.log(torch.where(bad, floor, raw))
        if op == "asin":
            raw = _eval(cur[1])
            finite = torch.isfinite(raw)
            bad = finite & ((raw < -1.0) | (raw > 1.0))
            if bool(bad.any().detach().cpu().item()):
                violation = torch.where(raw < -1.0, -1.0 - raw, raw - 1.0)
                violation = torch.clamp(violation, min=0.0)
                _record_projection(diag, op="asin", raw=raw, bad_mask=bad, violation=violation)
            return torch.asin(torch.clamp(raw, min=-1.0, max=1.0))
        if op == "acos":
            raw = _eval(cur[1])
            finite = torch.isfinite(raw)
            bad = finite & ((raw < -1.0) | (raw > 1.0))
            if bool(bad.any().detach().cpu().item()):
                violation = torch.where(raw < -1.0, -1.0 - raw, raw - 1.0)
                violation = torch.clamp(violation, min=0.0)
                _record_projection(diag, op="acos", raw=raw, bad_mask=bad, violation=violation)
            return torch.acos(torch.clamp(raw, min=-1.0, max=1.0))
        if op == "add":
            return _eval(cur[1]) + _eval(cur[2])
        if op == "sub":
            return _eval(cur[1]) - _eval(cur[2])
        if op == "mul":
            return _eval(cur[1]) * _eval(cur[2])
        if op == "div":
            return _eval(cur[1]) / _eval(cur[2])
        return eval_node(cur, x)

    return _eval(node), _finalize_diag(diag)
