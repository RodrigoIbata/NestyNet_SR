# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Core factorized symbolic search scoring implementation extracted from explorer."""

from __future__ import annotations

import math
import time
from typing import Any, Mapping

import torch

from ..basis_scoring import make_additive_basis_transition, snap_direct_coeff
from ..domain_projection import (
    domain_projection_is_acceptable,
    eval_node_with_domain_projection,
    merge_domain_projection_diagnostics,
)
from ..expr_ast import (
    BINARY_OPS,
    UNARY_OPS,
    collect_paths,
    dims_eq,
    eval_node,
    get_at,
    node_cost_physics_prior,
    node_depth,
    node_dims,
    node_size,
    node_str,
    simplify,
)
from ..expr_mapping import (
    _mapping_nparams,
    eval_mapping,
    fit_best,
    fit_poly,
    mapping_is_structural,
)

_LEGACY_REFINEMENT_HELPERS = [
    "_decorate_refine_variants",
    "_materialize_linearized_candidate",
    "_refine_hparams",
    "_variant_has_gate_potential",
]

_SCORE_LADDER_SCHEMA_VERSION = 1


def _coerce_refinement_helpers(cfg: Mapping | None) -> dict[str, object]:
    """Return explicit hooks for the refinement code that is not engine-owned yet."""
    hooks = None
    if isinstance(cfg, Mapping):
        hooks = cfg.get("_legacy_refinement_hooks", None)
    if not isinstance(hooks, Mapping):
        raise RuntimeError(
            "continuous skeleton refinement requires explicit _legacy_refinement_hooks; "
            "call via factorized_search.explorer.score_expr or pass refinement hooks directly"
        )
    out = {}
    for name in _LEGACY_REFINEMENT_HELPERS:
        helper = hooks.get(name, None)
        if helper is None:
            raise RuntimeError(f"missing refinement hook {name!r}")
        out[name] = helper
    return out


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def _probe_mse_or_none(y_true, y_pred) -> float | None:
    try:
        if y_pred is None or (not torch.is_tensor(y_pred)):
            return None
        r = (y_true - y_pred.reshape_as(y_true)).squeeze(-1)
        return _finite_or_none((r * r).mean().detach().cpu().item())
    except Exception:
        return None


def _as_score_col(value: Any, *, rows: int | None = None):
    if value is None or (not torch.is_tensor(value)):
        return None
    out = value
    if out.dim() == 1:
        out = out.reshape(-1, 1)
    elif out.dim() == 2 and int(out.shape[1]) == 1:
        pass
    else:
        if rows is None:
            return None
        try:
            out = out.reshape(int(rows), -1)
        except Exception:
            return None
        if out.dim() != 2 or int(out.shape[1]) != 1:
            return None
    if rows is not None and int(out.shape[0]) != int(rows):
        return None
    return out


def _finite_mask_threshold(cfg: Mapping | None, key: str, default: float) -> float:
    try:
        val = float((cfg or {}).get(key, default))
    except Exception:
        val = float(default)
    if not math.isfinite(val):
        val = float(default)
    return min(1.0, max(0.0, float(val)))


def _finite_mask_min_points(cfg: Mapping | None) -> int:
    try:
        out = int((cfg or {}).get("score_finite_mask_min_points", 8))
    except Exception:
        out = 8
    return max(1, int(out))


def _mask_fraction(mask) -> float:
    try:
        n = int(mask.numel())
        if n <= 0:
            return 0.0
        return float(mask.sum().detach().cpu().item()) / float(n)
    except Exception:
        return 0.0


def _finite_mask_ok(mask, *, min_frac: float, min_points: int) -> bool:
    try:
        n_valid = int(mask.sum().detach().cpu().item())
        n_total = int(mask.numel())
    except Exception:
        return False
    if n_valid < int(min_points):
        return False
    if n_total <= 0:
        return False
    return float(n_valid) / float(n_total) >= float(min_frac)


def _finite_mask_diag(
    *,
    enabled: bool,
    fit_mask=None,
    probe_mask=None,
    per_dataset: list[dict[str, Any]] | None = None,
    rejected_reason: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"enabled": bool(enabled)}
    if fit_mask is not None:
        out.update(
            {
                "fit_valid_frac": float(_mask_fraction(fit_mask)),
                "fit_valid_rows": int(fit_mask.sum().detach().cpu().item()),
                "fit_total_rows": int(fit_mask.numel()),
            }
        )
    if probe_mask is not None:
        out.update(
            {
                "probe_valid_frac": float(_mask_fraction(probe_mask)),
                "probe_valid_rows": int(probe_mask.sum().detach().cpu().item()),
                "probe_total_rows": int(probe_mask.numel()),
            }
        )
    if per_dataset:
        out["per_dataset"] = list(per_dataset)
        out["min_dataset_valid_frac"] = min(
            float(row.get("probe_valid_frac", 0.0)) for row in per_dataset
        )
    if rejected_reason:
        out["rejected_reason"] = str(rejected_reason)
    return out


def _attach_finite_mask_diag(mapping: Mapping | None, diag: Mapping | None) -> dict:
    out = dict(mapping or {})
    if isinstance(diag, Mapping) and bool(diag.get("enabled", False)):
        out["_finite_mask"] = dict(diag)
    return out


def _attach_domain_projection_diag(mapping: Mapping | None, diag: Mapping | None) -> dict:
    out = dict(mapping or {})
    if isinstance(diag, Mapping) and bool(diag.get("enabled", False)):
        out["_domain_projection"] = dict(diag)
    return out


def _with_score_diagnostics(score_tuple, finite_diag: Mapping | None, domain_diag: Mapping | None):
    if score_tuple is None or len(score_tuple) < 4:
        return score_tuple
    out = list(score_tuple)
    mapping = _attach_finite_mask_diag(out[3], finite_diag)
    mapping = _attach_domain_projection_diag(mapping, domain_diag)
    out[3] = mapping
    return tuple(out)


def _with_finite_mask_diag(score_tuple, diag: Mapping | None):
    if score_tuple is None or len(score_tuple) < 4:
        return score_tuple
    out = list(score_tuple)
    out[3] = _attach_finite_mask_diag(out[3], diag)
    return tuple(out)


def _head_term_count(head: Any) -> int:
    if not isinstance(head, Mapping):
        return 0
    terms = head.get("terms", None)
    if isinstance(terms, (list, tuple)):
        return int(len(terms))
    coeffs = head.get("coeffs", None)
    if isinstance(coeffs, (list, tuple)):
        return max(0, int(len(coeffs)) - 1)
    return 0


def _head_score_decomp(
    *,
    mse_core: Any,
    mse_with_head: Any,
    head: Any,
    core_pred_probe: Any,
    head_pred_probe: Any,
) -> dict[str, Any]:
    out = {
        "mse_core": _finite_or_none(mse_core),
        "mse_with_head": _finite_or_none(mse_with_head),
        "head_rel_gain": _rel_improve(mse_core, mse_with_head),
        "head_energy_frac": None,
        "n_head_terms": int(_head_term_count(head)),
    }
    try:
        if torch.is_tensor(core_pred_probe) and torch.is_tensor(head_pred_probe):
            full = core_pred_probe.reshape_as(head_pred_probe) + head_pred_probe
            head_energy = float((head_pred_probe.reshape(-1) ** 2).mean().detach().cpu())
            full_energy = float((full.reshape(-1) ** 2).mean().detach().cpu())
            if math.isfinite(head_energy) and math.isfinite(full_energy) and full_energy > 0.0:
                out["head_energy_frac"] = float(head_energy / max(full_energy, 1.0e-30))
    except Exception:
        pass
    return out


def _rel_improve(before: Any, after: Any) -> float | None:
    b = _finite_or_none(before)
    a = _finite_or_none(after)
    if b is None or a is None:
        return None
    return float((b - a) / max(abs(b), 1.0e-30))


def _score_ladder_template(expr: Any, carrier_probe_mse: Any, mapping: Mapping | None, mapped_probe_mse: Any) -> dict[str, Any]:
    mapping = mapping or {}
    return {
        "schema_version": _SCORE_LADDER_SCHEMA_VERSION,
        "carrier": {
            "expr": node_str(expr),
            "probe_mse_identity": _finite_or_none(carrier_probe_mse),
        },
        "mapped": {
            "available": True,
            "mapping_kind": str(mapping.get("kind", "")),
            "mapping_structural": bool(mapping_is_structural(mapping)),
            "mapping_nparams": int(_mapping_nparams(mapping)),
            "probe_mse": _finite_or_none(mapped_probe_mse),
            "improvement_vs_carrier": _rel_improve(carrier_probe_mse, mapped_probe_mse),
        },
        "head_augmented": {
            "available": False,
            "accepted": False,
            "probe_mse": None,
            "term_count": 0,
        },
        "compiled_structural": {
            "available": False,
            "accepted": False,
            "probe_mse": None,
            "expr": None,
        },
        "refined": {
            "enabled": False,
            "attempted": False,
            "accepted": False,
            "probe_mse": None,
            "expr": None,
        },
        "final_validation": {
            "available": False,
        },
    }


def _copy_score_ladder(ladder: Mapping | None) -> dict[str, Any]:
    if not isinstance(ladder, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key, value in ladder.items():
        if isinstance(value, Mapping):
            out[str(key)] = dict(value)
        else:
            out[str(key)] = value
    return out


def _score_ladder_from_mapping(mapping: Mapping | None) -> dict[str, Any]:
    if not isinstance(mapping, Mapping):
        return {}
    return _copy_score_ladder(mapping.get("_score_ladder", None))


def _acceptance_basis_for_mapping(mapping: Mapping | None, fallback: str = "mapped") -> str:
    if not isinstance(mapping, Mapping):
        return str(fallback)
    existing = mapping.get("_acceptance_basis", None)
    if existing:
        return str(existing)
    if mapping.get("_basis_transition", None) is not None:
        return "compiled_structural"
    if mapping.get("_lin_head", None) is not None:
        return "head_augmented"
    if mapping.get("_joint_linear_terms", None) is not None:
        return "joint_linear_terms"
    if mapping_is_structural(mapping):
        return "mapped_structural"
    return str(fallback)


def _attach_score_ladder(mapping: Mapping | None, ladder: Mapping | None, *, acceptance_basis: str | None = None) -> dict:
    out = dict(mapping or {})
    ladder_out = _copy_score_ladder(ladder)
    if ladder_out:
        out["_score_ladder"] = ladder_out
    out["_acceptance_basis"] = str(
        acceptance_basis or _acceptance_basis_for_mapping(out, fallback="mapped")
    )
    return out


def _mark_refinement_score(
    score_tuple,
    *,
    enabled: bool,
    attempted: bool,
    accepted: bool,
    source_expr: Any,
    accepted_expr: Any,
    base_probe_mse: Any,
):
    if score_tuple is None:
        return None
    if len(score_tuple) < 4:
        return score_tuple
    mapping = dict(score_tuple[3] or {})
    ladder = _score_ladder_from_mapping(mapping)
    refined = dict(ladder.get("refined", {}) if isinstance(ladder.get("refined", None), Mapping) else {})
    refined.update(
        {
            "enabled": bool(enabled),
            "attempted": bool(attempted),
            "accepted": bool(accepted),
            "probe_mse": _finite_or_none(score_tuple[0]) if accepted else None,
            "base_probe_mse": _finite_or_none(base_probe_mse),
            "improvement_vs_base": _rel_improve(base_probe_mse, score_tuple[0]) if accepted else None,
            "source_expr": node_str(source_expr),
            "expr": node_str(accepted_expr) if accepted else None,
        }
    )
    ladder["refined"] = refined
    basis = _acceptance_basis_for_mapping(mapping, fallback="mapped")
    if accepted:
        basis = "refined_structural" if mapping_is_structural(mapping) else "refined_mapped"
    mapping = _attach_score_ladder(mapping, ladder, acceptance_basis=basis)
    out = list(score_tuple)
    out[3] = mapping
    return tuple(out)


def _balanced_add_tree(terms):
    """Build a roughly balanced binary add tree from a list of term nodes."""
    if not terms:
        return None
    nodes = list(terms)
    while len(nodes) > 1:
        nxt = []
        i = 0
        n = len(nodes)
        while i < n:
            if i + 1 < n:
                nxt.append(("add", nodes[i], nodes[i + 1]))
                i += 2
            else:
                nxt.append(nodes[i])
                i += 1
        nodes = nxt
    return nodes[0]


def _strip_scalar_prefix(node):
    """Strip leading neg / mul-by-const wrappers for pool dedup."""
    out = node
    for _ in range(4):
        if not (isinstance(out, tuple) and out):
            break
        op = out[0]
        if op == "neg":
            out = out[1]
            continue
        if op == "mul":
            a, b = out[1], out[2]
            if isinstance(a, tuple) and a and a[0] == "const":
                out = b
                continue
            if isinstance(b, tuple) and b and b[0] == "const":
                out = a
                continue
        break
    return out


def _extract_scalar_core(node):
    """Split a node into (scalar, core) for simple linear-root canonicalization."""
    coeff = 1.0
    cur = node
    for _ in range(8):
        if not (isinstance(cur, tuple) and cur):
            break
        op = cur[0]
        if op == "neg":
            coeff = -coeff
            cur = cur[1]
            continue
        if op == "mul":
            a, b = cur[1], cur[2]
            if isinstance(a, tuple) and a and a[0] == "const":
                try:
                    coeff *= float(a[1])
                except Exception:
                    pass
                cur = b
                continue
            if isinstance(b, tuple) and b and b[0] == "const":
                try:
                    coeff *= float(b[1])
                except Exception:
                    pass
                cur = a
                continue
        break
    return float(coeff), cur


def _collect_linear_terms(node, sign=1.0, out=None):
    """Collect signed additive terms from a root expression."""
    if out is None:
        out = []
    if not (isinstance(node, tuple) and node):
        out.append((float(sign), node))
        return out
    op = node[0]
    if op == "add":
        _collect_linear_terms(node[1], sign, out)
        _collect_linear_terms(node[2], sign, out)
        return out
    if op == "sub":
        _collect_linear_terms(node[1], sign, out)
        _collect_linear_terms(node[2], -sign, out)
        return out
    if op == "neg":
        _collect_linear_terms(node[1], -sign, out)
        return out
    out.append((float(sign), node))
    return out


def _refine_diag(cfg):
    if not isinstance(cfg, dict):
        return None
    diag = cfg.get("diagnostics", cfg.get("refine_diagnostics", None))
    return diag if isinstance(diag, dict) else None


def _diag_inc(cfg, key, amount=1):
    diag = _refine_diag(cfg)
    if diag is not None:
        diag[str(key)] = int(diag.get(str(key), 0)) + int(amount)


def _diag_inc_context(cfg, suffix, amount=1):
    if not isinstance(cfg, dict):
        return
    context = str(cfg.get("refine_context", "") or "").strip().lower()
    if not context:
        return
    context = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in context)
    context = context.strip("_")
    if not context:
        return
    _diag_inc(cfg, f"{context}_{suffix}", amount)


def _diag_add_time(cfg, key, elapsed):
    diag = _refine_diag(cfg)
    if diag is not None:
        diag[str(key)] = float(diag.get(str(key), 0.0)) + float(max(0.0, elapsed))


def _node_var_indices(node, out=None):
    if out is None:
        out = set()
    if not (isinstance(node, tuple) and node):
        return out
    if node[0] == "var":
        try:
            out.add(int(node[1]))
        except Exception:
            pass
        return out
    if node[0] in ("const", "hparam"):
        return out
    if node[0] in UNARY_OPS or node[0] in ("asin", "acos"):
        return _node_var_indices(node[1], out)
    if node[0] in BINARY_OPS:
        _node_var_indices(node[1], out)
        _node_var_indices(node[2], out)
    return out


def _refine_tensor_signature(x_fit, y_fit):
    try:
        shape_x = tuple(int(v) for v in x_fit.shape)
        shape_y = tuple(int(v) for v in y_fit.shape)
        x_det = x_fit.detach()
        y_det = y_fit.detach()
        x_mean = float(torch.nanmean(x_det).detach().cpu())
        x_std = float(torch.sqrt(torch.nanmean((x_det - x_mean) ** 2)).detach().cpu())
        y_mean = float(torch.nanmean(y_det).detach().cpu())
        y_std = float(torch.sqrt(torch.nanmean((y_det - y_mean) ** 2)).detach().cpu())
        return (
            shape_x,
            shape_y,
            str(x_fit.dtype),
            str(y_fit.dtype),
            int(id(x_fit)),
            int(id(y_fit)),
            round(x_mean, 12),
            round(x_std, 12),
            round(y_mean, 12),
            round(y_std, 12),
        )
    except Exception:
        return None


def _refine_cfg_signature(cfg):
    keys = (
        "optimizer", "lbfgs_escalate_improve_factor", "lbfgs_steps", "fit_subset",
        "fit_subset_mode", "num_restarts", "max_params", "linear_combo_enable",
        "linear_terms_max", "linear_prune_rel", "linear_ridge", "safe_eps",
        "safe_penalty_weight", "safe_exp_clip", "theta_l2", "init_log_min",
        "init_log_max", "refine_grid_enable", "refine_grid_size", "refine_grid_size_2d",
        "refine_grid_passes", "refine_grid_topk", "refine_grid_max_evals",
        "joint_refine_enable", "joint_weight_mode",
    )
    return tuple((k, cfg.get(k, None)) for k in keys)


def _refine_attempt_cache_key(var_h, n_params, shift_slots, cfg, x_fit, y_fit):
    data_identity = (int(id(x_fit)), int(id(y_fit)))
    data_sig = cfg.get("_attempt_cache_data_signature", None)
    if data_sig is None or cfg.get("_attempt_cache_data_identity", None) != data_identity:
        data_sig = _refine_tensor_signature(x_fit, y_fit)
        cfg["_attempt_cache_data_signature"] = data_sig
        cfg["_attempt_cache_data_identity"] = data_identity
    joint = cfg.get("joint_fit_data", None)
    if isinstance(joint, (list, tuple)):
        joint_sig = tuple(
            str(row[0]) if isinstance(row, (tuple, list)) and len(row) == 3 else str(i)
            for i, row in enumerate(joint)
        )
    else:
        joint_sig = ()
    return (
        node_str(var_h),
        int(n_params),
        tuple(sorted(int(v) for v in shift_slots)),
        tuple(sorted(_node_var_indices(var_h))),
        data_sig,
        joint_sig,
        _refine_cfg_signature(cfg),
    )


def _refine_cache_get(cfg, key):
    if not bool(cfg.get("attempt_cache_enable", True)):
        return None
    cache = cfg.get("attempt_cache", None)
    if not isinstance(cache, dict):
        return None
    if key not in cache:
        _diag_inc(cfg, "attempt_cache_misses")
        return None
    _diag_inc(cfg, "attempt_cache_hits")
    return cache.get(key)


def _refine_cache_put(cfg, key, entry):
    if not bool(cfg.get("attempt_cache_enable", True)):
        return
    cache = cfg.get("attempt_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        cfg["attempt_cache"] = cache
    max_entries = max(0, int(cfg.get("attempt_cache_max_entries", 4096)))
    if max_entries <= 0:
        _diag_inc(cfg, "attempt_cache_skipped_full")
        return
    if len(cache) >= max_entries:
        try:
            cache.pop(next(iter(cache)))
            _diag_inc(cfg, "attempt_cache_evictions")
        except Exception:
            _diag_inc(cfg, "attempt_cache_skipped_full")
            return
    cache[key] = dict(entry)
    _diag_inc(cfg, "attempt_cache_stores")


def _eval_node_hparam(node, x, hparams):
    op = node[0]
    if op == "hparam":
        i = int(node[1])
        hp = hparams[i]
        if torch.is_tensor(hp):
            return torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device) * hp
        return torch.full((x.shape[0], 1), float(hp), dtype=x.dtype, device=x.device)
    if op == "var":
        i = node[1]
        return x[:, i:i + 1]
    if op == "const":
        return torch.full((x.shape[0], 1), node[1], dtype=x.dtype, device=x.device)
    if op == "sin":
        return torch.sin(_eval_node_hparam(node[1], x, hparams))
    if op == "cos":
        return torch.cos(_eval_node_hparam(node[1], x, hparams))
    if op == "asin":
        return torch.asin(_eval_node_hparam(node[1], x, hparams))
    if op == "acos":
        return torch.acos(_eval_node_hparam(node[1], x, hparams))
    if op == "exp":
        return torch.exp(_eval_node_hparam(node[1], x, hparams))
    if op == "log":
        return torch.log(_eval_node_hparam(node[1], x, hparams))
    if op == "sqrt":
        return torch.sqrt(_eval_node_hparam(node[1], x, hparams))
    if op == "sqr":
        c = _eval_node_hparam(node[1], x, hparams)
        return c * c
    if op == "neg":
        return -_eval_node_hparam(node[1], x, hparams)
    if op == "add":
        return _eval_node_hparam(node[1], x, hparams) + _eval_node_hparam(node[2], x, hparams)
    if op == "sub":
        return _eval_node_hparam(node[1], x, hparams) - _eval_node_hparam(node[2], x, hparams)
    if op == "mul":
        return _eval_node_hparam(node[1], x, hparams) * _eval_node_hparam(node[2], x, hparams)
    if op == "div":
        a = _eval_node_hparam(node[1], x, hparams)
        b = _eval_node_hparam(node[2], x, hparams)
        return a / b
    raise ValueError(op)


def _solve_linear_coeffs(Phi, y, ridge):
    ridge = max(0.0, float(ridge))
    if Phi is None or y is None or Phi.ndim != 2 or y.ndim != 2:
        return None
    if int(Phi.shape[0]) <= 0 or int(Phi.shape[1]) <= 0:
        return None
    if ridge > 0.0:
        try:
            k = int(Phi.shape[1])
            eye = torch.eye(k, dtype=Phi.dtype, device=Phi.device)
            gram = Phi.transpose(0, 1) @ Phi + ridge * eye
            rhs = Phi.transpose(0, 1) @ y
            return torch.linalg.solve(gram, rhs)
        except Exception:
            pass
    try:
        return torch.linalg.lstsq(Phi, y).solution
    except Exception:
        return None


def _joint_dataset_weights(joint_data, cfg):
    """Compute weights for joint multi-dataset refinement/scoring."""
    if not isinstance(joint_data, (list, tuple)):
        return None

    x0 = None
    for row in joint_data:
        if row is None:
            continue
        if isinstance(row, (tuple, list)) and len(row) == 3:
            x = row[1]
        elif isinstance(row, (tuple, list)) and len(row) == 2:
            x = row[0]
        else:
            continue
        if torch.is_tensor(x):
            x0 = x
            break

    if x0 is None:
        return None

    dtype = x0.dtype
    device = x0.device

    sizes = []
    for row in joint_data:
        if row is None:
            continue
        if isinstance(row, (tuple, list)) and len(row) == 3:
            x_d = row[1]
        elif isinstance(row, (tuple, list)) and len(row) == 2:
            x_d = row[0]
        else:
            continue
        if not torch.is_tensor(x_d):
            continue
        try:
            sizes.append(float(int(x_d.shape[0])))
        except Exception:
            sizes.append(0.0)

    D = len(sizes)
    if D <= 0:
        return None

    mode = str(cfg.get("joint_weight_mode", "points")).strip().lower()
    if mode in ("datasets", "dataset", "equal", "uniform"):
        return torch.full((D,), 1.0 / float(D), dtype=dtype, device=device)

    s = torch.as_tensor(sizes, dtype=dtype, device=device)
    tot = float(s.sum().detach().cpu())
    if not math.isfinite(tot) or tot <= 0.0:
        return torch.full((D,), 1.0 / float(D), dtype=dtype, device=device)
    return s / s.sum()


def _flatten_add_terms(node):
    op = node[0]
    if op == "add":
        return _flatten_add_terms(node[1]) + _flatten_add_terms(node[2])
    if op == "sub":
        return _flatten_add_terms(node[1]) + [("neg", t) for t in _flatten_add_terms(node[2])]
    return [node]


def _select_linear_basis_nodes(expr_h, cfg):
    if not bool(cfg.get("linear_combo_enable", True)):
        return [expr_h]
    terms = [t for t in _flatten_add_terms(expr_h) if t[0] != "const"]
    max_terms = max(1, int(cfg.get("linear_terms_max", 6)))
    if len(terms) < 2 or len(terms) > max_terms:
        return [expr_h]
    return terms



def _refine_budget_left(refine_state, max_refines: int) -> bool:
    if refine_state is None:
        return True
    if int(max_refines) > 0 and int(refine_state.get("trials_done", 0)) >= int(max_refines):
        return False
    depth_left = refine_state.get("depth_trials_left", None)
    if depth_left is not None and int(depth_left) <= 0:
        return False
    window_left = refine_state.get("window_trials_left", None)
    if window_left is not None and int(window_left) <= 0:
        return False
    return True


def _mapping_equiv_root(node, *, assume_simplified=False):
    """Canonicalize only top-level mapping-equivalent scalar/sign variants."""
    t = node if assume_simplified else simplify(node)

    terms = _collect_linear_terms(t, 1.0, [])
    if len(terms) >= 2:
        parsed = []
        for sgn, term in terms:
            c, core = _extract_scalar_core(term)
            parsed.append((float(sgn) * float(c), core))
        if parsed:
            core0 = parsed[0][1]
            key0 = node_str(core0)
            if all(node_str(core) == key0 for _, core in parsed):
                total = sum(float(c) for c, _ in parsed)
                if abs(total) <= 1.0e-14:
                    t = ("const", 0.0)
                else:
                    t = core0

    t = _strip_scalar_prefix(t)
    if isinstance(t, tuple) and t and t[0] == "sub" and node_str(t[1]) > node_str(t[2]):
        t = ("sub", t[2], t[1])
    return t

def _compile_linear_combo(term_nodes, coeffs, Phi_fit, prune_rel, max_depth):
    """Compile Σ c_i * term_i into a single AST, pruning tiny contributors and enforcing max_depth."""
    if term_nodes is None or coeffs is None or Phi_fit is None:
        return None
    try:
        K = min(int(len(term_nodes)), int(coeffs.shape[0]), int(Phi_fit.shape[1]))
    except Exception:
        return None
    if K <= 0:
        return None

    rel = max(0.0, float(prune_rel))

    contrib = []
    for j in range(K):
        try:
            c = float(coeffs[j])
        except Exception:
            c = float("nan")
        if (not math.isfinite(c)) or abs(c) < 1.0e-14:
            contrib.append(0.0)
            continue
        col = Phi_fit[:, j]
        rms = float(torch.sqrt((col * col).mean()))
        if not math.isfinite(rms):
            rms = 0.0
        contrib.append(abs(c) * rms)

    max_contrib = max(contrib) if contrib else 0.0

    keep = []
    keep_contrib = []
    for j in range(K):
        if contrib[j] <= 0.0:
            continue
        if max_contrib > 0.0 and contrib[j] < rel * max_contrib:
            continue
        keep.append(j)
        keep_contrib.append(contrib[j])

    if not keep:
        return None

    # Pre-build simplified scaled terms.
    scaled_terms = []
    scaled_contrib = []
    for jj, j in enumerate(keep):
        try:
            c = float(coeffs[j])
        except Exception:
            continue
        if (not math.isfinite(c)) or abs(c) < 1.0e-14:
            continue
        term = term_nodes[j]
        c = float(snap_direct_coeff(c))
        if abs(c - 1.0) < 1.0e-12:
            t = term
        elif abs(c + 1.0) < 1.0e-12:
            t = ("neg", term)
        else:
            t = ("mul", ("const", float(c)), term)
        t = simplify(t)
        scaled_terms.append(t)
        scaled_contrib.append(keep_contrib[jj])

    if not scaled_terms:
        return None

    # If depth is too large, drop the weakest contributors until feasible.
    active = list(range(len(scaled_terms)))
    while True:
        expr = _balanced_add_tree([scaled_terms[i] for i in active])
        if expr is None:
            return None
        expr = simplify(expr)
        if node_depth(expr) <= max_depth:
            return expr
        if len(active) <= 1:
            return None
        # Drop smallest-contribution term.
        drop_i = min(active, key=lambda ii: scaled_contrib[ii])
        active.remove(drop_i)


def _float_coeff_list(values: Any) -> list[float]:
    if values is None:
        return []
    if torch.is_tensor(values):
        try:
            values = values.detach().cpu().reshape(-1).tolist()
        except Exception:
            return []
    try:
        raw = list(values)
    except Exception:
        return []
    out: list[float] = []
    for value in raw:
        if torch.is_tensor(value):
            try:
                value = value.detach().cpu().item()
            except Exception:
                return []
        try:
            fv = float(value)
        except Exception:
            return []
        if not math.isfinite(fv):
            return []
        try:
            fv = float(snap_direct_coeff(fv))
        except Exception:
            pass
        out.append(float(fv))
    return out


def _active_degree(coeffs: list[float], tol: float) -> int:
    tol = max(0.0, float(tol))
    for idx in range(len(coeffs) - 1, -1, -1):
        if abs(float(coeffs[idx])) > tol:
            return int(idx)
    return 0


def _normalized_feature_ast(base_node: Any, *, mu: Any, std: Any) -> Any | None:
    try:
        mu_f = float(mu)
    except Exception:
        mu_f = 0.0
    try:
        std_f = float(std)
    except Exception:
        std_f = 1.0
    if (not math.isfinite(mu_f)) or (not math.isfinite(std_f)) or abs(std_f) < 1.0e-14:
        return None
    out = base_node
    if abs(mu_f) > 1.0e-12:
        out = ("sub", out, ("const", float(mu_f)))
    if abs(std_f - 1.0) > 1.0e-12:
        out = ("div", out, ("const", float(std_f)))
    return simplify(out)


def _poly_horner_ast(base_node: Any, coeffs: list[float], *, mu: Any, std: Any, coeff_tol: float) -> Any | None:
    coeffs = list(coeffs or [])
    if not coeffs:
        return ("const", 0.0)
    deg = _active_degree(coeffs, coeff_tol)
    coeffs = coeffs[: deg + 1]
    z = _normalized_feature_ast(base_node, mu=mu, std=std)
    if z is None:
        return None
    out = None
    for coeff in reversed(coeffs):
        c = 0.0 if abs(float(coeff)) <= float(coeff_tol) else float(coeff)
        coeff_node = ("const", c)
        if out is None:
            out = coeff_node
        else:
            out = ("add", coeff_node, ("mul", z, out))
    return simplify(out if out is not None else ("const", 0.0))


def _try_compile_structural_pade(
    *,
    expr: Any,
    mapping: Mapping | None,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    proj: torch.Tensor,
    fp_mode: str,
    q_scale: float,
    q_clip: int,
    cfg: Mapping,
    score_ladder: Mapping | None,
    mapped_probe_mse: float,
    y_var: float,
):
    """Materialize a low-order Padé map as a typed structural rational AST."""
    if not bool((cfg or {}).get("score_pade_structural_enable", False)):
        return None
    m = dict(mapping or {})
    if str(m.get("kind", "") or "").strip().lower() != "pade":
        return None

    coeff_tol = max(0.0, float((cfg or {}).get("score_pade_structural_coeff_tol", 1.0e-10) or 0.0))
    numer = _float_coeff_list(m.get("numer", ()))
    denom = _float_coeff_list(m.get("denom", ()))
    if not numer or not denom:
        return None

    deg_n = _active_degree(numer, coeff_tol)
    deg_d = _active_degree(denom, coeff_tol)
    max_degree = max(0, int((cfg or {}).get("score_pade_structural_max_degree", 2) or 0))
    max_total = max(0, int((cfg or {}).get("score_pade_structural_max_total_degree", 3) or 0))
    if deg_n > max_degree or deg_d > max_degree or (deg_n + deg_d) > max_total:
        return None

    numer_ast = _poly_horner_ast(
        expr,
        numer,
        mu=m.get("mu", 0.0),
        std=m.get("std", 1.0),
        coeff_tol=coeff_tol,
    )
    denom_ast = _poly_horner_ast(
        expr,
        denom,
        mu=m.get("mu", 0.0),
        std=m.get("std", 1.0),
        coeff_tol=coeff_tol,
    )
    if numer_ast is None or denom_ast is None:
        return None
    compiled = simplify(("div", numer_ast, denom_ast))

    max_depth = max(1, int((cfg or {}).get("score_pade_structural_max_depth", 8) or 8))
    max_size = max(1, int((cfg or {}).get("score_pade_structural_max_size", 64) or 64))
    if node_depth(compiled) > max_depth or node_size(compiled) > max_size:
        return None

    pred_fit = eval_node(compiled, x_fit)
    pred_probe = eval_node(compiled, x_probe)
    if pred_fit is None or pred_probe is None:
        return None
    if (not torch.isfinite(pred_fit).all()) or (not torch.isfinite(pred_probe).all()):
        return None

    r_fit = (y_fit - pred_fit.reshape_as(y_fit)).squeeze(-1)
    r_probe = (y_probe - pred_probe.reshape_as(y_probe)).squeeze(-1)
    mse_fit = float((r_fit * r_fit).mean())
    mse_probe = float((r_probe * r_probe).mean())
    if not math.isfinite(mse_fit) or not math.isfinite(mse_probe):
        return None

    rel_tol = max(0.0, float((cfg or {}).get("score_pade_structural_mse_rel_tol", 1.0e-6) or 0.0))
    abs_tol = max(1.0e-12, float(y_var) * 1.0e-12)
    if mse_probe > float(mapped_probe_mse) * (1.0 + rel_tol) + abs_tol:
        return None

    key, z = fingerprint(r_probe, proj, fp_mode, q_scale, q_clip)
    mapping_compiled = {
        "kind": "poly",
        "coeffs": [0.0, 1.0],
        "mu": 0.0,
        "std": 1.0,
        "_compiled_from_mapping": {
            "kind": "pade",
            "numer": numer[: deg_n + 1],
            "denom": denom[: deg_d + 1],
            "mu": _finite_or_none(m.get("mu", 0.0)),
            "std": _finite_or_none(m.get("std", 1.0)),
            "numer_degree": int(deg_n),
            "denom_degree": int(deg_d),
            "total_degree": int(deg_n + deg_d),
        },
        "_structural_rational": {
            "type": "pade_compiled_ast",
            "source": "score_mapping",
            "fit_mse": _finite_or_none(mse_fit),
            "probe_mse": _finite_or_none(mse_probe),
        },
    }

    ladder = _copy_score_ladder(score_ladder)
    compiled_stage = dict(
        ladder.get("compiled_structural", {})
        if isinstance(ladder.get("compiled_structural", None), Mapping)
        else {}
    )
    compiled_stage.update(
        {
            "available": True,
            "accepted": True,
            "probe_mse": _finite_or_none(mse_probe),
            "fit_mse": _finite_or_none(mse_fit),
            "expr": node_str(compiled),
            "source": "compiled_pade_mapping",
            "mapping_kind": "pade",
            "numer_degree": int(deg_n),
            "denom_degree": int(deg_d),
            "total_degree": int(deg_n + deg_d),
            "improvement_vs_mapped": _rel_improve(mapped_probe_mse, mse_probe),
        }
    )
    ladder["compiled_structural"] = compiled_stage
    mapping_compiled = _attach_score_ladder(
        mapping_compiled,
        ladder,
        acceptance_basis="typed_structural_rational",
    )
    return (mse_probe, key, z, mapping_compiled, compiled)


def _harvest_pool_from_archive(
    arch,
    rng,
    *,
    max_nodes=256,
    topk_residual_basins=50,
    elites_per_residual_basin=2,
    subtree_depth_max=3,
    subtree_size_max=12,
    base_seen=None,
    var_dims=None,
    target_dim=None,
):
    """Harvest simple subtrees from the archive to expand the residual pool."""
    if arch is None or not getattr(arch, "d", None):
        return []
    try:
        max_nodes = max(0, int(max_nodes))
    except Exception:
        max_nodes = 0
    if max_nodes <= 0:
        return []

    seen = set(base_seen) if base_seen else set()
    out = []

    try:
        recs = sorted(list(arch.d.values()), key=lambda r: float(getattr(r, "best_mse", 1e100)))
    except Exception:
        recs = list(getattr(arch, "d", {}).values())

    try:
        recs = recs[: max(1, int(topk_residual_basins))]
    except Exception:
        pass

    for r in recs:
        exprs = []
        try:
            els = list(getattr(r, "elites", []) or [])
            if els:
                els = sorted(
                    els,
                    key=lambda e: (
                        float(getattr(e, "mse", 1e100)),
                        float(getattr(e, "size", 1.0e100)),
                    ),
                )
                k = max(0, int(elites_per_residual_basin))
                if k > 0:
                    exprs.extend([el.expr for el in els[:k]])
        except Exception:
            pass
        try:
            exprs.append(r.best_expr)
        except Exception:
            pass

        for expr in exprs:
            if expr is None:
                continue
            try:
                paths = collect_paths(expr)
            except Exception:
                continue
            try:
                rng.shuffle(paths)
            except Exception:
                pass

            for p in paths:
                try:
                    sub = get_at(expr, p)
                except Exception:
                    continue
                if node_depth(sub) > subtree_depth_max:
                    continue
                if node_size(sub) > subtree_size_max:
                    continue
                sub = simplify(sub)
                sub = _strip_scalar_prefix(sub)
                if not (isinstance(sub, tuple) and sub):
                    continue
                if sub[0] == "const":
                    continue
                if var_dims is not None:
                    d = node_dims(sub, var_dims)
                    if d is None:
                        continue
                    if target_dim is not None and not dims_eq(d, target_dim):
                        continue
                key = node_str(sub)
                if key in seen:
                    continue
                seen.add(key)
                out.append(sub)
                if len(out) >= max_nodes:
                    return out

    return out

def fingerprint(r,proj,mode,scale,clip,eps=1e-12):
    r=r-r.mean(); r=r/(r.std()+eps)
    z=(r@proj)/math.sqrt(r.numel())
    if mode=="bits":
        bits=(z>0).to(torch.int8).tolist()
        k=0
        for i,b in enumerate(bits): k |= (int(b)&1)<<i
        return k,z
    q=torch.clamp((z*scale).round(), -clip, clip).to(torch.int16)
    return tuple(int(v) for v in q.tolist()), z

def _negate_smart(node):
    """Return a cheap equivalent tuple-AST for (-node), preferring forms that avoid an extra 'neg' node."""
    try:
        op = node[0]
    except Exception:
        return ("neg", node)
    if op == "const":
        return ("const", -float(node[1]))
    if op == "neg":
        return node[1]
    if op == "sub":
        # -(a-b) == (b-a) with no extra node
        return ("sub", node[2], node[1])
    if op == "mul":
        a, b = node[1], node[2]
        if isinstance(a, tuple) and a and a[0] == "const":
            return ("mul", ("const", -float(a[1])), b)
        if isinstance(b, tuple) and b and b[0] == "const":
            return ("mul", a, ("const", -float(b[1])))
    if op == "div":
        a, b = node[1], node[2]
        if isinstance(a, tuple) and a and a[0] == "const":
            return ("div", ("const", -float(a[1])), b)
    return ("neg", node)


def _mapping_family_hint(node):
    tokens = set()

    def _visit(cur):
        if not (isinstance(cur, tuple) and cur):
            return
        op = str(cur[0])
        if op in ("sin", "cos"):
            tokens.add("periodic")
        elif op in ("exp", "log"):
            tokens.add("exp")
        for child in cur[1:]:
            _visit(child)

    _visit(node)
    return ",".join(sorted(tokens))


def _mapping_fit_kwargs(cfg, expr):
    cfg = cfg or {}
    return {
        "family_mode": str(cfg.get("score_mapping_family_mode", "full") or "full"),
        "expensive_gate_best_mse": cfg.get("_mapping_family_best_mse", None),
        "expensive_gate_best_factor": float(cfg.get("score_mapping_expensive_gate_best_factor", 5.0) or 5.0),
        "expensive_gate_rel_y": float(cfg.get("score_mapping_expensive_rel_y", 0.10) or 0.10),
        "family_hint": _mapping_family_hint(expr),
    }


def _use_affine_fast_path(poly_degree, _family_mode=None) -> bool:
    return int(poly_degree) == 1


def _skip_negated_equiv_for_affine_poly_only(poly_degree, cfg) -> bool:
    if int(poly_degree) != 1:
        return False
    family_mode = str((cfg or {}).get("score_mapping_family_mode", "full") or "full").strip().lower()
    return family_mode == "poly_only"


def _fit_best_with_cfg(pred, y, poly_degree, cfg, *, expr=None, pred_probe=None, family_mode=None):
    """Apply cfg-driven mapping-fit restrictions at the scoring layer.

    Keeping this wrapper local to engine scoring avoids relying on callers to
    thread the right mapping family knobs through every internal fit_best(...)
    invocation.
    """
    fit_kwargs = _mapping_fit_kwargs(cfg, expr)
    if family_mode is not None:
        fit_kwargs["family_mode"] = str(family_mode or fit_kwargs.get("family_mode", "full"))
    if pred_probe is not None:
        fit_kwargs["pred_probe"] = pred_probe
    fit_kwargs["affine_fast"] = _use_affine_fast_path(poly_degree, fit_kwargs.get("family_mode"))
    fit_kwargs["diagnostics"] = _refine_diag(cfg)
    return fit_best(pred, y, poly_degree, **fit_kwargs)


def _score_prescreen_stats(cfg):
    stats = (cfg or {}).get("score_prescreen_stats", None)
    return stats if isinstance(stats, dict) else None


def _bump_score_stat(cfg, key, delta=1):
    stats = _score_prescreen_stats(cfg)
    if stats is None:
        return
    stats[key] = int(stats.get(key, 0)) + int(delta)


def _bump_full_score_stat(cfg):
    stats = _score_prescreen_stats(cfg)
    if stats is None:
        return
    stats["full_score_calls"] = int(stats.get("full_score_calls", 0)) + 1
    by_action = stats.get("full_score_calls_by_action", None)
    if not isinstance(by_action, dict):
        by_action = {}
        stats["full_score_calls_by_action"] = by_action
    action_name = str((cfg or {}).get("score_prescreen_action_name", "") or "unknown")
    by_action[action_name] = int(by_action.get(action_name, 0)) + 1


def _maybe_prescreen_candidate(pred_fit, y_fit, pred_probe, y_probe, poly_degree, expr, cfg):
    cfg = cfg or {}
    if bool(cfg.get("score_prescreen_force_full", False)):
        return True
    if not bool(cfg.get("score_prescreen_enable", False)):
        return True

    _bump_score_stat(cfg, "prescore_calls")
    family_mode = str(
        cfg.get(
            "score_prescreen_family_mode",
            _mapping_fit_kwargs(cfg, expr).get("family_mode", "cheap"),
        )
        or "cheap"
    )
    fb = _fit_best_with_cfg(
        pred_fit,
        y_fit,
        poly_degree,
        cfg,
        expr=expr,
        pred_probe=pred_probe,
        family_mode=family_mode,
    )
    if fb is None:
        _bump_score_stat(cfg, "prescore_dropped")
        return False
    _, mapping0 = fb
    y_hat_probe0 = eval_mapping(pred_probe, mapping0)
    if not torch.isfinite(y_hat_probe0).all():
        _bump_score_stat(cfg, "prescore_dropped")
        return False
    r0 = (y_probe - y_hat_probe0).squeeze(-1)
    mse0 = float((r0 * r0).mean())
    if not math.isfinite(mse0):
        _bump_score_stat(cfg, "prescore_dropped")
        return False

    family_hint = str(_mapping_fit_kwargs(cfg, expr).get("family_hint", "") or "")
    allow_hint = bool(cfg.get("score_prescreen_allow_hint", True))
    promoted_by_hint = bool(family_hint) and bool(allow_hint)

    parent_best_mse = cfg.get("score_prescreen_parent_mse", None)
    promoted_by_parent = False
    if parent_best_mse is not None:
        try:
            parent_best_mse = float(parent_best_mse)
        except Exception:
            parent_best_mse = None
        if parent_best_mse is not None and math.isfinite(parent_best_mse):
            parent_factor = max(1.0, float(cfg.get("score_prescreen_parent_best_factor", 1.5) or 1.5))
            promoted_by_parent = float(mse0) <= float(parent_best_mse) * parent_factor

    global_best_mse = cfg.get("score_prescreen_global_best_mse", None)
    promoted_by_global = False
    if bool(cfg.get("score_prescreen_use_global_best", True)) and global_best_mse is not None:
        try:
            global_best_mse = float(global_best_mse)
        except Exception:
            global_best_mse = None
        if global_best_mse is not None and math.isfinite(global_best_mse):
            global_factor = max(1.0, float(cfg.get("score_prescreen_global_best_factor", 3.0) or 3.0))
            promoted_by_global = float(mse0) <= float(global_best_mse) * global_factor

    no_thresholds = not (
        (parent_best_mse is not None and math.isfinite(parent_best_mse))
        or (global_best_mse is not None and math.isfinite(global_best_mse))
    )
    promoted = bool(promoted_by_hint or promoted_by_parent or promoted_by_global or no_thresholds)
    if promoted:
        _bump_score_stat(cfg, "prescore_promoted")
        if promoted_by_hint:
            _bump_score_stat(cfg, "prescore_promoted_by_hint")
        if promoted_by_parent:
            _bump_score_stat(cfg, "prescore_promoted_by_parent_threshold")
        if promoted_by_global:
            _bump_score_stat(cfg, "prescore_promoted_by_global_best_threshold")
        return True

    _bump_score_stat(cfg, "prescore_dropped")
    return False

def _pick_best_equiv_score(cands, y_var=None):
    """Pick the best candidate among equivalent-score variants.

    Preference order:
      1) Any *structural* mapping within a tolerant MSE window.
      2) Fewer mapping parameters.
      3) Simpler expression (physics prior cost, then size).
      4) Lowest MSE.
    """
    cands = [c for c in (cands or []) if c is not None]
    if not cands:
        return None
    best_mse = min(float(c[0]) for c in cands)
    if y_var is None:
        y_var = 1e-30
    y_var = max(float(y_var), 1e-30)
    mse_tol = max(best_mse * 3.0, y_var * 1e-8)
    close = [c for c in cands if float(c[0]) <= mse_tol]
    pool = close if close else cands

    def _rank(c):
        mse, _, _, mapping, expr = c
        return (
            0 if mapping_is_structural(mapping) else 1,
            _mapping_nparams(mapping),
            node_cost_physics_prior(expr),
            node_size(expr),
            float(mse),
        )

    pool.sort(key=_rank)
    return pool[0]


def _eval_score_col(node, x, cfg):
    try:
        value, domain_diag = eval_node_with_domain_projection(node, x, cfg or {})
    except Exception:
        return None, None
    out = _as_score_col(value, rows=int(x.shape[0]))
    if out is None:
        return None, domain_diag
    if not domain_projection_is_acceptable(domain_diag):
        return None, domain_diag
    return out, domain_diag


def _score_expr_base(node, x_fit, y_fit, x_probe, y_probe, proj, fp_mode, q_scale, q_clip, poly_degree, cfg=None):
    cfg = cfg or {}
    # Ensure a stable baseline form (also helps negate_smart produce nicer trees).
    node = simplify(node)

    p_fit, domain_fit_diag = _eval_score_col(node, x_fit, cfg)
    y_fit = _as_score_col(y_fit, rows=int(x_fit.shape[0]))
    if p_fit is None or y_fit is None:
        return None

    p_probe, domain_probe_diag = _eval_score_col(node, x_probe, cfg)
    y_probe = _as_score_col(y_probe, rows=int(x_probe.shape[0]))
    if p_probe is None or y_probe is None:
        return None
    domain_diag = merge_domain_projection_diagnostics(
        domain_fit_diag,
        domain_probe_diag,
        labels=("fit", "probe"),
    )
    if not domain_projection_is_acceptable(domain_diag):
        _bump_score_stat(cfg, "domain_projection_rejected")
        return None

    finite_mask_enabled = bool(cfg.get("score_finite_mask_enable", False))
    finite_diag = _finite_mask_diag(enabled=False)
    if finite_mask_enabled:
        fit_mask = (torch.isfinite(p_fit).reshape(-1)) & (torch.isfinite(y_fit).reshape(-1))
        probe_mask = (torch.isfinite(p_probe).reshape(-1)) & (torch.isfinite(y_probe).reshape(-1))
        min_fit_frac = _finite_mask_threshold(cfg, "score_finite_mask_min_fit_frac", 0.98)
        min_probe_frac = _finite_mask_threshold(cfg, "score_finite_mask_min_probe_frac", 0.98)
        min_points = _finite_mask_min_points(cfg)
        finite_diag = _finite_mask_diag(
            enabled=True,
            fit_mask=fit_mask,
            probe_mask=probe_mask,
        )
        if not _finite_mask_ok(fit_mask, min_frac=min_fit_frac, min_points=min_points):
            _bump_score_stat(cfg, "finite_mask_rejected_fit")
            return None
        if not _finite_mask_ok(probe_mask, min_frac=min_probe_frac, min_points=min_points):
            _bump_score_stat(cfg, "finite_mask_rejected_probe")
            return None
        x_fit = x_fit[fit_mask]
        y_fit = y_fit[fit_mask]
        p_fit = p_fit[fit_mask]
        x_probe = x_probe[probe_mask]
        y_probe = y_probe[probe_mask]
        p_probe = p_probe[probe_mask]
        proj = proj[probe_mask]
    else:
        if not torch.isfinite(p_fit).all():
            return None
        if not torch.isfinite(p_probe).all():
            return None

    if int(p_fit.shape[0]) <= 0 or int(p_probe.shape[0]) <= 0:
        return None
    if float(p_fit.std()) < 1e-12:
        return None

    y_var = max(float((y_fit ** 2).mean()), 1e-30)

    # Scoring augmentation: optional linear head on the residual that can soak up
    # simple unit-consistent additive terms (e.g. raw variables).
    head_enable = bool(cfg.get("score_head_enable", False))
    head_vars_enable = bool(cfg.get("score_head_vars_enable", True))
    head_omp_enable = bool(cfg.get("score_head_omp_enable", False))
    head_omp_max_terms = int(cfg.get("score_head_omp_max_terms", 0) or 0)
    head_omp_topk_try = int(cfg.get("score_head_omp_topk_try", 15) or 15)
    head_min_rel_improve = float(cfg.get("score_head_min_rel_improve", 0.0) or 0.0)

    head_ridge = cfg.get("score_head_ridge", None)
    if head_ridge is None:
        head_ridge = cfg.get("refine_linear_ridge", 1.0e-8)
    head_ridge = float(head_ridge) if head_ridge is not None else 0.0
    head_direct_combo_enable = bool(cfg.get("score_head_direct_combo_enable", True))
    head_direct_combo_prune_rel = float(cfg.get("score_head_direct_combo_prune_rel", 1.0e-6) or 0.0)
    head_direct_combo_tol = float(cfg.get("score_head_direct_combo_tol", 1.0e-6) or 0.0)
    max_depth = int(cfg.get("max_depth", 12) or 12)

    head_var_terms = []
    if head_enable and head_vars_enable:
        head_var_terms = list(cfg.get("score_head_var_terms", []) or [])

    # Optional OMP selection from the pool (requires the pool tensors in cfg).
    pool_nodes = cfg.get("score_head_pool_nodes", None)
    pool_phi_fit = cfg.get("score_head_pool_phi_fit", None)
    pool_phi_probe = cfg.get("score_head_pool_phi_probe", None)
    pool_norms_fit = cfg.get("score_head_pool_norms_fit", None)
    pool_valid_mask = cfg.get("score_head_pool_valid_mask", None)
    pool_node_to_idx = cfg.get("score_head_pool_node_to_idx", None)

    def _eval_term(term, X):
        # Fast-path raw variables.
        if isinstance(term, tuple) and len(term) == 2 and term[0] == "var":
            j = int(term[1])
            if j < 0 or j >= int(X.shape[1]):
                return None
            return X[:, j]
        v = eval_node(term, X)
        if v is None:
            return None
        if v.dim() == 2 and v.shape[1] == 1:
            v = v[:, 0]
        return v

    def _fit_head(resid_fit0, resid_probe0):
        """Fit a (bias + linear terms) head to resid_fit0, evaluate on resid_probe0.

        Returns:
            (mse_probe, r_probe, head_dict, pred_fit, pred_probe)
        where pred_* are the head contributions (shape Nx1), and r_probe is the final
        probe residual after subtracting the head (shape N,).
        """
        # resid_*0 are (N,1) tensors
        terms = []
        cols_fit = []
        cols_probe = []

        # Baseline terms (B1): unit-matching raw variables (pre-filtered upstream).
        for t in head_var_terms:
            v_fit = _eval_term(t, x_fit)
            v_probe = _eval_term(t, x_probe)
            if v_fit is None or v_probe is None:
                continue
            if (not torch.isfinite(v_fit).all()) or (not torch.isfinite(v_probe).all()):
                continue
            if float(v_fit.std()) < 1e-12:
                continue
            terms.append(t)
            cols_fit.append(v_fit)
            cols_probe.append(v_probe)

        # OMP extra terms from pool (B2): cheap greedy selection on residual.
        selected_pool = []
        if head_enable and head_omp_enable and head_omp_max_terms > 0:
            if (
                isinstance(pool_nodes, (list, tuple))
                and torch.is_tensor(pool_phi_fit)
                and torch.is_tensor(pool_phi_probe)
                and torch.is_tensor(pool_norms_fit)
                and torch.is_tensor(pool_valid_mask)
                and int(pool_phi_fit.shape[0]) == int(resid_fit0.shape[0])
                and int(pool_phi_probe.shape[0]) == int(resid_probe0.shape[0])
                and int(pool_phi_fit.shape[1]) == int(pool_norms_fit.shape[0])
                and int(pool_phi_fit.shape[1]) == int(pool_valid_mask.shape[0])
            ):
                # Exclude any pool terms already present in the baseline list.
                exclude = set()
                if isinstance(pool_node_to_idx, dict):
                    for t in terms:
                        try:
                            idx = pool_node_to_idx.get(t, None)
                        except Exception:
                            idx = None
                        if idx is not None:
                            exclude.add(int(idx))

                resid_v = resid_fit0.squeeze(-1)
                # Greedy select up to K terms.
                for _ in range(int(head_omp_max_terms)):
                    mask = pool_valid_mask.clone()
                    if exclude:
                        ex = torch.tensor(sorted(exclude), dtype=torch.long, device=mask.device)
                        mask[ex] = False
                    if selected_pool:
                        sel = torch.tensor(selected_pool, dtype=torch.long, device=mask.device)
                        mask[sel] = False
                    n_valid = int(mask.sum().item())
                    if n_valid <= 0:
                        break

                    # Correlation scores (normalized by column norm).
                    dots = torch.mv(pool_phi_fit.t(), resid_v)
                    denom = torch.sqrt(torch.clamp(pool_norms_fit, min=1e-30))
                    score = dots.abs() / denom
                    score = score.masked_fill(~mask, float("-inf"))

                    k = min(int(head_omp_topk_try), n_valid)
                    topk = torch.topk(score, k=k, largest=True).indices.tolist()

                    best_cand = None
                    best_mse = float("inf")
                    best_pred_fit = None

                    # Base Phi (ones + baseline columns).
                    base_cols_fit = cols_fit
                    base_cols_probe = cols_probe

                    for cand in topk:
                        cand = int(cand)
                        # Build Phi for this candidate set: ones | baseline | selected_pool | cand
                        phi_fit_parts = []
                        phi_probe_parts = []

                        if base_cols_fit:
                            phi_fit_parts.append(torch.stack(base_cols_fit, dim=1))
                            phi_probe_parts.append(torch.stack(base_cols_probe, dim=1))

                        if selected_pool:
                            sel = torch.tensor(selected_pool, dtype=torch.long, device=pool_phi_fit.device)
                            phi_fit_parts.append(pool_phi_fit[:, sel])
                            phi_probe_parts.append(pool_phi_probe[:, sel])

                        # Candidate column
                        phi_fit_parts.append(pool_phi_fit[:, cand:cand + 1])
                        phi_probe_parts.append(pool_phi_probe[:, cand:cand + 1])

                        phi_fit = torch.cat(phi_fit_parts, dim=1) if phi_fit_parts else pool_phi_fit[:, cand:cand + 1]
                        phi_probe = torch.cat(phi_probe_parts, dim=1) if phi_probe_parts else pool_phi_probe[:, cand:cand + 1]

                        # Add bias column.
                        ones_fit = torch.ones((phi_fit.shape[0], 1), dtype=phi_fit.dtype, device=phi_fit.device)
                        ones_probe = torch.ones((phi_probe.shape[0], 1), dtype=phi_probe.dtype, device=phi_probe.device)
                        Phi_fit = torch.cat([ones_fit, phi_fit], dim=1)
                        Phi_probe = torch.cat([ones_probe, phi_probe], dim=1)

                        sol = _solve_linear_coeffs(Phi_fit, resid_fit0, ridge=head_ridge)
                        if sol is None:
                            continue
                        pred_fit = Phi_fit @ sol
                        r_fit = resid_fit0 - pred_fit
                        mse_fit = float((r_fit.squeeze(-1) ** 2).mean())
                        if math.isfinite(mse_fit) and mse_fit < best_mse:
                            best_mse = mse_fit
                            best_cand = cand
                            best_pred_fit = pred_fit.detach()

                    if best_cand is None:
                        break
                    selected_pool.append(int(best_cand))
                    exclude.add(int(best_cand))
                    # Update residual for next round (on fit split).
                    if best_pred_fit is not None:
                        resid_v = (resid_fit0 - best_pred_fit).squeeze(-1)

        # Materialize the final term list.
        term_nodes = list(terms)
        if selected_pool and isinstance(pool_nodes, (list, tuple)):
            term_nodes.extend([pool_nodes[int(i)] for i in selected_pool])

        if not term_nodes:
            # Nothing to fit (we intentionally don't fit a bias-only head).
            return None

        # Materialize the final design matrices.
        phi_fit_cols = []
        phi_probe_cols = []
        # Baseline columns from earlier (same order as `terms`).
        if cols_fit:
            phi_fit_cols.append(torch.stack(cols_fit, dim=1))
            phi_probe_cols.append(torch.stack(cols_probe, dim=1))
        # Pool-selected columns (same order as selected_pool).
        if selected_pool:
            sel = torch.tensor(selected_pool, dtype=torch.long, device=pool_phi_fit.device)
            phi_fit_cols.append(pool_phi_fit[:, sel])
            phi_probe_cols.append(pool_phi_probe[:, sel])

        phi_fit = torch.cat(phi_fit_cols, dim=1) if phi_fit_cols else None
        phi_probe = torch.cat(phi_probe_cols, dim=1) if phi_probe_cols else None
        if phi_fit is None or phi_probe is None:
            return None

        ones_fit = torch.ones((phi_fit.shape[0], 1), dtype=phi_fit.dtype, device=phi_fit.device)
        ones_probe = torch.ones((phi_probe.shape[0], 1), dtype=phi_probe.dtype, device=phi_probe.device)
        Phi_fit = torch.cat([ones_fit, phi_fit], dim=1)
        Phi_probe = torch.cat([ones_probe, phi_probe], dim=1)

        sol = _solve_linear_coeffs(Phi_fit, resid_fit0, ridge=head_ridge)
        if sol is None:
            return None

        pred_fit = Phi_fit @ sol
        pred_probe = Phi_probe @ sol
        r_probe = (resid_probe0 - pred_probe).squeeze(-1)
        mse_probe = float((r_probe * r_probe).mean())
        if not math.isfinite(mse_probe):
            return None

        coeffs = sol.squeeze(-1).detach().cpu().tolist()
        coeffs = [float(v) for v in coeffs]
        head = {
            "terms": term_nodes,
            "coeffs": coeffs,  # [bias, a_0, ..., a_k]
            "ridge": float(head_ridge),
        }
        if selected_pool:
            head["pool_selected"] = [int(i) for i in selected_pool]

        return (
            mse_probe,
            r_probe,
            head,
            pred_fit.detach(),
            pred_probe.detach(),
            phi_fit.detach(),
            phi_probe.detach(),
        )

    def _try_direct_combo(
        expr_base,
        pred_fit_base,
        pred_probe_base,
        term_nodes,
        phi_fit_terms,
        phi_probe_terms,
        mse_ref,
        score_ladder=None,
    ):
        if not bool(head_direct_combo_enable):
            return None
        term_nodes = list(term_nodes or [])
        if not term_nodes:
            return None
        if (not torch.is_tensor(phi_fit_terms)) or (not torch.is_tensor(phi_probe_terms)):
            return None
        if int(phi_fit_terms.shape[1]) != int(len(term_nodes)):
            return None
        if int(phi_probe_terms.shape[1]) != int(len(term_nodes)):
            return None

        base_col_fit = pred_fit_base.squeeze(-1).unsqueeze(-1)
        ones_fit = torch.ones((int(y_fit.shape[0]), 1), dtype=y_fit.dtype, device=y_fit.device)
        Phi_fit = torch.cat([base_col_fit, phi_fit_terms, ones_fit], dim=1)
        sol = _solve_linear_coeffs(Phi_fit, y_fit, ridge=head_ridge)
        if sol is None:
            return None

        coeff_vec = sol.squeeze(-1)
        compiled = _compile_linear_combo(
            [expr_base, *term_nodes, ("const", 1.0)],
            coeff_vec,
            Phi_fit,
            head_direct_combo_prune_rel,
            max_depth,
        )
        if compiled is None:
            return None

        pred_fit_combo, combo_domain_fit = _eval_score_col(compiled, x_fit, cfg)
        pred_probe_combo, combo_domain_probe = _eval_score_col(compiled, x_probe, cfg)
        combo_domain_diag = merge_domain_projection_diagnostics(
            combo_domain_fit,
            combo_domain_probe,
            labels=("fit", "probe"),
        )
        if not domain_projection_is_acceptable(combo_domain_diag):
            return None
        if (pred_fit_combo is None) or (pred_probe_combo is None):
            return None
        if (not torch.isfinite(pred_fit_combo).all()) or (not torch.isfinite(pred_probe_combo).all()):
            return None

        r_probe_combo = (y_probe - pred_probe_combo).squeeze(-1)
        mse_combo = float((r_probe_combo * r_probe_combo).mean())
        if not math.isfinite(mse_combo):
            return None
        tol_abs = max(1.0e-12, float(y_var) * 1.0e-12)
        tol_rel = max(0.0, float(head_direct_combo_tol))
        if mse_combo > float(mse_ref) * (1.0 + tol_rel) + tol_abs:
            return None

        key_combo, z_combo = fingerprint(r_probe_combo, proj, fp_mode, q_scale, q_clip)
        mapping_combo = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
        mapping_combo["_basis_transition"] = make_additive_basis_transition(
            core_expr=expr_base,
            term_nodes=term_nodes,
            coeffs=coeff_vec.detach().cpu().tolist(),
            compiled_expr=compiled,
            ridge=float(head_ridge),
            prune_rel=float(head_direct_combo_prune_rel),
        )
        ladder_combo = _copy_score_ladder(score_ladder)
        compiled_stage = dict(ladder_combo.get("compiled_structural", {}) if isinstance(ladder_combo.get("compiled_structural", None), Mapping) else {})
        compiled_stage.update(
            {
                "available": True,
                "accepted": True,
                "probe_mse": _finite_or_none(mse_combo),
                "expr": node_str(compiled),
                "term_count": int(len(term_nodes) + 1),
                "improvement_vs_head_or_mapped": _rel_improve(mse_ref, mse_combo),
            }
        )
        ladder_combo["compiled_structural"] = compiled_stage
        mapping_combo = _attach_score_ladder(
            mapping_combo,
            ladder_combo,
            acceptance_basis="compiled_structural",
        )
        mapping_combo = _attach_domain_projection_diag(mapping_combo, combo_domain_diag)
        return (mse_combo, key_combo, z_combo, mapping_combo, compiled)

    cands = []
    full_score_counted = False

    def _record_full_score_once():
        nonlocal full_score_counted
        if full_score_counted:
            return
        _bump_full_score_stat(cfg)
        full_score_counted = True

    def _try(pred_fit, pred_probe, expr):
        if not _maybe_prescreen_candidate(pred_fit, y_fit, pred_probe, y_probe, poly_degree, expr, cfg):
            return None
        _record_full_score_once()
        fb = _fit_best_with_cfg(pred_fit, y_fit, poly_degree, cfg, expr=expr, pred_probe=pred_probe)
        if fb is None:
            return None
        _, mapping0 = fb

        # Base (univariate) mapping prediction.
        y_hat_fit0 = eval_mapping(pred_fit, mapping0)
        y_hat_probe0 = eval_mapping(pred_probe, mapping0)
        r0 = (y_probe - y_hat_probe0).squeeze(-1)
        mse0 = float((r0 * r0).mean())
        if not math.isfinite(mse0):
            return None

        carrier_mse = _probe_mse_or_none(y_probe, pred_probe)
        score_ladder = _score_ladder_template(expr, carrier_mse, mapping0, mse0)
        mse = mse0
        r = r0
        mapping = _attach_score_ladder(
            mapping0,
            score_ladder,
            acceptance_basis="mapped_structural" if mapping_is_structural(mapping0) else "mapped",
        )
        compiled_pade = _try_compile_structural_pade(
            expr=expr,
            mapping=mapping0,
            x_fit=x_fit,
            y_fit=y_fit,
            x_probe=x_probe,
            y_probe=y_probe,
            proj=proj,
            fp_mode=fp_mode,
            q_scale=q_scale,
            q_clip=q_clip,
            cfg=cfg,
            score_ladder=score_ladder,
            mapped_probe_mse=mse0,
            y_var=y_var,
        )

        # Optional linear head on the residual.
        refit_trigger_gain = max(head_min_rel_improve, 0.05)
        if head_enable and (head_var_terms or (head_omp_enable and head_omp_max_terms > 0)):
            resid_fit0 = (y_fit - y_hat_fit0)
            resid_probe0 = (y_probe - y_hat_probe0)

            head_fit1 = _fit_head(resid_fit0, resid_probe0)
            final_head = None
            final_phi_fit = None
            final_phi_probe = None
            head_refit_passes = 0
            if head_fit1 is not None:
                mse_h1, r_h1, head1, head_pred_fit1, head_pred_probe1, phi_fit_head1, phi_probe_head1 = head_fit1
                gain1 = (mse0 - mse_h1) / max(mse0, 1e-30)
                if math.isfinite(mse_h1) and (mse_h1 < mse0) and (gain1 >= head_min_rel_improve):
                    mse = mse_h1
                    r = r_h1
                    mapping = dict(mapping0)
                    mapping["_lin_head"] = head1
                    mapping["_score_decomp"] = _head_score_decomp(
                        mse_core=mse0,
                        mse_with_head=mse_h1,
                        head=head1,
                        core_pred_probe=y_hat_probe0,
                        head_pred_probe=head_pred_probe1,
                    )
                    head_stage = dict(score_ladder.get("head_augmented", {}))
                    head_stage.update(
                        {
                            "available": True,
                            "accepted": True,
                            "probe_mse": _finite_or_none(mse_h1),
                            "term_count": _head_term_count(head1),
                            "ridge": _finite_or_none(head1.get("ridge", None)) if isinstance(head1, Mapping) else None,
                            "improvement_vs_mapped": _rel_improve(mse0, mse_h1),
                            "outer_refit_passes": 0,
                        }
                    )
                    score_ladder["head_augmented"] = head_stage
                    mapping = _attach_score_ladder(mapping, score_ladder, acceptance_basis="head_augmented")
                    final_head = head1
                    final_phi_fit = phi_fit_head1
                    final_phi_probe = phi_probe_head1

                    # Alternating refinement: refit mapping <-> head until convergence.
                    # One pass is insufficient when f correlates with head variables.
                    _alt_gain = gain1
                    for _alt in range(4):
                        if _alt_gain < refit_trigger_gain:
                            break
                        y_fit_adj = y_fit - head_pred_fit1
                        fb2 = _fit_best_with_cfg(
                            pred_fit,
                            y_fit_adj,
                            poly_degree,
                            cfg,
                            expr=expr,
                            pred_probe=pred_probe,
                        )
                        if fb2 is None:
                            break
                        _, mapping1 = fb2
                        y_hat_fit1 = eval_mapping(pred_fit, mapping1)
                        y_hat_probe1 = eval_mapping(pred_probe, mapping1)
                        resid_fit1 = (y_fit - y_hat_fit1)
                        resid_probe1 = (y_probe - y_hat_probe1)

                        head_fit2 = _fit_head(resid_fit1, resid_probe1)
                        if head_fit2 is None:
                            break
                        mse_h2, r_h2, head2, head_pred_fit1, head_pred_probe1, phi_fit_head2, phi_probe_head2 = head_fit2
                        _alt_gain = (mse - mse_h2) / max(mse, 1e-30)
                        if not (math.isfinite(mse_h2) and mse_h2 < mse):
                            break
                        mse = mse_h2
                        r = r_h2
                        mapping = dict(mapping1)
                        mapping["_lin_head"] = head2
                        mapping["_score_decomp"] = _head_score_decomp(
                            mse_core=mse0,
                            mse_with_head=mse_h2,
                            head=head2,
                            core_pred_probe=y_hat_probe1,
                            head_pred_probe=head_pred_probe1,
                        )
                        head_refit_passes += 1
                        head_stage = dict(score_ladder.get("head_augmented", {}))
                        head_stage.update(
                            {
                                "available": True,
                                "accepted": True,
                                "probe_mse": _finite_or_none(mse_h2),
                                "term_count": _head_term_count(head2),
                                "ridge": _finite_or_none(head2.get("ridge", None)) if isinstance(head2, Mapping) else None,
                                "improvement_vs_mapped": _rel_improve(mse0, mse_h2),
                                "outer_refit_passes": int(head_refit_passes),
                            }
                        )
                        score_ladder["head_augmented"] = head_stage
                        mapping = _attach_score_ladder(mapping, score_ladder, acceptance_basis="head_augmented")
                        final_head = head2
                        final_phi_fit = phi_fit_head2
                        final_phi_probe = phi_probe_head2
                        if _alt_gain < 1e-3:
                            break  # converged

            direct_term_nodes = []
            direct_fit_cols = []
            direct_probe_cols = []
            for t in head_var_terms:
                v_fit = _eval_term(t, x_fit)
                v_probe = _eval_term(t, x_probe)
                if v_fit is None or v_probe is None:
                    continue
                if (not torch.isfinite(v_fit).all()) or (not torch.isfinite(v_probe).all()):
                    continue
                if float(v_fit.std()) < 1.0e-12:
                    continue
                direct_term_nodes.append(t)
                direct_fit_cols.append(v_fit)
                direct_probe_cols.append(v_probe)
            if final_head is not None:
                for jj, t in enumerate(list(final_head.get("terms", []) or [])):
                    if t in direct_term_nodes:
                        continue
                    try:
                        v_fit = final_phi_fit[:, jj]
                        v_probe = final_phi_probe[:, jj]
                    except Exception:
                        continue
                    direct_term_nodes.append(t)
                    direct_fit_cols.append(v_fit)
                    direct_probe_cols.append(v_probe)

            if direct_term_nodes:
                phi_fit_direct = torch.stack(direct_fit_cols, dim=1)
                phi_probe_direct = torch.stack(direct_probe_cols, dim=1)
                direct_combo = _try_direct_combo(
                    expr,
                    pred_fit,
                    pred_probe,
                    direct_term_nodes,
                    phi_fit_direct,
                    phi_probe_direct,
                    mse,
                    score_ladder=score_ladder,
                )
                if direct_combo is not None:
                    return _with_score_diagnostics(
                        _pick_best_equiv_score([direct_combo, compiled_pade], y_var=y_var),
                        finite_diag,
                        domain_diag,
                    )

        mapping = _attach_score_ladder(mapping, score_ladder, acceptance_basis=_acceptance_basis_for_mapping(mapping))
        key, z = fingerprint(r, proj, fp_mode, q_scale, q_clip)
        return _with_score_diagnostics(
            _pick_best_equiv_score([(mse, key, z, mapping, expr), compiled_pade], y_var=y_var),
            finite_diag,
            domain_diag,
        )

    # Variant 1: as-is
    sc0 = _try(p_fit, p_probe, node)
    if sc0 is not None:
        cands.append(sc0)

    # Variant 2: negated representative (covers the whole mapping-equivalence class)
    node_neg = simplify(_negate_smart(node))
    if node_str(node_neg) != node_str(node):
        if _skip_negated_equiv_for_affine_poly_only(poly_degree, cfg):
            _diag_inc(cfg, "negated_variant_skipped_affine_poly_only")
        else:
            _diag_inc(cfg, "negated_variant_scores")
            sc1 = _try(-p_fit, -p_probe, node_neg)
            if sc1 is not None:
                cands.append(sc1)

    return _pick_best_equiv_score(cands, y_var=y_var)

def _score_expr_base_joint_affine(node, x_fit, y_fit, x_probe, y_probe, proj, fp_mode, q_scale, q_clip, poly_degree, cfg):
    """Score an expression across multiple datasets using per-dataset affine maps.

    The affine maps are degree-1 poly maps fitted on each dataset's fit split and
    evaluated on each dataset's probe split. The final score is a weighted
    aggregation (points-weighted or datasets-weighted) of the per-dataset probe MSEs.
    """
    if cfg is None or (not bool(cfg.get("joint_score_enable", False))):
        return None
    joint_fit = cfg.get("joint_fit_data", None)
    joint_probe = cfg.get("joint_probe_data", None)
    if (not isinstance(joint_fit, (list, tuple))) or (not isinstance(joint_probe, (list, tuple))):
        return None
    if len(joint_fit) < 2 or len(joint_probe) < 2:
        return None

    # Build id-aligned datasets. If no explicit ids are provided, we align by index.
    fit_by_id = {}
    fit_order = []
    for i, row in enumerate(joint_fit):
        if row is None:
            continue
        did = None
        if isinstance(row, (tuple, list)) and len(row) == 3:
            did, x_d, y_d = row[0], row[1], row[2]
        elif isinstance(row, (tuple, list)) and len(row) == 2:
            did, x_d, y_d = str(i), row[0], row[1]
        else:
            continue
        if not (torch.is_tensor(x_d) and torch.is_tensor(y_d)):
            continue
        if y_d.dim() == 1:
            y_d = y_d.unsqueeze(-1)
        did_s = str(did)
        fit_by_id[did_s] = (x_d, y_d)
        fit_order.append(did_s)

    probe_by_id = {}
    for i, row in enumerate(joint_probe):
        if row is None:
            continue
        did = None
        if isinstance(row, (tuple, list)) and len(row) == 3:
            did, x_d, y_d = row[0], row[1], row[2]
        elif isinstance(row, (tuple, list)) and len(row) == 2:
            did, x_d, y_d = str(i), row[0], row[1]
        else:
            continue
        if not (torch.is_tensor(x_d) and torch.is_tensor(y_d)):
            continue
        if y_d.dim() == 1:
            y_d = y_d.unsqueeze(-1)
        did_s = str(did)
        probe_by_id[did_s] = (x_d, y_d)

    pairs = []
    for did in fit_order:
        if did not in probe_by_id:
            continue
        xf, yf = fit_by_id[did]
        xp, yp = probe_by_id[did]
        pairs.append((did, xf, yf, xp, yp))
    if len(pairs) < 2:
        return None

    # Dataset weights (default: points-weighted on the probe split).
    w = _joint_dataset_weights([(xp, yp) for (_did, _xf, _yf, xp, yp) in pairs], cfg)
    if w is None or int(w.numel()) != len(pairs):
        return None

    mse_total = torch.zeros((), dtype=w.dtype, device=w.device)
    r_parts = []
    proj_parts = []
    p_fit_parts = []
    y_fit_parts = []
    per_ds = []
    domain_ds_diags = []
    finite_mask_enabled = bool(cfg.get("score_finite_mask_enable", False))
    min_fit_frac = _finite_mask_threshold(cfg, "score_finite_mask_min_fit_frac", 0.98)
    min_probe_frac = _finite_mask_threshold(cfg, "score_finite_mask_min_dataset_frac", 0.95)
    min_points = _finite_mask_min_points(cfg)
    probe_offset = 0

    for wi, (did, xf, yf, xp, yp) in zip(w, pairs):
        p_fit, domain_fit_diag = _eval_score_col(node, xf, cfg)
        yf = _as_score_col(yf, rows=int(xf.shape[0]))
        p_probe, domain_probe_diag = _eval_score_col(node, xp, cfg)
        yp = _as_score_col(yp, rows=int(xp.shape[0]))
        if p_fit is None or yf is None or p_probe is None or yp is None:
            _bump_score_stat(cfg, "domain_projection_rejected_joint")
            return None
        domain_diag_d = merge_domain_projection_diagnostics(
            domain_fit_diag,
            domain_probe_diag,
            labels=(f"{did}:fit", f"{did}:probe"),
        )
        if not domain_projection_is_acceptable(domain_diag_d):
            _bump_score_stat(cfg, "domain_projection_rejected_joint")
            return None
        probe_slice = proj[probe_offset:probe_offset + int(xp.shape[0])]
        probe_offset += int(xp.shape[0])

        fit_mask = (torch.isfinite(p_fit).reshape(-1)) & (torch.isfinite(yf).reshape(-1))
        probe_mask = (torch.isfinite(p_probe).reshape(-1)) & (torch.isfinite(yp).reshape(-1))
        if finite_mask_enabled:
            if not _finite_mask_ok(fit_mask, min_frac=min_fit_frac, min_points=min_points):
                _bump_score_stat(cfg, "finite_mask_rejected_joint_fit")
                return None
            if not _finite_mask_ok(probe_mask, min_frac=min_probe_frac, min_points=min_points):
                _bump_score_stat(cfg, "finite_mask_rejected_joint_probe")
                return None
            p_fit_s = p_fit[fit_mask]
            yf_s = yf[fit_mask]
            p_probe_s = p_probe[probe_mask]
            yp_s = yp[probe_mask]
            probe_slice_s = probe_slice[probe_mask]
        else:
            if (not torch.isfinite(p_fit).all()) or (not torch.isfinite(p_probe).all()):
                return None
            p_fit_s = p_fit
            yf_s = yf
            p_probe_s = p_probe
            yp_s = yp
            probe_slice_s = probe_slice

        if float(p_fit_s.std()) < 1.0e-12:
            return None
        fb = fit_poly(p_fit_s, yf_s, degree=1, affine_fast=True, diagnostics=_refine_diag(cfg))
        if fb is None:
            return None
        sol, mu, std = fb
        mapping_d = {"kind": "poly", "coeffs": [float(sol[0]), float(sol[1])], "mu": float(mu), "std": float(std)}
        y_hat = eval_mapping(p_probe_s, mapping_d)
        r = (yp_s - y_hat).squeeze(-1)
        mse_d = (r * r).mean()
        if not torch.isfinite(mse_d):
            return None
        mse_total = mse_total + wi * mse_d
        r_parts.append(r)
        proj_parts.append(probe_slice_s)
        p_fit_parts.append(p_fit_s)
        y_fit_parts.append(yf_s)
        ds_row = {
            "id": did,
            "mapping": mapping_d,
            "mse": float(mse_d.detach().cpu()),
            "n_fit": int(yf_s.shape[0]),
            "n_probe": int(yp_s.shape[0]),
            "fit_valid_frac": float(_mask_fraction(fit_mask)),
            "probe_valid_frac": float(_mask_fraction(probe_mask)),
            "fit_total_rows": int(fit_mask.numel()),
            "probe_total_rows": int(probe_mask.numel()),
        }
        if isinstance(domain_diag_d, Mapping) and bool(domain_diag_d.get("enabled", False)):
            ds_row["domain_projection"] = dict(domain_diag_d)
            domain_ds_diags.append(domain_diag_d)
        per_ds.append(ds_row)

    if not torch.isfinite(mse_total):
        return None

    r_all = torch.cat(r_parts, dim=0) if r_parts else None
    proj_all = torch.cat(proj_parts, dim=0) if proj_parts else None
    if r_all is None or proj_all is None or int(r_all.numel()) != int(proj_all.shape[0]):
        return None

    # A representative (pooled) affine map for embedding/serialization.
    mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
    try:
        p_fit_all = torch.cat(p_fit_parts, dim=0)
        y_fit_all = torch.cat(y_fit_parts, dim=0)
        fb_all = fit_poly(p_fit_all, y_fit_all, degree=1, affine_fast=True, diagnostics=_refine_diag(cfg))
        if fb_all is not None:
            sol, mu, std = fb_all
            mapping = {"kind": "poly", "coeffs": [float(sol[0]), float(sol[1])], "mu": float(mu), "std": float(std)}
    except Exception:
        pass

    mapping["_joint_affine"] = {"weight_mode": str(cfg.get("joint_weight_mode", "points")), "datasets": per_ds}
    mapping = _attach_finite_mask_diag(
        mapping,
        _finite_mask_diag(enabled=finite_mask_enabled, per_dataset=per_ds),
    )
    mapping = _attach_domain_projection_diag(
        mapping,
        merge_domain_projection_diagnostics(*domain_ds_diags) if domain_ds_diags else None,
    )

    mse = float(mse_total.detach().cpu())
    if not math.isfinite(mse):
        return None
    ladder = _score_ladder_template(node, None, mapping, mse)
    mapping = _attach_score_ladder(
        mapping,
        ladder,
        acceptance_basis="joint_affine_structural" if mapping_is_structural(mapping) else "joint_affine",
    )
    key, z = fingerprint(r_all, proj_all, fp_mode, q_scale, q_clip)
    return mse, key, z, mapping

def _score_expr_base_joint_linear_terms(node, x_fit, y_fit, x_probe, y_probe, proj, fp_mode, q_scale, q_clip, poly_degree, cfg):
    """Score an expression across multiple datasets using per-dataset linear term coefficients.

    Uses the expression's additive terms as a basis (when enabled via
    ``linear_combo_enable``) and fits the linear coefficients independently per
    dataset on that dataset's fit split. The fitted coefficients are then
    evaluated on the corresponding probe split.

    This generalises the joint affine mapping (degree-1 poly on f(x)) to multiple
    per-dataset parameters (one coefficient per additive term, plus an intercept).
    """
    if cfg is None or (not bool(cfg.get("joint_score_enable", False))):
        return None
    if not bool(cfg.get("joint_terms_enable", False)):
        return None
    joint_fit = cfg.get("joint_fit_data", None)
    joint_probe = cfg.get("joint_probe_data", None)
    if (not isinstance(joint_fit, (list, tuple))) or (not isinstance(joint_probe, (list, tuple))):
        return None
    if len(joint_fit) < 2 or len(joint_probe) < 2:
        return None

    # Build id-aligned datasets. If no explicit ids are provided, we align by index.
    fit_by_id = {}
    fit_order = []
    for i, row in enumerate(joint_fit):
        if row is None:
            continue
        did = None
        if isinstance(row, (tuple, list)) and len(row) == 3:
            did, x_d, y_d = row[0], row[1], row[2]
        elif isinstance(row, (tuple, list)) and len(row) == 2:
            did, x_d, y_d = str(i), row[0], row[1]
        else:
            continue
        if not (torch.is_tensor(x_d) and torch.is_tensor(y_d)):
            continue
        if y_d.dim() == 1:
            y_d = y_d.unsqueeze(-1)
        did_s = str(did)
        fit_by_id[did_s] = (x_d, y_d)
        fit_order.append(did_s)

    probe_by_id = {}
    for i, row in enumerate(joint_probe):
        if row is None:
            continue
        did = None
        if isinstance(row, (tuple, list)) and len(row) == 3:
            did, x_d, y_d = row[0], row[1], row[2]
        elif isinstance(row, (tuple, list)) and len(row) == 2:
            did, x_d, y_d = str(i), row[0], row[1]
        else:
            continue
        if not (torch.is_tensor(x_d) and torch.is_tensor(y_d)):
            continue
        if y_d.dim() == 1:
            y_d = y_d.unsqueeze(-1)
        did_s = str(did)
        probe_by_id[did_s] = (x_d, y_d)

    pairs = []
    for did in fit_order:
        if did not in probe_by_id:
            continue
        xf, yf = fit_by_id[did]
        xp, yp = probe_by_id[did]
        pairs.append((did, xf, yf, xp, yp))
    if len(pairs) < 2:
        return None

    basis_nodes = _select_linear_basis_nodes(node, cfg)
    if not isinstance(basis_nodes, (list, tuple)) or len(basis_nodes) == 0:
        return None
    term_nodes = list(basis_nodes)

    # Dataset weights (default: points-weighted on the probe split).
    w = _joint_dataset_weights([(xp, yp) for (_did, _xf, _yf, xp, yp) in pairs], cfg)
    if w is None or int(w.numel()) != len(pairs):
        return None

    ridge = float(cfg.get("linear_ridge", 1.0e-8))
    mse_total = torch.zeros((), dtype=w.dtype, device=w.device)
    r_parts = []
    proj_parts = []
    per_ds = []
    p_fit_parts = []
    y_fit_parts = []
    domain_ds_diags = []
    finite_mask_enabled = bool(cfg.get("score_finite_mask_enable", False))
    min_fit_frac = _finite_mask_threshold(cfg, "score_finite_mask_min_fit_frac", 0.98)
    min_probe_frac = _finite_mask_threshold(cfg, "score_finite_mask_min_dataset_frac", 0.95)
    min_points = _finite_mask_min_points(cfg)
    probe_offset = 0

    for wi, (did, xf, yf, xp, yp) in zip(w, pairs):
        yf = _as_score_col(yf, rows=int(xf.shape[0]))
        yp = _as_score_col(yp, rows=int(xp.shape[0]))
        if yf is None or yp is None:
            return None
        cols_fit = []
        term_fit_diags = []
        for t in term_nodes:
            v, domain_t_diag = _eval_score_col(t, xf, cfg)
            if v is None:
                _bump_score_stat(cfg, "domain_projection_rejected_joint_terms")
                return None
            term_fit_diags.append(domain_t_diag)
            cols_fit.append(v.reshape(-1, 1))
        if not cols_fit:
            return None
        Phi_fit = torch.cat([torch.ones_like(cols_fit[0]), *cols_fit], dim=1)
        if int(Phi_fit.shape[1]) <= 1:
            return None

        fit_mask = torch.isfinite(yf).reshape(-1)
        for col in cols_fit:
            fit_mask = fit_mask & torch.isfinite(col).reshape(-1)
        if finite_mask_enabled:
            if not _finite_mask_ok(fit_mask, min_frac=min_fit_frac, min_points=min_points):
                _bump_score_stat(cfg, "finite_mask_rejected_joint_terms_fit")
                return None
            Phi_fit_s = Phi_fit[fit_mask]
            yf_s = yf[fit_mask]
        else:
            if not torch.isfinite(Phi_fit).all():
                return None
            Phi_fit_s = Phi_fit
            yf_s = yf
        col_std = float(Phi_fit_s[:, 1:].detach().std(unbiased=False))
        if (not math.isfinite(col_std)) or col_std < 1.0e-12:
            return None

        sol = _solve_linear_coeffs(Phi_fit_s, yf_s, ridge)
        if sol is None or (not torch.isfinite(sol).all()):
            return None

        cols_probe = []
        term_probe_diags = []
        for t in term_nodes:
            v, domain_t_diag = _eval_score_col(t, xp, cfg)
            if v is None:
                _bump_score_stat(cfg, "domain_projection_rejected_joint_terms")
                return None
            term_probe_diags.append(domain_t_diag)
            cols_probe.append(v.reshape(-1, 1))
        if not cols_probe:
            return None
        Phi_probe = torch.cat([torch.ones_like(cols_probe[0]), *cols_probe], dim=1)
        probe_slice = proj[probe_offset:probe_offset + int(xp.shape[0])]
        probe_offset += int(xp.shape[0])
        probe_mask = torch.isfinite(yp).reshape(-1)
        for col in cols_probe:
            probe_mask = probe_mask & torch.isfinite(col).reshape(-1)
        if finite_mask_enabled:
            if not _finite_mask_ok(probe_mask, min_frac=min_probe_frac, min_points=min_points):
                _bump_score_stat(cfg, "finite_mask_rejected_joint_terms_probe")
                return None
            Phi_probe_s = Phi_probe[probe_mask]
            yp_s = yp[probe_mask]
            probe_slice_s = probe_slice[probe_mask]
        else:
            if not torch.isfinite(Phi_probe).all():
                return None
            Phi_probe_s = Phi_probe
            yp_s = yp
            probe_slice_s = probe_slice

        y_hat = Phi_probe_s @ sol
        r = (yp_s - y_hat).squeeze(-1)
        mse_d = (r * r).mean()
        if not torch.isfinite(mse_d):
            return None

        mse_total = mse_total + wi * mse_d
        r_parts.append(r)
        proj_parts.append(probe_slice_s)

        try:
            coeffs = [float(v) for v in sol.squeeze(-1).detach().cpu().tolist()]
        except Exception:
            return None

        domain_diag_d = merge_domain_projection_diagnostics(
            *term_fit_diags,
            *term_probe_diags,
        )
        if not domain_projection_is_acceptable(domain_diag_d):
            _bump_score_stat(cfg, "domain_projection_rejected_joint_terms")
            return None

        ds_row = {
            "id": did,
            "coeffs": coeffs,
            "mse": float(mse_d.detach().cpu()),
            "n_fit": int(yf_s.shape[0]),
            "n_probe": int(yp_s.shape[0]),
            "fit_valid_frac": float(_mask_fraction(fit_mask)),
            "probe_valid_frac": float(_mask_fraction(probe_mask)),
            "fit_total_rows": int(fit_mask.numel()),
            "probe_total_rows": int(probe_mask.numel()),
        }
        if isinstance(domain_diag_d, Mapping) and bool(domain_diag_d.get("enabled", False)):
            ds_row["domain_projection"] = dict(domain_diag_d)
            domain_ds_diags.append(domain_diag_d)
        per_ds.append(ds_row)

        # For a representative pooled affine mapping (used for embedding/serialization).
        try:
            p_fit_node, _ = _eval_score_col(node, xf, cfg)
            if p_fit_node is not None:
                p_fit_parts.append(p_fit_node)
                y_fit_parts.append(yf)
        except Exception:
            pass

    if not torch.isfinite(mse_total):
        return None

    r_all = torch.cat(r_parts, dim=0) if r_parts else None
    proj_all = torch.cat(proj_parts, dim=0) if proj_parts else None
    if r_all is None or proj_all is None or int(r_all.numel()) != int(proj_all.shape[0]):
        return None

    # Representative (pooled) affine mapping for embedding/serialization (same policy as joint affine).
    mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
    try:
        if p_fit_parts and y_fit_parts:
            p_fit_all = torch.cat(p_fit_parts, dim=0)
            y_fit_all = torch.cat(y_fit_parts, dim=0)
            fb_all = fit_poly(p_fit_all, y_fit_all, degree=1, affine_fast=True, diagnostics=_refine_diag(cfg))
            if fb_all is not None:
                sol_aff, mu, std = fb_all
                mapping = {"kind": "poly", "coeffs": [float(sol_aff[0]), float(sol_aff[1])], "mu": float(mu), "std": float(std)}
    except Exception:
        pass

    mapping["_joint_linear_terms"] = {
        "weight_mode": str(cfg.get("joint_weight_mode", "points")),
        "terms": term_nodes,
        "datasets": per_ds,
    }
    mapping = _attach_finite_mask_diag(
        mapping,
        _finite_mask_diag(enabled=finite_mask_enabled, per_dataset=per_ds),
    )
    mapping = _attach_domain_projection_diag(
        mapping,
        merge_domain_projection_diagnostics(*domain_ds_diags) if domain_ds_diags else None,
    )

    mse = float(mse_total.detach().cpu())
    if not math.isfinite(mse):
        return None
    ladder = _score_ladder_template(node, None, mapping, None)
    head_stage = dict(ladder.get("head_augmented", {}))
    head_stage.update(
        {
            "available": True,
            "accepted": True,
            "probe_mse": _finite_or_none(mse),
            "term_count": int(len(term_nodes)),
            "kind": "joint_linear_terms",
        }
    )
    ladder["head_augmented"] = head_stage
    mapping = _attach_score_ladder(mapping, ladder, acceptance_basis="joint_linear_terms")
    key, z = fingerprint(r_all, proj_all, fp_mode, q_scale, q_clip)
    return mse, key, z, mapping

def _eval_node_hparam_safe(node, x, hparams, cfg):
    eps = max(float(cfg.get("safe_eps", 1.0e-6)), 1.0e-12)
    exp_clip = max(float(cfg.get("safe_exp_clip", 30.0)), 1.0)
    zero = torch.zeros((), dtype=x.dtype, device=x.device)

    op = node[0]
    if op in ("var", "const", "hparam"):
        return _eval_node_hparam(node, x, hparams), zero

    if op in UNARY_OPS or op in ("asin", "acos"):
        a, pa = _eval_node_hparam_safe(node[1], x, hparams, cfg)
        if op == "sin":
            return torch.sin(a), pa
        if op == "cos":
            return torch.cos(a), pa
        if op == "asin":
            corr = torch.nn.functional.softplus(a.abs() - 1.0)
            a_safe = torch.clamp(a, min=-1.0, max=1.0)
            return torch.asin(a_safe), pa + corr.mean()
        if op == "acos":
            corr = torch.nn.functional.softplus(a.abs() - 1.0)
            a_safe = torch.clamp(a, min=-1.0, max=1.0)
            return torch.acos(a_safe), pa + corr.mean()
        if op == "neg":
            return -a, pa
        if op == "sqr":
            return a * a, pa
        if op == "exp":
            over = torch.nn.functional.softplus(a.abs() - exp_clip)
            return torch.exp(torch.clamp(a, min=-exp_clip, max=exp_clip)), pa + over.mean()
        if op == "log":
            corr = torch.nn.functional.softplus(eps - a)
            a_safe = a + corr
            return torch.log(a_safe), pa + corr.mean()
        if op == "sqrt":
            corr = torch.nn.functional.softplus(eps - a)
            a_safe = a + corr
            return torch.sqrt(a_safe), pa + corr.mean()

    if op in BINARY_OPS:
        l, pl = _eval_node_hparam_safe(node[1], x, hparams, cfg)
        r, pr = _eval_node_hparam_safe(node[2], x, hparams, cfg)
        if op == "add":
            return l + r, pl + pr
        if op == "sub":
            return l - r, pl + pr
        if op == "mul":
            return l * r, pl + pr
        if op == "div":
            abs_r = r.abs()
            signed_floor = eps * torch.where(r >= 0, torch.ones_like(r), -torch.ones_like(r))
            denom = r + signed_floor
            out = l / denom
            near_zero = eps / (abs_r + eps)
            penalty = near_zero * near_zero
            return out, pl + pr + penalty.mean()

    # Fallback to strict op if we hit an unknown token; keep finite via penalty.
    try:
        out = _eval_node_hparam(node, x, hparams)
        if torch.isfinite(out).all():
            return out, zero
    except Exception:
        pass
    return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device), torch.tensor(1.0e6, dtype=x.dtype, device=x.device)

def score_expr(
    node, x_fit, y_fit, x_probe, y_probe, proj, fp_mode, q_scale, q_clip, poly_degree,
    refine_enable=False, refine_cfg=None, refine_best_mse=float("inf"), refine_state=None,
    return_expr=False,
):
    cfg = dict(refine_cfg or {})
    cfg["_mapping_family_best_mse"] = (
        float(refine_best_mse) if math.isfinite(float(refine_best_mse)) else None
    )
    use_joint = bool(cfg.get("joint_score_enable", False)) and isinstance(cfg.get("joint_fit_data", None), (list, tuple)) and isinstance(cfg.get("joint_probe_data", None), (list, tuple))
    _diag_inc(cfg, "score_calls")

    def _do_score(expr):
        if use_joint:
            if bool(cfg.get("joint_terms_enable", False)):
                sc = _score_expr_base_joint_linear_terms(expr, x_fit, y_fit, x_probe, y_probe, proj, fp_mode, q_scale, q_clip, poly_degree, cfg)
                if sc is not None:
                    return sc
            sc = _score_expr_base_joint_affine(expr, x_fit, y_fit, x_probe, y_probe, proj, fp_mode, q_scale, q_clip, poly_degree, cfg)
            if sc is not None:
                return sc
        return _score_expr_base(expr, x_fit, y_fit, x_probe, y_probe, proj, fp_mode, q_scale, q_clip, poly_degree, cfg)

    t_base = time.perf_counter()
    base = _do_score(node)
    _diag_add_time(cfg, "base_score_s", time.perf_counter() - t_base)
    if base is None:
        return None
    # Normalize to (mse, key, z, mapping, expr)
    if len(base) == 4:
        base = (base[0], base[1], base[2], base[3], simplify(node))
    node = base[4]

    if use_joint:
        # Joint scoring paths don't run the fast sign-sweep inside _score_expr_base.
        node_neg = simplify(_negate_smart(node))
        if node_str(node_neg) != node_str(node):
            _diag_inc(cfg, "negated_variant_scores")
            sc_neg = _do_score(node_neg)
            if sc_neg is not None:
                if len(sc_neg) == 4:
                    sc_neg = (sc_neg[0], sc_neg[1], sc_neg[2], sc_neg[3], node_neg)
                base = _pick_best_equiv_score([base, sc_neg], y_var=None)
                node = base[4]

    if not refine_enable:
        if return_expr:
            return base[0], base[1], base[2], base[3], node
        return base[0], base[1], base[2], base[3]
    refinement_helpers = _coerce_refinement_helpers(cfg)
    decorate_refine_variants = refinement_helpers["_decorate_refine_variants"]
    materialize_linearized_candidate = refinement_helpers["_materialize_linearized_candidate"]
    refine_hparams = refinement_helpers["_refine_hparams"]
    variant_has_gate_potential = refinement_helpers["_variant_has_gate_potential"]
    _diag_inc(cfg, "refine_score_calls")
    _diag_inc_context(cfg, "refine_score_calls")
    refine_attempted = False
    refined_accepted = False

    if bool(cfg.get("score_head_only", False)):
        base_marked = _mark_refinement_score(
            base,
            enabled=True,
            attempted=False,
            accepted=False,
            source_expr=node,
            accepted_expr=node,
            base_probe_mse=base[0],
        )
        if return_expr:
            return base_marked[0], base_marked[1], base_marked[2], base_marked[3], node
        return base_marked[0], base_marked[1], base_marked[2], base_marked[3]

    gate_factor = float(cfg.get("gate_best_factor", 10.0))
    gate_relax = 1.0
    if refine_state is not None:
        gate_relax = max(1.0, float(refine_state.get("gate_relax_factor", 1.0)))
    gate_factor = gate_factor * gate_relax
    base_mse = float(base[0])
    gate_triggered = False
    if math.isfinite(refine_best_mse) and math.isfinite(base_mse):
        gate_triggered = base_mse > refine_best_mse * max(gate_factor, 1.0)

    max_refines = int(cfg.get("max_refines", 0))
    if not _refine_budget_left(refine_state, max_refines):
        base_marked = _mark_refinement_score(
            base,
            enabled=True,
            attempted=False,
            accepted=False,
            source_expr=node,
            accepted_expr=node,
            base_probe_mse=base[0],
        )
        if return_expr:
            return base_marked[0], base_marked[1], base_marked[2], base_marked[3], node
        return base_marked[0], base_marked[1], base_marked[2], base_marked[3]

    _no_shift = frozenset()
    variants = decorate_refine_variants(
        node,
        max(1, int(cfg.get("max_variants", 4))),
        max(1, int(cfg.get("max_params", 2))),
        x_fit=x_fit,
        y_fit=y_fit,
        cfg=cfg,
    )
    if bool(cfg.get("linear_combo_enable", True)) and len(_select_linear_basis_nodes(node, cfg)) >= 2:
        variants = [(node, 0, _no_shift), *variants]
    if variants:
        seen = set()
        uniq = []
        for var_h, n_params, ss in variants:
            k = (node_str(var_h), int(n_params))
            if k in seen:
                continue
            seen.add(k)
            uniq.append((var_h, int(n_params), ss))
        variants = uniq
    _diag_inc(cfg, "variants_generated", len(variants))

    if gate_triggered:
        _diag_inc(cfg, "gate_triggered_score_calls")
        unlocked = []
        for var_h, n_params, ss in variants:
            if n_params <= 0:
                unlocked.append((var_h, n_params, ss))
                continue
            if variant_has_gate_potential(var_h, n_params, x_fit, y_fit, cfg):
                unlocked.append((var_h, n_params, ss))
        variants = unlocked
        _diag_inc(cfg, "variants_after_gate", len(variants))

    if not variants:
        base_marked = _mark_refinement_score(
            base,
            enabled=True,
            attempted=False,
            accepted=False,
            source_expr=node,
            accepted_expr=node,
            base_probe_mse=base[0],
        )
        if return_expr:
            return base_marked[0], base_marked[1], base_marked[2], base_marked[3], node
        return base_marked[0], base_marked[1], base_marked[2], base_marked[3]

    best = base
    best_expr = node
    for var_h, n_params, shift_slots in variants:
        cache_key = _refine_attempt_cache_key(var_h, n_params, shift_slots, cfg, x_fit, y_fit)
        cached = _refine_cache_get(cfg, cache_key)
        h_star = None
        cand = None
        if isinstance(cached, dict):
            refine_attempted = True
            if cached.get("status") != "ok":
                continue
            h_star = cached.get("hparams")
            cand = cached.get("candidate")
        if cached is None and refine_state is not None:
            if not _refine_budget_left(refine_state, max_refines):
                break
            done = int(refine_state.get("trials_done", 0))
            refine_state["trials_done"] = done + 1
            if refine_state.get("depth_trials_left", None) is not None:
                refine_state["depth_trials_left"] = int(refine_state["depth_trials_left"]) - 1
            if refine_state.get("window_trials_left", None) is not None:
                refine_state["window_trials_left"] = int(refine_state["window_trials_left"]) - 1

        if cached is None:
            refine_attempted = True
            _diag_inc(cfg, "refinement_attempts")
            _diag_inc_context(cfg, "refinement_attempts")
            h_star = refine_hparams(var_h, n_params, x_fit, y_fit, cfg, shift_slots=shift_slots)
        if h_star is None:
            if cached is None:
                _refine_cache_put(cfg, cache_key, {"status": "no_hparams"})
            continue
        # Scale slots must be positive; shift slots may be any finite value
        if any(
            (not math.isfinite(v)) or (i not in shift_slots and v <= 0.0)
            for i, v in enumerate(h_star)
        ):
            if cached is None:
                _refine_cache_put(
                    cfg,
                    cache_key,
                    {"status": "invalid_hparams", "hparams": list(h_star)},
                )
            continue
        if cand is None:
            cand = materialize_linearized_candidate(var_h, h_star, x_fit, y_fit, cfg)
            if cached is None:
                _refine_cache_put(
                    cfg,
                    cache_key,
                    {"status": "ok", "hparams": list(h_star), "candidate": cand},
                )
        _diag_inc(cfg, "materialized_rescores")
        _diag_inc_context(cfg, "materialized_rescores")
        sc = _do_score(cand)
        if sc is None:
            continue
        if len(sc) == 4:
            sc = (sc[0], sc[1], sc[2], sc[3], cand)
        cand_best = sc[4]
        if sc[0] < best[0]:
            # Only print when beating the global best, not just this skeleton's base
            if bool(cfg.get("verbose", True)) and sc[0] < refine_best_mse:
                src_raw = node_str(node)
                dst_raw = node_str(cand_best)
                src_can = node_str(_mapping_equiv_root(node))
                dst_can = node_str(_mapping_equiv_root(cand_best))
                if src_can != src_raw or dst_can != dst_raw:
                    print(
                        f"  [skeleton-refine] NEW BEST {src_can} -> {dst_can}  "
                        f"[raw {src_raw} -> {dst_raw}]  "
                        f"(mse {refine_best_mse:.6g} -> {sc[0]:.6g}, "
                        f"hparams={[f'{v:.4g}' for v in h_star]})"
                    )
                else:
                    print(
                        f"  [skeleton-refine] NEW BEST {src_raw} -> {dst_raw}  "
                        f"(mse {refine_best_mse:.6g} -> {sc[0]:.6g}, "
                        f"hparams={[f'{v:.4g}' for v in h_star]})"
                    )
            best = sc
            best_expr = cand_best
            refined_accepted = True
            _diag_inc(cfg, "accepted_refinements")
            _diag_inc_context(cfg, "accepted_refinements")

    best = _mark_refinement_score(
        best,
        enabled=True,
        attempted=refine_attempted,
        accepted=refined_accepted,
        source_expr=node,
        accepted_expr=best_expr,
        base_probe_mse=base[0],
    )
    if return_expr:
        return best[0], best[1], best[2], best[3], best_expr
    return best[0], best[1], best[2], best[3]

__all__ = [
    '_eval_node_hparam_safe',
    '_harvest_pool_from_archive',
    '_mapping_equiv_root',
    'fingerprint',
    'score_expr',
]
