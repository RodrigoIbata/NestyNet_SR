# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

"""Operator-factorized DE data structures, numerics, and lane policy."""

from typing import TYPE_CHECKING
import math
import multiprocessing as mp
import os
from functools import cmp_to_key
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import torch
from nestynet_sr.sr_core.ast_simplify import ast_node_count, simplify_ast
from nestynet_sr.sr_core.bridges import AbsNode, AcosNode, Add, AddNode, AsinNode, AtomNode, AtanNode, ConstNode, CosNode, D2U, DU, ExpNode, LogNode, Mul, MulNode, PowNode, SinNode, U, Var
from nestynet_sr.sr_de.de_search import ridge_lstsq
from nestynet_sr.sr_search.factorized_search.bridge import factorized_search_to_nestynet, embed_mapping_in_ast, remap_var_to_exprs
from nestynet_sr.sr_search.factorized_search.domain_projection import domain_projection_is_acceptable

from ._factorized_de_frontend import (
    DEFeatureGroup,
)

if TYPE_CHECKING:
    from ._factorized_de_lanes import (
        _finite_design_rows,
    )
    from ._factorized_de_explorer import (
        _run_one_typed_explorer_launch,
    )

@dataclass
class FactorizedDERescueConfig:
    mode: str = "never"  # never | auto | always
    trigger_val_rms: float = 1.0e-3
    trigger_cond: float = 1.0e8
    replace_rel_factor: float = 0.98
    ratio_rel_eps: float = 1.0e-2
    min_ratio_rows: int = 128
    shortlist_topk: int = 8
    base_modes: tuple[str, ...] = ("zero", "primary")
    two_block_shared_coord_mode: str = "never"  # never | auto | always
    typed_lane_workers: int = 1
    hp: Any | None = None


@dataclass(frozen=True)
class TypedExplorerLaunchTask:
    task_id: int
    lane: str
    base_mode: str
    order: int
    x_axis: int
    carrier_ast: Any
    coord_ast: Any
    rel_eps: float
    min_ratio_rows: int
    n_iter: int
    max_depth: int
    explorer_topk: int
    seed: int
    sample_seed: int
    dtype_name: str
    explorer_fit_cap: int | None
    explorer_probe_cap: int | None


@dataclass(frozen=True)
class TypedExplorerLaunchState:
    groups: Sequence[DEFeatureGroup]
    resid_fit_parts: Sequence[torch.Tensor]
    resid_probe_parts: Sequence[torch.Tensor]


@dataclass
class TypedExplorerLaunchResult:
    task_id: int
    ok: bool
    launched: bool
    rows: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    carrier_ast: Any | None = None
    coord_ast: Any | None = None
    fit_rows_full: int = 0
    probe_rows_full: int = 0
    fit_rows_search: int = 0
    probe_rows_search: int = 0
    error: str | None = None


_TYPED_EXPLORER_WORKER_STATE: TypedExplorerLaunchState | None = None


def _multiprocessing_start_method_name() -> str:
    try:
        return str(mp.get_context().get_start_method())
    except Exception:
        return ""


def _typed_explorer_worker_init(shared_state: TypedExplorerLaunchState) -> None:
    global _TYPED_EXPLORER_WORKER_STATE
    _TYPED_EXPLORER_WORKER_STATE = shared_state
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _typed_explorer_worker_run(task: TypedExplorerLaunchTask) -> TypedExplorerLaunchResult:
    state = _TYPED_EXPLORER_WORKER_STATE
    if state is None:
        raise RuntimeError("typed explorer worker state was not initialized")
    return _run_one_typed_explorer_launch(task, state)


@dataclass
class FactorizedDEBlock:
    role: str
    carrier_ast: Any
    coord_ast: Any
    coeff_ast: Any
    block_ast: Any
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class FactorizedDEResult:
    order: int
    x_axis: int
    nonanchor_ast: Any
    residual_ast: Any
    canonical_equation: str
    probe_mse: float
    probe_rms: float
    blocks: list[FactorizedDEBlock]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    engine: str = "factorized"
    residual_ast_raw: Any | None = None
    residual_ast_simplified: Any | None = None
    canonical_equation_raw: str | None = None
    canonical_equation_simplified: str | None = None

    def format_equation(self) -> str:
        return str(self.canonical_equation)


def _anchor_ast(order: int, *, x_axis: int):
    if int(order) == 1:
        return DU(int(x_axis))
    if int(order) == 2:
        return D2U(int(x_axis), int(x_axis))
    raise ValueError(f"Unsupported order={order}")


def _anchor_name(order: int, *, x_axis: int) -> str:
    var_name = f"x{int(x_axis)}"
    if int(order) == 1:
        return f"u_{var_name}"
    if int(order) == 2:
        return f"u_{var_name}{var_name}"
    return f"d^{int(order)}u/d{var_name}^{int(order)}"


def _canonical_equation(order: int, x_axis: int, nonanchor_ast) -> str:
    if nonanchor_ast is None:
        return f"{_anchor_name(order, x_axis=x_axis)} = 0"
    return f"{_anchor_name(order, x_axis=x_axis)} + {repr(nonanchor_ast)} = 0"


def _simplify_de_ast(node: Any | None) -> Any | None:
    if node is None:
        return None
    try:
        return simplify_ast(node)
    except Exception:
        return node


def _compiled_de_ast_payload(*, rhs_ast: Any | None = None, residual_ast: Any | None = None) -> dict[str, Any]:
    rhs_simplified = _simplify_de_ast(rhs_ast)
    residual_simplified = _simplify_de_ast(residual_ast)
    return {
        "rhs_ast_raw": rhs_ast,
        "residual_ast_raw": residual_ast,
        "rhs_ast_simplified": rhs_simplified,
        "residual_ast_simplified": residual_simplified,
        "symbolic_size_raw": None if residual_ast is None else int(ast_node_count(residual_ast)),
        "symbolic_size_simplified": None
        if residual_simplified is None
        else int(ast_node_count(residual_simplified)),
    }


def _compiled_de_row_payload(*, rhs_ast: Any | None = None, residual_ast: Any | None = None) -> dict[str, Any]:
    payload = _compiled_de_ast_payload(rhs_ast=rhs_ast, residual_ast=residual_ast)
    rhs_simplified = payload["rhs_ast_simplified"]
    residual_simplified = payload["residual_ast_simplified"]
    return {
        "rhs_ast_raw": None if rhs_ast is None else repr(rhs_ast),
        "residual_ast_raw": None if residual_ast is None else repr(residual_ast),
        "rhs_ast_simplified": None if rhs_simplified is None else repr(rhs_simplified),
        "residual_ast_simplified": None if residual_simplified is None else repr(residual_simplified),
        "symbolic_size_raw": payload["symbolic_size_raw"],
        "symbolic_size_simplified": payload["symbolic_size_simplified"],
    }


def _split_tensor(features, split: str, name: str) -> torch.Tensor:
    t = getattr(features, f"{name}_{split}")
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float64)
    return t


def _const_like(ref: torch.Tensor, value: Any) -> torch.Tensor:
    return torch.full_like(ref, ConstNode(value).value)


def _eval_ast_on_features(node, *, features, split: str, x_axis: int) -> torch.Tensor:
    x = _split_tensor(features, split, "x")
    u = _split_tensor(features, split, "u")
    du = _split_tensor(features, split, "du")
    d2u = _split_tensor(features, split, "d2u")

    if node is None:
        return torch.ones_like(u)

    if isinstance(node, ConstNode):
        return _const_like(u, node.value)

    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        kwargs = dict(getattr(node, "kwargs", {}) or {})
        if kind in ("var", "x", "input"):
            idxs = list(getattr(node, "var_idxs", ()) or ())
            if len(idxs) != 1 or int(idxs[0]) != int(x_axis):
                raise ValueError(f"Unsupported Var indices in factorized DE eval: {idxs!r}")
            return x[:, int(x_axis): int(x_axis) + 1]
        if kind in ("u", "field", "state"):
            return u
        if kind in ("du", "d1u", "grad_u"):
            axis = int(kwargs.get("axis", x_axis))
            if axis != int(x_axis):
                raise ValueError(f"Unsupported du axis={axis} for x_axis={x_axis}")
            return du
        if kind in ("d2u", "ddu", "hess_u"):
            a0 = int(kwargs.get("axis0", x_axis))
            a1 = int(kwargs.get("axis1", x_axis))
            if a0 != int(x_axis) or a1 != int(x_axis):
                raise ValueError(f"Unsupported d2u axes=({a0},{a1}) for x_axis={x_axis}")
            return d2u
        if kind in ("const", "constant"):
            return _const_like(u, kwargs.get("value", 1.0))
        if kind in ("free_const", "freeconst", "free_constant", "scale"):
            return _const_like(u, kwargs.get("init", 1.0))
        if kind in ("fixed_const", "fixedconst", "fixed_constant"):
            return _const_like(u, kwargs.get("value", 1.0))
        raise ValueError(f"Unsupported atom kind in factorized DE eval: {kind!r}")

    if isinstance(node, AddNode):
        return _eval_ast_on_features(node.left, features=features, split=split, x_axis=x_axis) + _eval_ast_on_features(
            node.right, features=features, split=split, x_axis=x_axis
        )
    if isinstance(node, MulNode):
        return _eval_ast_on_features(node.left, features=features, split=split, x_axis=x_axis) * _eval_ast_on_features(
            node.right, features=features, split=split, x_axis=x_axis
        )
    if isinstance(node, PowNode):
        base = _eval_ast_on_features(node.base, features=features, split=split, x_axis=x_axis)
        exponent = node.exponent
        if isinstance(exponent, dict):
            raise ValueError("Dict exponents are not supported in factorized DE eval")
        if isinstance(exponent, (int, float)):
            return torch.pow(base, float(exponent))
        expv = _eval_ast_on_features(exponent, features=features, split=split, x_axis=x_axis)
        if int(expv.numel()) == 1:
            return torch.pow(base, float(expv.reshape(-1)[0].detach().cpu().item()))
        return torch.pow(base, expv)
    if isinstance(node, LogNode):
        return torch.log(_eval_ast_on_features(node.arg, features=features, split=split, x_axis=x_axis))
    if isinstance(node, ExpNode):
        return torch.exp(_eval_ast_on_features(node.arg, features=features, split=split, x_axis=x_axis))
    if isinstance(node, SinNode):
        return torch.sin(_eval_ast_on_features(node.arg, features=features, split=split, x_axis=x_axis))
    if isinstance(node, CosNode):
        return torch.cos(_eval_ast_on_features(node.arg, features=features, split=split, x_axis=x_axis))
    if isinstance(node, AsinNode):
        return torch.asin(_eval_ast_on_features(node.arg, features=features, split=split, x_axis=x_axis))
    if isinstance(node, AcosNode):
        return torch.acos(_eval_ast_on_features(node.arg, features=features, split=split, x_axis=x_axis))
    if isinstance(node, AtanNode):
        return torch.atan(_eval_ast_on_features(node.arg, features=features, split=split, x_axis=x_axis))
    if isinstance(node, AbsNode):
        return torch.abs(_eval_ast_on_features(node.arg, features=features, split=split, x_axis=x_axis))

    raise TypeError(f"Unsupported AST node type in factorized DE eval: {type(node).__name__}")


def _anchor_tensor(features, *, order: int) -> tuple[torch.Tensor, torch.Tensor]:
    if int(order) == 1:
        return _split_tensor(features, "fit", "du"), _split_tensor(features, "probe", "du")
    if int(order) == 2:
        return _split_tensor(features, "fit", "d2u"), _split_tensor(features, "probe", "d2u")
    raise ValueError(f"Unsupported order={order}")


def _sum_linear_terms_ast(term_asts: Sequence[Any], coeffs: Sequence[float]):
    out = None
    for coeff, term_ast in zip(list(coeffs), list(term_asts)):
        c = float(coeff)
        if abs(c) < 1.0e-14:
            continue
        term = ConstNode(c) if term_ast is None else Mul(ConstNode(c), term_ast)
        out = term if out is None else Add(out, term)
    return out


def _shared_base_from_primary(
    primary,
    groups: Sequence[DEFeatureGroup],
    *,
    order: int,
    x_axis: int,
    dtype: torch.dtype,
):
    if primary is None or int(getattr(primary, "order", -1)) != int(order):
        base_fit = [torch.zeros_like(g.features.u_fit.reshape(-1)) for g in groups]
        base_probe = [torch.zeros_like(g.features.u_probe.reshape(-1)) for g in groups]
        return base_fit, base_probe, None, None

    term_asts = list(getattr(primary, "term_asts", []) or [])
    if not term_asts:
        base_fit = [torch.zeros_like(g.features.u_fit.reshape(-1)) for g in groups]
        base_probe = [torch.zeros_like(g.features.u_probe.reshape(-1)) for g in groups]
        return base_fit, base_probe, None, torch.zeros((0,), dtype=dtype)

    Phi_fit_parts: list[torch.Tensor] = []
    Phi_probe_parts: list[torch.Tensor] = []
    y_fit_parts: list[torch.Tensor] = []

    for group in groups:
        cols_fit = []
        cols_probe = []
        for term_ast in term_asts:
            if term_ast is None:
                cols_fit.append(torch.ones_like(group.features.u_fit.reshape(-1)))
                cols_probe.append(torch.ones_like(group.features.u_probe.reshape(-1)))
            else:
                cols_fit.append(
                    _eval_ast_on_features(term_ast, features=group.features, split="fit", x_axis=int(x_axis)).reshape(-1)
                )
                cols_probe.append(
                    _eval_ast_on_features(term_ast, features=group.features, split="probe", x_axis=int(x_axis)).reshape(-1)
                )
        Phi_fit_parts.append(torch.stack(cols_fit, dim=1))
        Phi_probe_parts.append(torch.stack(cols_probe, dim=1))
        anchor_fit, _ = _anchor_tensor(group.features, order=int(order))
        y_fit_parts.append((-anchor_fit).reshape(-1))

    coeffs_raw = getattr(primary, "coeffs", None)
    if coeffs_raw is None:
        coeffs_shared = ridge_lstsq(torch.cat(Phi_fit_parts, dim=0), torch.cat(y_fit_parts, dim=0), ridge=0.0)
    else:
        coeffs_t = (
            coeffs_raw.detach().to(dtype=dtype, device=Phi_fit_parts[0].device)
            if torch.is_tensor(coeffs_raw)
            else torch.as_tensor(coeffs_raw, dtype=dtype, device=Phi_fit_parts[0].device)
        )
        if coeffs_t.ndim == 2:
            coeffs_shared = ridge_lstsq(torch.cat(Phi_fit_parts, dim=0), torch.cat(y_fit_parts, dim=0), ridge=0.0)
        else:
            coeffs_shared = coeffs_t.reshape(-1)

    base_fit = [(Phi @ coeffs_shared).reshape(-1) for Phi in Phi_fit_parts]
    base_probe = [(Phi @ coeffs_shared).reshape(-1) for Phi in Phi_probe_parts]
    base_ast = _sum_linear_terms_ast(term_asts, coeffs_shared.detach().cpu().tolist())
    return base_fit, base_probe, base_ast, coeffs_shared.detach().cpu()


def _dedupe_ast_list(items: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for item in list(items):
        key = repr(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _coord_pool(*, cfg, order: int, x_axis: int) -> list[Any]:
    out: list[Any] = []
    x = Var(int(x_axis))
    u = U()
    if bool(getattr(cfg, "include_x", True)):
        out.extend([x, Add(ConstNode(1.0), x)])
    if bool(getattr(cfg, "include_u", True)):
        out.extend([u, Add(ConstNode(1.0), u)])
    if int(order) == 2 and bool(getattr(cfg, "include_du", True)):
        du = DU(int(x_axis))
        out.extend([du, Add(ConstNode(1.0), du)])
    return _dedupe_ast_list(out)


def _carrier_pool(*, cfg, order: int, x_axis: int) -> list[Any]:
    out: list[Any] = []
    if bool(getattr(cfg, "include_const", True)):
        out.append(ConstNode(1.0))
    if bool(getattr(cfg, "include_u", True)):
        out.append(U())
    if int(order) == 2 and bool(getattr(cfg, "include_du", True)):
        out.append(DU(int(x_axis)))
    return _dedupe_ast_list(out)


def _safe_ratio_target(resid: torch.Tensor, carrier: torch.Tensor, *, rel_eps: float):
    resid = resid.reshape(-1)
    carrier = carrier.reshape(-1)
    scale = torch.median(torch.abs(carrier)).clamp_min(1.0e-8)
    mask = torch.abs(carrier) > float(rel_eps) * scale
    if int(mask.sum()) <= 0:
        return None, None
    return (-resid[mask] / carrier[mask]).reshape(-1, 1), mask


def _normalized_group_quality_weights(
    groups: Sequence[DEFeatureGroup],
    *,
    min_weight: float = 0.25,
    max_weight: float = 4.0,
    power: float = 0.5,
) -> list[float]:
    if not groups:
        return []

    finite_losses = []
    for group in groups:
        loss = getattr(group, "surrogate_val_loss", None)
        if loss is None:
            continue
        try:
            loss_f = float(loss)
        except Exception:
            continue
        if math.isfinite(loss_f) and loss_f > 0.0:
            finite_losses.append(loss_f)

    if not finite_losses:
        return [1.0 for _ in groups]

    log_losses = torch.log(torch.as_tensor(finite_losses, dtype=torch.float64).clamp_min(1.0e-12))
    ref_loss = max(float(torch.exp(torch.mean(log_losses)).item()), 1.0e-12)
    raw: list[float] = []
    for group in groups:
        loss = getattr(group, "surrogate_val_loss", None)
        try:
            loss_f = float(loss)
        except Exception:
            loss_f = float("nan")
        if not math.isfinite(loss_f) or loss_f <= 0.0:
            raw.append(1.0)
            continue
        denom = max(loss_f, ref_loss * 1.0e-6, 1.0e-12)
        raw.append(math.pow(ref_loss / denom, float(power)))

    clipped = [min(float(max_weight), max(float(min_weight), w)) for w in raw]
    mean_w = sum(clipped) / max(len(clipped), 1)
    if not math.isfinite(mean_w) or mean_w <= 0.0:
        return [1.0 for _ in groups]
    return [float(w / mean_w) for w in clipped]


def _finite_xy_rows(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.isfinite(y.reshape(-1))
    if int(x.ndim) == 1:
        x = x.reshape(-1, 1)
    mask &= torch.isfinite(x).all(dim=1)
    if int(mask.sum()) <= 0:
        return x[:0], y[:0]
    return x[mask], y[mask]


def _prepare_curve_xy(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if int(x.ndim) != 2 or int(x.shape[1]) != 1:
        raise ValueError("Expected 1D coordinates encoded as an (n,1) tensor")
    x1 = x.reshape(-1)
    y1 = y.reshape(-1)
    if int(x1.numel()) <= 0:
        return x1, y1
    order = torch.argsort(x1)
    x_sorted = x1[order]
    y_sorted = y1[order]
    uniq_x, inverse, counts = torch.unique_consecutive(x_sorted, return_inverse=True, return_counts=True)
    if int(uniq_x.numel()) == int(x_sorted.numel()):
        return x_sorted, y_sorted
    sums = torch.zeros_like(uniq_x)
    sums.scatter_add_(0, inverse, y_sorted)
    means = sums / counts.to(dtype=sums.dtype)
    return uniq_x, means


def _prepare_curve_xy_aux(x: torch.Tensor, y: torch.Tensor, aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(x.ndim) != 2 or int(x.shape[1]) != 1:
        raise ValueError("Expected 1D coordinates encoded as an (n,1) tensor")
    x1 = x.reshape(-1)
    y1 = y.reshape(-1)
    aux1 = aux.reshape(-1)
    if int(x1.numel()) <= 0:
        return x1, y1, aux1
    order = torch.argsort(x1)
    x_sorted = x1[order]
    y_sorted = y1[order]
    aux_sorted = aux1[order]
    uniq_x, inverse, counts = torch.unique_consecutive(x_sorted, return_inverse=True, return_counts=True)
    if int(uniq_x.numel()) == int(x_sorted.numel()):
        return x_sorted, y_sorted, aux_sorted
    y_sums = torch.zeros_like(uniq_x)
    aux_sums = torch.zeros_like(uniq_x)
    y_sums.scatter_add_(0, inverse, y_sorted)
    aux_sums.scatter_add_(0, inverse, aux_sorted)
    counts_f = counts.to(dtype=y_sums.dtype)
    return uniq_x, y_sums / counts_f, aux_sums / counts_f


def _interp_1d_sorted(x_src: torch.Tensor, y_src: torch.Tensor, x_query: torch.Tensor) -> torch.Tensor:
    if int(x_src.numel()) < 2:
        raise ValueError("Need at least two source points for interpolation")
    idx = torch.searchsorted(x_src, x_query, right=False)
    idx = idx.clamp(1, int(x_src.numel()) - 1)
    x0 = x_src[idx - 1]
    x1 = x_src[idx]
    y0 = y_src[idx - 1]
    y1 = y_src[idx]
    denom = (x1 - x0).clamp_min(1.0e-12)
    t = (x_query - x0) / denom
    return y0 + t * (y1 - y0)


def _interp_or_exact_sorted(
    x_src: torch.Tensor,
    y_src: torch.Tensor,
    x_query: torch.Tensor,
) -> torch.Tensor:
    if int(x_src.numel()) == int(x_query.numel()) and int(x_src.numel()) > 0:
        max_scale = max(
            float(torch.max(torch.abs(x_src)).detach().cpu().item()),
            float(torch.max(torch.abs(x_query)).detach().cpu().item()),
            1.0,
        )
        max_err = float(torch.max(torch.abs(x_src - x_query)).detach().cpu().item())
        if max_err <= 1.0e-10 * max_scale:
            return y_src
    return _interp_1d_sorted(x_src, y_src, x_query)


def _eval_univariate_ast_on_values(node, z: torch.Tensor) -> torch.Tensor:
    z1 = z.reshape(-1)
    if node is None:
        return torch.ones_like(z1)
    if isinstance(node, ConstNode):
        return torch.full_like(z1, ConstNode(node.value).value)
    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        kwargs = dict(getattr(node, "kwargs", {}) or {})
        if kind in ("var", "x", "input"):
            idxs = list(getattr(node, "var_idxs", ()) or ())
            if len(idxs) != 1 or int(idxs[0]) != 0:
                raise ValueError(f"Unsupported univariate Var indices: {idxs!r}")
            return z1
        if kind in ("const", "constant"):
            return torch.full_like(z1, float(kwargs.get("value", 1.0)))
        if kind in ("free_const", "freeconst", "free_constant", "scale"):
            return torch.full_like(z1, float(kwargs.get("init", 1.0)))
        if kind in ("fixed_const", "fixedconst", "fixed_constant"):
            return torch.full_like(z1, float(kwargs.get("value", 1.0)))
        raise ValueError(f"Unsupported univariate atom kind: {kind!r}")
    if isinstance(node, AddNode):
        return _eval_univariate_ast_on_values(node.left, z1) + _eval_univariate_ast_on_values(node.right, z1)
    if isinstance(node, MulNode):
        return _eval_univariate_ast_on_values(node.left, z1) * _eval_univariate_ast_on_values(node.right, z1)
    if isinstance(node, PowNode):
        base = _eval_univariate_ast_on_values(node.base, z1)
        exponent = node.exponent
        if isinstance(exponent, dict):
            raise ValueError("Dict exponents are not supported in univariate eval")
        if isinstance(exponent, (int, float)):
            return torch.pow(base, float(exponent))
        expv = _eval_univariate_ast_on_values(exponent, z1)
        if int(expv.numel()) == 1:
            return torch.pow(base, float(expv.reshape(-1)[0].detach().cpu().item()))
        return torch.pow(base, expv)
    if isinstance(node, LogNode):
        return torch.log(_eval_univariate_ast_on_values(node.arg, z1))
    if isinstance(node, ExpNode):
        return torch.exp(_eval_univariate_ast_on_values(node.arg, z1))
    if isinstance(node, SinNode):
        return torch.sin(_eval_univariate_ast_on_values(node.arg, z1))
    if isinstance(node, CosNode):
        return torch.cos(_eval_univariate_ast_on_values(node.arg, z1))
    if isinstance(node, AsinNode):
        return torch.asin(_eval_univariate_ast_on_values(node.arg, z1))
    if isinstance(node, AcosNode):
        return torch.acos(_eval_univariate_ast_on_values(node.arg, z1))
    if isinstance(node, AtanNode):
        return torch.atan(_eval_univariate_ast_on_values(node.arg, z1))
    if isinstance(node, AbsNode):
        return torch.abs(_eval_univariate_ast_on_values(node.arg, z1))
    raise TypeError(f"Unsupported univariate AST node type: {type(node).__name__}")


def _weighted_median_1d(values: torch.Tensor, weights: torch.Tensor) -> float:
    v = values.reshape(-1)
    w = torch.clamp(weights.reshape(-1), min=0.0)
    finite = torch.isfinite(v) & torch.isfinite(w) & (w > 0.0)
    if int(finite.sum()) <= 0:
        raise ValueError("weighted median received no finite positive-weight entries")
    v = v[finite]
    w = w[finite]
    order = torch.argsort(v)
    v = v[order]
    w = w[order]
    total = float(torch.sum(w).detach().cpu().item())
    if not math.isfinite(total) or total <= 0.0:
        return float(torch.median(v).detach().cpu().item())
    cdf = torch.cumsum(w, dim=0)
    idx = int(torch.searchsorted(cdf, torch.tensor(0.5 * total, dtype=cdf.dtype, device=cdf.device), right=False).item())
    idx = max(0, min(idx, int(v.numel()) - 1))
    return float(v[idx].detach().cpu().item())


def _pooled_same_coord_coeff_target(
    *,
    groups: Sequence[DEFeatureGroup],
    resid_parts: Sequence[torch.Tensor],
    carrier_ast,
    coord_ast,
    split: str,
    x_axis: int,
    rel_eps: float,
    min_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not groups:
        return None

    group_weights = _normalized_group_quality_weights(groups)
    curves: list[tuple[torch.Tensor, torch.Tensor, float]] = []

    for group, resid_part, group_weight in zip(groups, resid_parts, group_weights):
        phi = _eval_ast_on_features(
            carrier_ast,
            features=group.features,
            split=str(split),
            x_axis=int(x_axis),
        ).reshape(-1)
        coord = _eval_ast_on_features(
            coord_ast,
            features=group.features,
            split=str(split),
            x_axis=int(x_axis),
        ).reshape(-1, 1)
        _, mask = _safe_ratio_target(resid_part, phi, rel_eps=float(rel_eps))
        if mask is None or int(mask.sum()) <= 1:
            continue

        coord_valid = coord[mask].reshape(-1)
        phi_valid = phi[mask].reshape(-1)
        resid_valid = resid_part[mask].reshape(-1)
        finite = torch.isfinite(coord_valid) & torch.isfinite(phi_valid) & torch.isfinite(resid_valid)
        if int(finite.sum()) <= 1:
            continue

        coord_valid = coord_valid[finite]
        phi_valid = phi_valid[finite]
        resid_valid = resid_valid[finite]
        coeff_valid = (-resid_valid / phi_valid).reshape(-1, 1)
        coord_curve, coeff_curve = _prepare_curve_xy(coord_valid.reshape(-1, 1), coeff_valid)
        if int(coord_curve.numel()) < 2:
            continue
        curves.append((coord_curve, coeff_curve, float(group_weight)))

    if not curves:
        return None

    ref_curve_idx = max(range(len(curves)), key=lambda i: int(curves[i][0].numel()))
    coord_ref = curves[ref_curve_idx][0]
    interp_values = []
    interp_weights = []

    for coord_curve, coeff_curve, curve_weight in curves:
        y_buf = torch.full_like(coord_ref, float("nan"))
        w_buf = torch.zeros_like(coord_ref)
        mask_ref = (coord_ref >= coord_curve[0]) & (coord_ref <= coord_curve[-1])
        if int(mask_ref.sum()) < 2:
            interp_values.append(y_buf)
            interp_weights.append(w_buf)
            continue
        coord_query = coord_ref[mask_ref]
        coeff_interp = _interp_or_exact_sorted(coord_curve, coeff_curve, coord_query)
        finite = torch.isfinite(coeff_interp)
        if int(finite.sum()) > 0:
            ref_idx = torch.nonzero(mask_ref, as_tuple=False).reshape(-1)[finite]
            y_buf[ref_idx] = coeff_interp[finite]
            w_buf[ref_idx] = float(max(curve_weight, 1.0e-12))
        interp_values.append(y_buf)
        interp_weights.append(w_buf)

    if not interp_values:
        return None

    value_mat = torch.stack(interp_values, dim=0)
    weight_mat = torch.stack(interp_weights, dim=0)
    valid_any = torch.isfinite(value_mat) & (weight_mat > 0.0)
    if not torch.any(valid_any):
        return None
    global_scale = max(
        float(torch.median(torch.abs(value_mat[valid_any])).detach().cpu().item()),
        1.0e-6,
    )
    dispersion_ref = max(0.05 * global_scale, 1.0e-6)

    coeff_out_vals: list[float] = []
    coord_out_vals: list[float] = []
    sample_weight_vals: list[float] = []

    min_coverage = 2 if len(curves) > 1 else 1
    for j in range(int(coord_ref.numel())):
        vals = value_mat[:, j]
        weights = weight_mat[:, j]
        valid = torch.isfinite(vals) & torch.isfinite(weights) & (weights > 0.0)
        if int(valid.sum()) < int(min_coverage):
            continue
        vals_v = vals[valid]
        weights_v = weights[valid]
        coeff_med = _weighted_median_1d(vals_v, weights_v)
        abs_dev = torch.abs(vals_v - coeff_med)
        disp = _weighted_median_1d(abs_dev, weights_v)
        coverage_weight = float(torch.sum(weights_v).detach().cpu().item())
        sample_weight = coverage_weight / (1.0 + (disp / dispersion_ref) ** 2)
        coord_out_vals.append(float(coord_ref[j].detach().cpu().item()))
        coeff_out_vals.append(float(coeff_med))
        sample_weight_vals.append(float(sample_weight))

    if len(coord_out_vals) < int(min_rows):
        return None

    coord_out = torch.as_tensor(coord_out_vals, dtype=coord_ref.dtype, device=coord_ref.device).reshape(-1, 1)
    coeff_out = torch.as_tensor(coeff_out_vals, dtype=coord_ref.dtype, device=coord_ref.device).reshape(-1, 1)
    sample_weight = torch.as_tensor(sample_weight_vals, dtype=coord_ref.dtype, device=coord_ref.device).reshape(-1, 1)
    mean_sample_weight = max(float(torch.mean(sample_weight).detach().cpu().item()), 1.0e-12)
    sample_weight = sample_weight / mean_sample_weight
    finite = (
        torch.isfinite(coord_out).all(dim=1)
        & torch.isfinite(coeff_out.reshape(-1))
        & torch.isfinite(sample_weight.reshape(-1))
        & (sample_weight.reshape(-1) > 0.0)
    )
    if int(finite.sum()) < int(min_rows):
        return None
    return coord_out[finite], coeff_out[finite], sample_weight[finite]


def _pairwise_curve_relative_mse(
    x_a: torch.Tensor,
    y_a: torch.Tensor,
    x_b: torch.Tensor,
    y_b: torch.Tensor,
) -> float | None:
    lo = max(float(x_a[0].detach().cpu().item()), float(x_b[0].detach().cpu().item()))
    hi = min(float(x_a[-1].detach().cpu().item()), float(x_b[-1].detach().cpu().item()))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return None

    mask_a = (x_a >= lo) & (x_a <= hi)
    mask_b = (x_b >= lo) & (x_b <= hi)
    x_a_overlap = x_a[mask_a]
    y_a_overlap = y_a[mask_a]
    x_b_overlap = x_b[mask_b]
    y_b_overlap = y_b[mask_b]
    if int(x_a_overlap.numel()) < 2 or int(x_b_overlap.numel()) < 2:
        return None

    y_b_on_a = _interp_1d_sorted(x_b_overlap, y_b_overlap, x_a_overlap)
    y_a_on_b = _interp_1d_sorted(x_a_overlap, y_a_overlap, x_b_overlap)
    scale_a = max(
        float(torch.mean(y_a_overlap.square()).detach().cpu().item()),
        float(torch.mean(y_b_on_a.square()).detach().cpu().item()),
        1.0e-12,
    )
    scale_b = max(
        float(torch.mean(y_b_overlap.square()).detach().cpu().item()),
        float(torch.mean(y_a_on_b.square()).detach().cpu().item()),
        1.0e-12,
    )
    mse_a = float(torch.mean((y_a_overlap - y_b_on_a).square()).detach().cpu().item()) / scale_a
    mse_b = float(torch.mean((y_b_overlap - y_a_on_b).square()).detach().cpu().item()) / scale_b
    return 0.5 * (mse_a + mse_b)


def _merge_weighted_curves_on_reference(
    curves: Sequence[tuple[torch.Tensor, torch.Tensor, float]],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not curves:
        return None
    ref_idx = max(range(len(curves)), key=lambda i: int(curves[i][0].numel()))
    x_ref = curves[ref_idx][0]
    y_sum = torch.zeros_like(x_ref)
    w_sum = torch.zeros_like(x_ref)
    for x_curve, y_curve, curve_weight in curves:
        mask_ref = (x_ref >= x_curve[0]) & (x_ref <= x_curve[-1])
        if int(mask_ref.sum()) < 2:
            continue
        x_query = x_ref[mask_ref]
        y_interp = _interp_or_exact_sorted(x_curve, y_curve, x_query)
        finite = torch.isfinite(y_interp)
        if int(finite.sum()) <= 0:
            continue
        ref_idx_valid = torch.nonzero(mask_ref, as_tuple=False).reshape(-1)[finite]
        w = float(max(curve_weight, 1.0e-12))
        y_sum[ref_idx_valid] += w * y_interp[finite]
        w_sum[ref_idx_valid] += w
    keep = torch.isfinite(y_sum) & torch.isfinite(w_sum) & (w_sum > 0.0)
    if int(keep.sum()) < 2:
        return None
    return x_ref[keep], (y_sum[keep] / w_sum[keep])


def _curve_shape_stats(x_curve: torch.Tensor, y_curve: torch.Tensor) -> dict[str, float]:
    if int(x_curve.numel()) < 3 or int(y_curve.numel()) < 3:
        return {
            "shape_score": 0.0,
            "sign_changes": 0.0,
            "curvature_ratio": 0.0,
            "tv_ratio": 1.0,
        }
    dx = torch.diff(x_curve).clamp_min(1.0e-12)
    dy = torch.diff(y_curve)
    slope = dy / dx
    slope_scale = max(float(torch.max(torch.abs(slope)).detach().cpu().item()), 1.0e-12)
    slope_tol = 1.0e-3 * slope_scale
    slope_sign = torch.sign(slope)
    slope_sign = torch.where(torch.abs(slope) >= slope_tol, slope_sign, torch.zeros_like(slope_sign))
    nz = slope_sign[slope_sign != 0.0]
    sign_changes = 0
    if int(nz.numel()) >= 2:
        sign_changes = int(torch.sum(nz[1:] * nz[:-1] < 0.0).detach().cpu().item())

    if int(slope.numel()) >= 2:
        ds = torch.diff(slope)
        curvature_ratio = float(
            torch.mean(ds.square()).detach().cpu().item()
            / max(float(torch.mean(slope.square()).detach().cpu().item()), 1.0e-12)
        )
    else:
        curvature_ratio = 0.0

    amp = max(
        float((torch.max(y_curve) - torch.min(y_curve)).detach().cpu().item()),
        1.0e-12,
    )
    tv_ratio = float(torch.sum(torch.abs(dy)).detach().cpu().item()) / amp
    shape_score = float(sign_changes) + 0.10 * max(0.0, tv_ratio - 1.0) + 0.05 * max(0.0, curvature_ratio)
    return {
        "shape_score": float(shape_score),
        "sign_changes": float(sign_changes),
        "curvature_ratio": float(curvature_ratio),
        "tv_ratio": float(tv_ratio),
    }


def _family_priority(row: dict[str, Any]) -> int:
    lane = str(row.get("lane", "") or "")
    family = str(row.get("family", "") or "")
    if lane == "x_coeff_on_u":
        priority = {
            "log": 0,
            "reciprocal": 1,
            "inv_square": 2,
            "poly2": 3,
            "explorer": 8,
        }
        return int(priority.get(family, 6))
    if lane == "state_nonlinearity":
        priority = {
            "poly2": 0,
            "poly3": 1,
            "reciprocal": 2,
            "log": 3,
            "exp": 4,
            "sin": 5,
            "cos": 5,
            "explorer": 8,
        }
        return int(priority.get(family, 6))
    return 10


def _row_domain_safe(row: Mapping[str, Any] | None) -> bool:
    if not isinstance(row, Mapping):
        return False
    if row.get("structural_ok", None) is False:
        return False
    if row.get("structural_hard_reject", None) is True:
        return False
    if row.get("domain_ok", None) is False:
        return False
    for key in ("domain_projection", "domain_projection_eval"):
        diag = row.get(key, None)
        if isinstance(diag, Mapping) and not domain_projection_is_acceptable(diag):
            return False
    mapping = row.get("mapping", None)
    if isinstance(mapping, Mapping):
        diag = mapping.get("_domain_projection", None)
        if isinstance(diag, Mapping) and not domain_projection_is_acceptable(diag):
            return False
    return True


def _x_lane_shape_stats(
    *,
    groups: Sequence[DEFeatureGroup],
    resid_probe_parts: Sequence[torch.Tensor],
    carrier_ast,
    coord_ast,
    coeff_ast,
    x_axis: int,
    rel_eps: float,
) -> dict[str, float]:
    curves: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    group_weights = _normalized_group_quality_weights(groups)
    for group, resid_probe, group_weight in zip(groups, resid_probe_parts, group_weights):
        phi_probe = _eval_ast_on_features(
            carrier_ast,
            features=group.features,
            split="probe",
            x_axis=int(x_axis),
        ).reshape(-1)
        coord_probe = _eval_ast_on_features(
            coord_ast,
            features=group.features,
            split="probe",
            x_axis=int(x_axis),
        ).reshape(-1, 1)
        coeff_probe = _eval_ast_on_features(
            coeff_ast,
            features=group.features,
            split="probe",
            x_axis=int(x_axis),
        ).reshape(-1, 1)
        _, mask_probe = _safe_ratio_target(resid_probe, phi_probe, rel_eps=float(rel_eps))
        if mask_probe is None or int(mask_probe.sum()) < 3:
            continue
        coord_valid, coeff_valid = _finite_xy_rows(coord_probe[mask_probe], coeff_probe[mask_probe])
        if int(coord_valid.shape[0]) < 3:
            continue
        x_curve, y_curve = _prepare_curve_xy(coord_valid, coeff_valid)
        if int(x_curve.numel()) < 3:
            continue
        curves.append((x_curve, y_curve, float(group_weight)))

    merged = _merge_weighted_curves_on_reference(curves)
    if merged is None:
        return {
            "shape_score": 0.0,
            "sign_changes": 0.0,
            "curvature_ratio": 0.0,
            "tv_ratio": 1.0,
        }
    return _curve_shape_stats(*merged)


def _lane_consistency_stats(
    *,
    groups: Sequence[DEFeatureGroup],
    resid_probe_parts: Sequence[torch.Tensor],
    carrier_ast,
    coord_ast,
    x_axis: int,
    rel_eps: float,
) -> tuple[float, int, int]:
    if int(len(groups)) <= 1:
        return 0.0, 0, 0

    curves: list[tuple[torch.Tensor, torch.Tensor]] = []
    for group, resid_probe in zip(groups, resid_probe_parts):
        phi_probe = _eval_ast_on_features(
            carrier_ast,
            features=group.features,
            split="probe",
            x_axis=int(x_axis),
        ).reshape(-1)
        z_probe = _eval_ast_on_features(
            coord_ast,
            features=group.features,
            split="probe",
            x_axis=int(x_axis),
        ).reshape(-1, 1)
        ratio_probe, mask_probe = _safe_ratio_target(resid_probe, phi_probe, rel_eps=float(rel_eps))
        if ratio_probe is None or mask_probe is None:
            continue
        z_probe_valid, ratio_probe_valid = _finite_xy_rows(z_probe[mask_probe], ratio_probe)
        if int(z_probe_valid.shape[0]) < 2:
            continue
        x_curve, y_curve = _prepare_curve_xy(z_probe_valid, ratio_probe_valid)
        if int(x_curve.numel()) < 2:
            continue
        curves.append((x_curve, y_curve))

    total_pairs = int(len(curves) * (len(curves) - 1) // 2)
    if total_pairs <= 0:
        return float("inf"), 0, total_pairs

    pair_scores: list[float] = []
    for i in range(len(curves)):
        x_i, y_i = curves[i]
        for j in range(i + 1, len(curves)):
            x_j, y_j = curves[j]
            score = _pairwise_curve_relative_mse(x_i, y_i, x_j, y_j)
            if score is not None and math.isfinite(score):
                pair_scores.append(float(score))

    if not pair_scores:
        return float("inf"), 0, total_pairs
    return float(sum(pair_scores) / len(pair_scores)), int(len(pair_scores)), int(total_pairs)


def _pairwise_same_x_witness(
    x_a: torch.Tensor,
    u_a: torch.Tensor,
    du_a: torch.Tensor,
    x_b: torch.Tensor,
    u_b: torch.Tensor,
    du_b: torch.Tensor,
) -> float | None:
    lo = max(float(x_a[0].detach().cpu().item()), float(x_b[0].detach().cpu().item()))
    hi = min(float(x_a[-1].detach().cpu().item()), float(x_b[-1].detach().cpu().item()))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return None
    mask_a = (x_a >= lo) & (x_a <= hi)
    mask_b = (x_b >= lo) & (x_b <= hi)
    x_a_overlap = x_a[mask_a]
    u_a_overlap = u_a[mask_a]
    du_a_overlap = du_a[mask_a]
    x_b_overlap = x_b[mask_b]
    u_b_overlap = u_b[mask_b]
    du_b_overlap = du_b[mask_b]
    if int(x_a_overlap.numel()) < 2 or int(x_b_overlap.numel()) < 2:
        return None
    u_b_on_a = _interp_1d_sorted(x_b_overlap, u_b_overlap, x_a_overlap)
    du_b_on_a = _interp_1d_sorted(x_b_overlap, du_b_overlap, x_a_overlap)
    witness = du_a_overlap * u_b_on_a - du_b_on_a * u_a_overlap
    scale = max(
        float(torch.mean((du_a_overlap * u_b_on_a).square()).detach().cpu().item()),
        float(torch.mean((du_b_on_a * u_a_overlap).square()).detach().cpu().item()),
        1.0e-12,
    )
    return float(torch.mean(witness.square()).detach().cpu().item()) / scale


def _pairwise_same_u_witness(
    u_a: torch.Tensor,
    du_a: torch.Tensor,
    x_a: torch.Tensor,
    u_b: torch.Tensor,
    du_b: torch.Tensor,
    x_b: torch.Tensor,
    *,
    min_x_sep_rel: float = 0.05,
) -> float | None:
    lo = max(float(u_a[0].detach().cpu().item()), float(u_b[0].detach().cpu().item()))
    hi = min(float(u_a[-1].detach().cpu().item()), float(u_b[-1].detach().cpu().item()))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return None
    mask_a = (u_a >= lo) & (u_a <= hi)
    mask_b = (u_b >= lo) & (u_b <= hi)
    u_a_overlap = u_a[mask_a]
    du_a_overlap = du_a[mask_a]
    x_a_overlap = x_a[mask_a]
    u_b_overlap = u_b[mask_b]
    du_b_overlap = du_b[mask_b]
    x_b_overlap = x_b[mask_b]
    if int(u_a_overlap.numel()) < 2 or int(u_b_overlap.numel()) < 2:
        return None
    du_b_on_a = _interp_1d_sorted(u_b_overlap, du_b_overlap, u_a_overlap)
    x_b_on_a = _interp_1d_sorted(u_b_overlap, x_b_overlap, u_a_overlap)
    x_span = max(
        float((x_a.max() - x_a.min()).detach().cpu().item()),
        float((x_b.max() - x_b.min()).detach().cpu().item()),
        1.0e-12,
    )
    x_sep_rel = float(torch.mean(torch.abs(x_a_overlap - x_b_on_a)).detach().cpu().item()) / x_span
    if not math.isfinite(x_sep_rel) or x_sep_rel < float(min_x_sep_rel):
        return None
    witness = du_a_overlap - du_b_on_a
    scale = max(
        float(torch.mean(du_a_overlap.square()).detach().cpu().item()),
        float(torch.mean(du_b_on_a.square()).detach().cpu().item()),
        1.0e-12,
    )
    return float(torch.mean(witness.square()).detach().cpu().item()) / scale


def _lane_witness_stats(
    *,
    lane: str,
    groups: Sequence[DEFeatureGroup],
    resid_probe_parts: Sequence[torch.Tensor],
    carrier_ast,
    coord_ast,
    x_axis: int,
    rel_eps: float,
) -> tuple[str, float, int, int]:
    lane_norm = str(lane or "")
    if int(len(groups)) <= 1:
        return "single_dataset", 0.0, 0, 0

    if lane_norm == "x_coeff_on_u":
        curves: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for group in groups:
            x_probe = _split_tensor(group.features, "probe", "x")[:, int(x_axis) : int(x_axis) + 1]
            u_probe = _split_tensor(group.features, "probe", "u")
            du_probe = _split_tensor(group.features, "probe", "du")
            x_curve, u_curve, du_curve = _prepare_curve_xy_aux(x_probe, u_probe, du_probe)
            if int(x_curve.numel()) < 2:
                continue
            curves.append((x_curve, u_curve, du_curve))
        total_pairs = int(len(curves) * (len(curves) - 1) // 2)
        if total_pairs <= 0:
            return "same_x_witness", float("inf"), 0, total_pairs
        pair_scores: list[float] = []
        for i in range(len(curves)):
            x_i, u_i, du_i = curves[i]
            for j in range(i + 1, len(curves)):
                x_j, u_j, du_j = curves[j]
                score = _pairwise_same_x_witness(x_i, u_i, du_i, x_j, u_j, du_j)
                if score is not None and math.isfinite(score):
                    pair_scores.append(float(score))
        if not pair_scores:
            return "same_x_witness", float("inf"), 0, total_pairs
        return "same_x_witness", float(sum(pair_scores) / len(pair_scores)), int(len(pair_scores)), int(total_pairs)

    if lane_norm == "state_nonlinearity":
        curves: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for group in groups:
            u_probe = _split_tensor(group.features, "probe", "u")
            du_probe = _split_tensor(group.features, "probe", "du")
            x_probe = _split_tensor(group.features, "probe", "x")[:, int(x_axis) : int(x_axis) + 1]
            u_curve, du_curve, x_curve = _prepare_curve_xy_aux(u_probe, du_probe, x_probe)
            if int(u_curve.numel()) < 2:
                continue
            curves.append((u_curve, du_curve, x_curve))
        total_pairs = int(len(curves) * (len(curves) - 1) // 2)
        if total_pairs <= 0:
            return "matched_u_witness", float("inf"), 0, total_pairs
        pair_scores: list[float] = []
        for i in range(len(curves)):
            u_i, du_i, x_i = curves[i]
            for j in range(i + 1, len(curves)):
                u_j, du_j, x_j = curves[j]
                score = _pairwise_same_u_witness(u_i, du_i, x_i, u_j, du_j, x_j)
                if score is not None and math.isfinite(score):
                    pair_scores.append(float(score))
        if not pair_scores:
            return "matched_u_witness", float("inf"), 0, total_pairs
        return "matched_u_witness", float(sum(pair_scores) / len(pair_scores)), int(len(pair_scores)), int(total_pairs)

    score, pairs, total_pairs = _lane_consistency_stats(
        groups=groups,
        resid_probe_parts=resid_probe_parts,
        carrier_ast=carrier_ast,
        coord_ast=coord_ast,
        x_axis=int(x_axis),
        rel_eps=float(rel_eps),
    )
    return "generic_ratio_curve", float(score), int(pairs), int(total_pairs)


def _probe_mse_from_residuals(residual_parts: Sequence[torch.Tensor]) -> float:
    if not residual_parts:
        return float("inf")
    resid_probe_cat = torch.cat([r.reshape(-1) for r in residual_parts], dim=0)
    if not torch.isfinite(resid_probe_cat).all():
        return float("inf")
    return float(torch.mean(resid_probe_cat.square()).detach().cpu().item())


_TRIM_FRAC = 0.01


_TRIM_K_SCALE = 8.0


_TRIM_MIN_KEEP = 32


def _outlier_trim_mask(residuals: torch.Tensor) -> torch.Tensor | None:
    """Mask of rows to KEEP, or None when nothing qualifies for trimming."""
    r = residuals.reshape(-1)
    n = int(r.numel())
    if n < _TRIM_MIN_KEEP:
        return None
    if not torch.isfinite(r).all():
        return None
    centered = (r - r.median()).abs()
    scale = 1.4826 * float(centered.median().detach().cpu().item())
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    try:
        tail = float(torch.quantile(centered, 1.0 - _TRIM_FRAC).detach().cpu().item())
    except Exception:
        return None
    threshold = max(tail, _TRIM_K_SCALE * scale)
    # Deviations at float-noise level (exact fits) need no robustness; trimming
    # there only perturbs tie-breaks between equally-exact candidates.
    if float(centered.max().detach().cpu().item()) < 1.0e-12:
        return None
    keep = centered <= threshold
    n_keep = int(keep.sum().detach().cpu().item())
    if n_keep >= n or n_keep < max(_TRIM_MIN_KEEP, int(n * (1.0 - 2.0 * _TRIM_FRAC))):
        return None
    return keep


_TRIM_LEVERAGE_CAP = 32.0


def _leverage_keep_mask(Phi: torch.Tensor, *, ridge: float) -> torch.Tensor | None:
    """Mask of rows to KEEP after dropping extreme-leverage rows, or None.

    Leverage outliers (a singular coordinate like du/x exploding on a few
    boundary-layer rows) pull the least-squares fit through themselves, so
    their residuals stay small and residual-based trimming cannot see them.
    Hat values depend on the design alone and expose them directly.
    """
    n, k = int(Phi.shape[0]), int(Phi.shape[1])
    if n < _TRIM_MIN_KEEP or k < 1:
        return None
    try:
        A = Phi.T @ Phi + max(float(ridge), 1.0e-12) * torch.eye(k, dtype=Phi.dtype, device=Phi.device)
        G = torch.linalg.solve(A, Phi.T)
        hat = (Phi * G.T).sum(dim=1)
    except Exception:
        return None
    if not bool(torch.isfinite(hat).all().detach().cpu().item()):
        return None
    cap = _TRIM_LEVERAGE_CAP * float(k) / float(n)
    flagged = hat > cap
    n_flagged = int(flagged.sum().detach().cpu().item())
    if n_flagged <= 0:
        return None
    max_trim = max(1, int(n * _TRIM_FRAC))
    if n_flagged > max_trim:
        threshold = torch.topk(hat, max_trim).values.min()
        flagged = hat >= threshold
    keep = ~flagged
    if int(keep.sum().detach().cpu().item()) < _TRIM_MIN_KEEP:
        return None
    return keep


def _ridge_lstsq_or_least_norm(Phi: torch.Tensor, y: torch.Tensor, *, ridge: float) -> torch.Tensor:
    """ridge_lstsq with a least-norm fallback for rank-deficient designs."""
    try:
        return ridge_lstsq(Phi, y, ridge=float(ridge))
    except Exception:
        return torch.linalg.lstsq(Phi, y.reshape(-1, 1)).solution


_RELATIVE_FIT_DYNAMIC_RANGE = 50.0


def _target_scale_row_weights(y: torch.Tensor) -> torch.Tensor | None:
    """Per-row weights for high-dynamic-range targets (relative least squares).

    Targets like u' = -u/x span several decades; absolute least squares then
    puts all statistical weight in the steep band where derivative features
    are least accurate, and the clean tail is ignored (de206). Returns None
    when the dynamic range is modest so well-scaled problems keep absolute
    fitting.
    """
    a = y.detach().reshape(-1).abs()
    if int(a.numel()) < _TRIM_MIN_KEEP or not torch.isfinite(a).all():
        return None
    try:
        q25 = float(torch.quantile(a, 0.25).item())
        q50 = float(torch.quantile(a, 0.50).item())
        q95 = float(torch.quantile(a, 0.95).item())
    except Exception:
        return None
    if not (q50 > 0.0) or q95 <= _RELATIVE_FIT_DYNAMIC_RANGE * q50:
        return None
    floor = max(q25, 1.0e-300)
    return 1.0 / a.clamp_min(floor)


def _scale_weighted_trimmed_lstsq(Phi: torch.Tensor, y: torch.Tensor, *, ridge: float = 0.0) -> torch.Tensor:
    """_trimmed_ridge_lstsq in relative units when the target spans decades."""
    w = _target_scale_row_weights(y)
    if w is None:
        return _trimmed_ridge_lstsq(Phi, y, ridge=float(ridge))
    y_w = (y.reshape(-1) * w).reshape(-1, 1)
    return _trimmed_ridge_lstsq(Phi * w.reshape(-1, 1), y_w, ridge=float(ridge))


def _trimmed_ridge_lstsq(Phi: torch.Tensor, y: torch.Tensor, *, ridge: float = 0.0) -> torch.Tensor:
    """ridge_lstsq with a leverage pre-trim and one residual-trimmed refit pass."""
    coeffs = _ridge_lstsq_or_least_norm(Phi, y, ridge=float(ridge))
    try:
        Phi_kept = Phi
        y_kept = y.reshape(-1)
        lev_keep = _leverage_keep_mask(Phi, ridge=float(ridge))
        if lev_keep is not None:
            Phi_kept = Phi[lev_keep]
            y_kept = y_kept[lev_keep]
            coeffs_lev = _ridge_lstsq_or_least_norm(Phi_kept, y_kept.reshape(-1, 1), ridge=float(ridge))
            if bool(torch.isfinite(coeffs_lev).all().detach().cpu().item()):
                coeffs = coeffs_lev
        resid = (y_kept.reshape(-1, 1) - Phi_kept @ coeffs.reshape(-1, 1)).reshape(-1)
        keep = _outlier_trim_mask(resid)
        if keep is None:
            return coeffs
        coeffs_trim = _ridge_lstsq_or_least_norm(Phi_kept[keep], y_kept[keep].reshape(-1, 1), ridge=float(ridge))
        if not bool(torch.isfinite(coeffs_trim).all().detach().cpu().item()):
            return coeffs
        return coeffs_trim
    except Exception:
        return coeffs


def _trimmed_mean_sq(residuals: torch.Tensor) -> float:
    """Mean squared residual with the outlier-trim rule applied."""
    r = residuals.reshape(-1)
    if not torch.isfinite(r).all():
        return float("inf")
    keep = _outlier_trim_mask(r)
    if keep is not None:
        r = r[keep]
    if int(r.numel()) <= 0:
        return float("inf")
    return float(torch.mean(r.square()).detach().cpu().item())


def _trimmed_probe_mse_from_residuals(residual_parts: Sequence[torch.Tensor]) -> float:
    if not residual_parts:
        return float("inf")
    resid_probe_cat = torch.cat([r.reshape(-1) for r in residual_parts], dim=0)
    if not torch.isfinite(resid_probe_cat).all():
        return float("inf")
    return _trimmed_mean_sq(resid_probe_cat)


def _quality_weighted_probe_mse_from_residuals(
    groups: Sequence[DEFeatureGroup],
    residual_parts: Sequence[torch.Tensor],
    *,
    robust: bool = False,
) -> float:
    if not groups or not residual_parts:
        return float("inf")
    group_weights = _normalized_group_quality_weights(
        groups,
        min_weight=0.05 if bool(robust) else 0.25,
        max_weight=4.0,
        power=1.0 if bool(robust) else 0.5,
    )
    group_scores: list[float] = []
    group_score_weights: list[float] = []
    weighted_sq_sum = 0.0
    weighted_count = 0.0
    for resid, group_weight in zip(residual_parts, group_weights):
        rr = resid.reshape(-1)
        if int(rr.numel()) <= 0 or not torch.isfinite(rr).all():
            continue
        if bool(robust):
            group_scores.append(float(torch.mean(rr.square()).detach().cpu().item()))
            group_score_weights.append(float(group_weight))
        else:
            weighted_sq_sum += float(group_weight) * float(torch.sum(rr.square()).detach().cpu().item())
            weighted_count += float(group_weight) * float(rr.numel())
    if bool(robust):
        if not group_scores:
            return float("inf")
        denom = max(sum(group_score_weights), 1.0e-12)
        return float(sum(w * s for w, s in zip(group_score_weights, group_scores)) / denom)
    if weighted_count <= 0.0:
        return float("inf")
    return float(weighted_sq_sum / weighted_count)


def _compose_nonanchor_ast(base_ast, block_asts: Sequence[Any]):
    out = base_ast
    for block_ast in list(block_asts):
        out = block_ast if out is None else Add(out, block_ast)
    return out


def _coeff_ast_from_row(row: dict[str, Any], coord_ast):
    inner = row.get("nestynet_ast", None)
    if inner is None:
        inner = factorized_search_to_nestynet(row["toy_ast"])
    if inner is None:
        return None
    mapping = dict(row.get("mapping", {}) or {})
    coeff_ast = None
    if mapping:
        coeff_ast = embed_mapping_in_ast(inner, mapping, [coord_ast], units_mode="raw")
    if coeff_ast is None:
        coeff_ast = remap_var_to_exprs(inner, [coord_ast])
    return coeff_ast


def _masked_original_scale_probe_mse(
    *,
    groups: Sequence[DEFeatureGroup],
    resid_probe_parts: Sequence[torch.Tensor],
    carrier_ast,
    coeff_ast,
    x_axis: int,
    rel_eps: float,
    robust: bool = False,
) -> float:
    group_weights = _normalized_group_quality_weights(
        groups,
        min_weight=0.05 if bool(robust) else 0.25,
        max_weight=4.0,
        power=1.0 if bool(robust) else 0.5,
    )
    group_scores: list[float] = []
    group_score_weights: list[float] = []
    weighted_sq_sum = 0.0
    weighted_count = 0.0
    for group, resid_probe, group_weight in zip(groups, resid_probe_parts, group_weights):
        phi_probe = _eval_ast_on_features(
            carrier_ast,
            features=group.features,
            split="probe",
            x_axis=int(x_axis),
        ).reshape(-1)
        _, mask_probe = _safe_ratio_target(resid_probe, phi_probe, rel_eps=float(rel_eps))
        if mask_probe is None or int(mask_probe.sum()) <= 0:
            continue
        coeff_probe = _eval_ast_on_features(
            coeff_ast,
            features=group.features,
            split="probe",
            x_axis=int(x_axis),
        ).reshape(-1)
        rr = resid_probe[mask_probe] + coeff_probe[mask_probe] * phi_probe[mask_probe]
        rr = rr[torch.isfinite(rr)]
        if int(rr.numel()) > 0:
            if bool(robust):
                sq = torch.sort(rr.square().reshape(-1))[0]
                trim_n = int(math.floor(0.15 * int(sq.numel())))
                keep = sq[: max(int(sq.numel()) - trim_n, 1)]
                group_scores.append(float(torch.mean(keep).detach().cpu().item()))
                group_score_weights.append(float(group_weight))
            else:
                weighted_sq_sum += float(group_weight) * float(torch.sum(rr.square()).detach().cpu().item())
                weighted_count += float(group_weight) * float(rr.numel())
    if bool(robust):
        if not group_scores:
            return float("inf")
        denom = max(sum(group_score_weights), 1.0e-12)
        return float(sum(w * s for w, s in zip(group_score_weights, group_scores)) / denom)
    if weighted_count <= 0.0:
        return float("inf")
    return float(weighted_sq_sum / weighted_count)


def _fit_original_scale_affine_explorer_head(
    *,
    row: dict[str, Any],
    groups: Sequence[DEFeatureGroup],
    order: int,
    x_axis: int,
    resid_fit_parts: Sequence[torch.Tensor],
    resid_probe_parts: Sequence[torch.Tensor],
    carrier_ast,
    coord_ast,
    rel_eps: float,
    min_ratio_rows: int,
) -> tuple[Any | None, dict[str, Any], float, Any | None]:
    inner_local = row.get("nestynet_ast", None)
    if inner_local is None:
        inner_local = factorized_search_to_nestynet(row["toy_ast"])
    if inner_local is None:
        return None, {}, float("inf"), None

    inner_ast = remap_var_to_exprs(inner_local, [coord_ast])
    inner_ast_local = remap_var_to_exprs(inner_local, [Var(0)])
    Phi_fit_parts: list[torch.Tensor] = []
    y_fit_parts: list[torch.Tensor] = []
    Phi_probe_parts: list[torch.Tensor] = []
    y_probe_parts: list[torch.Tensor] = []

    group_weights = _normalized_group_quality_weights(groups)
    for group, resid_fit, resid_probe, group_weight in zip(groups, resid_fit_parts, resid_probe_parts, group_weights):
        phi_fit = _eval_ast_on_features(
            carrier_ast,
            features=group.features,
            split="fit",
            x_axis=int(x_axis),
        ).reshape(-1)
        phi_probe = _eval_ast_on_features(
            carrier_ast,
            features=group.features,
            split="probe",
            x_axis=int(x_axis),
        ).reshape(-1)
        _, mask_fit = _safe_ratio_target(resid_fit, phi_fit, rel_eps=float(rel_eps))
        _, mask_probe = _safe_ratio_target(resid_probe, phi_probe, rel_eps=float(rel_eps))
        if mask_fit is None or mask_probe is None:
            continue

        pred_fit = _eval_ast_on_features(
            inner_ast,
            features=group.features,
            split="fit",
            x_axis=int(x_axis),
        ).reshape(-1)
        pred_probe = _eval_ast_on_features(
            inner_ast,
            features=group.features,
            split="probe",
            x_axis=int(x_axis),
        ).reshape(-1)

        Phi_fit = torch.stack([phi_fit, phi_fit * pred_fit], dim=1)[mask_fit]
        Phi_probe = torch.stack([phi_probe, phi_probe * pred_probe], dim=1)[mask_probe]
        y_fit = (-resid_fit[mask_fit]).reshape(-1, 1)
        y_probe = (-resid_probe[mask_probe]).reshape(-1, 1)

        Phi_fit_valid, y_fit_valid = _finite_design_rows(Phi_fit, y_fit)
        Phi_probe_valid, y_probe_valid = _finite_design_rows(Phi_probe, y_probe)
        if int(Phi_fit_valid.shape[0]) < int(min_ratio_rows) or int(Phi_probe_valid.shape[0]) < int(min_ratio_rows):
            continue

        scale = math.sqrt(max(float(group_weight), 1.0e-12))
        Phi_fit_parts.append(Phi_fit_valid * scale)
        y_fit_parts.append(y_fit_valid * scale)
        Phi_probe_parts.append(Phi_probe_valid * scale)
        y_probe_parts.append(y_probe_valid * scale)

    if not Phi_fit_parts or not Phi_probe_parts:
        return None, {}, float("inf"), None

    coeffs = (
        torch.linalg.lstsq(torch.cat(Phi_fit_parts, dim=0), torch.cat(y_fit_parts, dim=0)).solution.detach().cpu().reshape(-1)
    )
    coeff_ast = _sum_linear_terms_ast([None, inner_ast], coeffs.tolist())
    coeff_ast_local = _sum_linear_terms_ast([None, inner_ast_local], coeffs.tolist())
    if coeff_ast is None:
        return None, {}, float("inf"), None

    probe_pred = (
        torch.cat(Phi_probe_parts, dim=0)
        @ coeffs.to(dtype=Phi_probe_parts[0].dtype, device=Phi_probe_parts[0].device)
    ).reshape(-1, 1)
    probe_y = torch.cat(y_probe_parts, dim=0)
    probe_mse = float(torch.mean((probe_pred - probe_y) ** 2).detach().cpu().item())
    mapping = {
        "kind": "poly",
        "coeffs": [float(c) for c in coeffs.tolist()],
        "mu": 0.0,
        "std": 1.0,
        "_factorized_refit": "original_scale_affine",
    }
    return coeff_ast, mapping, float(probe_mse), coeff_ast_local


def _fit_original_scale_family_basis(
    *,
    basis_asts: Sequence[Any | None],
    basis_asts_local: Sequence[Any | None],
    groups: Sequence[DEFeatureGroup],
    x_axis: int,
    resid_fit_parts: Sequence[torch.Tensor],
    resid_probe_parts: Sequence[torch.Tensor],
    carrier_ast,
    rel_eps: float,
    min_ratio_rows: int,
) -> tuple[Any | None, Any | None, float, list[float]]:
    Phi_fit_parts: list[torch.Tensor] = []
    y_fit_parts: list[torch.Tensor] = []
    Phi_probe_parts: list[torch.Tensor] = []
    y_probe_parts: list[torch.Tensor] = []

    group_weights = _normalized_group_quality_weights(groups)
    for group, resid_fit, resid_probe, group_weight in zip(groups, resid_fit_parts, resid_probe_parts, group_weights):
        phi_fit = _eval_ast_on_features(
            carrier_ast,
            features=group.features,
            split="fit",
            x_axis=int(x_axis),
        ).reshape(-1)
        phi_probe = _eval_ast_on_features(
            carrier_ast,
            features=group.features,
            split="probe",
            x_axis=int(x_axis),
        ).reshape(-1)
        _, mask_fit = _safe_ratio_target(resid_fit, phi_fit, rel_eps=float(rel_eps))
        _, mask_probe = _safe_ratio_target(resid_probe, phi_probe, rel_eps=float(rel_eps))
        if mask_fit is None or mask_probe is None:
            continue

        cols_fit: list[torch.Tensor] = []
        cols_probe: list[torch.Tensor] = []
        for basis_ast in list(basis_asts):
            if basis_ast is None:
                cols_fit.append(phi_fit)
                cols_probe.append(phi_probe)
            else:
                basis_fit = _eval_ast_on_features(
                    basis_ast,
                    features=group.features,
                    split="fit",
                    x_axis=int(x_axis),
                ).reshape(-1)
                basis_probe = _eval_ast_on_features(
                    basis_ast,
                    features=group.features,
                    split="probe",
                    x_axis=int(x_axis),
                ).reshape(-1)
                cols_fit.append(phi_fit * basis_fit)
                cols_probe.append(phi_probe * basis_probe)

        Phi_fit = torch.stack(cols_fit, dim=1)[mask_fit]
        Phi_probe = torch.stack(cols_probe, dim=1)[mask_probe]
        y_fit = (-resid_fit[mask_fit]).reshape(-1, 1)
        y_probe = (-resid_probe[mask_probe]).reshape(-1, 1)

        Phi_fit_valid, y_fit_valid = _finite_design_rows(Phi_fit, y_fit)
        Phi_probe_valid, y_probe_valid = _finite_design_rows(Phi_probe, y_probe)
        if int(Phi_fit_valid.shape[0]) < int(min_ratio_rows) or int(Phi_probe_valid.shape[0]) < int(min_ratio_rows):
            continue

        scale = math.sqrt(max(float(group_weight), 1.0e-12))
        Phi_fit_parts.append(Phi_fit_valid * scale)
        y_fit_parts.append(y_fit_valid * scale)
        Phi_probe_parts.append(Phi_probe_valid * scale)
        y_probe_parts.append(y_probe_valid * scale)

    if not Phi_fit_parts or not Phi_probe_parts:
        return None, None, float("inf"), []

    coeffs = (
        _scale_weighted_trimmed_lstsq(torch.cat(Phi_fit_parts, dim=0), torch.cat(y_fit_parts, dim=0), ridge=0.0)
        .detach()
        .cpu()
        .reshape(-1)
    )
    coeff_ast = _sum_linear_terms_ast(list(basis_asts), coeffs.tolist())
    coeff_local_ast = _sum_linear_terms_ast(list(basis_asts_local), coeffs.tolist())
    if coeff_ast is None:
        return None, None, float("inf"), []

    probe_pred = (
        torch.cat(Phi_probe_parts, dim=0)
        @ coeffs.to(dtype=Phi_probe_parts[0].dtype, device=Phi_probe_parts[0].device)
    ).reshape(-1, 1)
    probe_y = torch.cat(y_probe_parts, dim=0)
    probe_mse = float(torch.mean((probe_pred - probe_y) ** 2).detach().cpu().item())
    return coeff_ast, coeff_local_ast, float(probe_mse), [float(c) for c in coeffs.tolist()]


def _base_variants(
    primary,
    groups: Sequence[DEFeatureGroup],
    *,
    order: int,
    x_axis: int,
    dtype: torch.dtype,
    base_modes: Sequence[str],
) -> list[dict[str, Any]]:
    zero_fit = [torch.zeros_like(g.features.u_fit.reshape(-1)) for g in groups]
    zero_probe = [torch.zeros_like(g.features.u_probe.reshape(-1)) for g in groups]
    variants: list[dict[str, Any]] = []
    seen: set[str] = set()

    for mode in list(base_modes):
        mode_norm = str(mode or "").strip().lower()
        if mode_norm in seen:
            continue
        seen.add(mode_norm)
        if mode_norm == "zero":
            variants.append(
                {
                    "mode": "zero",
                    "fit_parts": zero_fit,
                    "probe_parts": zero_probe,
                    "ast": None,
                }
            )
            continue
        if mode_norm == "primary":
            base_fit, base_probe, base_ast, _ = _shared_base_from_primary(
                primary,
                groups,
                order=int(order),
                x_axis=int(x_axis),
                dtype=dtype,
            )
            has_signal = base_ast is not None or any(
                bool(torch.isfinite(bp).any()) and float(torch.max(torch.abs(bp)).detach().cpu().item()) > 1.0e-12
                for bp in list(base_probe)
            )
            if has_signal:
                variants.append(
                    {
                        "mode": "primary",
                        "fit_parts": base_fit,
                        "probe_parts": base_probe,
                        "ast": base_ast,
                    }
                )
    if not variants:
        variants.append(
            {
                "mode": "zero",
                "fit_parts": zero_fit,
                "probe_parts": zero_probe,
                "ast": None,
            }
        )
    return variants


def _residual_parts_for_base(
    groups: Sequence[DEFeatureGroup],
    *,
    order: int,
    base_fit_parts: Sequence[torch.Tensor],
    base_probe_parts: Sequence[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    resid_fit_parts: list[torch.Tensor] = []
    resid_probe_parts: list[torch.Tensor] = []
    for group, base_fit, base_probe in zip(groups, base_fit_parts, base_probe_parts):
        anchor_fit, anchor_probe = _anchor_tensor(group.features, order=int(order))
        resid_fit_parts.append(anchor_fit.reshape(-1) + base_fit.reshape(-1))
        resid_probe_parts.append(anchor_probe.reshape(-1) + base_probe.reshape(-1))
    return resid_fit_parts, resid_probe_parts


def _material_improvement(
    probe_rms: float,
    baseline_probe_rms: float,
    *,
    replace_rel_factor: float,
) -> bool:
    if not math.isfinite(probe_rms):
        return False
    if not math.isfinite(baseline_probe_rms):
        return True
    return bool(probe_rms < float(replace_rel_factor) * baseline_probe_rms)


def _best_probe_rms(rows: Sequence[dict[str, Any]]) -> float:
    return min((float(row.get("probe_rms", float("inf"))) for row in list(rows)), default=float("inf"))


def _candidate_sort_key(row: dict[str, Any]) -> tuple[int, float, int, int, float, int, float]:
    base_penalty = 0 if str(row.get("base_mode", "zero")) == "zero" else 1
    return (
        0 if _row_domain_safe(row) else 1,
        float(row.get("probe_rms", float("inf"))),
        base_penalty,
        int(_family_priority(row)),
        float(row.get("shape_score", 0.0) or 0.0),
        int(row.get("symbolic_size_simplified", row.get("size", 10**9))),
        float(row.get("ratio_probe_mse", float("inf"))),
    )


def _consistency_profile(row: dict[str, Any]) -> tuple[bool, float, int, int]:
    score = row.get("consistency_score", None)
    pairs = int(row.get("consistency_pairs", 0) or 0)
    total_pairs = int(row.get("consistency_total_pairs", 0) or 0)
    if score is None:
        return False, float("inf"), pairs, total_pairs
    try:
        score_f = float(score)
    except Exception:
        return False, float("inf"), pairs, total_pairs
    if not math.isfinite(score_f):
        return False, float("inf"), pairs, total_pairs
    if total_pairs <= 0 or pairs <= 0:
        return False, float("inf"), pairs, total_pairs
    coverage = float(pairs) / float(total_pairs)
    return bool(coverage >= 0.5), score_f, pairs, total_pairs


def _consistency_evidence_tier(row: dict[str, Any]) -> tuple[int, str]:
    has_consistency, _score, pairs, total_pairs = _consistency_profile(row)
    if total_pairs <= 0 or pairs <= 0:
        return 0, "unverified"
    coverage = float(pairs) / float(total_pairs)
    score = row.get("consistency_score", None)
    score_f = float(score) if score is not None and math.isfinite(float(score)) else float("inf")
    if has_consistency and score_f <= 0.8:
        return 2, "verified"
    if coverage > 0.0:
        return 1, "weakly_verified"
    return 0, "unverified"


def _candidate_preferred(lhs: dict[str, Any], rhs: dict[str, Any]) -> bool:
    lhs_domain_ok = _row_domain_safe(lhs)
    rhs_domain_ok = _row_domain_safe(rhs)
    if lhs_domain_ok != rhs_domain_ok:
        return bool(lhs_domain_ok)
    lhs_probe = float(lhs.get("probe_rms", float("inf")))
    rhs_probe = float(rhs.get("probe_rms", float("inf")))
    lhs_lane = str(lhs.get("lane", "") or "")
    rhs_lane = str(rhs.get("lane", "") or "")
    lhs_has_consistency, lhs_consistency, _lhs_pairs, _lhs_total_pairs = _consistency_profile(lhs)
    rhs_has_consistency, rhs_consistency, _rhs_pairs, _rhs_total_pairs = _consistency_profile(rhs)
    lhs_tier, _lhs_tier_name = _consistency_evidence_tier(lhs)
    rhs_tier, _rhs_tier_name = _consistency_evidence_tier(rhs)
    lhs_shape = float(lhs.get("shape_score", 0.0) or 0.0)
    rhs_shape = float(rhs.get("shape_score", 0.0) or 0.0)
    lhs_family_priority = _family_priority(lhs)
    rhs_family_priority = _family_priority(rhs)

    first_order_lane_pref = _first_order_lane_preference(lhs, rhs)
    if first_order_lane_pref is not None:
        return bool(first_order_lane_pref)

    if lhs_lane == rhs_lane == "x_coeff_on_u":
        if lhs_family_priority < rhs_family_priority:
            if _x_family_can_replace_explorer(lhs, rhs):
                return True
        if rhs_family_priority < lhs_family_priority:
            if _x_family_can_replace_explorer(rhs, lhs):
                return False
        if lhs_shape + 1.0 < rhs_shape and lhs_probe <= 1.10 * rhs_probe:
            return True
        if rhs_shape + 1.0 < lhs_shape and rhs_probe <= 1.10 * lhs_probe:
            return False

    if lhs_tier != rhs_tier:
        lhs_primary = str(lhs.get("base_mode", "zero")) == "primary"
        rhs_primary = str(rhs.get("base_mode", "zero")) == "primary"
        if lhs_tier > rhs_tier:
            rel = 4.0 if rhs_tier == 0 and rhs_primary else 3.0 if rhs_tier == 0 else 2.0
            if lhs_probe <= rel * rhs_probe:
                return True
        else:
            rel = 4.0 if lhs_tier == 0 and lhs_primary else 3.0 if lhs_tier == 0 else 2.0
            if rhs_probe <= rel * lhs_probe:
                return False

    if lhs_has_consistency and rhs_has_consistency:
        if lhs_consistency <= 0.85 * rhs_consistency and lhs_probe <= 2.0 * rhs_probe:
            return True
        if rhs_consistency <= 0.85 * lhs_consistency and rhs_probe <= 2.0 * lhs_probe:
            return False

    lhs_zero = str(lhs.get("base_mode", "zero")) == "zero"
    rhs_zero = str(rhs.get("base_mode", "zero")) == "zero"
    if lhs_zero != rhs_zero:
        if lhs_zero and lhs_probe <= 1.1 * rhs_probe:
            return True
        if rhs_zero and rhs_probe <= 1.1 * lhs_probe:
            return False

    return _candidate_sort_key(lhs) < _candidate_sort_key(rhs)


def _diverse_candidate_shortlist(
    sorted_rows: Sequence[dict[str, Any]],
    topk: int,
) -> list[dict[str, Any]]:
    """Pick a shortlist with structural diversity from rank-sorted candidates.

    Filling all slots strictly by rank lets one basin's near-clones occupy the
    whole shortlist, so downstream rollout validation has no alternative
    structure to adjudicate. First pass keeps the best row per structural
    signature in rank order; remaining slots are filled by rank. Projection
    variants use their support or snap signature so rollout can adjudicate
    simpler coefficient laws.
    """
    rows = list(sorted_rows)
    topk_i = max(1, int(topk))
    seen: set[tuple[str, str, str, str]] = set()
    primary: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("lane", "") or ""),
            str(row.get("family", "") or ""),
            str(row.get("base_mode", "") or ""),
            str(row.get("projection_signature", "") or ""),
        )
        if key in seen:
            rest.append(row)
        else:
            seen.add(key)
            primary.append(row)
    out = primary[:topk_i]
    if len(out) < topk_i:
        out.extend(rest[: topk_i - len(out)])
    return out


def _compare_candidate_rows(lhs: dict[str, Any], rhs: dict[str, Any]) -> int:
    lhs_pref = _candidate_preferred(lhs, rhs)
    rhs_pref = _candidate_preferred(rhs, lhs)
    if lhs_pref and not rhs_pref:
        return -1
    if rhs_pref and not lhs_pref:
        return 1
    lhs_key = _candidate_sort_key(lhs)
    rhs_key = _candidate_sort_key(rhs)
    if lhs_key < rhs_key:
        return -1
    if rhs_key < lhs_key:
        return 1
    return 0


def _best_probe_row(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    rows_list = list(rows)
    if not rows_list:
        return None
    return min(
        rows_list,
        key=lambda row: (
            float(row.get("probe_rms", float("inf"))),
            float(row.get("shape_score", 0.0) or 0.0),
            int(_family_priority(row)),
            int(row.get("size", 10**9) or 10**9),
            float(row.get("ratio_probe_mse", float("inf"))),
        ),
    )


def _select_lane_representative(
    rows: Sequence[dict[str, Any]],
    *,
    prefer_explorer_rel_factor: float = 1.10,
) -> dict[str, Any] | None:
    rows_sorted = sorted(list(rows), key=cmp_to_key(_compare_candidate_rows))
    if not rows_sorted:
        return None
    best = rows_sorted[0]
    best_probe_rms = float(best.get("probe_rms", float("inf")))
    lane = str(best.get("lane", "") or "")
    explorer_rows = [row for row in rows_sorted if str(row.get("family", "")) == "explorer"]
    family_rows = [row for row in rows_sorted if str(row.get("family", "")) != "explorer"]
    if lane == "x_coeff_on_u" and family_rows:
        if not explorer_rows:
            best_family = _best_global_x_family(family_rows)
            return best if best_family is None else best_family
        best_explorer = _best_probe_row(explorer_rows)
        if best_explorer is None:
            return best
        replacement_family = _replacement_global_x_family(family_rows, best_explorer)
        if replacement_family is not None:
            return replacement_family
        return best_explorer
    if lane == "state_nonlinearity" and family_rows:
        if not explorer_rows:
            best_family = _best_global_state_family(family_rows)
            return best if best_family is None else best_family
        best_explorer = _best_probe_row(explorer_rows)
        if best_explorer is None:
            return best
        replacement_family = _replacement_global_state_family(family_rows, best_explorer)
        if replacement_family is not None:
            return replacement_family
    if not explorer_rows:
        return best
    best_explorer = _best_probe_row(explorer_rows)
    if best_explorer is None:
        return best
    best_explorer_probe_rms = float(best_explorer.get("probe_rms", float("inf")))
    if math.isfinite(best_explorer_probe_rms) and (
        not math.isfinite(best_probe_rms)
        or best_explorer_probe_rms <= float(prefer_explorer_rel_factor) * best_probe_rms
    ):
        return best_explorer
    return best


def _x_family_can_replace_explorer(
    family_row: dict[str, Any],
    explorer_row: dict[str, Any],
    *,
    same_coord_only: bool = True,
) -> bool:
    if same_coord_only and _same_x_lane_coord(family_row) != _same_x_lane_coord(explorer_row):
        return False
    if str(family_row.get("base_mode", "")) != str(explorer_row.get("base_mode", "")):
        return False
    family_probe_rms = float(family_row.get("probe_rms", float("inf")))
    explorer_probe_rms = float(explorer_row.get("probe_rms", float("inf")))
    family_penalized_probe = _x_family_penalized_probe(family_row)
    if not math.isfinite(family_probe_rms):
        return False
    if not math.isfinite(explorer_probe_rms):
        return True
    family_shape = float(family_row.get("shape_score", 0.0) or 0.0)
    explorer_shape = float(explorer_row.get("shape_score", 0.0) or 0.0)
    family_tier, _ = _consistency_evidence_tier(family_row)
    explorer_tier, _ = _consistency_evidence_tier(explorer_row)
    family_priority = _family_priority(family_row)
    family_fit_target = float(family_row.get("fit_target_mse", float("inf")) or float("inf"))
    explorer_fit_target = float(explorer_row.get("fit_target_mse", float("inf")) or float("inf"))
    strong_fit_target_advantage = (
        math.isfinite(family_fit_target)
        and math.isfinite(explorer_fit_target)
        and family_fit_target <= 0.10 * max(explorer_fit_target, 1.0e-12)
    )
    if not same_coord_only and not strong_fit_target_advantage:
        return False
    replace_rel = {
        0: 1.25,  # log
        1: 1.20,  # reciprocal
        2: 1.15,  # inverse-square
        3: 1.10,  # poly2
    }.get(int(family_priority), 1.05)
    if family_tier > explorer_tier:
        return bool(
            family_penalized_probe <= (replace_rel + 0.05) * explorer_probe_rms
            and family_shape <= explorer_shape + 0.25
        )
    if family_tier == explorer_tier:
        if family_penalized_probe <= replace_rel * explorer_probe_rms and family_shape <= explorer_shape + 0.25:
            return True
        if strong_fit_target_advantage and family_penalized_probe <= 2.0 * explorer_probe_rms:
            return True
    return False


def _candidate_identity_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str, str]:
    return (
        str(row.get("lane", "") or ""),
        str(row.get("family", "") or ""),
        str(row.get("base_mode", "") or ""),
        repr(row.get("coord_ast", None)),
        repr(row.get("carrier_ast", None)),
        repr(row.get("coeff_ast", None)),
        repr(row.get("block_ast", None)),
    )


def _same_x_lane_coord(row: dict[str, Any]) -> str:
    return repr(row.get("coord_ast", None))


def _x_family_penalized_probe(row: dict[str, Any]) -> float:
    probe = float(row.get("probe_rms", float("inf")))
    size = max(int(row.get("size", 1) or 1), 1)
    return float(probe * max(1.0, float(size) / 2.0))


def _sorted_global_x_families(
    family_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        list(family_rows),
        key=lambda row: (
            float(row.get("fit_target_mse", float("inf")) or float("inf")),
            int(_family_priority(row)),
            _x_family_penalized_probe(row),
            float(row.get("probe_rms", float("inf"))),
            float(row.get("ratio_probe_mse", float("inf"))),
            float(row.get("shape_score", 0.0) or 0.0),
        ),
    )


def _best_global_x_family(
    family_rows: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    rows = _sorted_global_x_families(family_rows)
    return rows[0] if rows else None


def _replacement_global_x_family(
    family_rows: Sequence[dict[str, Any]],
    explorer_row: dict[str, Any],
) -> dict[str, Any] | None:
    for row in _sorted_global_x_families(family_rows):
        same_coord = _same_x_lane_coord(row) == _same_x_lane_coord(explorer_row)
        if _x_family_can_replace_explorer(row, explorer_row, same_coord_only=same_coord):
            return row
    return None


def _select_x_lane_candidates(
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    rows_sorted = sorted(list(rows), key=cmp_to_key(_compare_candidate_rows))
    if not rows_sorted:
        return None, []
    explorer_rows = [row for row in rows_sorted if str(row.get("family", "")) == "explorer"]
    family_rows = [row for row in rows_sorted if str(row.get("family", "")) != "explorer"]
    representative = _select_lane_representative(rows_sorted)
    if not explorer_rows or not family_rows:
        return representative, [] if representative is None else [representative]

    best_explorer = _best_probe_row(explorer_rows)
    if best_explorer is None:
        return representative, [] if representative is None else [representative]
    best_family = _replacement_global_x_family(family_rows, best_explorer)
    if best_family is None:
        best_family = _best_global_x_family(family_rows)
    kept: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str, str, str]] = set()
    for row in (best_explorer, best_family, representative):
        if row is None:
            continue
        key = _candidate_identity_key(row)
        if key in seen:
            continue
        kept.append(row)
        seen.add(key)
    return representative, kept


def _select_state_lane_candidates(
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    rows_sorted = sorted(list(rows), key=cmp_to_key(_compare_candidate_rows))
    if not rows_sorted:
        return None, []
    explorer_rows = [row for row in rows_sorted if str(row.get("family", "")) == "explorer"]
    family_rows = [row for row in rows_sorted if str(row.get("family", "")) != "explorer"]
    representative = _select_lane_representative(rows_sorted)
    if not explorer_rows or not family_rows:
        return representative, [] if representative is None else [representative]

    best_explorer = _best_probe_row(explorer_rows)
    if best_explorer is None:
        return representative, [] if representative is None else [representative]
    best_family = _replacement_global_state_family(family_rows, best_explorer)
    if best_family is None:
        best_family = _best_global_state_family(family_rows)
    diverse_families = _best_state_family_per_name(family_rows)
    direct_exp_family = _best_direct_state_family(family_rows, "exp")
    kept: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str, str, str]] = set()
    for row in (best_explorer, direct_exp_family, best_family, representative, *diverse_families[:4]):
        if row is None:
            continue
        key = _candidate_identity_key(row)
        if key in seen:
            continue
        kept.append(row)
        seen.add(key)
    return representative, kept


def _finite_row_float(row: dict[str, Any], name: str, default: float = float("inf")) -> float:
    try:
        value = float(row.get(name, default))
    except Exception:
        return float(default)
    return value if math.isfinite(value) else float(default)


def _is_closed_state_family(row: dict[str, Any]) -> bool:
    if str(row.get("lane", "") or "") != "state_nonlinearity":
        return False
    family = str(row.get("family", "") or "")
    return family not in ("", "explorer")


def _is_distilled_state_family(row: dict[str, Any]) -> bool:
    mapping = row.get("mapping", None)
    if isinstance(mapping, dict) and str(mapping.get("_distilled_from", "") or ""):
        return True
    return str(row.get("coeff_expr", "") or "").endswith("[distilled]")


def _state_family_coord_priority(row: dict[str, Any]) -> int:
    coord_key = repr(row.get("coord_ast", None))
    if coord_key == repr(U()):
        return 0
    if coord_key == repr(Add(ConstNode(1.0), U())):
        return 1
    return 2


def _state_family_penalized_probe(row: dict[str, Any]) -> float:
    probe = _finite_row_float(row, "probe_rms")
    size = max(int(row.get("size", 1) or 1), 1)
    return float(probe * max(1.0, float(size) / 2.0))


def _sorted_global_state_families(
    family_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        list(family_rows),
        key=lambda row: (
            _finite_row_float(row, "fit_target_mse"),
            _finite_row_float(row, "probe_target_mse"),
            int(_family_priority(row)),
            _state_family_penalized_probe(row),
            _finite_row_float(row, "probe_rms"),
            _finite_row_float(row, "ratio_probe_mse"),
        ),
    )


def _best_global_state_family(
    family_rows: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    rows = _sorted_global_state_families(family_rows)
    return rows[0] if rows else None


def _best_state_family_per_name(
    family_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    best_by_name: dict[str, dict[str, Any]] = {}
    for row in _sorted_global_state_families(family_rows):
        family = str(row.get("family", "") or "")
        if not family or family == "explorer":
            continue
        if family not in best_by_name:
            best_by_name[family] = row
    return sorted(
        list(best_by_name.values()),
        key=lambda row: (
            int(_family_priority(row)),
            _finite_row_float(row, "fit_target_mse"),
            _finite_row_float(row, "probe_target_mse"),
            _state_family_penalized_probe(row),
            _finite_row_float(row, "probe_rms"),
        ),
    )


def _best_direct_state_family(
    family_rows: Sequence[dict[str, Any]],
    family_name: str,
) -> dict[str, Any] | None:
    family_norm = str(family_name or "")
    direct_rows = [
        row
        for row in list(family_rows)
        if str(row.get("family", "") or "") == family_norm and not _is_distilled_state_family(row)
    ]
    if not direct_rows:
        return None
    return sorted(
        direct_rows,
        key=lambda row: (
            _state_family_coord_priority(row),
            _finite_row_float(row, "fit_target_mse"),
            _finite_row_float(row, "probe_target_mse"),
            _state_family_penalized_probe(row),
            _finite_row_float(row, "probe_rms"),
        ),
    )[0]


def _state_family_can_replace_explorer(
    family_row: dict[str, Any],
    explorer_row: dict[str, Any],
) -> bool:
    if str(family_row.get("lane", "") or "") != "state_nonlinearity":
        return False
    if str(explorer_row.get("lane", "") or "") != "state_nonlinearity":
        return False
    if str(family_row.get("base_mode", "")) != str(explorer_row.get("base_mode", "")):
        return False

    family_probe = _finite_row_float(family_row, "probe_rms")
    explorer_probe = _finite_row_float(explorer_row, "probe_rms")
    if not math.isfinite(family_probe) or not math.isfinite(explorer_probe):
        return False

    family_tier, _ = _consistency_evidence_tier(family_row)
    explorer_tier, _ = _consistency_evidence_tier(explorer_row)
    family_priority = int(_family_priority(family_row))
    family_fit = _finite_row_float(family_row, "fit_target_mse")
    explorer_fit = _finite_row_float(explorer_row, "fit_target_mse")
    strong_fit_target_advantage = (
        math.isfinite(family_fit)
        and math.isfinite(explorer_fit)
        and family_fit <= 0.10 * max(explorer_fit, 1.0e-12)
    )

    if family_tier > explorer_tier and family_probe <= 2.0 * explorer_probe:
        return True
    if family_tier == explorer_tier:
        rel = {0: 1.35, 1: 1.50, 2: 1.75}.get(int(family_priority), 1.20)
        if family_probe <= rel * explorer_probe:
            return True
        if strong_fit_target_advantage and family_probe <= 2.0 * explorer_probe:
            return True
    if strong_fit_target_advantage and family_priority <= 1 and family_probe <= 2.5 * explorer_probe:
        return True
    return False


def _replacement_global_state_family(
    family_rows: Sequence[dict[str, Any]],
    explorer_row: dict[str, Any],
) -> dict[str, Any] | None:
    for row in _sorted_global_state_families(family_rows):
        if _state_family_can_replace_explorer(row, explorer_row):
            return row
    return None


def _state_lane_can_override_x_lane(
    state_row: dict[str, Any],
    x_row: dict[str, Any],
) -> bool:
    if str(state_row.get("lane", "") or "") != "state_nonlinearity":
        return False
    if str(x_row.get("lane", "") or "") != "x_coeff_on_u":
        return False
    if not _is_closed_state_family(state_row):
        return False

    state_probe = _finite_row_float(state_row, "probe_rms")
    x_probe = _finite_row_float(x_row, "probe_rms")
    if not math.isfinite(state_probe) or not math.isfinite(x_probe):
        return False

    state_tier, _ = _consistency_evidence_tier(state_row)
    x_tier, _ = _consistency_evidence_tier(x_row)
    state_family = str(state_row.get("family", "") or "")

    # A verified x-coefficient law should not be displaced by an unverified
    # autonomous alias.  The override is meant for cases where the x lane is an
    # interpolation-shaped disguise and a simple autonomous family explains the
    # same residual at comparable accuracy.
    if x_tier >= 2 and state_tier <= 0:
        return False
    if state_tier > x_tier and state_probe <= 2.0 * x_probe:
        return True
    if state_tier == x_tier and state_probe <= 1.25 * x_probe:
        return True
    if state_family == "exp" and x_tier <= 1 and state_probe <= 2.0 * x_probe:
        return True
    if x_tier <= 0 and state_probe <= 1.60 * x_probe:
        return True
    return False


def _first_order_lane_preference(lhs: dict[str, Any], rhs: dict[str, Any]) -> bool | None:
    lhs_lane = str(lhs.get("lane", "") or "")
    rhs_lane = str(rhs.get("lane", "") or "")
    if lhs_lane == "state_nonlinearity" and rhs_lane == "x_coeff_on_u":
        if _state_lane_can_override_x_lane(lhs, rhs):
            return True
        return None
    if lhs_lane == "x_coeff_on_u" and rhs_lane == "state_nonlinearity":
        if _state_lane_can_override_x_lane(rhs, lhs):
            return False
        return None
    return None


def _evenly_spaced_indices(n_rows: int, max_points: int) -> list[int]:
    n = int(n_rows)
    k = max(1, int(max_points))
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    idxs = {
        int(round(i * float(n - 1) / float(max(k - 1, 1))))
        for i in range(k)
    }
    return sorted(min(max(int(idx), 0), n - 1) for idx in idxs)


def _pooled_target_summary(
    *,
    groups: Sequence[DEFeatureGroup],
    resid_parts: Sequence[torch.Tensor],
    carrier_ast,
    coord_ast,
    split: str,
    x_axis: int,
    rel_eps: float,
    min_rows: int,
    max_points: int = 64,
) -> dict[str, Any] | None:
    pooled = _pooled_same_coord_coeff_target(
        groups=groups,
        resid_parts=resid_parts,
        carrier_ast=carrier_ast,
        coord_ast=coord_ast,
        split=str(split),
        x_axis=int(x_axis),
        rel_eps=float(rel_eps),
        min_rows=int(min_rows),
    )
    if pooled is None:
        return None
    z_vals, y_vals, w_vals = pooled
    z = z_vals.reshape(-1)
    y = y_vals.reshape(-1)
    w = w_vals.reshape(-1)
    mask = torch.isfinite(z) & torch.isfinite(y) & torch.isfinite(w)
    z = z[mask]
    y = y[mask]
    w = w[mask]
    if int(z.numel()) <= 0:
        return None
    order = torch.argsort(z)
    z = z[order]
    y = y[order]
    w = w[order]
    sample_idxs = _evenly_spaced_indices(int(z.numel()), int(max_points))
    samples = [
        {
            "z": float(z[idx].detach().cpu().item()),
            "y": float(y[idx].detach().cpu().item()),
            "weight": float(w[idx].detach().cpu().item()),
        }
        for idx in sample_idxs
    ]
    return {
        "split": str(split),
        "n_rows": int(z.numel()),
        "z_min": float(torch.min(z).detach().cpu().item()),
        "z_max": float(torch.max(z).detach().cpu().item()),
        "y_min": float(torch.min(y).detach().cpu().item()),
        "y_max": float(torch.max(y).detach().cpu().item()),
        "weight_min": float(torch.min(w).detach().cpu().item()),
        "weight_max": float(torch.max(w).detach().cpu().item()),
        "samples": samples,
    }


def _pooled_target_mse_from_local_ast(
    *,
    groups: Sequence[DEFeatureGroup],
    resid_parts: Sequence[torch.Tensor],
    carrier_ast,
    coord_ast,
    coeff_local_ast,
    split: str,
    x_axis: int,
    rel_eps: float,
    min_rows: int,
    robust: bool = False,
) -> float:
    if coeff_local_ast is None:
        return float("inf")
    pooled = _pooled_same_coord_coeff_target(
        groups=groups,
        resid_parts=resid_parts,
        carrier_ast=carrier_ast,
        coord_ast=coord_ast,
        split=str(split),
        x_axis=int(x_axis),
        rel_eps=float(rel_eps),
        min_rows=int(min_rows),
    )
    if pooled is None:
        return float("inf")
    z_vals, y_vals, w_vals = pooled
    z = z_vals.reshape(-1)
    y = y_vals.reshape(-1)
    w = w_vals.reshape(-1)
    try:
        pred = _eval_univariate_ast_on_values(coeff_local_ast, z).reshape(-1)
    except Exception:
        return float("inf")
    finite = torch.isfinite(z) & torch.isfinite(y) & torch.isfinite(w) & torch.isfinite(pred) & (w > 0.0)
    if int(finite.sum()) <= 0:
        return float("inf")
    sq = (pred[finite] - y[finite]).square()
    weights = w[finite]
    if bool(robust):
        order = torch.argsort(sq)
        sq = sq[order]
        weights = weights[order]
        trim_n = int(math.floor(0.15 * int(sq.numel())))
        keep = max(int(sq.numel()) - trim_n, 1)
        sq = sq[:keep]
        weights = weights[:keep]
    denom = max(float(torch.sum(weights).detach().cpu().item()), 1.0e-12)
    return float(torch.sum(weights * sq).detach().cpu().item()) / denom


def _binned_ratio_collapse_variance(
    grouped_rows: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    n_bins: int = 16,
) -> tuple[float | None, float | None]:
    if not grouped_rows:
        return None, None
    z_all = torch.cat([z.reshape(-1) for z, _ in grouped_rows], dim=0)
    y_all = torch.cat([y.reshape(-1) for _, y in grouped_rows], dim=0)
    finite_all = torch.isfinite(z_all) & torch.isfinite(y_all)
    if int(finite_all.sum()) < 2:
        return None, None
    z_all = z_all[finite_all]
    y_all = y_all[finite_all]
    z_min = float(torch.min(z_all).detach().cpu().item())
    z_max = float(torch.max(z_all).detach().cpu().item())
    if not math.isfinite(z_min) or not math.isfinite(z_max) or z_max <= z_min:
        return None, None

    scale = max(float(torch.median(torch.abs(y_all)).detach().cpu().item()), 1.0e-6)
    bins = max(2, int(n_bins))
    edges = torch.linspace(z_min, z_max, bins + 1, dtype=z_all.dtype, device=z_all.device)

    within_vars: list[float] = []
    bin_group_means: list[list[float]] = [[] for _ in range(bins)]
    for z_raw, y_raw in grouped_rows:
        z = z_raw.reshape(-1)
        y = y_raw.reshape(-1)
        finite = torch.isfinite(z) & torch.isfinite(y)
        if int(finite.sum()) < 2:
            continue
        z = z[finite]
        y = y[finite]
        bin_idx = torch.bucketize(z, edges, right=False) - 1
        bin_idx = torch.clamp(bin_idx, 0, bins - 1)
        for b in range(bins):
            mask = bin_idx == int(b)
            if int(mask.sum()) <= 0:
                continue
            vals = y[mask]
            if int(vals.numel()) >= 2:
                within_vars.append(float(torch.var(vals, unbiased=False).detach().cpu().item()) / (scale * scale))
            bin_group_means[b].append(float(torch.mean(vals).detach().cpu().item()))

    cross_vars: list[float] = []
    for means in bin_group_means:
        if len(means) < 2:
            continue
        means_t = torch.as_tensor(means, dtype=y_all.dtype, device=y_all.device)
        cross_vars.append(float(torch.var(means_t, unbiased=False).detach().cpu().item()) / (scale * scale))

    within = None if not within_vars else float(sum(within_vars) / len(within_vars))
    cross = None if not cross_vars else float(sum(cross_vars) / len(cross_vars))
    return within, cross


def _curve_monotonic_and_sign_stats(curves: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> tuple[float, float, float]:
    if not curves:
        return 0.0, 0.0, 0.0
    monotonic = 0
    sign_changes: list[float] = []
    mixed_sign = 0
    for _x_curve, y_curve in curves:
        y = y_curve.reshape(-1)
        finite = torch.isfinite(y)
        y = y[finite]
        if int(y.numel()) < 2:
            continue
        dy = torch.diff(y)
        tol = 1.0e-6 * max(float(torch.max(torch.abs(y)).detach().cpu().item()), 1.0)
        if bool(torch.all(dy >= -tol)) or bool(torch.all(dy <= tol)):
            monotonic += 1
        signs = torch.sign(torch.where(torch.abs(y) >= tol, y, torch.zeros_like(y)))
        nz = signs[signs != 0.0]
        if int(nz.numel()) >= 2:
            sign_changes.append(float(torch.sum(nz[1:] * nz[:-1] < 0.0).detach().cpu().item()))
        else:
            sign_changes.append(0.0)
        if bool(torch.any(y > tol)) and bool(torch.any(y < -tol)):
            mixed_sign += 1
    denom = max(int(len(curves)), 1)
    return (
        float(monotonic) / float(denom),
        float(sum(sign_changes) / max(len(sign_changes), 1)),
        float(mixed_sign) / float(denom),
    )


def _residual_ratio_collapse_diagnostics(
    *,
    groups: Sequence[DEFeatureGroup],
    resid_parts: Sequence[torch.Tensor],
    carrier_ast,
    coord_ast,
    split: str,
    x_axis: int,
    rel_eps: float,
    min_rows: int = 2,
) -> dict[str, Any]:
    total_rows = 0
    safe_rows = 0
    curves: list[tuple[torch.Tensor, torch.Tensor]] = []
    grouped_rows: list[tuple[torch.Tensor, torch.Tensor]] = []

    for group, resid_part in zip(groups, resid_parts):
        resid = resid_part.reshape(-1)
        phi = _eval_ast_on_features(
            carrier_ast,
            features=group.features,
            split=str(split),
            x_axis=int(x_axis),
        ).reshape(-1)
        coord = _eval_ast_on_features(
            coord_ast,
            features=group.features,
            split=str(split),
            x_axis=int(x_axis),
        ).reshape(-1, 1)
        n_rows = min(int(resid.numel()), int(phi.numel()), int(coord.shape[0]))
        if n_rows <= 0:
            continue
        total_rows += n_rows
        resid = resid[:n_rows]
        phi = phi[:n_rows]
        coord = coord[:n_rows]
        ratio, mask = _safe_ratio_target(resid, phi, rel_eps=float(rel_eps))
        if ratio is None or mask is None:
            continue
        safe_rows += int(mask.sum().detach().cpu().item())
        z_valid, ratio_valid = _finite_xy_rows(coord[mask], ratio)
        if int(z_valid.shape[0]) < int(min_rows):
            continue
        z_curve, q_curve = _prepare_curve_xy(z_valid, ratio_valid)
        if int(z_curve.numel()) < int(min_rows):
            continue
        curves.append((z_curve, q_curve))
        grouped_rows.append((z_valid.reshape(-1), ratio_valid.reshape(-1)))

    n_groups = int(len(groups))
    total_pairs = int(len(curves) * (len(curves) - 1) // 2)
    pair_scores: list[float] = []
    for i in range(len(curves)):
        z_i, q_i = curves[i]
        for j in range(i + 1, len(curves)):
            z_j, q_j = curves[j]
            score = _pairwise_curve_relative_mse(z_i, q_i, z_j, q_j)
            if score is not None and math.isfinite(score):
                pair_scores.append(float(score))

    pairs = int(len(pair_scores))
    score = float(sum(pair_scores) / pairs) if pairs > 0 else float("inf")
    coverage = float(pairs) / float(total_pairs) if total_pairs > 0 else 0.0
    group_coverage = float(len(curves)) / float(n_groups) if n_groups > 0 else 0.0
    domain_safe_fraction = float(safe_rows) / float(total_rows) if total_rows > 0 else 0.0
    within_var, cross_var = _binned_ratio_collapse_variance(grouped_rows)
    monotonic_support, sign_changes_mean, mixed_sign_fraction = _curve_monotonic_and_sign_stats(curves)

    if n_groups <= 1:
        reason = "single_dataset_low_confidence"
    elif len(curves) < 2:
        reason = "insufficient_safe_rows"
    elif total_pairs > 0 and pairs <= 0:
        reason = "no_overlap"
    else:
        reason = "ok"

    if reason == "ok" and coverage >= 0.5 and math.isfinite(score) and score <= 0.8:
        confidence = "high"
    elif pairs > 0:
        confidence = "weak"
    else:
        confidence = "low"

    return {
        "collapse_score": float(score),
        "collapse_coverage": float(coverage),
        "collapse_group_coverage": float(group_coverage),
        "collapse_pairs": int(pairs),
        "collapse_total_pairs": int(total_pairs),
        "collapse_reason": str(reason),
        "collapse_confidence": str(confidence),
        "collapse_safe_rows": int(safe_rows),
        "collapse_total_rows": int(total_rows),
        "collapse_domain_safe_fraction": float(domain_safe_fraction),
        "collapse_within_bin_variance": None if within_var is None else float(within_var),
        "collapse_cross_trajectory_variance": None if cross_var is None else float(cross_var),
        "collapse_monotonic_support": float(monotonic_support),
        "collapse_sign_changes_mean": float(sign_changes_mean),
        "collapse_mixed_sign_fraction": float(mixed_sign_fraction),
    }


def _x_lane_candidate_report_row(
    row: dict[str, Any],
    *,
    selected_key: tuple[str, str, str, str, str, str, str] | None,
) -> dict[str, Any]:
    key = _candidate_identity_key(row)
    return {
        "lane": str(row.get("lane", "")),
        "family": str(row.get("family", "")),
        "base_mode": str(row.get("base_mode", "")),
        "coord_ast": None if row.get("coord_ast", None) is None else repr(row.get("coord_ast", None)),
        "carrier_ast": None if row.get("carrier_ast", None) is None else repr(row.get("carrier_ast", None)),
        "coeff_ast": None if row.get("coeff_ast", None) is None else repr(row.get("coeff_ast", None)),
        "coeff_expr": str(row.get("coeff_expr", "")),
        "probe_rms": float(row.get("probe_rms", float("inf"))),
        "probe_mse": float(row.get("probe_mse", float("inf"))),
        "raw_probe_rms": float(row.get("raw_probe_rms", float("inf"))),
        "raw_probe_mse": float(row.get("raw_probe_mse", float("inf"))),
        "ratio_probe_mse": float(row.get("ratio_probe_mse", float("inf"))),
        "fit_target_mse": float(row.get("fit_target_mse", float("inf")))
        if row.get("fit_target_mse", None) is not None and math.isfinite(float(row.get("fit_target_mse")))
        else None,
        "probe_target_mse": float(row.get("probe_target_mse", float("inf")))
        if row.get("probe_target_mse", None) is not None and math.isfinite(float(row.get("probe_target_mse")))
        else None,
        "consistency_score": float(row.get("consistency_score", float("inf")))
        if row.get("consistency_score", None) is not None and math.isfinite(float(row.get("consistency_score")))
        else None,
        "consistency_pairs": int(row.get("consistency_pairs", 0)),
        "consistency_total_pairs": int(row.get("consistency_total_pairs", 0)),
        "evidence_tier": str(row.get("evidence_tier", "")),
        "shape_score": float(row.get("shape_score", 0.0) or 0.0),
        "sign_changes": float(row.get("sign_changes", 0.0) or 0.0),
        "curvature_ratio": float(row.get("curvature_ratio", 0.0) or 0.0),
        "tv_ratio": float(row.get("tv_ratio", 1.0) or 1.0),
        "collapse_score": None
        if not math.isfinite(_finite_row_float(row, "collapse_score"))
        else _finite_row_float(row, "collapse_score"),
        "collapse_coverage": float(row.get("collapse_coverage", 0.0) or 0.0),
        "collapse_group_coverage": float(row.get("collapse_group_coverage", 0.0) or 0.0),
        "collapse_pairs": int(row.get("collapse_pairs", 0) or 0),
        "collapse_total_pairs": int(row.get("collapse_total_pairs", 0) or 0),
        "collapse_reason": str(row.get("collapse_reason", "")),
        "collapse_confidence": str(row.get("collapse_confidence", "")),
        "collapse_domain_safe_fraction": float(row.get("collapse_domain_safe_fraction", 0.0) or 0.0),
        "collapse_within_bin_variance": row.get("collapse_within_bin_variance", None),
        "collapse_cross_trajectory_variance": row.get("collapse_cross_trajectory_variance", None),
        "collapse_monotonic_support": float(row.get("collapse_monotonic_support", 0.0) or 0.0),
        "selected_global": bool(selected_key is not None and key == selected_key),
    }


def _build_zero_base_x_lane_diagnostics(
    *,
    groups: Sequence[DEFeatureGroup],
    x_axis: int,
    rel_eps: float,
    min_ratio_rows: int,
    resid_fit_parts: Sequence[torch.Tensor],
    resid_probe_parts: Sequence[torch.Tensor],
    coord_asts: Sequence[Any],
    x_rows: Sequence[dict[str, Any]],
    selected_row: dict[str, Any] | None,
) -> dict[str, Any] | None:
    zero_rows = [
        row
        for row in list(x_rows)
        if str(row.get("lane", "")) == "x_coeff_on_u" and str(row.get("base_mode", "")) == "zero"
    ]
    if not zero_rows:
        return None
    selected_key = None if selected_row is None else _candidate_identity_key(selected_row)
    score_rows = sorted(zero_rows, key=lambda row: (float(row.get("ratio_probe_mse", float("inf"))), _candidate_sort_key(row)))
    best_ratio_row = score_rows[0] if score_rows else None
    best_probe_row = min(zero_rows, key=lambda row: float(row.get("probe_rms", float("inf"))))
    coord_payloads: list[dict[str, Any]] = []
    for coord_ast in _dedupe_ast_list(coord_asts):
        coord_key = repr(coord_ast)
        coord_rows = [row for row in zero_rows if repr(row.get("coord_ast", None)) == coord_key]
        fit_summary = _pooled_target_summary(
            groups=groups,
            resid_parts=resid_fit_parts,
            carrier_ast=U(),
            coord_ast=coord_ast,
            split="fit",
            x_axis=int(x_axis),
            rel_eps=float(rel_eps),
            min_rows=int(min_ratio_rows),
        )
        probe_summary = _pooled_target_summary(
            groups=groups,
            resid_parts=resid_probe_parts,
            carrier_ast=U(),
            coord_ast=coord_ast,
            split="probe",
            x_axis=int(x_axis),
            rel_eps=float(rel_eps),
            min_rows=int(min_ratio_rows),
        )
        coord_payloads.append(
            {
                "coord_ast": coord_key,
                "fit_target": fit_summary,
                "probe_target": probe_summary,
                "candidate_scores": [
                    _x_lane_candidate_report_row(row, selected_key=selected_key)
                    for row in sorted(
                        coord_rows,
                        key=lambda row: (float(row.get("ratio_probe_mse", float("inf"))), _candidate_sort_key(row)),
                    )
                ],
            }
        )
    return {
        "selected_row_key": None if selected_key is None else list(selected_key),
        "best_by_ratio_probe_mse": None if best_ratio_row is None else _x_lane_candidate_report_row(best_ratio_row, selected_key=selected_key),
        "best_by_probe_rms": None if best_probe_row is None else _x_lane_candidate_report_row(best_probe_row, selected_key=selected_key),
        "coord_reports": coord_payloads,
    }


def _choose_preferred_zero_lane(
    *,
    state_rows: Sequence[dict[str, Any]],
    x_coeff_rows: Sequence[dict[str, Any]],
) -> str | None:
    state_choice = _select_lane_representative(state_rows)
    x_choice = _select_lane_representative(x_coeff_rows)
    if state_choice is not None and x_choice is not None:
        state_tier, _state_tier_name = _consistency_evidence_tier(state_choice)
        x_tier, _x_tier_name = _consistency_evidence_tier(x_choice)
        state_has_consistency, state_consistency, state_pairs, state_total_pairs = _consistency_profile(state_choice)
        x_has_consistency, x_consistency, x_pairs, x_total_pairs = _consistency_profile(x_choice)

        if x_tier != state_tier:
            return "x_coeff_on_u" if x_tier > state_tier else "state_nonlinearity"
        if x_has_consistency != state_has_consistency:
            return "x_coeff_on_u" if x_has_consistency else "state_nonlinearity"
        if x_has_consistency and state_has_consistency:
            if x_consistency <= 0.85 * state_consistency:
                return "x_coeff_on_u"
            if state_consistency <= 0.85 * x_consistency:
                return "state_nonlinearity"
            x_coverage = float(x_pairs) / float(x_total_pairs) if x_total_pairs > 0 else 0.0
            state_coverage = float(state_pairs) / float(state_total_pairs) if state_total_pairs > 0 else 0.0
            if x_coverage > state_coverage + 1.0e-12:
                return "x_coeff_on_u"
            if state_coverage > x_coverage + 1.0e-12:
                return "state_nonlinearity"
            if x_pairs != state_pairs:
                return "x_coeff_on_u" if x_pairs > state_pairs else "state_nonlinearity"
        return "x_coeff_on_u" if _candidate_sort_key(x_choice) < _candidate_sort_key(state_choice) else "state_nonlinearity"
    if x_choice is not None:
        return "x_coeff_on_u"
    if state_choice is not None:
        return "state_nonlinearity"
    return None


def _active_first_order_typed_lanes(
    *,
    base_mode: str,
    preferred_zero_lane: str | None,
    allow_state_lane: bool,
    allow_x_coeff_lane: bool,
) -> tuple[bool, bool]:
    if str(base_mode or "zero") == "zero":
        return bool(allow_state_lane), bool(allow_x_coeff_lane)
    if preferred_zero_lane is None:
        return False, False
    return (
        bool(allow_state_lane and preferred_zero_lane == "state_nonlinearity"),
        bool(allow_x_coeff_lane and preferred_zero_lane == "x_coeff_on_u"),
    )

__factorized_de_definitions__ = (
    "FactorizedDERescueConfig",
    "TypedExplorerLaunchTask",
    "TypedExplorerLaunchState",
    "TypedExplorerLaunchResult",
    "_multiprocessing_start_method_name",
    "_typed_explorer_worker_init",
    "_typed_explorer_worker_run",
    "FactorizedDEBlock",
    "FactorizedDEResult",
    "_anchor_ast",
    "_anchor_name",
    "_canonical_equation",
    "_simplify_de_ast",
    "_compiled_de_ast_payload",
    "_compiled_de_row_payload",
    "_split_tensor",
    "_const_like",
    "_eval_ast_on_features",
    "_anchor_tensor",
    "_sum_linear_terms_ast",
    "_shared_base_from_primary",
    "_dedupe_ast_list",
    "_coord_pool",
    "_carrier_pool",
    "_safe_ratio_target",
    "_normalized_group_quality_weights",
    "_finite_xy_rows",
    "_prepare_curve_xy",
    "_prepare_curve_xy_aux",
    "_interp_1d_sorted",
    "_interp_or_exact_sorted",
    "_eval_univariate_ast_on_values",
    "_weighted_median_1d",
    "_pooled_same_coord_coeff_target",
    "_pairwise_curve_relative_mse",
    "_merge_weighted_curves_on_reference",
    "_curve_shape_stats",
    "_family_priority",
    "_row_domain_safe",
    "_x_lane_shape_stats",
    "_lane_consistency_stats",
    "_pairwise_same_x_witness",
    "_pairwise_same_u_witness",
    "_lane_witness_stats",
    "_probe_mse_from_residuals",
    "_outlier_trim_mask",
    "_leverage_keep_mask",
    "_ridge_lstsq_or_least_norm",
    "_target_scale_row_weights",
    "_scale_weighted_trimmed_lstsq",
    "_trimmed_ridge_lstsq",
    "_trimmed_mean_sq",
    "_trimmed_probe_mse_from_residuals",
    "_quality_weighted_probe_mse_from_residuals",
    "_compose_nonanchor_ast",
    "_coeff_ast_from_row",
    "_masked_original_scale_probe_mse",
    "_fit_original_scale_affine_explorer_head",
    "_fit_original_scale_family_basis",
    "_base_variants",
    "_residual_parts_for_base",
    "_material_improvement",
    "_best_probe_rms",
    "_candidate_sort_key",
    "_consistency_profile",
    "_consistency_evidence_tier",
    "_candidate_preferred",
    "_diverse_candidate_shortlist",
    "_compare_candidate_rows",
    "_best_probe_row",
    "_select_lane_representative",
    "_x_family_can_replace_explorer",
    "_candidate_identity_key",
    "_same_x_lane_coord",
    "_x_family_penalized_probe",
    "_sorted_global_x_families",
    "_best_global_x_family",
    "_replacement_global_x_family",
    "_select_x_lane_candidates",
    "_select_state_lane_candidates",
    "_finite_row_float",
    "_is_closed_state_family",
    "_is_distilled_state_family",
    "_state_family_coord_priority",
    "_state_family_penalized_probe",
    "_sorted_global_state_families",
    "_best_global_state_family",
    "_best_state_family_per_name",
    "_best_direct_state_family",
    "_state_family_can_replace_explorer",
    "_replacement_global_state_family",
    "_state_lane_can_override_x_lane",
    "_first_order_lane_preference",
    "_evenly_spaced_indices",
    "_pooled_target_summary",
    "_pooled_target_mse_from_local_ast",
    "_binned_ratio_collapse_variance",
    "_curve_monotonic_and_sign_stats",
    "_residual_ratio_collapse_diagnostics",
    "_x_lane_candidate_report_row",
    "_build_zero_base_x_lane_diagnostics",
    "_choose_preferred_zero_lane",
    "_active_first_order_typed_lanes",
)

__factorized_de_constants__ = (
    "_TYPED_EXPLORER_WORKER_STATE",
    "_TRIM_FRAC",
    "_TRIM_K_SCALE",
    "_TRIM_MIN_KEEP",
    "_TRIM_LEVERAGE_CAP",
    "_RELATIVE_FIT_DYNAMIC_RANGE",
)

__factorized_de_late_bindings__ = (
    "_finite_design_rows",
    "_run_one_typed_explorer_launch",
)
