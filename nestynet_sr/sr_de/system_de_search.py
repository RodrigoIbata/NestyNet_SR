# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""System DE discovery from a frozen surrogate u(x).

This is the multi-output extension of :mod:`nestynet_sr.sr_de.de_search`.

We discover **systems** of implicit equations that are linear-in-coefficients:

    anchor_i(x,u,du,...) + Σ_k c_{i,k} φ_k(x,u,du,...) = 0

where `i` indexes the equation/component (typically a state vector component).

Key design choice
-----------------
We keep the internal representation *scalar*: each equation residual is a
scalar AST (and each term evaluates to a scalar column). Vector-valued
surrogates are supported via an `out_idx` selector stored in AtomNode.kwargs.

For stable discovery across components, the default sparsification uses a
**group STLSQ** loop that enforces a shared active term support across all
equations (useful for vector PDEs where each component shares the same
operators).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from nestynet_sr.adaptors.u_feature_leaf import UFeatureCache
from nestynet_sr.sr_core.bridges import (
    D2U,
    DU,
    Add,
    AddNode,
    AtomNode,
    ConstNode,
    CosNode,
    ExpNode,
    FreeConst,
    LogNode,
    Mul,
    MulNode,
    Node,
    Pow,
    PowNode,
    SinNode,
    U,
    Var,
)

# Reuse the numerics + units helpers from de_search
from nestynet_sr.sr_de.de_search import (
    group_stlsq,
    rank_aware_lstsq,
    required_coeff_dim_for_term,
    ridge_lstsq,
    stlsq,
    term_units_feasible,
)

# ──────────────────────────────────────────────────────────────
# Configuration & results
# ──────────────────────────────────────────────────────────────


@dataclass
class SystemDESearchConfig:
    """Configuration for system DE discovery from a surrogate.

    Notes
    -----
    * `x_axis` is the independent coordinate axis for the anchor derivative.
      For ODE discovery this is typically the time coordinate.
    * `out_idxs` selects which surrogate outputs are treated as the state.
      If None, defaults to all outputs.
    """

    # Anchor derivative order w.r.t x_axis
    x_axis: int = 0
    order_candidates: Tuple[int, ...] = (1,)

    # Which components of the surrogate output participate in the system
    out_idxs: Optional[Tuple[int, ...]] = None

    # Library controls (kept conservative by default)
    max_x_power: int = 1
    max_u_power: int = 2
    include_const: bool = True
    include_x: bool = True
    include_u: bool = True

    # Cross terms between state components
    include_u_cross: bool = True  # include u_i * u_j (i<j)
    include_xu: bool = True  # include x^p * u_i^q cross terms

    # Optional derivative features in the library
    include_du: bool = False
    du_axes: Tuple[int, ...] = ()  # if empty, no derivative features are added
    include_d2u: bool = False
    d2u_axes: Tuple[Tuple[int, int], ...] = ()

    # STLSQ
    ridge: float = 1e-10
    stlsq_lambda: float = 1e-3
    stlsq_max_iter: int = 10

    # Sampling
    max_batches: int = 32
    max_points: int = 20000

    # Model-selection heuristic
    sparsity_penalty: float = 1e-3

    # Sparsity structure
    share_support_across_equations: bool = True

    # Fail-closed autonomous-vector geometry escalation.
    poisson_auto: bool = True
    poisson_auto_config: Any = None


@dataclass
class SystemDESearchResult:
    """Result of system DE discovery (single dataset).

    Attributes
    ----------
    order : int
        Anchor order (1 for DU, 2 for D2U)
    x_axis : int
        Independent axis used for the anchor
    out_idxs : Tuple[int,...]
        Output components used as equations
    term_asts : List[Node]
        Selected library terms; `None` denotes constant term.
    coeffs : torch.Tensor
        Coeff matrix shape (M, K_sel) where M=len(out_idxs)
    rms_train : List[float]
        RMS residual per equation on training points
    rms_val : List[float] | None
        RMS residual per equation on validation points
    residual_asts : List[Node] | None
        Residual ASTs per equation (anchor + Σ c term)
    """

    order: int
    x_axis: int
    out_idxs: Tuple[int, ...]
    term_asts: List[Node]
    coeffs: torch.Tensor  # (M, K_sel)
    rms_train: List[float]
    rms_val: Optional[List[float]] = None
    residual_asts: Optional[List[Node]] = None
    poisson_report: Any = None

    def format_equation(self, eq: int, tol: float = 1e-3, var_name: str = "x0") -> str:
        """Human-readable single equation string."""
        m = int(eq)
        if m < 0 or m >= len(self.out_idxs):
            raise IndexError("eq index out of range")

        out_idx = int(self.out_idxs[m])
        if self.order == 0:
            lhs = f"u{out_idx}"
        elif self.order == 1:
            lhs = f"u{out_idx}_{var_name}"
        elif self.order == 2:
            lhs = f"u{out_idx}_{var_name}{var_name}"
        else:
            lhs = f"d^{self.order}u{out_idx}/d{var_name}^{self.order}"

        terms_str = []
        for c, term in zip(self.coeffs[m].tolist(), self.term_asts):
            # Snap to simple rationals/integers for printing
            c_snap = c
            for target in [0.0, 0.5, 1.0, 2.0, 3.0, -0.5, -1.0, -2.0, -3.0]:
                if abs(c - target) < tol:
                    c_snap = target
                    break
            if abs(c_snap) < 1e-12:
                continue
            if term is None:
                terms_str.append(f"{c_snap:g}")
                continue

            try:
                term_str = repr(term)
            except Exception:
                term_str = str(type(term).__name__)

            if abs(abs(c_snap) - 1.0) < 1e-12:
                terms_str.append(term_str if c_snap > 0 else f"-{term_str}")
            else:
                terms_str.append(f"{c_snap:g}*{term_str}")

        rhs = " + ".join(terms_str).replace("+ -", "- ")
        if rhs.strip() == "":
            rhs = "0"
        return f"{lhs} + {rhs} = 0"

    def format_system(self, tol: float = 1e-3, var_name: str = "x0") -> str:
        return "\n".join(self.format_equation(i, tol=tol, var_name=var_name) for i in range(len(self.out_idxs)))


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _flatten_x(batch) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        x = batch[0]
    else:
        x = batch
    if x is None:
        raise ValueError("Dataset returned x=None")
    if x.ndim == 1:
        return x.unsqueeze(1)
    if x.ndim == 2:
        return x
    return x.view(x.shape[0], -1)


def _as_N(t: torch.Tensor) -> torch.Tensor:
    """(N,1) -> (N,)"""
    if t.ndim == 2 and t.shape[1] == 1:
        return t[:, 0]
    if t.ndim == 1:
        return t
    raise ValueError(f"Expected (N,) or (N,1), got {tuple(t.shape)}")


def _get_out_idx(atom: AtomNode) -> int:
    kw = getattr(atom, "kwargs", None) or {}
    return int(kw.get("out_idx", kw.get("out", kw.get("component", 0))))


def _eval_ast(node: Node, x: torch.Tensor, cache: UFeatureCache) -> torch.Tensor:
    """Evaluate an SR AST numerically using (x, u, du, d2u) from cache.

    This variant supports vector-valued surrogates by honoring AtomNode.kwargs
    key `out_idx` for u/du/d2u atoms.
    """
    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        kw = getattr(node, "kwargs", None) or {}

        if kind in ("var", "x", "input"):
            if len(node.var_idxs) != 1:
                raise ValueError(f"var atom expects one index; got {node.var_idxs}")
            j = int(node.var_idxs[0])
            return x[:, j : j + 1]

        if kind in ("u", "field", "state"):
            out_idx = _get_out_idx(node)
            cache.ensure(x, need_grad=False, need_hess=False)
            u = cache.u
            if u is None:
                raise RuntimeError("UFeatureCache.u is None after ensure")
            if u.ndim == 2 and u.shape[1] > 1:
                return u[:, out_idx : out_idx + 1]
            return u

        if kind in ("du", "d1u", "grad_u"):
            axis = int(kw.get("axis", 0))
            out_idx = _get_out_idx(node)
            cache.ensure(x, need_grad=True, need_hess=False)
            g = cache.g
            if g is None:
                raise RuntimeError("UFeatureCache.g is None after ensure")
            # g: (N,Ny,Nx) or (N,1,Nx)
            if g.ndim != 3:
                raise ValueError(f"Expected cache.g ndim=3, got {g.ndim}")
            if g.shape[1] > 1:
                return g[:, out_idx : out_idx + 1, axis]
            return g[:, 0:1, axis]

        if kind in ("d2u", "ddu", "hess_u"):
            a0 = int(kw.get("axis0", 0))
            a1 = int(kw.get("axis1", 0))
            out_idx = _get_out_idx(node)
            cache.ensure(x, need_grad=False, need_hess=True)
            H = cache.H
            if H is None:
                raise RuntimeError("UFeatureCache.H is None after ensure")
            if H.ndim != 4:
                raise ValueError(f"Expected cache.H ndim=4, got {H.ndim}")
            if H.shape[1] > 1:
                return H[:, out_idx : out_idx + 1, a0, a1]
            return H[:, 0:1, a0, a1]

        if kind in ("const", "constant"):
            val = float(kw.get("value", 1.0))
            return torch.full((x.shape[0], 1), val, device=x.device, dtype=x.dtype)

        if kind in ("free_const", "freeconst", "free_constant"):
            raise ValueError(
                "free_const atoms should not appear inside library term ASTs (use constants column separately)"
            )

        raise ValueError(f"Unsupported atom kind in system DE eval: {kind!r}")

    if isinstance(node, ConstNode):
        val = float(getattr(node, 'value', 0.0))
        return torch.full((x.shape[0], 1), val, device=x.device, dtype=x.dtype)

    if isinstance(node, AddNode):
        return _eval_ast(node.left, x, cache) + _eval_ast(node.right, x, cache)
    if isinstance(node, MulNode):
        return _eval_ast(node.left, x, cache) * _eval_ast(node.right, x, cache)
    if isinstance(node, PowNode):
        base_val = _eval_ast(node.base, x, cache)
        if isinstance(node.exponent, (int, float)):
            return base_val.pow(node.exponent)
        exp_val = _eval_ast(node.exponent, x, cache)
        if exp_val.numel() == 1:
            return base_val.pow(exp_val.item())
        return torch.pow(base_val, exp_val)
    if isinstance(node, LogNode):
        return torch.log(_eval_ast(node.arg, x, cache))
    if isinstance(node, ExpNode):
        return torch.exp(_eval_ast(node.arg, x, cache))
    if isinstance(node, SinNode):
        return torch.sin(_eval_ast(node.arg, x, cache))
    if isinstance(node, CosNode):
        return torch.cos(_eval_ast(node.arg, x, cache))

    raise TypeError(f"Unknown node type: {type(node)}")


def _gather_x(dataloader, *, max_batches: int, max_points: int, device=None) -> torch.Tensor:
    xs = []
    n = 0
    for bi, batch in enumerate(dataloader):
        if bi >= max_batches or n >= max_points:
            break
        x = _flatten_x(batch)
        if device is not None:
            x = x.to(device)
        xs.append(x)
        n += x.shape[0]
    if not xs:
        raise ValueError("No batches found in dataloader")
    X = torch.cat(xs, dim=0)
    if X.shape[0] > max_points:
        X = X[:max_points]
    return X


# ──────────────────────────────────────────────────────────────
# Library generation
# ──────────────────────────────────────────────────────────────


def _pow_if(node: Node, p: int) -> Node:
    if p == 1:
        return node
    return Pow(node, p)


def build_system_library_terms(cfg: SystemDESearchConfig, *, order: int) -> List[Node]:
    """Return a list of candidate term ASTs φ_k for system search."""
    xj = Var(cfg.x_axis)

    # out_idxs may be None at config time; caller should resolve.
    if cfg.out_idxs is None:
        raise ValueError("build_system_library_terms requires cfg.out_idxs to be resolved")
    outs = tuple(int(i) for i in cfg.out_idxs)

    terms: List[Node] = []

    # x^p
    if cfg.include_x:
        for p in range(1, max(1, int(cfg.max_x_power)) + 1):
            terms.append(_pow_if(xj, p))

    # u_i^q
    if cfg.include_u:
        for oi in outs:
            ui = U(out_idx=int(oi))
            for q in range(1, max(1, int(cfg.max_u_power)) + 1):
                terms.append(_pow_if(ui, q))

    # pairwise cross terms u_i*u_j
    if cfg.include_u_cross and cfg.include_u and len(outs) > 1:
        for a in range(len(outs)):
            for b in range(a + 1, len(outs)):
                terms.append(Mul(U(out_idx=int(outs[a])), U(out_idx=int(outs[b]))))

    # x^p * u_i^q
    if cfg.include_xu and cfg.include_x and cfg.include_u:
        for oi in outs:
            ui = U(out_idx=int(oi))
            for p in range(1, max(1, int(cfg.max_x_power)) + 1):
                for q in range(1, max(1, int(cfg.max_u_power)) + 1):
                    terms.append(Mul(_pow_if(xj, p), _pow_if(ui, q)))

    # Optional derivative features
    if cfg.include_du and len(cfg.du_axes) > 0:
        for oi in outs:
            for ax in cfg.du_axes:
                # Avoid trivial inclusion of the anchor derivative when order==1
                if order == 1 and int(ax) == int(cfg.x_axis):
                    continue
                terms.append(DU(int(ax), out_idx=int(oi)))

    if cfg.include_d2u and len(cfg.d2u_axes) > 0:
        for oi in outs:
            for (a0, a1) in cfg.d2u_axes:
                # Avoid trivial inclusion of the anchor second derivative when order==2
                if order == 2 and int(a0) == int(cfg.x_axis) and int(a1) == int(cfg.x_axis):
                    continue
                terms.append(D2U(int(a0), int(a1), out_idx=int(oi)))

    # Deduplicate
    uniq: Dict[str, Node] = {}
    for t in terms:
        uniq[repr(t)] = t
    return list(uniq.values())


# ──────────────────────────────────────────────────────────────
# Main search
# ──────────────────────────────────────────────────────────────


def discover_system_de_from_surrogate(
    surrogate,
    train_dataloader,
    val_dataloader=None,
    *,
    cfg: Optional[SystemDESearchConfig] = None,
    device=None,
    dataset=None,
    extra_terms: Optional[Sequence[Node]] = None,
    library_terms: Optional[Sequence[Node]] = None,
) -> SystemDESearchResult:
    """Discover a sparse system of implicit DEs from a frozen surrogate u(x)."""

    if cfg is None:
        cfg = SystemDESearchConfig()

    # Auto-detect x_axis from dataset coordinate metadata if available
    if dataset is not None:
        try:
            if hasattr(dataset, "has_coord_metadata") and dataset.has_coord_metadata():
                time_coords = dataset.get_time_coords()
                if time_coords and len(time_coords) > 0:
                    detected = int(time_coords[0])
                    if cfg.x_axis == 0 or cfg.x_axis is None:
                        cfg = SystemDESearchConfig(
                            x_axis=detected,
                            order_candidates=cfg.order_candidates,
                            out_idxs=cfg.out_idxs,
                            max_x_power=cfg.max_x_power,
                            max_u_power=cfg.max_u_power,
                            include_const=cfg.include_const,
                            include_x=cfg.include_x,
                            include_u=cfg.include_u,
                            include_u_cross=cfg.include_u_cross,
                            include_xu=cfg.include_xu,
                            include_du=cfg.include_du,
                            du_axes=cfg.du_axes,
                            include_d2u=cfg.include_d2u,
                            d2u_axes=cfg.d2u_axes,
                            ridge=cfg.ridge,
                            stlsq_lambda=cfg.stlsq_lambda,
                            stlsq_max_iter=cfg.stlsq_max_iter,
                            max_batches=cfg.max_batches,
                            max_points=cfg.max_points,
                            sparsity_penalty=cfg.sparsity_penalty,
                            share_support_across_equations=cfg.share_support_across_equations,
                            poisson_auto=cfg.poisson_auto,
                            poisson_auto_config=cfg.poisson_auto_config,
                        )
        except Exception:
            pass

    dev = device
    if dev is None:
        try:
            dev = next(surrogate.parameters()).device
        except Exception:
            dev = torch.device("cpu")

    Xtr = _gather_x(
        train_dataloader, max_batches=cfg.max_batches, max_points=cfg.max_points, device=dev
    )
    Xva = None
    if val_dataloader is not None:
        Xva = _gather_x(
            val_dataloader, max_batches=cfg.max_batches, max_points=cfg.max_points, device=dev
        )

    cache = UFeatureCache(surrogate)
    cache.ensure(Xtr, need_grad=False, need_hess=False)
    if cache.u is None:
        raise RuntimeError("Failed to evaluate surrogate output")
    Ny = int(cache.u.shape[1]) if cache.u.ndim == 2 else 1

    out_idxs = cfg.out_idxs
    if out_idxs is None:
        out_idxs = tuple(range(Ny))
    else:
        out_idxs = tuple(int(i) for i in out_idxs)
        for i in out_idxs:
            if i < 0 or i >= Ny:
                raise ValueError(f"out_idx {i} out of range (Ny={Ny})")

    # Ensure cfg has resolved out_idxs for library building
    cfg_resolved = SystemDESearchConfig(
        x_axis=int(cfg.x_axis),
        order_candidates=tuple(int(o) for o in cfg.order_candidates),
        out_idxs=out_idxs,
        max_x_power=int(cfg.max_x_power),
        max_u_power=int(cfg.max_u_power),
        include_const=bool(cfg.include_const),
        include_x=bool(cfg.include_x),
        include_u=bool(cfg.include_u),
        include_u_cross=bool(cfg.include_u_cross),
        include_xu=bool(cfg.include_xu),
        include_du=bool(cfg.include_du),
        du_axes=tuple(int(a) for a in (cfg.du_axes or ())),
        include_d2u=bool(cfg.include_d2u),
        d2u_axes=tuple((int(a0), int(a1)) for (a0, a1) in (cfg.d2u_axes or ())),
        ridge=float(cfg.ridge),
        stlsq_lambda=float(cfg.stlsq_lambda),
        stlsq_max_iter=int(cfg.stlsq_max_iter),
        max_batches=int(cfg.max_batches),
        max_points=int(cfg.max_points),
        sparsity_penalty=float(cfg.sparsity_penalty),
        share_support_across_equations=bool(cfg.share_support_across_equations),
        poisson_auto=bool(cfg.poisson_auto),
        poisson_auto_config=cfg.poisson_auto_config,
    )

    best: Optional[SystemDESearchResult] = None
    best_score = float("inf")

    for order in cfg_resolved.order_candidates:
        if int(order) not in (0, 1, 2):
            continue

        cache.reset()
        if int(order) == 0:
            cache.ensure(Xtr, need_grad=False, need_hess=False)
            if cache.u is None:
                raise RuntimeError("Surrogate output unavailable")
            anchor_full = cache.u if cache.u.ndim == 2 else cache.u.unsqueeze(1)  # (N,Ny)
        elif int(order) == 1:
            cache.ensure(Xtr, need_grad=True, need_hess=False)
            if cache.g is None:
                raise RuntimeError("Surrogate grad unavailable")
            anchor_full = cache.g[:, :, int(cfg_resolved.x_axis)]  # (N,Ny)
        else:
            cache.ensure(Xtr, need_grad=False, need_hess=True)
            if cache.H is None:
                raise RuntimeError("Surrogate grad_grad unavailable")
            anchor_full = cache.H[:, :, int(cfg_resolved.x_axis), int(cfg_resolved.x_axis)]  # (N,Ny)

        # Build library terms
        if library_terms is None:
            terms = build_system_library_terms(cfg_resolved, order=int(order))
        else:
            terms = [t for t in library_terms if t is not None]
        if extra_terms is not None:
            terms.extend([t for t in extra_terms if t is not None])

        # Deduplicate after optional extension/override
        if len(terms) > 1:
            uniq: Dict[str, Node] = {}
            for t in terms:
                uniq[repr(t)] = t
            terms = list(uniq.values())

        # Design matrix Phi (shared across equations)
        cols = []
        term_asts: List[Node] = []
        if cfg_resolved.include_const:
            cols.append(torch.ones(Xtr.shape[0], device=dev, dtype=Xtr.dtype))
            term_asts.append(None)
        for t in terms:
            v = _as_N(_eval_ast(t, Xtr, cache))
            cols.append(v)
            term_asts.append(t)
        if not cols:
            continue
        Phi = torch.stack(cols, dim=1)  # (N,K)

        # Build per-equation targets
        Phis: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        for oi in out_idxs:
            y = -anchor_full[:, int(oi)]
            m = torch.isfinite(y)
            m &= torch.isfinite(Phi).all(dim=1)
            if int(m.sum()) < 10:
                raise RuntimeError(f"Too few finite rows for out_idx={oi} (order={order})")
            Phis.append(Phi[m])
            ys.append(y[m])

        # Solve
        if cfg_resolved.share_support_across_equations:
            C, keep = group_stlsq(
                Phis,
                ys,
                ridge=cfg_resolved.ridge,
                lam=cfg_resolved.stlsq_lambda,
                max_iter=cfg_resolved.stlsq_max_iter,
            )
        else:
            # Independent STLSQ per equation, then union the supports.
            keeps = []
            Cs = []
            for i in range(len(out_idxs)):
                c_i, keep_i = stlsq(
                    Phis[i],
                    ys[i],
                    ridge=cfg_resolved.ridge,
                    lam=cfg_resolved.stlsq_lambda,
                    max_iter=cfg_resolved.stlsq_max_iter,
                )
                keeps.append(keep_i)
                Cs.append(c_i)
            keep = torch.stack(keeps, dim=0).any(dim=0)
            torch.stack(Cs, dim=0)

        Ksel = int(keep.sum())
        if Ksel == 0:
            continue

        # Unbiased refit on selected terms (per equation)
        Csel = torch.zeros((len(out_idxs), Ksel), device=dev, dtype=Phi.dtype)
        for i in range(len(out_idxs)):
            Csel[i] = ridge_lstsq(Phis[i][:, keep], ys[i], ridge=0.0)

        term_sel = [t for t, k in zip(term_asts, keep.tolist()) if k]

        # Training RMS per equation
        rms_tr: List[float] = []
        for i in range(len(out_idxs)):
            r = (-ys[i]) + Phis[i][:, keep] @ Csel[i]
            rms_tr.append(float((r.square().mean().sqrt()).detach().cpu()))

        # Validation RMS per equation
        rms_va = None
        if Xva is not None:
            cache.reset()
            if int(order) == 0:
                cache.ensure(Xva, need_grad=False, need_hess=False)
                anchor_va_full = cache.u if cache.u.ndim == 2 else cache.u.unsqueeze(1)
            elif int(order) == 1:
                cache.ensure(Xva, need_grad=True, need_hess=False)
                anchor_va_full = cache.g[:, :, int(cfg_resolved.x_axis)]
            else:
                cache.ensure(Xva, need_grad=False, need_hess=True)
                anchor_va_full = cache.H[:, :, int(cfg_resolved.x_axis), int(cfg_resolved.x_axis)]

            cols_va = []
            if cfg_resolved.include_const:
                cols_va.append(torch.ones(Xva.shape[0], device=dev, dtype=Xva.dtype))
            for t in terms:
                cols_va.append(_as_N(_eval_ast(t, Xva, cache)))
            Phi_va_full = torch.stack(cols_va, dim=1)
            Phi_va_sel = Phi_va_full[:, keep]

            rms_va = []
            for i, oi in enumerate(out_idxs):
                yva = -anchor_va_full[:, int(oi)]
                m = torch.isfinite(yva)
                m &= torch.isfinite(Phi_va_sel).all(dim=1)
                if int(m.sum()) < 10:
                    rms_va.append(float("nan"))
                    continue
                rva = (-yva[m]) + Phi_va_sel[m] @ Csel[i]
                rms_va.append(float((rva.square().mean().sqrt()).detach().cpu()))

        ref = rms_va if rms_va is not None else rms_tr
        # Penalise NaNs heavily
        ref2 = [v if (v == v) else 1e9 for v in ref]
        score = float(sum(ref2) / max(1, len(ref2))) + cfg_resolved.sparsity_penalty * len(term_sel)

        if best is None or score < best_score:
            best_score = score
            best = SystemDESearchResult(
                order=int(order),
                x_axis=int(cfg_resolved.x_axis),
                out_idxs=out_idxs,
                term_asts=term_sel,
                coeffs=Csel.detach().cpu(),
                rms_train=rms_tr,
                rms_val=rms_va,
            )

    if best is None:
        raise RuntimeError("System DE discovery failed to produce any candidate")

    best.residual_asts = build_system_residual_asts(best)
    try:
        from nestynet_sr.sr_de.poisson_auto import (
            AutoPoissonConfig,
            auto_discover_poisson_from_system_result,
        )

        cache.reset()
        cache.ensure(Xtr, need_grad=False, need_hess=False)
        if cache.u is None:
            raise RuntimeError("surrogate state values unavailable for Poisson routing")
        state_points = cache.u[:, list(out_idxs)].detach()
        auto_cfg = cfg_resolved.poisson_auto_config
        if auto_cfg is None:
            auto_cfg = AutoPoissonConfig(enabled=bool(cfg_resolved.poisson_auto))
        elif not bool(cfg_resolved.poisson_auto):
            auto_cfg = replace(auto_cfg, enabled=False)
        best.poisson_report = auto_discover_poisson_from_system_result(
            best,
            state_points,
            auto_cfg,
        )
    except Exception as exc:
        try:
            from nestynet_sr.sr_de.poisson_auto import AutoPoissonReport

            best.poisson_report = AutoPoissonReport(
                status="failed",
                reason=f"system_integration_failed:{type(exc).__name__}:{str(exc)[:160]}",
                enabled=bool(cfg_resolved.poisson_auto),
                state_dim=len(out_idxs),
                dataset_count=1,
                sample_counts=(int(Xtr.shape[0]),),
            )
        except Exception:
            best.poisson_report = None
    return best


def build_system_residual_asts(result: SystemDESearchResult, *, coeff_prefix: str = "c") -> List[Node]:
    """Build residual ASTs per equation: anchor_i + Σ c_{i,k} term_k."""
    residuals: List[Node] = []
    x_axis = int(result.x_axis)
    for i, oi in enumerate(result.out_idxs):
        oi = int(oi)
        if int(result.order) == 0:
            root: Node = U(out_idx=oi)
        elif int(result.order) == 1:
            root = DU(x_axis, out_idx=oi)
        elif int(result.order) == 2:
            root = D2U(x_axis, x_axis, out_idx=oi)
        else:
            raise ValueError(f"Unsupported anchor order: {result.order}")

        for k, (term, c) in enumerate(zip(result.term_asts, result.coeffs[i].tolist())):
            name = f"{coeff_prefix}{i}_{k}"
            if term is None:
                root = Add(root, FreeConst(name, init=float(c)))
            else:
                root = Add(root, Mul(FreeConst(name, init=float(c)), term))
        residuals.append(root)
    return residuals


# ──────────────────────────────────────────────────────────────
# Vector equation discovery (tied coefficients across components)
# ──────────────────────────────────────────────────────────────


@dataclass
class VectorDESearchConfig:
    """Configuration for *vector* DE discovery from a surrogate.

    This is a thin, opinionated layer over :func:`discover_system_de_from_surrogate`
    for the common case where you want to discover a **single vector equation**
    (e.g. Maxwell, Navier–Stokes momentum, MHD induction), where all components
    share the same scalar coefficients.

    You provide a list of vector-valued candidate terms (each term is a sequence
    of scalar AST Nodes with length equal to ``len(out_idxs)``). The solver then
    fits a *shared* coefficient vector ``c_k`` such that for each component i:

        anchor_i + Σ_k c_k * term_k[i] = 0.

    Notes
    -----
    * ``out_idxs`` must be provided and defines the component ordering.
    * Sparsification is done with standard STLSQ on the stacked component system.
    """

    x_axis: int = 0
    order_candidates: Tuple[int, ...] = (1,)

    out_idxs: Optional[Tuple[int, ...]] = None

    include_const: bool = False

    ridge: float = 1e-10
    stlsq_lambda: float = 1e-3
    stlsq_max_iter: int = 10

    max_batches: int = 32
    max_points: int = 20000

    sparsity_penalty: float = 1e-3

    # Optional dimensional-analysis context for vector terms / tied coefficients.
    units_spec: Any = None
    enforce_units: bool = False


@dataclass
class VectorDESearchResult:
    """Result of vector DE discovery."""

    order: int
    x_axis: int
    out_idxs: Tuple[int, ...]

    # Selected vector terms; `None` denotes a constant vector term (all-ones).
    term_vecs: List[Optional[Sequence[Node]]]

    # Shared coefficient vector, shape (K_sel,)
    coeffs: torch.Tensor

    rms_train: List[float]
    rms_val: Optional[List[float]] = None
    residual_asts: Optional[List[Node]] = None

    def format_equation(self, eq: int, tol: float = 1e-3, var_name: str = "x0") -> str:
        m = int(eq)
        if m < 0 or m >= len(self.out_idxs):
            raise IndexError("eq index out of range")

        out_idx = int(self.out_idxs[m])
        if self.order == 0:
            lhs = f"u{out_idx}"
        elif self.order == 1:
            lhs = f"u{out_idx}_{var_name}"
        elif self.order == 2:
            lhs = f"u{out_idx}_{var_name}{var_name}"
        else:
            lhs = f"d^{self.order}u{out_idx}/d{var_name}^{self.order}"

        terms_str = []
        for c, tvec in zip(self.coeffs.tolist(), self.term_vecs):
            c_snap = c
            for target in [0.0, 0.5, 1.0, 2.0, 3.0, -0.5, -1.0, -2.0, -3.0]:
                if abs(c - target) < tol:
                    c_snap = target
                    break
            if abs(c_snap) < 1e-12:
                continue

            if tvec is None:
                terms_str.append(f"{c_snap:g}")
                continue

            try:
                term_str = repr(tvec[m])
            except Exception:
                term_str = str(type(tvec[m]).__name__)

            if abs(abs(c_snap) - 1.0) < 1e-12:
                terms_str.append(term_str if c_snap > 0 else f"-{term_str}")
            else:
                terms_str.append(f"{c_snap:g}*{term_str}")

        rhs = " + ".join(terms_str).replace("+ -", "- ")
        if rhs.strip() == "":
            rhs = "0"
        return f"{lhs} + {rhs} = 0"

    def format_system(self, tol: float = 1e-3, var_name: str = "x0") -> str:
        return "\n".join(self.format_equation(i, tol=tol, var_name=var_name) for i in range(len(self.out_idxs)))


def _dedup_vector_terms(terms: Sequence[Sequence[Node]]) -> List[Sequence[Node]]:
    uniq: Dict[str, Sequence[Node]] = {}
    for t in terms:
        key = "|".join(repr(c) for c in t)
        uniq[key] = t
    return list(uniq.values())


def discover_vector_de_from_surrogate(
    surrogate,
    train_dataloader,
    val_dataloader=None,
    *,
    cfg: Optional[VectorDESearchConfig] = None,
    device=None,
    dataset=None,
    vector_terms: Optional[Sequence[Sequence[Node]]] = None,
    extra_vector_terms: Optional[Sequence[Sequence[Node]]] = None,
    library_vector_terms: Optional[Sequence[Sequence[Node]]] = None,
) -> VectorDESearchResult:
    """Discover a sparse **vector** DE from a frozen surrogate.

    Parameters
    ----------
    vector_terms : sequence[sequence[Node]]
        Vector-valued candidate terms. Each term is a sequence of scalar AST
        Nodes with length equal to ``len(cfg.out_idxs)``.

    Notes
    -----
    This enforces coefficient tying across components by fitting a single
    coefficient vector to the stacked component regression.
    """

    if cfg is None:
        cfg = VectorDESearchConfig()

    # Optionally auto-detect x_axis from dataset metadata.
    if dataset is not None:
        try:
            if hasattr(dataset, "has_coord_metadata") and dataset.has_coord_metadata():
                time_coords = dataset.get_time_coords()
                if time_coords and len(time_coords) > 0:
                    detected = int(time_coords[0])
                    if cfg.x_axis == 0 or cfg.x_axis is None:
                        cfg = VectorDESearchConfig(
                            x_axis=detected,
                            order_candidates=cfg.order_candidates,
                            out_idxs=cfg.out_idxs,
                            include_const=cfg.include_const,
                            ridge=cfg.ridge,
                            stlsq_lambda=cfg.stlsq_lambda,
                            stlsq_max_iter=cfg.stlsq_max_iter,
                            max_batches=cfg.max_batches,
                            max_points=cfg.max_points,
                            sparsity_penalty=cfg.sparsity_penalty,
                            units_spec=cfg.units_spec,
                            enforce_units=cfg.enforce_units,
                        )
        except Exception:
            pass

    if cfg.out_idxs is None:
        raise ValueError("VectorDESearchConfig.out_idxs must be provided")

    dev = device
    if dev is None:
        try:
            dev = next(surrogate.parameters()).device
        except Exception:
            dev = torch.device("cpu")

    Xtr = _gather_x(train_dataloader, max_batches=int(cfg.max_batches), max_points=int(cfg.max_points), device=dev)
    Xva = None
    if val_dataloader is not None:
        Xva = _gather_x(val_dataloader, max_batches=int(cfg.max_batches), max_points=int(cfg.max_points), device=dev)

    cache = UFeatureCache(surrogate)
    cache.ensure(Xtr, need_grad=False, need_hess=False)
    if cache.u is None:
        raise RuntimeError("Failed to evaluate surrogate output")
    Ny = int(cache.u.shape[1]) if cache.u.ndim == 2 else 1

    out_idxs = tuple(int(i) for i in cfg.out_idxs)
    for i in out_idxs:
        if i < 0 or i >= Ny:
            raise ValueError(f"out_idx {i} out of range (Ny={Ny})")

    dim = len(out_idxs)
    if dim < 1:
        raise ValueError("out_idxs must be non-empty")

    # Choose the term set.
    if library_vector_terms is not None:
        terms_v = list(library_vector_terms)
    else:
        if vector_terms is None:
            raise ValueError("vector_terms must be provided (or pass library_vector_terms)")
        terms_v = list(vector_terms)

    if extra_vector_terms is not None:
        terms_v.extend(list(extra_vector_terms))

    # Validate term dimensions.
    for t in terms_v:
        if len(t) != dim:
            raise ValueError(f"Vector term has len={len(t)} but system dim={dim}")

    terms_v = _dedup_vector_terms(terms_v)
    term_vecs_all: List[Optional[Sequence[Node]]] = []
    if bool(cfg.include_const):
        term_vecs_all.append(None)
    term_vecs_all.extend(terms_v)
    output_dims = _vector_output_dims(getattr(cfg, "units_spec", None), Ny=Ny)
    eq_spec_units = VectorEquationSpec(out_idxs=out_idxs, name="vector")

    best: Optional[VectorDESearchResult] = None
    best_score = float("inf")

    for order in tuple(int(o) for o in cfg.order_candidates):
        if order not in (0, 1, 2):
            continue

        term_ok_tbl, _, _ = _vector_term_units_tables(
            equations=(eq_spec_units,),
            term_vecs_all=term_vecs_all,
            order=int(order),
            x_axis=int(cfg.x_axis),
            units_spec=getattr(cfg, "units_spec", None),
            output_dims=output_dims,
            enforce_units=bool(getattr(cfg, "enforce_units", False)),
        )
        if term_ok_tbl is not None and not any(bool(v) for v in term_ok_tbl[0]):
            continue

        cache.reset()
        if order == 0:
            cache.ensure(Xtr, need_grad=False, need_hess=False)
            if cache.u is None:
                raise RuntimeError("Surrogate output unavailable")
            anchor_full = cache.u if cache.u.ndim == 2 else cache.u.unsqueeze(1)  # (N,Ny)
        elif order == 1:
            cache.ensure(Xtr, need_grad=True, need_hess=False)
            if cache.g is None:
                raise RuntimeError("Surrogate grad unavailable")
            anchor_full = cache.g[:, :, int(cfg.x_axis)]  # (N,Ny)
        else:
            cache.ensure(Xtr, need_grad=False, need_hess=True)
            if cache.H is None:
                raise RuntimeError("Surrogate grad_grad unavailable")
            anchor_full = cache.H[:, :, int(cfg.x_axis), int(cfg.x_axis)]  # (N,Ny)

        # Build per-component Phi_i and y_i, then stack.
        Phis = []
        ys = []
        masks = []
        for ci, oi in enumerate(out_idxs):
            cols = []
            if bool(cfg.include_const):
                if term_ok_tbl is None or bool(term_ok_tbl[0][0]):
                    cols.append(torch.ones(Xtr.shape[0], device=dev, dtype=Xtr.dtype))
                else:
                    cols.append(torch.zeros(Xtr.shape[0], device=dev, dtype=Xtr.dtype))
            for tk, t in enumerate(terms_v):
                full_k = (1 if bool(cfg.include_const) else 0) + tk
                if term_ok_tbl is not None and not bool(term_ok_tbl[0][full_k]):
                    cols.append(torch.zeros(Xtr.shape[0], device=dev, dtype=Xtr.dtype))
                else:
                    cols.append(_as_N(_eval_ast(t[ci], Xtr, cache)))
            Phi_i = torch.stack(cols, dim=1) if cols else Xtr.new_zeros(Xtr.shape[0], 0)
            y_i = -anchor_full[:, int(oi)]
            m = torch.isfinite(y_i)
            m &= torch.isfinite(Phi_i).all(dim=1)
            if int(m.sum()) < 10:
                raise RuntimeError(f"Too few finite rows for out_idx={oi} (order={order})")
            Phis.append(Phi_i[m])
            ys.append(y_i[m])
            masks.append(m)

        Phi_stack = torch.cat(Phis, dim=0)
        y_stack = torch.cat(ys, dim=0)

        c, keep = stlsq(
            Phi_stack,
            y_stack,
            ridge=float(cfg.ridge),
            lam=float(cfg.stlsq_lambda),
            max_iter=int(cfg.stlsq_max_iter),
        )

        Ksel = int(keep.sum())
        if Ksel == 0:
            continue

        # Unbiased refit on selected terms.
        csel = ridge_lstsq(Phi_stack[:, keep], y_stack, ridge=0.0)

        # Selected term vectors (with optional const sentinel).
        term_sel = [t for t, k in zip(term_vecs_all, keep.tolist()) if k]

        # Per-component RMS (train)
        rms_tr: List[float] = []
        for i in range(dim):
            r = (-ys[i]) + Phis[i][:, keep] @ csel
            rms_tr.append(float((r.square().mean().sqrt()).detach().cpu()))

        # Validation RMS
        rms_va = None
        if Xva is not None:
            cache.reset()
            if order == 0:
                cache.ensure(Xva, need_grad=False, need_hess=False)
                anchor_va_full = cache.u if cache.u.ndim == 2 else cache.u.unsqueeze(1)
            elif order == 1:
                cache.ensure(Xva, need_grad=True, need_hess=False)
                anchor_va_full = cache.g[:, :, int(cfg.x_axis)]
            else:
                cache.ensure(Xva, need_grad=False, need_hess=True)
                anchor_va_full = cache.H[:, :, int(cfg.x_axis), int(cfg.x_axis)]

            rms_va = []
            for ci, oi in enumerate(out_idxs):
                cols = []
                if bool(cfg.include_const):
                    if term_ok_tbl is None or bool(term_ok_tbl[0][0]):
                        cols.append(torch.ones(Xva.shape[0], device=dev, dtype=Xva.dtype))
                    else:
                        cols.append(torch.zeros(Xva.shape[0], device=dev, dtype=Xva.dtype))
                for tk, t in enumerate(terms_v):
                    full_k = (1 if bool(cfg.include_const) else 0) + tk
                    if term_ok_tbl is not None and not bool(term_ok_tbl[0][full_k]):
                        cols.append(torch.zeros(Xva.shape[0], device=dev, dtype=Xva.dtype))
                    else:
                        cols.append(_as_N(_eval_ast(t[ci], Xva, cache)))
                Phi_va_i = torch.stack(cols, dim=1) if cols else Xva.new_zeros(Xva.shape[0], 0)
                yva = -anchor_va_full[:, int(oi)]
                m = torch.isfinite(yva)
                m &= torch.isfinite(Phi_va_i[:, keep]).all(dim=1)
                if int(m.sum()) < 10:
                    rms_va.append(float("nan"))
                    continue
                rva = (-yva[m]) + Phi_va_i[m][:, keep] @ csel
                rms_va.append(float((rva.square().mean().sqrt()).detach().cpu()))

        ref = rms_va if rms_va is not None else rms_tr
        ref2 = [v if (v == v) else 1e9 for v in ref]
        score = float(sum(ref2) / max(1, len(ref2))) + float(cfg.sparsity_penalty) * len(term_sel)

        if best is None or score < best_score:
            best_score = score
            best = VectorDESearchResult(
                order=int(order),
                x_axis=int(cfg.x_axis),
                out_idxs=out_idxs,
                term_vecs=term_sel,
                coeffs=csel.detach().cpu(),
                rms_train=rms_tr,
                rms_val=rms_va,
            )

    if best is None:
        raise RuntimeError("Vector DE discovery failed to produce any candidate")

    best.residual_asts = build_vector_residual_asts(best)
    return best


def build_vector_residual_asts(result: VectorDESearchResult, *, coeff_prefix: str = "c") -> List[Node]:
    """Build tied-coefficient residual ASTs per component."""

    residuals: List[Node] = []
    x_axis = int(result.x_axis)
    dim = len(result.out_idxs)

    for ci, oi in enumerate(result.out_idxs):
        oi = int(oi)
        if int(result.order) == 0:
            root: Node = U(out_idx=oi)
        elif int(result.order) == 1:
            root = DU(x_axis, out_idx=oi)
        elif int(result.order) == 2:
            root = D2U(x_axis, x_axis, out_idx=oi)
        else:
            raise ValueError(f"Unsupported anchor order: {result.order}")

        for k, (tvec, c) in enumerate(zip(result.term_vecs, result.coeffs.tolist())):
            name = f"{coeff_prefix}{k}"
            if tvec is None:
                root = Add(root, FreeConst(name, init=float(c)))
            else:
                root = Add(root, Mul(FreeConst(name, init=float(c)), tvec[ci]))

        residuals.append(root)

    if len(residuals) != dim:
        raise RuntimeError("Internal error: residual count mismatch")
    return residuals


# ──────────────────────────────────────────────────────────────
# Multiple vector equations discovery (system of vector equations)
# ──────────────────────────────────────────────────────────────


@dataclass
class VectorEquationSpec:
    """Specification for one vector equation in a coupled system search."""

    out_idxs: Tuple[int, ...]
    name: str | None = None


@dataclass
class CoeffShareGroup:
    """A coefficient-sharing constraint across equations.

    Each member identifies a coefficient by ``(eq_index, term_index)`` where:

    * ``eq_index`` indexes the ``equations`` passed to the solver (0..Q-1)
    * ``term_index`` indexes the **full** library column ordering used in regression:
        - if ``include_const=True``, ``term_index=0`` is the constant column
        - remaining term indices follow the order of ``terms_v`` after deduplication

    The constraint enforces:

        coeff(eq, term) = scale * g[name]

    where ``g[name]`` is a single shared scalar parameter.

    Notes
    -----
    This is enforced at regression time (STLSQ), and also reflected in the
    generated residual ASTs by reusing the same FreeConst name.
    """

    name: str
    members: Tuple[Tuple[int, int], ...]
    scales: Optional[Tuple[float, ...]] = None


def share_coeff_by_term(
    name: str,
    term_idx: int,
    eq_idxs: Sequence[int],
    *,
    scales: Optional[Sequence[float]] = None,
) -> CoeffShareGroup:
    """Convenience helper: share one term's coefficient across multiple equations."""

    eqs = tuple(int(q) for q in eq_idxs)
    mem = tuple((q, int(term_idx)) for q in eqs)
    sc = None
    if scales is not None:
        sc = tuple(float(s) for s in scales)
        if len(sc) != len(mem):
            raise ValueError("scales length must match eq_idxs length")
    return CoeffShareGroup(name=str(name), members=mem, scales=sc)


@dataclass
class VectorSystemDESearchConfig:
    """Configuration for simultaneous discovery of multiple vector equations.

    This generalizes :class:`VectorDESearchConfig` from a single vector equation
    to a *set* of vector equations, each with its own coefficient vector, but
    with optional shared sparsity support across equations.

    Notes
    -----
    * All equations in a single call must share the same vector dimension
      (i.e. the same length of ``out_idxs``).
    * Candidate terms are provided as vector terms (sequence of scalar AST
      nodes, one per component). These are shared across all equations.
    """

    x_axis: int = 0
    order_candidates: Tuple[int, ...] = (1,)

    include_const: bool = False

    ridge: float = 1e-10
    stlsq_lambda: float = 1e-3
    stlsq_max_iter: int = 10

    max_batches: int = 32
    max_points: int = 20000

    sparsity_penalty: float = 1e-3

    # If True, enforce a shared support (active term set) across all vector equations.
    share_support_across_equations: bool = False

    # Optional coefficient-sharing constraints across equations.
    coeff_share_groups: Optional[Tuple[CoeffShareGroup, ...]] = None

    # Optional dimensional-analysis context for vector terms / tied coefficients.
    units_spec: Any = None
    enforce_units: bool = False


@dataclass
class VectorSystemDESearchResult:
    """Result of a simultaneous multi-vector-equation discovery."""

    order: int
    x_axis: int

    equations: Tuple[VectorEquationSpec, ...]

    # Selected vector terms; `None` denotes a constant vector term (all-ones).
    term_vecs: List[Optional[Sequence[Node]]]

    # Coefficients per equation, shape (Q, K_sel)
    coeffs: torch.Tensor

    # RMS residuals per equation per component: shape (Q, dim)
    rms_train: List[List[float]]
    rms_val: Optional[List[List[float]]] = None

    # Residual ASTs per equation per component: shape (Q, dim)
    residual_asts: Optional[List[List[Node]]] = None

    # --- Optional coefficient-sharing metadata (for cross-equation tying) ---
    # Full-term indices (0..K-1) corresponding to `term_vecs` entries.
    term_full_idxs: Optional[List[int]] = None

    # Global coefficient vector metadata when coeff_share_groups are used.
    global_names: Optional[List[str]] = None
    global_coeffs: Optional[torch.Tensor] = None  # shape (P,)
    map_gidx: Optional[torch.Tensor] = None  # shape (Q,K)
    map_scale: Optional[torch.Tensor] = None  # shape (Q,K)

    # True if the unbiased refit hit a rank-deficient (collinear) selected
    # support and was solved by the minimum-norm fallback instead of crashing.
    # The recovered coefficient split among collinear columns is then a
    # convention, not data-determined (see rank_aware_lstsq).
    rank_deficient: bool = False

    @property
    def n_equations(self) -> int:
        return len(self.equations)

    @property
    def dim(self) -> int:
        return len(self.equations[0].out_idxs) if self.equations else 0

    def format_component(self, eq: int, comp: int, tol: float = 1e-3, var_name: str = "x0") -> str:
        q = int(eq)
        c = int(comp)
        if q < 0 or q >= len(self.equations):
            raise IndexError("eq index out of range")
        if c < 0 or c >= self.dim:
            raise IndexError("component index out of range")

        eqs = self.equations[q]
        out_idx = int(eqs.out_idxs[c])

        if self.order == 0:
            lhs = f"u{out_idx}"
        elif self.order == 1:
            lhs = f"u{out_idx}_{var_name}"
        elif self.order == 2:
            lhs = f"u{out_idx}_{var_name}{var_name}"
        else:
            lhs = f"d^{self.order}u{out_idx}/d{var_name}^{self.order}"

        terms_str = []
        for ck, tvec in zip(self.coeffs[q].tolist(), self.term_vecs):
            # Snap to a few simple values for printing.
            c_snap = ck
            for target in [0.0, 0.5, 1.0, 2.0, 3.0, -0.5, -1.0, -2.0, -3.0]:
                if abs(ck - target) < tol:
                    c_snap = target
                    break
            if abs(c_snap) < 1e-12:
                continue

            if tvec is None:
                terms_str.append(f"{c_snap:g}")
                continue

            try:
                term_str = repr(tvec[c])
            except Exception:
                term_str = str(type(tvec[c]).__name__)

            if abs(abs(c_snap) - 1.0) < 1e-12:
                terms_str.append(term_str if c_snap > 0 else f"-{term_str}")
            else:
                terms_str.append(f"{c_snap:g}*{term_str}")

        rhs = " + ".join(terms_str).replace("+ -", "- ")
        if rhs.strip() == "":
            rhs = "0"

        prefix = f"[{eqs.name}] " if (eqs.name is not None and str(eqs.name).strip() != "") else ""
        return f"{prefix}{lhs} + {rhs} = 0"

    def format_equation(self, eq: int, tol: float = 1e-3, var_name: str = "x0") -> str:
        q = int(eq)
        if q < 0 or q >= len(self.equations):
            raise IndexError("eq index out of range")
        lines = [self.format_component(q, c, tol=tol, var_name=var_name) for c in range(self.dim)]
        return "\n".join(lines)

    def format_system(self, tol: float = 1e-3, var_name: str = "x0") -> str:
        blocks = [self.format_equation(q, tol=tol, var_name=var_name) for q in range(self.n_equations)]
        return "\n\n".join(blocks)


def _sanitize_ident(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    ident = "".join(out).strip("_")
    return ident if ident != "" else "eq"


def _vector_output_dims(units_spec: Any, *, Ny: int) -> tuple[Any, ...] | None:
    """Resolve per-output physical dimensions for a vector surrogate."""
    if units_spec is None:
        return None
    out_dims = getattr(units_spec, "output_dims", None)
    if out_dims is None:
        return tuple(tuple(units_spec.y_dim) for _ in range(int(Ny)))
    if len(out_dims) != int(Ny):
        raise ValueError(f"UnitsSpec.output_dims has len={len(out_dims)} but surrogate Ny={Ny}")
    return tuple(tuple(d) for d in out_dims)


def _local_units_spec_for_output(units_spec: Any, output_dims: Sequence[Any], out_idx: int):
    return replace(units_spec, y_dim=tuple(output_dims[int(out_idx)]))


def _vector_term_units_for_equation(
    term_vec: Optional[Sequence[Node]],
    *,
    equation: VectorEquationSpec,
    order: int,
    x_axis: int,
    units_spec: Any,
    output_dims: Sequence[Any],
    enforce_units: bool,
) -> tuple[bool, Any, str]:
    """Check one tied vector term against all components of one equation."""
    if units_spec is None or not bool(enforce_units):
        return True, None, ""

    us = units_spec.unit_system
    req_dims: list[Any] = []
    detail_rows: list[tuple[int, int, Any]] = []

    for ci, oi in enumerate(tuple(int(i) for i in equation.out_idxs)):
        local_spec = _local_units_spec_for_output(units_spec, output_dims, oi)
        scalar_term = None if term_vec is None else term_vec[ci]
        ok_term, why_term = term_units_feasible(
            scalar_term,
            order=int(order),
            x_axis=int(x_axis),
            units_spec=local_spec,
            enforce_units=True,
        )
        if not ok_term:
            return False, None, f"component {ci} (out_idx={oi}) is inadmissible: {why_term}"
        try:
            req_dim = required_coeff_dim_for_term(
                scalar_term,
                order=int(order),
                x_axis=int(x_axis),
                units_spec=local_spec,
            )
        except Exception as exc:
            return False, None, f"component {ci} (out_idx={oi}) units inference failed: {exc}"
        req_dims.append(tuple(req_dim))
        detail_rows.append((ci, oi, tuple(req_dim)))

    if req_dims:
        ref_dim = tuple(req_dims[0])
        if any(tuple(d) != ref_dim for d in req_dims[1:]):
            detail = ", ".join(
                f"comp {ci} (out_idx={oi}) -> {us.format_dim(dim)}"
                for ci, oi, dim in detail_rows
            )
            return (
                False,
                None,
                f"tied coefficient requires inconsistent dimensions across components: {detail}",
            )
        return True, ref_dim, ""

    return True, None, ""


def _vector_term_units_tables(
    *,
    equations: Sequence[VectorEquationSpec],
    term_vecs_all: Sequence[Optional[Sequence[Node]]],
    order: int,
    x_axis: int,
    units_spec: Any,
    output_dims: Sequence[Any] | None,
    enforce_units: bool,
) -> tuple[list[list[bool]] | None, list[list[Any]] | None, list[list[str]] | None]:
    """Per-equation admissibility / required-dim tables for vector terms."""
    if units_spec is None or output_dims is None or not bool(enforce_units):
        return None, None, None

    Q = len(equations)
    K = len(term_vecs_all)
    ok_tbl = [[True for _ in range(K)] for _ in range(Q)]
    req_tbl: list[list[Any]] = [[None for _ in range(K)] for _ in range(Q)]
    why_tbl = [["" for _ in range(K)] for _ in range(Q)]

    for q, eq in enumerate(equations):
        for k, term_vec in enumerate(term_vecs_all):
            ok_k, req_k, why_k = _vector_term_units_for_equation(
                term_vec,
                equation=eq,
                order=int(order),
                x_axis=int(x_axis),
                units_spec=units_spec,
                output_dims=output_dims,
                enforce_units=bool(enforce_units),
            )
            ok_tbl[q][k] = bool(ok_k)
            req_tbl[q][k] = req_k
            why_tbl[q][k] = str(why_k or "")

    return ok_tbl, req_tbl, why_tbl


def _validate_coeff_share_groups_units(
    *,
    groups: Sequence[CoeffShareGroup],
    term_ok_tbl: list[list[bool]] | None,
    term_req_tbl: list[list[Any]] | None,
    term_why_tbl: list[list[str]] | None,
    units_spec: Any,
) -> None:
    """Reject shared coefficient groups whose members imply different units."""
    if not groups or units_spec is None or term_ok_tbl is None or term_req_tbl is None:
        return

    us = units_spec.unit_system
    Q = len(term_ok_tbl)
    K = len(term_ok_tbl[0]) if Q > 0 else 0

    for grp in groups:
        req_rows: list[tuple[int, int, Any]] = []
        for q, k in tuple(grp.members):
            q = int(q)
            k = int(k)
            if q < 0 or q >= Q or k < 0 or k >= K:
                continue
            if not bool(term_ok_tbl[q][k]):
                why = ""
                if term_why_tbl is not None:
                    why = str(term_why_tbl[q][k] or "")
                raise ValueError(
                    f"Coeff share group {grp.name!r} is dimensionally inadmissible at "
                    f"(eq={q}, term={k}): {why or 'term is not unit-compatible'}"
                )
            req_dim = term_req_tbl[q][k]
            if req_dim is None:
                raise ValueError(
                    f"Coeff share group {grp.name!r} has no resolved coefficient dimension "
                    f"for member (eq={q}, term={k})"
                )
            req_rows.append((q, k, tuple(req_dim)))

        if len(req_rows) <= 1:
            continue

        ref_dim = tuple(req_rows[0][2])
        if any(tuple(dim) != ref_dim for _, _, dim in req_rows[1:]):
            detail = "; ".join(
                f"(eq={q}, term={k}) -> {us.format_dim(dim)}"
                for q, k, dim in req_rows
            )
            raise ValueError(
                f"Coeff share group {grp.name!r} ties incompatible coefficient dimensions: {detail}"
            )


def _build_coeff_share_mapping(
    *,
    equations: Tuple[VectorEquationSpec, ...],
    K: int,
    groups: Sequence[CoeffShareGroup],
    coeff_prefix: str = "c",
    device=None,
    dtype: torch.dtype | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Build a coefficient-sharing mapping for a multi-vector system.

    The solver's native coefficients are ``c[q, k]`` where:

    * ``q`` indexes vector equations (0..Q-1)
    * ``k`` indexes library columns (0..K-1) in the **full** regression library

    This routine constructs a mapping to a global coefficient vector ``g[j]`` such that:

        c[q, k] = scale[q, k] * g[gidx[q, k]]

    Coefficients in the same :class:`CoeffShareGroup` share the same global index.
    All remaining coefficients get unique global indices.

    Returns
    -------
    gidx : LongTensor, shape (Q, K)
        Global coefficient index per (equation, term).
    scale : Tensor, shape (Q, K)
        Scaling factor applied to the corresponding global coefficient.
    names : list[str], length P
        Parameter name for each global coefficient index.
    """

    Q = len(equations)
    if Q < 1:
        raise ValueError("Need at least one equation")
    if int(K) < 1:
        raise ValueError("K must be >= 1")

    dev = device
    dt = dtype or torch.float64

    gidx = torch.full((Q, int(K)), -1, dtype=torch.long, device=dev)
    scale = torch.ones((Q, int(K)), dtype=dt, device=dev)

    names: List[str] = []
    used_names = set()
    used_members = set()

    # Allocate group parameters first.
    for grp in groups:
        gname = _sanitize_ident(grp.name)
        if gname in used_names:
            raise ValueError(f"Duplicate coeff-share group name: {gname!r}")
        used_names.add(gname)

        gid = len(names)
        names.append(gname)

        mem = tuple(grp.members)
        if len(mem) == 0:
            continue

        if grp.scales is None:
            scales = [1.0] * len(mem)
        else:
            if len(grp.scales) != len(mem):
                raise ValueError(f"Group {grp.name!r}: scales length must match members length")
            scales = [float(s) for s in grp.scales]

        for (q, k), s in zip(mem, scales):
            q = int(q)
            k = int(k)
            if q < 0 or q >= Q:
                raise ValueError(f"Group {grp.name!r}: eq_index {q} out of range (Q={Q})")
            if k < 0 or k >= int(K):
                raise ValueError(f"Group {grp.name!r}: term_index {k} out of range (K={K})")
            if (q, k) in used_members:
                raise ValueError(
                    f"Coefficient (eq={q}, term={k}) is assigned to multiple coeff-share groups"
                )
            used_members.add((q, k))
            gidx[q, k] = gid
            scale[q, k] = float(s)

    # Allocate remaining coefficients as unique parameters.
    for q in range(Q):
        eqname = _sanitize_ident(equations[q].name) if equations[q].name is not None else str(q)
        for k in range(int(K)):
            if int(gidx[q, k].item()) >= 0:
                continue
            gid = len(names)
            names.append(f"{coeff_prefix}{eqname}_{k}")
            gidx[q, k] = gid

    return gidx, scale, names


def discover_vector_system_de_from_surrogate(
    surrogate,
    train_dataloader,
    val_dataloader=None,
    *,
    cfg: Optional[VectorSystemDESearchConfig] = None,
    equations: Sequence[VectorEquationSpec],
    device=None,
    dataset=None,
    vector_terms: Optional[Sequence[Sequence[Node]]] = None,
    extra_vector_terms: Optional[Sequence[Sequence[Node]]] = None,
    library_vector_terms: Optional[Sequence[Sequence[Node]]] = None,
) -> VectorSystemDESearchResult:
    """Discover a coupled **system** of multiple vector equations.

    Parameters
    ----------
    equations : sequence[VectorEquationSpec]
        The vector equations to discover. Each equation is defined by the
        surrogate output indices of its components.
    vector_terms : sequence[sequence[Node]]
        Shared candidate vector terms. Each term is a sequence of scalar AST
        nodes with length equal to the vector dimension.

    Notes
    -----
    Coefficients are tied across vector components *within* each equation.
    Optionally, a shared support can be enforced across equations.
    """

    if cfg is None:
        cfg = VectorSystemDESearchConfig()

    if equations is None or len(equations) == 0:
        raise ValueError("equations must be a non-empty sequence")

    eqs = tuple(equations)
    dim = len(eqs[0].out_idxs)
    if dim < 1:
        raise ValueError("equation out_idxs must be non-empty")
    for e in eqs:
        if len(e.out_idxs) != dim:
            raise ValueError("All vector equations must share the same dimension")

    # Optionally auto-detect x_axis from dataset metadata.
    if dataset is not None:
        try:
            if hasattr(dataset, "has_coord_metadata") and dataset.has_coord_metadata():
                time_coords = dataset.get_time_coords()
                if time_coords and len(time_coords) > 0:
                    detected = int(time_coords[0])
                    if cfg.x_axis == 0 or cfg.x_axis is None:
                        cfg = VectorSystemDESearchConfig(
                            x_axis=detected,
                            order_candidates=cfg.order_candidates,
                            include_const=cfg.include_const,
                            ridge=cfg.ridge,
                            stlsq_lambda=cfg.stlsq_lambda,
                            stlsq_max_iter=cfg.stlsq_max_iter,
                            max_batches=cfg.max_batches,
                            max_points=cfg.max_points,
                            sparsity_penalty=cfg.sparsity_penalty,
                            share_support_across_equations=cfg.share_support_across_equations,
                            coeff_share_groups=cfg.coeff_share_groups,
                            units_spec=cfg.units_spec,
                            enforce_units=cfg.enforce_units,
                        )
        except Exception:
            pass

    dev = device
    if dev is None:
        try:
            dev = next(surrogate.parameters()).device
        except Exception:
            dev = torch.device("cpu")

    Xtr = _gather_x(train_dataloader, max_batches=int(cfg.max_batches), max_points=int(cfg.max_points), device=dev)
    Xva = None
    if val_dataloader is not None:
        Xva = _gather_x(val_dataloader, max_batches=int(cfg.max_batches), max_points=int(cfg.max_points), device=dev)

    cache = UFeatureCache(surrogate)
    cache.ensure(Xtr, need_grad=False, need_hess=False)
    if cache.u is None:
        raise RuntimeError("Failed to evaluate surrogate output")
    Ny = int(cache.u.shape[1]) if cache.u.ndim == 2 else 1

    # Validate out_idxs.
    for e in eqs:
        for oi in e.out_idxs:
            i = int(oi)
            if i < 0 or i >= Ny:
                raise ValueError(f"out_idx {i} out of range (Ny={Ny})")

    # Choose the term set.
    if library_vector_terms is not None:
        terms_v = list(library_vector_terms)
    else:
        if vector_terms is None:
            raise ValueError("vector_terms must be provided (or pass library_vector_terms)")
        terms_v = list(vector_terms)

    if extra_vector_terms is not None:
        terms_v.extend(list(extra_vector_terms))

    # Validate term dimensions.
    for t in terms_v:
        if len(t) != dim:
            raise ValueError(f"Vector term has len={len(t)} but system dim={dim}")

    terms_v = _dedup_vector_terms(terms_v)
    term_vecs_all: List[Optional[Sequence[Node]]] = []
    if bool(cfg.include_const):
        term_vecs_all.append(None)
    term_vecs_all.extend(terms_v)
    output_dims = _vector_output_dims(getattr(cfg, "units_spec", None), Ny=Ny)

    Q = len(eqs)
    K_full = len(term_vecs_all)

    # Optional coefficient sharing across equations (ties some coefficients to a shared parameter).
    use_coeff_share = cfg.coeff_share_groups is not None and len(tuple(cfg.coeff_share_groups)) > 0
    gidx_map = None
    gscale_map = None
    global_names = None
    P_global = 0
    if use_coeff_share:
        gidx_map, gscale_map, global_names = _build_coeff_share_mapping(
            equations=eqs,
            K=int(K_full),
            groups=tuple(cfg.coeff_share_groups or ()),
            coeff_prefix="c",
            device=dev,
            dtype=Xtr.dtype,
        )
        P_global = len(global_names)

    best: Optional[VectorSystemDESearchResult] = None
    best_score = float("inf")

    for order in tuple(int(o) for o in cfg.order_candidates):
        if order not in (0, 1, 2):
            continue

        term_ok_tbl, term_req_tbl, term_why_tbl = _vector_term_units_tables(
            equations=eqs,
            term_vecs_all=term_vecs_all,
            order=int(order),
            x_axis=int(cfg.x_axis),
            units_spec=getattr(cfg, "units_spec", None),
            output_dims=output_dims,
            enforce_units=bool(getattr(cfg, "enforce_units", False)),
        )
        if term_ok_tbl is not None and not any(any(bool(v) for v in row) for row in term_ok_tbl):
            continue
        if use_coeff_share and term_ok_tbl is not None:
            _validate_coeff_share_groups_units(
                groups=tuple(cfg.coeff_share_groups or ()),
                term_ok_tbl=term_ok_tbl,
                term_req_tbl=term_req_tbl,
                term_why_tbl=term_why_tbl,
                units_spec=getattr(cfg, "units_spec", None),
            )

        cache.reset()
        if order == 0:
            cache.ensure(Xtr, need_grad=False, need_hess=False)
            if cache.u is None:
                raise RuntimeError("Surrogate output unavailable")
            anchor_full = cache.u if cache.u.ndim == 2 else cache.u.unsqueeze(1)  # (N,Ny)
        elif order == 1:
            cache.ensure(Xtr, need_grad=True, need_hess=False)
            if cache.g is None:
                raise RuntimeError("Surrogate grad unavailable")
            anchor_full = cache.g[:, :, int(cfg.x_axis)]  # (N,Ny)
        else:
            cache.ensure(Xtr, need_grad=False, need_hess=True)
            if cache.H is None:
                raise RuntimeError("Surrogate grad_grad unavailable")
            anchor_full = cache.H[:, :, int(cfg.x_axis), int(cfg.x_axis)]  # (N,Ny)

        # Build Phi/y per equation (stacking components enforces coefficient tying).
        Phis_eq: List[torch.Tensor] = []
        ys_eq: List[torch.Tensor] = []
        Phis_comp: List[List[torch.Tensor]] = []
        ys_comp: List[List[torch.Tensor]] = []

        for e in eqs:
            phis_c: List[torch.Tensor] = []
            ys_c: List[torch.Tensor] = []
            for ci, oi in enumerate(tuple(int(i) for i in e.out_idxs)):
                cols = []
                q_idx = len(Phis_eq)
                if bool(cfg.include_const):
                    if term_ok_tbl is None or bool(term_ok_tbl[q_idx][0]):
                        cols.append(torch.ones(Xtr.shape[0], device=dev, dtype=Xtr.dtype))
                    else:
                        cols.append(torch.zeros(Xtr.shape[0], device=dev, dtype=Xtr.dtype))
                for tk, t in enumerate(terms_v):
                    full_k = (1 if bool(cfg.include_const) else 0) + tk
                    if term_ok_tbl is not None and not bool(term_ok_tbl[q_idx][full_k]):
                        cols.append(torch.zeros(Xtr.shape[0], device=dev, dtype=Xtr.dtype))
                    else:
                        cols.append(_as_N(_eval_ast(t[ci], Xtr, cache)))
                Phi_i = torch.stack(cols, dim=1) if cols else Xtr.new_zeros(Xtr.shape[0], 0)

                y_i = -anchor_full[:, int(oi)]
                m = torch.isfinite(y_i)
                m &= torch.isfinite(Phi_i).all(dim=1)
                if int(m.sum()) < 10:
                    raise RuntimeError(f"Too few finite rows for out_idx={oi} (order={order})")

                phis_c.append(Phi_i[m])
                ys_c.append(y_i[m])

            Phi_stack = torch.cat(phis_c, dim=0)
            y_stack = torch.cat(ys_c, dim=0)

            Phis_eq.append(Phi_stack)
            ys_eq.append(y_stack)
            Phis_comp.append(phis_c)
            ys_comp.append(ys_c)

        # Solve for coefficients per equation.
        if use_coeff_share:
            # Global constrained solve using shared coefficient parameters g.
            if gidx_map is None or gscale_map is None or global_names is None:
                raise RuntimeError("Internal error: coeff-share mapping missing")

            Phi_glob_blocks: List[torch.Tensor] = []
            y_glob_blocks: List[torch.Tensor] = []

            for q in range(Q):
                Phi_q = Phis_eq[q]  # (n_q, K_full)
                idx_q = gidx_map[q].to(device=Phi_q.device)
                sc_q = gscale_map[q].to(device=Phi_q.device, dtype=Phi_q.dtype)

                n_q = int(Phi_q.shape[0])
                Phi_qg = Phi_q.new_zeros((n_q, int(P_global)))
                Phi_qg.scatter_add_(
                    1,
                    idx_q.view(1, -1).expand(n_q, -1),
                    Phi_q * sc_q.view(1, -1),
                )

                Phi_glob_blocks.append(Phi_qg)
                y_glob_blocks.append(ys_eq[q])

            Phi_glob = torch.cat(Phi_glob_blocks, dim=0)
            y_glob = torch.cat(y_glob_blocks, dim=0)

            g, keep_g = stlsq(
                Phi_glob,
                y_glob,
                ridge=float(cfg.ridge),
                lam=float(cfg.stlsq_lambda),
                max_iter=int(cfg.stlsq_max_iter),
            )

            Ksel_g = int(keep_g.sum())
            if Ksel_g == 0:
                continue

            # Unbiased refit on selected *global* coefficients (rank-safe).
            gsel, ginfo = rank_aware_lstsq(Phi_glob[:, keep_g], y_glob, ridge=0.0)
            rank_deficient = bool(ginfo["rank_deficient"])
            g_full = torch.zeros((int(P_global),), device=dev, dtype=Phi_glob.dtype)
            g_full[keep_g] = gsel

            # Keep a term if any equation uses it.
            term_keep = torch.zeros((int(K_full),), device=dev, dtype=torch.bool)
            for q in range(Q):
                term_keep |= keep_g[gidx_map[q]]

            Ksel = int(term_keep.sum())
            if Ksel == 0:
                continue

            # Build per-equation coefficient vectors for the selected *terms*.
            Csel = torch.zeros((Q, Ksel), device=dev, dtype=Phi_glob.dtype)
            for q in range(Q):
                c_q_full = g_full[gidx_map[q]] * gscale_map[q].to(device=dev, dtype=Phi_glob.dtype)
                Csel[q] = c_q_full[term_keep]

            # Selected vector terms (with optional const sentinel).
            term_sel = [t for t, k in zip(term_vecs_all, term_keep.tolist()) if k]

            term_full_idxs = (
                torch.nonzero(term_keep, as_tuple=False).view(-1).detach().cpu().tolist()
            )

            # Training RMS per equation per component.
            rms_tr: List[List[float]] = []
            for q in range(Q):
                eq_rms = []
                for ci in range(dim):
                    r = (-ys_comp[q][ci]) + Phis_comp[q][ci][:, term_keep] @ Csel[q]
                    eq_rms.append(float((r.square().mean().sqrt()).detach().cpu()))
                rms_tr.append(eq_rms)

            # Validation RMS.
            rms_va = None
            if Xva is not None:
                cache.reset()
                if order == 0:
                    cache.ensure(Xva, need_grad=False, need_hess=False)
                    anchor_va_full = cache.u if cache.u.ndim == 2 else cache.u.unsqueeze(1)
                elif order == 1:
                    cache.ensure(Xva, need_grad=True, need_hess=False)
                    anchor_va_full = cache.g[:, :, int(cfg.x_axis)]
                else:
                    cache.ensure(Xva, need_grad=False, need_hess=True)
                    anchor_va_full = cache.H[:, :, int(cfg.x_axis), int(cfg.x_axis)]

                rms_va = []
                for q, e in enumerate(eqs):
                    eq_rms = []
                    for ci, oi in enumerate(tuple(int(i) for i in e.out_idxs)):
                        cols = []
                        if bool(cfg.include_const):
                            if term_ok_tbl is None or bool(term_ok_tbl[q][0]):
                                cols.append(torch.ones(Xva.shape[0], device=dev, dtype=Xva.dtype))
                            else:
                                cols.append(torch.zeros(Xva.shape[0], device=dev, dtype=Xva.dtype))
                        for tk, t in enumerate(terms_v):
                            full_k = (1 if bool(cfg.include_const) else 0) + tk
                            if term_ok_tbl is not None and not bool(term_ok_tbl[q][full_k]):
                                cols.append(torch.zeros(Xva.shape[0], device=dev, dtype=Xva.dtype))
                            else:
                                cols.append(_as_N(_eval_ast(t[ci], Xva, cache)))
                        Phi_va_i = torch.stack(cols, dim=1) if cols else Xva.new_zeros(Xva.shape[0], 0)

                        yva = -anchor_va_full[:, int(oi)]
                        m = torch.isfinite(yva)
                        m &= torch.isfinite(Phi_va_i[:, term_keep]).all(dim=1)
                        if int(m.sum()) < 10:
                            eq_rms.append(float("nan"))
                            continue
                        rva = (-yva[m]) + Phi_va_i[m][:, term_keep] @ Csel[q]
                        eq_rms.append(float((rva.square().mean().sqrt()).detach().cpu()))
                    rms_va.append(eq_rms)

            # Score (mean RMS across all eqs/components + sparsity penalty per *global* coefficient).
            ref = rms_va if rms_va is not None else rms_tr
            flat = []
            for q in range(Q):
                for ci in range(dim):
                    v = float(ref[q][ci])
                    flat.append(v if (v == v) else 1e9)
            score = float(sum(flat) / max(1, len(flat))) + float(cfg.sparsity_penalty) * float(Ksel_g)

            if best is None or score < best_score:
                best_score = score
                best = VectorSystemDESearchResult(
                    order=int(order),
                    x_axis=int(cfg.x_axis),
                    equations=eqs,
                    term_vecs=term_sel,
                    coeffs=Csel.detach().cpu(),
                    rms_train=rms_tr,
                    rms_val=rms_va,
                    term_full_idxs=[int(i) for i in term_full_idxs],
                    global_names=list(global_names),
                    global_coeffs=g_full.detach().cpu(),
                    map_gidx=gidx_map.detach().cpu(),
                    map_scale=gscale_map.detach().cpu(),
                    rank_deficient=bool(rank_deficient),
                )

        else:
            # Optional shared support (active term set) across equations.
            if bool(cfg.share_support_across_equations):
                C, keep = group_stlsq(
                    Phis_eq,
                    ys_eq,
                    ridge=float(cfg.ridge),
                    lam=float(cfg.stlsq_lambda),
                    max_iter=int(cfg.stlsq_max_iter),
                )
            else:
                Cs = []
                keeps = []
                for q in range(Q):
                    c_q, keep_q = stlsq(
                        Phis_eq[q],
                        ys_eq[q],
                        ridge=float(cfg.ridge),
                        lam=float(cfg.stlsq_lambda),
                        max_iter=int(cfg.stlsq_max_iter),
                    )
                    Cs.append(c_q)
                    keeps.append(keep_q)
                keep = torch.stack(keeps, dim=0).any(dim=0)
                torch.stack(Cs, dim=0)

            Ksel = int(keep.sum())
            if Ksel == 0:
                continue

            # Unbiased refit on the selected terms (rank-safe: a collinear
            # support is solved by the min-norm fallback and flagged rather than
            # raising a singular-matrix error).
            Csel = torch.zeros((Q, Ksel), device=dev, dtype=Phis_eq[0].dtype)
            rank_deficient = False
            for q in range(Q):
                c_q, info_q = rank_aware_lstsq(Phis_eq[q][:, keep], ys_eq[q], ridge=0.0)
                Csel[q] = c_q
                rank_deficient = rank_deficient or bool(info_q["rank_deficient"])

            # Selected vector terms (with optional const sentinel).
            term_sel = [t for t, k in zip(term_vecs_all, keep.tolist()) if k]

            # Training RMS per equation per component.
            rms_tr: List[List[float]] = []
            for q in range(Q):
                eq_rms = []
                for ci in range(dim):
                    r = (-ys_comp[q][ci]) + Phis_comp[q][ci][:, keep] @ Csel[q]
                    eq_rms.append(float((r.square().mean().sqrt()).detach().cpu()))
                rms_tr.append(eq_rms)

            # Validation RMS.
            rms_va = None
            if Xva is not None:
                cache.reset()
                if order == 0:
                    cache.ensure(Xva, need_grad=False, need_hess=False)
                    anchor_va_full = cache.u if cache.u.ndim == 2 else cache.u.unsqueeze(1)
                elif order == 1:
                    cache.ensure(Xva, need_grad=True, need_hess=False)
                    anchor_va_full = cache.g[:, :, int(cfg.x_axis)]
                else:
                    cache.ensure(Xva, need_grad=False, need_hess=True)
                    anchor_va_full = cache.H[:, :, int(cfg.x_axis), int(cfg.x_axis)]

                rms_va = []
                for q, e in enumerate(eqs):
                    eq_rms = []
                    for ci, oi in enumerate(tuple(int(i) for i in e.out_idxs)):
                        cols = []
                        if bool(cfg.include_const):
                            if term_ok_tbl is None or bool(term_ok_tbl[q][0]):
                                cols.append(torch.ones(Xva.shape[0], device=dev, dtype=Xva.dtype))
                            else:
                                cols.append(torch.zeros(Xva.shape[0], device=dev, dtype=Xva.dtype))
                        for tk, t in enumerate(terms_v):
                            full_k = (1 if bool(cfg.include_const) else 0) + tk
                            if term_ok_tbl is not None and not bool(term_ok_tbl[q][full_k]):
                                cols.append(torch.zeros(Xva.shape[0], device=dev, dtype=Xva.dtype))
                            else:
                                cols.append(_as_N(_eval_ast(t[ci], Xva, cache)))
                        Phi_va_i = torch.stack(cols, dim=1) if cols else Xva.new_zeros(Xva.shape[0], 0)

                        yva = -anchor_va_full[:, int(oi)]
                        m = torch.isfinite(yva)
                        m &= torch.isfinite(Phi_va_i[:, keep]).all(dim=1)
                        if int(m.sum()) < 10:
                            eq_rms.append(float("nan"))
                            continue
                        rva = (-yva[m]) + Phi_va_i[m][:, keep] @ Csel[q]
                        eq_rms.append(float((rva.square().mean().sqrt()).detach().cpu()))
                    rms_va.append(eq_rms)

            # Score (mean RMS across all eqs/components + sparsity penalty per-coefficient vector).
            ref = rms_va if rms_va is not None else rms_tr
            flat = []
            for q in range(Q):
                for ci in range(dim):
                    v = float(ref[q][ci])
                    flat.append(v if (v == v) else 1e9)
            score = float(sum(flat) / max(1, len(flat))) + float(cfg.sparsity_penalty) * float(len(term_sel) * Q)

            if best is None or score < best_score:
                best_score = score
                best = VectorSystemDESearchResult(
                    order=int(order),
                    x_axis=int(cfg.x_axis),
                    equations=eqs,
                    term_vecs=term_sel,
                    coeffs=Csel.detach().cpu(),
                    rms_train=rms_tr,
                    rms_val=rms_va,
                    rank_deficient=bool(rank_deficient),
                )


    if best is None:
        raise RuntimeError("Vector system DE discovery failed to produce any candidate")

    best.residual_asts = build_vector_system_residual_asts(best)
    return best


def build_vector_system_residual_asts(
    result: VectorSystemDESearchResult, *, coeff_prefix: str = "c"
) -> List[List[Node]]:
    """Build residual ASTs per equation per component."""

    residuals: List[List[Node]] = []
    x_axis = int(result.x_axis)
    dim = result.dim

    use_map = (
        result.global_names is not None
        and result.global_coeffs is not None
        and result.map_gidx is not None
        and result.map_scale is not None
        and result.term_full_idxs is not None
    )

    if use_map:
        g_names = result.global_names  # type: ignore
        g_vals = result.global_coeffs  # type: ignore
        gidx = result.map_gidx  # type: ignore
        gscale = result.map_scale  # type: ignore
        term_full_idxs = result.term_full_idxs  # type: ignore

    for q, eq in enumerate(result.equations):
        eq_name = _sanitize_ident(eq.name) if eq.name is not None else str(q)
        eq_res: List[Node] = []

        for ci, oi in enumerate(eq.out_idxs):
            oi = int(oi)
            if int(result.order) == 0:
                root: Node = U(out_idx=oi)
            elif int(result.order) == 1:
                root = DU(x_axis, out_idx=oi)
            elif int(result.order) == 2:
                root = D2U(x_axis, x_axis, out_idx=oi)
            else:
                raise ValueError(f"Unsupported anchor order: {result.order}")

            coeff_row = result.coeffs[q].tolist()
            for k, (tvec, c) in enumerate(zip(result.term_vecs, coeff_row)):
                if use_map:
                    full_k = int(term_full_idxs[k])
                    gid = int(gidx[q, full_k].item())
                    pname = str(g_names[gid])
                    init = float(g_vals[gid].item()) if hasattr(g_vals, "numel") else float(g_vals[gid])
                    sc = float(gscale[q, full_k].item())

                    coef = FreeConst(pname, init=init)
                    if abs(sc - 1.0) > 1e-12:
                        coef = Mul(ConstNode(float(sc)), coef)

                    if tvec is None:
                        root = Add(root, coef)
                    else:
                        root = Add(root, Mul(coef, tvec[ci]))
                else:
                    name = f"{coeff_prefix}{eq_name}_{k}"
                    if tvec is None:
                        root = Add(root, FreeConst(name, init=float(c)))
                    else:
                        root = Add(root, Mul(FreeConst(name, init=float(c)), tvec[ci]))

            eq_res.append(root)

        if len(eq_res) != dim:
            raise RuntimeError("Internal error: vector residual component mismatch")

        residuals.append(eq_res)

    return residuals
