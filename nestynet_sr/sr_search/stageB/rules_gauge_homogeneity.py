# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Gauge, homogeneity, prefactor, and ratio Stage-B rewrite rules."""

from __future__ import annotations

import copy
import math
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from nestynet_sr.sr_core.bridges import (
    AcosNode,
    AddNode,
    AsinNode,
    AtanNode,
    AtomNode,
    ConstNode,
    CosNode,
    ExpNode,
    LogNode,
    MulNode,
    Node,
    PowNode,
    SinNode,
    Var,
    _collect_var_idxs_from_node,
    ast_to_human_readable,
    clone_ast,
    effective_arity,
    get_input_exprs,
    has_nontrivial_input,
    is_trivial_input,
    replace_atom_in_ast,
)
from nestynet_sr.sr_core.constants import (
    build_scalar_atom_from_variant as _build_scalar_atom_from_variant,
    scalar_constant_variants as _scalar_constant_variants,
)
from nestynet_sr.sr_core.separability_math import (
    build_monomial_ast,
    check_coupled_leaf_ratio_from_derivs,
)
from nestynet_sr.sr_search.candidate_builders import _build_atom_input_tensor

from .engine import Candidate, StageBContext, StageBRule, atom_content_hash, candidate_pattern_name
from .helpers import (
    _build_counterfactor_add_split_candidate,
    _build_counterterm_mul_split_candidate,
    _build_coupled_ratio_candidate,
    _build_overlap_counterterm_peel_candidates,
    _build_overlap_prefactor_peel_candidates,
    _build_product_homogeneity_candidate,
    _build_quadratic_poly_candidate,
    _build_ratio_invariance_candidate,
    _collect_all_atoms,
    _collect_multivariate_nn_atoms,
    _set_constant_leaf_value,
    build_atom_to_leaf_map,
)
from .homogeneous_gauge_scope import HomogeneousGaugeScope
from .rules_common import (
    _HomogeneousGaugeTeacher,
    _effective_input_dims_for_atom,
    _subtree_content_hash,
)
from .rules_nn_leaf import (
    _eval_subtree_with_leaf_map,
    _flatten_mul,
    _iter_add_nodes,
    _iter_mul_nodes,
    _rebuild_mul,
    _replace_node_in_ast,
    _vars_in_subtree_simple,
)
from .rules_univariate import RuleUniNN
from .transfer_basis import build_transfer_basis


_UNARY_AST_NODES = (LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode)


def _sync_stageb_rules_compat_overrides() -> None:
    """Honor legacy tests/tools that monkeypatch helpers through ``stageB.rules``."""
    rules_mod = sys.modules.get("nestynet_sr.sr_search.stageB.rules")
    if rules_mod is None:
        return
    for name in (
        "RuleUniNN",
        "_build_counterfactor_add_split_candidate",
        "_build_counterterm_mul_split_candidate",
        "_build_coupled_ratio_candidate",
        "_build_overlap_counterterm_peel_candidates",
        "_build_overlap_prefactor_peel_candidates",
        "_build_product_homogeneity_candidate",
        "_build_quadratic_poly_candidate",
        "_build_ratio_invariance_candidate",
        "_collect_multivariate_nn_atoms",
        "_scalar_constant_variants",
        "_build_scalar_atom_from_variant",
        "_set_constant_leaf_value",
        "build_atom_to_leaf_map",
        "check_coupled_leaf_ratio_from_derivs",
    ):
        if hasattr(rules_mod, name):
            globals()[name] = getattr(rules_mod, name)


def _check_ratio_invariance_on_leaf_inputs(
    X_in: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    xi_local_idx: int,
    xj_local_idx: int,
    *,
    threshold: float = 0.05,
) -> Dict[str, Any]:
    """Evaluate the ratio-invariance Euler test in leaf-local coordinates."""
    xi = X_in[:, int(xi_local_idx)]
    xj = X_in[:, int(xj_local_idx)]
    df_dxi = g[:, int(xi_local_idx)]
    df_dxj = g[:, int(xj_local_idx)]

    euler = xi * df_dxi + xj * df_dxj
    f_scale = f.abs().median().clamp_min(1e-12)
    euler_score = float((euler.abs().median() / f_scale).item())

    def _ratio_range(num: torch.Tensor, den: torch.Tensor) -> float:
        mask = torch.isfinite(num) & torch.isfinite(den) & (den.abs() > 1e-12)
        if int(mask.sum().item()) < 2:
            return float("inf")
        r = num[mask] / den[mask]
        r = r[torch.isfinite(r)]
        if int(r.numel()) < 2:
            return float("inf")
        return float((r.max() - r.min()).abs().item())

    range_xj_xi = _ratio_range(xj, xi)
    range_xi_xj = _ratio_range(xi, xj)
    ratio_direction = "xj/xi" if range_xj_xi <= range_xi_xj else "xi/xj"

    return {
        "ok": bool(euler_score < threshold),
        "xi_local_idx": int(xi_local_idx),
        "xj_local_idx": int(xj_local_idx),
        "euler_score": float(euler_score),
        "ratio_direction": ratio_direction,
    }


def _univariate_collapse_score(
    coord: torch.Tensor,
    values: torch.Tensor,
    *,
    n_bins: int = 48,
    min_points: int = 256,
) -> float:
    """Return within-bin RMS / total RMS for ``values ≈ g(coord)``."""
    try:
        coord = coord.detach().reshape(-1).to(dtype=torch.float64)
        values = values.detach().reshape(-1).to(dtype=torch.float64)
        mask = torch.isfinite(coord) & torch.isfinite(values)
        coord = coord[mask]
        values = values[mask]
        n = int(coord.numel())
        if n < int(min_points):
            return float("inf")
        order = torch.argsort(coord)
        y = values[order]
        total = y - torch.mean(y)
        total_rms = torch.sqrt(torch.mean(total * total)).clamp_min(1e-30)
        nb = int(max(4, min(int(n_bins), n // 16)))
        x_sorted = coord[order]
        x_chunks = torch.chunk(x_sorted, nb)
        y_chunks = torch.chunk(y, nb)
        residuals = []
        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            if int(y_chunk.numel()) < 4:
                continue
            xc = x_chunk - torch.mean(x_chunk)
            yc = y_chunk - torch.mean(y_chunk)
            den = torch.sum(xc * xc)
            if float(den.detach().cpu()) > 1e-30:
                slope = torch.sum(xc * yc) / den
                fitted = torch.mean(y_chunk) + slope * xc
                residuals.append(y_chunk - fitted)
            else:
                residuals.append(y_chunk - torch.mean(y_chunk))
        if not residuals:
            return float("inf")
        res = torch.cat(residuals)
        return float((torch.sqrt(torch.mean(res * res)) / total_rms).item())
    except Exception:
        return float("inf")


def _homogeneity_ratio_units_ok(
    ctx: StageBContext,
    target: AtomNode,
    *,
    power_dim: int,
    ratio_dim: int,
) -> bool:
    """Check that the proposed ratio input is dimensionless when units apply."""
    if not getattr(ctx, "enforce_units", False):
        return True
    units_spec = getattr(ctx, "units_spec", None)
    if units_spec is None:
        return True
    dims = _effective_input_dims_for_atom(target, units_spec)
    if len(dims) <= max(int(power_dim), int(ratio_dim)):
        return True
    power_d = tuple(dims[int(power_dim)])
    ratio_d = tuple(dims[int(ratio_dim)])
    if power_d == ratio_d:
        return True
    try:
        power_label = ast_to_human_readable(get_input_exprs(target)[int(power_dim)])
        ratio_label = ast_to_human_readable(get_input_exprs(target)[int(ratio_dim)])
    except Exception:
        power_label = f"input{int(power_dim)}"
        ratio_label = f"input{int(ratio_dim)}"
    ctx.log(
        "[Stage B] homogeneity_peel: skipping ratio "
        f"{ratio_label}/{power_label} due to incompatible units"
    )
    return False


def _make_homogeneity_peel_values_init_fn(
    *,
    new_tag: str,
    degree: float,
    q_values: torch.Tensor,
    numerator_values: torch.Tensor,
    f: torch.Tensor,
    ctx: StageBContext,
):
    """Warm-start the residual 1D NN with ``f / power**degree`` as teacher."""

    q = q_values.detach()
    r_num = numerator_values.detach()
    y = f.detach()
    eps = torch.as_tensor(1e-12, device=q.device, dtype=q.dtype)
    finite = torch.isfinite(q) & torch.isfinite(r_num) & torch.isfinite(y) & (q.abs() > eps)
    if not float(degree).is_integer():
        finite = finite & (q > eps)
    if int(finite.sum().item()) < 128:
        return None

    q = q[finite]
    r_num = r_num[finite]
    y = y[finite]
    try:
        power = torch.pow(q, float(degree))
        ratio = r_num / q
        teacher = y / (power + 1e-30)
    except Exception:
        return None
    mask = torch.isfinite(ratio) & torch.isfinite(teacher)
    ratio = ratio[mask].reshape(-1, 1)
    teacher = teacher[mask].reshape(-1, 1)
    if int(ratio.shape[0]) < 128:
        return None

    ratio_cpu = ratio.detach().to(device=torch.device("cpu"), dtype=torch.float64)
    teacher_cpu = teacher.detach().to(device=torch.device("cpu"), dtype=torch.float64)
    epochs = int(getattr(ctx.lm_hp, "stageB_homogeneity_pretrain_epochs", 1000))
    epochs = max(50, min(2000, epochs))

    def _init_fn(root_inner: Node, model_inner: torch.nn.Module):
        try:
            atom_to_leaf = build_atom_to_leaf_map(root_inner, model_inner)
        except Exception as e:
            print(f"[Stage B] homogeneity_peel init: atom-to-leaf map failed: {e}")
            return

        new_leaf = None
        for atom in _collect_all_atoms(root_inner):
            if isinstance(atom, AtomNode) and getattr(atom, "tag", None) == new_tag:
                new_leaf = atom_to_leaf.get(id(atom), None)
                break
        if new_leaf is None:
            print(f"[Stage B] homogeneity_peel init: missing residual leaf tag={new_tag}")
            return

        try:
            dev = next(new_leaf.parameters()).device
            dt = next(new_leaf.parameters()).dtype
        except StopIteration:
            return
        x_fit = ratio_cpu.to(device=dev, dtype=dt)
        y_fit = teacher_cpu.to(device=dev, dtype=dt)
        try:
            opt = torch.optim.Adam(new_leaf.parameters(), lr=1e-2)
            for epoch in range(epochs):
                pred = new_leaf(x_fit)
                if pred.dim() == 1:
                    pred = pred.reshape(-1, 1)
                loss = torch.mean((pred - y_fit) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if float(loss.detach().cpu()) < 1e-10:
                    break
            if getattr(ctx, "verbose", False):
                print(
                    "[Stage B] homogeneity_peel init: "
                    f"tag={new_tag}, loss={float(loss.detach().cpu()):.4e}, "
                    f"epochs={epoch + 1}"
                )
        except Exception as e:
            print(f"[Stage B] homogeneity_peel init failed: {e}")

    return _init_fn


def _make_homogeneity_peel_init_fn(
    *,
    target: AtomNode,
    new_tag: str,
    power_dim: int,
    ratio_dim: int,
    degree: float,
    X_in: torch.Tensor,
    f: torch.Tensor,
    ctx: StageBContext,
):
    return _make_homogeneity_peel_values_init_fn(
        new_tag=new_tag,
        degree=degree,
        q_values=X_in[:, int(power_dim)],
        numerator_values=X_in[:, int(ratio_dim)],
        f=f,
        ctx=ctx,
    )


def _multiply_exprs(exprs: Tuple[Node, ...]) -> Optional[Node]:
    """Build a left-associated product AST from one or more expressions."""
    if not exprs:
        return None
    out = clone_ast(exprs[0])
    for expr in exprs[1:]:
        out = MulNode(out, clone_ast(expr))
    return out


def _multiply_dims(dims: List[Tuple[Any, ...]], idxs: Tuple[int, ...]) -> Optional[Tuple[Any, ...]]:
    if not idxs or not dims:
        return None
    out = [d for d in dims[int(idxs[0])]]
    for idx in idxs[1:]:
        d = dims[int(idx)]
        if len(d) != len(out):
            return None
        out = [a + b for a, b in zip(out, d)]
    return tuple(out)


def _homogeneity_product_ratio_units_ok(
    ctx: StageBContext,
    target: AtomNode,
    *,
    power_dim: int,
    numerator_dims: Tuple[int, ...],
) -> bool:
    """Check that product(numerator_dims) / power_dim is dimensionless."""
    if not getattr(ctx, "enforce_units", False):
        return True
    units_spec = getattr(ctx, "units_spec", None)
    if units_spec is None:
        return True
    dims = _effective_input_dims_for_atom(target, units_spec)
    if not dims:
        return True
    max_idx = max((int(power_dim), *(int(i) for i in numerator_dims)))
    if len(dims) <= max_idx:
        return True
    power_d = tuple(dims[int(power_dim)])
    num_d = _multiply_dims(dims, tuple(int(i) for i in numerator_dims))
    if num_d is None or num_d == power_d:
        return True
    try:
        input_exprs = get_input_exprs(target)
        power_label = ast_to_human_readable(input_exprs[int(power_dim)])
        numerator_label = "*".join(
            ast_to_human_readable(input_exprs[int(i)])
            for i in numerator_dims
        )
    except Exception:
        power_label = f"input{int(power_dim)}"
        numerator_label = "*".join(f"input{int(i)}" for i in numerator_dims)
    ctx.log(
        "[Stage B] homogeneity_peel: skipping product ratio "
        f"({numerator_label})/{power_label} due to incompatible units"
    )
    return False


def _has_nn_atom(node: Node) -> bool:
    """Check if a subtree contains any NN atom."""
    if isinstance(node, AtomNode):
        return str(getattr(node, "kind", "")).lower() == "nn"
    if isinstance(node, (AddNode, MulNode)):
        return _has_nn_atom(node.left) or _has_nn_atom(node.right)
    if isinstance(node, PowNode):
        return _has_nn_atom(node.base)
    if isinstance(node, _UNARY_AST_NODES):
        return _has_nn_atom(node.arg)
    return False


class RuleAdditiveGaugeTransfer(StageBRule):
    """Resolve additive gauge scopes with a visible analytic transfer.

    For an unresolved scope like ``NN(u, s) + NN(s, v)`` or the same analytic
    prefactor multiplying both terms, a leaf-local rewrite can choose an
    arbitrary representative for the shared variables ``s``.  This rule only
    proposes visible, scope-simplifying representatives:

        NN(u, s) + NN(s, v) -> (NN(u) + h(s)) + NN(s, v)
        A*NN(u, s) + A*NN(s, v) -> A*(NN(u) + h(s)) + A*NN(s, v)

    where ``h`` comes from a small units-aware analytic transfer basis.  It
    never emits a hidden shifted-leaf-only candidate; validation and the
    fitted-candidate gauge gate still decide whether the proposal is accepted.
    """

    name = "additive_gauge_transfer"

    def iter_targets(self, ctx: StageBContext):
        _sync_stageb_rules_compat_overrides()
        try:
            return [scope.add_node for scope in ctx.additive_gauge_index().unresolved_scopes]
        except Exception:
            return []

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        _sync_stageb_rules_compat_overrides()
        if not isinstance(target, AddNode):
            return []

        try:
            scope = next(
                scope
                for scope in ctx.additive_gauge_index().unresolved_scopes
                if scope.add_node is target
            )
        except Exception:
            return []

        # v1 is intentionally narrow: direct NN + NN, or identical analytic
        # prefactor times each NN term.  More general prefactor-aware transfers
        # need explicit coefficient balancing and should be added separately.
        terms = tuple(getattr(scope, "terms", ()) or ())
        if len(terms) != 2:
            return []
        if any(len(getattr(term, "nn_atoms", ()) or ()) != 1 for term in terms):
            return []
        term_views = tuple(_gauge_transfer_term_view(term.node) for term in terms)
        if any(view is None for view in term_views):
            return []
        atoms = tuple(view[1] for view in term_views if view is not None)
        if len(atoms) != 2:
            return []
        if not _same_gauge_prefactor(term_views[0][0], term_views[1][0]):
            return []

        shared_vars = frozenset(set(_raw_vars_for_atom(atoms[0])) & set(_raw_vars_for_atom(atoms[1])))
        if not shared_vars:
            return []

        try:
            before_score = ctx.additive_gauge_global_score()
        except Exception:
            before_score = None

        max_features = int(getattr(ctx.lm_hp, "stageB_gauge_transfer_max_features", 16) or 16)
        max_features = max(0, min(max_features, 32))
        candidates: List[Candidate] = []

        for side_idx, atom in enumerate(atoms):
            raw_vars = frozenset(_raw_vars_for_atom(atom))
            private_vars = tuple(sorted(raw_vars - shared_vars))
            if not private_vars:
                continue

            replacement_atom = _make_gauge_private_atom(atom, private_vars, side_idx=side_idx)
            if replacement_atom is None:
                continue
            try:
                if effective_arity(replacement_atom) >= effective_arity(atom):
                    continue
            except Exception:
                pass

            try:
                required_dim = ctx.infer_target_dim(atom)
            except Exception:
                required_dim = None
            shared_inputs = _shared_input_exprs_for_atom(atom, shared_vars)
            features = build_transfer_basis(
                shared_vars=tuple(sorted(shared_vars)),
                shared_inputs=shared_inputs,
                ctx=ctx,
                required_dim=required_dim,
                max_features=max_features,
                strict_units=bool(getattr(ctx, "enforce_units", False)),
            )
            if not features:
                continue

            side_name = "left" if side_idx == 0 else "right"
            skipped_domain = 0
            for feature in features:
                domain_ok_frac = _transfer_feature_domain_ok_frac(ctx, feature)
                min_domain_ok = float(
                    getattr(
                        ctx.lm_hp,
                        "stageB_gauge_transfer_min_domain_ok_frac",
                        getattr(ctx.lm_hp, "macro_domain_ok_frac", 0.98),
                    )
                )
                if domain_ok_frac < min_domain_ok:
                    skipped_domain += 1
                    continue

                h_expr = clone_ast(feature.expr)
                new_private = clone_ast(replacement_atom)
                replacement = AddNode(new_private, h_expr)
                new_root = _replace_node_in_ast(ctx.state.root, atom, replacement)
                if new_root is None:
                    continue
                try:
                    after_score = ctx.additive_gauge_global_score(new_root)
                except Exception:
                    after_score = None
                if before_score is not None and after_score is not None and not (after_score < before_score):
                    continue

                meta = {
                    "pattern": self.name,
                    "structural": True,
                    "separability_like": True,
                    "partial_sep": True,
                    "additive_gauge_confirmed": True,
                    "additive_gauge_scope_simplified": True,
                    "additive_gauge_requires_scope_improvement": True,
                    "additive_gauge_scope_uid": getattr(scope, "uid", ""),
                    "additive_gauge_transfer_basis": str(feature.desc),
                    "additive_gauge_transfer_domain_ok_frac": float(domain_ok_frac),
                    "additive_gauge_transfer_prefactor": "same" if not _is_one_like(term_views[side_idx][0]) else "none",
                    "additive_gauge_transfer_shared_vars": tuple(sorted(shared_vars)),
                    "additive_gauge_transfer_side": side_name,
                    "log": (
                        f"[Stage B] additive_gauge_transfer: {side_name} "
                        f"NN{tuple(sorted(raw_vars))} -> NN{private_vars} + {feature.desc} "
                        f"(domain={domain_ok_frac:.2f}, prefactor={meta_prefactor_label(term_views[side_idx][0])})"
                    ),
                }
                if before_score is not None:
                    meta["additive_gauge_score_before"] = before_score
                if after_score is not None:
                    meta["additive_gauge_score_after_predicted"] = after_score

                sig = (
                    _subtree_content_hash(target),
                    hash(getattr(atom, "tag", None)),
                    hash(private_vars),
                    hash(str(feature.desc)),
                    side_idx,
                )
                candidates.append(
                    Candidate(
                        label=self.name,
                        root=new_root,
                        meta=meta,
                        signature=sig,
                    )
                )
            if skipped_domain:
                ctx.log(
                    f"[Stage B] additive_gauge_transfer skipped {skipped_domain} "
                    f"basis term(s) for bad sampled domain on {side_name}"
                )

        if candidates:
            ctx.log(
                f"[Stage B] additive_gauge_transfer proposed {len(candidates)} "
                f"visible candidate(s) for scope {getattr(scope, 'uid', '')}"
            )
        return candidates


class RuleMultiplicativeHomogeneityTransfer(StageBRule):
    """Try the reciprocal homogeneous representative before generic 1D fits.

    A product like ``x_i**k * NN(x_j/x_i)`` has an equivalent representative
    ``x_j**k * NN(x_i/x_j)``.  This rule is a bounded meta-rule: it builds the
    reciprocal representative only as a probe, then immediately runs the normal
    univariate analytic closures on that visible AST.  It never returns a
    hidden shifted/reparameterized NN state as an accepted candidate.
    """

    name = "multiplicative_homogeneity_transfer"
    multi_probe_native = True

    _ALLOWED_UNIVARIATE_FAMILIES = {
        "monomial_deg1",
        "monomial_deg2",
        "monomial_deg3",
        "monomial_deg4",
        "monomial_deg5",
        "monomial_deg6",
        "poly",
        "planck",
        "planck_full",
        "expm1",
        "symexp_denom",
        "1d_powexp",
        "polylog",
        "logshifted",
        "inv_poly",
        "sqrt_poly",
        "exp",
        "trig",
        "tanh",
    }

    def iter_targets(self, ctx: StageBContext):
        _sync_stageb_rules_compat_overrides()
        try:
            return [scope.mul_node for scope in ctx.homogeneous_gauge_index().unresolved_scopes]
        except Exception:
            return []

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        _sync_stageb_rules_compat_overrides()
        if not isinstance(target, MulNode):
            return []
        try:
            scopes = [
                scope for scope in ctx.homogeneous_gauge_index().unresolved_scopes
                if scope.mul_node is target
            ]
        except Exception:
            return []
        if not scopes:
            return []

        out: List[Candidate] = []
        for scope in scopes:
            out.extend(self._propose_for_scope(ctx, scope))
        if out:
            ctx.log(
                f"[Stage B] multiplicative_homogeneity_transfer proposed {len(out)} "
                "visible reciprocal-representative closure candidate(s)"
            )
        return out

    def _propose_for_scope(self, ctx: StageBContext, scope: HomogeneousGaugeScope) -> List[Candidate]:
        atom = scope.ratio_atom
        tag = getattr(atom, "tag", None)
        if tag is None or not isinstance(getattr(ctx.state, "reuse", None), dict):
            return []
        if tag not in ctx.state.reuse:
            return []
        degree = float(scope.power_factor.degree)
        if not float(degree).is_integer():
            return []
        ratio_power = float(getattr(scope, "ratio_power", 1.0))
        if abs(ratio_power) < 1.0e-12 or not math.isfinite(ratio_power):
            return []
        if not _homogeneous_scope_units_ok(ctx, scope):
            return []

        alt_power_var = int(scope.alternate_power_var)
        alt_ratio_expr = scope.alternate_ratio_expr
        alt_tag = f"{tag}_HG_{scope.power_factor.var_idx}_{alt_power_var}_{int(round(degree))}"
        nn_kwargs: Dict[str, Any] = {}
        for kk in ("num_segments", "dual_layer", "seg_width"):
            if kk in (getattr(atom, "kwargs", {}) or {}):
                nn_kwargs[kk] = (getattr(atom, "kwargs", {}) or {})[kk]
        alt_atom = AtomNode(
            kind="nn",
            var_idxs=tuple(int(j) for j in atom.var_idxs),
            kwargs=nn_kwargs,
            tag=alt_tag,
            inputs=(clone_ast(alt_ratio_expr),),
        )
        alt_power = _power_factor_for_var(alt_power_var, degree)
        alt_ratio_factor: Node
        if abs(ratio_power - 1.0) < 1.0e-12:
            alt_ratio_factor = alt_atom
        else:
            alt_ratio_factor = PowNode(alt_atom, ratio_power)
        factors = []
        replaced_power = False
        replaced_atom = False
        for factor in scope.factors:
            if factor is scope.power_factor.node:
                factors.append(alt_power)
                replaced_power = True
            elif factor is getattr(scope, "ratio_factor", atom):
                factors.append(alt_ratio_factor)
                replaced_atom = True
            else:
                factors.append(clone_ast(factor))
        if not replaced_power or not replaced_atom:
            return []
        temp_subtree = _rebuild_mul(factors)
        temp_root = _replace_node_in_ast(ctx.state.root, scope.mul_node, temp_subtree)
        if temp_root is None:
            return []

        if scope.power_factor.var_idx == scope.ratio.denominator_var:
            transfer_degree = degree / ratio_power
        else:
            transfer_degree = -degree / ratio_power
        reuse_override = dict(ctx.state.reuse)
        reuse_override[alt_tag] = _HomogeneousGaugeTeacher(ctx.state.reuse[tag], transfer_degree)
        reuses_override = _homogeneous_reuses_override(ctx, tag, alt_tag, transfer_degree)

        temp_state = copy.copy(ctx.state)
        temp_state.root = temp_root
        temp_state.reuse = reuse_override
        if reuses_override is not None:
            temp_state.reuses = reuses_override
        temp_ctx = copy.copy(ctx)
        temp_ctx.state = temp_state
        temp_ctx._stageB_coord_alias_active = True
        for cache_name in ("_cache", "_dim_cache"):
            if hasattr(temp_ctx, cache_name):
                try:
                    setattr(temp_ctx, cache_name, {})
                except Exception:
                    pass
        if hasattr(temp_ctx, "_dim_cache_root_id"):
            try:
                temp_ctx._dim_cache_root_id = None
            except Exception:
                pass

        try:
            before_score = ctx.homogeneous_gauge_global_score()
        except Exception:
            before_score = None

        try:
            raw_cands = RuleUniNN(factorized_search_rule=None).propose(temp_ctx, alt_atom) or []
        except Exception as exc:
            if bool(getattr(ctx, "verbose", False)):
                ctx.log(f"[Stage B] multiplicative_homogeneity_transfer probe failed: {exc}")
            return []

        candidates: List[Candidate] = []
        for cand in raw_cands:
            if cand is None:
                continue
            try:
                if not cand.materialise():
                    continue
            except Exception:
                continue
            family = str(candidate_pattern_name(cand) or cand.label)
            base_family = family.split("[", 1)[0]
            if base_family not in self._ALLOWED_UNIVARIATE_FAMILIES:
                continue
            if cand.root is None:
                continue
            try:
                after_score = ctx.homogeneous_gauge_global_score(cand.root)
            except Exception:
                after_score = None
            if before_score is not None and after_score is not None and not (after_score < before_score):
                continue
            meta = dict(cand.meta) if isinstance(getattr(cand, "meta", None), dict) else {}
            old_log = meta.get("log", "")
            meta.update(
                {
                    "pattern": self.name,
                    "pattern_family": base_family,
                    "structural": True,
                    "separability_like": True,
                    "homogeneous_gauge_confirmed": True,
                    "homogeneous_gauge_scope_simplified": True,
                    "homogeneous_gauge_requires_scope_improvement": True,
                    "homogeneous_gauge_scope_uid": getattr(scope, "uid", ""),
                    "homogeneous_gauge_score_before": before_score,
                    "homogeneous_gauge_score_after_predicted": after_score,
                    "homogeneous_gauge_transfer_from": scope.direction,
                    "homogeneous_gauge_transfer_to": (
                        f"x{alt_power_var}^{degree:g} * NN("
                        f"x{scope.ratio.denominator_var}/x{scope.ratio.numerator_var})"
                    ),
                    "homogeneous_gauge_transfer_degree": float(transfer_degree),
                    "homogeneous_gauge_ratio_power": float(ratio_power),
                    "log": (
                        "[Stage B] multiplicative_homogeneity_transfer: "
                        f"{scope.direction} -> x{alt_power_var}^{degree:g}*NN(1/z); "
                        f"then trying {cand.label}"
                        + (f" | {old_log}" if isinstance(old_log, str) and old_log else "")
                    ),
                }
            )
            label = f"homogeneous_{cand.label}"
            sig = (
                _subtree_content_hash(scope.mul_node),
                hash(str(getattr(scope, "uid", ""))),
                hash(str(cand.label)),
                int(scope.power_factor.var_idx),
                int(alt_power_var),
                int(round(degree * 1000.0)),
            )
            candidates.append(Candidate(label=label, root=cand.root, init_fn=cand.init_fn, meta=meta, signature=sig))

        priority = {
            "sqrt_poly": 0,
            "inv_poly": 1,
            "planck": 2,
            "planck_full": 3,
            "expm1": 4,
            "1d_powexp": 5,
            "polylog": 6,
            "logshifted": 7,
            "exp": 8,
            "trig": 9,
            "tanh": 10,
            "monomial_deg1": 11,
            "monomial_deg2": 12,
            "monomial_deg3": 13,
            "monomial_deg4": 14,
            "monomial_deg5": 15,
            "monomial_deg6": 16,
            "poly": 17,
        }
        candidates.sort(
            key=lambda c: (
                priority.get(str((c.meta or {}).get("pattern_family", c.label)), 50),
                str(c.label),
            )
        )
        return candidates[:8]


def _power_factor_for_var(var_idx: int, degree: float) -> Node:
    base = Var(int(var_idx))
    if abs(float(degree) - 1.0) < 1.0e-12:
        return base
    return PowNode(base, float(degree))


def _homogeneous_reuses_override(
    ctx: StageBContext,
    old_tag: str,
    new_tag: str,
    transfer_degree: float,
):
    state_reuses = getattr(ctx.state, "reuses", None)
    if not isinstance(state_reuses, (list, tuple)):
        return None
    wrapped_reuses = []
    for reuse_i in state_reuses:
        if not isinstance(reuse_i, dict) or old_tag not in reuse_i:
            return None
        wrapped = dict(reuse_i)
        wrapped[new_tag] = _HomogeneousGaugeTeacher(reuse_i[old_tag], transfer_degree)
        wrapped_reuses.append(wrapped)
    return wrapped_reuses


def _homogeneous_scope_units_ok(ctx: StageBContext, scope: HomogeneousGaugeScope) -> bool:
    """Require the two ratio variables to be compatible before transfer."""
    if not getattr(ctx, "enforce_units", False):
        return True
    units_spec = getattr(ctx, "units_spec", None)
    if units_spec is None:
        return True
    try:
        dims = units_spec.x_dims
        n = int(scope.ratio.numerator_var)
        d = int(scope.ratio.denominator_var)
        if tuple(dims[n]) == tuple(dims[d]):
            return True
    except Exception:
        return True
    ctx.log(
        "[Stage B] multiplicative_homogeneity_transfer: skipping "
        f"x{scope.ratio.numerator_var}/x{scope.ratio.denominator_var} "
        "because ratio variables have incompatible units"
    )
    return False


def _raw_vars_for_atom(atom: AtomNode) -> Tuple[int, ...]:
    try:
        return tuple(int(v) for v in atom.raw_var_idxs)
    except Exception:
        return tuple(int(v) for v in getattr(atom, "var_idxs", ()) or ())


def _gauge_transfer_term_view(term_node: Node) -> Optional[Tuple[Node, AtomNode]]:
    """Return ``(analytic_prefactor, nn_atom)`` for a supported gauge term."""
    if isinstance(term_node, AtomNode) and str(getattr(term_node, "kind", "")).lower() == "nn":
        return ConstNode(1.0), term_node
    if not isinstance(term_node, MulNode):
        return None

    factors = _flatten_mul(term_node)
    nn_positions = [
        i
        for i, factor in enumerate(factors)
        if isinstance(factor, AtomNode) and str(getattr(factor, "kind", "")).lower() == "nn"
    ]
    if len(nn_positions) != 1:
        return None
    nn_pos = nn_positions[0]
    prefactors = []
    for i, factor in enumerate(factors):
        if i == nn_pos:
            continue
        if _has_nn_atom(factor):
            return None
        prefactors.append(factor)
    prefactor = ConstNode(1.0) if not prefactors else _rebuild_mul([clone_ast(f) for f in prefactors])
    return prefactor, factors[nn_pos]


def _same_gauge_prefactor(left: Node, right: Node) -> bool:
    try:
        return _subtree_content_hash(left) == _subtree_content_hash(right)
    except Exception:
        return False


def _is_one_like(node: Node) -> bool:
    return isinstance(node, ConstNode) and abs(float(node.value) - 1.0) < 1e-12


def meta_prefactor_label(node: Node) -> str:
    return "none" if _is_one_like(node) else "same"


def _make_gauge_private_atom(atom: AtomNode, private_vars: Tuple[int, ...], *, side_idx: int) -> Optional[AtomNode]:
    inputs = _select_inputs_subset(atom, set(private_vars))
    kwargs = copy.deepcopy(getattr(atom, "kwargs", {}) or {})
    tag_base = getattr(atom, "tag", None) or f"gauge_{id(atom)}"
    try:
        return AtomNode(
            kind="nn",
            var_idxs=tuple(private_vars),
            kwargs=kwargs,
            tag=f"{tag_base}_gpriv{side_idx}",
            inputs=inputs,
        )
    except Exception:
        return None


def _shared_input_exprs_for_atom(atom: AtomNode, shared_vars: frozenset[int]) -> Tuple[Node, ...]:
    return _select_inputs_subset(atom, set(shared_vars)) or tuple(Var(int(v)) for v in sorted(shared_vars))


def _select_inputs_subset(atom: AtomNode, raw_vars: set[int]) -> Optional[Tuple[Node, ...]]:
    selected: List[Node] = []
    try:
        inputs = get_input_exprs(atom)
    except Exception:
        inputs = ()
    for inp in inputs:
        try:
            inp_vars = set(int(v) for v in _collect_var_idxs_from_node(inp))
        except Exception:
            inp_vars = set()
        if inp_vars and inp_vars <= raw_vars:
            selected.append(clone_ast(inp))
    if not selected:
        return None
    if all(is_trivial_input(inp) for inp in selected):
        return None
    return tuple(selected)


def _transfer_feature_domain_ok_frac(ctx: StageBContext, feature: Any) -> float:
    """Sample-check transfer basis domain before launching an LM fit."""
    loader = getattr(ctx, "train_loader_probe", None) or getattr(ctx, "train_loader", None)
    if loader is None:
        return 1.0
    if isinstance(loader, (list, tuple)):
        loaders = list(loader)
    else:
        loaders = [loader]

    total = 0
    ok = 0
    max_points = int(getattr(ctx.lm_hp, "stageB_gauge_transfer_domain_points", 2048) or 2048)
    for loader_i in loaders:
        if loader_i is None:
            continue
        for batch in loader_i:
            try:
                xb = batch[0] if isinstance(batch, (list, tuple)) else batch
                X = xb.to(device=ctx.device, dtype=ctx.dtype)
                y = _eval_transfer_basis_expr(feature.basis_expr, X)
                mask = torch.isfinite(y)
            except Exception:
                return 0.0
            n = int(mask.numel())
            if n <= 0:
                continue
            ok += int(mask.sum().item())
            total += n
            if total >= max_points:
                break
        if total >= max_points:
            break
    if total <= 0:
        return 1.0
    return float(ok) / float(total)


def _eval_transfer_basis_expr(node: Node, X: torch.Tensor) -> torch.Tensor:
    """Evaluate a pure analytic transfer-basis expression on raw inputs."""
    if isinstance(node, ConstNode):
        return torch.full((X.shape[0],), float(node.value), device=X.device, dtype=X.dtype)
    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        if kind in ("var", "x", "input") and len(getattr(node, "var_idxs", ()) or ()) == 1:
            return X[:, int(node.var_idxs[0])]
        raise TypeError(f"unsupported transfer-basis atom kind={kind}")
    if isinstance(node, AddNode):
        return _eval_transfer_basis_expr(node.left, X) + _eval_transfer_basis_expr(node.right, X)
    if isinstance(node, MulNode):
        return _eval_transfer_basis_expr(node.left, X) * _eval_transfer_basis_expr(node.right, X)
    if isinstance(node, PowNode):
        return torch.pow(_eval_transfer_basis_expr(node.base, X), float(node.exponent))
    if isinstance(node, LogNode):
        return torch.log(_eval_transfer_basis_expr(node.arg, X))
    if isinstance(node, ExpNode):
        return torch.exp(_eval_transfer_basis_expr(node.arg, X))
    if isinstance(node, SinNode):
        return torch.sin(_eval_transfer_basis_expr(node.arg, X))
    if isinstance(node, CosNode):
        return torch.cos(_eval_transfer_basis_expr(node.arg, X))
    if isinstance(node, AsinNode):
        return torch.asin(torch.clamp(_eval_transfer_basis_expr(node.arg, X), -1.0, 1.0))
    if isinstance(node, AcosNode):
        return torch.acos(torch.clamp(_eval_transfer_basis_expr(node.arg, X), -1.0, 1.0))
    if isinstance(node, AtanNode):
        return torch.atan(_eval_transfer_basis_expr(node.arg, X))
    raise TypeError(f"unsupported transfer-basis node {type(node).__name__}")


class RuleCommonPrefactor(StageBRule):
    """Factor shared multiplicative structure from AddNode children.

    Targets ``AddNode(L, R)`` patterns where both branches share variables and
    at least one contains an NN atom.  After demodulating any trig (cos/sin)
    factors from the branches, computes the ratio of one branch's core value
    to the other and probes for simple analytical structure:

    (a) Near-constant ratio → ``L + R  =  L * (1 + c)``
    (b) Power-product ratio → ``L + R  =  L * (1 + c·∏xᵢ^aᵢ)``
    (c) sqrt(1+PP) ratio   → ``L + R  =  L * (1 + sqrt(1 + c·∏xᵢ^aᵢ) * trig)``

    Primary use case: physics expressions like ``P(u)·(1 + Q(u)·cos(θ))``
    where ``P`` is a common prefactor (orbital mechanics, interference,
    scattering cross-sections).

    Pattern label: common_prefactor
    """

    name = "common_prefactor"

    # ── targeting ──────────────────────────────────────────────────

    def iter_targets(self, ctx: StageBContext):
        _sync_stageb_rules_compat_overrides()
        add_nodes = _iter_add_nodes(ctx.state.root)
        targets = []
        for an in add_nodes:
            left_vars = set(_vars_in_subtree_simple(an.left))
            right_vars = set(_vars_in_subtree_simple(an.right))
            if not left_vars & right_vars:
                continue
            if not _has_nn_atom(an.left) and not _has_nn_atom(an.right):
                continue
            targets.append(an)
        return targets

    # ── power-product probe ───────────────────────────────────────

    @staticmethod
    def _probe_pp(ratio_np: np.ndarray, X_np: np.ndarray):
        """Log-space OLS: ``log|ratio| = Σ aᵢ·log|xᵢ| + c``.

        Returns ``(exponents, log_c, r2)`` or ``None``.
        """
        N, m = X_np.shape
        eps = 1e-12
        abs_ratio = np.abs(ratio_np)
        mask = abs_ratio > eps
        for j in range(m):
            mask &= X_np[:, j] > eps
        if mask.sum() < max(100, N // 4):
            return None

        log_r = np.log(abs_ratio[mask])
        log_x = np.log(X_np[mask])
        ones = np.ones((log_x.shape[0], 1), dtype=np.float64)
        A = np.concatenate([log_x, ones], axis=1)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, log_r, rcond=None)
        except np.linalg.LinAlgError:
            return None

        exponents = coeffs[:m]
        log_c = coeffs[m]
        predicted = A @ coeffs
        if not np.all(np.isfinite(predicted)):
            return None
        ss_res = float(np.sum((log_r - predicted) ** 2))
        ss_tot = float(np.sum((log_r - np.mean(log_r)) ** 2))
        r2 = 1.0 if ss_tot < eps else 1.0 - ss_res / ss_tot
        return exponents, log_c, r2

    # ── snap exponents to half-integers ───────────────────────────

    @staticmethod
    def _snap(exponents: np.ndarray, log_X: np.ndarray, log_r: np.ndarray, log_c: float):
        """Snap exponents to nearest half-integer; fall back to raw if fit degrades."""
        snapped = np.zeros_like(exponents)
        for j in range(len(exponents)):
            half = round(exponents[j] * 2.0) / 2.0
            snapped[j] = half if abs(exponents[j] - half) < 0.25 else exponents[j]
        predicted = log_X @ snapped + log_c
        if np.all(np.isfinite(predicted)):
            ss_res = float(np.sum((log_r - predicted) ** 2))
            ss_tot = float(np.sum((log_r - np.mean(log_r)) ** 2))
            r2_snap = 1.0 if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
            if r2_snap >= 0.95:
                return snapped
        return exponents

    # ── AST builder ───────────────────────────────────────────────

    def _build_candidate(
        self,
        ctx: StageBContext,
        target: AddNode,
        factor_branch: Node,
        trig_factors: List[Node],
        probe_kind: str,
        direction: str,
        c_val: float,
        exponents: Optional[np.ndarray],
        X_valid: Optional[np.ndarray],
        const_variant: Optional[Dict[str, Any]] = None,
    ) -> Optional[Candidate]:
        """Build ``factor_branch * (1 + Q * trig_product)`` AST."""
        st = ctx.state
        parent_tag = f"cpf_{id(target)}"
        new_atom_ids: List[int] = []
        const_tag = f"{parent_tag}_c"
        const_variant_eff = (
            const_variant
            if isinstance(const_variant, dict)
            else {
                "mode": "scale",
                "name": const_tag,
                "tag": const_tag,
                "value": float(c_val),
                "label_suffix": "",
            }
        )
        const_leaf = _build_scalar_atom_from_variant(const_variant_eff)
        new_atom_ids.append(id(const_leaf))
        init_c = float(const_variant_eff.get("value", c_val))
        label_suffix = str(const_variant_eff.get("label_suffix", ""))

        # ── Q_ast construction ────────────────────────────────────
        if probe_kind == "constant":
            Q_ast: Node = const_leaf

        elif probe_kind == "power_product":
            monomial = self._build_rpoly_monomial(
                exponents, parent_tag, new_atom_ids,
            )
            if monomial is None:
                return None
            Q_ast = MulNode(left=const_leaf, right=monomial)

        elif probe_kind == "sqrt_1_refine_pp":
            monomial = self._build_rpoly_monomial(
                exponents, parent_tag, new_atom_ids,
            )
            if monomial is None:
                return None
            inner_pp = MulNode(left=const_leaf, right=monomial)
            Q_ast = PowNode(
                base=AddNode(left=ConstNode(1.0), right=inner_pp),
                exponent=0.5,
            )
        else:
            return None

        # ── rest = Q_ast * trig_product (if any) ─────────────────
        if trig_factors:
            trig_product: Node = clone_ast(trig_factors[0])
            for tf in trig_factors[1:]:
                trig_product = MulNode(left=trig_product, right=clone_ast(tf))
            rest: Node = MulNode(left=Q_ast, right=trig_product)
        else:
            rest = Q_ast

        # ── factored AST: factor_branch * (1 + rest) ─────────────
        one_refine_rest = AddNode(left=ConstNode(1.0), right=rest)
        factored = MulNode(left=clone_ast(factor_branch), right=one_refine_rest)

        new_root = _replace_node_in_ast(st.root, target, factored)

        # ── init_fn ──────────────────────────────────────────────
        _aids = list(new_atom_ids)
        _cval = float(init_c)

        def _init_cpf(root_new, model_new, *, _a=_aids, _c=_cval):
            atl = build_atom_to_leaf_map(root_new, model_new)
            with torch.no_grad():
                for aid in _a:
                    leaf = atl.get(aid)
                    if leaf is None:
                        continue
                    if _set_constant_leaf_value(leaf, float(_c)):
                        continue
                    core = getattr(leaf, "model", leaf)
                    if not hasattr(core, "coeffs"):
                        continue
                    if core.coeffs.numel() > 0:
                        core.coeffs.fill_(0.0)
                        core.coeffs[-1] = 1.0

        _init_cpf._after_analytic_init = True

        sig = (
            _subtree_content_hash(target),
            hash(probe_kind),
            hash(direction),
            hash(str(const_variant_eff.get("tag", ""))),
        )
        return Candidate(
            label=f"common_prefactor{label_suffix}",
            root=new_root,
            init_fn=_init_cpf,
            meta={
                "structural": True,
                "noisy_gauge_requires_strict_improvement": True,
                "noisy_gauge_family": "common_prefactor",
                "log": (
                    f"[Stage B] common_prefactor ({probe_kind}, {direction}) "
                    f"c≈{init_c:.4g}{label_suffix}"
                ),
            },
            signature=sig,
        )

    @staticmethod
    def _build_rpoly_monomial(
        exponents: np.ndarray, parent_tag: str, new_atom_ids: List[int],
    ) -> Optional[Node]:
        """Build ``∏ rpoly(xⱼ)^aⱼ`` monomial from snapped exponents."""
        factors: List[Node] = []
        for j in range(len(exponents)):
            if abs(exponents[j]) < 1e-10:
                continue
            var_atom = AtomNode(
                kind="rpoly", var_idxs=(j,), kwargs={"degree": 1},
                tag=f"{parent_tag}_v{j}",
            )
            new_atom_ids.append(id(var_atom))
            if abs(exponents[j] - 1.0) < 1e-10:
                factors.append(var_atom)
            else:
                factors.append(PowNode(base=var_atom, exponent=float(exponents[j])))
        if not factors:
            return None
        product: Node = factors[0]
        for f in factors[1:]:
            product = MulNode(left=product, right=f)
        return product

    # ── propose ───────────────────────────────────────────────────

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        _sync_stageb_rules_compat_overrides()
        if not isinstance(target, AddNode):
            return []

        st = ctx.state
        try:
            atom_to_leaf = build_atom_to_leaf_map(st.root, st.model)
        except Exception:
            return []

        # ── gather data (up to 4096 points) ──────────────────────
        X_batches: List[torch.Tensor] = []
        n_total = 0
        for batch in ctx.train_loader_probe:
            xb = batch[0].to(ctx.device, ctx.dtype)
            X_batches.append(xb)
            n_total += xb.shape[0]
            if n_total >= 4096:
                break
        if not X_batches:
            return []
        X = torch.cat(X_batches, dim=0)[:4096]
        N = X.shape[0]
        if N < 200:
            return []

        L, R = target.left, target.right

        # ── evaluate both branches ───────────────────────────────
        try:
            with torch.no_grad():
                L_val = _eval_subtree_with_leaf_map(L, X, atom_to_leaf)
                R_val = _eval_subtree_with_leaf_map(R, X, atom_to_leaf)
        except Exception:
            return []

        # ── demodulate trig factors ──────────────────────────────
        L_factors = _flatten_mul(L)
        R_factors = _flatten_mul(R)
        L_trig = [f for f in L_factors if isinstance(f, (CosNode, SinNode))]
        R_trig = [f for f in R_factors if isinstance(f, (CosNode, SinNode))]

        if L_trig and R_trig:
            ctx.log("[Stage B] common_prefactor: both branches have trig — skip")
            return []

        L_core = L_val
        R_core = R_val

        if R_trig:
            try:
                with torch.no_grad():
                    tv = torch.ones(N, device=X.device, dtype=X.dtype)
                    for tf in R_trig:
                        tv = tv * _eval_subtree_with_leaf_map(tf, X, atom_to_leaf)
                safe = tv.abs() > 0.15
                if safe.sum() < 200:
                    ctx.log("[Stage B] common_prefactor: too few safe trig points on R")
                    return []
                R_core = torch.full_like(R_val, float("nan"))
                R_core[safe] = R_val[safe] / tv[safe]
            except Exception:
                return []

        if L_trig:
            try:
                with torch.no_grad():
                    tv = torch.ones(N, device=X.device, dtype=X.dtype)
                    for tf in L_trig:
                        tv = tv * _eval_subtree_with_leaf_map(tf, X, atom_to_leaf)
                safe = tv.abs() > 0.15
                if safe.sum() < 200:
                    ctx.log("[Stage B] common_prefactor: too few safe trig points on L")
                    return []
                L_core = torch.full_like(L_val, float("nan"))
                L_core[safe] = L_val[safe] / tv[safe]
            except Exception:
                return []

        X_np = X.detach().cpu().numpy().astype(np.float64)

        candidates: List[Candidate] = []

        # ── try both factoring directions ────────────────────────
        for direction, numer, denom, factor_branch, trig_fs in [
            ("R/L", R_core, L_core, L, R_trig),
            ("L/R", L_core, R_core, R, L_trig),
        ]:
            valid = (
                torch.isfinite(numer)
                & torch.isfinite(denom)
                & (denom.abs() > 1e-12)
            )
            n_valid = int(valid.sum())
            if n_valid < 200:
                continue

            ratio_np = (
                (numer[valid] / denom[valid])
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .ravel()
            )
            X_valid = X_np[valid.cpu().numpy()]

            # (a) near-constant ratio
            mean_r = float(np.mean(ratio_np))
            rel_std = float(np.std(ratio_np)) / (abs(mean_r) + 1e-30)
            if rel_std < 0.01 and abs(mean_r) > 1e-6:
                ctx.log(
                    f"[Stage B] common_prefactor: constant ratio={mean_r:.4f} "
                    f"(rel_std={rel_std:.4f}) direction={direction}"
                )
                const_variants = _scalar_constant_variants(
                    getattr(ctx, "units_spec", None),
                    base_tag=f"cpf_{id(target)}_c",
                    scale_init=float(mean_r),
                )
                for cvar in const_variants:
                    cand = self._build_candidate(
                        ctx,
                        target,
                        factor_branch,
                        trig_fs,
                        "constant",
                        direction,
                        mean_r,
                        None,
                        None,
                        const_variant=cvar,
                    )
                    if cand is not None:
                        candidates.append(cand)
                continue

            # (b) power-product ratio
            pp = self._probe_pp(ratio_np, X_valid)
            if pp is not None:
                exponents, log_c, r2 = pp
                if r2 > 0.98:
                    # Prepare log-space arrays for snapping
                    eps = 1e-12
                    abs_ratio = np.abs(ratio_np)
                    pp_mask = abs_ratio > eps
                    for j in range(X_valid.shape[1]):
                        pp_mask &= X_valid[:, j] > eps
                    log_r = np.log(abs_ratio[pp_mask])
                    log_x = np.log(X_valid[pp_mask])
                    snapped = self._snap(exponents, log_x, log_r, log_c)
                    sign_c = 1.0 if np.median(ratio_np) >= 0 else -1.0
                    c_val = sign_c * math.exp(log_c)
                    ctx.log(
                        f"[Stage B] common_prefactor: PP ratio R²={r2:.4f} "
                        f"exp={[f'{e:.2f}' for e in snapped]} dir={direction}"
                    )
                    const_variants = _scalar_constant_variants(
                        getattr(ctx, "units_spec", None),
                        base_tag=f"cpf_{id(target)}_c",
                        scale_init=float(c_val),
                    )
                    for cvar in const_variants:
                        cand = self._build_candidate(
                            ctx,
                            target,
                            factor_branch,
                            trig_fs,
                            "power_product",
                            direction,
                            c_val,
                            snapped,
                            X_valid,
                            const_variant=cvar,
                        )
                        if cand is not None:
                            candidates.append(cand)
                    continue

            # (c) sqrt(1+PP) — check if ratio²-1 is a power product
            rsq_m1 = ratio_np ** 2 - 1.0
            if np.mean(rsq_m1 > 1e-6) > 0.7:
                pp2 = self._probe_pp(rsq_m1, X_valid)
                if pp2 is not None:
                    exponents2, log_c2, r2_2 = pp2
                    if r2_2 > 0.98:
                        eps = 1e-12
                        abs_rsq = np.abs(rsq_m1)
                        pp2_mask = abs_rsq > eps
                        for j in range(X_valid.shape[1]):
                            pp2_mask &= X_valid[:, j] > eps
                        log_r2 = np.log(abs_rsq[pp2_mask])
                        log_x2 = np.log(X_valid[pp2_mask])
                        snapped2 = self._snap(exponents2, log_x2, log_r2, log_c2)
                        c_val2 = math.exp(log_c2)
                        ctx.log(
                            f"[Stage B] common_prefactor: sqrt(1+PP) R²={r2_2:.4f} "
                            f"exp={[f'{e:.2f}' for e in snapped2]} dir={direction}"
                        )
                        const_variants = _scalar_constant_variants(
                            getattr(ctx, "units_spec", None),
                            base_tag=f"cpf_{id(target)}_c",
                            scale_init=float(c_val2),
                        )
                        for cvar in const_variants:
                            cand = self._build_candidate(
                                ctx,
                                target,
                                factor_branch,
                                trig_fs,
                                "sqrt_1_refine_pp",
                                direction,
                                c_val2,
                                snapped2,
                                X_valid,
                                const_variant=cvar,
                            )
                            if cand is not None:
                                candidates.append(cand)

        return candidates


class RuleOverlapPrefactorPeelNN(StageBRule):
    """Reduce overlap in direct additive NN siblings by peeling a shared factor.

    v1 targets only direct ``AddNode(nn(...), nn(...))`` structures with
    singleton shared-variable peels:

        nn_L(u, r, t) + nn_R(v, r, t)
            -> nn_M(t) * (nn_A(u, r) + nn_B(v, r, t))

    and the mirrored right-peel variant. Candidate screening uses the peeled
    leaf's derivative/Hessian structure; LM validation decides the final fit.
    """

    name = "overlap_prefactor_peel"

    def iter_targets(self, ctx: StageBContext):
        _sync_stageb_rules_compat_overrides()
        targets: List[AddNode] = []
        for an in _iter_add_nodes(ctx.state.root):
            if not isinstance(an.left, AtomNode) or not isinstance(an.right, AtomNode):
                continue
            if str(getattr(an.left, "kind", "")).lower() != "nn":
                continue
            if str(getattr(an.right, "kind", "")).lower() != "nn":
                continue
            left_vars = set(int(v) for v in getattr(an.left, "var_idxs", ()) or ())
            right_vars = set(int(v) for v in getattr(an.right, "var_idxs", ()) or ())
            shared = left_vars & right_vars
            if not shared:
                continue
            if not ((left_vars - right_vars) or (right_vars - left_vars)):
                continue
            targets.append(an)
        return targets

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        _sync_stageb_rules_compat_overrides()
        if not isinstance(target, AddNode):
            return []

        st = ctx.state
        try:
            left_vars = tuple(int(v) for v in getattr(target.left, "var_idxs", ()) or ())
            right_vars = tuple(int(v) for v in getattr(target.right, "var_idxs", ()) or ())
            ctx.log(
                f"[Stage B] Probing overlap_prefactor_peel on AddNode "
                f"left={left_vars} right={right_vars}"
            )
        except Exception:
            pass
        raw_candidates = _build_overlap_prefactor_peel_candidates(
            root=st.root,
            target=target,
            model=st.model,
            train_loader=ctx.train_loader_probe,
            device=ctx.device,
            dtype=ctx.dtype,
            log_fn=ctx.log,
            max_points=2048,
            max_per_direction=1,
        )
        if not raw_candidates:
            ctx.log("[Stage B] overlap_prefactor_peel: no proposals survived builder")
            return []

        out: List[Candidate] = []
        for cand_root, init_fn, metadata in raw_candidates:
            signature = metadata.get("signature", None) if metadata else None
            out.append(
                Candidate(
                    self.name,
                    cand_root,
                    init_fn,
                    meta=metadata if metadata else {"structural": True},
                    signature=signature,
                )
            )
        return out


class RuleOverlapCountertermPeelNN(StageBRule):
    """Reduce overlap in direct multiplicative NN siblings by peeling a counterterm.

    v1 targets only direct ``MulNode(nn(...), nn(...))`` structures with
    singleton shared-variable peels:

        nn_L(u, r, t) * nn_R(v, r, t)
            -> (nn_C(t) + nn_A(u, r)) * nn_R(v, r, t)

    and the mirrored right-peel variant. Candidate screening uses the peeled
    leaf's mixed-Hessian structure; LM validation decides the final fit.
    """

    name = "overlap_counterterm_peel"

    def iter_targets(self, ctx: StageBContext):
        _sync_stageb_rules_compat_overrides()
        targets: List[MulNode] = []
        for mn in _iter_mul_nodes(ctx.state.root):
            if not isinstance(mn.left, AtomNode) or not isinstance(mn.right, AtomNode):
                continue
            if str(getattr(mn.left, "kind", "")).lower() != "nn":
                continue
            if str(getattr(mn.right, "kind", "")).lower() != "nn":
                continue
            left_vars = set(int(v) for v in getattr(mn.left, "var_idxs", ()) or ())
            right_vars = set(int(v) for v in getattr(mn.right, "var_idxs", ()) or ())
            shared = left_vars & right_vars
            if not shared:
                continue
            if not ((left_vars - right_vars) or (right_vars - left_vars)):
                continue
            targets.append(mn)
        return targets

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        _sync_stageb_rules_compat_overrides()
        if not isinstance(target, MulNode):
            return []

        st = ctx.state
        try:
            left_vars = tuple(int(v) for v in getattr(target.left, "var_idxs", ()) or ())
            right_vars = tuple(int(v) for v in getattr(target.right, "var_idxs", ()) or ())
            ctx.log(
                f"[Stage B] Probing overlap_counterterm_peel on MulNode "
                f"left={left_vars} right={right_vars}"
            )
        except Exception:
            pass
        raw_candidates = _build_overlap_counterterm_peel_candidates(
            root=st.root,
            target=target,
            model=st.model,
            train_loader=ctx.train_loader_probe,
            device=ctx.device,
            dtype=ctx.dtype,
            log_fn=ctx.log,
            max_points=2048,
            max_per_direction=1,
        )
        if not raw_candidates:
            ctx.log("[Stage B] overlap_counterterm_peel: no proposals survived builder")
            return []

        out: List[Candidate] = []
        for cand_root, init_fn, metadata in raw_candidates:
            signature = metadata.get("signature", None) if metadata else None
            out.append(
                Candidate(
                    self.name,
                    cand_root,
                    init_fn,
                    meta=metadata if metadata else {"structural": True},
                    signature=signature,
                )
            )
        return out


class RuleCounterfactorAddSplitNN(StageBRule):
    """Late-stage fallback: try to split a stubborn multivariate nn(...) leaf as

       nn(x)  →  poly(x_A) * poly(x_B) * (nn(x_A) + nn(x_B))

    where poly(x_A), poly(x_B) are *multiplicative* counterfactors chosen so that
    r(x) = nn(x) / (poly(x_A)*poly(x_B)) is approximately additively separable
    between A and B.

    Note: the additive split is only defined up to a constant shift
    (g+h == (g+c)+(h-c)). We do not introduce extra scalar (α, β) leaves here;
    any constant offset/overall scaling can be absorbed into the two NN leaves
    and/or the counterfactors.

    Pattern label: counterfactor_add_split
    """

    name = "counterfactor_add_split"

    def iter_targets(self, ctx: StageBContext):
        _sync_stageb_rules_compat_overrides()
        return _collect_multivariate_nn_atoms(ctx.state.root)

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        _sync_stageb_rules_compat_overrides()
        if not isinstance(target, AtomNode):
            return []
        if str(target.kind).lower() != "nn":
            return []
        if effective_arity(target) < 2:
            return []

        st = ctx.state
        ctx.log(f"[Stage B] Probing counterfactor add split on NN vars={target.var_idxs}")
        cand_root, init_fn, metadata = _build_counterfactor_add_split_candidate(
            root=st.root,
            target=target,
            model=st.model,
            reuse=st.reuse,
            train_loader=ctx.train_loader_probe,
            device=ctx.device,
            dtype=ctx.dtype,
            # Allow degree-0 counterfactors (constants) so one-sided counterfactors
            # like z**2 * (g(z) + h(t)) are discoverable.
            degrees_A=(0, 1, 2),
            degrees_B=(0, 1, 2),
            n_alt=5,
            max_points=4096,
            # v2: loosen identity tolerance; LM will validate the rewrite.
            rel_err_tol=5e-2,
        )
        if cand_root is None:
            return []

        # Option-B refactor: rules must be pure proposal generators.
        # Deduplication/attempt bookkeeping happens in StageBEngine.
        signature = metadata.get("signature", None) if metadata else None

        # Return candidate with metadata from builder
        return [
            Candidate(
                self.name,
                cand_root,
                init_fn,
                meta=metadata if metadata else {"structural": True},
                signature=signature,
            )
        ]


class RuleCountertermMulSplitNN(StageBRule):
    """Late-stage fallback: try to split a stubborn multivariate nn(...) leaf as

        nn(x)  →  poly(x_A) + nn(x_A)*nn(x_B)
        nn(x)  →  poly(x_B) + nn(x_A)*nn(x_B)
        nn(x)  →  poly(x_A) + poly(x_B) + nn(x_A)*nn(x_B)

    by solving for polynomials P_A(x_A) + P_B(x_B) that cancel the additive
    contamination preventing multiplicative separability. Uses alternating minimisation
    to fit both counterterm polynomials.

    This is particularly useful after an outer y-transform (e.g. square(y) for
    sqrt(·) outputs) has already exposed an interior structure of the form

        u(x) = P_A(x_A) + P_B(x_B) + g(x_A) h(x_B).

    Pattern label: counterterm_mul_split
    """

    name = "counterterm_mul_split"

    def iter_targets(self, ctx: StageBContext):
        _sync_stageb_rules_compat_overrides()
        # Only multivariate NN atoms are eligible.
        return _collect_multivariate_nn_atoms(ctx.state.root)

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        _sync_stageb_rules_compat_overrides()
        if not isinstance(target, AtomNode):
            return []
        if str(target.kind).lower() != "nn":
            return []
        if effective_arity(target) < 2:
            return []

        st = ctx.state

        # Guard: if this leaf is already extremely well-approximated by a low-degree
        # polynomial (typically quadratic) in its *own* inputs, prefer letting the
        # dedicated polynomial rewrite (RuleMultiDNN → quad_poly) handle it. This
        # prevents counterterm factorizations from pre-empting clean polynomial cases
        # like (x1-x0)^2.
        #
        # Important: we only apply this guard if quad_poly is not disabled, otherwise
        # we'd risk skipping a useful structural rewrite with no replacement.
        if "quad_poly" not in ctx.disabled_patterns:
            _us_g = getattr(ctx, "units_spec", None)
            _eu_g = getattr(ctx, "enforce_units", False)
            _homo_g = False
            if _eu_g and _us_g is not None:
                _dimless_g = tuple(0 for _ in _us_g.unit_system.base)
                _homo_g = all(
                    tuple(_us_g.x_dims[int(j)]) != _dimless_g
                    for j in target.var_idxs
                )

            def _quad_guard(_h=_homo_g):
                try:
                    r, _init = _build_quadratic_poly_candidate(
                        root=st.root,
                        target=target,
                        reuse=st.reuse,
                        train_loader=ctx.train_loader_probe,
                        device=ctx.device,
                        dtype=ctx.dtype,
                        degree=2,
                        max_points=4096,
                        rel_rms_threshold=1e-3,
                        homogeneous=_h,
                    )
                    return r is not None
                except Exception:
                    return False

            looks_quadratic = ctx.cached(("guard_quad_for_counterterm", id(target)), _quad_guard)
            if looks_quadratic:
                ctx.log(
                    f"[Stage B]  Skipping counterterm mul split on NN vars={target.var_idxs}: "
                    "leaf is quadratic-like (quad_poly available)."
                )
                return []

        ctx.log(f"[Stage B] Probing counterterm mul split on NN vars={target.var_idxs}")
        m_eff = int(effective_arity(target))
        # Quadratic counterterms cover many cases, but some AIF problems (e.g. Klein–Nishina variants)
        # have a cubic-only additive contamination (e.g. z + z^3) that blocks the rank-1 cross-Hessian test.
        # For bivariate leaves, cheaply extend the search to cubic polynomials.
        if m_eff == 2:
            degrees_A = (2, 3)
            degrees_B = (2, 3)
        else:
            degrees_A = (2,)
            degrees_B = (2,)
        cand_root, init_fn, metadata = _build_counterterm_mul_split_candidate(
            root=st.root,
            target=target,
            model=st.model,
            reuse=st.reuse,
            train_loader=ctx.train_loader_probe,
            device=ctx.device,
            dtype=ctx.dtype,
            degrees_A=degrees_A,
            degrees_B=degrees_B,
            n_alt=10,  # Increased from 3 to allow better convergence
            max_points=4096,
            rel_err_tol=2e-2,  # 2% tolerance (relaxed from 0.5% to accept NN approximation errors)
            ridge=1e-8,
        )
        if cand_root is None:
            return []

        # Option-B refactor: rules must be pure proposal generators.
        # Deduplication/attempt bookkeeping happens in StageBEngine.
        signature = metadata.get("signature", None) if metadata else None

        # Return candidate with metadata from builder
        return [
            Candidate(
                self.name,
                cand_root,
                init_fn,
                meta=metadata if metadata else {"structural": True},
                signature=signature,
            )
        ]


class RuleHomogeneityPeel(StageBRule):
    """
    Detect and exploit homogeneous functions of non-zero degree k.

    When a bivariate NN leaf f(xi, xj) is homogeneous of degree k != 0:
        xi * ∂f/∂xi + xj * ∂f/∂xj = k * f

    This rule rewrites:
        NN(xi, xj) → xi^k * NN_univariate(xj/xi)

    The residual univariate NN can then be collapsed by existing ratio-invariance
    logic or polynomial rewrites.

    This is useful for expressions like:
        f(x0, x1) = x0 * (1 + x1/x0)^2  (degree 1)
        f(x0, x1) = x0^2 * sin(x1/x0)   (degree 2)

    Pattern label: homogeneity_peel
    """

    name = "homogeneity_peel"

    def iter_targets(self, ctx: StageBContext):
        """Return NN atoms where a homogeneous ratio peel is plausible."""
        _sync_stageb_rules_compat_overrides()
        atoms = _collect_multivariate_nn_atoms(ctx.state.root)
        return [
            a
            for a in atoms
            if (
                effective_arity(a) == 2
                or (has_nontrivial_input(a) and effective_arity(a) >= 3)
            )
        ]

    def _propose_compound_product_ratio(
        self,
        ctx: StageBContext,
        target: AtomNode,
        X_in: torch.Tensor,
        f: torch.Tensor,
        input_exprs: Tuple[Node, ...],
    ) -> List[Candidate]:
        """Try q^k * NN(product(extra inputs) / q) for compound atoms.

        This covers Planck/Gaussian-scale shapes after Stage A has already
        surfaced one compound coordinate, e.g. NN[z=x2*x3, x0, x1] where the
        natural residual coordinate is (x0*x1)/z.
        """
        if not has_nontrivial_input(target):
            return []
        m_eff = int(X_in.shape[1])
        if m_eff < 3 or len(input_exprs) != m_eff:
            return []

        st = ctx.state
        kw = getattr(target, "kwargs", None) or {}
        nn_kwargs: Dict[str, Any] = {}
        for kk in ("num_segments", "dual_layer", "seg_width"):
            if kk in kw:
                nn_kwargs[kk] = kw[kk]

        thr = 0.05
        degree_grid = (1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0)
        proposals: List[Tuple[float, int, Tuple[int, ...], float, Candidate]] = []

        for power_dim in range(m_eff):
            other_dims = [i for i in range(m_eff) if i != power_dim]
            numerator_sets: List[Tuple[int, ...]] = []
            for size in range(1, min(2, len(other_dims)) + 1):
                for start_i, first in enumerate(other_dims):
                    if size == 1:
                        numerator_sets.append((first,))
                    else:
                        for second in other_dims[start_i + 1:]:
                            numerator_sets.append((first, second))

            q_vals = X_in[:, int(power_dim)]
            q_abs_ok = q_vals.abs() > 1e-12
            if int(q_abs_ok.sum().item()) < 512:
                continue

            for numerator_dims in numerator_sets:
                if not _homogeneity_product_ratio_units_ok(
                    ctx,
                    target,
                    power_dim=power_dim,
                    numerator_dims=numerator_dims,
                ):
                    continue

                numerator_vals = torch.ones_like(q_vals)
                for idx in numerator_dims:
                    numerator_vals = numerator_vals * X_in[:, int(idx)]

                numerator_expr = _multiply_exprs(
                    tuple(input_exprs[int(idx)] for idx in numerator_dims)
                )
                if numerator_expr is None:
                    continue

                for degree in degree_grid:
                    if not float(degree).is_integer():
                        if float(q_vals.min().item()) <= 0.0:
                            continue
                    try:
                        ratio_vals = numerator_vals / q_vals
                        norm_vals = f / (torch.pow(q_vals, float(degree)) + 1e-30)
                        collapse_score = _univariate_collapse_score(ratio_vals, norm_vals)
                    except Exception:
                        collapse_score = float("inf")
                    if not math.isfinite(collapse_score) or collapse_score > thr:
                        continue

                    ratio_ast = MulNode(
                        clone_ast(numerator_expr),
                        PowNode(base=clone_ast(input_exprs[int(power_dim)]), exponent=-1.0),
                    )
                    base_tag = getattr(target, "tag", None)
                    nums_tag = "_".join(str(int(i)) for i in numerator_dims)
                    degree_tag = str(float(degree)).replace("-", "m").replace(".", "p")
                    new_tag = (
                        f"{base_tag}_HP_{power_dim}_{nums_tag}_{degree_tag}"
                        if base_tag
                        else f"HP_{power_dim}_{nums_tag}_{degree_tag}"
                    )
                    residual_nn = AtomNode(
                        kind="nn",
                        var_idxs=target.var_idxs,
                        kwargs=nn_kwargs,
                        tag=new_tag,
                        inputs=(ratio_ast,),
                    )
                    if abs(float(degree) - 1.0) < 1e-9:
                        power_factor = clone_ast(input_exprs[int(power_dim)])
                    else:
                        power_factor = PowNode(
                            base=clone_ast(input_exprs[int(power_dim)]),
                            exponent=float(degree),
                        )
                    cand_root = replace_atom_in_ast(
                        st.root,
                        target,
                        MulNode(power_factor, residual_nn),
                    )
                    init_fn = _make_homogeneity_peel_values_init_fn(
                        new_tag=new_tag,
                        degree=float(degree),
                        q_values=q_vals,
                        numerator_values=numerator_vals,
                        f=f,
                        ctx=ctx,
                    )
                    try:
                        numerator_label = "*".join(
                            ast_to_human_readable(input_exprs[int(i)])
                            for i in numerator_dims
                        )
                        power_label = ast_to_human_readable(input_exprs[int(power_dim)])
                    except Exception:
                        numerator_label = "*".join(f"in{int(i)}" for i in numerator_dims)
                        power_label = f"in{int(power_dim)}"

                    metadata = {
                        "pattern": "homogeneity_peel",
                        "degree": float(degree),
                        "power_dim": int(power_dim),
                        "numerator_dims": tuple(int(i) for i in numerator_dims),
                        "structural": True,
                        "separability_like": True,
                        "collapse_score": float(collapse_score),
                        "mode": "compound_product_ratio",
                        "direction": f"({numerator_label})/{power_label}",
                    }
                    sig = (
                        atom_content_hash(target),
                        71237,
                        int(power_dim),
                        len(numerator_dims),
                        *(int(i) for i in numerator_dims),
                        int(round(float(degree) * 1000.0)),
                    )
                    proposals.append(
                        (
                            float(collapse_score),
                            len(numerator_dims),
                            tuple(int(i) for i in numerator_dims),
                            abs(float(degree) - 1.0),
                            Candidate(self.name, cand_root, init_fn, meta=metadata, signature=sig),
                        )
                    )

        proposals.sort(key=lambda item: (item[0], item[1], item[3], item[2]))
        out = [cand for *_prefix, cand in proposals[:4]]
        for cand in out:
            ctx.log(
                "[Stage B] homogeneity_peel product-ratio candidate: "
                f"vars={target.var_idxs}, direction={cand.meta.get('direction')}, "
                f"degree={cand.meta.get('degree'):.3g}, "
                f"collapse={cand.meta.get('collapse_score'):.4f}"
            )
        return out

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        """
        Test if the NN leaf is homogeneous of non-zero degree k.

        Uses the generalized Euler criterion in atom-local coordinates to
        detect homogeneous degree-k functions.  Works for both simple
        bivariate atoms and compound atoms (e.g. NN[(x1-x2), x0]).

        Emits candidates for BOTH ratio directions to let Stage B scoring
        pick the winner.
        """
        _sync_stageb_rules_compat_overrides()
        import torch

        if not isinstance(target, AtomNode):
            return []
        if str(target.kind).lower() != "nn":
            return []
        eff_arity = int(effective_arity(target))
        if eff_arity < 2:
            return []
        if eff_arity > 2 and not has_nontrivial_input(target):
            return []

        st = ctx.state

        # --- get leaf & atom-local data (mirrors RuleProductHomogeneity) ---
        try:
            atom_to_leaf = build_atom_to_leaf_map(st.root, st.model)
            leaf = atom_to_leaf.get(id(target), None)
        except Exception as e:
            ctx.log(f"[Stage B] homogeneity_peel: failed atom_to_leaf map: {e}")
            return []
        if leaf is None:
            return []

        Xs: List[torch.Tensor] = []
        for batch in ctx.train_loader_probe:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            Xs.append(x.to(device=ctx.device, dtype=ctx.dtype))
            if sum(t.size(0) for t in Xs) >= 2048:
                break
        if not Xs:
            return []
        X_full = torch.cat(Xs, dim=0)[:2048]

        try:
            X_in = _build_atom_input_tensor(target, X_full)  # [B, 2]
        except Exception as e:
            ctx.log(f"[Stage B] homogeneity_peel: failed to build leaf-input tensor: {e}")
            return []
        if X_in.ndim != 2 or X_in.size(1) != eff_arity:
            return []

        # Evaluate leaf in atom-local coords.  The classic bivariate path also
        # needs the local gradient; the compound-product path only needs the
        # scatter-collapse certificate.
        try:
            with torch.no_grad():
                f = leaf(X_in)
                if f.dim() == 2:
                    f = f[:, 0]
                else:
                    f = f.view(-1)
        except Exception as e:
            ctx.log(f"[Stage B] homogeneity_peel: eval failed: {e}")
            return []

        m = torch.isfinite(f) & torch.isfinite(X_in).all(dim=1)
        if int(m.sum().item()) < 512:
            return []
        f = f[m]
        X_in = X_in[m]
        input_exprs = get_input_exprs(target)

        if eff_arity > 2:
            return self._propose_compound_product_ratio(ctx, target, X_in, f, input_exprs)

        try:
            g = leaf.grad({"x": X_in}, allow_unused=True)
            if g is None:
                return []
            if g.dim() == 3:
                g = g[:, 0, :]
        except Exception as e:
            ctx.log(f"[Stage B] homogeneity_peel: eval/grad failed: {e}")
            return []

        # Finite-value mask for gradients
        m = torch.isfinite(g).all(dim=1)
        if int(m.sum().item()) < 512:
            return []
        f = f[m]
        X_in = X_in[m]
        g = g[m]
        f_scale = f.abs().median().clamp_min(1e-12)

        # --- Euler test: X_in[:,0]*g[:,0] + X_in[:,1]*g[:,1] = k*f ---
        lhs = X_in[:, 0] * g[:, 0] + X_in[:, 1] * g[:, 1]
        m0 = torch.isfinite(lhs) & (f.abs() > 1e-12)
        if int(m0.sum().item()) < 512:
            return []

        k_hat = float(torch.median((lhs[m0] / f[m0]).clamp(-50.0, 50.0)).item())

        base_degs = [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 0.5, -0.5]
        degs = set(float(d) for d in base_degs)
        for den in (1, 2, 3, 4):
            degs.add(round(k_hat * den) / den)

        thr = 0.05
        best_k: Optional[float] = None
        best_s = float("inf")
        for k in degs:
            if abs(k) < 1e-12:
                continue  # degree-0 is ratio-invariance, handled elsewhere
            s = float(((lhs - k * f).abs().median() / f_scale).item())
            if s < best_s:
                best_s = s
                best_k = k

        if best_k is None or best_s > thr:
            return []

        degree = float(best_k)
        ctx.log(
            f"[Stage B] Homogeneity degree-{degree:.2g} detected for NN vars={target.var_idxs}: "
            f"euler_residual={best_s:.4f}"
        )

        # --- Build candidates for BOTH ratio directions ---
        degree_is_fractional = not float(degree).is_integer()
        kw = getattr(target, "kwargs", None) or {}
        is_compound = has_nontrivial_input(target)

        # Direction ordering: prefer the ratio direction with values closer to O(1)
        dirs = [(0, 1), (1, 0)]  # (power_dim, ratio_dim) in atom-local space
        try:
            eps = 1e-12

            def _ratio_med(p: int, r: int) -> float:
                den = X_in[:, p].abs().clamp_min(eps)
                num = X_in[:, r].abs()
                return float(torch.median(num / den).item())

            dirs.sort(key=lambda pr: _ratio_med(pr[0], pr[1]))
        except Exception:
            pass

        candidates: List[Candidate] = []
        for power_dim, ratio_dim in dirs:
            if not _homogeneity_ratio_units_ok(
                ctx,
                target,
                power_dim=power_dim,
                ratio_dim=ratio_dim,
            ):
                continue

            # Gate fractional degrees: power variable must be strictly positive
            if degree_is_fractional:
                min_val = float(X_in[:, power_dim].min().item())
                if min_val <= 0:
                    ctx.log(
                        f"[Stage B] homogeneity_peel: skipping power_dim={power_dim} for "
                        f"fractional degree {degree:.2g} (min={min_val:.3g} <= 0)"
                    )
                    continue

            # Certificate: after peeling q^k, the target must collapse onto
            # a univariate function of ratio_dim / power_dim.  This is the
            # concrete payoff for the counter-factor/homogeneity move.
            try:
                q_vals = X_in[:, int(power_dim)]
                r_vals = X_in[:, int(ratio_dim)] / q_vals
                norm_vals = f / (torch.pow(q_vals, float(degree)) + 1e-30)
                collapse_score = _univariate_collapse_score(r_vals, norm_vals)
            except Exception:
                collapse_score = float("inf")
            if not math.isfinite(collapse_score) or collapse_score > thr:
                ctx.log(
                    "[Stage B] homogeneity_peel: normalized ratio collapse failed "
                    f"for power_dim={power_dim}, ratio_dim={ratio_dim}, "
                    f"score={collapse_score:.4f} > {thr:.4f}"
                )
                continue

            # Build the AST: input_exprs[power_dim]^k * NN(input_exprs[ratio_dim] / input_exprs[power_dim])
            ratio_num_expr = clone_ast(input_exprs[ratio_dim])

            # ratio = ratio_num / power
            # We build it via build_monomial_ast for simple atoms, or
            # MulNode(ratio_num, PowNode(power, -1)) for compound.
            if not is_compound:
                # Simple atom: input_exprs are Var(i), Var(j)
                pv = int(target.var_idxs[power_dim])
                rv = int(target.var_idxs[ratio_dim])
                ratio_ast = build_monomial_ast(
                    var_idxs=(rv, pv),
                    exponents=(1, -1),
                )
            else:
                # Compound atom: build ratio from AST expressions
                ratio_ast = MulNode(
                    ratio_num_expr,
                    PowNode(base=clone_ast(input_exprs[power_dim]), exponent=-1.0),
                )

            # Build NN kwargs from original
            nn_kwargs: Dict[str, Any] = {}
            for kk in ("num_segments", "dual_layer", "seg_width"):
                if kk in kw:
                    nn_kwargs[kk] = kw[kk]

            # Tag for teacher-based univariate rewrites
            base_tag = getattr(target, "tag", None)
            new_tag = (
                f"{base_tag}_H_{power_dim}_{ratio_dim}"
                if base_tag
                else f"H_{power_dim}_{ratio_dim}"
            )

            compound_nn = AtomNode(
                kind="nn",
                var_idxs=target.var_idxs,  # keep all vars for data routing
                kwargs=nn_kwargs,
                tag=new_tag,
                inputs=(ratio_ast,),
            )

            # Power factor: power_expr^k
            if abs(degree - 1.0) < 1e-9:
                power_factor = clone_ast(input_exprs[power_dim])
            else:
                power_factor = PowNode(
                    base=clone_ast(input_exprs[power_dim]),
                    exponent=degree,
                )

            new_subtree = MulNode(left=power_factor, right=compound_nn)
            cand_root = replace_atom_in_ast(st.root, target, new_subtree)

            metadata = {
                "pattern": "homogeneity_peel",
                "degree": degree,
                "power_dim": power_dim,
                "ratio_dim": ratio_dim,
                "structural": True,
                "separability_like": True,
                "score": best_s,
                "collapse_score": float(collapse_score),
                "mode": "compound" if is_compound else "plain",
            }

            ctx.log(
                f"[Stage B] homogeneity_peel candidate: vars={target.var_idxs}, "
                f"power_dim={power_dim}, ratio_dim={ratio_dim}, degree={degree:.2g}, "
                f"score={best_s:.4f}, collapse={collapse_score:.4f}, "
                f"mode={'compound' if is_compound else 'plain'}"
            )

            init_fn = _make_homogeneity_peel_init_fn(
                target=target,
                new_tag=new_tag,
                power_dim=power_dim,
                ratio_dim=ratio_dim,
                degree=degree,
                X_in=X_in,
                f=f,
                ctx=ctx,
            )

            candidates.append(
                Candidate(self.name, cand_root, init_fn, meta=metadata)
            )

        return candidates


class RuleProductHomogeneity(StageBRule):
    """
    Detect patterns of form f(xi, xj) = xi^k * h(xi*xj) (product inside).

    When a bivariate NN leaf has the form f(xi, xj) = xi^k * h(xi*xj),
    the function h depends only on the product w = xi*xj (not a ratio).

    Mathematical test: For f = xi^k * h(xi*xj):
        xj * ∂f/∂xj = xi * ∂f/∂xi - k*f

    This rule rewrites:
        NN(xi, xj) → xi^k * NN_1D(xi*xj)

    This is useful for expressions like:
        f(x, z) = x * tanh(x*z)  (degree 1)
        f(x, z) = x^2 * sin(x*z) (degree 2)

    Pattern label: product_homogeneity
    """

    name = "product_homogeneity"

    def iter_targets(self, ctx: StageBContext):
        """Return all bivariate NN atoms in the current AST."""
        _sync_stageb_rules_compat_overrides()
        atoms = _collect_multivariate_nn_atoms(ctx.state.root)
        return [a for a in atoms if effective_arity(a) == 2]

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        _sync_stageb_rules_compat_overrides()
        import torch

        from nestynet_sr.sr_core.bridges import clone_ast
        if not isinstance(target, AtomNode):
            return []
        if str(target.kind).lower() != "nn":
            return []
        if effective_arity(target) != 2:
            return []
        st = ctx.state
        try:
            atom_to_leaf = build_atom_to_leaf_map(st.root, st.model)
            leaf = atom_to_leaf.get(id(target), None)
        except Exception as e:
            ctx.log(f"[Stage B] product_homogeneity: failed atom_to_leaf map: {e}")
            return []
        if leaf is None:
            return []
        Xs = []
        for batch in ctx.train_loader_probe:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            Xs.append(x.to(device=ctx.device, dtype=ctx.dtype))
            if sum(t.size(0) for t in Xs) >= 2048:
                break
        if not Xs:
            return []
        X_full = torch.cat(Xs, dim=0)[:2048]
        try:
            X_in = _build_atom_input_tensor(target, X_full)
        except Exception as e:
            ctx.log(f"[Stage B] product_homogeneity: failed to build leaf-input tensor: {e}")
            return []
        if X_in.ndim != 2 or X_in.size(1) != 2:
            return []
        try:
            with torch.no_grad():
                f = leaf(X_in)
                if f.dim() == 2:
                    f = f[:, 0]
                else:
                    f = f.view(-1)
            g = leaf.grad({"x": X_in}, allow_unused=True)
            if g is None:
                return []
            if g.dim() == 3:
                g = g[:, 0, :]
        except Exception as e:
            ctx.log(f"[Stage B] product_homogeneity: eval/grad failed: {e}")
            return []
        m = torch.isfinite(f) & torch.isfinite(X_in).all(dim=1) & torch.isfinite(g).all(dim=1)
        if int(m.sum().item()) < 512:
            return []
        f = f[m]
        X_in = X_in[m]
        g = g[m]
        f_scale = f.abs().median().clamp_min(1e-12)
        kw = getattr(target, "kwargs", None) or {}
        input_exprs = tuple(get_input_exprs(target))
        base_degs = [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 0.5, -0.5]
        thr = 0.05
        def _best_degree(power_dim: int, other_dim: int):
            a = X_in[:, power_dim]
            b = X_in[:, other_dim]
            lhs = a * g[:, power_dim] - b * g[:, other_dim]
            m0 = torch.isfinite(lhs) & torch.isfinite(f) & (f.abs() > 1e-12)
            if int(m0.sum().item()) < 512:
                return None
            k_hat = float(torch.median((lhs[m0] / f[m0]).clamp(-50.0, 50.0)).item())
            degs = set(float(d) for d in base_degs)
            for den in (1, 2, 3, 4):
                degs.add(round(k_hat * den) / den)
            best_k = None
            best_s = float("inf")
            for k in degs:
                s = float(((lhs - k * f).abs().median() / f_scale).item())
                if s < best_s:
                    best_s = s
                    best_k = k
            return best_s, best_k
        cands: List[Candidate] = []
        if has_nontrivial_input(target):
            if len(input_exprs) != 2:
                return []
            nn_kwargs: Dict[str, Any] = {}
            for kk in ("num_segments", "dual_layer", "seg_width"):
                if kk in kw:
                    nn_kwargs[kk] = kw[kk]
            for p, q in [(0, 1), (1, 0)]:
                out = _best_degree(p, q)
                if out is None:
                    continue
                score, k = out
                if score > thr:
                    continue
                if (not float(k).is_integer()) and float(X_in[:, p].min().item()) <= 0:
                    continue
                prod_expr = _multiply_exprs((input_exprs[p], input_exprs[q]))
                if prod_expr is None:
                    continue
                new_nn = AtomNode(
                    kind="nn",
                    var_idxs=target.var_idxs,
                    kwargs=nn_kwargs,
                    tag=None,
                    inputs=(prod_expr,),
                )
                if abs(float(k)) < 1e-12:
                    new_sub = new_nn
                else:
                    power_base = clone_ast(input_exprs[p])
                    pf = (
                        power_base
                        if abs(float(k) - 1.0) < 1e-9
                        else PowNode(power_base, float(k))
                    )
                    new_sub = MulNode(pf, new_nn)
                cand_root = replace_atom_in_ast(st.root, target, new_sub)
                try:
                    power_label = ast_to_human_readable(input_exprs[p])
                    other_label = ast_to_human_readable(input_exprs[q])
                except Exception:
                    power_label = f"input{p}"
                    other_label = f"input{q}"
                meta = {
                    "pattern": "product_homogeneity",
                    "degree": float(k),
                    "power_dim": int(p),
                    "product_dim": int(q),
                    "structural": True,
                    "score": float(score),
                    "mode": "compound_effective",
                    "direction": f"{power_label} * NN({power_label}*{other_label})",
                }
                ctx.log(
                    "[Stage B] Product-homogeneity detected (compound effective inputs) "
                    f"vars={target.var_idxs}, power={power_label}, degree={k:.2g}, "
                    f"score={score:.4f}"
                )
                sig = (
                    atom_content_hash(target),
                    81721,
                    int(p),
                    int(q),
                    int(round(float(k) * 1000.0)),
                )
                cands.append(Candidate(self.name, cand_root, None, meta=meta, signature=sig))
            return cands
        cols = [int(j) for j in target.var_idxs]
        if len(cols) != 2:
            return []
        for p, q in [(0, 1), (1, 0)]:
            out = _best_degree(p, q)
            if out is None:
                continue
            score, k = out
            if score > thr:
                continue
            power_var = cols[p]
            prod_var = cols[q]
            if (not float(k).is_integer()) and float(X_full[:, power_var].min().item()) <= 0:
                continue
            ctx.log(f"[Stage B] Product-homogeneity detected vars={target.var_idxs}, power=x{power_var}, other=x{prod_var}, degree={k:.2g}, score={score:.4f}")
            cand_root, init_fn, meta = _build_product_homogeneity_candidate(
                root=st.root, target=target, reuse=st.reuse, train_loader=ctx.train_loader_probe,
                device=ctx.device, dtype=ctx.dtype, degree=float(k), power_var_idx=power_var, product_var_idx=prod_var
            )
            if cand_root is None:
                continue
            if meta is None:
                meta = {"structural": True, "pattern": "product_homogeneity"}
            meta.update({"degree": float(k), "power_var_idx": power_var, "product_var_idx": prod_var, "score": float(score), "mode": "plain"})
            cands.append(Candidate(self.name, cand_root, init_fn, meta=meta))
        return cands




class RuleRatioInvariance(StageBRule):
    """
    Detect and exploit ratio-invariance (homogeneous degree-0 functions).

    When an NN leaf f(xi, xj, ...) depends on two of its variables only
    through their ratio r = xj/xi, it satisfies the Euler criterion:
        xi·∂f/∂xi + xj·∂f/∂xj = 0.

    **Bivariate** (len == 2):
        NN(xi, xj) → (poly(xj/xi))^exponent

    **≥3-var** (len >= 3):
        NN(xi, xj, xk, ...) → NN[r=xj/xi, xk, ...]   (arity reduction)
        Collapses the ratio-invariant pair into a compound variable,
        reducing effective arity by 1.

    Pattern label: ratio_invariance
    """

    name = "ratio_invariance"

    def iter_targets(self, ctx: StageBContext):
        """Return all NN atoms with effective arity ≥2 in the current AST."""
        _sync_stageb_rules_compat_overrides()
        atoms = _collect_multivariate_nn_atoms(ctx.state.root)
        return [a for a in atoms if effective_arity(a) >= 2]

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        """
        Test if any variable pair in the NN leaf is ratio-invariant.

        Uses the Euler derivative criterion to detect homogeneous degree-0
        pairs in the leaf's effective inputs. For bivariate leaves, builds
        poly(ratio)^exponent candidates. For ≥3-input leaves, collapses the
        detected pair into a new compound variable.
        """
        _sync_stageb_rules_compat_overrides()
        if not isinstance(target, AtomNode):
            return []
        if str(target.kind).lower() != "nn":
            return []
        eff_arity = int(effective_arity(target))
        if eff_arity < 2:
            return []

        st = ctx.state
        try:
            atom_to_leaf = build_atom_to_leaf_map(st.root, st.model)
        except Exception as e:
            ctx.log(f"[Stage B] ratio_invariance: failed to build subtree model: {e}")
            return []
        leaf = atom_to_leaf.get(id(target), None)
        if leaf is None:
            return []

        Xs: List[torch.Tensor] = []
        for batch in ctx.train_loader_probe:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            Xs.append(x.to(device=ctx.device, dtype=ctx.dtype))
            if sum(t.size(0) for t in Xs) >= 2048:
                break
        if not Xs:
            return []
        X_full = torch.cat(Xs, dim=0)[:2048]

        try:
            X_in = _build_atom_input_tensor(target, X_full)
        except Exception as e:
            ctx.log(f"[Stage B] ratio_invariance: failed to build leaf-input tensor: {e}")
            return []
        if X_in.ndim != 2 or X_in.size(1) != eff_arity:
            return []

        try:
            with torch.no_grad():
                f = leaf(X_in)
                if f.dim() == 2:
                    f = f[:, 0]
                else:
                    f = f.view(-1)
            g = leaf.grad({"x": X_in}, allow_unused=True)
            if g is None:
                return []
            if g.dim() == 3:
                g = g[:, 0, :]
        except Exception as e:
            ctx.log(f"[Stage B] ratio_invariance: eval/grad failed: {e}")
            return []

        m = (
            torch.isfinite(f)
            & torch.isfinite(X_in).all(dim=1)
            & torch.isfinite(g).all(dim=1)
        )
        if int(m.sum().item()) < 512:
            return []
        f = f[m]
        X_in = X_in[m]
        g = g[m]
        input_exprs = get_input_exprs(target)

        # --- Bivariate path: existing poly(ratio)^exponent candidates ---
        if eff_arity == 2:
            return self._propose_bivariate(ctx, target, X_in, f, g, input_exprs)

        # --- ≥3-input path: compound variable collapse ---
        return self._propose_multivar(ctx, target, X_in, f, g, input_exprs)

    # ------------------------------------------------------------------
    # Bivariate: unchanged logic factored into a helper
    # ------------------------------------------------------------------
    def _propose_bivariate(
        self,
        ctx: StageBContext,
        target: AtomNode,
        X_in: torch.Tensor,
        f: torch.Tensor,
        g: torch.Tensor,
        input_exprs: Tuple[Node, ...],
    ) -> List[Candidate]:
        st = ctx.state
        try:
            input_labels = tuple(ast_to_human_readable(inp) for inp in input_exprs)
        except Exception:
            input_labels = tuple(f"in{i}" for i in range(len(input_exprs)))

        candidates = []
        for xi_local_idx, xj_local_idx in ((0, 1), (1, 0)):
            result = _check_ratio_invariance_on_leaf_inputs(
                X_in,
                f,
                g,
                xi_local_idx,
                xj_local_idx,
                threshold=0.05,
            )
            if not result["ok"]:
                continue
            direction = f"{input_labels[xj_local_idx]}/{input_labels[xi_local_idx]}"

            ctx.log(
                f"[Stage B] Ratio-invariance detected for NN inputs={direction} "
                f"(vars={target.var_idxs}): euler_score={float(result['euler_score']):.4f}"
            )

            for exponent in [-0.5, 0.5, 1.0, -1.0, -2.0]:
                for degree in [2, 3, 4]:
                    cand_root, init_fn = _build_ratio_invariance_candidate(
                        root=st.root,
                        target=target,
                        reuse=st.reuse,
                        train_loader=ctx.train_loader_probe,
                        device=ctx.device,
                        dtype=ctx.dtype,
                        xi_local_idx=xi_local_idx,
                        xj_local_idx=xj_local_idx,
                        degree=degree,
                        exponent=exponent,
                        min_points=400,
                        rel_rms_threshold=0.02,
                    )

                    if cand_root is None:
                        continue

                    ctx.log(
                        f"[Stage B] ratio_invariance candidate: "
                        f"vars={target.var_idxs}, deg={degree}, exp={exponent}"
                    )

                    candidates.append(
                        Candidate(
                            self.name,
                            cand_root,
                            init_fn,
                            meta={
                                "pattern": "ratio_invariance",
                                "direction": direction,
                                "local_pair": (xi_local_idx, xj_local_idx),
                                "degree": degree,
                                "exponent": exponent,
                            },
                        )
                    )

        return candidates

    # ------------------------------------------------------------------
    # ≥3-var: collapse ratio-invariant pair into a compound variable
    # ------------------------------------------------------------------
    def _propose_multivar(
        self,
        ctx: StageBContext,
        target: AtomNode,
        X_in: torch.Tensor,
        f: torch.Tensor,
        g: torch.Tensor,
        input_exprs: Tuple[Node, ...],
    ) -> List[Candidate]:
        st = ctx.state
        candidates = []

        try:
            input_labels = tuple(ast_to_human_readable(inp) for inp in input_exprs)
        except Exception:
            input_labels = tuple(f"in{i}" for i in range(len(input_exprs)))

        m_eff = int(X_in.shape[1])
        # Try every ordered pair (den, num) looking for ratio-invariance in local coords
        for xi_local_idx in range(m_eff):
            for xj_local_idx in range(m_eff):
                if xi_local_idx == xj_local_idx:
                    continue
                result = _check_ratio_invariance_on_leaf_inputs(
                    X_in,
                    f,
                    g,
                    xi_local_idx,
                    xj_local_idx,
                    threshold=0.05,
                )
                if not result["ok"]:
                    continue
                direction = f"{input_labels[xj_local_idx]}/{input_labels[xi_local_idx]}"

                ctx.log(
                    f"[Stage B] Ratio-invariance detected in ≥3-input NN "
                    f"vars={target.var_idxs}: pair {direction}, "
                    f"euler_score={float(result['euler_score']):.4f}"
                )

                # Build compound collapse candidate:
                #   NN[input_0,...] → NN[r=input_j/input_i, remaining_inputs...]
                remaining_inputs = tuple(
                    clone_ast(inp)
                    for idx, inp in enumerate(input_exprs)
                    if idx not in (xi_local_idx, xj_local_idx)
                )
                ratio_ast = MulNode(
                    clone_ast(input_exprs[xj_local_idx]),
                    PowNode(base=clone_ast(input_exprs[xi_local_idx]), exponent=-1.0),
                )

                # Build compound NN atom
                nn_kwargs: Dict[str, Any] = {}
                # Copy relevant hyperparams from original target
                if target.kwargs:
                    for key in ("num_segments", "dual_layer", "seg_width"):
                        if key in target.kwargs:
                            nn_kwargs[key] = target.kwargs[key]

                # Build inputs tuple: ratio expr + remaining effective inputs
                nn_inputs: Tuple[Node, ...] = (ratio_ast,) + remaining_inputs

                # Tag for teacher extraction in subsequent rewrites
                base_tag = getattr(target, "tag", None)
                new_tag = (
                    f"{base_tag}_R_{xj_local_idx}_{xi_local_idx}"
                    if base_tag
                    else f"R_{xj_local_idx}_{xi_local_idx}"
                )

                compound_nn = AtomNode(
                    kind="nn",
                    var_idxs=target.var_idxs,  # keep all for data routing
                    kwargs=nn_kwargs,
                    tag=new_tag,
                    inputs=nn_inputs,
                )

                cand_root = _replace_node_in_ast(st.root, target, compound_nn)

                # No custom init — the NN will be retrained via LM from
                # its current weights, which already encode the function.
                # The leaf dimensionality changes (n_in decreases by 1)
                # so existing weights are not directly reusable; LM will
                # handle the fresh fit.

                sig = (atom_content_hash(target), int(xi_local_idx), int(xj_local_idx))
                candidates.append(
                    Candidate(
                        self.name,
                        cand_root,
                        None,  # init_fn: retrain from scratch
                        meta={
                            "pattern": "ratio_invariance",
                            "ratio_collapse": True,
                            "structural": True,
                            "direction": direction,
                            "local_pair": (xi_local_idx, xj_local_idx),
                            "euler_score": float(result["euler_score"]),
                        },
                        signature=sig,
                    )
                )

                # Return on first hit — one ratio collapse per iteration
                # is enough; further pairs can be caught in the next pass.
                return candidates

        return candidates


class RuleCoupledLeafRatio(StageBRule):
    """
    Detect when two NN leaves have a simple polynomial ratio.

    When Stage B reaches a structure like:
        (poly0(x0) * F(x1,x2)) + (poly1(x3) * G(x1,x2))

    and F/G is a simple polynomial (e.g., -x2/x1²), we can factor out the
    common structure by replacing F with poly * G.

    This is particularly useful for the Lorentz velocity addition formula:
        y = (x3 - x0*x2/x1²) / sqrt(1 - x2²/x1²)

    where F/G = -x2/x1² is a simple monomial.

    Pattern label: coupled_leaf_ratio
    """

    name = "coupled_leaf_ratio"

    def iter_targets(self, ctx: StageBContext):
        """Return all pairs of bivariate NN atoms with shared variables."""
        _sync_stageb_rules_compat_overrides()
        atoms = _collect_multivariate_nn_atoms(ctx.state.root)
        # Only bivariate NN leaves
        bivar_atoms = [a for a in atoms if len(a.var_idxs) == 2]

        # Find pairs with shared variables
        pairs = []
        for i, a1 in enumerate(bivar_atoms):
            vars1 = set(int(v) for v in a1.var_idxs)
            for a2 in bivar_atoms[i + 1 :]:
                vars2 = set(int(v) for v in a2.var_idxs)
                shared = vars1 & vars2
                if shared:
                    pairs.append((a1, a2, shared))

        return pairs

    def propose(self, ctx: StageBContext, target) -> List[Candidate]:
        """
        Test if F/G is a simple polynomial for each pair of NN leaves.

        Uses derivative ratios ∂y/∂x_affine to compute F/G without isolating
        individual leaf outputs.
        """
        _sync_stageb_rules_compat_overrides()
        if not isinstance(target, tuple) or len(target) != 3:
            return []

        target_F, target_G, shared_vars = target

        if not isinstance(target_F, AtomNode) or not isinstance(target_G, AtomNode):
            return []

        st = ctx.state

        # Find variables that appear in additive terms with the NN leaves
        # For the pattern: poly0(x0) * F + poly1(x3) * G
        # we need to identify x0 and x3 as "affine" variables

        # Search for additive structure containing these NN leaves
        def find_affine_var_for_nn(root: Node, nn_atom: AtomNode) -> Optional[int]:
            """
            Find the affine variable index for a NN atom in an additive structure.
            Looks for patterns like: poly(x_i) * NN(...) where poly is linear.
            """

            def search(n: Node) -> Optional[int]:
                if isinstance(n, MulNode):
                    # Check if one child is the NN and the other is a linear poly
                    left_is_nn = n.left is nn_atom
                    right_is_nn = n.right is nn_atom
                    if left_is_nn or right_is_nn:
                        poly_side = n.right if left_is_nn else n.left
                        if isinstance(poly_side, AtomNode) and str(poly_side.kind).lower() in ("poly", "polynomial", "rpoly", "rpolynomial", "r_polynomial"):
                            if effective_arity(poly_side) == 1:
                                return int(poly_side.var_idxs[0])
                    result = search(n.left)
                    if result is not None:
                        return result
                    return search(n.right)
                if isinstance(n, AddNode):
                    result = search(n.left)
                    if result is not None:
                        return result
                    return search(n.right)
                if isinstance(n, PowNode):
                    return search(n.base)
                if isinstance(n, _UNARY_AST_NODES):
                    return search(n.arg)
                return None

            return search(root)

        affine_F = find_affine_var_for_nn(st.root, target_F)
        affine_G = find_affine_var_for_nn(st.root, target_G)

        if affine_F is None or affine_G is None:
            ctx.log(
                f"[Stage B] coupled_leaf_ratio: no affine vars found for NN pair "
                f"F={target_F.var_idxs}, G={target_G.var_idxs}"
            )
            return []

        ctx.log(
            f"[Stage B] Testing coupled-leaf ratio for NN pair "
            f"F={target_F.var_idxs} (affine={affine_F}), "
            f"G={target_G.var_idxs} (affine={affine_G})"
        )

        # Affinity check: verify linear coefficients are truly constant (d²y/dx²≈0)
        try:
            X_aff = next(iter(ctx.train_loader_probe))[0].to(ctx.device, ctx.dtype)[:512]
            H = st.model.grad_grad(X_aff)  # [B, 1, Nx, Nx]
            d2y_dxF2 = H[:, 0, affine_F, affine_F]
            d2y_dxG2 = H[:, 0, affine_G, affine_G]
            scale_F = H[:, 0, affine_F, :].abs().max().clamp(min=1e-12)
            scale_G = H[:, 0, affine_G, :].abs().max().clamp(min=1e-12)
            rel_F = d2y_dxF2.abs().max() / scale_F
            rel_G = d2y_dxG2.abs().max() / scale_G
            if rel_F > 0.01 or rel_G > 0.01:
                ctx.log(
                    f"[Stage B] coupled_leaf_ratio: affinity check failed - "
                    f"non-constant coefficients (rel_F={rel_F:.4f}, rel_G={rel_G:.4f})"
                )
                return []
        except Exception as e:
            ctx.log(f"[Stage B] coupled_leaf_ratio: affinity check error: {e}")
            # Proceed anyway - the ratio check will reject bad cases

        # Check if F/G is a simple polynomial using full model derivatives
        try:
            result = check_coupled_leaf_ratio_from_derivs(
                model=st.model,
                datagen=ctx.train_loader_probe,
                affine_idx_F=affine_F,
                affine_idx_G=affine_G,
                shared_var_idxs=list(shared_vars),
                device=ctx.device,
                dtype=ctx.dtype,
                threshold=0.02,
                n_points=2048,
            )
        except Exception as e:
            ctx.log(f"[Stage B] coupled_leaf_ratio check failed: {e}")
            return []

        if not result.ok:
            ctx.log(
                f"[Stage B] coupled_leaf_ratio rejected: "
                f"rel_rms={result.ratio_rel_rms:.4f}, form={result.poly_form}"
            )
            return []

        ctx.log(
            f"[Stage B] Coupled-leaf ratio detected: F/G ≈ {result.poly_form}, "
            f"coeffs={result.poly_coeffs}, rel_rms={result.ratio_rel_rms:.4f}"
        )

        # Build the candidate: replace F with poly * G
        cand_root, init_fn = _build_coupled_ratio_candidate(
            root=st.root,
            target_F=target_F,
            target_G=target_G,
            reuse=st.reuse,
            train_loader=ctx.train_loader_probe,
            device=ctx.device,
            dtype=ctx.dtype,
            poly_form=result.poly_form,
            poly_coeffs=result.poly_coeffs,
        )

        if cand_root is None:
            return []

        return [
            Candidate(
                self.name,
                cand_root,
                init_fn,
                meta={
                    "pattern": "coupled_leaf_ratio",
                    "poly_form": result.poly_form,
                    "poly_coeffs": result.poly_coeffs,
                },
            )
        ]
