# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Compound-coordinate Stage-B rewrite rules."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from nestynet_sr.sr_core.atoms import PlanckLeaf
from nestynet_sr.sr_core.bridges import (
    AtomNode,
    MulNode,
    Node,
    ast_to_human_readable,
    clone_ast,
    effective_arity,
    get_input_exprs,
    is_trivial_input,
    replace_atom_in_ast,
)
from nestynet_sr.sr_core.constants import make_unit_aware_scalar_atom as _make_unit_aware_scalar_atom
from nestynet_sr.sr_core.separability_math import (
    build_monomial_ast,
    check_monomial_compound_logderiv,
)
from nestynet_sr.sr_search.compound_proposals import (
    build_barycentric_compound_proposals,
    build_logexp_compound_proposals,
    build_metric_distance_compound_proposals,
    stageB_meta_from_proposal,
)

from .engine import Candidate, StageBContext, StageBRule, atom_content_hash
from .helpers import (
    _collect_all_atoms,
    _collect_multivariate_nn_atoms,
    _set_constant_leaf_value,
    build_atom_to_leaf_map,
)
from .rules_common import _effective_input_dims_for_atom, _subtree_content_hash
from .splits import _gather_nn_atom_value_grad_hess


class RuleCompoundFunctionMacros(StageBRule):
    """Rule: try a small library of *compound-function* macros on NN leaves.

    This is deliberately conservative:
    - expands macros to ordinary AST nodes (no new runtime primitives)
    - screens candidates cheaply on a cached batch (affine fit)
    - returns only a handful of the best candidates

    Pattern labels: cf_<macro_name>
    """

    name = "compound_fn_macros"

    def iter_targets(self, ctx: StageBContext):
        # Apply to both uni- and multi-variate NN atoms; the macro proposer
        # caps the usable variable subset itself.
        return [
            a
            for a in _collect_all_atoms(ctx.state.root)
            if isinstance(a, AtomNode) and str(getattr(a, "kind", "")).lower() == "nn" and len(getattr(a, "var_idxs", ()) or ()) >= 1
        ]

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        if not isinstance(target, AtomNode) or str(getattr(target, "kind", "")).lower() != "nn":
            return []
        # Note: tier-2/tier-3 macro forms (factor(x)*macro, p(x)*(a*macro+b))
        # can produce unitful output via their prefactors.  Don't gate here;
        # the per-candidate check_units_ast precheck handles this correctly.
        try:
            from nestynet_sr.sr_search.compound_functions import propose_compound_function_macros

            return list(propose_compound_function_macros(ctx, target) or [])
        except Exception as e:
            # Don't fail the whole Stage-B pass if macros error-out, but do emit
            # a breadcrumb in verbose mode (this rule is otherwise silent).
            if getattr(ctx, "verbose", False):
                try:
                    ctx.log(f"[Stage B] Rule {self.name} failed: {type(e).__name__}: {e}")
                except Exception:
                    pass
            return []


def _dim_difference(a, b):
    if a is None:
        return None
    if b is None:
        return tuple(a)
    try:
        return tuple(x - y for x, y in zip(tuple(a), tuple(b)))
    except Exception:
        return None


class RuleMetricDistance(StageBRule):
    """Try visible metric-distance closures on NN leaves.

    This catches law-of-cosines / reciprocal-distance atoms such as
    ``1/sqrt(a**2 + b**2 - 2*a*b*cos(theta))`` and Cartesian distance atoms
    built from coordinate differences.  It is a terminal analytic rule: it does
    not commit hidden coordinates, and normal LM validation remains the accept
    gate.
    """

    name = "metric_distance"

    def iter_targets(self, ctx: StageBContext):
        out = []
        for atom in _collect_all_atoms(ctx.state.root):
            if not isinstance(atom, AtomNode):
                continue
            if str(getattr(atom, "kind", "")).lower() != "nn":
                continue
            if int(effective_arity(atom)) >= 2:
                out.append(atom)
        return out

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        if not isinstance(target, AtomNode):
            return []
        if str(getattr(target, "kind", "")).lower() != "nn":
            return []

        units_spec = getattr(ctx, "units_spec", None) if bool(getattr(ctx, "enforce_units", False)) else None
        try:
            proposals = build_metric_distance_compound_proposals(
                target,
                units_spec=units_spec,
                include_polar=True,
                include_cartesian=True,
                wrappers=("q", "sqrt_q", "inv_sqrt_q", "inv_q"),
                max_cartesian_pairs=3,
                max_proposals=20,
            )
        except Exception as e:
            if getattr(ctx, "verbose", False):
                ctx.log(f"[Stage B] Rule {self.name} failed: {type(e).__name__}: {e}")
            return []

        if not proposals:
            return []

        target_dim = None
        try:
            target_dim = ctx.infer_target_dim(target)
            if target_dim is not None:
                target_dim = tuple(target_dim)
        except Exception:
            target_dim = None

        cands: List[Candidate] = []
        wrapper_code = {"q": 1, "sqrt_q": 2, "inv_sqrt_q": 3, "inv_q": 4}
        family_code = {
            "lawcos": 101,
            "lawcos_sq_plus": 102,
            "lawcos_sq_minus": 103,
            "cartdist": 104,
        }

        for mp in proposals:
            scale_dim = _dim_difference(target_dim, mp.z_dim)
            scale_tag = f"metric_{getattr(target, 'tag', 'leaf')}_{mp.family}_{mp.wrapper}"
            try:
                scale_node = _make_unit_aware_scalar_atom(
                    scale_dim,
                    units_spec,
                    base_tag=scale_tag,
                    init=1.0,
                    strict=bool(getattr(ctx, "enforce_units", False)),
                )
            except Exception:
                continue

            try:
                replacement = MulNode(scale_node, clone_ast(mp.z_ast))
                root_new = replace_atom_in_ast(ctx.state.root, target, replacement)
            except Exception:
                continue

            scale_atom_tag = getattr(scale_node, "tag", None)

            def _init_fn(root_inner: Node, model_inner: nn.Module, _tag=scale_atom_tag):
                if _tag is None:
                    return
                try:
                    atom_to_leaf = build_atom_to_leaf_map(root_inner, model_inner)
                except Exception:
                    return
                for atom_inner in _collect_all_atoms(root_inner):
                    if getattr(atom_inner, "tag", None) != _tag:
                        continue
                    leaf = atom_to_leaf.get(id(atom_inner), None)
                    if leaf is not None:
                        _set_constant_leaf_value(leaf, 1.0)
                    return

            try:
                z_desc = ast_to_human_readable(mp.z_ast)
            except Exception:
                z_desc = str(mp.label)
            flat_idxs = []
            meta_src = mp.meta or {}
            if "indices" in meta_src:
                flat_idxs = [int(v) for v in meta_src.get("indices", ())]
            elif "pairs" in meta_src:
                for pair in meta_src.get("pairs", ()):
                    flat_idxs.extend(int(v) for v in pair)
            sig = (
                atom_content_hash(target),
                int(family_code.get(mp.family, 999)),
                int(wrapper_code.get(mp.wrapper, 0)),
                *flat_idxs,
            )
            label = f"metric_{mp.family}_{mp.wrapper}"
            cand_meta = stageB_meta_from_proposal(
                mp,
                pattern="metric_distance",
            )
            cand_meta.update(
                {
                    "structural": True,
                    "terminal_macro": True,
                    "log": (
                        "[Stage B]  Trying metric-distance closure "
                        f"{mp.family}:{mp.wrapper} on NN vars={target.var_idxs}: {z_desc}"
                    ),
                }
            )
            cands.append(
                Candidate(
                    label,
                    root_new,
                    _init_fn,
                    signature=sig,
                    meta=cand_meta,
                )
            )

        return cands


class RuleBarycentricCompound(StageBRule):
    """Try visible weighted-average / barycentric closures on NN leaves."""

    name = "barycentric"

    def iter_targets(self, ctx: StageBContext):
        out = []
        for atom in _collect_all_atoms(ctx.state.root):
            if not isinstance(atom, AtomNode):
                continue
            if str(getattr(atom, "kind", "")).lower() != "nn":
                continue
            if int(effective_arity(atom)) >= 4:
                out.append(atom)
        return out

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        if not isinstance(target, AtomNode):
            return []
        if str(getattr(target, "kind", "")).lower() != "nn":
            return []

        units_spec = getattr(ctx, "units_spec", None) if bool(getattr(ctx, "enforce_units", False)) else None
        try:
            proposals = build_barycentric_compound_proposals(
                target,
                units_spec=units_spec,
                wrappers=("z", "inv_z"),
                max_proposals=20,
            )
        except Exception as e:
            if getattr(ctx, "verbose", False):
                ctx.log(f"[Stage B] Rule {self.name} failed: {type(e).__name__}: {e}")
            return []

        if not proposals:
            return []

        target_dim = None
        try:
            target_dim = ctx.infer_target_dim(target)
            if target_dim is not None:
                target_dim = tuple(target_dim)
        except Exception:
            target_dim = None

        cands: List[Candidate] = []
        for bp in proposals:
            scale_dim = _dim_difference(target_dim, bp.z_dim)
            scale_tag = f"bary_{getattr(target, 'tag', 'leaf')}_{bp.family}_{bp.wrapper}"
            try:
                scale_node = _make_unit_aware_scalar_atom(
                    scale_dim,
                    units_spec,
                    base_tag=scale_tag,
                    init=1.0,
                    strict=bool(getattr(ctx, "enforce_units", False)),
                )
            except Exception:
                continue

            try:
                replacement = MulNode(scale_node, clone_ast(bp.z_ast))
                root_new = replace_atom_in_ast(ctx.state.root, target, replacement)
            except Exception:
                continue
            if root_new is None:
                continue

            scale_atom_tag = getattr(scale_node, "tag", None)

            def _init_fn(root_inner: Node, model_inner: nn.Module, _tag=scale_atom_tag):
                if _tag is None:
                    return
                try:
                    atom_to_leaf = build_atom_to_leaf_map(root_inner, model_inner)
                except Exception:
                    return
                for atom_inner in _collect_all_atoms(root_inner):
                    if getattr(atom_inner, "tag", None) != _tag:
                        continue
                    leaf = atom_to_leaf.get(id(atom_inner), None)
                    if leaf is not None:
                        _set_constant_leaf_value(leaf, 1.0)
                    return

            try:
                z_desc = ast_to_human_readable(bp.z_ast)
            except Exception:
                z_desc = str(bp.label)
            sig = (
                atom_content_hash(target),
                "barycentric",
                str(bp.family),
                str(bp.wrapper),
                tuple(int(v) for v in bp.consumed_inputs),
            )
            label = f"bary_{bp.family}_{bp.wrapper}"
            cand_meta = stageB_meta_from_proposal(bp, pattern="barycentric")
            cand_meta.update(
                {
                    "structural": True,
                    "terminal_macro": True,
                    "log": (
                        "[Stage B]  Trying barycentric closure "
                        f"{bp.family}:{bp.wrapper} on NN vars={target.var_idxs}: {z_desc}"
                    ),
                }
            )
            cands.append(
                Candidate(
                    label,
                    root_new,
                    _init_fn,
                    signature=sig,
                    meta=cand_meta,
                )
            )

        return cands


class RuleLogExpCompound(StageBRule):
    """Try visible log/exp closures on dimensionless compound coordinates."""

    name = "logexp_compound"

    def iter_targets(self, ctx: StageBContext):
        out = []
        for atom in _collect_all_atoms(ctx.state.root):
            if not isinstance(atom, AtomNode):
                continue
            if str(getattr(atom, "kind", "")).lower() != "nn":
                continue
            if int(effective_arity(atom)) >= 1:
                out.append(atom)
        return out

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        if not isinstance(target, AtomNode):
            return []
        if str(getattr(target, "kind", "")).lower() != "nn":
            return []

        units_spec = getattr(ctx, "units_spec", None) if bool(getattr(ctx, "enforce_units", False)) else None
        try:
            proposals = build_logexp_compound_proposals(
                target,
                units_spec=units_spec,
                wrappers=("log", "exp"),
                max_proposals=16,
            )
        except Exception as e:
            if getattr(ctx, "verbose", False):
                ctx.log(f"[Stage B] Rule {self.name} failed: {type(e).__name__}: {e}")
            return []

        if not proposals:
            return []

        target_dim = None
        try:
            target_dim = ctx.infer_target_dim(target)
            if target_dim is not None:
                target_dim = tuple(target_dim)
        except Exception:
            target_dim = None

        cands: List[Candidate] = []
        for lp in proposals:
            scale_dim = _dim_difference(target_dim, lp.z_dim)
            scale_tag = f"logexp_{getattr(target, 'tag', 'leaf')}_{lp.family}_{lp.wrapper}"
            try:
                scale_node = _make_unit_aware_scalar_atom(
                    scale_dim,
                    units_spec,
                    base_tag=scale_tag,
                    init=1.0,
                    strict=bool(getattr(ctx, "enforce_units", False)),
                )
            except Exception:
                continue

            try:
                replacement = MulNode(scale_node, clone_ast(lp.z_ast))
                root_new = replace_atom_in_ast(ctx.state.root, target, replacement)
            except Exception:
                continue
            if root_new is None:
                continue

            scale_atom_tag = getattr(scale_node, "tag", None)

            def _init_fn(root_inner: Node, model_inner: nn.Module, _tag=scale_atom_tag):
                if _tag is None:
                    return
                try:
                    atom_to_leaf = build_atom_to_leaf_map(root_inner, model_inner)
                except Exception:
                    return
                for atom_inner in _collect_all_atoms(root_inner):
                    if getattr(atom_inner, "tag", None) != _tag:
                        continue
                    leaf = atom_to_leaf.get(id(atom_inner), None)
                    if leaf is not None:
                        _set_constant_leaf_value(leaf, 1.0)
                    return

            try:
                z_desc = ast_to_human_readable(lp.z_ast)
            except Exception:
                z_desc = str(lp.label)
            sig = (
                atom_content_hash(target),
                "logexp",
                str(lp.family),
                str(lp.wrapper),
                tuple(int(v) for v in lp.consumed_inputs),
            )
            label = f"logexp_{lp.family}_{lp.wrapper}"
            cand_meta = stageB_meta_from_proposal(lp, pattern="logexp")
            cand_meta.update(
                {
                    "structural": True,
                    "terminal_macro": True,
                    "log": (
                        "[Stage B]  Trying log/exp compound closure "
                        f"{lp.family}:{lp.wrapper} on NN vars={target.var_idxs}: {z_desc}"
                    ),
                }
            )
            cands.append(
                Candidate(
                    label,
                    root_new,
                    _init_fn,
                    signature=sig,
                    meta=cand_meta,
                )
            )

        return cands


class RuleMonomialPrefactorCompound(StageBRule):
    """Detect a monomial compound variable with an *outer monomial prefactor*.

    This targets expressions of the form:

        f(x) ≈ m(x) * g(z),   z = ∏ x_i^{a_i},   m(x) = ∏ x_i^{b_i}

    using log-derivative collinearity (Stage-A style) to recover the exponent
    vector a for z and an orthogonal offset b_perp that can be rounded to a
    simple integer prefactor b. When it succeeds, we immediately propose an
    analytic 1D rewrite g(z) using a Planck-like leaf:

        g(z) ≈ A * z^p / (exp(α z + β) - 1)

    This combination is particularly effective on AI-Feynman #043-style targets.

    Pattern label: planck_compound_prefactor
    """

    name = "monomial_prefactor_compound"

    def iter_targets(self, ctx: StageBContext):
        # Only multivariate *non-compound* NN leaves.
        out = []
        for a in _collect_multivariate_nn_atoms(ctx.state.root):
            try:
                n = len(getattr(a, "var_idxs", ()) or ())
            except Exception:
                continue
            if n < 2:
                continue
            # Keep the combinatorics bounded.
            if n > 8:
                continue
            out.append(a)
        return out

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        if not isinstance(target, AtomNode) or str(getattr(target, "kind", "")).lower() != "nn":
            return []

        # Gather a moderate batch of (x, f, ∂f/∂x) for the *leaf*.
        try:
            gathered = _gather_nn_atom_value_grad_hess(
                root=ctx.state.root,
                model=ctx.state.model,
                atom=target,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
                max_points=4096,
            )
        except Exception:
            gathered = None
        if gathered is None:
            return []
        X, X_raw, u, du, _Hu = gathered
        if X_raw is None or u is None or du is None:
            return []

        m = int(X.shape[1])
        if m < 2 or m > 8:
            return []

        # Run log-derivative monomial compound detection.
        # Use X (local inputs aligned with du) instead of X_raw (base vars only),
        # so that x_vals and dydx_vals have matching column counts.
        try:
            results, sigma_ratio, b_perp = check_monomial_compound_logderiv(
                var_idxs=tuple(range(m)),
                x_vals=X,
                y_vals=u,
                dydx_vals=du,
                max_exponent=3,
                precision=0.05,
            )
        except Exception:
            return []
        # Keep only strong matches. The core detector already ranks proposals,
        # but this threshold helps avoid spending LM budget on spurious integerisations.
        min_conf = 0.55
        if results:
            results = [(a, c) for (a, c) in results if float(c) >= min_conf]
        if not results or b_perp is None:
            return []

        # Helper: integerise b_perp, allowing a small shift along 'a' (absorbed into g(z)).
        def _integerise_b_perp(b_perp_vec, a_vec, max_abs=6, max_shift=4, round_tol=0.35):
            try:
                b = np.asarray(b_perp_vec, dtype=float).reshape(-1)
                a = np.asarray(a_vec, dtype=float).reshape(-1)
                if b.shape[0] != a.shape[0]:
                    return None
            except Exception:
                return None

            best = None
            best_score = float("inf")
            for k in range(-int(max_shift), int(max_shift) + 1):
                b_try = b + float(k) * a
                b_round = np.round(b_try)
                if np.max(np.abs(b_try - b_round)) > float(round_tol):
                    continue
                b_int = b_round.astype(int)
                if np.max(np.abs(b_int)) > int(max_abs):
                    continue
                score = float(np.sum(np.abs(b_int))) + 0.05 * float(np.max(np.abs(b_try - b_round)))
                if score < best_score:
                    best_score = score
                    best = tuple(int(x) for x in b_int.tolist())
            if best is None:
                b_round = np.round(b)
                if np.max(np.abs(b - b_round)) <= float(round_tol):
                    b_int = b_round.astype(int)
                    if np.max(np.abs(b_int)) <= int(max_abs):
                        best = tuple(int(x) for x in b_int.tolist())
            return best

        # Use the reduced Planck helper: p is a scanned structural choice, not
        # an LM-fitted parameter.
        from nestynet_sr.sr_search.fitting_utils import _fit_planck_tail_discrete_power

        target_vars = tuple(int(j) for j in getattr(target, "var_idxs", ()) or ())
        out: List[Candidate] = []

        for a_int, conf in (results[:3] if len(results) > 0 else []):
            try:
                a_int = tuple(int(e) for e in a_int)
            except Exception:
                continue
            if len(a_int) != m:
                continue
            if sum(1 for e in a_int if int(e) != 0) < 2:
                continue

            b_int = _integerise_b_perp(b_perp, a_int, max_abs=max(6, 2 * 3), max_shift=4, round_tol=0.35)
            if b_int is None or len(b_int) != m:
                continue

            # Evaluate z and prefactor directly on the gathered local X_raw.
            try:
                z = torch.ones((X_raw.shape[0],), device=X_raw.device, dtype=X_raw.dtype)
                for j, e in enumerate(a_int):
                    if int(e) == 0:
                        continue
                    z = z * (X_raw[:, j] ** int(e))
                pref = torch.ones_like(z)
                if any(int(e) != 0 for e in b_int):
                    for j, e in enumerate(b_int):
                        if int(e) == 0:
                            continue
                        pref = pref * (X_raw[:, j] ** int(e))
                y_res = (u.reshape(-1) / (pref + 1.0e-30)).reshape(-1)
            except Exception:
                continue

            fit = _fit_planck_tail_discrete_power(
                z, y_res,
                min_points=256,
                tail_fraction=0.5,
                rel_rms_threshold=0.08,
            )
            if fit is None:
                continue
            p_est, a_est, b0, rms_rel_log = fit

            # Build AST: m(x) * planck(z(x)).
            try:
                z_ast = build_monomial_ast(target_vars, a_int)
            except Exception:
                continue

            planck_tag = None
            try:
                if getattr(target, "tag", None) is not None:
                    planck_tag = f"{str(getattr(target, 'tag'))}_planck"
            except Exception:
                planck_tag = None

            planck_atom = AtomNode(
                kind="planck",
                var_idxs=target_vars,
                kwargs={"p": float(p_est)},
                tag=planck_tag,
                inputs=(z_ast,),
            )

            repl: Node = planck_atom
            if any(int(e) != 0 for e in b_int):
                try:
                    pref_ast = build_monomial_ast(target_vars, b_int)
                    repl = MulNode(pref_ast, planck_atom)
                except Exception:
                    repl = planck_atom

            try:
                cand_root = replace_atom_in_ast(ctx.state.root, target, repl)
            except Exception:
                continue

            # Custom init: seed the Planck leaf close to a tail-fit.
            def _init_fn(root_inner: Node, model_inner: nn.Module, _tag=planck_tag, _vars=target_vars, _b0=b0, _p=p_est, _a=a_est):
                try:
                    atoms = _collect_all_atoms(root_inner)
                    leaves = list(getattr(model_inner, "leaf", []) or [])
                    for atom_i, leaf_mod in zip(atoms, leaves):
                        if not isinstance(atom_i, AtomNode) or str(getattr(atom_i, "kind", "")).lower() != "planck":
                            continue
                        if _tag is not None:
                            if str(getattr(atom_i, "tag", "")) != str(_tag):
                                continue
                        else:
                            if tuple(int(j) for j in getattr(atom_i, "var_idxs", ()) or ()) != tuple(int(j) for j in _vars):
                                continue

                        # Analytic leaves are typically wrapped (e.g. AutogradAdaptor), so
                        # unwrap common container attributes before type-checking / init.
                        core = getattr(
                            leaf_mod,
                            "core",
                            getattr(leaf_mod, "model", getattr(leaf_mod, "base_model", leaf_mod)),
                        )

                        if not isinstance(core, PlanckLeaf):
                            continue

                        with torch.no_grad():
                            # Clamp to sane initial values to avoid overflow.
                            p0 = float(max(-8.0, min(8.0, float(_p))))
                            a0 = float(max(1.0e-12, min(1.0e12, float(_a))))
                            b00 = float(max(-20.0, min(20.0, float(_b0))))
                            core.p.copy_(torch.tensor(p0, device=core.p.device, dtype=core.p.dtype))
                            core.log_a.copy_(torch.tensor(math.log(a0), device=core.log_a.device, dtype=core.log_a.dtype).clamp(-20.0, 20.0))
                            core.log_amp.copy_(torch.tensor(b00, device=core.log_amp.device, dtype=core.log_amp.dtype).clamp(-20.0, 20.0))
                        break
                except Exception:
                    return

            sig = (atom_content_hash(target),) + tuple(int(x) for x in a_int) + tuple(int(x) for x in b_int)
            out.append(
                Candidate(
                    "planck_compound_prefactor",
                    cand_root,
                    _init_fn,
                    signature=sig,
                    meta={
                        "structural": True,
                        "min_free_params": 2,
                        "planck_power": float(p_est),
                        "log": (
                            f"[Stage B]  Trying Planck(mon) with prefactor on NN vars={target_vars} "
                            f"a={a_int}, b={b_int}, p={p_est:g}, rms_rel_log≈{rms_rel_log:.3g}, sigma≈{sigma_ratio}"
                        ),
                    },
                )
            )

        return out


class RuleCompoundPlanck(StageBRule):
    """Planck/exponential templates for compound atoms with extra variables.

    Targets atoms like NN[z=(x2*x3), x0, x1] where:
    - z is the compound variable (a product of some subset of vars)
    - x0, x1 are the "extra" raw input axes still fed to the leaf

    This rule proposes Planck templates where the function is univariate in a
    derived feature:

        w = (extra_prod) / z = (x0*x1) / (x2*x3)

    Then fits:
        y ≈ z * A * w^p / (exp(α*w + β) - 1)    [Planck]

    Pattern label: compound_planck
    """

    name = "compound_planck"

    def iter_targets(self, ctx: StageBContext):
        """Return compound-coordinate atoms with at least two effective inputs."""
        out = []
        for a in _collect_all_atoms(ctx.state.root):
            if not isinstance(a, AtomNode) or str(a.kind).lower() != "nn":
                continue
            inputs = get_input_exprs(a)
            if len(inputs) < 2 or len(inputs) > 3:
                continue
            if not any(not is_trivial_input(inp) for inp in inputs):
                continue
            out.append(a)
        return out

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        """Generate Planck candidates for derived features."""
        if not isinstance(target, AtomNode) or str(target.kind).lower() != "nn":
            return []
        # Note: the Planck template builds z_expr * A * w^p / (exp(...) - 1)
        # where z_expr can carry units.  Don't gate here; the per-candidate
        # check_units_ast precheck handles dimensional validation correctly.

        inputs = tuple(get_input_exprs(target))
        if len(inputs) < 2 or len(inputs) > 3:
            return []
        if not any(not is_trivial_input(inp) for inp in inputs):
            return []

        from .helpers import _build_planck_derived_feature_candidate

        st = ctx.state
        cands = []

        def _product_expr(nodes: Tuple[Node, ...]) -> Node:
            if len(nodes) == 1:
                return clone_ast(nodes[0])
            out: Node = MulNode(clone_ast(nodes[0]), clone_ast(nodes[1]))
            for node in nodes[2:]:
                out = MulNode(out, clone_ast(node))
            return out

        seen_specs: set[Tuple[int, int]] = set()
        for pref_idx, z_expr_raw in enumerate(inputs):
            other_inputs = tuple(inp for j, inp in enumerate(inputs) if j != pref_idx)
            if not other_inputs:
                continue
            z_expr = clone_ast(z_expr_raw)
            extra_prod = _product_expr(other_inputs)
            try:
                sig_extra = (
                    _subtree_content_hash(z_expr),
                    _subtree_content_hash(extra_prod),
                )
            except Exception:
                sig_extra = (hash(repr(z_expr)), hash(repr(extra_prod)))
            if sig_extra in seen_specs:
                continue
            seen_specs.add(sig_extra)

            planck_cand = _build_planck_derived_feature_candidate(
                root=st.root,
                target=target,
                extra_prod=extra_prod,
                z_expr=z_expr,
                reuse=st.reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
                label=f"compound_planck[{pref_idx}]",
                signature_extra=sig_extra,
            )
            if planck_cand is not None:
                cands.append(planck_cand)

        return cands


class RuleNonlinearSubstitution(StageBRule):
    """Detect nonlinear variable substitutions that render an NN leaf rational.

    For each multivariate NN leaf, try replacing each input variable v
    with T(v) for T in {cos, sin, exp, log}. If the leaf becomes a
    low-degree rational P(..)/Q(..) in the transformed coordinates,
    propose a RationalPolyLeaf candidate.

    Cheap screening (parity + SVD rational probe) runs before any LM fit.

    Pattern labels: nls_cos, nls_sin, nls_exp, nls_log
    """

    name = "nonlinear_substitution"
    # When a single multivariate NN target remains, evaluate all generated
    # nonlinear-substitution variants (instead of greedy first-accept) so the
    # selector can choose the simplest below-floor candidate.
    exhaustive = True

    def iter_targets(self, ctx: StageBContext):
        return _collect_multivariate_nn_atoms(ctx.state.root)

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        if not isinstance(target, AtomNode) or target.kind.lower() != "nn":
            return []

        st = ctx.state
        tag = target.tag
        if tag is None or tag not in st.reuse:
            return []

        teacher = st.reuse[tag]
        print(f"[Stage B] RuleNonlinearSubstitution: screening NN vars={target.var_idxs}, tag={tag}")

        from nestynet_sr.sr_search.candidate_builders import (
            _build_nonlinear_sub_candidate,
            _gather_atom_teacher_data,
        )
        from nestynet_sr.sr_search.fitting_utils import _nonlinear_substitution_screen

        # Gather leaf input data
        data = _gather_atom_teacher_data(
            train_loader=ctx.train_loader_probe,
            atom=target,
            teacher=teacher,
            device=ctx.device,
            dtype=ctx.dtype,
            max_points=3000,
        )
        if data is None:
            return []
        X, F = data

        # Build trig_hints from Stage A trig detection (overrides parity pre-screen)
        trig_hints: Optional[Dict[int, str]] = None
        if getattr(ctx, "trig_by_axis", None):
            trig_hints = {}
            # Build trig hints uniformly from input expressions.
            # For each local axis, find the global variables it depends on
            # and check for trig hints.
            all_inputs = get_input_exprs(target)
            for k, inp in enumerate(all_inputs):
                if is_trivial_input(inp):
                    ax = int(inp.var_idxs[0])
                    spec = ctx.trig_by_axis.get(ax)
                    if spec is not None:
                        trig_hints[k] = "cos" if abs(spec.phase) < 0.5 else "sin"
                else:
                    # Nontrivial expression: check participating variables
                    from nestynet_sr.sr_core.bridges import _collect_var_idxs_from_node
                    for ax in _collect_var_idxs_from_node(inp):
                        spec = ctx.trig_by_axis.get(int(ax))
                        if spec is not None:
                            trig_hints[k] = "cos" if abs(spec.phase) < 0.5 else "sin"
                            break

            if not trig_hints:
                trig_hints = None
            else:
                ctx.log(f"[Stage B] NonlinearSub: trig hints: {trig_hints}")

        requested_count = max(
            0,
            int(getattr(ctx.lm_hp, "stageB_nls_requested_count", 3) or 0),
        )
        max_attempts = max(
            0,
            int(getattr(ctx.lm_hp, "stageB_nls_max_attempts", 1024) or 0),
        )
        max_support_attempts = max(
            0,
            int(
                getattr(
                    ctx.lm_hp,
                    "stageB_nls_max_support_attempts",
                    2048,
                )
                or 0
            ),
        )
        screen_diagnostics: Dict[str, Any] = {}

        target_dim = None
        effective_input_dims = None
        coefficient_policy = "free_const_only"
        if bool(getattr(ctx, "enforce_units", False)):
            units_spec = getattr(ctx, "units_spec", None)
            try:
                target_dim = tuple(ctx.infer_target_dim(target) or ())
                effective_input_dims = tuple(
                    _effective_input_dims_for_atom(target, units_spec)
                )
                coefficient_policy = str(
                    getattr(units_spec, "policy", "free_const_only")
                )
            except Exception:
                target_dim = None
                effective_input_dims = None
            if (
                not target_dim
                or not effective_input_dims
                or len(effective_input_dims) != int(effective_arity(target))
            ):
                proposal_budget = {
                    "requested_count": int(requested_count),
                    "raw_attempted": 0,
                    "baseline_attempted": 0,
                    "support_raw_attempted": 0,
                    "unit_rejected": 1,
                    "build_rejected": 0,
                    "deduplicated": 0,
                    "emitted": 0,
                    "exhausted": True,
                    "exhaustion_reason": "unit_dimensions_unavailable",
                    "truncated_by_attempt_budget": False,
                    "max_attempts": int(max_attempts),
                    "max_support_attempts": int(max_support_attempts),
                }
                ctx._cache[("proposal_budget", self.name, id(target))] = dict(
                    proposal_budget
                )
                ctx.log(
                    "[Stage B] NonlinearSub: unit-aware support planning "
                    f"unavailable for NN vars={target.var_idxs}"
                )
                return []

        # Run cheap screening (including outer-transform probing). Exact units
        # constrain both the transformed candidates and their raw-space
        # comparator before any numerical ranking occurs.
        hits = _nonlinear_substitution_screen(
            X, F, teacher=teacher, threshold=0.02, trig_hints=trig_hints,
            max_deg_num=3, max_deg_den=3,
            # Reciprocal ratio coordinates can turn an otherwise cubic law
            # into a quartic mixed numerator. Add only the 4/1 family rather
            # than opening the substantially larger 4/2 and 4/3 rectangle.
            extra_degree_pairs=[(4, 1)],
            outer_transforms=["square", "reciprocal"],
            target_dim=target_dim,
            input_dims=effective_input_dims,
            coefficient_policy=coefficient_policy,
            max_attempts=max_attempts,
            max_support_attempts=max_support_attempts,
            diagnostics=screen_diagnostics,
        )

        if not hits:
            proposal_budget = {
                "requested_count": int(requested_count),
                "raw_attempted": int(screen_diagnostics.get("raw_attempted", 0)),
                "baseline_attempted": int(
                    screen_diagnostics.get("baseline_attempted", 0)
                ),
                "support_raw_attempted": int(
                    screen_diagnostics.get("support_raw_attempted", 0)
                ),
                "screen_emitted": int(screen_diagnostics.get("emitted", 0)),
                "unit_rejected": int(screen_diagnostics.get("unit_rejected", 0)),
                "numeric_rejected": int(screen_diagnostics.get("numeric_rejected", 0)),
                "build_rejected": 0,
                "deduplicated": int(screen_diagnostics.get("deduplicated", 0)),
                "emitted": 0,
                "exhausted": True,
                "exhaustion_reason": str(
                    screen_diagnostics.get(
                        "exhaustion_reason", "candidate_space_exhausted"
                    )
                ),
                "truncated_by_attempt_budget": bool(
                    screen_diagnostics.get("truncated_by_attempt_budget", False)
                ),
                "max_attempts": int(max_attempts),
                "max_support_attempts": int(max_support_attempts),
                "screen": dict(screen_diagnostics),
            }
            ctx._cache[("proposal_budget", self.name, id(target))] = dict(
                proposal_budget
            )
            ctx.log(
                f"[Stage B] NonlinearSub: no substitutions found "
                f"for NN vars={target.var_idxs}; budget={proposal_budget}"
            )
            return []

        # Log findings
        for h in hits[:12]:
            ot = h.get("outer_transform", "identity")
            ot_tag = f", outer={ot}" if ot != "identity" else ""
            ctx.log(
                f"[Stage B] NonlinearSub: {h['transform']}(col {h['col_idx']}) "
                f"-> ratpoly({h['deg_num']}/{h['deg_den']}), "
                f"err={h['error']:.4g}, parity={h['parity']}{ot_tag}"
            )
        if len(hits) > 12:
            ctx.log(
                f"[Stage B] NonlinearSub: {len(hits) - 12} additional "
                "unit-admissible screened support(s) retained as fallbacks"
            )

        # Count requested *emissions*, not the first raw hits. A failed build,
        # duplicate, or final whole-AST unit assertion advances to the next
        # screened support until the request is filled or search is exhausted.
        import zlib

        cands: List[Candidate] = []
        build_attempted = 0
        build_rejected = 0
        final_unit_rejected = 0
        candidate_deduplicated = 0
        label_counts: Dict[str, int] = {}
        seen_signatures = set()
        for h in hits:
            if len(cands) >= requested_count:
                break
            build_attempted += 1
            result = _build_nonlinear_sub_candidate(
                root=st.root,
                target=target,
                reuse=st.reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
                hit=h,
            )
            if result is None:
                build_rejected += 1
                continue
            root_new, init_fn, meta = result
            if root_new is None:
                build_rejected += 1
                continue

            if bool(getattr(ctx, "enforce_units", False)):
                coefficient_certificate = meta.get(
                    "coefficient_unit_certificate"
                )
                if not (
                    isinstance(coefficient_certificate, dict)
                    and coefficient_certificate.get("valid") is True
                ):
                    final_unit_rejected += 1
                    continue
                try:
                    from nestynet_sr.sr_core.units import check_units_ast

                    whole_ast_units = check_units_ast(root_new, ctx.units_spec)
                except Exception:
                    whole_ast_units = None
                if whole_ast_units is None or not bool(
                    getattr(whole_ast_units, "ok", False)
                ):
                    final_unit_rejected += 1
                    continue
                meta["unit_admissibility"] = {
                    "checked": True,
                    "valid": True,
                    "code": "stageb_nls_unit_support_valid",
                    "reason": "exact coefficient support and whole AST are dimensionally admissible",
                    "coefficient_units": coefficient_certificate,
                }

            # Stall-retry policy: a strong substitution screen certifies that
            # this rational family can reproduce the teacher near-exactly, so
            # a fit that stalls far above the current model loss is an
            # optimization failure, not a family mismatch (pb119-class).
            # Allow bounded jittered restarts; candidates without these
            # attributes keep the single-start behavior.
            screen_err = float(h.get("error", float("inf")))
            base_loss = float(getattr(st, "val_loss", float("inf")))
            if (
                math.isfinite(screen_err)
                and screen_err
                <= float(
                    getattr(ctx.lm_hp, "stageB_nls_retry_screen_err_max", 0.02)
                )
                and math.isfinite(base_loss)
            ):
                init_fn._candidate_max_starts = int(
                    getattr(ctx.lm_hp, "stageB_nls_max_starts", 3)
                )
                init_fn._candidate_retry_nonfinite = True
                init_fn._candidate_retry_stall_loss = 10.0 * max(
                    base_loss, 1.0e-300
                )

            outer_t = h.get("outer_transform", "identity")
            base_label = (
                f"nls_{h['transform']}"
                if outer_t == "identity"
                else f"nls_{h['transform']}_{outer_t}"
            )
            label_index = int(label_counts.get(base_label, 0))
            label_counts[base_label] = label_index + 1
            label = base_label if label_index == 0 else f"{base_label}[{label_index}]"
            # Propagate probe error so the floor-lock bypass can fire
            # when a higher-param candidate has near-perfect screening fit.
            meta["precheck_rel_rms"] = float(h.get("error", float("inf")))
            # A fit may sparsify two different planned supports to the same
            # final rational AST. Count only distinct final emissions and keep
            # consuming screened fallbacks until the requested count is met.
            support_payload = (
                tuple(
                    tuple(int(v) for v in row)
                    for row in meta.get(
                        "exps_num_override",
                        h.get("exps_num_override", ()),
                    )
                ),
                tuple(
                    tuple(int(v) for v in row)
                    for row in meta.get(
                        "exps_den_override",
                        h.get("exps_den_override", ()),
                    )
                ),
            )
            support_sig = int(
                zlib.crc32(repr(support_payload).encode("utf-8")) & 0xFFFFFFFF
            )
            transform_sig = int(
                zlib.crc32(str(h["transform"]).encode("utf-8")) & 0xFFFFFFFF
            )
            outer_sig = int(
                zlib.crc32(str(outer_t).encode("utf-8")) & 0xFFFFFFFF
            )
            sig = (
                atom_content_hash(target),
                transform_sig,
                int(meta.get("col_idx", h["col_idx"])),
                int(meta.get("deg_num", h["deg_num"])),
                int(meta.get("deg_den", h["deg_den"])),
                outer_sig,
                int(
                    zlib.crc32(
                        str(meta.get("leaf_kind", "ratpoly")).encode("utf-8")
                    )
                    & 0xFFFFFFFF
                ),
                int(bool(meta.get("trial_inv_z", False))),
                int(float(meta.get("sign_hint", 1.0)) < 0.0),
                support_sig,
            )
            if sig in seen_signatures:
                candidate_deduplicated += 1
                continue
            seen_signatures.add(sig)
            cands.append(Candidate(label, root_new, init_fn, meta=meta, signature=sig))

        total_unit_rejected = int(
            screen_diagnostics.get("unit_rejected", 0)
        ) + int(final_unit_rejected)
        emitted = int(len(cands))
        truncated = bool(
            screen_diagnostics.get("truncated_by_attempt_budget", False)
        )
        if emitted >= requested_count:
            exhausted = False
            exhaustion_reason = None
        elif truncated:
            exhausted = True
            exhaustion_reason = "attempt_budget_exhausted"
        else:
            exhausted = True
            exhaustion_reason = "candidate_space_exhausted"
        proposal_budget = {
            "requested_count": int(requested_count),
            "raw_attempted": int(screen_diagnostics.get("raw_attempted", 0)),
            "baseline_attempted": int(
                screen_diagnostics.get("baseline_attempted", 0)
            ),
            "support_raw_attempted": int(
                screen_diagnostics.get("support_raw_attempted", 0)
            ),
            "screen_emitted": int(screen_diagnostics.get("emitted", 0)),
            "candidate_build_attempted": int(build_attempted),
            "unit_rejected": int(total_unit_rejected),
            "numeric_rejected": int(screen_diagnostics.get("numeric_rejected", 0)),
            "build_rejected": int(build_rejected),
            "deduplicated": int(screen_diagnostics.get("deduplicated", 0))
            + int(candidate_deduplicated),
            "candidate_deduplicated": int(candidate_deduplicated),
            "emitted": emitted,
            "exhausted": bool(exhausted),
            "exhaustion_reason": exhaustion_reason,
            "truncated_by_attempt_budget": truncated,
            "max_attempts": int(max_attempts),
            "max_support_attempts": int(max_support_attempts),
            "screen": dict(screen_diagnostics),
        }
        for cand in cands:
            cand.meta["proposal_budget"] = dict(proposal_budget)
        ctx._cache[("proposal_budget", self.name, id(target))] = dict(
            proposal_budget
        )
        ctx.log(
            "[Stage B] NonlinearSub proposal budget: "
            f"requested={requested_count}, attempted={proposal_budget['raw_attempted']}, "
            f"unit_rejected={total_unit_rejected}, built={build_attempted}, "
            f"emitted={emitted}, exhaustion={exhaustion_reason or 'none'}"
        )
        return cands


class RuleAffineDecomposition(StageBRule):
    """Detect affine decomposition: g(f(z, w)) = a(z) + b(z) * h(w).

    For each 2D NN leaf, try output transforms g in {identity, reciprocal}
    and variable transforms h in {identity, cos, sin} on the second input.
    If the transformed output is affine in h(w) across z-bins, propose a
    structural rewrite that replaces the 2D atom with two 1D NN atoms.

    Pattern label: affine_decomp
    """

    name = "affine_decomp"
    multi_probe_native = True

    def iter_targets(self, ctx: StageBContext):
        return [a for a in _collect_multivariate_nn_atoms(ctx.state.root)
                if effective_arity(a) == 2]

    def _build_trig_hints(self, ctx: StageBContext, target: AtomNode) -> Optional[Dict]:
        trig_hints: Optional[Dict] = None
        if getattr(ctx, "trig_by_axis", None):
            trig_hints = {}
            all_inputs = get_input_exprs(target)
            for k, inp in enumerate(all_inputs):
                if is_trivial_input(inp):
                    spec = ctx.trig_by_axis.get(int(inp.var_idxs[0]))
                    if spec is not None:
                        trig_hints[k] = spec
                else:
                    from nestynet_sr.sr_core.bridges import _collect_var_idxs_from_node
                    for ax in _collect_var_idxs_from_node(inp):
                        spec = ctx.trig_by_axis.get(int(ax))
                        if spec is not None:
                            trig_hints[k] = spec
                            break
        return trig_hints or None

    def _screen_hits_for_dataset(
        self,
        ctx: StageBContext,
        target: AtomNode,
        teacher,
        train_loader_probe,
        *,
        ds_name: str,
    ) -> List[Dict]:
        from nestynet_sr.sr_search.candidate_builders import _gather_atom_teacher_data
        from nestynet_sr.sr_search.fitting_utils import _affine_decomposition_screen

        trig_hints = self._build_trig_hints(ctx, target)
        if trig_hints:
            ctx.log(f"[Stage B] AffineDecomp[{ds_name}]: trig hints: {trig_hints}")

        data = _gather_atom_teacher_data(
            train_loader=train_loader_probe,
            atom=target,
            teacher=teacher,
            device=ctx.device,
            dtype=ctx.dtype,
            max_points=3000,
        )
        if data is None:
            return []
        X, F = data
        hits = _affine_decomposition_screen(X, F, trig_hints=trig_hints)
        if not hits:
            ctx.log(
                f"[Stage B] AffineDecomp[{ds_name}]: no decomposition found "
                f"for NN vars={target.var_idxs}"
            )
            return []
        for h in hits:
            ctx.log(
                f"[Stage B] AffineDecomp[{ds_name}]: g={h['g_name']}, h={h['h_name']}, "
                f"omega={h['omega']:.4g}, R²={h['median_r2']:.6f}"
            )
        return hits

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        if not isinstance(target, AtomNode) or target.kind.lower() != "nn":
            return []
        if effective_arity(target) != 2:
            return []

        st = ctx.state
        tag = target.tag
        if tag is None or tag not in st.reuse:
            return []

        print(f"[Stage B] RuleAffineDecomposition: screening NN vars={target.var_idxs}, tag={tag}")
        from nestynet_sr.sr_search.candidate_builders import _build_affine_decomp_candidate

        train_loaders = list(ctx.train_loader_probes)
        reuses = list(getattr(st, "reuses", None) or [])
        if len(reuses) < len(train_loaders):
            reuses = reuses + [st.reuse for _ in range(len(train_loaders) - len(reuses))]
        dataset_names = (
            [str(x) for x in ctx.dataset_ids]
            if isinstance(ctx.dataset_ids, (list, tuple)) and len(ctx.dataset_ids) == len(train_loaders)
            else [f"ds{i}" for i in range(len(train_loaders))]
        )

        pooled_hits: Dict[Tuple[str, str, float, int], Dict[str, Any]] = {}
        raw_count = 0
        for ds_idx, (ds_name, train_loader_probe, reuse_i) in enumerate(
            zip(dataset_names, train_loaders, reuses)
        ):
            teacher = reuse_i.get(tag, None) if isinstance(reuse_i, dict) else None
            if teacher is None:
                continue
            hits_i = self._screen_hits_for_dataset(
                ctx, target, teacher, train_loader_probe, ds_name=str(ds_name)
            )
            raw_count += len(hits_i)
            for h in hits_i[:2]:
                key = (
                    str(h["g_name"]),
                    str(h["h_name"]),
                    round(float(h.get("omega", 1.0)), 12),
                    int(h.get("col_w", 1)),
                )
                bucket = pooled_hits.setdefault(
                    key,
                    {
                        "representative_hit": h,
                        "dataset_hit_map": {},
                        "probe_dataset_idxs": [],
                        "probe_datasets": [],
                    },
                )
                rep = bucket["representative_hit"]
                h_rank = (
                    float(h.get("median_r2", 0.0)),
                    int(bool(h.get("global_affine", False))),
                )
                rep_rank = (
                    float(rep.get("median_r2", 0.0)),
                    int(bool(rep.get("global_affine", False))),
                )
                if h_rank > rep_rank:
                    bucket["representative_hit"] = h
                bucket["dataset_hit_map"][str(ds_name)] = h
                bucket["probe_dataset_idxs"].append(int(ds_idx))
                bucket["probe_datasets"].append(str(ds_name))

        if not pooled_hits:
            return []

        ctx.log(
            f"[Stage B]  conjoint probe affine_decomp: "
            f"{raw_count} raw -> {len(pooled_hits)} pooled candidates over {len(train_loaders)} datasets"
        )

        ranked_hits = sorted(
            pooled_hits.values(),
            key=lambda b: (
                -len(set(int(i) for i in b["probe_dataset_idxs"])),
                -float(b["representative_hit"].get("median_r2", 0.0)),
            ),
        )

        cands = []
        for bucket in ranked_hits[:2]:
            h = bucket["representative_hit"]
            result = _build_affine_decomp_candidate(
                root=st.root,
                target=target,
                reuse=st.reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
                hit=h,
                dataset_hit_map=bucket["dataset_hit_map"],
            )
            if result is None:
                continue
            root_new, init_fn, meta = result
            if root_new is None:
                continue

            sig = (
                id(target),
                hash(h["g_name"]),
                hash(h["h_name"]),
                h.get("omega", 1.0),
            )
            if not isinstance(meta, dict):
                meta = {}
            meta["probe_dataset_idxs"] = sorted(set(int(i) for i in bucket["probe_dataset_idxs"]))
            meta["probe_datasets"] = sorted(set(str(n) for n in bucket["probe_datasets"]))
            meta["probe_dataset_count"] = len(meta["probe_dataset_idxs"])
            cands.append(Candidate(
                label="affine_decomp",
                root=root_new,
                init_fn=init_fn,
                meta=meta,
                signature=sig,
            ))

        return cands
