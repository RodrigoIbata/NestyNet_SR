# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Late preconditioner fallback Stage-B rule."""

from __future__ import annotations

import math

import torch

from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    CosNode,
    MulNode,
    PowNode,
    SinNode,
    has_nontrivial_input,
    replace_atom_in_ast,
)
from nestynet_sr.sr_core.constants import (
    build_scalar_atom_from_variant as _build_scalar_atom_from_variant,
    make_unit_aware_scalar_atom as _make_unit_aware_scalar_atom,
    scalar_constant_variants as _scalar_constant_variants,
)
from nestynet_sr.sr_search.candidate_builders import _build_atom_input_tensor
from nestynet_sr.sr_search.features import (
    _cross_hess_rel,
    _hess_const_rel,
    _poly_fit_rms_rel,
    _probe_score,
    _scaling_rel_std,
)

from .engine import Candidate, StageBRule
from .helpers import (
    _collect_all_atoms,
    _collect_multivariate_nn_atoms,
    _collect_univariate_nn_atoms,
    _poly_zero_and_set,
    _set_constant_leaf_value,
    build_atom_to_leaf_map,
)


class RulePreconditionerFallbackNN(StageBRule):
    name = "preconditioner_fallback_nn"

    def __init__(
        self,
        *,
        tierA_keep=6,
        topk=2,
        max_points=2048,
        min_domain_frac=0.92,
        eps_C=1e-3,
        phases=(0.0, math.pi / 2),
        omega_scales=(1.0, 0.5, 2.0),
        seg_frac_probe=0.25,
        seg_frac_fallback=0.5,
        seg_min=4,
        tierB_steps=120,
        tierB_lr=3e-2,
        tierB_batch=256,
        tierB_max=4,
        min_improve=0.25,
        max_axes=2,
        add_amp_clip=1e3,
    ):
        self.tierA_keep = int(tierA_keep)
        self.topk = int(topk)
        self.max_points = int(max_points)
        self.min_domain_frac = float(min_domain_frac)
        self.eps_C = float(eps_C)
        self.phases = tuple(float(p) for p in phases)
        self.omega_scales = tuple(float(s) for s in omega_scales)
        self.seg_frac_probe = float(seg_frac_probe)
        self.seg_frac_fallback = float(seg_frac_fallback)
        self.seg_min = int(seg_min)
        self.tierB_steps = int(tierB_steps)
        self.tierB_lr = float(tierB_lr)
        self.tierB_batch = int(tierB_batch)
        self.tierB_max = int(tierB_max)
        self.min_improve = float(min_improve)
        self.max_axes = int(max_axes)
        self.add_amp_clip = float(add_amp_clip)

    def iter_targets(self, ctx):
        # Skip nontrivial-input atoms - axis mapping is too complex for preconditioner fallback
        all_targets = _collect_univariate_nn_atoms(ctx.state.root) + _collect_multivariate_nn_atoms(
            ctx.state.root
        )
        return [t for t in all_targets if not has_nontrivial_input(t)]

    def propose(self, ctx, target):
        if not isinstance(target, AtomNode) or str(target.kind).lower() != "nn":
            return []
        if getattr(ctx, "fresh_nn_factory", None) is None:
            return []
        st = ctx.state
        atom_to_leaf = build_atom_to_leaf_map(st.root, st.model)
        leaf = atom_to_leaf.get(id(target), None)
        if leaf is None:
            return []

        X_full = []
        n = 0
        for batch in ctx.train_loader_probe:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if x is None:
                continue
            x = x.to(device=ctx.device, dtype=ctx.dtype)
            X_full.append(x)
            n += int(x.shape[0])
            if n >= self.max_points:
                break
        if not X_full:
            return []
        X_full = torch.cat(X_full, dim=0)[: self.max_points]

        # Unified eval for both compound and simple atoms
        X = _build_atom_input_tensor(target, X_full)
        d = int(X.shape[1])

        if d < 1:
            return []

        with torch.no_grad():
            try:
                u = leaf(X)
                u = u[:, 0] if u.dim() == 2 else u.view(-1)
                cache = {"x": X}
                gu = leaf.grad(cache)
                gu = gu[:, 0, :] if gu.dim() == 3 else gu.view(-1, d)
                Hu = leaf.grad_grad(cache)
                Hu = Hu[:, 0, :, :] if Hu.dim() == 4 else Hu.view(-1, d, d)
            except Exception:
                return []

        from nestynet_sr.sr_search.fitting_utils import _rational_probe_nd

        def _score_arrays(Xv, yv, gv, Hv, mask_extra=None):
            m = torch.isfinite(yv)
            if gv is not None:
                m = m & torch.isfinite(gv).all(dim=1)
            if Hv is not None:
                m = m & torch.isfinite(Hv).all(dim=(1, 2))
            if mask_extra is not None:
                m = m & mask_extra
            frac = float(m.float().mean().item()) if m.numel() else 0.0
            if frac < 1e-12:
                return -1e9, frac
            Xm = Xv[m]
            ym = yv[m]
            gm = gv[m] if gv is not None else None
            Hm = Hv[m] if Hv is not None else None
            try:
                poly2 = _poly_fit_rms_rel(Xm, ym, degree=2)
            except Exception:
                poly2 = float("inf")
            try:
                rat = _rational_probe_nd(
                    Xm, ym, deg_num=1, deg_den=1, min_points=50, max_points=600
                )
            except Exception:
                rat = float("inf")
            try:
                hess_rel, _ = _hess_const_rel(Hm)
            except Exception:
                hess_rel = float("inf")
            try:
                sc_rel = _scaling_rel_std(Xm, ym, gm)
            except Exception:
                sc_rel = float("inf")
            try:
                cross = _cross_hess_rel(Hm)
            except Exception:
                cross = float("inf")
            return float(
                _probe_score(
                    domain_ok_frac=frac,
                    poly2_rms_rel=poly2,
                    rat_rms_rel=rat,
                    hess_const_rel=hess_rel,
                    scaling_rel_std=sc_rel,
                    cross_hess_rel=cross,
                )
            ), frac

        base_score, base_dom = _score_arrays(X, u, gu, Hu)
        if base_dom < self.min_domain_frac:
            return []

        def _C_eval(kind, axis_k, omega=0.0, phase=0.0, slope=0.0):
            x = X[:, axis_k]
            g = torch.zeros((X.shape[0], d), device=X.device, dtype=X.dtype)
            H = torch.zeros((X.shape[0], d, d), device=X.device, dtype=X.dtype)
            if kind == "cos":
                z = omega * x + phase
                c = torch.cos(z)
                s = torch.sin(z)
                g[:, axis_k] = -s * omega
                H[:, axis_k, axis_k] = -c * (omega * omega)
                return c, g, H
            if kind == "sin":
                z = omega * x + phase
                c = torch.cos(z)
                s = torch.sin(z)
                g[:, axis_k] = c * omega
                H[:, axis_k, axis_k] = -s * (omega * omega)
                return s, g, H
            if kind == "poly1":
                z = slope * x
                g[:, axis_k] = slope
                return z, g, H
            if kind == "exp1":
                z = (slope * x).clamp(-60.0, 60.0)
                e = torch.exp(z)
                g[:, axis_k] = e * slope
                H[:, axis_k, axis_k] = e * (slope * slope)
                return e, g, H
            raise ValueError(kind)

        axes = list(range(d))
        if getattr(ctx, "trig_by_axis", None):
            scored = []
            for k in axes:
                j = int(target.var_idxs[k])
                spec = ctx.trig_by_axis.get(j, None)
                scored.append(
                    (k, float(getattr(spec, "strength", 0.0)) if spec is not None else 0.0)
                )
            scored.sort(key=lambda t: t[1], reverse=True)
            axes = [k for k, _ in scored[: max(1, min(self.max_axes, d))]]
        else:
            axes = axes[: max(1, min(self.max_axes, d))]

        cand_specs = []
        for k in axes:
            j = int(target.var_idxs[k])
            spec = ctx.trig_by_axis.get(j, None) if getattr(ctx, "trig_by_axis", None) else None
            if spec is not None and math.isfinite(float(spec.omega)) and float(spec.omega) > 0:
                for s in self.omega_scales:
                    w = float(spec.omega) * float(s)
                    for ph in self.phases:
                        cand_specs.append(("cos", k, w, float(ph), 0.0))
                        cand_specs.append(("sin", k, w, float(ph), 0.0))
            xk = X[:, k]
            scale_x = float(torch.median(xk.abs()).clamp_min(1e-6).item())
            cand_specs.append(("poly1", k, 0.0, 0.0, 1.0 / scale_x))
            med_u = float(torch.median(u.abs()).clamp_min(1e-12).item())
            mask_u = torch.isfinite(u) & (u.abs() > 1e-9 * med_u)
            if mask_u.any():
                ratio = (gu[:, k] / u).masked_select(mask_u)
                if ratio.numel() >= 50 and torch.isfinite(ratio).any():
                    s0 = float(torch.median(ratio[torch.isfinite(ratio)]).clamp(-10.0, 10.0).item())
                    cand_specs.append(("exp1", k, 0.0, 0.0, -s0))

        if not cand_specs:
            return []

        scored_cands = []
        for kind, k, w, ph, slope in cand_specs:
            try:
                C0, gC0, HC0 = _C_eval(kind, k, omega=w, phase=ph, slope=slope)
            except Exception:
                continue
            mC = (
                torch.isfinite(C0)
                & torch.isfinite(gC0).all(dim=1)
                & torch.isfinite(HC0).all(dim=(1, 2))
            )
            mC_div = mC & (C0.abs() > self.eps_C)

            r_mul = u * C0
            g_mul = C0.view(-1, 1) * gu + u.view(-1, 1) * gC0
            outer = gu.unsqueeze(2) * gC0.unsqueeze(1)
            H_mul = C0.view(-1, 1, 1) * Hu + u.view(-1, 1, 1) * HC0 + outer + outer.transpose(1, 2)
            s_mul, dom_mul = _score_arrays(X, r_mul, g_mul, H_mul, mask_extra=mC_div)
            imp_mul = float(s_mul - base_score)
            if dom_mul >= self.min_domain_frac and imp_mul >= self.min_improve:
                scored_cands.append(
                    (imp_mul, s_mul, "div", kind, k, w, ph, slope, None, None, None)
                )

            alpha = None
            if kind in ("cos", "sin", "exp1"):
                mm = mC
                num = (u[mm] * C0[mm]).sum()
                den = (C0[mm] * C0[mm]).sum().clamp_min(1e-12)
                alpha = float((num / den).clamp(-self.add_amp_clip, self.add_amp_clip).item())

            C_add = C0 if alpha is None else (C0 * alpha)
            gC_add = gC0 if alpha is None else (gC0 * alpha)
            HC_add = HC0 if alpha is None else (HC0 * alpha)
            r_add = u - C_add
            g_add = gu - gC_add
            H_add = Hu - HC_add
            s_add, dom_add = _score_arrays(X, r_add, g_add, H_add)
            imp_add = float(s_add - base_score)
            if dom_add >= self.min_domain_frac and imp_add >= self.min_improve:
                scored_cands.append(
                    (imp_add, s_add, "add", kind, k, w, ph, slope, alpha, None, None)
                )

        if not scored_cands:
            return []
        scored_cands.sort(key=lambda t: (t[0], t[1]), reverse=True)
        scored_cands = scored_cands[: max(1, self.tierA_keep)]

        def _quick_fit(seg_count, y_target):
            nn_kwargs = dict(getattr(target, "kwargs", {}) or {})
            nn_kwargs["num_segments"] = int(seg_count)
            atom_tmp = AtomNode(
                "nn", tuple(int(i) for i in target.var_idxs), kwargs=nn_kwargs, tag=None
            )
            try:
                m = ctx.fresh_nn_factory(atom_tmp, None).to(device=ctx.device, dtype=ctx.dtype)
            except Exception:
                return float("inf"), None
            if not list(m.parameters()):
                return float("inf"), None
            m.train()
            opt = torch.optim.Adam(m.parameters(), lr=float(self.tierB_lr))
            N = int(X.shape[0])
            bs = min(int(self.tierB_batch), N)
            for _ in range(int(self.tierB_steps)):
                idx = torch.randint(0, N, (bs,), device=ctx.device)
                xb = X[idx]
                yb = y_target[idx]
                pred = m(xb)
                pred = pred[:, 0] if pred.dim() == 2 else pred.view(-1)
                loss = ((pred - yb) ** 2).mean()
                if not torch.isfinite(loss):
                    break
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                pred = m(X)
                pred = pred[:, 0] if pred.dim() == 2 else pred.view(-1)
                mse = float(((pred - y_target) ** 2).mean().item())
            return mse, {k: v.detach().clone() for k, v in m.state_dict().items()}

        seg_old = int(getattr(target, "kwargs", {}).get("num_segments", 32))
        seg_probe = max(self.seg_min, int(round(seg_old * self.seg_frac_probe)))
        seg_fb = max(seg_probe, max(self.seg_min, int(round(seg_old * self.seg_frac_fallback))))
        seg_fb = seg_fb if seg_fb != seg_probe else None

        tierB_pool = scored_cands[: max(1, min(self.tierB_max, len(scored_cands)))]
        scored2 = []
        for imp, sA, form, kind, k, w, ph, slope, alpha, _, _ in tierB_pool:
            C0, _, _ = _C_eval(kind, k, omega=w, phase=ph, slope=slope)
            C_use = C0 if (form == "div" or alpha is None) else (C0 * alpha)
            y_rem = (u * C_use) if form == "div" else (u - C_use)
            var = float(y_rem.var(unbiased=False).clamp_min(1e-12).item())
            mse1, sd1 = _quick_fit(seg_probe, y_rem)
            rel1 = mse1 / var if math.isfinite(mse1) else float("inf")
            seg_use, mse_use, sd_use, rel_use = seg_probe, mse1, sd1, rel1
            if seg_fb is not None and (not math.isfinite(rel1) or rel1 > 0.25):
                mse2, sd2 = _quick_fit(seg_fb, y_rem)
                rel2 = mse2 / var if math.isfinite(mse2) else float("inf")
                if math.isfinite(rel2) and rel2 < rel_use:
                    seg_use, mse_use, sd_use, rel_use = int(seg_fb), mse2, sd2, rel2
            scored2.append(
                (
                    imp,
                    sA,
                    -rel_use,
                    form,
                    kind,
                    k,
                    w,
                    ph,
                    slope,
                    alpha,
                    seg_use,
                    mse_use,
                    sd_use,
                    rel_use,
                )
            )
        scored2.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        best = [
            (imp, sA, form, kind, k, w, ph, slope, alpha, seg_use, mse_use, sd_use, rel_use)
            for (
                imp,
                sA,
                _,
                form,
                kind,
                k,
                w,
                ph,
                slope,
                alpha,
                seg_use,
                mse_use,
                sd_use,
                rel_use,
            ) in scored2
        ]

        def _build_C_node(kind, k_local, w, ph, slope, alpha, tag_base, *, for_div, alpha_variant=None):
            axis = int(target.var_idxs[k_local])
            init = {}
            if kind in ("cos", "sin"):
                tpoly = f"{tag_base}_poly"
                poly = AtomNode("poly", (axis,), kwargs={"degree": 1, "min_total": 0}, tag=tpoly)
                base = CosNode(poly) if kind == "cos" else SinNode(poly)
                init[tpoly] = {(0,): float(ph), (1,): float(w)}
                node = base
                if (not for_div) and (alpha is not None):
                    if alpha_variant is None:
                        ta = f"{tag_base}_a"
                        a = _make_unit_aware_scalar_atom(
                            None,
                            getattr(ctx, "units_spec", None),
                            base_tag=ta,
                            init=float(alpha),
                        )
                        init[ta] = float(alpha)
                    else:
                        a = _build_scalar_atom_from_variant(alpha_variant)
                        init[str(alpha_variant["tag"])] = float(alpha_variant["value"])
                    node = MulNode(a, base)
                return node, init
            if kind == "poly1":
                tpoly = f"{tag_base}_poly"
                poly = AtomNode("poly", (axis,), kwargs={"degree": 1, "min_total": 0}, tag=tpoly)
                init[tpoly] = {(0,): 0.0, (1,): float(slope)}
                return poly, init
            if kind == "exp1":
                texp = f"{tag_base}_exp"
                base = AtomNode("exp_poly", (axis,), kwargs={"degree": 1}, tag=texp)
                init[texp] = {(0,): 0.0, (1,): float(slope)}
                node = base
                if (not for_div) and (alpha is not None):
                    if alpha_variant is None:
                        ta = f"{tag_base}_a"
                        a = _make_unit_aware_scalar_atom(
                            None,
                            getattr(ctx, "units_spec", None),
                            base_tag=ta,
                            init=float(alpha),
                        )
                        init[ta] = float(alpha)
                    else:
                        a = _build_scalar_atom_from_variant(alpha_variant)
                        init[str(alpha_variant["tag"])] = float(alpha_variant["value"])
                    node = MulNode(a, base)
                return node, init
            raise ValueError(kind)

        out = []
        for idx_c, (
            imp,
            sA,
            form,
            kind,
            k,
            w,
            ph,
            slope,
            alpha,
            seg_use,
            mse_use,
            sd_use,
            rel_use,
        ) in enumerate(best[: max(1, self.topk)]):
            seg_use = int(seg_use) if seg_use is not None else seg_probe
            nn_tag = f"preR_{id(target)}_{idx_c}"
            nn_kwargs = dict(getattr(target, "kwargs", {}) or {})
            nn_kwargs["num_segments"] = int(seg_use)
            nn_r = AtomNode(
                "nn", tuple(int(i) for i in target.var_idxs), kwargs=nn_kwargs, tag=nn_tag
            )
            C_tag_base = f"{nn_tag}_C"
            alpha_variants = [None]
            if (
                (form != "div")
                and (alpha is not None)
                and (kind in ("cos", "sin", "exp1"))
            ):
                alpha_variants = _scalar_constant_variants(
                    getattr(ctx, "units_spec", None),
                    base_tag=f"{C_tag_base}_a",
                    scale_init=float(alpha),
                )

            for alpha_variant in alpha_variants:
                C_node, C_init = _build_C_node(
                    kind,
                    k,
                    w,
                    ph,
                    slope,
                    alpha,
                    C_tag_base,
                    for_div=(form == "div"),
                    alpha_variant=alpha_variant,
                )
                new_sub = (
                    MulNode(nn_r, PowNode(C_node, -1.0)) if form == "div" else AddNode(nn_r, C_node)
                )
                new_root = replace_atom_in_ast(st.root, target, new_sub)

                def _init_fn(root_new, model_new, *, nn_tag=nn_tag, sd_use=sd_use, C_init=C_init):
                    atom_to_leaf2 = build_atom_to_leaf_map(root_new, model_new)
                    for atom in _collect_all_atoms(root_new):
                        if not isinstance(atom, AtomNode):
                            continue
                        if atom.tag == nn_tag:
                            leaf2 = atom_to_leaf2.get(id(atom), None)
                            if leaf2 is not None and sd_use is not None:
                                try:
                                    leaf2.load_state_dict(sd_use, strict=False)
                                except Exception:
                                    pass
                        if atom.tag in C_init:
                            leaf2 = atom_to_leaf2.get(id(atom), None)
                            if leaf2 is not None:
                                try:
                                    init_spec = C_init[atom.tag]
                                    if isinstance(init_spec, dict):
                                        _poly_zero_and_set(leaf2, init_spec)
                                    else:
                                        _set_constant_leaf_value(leaf2, float(init_spec))
                                except Exception:
                                    pass

                label_suffix = (
                    str(alpha_variant.get("label_suffix", ""))
                    if isinstance(alpha_variant, dict)
                    else ""
                )
                log = f"[Stage B]  Trying preconditioner({form}) on NN vars={target.var_idxs}: C={kind}@x{int(target.var_idxs[k])} TierA=+{imp:.2f} seg={seg_use} relMSE~{(rel_use if (rel_use is not None and math.isfinite(float(rel_use))) else float('nan')):.2e}{label_suffix}"
                out.append(
                    Candidate(
                        "precond_fallback_" + form + label_suffix,
                        new_root,
                        _init_fn,
                        meta={
                            "log": log,
                            "tierA": float(imp),
                            "seg": int(seg_use),
                            "rel_mse": float(rel_use)
                            if rel_use is not None and math.isfinite(float(rel_use))
                            else float("inf"),
                        },
                    )
                )
        return out
