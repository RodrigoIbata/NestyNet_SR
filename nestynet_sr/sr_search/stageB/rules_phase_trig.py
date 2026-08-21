# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Phase-hint and last-hard trigonometric Stage-B rules."""

from __future__ import annotations

import copy
import math
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from nestynet_sr.sr_core.bridges import (
    AcosNode,
    AddNode,
    AsinNode,
    AtanNode,
    AtomNode,
    ConstNode,
    CosNode,
    MulNode,
    Node,
    PowNode,
    SinNode,
    _collect_var_idxs_from_node,
    ast_to_human_readable,
    atom_problem_label,
    clone_ast,
    clone_inputs,
    compound_input_expr,
    effective_arity,
    eval_input_expr,
    get_input_exprs,
    replace_atom_in_ast,
)
from nestynet_sr.sr_core.constants import make_unit_aware_scalar_atom as _make_unit_aware_scalar_atom
from nestynet_sr.sr_search.candidate_builders import _gather_atom_teacher_data
from nestynet_sr.sr_search.phase_scan import (
    PhaseScanHyperparams,
    phase_hint_omega_candidates,
    run_phase_prescan,
    stable_int_hash,
)
from nestynet_sr.sr_search.wrapper_policy import macro_arg_wrapper_policy, snap_omega

from .engine import Candidate, StageBContext, StageBRule, atom_content_hash
from .helpers import (
    _collect_all_atoms,
    _poly_zero_and_set,
    _set_constant_leaf_value,
    build_atom_to_leaf_map,
)
from .rules_common import (
    _effective_input_dims_for_atom,
    _mark_reciprocal_coordinate_candidate,
    _merge_reciprocal_aliases_pairwise,
    _stageB_noisy_rel_rms_threshold,
    _wrap_reuse_for_reciprocal_coordinate,
)

try:
    from nestynet_sr.sr_core.units import (
        is_dimless as _is_dimless,
        scale_dim as _scale_dim,
    )
except Exception:  # pragma: no cover
    _is_dimless = None  # type: ignore
    _scale_dim = None  # type: ignore


def _ctx_pattern_disabled(ctx: StageBContext, name: str) -> bool:
    checker = getattr(ctx, "is_pattern_disabled", None)
    if checker is None:
        return False
    return bool(checker(name))


def _last_hard_atom_context(
    ctx: StageBContext,
    target: AtomNode,
    *,
    max_arity: int = 2,
) -> Optional[Tuple[Tuple[Any, ...], List[Tuple[Any, ...]]]]:
    """Return ``(target_dim, input_dims)`` when the final-atom rescue is allowed."""
    if (
        not isinstance(target, AtomNode)
        or str(getattr(target, "kind", "")).lower() != "nn"
        or atom_problem_label(target) is not None
    ):
        return None
    if not bool(getattr(ctx, "enforce_units", False)):
        return None
    units_spec = getattr(ctx, "units_spec", None)
    if units_spec is None:
        return None

    nn_atoms = [
        atom
        for atom in _collect_all_atoms(ctx.state.root)
        if isinstance(atom, AtomNode)
        and str(getattr(atom, "kind", "")).lower() == "nn"
    ]
    if len(nn_atoms) != 1 or nn_atoms[0] is not target:
        return None

    arity = int(effective_arity(target))
    if arity < 1 or arity > int(max_arity):
        return None

    try:
        target_dim = tuple(ctx.infer_target_dim(target) or ())
    except Exception:
        return None
    if not target_dim:
        return None

    x_dims = _effective_input_dims_for_atom(target, units_spec)
    if len(x_dims) != arity:
        return None
    return target_dim, x_dims


_LAST_HARD_TRIG_POWER_POWERS = (-4, -2)
_LAST_HARD_TRIG_POWER_OMEGAS = (
    0.25,
    1.0 / 3.0,
    0.5,
    2.0 / 3.0,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    math.pi / 4.0,
    math.pi / 3.0,
    math.pi / 2.0,
    math.pi,
    2.0 * math.pi,
)


def _wrap_phase(phi: float) -> float:
    try:
        v = float(phi)
    except Exception:
        return 0.0
    two_pi = 2.0 * math.pi
    return float((v + math.pi) % two_pi - math.pi)


def _snap_phase_for_even_power(phi: float, *, tol: float = 5.0e-2) -> float:
    """Snap phases that are equivalent for even trig powers."""
    v = _wrap_phase(phi)
    candidates = (0.0, math.pi / 2.0, -math.pi / 2.0, math.pi, -math.pi)
    best = min(candidates, key=lambda c: abs(_wrap_phase(v - c)))
    if abs(_wrap_phase(v - best)) <= float(tol):
        v = _wrap_phase(best)
    # sin/cos even negative powers are invariant under phase -> phase + pi.
    if abs(abs(v) - math.pi) <= float(tol):
        return 0.0
    return float(v)


def _dedupe_positive_floats(vals: List[float]) -> List[float]:
    out: List[float] = []
    for raw in vals:
        try:
            v = abs(float(raw))
        except Exception:
            continue
        if (not math.isfinite(v)) or v <= 1.0e-12:
            continue
        if any(abs(v - old) <= max(1.0e-9, 1.0e-4 * max(abs(old), 1.0)) for old in out):
            continue
        out.append(v)
    return sorted(out)


def _stageB_phase_scan_units_payload_for_atom(ctx: StageBContext, target: AtomNode):
    if not bool(getattr(ctx, "enforce_units", False)):
        return None, True
    units_spec = getattr(ctx, "units_spec", None)
    if units_spec is None:
        return None, False
    dims = _effective_input_dims_for_atom(target, units_spec)
    if not dims:
        return None, False
    return {"x_dims": tuple(tuple(d) for d in dims)}, False


def _stageB_phase_hints_for_atom(ctx: StageBContext, target: AtomNode) -> List[Any]:
    """Run a capped PhaseScan on one Stage-B NN atom's teacher data.

    This is proposal evidence only.  The returned hints are used to seed the
    deterministic trig screens below; accepted expressions still pass through
    normal Stage-B fitting and validation.
    """
    if not isinstance(target, AtomNode) or str(getattr(target, "kind", "")).lower() != "nn":
        return []
    if not bool(getattr(ctx.lm_hp, "stageB_phase_scan_enabled", True)):
        return []
    tag = getattr(target, "tag", None)
    teacher = ctx.state.reuse.get(tag, None) if isinstance(getattr(ctx.state, "reuse", None), dict) else None
    if teacher is None:
        return []

    key = (
        "stageB_phase_hints_for_atom",
        id(ctx.state.root),
        int(atom_content_hash(target)),
        str(tag),
    )

    def _compute() -> List[Any]:
        try:
            max_points = int(getattr(ctx.lm_hp, "stageB_last_hard_trig_power_max_points", 5000) or 5000)
        except Exception:
            max_points = 5000
        data = _gather_atom_teacher_data(
            train_loader=ctx.train_loader_probe,
            atom=target,
            teacher=teacher,
            device=ctx.device,
            dtype=ctx.dtype,
            max_points=max_points,
        )
        if data is None:
            return []
        X, F = data
        if X.ndim != 2 or X.shape[0] < 128 or X.shape[1] < 1:
            return []
        f = F.detach().to(dtype=torch.float64).reshape(-1)
        X64 = X.detach().to(dtype=torch.float64)
        n = min(int(X64.shape[0]), int(f.shape[0]))
        X64 = X64[:n]
        f = f[:n]
        finite = torch.isfinite(f) & torch.all(torch.isfinite(X64), dim=1)
        if int(finite.sum().item()) < 128:
            return []
        X64 = X64[finite]
        f = f[finite]
        if float(torch.std(f).item()) <= 1.0e-14:
            return []

        units_payload, ignore_units = _stageB_phase_scan_units_payload_for_atom(ctx, target)
        if bool(getattr(ctx, "enforce_units", False)) and units_payload is None:
            return []

        try:
            min_domain = float(getattr(ctx.lm_hp, "macro_domain_ok_frac", 0.98))
        except Exception:
            min_domain = 0.98
        hp = PhaseScanHyperparams(
            enabled=True,
            sample_size=max(128, min(int(max_points), 4096)),
            max_support=max(1, min(3, int(X64.shape[1]))),
            max_candidates=64,
            max_candidates_per_support=16,
            min_domain_frac=float(min_domain),
            max_harmonic=3,
            fft_grid_size=384,
            fft_top_k=8,
            log_top_k=0,
            context_enabled=False,
        )
        try:
            hints = run_phase_prescan(
                X64.detach().cpu().numpy(),
                f.detach().cpu().numpy(),
                Nxvars=int(X64.shape[1]),
                units_payload=units_payload,
                ignore_units=bool(ignore_units),
                hp=hp,
            )
        except Exception:
            return []
        hints = list(hints or [])[:4]
        if hints and bool(getattr(ctx, "verbose", False)):
            try:
                top = hints[0]
                ctx.log(
                    "[Stage B PhaseScan] local atom hint "
                    f"target=NN vars={target.var_idxs}: z={getattr(top, 'carrier_label', '?')}, "
                    f"R2={float(getattr(top, 'r2_phase', float('nan'))):.3g}, "
                    f"omega≈{float(getattr(top, 'observed_omega', float('nan'))):.4g}"
                )
            except Exception:
                pass
        return hints

    try:
        return list(ctx.cached(key, _compute) or [])
    except Exception:
        return _compute()


def _last_hard_trig_power_omega_grid(ctx: StageBContext, target: AtomNode) -> List[float]:
    raw = list(_LAST_HARD_TRIG_POWER_OMEGAS)

    # Add oracle trig hints when available.  This remains a non-FFT rescue:
    # the grid fit below is deterministic least squares, and the hint is only
    # an extra seed.
    try:
        trig_by_axis = getattr(ctx, "trig_by_axis", {}) or {}
        for vi in target.var_idxs:
            spec = trig_by_axis.get(int(vi), None)
            if spec is None:
                continue
            w = float(getattr(spec, "omega", 0.0))
            if math.isfinite(w) and w > 0.0:
                raw.append(float(snap_omega(w)))
                raw.extend([0.5 * w, 2.0 * w])
    except Exception:
        pass

    try:
        phase_hints = list(getattr(ctx, "phase_hints", []) or [])[:8]
        raw.extend(phase_hint_omega_candidates(phase_hints, for_square=False))
    except Exception:
        pass

    try:
        local_phase_hints = _stageB_phase_hints_for_atom(ctx, target)
        raw.extend(phase_hint_omega_candidates(local_phase_hints, for_square=False))
    except Exception:
        pass

    return _dedupe_positive_floats(raw)


def _solve_affine_trig_1d(
    z: torch.Tensor,
    y: torch.Tensor,
    omega: float,
) -> Optional[Dict[str, float]]:
    try:
        w = float(omega)
        wz = z * w
        design = torch.stack(
            [torch.sin(wz), torch.cos(wz), torch.ones_like(wz)],
            dim=1,
        )
        sol = torch.linalg.lstsq(design, y).solution
        if sol.numel() < 3 or not torch.isfinite(sol).all():
            return None
        pred = design @ sol
        resid = y - pred
        centered = y - torch.mean(y)
        denom = torch.sqrt(torch.mean(centered * centered)).clamp_min(1.0e-30)
        rel = torch.sqrt(torch.mean(resid * resid)) / denom
        A, B, C = [float(v) for v in sol[:3]]
        amp = math.hypot(A, B)
        if (not math.isfinite(amp)) or amp <= 1.0e-14:
            return None
        return {
            "omega": w,
            "A": A,
            "B": B,
            "offset": float(C),
            "amp": float(amp),
            "rel_rms": float(rel.item()),
            "offset_rel": float(abs(C) / max(abs(amp), 1.0e-30)),
        }
    except Exception:
        return None


def _last_hard_trig_power_screen(
    ctx: StageBContext,
    target: AtomNode,
    *,
    min_points: int = 200,
    max_candidates: int = 4,
) -> List[Dict[str, Any]]:
    tag = getattr(target, "tag", None)
    teacher = ctx.state.reuse.get(tag, None) if isinstance(getattr(ctx.state, "reuse", None), dict) else None
    if teacher is None:
        return []

    try:
        max_points = int(getattr(ctx.lm_hp, "stageB_last_hard_trig_power_max_points", 5000) or 5000)
    except Exception:
        max_points = 5000
    data = _gather_atom_teacher_data(
        train_loader=ctx.train_loader_probe,
        atom=target,
        teacher=teacher,
        device=ctx.device,
        dtype=ctx.dtype,
        max_points=max_points,
    )
    if data is None:
        return []
    X, F = data
    if X.ndim != 2 or X.shape[1] != 1:
        return []

    z = X[:, 0].detach().to(dtype=torch.float64).reshape(-1)
    f = F.detach().to(dtype=torch.float64).reshape(-1)
    finite = torch.isfinite(z) & torch.isfinite(f)
    scale = torch.quantile(f[finite].abs(), 0.75).clamp_min(1.0) if bool(finite.any()) else torch.tensor(1.0)
    finite = finite & (f.abs() > (1.0e-14 * scale))
    domain_frac = float(finite.to(dtype=torch.float64).mean().item()) if int(f.numel()) else 0.0
    try:
        min_domain = float(getattr(ctx.lm_hp, "macro_domain_ok_frac", 0.98))
    except Exception:
        min_domain = 0.98
    if domain_frac < min_domain:
        return []

    z = z[finite]
    f = f[finite]
    if int(z.numel()) < int(min_points):
        return []

    signs = torch.sign(f)
    sign_ref = float(torch.sign(torch.median(f)).item())
    if sign_ref == 0.0:
        sign_ref = 1.0
    sign_frac = float((signs == sign_ref).to(dtype=torch.float64).mean().item())
    if sign_frac < min_domain:
        return []

    abs_f = f.abs().clamp_min(1.0e-300)
    omega_grid = _last_hard_trig_power_omega_grid(ctx, target)
    if not omega_grid:
        return []

    try:
        max_rel = float(getattr(ctx.lm_hp, "stageB_last_hard_trig_power_screen_rel_rms", 2.0e-2))
    except Exception:
        max_rel = 2.0e-2
    max_rel = _stageB_noisy_rel_rms_threshold(ctx, max_rel, y_values=f)
    try:
        max_offset_rel = float(getattr(ctx.lm_hp, "stageB_last_hard_trig_power_max_offset_rel", 0.15))
    except Exception:
        max_offset_rel = 0.15

    rows: List[Dict[str, Any]] = []
    for power in _LAST_HARD_TRIG_POWER_POWERS:
        q = abs(int(power))
        y = torch.pow(abs_f, -1.0 / float(q))
        if not torch.isfinite(y).all():
            continue
        if float(torch.std(y).item()) <= 1.0e-14:
            continue
        for omega in omega_grid:
            fit = _solve_affine_trig_1d(z, y, omega)
            if fit is None:
                continue
            rel = float(fit["rel_rms"])
            offset_rel = float(fit["offset_rel"])
            if rel > max_rel or offset_rel > max_offset_rel:
                continue

            A = float(fit["A"])
            B = float(fit["B"])
            phase_sin = _snap_phase_for_even_power(math.atan2(B, A))
            phase_cos = _snap_phase_for_even_power(math.atan2(-A, B))
            use_cos = abs(phase_cos) + 1.0e-12 < abs(phase_sin)
            trig_kind = "cos" if use_cos else "sin"
            phase = phase_cos if use_cos else phase_sin
            amp = float(fit["amp"])
            scale_init = float(sign_ref) * (amp ** (-float(q)))
            if not math.isfinite(scale_init):
                continue
            rows.append(
                {
                    "power": int(power),
                    "omega": float(omega),
                    "phase": float(phase),
                    "trig_kind": trig_kind,
                    "scale_init": scale_init,
                    "rel_rms": rel,
                    "offset_rel": offset_rel,
                    "domain_frac": domain_frac,
                    "sign_frac": sign_frac,
                    "screen_score": rel + 0.05 * offset_rel + 0.001 * float(q),
                }
            )

    rows.sort(key=lambda r: float(r["screen_score"]))
    return rows[: max(0, int(max_candidates))]


def _make_last_hard_trig_power_candidate(
    ctx: StageBContext,
    target: AtomNode,
    target_dim: Tuple[Any, ...],
    hit: Dict[str, Any],
) -> Optional[Candidate]:
    base_tag = str(getattr(target, "tag", None) or "leaf")
    power = int(hit["power"])
    omega = float(hit["omega"])
    phase = float(hit["phase"])
    trig_kind = str(hit["trig_kind"])
    scale_init = float(hit["scale_init"])

    arg_tag = f"{base_tag}_lhtrig_arg"
    scale_tag = f"{base_tag}_lhtrig_scale"
    arg_atom = AtomNode(
        kind="poly",
        var_idxs=tuple(int(j) for j in target.var_idxs),
        kwargs={"degree": 1, "min_total": 0},
        tag=arg_tag,
        inputs=clone_inputs(target),
    )
    trig_node = CosNode(arg_atom) if trig_kind == "cos" else SinNode(arg_atom)
    core = PowNode(trig_node, float(power))
    scale_node = _make_unit_aware_scalar_atom(
        target_dim,
        getattr(ctx, "units_spec", None),
        base_tag=scale_tag,
        init=scale_init,
    )
    new_subtree = MulNode(scale_node, core)
    root_new = replace_atom_in_ast(ctx.state.root, target, new_subtree)
    if root_new is None:
        return None

    scale_node_tag = getattr(scale_node, "tag", scale_tag)

    def _init(root_new_inner: Node, model_new: nn.Module, *, _phase=phase, _omega=omega, _scale=scale_init):
        atom_to_leaf = build_atom_to_leaf_map(root_new_inner, model_new)
        for atom in _collect_all_atoms(root_new_inner):
            if not isinstance(atom, AtomNode):
                continue
            leaf = atom_to_leaf.get(id(atom), None)
            if leaf is None:
                continue
            if getattr(atom, "tag", None) == arg_tag:
                try:
                    _poly_zero_and_set(leaf, {(0,): float(_phase), (1,): float(_omega)})
                except Exception:
                    pass
            elif getattr(atom, "tag", None) == scale_node_tag:
                try:
                    _set_constant_leaf_value(leaf, float(_scale))
                except Exception:
                    pass

    _init._after_analytic_init = True

    label = f"last_trig_power_{trig_kind}_p{power}"
    meta = {
        "structural": True,
        "pattern": "last_hard_trig_power",
        "pattern_family": "last_hard_trig_power",
        "last_hard_trig_power": True,
        "trig_kind": trig_kind,
        "trig_power": int(power),
        "omega": float(omega),
        "phase": float(phase),
        "screen_rel_rms": float(hit.get("rel_rms", float("inf"))),
        "screen_offset_rel": float(hit.get("offset_rel", float("inf"))),
        "signature": (
            int(atom_content_hash(target)),
            stable_int_hash("last_hard_trig_power"),
            stable_int_hash(trig_kind),
            int(power),
            int(round(float(omega) * 1.0e6)),
            int(round(float(phase) * 1.0e6)),
        ),
        "log": (
            f"[Stage B]  Trying last-hard trig-power {trig_kind}(ωz+φ)^{power} "
            f"on NN vars={target.var_idxs}: ω≈{omega:.4g}, φ≈{phase:.4g}, "
            f"rel={float(hit.get('rel_rms', float('inf'))):.2e}"
        ),
    }
    return Candidate(label=label, root=root_new, init_fn=_init, meta=meta)


def _last_hard_trig_square_omega_grid(ctx: StageBContext, target: AtomNode) -> List[float]:
    """Harmonic frequencies for f(z)≈C+A*sin(Wz)+B*cos(Wz).

    A pure square ``sin(a*z+b)^2`` appears at harmonic frequency ``W=2a``.
    Reuse the existing no-FFT trig grid and add its doubled frequencies so
    physics constants like ``a=2*pi`` are represented by ``W=4*pi``.
    """
    arg_grid = _last_hard_trig_power_omega_grid(ctx, target)
    raw: List[float] = []
    raw.extend(arg_grid)
    raw.extend([2.0 * w for w in arg_grid])
    try:
        phase_hints = list(getattr(ctx, "phase_hints", []) or [])[:8]
        raw.extend(phase_hint_omega_candidates(phase_hints, for_square=True))
    except Exception:
        pass
    try:
        local_phase_hints = _stageB_phase_hints_for_atom(ctx, target)
        raw.extend(phase_hint_omega_candidates(local_phase_hints, for_square=True))
    except Exception:
        pass
    return _dedupe_positive_floats(raw)


def _phase_hint_vars(hint: Any) -> set[int]:
    try:
        ast = getattr(hint, "carrier_ast", None)
        return set(int(v) for v in _collect_var_idxs_from_node(ast))
    except Exception:
        return set()


def _phase_hint_carrier_is_dimless(ctx: StageBContext, hint: Any) -> bool:
    if not bool(getattr(ctx, "enforce_units", False)):
        return True
    units_spec = getattr(ctx, "units_spec", None)
    if units_spec is None:
        return False
    try:
        from nestynet_sr.sr_core.units import eval_analytic_expr_dim

        dim = eval_analytic_expr_dim(getattr(hint, "carrier_ast", None), units_spec.x_dims)
        return dim is not None and (_is_dimless is None or bool(_is_dimless(dim)))
    except Exception:
        return False


def _phase_hint_arg_omegas(hint: Any) -> List[float]:
    vals: List[float] = []
    try:
        vals.extend(phase_hint_omega_candidates([hint], for_square=False))
    except Exception:
        pass
    try:
        obs = getattr(hint, "observed_omega", None)
        if obs is not None:
            vals.append(0.5 * float(obs))
    except Exception:
        pass
    return _dedupe_positive_floats(vals)


def _phase_hint_fourier_omegas(hint: Any) -> List[float]:
    vals: List[float] = []
    try:
        vals.extend(phase_hint_omega_candidates([hint], for_square=True))
    except Exception:
        pass
    return _dedupe_positive_floats(vals)


def _gather_phase_direct_xy(ctx: StageBContext, *, max_points: int = 5000):
    """Gather raw Stage-B training pairs for direct phase-closure screening."""
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    n = 0
    try:
        loader = ctx.train_loader_probe
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                xb, yb = batch[0], batch[1]
            else:
                continue
            xb = xb.to(device=ctx.device, dtype=ctx.dtype)
            yb = yb.to(device=ctx.device, dtype=ctx.dtype)
            remaining = int(max_points) - int(n)
            if remaining <= 0:
                break
            if int(xb.shape[0]) > remaining:
                xb = xb[:remaining]
                yb = yb[:remaining]
            xs.append(xb)
            ys.append(yb.reshape(int(yb.shape[0]), -1)[:, :1])
            n += int(xb.shape[0])
    except Exception:
        return None
    if not xs or not ys:
        return None
    try:
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0).reshape(-1)
    except Exception:
        return None


def _phase_direct_affine_fit(z: torch.Tensor, y: torch.Tensor, omega: float) -> Optional[Dict[str, float]]:
    fit = _solve_affine_trig_1d(z, y, omega)
    if fit is None:
        return None
    return fit


def _phase_direct_square_fit(z: torch.Tensor, y: torch.Tensor, arg_omega: float, trig_kind: str) -> Optional[Dict[str, float]]:
    harmonic_omega = 2.0 * float(arg_omega)
    fit = _solve_affine_trig_1d(z, y, harmonic_omega)
    if fit is None:
        return None
    A = float(fit["A"])
    B = float(fit["B"])
    C = float(fit["offset"])
    amp = float(fit["amp"])
    if not math.isfinite(amp) or amp <= 1.0e-14:
        return None
    if str(trig_kind) == "cos":
        phase = _snap_phase_for_even_power(0.5 * math.atan2(-A, B))
    else:
        phase = _snap_phase_for_even_power(0.5 * math.atan2(A, -B))
    scale_init = 2.0 * amp
    offset_init = C - 0.5 * scale_init
    if not all(math.isfinite(v) for v in (phase, scale_init, offset_init)):
        return None
    out = dict(fit)
    out.update(
        {
            "omega": float(arg_omega),
            "harmonic_omega": float(harmonic_omega),
            "phase": float(phase),
            "trig_kind": str(trig_kind),
            "scale_init": float(scale_init),
            "offset_init": float(offset_init),
            "screen_score": float(fit.get("rel_rms", float("inf"))),
        }
    )
    return out


def _phase_direct_target_dim(ctx: StageBContext, target: AtomNode):
    try:
        dim = ctx.infer_target_dim(target)
        if dim is None:
            return None
        return tuple(dim)
    except Exception:
        return None


def _make_phase_direct_square_candidate(
    ctx: StageBContext,
    target: AtomNode,
    hint: Any,
    target_dim,
    hit: Dict[str, Any],
) -> Optional[Candidate]:
    carrier_label = str(getattr(hint, "carrier_label", "z"))
    try:
        carrier_ast = clone_ast(getattr(hint, "carrier_ast"))
    except Exception:
        return None
    trig_kind = str(hit["trig_kind"])
    omega = float(hit["omega"])
    phase = float(hit["phase"])
    scale_init = float(hit["scale_init"])
    offset_init = float(hit["offset_init"])
    base_tag = str(getattr(target, "tag", None) or "leaf")
    arg_tag = f"{base_tag}_phase_arg"
    scale_tag = f"{base_tag}_phase_scale"
    offset_tag = f"{base_tag}_phase_offset"
    var_idxs = tuple(sorted(int(v) for v in _collect_var_idxs_from_node(carrier_ast)))
    arg_atom = AtomNode(
        kind="poly",
        var_idxs=var_idxs,
        kwargs={"degree": 1, "min_total": 0},
        tag=arg_tag,
        inputs=(carrier_ast,),
    )
    trig_node = CosNode(arg_atom) if trig_kind == "cos" else SinNode(arg_atom)
    try:
        scale_node = _make_unit_aware_scalar_atom(
            target_dim,
            getattr(ctx, "units_spec", None),
            base_tag=scale_tag,
            init=scale_init,
            strict=bool(getattr(ctx, "enforce_units", False)),
        )
        offset_node = _make_unit_aware_scalar_atom(
            target_dim,
            getattr(ctx, "units_spec", None),
            base_tag=offset_tag,
            init=offset_init,
            strict=bool(getattr(ctx, "enforce_units", False)),
        )
    except Exception:
        return None
    new_subtree = AddNode(MulNode(scale_node, PowNode(trig_node, 2.0)), offset_node)
    root_new = replace_atom_in_ast(ctx.state.root, target, new_subtree)
    if root_new is None:
        return None

    scale_node_tag = getattr(scale_node, "tag", scale_tag)
    offset_node_tag = getattr(offset_node, "tag", offset_tag)

    def _init(root_new_inner: Node, model_new: nn.Module, *, _omega=omega, _phase=phase, _scale=scale_init, _offset=offset_init):
        atom_to_leaf = build_atom_to_leaf_map(root_new_inner, model_new)
        for atom in _collect_all_atoms(root_new_inner):
            if not isinstance(atom, AtomNode):
                continue
            leaf = atom_to_leaf.get(id(atom), None)
            if leaf is None:
                continue
            tag = getattr(atom, "tag", None)
            if tag == arg_tag:
                try:
                    _poly_zero_and_set(leaf, {(0,): float(_phase), (1,): float(_omega)})
                except Exception:
                    pass
            elif tag == scale_node_tag:
                try:
                    _set_constant_leaf_value(leaf, float(_scale))
                except Exception:
                    pass
            elif tag == offset_node_tag:
                try:
                    _set_constant_leaf_value(leaf, float(_offset))
                except Exception:
                    pass

    _init._after_analytic_init = True
    label = f"phase_hint_{trig_kind}_square"
    meta = {
        "structural": True,
        "pattern": "phase_hint_trig_closure",
        "pattern_family": "phase_hint_trig_closure",
        "phase_hint_trig_closure": True,
        "trig_kind": trig_kind,
        "trig_power": 2,
        "omega": float(omega),
        "phase": float(phase),
        "carrier_label": carrier_label,
        "screen_rel_rms": float(hit.get("rel_rms", float("inf"))),
        "signature": (
            int(atom_content_hash(target)),
            stable_int_hash("phase_hint_trig_closure"),
            stable_int_hash(carrier_label),
            stable_int_hash(trig_kind),
            2,
            int(round(float(omega) * 1.0e6)),
            int(round(float(phase) * 1.0e6)),
        ),
        "log": (
            f"[Stage B PhaseHint] Trying direct {trig_kind}_square closure "
            f"z={carrier_label} omega≈{omega:.6g}, phase≈{phase:.4g}, "
            f"rel={float(hit.get('rel_rms', float('inf'))):.2e}"
        ),
    }
    return Candidate(label=label, root=root_new, init_fn=_init, meta=meta)


def _make_phase_direct_fourier_candidate(
    ctx: StageBContext,
    target: AtomNode,
    hint: Any,
    target_dim,
    hit: Dict[str, Any],
) -> Optional[Candidate]:
    carrier_label = str(getattr(hint, "carrier_label", "z"))
    try:
        carrier_ast = clone_ast(getattr(hint, "carrier_ast"))
    except Exception:
        return None
    omega = float(hit["omega"])
    A = float(hit["A"])
    B = float(hit["B"])
    C = float(hit["offset"])
    base_tag = str(getattr(target, "tag", None) or "leaf")
    arg_cos_tag = f"{base_tag}_phase_fourier_cos_arg"
    arg_sin_tag = f"{base_tag}_phase_fourier_sin_arg"
    cos_tag = f"{base_tag}_phase_cos_scale"
    sin_tag = f"{base_tag}_phase_sin_scale"
    offset_tag = f"{base_tag}_phase_fourier_offset"
    var_idxs = tuple(sorted(int(v) for v in _collect_var_idxs_from_node(carrier_ast)))
    arg_cos_atom = AtomNode(
        kind="poly",
        var_idxs=var_idxs,
        kwargs={"degree": 1, "min_total": 0},
        tag=arg_cos_tag,
        inputs=(carrier_ast,),
    )
    arg_sin_atom = AtomNode(
        kind="poly",
        var_idxs=var_idxs,
        kwargs={"degree": 1, "min_total": 0},
        tag=arg_sin_tag,
        inputs=(clone_ast(carrier_ast),),
    )
    try:
        cos_scale = _make_unit_aware_scalar_atom(
            target_dim,
            getattr(ctx, "units_spec", None),
            base_tag=cos_tag,
            init=B,
            strict=bool(getattr(ctx, "enforce_units", False)),
        )
        sin_scale = _make_unit_aware_scalar_atom(
            target_dim,
            getattr(ctx, "units_spec", None),
            base_tag=sin_tag,
            init=A,
            strict=bool(getattr(ctx, "enforce_units", False)),
        )
        offset_node = _make_unit_aware_scalar_atom(
            target_dim,
            getattr(ctx, "units_spec", None),
            base_tag=offset_tag,
            init=C,
            strict=bool(getattr(ctx, "enforce_units", False)),
        )
    except Exception:
        return None
    new_subtree = AddNode(
        AddNode(MulNode(cos_scale, CosNode(arg_cos_atom)), MulNode(sin_scale, SinNode(arg_sin_atom))),
        offset_node,
    )
    root_new = replace_atom_in_ast(ctx.state.root, target, new_subtree)
    if root_new is None:
        return None

    cos_tag_eff = getattr(cos_scale, "tag", cos_tag)
    sin_tag_eff = getattr(sin_scale, "tag", sin_tag)
    offset_tag_eff = getattr(offset_node, "tag", offset_tag)

    def _init(root_new_inner: Node, model_new: nn.Module, *, _omega=omega, _A=A, _B=B, _C=C):
        atom_to_leaf = build_atom_to_leaf_map(root_new_inner, model_new)
        for atom in _collect_all_atoms(root_new_inner):
            if not isinstance(atom, AtomNode):
                continue
            leaf = atom_to_leaf.get(id(atom), None)
            if leaf is None:
                continue
            tag = getattr(atom, "tag", None)
            if tag == arg_cos_tag or tag == arg_sin_tag:
                try:
                    _poly_zero_and_set(leaf, {(0,): 0.0, (1,): float(_omega)})
                except Exception:
                    pass
            elif tag == cos_tag_eff:
                try:
                    _set_constant_leaf_value(leaf, float(_B))
                except Exception:
                    pass
            elif tag == sin_tag_eff:
                try:
                    _set_constant_leaf_value(leaf, float(_A))
                except Exception:
                    pass
            elif tag == offset_tag_eff:
                try:
                    _set_constant_leaf_value(leaf, float(_C))
                except Exception:
                    pass

    _init._after_analytic_init = True
    label = "phase_hint_fourier1"
    meta = {
        "structural": True,
        "pattern": "phase_hint_trig_closure",
        "pattern_family": "phase_hint_trig_closure",
        "phase_hint_trig_closure": True,
        "omega": float(omega),
        "carrier_label": carrier_label,
        "screen_rel_rms": float(hit.get("rel_rms", float("inf"))),
        "signature": (
            int(atom_content_hash(target)),
            stable_int_hash("phase_hint_trig_closure"),
            stable_int_hash(carrier_label),
            stable_int_hash("fourier1"),
            int(round(float(omega) * 1.0e6)),
        ),
        "log": (
            f"[Stage B PhaseHint] Trying direct Fourier closure "
            f"z={carrier_label} omega≈{omega:.6g}, rel={float(hit.get('rel_rms', float('inf'))):.2e}"
        ),
    }
    return Candidate(label=label, root=root_new, init_fn=_init, meta=meta)


def _phase_direct_target_is_dimless(ctx: StageBContext, target: AtomNode) -> bool:
    if not bool(getattr(ctx, "enforce_units", False)):
        return True
    dim = _phase_direct_target_dim(ctx, target)
    if dim is None:
        return False
    return _is_dimless is None or bool(_is_dimless(dim))


def _outer_link_hint_vars(hint: Any) -> set[int]:
    return _phase_hint_vars(hint)


def _outer_link_predict_y(link_name: str, arg: torch.Tensor) -> Optional[torch.Tensor]:
    link = str(link_name)
    if link == "arcsin":
        return torch.asin(torch.clamp(arg, -1.0 + 1.0e-12, 1.0 - 1.0e-12))
    if link == "arccos":
        return torch.acos(torch.clamp(arg, -1.0 + 1.0e-12, 1.0 - 1.0e-12))
    if link == "arctan":
        return torch.atan(arg)
    return None


def _outer_link_direct_fit_error(
    z: torch.Tensor,
    y: torch.Tensor,
    hint: Any,
) -> Optional[Dict[str, float]]:
    try:
        a = float(getattr(hint, "affine_a"))
        b = float(getattr(hint, "affine_b"))
        link = str(getattr(hint, "link_name"))
    except Exception:
        return None
    arg = a * z + b
    finite = torch.isfinite(arg) & torch.isfinite(y)
    if link in {"arcsin", "arccos"}:
        finite = finite & (arg >= -1.0 - 1.0e-8) & (arg <= 1.0 + 1.0e-8)
    if int(finite.sum().item()) < 200:
        return None
    argf = arg[finite]
    yf = y[finite]
    pred = _outer_link_predict_y(link, argf)
    if pred is None or not torch.isfinite(pred).all():
        return None
    resid = pred - yf
    centered = yf - torch.mean(yf)
    denom = torch.sqrt(torch.mean(centered * centered)).clamp_min(1.0e-30)
    rel = torch.sqrt(torch.mean(resid * resid)) / denom
    if not torch.isfinite(rel):
        return None
    return {
        "rel_rms": float(rel.item()),
        "domain_frac": float(finite.to(dtype=torch.float64).mean().item()),
        "affine_a": float(a),
        "affine_b": float(b),
    }


def _outer_link_branch_ok_torch(link_name: str, y: torch.Tensor) -> torch.Tensor:
    link = str(link_name)
    yy = y.reshape(-1)
    if link == "arcsin":
        ok = (yy >= (-0.5 * math.pi - 1.0e-10)) & (yy <= (0.5 * math.pi + 1.0e-10))
        ok = ok & (torch.cos(yy) >= -1.0e-8)
    elif link == "arccos":
        ok = (yy >= (-1.0e-10)) & (yy <= (math.pi + 1.0e-10))
        ok = ok & (torch.sin(yy) >= -1.0e-8)
    elif link == "arctan":
        ok = (yy > (-0.5 * math.pi + 1.0e-6)) & (yy < (0.5 * math.pi - 1.0e-6))
        ok = ok & (torch.abs(torch.cos(yy)) > 1.0e-6)
    else:
        ok = torch.zeros_like(yy, dtype=torch.bool)
    return ok & torch.isfinite(yy)


def _outer_link_transform_y_torch(link_name: str, y: torch.Tensor) -> Optional[torch.Tensor]:
    link = str(link_name)
    yy = y.reshape(-1)
    if link == "arcsin":
        return torch.sin(yy)
    if link == "arccos":
        return torch.cos(yy)
    if link == "arctan":
        return torch.tan(yy)
    return None


def _stageB_outer_feature_is_dimless(ctx: StageBContext, expr: Node) -> bool:
    if not bool(getattr(ctx, "enforce_units", False)):
        return True
    units_spec = getattr(ctx, "units_spec", None)
    if units_spec is None:
        return False
    try:
        from nestynet_sr.sr_core.units import eval_analytic_expr_dim

        dim = eval_analytic_expr_dim(expr, units_spec.x_dims)
        return dim is not None and (_is_dimless is None or bool(_is_dimless(dim)))
    except Exception:
        return False


def _stageB_outer_link_feature_groups(ctx: StageBContext, target: AtomNode) -> List[List[Tuple[str, Node]]]:
    """Build small dimensionless per-coordinate feature variants.

    The variants are deliberately local to the current effective atom inputs:
    after Stage A accepts a compound such as ``z=x0/x1``, this helper sees
    ``z`` as one coordinate and can pair ``1/z`` with ``cos(x2)`` without an
    expression-specific carrier.
    """

    groups: List[List[Tuple[str, Node]]] = []
    for expr in get_input_exprs(target):
        if not _stageB_outer_feature_is_dimless(ctx, expr):
            groups.append([])
            continue
        base = clone_ast(expr)
        label = ast_to_human_readable(base)
        variants: List[Tuple[str, Node]] = [(label, base)]
        variants.append((f"({label})^-1", PowNode(clone_ast(expr), -1.0)))
        variants.append((f"sin({label})", SinNode(clone_ast(expr))))
        variants.append((f"cos({label})", CosNode(clone_ast(expr))))

        dedup: List[Tuple[str, Node]] = []
        seen: set[str] = set()
        for lab, ast in variants:
            key = repr(ast)
            if key in seen:
                continue
            seen.add(key)
            dedup.append((lab, ast))
        groups.append(dedup)
    return groups


def _outer_ratpoly_exponents(n_features: int, degree: int) -> List[Tuple[int, ...]]:
    n = int(n_features)
    deg = int(degree)
    if n <= 0 or deg < 0:
        return []
    out: List[Tuple[int, ...]] = []

    def _rec(pos: int, remaining: int, cur: List[int]) -> None:
        if pos == n:
            out.append(tuple(int(v) for v in cur))
            return
        for p in range(remaining + 1):
            cur.append(p)
            _rec(pos + 1, remaining - p, cur)
            cur.pop()

    _rec(0, deg, [])
    out.sort(key=lambda e: (sum(e), e))
    return out


def _outer_ratpoly_design(values: Sequence[torch.Tensor], exps: Sequence[Tuple[int, ...]]) -> Optional[torch.Tensor]:
    if not values or not exps:
        return None
    cols: List[torch.Tensor] = []
    for exp in exps:
        if len(exp) != len(values):
            return None
        col = torch.ones_like(values[0], dtype=torch.float64)
        for v, p in zip(values, exp):
            pp = int(p)
            if pp == 1:
                col = col * v
            elif pp != 0:
                col = col * torch.pow(v, pp)
        cols.append(col)
    try:
        return torch.stack(cols, dim=1)
    except Exception:
        return None


def _outer_ratpoly_normalize(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    aa = np.asarray(a, dtype=np.float64).copy()
    bb = np.asarray(b, dtype=np.float64).copy()
    scale = None
    if bb.size and abs(float(bb[0])) > 1.0e-12:
        scale = float(bb[0])
    else:
        vals = np.concatenate([np.abs(aa.reshape(-1)), np.abs(bb.reshape(-1))])
        vmax = float(np.max(vals)) if vals.size else 0.0
        if vmax > 1.0e-12:
            scale = vmax
    if scale is None or not math.isfinite(scale) or abs(scale) <= 1.0e-12:
        return aa, bb
    aa = aa / scale
    bb = bb / scale
    if bb.size and bb[0] < 0.0:
        aa = -aa
        bb = -bb
    maxabs = float(np.max(np.abs(np.concatenate([aa.reshape(-1), bb.reshape(-1)])))) if (aa.size + bb.size) else 0.0
    tol = max(1.0e-10, 1.0e-8 * maxabs)
    aa[np.abs(aa) < tol] = 0.0
    bb[np.abs(bb) < tol] = 0.0
    return aa, bb


def _outer_ratpoly_snap_coeff(v: float) -> float:
    vv = float(v)
    if not math.isfinite(vv):
        return vv
    if abs(vv) < 1.0e-10:
        return 0.0
    q = Fraction(vv).limit_denominator(12)
    qf = float(q)
    if abs(vv - qf) <= max(2.0e-8, 2.0e-6 * max(1.0, abs(vv))):
        return qf
    return vv


def _outer_ratpoly_fit(
    feature_values: Sequence[torch.Tensor],
    y: torch.Tensor,
    link_name: str,
    *,
    degree: int,
    min_domain: float,
) -> Optional[Dict[str, Any]]:
    target = _outer_link_transform_y_torch(link_name, y)
    if target is None:
        return None
    vals = [v.detach().to(dtype=torch.float64).reshape(-1) for v in feature_values]
    yy = y.detach().to(dtype=torch.float64).reshape(-1)
    tt = target.detach().to(dtype=torch.float64).reshape(-1)
    n = min([int(yy.shape[0]), int(tt.shape[0]), *(int(v.shape[0]) for v in vals)])
    if n < 256:
        return None
    vals = [v[:n] for v in vals]
    yy = yy[:n]
    tt = tt[:n]
    branch_ok = _outer_link_branch_ok_torch(link_name, yy)
    finite = branch_ok & torch.isfinite(tt)
    for v in vals:
        finite = finite & torch.isfinite(v)
    domain_frac = float(finite.to(dtype=torch.float64).mean().item()) if n else 0.0
    if domain_frac < float(min_domain):
        return None
    vals = [v[finite] for v in vals]
    yy = yy[finite]
    tt = tt[finite]
    n_ok = int(yy.shape[0])
    if n_ok < 256 or float(torch.std(tt).item()) <= 1.0e-14:
        return None

    exps = _outer_ratpoly_exponents(len(vals), int(degree))
    Phi = _outer_ratpoly_design(vals, exps)
    if Phi is None or Phi.ndim != 2 or Phi.shape[1] < 2:
        return None
    m = int(Phi.shape[1])
    if n_ok < 2 * m + 16:
        return None

    # Deterministic probe split.  This is proposal evidence; official Stage-B
    # validation still decides whether the visible closure is accepted.
    rng = np.random.default_rng(39017 + 97 * int(degree) + 17 * len(vals))
    perm = torch.as_tensor(rng.permutation(n_ok), dtype=torch.long, device=Phi.device)
    n_train = max(128, int(0.65 * n_ok))
    if n_ok - n_train < 96:
        return None
    tr = perm[:n_train]
    va = perm[n_train:]
    Phi_tr = Phi[tr]
    Phi_va = Phi[va]
    t_tr = tt[tr]
    t_va = tt[va]
    y_va = yy[va]

    A = torch.cat([Phi_tr, -t_tr.unsqueeze(1) * Phi_tr], dim=1)
    if not torch.isfinite(A).all():
        return None
    try:
        _, _, vh = torch.linalg.svd(A, full_matrices=False)
    except Exception:
        return None
    sol = vh[-1, :]
    a = sol[:m].detach().cpu().numpy()
    b = sol[m:].detach().cpu().numpy()
    a, b = _outer_ratpoly_normalize(a, b)
    a_t = torch.as_tensor(a, dtype=torch.float64, device=Phi_va.device)
    b_t = torch.as_tensor(b, dtype=torch.float64, device=Phi_va.device)
    P = Phi_va @ a_t
    Q = Phi_va @ b_t
    q_abs = torch.abs(Q)
    q_max = float(torch.max(q_abs).item()) if Q.numel() else 0.0
    q_eps = max(1.0e-10, 1.0e-8 * q_max)
    q_ok = torch.isfinite(Q) & (q_abs > q_eps)
    if float(q_ok.to(dtype=torch.float64).mean().item()) < float(min_domain):
        return None
    arg = P[q_ok] / Q[q_ok]
    y_ref = y_va[q_ok]
    if link_name in {"arcsin", "arccos"}:
        arg_ok = torch.isfinite(arg) & (arg >= -1.0 - 1.0e-8) & (arg <= 1.0 + 1.0e-8)
    else:
        arg_ok = torch.isfinite(arg)
    if float(arg_ok.to(dtype=torch.float64).mean().item()) < float(min_domain):
        return None
    arg = torch.clamp(arg[arg_ok], -1.0 + 1.0e-12, 1.0 - 1.0e-12) if link_name in {"arcsin", "arccos"} else arg[arg_ok]
    y_ref = y_ref[arg_ok]
    pred_y = _outer_link_predict_y(link_name, arg)
    if pred_y is None or not torch.isfinite(pred_y).all():
        return None
    resid_y = pred_y - y_ref
    denom_y = torch.sqrt(torch.mean((y_ref - torch.mean(y_ref)) ** 2)).clamp_min(1.0e-30)
    rel_y = torch.sqrt(torch.mean(resid_y * resid_y)) / denom_y
    if not torch.isfinite(rel_y):
        return None

    pred_t = (Phi_va[q_ok] @ a_t) / (Phi_va[q_ok] @ b_t)
    resid_t = pred_t - t_va[q_ok]
    denom_t = torch.sqrt(torch.mean((t_va[q_ok] - torch.mean(t_va[q_ok])) ** 2)).clamp_min(1.0e-30)
    rel_t = torch.sqrt(torch.mean(resid_t * resid_t)) / denom_t

    a = np.asarray([_outer_ratpoly_snap_coeff(float(v)) for v in a], dtype=np.float64)
    b = np.asarray([_outer_ratpoly_snap_coeff(float(v)) for v in b], dtype=np.float64)
    return {
        "degree": int(degree),
        "exps": tuple(tuple(int(p) for p in e) for e in exps),
        "coeff_num": tuple(float(v) for v in a),
        "coeff_den": tuple(float(v) for v in b),
        "rel_rms": float(rel_y.item()),
        "rel_rms_transform": float(rel_t.item()) if torch.isfinite(rel_t) else float("inf"),
        "domain_frac": float(domain_frac),
        "arg_domain_frac": float(arg_ok.to(dtype=torch.float64).mean().item()),
        "den_domain_frac": float(q_ok.to(dtype=torch.float64).mean().item()),
    }


def _outer_ratpoly_monomial_ast(features: Sequence[Node], exp: Sequence[int]) -> Node:
    factors: List[Node] = []
    for feat, pp in zip(features, exp):
        p = int(pp)
        if p == 0:
            continue
        if p == 1:
            factors.append(clone_ast(feat))
        else:
            factors.append(PowNode(clone_ast(feat), float(p)))
    if not factors:
        return ConstNode(1.0)
    out = factors[0]
    for f in factors[1:]:
        out = MulNode(out, f)
    return out


def _outer_ratpoly_sum_ast(features: Sequence[Node], exps: Sequence[Tuple[int, ...]], coeffs: Sequence[float]) -> Optional[Node]:
    terms: List[Node] = []
    for exp, coeff in zip(exps, coeffs):
        c = float(coeff)
        if not math.isfinite(c) or abs(c) <= 1.0e-12:
            continue
        mono = _outer_ratpoly_monomial_ast(features, exp)
        if isinstance(mono, ConstNode) and abs(float(mono.value) - 1.0) <= 1.0e-14:
            term = ConstNode(c)
        elif abs(c - 1.0) <= 1.0e-12:
            term = mono
        elif abs(c + 1.0) <= 1.0e-12:
            term = MulNode(ConstNode(-1.0), mono)
        else:
            term = MulNode(ConstNode(c), mono)
        terms.append(term)
    if not terms:
        return None
    out = terms[0]
    for t in terms[1:]:
        out = AddNode(out, t)
    return out


def _make_outer_inverse_trig_rational_candidate(
    ctx: StageBContext,
    target: AtomNode,
    link: str,
    feature_labels: Sequence[str],
    feature_asts: Sequence[Node],
    fit: Dict[str, Any],
) -> Optional[Candidate]:
    inv_node = {
        "arcsin": AsinNode,
        "arccos": AcosNode,
        "arctan": AtanNode,
    }.get(str(link))
    if inv_node is None:
        return None
    exps = tuple(tuple(int(p) for p in e) for e in fit.get("exps", ()))
    coeff_num = tuple(float(v) for v in fit.get("coeff_num", ()))
    coeff_den = tuple(float(v) for v in fit.get("coeff_den", ()))
    if not exps or len(coeff_num) != len(exps) or len(coeff_den) != len(exps):
        return None
    num = _outer_ratpoly_sum_ast(feature_asts, exps, coeff_num)
    den = _outer_ratpoly_sum_ast(feature_asts, exps, coeff_den)
    if num is None or den is None:
        return None
    arg = MulNode(num, PowNode(den, -1.0))
    root_new = replace_atom_in_ast(ctx.state.root, target, inv_node(arg))
    if root_new is None:
        return None
    degree = int(fit.get("degree", 0))
    labels = tuple(str(v) for v in feature_labels)
    meta = {
        "structural": True,
        "pattern": "inverse_trig_outer_rational_closure",
        "pattern_family": "inverse_trig_outer_rational_closure",
        "inverse_trig_outer_rational_closure": True,
        "link_name": str(link),
        "rational_degree": int(degree),
        "feature_labels": labels,
        "screen_rel_rms": float(fit.get("rel_rms", float("inf"))),
        "screen_rel_rms_transform": float(fit.get("rel_rms_transform", float("inf"))),
        "signature": (
            int(atom_content_hash(target)),
            stable_int_hash("inverse_trig_outer_rational_closure"),
            stable_int_hash(str(link)),
            int(degree),
            tuple(stable_int_hash(v) for v in labels),
        ),
        "log": (
            f"[Stage B OuterLink] Trying rational {str(link)} closure "
            f"features={labels}, deg={degree}, rel={float(fit.get('rel_rms', float('inf'))):.2e}"
        ),
    }
    return Candidate(label=f"inverse_trig_outer_rational_{str(link)}_deg{degree}", root=root_new, init_fn=None, meta=meta)


def _make_outer_inverse_trig_candidate(
    ctx: StageBContext,
    target: AtomNode,
    hint: Any,
    hit: Dict[str, float],
) -> Optional[Candidate]:
    carrier_label = str(getattr(hint, "carrier_label", "z"))
    link = str(getattr(hint, "link_name", ""))
    try:
        carrier_ast = clone_ast(getattr(hint, "carrier_ast"))
    except Exception:
        return None
    inv_node = {
        "arcsin": AsinNode,
        "arccos": AcosNode,
        "arctan": AtanNode,
    }.get(link)
    if inv_node is None:
        return None

    base_tag = str(getattr(target, "tag", None) or "leaf")
    arg_tag = f"{base_tag}_outer_{link}_arg"
    var_idxs = tuple(sorted(int(v) for v in _collect_var_idxs_from_node(carrier_ast)))
    if not var_idxs:
        return None
    arg_atom = AtomNode(
        kind="poly",
        var_idxs=var_idxs,
        kwargs={"degree": 1, "min_total": 0},
        tag=arg_tag,
        inputs=(carrier_ast,),
    )
    root_new = replace_atom_in_ast(ctx.state.root, target, inv_node(arg_atom))
    if root_new is None:
        return None
    a = float(hit.get("affine_a", getattr(hint, "affine_a", 1.0)))
    b = float(hit.get("affine_b", getattr(hint, "affine_b", 0.0)))

    def _init(root_new_inner: Node, model_new: nn.Module, *, _a=a, _b=b):
        atom_to_leaf = build_atom_to_leaf_map(root_new_inner, model_new)
        for atom in _collect_all_atoms(root_new_inner):
            if not isinstance(atom, AtomNode):
                continue
            if getattr(atom, "tag", None) != arg_tag:
                continue
            leaf = atom_to_leaf.get(id(atom), None)
            if leaf is None:
                continue
            try:
                _poly_zero_and_set(leaf, {(0,): float(_b), (1,): float(_a)})
            except Exception:
                pass

    _init._after_analytic_init = True
    label = f"inverse_trig_outer_{link}"
    meta = {
        "structural": True,
        "pattern": "inverse_trig_outer_closure",
        "pattern_family": "inverse_trig_outer_closure",
        "inverse_trig_outer_closure": True,
        "link_name": link,
        "transform_name": str(getattr(hint, "transform_name", "")),
        "carrier_label": carrier_label,
        "screen_rel_rms": float(hit.get("rel_rms", float("inf"))),
        "signature": (
            int(atom_content_hash(target)),
            stable_int_hash("inverse_trig_outer_closure"),
            stable_int_hash(link),
            stable_int_hash(carrier_label),
            int(round(float(a) * 1.0e6)),
            int(round(float(b) * 1.0e6)),
        ),
        "log": (
            f"[Stage B OuterLink] Trying direct {link} closure "
            f"z={carrier_label}, a≈{a:.6g}, b≈{b:.6g}, "
            f"rel={float(hit.get('rel_rms', float('inf'))):.2e}"
        ),
    }
    return Candidate(label=label, root=root_new, init_fn=_init, meta=meta)


class RuleInverseTrigOuterClosure(StageBRule):
    """Visible inverse-trig closures from inside-out outer-link hints."""

    name = "inverse_trig_outer_closure"
    exhaustive = True

    def iter_targets(self, ctx: StageBContext):
        if not getattr(ctx, "outer_link_hints", None):
            return []
        atoms = [
            atom
            for atom in _collect_all_atoms(ctx.state.root)
            if isinstance(atom, AtomNode)
            and str(getattr(atom, "kind", "")).lower() == "nn"
        ]
        if len(atoms) != 1:
            return []
        target = atoms[0]
        if ctx.state.root is not target:
            return []
        return [target]

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        if not isinstance(target, AtomNode):
            return []
        hints = list(getattr(ctx, "outer_link_hints", []) or [])
        if not hints:
            return []
        if not _phase_direct_target_is_dimless(ctx, target):
            return []
        target_vars = set(int(v) for v in getattr(target, "var_idxs", ()) or ())
        if not target_vars:
            return []

        try:
            max_points = int(getattr(ctx.lm_hp, "stageB_last_hard_trig_power_max_points", 5000) or 5000)
        except Exception:
            max_points = 5000
        data = _gather_phase_direct_xy(ctx, max_points=max_points)
        if data is None:
            return []
        X, y = data
        y = y.detach().to(dtype=torch.float64).reshape(-1)
        finite_y = torch.isfinite(y)
        if int(finite_y.sum().item()) < 200:
            return []

        try:
            max_rel = float(getattr(ctx.lm_hp, "stageB_last_hard_trig_power_screen_rel_rms", 2.0e-2))
        except Exception:
            max_rel = 2.0e-2
        max_rel = _stageB_noisy_rel_rms_threshold(ctx, max_rel, y_values=y[finite_y])
        try:
            min_domain = float(getattr(ctx.lm_hp, "macro_domain_ok_frac", 0.98))
        except Exception:
            min_domain = 0.98

        cands: List[Candidate] = []
        for hint in hints[:8]:
            hint_vars = _outer_link_hint_vars(hint)
            if not hint_vars or not hint_vars.issubset(target_vars):
                continue
            if not _phase_hint_carrier_is_dimless(ctx, hint):
                continue
            try:
                if float(getattr(hint, "domain_ok_frac", 0.0)) < min_domain:
                    continue
                if float(getattr(hint, "branch_ok_frac", 0.0)) < min_domain:
                    continue
            except Exception:
                continue
            try:
                z = eval_input_expr(getattr(hint, "carrier_ast"), X).reshape(-1).detach().to(dtype=torch.float64)
            except Exception:
                continue
            finite = finite_y & torch.isfinite(z)
            if int(finite.sum().item()) < 200:
                continue
            hit = _outer_link_direct_fit_error(z[finite], y[finite], hint)
            if hit is None:
                continue
            if float(hit.get("domain_frac", 0.0)) < min_domain:
                continue
            if float(hit.get("rel_rms", float("inf"))) > max_rel:
                continue
            cand = _make_outer_inverse_trig_candidate(ctx, target, hint, hit)
            if cand is not None:
                cands.append(cand)

        cands.sort(key=lambda c: float((c.meta or {}).get("screen_rel_rms", float("inf"))))
        return cands[:8]


class RuleInverseTrigRationalOuterClosure(StageBRule):
    """Low-arity rational closure for inverse-trig outer links.

    This is the dynamic counterpart to the Stage-0 inverse-link scan.  It runs
    on the current accepted Stage-B atom, so compounds discovered by Stage A
    are visible as effective inputs.  The screen is intentionally bounded:
    arity <= 2, degree 1 first, degree 2 only if degree 1 has no validated
    proposal.
    """

    name = "inverse_trig_outer_rational_closure"
    exhaustive = True

    def iter_targets(self, ctx: StageBContext):
        atoms = [
            atom
            for atom in _collect_all_atoms(ctx.state.root)
            if isinstance(atom, AtomNode)
            and str(getattr(atom, "kind", "")).lower() == "nn"
        ]
        if len(atoms) != 1:
            return []
        target = atoms[0]
        if ctx.state.root is not target:
            return []
        if int(effective_arity(target)) > 2:
            return []
        return [target]

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        if not isinstance(target, AtomNode):
            return []
        if not _phase_direct_target_is_dimless(ctx, target):
            return []
        input_exprs = tuple(get_input_exprs(target))
        if not input_exprs or len(input_exprs) > 2:
            return []

        groups = _stageB_outer_link_feature_groups(ctx, target)
        if not groups or any(not g for g in groups):
            return []

        try:
            max_points = int(getattr(ctx.lm_hp, "stageB_last_hard_trig_power_max_points", 5000) or 5000)
        except Exception:
            max_points = 5000
        data = _gather_phase_direct_xy(ctx, max_points=max_points)
        if data is None:
            return []
        X, y = data
        X = X.detach().to(dtype=torch.float64)
        y = y.detach().to(dtype=torch.float64).reshape(-1)
        if X.ndim != 2 or int(X.shape[0]) < 256:
            return []

        try:
            max_rel = float(getattr(ctx.lm_hp, "stageB_last_hard_trig_power_screen_rel_rms", 2.0e-2))
        except Exception:
            max_rel = 2.0e-2
        max_rel = _stageB_noisy_rel_rms_threshold(ctx, max_rel, y_values=y)
        try:
            min_domain = float(getattr(ctx.lm_hp, "macro_domain_ok_frac", 0.98))
        except Exception:
            min_domain = 0.98

        feature_sets: List[Tuple[Tuple[str, ...], Tuple[Node, ...], Tuple[torch.Tensor, ...]]] = []
        if len(groups) == 1:
            for lab, ast in groups[0]:
                try:
                    vals = eval_input_expr(ast, X).reshape(-1).detach().to(dtype=torch.float64)
                except Exception:
                    continue
                feature_sets.append(((lab,), (clone_ast(ast),), (vals,)))
        else:
            # Prefer two-coordinate evidence; single-coordinate fallbacks are
            # still allowed but are less likely to beat the simpler direct scan.
            for lab0, ast0 in groups[0]:
                for lab1, ast1 in groups[1]:
                    try:
                        v0 = eval_input_expr(ast0, X).reshape(-1).detach().to(dtype=torch.float64)
                        v1 = eval_input_expr(ast1, X).reshape(-1).detach().to(dtype=torch.float64)
                    except Exception:
                        continue
                    feature_sets.append(((lab0, lab1), (clone_ast(ast0), clone_ast(ast1)), (v0, v1)))
            for group in groups:
                for lab, ast in group:
                    try:
                        vals = eval_input_expr(ast, X).reshape(-1).detach().to(dtype=torch.float64)
                    except Exception:
                        continue
                    feature_sets.append(((lab,), (clone_ast(ast),), (vals,)))

        if not feature_sets:
            return []

        cands: List[Candidate] = []
        for degree in (1, 2):
            degree_cands: List[Candidate] = []
            for link in ("arcsin", "arccos", "arctan"):
                if float(_outer_link_branch_ok_torch(link, y).to(dtype=torch.float64).mean().item()) < min_domain:
                    continue
                for labels, asts, vals in feature_sets:
                    fit = _outer_ratpoly_fit(vals, y, link, degree=int(degree), min_domain=float(min_domain))
                    if fit is None:
                        continue
                    if float(fit.get("rel_rms", float("inf"))) > max_rel:
                        continue
                    cand = _make_outer_inverse_trig_rational_candidate(ctx, target, link, labels, asts, fit)
                    if cand is not None:
                        degree_cands.append(cand)
            degree_cands.sort(
                key=lambda c: (
                    float((c.meta or {}).get("screen_rel_rms", float("inf"))),
                    len(tuple((c.meta or {}).get("feature_labels", ()) or ())),
                    str(c.label),
                )
            )
            if degree_cands:
                cands = degree_cands
                break
        return cands[:8]


class RulePhaseHintTrigClosure(StageBRule):
    """Direct visible trig/Fourier closures from Stage-0 phase hints."""

    name = "phase_hint_trig_closure"
    exhaustive = True

    def iter_targets(self, ctx: StageBContext):
        if not getattr(ctx, "phase_hints", None):
            return []
        atoms = [
            atom
            for atom in _collect_all_atoms(ctx.state.root)
            if isinstance(atom, AtomNode)
            and str(getattr(atom, "kind", "")).lower() == "nn"
        ]
        if len(atoms) != 1:
            return []
        target = atoms[0]
        # v1 deliberately handles the pure unresolved surrogate case only.
        # Mixed/prefactored direct phase closure needs residual isolation.
        if ctx.state.root is not target:
            return []
        return [target]

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        if not isinstance(target, AtomNode):
            return []
        hints = list(getattr(ctx, "phase_hints", []) or [])
        if not hints:
            return []
        target_vars = set(int(v) for v in getattr(target, "var_idxs", ()) or ())
        if not target_vars:
            return []

        try:
            max_points = int(getattr(ctx.lm_hp, "stageB_last_hard_trig_power_max_points", 5000) or 5000)
        except Exception:
            max_points = 5000
        data = _gather_phase_direct_xy(ctx, max_points=max_points)
        if data is None:
            return []
        X, y = data
        y = y.detach().to(dtype=torch.float64).reshape(-1)
        finite_y = torch.isfinite(y)
        if int(finite_y.sum().item()) < 200:
            return []

        try:
            max_rel = float(getattr(ctx.lm_hp, "stageB_last_hard_trig_power_screen_rel_rms", 2.0e-2))
        except Exception:
            max_rel = 2.0e-2
        max_rel = _stageB_noisy_rel_rms_threshold(ctx, max_rel, y_values=y[finite_y])
        target_dim = _phase_direct_target_dim(ctx, target)
        if bool(getattr(ctx, "enforce_units", False)) and target_dim is None:
            return []

        cands: List[Candidate] = []
        for hint in hints[:8]:
            hint_vars = _phase_hint_vars(hint)
            if not hint_vars or not hint_vars.issubset(target_vars):
                continue
            if not _phase_hint_carrier_is_dimless(ctx, hint):
                if bool(getattr(ctx, "verbose", False)):
                    ctx.log(
                        "[Stage B PhaseHint] Skipping direct closure: "
                        f"carrier {getattr(hint, 'carrier_label', 'z')} is not dimensionless."
                    )
                continue
            try:
                z = eval_input_expr(getattr(hint, "carrier_ast"), X).reshape(-1).detach().to(dtype=torch.float64)
            except Exception:
                continue
            finite = finite_y & torch.isfinite(z)
            if int(finite.sum().item()) < 200:
                continue
            zf = z[finite]
            yf = y[finite]
            if float(torch.std(yf).item()) <= 1.0e-14:
                continue

            for omega in _phase_hint_arg_omegas(hint)[:6]:
                for trig_kind in ("sin", "cos"):
                    hit = _phase_direct_square_fit(zf, yf, float(omega), trig_kind)
                    if hit is None:
                        continue
                    if float(hit.get("rel_rms", float("inf"))) > max_rel:
                        continue
                    cand = _make_phase_direct_square_candidate(ctx, target, hint, target_dim, hit)
                    if cand is not None:
                        cands.append(cand)
            for omega in _phase_hint_fourier_omegas(hint)[:6]:
                hit = _phase_direct_affine_fit(zf, yf, float(omega))
                if hit is None:
                    continue
                if float(hit.get("rel_rms", float("inf"))) > max_rel:
                    continue
                hit = dict(hit)
                hit["omega"] = float(omega)
                cand = _make_phase_direct_fourier_candidate(ctx, target, hint, target_dim, hit)
                if cand is not None:
                    cands.append(cand)

        cands.sort(key=lambda c: float((c.meta or {}).get("screen_rel_rms", float("inf"))))
        return cands[:8]


def _phase_reciprocal_trig_power_fit(
    z: torch.Tensor,
    y: torch.Tensor,
    omega: float,
    *,
    power: int,
    max_offset_rel: float,
) -> Optional[Dict[str, float]]:
    try:
        q = abs(int(power))
        if q <= 0:
            return None
        z = z.detach().to(dtype=torch.float64).reshape(-1)
        y = y.detach().to(dtype=torch.float64).reshape(-1)
        finite = torch.isfinite(z) & torch.isfinite(y)
        scale = torch.quantile(y[finite].abs(), 0.75).clamp_min(1.0) if bool(finite.any()) else torch.tensor(1.0)
        finite = finite & (y.abs() > (1.0e-14 * scale))
        if int(finite.sum().item()) < 200:
            return None
        zf = z[finite]
        yf = y[finite]
        signs = torch.sign(yf)
        sign_ref = float(torch.sign(torch.median(yf)).item())
        if sign_ref == 0.0:
            sign_ref = 1.0
        sign_frac = float((signs == sign_ref).to(dtype=torch.float64).mean().item())
        if sign_frac < 0.98:
            return None
        yy = torch.pow(yf.abs().clamp_min(1.0e-300), -1.0 / float(q))
        if not torch.isfinite(yy).all() or float(torch.std(yy).item()) <= 1.0e-14:
            return None
        fit = _solve_affine_trig_1d(zf, yy, float(omega))
        if fit is None:
            return None
        offset_rel = float(fit.get("offset_rel", float("inf")))
        if offset_rel > float(max_offset_rel):
            return None
        A = float(fit["A"])
        B = float(fit["B"])
        phase_sin = _snap_phase_for_even_power(math.atan2(B, A))
        phase_cos = _snap_phase_for_even_power(math.atan2(-A, B))
        use_cos = abs(phase_cos) + 1.0e-12 < abs(phase_sin)
        trig_kind = "cos" if use_cos else "sin"
        phase = phase_cos if use_cos else phase_sin
        amp = float(fit["amp"])
        scale_init = float(sign_ref) * (amp ** (-float(q)))
        if not all(math.isfinite(v) for v in (phase, amp, scale_init)):
            return None
        return {
            "power": int(power),
            "omega": float(omega),
            "phase": float(phase),
            "trig_kind": trig_kind,
            "scale_init": float(scale_init),
            "rel_rms": float(fit["rel_rms"]),
            "offset_rel": float(offset_rel),
            "sign_frac": float(sign_frac),
            "screen_score": float(fit["rel_rms"]) + 0.05 * offset_rel + 0.001 * float(q),
        }
    except Exception:
        return None


def _make_phase_reciprocal_trig_power_candidate(
    ctx: StageBContext,
    target: AtomNode,
    hint: Any,
    target_dim,
    hit: Dict[str, Any],
) -> Optional[Candidate]:
    carrier_label = str(getattr(hint, "carrier_label", "z"))
    try:
        carrier_ast = clone_ast(getattr(hint, "carrier_ast"))
    except Exception:
        return None
    power = int(hit["power"])
    omega = float(hit["omega"])
    phase = float(hit["phase"])
    trig_kind = str(hit["trig_kind"])
    scale_init = float(hit["scale_init"])
    base_tag = str(getattr(target, "tag", None) or "leaf")
    arg_tag = f"{base_tag}_phase_recip_arg"
    scale_tag = f"{base_tag}_phase_recip_scale"
    var_idxs = tuple(sorted(int(v) for v in _collect_var_idxs_from_node(carrier_ast)))
    if not var_idxs:
        return None
    arg_atom = AtomNode(
        kind="poly",
        var_idxs=var_idxs,
        kwargs={"degree": 1, "min_total": 0},
        tag=arg_tag,
        inputs=(carrier_ast,),
    )
    trig_node = CosNode(arg_atom) if trig_kind == "cos" else SinNode(arg_atom)
    scale_node = _make_unit_aware_scalar_atom(
        target_dim,
        getattr(ctx, "units_spec", None),
        base_tag=scale_tag,
        init=scale_init,
        strict=bool(getattr(ctx, "enforce_units", False)),
    )
    new_subtree = MulNode(scale_node, PowNode(trig_node, float(power)))
    root_new = replace_atom_in_ast(ctx.state.root, target, new_subtree)
    if root_new is None:
        return None

    scale_node_tag = getattr(scale_node, "tag", scale_tag)

    def _init(root_new_inner: Node, model_new: nn.Module, *, _phase=phase, _omega=omega, _scale=scale_init):
        atom_to_leaf = build_atom_to_leaf_map(root_new_inner, model_new)
        for atom in _collect_all_atoms(root_new_inner):
            if not isinstance(atom, AtomNode):
                continue
            leaf = atom_to_leaf.get(id(atom), None)
            if leaf is None:
                continue
            tag = getattr(atom, "tag", None)
            if tag == arg_tag:
                try:
                    _poly_zero_and_set(leaf, {(0,): float(_phase), (1,): float(_omega)})
                except Exception:
                    pass
            elif tag == scale_node_tag:
                try:
                    _set_constant_leaf_value(leaf, float(_scale))
                except Exception:
                    pass

    _init._after_analytic_init = True
    label = f"phase_hint_reciprocal_{trig_kind}_p{power}"
    meta = {
        "structural": True,
        "pattern": "phase_hint_reciprocal_trig_power",
        "pattern_family": "phase_hint_reciprocal_trig_power",
        "phase_hint_reciprocal_trig_power": True,
        "trig_kind": trig_kind,
        "trig_power": int(power),
        "omega": float(omega),
        "phase": float(phase),
        "carrier_label": carrier_label,
        "screen_rel_rms": float(hit.get("rel_rms", float("inf"))),
        "signature": (
            int(atom_content_hash(target)),
            stable_int_hash("phase_hint_reciprocal_trig_power"),
            stable_int_hash(carrier_label),
            stable_int_hash(trig_kind),
            int(power),
            int(round(float(omega) * 1.0e6)),
            int(round(float(phase) * 1.0e6)),
        ),
        "log": (
            f"[Stage B PhaseHint] Trying reciprocal trig-power closure "
            f"{trig_kind}(ωz+φ)^{power}, z={carrier_label}, "
            f"ω≈{omega:.6g}, φ≈{phase:.4g}, rel={float(hit.get('rel_rms', float('inf'))):.2e}"
        ),
    }
    return Candidate(label=label, root=root_new, init_fn=_init, meta=meta)


class RulePhaseHintReciprocalTrigPower(StageBRule):
    """Direct visible reciprocal trig-power closures from PhaseScan hints."""

    name = "phase_hint_reciprocal_trig_power"
    exhaustive = True

    def iter_targets(self, ctx: StageBContext):
        if not getattr(ctx, "phase_hints", None):
            return []
        atoms = [
            atom
            for atom in _collect_all_atoms(ctx.state.root)
            if isinstance(atom, AtomNode)
            and str(getattr(atom, "kind", "")).lower() == "nn"
        ]
        if len(atoms) != 1:
            return []
        target = atoms[0]
        if ctx.state.root is not target:
            return []
        return [target]

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        if not isinstance(target, AtomNode):
            return []
        hints = list(getattr(ctx, "phase_hints", []) or [])
        if not hints:
            return []
        target_vars = set(int(v) for v in getattr(target, "var_idxs", ()) or ())
        if not target_vars:
            return []

        try:
            max_points = int(getattr(ctx.lm_hp, "stageB_last_hard_trig_power_max_points", 5000) or 5000)
        except Exception:
            max_points = 5000
        data = _gather_phase_direct_xy(ctx, max_points=max_points)
        if data is None:
            return []
        X, y = data
        y = y.detach().to(dtype=torch.float64).reshape(-1)
        finite_y = torch.isfinite(y)
        if int(finite_y.sum().item()) < 200:
            return []

        try:
            max_rel = float(getattr(ctx.lm_hp, "stageB_last_hard_trig_power_screen_rel_rms", 2.0e-2))
        except Exception:
            max_rel = 2.0e-2
        max_rel = _stageB_noisy_rel_rms_threshold(ctx, max_rel, y_values=y[finite_y])
        try:
            max_offset_rel = float(getattr(ctx.lm_hp, "stageB_last_hard_trig_power_max_offset_rel", 0.15))
        except Exception:
            max_offset_rel = 0.15
        target_dim = _phase_direct_target_dim(ctx, target)
        if bool(getattr(ctx, "enforce_units", False)) and target_dim is None:
            return []

        cands: List[Candidate] = []
        for hint in hints[:8]:
            hint_vars = _phase_hint_vars(hint)
            if not hint_vars or not hint_vars.issubset(target_vars):
                continue
            if not _phase_hint_carrier_is_dimless(ctx, hint):
                continue
            try:
                z = eval_input_expr(getattr(hint, "carrier_ast"), X).reshape(-1).detach().to(dtype=torch.float64)
            except Exception:
                continue
            finite = finite_y & torch.isfinite(z)
            if int(finite.sum().item()) < 200:
                continue
            omega_vals = _dedupe_positive_floats(
                _phase_hint_arg_omegas(hint) + _phase_hint_fourier_omegas(hint)
            )
            for power in _LAST_HARD_TRIG_POWER_POWERS:
                for omega in omega_vals[:8]:
                    hit = _phase_reciprocal_trig_power_fit(
                        z[finite],
                        y[finite],
                        float(omega),
                        power=int(power),
                        max_offset_rel=max_offset_rel,
                    )
                    if hit is None:
                        continue
                    if float(hit.get("rel_rms", float("inf"))) > max_rel:
                        continue
                    cand = _make_phase_reciprocal_trig_power_candidate(
                        ctx,
                        target,
                        hint,
                        target_dim,
                        hit,
                    )
                    if cand is not None:
                        cands.append(cand)

        cands.sort(key=lambda c: float((c.meta or {}).get("screen_rel_rms", float("inf"))))
        return cands[:8]


def _phase_context_hint_vars(hint: Any) -> set[int]:
    out = _phase_hint_vars(hint)
    try:
        for ast in tuple(getattr(hint, "context_asts", ()) or ()):
            out.update(int(v) for v in _collect_var_idxs_from_node(ast))
    except Exception:
        pass
    return out


def _phase_context_omegas(hint: Any) -> List[float]:
    vals: List[float] = []
    details = getattr(hint, "details", {}) if isinstance(getattr(hint, "details", None), dict) else {}
    try:
        w = details.get("omega", None)
        if w is not None:
            vals.append(float(w))
    except Exception:
        pass
    for term in tuple(getattr(hint, "omega_terms", ()) or ()):
        try:
            fam = str(getattr(term, "family", ""))
            if fam == "fourier":
                vals.append(float(getattr(term, "actual_omega")))
            vals.append(float(getattr(term, "base_omega")))
        except Exception:
            continue
    return _dedupe_positive_floats(vals)


def _phase_context_scale_dim(ctx: StageBContext):
    if not bool(getattr(ctx, "enforce_units", False)):
        return None
    units_spec = getattr(ctx, "units_spec", None)
    if units_spec is None:
        return None
    try:
        return tuple(units_spec.unit_system.dimless())
    except Exception:
        return None


def _phase_context_feature_units_ok(ctx: StageBContext, context_ast: Node, target_dim) -> bool:
    if not bool(getattr(ctx, "enforce_units", False)):
        return True
    units_spec = getattr(ctx, "units_spec", None)
    if units_spec is None or target_dim is None:
        return False
    try:
        from nestynet_sr.sr_core.units import eval_analytic_expr_dim

        dim = eval_analytic_expr_dim(context_ast, units_spec.x_dims)
        return dim is not None and tuple(dim) == tuple(target_dim)
    except Exception:
        return False


def _fit_context_one_minus_cos(
    z: torch.Tensor,
    p: torch.Tensor,
    y: torch.Tensor,
    omega: float,
) -> Optional[Dict[str, float]]:
    """Fit and score y ≈ scale * p * (1 - cos(omega*z + phase))."""
    try:
        w = float(omega)
        z = z.detach().to(dtype=torch.float64).reshape(-1)
        p = p.detach().to(dtype=torch.float64).reshape(-1)
        y = y.detach().to(dtype=torch.float64).reshape(-1)
        finite = torch.isfinite(z) & torch.isfinite(p) & torch.isfinite(y)
        if int(finite.sum().item()) < 200:
            return None
        z = z[finite]
        p = p[finite]
        y = y[finite]
        if float(torch.sqrt(torch.mean(p * p)).item()) <= 1.0e-14:
            return None
        wz = w * z
        design = torch.stack([p, p * torch.cos(wz), p * torch.sin(wz)], dim=1)
        sol = torch.linalg.lstsq(design, y).solution
        if sol.numel() < 3 or not torch.isfinite(sol).all():
            return None
        base, cos_coef, sin_coef = [float(v) for v in sol[:3]]
        amp = math.hypot(cos_coef, sin_coef)
        if (not math.isfinite(base)) or (not math.isfinite(amp)) or max(abs(base), amp) <= 1.0e-14:
            return None

        if abs(base) <= 1.0e-14:
            return None
        phase = math.atan2(sin_coef / base, -cos_coef / base)
        phase = _wrap_phase(float(phase))
        pred = float(base) * p * (1.0 - torch.cos(wz + float(phase)))
        resid = y - pred
        centered = y - torch.mean(y)
        denom = torch.sqrt(torch.mean(centered * centered)).clamp_min(1.0e-30)
        rel = torch.sqrt(torch.mean(resid * resid)) / denom
        if not torch.isfinite(rel):
            return None
        return {
            "omega": float(w),
            "phase": float(phase),
            "scale_init": float(base),
            "screen_rel_rms": float(rel.item()),
            "linear_base": float(base),
            "linear_cos": float(cos_coef),
            "linear_sin": float(sin_coef),
            "linear_amp": float(amp),
        }
    except Exception:
        return None


def _make_phase_context_one_minus_cos_candidate(
    ctx: StageBContext,
    target: AtomNode,
    hint: Any,
    context_ast: Node,
    target_dim,
    hit: Dict[str, Any],
) -> Optional[Candidate]:
    try:
        carrier_ast = clone_ast(getattr(hint, "carrier_ast"))
        context_ast = clone_ast(context_ast)
    except Exception:
        return None
    if not _phase_context_feature_units_ok(ctx, context_ast, target_dim):
        return None

    carrier_label = str(getattr(hint, "carrier_label", "z"))
    try:
        context_label = ast_to_human_readable(context_ast)
    except Exception:
        context_label = "context"
    omega = float(hit["omega"])
    phase = float(hit["phase"])
    scale_init = float(hit["scale_init"])
    base_tag = str(getattr(target, "tag", None) or "leaf")
    arg_tag = f"{base_tag}_phase_ctx_arg"
    scale_tag = f"{base_tag}_phase_ctx_scale"
    var_idxs = tuple(sorted(int(v) for v in _collect_var_idxs_from_node(carrier_ast)))
    if not var_idxs:
        return None
    arg_atom = AtomNode(
        kind="poly",
        var_idxs=var_idxs,
        kwargs={"degree": 1, "min_total": 0},
        tag=arg_tag,
        inputs=(carrier_ast,),
    )
    one_minus_cos = AddNode(ConstNode(1.0), MulNode(ConstNode(-1.0), CosNode(arg_atom)))
    try:
        scale_node = _make_unit_aware_scalar_atom(
            _phase_context_scale_dim(ctx),
            getattr(ctx, "units_spec", None),
            base_tag=scale_tag,
            init=scale_init,
            strict=bool(getattr(ctx, "enforce_units", False)),
        )
    except Exception:
        return None
    new_subtree = MulNode(MulNode(scale_node, context_ast), one_minus_cos)
    root_new = replace_atom_in_ast(ctx.state.root, target, new_subtree)
    if root_new is None:
        return None

    scale_node_tag = getattr(scale_node, "tag", scale_tag)

    def _init(root_new_inner: Node, model_new: nn.Module, *, _omega=omega, _phase=phase, _scale=scale_init):
        atom_to_leaf = build_atom_to_leaf_map(root_new_inner, model_new)
        for atom in _collect_all_atoms(root_new_inner):
            if not isinstance(atom, AtomNode):
                continue
            leaf = atom_to_leaf.get(id(atom), None)
            if leaf is None:
                continue
            tag = getattr(atom, "tag", None)
            if tag == arg_tag:
                try:
                    _poly_zero_and_set(leaf, {(0,): float(_phase), (1,): float(_omega)})
                except Exception:
                    pass
            elif tag == scale_node_tag:
                try:
                    _set_constant_leaf_value(leaf, float(_scale))
                except Exception:
                    pass

    _init._after_analytic_init = True
    label = "phase_context_one_minus_cos"
    meta = {
        "structural": True,
        "pattern": "phase_context_trig_closure",
        "pattern_family": "phase_context_trig_closure",
        "phase_context_trig_closure": True,
        "omega": float(omega),
        "phase": float(phase),
        "carrier_label": carrier_label,
        "context_label": context_label,
        "screen_rel_rms": float(hit.get("screen_rel_rms", float("inf"))),
        "signature": (
            int(atom_content_hash(target)),
            stable_int_hash("phase_context_trig_closure"),
            stable_int_hash(carrier_label),
            stable_int_hash(context_label),
            stable_int_hash("one_minus_cos"),
            int(round(float(omega) * 1.0e6)),
            int(round(float(phase) * 1.0e6)),
        ),
        "log": (
            f"[Stage B PhaseContext] Trying direct one_minus_cos closure "
            f"context={context_label}, z={carrier_label}, omega≈{omega:.6g}, "
            f"phase≈{phase:.4g}, rel={float(hit.get('screen_rel_rms', float('inf'))):.2e}"
        ),
    }
    return Candidate(label=label, root=root_new, init_fn=_init, meta=meta)


class RulePhaseContextTrigClosure(StageBRule):
    """Visible contextual trig closures from Stage-0 phase-context hints."""

    name = "phase_context_trig_closure"
    exhaustive = True

    def iter_targets(self, ctx: StageBContext):
        if not getattr(ctx, "phase_context_hints", None):
            return []
        atoms = [
            atom
            for atom in _collect_all_atoms(ctx.state.root)
            if isinstance(atom, AtomNode)
            and str(getattr(atom, "kind", "")).lower() == "nn"
        ]
        if len(atoms) != 1:
            return []
        target = atoms[0]
        if ctx.state.root is not target:
            return []
        return [target]

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        if not isinstance(target, AtomNode):
            return []
        hints = list(getattr(ctx, "phase_context_hints", []) or [])
        if not hints:
            return []
        target_vars = set(int(v) for v in getattr(target, "var_idxs", ()) or ())
        if not target_vars:
            return []

        try:
            max_points = int(getattr(ctx.lm_hp, "stageB_last_hard_trig_power_max_points", 5000) or 5000)
        except Exception:
            max_points = 5000
        data = _gather_phase_direct_xy(ctx, max_points=max_points)
        if data is None:
            return []
        X, y = data
        y = y.detach().to(dtype=torch.float64).reshape(-1)
        finite_y = torch.isfinite(y)
        if int(finite_y.sum().item()) < 200:
            return []

        try:
            max_rel = float(getattr(ctx.lm_hp, "stageB_last_hard_trig_power_screen_rel_rms", 2.0e-2))
        except Exception:
            max_rel = 2.0e-2
        max_rel = _stageB_noisy_rel_rms_threshold(ctx, max_rel, y_values=y[finite_y])
        target_dim = _phase_direct_target_dim(ctx, target)
        if bool(getattr(ctx, "enforce_units", False)) and target_dim is None:
            return []

        cands: List[Candidate] = []
        for hint in hints[:8]:
            if str(getattr(hint, "coupling_mode", "")) not in {"prefactor", "pure", ""}:
                continue
            if str(getattr(hint, "waveform_family", "")) not in {"one_minus_cos", "contextual_fourier"}:
                continue
            hint_vars = _phase_context_hint_vars(hint)
            if not hint_vars or not hint_vars.issubset(target_vars):
                continue
            if not _phase_hint_carrier_is_dimless(ctx, hint):
                if bool(getattr(ctx, "verbose", False)):
                    ctx.log(
                        "[Stage B PhaseContext] Skipping direct closure: "
                        f"carrier {getattr(hint, 'carrier_label', 'z')} is not dimensionless."
                    )
                continue
            try:
                z = eval_input_expr(getattr(hint, "carrier_ast"), X).reshape(-1).detach().to(dtype=torch.float64)
            except Exception:
                continue
            for context_ast in tuple(getattr(hint, "context_asts", ()) or ())[:6]:
                if context_ast is None:
                    continue
                if not _phase_context_feature_units_ok(ctx, context_ast, target_dim):
                    continue
                try:
                    p = eval_input_expr(context_ast, X).reshape(-1).detach().to(dtype=torch.float64)
                except Exception:
                    continue
                finite = finite_y & torch.isfinite(z) & torch.isfinite(p)
                if int(finite.sum().item()) < 200:
                    continue
                zf = z[finite]
                pf = p[finite]
                yf = y[finite]
                if float(torch.std(yf).item()) <= 1.0e-14:
                    continue
                for omega in _phase_context_omegas(hint)[:6]:
                    hit = _fit_context_one_minus_cos(zf, pf, yf, float(omega))
                    if hit is None:
                        continue
                    if float(hit.get("screen_rel_rms", float("inf"))) > max_rel:
                        continue
                    cand = _make_phase_context_one_minus_cos_candidate(
                        ctx,
                        target,
                        hint,
                        context_ast,
                        target_dim,
                        hit,
                    )
                    if cand is not None:
                        cands.append(cand)

        cands.sort(key=lambda c: float((c.meta or {}).get("screen_rel_rms", float("inf"))))
        return cands[:8]


def _last_hard_trig_square_screen(
    ctx: StageBContext,
    target: AtomNode,
    *,
    min_points: int = 200,
    max_candidates: int = 4,
) -> List[Dict[str, Any]]:
    tag = getattr(target, "tag", None)
    teacher = ctx.state.reuse.get(tag, None) if isinstance(getattr(ctx.state, "reuse", None), dict) else None
    if teacher is None:
        return []

    try:
        max_points = int(getattr(ctx.lm_hp, "stageB_last_hard_trig_power_max_points", 5000) or 5000)
    except Exception:
        max_points = 5000
    data = _gather_atom_teacher_data(
        train_loader=ctx.train_loader_probe,
        atom=target,
        teacher=teacher,
        device=ctx.device,
        dtype=ctx.dtype,
        max_points=max_points,
    )
    if data is None:
        return []
    X, F = data
    if X.ndim != 2 or X.shape[1] != 1:
        return []

    z = X[:, 0].detach().to(dtype=torch.float64).reshape(-1)
    f = F.detach().to(dtype=torch.float64).reshape(-1)
    finite = torch.isfinite(z) & torch.isfinite(f)
    domain_frac = float(finite.to(dtype=torch.float64).mean().item()) if int(f.numel()) else 0.0
    try:
        min_domain = float(getattr(ctx.lm_hp, "macro_domain_ok_frac", 0.98))
    except Exception:
        min_domain = 0.98
    if domain_frac < min_domain:
        return []

    z = z[finite]
    f = f[finite]
    if int(z.numel()) < int(min_points):
        return []
    centered = f - torch.mean(f)
    if float(torch.sqrt(torch.mean(centered * centered)).item()) <= 1.0e-14:
        return []

    omega_grid = _last_hard_trig_square_omega_grid(ctx, target)
    if not omega_grid:
        return []

    try:
        max_rel = float(getattr(ctx.lm_hp, "stageB_last_hard_trig_power_screen_rel_rms", 2.0e-2))
    except Exception:
        max_rel = 2.0e-2
    max_rel = _stageB_noisy_rel_rms_threshold(ctx, max_rel, y_values=f)
    try:
        max_square_offset_rel = float(getattr(ctx.lm_hp, "stageB_last_hard_trig_power_max_offset_rel", 0.15))
    except Exception:
        max_square_offset_rel = 0.15

    rows: List[Dict[str, Any]] = []
    for harmonic_omega in omega_grid:
        fit = _solve_affine_trig_1d(z, f, harmonic_omega)
        if fit is None:
            continue
        rel = float(fit["rel_rms"])
        if rel > max_rel:
            continue

        A = float(fit["A"])
        B = float(fit["B"])
        C = float(fit["offset"])
        amp = float(fit["amp"])
        if (not math.isfinite(C)) or abs(C) <= 1.0e-14:
            continue
        square_offset_rel = abs(abs(C) - amp) / max(amp, 1.0e-30)
        if square_offset_rel > max_square_offset_rel:
            continue

        # For scale*S=2C:
        # S*sin(arg)^2 = C + C*sin(2arg_phase)*sin(Wz) - C*cos(2arg_phase)*cos(Wz)
        # S*cos(arg)^2 = C - C*sin(2arg_phase)*sin(Wz) + C*cos(2arg_phase)*cos(Wz)
        sign_c = 1.0 if C >= 0.0 else -1.0
        phase_sin = _snap_phase_for_even_power(
            0.5 * math.atan2(sign_c * A, sign_c * (-B))
        )
        phase_cos = _snap_phase_for_even_power(
            0.5 * math.atan2(sign_c * (-A), sign_c * B)
        )
        use_cos = abs(phase_cos) + 1.0e-12 < abs(phase_sin)
        trig_kind = "cos" if use_cos else "sin"
        phase = phase_cos if use_cos else phase_sin
        arg_omega = 0.5 * float(harmonic_omega)
        try:
            arg_omega = float(snap_omega(arg_omega))
        except Exception:
            arg_omega = 0.5 * float(harmonic_omega)
        scale_init = 2.0 * C
        if not (math.isfinite(arg_omega) and math.isfinite(scale_init)):
            continue

        rows.append(
            {
                "omega": float(arg_omega),
                "harmonic_omega": float(harmonic_omega),
                "phase": float(phase),
                "trig_kind": trig_kind,
                "scale_init": float(scale_init),
                "rel_rms": rel,
                "square_offset_rel": float(square_offset_rel),
                "domain_frac": domain_frac,
                "screen_score": rel + 0.05 * square_offset_rel,
            }
        )

    rows.sort(key=lambda r: float(r["screen_score"]))
    return rows[: max(0, int(max_candidates))]


def _make_last_hard_trig_square_candidate(
    ctx: StageBContext,
    target: AtomNode,
    target_dim: Tuple[Any, ...],
    hit: Dict[str, Any],
) -> Optional[Candidate]:
    base_tag = str(getattr(target, "tag", None) or "leaf")
    omega = float(hit["omega"])
    harmonic_omega = float(hit["harmonic_omega"])
    phase = float(hit["phase"])
    trig_kind = str(hit["trig_kind"])
    scale_init = float(hit["scale_init"])

    arg_tag = f"{base_tag}_lhtrigsq_arg"
    scale_tag = f"{base_tag}_lhtrigsq_scale"
    arg_atom = AtomNode(
        kind="poly",
        var_idxs=tuple(int(j) for j in target.var_idxs),
        kwargs={"degree": 1, "min_total": 0},
        tag=arg_tag,
        inputs=clone_inputs(target),
    )
    trig_node = CosNode(arg_atom) if trig_kind == "cos" else SinNode(arg_atom)
    core = PowNode(trig_node, 2.0)
    scale_node = _make_unit_aware_scalar_atom(
        target_dim,
        getattr(ctx, "units_spec", None),
        base_tag=scale_tag,
        init=scale_init,
    )
    new_subtree = MulNode(scale_node, core)
    root_new = replace_atom_in_ast(ctx.state.root, target, new_subtree)
    if root_new is None:
        return None

    scale_node_tag = getattr(scale_node, "tag", scale_tag)

    def _init(root_new_inner: Node, model_new: nn.Module, *, _phase=phase, _omega=omega, _scale=scale_init):
        atom_to_leaf = build_atom_to_leaf_map(root_new_inner, model_new)
        for atom in _collect_all_atoms(root_new_inner):
            if not isinstance(atom, AtomNode):
                continue
            leaf = atom_to_leaf.get(id(atom), None)
            if leaf is None:
                continue
            if getattr(atom, "tag", None) == arg_tag:
                try:
                    _poly_zero_and_set(leaf, {(0,): float(_phase), (1,): float(_omega)})
                except Exception:
                    pass
            elif getattr(atom, "tag", None) == scale_node_tag:
                try:
                    _set_constant_leaf_value(leaf, float(_scale))
                except Exception:
                    pass

    _init._after_analytic_init = True

    label = f"last_trig_square_{trig_kind}"
    meta = {
        "structural": True,
        "pattern": "last_hard_trig_square",
        "pattern_family": "last_hard_trig_square",
        "last_hard_trig_square": True,
        "trig_kind": trig_kind,
        "trig_power": 2,
        "omega": float(omega),
        "harmonic_omega": float(harmonic_omega),
        "phase": float(phase),
        "scale_init": float(scale_init),
        "screen_rel_rms": float(hit.get("rel_rms", float("inf"))),
        "screen_square_offset_rel": float(hit.get("square_offset_rel", float("inf"))),
        "signature": (
            int(atom_content_hash(target)),
            stable_int_hash("last_hard_trig_square"),
            stable_int_hash(trig_kind),
            int(round(float(omega) * 1.0e6)),
            int(round(float(phase) * 1.0e6)),
        ),
        "log": (
            f"[Stage B]  Trying last-hard trig-square {trig_kind}(ωz+φ)^2 "
            f"on NN vars={target.var_idxs}: ω≈{omega:.4g}, φ≈{phase:.4g}, "
            f"scale≈{scale_init:.4g}, rel={float(hit.get('rel_rms', float('inf'))):.2e}"
        ),
    }
    return Candidate(label=label, root=root_new, init_fn=_init, meta=meta)


class RuleLastHardTrigSquare1D(StageBRule):
    """Final 1D rescue for pure harmonic squares such as sin(2*pi*z)^2."""

    name = "last_hard_trig_square"
    exhaustive = True

    def iter_targets(self, ctx: StageBContext):
        atoms = [
            atom
            for atom in _collect_all_atoms(ctx.state.root)
            if isinstance(atom, AtomNode)
            and str(getattr(atom, "kind", "")).lower() == "nn"
        ]
        if len(atoms) != 1:
            return []
        target = atoms[0]
        gate = _last_hard_atom_context(ctx, target, max_arity=1)
        if gate is None:
            return []
        _target_dim, x_dims = gate
        if _is_dimless is not None and x_dims and not _is_dimless(x_dims[0]):
            return []
        return [target]

    def _propose_reciprocal_coordinate_alias(self, ctx: StageBContext, target: AtomNode) -> List[Candidate]:
        if bool(getattr(ctx, "_stageB_coord_alias_active", False)):
            return []
        if not isinstance(target, AtomNode) or effective_arity(target) != 1:
            return []
        tag = getattr(target, "tag", None)
        if tag is None:
            return []
        z_expr = compound_input_expr(target)
        if z_expr is None:
            return []

        reuse_override = _wrap_reuse_for_reciprocal_coordinate(ctx.state.reuse, tag)
        if reuse_override is None:
            return []

        reuses_override = None
        state_reuses = getattr(ctx.state, "reuses", None)
        if isinstance(state_reuses, (list, tuple)):
            wrapped_reuses = []
            ok_all = True
            for reuse_i in state_reuses:
                wrapped = _wrap_reuse_for_reciprocal_coordinate(reuse_i, tag)
                if wrapped is None:
                    ok_all = False
                    break
                wrapped_reuses.append(wrapped)
            if ok_all:
                reuses_override = wrapped_reuses

        alias_target = AtomNode(
            kind="nn",
            var_idxs=tuple(int(j) for j in target.var_idxs),
            kwargs=copy.deepcopy(getattr(target, "kwargs", {}) or {}),
            tag=tag,
            inputs=(PowNode(z_expr, -1.0),),
        )
        alias_root = replace_atom_in_ast(ctx.state.root, target, alias_target)
        if alias_root is None:
            return []

        alias_state = copy.copy(ctx.state)
        alias_state.root = alias_root
        alias_state.reuse = reuse_override
        if reuses_override is not None:
            alias_state.reuses = reuses_override

        alias_ctx = copy.copy(ctx)
        alias_ctx.state = alias_state
        alias_ctx._stageB_coord_alias_active = True
        for cache_name in ("_cache", "_dim_cache"):
            if hasattr(alias_ctx, cache_name):
                try:
                    setattr(alias_ctx, cache_name, {})
                except Exception:
                    pass
        if hasattr(alias_ctx, "_dim_cache_root_id"):
            try:
                alias_ctx._dim_cache_root_id = None
            except Exception:
                pass

        try:
            alias_cands = self.propose(alias_ctx, alias_target) or []
        except Exception as exc:
            if bool(getattr(ctx, "verbose", False)):
                ctx.log(f"[Stage B]  reciprocal-coordinate trig-square proposal failed: {exc}")
            return []

        out: List[Candidate] = []
        for cand in alias_cands:
            if cand is None:
                continue
            out.append(
                _mark_reciprocal_coordinate_candidate(
                    cand,
                    reuse_override=reuse_override,
                    reuses_override=reuses_override,
                )
            )
        return out

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        if not isinstance(target, AtomNode):
            return []
        if _ctx_pattern_disabled(ctx, self.name) or _ctx_pattern_disabled(ctx, "trig"):
            return []
        gate = _last_hard_atom_context(ctx, target, max_arity=1)
        if gate is None:
            return []
        target_dim, x_dims = gate
        if _is_dimless is not None and x_dims and not _is_dimless(x_dims[0]):
            return []

        try:
            policy = macro_arg_wrapper_policy(ctx, ctx.lm_hp, target)
            if not bool(getattr(policy, "trig", True)):
                return []
        except Exception:
            pass

        hits = _last_hard_trig_square_screen(ctx, target)
        cands: List[Candidate] = []
        for hit in hits:
            cand = _make_last_hard_trig_square_candidate(ctx, target, target_dim, hit)
            if cand is not None:
                cands.append(cand)

        cands = _merge_reciprocal_aliases_pairwise(
            cands,
            self._propose_reciprocal_coordinate_alias(ctx, target),
        )

        if cands:
            ctx.log(
                f"[Stage B] RuleLastHardTrigSquare1D proposing {len(cands)} candidate(s) "
                f"for final NN vars={target.var_idxs}: {[c.label for c in cands]}"
            )
        return cands


class RuleLastHardTrigPower1D(StageBRule):
    """Final 1D rescue for singular trig envelopes such as sin(x/2)^-4."""

    name = "last_hard_trig_power"
    exhaustive = True

    def iter_targets(self, ctx: StageBContext):
        atoms = [
            atom
            for atom in _collect_all_atoms(ctx.state.root)
            if isinstance(atom, AtomNode)
            and str(getattr(atom, "kind", "")).lower() == "nn"
        ]
        if len(atoms) != 1:
            return []
        target = atoms[0]
        gate = _last_hard_atom_context(ctx, target, max_arity=1)
        if gate is None:
            return []
        _target_dim, x_dims = gate
        if _is_dimless is not None and x_dims and not _is_dimless(x_dims[0]):
            return []
        return [target]

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        if not isinstance(target, AtomNode):
            return []
        if _ctx_pattern_disabled(ctx, self.name) or _ctx_pattern_disabled(ctx, "trig"):
            return []
        gate = _last_hard_atom_context(ctx, target, max_arity=1)
        if gate is None:
            return []
        target_dim, x_dims = gate
        if _is_dimless is not None and x_dims and not _is_dimless(x_dims[0]):
            return []

        try:
            policy = macro_arg_wrapper_policy(ctx, ctx.lm_hp, target)
            if not bool(getattr(policy, "trig", True)):
                return []
        except Exception:
            pass

        hits = _last_hard_trig_power_screen(ctx, target)
        cands: List[Candidate] = []
        for hit in hits:
            cand = _make_last_hard_trig_power_candidate(ctx, target, target_dim, hit)
            if cand is not None:
                cands.append(cand)

        if cands:
            ctx.log(
                f"[Stage B] RuleLastHardTrigPower1D proposing {len(cands)} candidate(s) "
                f"for final NN vars={target.var_idxs}: {[c.label for c in cands]}"
            )
        return cands
