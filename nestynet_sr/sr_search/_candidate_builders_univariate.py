# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

"""Univariate analytic-family candidate builders and AST rewrites."""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from nestynet_sr.sr_search.gate_telemetry import record_gate

from nestynet_sr.sr_core.atoms import (
    ExpPolyLeaf,
    PlanckFullLeaf,
    PlanckLeaf,
    PolyLeaf,
    PowerLeaf,
    RExpPolyLeaf,
    RRationalPolyLeaf,
    _enumerate_exponents,
    _eval_monomials,
)
from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    CosNode,
    ExpNode,
    MulNode,
    Node,
    PowNode,
    SinNode,
    ast_to_human_readable,
    clone_ast,
    clone_inputs,
    compound_input_expr,
    effective_arity,
    get_input_exprs,
    has_nontrivial_input,
    is_trivial_input,
    trivial_input_position,
    _collect_var_idxs_from_node,
)
from nestynet_sr.sr_core.constants import (
    build_scalar_atom_from_variant as _build_scalar_atom_from_variant,
    scalar_constant_variants as _scalar_constant_variants,
)

from ._candidate_builders_common import (
    _atom_inputs_match,
    _find_matching_core,
    _replace_node,
    _single_power_coordinate_inputs,
)
from ._candidate_builders_multivariate import _make_power_exp_poly_rewrite
from .features import ScaleSpec, TrigAxisSpec
from .fitting_utils import (
    PLANCK_STRUCTURAL_POWERS,
    _fit_planck_tail,
    _fit_planck_tail_fixed_power,
    _fit_power_coeffs_1d,
    _fit_rational_coeffs_1d,
    _gather_teacher_data_1d,
)

def _build_power_exp_1d_candidate(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    degree: int = 2,
    min_points: int = 400,
    eps: float = 1e-8,
    max_abs_exponent: float = 20.0,
) -> Tuple[Optional[Node], Optional[callable]]:
    """
    1D analogue of the multi-D exp-branch rewrite:

        f(x_j) ≈ A * x_j^p * exp( poly(x_j) )

    We:
      - fit a power law A*x^p to the Stage-A leaf,
      - examine g = f / (A*x^p), and only proceed if log g is tame,
      - propose power(x_j)^p * exp_poly(x_j) with conservative init.
    """
    # Import locally to avoid circular dependency
    from .stageB import _collect_all_atoms

    if target.kind.lower() != "nn" or effective_arity(target) != 1:
        return None, None

    input_expr = compound_input_expr(target)
    axis = None
    if input_expr is not None and is_trivial_input(input_expr):
        axis = int(input_expr.var_idxs[0])
        input_expr_for_data = None
    else:
        input_expr_for_data = input_expr

    tag = target.tag
    if tag is None or tag not in reuse:
        return None, None
    teacher = reuse[tag]

    data = _gather_teacher_data_1d(
        train_loader,
        teacher,
        device,
        dtype,
        axis=axis,
        input_expr=input_expr_for_data,
        max_points=5000,
    )
    if data is None:
        return None, None
    X, F = data  # X:[N,1], F:[N]
    X = X.view(-1).to(dtype=torch.float64)
    F = F.view(-1).to(dtype=torch.float64)

    coeffs = _fit_power_coeffs_1d(X, F, min_points=min_points, dtype=torch.float64, eps=eps)
    if coeffs is None:
        return None, None
    A_est, p_est = coeffs

    mask = (X > eps) & (F > eps)
    if mask.sum().item() < min_points:
        return None, None

    Xm = X[mask]
    Fm = F[mask]
    with torch.no_grad():
        g = Fm / (A_est * torch.pow(Xm, p_est))
        log_g = torch.log(g)
        mu = float(log_g.mean().item())
        log_g_c = log_g - mu
        max_abs_log = float(log_g_c.abs().max().item())
        if (not math.isfinite(max_abs_log)) or max_abs_log > max_abs_exponent:
            return None, None

    # Build AST: power(xj)^p_est * exp_poly(xj; degree)
    root_cand = _make_power_exp_poly_rewrite(
        root=root,
        target=target,
        pivot_axis=axis,
        exponent=p_est,
        degree=degree,
        pivot_input_expr=input_expr_for_data,
    )

    target_inputs = get_input_exprs(target)

    def _custom_init(root_inner: Node, model_inner: torch.nn.Module):
        atoms = _collect_all_atoms(root_inner)
        leaves = list(model_inner.leaf)
        power_core = _find_matching_core(
            atoms,
            leaves,
            core_types=PowerLeaf,
            expected_kind="power",
            expected_inputs=target_inputs,
        )
        exp_core = _find_matching_core(
            atoms,
            leaves,
            core_types=(ExpPolyLeaf, RExpPolyLeaf),
            expected_kind="exp_poly",
            expected_inputs=target_inputs,
        )

        if power_core is None or exp_core is None:
            return

        with torch.no_grad():
            # Power: A_est * x^p_est
            if hasattr(power_core, "exponent"):
                power_core.exponent.copy_(
                    torch.as_tensor(
                        p_est,
                        dtype=power_core.exponent.dtype,
                        device=power_core.exponent.device,
                    )
                )
            if hasattr(power_core, "amp"):
                power_core.amp.copy_(
                    torch.as_tensor(
                        A_est,
                        dtype=power_core.amp.dtype,
                        device=power_core.amp.device,
                    )
                )

            # ExpPoly: start near zero exponent (flat), small random deviations
            exp_core.coeffs.zero_()
            exp_core.coeffs.normal_(mean=0.0, std=0.01)

        print(
            f"[Stage B custom_init 1D] Power*exp-poly init on "
            f"{ast_to_human_readable(input_expr) if input_expr is not None else f'axis {axis}'}: "
            f"A={A_est:.3e}, p={p_est:.3f}"
        )

    return root_cand, _custom_init


def _planck_power_label(p: float) -> str:
    if abs(float(p) - round(float(p))) < 1e-12:
        return str(int(round(float(p))))
    return f"{float(p):g}".replace("-", "m").replace(".", "p")


def _build_planck_1d_candidates(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    min_points: int = 400,
    eps: float = 1e-8,
    tail_fraction: float = 0.5,
    max_abs_p: float = 10.0,
    rel_rms_threshold: float = 5e-2,
) -> List[Tuple[str, Node, callable, Dict[str, Any]]]:
    """
    1D Planck / Bose–Einstein rewrite:

        f(x_j) ≈ A * x_j^p / (exp(a x_j + b) - 1)

    We fit on the high‑x tail in log‑space:
        log f ≈ β0 + p log x - a x

    The reduced Planck leaf treats ``p`` as a structural dictionary choice,
    not an LM-fitted parameter.  Emit the small fixed dictionary as separate
    two-parameter candidates; the tail fit only initialises and ranks them.
    """
    from .stageB import _collect_all_atoms

    if target.kind.lower() != "nn" or effective_arity(target) != 1:
        return []

    # Check for compound atom
    input_expr = compound_input_expr(target)
    axis = int(target.var_idxs[0]) if input_expr is None else None

    tag = target.tag
    if tag is None or tag not in reuse:
        return []
    teacher = reuse[tag]

    # Gather data (unified for both compound and univariate)
    data = _gather_teacher_data_1d(
        train_loader, teacher, device, dtype,
        axis=axis, input_expr=input_expr, max_points=5000
    )
    if data is None:
        return []

    X, F = data

    out: List[Tuple[str, Node, callable, Dict[str, Any]]] = []
    target_var_idxs = tuple(int(i) for i in target.var_idxs)
    target_inputs = get_input_exprs(target)

    for p in PLANCK_STRUCTURAL_POWERS:
        fit_result = _fit_planck_tail_fixed_power(
            X,
            F,
            p_fixed=float(p),
            min_points=min_points,
            eps=eps,
            tail_fraction=tail_fraction,
            rel_rms_threshold=float("inf"),
        )
        if fit_result is None:
            continue

        p_est, a_est, b0, rms_rel = fit_result
        if abs(float(p_est)) > float(max_abs_p):
            continue

        planck_atom = AtomNode(
            kind="planck",
            var_idxs=target.var_idxs,
            kwargs={"p": float(p_est)},
            tag=None,
            inputs=clone_inputs(target),
        )
        cand_root = _replace_node(root, target, planck_atom)
        if cand_root is None:
            continue

        b0_clamped = max(-20.0, min(20.0, float(b0)))
        a_est_clamped = max(1e-4, min(60.0, float(a_est)))
        p_est_clamped = max(-float(max_abs_p), min(float(max_abs_p), float(p_est)))

        def _custom_init(
            root_inner: Node,
            model_inner: torch.nn.Module,
            _p: float = p_est_clamped,
            _a: float = a_est_clamped,
            _b0: float = b0_clamped,
            _rms: float = float(rms_rel),
        ):
            atoms = _collect_all_atoms(root_inner)
            leaves = list(model_inner.leaf)
            core_planck = _find_matching_core(
                atoms,
                leaves,
                core_types=PlanckLeaf,
                expected_kind="planck",
                expected_inputs=target_inputs,
            )

            if core_planck is None:
                return

            with torch.no_grad():
                core_planck.p.copy_(
                    torch.as_tensor(
                        _p,
                        dtype=core_planck.p.dtype,
                        device=core_planck.p.device,
                    )
                )
                core_planck.log_a.copy_(
                    torch.log(
                        torch.as_tensor(
                            _a,
                            dtype=core_planck.log_a.dtype,
                            device=core_planck.log_a.device,
                        )
                    )
                )
                core_planck.log_amp.copy_(
                    torch.as_tensor(
                        _b0,
                        dtype=core_planck.log_amp.dtype,
                        device=core_planck.log_amp.device,
                    )
                )

            var_desc = f"axis={axis}" if axis is not None else f"compound vars={target_var_idxs}"
            print(
                f"[Stage B custom_init Planck] {var_desc}, "
                f"p={_p:.3g}, a≈{_a:.3g}, logA≈{_b0:.3g}, tail_rms≈{_rms:.3g}"
            )

        label = f"planck_p{_planck_power_label(float(p_est))}"
        out.append(
            (
                label,
                cand_root,
                _custom_init,
                {
                    "pattern_family": "planck",
                    "min_free_params": 2,
                    "planck_power": float(p_est),
                    "planck_tail_rms_rel": float(rms_rel),
                    "log": (
                        f"[Stage B]  Trying reduced 1D Planck rewrite on NN leaf vars {target.var_idxs} "
                        f"(p={float(p_est):.3g}, tail_rms≈{float(rms_rel):.3g})"
                    ),
                },
            )
        )

    return out


def _build_planck_1d_candidate(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    min_points: int = 400,
    eps: float = 1e-8,
    tail_fraction: float = 0.5,
    max_abs_p: float = 10.0,
    rel_rms_threshold: float = 5e-2,
) -> Tuple[Optional[Node], Optional[callable]]:
    candidates = _build_planck_1d_candidates(
        root=root,
        target=target,
        reuse=reuse,
        train_loader=train_loader,
        device=device,
        dtype=dtype,
        min_points=min_points,
        eps=eps,
        tail_fraction=tail_fraction,
        max_abs_p=max_abs_p,
        rel_rms_threshold=rel_rms_threshold,
    )
    if not candidates:
        return None, None
    _label, cand_root, init_fn, _meta = min(
        candidates,
        key=lambda item: float((item[3] or {}).get("planck_tail_rms_rel", float("inf"))),
    )
    return cand_root, init_fn


def _build_planck_full_1d_candidate(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    min_points: int = 400,
    eps: float = 1e-8,
    tail_fraction: float = 0.5,
    max_abs_p: float = 10.0,
    rel_rms_threshold: float = 5e-2,
) -> Tuple[Optional[Node], Optional[callable]]:
    """Legacy flexible Planck rewrite with trainable p and denominator shift."""
    from .stageB import _collect_all_atoms

    if target.kind.lower() != "nn" or effective_arity(target) != 1:
        return None, None

    input_expr = compound_input_expr(target)
    axis = int(target.var_idxs[0]) if input_expr is None else None

    tag = target.tag
    if tag is None or tag not in reuse:
        return None, None
    teacher = reuse[tag]

    data = _gather_teacher_data_1d(
        train_loader, teacher, device, dtype,
        axis=axis, input_expr=input_expr, max_points=5000
    )
    if data is None:
        return None, None

    X, F = data
    fit_result = _fit_planck_tail(
        X, F,
        min_points=min_points,
        eps=eps,
        tail_fraction=tail_fraction,
        max_abs_p=max_abs_p,
        rel_rms_threshold=rel_rms_threshold,
    )
    if fit_result is None:
        return None, None

    p_est, a_est, b0, _rms_rel = fit_result
    planck_atom = AtomNode(
        kind="planck_full",
        var_idxs=target.var_idxs,
        kwargs={},
        tag=None,
        inputs=clone_inputs(target),
    )
    cand_root = _replace_node(root, target, planck_atom)

    b0_clamped = max(-20.0, min(20.0, b0))
    a_est_clamped = max(1e-4, min(60.0, a_est))
    p_est_clamped = max(-max_abs_p, min(max_abs_p, p_est))
    target_var_idxs = tuple(int(i) for i in target.var_idxs)
    target_inputs = get_input_exprs(target)

    def _custom_init(root_inner: Node, model_inner: torch.nn.Module):
        atoms = _collect_all_atoms(root_inner)
        leaves = list(model_inner.leaf)
        core_planck = _find_matching_core(
            atoms,
            leaves,
            core_types=PlanckFullLeaf,
            expected_kind="planck_full",
            expected_inputs=target_inputs,
        )
        if core_planck is None:
            return
        with torch.no_grad():
            core_planck.p.copy_(
                torch.as_tensor(
                    p_est_clamped,
                    dtype=core_planck.p.dtype,
                    device=core_planck.p.device,
                )
            )
            core_planck.log_a.copy_(
                torch.log(
                    torch.as_tensor(
                        a_est_clamped,
                        dtype=core_planck.log_a.dtype,
                        device=core_planck.log_a.device,
                    )
                )
            )
            core_planck.log_amp.copy_(
                torch.as_tensor(
                    b0_clamped,
                    dtype=core_planck.log_amp.dtype,
                    device=core_planck.log_amp.device,
                )
            )
            core_planck.b.zero_()

        var_desc = f"axis={axis}" if axis is not None else f"compound vars={target_var_idxs}"
        print(
            f"[Stage B custom_init PlanckFull] {var_desc}, "
            f"p≈{p_est_clamped:.3g}, a≈{a_est_clamped:.3g}, logA≈{b0_clamped:.3g}"
        )

    return cand_root, _custom_init


def _build_expm1_1d_candidate(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    min_points: int = 400,
    rel_rms_threshold: float = 0.10,  # Log-space fit threshold; actual fit validates later
) -> Tuple[Optional[Node], Optional[callable]]:
    """
    1D Expm1 rewrite: f(x) ≈ A * (exp(a*x + b) - 1)

    Detection strategy:
    - For large x: f(x) ≈ A*exp(a*x), so log(f) ≈ log(A) + a*x
    - Fit log(|f|) vs x in high-|f| tail to get (a, logA)
    - Then fit full data to refine b offset
    """
    from .stageB import _collect_all_atoms

    if target.kind.lower() != "nn" or effective_arity(target) != 1:
        return None, None

    input_expr = compound_input_expr(target)
    axis = int(target.var_idxs[0]) if input_expr is None else None

    tag = target.tag
    if tag is None or tag not in reuse:
        return None, None
    teacher = reuse[tag]

    # Gather data (unified for both compound and univariate)
    data = _gather_teacher_data_1d(
        train_loader, teacher, device, dtype,
        axis=axis, input_expr=input_expr, max_points=5000
    )
    if data is None:
        return None, None

    X, F = data
    X = X.view(-1).to(dtype=torch.float64)
    F = F.view(-1).to(dtype=torch.float64)

    # Filter to reasonable values
    eps = 1e-8
    m = torch.isfinite(F) & (F.abs() > eps)
    if m.sum().item() < min_points:
        return None, None
    X, F = X[m], F[m]

    # Detect sign: expm1 can be positive (z>0) or negative (z<0)
    sign = 1.0 if (F > 0).sum() > (F < 0).sum() else -1.0
    F_signed = sign * F

    # Filter to positive values for log fit
    m_pos = F_signed > eps
    if m_pos.sum().item() < min_points:
        return None, None

    X_pos, F_pos = X[m_pos], F_signed[m_pos]

    # Fit in tail: log(f) ≈ log(A) + a*x
    order = torch.argsort(F_pos)
    k_tail = int(0.5 * len(order))  # top 50% by magnitude
    tail_mask = torch.zeros_like(F_pos, dtype=torch.bool)
    tail_mask[order[k_tail:]] = True

    if tail_mask.sum().item() < min_points:
        return None, None

    X_tail = X_pos[tail_mask]
    logF_tail = torch.log(F_pos[tail_mask])

    # Linear fit: log(f) = beta0 + beta1 * x
    Phi = torch.stack([torch.ones_like(X_tail), X_tail], dim=1)
    try:
        beta = torch.linalg.lstsq(Phi, logF_tail.unsqueeze(1)).solution.squeeze(1)
    except RuntimeError:
        return None, None

    log_A_est = float(beta[0])
    a_est = float(beta[1])

    if not (math.isfinite(a_est) and math.isfinite(log_A_est)):
        return None, None
    if a_est <= 0:  # Need positive a for exp growth
        return None, None

    # Validate fit quality in log-space (actual training validates properly later)
    logF_fit = (Phi @ beta).view(-1)
    resid = logF_tail - logF_fit
    rms_rel = float(torch.sqrt(torch.mean(resid**2)) / torch.std(logF_tail))
    record_gate(
        "expm1_1d", "log_rel_rms", rms_rel, rel_rms_threshold,
        accepted=not (rms_rel > rel_rms_threshold),
    )
    if rms_rel > rel_rms_threshold:
        return None, None

    # Build candidate
    expm1_atom = AtomNode(
        kind="expm1",
        var_idxs=target.var_idxs,
        kwargs={},
        tag=None,
        inputs=clone_inputs(target),
    )
    cand_root = _replace_node(root, target, expm1_atom)

    # Clamp estimates
    log_amp_est = log_A_est + math.log(abs(sign)) if sign != 1.0 else log_A_est
    log_amp_clamped = max(-20.0, min(20.0, log_amp_est))
    log_a_clamped = math.log(max(1e-4, min(60.0, a_est)))

    target_var_idxs = tuple(int(i) for i in target.var_idxs)
    target_inputs = get_input_exprs(target)

    def _custom_init(root_inner: Node, model_inner: torch.nn.Module):
        from nestynet_sr.sr_core.atoms import Expm1Leaf

        atoms = _collect_all_atoms(root_inner)
        leaves = list(model_inner.leaf)

        core = _find_matching_core(
            atoms,
            leaves,
            core_types=Expm1Leaf,
            expected_kind="expm1",
            expected_inputs=target_inputs,
        )
        if core is not None:
            with torch.no_grad():
                core.log_amp.copy_(
                    torch.tensor(
                        log_amp_clamped, dtype=core.log_amp.dtype, device=core.log_amp.device
                    )
                )
                core.log_a.copy_(
                    torch.tensor(
                        log_a_clamped, dtype=core.log_a.dtype, device=core.log_a.device
                    )
                )
                core.b.zero_()
            var_desc = f"axis={axis}" if axis is not None else f"compound vars={target_var_idxs}"
            print(f"[Stage B custom_init Expm1] {var_desc}, a≈{a_est:.3g}, logA≈{log_amp_clamped:.3g}")

    return cand_root, _custom_init


def _symexp_scale_and_u(F_pos: torch.Tensor) -> Tuple[float, torch.Tensor, int]:
    """Estimate the sech scale and acosh argument u = scale/(2f).

    Returns (scale0, u, scale_doublings). scale_doublings > 0 means the 0.99
    quantile estimate had to be inflated for u > 1 to hold on most points,
    i.e. the data never approach the cosh turnover: the template is then in
    its tail-degenerate regime where it can reproduce any positive f.
    """
    scale0 = float(torch.quantile(F_pos, 0.99).item()) * 2.0 * 1.05
    scale0 = max(scale0, 1e-8)
    scale_doublings = 0
    for _ in range(6):
        u = scale0 / (2.0 * F_pos)
        frac_valid = float((u > 1.0 + 1e-6).to(dtype=torch.float64).mean().item())
        if frac_valid >= 0.6:
            break
        scale0 *= 2.0
        scale_doublings += 1
    u = scale0 / (2.0 * F_pos)
    return scale0, u, scale_doublings


def _eval_ratpoly_1d(X: torch.Tensor, a_coeffs: torch.Tensor, b_coeffs: torch.Tensor) -> torch.Tensor:
    """Evaluate (a0 + a1*x + ...)/(b0 + b1*x + ...) with a guarded denominator."""
    num = torch.zeros_like(X)
    den = torch.zeros_like(X)
    for i in range(int(a_coeffs.numel())):
        num = num + a_coeffs[i] * X.pow(i)
    for i in range(int(b_coeffs.numel())):
        den = den + b_coeffs[i] * X.pow(i)
    sgn = torch.sign(den)
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    den = sgn * torch.clamp(den.abs(), min=1e-12)
    return num / den


def _symexp_gate_decision(
    rms: float,
    *,
    eff_threshold: float,
    null_rms: Optional[float],
    null_margin: float,
) -> bool:
    """Accept the symexp candidate only if it fits well enough (eff_threshold,
    tightened in the tail-degenerate regime) AND beats the simpler rational
    null by a real factor (null_margin) when that null is available."""
    if not math.isfinite(rms) or rms > eff_threshold:
        return False
    if null_rms is not None and math.isfinite(null_rms) and null_rms >= 0.0:
        return rms <= null_margin * null_rms
    return True


def _build_symexp_denom_1d_candidate(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    min_points: int = 250,
    rel_rms_threshold: float = 0.05,
    units_spec: Any = None,
    max_fixed_consts: int = 4,
    u_q01_max: float = 1.5,
    tail_rel_rms_threshold: float = 0.01,
    null_margin: float = 0.5,
) -> List[Tuple[Node, Optional[callable], str]]:
    """
    1D symmetric exponential denominator rewrite:

        f(z) ≈ scale / (exp(r(z)) + exp(-r(z)))  =  scale / (2 * cosh(r(z)))

    where r(z) is a simple 1D rational polynomial (deg_num=1, deg_den=1).

    This template can fit sech-like forms including:
    - 1 / (exp(z) + exp(-z))  ->  r(z) = z
    - 1 / (exp(1/z) + exp(-1/z))  ->  r(z) = 1/z = (a)/(c + dz)

    For compound atoms, also tries the reciprocal variant:
        f(z) ≈ scale / (exp(r(1/z)) + exp(-r(1/z)))
    which can fit forms like cosh(1/z) with a simple linear r.

    Detection strategy:
    1. Check if NN output is sign-definite (positive or negative)
    2. Estimate scale from upper quantile of |f|
    3. Compute u = scale / (2*|f|), then r = acosh(u) = log(u + sqrt(u^2 - 1))
    4. Fit r(z) as a rational polynomial and check quality
    5. For compound atoms, also try fitting r(1/z)

    Returns a list of (candidate_root, custom_init, label) tuples.
    When ``units_spec`` declares fixed dimensionless constants, emits
    extra variants with those constants as the outer ``scale`` leaf.
    Also supports compound atoms with input_expr.
    """
    from .fitting_utils import _fit_rational_coeffs_1d
    from .stageB import _collect_all_atoms

    candidates: List[Tuple[Node, Optional[callable], str]] = []

    if target.kind.lower() != "nn" or effective_arity(target) != 1:
        return candidates

    # Check for compound atom
    input_expr = compound_input_expr(target)
    axis = int(target.var_idxs[0]) if input_expr is None else None

    tag = target.tag
    if tag is None or tag not in reuse:
        return candidates
    teacher = reuse[tag]

    # Gather teacher data (unified for both compound and univariate)
    data = _gather_teacher_data_1d(
        train_loader, teacher, device, dtype,
        axis=axis, input_expr=input_expr, max_points=5000
    )
    if data is None:
        return candidates

    X, F = data
    X = X.view(-1).to(dtype=torch.float64)
    F = F.view(-1).to(dtype=torch.float64)

    # Filter to finite values
    m_fin = torch.isfinite(F) & torch.isfinite(X)
    if m_fin.sum().item() < min_points:
        return candidates
    X, F = X[m_fin], F[m_fin]

    # Check sign-definiteness
    eps = 1e-12
    pos_frac = float((F > eps).to(dtype=torch.float64).mean().item())
    neg_frac = float((F < -eps).to(dtype=torch.float64).mean().item())

    if max(pos_frac, neg_frac) < 0.85:
        # Not sign-definite enough
        return candidates

    sign = 1.0 if pos_frac >= neg_frac else -1.0
    F_pos = sign * F
    m_pos = F_pos > eps
    if m_pos.sum().item() < min_points:
        return candidates
    X_pos, F_pos = X[m_pos], F_pos[m_pos]

    # Estimate scale and the acosh argument u (extracted so identifiability is testable)
    scale0, u, scale_doublings = _symexp_scale_and_u(F_pos)

    m_valid = (u > 1.0 + 1e-6) & torch.isfinite(u)
    if m_valid.sum().item() < min_points:
        return candidates

    u_valid = u[m_valid]
    X_valid = X_pos[m_valid]

    # --- Tail-regime identifiability guard ------------------------------------
    # A genuine sech/cosh form needs no scale inflation and its data approach the
    # cosh turnover (u -> 1 somewhere). Once scale is inflated past the data's
    # maximum, scale/(2cosh(r)) degenerates to scale*exp(-r), which reproduces
    # ANY positive f with r = log(scale/f): the rel_rms gate then only tests
    # "is log(1/f) Mobius-like" and accepts unrelated positive slice families.
    # In that tail regime we do not reject outright (r(z)=1/z cases can still be
    # exact) but demand a much tighter fit.
    try:
        u_q01 = float(torch.quantile(u_valid, 0.01).item())
    except RuntimeError:
        u_q01 = None
    identifiable = (scale_doublings == 0) and (u_q01 is not None) and (u_q01 <= u_q01_max)
    eff_rel_rms_threshold = (
        rel_rms_threshold if identifiable else min(rel_rms_threshold, tail_rel_rms_threshold)
    )

    # --- Comparative simpler-family null --------------------------------------
    # A simple rational fit directly on f. If that simpler family explains the
    # slice comparably well, the symexp rewrite adds no structural information
    # and must not win ("more expressive does not imply more informative").
    null_rms: Optional[float] = None
    try:
        null_fit = _fit_rational_coeffs_1d(
            X_pos, F_pos, deg_num=1, deg_den=1, min_points=min_points
        )
        if null_fit is not None:
            _na, _nb = null_fit
            _pred = _eval_ratpoly_1d(X_pos, _na, _nb)
            _res = F_pos - _pred
            _sc = float(torch.sqrt(torch.mean(F_pos * F_pos)))
            if _sc > 0:
                null_rms = float(torch.sqrt(torch.mean(_res * _res)) / _sc)
    except (RuntimeError, ValueError):
        null_rms = None

    def _symexp_gate_ok(rms: float) -> bool:
        return _symexp_gate_decision(
            rms,
            eff_threshold=eff_rel_rms_threshold,
            null_rms=null_rms,
            null_margin=null_margin,
        )

    # r = acosh(u) = log(u + sqrt(u^2 - 1))
    r_target = torch.log(u_valid + torch.sqrt(torch.clamp(u_valid * u_valid - 1.0, min=1e-12)))
    m_fin_r = torch.isfinite(r_target)
    if m_fin_r.sum().item() < min_points:
        return candidates
    X_fit = X_valid[m_fin_r]
    r_fit = r_target[m_fin_r]

    target_var_idxs = tuple(int(i) for i in target.var_idxs)
    key = "_".join(str(i) for i in target_var_idxs)
    var_desc = f"axis={axis}" if axis is not None else f"compound vars={target_var_idxs}"
    scale_variants = _scalar_constant_variants(
        units_spec,
        base_tag=f"symexp1d_scale_{key}",
        scale_init=float(sign * scale0),
        max_fixed=int(max_fixed_consts),
    )

    # Helper function to build a candidate with PolyLeaf for r(z) = a*z (standard variant)
    def _build_candidate_poly(
        linear_coeff: float,
        tag_suffix: str,
        label: str,
        scale_variant: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[Node, Optional[callable], str]]:
        """Build a symexp_denom candidate using PolyLeaf for r(z) = a*z."""
        # Tags for the atoms (unique per variant)
        tag_scale = f"symexp1d_scale_{key}{tag_suffix}"
        # NOTE: we intentionally *do not* reuse the same AtomNode objects in multiple
        # places in the AST. The Stage-B linear_refinement path uses a cached evaluator
        # keyed by id(node); reusing Node objects (DAG-style) breaks leaf indexing and
        # can yield x=None in get_data_batch(), which then crashes SegmentedAdaptor/
        # DualSegmentedAdaptor.linear_refinement.
        #
        # So we create *distinct* atoms for the +r and -r branches, and initialize them
        # to identical values.
        tag_r_p = f"symexp1d_rp_{key}{tag_suffix}"
        tag_r_m = f"symexp1d_rm_{key}{tag_suffix}"
        tag_c_p = f"symexp1d_cp_{key}{tag_suffix}"
        tag_c_m = f"symexp1d_cm_{key}{tag_suffix}"
        tag_m1 = f"symexp1d_m1_{key}{tag_suffix}"

        # Build atoms (each gets its own kwargs dict and deep-copied inputs)
        r_atom_p = AtomNode(kind="poly", var_idxs=target.var_idxs, kwargs={"degree": 1, "min_total": 1}, tag=tag_r_p, inputs=clone_inputs(target))
        r_atom_m = AtomNode(kind="poly", var_idxs=target.var_idxs, kwargs={"degree": 1, "min_total": 1}, tag=tag_r_m, inputs=clone_inputs(target))
        c_atom_p = _build_scalar_atom_from_variant(
            {"mode": "scale", "name": tag_c_p, "tag": tag_c_p, "value": 1.0}
        )
        c_atom_m = _build_scalar_atom_from_variant(
            {"mode": "scale", "name": tag_c_m, "tag": tag_c_m, "value": 1.0}
        )
        m1_atom = _build_scalar_atom_from_variant(
            {"mode": "scale", "name": tag_m1, "tag": tag_m1, "value": -1.0}
        )
        scale_variant_eff = (
            scale_variant
            if isinstance(scale_variant, dict)
            else {
                "mode": "scale",
                "name": tag_scale,
                "tag": tag_scale,
                "value": float(sign * scale0),
                "label_suffix": "",
            }
        )
        scale_atom = _build_scalar_atom_from_variant(
            {
                "mode": scale_variant_eff.get("mode", "scale"),
                "name": scale_variant_eff.get("name", tag_scale),
                "tag": tag_scale,
                "value": float(scale_variant_eff.get("value", sign * scale0)),
            }
        )
        label_eff = f"{label}{str(scale_variant_eff.get('label_suffix', ''))}"

        # Build tree: scale / (exp(c*r) + exp(-1*c*r))
        cr_p = MulNode(c_atom_p, r_atom_p)
        cr_m = MulNode(c_atom_m, r_atom_m)
        exp_p = ExpNode(cr_p)
        exp_m = ExpNode(MulNode(m1_atom, cr_m))
        denom = AddNode(exp_p, exp_m)
        inv_denom = PowNode(denom, exponent=-1.0)
        new_sub = MulNode(scale_atom, inv_denom)

        cand_root = _replace_node(root, target, new_sub)
        if cand_root is None:
            return None

        # Capture fitted values for custom init
        scale_init = float(scale_variant_eff.get("value", sign * scale0))
        linear_init = float(linear_coeff)

        def _custom_init(root_inner: Node, model_inner: torch.nn.Module):
            """Initialize the symexp_denom_1d atoms from fitted values."""
            from nestynet_sr.sr_core.atoms import PolyLeaf

            from .stageB import _poly_zero_and_set, build_atom_to_leaf_map
            from .stageB.leaf_utils import _set_constant_leaf_value

            atom_to_leaf = build_atom_to_leaf_map(root_inner, model_inner)

            # Find leaves by walking atoms and matching tags
            atoms = _collect_all_atoms(root_inner)
            leaf_scale = None
            leaf_r_p = None
            leaf_r_m = None
            leaf_c_p = None
            leaf_c_m = None
            leaf_m1 = None

            for atom in atoms:
                if not isinstance(atom, AtomNode):
                    continue
                leaf = atom_to_leaf.get(id(atom), None)
                if leaf is None:
                    continue
                if atom.tag == tag_scale:
                    leaf_scale = leaf
                elif atom.tag == tag_r_p:
                    leaf_r_p = leaf
                elif atom.tag == tag_r_m:
                    leaf_r_m = leaf
                elif atom.tag == tag_c_p:
                    leaf_c_p = leaf
                elif atom.tag == tag_c_m:
                    leaf_c_m = leaf
                elif atom.tag == tag_m1:
                    leaf_m1 = leaf

            # Initialize scalar constants on the scale-style leaves.
            if leaf_scale is not None:
                _set_constant_leaf_value(leaf_scale, float(scale_init))

            if leaf_c_p is not None:
                _set_constant_leaf_value(leaf_c_p, 1.0)
            if leaf_c_m is not None:
                _set_constant_leaf_value(leaf_c_m, 1.0)

            if leaf_m1 is not None:
                _set_constant_leaf_value(leaf_m1, -1.0)

            # Initialize poly r(z) = a*z with fitted linear coefficient
            # With min_total=1 and degree=1, exps = [[1]] so key is (1,)
            if leaf_r_p is not None:
                core = getattr(leaf_r_p, "core", getattr(leaf_r_p, "model", leaf_r_p))
                if isinstance(core, PolyLeaf):
                    _poly_zero_and_set(leaf_r_p, {(1,): linear_init})
            if leaf_r_m is not None:
                core = getattr(leaf_r_m, "core", getattr(leaf_r_m, "model", leaf_r_m))
                if isinstance(core, PolyLeaf):
                    _poly_zero_and_set(leaf_r_m, {(1,): linear_init})

            print(f"[Stage B custom_init {label_eff}] {var_desc}, scale≈{scale_init:.3g}, r±(z)≈{linear_init:.3g}*z")

        _custom_init._after_analytic_init = True
        return cand_root, _custom_init, label_eff

    # Helper function to build a reciprocal-coordinate candidate with PolyLeaf:
    # r(z) = a/z is represented as poly(w), w = 1/z.  This keeps reciprocal
    # handling in the coordinate layer instead of introducing a distinct inverse
    # monomial atom type.
    def _build_candidate_recip_poly(
        amp_init: float,
        tag_suffix: str,
        label: str,
        scale_variant: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[Node, Optional[callable], str]]:
        """Build a symexp_denom candidate using PolyLeaf on w=1/z."""
        recip_inputs = _single_power_coordinate_inputs(target, -1.0)
        if recip_inputs is None:
            return None

        # Tags for the atoms (unique per variant)
        tag_scale = f"symexp1d_scale_{key}{tag_suffix}"
        tag_r_p = f"symexp1d_rp_{key}{tag_suffix}"
        tag_r_m = f"symexp1d_rm_{key}{tag_suffix}"
        tag_c_p = f"symexp1d_cp_{key}{tag_suffix}"
        tag_c_m = f"symexp1d_cm_{key}{tag_suffix}"
        tag_m1 = f"symexp1d_m1_{key}{tag_suffix}"

        # Build atoms (each gets its own kwargs dict and deep-copied inputs)
        r_atom_p = AtomNode(
            kind="poly",
            var_idxs=target.var_idxs,
            kwargs={"degree": 1, "min_total": 1},
            tag=tag_r_p,
            inputs=tuple(clone_ast(inp) for inp in recip_inputs),
        )
        r_atom_m = AtomNode(
            kind="poly",
            var_idxs=target.var_idxs,
            kwargs={"degree": 1, "min_total": 1},
            tag=tag_r_m,
            inputs=tuple(clone_ast(inp) for inp in recip_inputs),
        )
        c_atom_p = _build_scalar_atom_from_variant(
            {"mode": "scale", "name": tag_c_p, "tag": tag_c_p, "value": 1.0}
        )
        c_atom_m = _build_scalar_atom_from_variant(
            {"mode": "scale", "name": tag_c_m, "tag": tag_c_m, "value": 1.0}
        )
        m1_atom = _build_scalar_atom_from_variant(
            {"mode": "scale", "name": tag_m1, "tag": tag_m1, "value": -1.0}
        )
        scale_variant_eff = (
            scale_variant
            if isinstance(scale_variant, dict)
            else {
                "mode": "scale",
                "name": tag_scale,
                "tag": tag_scale,
                "value": float(sign * scale0),
                "label_suffix": "",
            }
        )
        scale_atom = _build_scalar_atom_from_variant(
            {
                "mode": scale_variant_eff.get("mode", "scale"),
                "name": scale_variant_eff.get("name", tag_scale),
                "tag": tag_scale,
                "value": float(scale_variant_eff.get("value", sign * scale0)),
            }
        )
        label_eff = f"{label}{str(scale_variant_eff.get('label_suffix', ''))}"

        # Build tree: scale / (exp(c*r) + exp(-1*c*r))
        cr_p = MulNode(c_atom_p, r_atom_p)
        cr_m = MulNode(c_atom_m, r_atom_m)
        exp_p = ExpNode(cr_p)
        exp_m = ExpNode(MulNode(m1_atom, cr_m))
        denom = AddNode(exp_p, exp_m)
        inv_denom = PowNode(denom, exponent=-1.0)
        new_sub = MulNode(scale_atom, inv_denom)

        cand_root = _replace_node(root, target, new_sub)
        if cand_root is None:
            return None

        # Capture fitted values for custom init
        scale_init = float(scale_variant_eff.get("value", sign * scale0))
        amp_val = float(amp_init)

        def _custom_init(root_inner: Node, model_inner: torch.nn.Module):
            """Initialize the symexp_denom_1d atoms from fitted values."""
            from .stageB import _poly_zero_and_set, build_atom_to_leaf_map
            from .stageB.leaf_utils import _set_constant_leaf_value

            atom_to_leaf = build_atom_to_leaf_map(root_inner, model_inner)

            # Find leaves by walking atoms and matching tags
            atoms = _collect_all_atoms(root_inner)
            leaf_scale = None
            leaf_r_p = None
            leaf_r_m = None
            leaf_c_p = None
            leaf_c_m = None
            leaf_m1 = None

            for atom in atoms:
                if not isinstance(atom, AtomNode):
                    continue
                leaf = atom_to_leaf.get(id(atom), None)
                if leaf is None:
                    continue
                if atom.tag == tag_scale:
                    leaf_scale = leaf
                elif atom.tag == tag_r_p:
                    leaf_r_p = leaf
                elif atom.tag == tag_r_m:
                    leaf_r_m = leaf
                elif atom.tag == tag_c_p:
                    leaf_c_p = leaf
                elif atom.tag == tag_c_m:
                    leaf_c_m = leaf
                elif atom.tag == tag_m1:
                    leaf_m1 = leaf

            # Initialize scalar constants on the scale-style leaves.
            if leaf_scale is not None:
                _set_constant_leaf_value(leaf_scale, float(scale_init))

            if leaf_c_p is not None:
                _set_constant_leaf_value(leaf_c_p, 1.0)
            if leaf_c_m is not None:
                _set_constant_leaf_value(leaf_c_m, 1.0)

            if leaf_m1 is not None:
                _set_constant_leaf_value(leaf_m1, -1.0)

            # Initialize poly(w) = amp*w, where w = 1/z.
            if leaf_r_p is not None:
                core = getattr(leaf_r_p, "core", getattr(leaf_r_p, "model", leaf_r_p))
                if isinstance(core, PolyLeaf):
                    _poly_zero_and_set(leaf_r_p, {(1,): amp_val})
            if leaf_r_m is not None:
                core = getattr(leaf_r_m, "core", getattr(leaf_r_m, "model", leaf_r_m))
                if isinstance(core, PolyLeaf):
                    _poly_zero_and_set(leaf_r_m, {(1,): amp_val})

            print(
                f"[Stage B custom_init {label_eff}] {var_desc}, "
                f"scale≈{scale_init:.3g}, r±(1/z)≈{amp_val:.3g}*(1/z)"
            )

        _custom_init._after_analytic_init = True
        return cand_root, _custom_init, label_eff

    # Evaluate fit quality helper
    def _evaluate_fit(X_fit_eval: torch.Tensor, r_fit_eval: torch.Tensor, a_coeffs: torch.Tensor, b_coeffs: torch.Tensor) -> float:
        """Compute relative RMS error of the rational fit."""
        from .fitting_utils import _enumerate_exponents, _eval_monomials
        exps_num = _enumerate_exponents(1, 1)
        exps_den = _enumerate_exponents(1, 1)
        exps_num_t = torch.tensor(exps_num, dtype=torch.int64, device=X_fit_eval.device)
        exps_den_t = torch.tensor(exps_den, dtype=torch.int64, device=X_fit_eval.device)

        Phi_num = _eval_monomials(X_fit_eval.view(-1, 1), exps_num_t)
        Phi_den = _eval_monomials(X_fit_eval.view(-1, 1), exps_den_t)

        P = Phi_num @ a_coeffs
        Q = Phi_den @ b_coeffs
        Q_safe = Q.clamp_min(1e-8)
        r_hat = P / Q_safe

        resid = r_hat - r_fit_eval
        std_r = float(r_fit_eval.std(unbiased=False).item())
        if std_r < 1e-12:
            return float('inf')
        return float(torch.sqrt(torch.mean(resid ** 2)).item()) / std_r

    # --- Standard fit: r(z) ---
    # Track candidates with their rel_rms for sorting
    candidates_with_rms: List[Tuple[Tuple[Node, Optional[callable], str], float]] = []

    fit_result = _fit_rational_coeffs_1d(
        X_fit, r_fit, deg_num=1, deg_den=1, min_points=min_points
    )
    if fit_result is not None:
        a_coeffs, b_coeffs = fit_result
        rms_rel = _evaluate_fit(X_fit, r_fit, a_coeffs, b_coeffs)
        _accept_std = _symexp_gate_ok(rms_rel)
        record_gate(
            "symexp_denom_1d", "rel_rms", rms_rel, eff_rel_rms_threshold,
            accepted=_accept_std,
            context={
                "variant": "std",
                "scale_doublings": scale_doublings,
                "u_q01": u_q01,
                "identifiable": identifiable,
                "null_rms": null_rms,
                "var_desc": var_desc,
            },
        )
        if _accept_std:
            # Extract linear coefficient: from ratpoly fit (a0 + a1*z)/(b0 + b1*z)
            # We simplify to r(z) = a*z by extracting the effective linear slope
            # a_coeffs = [a0, a1], b_coeffs = [b0, b1]
            _a0, a1 = float(a_coeffs[0].item()), float(a_coeffs[1].item())
            b0, _b1 = float(b_coeffs[0].item()), float(b_coeffs[1].item())
            # Approximate linear coefficient: if b0 >> b1*z_typical, then r ≈ (a0 + a1*z)/b0
            # The effective slope is a1/b0 (assuming b0 normalizes the denominator)
            linear_coeff = a1 / max(abs(b0), 1e-8) if abs(b0) > 1e-8 else a1
            print(f"[Stage B] symexp_denom_1d accepted: rel_rms={rms_rel:.4f}, scale≈{sign * scale0:.3g}, linear≈{linear_coeff:.3g}, {var_desc}")
            for vi, s_var in enumerate(scale_variants):
                suffix = "" if vi == 0 else f"_sv{vi}"
                cand = _build_candidate_poly(
                    linear_coeff,
                    suffix,
                    "symexp_denom_1d",
                    scale_variant=s_var,
                )
                if cand is not None:
                    candidates_with_rms.append((cand, rms_rel))
        else:
            print(
                f"[Stage B] symexp_denom_1d rejected: rel_rms={rms_rel:.4f} "
                f"(eff_thr={eff_rel_rms_threshold}, identifiable={identifiable}, "
                f"null_rms={'%.4f' % null_rms if null_rms is not None else 'n/a'}), {var_desc}"
            )

    # --- Reciprocal fit: r(1/z) for nontrivial-input atoms ---
    # This helps when the true function is cosh(1/z) - fitting r(w) where w=1/z is trivial
    if has_nontrivial_input(target):
        # Compute w = 1/z
        eps_recip = 1e-8
        m_nonzero = torch.abs(X_fit) > eps_recip
        if m_nonzero.sum().item() >= min_points:
            X_recip = 1.0 / X_fit[m_nonzero]
            r_fit_recip = r_fit[m_nonzero]

            # Filter out non-finite values from reciprocal
            m_fin_recip = torch.isfinite(X_recip)
            if m_fin_recip.sum().item() >= min_points:
                X_recip = X_recip[m_fin_recip]
                r_fit_recip = r_fit_recip[m_fin_recip]

                fit_result_recip = _fit_rational_coeffs_1d(
                    X_recip, r_fit_recip, deg_num=1, deg_den=1, min_points=min_points
                )
                if fit_result_recip is not None:
                    a_coeffs_recip, b_coeffs_recip = fit_result_recip
                    rms_rel_recip = _evaluate_fit(X_recip, r_fit_recip, a_coeffs_recip, b_coeffs_recip)
                    _accept_recip = _symexp_gate_ok(rms_rel_recip)
                    record_gate(
                        "symexp_denom_1d", "rel_rms", rms_rel_recip, eff_rel_rms_threshold,
                        accepted=_accept_recip,
                        context={
                            "variant": "recip",
                            "scale_doublings": scale_doublings,
                            "u_q01": u_q01,
                            "identifiable": identifiable,
                            "null_rms": null_rms,
                            "var_desc": var_desc,
                        },
                    )
                    if _accept_recip:
                        # Extract coefficients: r(w) = (a0 + a1*w)/(b0 + b1*w)
                        a0_r = float(a_coeffs_recip[0].item())
                        a1_r = float(a_coeffs_recip[1].item())
                        b0_r = float(b_coeffs_recip[0].item())
                        b1_r = float(b_coeffs_recip[1].item())

                        # Check if fit shows pure 1/z form: |a0| < eps and |b1| < eps
                        # meaning r(w) ≈ a1*w/b0 = (a1/b0)/z
                        eps_pure = 0.1  # Relative threshold for "small" coefficients
                        a1_scale = max(abs(a1_r), 1e-8)
                        b0_scale = max(abs(b0_r), 1e-8)
                        is_pure_inverse = (abs(a0_r) < eps_pure * a1_scale) and (abs(b1_r) < eps_pure * b0_scale)

                        if is_pure_inverse:
                            # Use a reciprocal-coordinate PolyLeaf for r(z) = amp/z.
                            amp_init = a1_r / b0_r if abs(b0_r) > 1e-8 else a1_r
                            print(
                                "[Stage B] symexp_denom_1d_recip (poly on 1/z) "
                                f"accepted: rel_rms={rms_rel_recip:.4f}, "
                                f"scale≈{sign * scale0:.3g}, amp≈{amp_init:.3g}, {var_desc}"
                            )
                            for vi, s_var in enumerate(scale_variants):
                                suffix = "_recip" if vi == 0 else f"_recip_sv{vi}"
                                cand_recip = _build_candidate_recip_poly(
                                    amp_init,
                                    suffix,
                                    "symexp_denom_1d_recip",
                                    scale_variant=s_var,
                                )
                                if cand_recip is not None:
                                    candidates_with_rms.append((cand_recip, rms_rel_recip))
                        else:
                            # Use PolyLeaf for r(z) = a*z with coefficient from swapped fit
                            # From r(w) = (a0 + a1*w)/(b0 + b1*w), r(1/z) = (a0*z + a1)/(b0*z + b1)
                            # The swapped coefficients give r(z) = (a1 + a0*z)/(b1 + b0*z)
                            # Effective linear: a0/b1 (the slope of z term in swapped form)
                            if abs(b1_r) > 1e-8:
                                linear_coeff = a0_r / b1_r
                            else:
                                linear_coeff = a0_r
                            print(f"[Stage B] symexp_denom_1d_recip (poly) accepted: rel_rms={rms_rel_recip:.4f}, scale≈{sign * scale0:.3g}, linear≈{linear_coeff:.3g}, {var_desc}")
                            for vi, s_var in enumerate(scale_variants):
                                suffix = "_recip" if vi == 0 else f"_recip_sv{vi}"
                                cand_recip = _build_candidate_poly(
                                    linear_coeff,
                                    suffix,
                                    "symexp_denom_1d_recip",
                                    scale_variant=s_var,
                                )
                                if cand_recip is not None:
                                    candidates_with_rms.append((cand_recip, rms_rel_recip))
                    else:
                        print(
                            f"[Stage B] symexp_denom_1d_recip rejected: rel_rms={rms_rel_recip:.4f} "
                            f"(eff_thr={eff_rel_rms_threshold}, identifiable={identifiable}, "
                            f"null_rms={'%.4f' % null_rms if null_rms is not None else 'n/a'}), {var_desc}"
                        )

    # Sort candidates by pre-fit quality (best rel_rms first) so the better variant is tried first
    candidates_with_rms.sort(key=lambda x: x[1])
    candidates = [c[0] for c in candidates_with_rms]

    return candidates


def _make_scaling_based_rewrite(
    root: Node,
    target: AtomNode,
    spec: ScaleSpec,
    k_tol: float = 0.25,
) -> Optional[Node]:
    """
    Use a single-axis ScaleSpec to propose a NN -> {poly, ratpoly} rewrite.

    Rules (very conservative for now):
        k ≈ +1  -> degree-1 polynomial in that axis
        k ≈ +2  -> degree-2 polynomial
        k ≈ -1  -> small rational poly (deg 1 / deg 1)
    """
    k = float(spec.k_hat)
    var_idxs = target.var_idxs
    kind_new: Optional[str] = None
    kwargs_new: Dict[str, Any] = {}

    if abs(k - 1.0) <= k_tol:
        kind_new = "poly"
        kwargs_new = {"degree": 1}
    elif abs(k - 2.0) <= k_tol:
        kind_new = "poly"
        kwargs_new = {"degree": 2}
    elif abs(abs(k) - 0.5) <= k_tol:
        # Half‑power: x^{±1/2} → PowerLeaf
        kind_new = "power"
        # Seed exponent close to the measured k (e.g., 0.5 or -0.5)
        kwargs_new = {"exponent_init": k}
    elif abs(k + 1.0) <= k_tol or k < 0.0:
        # Other negative degrees → reciprocal‑like behaviour
        kind_new = "ratpoly"
        kwargs_new = {
            "deg_num": 0,
            "deg_den": 1,
            "exps_num_override": [[0]],
            "exps_den_override": [[1]],
        }

    if kind_new is None:
        return None

    # Preserve the original tag so we know which NN leaf this came from
    new_atom = AtomNode(
        kind=kind_new,
        var_idxs=var_idxs,
        kwargs=kwargs_new,
        tag=target.tag,
        inputs=clone_inputs(target),
    )
    return _replace_node(root, target, new_atom)


def _make_trig_based_rewrite(root: Node, target: AtomNode, spec: TrigAxisSpec) -> Optional[Node]:
    """
    Existing 1D trig rewrite: NN[x_j] -> SinLinearLeaf(x_j) along a trig axis.

    Also supports compound atoms with input_expr (effective_arity=1).
    """
    if effective_arity(target) != 1:
        return None

    # For compound atoms, skip axis matching (hint comes from underlying variables)
    input_expr = compound_input_expr(target)
    if input_expr is None:
        axis = int(target.var_idxs[0])
        if axis != int(spec.axis):
            return None

    # Build atom, preserving compound inputs
    new_atom = AtomNode(
        kind="sin_linear",
        var_idxs=target.var_idxs,
        kwargs={},
        tag=target.tag,
        inputs=clone_inputs(target),
    )
    return _replace_node(root, target, new_atom)


def _make_tanh_based_rewrite(root: Node, target: AtomNode) -> Optional[Node]:
    """1D tanh rewrite: NN[z] -> TanhLinearLeaf(z).

    Supports compound atoms with input_expr (effective_arity=1), i.e. cases
    where the NN is already parameterized by a detected compound coordinate
    z(x).
    """
    if effective_arity(target) != 1:
        return None

    new_atom = AtomNode(
        kind="tanh_linear",
        var_idxs=target.var_idxs,
        kwargs={},
        tag=target.tag,
        inputs=clone_inputs(target),
    )
    return _replace_node(root, target, new_atom)


def _make_affine_trig_rewrite(root: Node, target: AtomNode, spec: TrigAxisSpec) -> Optional[Node]:
    """
    Build affine trig rewrite: c + A*sin(ωz+φ)

    This can represent shifted cosine forms like 2 - 2*cos(z), since
    cos(z) = sin(z + π/2). The constant term allows fitting the offset.

    Also supports compound atoms with input_expr (effective_arity=1).
    """
    if effective_arity(target) != 1:
        return None

    # For compound atoms, skip axis matching (hint comes from underlying variables)
    input_expr = compound_input_expr(target)
    if input_expr is None:
        axis = int(target.var_idxs[0])
        if axis != int(spec.axis):
            return None

    # Build dimensionless offset atom.
    const_tag = f"{target.tag}_c" if target.tag else None
    const_name = const_tag if const_tag is not None else "s"
    const_atom = _build_scalar_atom_from_variant(
        {"mode": "scale", "name": const_name, "tag": const_tag, "value": 1.0}
    )

    # Build sin_linear atom (preserving compound inputs)
    trig_atom = AtomNode(
        kind="sin_linear",
        var_idxs=target.var_idxs,
        kwargs={},
        tag=target.tag,
        inputs=clone_inputs(target),
    )

    # Combine: c + A*sin(ωz+φ)
    new_sub = AddNode(const_atom, trig_atom)

    return _replace_node(root, target, new_sub)


def _make_multid_trig_rewrite(
    root: Node,
    target: AtomNode,
    spec: TrigAxisSpec,
    degree_arg: int = 1,
    degree_amp: int = 1,
    trig_kind: str = "sin",
    homogeneous: bool = False,
) -> Optional[Node]:
    """
    Rewrite a multi-D NN atom into amplitude(x_other) * trig( arg(x_axis) ),
    using generic unary SinNode/CosNode and PolyLeafs for amplitude/argument.

    - target.kind is expected to be "nn"
    - spec.axis must be one of target.var_idxs
    - Only used when len(target.var_idxs) > 1; 1D case is handled by _make_trig_based_rewrite.
    """
    # Only meaningful if we truly have >1 variable
    if len(target.var_idxs) <= 1:
        return None

    axis = int(spec.axis)
    if axis not in target.var_idxs:
        return None

    compound = has_nontrivial_input(target)
    axis_pos = trivial_input_position(target, axis) if compound else None
    if compound and axis_pos is None:
        # The trig axis lives inside a nontrivial compound input; this family
        # cannot isolate it.  Skip — never fall back to raw var_idxs.
        return None

    # Argument depends (for now) only on that axis - include constant for phase shift
    arg_atom = AtomNode(
        kind="poly",
        var_idxs=[axis],
        kwargs={"degree": degree_arg, "min_total": 0},
        tag=None,
    )

    if trig_kind == "cos":
        trig_node: Node = CosNode(arg_atom)
    else:
        trig_node = SinNode(arg_atom)

    # Amplitude depends on the remaining variables, if any.  For compound
    # targets it consumes the REMAINING input expressions (e.g. z = x0/x1)
    # rather than the raw variables, matching units semantics and eval.
    _mt_amp = degree_amp if homogeneous else 0
    if compound:
        other_inputs = tuple(
            clone_ast(e)
            for j, e in enumerate(get_input_exprs(target))
            if j != axis_pos
        )
        if other_inputs:
            other_axes = sorted(
                {
                    int(v)
                    for e in other_inputs
                    for v in _collect_var_idxs_from_node(e)
                }
            )
            amp_atom = AtomNode(
                kind="poly",
                var_idxs=other_axes,
                kwargs={"degree": degree_amp, "min_total": _mt_amp},
                tag=None,
                inputs=other_inputs,
            )
            new_subtree: Node = MulNode(amp_atom, trig_node)
        else:
            new_subtree = trig_node
        return _replace_node(root, target, new_subtree)

    other_axes = [i for i in target.var_idxs if int(i) != axis]
    if other_axes:
        amp_atom = AtomNode(
            kind="poly",
            var_idxs=other_axes,
            kwargs={"degree": degree_amp, "min_total": _mt_amp},
            tag=None,
        )
        new_subtree = MulNode(amp_atom, trig_node)
    else:
        # This case should normally be handled by _make_trig_based_rewrite,
        # but keep it for completeness.
        new_subtree = trig_node

    return _replace_node(root, target, new_subtree)


def _make_multid_trig_pair_rewrite(
    root: Node,
    target: AtomNode,
    *,
    axis: int,
    partner_axis: int,
    degree_arg: int = 1,
    degree_amp: int = 2,
    trig_kind: str = "cos",
    init_arg_coeffs: Optional[List[float]] = None,
    init_amp_coeffs: Optional[List[float]] = None,
    homogeneous: bool = False,
) -> Optional[Node]:
    """
    Rewrite a multi-D NN atom into amplitude(x_other) * trig(arg(x_axis, x_partner)).

    Parameters
    ----------
    init_arg_coeffs : list of float, optional
        Initial coefficients for the argument polynomial. For degree=1 with 2 vars,
        the order is typically [const, coeff_axis, coeff_partner].
        E.g., [phase, omega, -omega] gives phase + omega*(x_axis - x_partner).
    init_amp_coeffs : list of float, optional
        Initial coefficients for the amplitude polynomial. If amp_vars is empty
        (constant amplitude), this should be [amplitude].
    """
    if axis == partner_axis:
        return None
    if axis not in target.var_idxs or partner_axis not in target.var_idxs:
        return None

    compound = has_nontrivial_input(target)
    axis_pos = trivial_input_position(target, axis) if compound else None
    partner_pos = (
        trivial_input_position(target, partner_axis) if compound else None
    )
    if compound and (axis_pos is None or partner_pos is None):
        # Either axis lives inside a nontrivial compound input; the pair
        # rewrite cannot isolate it.  Skip — never fall back to raw var_idxs.
        return None

    # argument: linear/poly in (axis, partner) - include constant term for phase shifts
    arg_kwargs = {"degree": degree_arg, "min_total": 0}
    if init_arg_coeffs is not None:
        arg_kwargs["init_coeffs"] = init_arg_coeffs
    arg_atom = AtomNode(
        kind="poly",
        var_idxs=[axis, partner_axis],
        kwargs=arg_kwargs,
        tag=f"{target.tag}_arg",
    )

    # amplitude: poly in remaining vars (possibly empty -> constant poly).
    # For compound targets it consumes the REMAINING input expressions.
    _mt_amp_p = degree_amp if homogeneous else 0
    amp_kwargs = {"degree": degree_amp, "min_total": _mt_amp_p}
    if init_amp_coeffs is not None:
        amp_kwargs["init_coeffs"] = init_amp_coeffs
    if compound:
        amp_inputs = tuple(
            clone_ast(e)
            for j, e in enumerate(get_input_exprs(target))
            if j not in (axis_pos, partner_pos)
        )
        amp_vars = sorted(
            {
                int(v)
                for e in amp_inputs
                for v in _collect_var_idxs_from_node(e)
            }
        )
        amp_atom = AtomNode(
            kind="poly",
            var_idxs=amp_vars,
            kwargs=amp_kwargs,
            tag=f"{target.tag}_amp",
            inputs=amp_inputs if amp_inputs else None,
        )
    else:
        amp_vars = [v for v in target.var_idxs if v not in (axis, partner_axis)]
        amp_atom = AtomNode(
            kind="poly",
            var_idxs=amp_vars,
            kwargs=amp_kwargs,
            tag=f"{target.tag}_amp",
        )

    trig_node = CosNode(arg_atom) if trig_kind == "cos" else SinNode(arg_atom)
    new_subtree = MulNode(amp_atom, trig_node)
    return _replace_node(root, target, new_subtree)


def _build_trig_affine_envelope_candidate(
    root: Node,
    target: AtomNode,
    trig_spec: TrigAxisSpec,
    model: torch.nn.Module,
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    deg_offset: int = 2,
    deg_amp2: int = 3,
    min_points: int = 400,
    max_points: int = 5000,
    rel_rms_threshold: float = 0.05,
    variance_ratio_threshold: float = 0.1,
    homogeneous: bool = False,
) -> Tuple[Optional[Node], Optional[Callable]]:
    """
    Build a trig-affine-envelope candidate for multi-D NN atoms:

        NN(x_S) -> A(x_other) + sqrt(B2(x_other)) * cos(arg(x_axis))

    where A and B2 are polynomials in the non-trig axes.

    This uses derivative-based identities to extract the offset A and
    envelope squared B^2 from the NN surrogate:

        y = A(u) + B(u)*cos(ω*t + φ)
        y' = -ωB(u)sin(ω*t + φ)
        y'' = -ω²B(u)cos(ω*t + φ)

    Therefore:
        A(u) = y + y''/ω²
        B(u)² = (y')²/ω² + (y'')²/ω⁴

    Parameters
    ----------
    trig_spec : TrigAxisSpec
        Contains axis index and omega frequency estimate.
    model : torch.nn.Module
        The compiled model containing the NN leaf.
    deg_offset : int
        Polynomial degree for the offset A(x_other).
    deg_amp2 : int
        Polynomial degree for the squared envelope B²(x_other).
    variance_ratio_threshold : float
        Maximum allowed ratio of variance along trig axis vs total variance
        for A and B² to be considered independent of the trig axis.
    """
    from .stageB import _collect_all_atoms
    from .stageB.splits import _gather_nn_atom_value_grad_hess

    if target.kind.lower() != "nn":
        return None, None

    # Skip nontrivial-input atoms - the effective input structure doesn't map cleanly
    # to the original var_idxs, making polynomial construction unreliable.
    if has_nontrivial_input(target):
        return None, None

    var_idxs = tuple(int(i) for i in target.var_idxs)
    dim = len(var_idxs)
    if dim < 2:  # Need at least 2D (trig axis + other axes)
        return None, None

    axis = int(trig_spec.axis)
    if axis not in var_idxs:
        return None, None

    omega = float(trig_spec.omega)
    if omega <= 0 or not math.isfinite(omega):
        return None, None

    # Get derivatives from the NN
    data = _gather_nn_atom_value_grad_hess(
        root=root,
        model=model,
        atom=target,
        train_loader=train_loader,
        device=device,
        dtype=dtype,
        max_points=max_points,
    )

    if data is None:
        return None, None

    X, _, u, du, H = data  # Ignore X_raw (not needed here)
    N = X.shape[0]
    if N < min_points:
        return None, None

    # Use actual dimension of input data (handles compound atoms correctly)
    dim = X.shape[1]

    # Find local axis index within the atom's effective inputs.
    # For compound atoms, derivatives are w.r.t. [z, extra_vars...], not original var_idxs.
    # Plain NN atom (nontrivial inputs already excluded above):
    # axis_local is position in var_idxs
    try:
        axis_local = var_idxs.index(axis)
    except ValueError:
        return None, None

    # Extract derivatives w.r.t. the trig axis
    du_axis = du[:, axis_local]  # [N]
    d2u_axis = H[:, axis_local, axis_local]  # [N]

    # Helper to compute axis dependence (fraction of variance explained by trig axis)
    def _compute_axis_dependence(Y: torch.Tensor, x: torch.Tensor, n_bins: int = 10):
        """Estimate fraction of variance explained by x (trig axis)."""
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-10:
            return 0.0  # All same x, can't estimate

        # Bin the data and compute mean per bin
        bin_edges = torch.linspace(float(x_min), float(x_max), n_bins + 1, device=x.device)
        bin_means = torch.zeros(n_bins, device=Y.device, dtype=Y.dtype)
        bin_counts = torch.zeros(n_bins, device=Y.device)

        for i in range(n_bins):
            if i < n_bins - 1:
                mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
            else:
                mask = (x >= bin_edges[i]) & (x <= bin_edges[i + 1])
            if mask.sum() > 0:
                bin_means[i] = Y[mask].mean()
                bin_counts[i] = mask.sum()

        # Variance between bin means vs total variance
        valid_bins = bin_counts > 0
        if valid_bins.sum() < 2:
            return 0.0

        between_var = (bin_means[valid_bins] - Y.mean()).pow(2).mean()
        total_var = Y.var()
        if total_var < 1e-12:
            return 0.0
        return float(between_var / total_var)

    # Helper to compute A_data and B2_data for a given omega
    def _compute_AB2_for_omega(w: float):
        w_sq = w * w
        w_4 = w_sq * w_sq
        A = u + d2u_axis / w_sq
        B2 = (du_axis * du_axis) / w_sq + (d2u_axis * d2u_axis) / w_4
        return A, B2

    # Refine omega by searching for the value that minimizes axis dependence.
    # The FFT-based estimate can be off when data spans less than a full period.
    omega_init = omega
    best_omega = omega
    best_score = float("inf")

    # Coarse search over a wide range
    n_coarse = 21
    omega_lo, omega_hi = 0.3 * omega_init, 3.0 * omega_init
    for w in torch.linspace(omega_lo, omega_hi, n_coarse).tolist():
        if w <= 0:
            continue
        A_trial, B2_trial = _compute_AB2_for_omega(w)
        fmask = torch.isfinite(A_trial) & torch.isfinite(B2_trial) & (B2_trial >= 0)
        if fmask.sum() < min_points:
            continue
        x_ax = X[:, axis_local][fmask]
        score = _compute_axis_dependence(A_trial[fmask], x_ax) + _compute_axis_dependence(B2_trial[fmask], x_ax)
        if score < best_score:
            best_score = score
            best_omega = w

    # Fine search around the best coarse value
    omega_fine_lo = max(0.1, best_omega * 0.8)
    omega_fine_hi = best_omega * 1.2
    n_fine = 11
    for w in torch.linspace(omega_fine_lo, omega_fine_hi, n_fine).tolist():
        if w <= 0:
            continue
        A_trial, B2_trial = _compute_AB2_for_omega(w)
        fmask = torch.isfinite(A_trial) & torch.isfinite(B2_trial) & (B2_trial >= 0)
        if fmask.sum() < min_points:
            continue
        x_ax = X[:, axis_local][fmask]
        score = _compute_axis_dependence(A_trial[fmask], x_ax) + _compute_axis_dependence(B2_trial[fmask], x_ax)
        if score < best_score:
            best_score = score
            best_omega = w

    omega = best_omega
    if abs(omega - omega_init) > 1e-6:
        print(f"[Stage B] trig_affine_env: refined omega {omega_init:.4f} -> {omega:.4f}")

    # Apply derivative identities to extract A and B² with refined omega
    omega_sq = omega * omega
    omega_4 = omega_sq * omega_sq

    A_data = u + d2u_axis / omega_sq  # [N]
    B2_data = (du_axis * du_axis) / omega_sq + (d2u_axis * d2u_axis) / omega_4  # [N]

    # Filter non-finite values
    finite_mask = torch.isfinite(A_data) & torch.isfinite(B2_data) & (B2_data >= 0)
    if finite_mask.sum() < min_points:
        return None, None

    X = X[finite_mask]
    A_data = A_data[finite_mask]
    B2_data = B2_data[finite_mask]
    x_axis = X[:, axis_local]

    # Validate: A and B² should be (approximately) independent of the trig axis
    A_axis_dep = _compute_axis_dependence(A_data, x_axis)
    B2_axis_dep = _compute_axis_dependence(B2_data, x_axis)

    if A_axis_dep > variance_ratio_threshold or B2_axis_dep > variance_ratio_threshold:
        print(
            f"[Stage B] trig_affine_env rejected: A depends on axis ({A_axis_dep:.3f}), "
            f"B² depends on axis ({B2_axis_dep:.3f}), threshold={variance_ratio_threshold}"
        )
        return None, None

    # Build input for polynomial fitting (other axes only)
    other_axes = [i for i in range(dim) if i != axis_local]
    if not other_axes:
        # Only trig axis - can still fit constants
        other_axes = []
        X_other = torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)
        dim_other = 0
    else:
        X_other = X[:, other_axes]
        dim_other = len(other_axes)

    # Convert to float64 for fitting stability
    X_other = X_other.to(dtype=torch.float64)
    A_data = A_data.to(dtype=torch.float64)
    B2_data = B2_data.to(dtype=torch.float64)

    # Fit polynomial to A_data — iterate over degrees, pick lowest good fit.
    best_A_deg = None
    coeffs_A = None
    A_rms_rel = float("inf")
    if dim_other > 0:
        for _da in range(1, deg_offset + 1):
            _mt_a = _da if homogeneous else 0
            _exps = _enumerate_exponents(dim_other, _da, min_total=_mt_a)
            if not _exps:
                continue
            _exps_t = torch.tensor(_exps, dtype=torch.int64, device=X_other.device)
            _Phi = _eval_monomials(X_other, _exps_t)
            try:
                _c = (torch.linalg.pinv(_Phi) @ A_data.unsqueeze(1)).squeeze(1)
                _pred = _Phi @ _c
                _resid = A_data - _pred
                _scale = float(torch.sqrt(torch.mean(A_data * A_data)))
                _rr = float(torch.sqrt(torch.mean(_resid * _resid)) / _scale) if _scale > 0 else 0.0
            except RuntimeError:
                continue
            if _rr <= rel_rms_threshold:
                best_A_deg, coeffs_A, A_rms_rel = _da, _c, _rr
                break  # Occam: take the lowest passing degree
            if _rr < A_rms_rel:
                best_A_deg, coeffs_A, A_rms_rel = _da, _c, _rr
    else:
        # Constant polynomial (0D)
        Phi_A = torch.ones(X_other.shape[0], 1, device=X_other.device, dtype=X_other.dtype)
        try:
            coeffs_A = (torch.linalg.pinv(Phi_A) @ A_data.unsqueeze(1)).squeeze(1)
            A_pred = Phi_A @ coeffs_A
            A_resid = A_data - A_pred
            A_scale = float(torch.sqrt(torch.mean(A_data * A_data)))
            A_rms_rel = float(torch.sqrt(torch.mean(A_resid * A_resid)) / A_scale) if A_scale > 0 else 0.0
        except RuntimeError:
            return None, None
        best_A_deg = 0

    if coeffs_A is None:
        return None, None

    # Fit polynomial to B2_data — iterate over degrees, pick lowest good fit.
    best_B2_deg = None
    coeffs_B2 = None
    B2_rms_rel = float("inf")
    if dim_other > 0:
        for _db in range(1, deg_amp2 + 1):
            _mt_b = _db if homogeneous else 0
            _exps = _enumerate_exponents(dim_other, _db, min_total=_mt_b)
            if not _exps:
                continue
            _exps_t = torch.tensor(_exps, dtype=torch.int64, device=X_other.device)
            _Phi = _eval_monomials(X_other, _exps_t)
            try:
                _c = (torch.linalg.pinv(_Phi) @ B2_data.unsqueeze(1)).squeeze(1)
                _pred = _Phi @ _c
                _resid = B2_data - _pred
                _scale = float(torch.sqrt(torch.mean(B2_data * B2_data)))
                _rr = float(torch.sqrt(torch.mean(_resid * _resid)) / _scale) if _scale > 0 else 0.0
            except RuntimeError:
                continue
            if _rr <= rel_rms_threshold:
                best_B2_deg, coeffs_B2, B2_rms_rel = _db, _c, _rr
                break  # Occam: take the lowest passing degree
            if _rr < B2_rms_rel:
                best_B2_deg, coeffs_B2, B2_rms_rel = _db, _c, _rr
    else:
        Phi_B2 = torch.ones(X_other.shape[0], 1, device=X_other.device, dtype=X_other.dtype)
        try:
            coeffs_B2 = (torch.linalg.pinv(Phi_B2) @ B2_data.unsqueeze(1)).squeeze(1)
            B2_pred = Phi_B2 @ coeffs_B2
            B2_resid = B2_data - B2_pred
            B2_scale = float(torch.sqrt(torch.mean(B2_data * B2_data)))
            B2_rms_rel = float(torch.sqrt(torch.mean(B2_resid * B2_resid)) / B2_scale) if B2_scale > 0 else 0.0
        except RuntimeError:
            return None, None
        best_B2_deg = 0

    if coeffs_B2 is None:
        return None, None

    # Check if fits are good enough
    record_gate(
        "trig_affine_env", "A_rel_rms", A_rms_rel, rel_rms_threshold,
        accepted=not (A_rms_rel > rel_rms_threshold),
    )
    record_gate(
        "trig_affine_env", "B2_rel_rms", B2_rms_rel, rel_rms_threshold,
        accepted=not (B2_rms_rel > rel_rms_threshold),
    )
    if A_rms_rel > rel_rms_threshold or B2_rms_rel > rel_rms_threshold:
        print(
            f"[Stage B] trig_affine_env rejected: A fit rms_rel={A_rms_rel:.4f}, "
            f"B² fit rms_rel={B2_rms_rel:.4f}, threshold={rel_rms_threshold}"
        )
        return None, None

    # Estimate phase from derivatives
    # sin(θ) = -y'/(ω*B), cos(θ) = -y''/(ω²*B), θ = ω*t + φ
    B_data = torch.sqrt(B2_data.clamp(min=1e-12))
    sin_theta = -du[:, axis_local][finite_mask].to(torch.float64) / (omega * B_data)
    cos_theta = -H[:, axis_local, axis_local][finite_mask].to(torch.float64) / (omega_sq * B_data)
    theta = torch.atan2(sin_theta, cos_theta)
    unwrapped_phase = theta - omega * x_axis.to(torch.float64)

    # Circular mean for phase
    phase_sin = torch.sin(unwrapped_phase).mean()
    phase_cos = torch.cos(unwrapped_phase).mean()
    phi_est = float(torch.atan2(phase_sin, phase_cos))

    # Build the AST: A(other) + sqrt(B2(other)) * cos(arg(axis))
    other_var_idxs = [var_idxs[i] for i in other_axes] if other_axes else []

    # Offset polynomial A(x_other)
    _off_deg = best_A_deg if other_var_idxs else 0
    _off_mt = _off_deg if homogeneous else 0
    offset_atom = AtomNode(
        kind="poly",
        var_idxs=other_var_idxs if other_var_idxs else [var_idxs[0]],  # Fallback to constant
        kwargs={"degree": _off_deg, "min_total": _off_mt},
        tag=f"{target.tag}_trig_off",
    )

    # Envelope squared polynomial B²(x_other)
    _amp_deg = best_B2_deg if other_var_idxs else 0
    _amp_mt = _amp_deg if homogeneous else 0
    amp2_atom = AtomNode(
        kind="poly",
        var_idxs=other_var_idxs if other_var_idxs else [var_idxs[0]],
        kwargs={"degree": _amp_deg, "min_total": _amp_mt},
        tag=f"{target.tag}_trig_amp2",
    )

    # Argument polynomial: phi + omega * x_axis
    # The trig argument must be dimensionless; x_axis is already dimensionless
    # when the trig probe accepted it, so min_total=0 is fine here.
    arg_atom = AtomNode(
        kind="poly",
        var_idxs=[axis],
        kwargs={"degree": 1, "min_total": 0},
        tag=f"{target.tag}_trig_arg",
    )

    # Build: sqrt(B2) * cos(arg)
    sqrt_amp = PowNode(base=amp2_atom, exponent=0.5)
    trig_node = CosNode(arg=arg_atom)
    trig_term = MulNode(sqrt_amp, trig_node)

    # Build: A + sqrt(B2) * cos(arg)
    new_subtree = AddNode(offset_atom, trig_term)
    cand_root = _replace_node(root, target, new_subtree)

    # Store fitted values for custom init
    _coeffs_A = coeffs_A.detach().clone()
    _coeffs_B2 = coeffs_B2.detach().clone()
    _omega_est = omega
    _phi_est = phi_est

    def _custom_init(root_inner: Node, model_inner: torch.nn.Module):
        """Initialize the polynomial leaves with fitted coefficients."""
        atoms = _collect_all_atoms(root_inner)
        leaves = list(model_inner.leaf)

        for atom_i, leaf_mod in zip(atoms, leaves):
            core = getattr(leaf_mod, "core", getattr(leaf_mod, "model", leaf_mod))
            if not isinstance(core, PolyLeaf):
                continue

            tag_i = atom_i.tag
            if tag_i is None:
                continue

            try:
                if tag_i.endswith("_trig_off"):
                    # Initialize offset polynomial
                    with torch.no_grad():
                        if core.coeffs.numel() == _coeffs_A.numel():
                            core.coeffs.copy_(
                                _coeffs_A.to(device=core.coeffs.device, dtype=core.coeffs.dtype)
                            )
                        elif core.coeffs.numel() == 1:
                            # Constant poly
                            core.coeffs.fill_(float(_coeffs_A[0]))
                elif tag_i.endswith("_trig_amp2"):
                    # Initialize amplitude squared polynomial
                    with torch.no_grad():
                        if core.coeffs.numel() == _coeffs_B2.numel():
                            core.coeffs.copy_(
                                _coeffs_B2.to(device=core.coeffs.device, dtype=core.coeffs.dtype)
                            )
                        elif core.coeffs.numel() == 1:
                            core.coeffs.fill_(float(_coeffs_B2[0]))
                elif tag_i.endswith("_trig_arg"):
                    # Initialize argument polynomial: [phi, omega]
                    with torch.no_grad():
                        if core.coeffs.numel() >= 2:
                            core.coeffs[0] = _phi_est
                            core.coeffs[1] = _omega_est
                        elif core.coeffs.numel() == 1:
                            core.coeffs[0] = _omega_est
            except Exception as e:
                print(f"[Stage B] trig_affine_env init failed for {tag_i}: {e}")

    print(
        f"[Stage B] trig_affine_env candidate built: A rms_rel={A_rms_rel:.4f} (deg={best_A_deg}), "
        f"B² rms_rel={B2_rms_rel:.4f} (deg={best_B2_deg}), omega={omega:.4f}, phi={phi_est:.4f}"
        + (", homogeneous" if homogeneous else "")
    )

    return cand_root, _custom_init


def _make_exp_poly_rewrite(
    root: Node,
    target: AtomNode,
    degree: int = 2,
) -> Optional[Node]:
    """
    Rewrite a univariate NN atom into an ExpPolyLeaf of given degree:

        NN(x_j) -> exp_poly(x_j; degree)

    Currently only used for 1D axes to capture Gaussian‑like shapes.
    Also supports compound atoms with input_expr (effective_arity=1).
    """
    if effective_arity(target) != 1:
        return None

    new_atom = AtomNode(
        kind="exp_poly",  # handled by _build_leaf_module
        var_idxs=target.var_idxs,
        kwargs={"degree": int(degree)},
        tag=target.tag,
        inputs=clone_inputs(target),
    )
    return _replace_node(root, target, new_atom)


def _make_exp_ratpoly_rewrite(root, target, deg_num=2, deg_den=2):
    if effective_arity(target) != 1:
        return None

    new_atom = AtomNode(
        kind="exp_ratpoly",
        var_idxs=target.var_idxs,
        kwargs={"deg_num": int(deg_num), "deg_den": int(deg_den)},
        tag=target.tag,
        inputs=clone_inputs(target),
    )
    return _replace_node(root, target, new_atom)


def _build_ratpoly_1d_candidates(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    max_deg_num: int = 4,
    max_deg_den: int = 4,
    min_points: int = 200,
    rel_rms_threshold: float = 0.02,
    enforce_units: bool = False,
    target_dim: Optional[tuple] = None,
    x_dims: Optional[List[tuple]] = None,
) -> List[Tuple[Node, callable, Dict[str, Any]]]:
    """
    Build 1D rational polynomial candidates: nn(x) -> scale * P_red(x)/Q(x).

    Returns a list of ``(root, init_fn, meta)`` tuples ordered by complexity
    (simplest first). Each tuple corresponds to a degree pair whose precheck
    fit passed ``rel_rms_threshold``. This lets Stage B fall back to a more
    expressive 1D ratpoly when the simplest candidate is rejected later.

    When no degree pair passes the threshold, returns an empty list.
    """
    from .fitting_utils import _enumerate_exponents, _eval_monomials
    from .stageB import _collect_all_atoms

    target_inputs = get_input_exprs(target)

    def _stable_target_sig_1d() -> int:
        import zlib

        payload = (
            str(getattr(target, "tag", "") or ""),
            tuple(int(i) for i in target.var_idxs),
            tuple(repr(inp) for inp in target_inputs),
        )
        return int(zlib.crc32(repr(payload).encode("utf-8")) & 0xFFFFFFFF)

    def _support_bitvec_1d(exps: torch.Tensor) -> Tuple[int, ...]:
        if exps.ndim != 2 or int(exps.shape[0]) <= 0:
            return (0,)
        degs = exps.sum(dim=1).to(dtype=torch.int64)
        max_deg = int(degs.max().item())
        bits = [0] * (max_deg + 1)
        for deg in degs.tolist():
            bits[int(deg)] = 1
        return tuple(int(b) for b in bits)

    def _support_signature_1d(target_sig: int, exps_num: torch.Tensor, exps_den: torch.Tensor) -> Tuple[int, ...]:
        num_bits = _support_bitvec_1d(exps_num)
        den_bits = _support_bitvec_1d(exps_den)
        return (int(target_sig), len(num_bits), *num_bits, -1, len(den_bits), *den_bits)

    if target.kind.lower() != "nn" or effective_arity(target) != 1:
        return []
    if bool(enforce_units) and (
        target_dim is None
        or x_dims is None
        or len(x_dims) < 1
    ):
        return []

    # Check for compound atom
    input_expr = compound_input_expr(target)
    axis = int(target.var_idxs[0]) if input_expr is None else None

    tag = target.tag
    if tag is None or tag not in reuse:
        return []
    teacher = reuse[tag]

    # Gather teacher data (unified for both compound and univariate)
    data = _gather_teacher_data_1d(
        train_loader, teacher, device, dtype,
        axis=axis, input_expr=input_expr, max_points=5000
    )
    if data is None:
        return []

    X, F = data
    X = X.view(-1).to(dtype=torch.float64)
    F = F.view(-1).to(dtype=torch.float64)
    N = X.numel()
    if N < min_points:
        return []

    std_F = float(F.std(unbiased=False).item())
    if std_F < 1e-12:
        return []

    # Unit-aware runs construct exact supports before fitting.  A dimensionless
    # input follows this same path and simply puts every monomial in the zero
    # dimension class.
    degree_trials: List[Dict[str, Any]] = []
    support_plan_diagnostics: Optional[Dict[str, Any]] = None
    if bool(enforce_units):
        from .rational_supports import plan_unit_consistent_rational_supports

        support_plan = plan_unit_consistent_rational_supports(
            target_dim=target_dim,
            input_dims=x_dims[:1],
            # Preserve the former dimensional degree probe's bounded
            # auto-raise so, for example, an L^6 target is not lost merely
            # because the numerical caller kept its degree-4 default.
            max_deg_num=max(int(max_deg_num), 8),
            max_deg_den=max(int(max_deg_den), 8),
            coefficient_policy="free_const_only",
            max_attempts=2048,
        )
        support_plan_diagnostics = support_plan.diagnostics()
        for support in support_plan.supports:
            degree_trials.append({
                "complexity": int(support.complexity),
                "deg_num": int(support.degree_num),
                "deg_den": int(support.degree_den),
                "mt_num": 0,
                "mt_den": 0,
                "exps_num": support.numerator_exponents,
                "exps_den": support.denominator_exponents,
                "coefficient_unit_certificate": support.certificate.to_dict(),
            })
        if not degree_trials:
            print(
                "[Stage B] ratpoly_1d support planner found no admissible support "
                f"for vars={target.var_idxs}: {support_plan_diagnostics}"
            )
            return []
    else:
        # No unit payload: preserve the historical dense degree grid.
        for deg_den in range(1, max_deg_den + 1):
            for deg_num in range(0, max_deg_num + 1):
                mt_num = 0
                mt_den = 0
                n_num = len(_enumerate_exponents(1, deg_num, min_total=mt_num))
                n_den = len(_enumerate_exponents(1, deg_den, min_total=mt_den))
                complexity = n_num + n_den
                degree_trials.append({
                    "complexity": int(complexity),
                    "deg_num": int(deg_num),
                    "deg_den": int(deg_den),
                    "mt_num": int(mt_num),
                    "mt_den": int(mt_den),
                    "exps_num": None,
                    "exps_den": None,
                    "coefficient_unit_certificate": None,
                })

    degree_trials.sort(
        key=lambda trial: (
            int(trial["complexity"]),
            int(trial["deg_num"]),
            int(trial["deg_den"]),
            tuple(trial.get("exps_num") or ()),
            tuple(trial.get("exps_den") or ()),
        )
    )

    accepted_trials: List[Dict[str, Any]] = []
    best_reject = None
    for degree_trial in degree_trials:
        complexity = int(degree_trial["complexity"])
        deg_num = int(degree_trial["deg_num"])
        deg_den = int(degree_trial["deg_den"])
        mt_num = int(degree_trial["mt_num"])
        mt_den = int(degree_trial["mt_den"])
        fit_kwargs: Dict[str, Any] = {
            "deg_num": deg_num,
            "deg_den": deg_den,
            "min_points": min_points,
            "min_total_num": mt_num,
            "min_total_den": mt_den,
            "return_support": True,
        }
        if degree_trial.get("exps_num") is not None:
            fit_kwargs["exps_num_override"] = degree_trial["exps_num"]
            fit_kwargs["exps_den_override"] = degree_trial["exps_den"]
        result = _fit_rational_coeffs_1d(X, F, **fit_kwargs)
        if result is None:
            continue
        a_coeffs, b_coeffs, exps_num_t, exps_den_t = result
        exps_num_t = exps_num_t.to(device=X.device, dtype=torch.int64)
        exps_den_t = exps_den_t.to(device=X.device, dtype=torch.int64)

        Phi_num = _eval_monomials(X.view(-1, 1), exps_num_t)
        Phi_den = _eval_monomials(X.view(-1, 1), exps_den_t)

        P = Phi_num @ a_coeffs
        Q = Phi_den @ b_coeffs
        Q_safe = Q.clamp_min(1e-8)
        F_hat = P / Q_safe

        resid = F_hat - F
        rms = float(torch.sqrt(torch.mean(resid ** 2)).item())
        rel_rms = rms / std_F

        if rel_rms <= rel_rms_threshold:
            deg_num_eff = int(exps_num_t.sum(dim=1).max().item()) if int(exps_num_t.shape[0]) > 0 else int(mt_num)
            deg_den_eff = int(exps_den_t.sum(dim=1).max().item()) if int(exps_den_t.shape[0]) > 0 else int(mt_den)
            accepted_trials.append({
                "complexity": int(complexity),
                "deg_num": int(deg_num),
                "deg_den": int(deg_den),
                "deg_num_eff": int(deg_num_eff),
                "deg_den_eff": int(deg_den_eff),
                "mt_num": int(mt_num),
                "mt_den": int(mt_den),
                "n_terms_num": int(exps_num_t.shape[0]),
                "n_terms_den": int(exps_den_t.shape[0]),
                "rel_rms": float(rel_rms),
                "a_coeffs": a_coeffs.detach().cpu().clone(),
                "b_coeffs": b_coeffs.detach().cpu().clone(),
                "exps_num": exps_num_t.detach().cpu().clone(),
                "exps_den": exps_den_t.detach().cpu().clone(),
                "coefficient_unit_certificate": degree_trial.get(
                    "coefficient_unit_certificate"
                ),
            })
        elif best_reject is None or rel_rms < best_reject["rel_rms"]:
            best_reject = {
                "complexity": int(complexity),
                "deg_num": int(deg_num),
                "deg_den": int(deg_den),
                "mt_num": int(mt_num),
                "mt_den": int(mt_den),
                "rel_rms": float(rel_rms),
            }

    if not accepted_trials:
        if best_reject is not None:
            print(
                f"[Stage B] ratpoly_1d rejected: best rel_rms={best_reject['rel_rms']:.4f} "
                f"> {rel_rms_threshold}, deg=({best_reject['deg_num']},{best_reject['deg_den']}), "
                f"vars={target.var_idxs}"
            )
        return []

    _best_per_support: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    for trial in accepted_trials:
        target_sig = _stable_target_sig_1d()
        support_sig = _support_signature_1d(target_sig, trial["exps_num"], trial["exps_den"])
        keep = _best_per_support.get(support_sig)
        if keep is None or float(trial["rel_rms"]) < float(keep["rel_rms"]):
            trial = dict(trial)
            trial["target_signature"] = int(target_sig)
            trial["support_signature"] = support_sig
            _best_per_support[support_sig] = trial

    accepted_trials = list(_best_per_support.values())
    accepted_trials.sort(
        key=lambda t: (
            int(t["n_terms_num"] + t["n_terms_den"]),
            int(t["deg_num_eff"] + t["deg_den_eff"]),
            int(t["deg_num_eff"]),
            int(t["deg_den_eff"]),
            float(t["rel_rms"]),
            int(t["complexity"]),
            int(t["deg_num"]),
            int(t["deg_den"]),
        )
    )

    print(
        f"[Stage B] ratpoly_1d probe: {len(accepted_trials)} sub-threshold trial(s) for vars={target.var_idxs}: "
        + ", ".join(
            f"deg=({t['deg_num']},{t['deg_den']})->({t['deg_num_eff']},{t['deg_den_eff']}) "
            f"nnz=({t['n_terms_num']},{t['n_terms_den']}) rms={t['rel_rms']:.4f}"
            for t in accepted_trials
        )
    )

    var_idxs = tuple(target.var_idxs)
    results: List[Tuple[Node, callable, Dict[str, Any]]] = []
    for trial in accepted_trials:
        deg_num_screen = int(trial["deg_num"])
        deg_den_screen = int(trial["deg_den"])
        deg_num = int(trial["deg_num_eff"])
        deg_den = int(trial["deg_den_eff"])
        mt_num = int(trial["mt_num"])
        mt_den = int(trial["mt_den"])
        rel_rms = float(trial["rel_rms"])
        a_cpu = trial["a_coeffs"]
        b_cpu = trial["b_coeffs"]
        exps_num_cpu = trial["exps_num"]
        exps_den_cpu = trial["exps_den"]
        n_terms_num = int(trial["n_terms_num"])
        n_terms_den = int(trial["n_terms_den"])

        unit_meta: Dict[str, Any] = {}
        if bool(enforce_units):
            from nestynet_sr.sr_core.coefficient_units import (
                solve_rational_coefficient_gauge,
            )

            final_solution = solve_rational_coefficient_gauge(
                target_dim=target_dim,
                input_dims=x_dims[:1],
                numerator_exponents=exps_num_cpu.tolist(),
                denominator_exponents=exps_den_cpu.tolist(),
                numerator_pivot=int(exps_num_cpu.shape[0]) - 1,
                coefficient_policy="free_const_only",
            )
            if not final_solution.ok:
                print(
                    "[Stage B] ratpoly_1d final support rejected by coefficient-unit solver: "
                    f"{final_solution.code}: {final_solution.reason}, vars={target.var_idxs}"
                )
                continue
            unit_meta = {
                "unit_support_planned": True,
                "coefficient_unit_certificate": final_solution.to_dict(),
                "unit_support_diagnostics": dict(
                    support_plan_diagnostics or {}
                ),
            }

        if a_cpu.numel() == 0:
            continue
        lead_val = a_cpu[-1]
        if float(lead_val.abs().item()) < 1e-30:
            print(
                f"[Stage B] ratpoly_1d skipped deg=({deg_num_screen},{deg_den_screen}) "
                f"-> ({deg_num},{deg_den}) on vars={target.var_idxs}: "
                "reduced numerator lead is ~0"
            )
            continue

        _kw: Dict[str, Any] = {
            "deg_num": deg_num,
            "deg_den": deg_den,
            "exps_num_override": [
                [int(v) for v in row]
                for row in exps_num_cpu.tolist()
            ],
            "exps_den_override": [
                [int(v) for v in row]
                for row in exps_den_cpu.tolist()
            ],
        }
        if mt_num > 0:
            _kw["min_total_num"] = mt_num
        if mt_den > 0:
            _kw["min_total_den"] = mt_den
        scale_tag = (
            f"ratpoly1d_scale_{str(target.tag) if target.tag is not None else 'anon'}"
            f"_v{'_'.join(str(int(i)) for i in target.var_idxs)}"
            f"_sn{deg_num_screen}_sd{deg_den_screen}_n{deg_num}_d{deg_den}_mn{mt_num}_md{mt_den}"
        )
        scale_atom = AtomNode(
            kind="scale",
            var_idxs=(),
            kwargs={"init": 1.0, "name": "s"},
            tag=scale_tag,
        )
        _kw["_mul_scale_tag"] = scale_tag
        new_atom = AtomNode(
            kind="rratpoly",
            var_idxs=target.var_idxs,
            kwargs=_kw,
            tag=target.tag,
            inputs=clone_inputs(target),
        )
        cand_root = _replace_node(root, target, MulNode(left=scale_atom, right=new_atom))

        def _make_custom_init(
            _deg_num=deg_num,
            _deg_den=deg_den,
            _deg_num_screen=deg_num_screen,
            _deg_den_screen=deg_den_screen,
            _mt_num=mt_num,
            _mt_den=mt_den,
            _a_cpu=a_cpu,
            _b_cpu=b_cpu,
            _exps_num_cpu=exps_num_cpu.clone(),
            _exps_den_cpu=exps_den_cpu.clone(),
            _n_terms_num=n_terms_num,
            _n_terms_den=n_terms_den,
            _rel_rms=rel_rms,
            _tag=target.tag,
            _scale_tag=scale_tag,
        ):
            def _custom_init(root_inner: Node, model_inner: torch.nn.Module):
                """Copy fitted rational coefficients into the reduced 1D rational leaf."""
                atoms = _collect_all_atoms(root_inner)
                leaves = list(model_inner.leaf)
                rat_core = _find_matching_core(
                    atoms,
                    leaves,
                    core_types=RRationalPolyLeaf,
                    expected_kind="rratpoly",
                    expected_tag=_tag if _tag is not None else None,
                    expected_inputs=target_inputs,
                    predicate=lambda _atom, core: (
                        int(getattr(core, "deg_num", -1)) == _deg_num
                        and int(getattr(core, "deg_den", -1)) == _deg_den
                        and int(getattr(core, "min_total_num", -1)) == _mt_num
                        and int(getattr(core, "min_total_den", -1)) == _mt_den
                        and torch.equal(
                            core.exps_num_full.detach().cpu(),
                            _exps_num_cpu.to(dtype=torch.int64),
                        )
                        and torch.equal(
                            core.exps_den.detach().cpu(),
                            _exps_den_cpu.to(dtype=torch.int64),
                        )
                    ),
                )
                scale_core = _find_matching_core(
                    atoms,
                    leaves,
                    core_types=torch.nn.Module,
                    expected_kind="scale",
                    expected_tag=_scale_tag,
                )

                if rat_core is None:
                    print(
                        f"[Stage B custom_init ratpoly_1d] No RRationalPolyLeaf found for vars {var_idxs}, "
                        f"deg=({_deg_num},{_deg_den})"
                    )
                    return

                lead = _a_cpu[int(rat_core.lead_pos_num)] if _a_cpu.numel() > rat_core.lead_pos_num else None
                if lead is None or float(lead.abs().item()) < 1e-30:
                    print(
                        f"[Stage B custom_init ratpoly_1d] Reduced lead vanished for vars {var_idxs}, "
                        f"deg=({_deg_num_screen},{_deg_den_screen})->({_deg_num},{_deg_den})"
                    )
                    return

                with torch.no_grad():
                    if _a_cpu.numel() == int(rat_core.exps_num_full.shape[0]) and rat_core.free_pos_num.numel() > 0:
                        idx = rat_core.free_pos_num.to(device=_a_cpu.device)
                        free_a = _a_cpu[idx] / lead
                        rat_core.coeffs_num.copy_(
                            free_a.to(device=rat_core.coeffs_num.device, dtype=rat_core.coeffs_num.dtype)
                        )
                    if _b_cpu.numel() == rat_core.coeffs_den.numel():
                        rat_core.coeffs_den.copy_(
                            _b_cpu.to(device=rat_core.coeffs_den.device, dtype=rat_core.coeffs_den.dtype)
                        )
                    if scale_core is not None and hasattr(scale_core, "value"):
                        scale_core.value.copy_(
                            torch.as_tensor(
                                float(lead.item()),
                                dtype=scale_core.value.dtype,
                                device=scale_core.value.device,
                            )
                        )
                print(
                    f"[Stage B custom_init ratpoly_1d] Init on vars {var_idxs}: "
                    f"deg=({_deg_num_screen},{_deg_den_screen})->({_deg_num},{_deg_den}), "
                    f"nnz=({_n_terms_num},{_n_terms_den}), rel_rms≈{_rel_rms:.2e}"
                )
            return _custom_init

        results.append((
            cand_root,
            _make_custom_init(),
            {
                **unit_meta,
                "pattern_family": "ratpoly_1d",
                "terminal_family": "ratpoly_1d",
                "terminal_protected": bool(
                    deg_num <= 1
                    and deg_den <= 1
                    and n_terms_num <= 2
                    and n_terms_den <= 2
                ),
                "terminal_priority_family": (
                    "sparse_mobius_1d"
                    if (
                        deg_num <= 1
                        and deg_den <= 1
                        and n_terms_num <= 2
                        and n_terms_den <= 2
                    )
                    else "ratpoly_1d"
                ),
                "terminal_n_terms": int(n_terms_num + n_terms_den),
                "ratpoly_scale_tag": scale_tag,
                "ratpoly_target_tag": target.tag,
                "ratpoly_target_sig": int(trial["target_signature"]),
                "ratpoly_var_idxs": tuple(int(i) for i in target.var_idxs),
                "deg_num": deg_num,
                "deg_den": deg_den,
                "deg_num_screen": deg_num_screen,
                "deg_den_screen": deg_den_screen,
                "n_terms_num": n_terms_num,
                "n_terms_den": n_terms_den,
                "precheck_rel_rms": rel_rms,
                "probe_rms_rel": rel_rms,
                "reuse_blacklist_tags": [
                    str(t)
                    for t in (target.tag, scale_tag)
                    if t is not None
                ],
                "log": (
                    f"[Stage B]  Trying 1D ratpoly deg=({deg_num_screen},{deg_den_screen})"
                    f"->({deg_num},{deg_den}) nnz=({n_terms_num},{n_terms_den}) "
                    f"(scale*P_red(x)/Q(x)) rewrite on NN leaf vars {target.var_idxs}"
                ),
                "signature": tuple(int(v) for v in trial["support_signature"]),
            },
        ))

    return results


def _build_ratpoly_1d_candidate(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    max_deg_num: int = 4,
    max_deg_den: int = 4,
    min_points: int = 200,
    rel_rms_threshold: float = 0.02,
    enforce_units: bool = False,
    target_dim: Optional[tuple] = None,
    x_dims: Optional[List[tuple]] = None,
) -> Tuple[Optional[Node], Optional[callable], Optional[float]]:
    """Backward-compat wrapper: returns the first (simplest) 1D ratpoly candidate."""
    results = _build_ratpoly_1d_candidates(
        root=root,
        target=target,
        reuse=reuse,
        train_loader=train_loader,
        device=device,
        dtype=dtype,
        max_deg_num=max_deg_num,
        max_deg_den=max_deg_den,
        min_points=min_points,
        rel_rms_threshold=rel_rms_threshold,
        enforce_units=enforce_units,
        target_dim=target_dim,
        x_dims=x_dims,
    )
    if results:
        cand_root, init_fn, meta = results[0]
        return cand_root, init_fn, meta.get("precheck_rel_rms")
    return None, None, None


def _build_sqrt_ratpoly_1d_candidates(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    max_deg_num: int = 4,
    max_deg_den: int = 4,
    min_points: int = 200,
    rel_rms_threshold: float = 1.0e-3,
    enforce_units: bool = False,
    target_dim: Optional[tuple] = None,
    x_dims: Optional[List[tuple]] = None,
) -> List[Tuple[Node, callable, Dict[str, Any]]]:
    """Build 1D ``sqrt(ratpoly)`` and ``1/sqrt(ratpoly)`` candidates.

    Greedy Stage-A compound construction can turn a formerly bivariate motif
    into a scalar leaf ``NN[z(x)]``.  The multivariate ``sqrt_ratpoly`` rule no
    longer sees that case, while ``sqrt_poly`` only covers polynomial radicands.
    This helper reuses the reduced 1D rational fitter on the lifted targets
    ``f(z)^2`` and ``1/f(z)^2`` and wraps the fitted rational visibly.
    """

    if target.kind.lower() != "nn" or effective_arity(target) != 1:
        return []
    tag = getattr(target, "tag", None)
    if tag is None or tag not in reuse:
        return []

    teacher = reuse[tag]

    class _LiftedTeacher(torch.nn.Module):
        def __init__(self, base: torch.nn.Module, mode: str):
            super().__init__()
            self.base = base
            self.mode = str(mode)

        def forward(self, x):
            y = self.base(x)
            y2 = y * y
            if self.mode == "inv_square":
                return torch.reciprocal(torch.clamp(y2, min=1.0e-24))
            return y2

    def _scaled_dim(dim: Optional[tuple], factor: int) -> Optional[tuple]:
        if dim is None:
            return None
        try:
            return tuple(int(factor) * d for d in tuple(dim))
        except Exception:
            return None

    target_inputs = get_input_exprs(target)
    branch_specs = (
        ("sqrt", 0.5, "square", _scaled_dim(target_dim, 2)),
        ("inv_sqrt", -0.5, "inv_square", _scaled_dim(target_dim, -2)),
    )
    out: List[Tuple[Node, callable, Dict[str, Any]]] = []

    for branch_name, exponent, lift_mode, lifted_dim in branch_specs:
        lifted_reuse = dict(reuse)
        lifted_reuse[tag] = _LiftedTeacher(teacher, lift_mode)
        lifted_results = _build_ratpoly_1d_candidates(
            root=target,
            target=target,
            reuse=lifted_reuse,
            train_loader=train_loader,
            device=device,
            dtype=dtype,
            max_deg_num=max_deg_num,
            max_deg_den=max_deg_den,
            min_points=min_points,
            rel_rms_threshold=rel_rms_threshold,
            enforce_units=enforce_units,
            target_dim=lifted_dim,
            x_dims=x_dims,
        )
        for lifted_root, lifted_init, lifted_meta in lifted_results:
            if lifted_root is None:
                continue
            new_subtree = PowNode(lifted_root, float(exponent))
            cand_root = _replace_node(root, target, new_subtree)
            if cand_root is None:
                continue

            meta = dict(lifted_meta) if isinstance(lifted_meta, dict) else {}
            meta["pattern_family"] = "sqrt_ratpoly_1d"
            meta["sqrt_ratpoly_kind"] = branch_name
            meta["sqrt_ratpoly_inner_family"] = "ratpoly_1d"
            meta["sqrt_ratpoly_target_tag"] = tag
            meta["sqrt_ratpoly_target_inputs"] = tuple(repr(inp) for inp in target_inputs)
            meta["log"] = (
                f"[Stage B]  Trying 1D sqrt-ratpoly ({branch_name}) "
                f"deg=({meta.get('deg_num_screen', '?')},{meta.get('deg_den_screen', '?')})"
                f"->({meta.get('deg_num', '?')},{meta.get('deg_den', '?')}) "
                f"nnz=({meta.get('n_terms_num', '?')},{meta.get('n_terms_den', '?')}) "
                f"on NN leaf vars {target.var_idxs}"
            )

            def _make_init(_inner_init=lifted_init, _link=lift_mode):
                def _custom_init(root_inner: Node, model_inner: torch.nn.Module):
                    if _inner_init is not None:
                        _inner_init(root_inner, model_inner)

                _custom_init._after_analytic_init = True
                _custom_init._fit_lift_link = _link
                return _custom_init

            out.append((cand_root, _make_init(), meta))

    return out


def _build_power_1d_candidate(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    exponent: float = -1.0,
    min_points: int = 200,
    rel_rms_threshold: float = 0.05,
) -> Tuple[Optional[Node], Optional[callable]]:
    """
    Build a 1D power-law candidate: nn(x) -> amp * x^exponent.

    Fits amp via least squares from teacher NN data:
        amp = sum(f * x^exp) / sum(x^(2*exp))

    The candidate is represented as a linear PolyLeaf on an explicit coordinate
    z = x^exponent, so inverse powers do not need a separate inv_monomial atom.
    """

    if target.kind.lower() != "nn" or effective_arity(target) != 1:
        return None, None

    # Check for compound atom
    input_expr = compound_input_expr(target)
    axis = int(target.var_idxs[0]) if input_expr is None else None

    tag = target.tag
    if tag is None or tag not in reuse:
        return None, None
    teacher = reuse[tag]

    # Gather teacher data (unified for both compound and univariate)
    data = _gather_teacher_data_1d(
        train_loader, teacher, device, dtype,
        axis=axis, input_expr=input_expr, max_points=5000
    )
    if data is None:
        return None, None

    X, F = data
    X = X.view(-1).to(dtype=torch.float64)
    F = F.view(-1).to(dtype=torch.float64)
    N = X.numel()
    if N < min_points:
        return None, None

    # For negative exponents, x must be positive (can't compute x^(-1) for x<=0)
    if exponent < 0:
        mask = X > 1e-8
        X = X[mask]
        F = F[mask]
        N = X.numel()
        if N < min_points:
            return None, None

    # Fit amp * x^k + c to handle gauge offsets from additive splits.
    # The constant c will be absorbed by sibling NNs during LM optimization,
    # so we use it only for pattern detection, not in the candidate.
    x_pow = X.pow(exponent)
    ones = torch.ones_like(x_pow)
    A = torch.stack([x_pow, ones], dim=1)  # [N, 2]
    result = torch.linalg.lstsq(A, F.unsqueeze(1))
    coeffs = result.solution.squeeze()
    amp = float(coeffs[0])
    c = float(coeffs[1])  # gauge offset - absorbed by siblings

    if not math.isfinite(amp) or not math.isfinite(c):
        return None, None

    # Evaluate fit quality WITH constant (to detect pattern)
    F_hat = amp * x_pow + c
    resid = F_hat - F
    rms = float(torch.sqrt(torch.mean(resid ** 2)).item())
    std_F = float(F.std(unbiased=False).item())
    rel_rms = rms / max(std_F, 1e-12)

    if rel_rms > rel_rms_threshold:
        print(
            "[Stage B] power-coordinate monomial rejected: "
            f"rel_rms={rel_rms:.4f} > {rel_rms_threshold}, "
            f"exponent={float(exponent):.4g}, vars={target.var_idxs}, amp={amp:.4g}, c={c:.4g}"
        )
        return None, None

    coord_inputs = _single_power_coordinate_inputs(target, float(exponent))
    if coord_inputs is None:
        return None, None

    new_atom = AtomNode(
        kind="poly",
        var_idxs=target.var_idxs,
        kwargs={"degree": 1, "min_total": 1},
        tag=target.tag,
        inputs=coord_inputs,
    )
    cand_root = _replace_node(root, target, new_atom)
    if cand_root is None:
        return None, None

    def _custom_init(root_inner: Node, model_inner: torch.nn.Module):
        from .stageB import _collect_all_atoms, _poly_zero_and_set, build_atom_to_leaf_map

        atom_to_leaf = build_atom_to_leaf_map(root_inner, model_inner)
        atoms = _collect_all_atoms(root_inner)
        for atom in atoms:
            if not isinstance(atom, AtomNode):
                continue
            if str(atom.kind).lower() != "poly":
                continue
            if getattr(atom, "tag", None) != target.tag:
                continue
            if not _atom_inputs_match(atom, coord_inputs):
                continue
            leaf = atom_to_leaf.get(id(atom), None)
            if leaf is None:
                continue
            core = getattr(leaf, "core", getattr(leaf, "model", leaf))
            if isinstance(core, PolyLeaf):
                _poly_zero_and_set(leaf, {(1,): amp})
                print(
                    f"[Stage B custom_init power_coordinate] vars={target.var_idxs}, "
                    f"exponent={float(exponent):.4g}, amp≈{amp:.4g}"
                )
                break

    _custom_init._after_analytic_init = True
    return cand_root, _custom_init
