# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

"""Multivariate polynomial, rational, and exponential candidate builders."""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from nestynet_sr.sr_core.atoms import (
    ExpRationalPolyLeaf,
    PolyLeaf,
    PowerLeaf,
    RationalPolyLeaf,
    RRationalPolyLeaf,
    _enumerate_exponents,
    _eval_monomials,
)
from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    ConstNode,
    CosNode,
    ExpNode,
    LogNode,
    MulNode,
    Node,
    PowNode,
    SinNode,
    _collect_var_idxs_from_node,
    _select_inputs_for_var_group,
    ast_equals,
    clone_ast,
    clone_inputs,
    effective_arity,
    get_input_exprs,
    has_nontrivial_input,
    is_trivial_input,
)
from nestynet_sr.sr_core.constants import (
    build_scalar_atom_from_variant as _build_scalar_atom_from_variant,
)

from ._candidate_builders_common import (
    _exps_key,
    _exps_override_from_tensor,
    _find_matching_core,
    _gather_atom_teacher_data,
    _max_total_degree_from_exps,
    _move_sparse_pivot_to_end,
    _parse_pure_difference_expr,
    _replace_node,
    _select_clear_rratpoly_pivot,
    _select_sign_region,
    _support_is_valid,
)
from .features import ScaleSpec, TrigAxisSpec
from .fitting_utils import _fit_rational_coeffs_nd, _rational_probe_nd
from .model_selection import noisy_rel_rms_threshold as _noisy_rel_rms_threshold

# Keep Stage-B imports local in the builders below to avoid a package cycle.

def _make_power_exp_ratpoly_rewrite(
    root: Node,
    target: AtomNode,
    pivot_axis: Optional[int],
    exponent: float,
    deg_num: int = 2,
    deg_den: int = 2,
    pivot_input_expr: Optional[Node] = None,
    power_tag: Optional[str] = None,
    exp_tag: Optional[str] = None,
) -> Node:
    """
    Replace a multi-D NN atom by:

        power(x_pivot)^k * exp_ratpoly(vars_group)

    where the exponent k is typically inferred from a scaling law.
    """
    if pivot_input_expr is not None:
        power_var_idxs = _collect_var_idxs_from_node(pivot_input_expr)
        power_inputs = (clone_ast(pivot_input_expr),)
    else:
        if pivot_axis is None:
            raise ValueError("pivot_axis or pivot_input_expr must be provided")
        power_var_idxs = (int(pivot_axis),)
        power_inputs = None

    power_atom = AtomNode(
        kind="power",
        var_idxs=power_var_idxs,
        kwargs={"exponent_init": float(exponent)},
        tag=power_tag,
        inputs=power_inputs,
    )
    exp_atom = AtomNode(
        kind="exp_ratpoly",
        var_idxs=target.var_idxs,
        kwargs={"deg_num": int(deg_num), "deg_den": int(deg_den)},
        tag=exp_tag,
        inputs=clone_inputs(target),
    )
    new_subtree = MulNode(left=power_atom, right=exp_atom)
    return _replace_node(root, target, new_subtree)


def _make_power_exp_poly_rewrite(
    root: Node,
    target: AtomNode,
    pivot_axis: Optional[int],
    exponent: float,
    degree: int = 2,
    pivot_input_expr: Optional[Node] = None,
) -> Node:
    """
    1D version for univariate NN leaves:

        NN(z) -> power(z)^k * exp_poly(z; degree)
    """
    if pivot_input_expr is not None:
        power_var_idxs = _collect_var_idxs_from_node(pivot_input_expr)
        power_inputs = None if is_trivial_input(pivot_input_expr) else (clone_ast(pivot_input_expr),)
    else:
        if pivot_axis is None:
            raise ValueError("pivot_axis or pivot_input_expr must be provided")
        power_var_idxs = (int(pivot_axis),)
        power_inputs = None

    power_atom = AtomNode(
        kind="power",
        var_idxs=power_var_idxs,
        kwargs={"exponent_init": float(exponent)},
        tag=None,
        inputs=power_inputs,
    )
    exp_atom = AtomNode(
        kind="exp_poly",
        var_idxs=target.var_idxs,
        kwargs={"degree": int(degree)},
        tag=None,
        inputs=clone_inputs(target),
    )
    new_subtree = MulNode(left=power_atom, right=exp_atom)
    return _replace_node(root, target, new_subtree)


def _build_quadratic_poly_candidate(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    degree: int = 2,
    min_points: int = 400,
    max_points: int = 5000,
    rel_rms_threshold: float = 1e-3,
    homogeneous: bool = False,
    target_dim: Optional[tuple] = None,
    x_dims: Optional[List[tuple]] = None,
) -> Tuple[Optional[Node], Optional[callable]]:
    """
    Try to replace a multi-D NN atom by a low-degree polynomial in its
    inputs:

        NN[x_i,...]  →  PolyLeaf(x_i, ...; degree=2)

    We fit the polynomial directly to the Stage-A leaf output using
    least-squares over the monomial basis produced by _enumerate_exponents.

    The candidate is only accepted if the relative RMS residual is small
    (rel_rms_threshold).

    When ``target_dim`` and ``x_dims`` are provided and units are enforced
    (``homogeneous=True``), a dimensional degree probe determines the exact
    set of valid monomials — replacing the crude homogeneous basis which
    is only correct when all variables share the same dimension.
    """
    # Import locally to avoid circular dependency
    from .stageB import _collect_all_atoms

    var_idxs = tuple(int(i) for i in target.var_idxs)
    dim = len(var_idxs)

    # For now, restrict to small-dimensional leaves; Eq.#3 uses dim=2.
    if dim < 2:
        return None, None

    tag = target.tag
    if tag is None or tag not in reuse:
        return None, None
    teacher = reuse[tag]

    data = _gather_atom_teacher_data(
        train_loader=train_loader,
        atom=target,
        teacher=teacher,
        device=device,
        dtype=dtype,
        max_points=max_points,
    )
    if data is None:
        return None, None
    X, F = data
    if X.numel() == 0 or F.numel() == 0:
        return None, None

    X = X.to(dtype=torch.float64)
    F = F.to(dtype=torch.float64).view(-1)
    N, dim = X.shape
    if N < min_points:
        return None, None

    # ── Dimensional degree probe ──
    # Gate on availability of dimensional data, not on `homogeneous`.
    # `homogeneous` only controls the fallback basis when the probe
    # is unavailable.
    _use_dim_probe = (
        target_dim is not None
        and x_dims is not None
        and len(x_dims) == dim
    )
    _probe_exps = None
    if _use_dim_probe:
        from .ratpoly_degree_probe import probe_poly_exponents
        _probe_exps = probe_poly_exponents(target_dim, x_dims, max_degree=degree)

    if _probe_exps is not None:
        # Use dimensionally filtered monomials for the pre-fit.
        exps_list = []
        for k in sorted(_probe_exps):
            if k <= degree:
                exps_list.extend(_probe_exps[k])
        if not exps_list:
            return None, None
        _mt = min(_probe_exps)  # min total degree with valid monomials
        print(
            f"[Stage B] quad_poly: using degree-probe monomials, "
            f"{len(exps_list)} terms, min_deg={_mt}, vars={var_idxs}"
        )
    else:
        _mt = degree if homogeneous else 0
        exps_list = _enumerate_exponents(dim, degree, min_total=_mt)

    exps = torch.tensor(exps_list, dtype=torch.int64, device=X.device)
    Phi = _eval_monomials(X, exps)  # [N, n_terms]

    try:
        coeffs = (torch.linalg.pinv(Phi) @ F.unsqueeze(1)).squeeze(1)  # [n_terms]
    except RuntimeError:
        return None, None

    F_fit = (Phi @ coeffs).view(-1)
    resid = F - F_fit
    rms_abs = float(torch.sqrt(torch.mean(resid * resid)))
    scale = float(torch.sqrt(torch.mean(F * F)))
    if scale < 1e-12:
        rms_rel = 0.0 if rms_abs < 1e-12 else float("inf")
    else:
        rms_rel = rms_abs / scale

    if (not math.isfinite(rms_rel)) or (rms_rel > rel_rms_threshold):
        return None, None

    # Build the replacement AST node
    target_var_idxs = tuple(int(i) for i in target.var_idxs)
    target_inputs = get_input_exprs(target)
    poly_atom = AtomNode(
        kind="poly",
        var_idxs=target_var_idxs,
        kwargs={"degree": int(degree), "min_total": _mt},
        tag=None,
        inputs=clone_inputs(target),
    )
    cand_root = _replace_node(root, target, poly_atom)

    def _custom_init(root_inner: Node, model_inner: torch.nn.Module):
        """
        Copy the fitted polynomial coefficients into the new PolyLeaf.
        """
        atoms = _collect_all_atoms(root_inner)
        leaves = list(model_inner.leaf)
        poly_core = _find_matching_core(
            atoms,
            leaves,
            core_types=PolyLeaf,
            expected_kind="poly",
            expected_inputs=target_inputs,
        )

        if poly_core is None:
            print("[Stage B custom_init quad] No PolyLeaf found for vars", target_var_idxs)
            return

        dev = poly_core.coeffs.device
        dt = poly_core.coeffs.dtype
        with torch.no_grad():
            if coeffs.numel() == poly_core.coeffs.numel():
                poly_core.coeffs.copy_(coeffs.to(device=dev, dtype=dt))
            else:
                print(
                    "[Stage B custom_init quad] Coeff count mismatch; "
                    f"got {coeffs.numel()}, expected {poly_core.coeffs.numel()}"
                )

    return cand_root, _custom_init


def _build_trig_diff_affine_envelope_candidate(
    root: Node,
    target: AtomNode,
    trig_spec: TrigAxisSpec,
    model: torch.nn.Module,
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    partner_axis: int,
    degree_arg: int = 1,
    max_points: int = 5000,
    min_points: int = 400,
    cos_eps: float = 0.15,
    sin_eps: float = 0.15,
    homogeneous: bool = False,
) -> Tuple[Optional[Node], Optional[Callable]]:
    """
    Build a trig-difference affine-envelope candidate for multi-D NN atoms:

        NN(x) -> A(u) + B(u) * cos(arg(x_axis, x_partner))

    with arg ~ omega * (x_axis - x_partner) + phi.

    This targets functions that are affine in a trigonometric difference term,
    while the offset and amplitude depend on the remaining variables.

    The offset signal is extracted from the teacher NN using the identity
    A = y + y''/omega^2 (derivatives w.r.t. the trig axis).

    The amplitude is estimated pointwise using safe divisions (preferring the
    cos-based estimate when |cos(arg)| is not too small).

    Returns (cand_root, custom_init_fn) or (None, None).
    """
    from .stageB import _collect_all_atoms
    from .stageB.splits import _gather_nn_atom_value_grad_hess, build_atom_to_leaf_map
    from .stageB.subtree_utils import _infer_nn_hyperparams_from_root

    if not isinstance(target, AtomNode) or str(target.kind).lower() != "nn":
        return None, None

    var_idxs = tuple(int(i) for i in target.var_idxs)
    input_exprs = tuple(get_input_exprs(target))
    has_compound_inputs = has_nontrivial_input(target)
    if (
        (has_compound_inputs and not input_exprs)
        or ((not has_compound_inputs) and len(var_idxs) < 2)
    ):
        return None, None

    axis = int(trig_spec.axis)
    partner = int(partner_axis)
    if axis == partner:
        return None, None
    if axis not in var_idxs or partner not in var_idxs:
        return None, None

    omega = float(trig_spec.omega)
    if omega <= 0 or (not math.isfinite(omega)):
        return None, None

    # -------------------------------------------------------------------------
    # Compound atom handling: check whether any effective input absorbs
    # (axis, partner) as z = axis - partner.
    # -------------------------------------------------------------------------
    compound_absorbs_trig = False
    compound_sign = 1.0
    trig_local_idx: Optional[int] = None

    if has_compound_inputs:
        for local_idx, expr in enumerate(input_exprs):
            # Check if this effective input is a pure difference that matches
            # (axis, partner).
            diff_pair = _parse_pure_difference_expr(expr)
            if diff_pair is None:
                continue
            i_diff, j_diff = diff_pair
            if not ((i_diff == axis and j_diff == partner) or (i_diff == partner and j_diff == axis)):
                continue

            # The remaining effective inputs must not still depend on the trig
            # pair; otherwise the offset/amplitude are not functions of clean
            # "other" coordinates.
            other_raw_vars: set[int] = set()
            for j, other_expr in enumerate(input_exprs):
                if j == local_idx:
                    continue
                other_raw_vars.update(int(v) for v in _collect_var_idxs_from_node(other_expr))
            if axis in other_raw_vars or partner in other_raw_vars:
                continue

            compound_absorbs_trig = True
            trig_local_idx = int(local_idx)
            # Track sign: if z = partner - axis, omega should be negated.
            compound_sign = 1.0 if i_diff == axis else -1.0
            break

        if not compound_absorbs_trig:
            # Compound exists but doesn't absorb trig axes - can't handle
            return None, None

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

    X, _, u, du, H = data
    N = int(X.shape[0])
    if N < min_points:
        return None, None

    # -------------------------------------------------------------------------
    # Derivative extraction depends on compound vs non-compound
    # -------------------------------------------------------------------------
    if compound_absorbs_trig:
        # For compound atoms, derivatives are in effective-input coordinates.
        axis_local = int(trig_local_idx)
        du_axis = du[:, axis_local]
        d2u_axis = H[:, axis_local, axis_local]
    else:
        try:
            axis_local = list(var_idxs).index(axis)
            partner_local = list(var_idxs).index(partner)
        except ValueError:
            return None, None

        du_axis = du[:, axis_local]
        d2u_axis = H[:, axis_local, axis_local]

    omega_sq = omega * omega

    # Offset: A = y + y''/omega^2
    A_data = u + d2u_axis / omega_sq

    # Initial argument: omega*(x_axis - x_partner) + phi0
    # Keep phi0=0 here; training can adjust the constant term.
    phi0 = 0.0
    if compound_absorbs_trig:
        # For compound: z = x_axis - x_partner (or with sign flip),
        # possibly at any effective-input position.
        z_vals = X[:, int(trig_local_idx)]
        effective_omega = omega * compound_sign
        arg_vals = effective_omega * z_vals + phi0
    else:
        x_axis = X[:, axis_local]
        x_partner = X[:, partner_local]
        arg_vals = omega * (x_axis - x_partner) + phi0
        effective_omega = omega

    cos_vals = torch.cos(arg_vals)
    sin_vals = torch.sin(arg_vals)

    # Signed amplitude estimate (pointwise)
    B_data = torch.full_like(u, float("nan"))

    mask_cos = torch.abs(cos_vals) > float(cos_eps)
    if mask_cos.any():
        B_data[mask_cos] = ((u - A_data) / cos_vals)[mask_cos]

    mask_sin = torch.abs(sin_vals) > float(sin_eps)
    if mask_sin.any():
        B_sin = (-du_axis) / (effective_omega * sin_vals)
        # Fill missing values first
        m = mask_sin & (~torch.isfinite(B_data))
        if m.any():
            B_data[m] = B_sin[m]
        # Where both are available, average for stability
        m2 = mask_sin & torch.isfinite(B_data)
        if m2.any():
            B_data[m2] = 0.5 * (B_data[m2] + B_sin[m2])

    finite = torch.isfinite(A_data) & torch.isfinite(B_data)
    if int(finite.sum().item()) < min_points:
        return None, None

    # -------------------------------------------------------------------------
    # Identify "other" variables for offset/amplitude NNs
    # -------------------------------------------------------------------------
    if compound_absorbs_trig:
        other_locals = [i for i in range(X.shape[1]) if i != int(trig_local_idx)]
        other_input_exprs = tuple(clone_ast(input_exprs[i]) for i in other_locals)
        other_vars = sorted(
            {
                int(v)
                for expr in other_input_exprs
                for v in _collect_var_idxs_from_node(expr)
            }
        )
    else:
        other_locals = [i for i in range(X.shape[1]) if i not in (axis_local, partner_local)]
        other_vars = [int(var_idxs[i]) for i in other_locals]
        other_input_exprs = tuple()

    # Teacher data for init (CPU)
    if other_locals:
        X_other_cpu = X[finite][:, other_locals].detach().cpu()
    else:
        X_other_cpu = torch.zeros((int(finite.sum().item()), 1), dtype=torch.float32)
    A_cpu = A_data[finite].detach().cpu()
    B_cpu = B_data[finite].detach().cpu()

    # Build candidate AST: A(other) + B(other) * cos(poly(axis, partner))
    num_segments, dual_layer = _infer_nn_hyperparams_from_root(root)
    nn_kwargs = {"num_segments": int(num_segments), "dual_layer": bool(dual_layer)}

    parent_tag = target.tag if target.tag is not None else f"trigdiff_{id(target)}"
    tag_off = f"{parent_tag}_tdiff_off"
    tag_amp = f"{parent_tag}_tdiff_amp"
    tag_arg = f"{parent_tag}_tdiff_arg"

    if other_vars:
        if compound_absorbs_trig:
            other_inputs = other_input_exprs
            other_atom_var_idxs = tuple(other_vars) if other_vars else tuple(var_idxs)
        else:
            other_inputs = _select_inputs_for_var_group(target, other_vars)
            other_atom_var_idxs = tuple(other_vars)
        off_atom = AtomNode("nn", other_atom_var_idxs, kwargs=nn_kwargs, tag=tag_off,
                            inputs=other_inputs)
        amp_atom = AtomNode("nn", other_atom_var_idxs, kwargs=nn_kwargs, tag=tag_amp,
                            inputs=other_inputs)
    else:
        off_atom = _build_scalar_atom_from_variant(
            {"mode": "scale", "name": tag_off, "tag": tag_off, "value": 1.0}
        )
        amp_atom = _build_scalar_atom_from_variant(
            {"mode": "scale", "name": tag_amp, "tag": tag_amp, "value": 1.0}
        )

    # Build the argument atom: poly(axis, partner) with init_coeffs for omega*(axis - partner)
    _mt_arg = degree_arg if homogeneous else 0
    arg_kwargs = {"degree": int(degree_arg), "min_total": _mt_arg}
    if int(degree_arg) == 1:
        if compound_absorbs_trig:
            arg_kwargs["init_coeffs"] = [phi0, effective_omega]
        else:
            # Coefficients for: c0 + c1*x_axis + c2*x_partner
            # We want: phi0 + omega*x_axis - omega*x_partner
            arg_kwargs["init_coeffs"] = [phi0, omega, -omega]

    if compound_absorbs_trig:
        trig_expr = clone_ast(input_exprs[int(trig_local_idx)])
        arg_var_idxs = tuple(
            sorted(int(v) for v in _collect_var_idxs_from_node(trig_expr))
        ) or tuple(var_idxs)
        arg_inputs = (trig_expr,)
    else:
        arg_var_idxs = [axis, partner]
        arg_inputs = None

    arg_atom = AtomNode(
        kind="poly",
        var_idxs=arg_var_idxs,
        kwargs=arg_kwargs,
        tag=tag_arg,
        inputs=arg_inputs,
    )

    trig_node = CosNode(arg_atom)
    new_subtree = AddNode(off_atom, MulNode(amp_atom, trig_node))
    cand_root = _replace_node(root, target, new_subtree)

    def _custom_init(root_inner: Node, model_inner: torch.nn.Module):
        # Only init NN leaves (constant scale leaves don't need init here).
        if not other_vars:
            return

        try:
            atom_to_leaf = build_atom_to_leaf_map(root_inner, model_inner)
        except Exception:
            return

        def _leaf_param_device_dtype(mod: torch.nn.Module) -> Tuple[torch.device, torch.dtype]:
            for p in mod.parameters(recurse=True):
                if isinstance(p, torch.Tensor):
                    return p.device, p.dtype
            return device, dtype

        def _lm_fit_leaf_to_data(
            leaf_mod: torch.nn.Module,
            x_data: torch.Tensor,
            y_data: torch.Tensor,
            *,
            epochs: int = 60,
            chisq_tol: float = 1e-12,
        ) -> None:
            try:
                import nestynet
                from torch.utils.data import DataLoader, TensorDataset

                dev, dt = _leaf_param_device_dtype(leaf_mod)
                x_all = x_data.to(dev, dt)
                y_all = y_data.to(dev, dt).reshape(-1, 1)

                n = x_all.shape[0]
                if n < 64:
                    return

                perm = torch.randperm(n, device=x_all.device)
                n_use = min(n, 2048)
                n_train = int(n_use * 0.8)
                idx_train = perm[:n_train]
                idx_val = perm[n_train:n_use]

                x_train, y_train = x_all[idx_train], y_all[idx_train]
                x_val, y_val = x_all[idx_val], y_all[idx_val]

                dl_train = DataLoader(
                    TensorDataset(x_train, y_train),
                    batch_size=x_train.shape[0],
                    shuffle=False,
                )
                dl_val = DataLoader(
                    TensorDataset(x_val, y_val),
                    batch_size=x_val.shape[0],
                    shuffle=False,
                )

                def fac(dl):
                    def f(_):
                        return nestynet.optimizer.ResidualsModule(
                            providers=[leaf_mod],
                            dataloader=dl,
                            device=dev,
                        )

                    return f

                from nestynet_sr.sr_search.training import SR_LM_OVERRIDES

                cfg = nestynet.optimizer.LMConfig(
                    verbose=False,
                    LM_strategy="direct_solve",
                    chisq_tol=chisq_tol,
                    log_to_console=False,
                    **SR_LM_OVERRIDES,
                )
                lm_opt = nestynet.optimizer.Predictive_LM_Optimizer(
                    list(leaf_mod.parameters()),
                    [fac(dl_train)],
                    residual_module_factories_val=[fac(dl_val)],
                    cfg=cfg,
                )

                for _ in range(int(epochs)):
                    lm_opt.step()
                    if lm_opt.state.get("halt"):
                        break

            except Exception:
                return

        # Initialize the two NN leaves by tag
        for a in _collect_all_atoms(root_inner):
            if not isinstance(a, AtomNode) or str(a.kind).lower() != "nn":
                continue
            leaf_mod = atom_to_leaf.get(id(a))
            if leaf_mod is None:
                continue
            if getattr(a, "tag", None) == tag_off:
                _lm_fit_leaf_to_data(leaf_mod, X_other_cpu, A_cpu)
            elif getattr(a, "tag", None) == tag_amp:
                _lm_fit_leaf_to_data(leaf_mod, X_other_cpu, B_cpu)

    return cand_root, _custom_init


def _build_sqrt_ratpoly_candidate(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    deg_num: int = 2,
    deg_den: int = 2,
    min_points: int = 400,
    max_points: int = 5000,
    rel_rms_threshold: float = 1e-3,
    eps: float = 1e-8,
    enforce_units: bool = False,
    target_dim: Optional[tuple] = None,
    x_dims: Optional[List[tuple]] = None,
) -> Tuple[Optional[Node], Optional[callable]]:
    """
    Try to replace a multi-D NN atom by a sqrt(rational) or 1/sqrt(rational)
    in its own inputs:

        f(x_S) ≈ sqrt(P(x_S)/Q(x_S))        or
        f(x_S) ≈ 1/sqrt(P(x_S)/Q(x_S))

    where P/Q is a low-degree rational polynomial. We *detect* this by
    testing rational fits to f^2 and 1/f^2 via _rational_probe_nd, and
    if successful, build a PowNode(ratpoly, ±1/2). Coefficients are
    initialised conservatively and refined by LM.
    """
    # Import locally to avoid circular dependency
    from .stageB import _collect_all_atoms

    if target.kind.lower() != "nn":
        return None, None

    var_idxs = tuple(int(i) for i in target.var_idxs)
    dim = int(effective_arity(target))
    # Restrict to genuine multi-D leaves; 1D sqrt(poly) handles the 1D case.
    if dim < 2:
        return None, None
    if bool(enforce_units) and (
        target_dim is None
        or x_dims is None
        or len(x_dims) != dim
    ):
        return None, None

    # Adaptive degrees for multi-D: allow full cross-terms.
    # For dim=2 with individual degree 2, need total degree 4 in numerator.
    # E.g., f² = x1²*x2²/(x2² - x1²) requires deg_num=4 to represent x1²*x2².
    # For dim=3, denominator like x1⁴ - x1²*x2² also needs degree 4.
    effective_deg_num = max(deg_num, 2 * dim)
    effective_deg_den = max(deg_den, 2 * dim)

    tag = target.tag
    if tag is None or tag not in reuse:
        return None, None
    teacher = reuse[tag]

    data = _gather_atom_teacher_data(
        train_loader=train_loader,
        atom=target,
        teacher=teacher,
        device=device,
        dtype=dtype,
        max_points=max_points,
    )
    if data is None:
        return None, None
    X, F = data
    if X.numel() == 0 or F.numel() == 0:
        return None, None

    X = X.to(dtype=torch.float64)
    F = F.to(dtype=torch.float64).view(-1)
    N, dim2 = X.shape
    if N < min_points or dim2 != dim:
        return None, None

    # Restrict to a region where f has a consistent sign so that
    # sqrt and 1/sqrt are well-defined. If the leaf is everywhere
    # negative, we flip it and analyse |f|.
    mask, sign = _select_sign_region(F, min_points=min_points, eps=eps)
    if mask is None:
        return None, None
    X = X[mask]
    F = sign * F[mask]

    # Targets: P₁ ≈ f²,  P₂ ≈ 1/f²
    Y1 = F * F
    Y2 = 1.0 / torch.clamp(Y1, min=eps * eps)

    # Probe only structurally admissible supports when units are active.  The
    # square and reciprocal-square lifts have dimensions +2D and -2D.
    selected_support1 = None
    selected_support2 = None
    support_diagnostics: Dict[str, Any] = {}

    def _probe_supports(y_values, rational_target_dim, label):
        from .rational_supports import plan_unit_consistent_rational_supports

        plan = plan_unit_consistent_rational_supports(
            target_dim=rational_target_dim,
            input_dims=x_dims,
            max_deg_num=int(effective_deg_num),
            max_deg_den=int(effective_deg_den),
            coefficient_policy="free_const_only",
            max_attempts=2048,
        )
        support_diagnostics[label] = plan.diagnostics()
        best = None
        for support in plan.supports:
            try:
                result = _rational_probe_nd(
                    X,
                    y_values,
                    deg_num=int(support.degree_num),
                    deg_den=int(support.degree_den),
                    min_points=max(1, min_points // 2),
                    max_points=min(min_points, 1000),
                    dtype=torch.float64,
                    return_coeffs=True,
                    filter_outliers=True,
                    error_metric="median_rel",
                    exps_num_override=support.numerator_exponents,
                    exps_den_override=support.denominator_exponents,
                )
                rms_rel, coeffs_num, coeffs_den = result
            except Exception:
                continue
            if not math.isfinite(float(rms_rel)):
                continue
            key = (float(rms_rel), int(support.complexity))
            if best is None or key < best[0]:
                best = (key, support, coeffs_num, coeffs_den)
        if best is None:
            return float("inf"), None, None, None
        return float(best[0][0]), best[2], best[3], best[1]

    if bool(enforce_units):
        from fractions import Fraction
        from nestynet_sr.sr_core.units import scale_dim

        square_target_dim = scale_dim(tuple(target_dim), Fraction(2))
        inverse_square_target_dim = scale_dim(tuple(target_dim), Fraction(-2))
        rms_rel1, coeffs_num1, coeffs_den1, selected_support1 = _probe_supports(
            Y1, square_target_dim, "square"
        )
        rms_rel2, coeffs_num2, coeffs_den2, selected_support2 = _probe_supports(
            Y2, inverse_square_target_dim, "inverse_square"
        )
    else:
        def _probe_dense(y_values):
            try:
                result = _rational_probe_nd(
                    X,
                    y_values,
                    deg_num=effective_deg_num,
                    deg_den=effective_deg_den,
                    min_points=max(1, min_points // 2),
                    max_points=min(min_points, 1000),
                    dtype=torch.float64,
                    return_coeffs=True,
                    filter_outliers=True,
                    error_metric="median_rel",
                )
                return result
            except Exception:
                return float("inf"), None, None

        rms_rel1, coeffs_num1, coeffs_den1 = _probe_dense(Y1)
        rms_rel2, coeffs_num2, coeffs_den2 = _probe_dense(Y2)

    best_kind: Optional[str] = None
    y_target: Optional[torch.Tensor] = None
    fitted_coeffs_num: Optional[torch.Tensor] = None
    fitted_coeffs_den: Optional[torch.Tensor] = None
    selected_support = None

    if rms_rel1 < rms_rel2 and rms_rel1 < rel_rms_threshold:
        best_kind = "sqrt"
        y_target = Y1
        fitted_coeffs_num = coeffs_num1
        fitted_coeffs_den = coeffs_den1
        selected_support = selected_support1
    elif rms_rel2 < rel_rms_threshold:
        best_kind = "inv_sqrt"
        y_target = Y2
        fitted_coeffs_num = coeffs_num2
        fitted_coeffs_den = coeffs_den2
        selected_support = selected_support2

    # For inv_sqrt, we're fitting P/Q = 1/f² which has swapped degree requirements
    # compared to sqrt where P/Q = f². Swap the degrees for inv_sqrt.
    if best_kind == "inv_sqrt" and not bool(enforce_units):
        effective_deg_num, effective_deg_den = effective_deg_den, effective_deg_num

    if best_kind is None or y_target is None:
        print(
            f"[Stage B] sqrt_ratpoly candidate rejected: rms_rel(f^2)={rms_rel1:.2%}, rms_rel(1/f^2)={rms_rel2:.2%}, "
            f"threshold={rel_rms_threshold:.1%}, vars={var_idxs}"
        )
        return None, None

    exponent = 0.5 if best_kind == "sqrt" else -0.5
    if selected_support is not None:
        exps_num_dense = torch.tensor(
            selected_support.numerator_exponents,
            dtype=torch.int64,
        )
        exps_den_dense = torch.tensor(
            selected_support.denominator_exponents,
            dtype=torch.int64,
        )
    else:
        exps_num_dense = torch.tensor(
            _enumerate_exponents(dim, effective_deg_num),
            dtype=torch.int64,
        )
        exps_den_dense = torch.tensor(
            _enumerate_exponents(dim, effective_deg_den),
            dtype=torch.int64,
        )
    sparse_fit = _fit_rational_coeffs_nd(
        X,
        y_target,
        exps_num=exps_num_dense.to(device=X.device),
        exps_den=exps_den_dense.to(device=X.device),
        min_points=max(1, min_points // 2),
        return_support_indices=True,
    )
    exps_num_selected = (
        exps_num_dense[sparse_fit[2].detach().cpu()].detach().cpu().clone()
        if sparse_fit is not None and _support_is_valid(sparse_fit[2].detach().cpu(), exps_num_dense)
        else exps_num_dense.detach().cpu().clone()
    )
    exps_den_selected = (
        exps_den_dense[sparse_fit[3].detach().cpu()].detach().cpu().clone()
        if sparse_fit is not None and _support_is_valid(sparse_fit[3].detach().cpu(), exps_den_dense)
        else exps_den_dense.detach().cpu().clone()
    )
    model_deg_num = _max_total_degree_from_exps(exps_num_selected, fallback=effective_deg_num)
    model_deg_den = _max_total_degree_from_exps(exps_den_selected, fallback=effective_deg_den)

    coefficient_unit_certificate = None
    if bool(enforce_units):
        from fractions import Fraction
        from nestynet_sr.sr_core.coefficient_units import (
            solve_rational_coefficient_gauge,
        )
        from nestynet_sr.sr_core.units import scale_dim

        rational_target_dim = scale_dim(
            tuple(target_dim),
            Fraction(2 if best_kind == "sqrt" else -2),
        )
        final_solution = solve_rational_coefficient_gauge(
            target_dim=rational_target_dim,
            input_dims=x_dims,
            numerator_exponents=exps_num_selected.tolist(),
            denominator_exponents=exps_den_selected.tolist(),
            coefficient_policy="free_const_only",
        )
        if not final_solution.ok:
            print(
                "[Stage B] sqrt-ratpoly final support rejected by coefficient-unit solver: "
                f"{final_solution.code}: {final_solution.reason}, vars={var_idxs}"
            )
            return None, None
        coefficient_unit_certificate = final_solution.to_dict()

    # Build AST: Pow( ratpoly(x_S), exponent )
    _kw: Dict[str, Any] = {"deg_num": int(model_deg_num), "deg_den": int(model_deg_den)}
    exps_num_override = _exps_override_from_tensor(exps_num_selected)
    exps_den_override = _exps_override_from_tensor(exps_den_selected)
    _kw["exps_num_override"] = exps_num_override
    _kw["exps_den_override"] = exps_den_override
    rat_atom = AtomNode(
        kind="ratpoly",
        var_idxs=target.var_idxs,
        kwargs=_kw,
        tag=None,
        inputs=clone_inputs(target),
    )
    new_subtree = PowNode(base=rat_atom, exponent=float(exponent))
    cand_root = _replace_node(root, target, new_subtree)

    # Cache target values and fitted coefficients for initialization.
    y_target_cpu = y_target.detach().cpu()
    target_var_idxs = var_idxs
    target_inputs = get_input_exprs(target)
    fitted_num_cpu = sparse_fit[0].detach().cpu().clone() if sparse_fit is not None else (
        fitted_coeffs_num.detach().cpu() if fitted_coeffs_num is not None else None
    )
    fitted_den_cpu = sparse_fit[1].detach().cpu().clone() if sparse_fit is not None else (
        fitted_coeffs_den.detach().cpu() if fitted_coeffs_den is not None else None
    )
    exps_num_key = _exps_key(exps_num_selected)
    exps_den_key = _exps_key(exps_den_selected)

    def _custom_init(root_inner: Node, model_inner: torch.nn.Module):
        """
        Initialize sqrt-rational coefficients using the fitted probe coefficients
        when available, falling back to conservative initialization otherwise.

            f(x_S) ≈ (P(x_S)/Q(x_S))^{±1/2}
        """
        atoms = _collect_all_atoms(root_inner)
        leaves = list(model_inner.leaf)
        rat_core = _find_matching_core(
            atoms,
            leaves,
            core_types=RationalPolyLeaf,
            expected_kind="ratpoly",
            expected_inputs=target_inputs,
            predicate=lambda _atom, core: (
                int(getattr(core, "deg_num", -1)) == model_deg_num
                and int(getattr(core, "deg_den", -1)) == model_deg_den
                and (
                    exps_num_key is None
                    or exps_den_key is None
                    or (
                        _exps_key(core.exps_num.detach().cpu()) == exps_num_key
                        and _exps_key(core.exps_den.detach().cpu()) == exps_den_key
                    )
                )
            ),
        )

        if rat_core is None:
            print(
                "[Stage B custom_init sqrt-ratpoly] No RationalPolyLeaf found for vars",
                target_var_idxs,
            )
            return

        dev = rat_core.coeffs_num.device
        dt = rat_core.coeffs_num.dtype

        # Try to use fitted coefficients from the probe if available and shapes match
        if (fitted_num_cpu is not None and fitted_den_cpu is not None and
            fitted_num_cpu.shape[0] == rat_core.coeffs_num.shape[0] and
            fitted_den_cpu.shape[0] == rat_core.coeffs_den.shape[0]):
            with torch.no_grad():
                rat_core.coeffs_num.copy_(fitted_num_cpu.to(dtype=dt, device=dev))
                rat_core.coeffs_den.copy_(fitted_den_cpu.to(dtype=dt, device=dev))
            print(
                f"[Stage B custom_init sqrt-ratpoly] Using probe coefficients for vars {target_var_idxs}: "
                f"kind={best_kind}"
            )
            return

        # Fallback: conservative initialization (constant P, Q≈1)
        exps_num = rat_core.exps_num.detach().cpu()
        exps_den = rat_core.exps_den.detach().cpu()
        idx_num_const = None
        idx_den_const = None
        for k, e in enumerate(exps_num):
            if int(e.sum().item()) == 0:
                idx_num_const = k
                break
        for k, e in enumerate(exps_den):
            if int(e.sum().item()) == 0:
                idx_den_const = k
                break

        # Target mean value for P(x_S)
        target_mean = float(y_target_cpu.mean().item())
        if not math.isfinite(target_mean) or target_mean <= 0.0:
            target_mean = 1.0

        with torch.no_grad():
            rat_core.coeffs_num.zero_()
            rat_core.coeffs_den.zero_()

            # Denominator: Q(x) ≈ 1
            if idx_den_const is not None:
                rat_core.coeffs_den[idx_den_const] = 1.0
            elif rat_core.coeffs_den.numel() > 0:
                rat_core.coeffs_den[0] = 1.0
            if rat_core.coeffs_den.numel() > 1:
                rat_core.coeffs_den[1:].normal_(mean=0.0, std=0.01)

            # Numerator: P(x) ≈ mean(y_target)
            if idx_num_const is not None:
                rat_core.coeffs_num[idx_num_const] = torch.as_tensor(
                    target_mean, dtype=dt, device=dev
                )
            elif rat_core.coeffs_num.numel() > 0:
                rat_core.coeffs_num[0] = torch.as_tensor(target_mean, dtype=dt, device=dev)
            if rat_core.coeffs_num.numel() > 1:
                rat_core.coeffs_num[1:].normal_(mean=0.0, std=0.01)

        print(
            f"[Stage B custom_init sqrt-ratpoly] Fallback init on vars {target_var_idxs}: "
            f"kind={best_kind}, mean_P≈{target_mean:.3g}"
        )

    _best_rms = float(rms_rel1 if best_kind == "sqrt" else rms_rel2)
    # Hint for numerically-stable fitting: prefit in a lifted space first.
    # sqrt(u)  -> fit u via square-link; 1/sqrt(u) -> fit u via inv_square-link.
    if best_kind == "sqrt":
        _custom_init._fit_lift_link = "square"
    else:
        _custom_init._fit_lift_link = "inv_square"

    meta: Dict[str, Any] = {"probe_rms_rel": _best_rms}
    if coefficient_unit_certificate is not None:
        meta.update({
            "unit_support_planned": True,
            "coefficient_unit_certificate": coefficient_unit_certificate,
            "unit_support_diagnostics": support_diagnostics,
        })
    return cand_root, _custom_init, meta


def _build_log_ratpoly_candidate(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    deg_num: int = 1,
    deg_den: int = 1,
    min_points: int = 400,
    max_points: int = 5000,
    max_abs_exponent: float = 20.0,
    rel_rms_threshold: float = 1e-3,
    enforce_units: bool = False,
    target_dim: Optional[tuple] = None,
    x_dims: Optional[List[tuple]] = None,
) -> Tuple[Optional[Node], Optional[callable]]:
    if target.kind.lower() != "nn":
        return None, None
    var_idxs = tuple(int(i) for i in target.var_idxs)
    tag = target.tag
    if tag is None or tag not in reuse:
        return None, None
    teacher = reuse[tag]
    data = _gather_atom_teacher_data(
        train_loader=train_loader,
        atom=target,
        teacher=teacher,
        device=device,
        dtype=dtype,
        max_points=max_points,
    )
    if data is None:
        return None, None
    X, F = data
    if X.numel() == 0 or F.numel() == 0:
        return None, None
    X = X.to(dtype=torch.float64)
    F = F.to(dtype=torch.float64).view(-1)
    N, dim = X.shape
    # Adaptive degrees for multi-D: need higher degrees for cross-terms
    # E.g., x4^4 / (x4^2 - x5^2)^2 requires deg_num=4, deg_den=4 for dim=2
    effective_deg_num = max(deg_num, 2 * dim) if dim >= 2 else deg_num
    effective_deg_den = max(deg_den, 2 * dim) if dim >= 2 else deg_den
    if bool(enforce_units) and (
        target_dim is None
        or x_dims is None
        or len(x_dims) != dim
        or any(value != 0 for value in target_dim)
    ):
        # A logarithm's output and argument must both be dimensionless.
        return None, None
    if N < min_points:
        return None, None
    max_abs = float(F.abs().max().item())
    if (not math.isfinite(max_abs)) or max_abs > max_abs_exponent:
        return None, None
    g = torch.exp(F)
    if not torch.isfinite(g).all():
        return None, None
    planned_support = None
    if bool(enforce_units):
        from .rational_supports import plan_unit_consistent_rational_supports

        support_plan = plan_unit_consistent_rational_supports(
            target_dim=target_dim,
            input_dims=x_dims,
            max_deg_num=int(effective_deg_num),
            max_deg_den=int(effective_deg_den),
            coefficient_policy="free_const_only",
            max_attempts=2048,
        )
        best_probe = None
        for support in support_plan.supports:
            try:
                error = _rational_probe_nd(
                    X,
                    g,
                    deg_num=int(support.degree_num),
                    deg_den=int(support.degree_den),
                    min_points=min_points,
                    max_points=min(min_points, 1000),
                    dtype=torch.float64,
                    filter_outliers=True,
                    error_metric="median_rel",
                    exps_num_override=support.numerator_exponents,
                    exps_den_override=support.denominator_exponents,
                )
            except Exception:
                continue
            if not math.isfinite(float(error)):
                continue
            key = (float(error), int(support.complexity))
            if best_probe is None or key < best_probe[0]:
                best_probe = (key, support)
        if best_probe is None:
            return None, None
        rms_rel = float(best_probe[0][0])
        planned_support = best_probe[1]
    else:
        try:
            rms_rel = _rational_probe_nd(
                X,
                g,
                deg_num=effective_deg_num,
                deg_den=effective_deg_den,
                min_points=min_points,
                max_points=min(min_points, 1000),
                dtype=torch.float64,
                filter_outliers=True,
                error_metric="median_rel",
            )
        except Exception:
            return None, None
    if (not math.isfinite(rms_rel)) or rms_rel > rel_rms_threshold:
        print(
            f"[Stage B] log_ratpoly candidate rejected: rms_rel(exp(f))={rms_rel:.2%}, "
            f"threshold={rel_rms_threshold:.1%}, vars={var_idxs}, deg={effective_deg_num}/{effective_deg_den}"
        )
        return None, None
    target_var_idxs = var_idxs
    target_inputs = get_input_exprs(target)
    y_target_cpu = g.detach().cpu()
    if planned_support is not None:
        exps_num_dense = torch.tensor(
            planned_support.numerator_exponents,
            dtype=torch.int64,
        )
        exps_den_dense = torch.tensor(
            planned_support.denominator_exponents,
            dtype=torch.int64,
        )
    else:
        exps_num_dense = torch.tensor(
            _enumerate_exponents(dim, effective_deg_num),
            dtype=torch.int64,
        )
        exps_den_dense = torch.tensor(
            _enumerate_exponents(dim, effective_deg_den),
            dtype=torch.int64,
        )
    sparse_fit = _fit_rational_coeffs_nd(
        X,
        g,
        exps_num=exps_num_dense.to(device=X.device),
        exps_den=exps_den_dense.to(device=X.device),
        min_points=min_points,
        return_support_indices=True,
    )
    exps_num_selected = (
        exps_num_dense[sparse_fit[2].detach().cpu()].detach().cpu().clone()
        if sparse_fit is not None and _support_is_valid(sparse_fit[2].detach().cpu(), exps_num_dense)
        else exps_num_dense.detach().cpu().clone()
    )
    exps_den_selected = (
        exps_den_dense[sparse_fit[3].detach().cpu()].detach().cpu().clone()
        if sparse_fit is not None and _support_is_valid(sparse_fit[3].detach().cpu(), exps_den_dense)
        else exps_den_dense.detach().cpu().clone()
    )
    model_deg_num = _max_total_degree_from_exps(exps_num_selected, fallback=effective_deg_num)
    model_deg_den = _max_total_degree_from_exps(exps_den_selected, fallback=effective_deg_den)
    if bool(enforce_units):
        from nestynet_sr.sr_core.coefficient_units import (
            solve_rational_coefficient_gauge,
        )

        final_solution = solve_rational_coefficient_gauge(
            target_dim=target_dim,
            input_dims=x_dims,
            numerator_exponents=exps_num_selected.tolist(),
            denominator_exponents=exps_den_selected.tolist(),
            coefficient_policy="free_const_only",
        )
        if not final_solution.ok:
            print(
                "[Stage B] log-ratpoly final support rejected by coefficient-unit solver: "
                f"{final_solution.code}: {final_solution.reason}, vars={var_idxs}"
            )
            return None, None
    _kw_lr: Dict[str, Any] = {"deg_num": int(model_deg_num), "deg_den": int(model_deg_den)}
    exps_num_override = _exps_override_from_tensor(exps_num_selected)
    exps_den_override = _exps_override_from_tensor(exps_den_selected)
    _kw_lr["exps_num_override"] = exps_num_override
    _kw_lr["exps_den_override"] = exps_den_override
    rat_atom = AtomNode(
        kind="ratpoly",
        var_idxs=target.var_idxs,
        kwargs=_kw_lr,
        tag=None,
        inputs=clone_inputs(target),
    )
    new_subtree = LogNode(arg=rat_atom)
    cand_root = _replace_node(root, target, new_subtree)
    fitted_num_cpu = sparse_fit[0].detach().cpu().clone() if sparse_fit is not None else None
    fitted_den_cpu = sparse_fit[1].detach().cpu().clone() if sparse_fit is not None else None
    exps_num_key = _exps_key(exps_num_selected)
    exps_den_key = _exps_key(exps_den_selected)

    def _custom_init(root_inner, model_inner):
        from .stageB import _collect_all_atoms

        atoms = _collect_all_atoms(root_inner)
        leaves = list(model_inner.leaf)
        rat_core = _find_matching_core(
            atoms,
            leaves,
            core_types=RationalPolyLeaf,
            expected_kind="ratpoly",
            expected_inputs=target_inputs,
            predicate=lambda _atom, core: (
                int(getattr(core, "deg_num", -1)) == model_deg_num
                and int(getattr(core, "deg_den", -1)) == model_deg_den
                and (
                    exps_num_key is None
                    or exps_den_key is None
                    or (
                        _exps_key(core.exps_num.detach().cpu()) == exps_num_key
                        and _exps_key(core.exps_den.detach().cpu()) == exps_den_key
                    )
                )
            ),
        )
        if rat_core is None:
            print(
                "[Stage B custom_init log-ratpoly] No RationalPolyLeaf found for vars",
                target_var_idxs,
            )
            return
        if (
            fitted_num_cpu is not None
            and fitted_den_cpu is not None
            and fitted_num_cpu.numel() == rat_core.coeffs_num.numel()
            and fitted_den_cpu.numel() == rat_core.coeffs_den.numel()
        ):
            with torch.no_grad():
                rat_core.coeffs_num.copy_(
                    fitted_num_cpu.to(dtype=rat_core.coeffs_num.dtype, device=rat_core.coeffs_num.device)
                )
                rat_core.coeffs_den.copy_(
                    fitted_den_cpu.to(dtype=rat_core.coeffs_den.dtype, device=rat_core.coeffs_den.device)
                )
            return
        exps_num = rat_core.exps_num.detach().cpu()
        exps_den = rat_core.exps_den.detach().cpu()
        idx_num_const = None
        idx_den_const = None
        for k, e in enumerate(exps_num):
            if int(e.sum().item()) == 0:
                idx_num_const = k
                break
        for k, e in enumerate(exps_den):
            if int(e.sum().item()) == 0:
                idx_den_const = k
                break
        dev = rat_core.coeffs_num.device
        dt = rat_core.coeffs_num.dtype
        target_mean = float(y_target_cpu.mean().item())
        if (not math.isfinite(target_mean)) or target_mean <= 0.0:
            target_mean = 1.0
        with torch.no_grad():
            rat_core.coeffs_num.zero_()
            rat_core.coeffs_den.zero_()
            if idx_den_const is not None:
                rat_core.coeffs_den[idx_den_const] = 1.0
            elif rat_core.coeffs_den.numel() > 0:
                rat_core.coeffs_den[0] = 1.0
            if rat_core.coeffs_den.numel() > 1:
                rat_core.coeffs_den[1:].normal_(mean=0.0, std=0.01)
            if idx_num_const is not None:
                rat_core.coeffs_num[idx_num_const] = torch.as_tensor(
                    target_mean, dtype=dt, device=dev
                )
            elif rat_core.coeffs_num.numel() > 0:
                rat_core.coeffs_num[0] = torch.as_tensor(target_mean, dtype=dt, device=dev)
            if rat_core.coeffs_num.numel() > 1:
                rat_core.coeffs_num[1:].normal_(mean=0.0, std=0.01)
        print(
            f"[Stage B custom_init log-ratpoly] Init on vars {target_var_idxs}: mean_P≈{target_mean:.3g}"
        )

    # Hint for numerically-stable fitting: log(u) -> fit u via exp-link first.
    _custom_init._fit_lift_link = "exp"

    return cand_root, _custom_init


def _build_ratpoly_candidates(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    max_deg_num: int = 4,
    max_deg_den: int = 4,
    min_points: int = 400,
    max_points: int = 5000,
    fit_frac: float = 0.7,
    rel_rms_threshold: float = 2e-2,
    max_terms_total: int = 180,
    eps_Q: float = 1e-10,
    enforce_units: bool = False,
    target_dim: Optional[tuple] = None,
    x_dims: Optional[List[tuple]] = None,
) -> List[Tuple[Node, callable, Dict[str, Any]]]:
    """Try to replace a (multi-D) NN atom by rational polynomial leaves.

    Returns a list of ``(root, custom_init, meta)`` tuples ordered by
    complexity (simplest first).  Each tuple corresponds to a different
    degree pair that passes the probe threshold.  The caller can feed
    these into the Stage B engine so that if the simplest candidate is
    rejected (e.g. by the asinh y-space sanity check), higher-degree
    candidates get a chance.

    When no degree pair passes the threshold, returns an empty list.
    """

    def _stable_target_sig_nd(_target: AtomNode) -> int:
        import zlib

        payload = (
            str(getattr(_target, "tag", "") or ""),
            tuple(int(i) for i in _target.var_idxs),
            tuple(repr(inp) for inp in get_input_exprs(_target)),
        )
        return int(zlib.crc32(repr(payload).encode("utf-8")) & 0xFFFFFFFF)

    def _support_signature_nd(
        *,
        target_sig: int,
        leaf_kind: str,
        exps_num: Optional[torch.Tensor],
        exps_den: Optional[torch.Tensor],
    ) -> Tuple[int, ...]:
        def _flat(exps_t: Optional[torch.Tensor]) -> Tuple[int, ...]:
            if exps_t is None or exps_t.ndim != 2:
                return (0, 0)
            rows = int(exps_t.shape[0])
            dim_local = int(exps_t.shape[1])
            return (rows, dim_local, *(int(v) for v in exps_t.reshape(-1).tolist()))

        kind_code = 1 if str(leaf_kind).lower() == "rratpoly" else 0
        return (int(target_sig), int(kind_code), *_flat(exps_num), -1, *_flat(exps_den))

    if target.kind.lower() != "nn":
        return []

    var_idxs = tuple(int(i) for i in target.var_idxs)
    dim = int(effective_arity(target))
    if dim < 1:
        return []
    if bool(enforce_units) and (
        target_dim is None
        or x_dims is None
        or len(x_dims) != dim
    ):
        return []

    tag = target.tag
    if tag is None or tag not in reuse:
        return []
    teacher = reuse[tag]

    data = _gather_atom_teacher_data(
        train_loader=train_loader,
        atom=target,
        teacher=teacher,
        device=device,
        dtype=dtype,
        max_points=max_points,
    )
    if data is None:
        return []
    X, F = data
    if X.numel() == 0 or F.numel() == 0:
        return []

    X = X.to(dtype=torch.float64)
    F = F.to(dtype=torch.float64).view(-1)
    N, dim2 = X.shape
    if N < min_points or dim2 != dim:
        return []

    def _n_terms_poly(deg: int, min_total: int = 0) -> int:
        # Count monomials with min_total <= total degree <= deg in `dim` variables.
        total = 0
        for d in range(min_total, deg + 1):
            total += int(math.comb(d + dim - 1, dim - 1))
        return total

    # ── Build list of degree/support configurations to try ──
    # Unit-aware runs construct exact monomial supports before any numerical
    # probe.  Unitless dimensions use this same planner: zero is just another
    # exact dimension vector, not a separate algorithm.
    degree_trials: List[Dict[str, Any]] = []
    support_plan_diagnostics: Optional[Dict[str, Any]] = None

    if bool(enforce_units):
        from .rational_supports import plan_unit_consistent_rational_supports

        support_plan = plan_unit_consistent_rational_supports(
            target_dim=target_dim,
            input_dims=x_dims,
            # Preserve the former dimensional degree probe's bounded
            # auto-raise: caller defaults are screening defaults, while units
            # may prove that the first viable support begins above them.
            max_deg_num=max(int(max_deg_num), 8),
            max_deg_den=max(int(max_deg_den), 8),
            coefficient_policy="free_const_only",
            max_attempts=2048,
        )
        support_plan_diagnostics = support_plan.diagnostics()
        for support in support_plan.supports:
            if int(support.complexity) > int(max_terms_total):
                continue
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
                "[Stage B] ratpoly support planner found no admissible support "
                f"for vars={var_idxs}: {support_plan_diagnostics}"
            )
            return []
    else:
        # No unit payload: preserve the historical dense numerical search.
        for deg_den in range(1, int(max_deg_den) + 1):
            for deg_num in range(1, int(max_deg_num) + 1):
                mt_num = 0
                mt_den = 0
                complexity = _n_terms_poly(deg_num, mt_num) + _n_terms_poly(deg_den, mt_den)
                if complexity > int(max_terms_total):
                    continue
                degree_trials.append({
                    "complexity": int(complexity),
                    "deg_num": int(deg_num),
                    "deg_den": int(deg_den),
                    "mt_num": int(mt_num),
                    "mt_den": int(mt_den),
                    "exps_num": None,
                    "exps_den": None,
                })

    # Sort by (complexity, deg_num, deg_den) — simplest first.
    degree_trials.sort(key=lambda t: (t["complexity"], t["deg_num"], t["deg_den"]))

    # ── Probe all degree trials and collect sub-threshold hits ──
    accepted_trials: List[Dict[str, Any]] = []
    best_reject = None  # best trial that didn't pass threshold (diagnostics)
    n_tried = 0
    n_exceptions = 0
    n_nonfinite = 0
    for trial in degree_trials:
        n_tried += 1
        try:
            rms_rel = _rational_probe_nd(
                X,
                F,
                deg_num=int(trial["deg_num"]),
                deg_den=int(trial["deg_den"]),
                min_points=min_points,
                max_points=min(1000, max(min_points, 1000)),
                dtype=torch.float64,
                fit_frac=float(fit_frac),
                eps_Q=float(eps_Q),
                seed=0,
                filter_outliers=True,
                error_metric="median_rel",
                min_total_num=int(trial["mt_num"]),
                min_total_den=int(trial["mt_den"]),
                exps_num_override=trial["exps_num"],
                exps_den_override=trial["exps_den"],
            )
        except Exception:
            n_exceptions += 1
            continue
        if not math.isfinite(rms_rel):
            n_nonfinite += 1
            continue
        cand = dict(trial)
        cand["rms_rel"] = float(rms_rel)
        if rms_rel <= float(rel_rms_threshold):
            accepted_trials.append(cand)
        elif best_reject is None or cand["rms_rel"] < best_reject["rms_rel"]:
            best_reject = cand

    if not accepted_trials:
        if best_reject is not None:
            print(
                f"[Stage B] ratpoly candidate rejected: best_rms_rel={best_reject['rms_rel']:.2%} "
                f"> threshold={rel_rms_threshold:.1%}, best_deg=({best_reject['deg_num']}, "
                f"{best_reject['deg_den']}), n_terms={best_reject['complexity']}, vars={var_idxs}"
            )
            return []
        print(
            f"[Stage B] ratpoly candidate failed: no valid fit found after {n_tried} attempts "
            f"(exceptions={n_exceptions}, non-finite={n_nonfinite}), vars={var_idxs}"
        )
        return []

    # Deduplicate identical screening supports.  Distinct dimensional classes
    # can share the same degree pair and must each retain a chance to fit.
    _best_per_deg: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for t in accepted_trials:
        key = (
            int(t["deg_num"]),
            int(t["deg_den"]),
            tuple(tuple(int(v) for v in row) for row in (t.get("exps_num") or ())),
            tuple(tuple(int(v) for v in row) for row in (t.get("exps_den") or ())),
        )
        if key not in _best_per_deg or t["rms_rel"] < _best_per_deg[key]["rms_rel"]:
            _best_per_deg[key] = t

    enriched_trials: List[Dict[str, Any]] = []
    for trial in _best_per_deg.values():
        t_deg_num = int(trial["deg_num"])
        t_deg_den = int(trial["deg_den"])
        t_mt_num = int(trial["mt_num"])
        t_mt_den = int(trial["mt_den"])

        if trial.get("exps_num") is not None and trial.get("exps_den") is not None:
            exps_num_dense = torch.tensor(trial["exps_num"], dtype=torch.int64)
            exps_den_dense = torch.tensor(trial["exps_den"], dtype=torch.int64)
        else:
            exps_num_dense = torch.tensor(
                _enumerate_exponents(dim, t_deg_num, min_total=t_mt_num),
                dtype=torch.int64,
            )
            exps_den_dense = torch.tensor(
                _enumerate_exponents(dim, t_deg_den, min_total=t_mt_den),
                dtype=torch.int64,
            )

        sparse_fit = _fit_rational_coeffs_nd(
            X,
            F,
            exps_num=exps_num_dense.to(device=X.device),
            exps_den=exps_den_dense.to(device=X.device),
            min_points=min_points,
            subsample_frac=1.0,
            eps_Q=float(eps_Q),
            seed=0,
            return_support_indices=True,
        )

        fit_a_cpu = None
        fit_b_cpu = None
        # Even a no-units run must emit explicit supports now that the span
        # fallback is gone.  Sparsification may replace these dense supports
        # below, but it may never erase them.
        exps_num_final: Optional[torch.Tensor] = exps_num_dense.detach().cpu().clone()
        exps_den_final: Optional[torch.Tensor] = exps_den_dense.detach().cpu().clone()
        deg_num_eff = _max_total_degree_from_exps(exps_num_final, fallback=t_deg_num)
        deg_den_eff = _max_total_degree_from_exps(exps_den_final, fallback=t_deg_den)
        n_terms_num = int(exps_num_final.shape[0])
        n_terms_den = int(exps_den_final.shape[0])
        candidate_kind = "ratpoly"
        pivot_reason = None

        if sparse_fit is not None:
            support_num = sparse_fit[2].detach().cpu()
            support_den = sparse_fit[3].detach().cpu()
            valid_num = _support_is_valid(support_num, exps_num_dense)
            valid_den = _support_is_valid(support_den, exps_den_dense)

            if valid_num and valid_den:
                fit_a_cpu = sparse_fit[0].detach().cpu().clone()
                fit_b_cpu = sparse_fit[1].detach().cpu().clone()
                exps_num_final = exps_num_dense[support_num].detach().cpu().clone()
                deg_num_eff = _max_total_degree_from_exps(exps_num_final, fallback=t_deg_num)
                n_terms_num = int(exps_num_final.shape[0])
                exps_den_final = exps_den_dense[support_den].detach().cpu().clone()
                deg_den_eff = _max_total_degree_from_exps(exps_den_final, fallback=t_deg_den)
                n_terms_den = int(exps_den_final.shape[0])

            if fit_a_cpu is not None and exps_num_final is not None and dim > 1:
                pivot_idx, pivot_reason = _select_clear_rratpoly_pivot(exps_num_final, fit_a_cpu)
                if pivot_idx is not None:
                    exps_num_final, fit_a_cpu = _move_sparse_pivot_to_end(
                        exps_num_final,
                        fit_a_cpu,
                        int(pivot_idx),
                    )
                    candidate_kind = "rratpoly"
        coefficient_unit_certificate = None
        if bool(enforce_units):
            from nestynet_sr.sr_core.coefficient_units import (
                solve_rational_coefficient_gauge,
            )

            final_solution = solve_rational_coefficient_gauge(
                target_dim=target_dim,
                input_dims=x_dims,
                numerator_exponents=exps_num_final.tolist(),
                denominator_exponents=exps_den_final.tolist(),
                numerator_pivot=(
                    int(exps_num_final.shape[0]) - 1
                    if candidate_kind == "rratpoly"
                    else None
                ),
                coefficient_policy="free_const_only",
            )
            if not final_solution.ok:
                print(
                    "[Stage B] ratpoly final support rejected by coefficient-unit solver: "
                    f"{final_solution.code}: {final_solution.reason}, vars={var_idxs}"
                )
                continue
            coefficient_unit_certificate = final_solution.to_dict()

        trial_enriched = dict(trial)
        trial_enriched["candidate_kind"] = candidate_kind
        trial_enriched["deg_num_eff"] = int(deg_num_eff)
        trial_enriched["deg_den_eff"] = int(deg_den_eff)
        trial_enriched["n_terms_num"] = int(n_terms_num)
        trial_enriched["n_terms_den"] = int(
            n_terms_den if n_terms_den > 0 else int(exps_den_final.shape[0]) if exps_den_final is not None else 0
        )
        trial_enriched["fit_a_cpu"] = fit_a_cpu
        trial_enriched["fit_b_cpu"] = fit_b_cpu
        trial_enriched["exps_num_final"] = exps_num_final
        trial_enriched["exps_den_final"] = exps_den_final
        trial_enriched["pivot_reason"] = pivot_reason
        trial_enriched["coefficient_unit_certificate"] = coefficient_unit_certificate
        enriched_trials.append(trial_enriched)

    # Deduplicate by final support signature after sparsification/reduction.
    _best_per_signature: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for t in enriched_trials:
        key = (
            str(t["candidate_kind"]),
            int(t["deg_num_eff"]),
            int(t["deg_den_eff"]),
            _exps_key(t.get("exps_num_final")),
            _exps_key(t.get("exps_den_final")),
        )
        if key not in _best_per_signature or float(t["rms_rel"]) < float(_best_per_signature[key]["rms_rel"]):
            _best_per_signature[key] = t

    accepted_trials = sorted(
        _best_per_signature.values(),
        key=lambda t: (
            int(t["n_terms_num"] + t["n_terms_den"]),
            int(t["deg_num_eff"] + t["deg_den_eff"]),
            int(t["deg_num_eff"]),
            int(t["deg_den_eff"]),
            float(t["rms_rel"]),
            int(t["complexity"]),
            int(t["deg_num"]),
            int(t["deg_den"]),
        ),
    )

    print(
        f"[Stage B] ratpoly probe: {len(accepted_trials)} sub-threshold trial(s) for vars={var_idxs}: "
        + ", ".join(
            f"{t['candidate_kind']} deg=({t['deg_num']},{t['deg_den']})->({t['deg_num_eff']},{t['deg_den_eff']}) "
            f"nnz=({t['n_terms_num']},{t['n_terms_den']}) rms={t['rms_rel']:.4f}"
            for t in accepted_trials
        )
    )

    # ── Build candidate ASTs for each accepted trial ──
    X_cpu = X.detach().cpu()
    F_cpu = F.detach().cpu()
    target_var_idxs = var_idxs
    target_inputs = get_input_exprs(target)
    target_sig = _stable_target_sig_nd(target)

    results: List[Tuple[Node, callable, Dict[str, Any]]] = []
    for trial_i in accepted_trials:
        t_kind = str(trial_i["candidate_kind"])
        t_deg_num_screen = int(trial_i["deg_num"])
        t_deg_den_screen = int(trial_i["deg_den"])
        t_deg_num = int(trial_i["deg_num_eff"])
        t_deg_den = int(trial_i["deg_den_eff"])
        t_mt_num = int(trial_i["mt_num"])
        t_mt_den = int(trial_i["mt_den"])
        t_rms = float(trial_i["rms_rel"])
        t_n_terms_num = int(trial_i["n_terms_num"])
        t_n_terms_den = int(trial_i["n_terms_den"])
        t_exps_num_t = trial_i.get("exps_num_final")
        t_exps_den_t = trial_i.get("exps_den_final")
        t_fit_a_cpu = trial_i.get("fit_a_cpu")
        t_fit_b_cpu = trial_i.get("fit_b_cpu")
        t_pivot_reason = trial_i.get("pivot_reason")
        t_coefficient_unit_certificate = trial_i.get("coefficient_unit_certificate")

        t_exps_num_key = _exps_key(t_exps_num_t)
        t_exps_den_key = _exps_key(t_exps_den_t)

        _kw: Dict[str, Any] = {"deg_num": t_deg_num, "deg_den": t_deg_den}
        if t_mt_num > 0:
            _kw["min_total_num"] = t_mt_num
        if t_mt_den > 0:
            _kw["min_total_den"] = t_mt_den
        exps_num_override = _exps_override_from_tensor(t_exps_num_t)
        exps_den_override = _exps_override_from_tensor(t_exps_den_t)
        if exps_num_override is not None:
            _kw["exps_num_override"] = exps_num_override
        if exps_den_override is not None:
            _kw["exps_den_override"] = exps_den_override

        scale_tag = None
        if t_kind == "rratpoly":
            scale_tag = (
                f"ratpolynd_scale_{str(target.tag) if target.tag is not None else 'anon'}"
                f"_v{'_'.join(str(int(i)) for i in target.var_idxs)}"
                f"_sn{t_deg_num_screen}_sd{t_deg_den_screen}_n{t_deg_num}_d{t_deg_den}"
            )
            scale_atom = AtomNode(
                kind="scale",
                var_idxs=(),
                kwargs={"init": 1.0, "name": "s"},
                tag=scale_tag,
            )
            _kw["_mul_scale_tag"] = scale_tag
            rat_atom = AtomNode(
                kind="rratpoly",
                var_idxs=target.var_idxs,
                kwargs=_kw,
                tag=target.tag,
                inputs=clone_inputs(target),
            )
            cand_root = _replace_node(root, target, MulNode(left=scale_atom, right=rat_atom))
        else:
            rat_atom = AtomNode(
                kind="ratpoly",
                var_idxs=target.var_idxs,
                kwargs=_kw,
                tag=target.tag,
                inputs=clone_inputs(target),
            )
            cand_root = _replace_node(root, target, rat_atom)

        def _make_custom_init(
            _kind=t_kind,
            _deg_num=t_deg_num,
            _deg_den=t_deg_den,
            _exps_num_key=t_exps_num_key,
            _exps_den_key=t_exps_den_key,
            _fit_a_cpu=t_fit_a_cpu,
            _fit_b_cpu=t_fit_b_cpu,
            _rms=t_rms,
            _scale_tag=scale_tag,
        ):
            def _custom_init(root_inner, model_inner):
                from .stageB import _collect_all_atoms

                atoms = _collect_all_atoms(root_inner)
                leaves = list(model_inner.leaf)

                if _kind == "rratpoly":
                    rat_core = _find_matching_core(
                        atoms,
                        leaves,
                        core_types=RRationalPolyLeaf,
                        expected_kind="rratpoly",
                        expected_tag=target.tag if target.tag is not None else None,
                        expected_inputs=target_inputs,
                        predicate=lambda _atom, core: (
                            int(getattr(core, "deg_num", -1)) == _deg_num
                            and int(getattr(core, "deg_den", -1)) == _deg_den
                            and (
                                _exps_num_key is None
                                or _exps_den_key is None
                                or (
                                    _exps_key(core.exps_num_full.detach().cpu()) == _exps_num_key
                                    and _exps_key(core.exps_den.detach().cpu()) == _exps_den_key
                                )
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
                            "[Stage B custom_init rratpoly] No matching RRationalPolyLeaf "
                            f"for vars {target_var_idxs}, deg_num={_deg_num}, deg_den={_deg_den}"
                        )
                        return

                    dev = rat_core.coeffs_num.device
                    dt = rat_core.coeffs_num.dtype

                    a_full = _fit_a_cpu
                    b = _fit_b_cpu
                    if a_full is None or b is None:
                        try:
                            a_full, b = _fit_rational_coeffs_nd(
                                X_cpu.to(device=dev, dtype=torch.float64),
                                F_cpu.to(device=dev, dtype=torch.float64),
                                exps_num=rat_core.exps_num_full.detach().to(device=dev, dtype=torch.long),
                                exps_den=rat_core.exps_den.detach().to(device=dev, dtype=torch.long),
                                subsample_frac=1.0,
                                eps_Q=float(eps_Q),
                                seed=0,
                            )
                        except Exception as e:
                            print("[Stage B custom_init rratpoly] Fit failed:", repr(e))
                            return
                    if a_full is None or b is None:
                        print("[Stage B custom_init rratpoly] Fit returned None")
                        return

                    lead = a_full[int(rat_core.lead_pos_num)] if a_full.numel() > rat_core.lead_pos_num else None
                    if lead is None or float(lead.abs().item()) < 1e-30:
                        print("[Stage B custom_init rratpoly] Reduced lead vanished")
                        return

                    with torch.no_grad():
                        if rat_core.free_pos_num.numel() > 0 and a_full.numel() == int(rat_core.exps_num_full.shape[0]):
                            idx = rat_core.free_pos_num.to(device=a_full.device)
                            free_a = a_full[idx] / lead
                            rat_core.coeffs_num.copy_(free_a.to(dtype=dt, device=dev))
                        if b.numel() == rat_core.coeffs_den.numel():
                            rat_core.coeffs_den.copy_(b.to(dtype=rat_core.coeffs_den.dtype, device=rat_core.coeffs_den.device))
                        if scale_core is not None and hasattr(scale_core, "value"):
                            scale_core.value.copy_(
                                torch.as_tensor(
                                    float(lead.item()),
                                    dtype=scale_core.value.dtype,
                                    device=scale_core.value.device,
                                )
                            )
                        if rat_core.coeffs_den.numel() > 0:
                            s = rat_core.coeffs_den.abs().max()
                            if torch.isfinite(s) and float(s) > 0.0:
                                rat_core.coeffs_den[rat_core.coeffs_den.abs() < (1e-12 * s)] = 0.0

                    print(
                        f"[Stage B custom_init rratpoly] Init on vars {target_var_idxs}: "
                        f"degN={_deg_num}, degD={_deg_den}, probe_rms_rel≈{_rms:.3g}"
                    )
                    return

                rat_core = _find_matching_core(
                    atoms,
                    leaves,
                    core_types=RationalPolyLeaf,
                    expected_kind="ratpoly",
                    expected_inputs=target_inputs,
                    predicate=lambda _atom, core: (
                        int(getattr(core, "deg_num", -1)) == _deg_num
                        and int(getattr(core, "deg_den", -1)) == _deg_den
                        and (
                            _exps_num_key is None
                            or _exps_den_key is None
                            or (
                                _exps_key(core.exps_num.detach().cpu()) == _exps_num_key
                                and _exps_key(core.exps_den.detach().cpu()) == _exps_den_key
                            )
                        )
                    ),
                )

                if rat_core is None:
                    print(
                        "[Stage B custom_init ratpoly] No matching RationalPolyLeaf "
                        f"for vars {target_var_idxs}, deg_num={_deg_num}, deg_den={_deg_den}"
                    )
                    return

                dev = rat_core.coeffs_num.device
                dt = rat_core.coeffs_num.dtype
                if (
                    _fit_a_cpu is not None
                    and _fit_b_cpu is not None
                    and _fit_a_cpu.numel() == rat_core.coeffs_num.numel()
                    and _fit_b_cpu.numel() == rat_core.coeffs_den.numel()
                ):
                    a = _fit_a_cpu
                    b = _fit_b_cpu
                else:
                    try:
                        a, b = _fit_rational_coeffs_nd(
                            X_cpu.to(device=dev, dtype=torch.float64),
                            F_cpu.to(device=dev, dtype=torch.float64),
                            exps_num=rat_core.exps_num.detach().to(device=dev, dtype=torch.long),
                            exps_den=rat_core.exps_den.detach().to(device=dev, dtype=torch.long),
                            subsample_frac=1.0,
                            eps_Q=float(eps_Q),
                            seed=0,
                        )
                    except Exception as e:
                        print("[Stage B custom_init ratpoly] Fit failed:", repr(e))
                        return
                if a is None or b is None:
                    print("[Stage B custom_init ratpoly] Fit returned None")
                    return

                with torch.no_grad():
                    rat_core.coeffs_num.copy_(a.to(dtype=dt, device=dev))
                    rat_core.coeffs_den.copy_(b.to(dtype=dt, device=dev))
                    for p in (rat_core.coeffs_num, rat_core.coeffs_den):
                        s = p.abs().max()
                        if torch.isfinite(s) and float(s) > 0.0:
                            p[p.abs() < (1e-12 * s)] = 0.0

                print(
                    f"[Stage B custom_init ratpoly] Init on vars {target_var_idxs}: "
                    f"degN={_deg_num}, degD={_deg_den}, probe_rms_rel≈{_rms:.3g}"
                )
            return _custom_init

        leaf_kind = "rratpoly" if t_kind == "rratpoly" else "ratpoly"
        support_signature = _support_signature_nd(
            target_sig=target_sig,
            leaf_kind=leaf_kind,
            exps_num=t_exps_num_t,
            exps_den=t_exps_den_t,
        )
        meta: Dict[str, Any] = {
            "deg_num": t_deg_num,
            "deg_den": t_deg_den,
            "pattern_family": "ratpoly",
            "leaf_kind": leaf_kind,
            "ratpoly_target_sig": int(target_sig),
            "ratpoly_target_tag": target.tag,
            "ratpoly_var_idxs": tuple(int(i) for i in target.var_idxs),
            "ratpoly_exps_num_key": t_exps_num_key,
            "ratpoly_exps_den_key": t_exps_den_key,
            "precheck_rel_rms": t_rms,
            "probe_rms_rel": t_rms,
            "n_terms_num": t_n_terms_num,
            "n_terms_den": t_n_terms_den,
            "signature": support_signature,
            "log": (
                f"[Stage B]  Trying {leaf_kind} deg=({t_deg_num_screen},{t_deg_den_screen})"
                f"->({t_deg_num},{t_deg_den}) nnz=({t_n_terms_num},{t_n_terms_den}) "
                f"on {target.tag} vars={var_idxs}"
            ),
        }
        if t_kind == "rratpoly":
            meta["reduced"] = True
            meta["ratpoly_scale_tag"] = scale_tag
            if t_pivot_reason is not None:
                meta["pivot_reason"] = str(t_pivot_reason)
        if t_deg_num != t_deg_num_screen:
            meta["deg_num_screen"] = t_deg_num_screen
        if t_deg_den != t_deg_den_screen:
            meta["deg_den_screen"] = t_deg_den_screen
        if t_exps_num_t is not None and t_exps_den_t is not None:
            meta["degree_probe_exact_support"] = bool(trial_i.get("exps_num") is not None and trial_i.get("exps_den") is not None)
        if t_coefficient_unit_certificate is not None:
            meta["unit_support_planned"] = True
            meta["coefficient_unit_certificate"] = dict(
                t_coefficient_unit_certificate
            )
            meta["unit_support_diagnostics"] = dict(
                support_plan_diagnostics or {}
            )

        results.append((cand_root, _make_custom_init(), meta))

    return results


def _build_ratpoly_candidate(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    max_deg_num: int = 4,
    max_deg_den: int = 4,
    min_points: int = 400,
    max_points: int = 5000,
    fit_frac: float = 0.7,
    rel_rms_threshold: float = 2e-2,
    max_terms_total: int = 180,
    eps_Q: float = 1e-10,
    enforce_units: bool = False,
    target_dim: Optional[tuple] = None,
    x_dims: Optional[List[tuple]] = None,
) -> Tuple[Optional[Node], Optional[callable], Dict[str, Any]]:
    """Backward-compat wrapper: returns the first (simplest) candidate."""
    results = _build_ratpoly_candidates(
        root=root, target=target, reuse=reuse, train_loader=train_loader,
        device=device, dtype=dtype, max_deg_num=max_deg_num,
        max_deg_den=max_deg_den, min_points=min_points, max_points=max_points,
        fit_frac=fit_frac, rel_rms_threshold=rel_rms_threshold,
        max_terms_total=max_terms_total, eps_Q=eps_Q,
        enforce_units=enforce_units, target_dim=target_dim, x_dims=x_dims,
    )
    if results:
        return results[0]
    return None, None, {}


def _dim_tuple_is_zero(dim: Optional[tuple]) -> bool:
    if dim is None:
        return False
    try:
        return all(v == 0 for v in tuple(dim))
    except Exception:
        return False


def _stable_last_hard_ratio_sig(target: AtomNode) -> int:
    import zlib

    payload = (
        str(getattr(target, "tag", "") or ""),
        tuple(int(i) for i in target.var_idxs),
        tuple(repr(inp) for inp in get_input_exprs(target)),
    )
    return int(zlib.crc32(repr(payload).encode("utf-8")) & 0xFFFFFFFF)


def _build_last_hard_ratio_candidates(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    *,
    target_dim: Optional[tuple],
    x_dims: Optional[List[tuple]],
    min_points: int = 200,
    max_points: int = 5000,
    rel_rms_threshold: float = 2e-2,
    max_p: int = 4,
    max_q: int = 2,
) -> List[Tuple[str, Node, Optional[callable], Dict[str, Any]]]:
    """Build final-atom ratio-coordinate rational candidates.

    This is deliberately narrow: it only handles a single dimensionless
    ratio coordinate z = u/v, with small explicit supports
    ``z**p / (1 - z**2)**q``.  Unitful final atoms are left to the generic
    units-aware ratpoly rescue because the ratio coordinate itself is
    dimensionless.
    """
    if target.kind.lower() != "nn" or int(effective_arity(target)) != 2:
        return []
    if not _dim_tuple_is_zero(target_dim):
        return []
    if x_dims is None or len(x_dims) != 2:
        return []
    if tuple(x_dims[0]) != tuple(x_dims[1]):
        return []

    tag = target.tag
    if tag is None or tag not in reuse:
        return []
    teacher = reuse[tag]

    data = _gather_atom_teacher_data(
        train_loader=train_loader,
        atom=target,
        teacher=teacher,
        device=device,
        dtype=dtype,
        max_points=max_points,
    )
    if data is None:
        return []
    X, F = data
    X = X.to(dtype=torch.float64)
    F = F.to(dtype=torch.float64).view(-1)
    if X.ndim != 2 or int(X.shape[1]) != 2 or int(X.shape[0]) < int(min_points):
        return []

    std_F = float(F.std(unbiased=False).item())
    if not math.isfinite(std_F) or std_F < 1e-12:
        return []

    inputs = get_input_exprs(target)
    if len(inputs) != 2:
        return []

    results: List[Tuple[str, Node, Optional[callable], Dict[str, Any]]] = []
    target_sig = _stable_last_hard_ratio_sig(target)

    trials: List[Dict[str, Any]] = []
    eps = 1e-10
    for den_idx, num_idx in ((0, 1), (1, 0)):
        denom = X[:, den_idx]
        numer = X[:, num_idx]
        mask = torch.isfinite(denom) & torch.isfinite(numer) & torch.isfinite(F) & (denom.abs() > eps)
        if int(mask.sum().item()) < int(min_points):
            continue
        z = (numer[mask] / denom[mask]).view(-1, 1)
        f = F[mask]
        if int(z.shape[0]) < int(min_points):
            continue

        for q in range(1, int(max_q) + 1):
            exps_den = torch.tensor([[2 * k] for k in range(0, q + 1)], dtype=torch.int64)
            for p in range(0, int(max_p) + 1):
                exps_num = torch.tensor([[p]], dtype=torch.int64)
                fit = _fit_rational_coeffs_nd(
                    z,
                    f,
                    exps_num=exps_num,
                    exps_den=exps_den,
                    min_points=min_points,
                    dtype=torch.float64,
                    subsample_frac=1.0,
                    eps_Q=1e-10,
                    seed=0,
                )
                if fit is None:
                    continue
                a, b = fit
                if a is None or b is None:
                    continue
                Phi_num = _eval_monomials(z, exps_num.to(device=z.device))
                Phi_den = _eval_monomials(z, exps_den.to(device=z.device))
                den_val = Phi_den @ b.to(device=z.device, dtype=torch.float64)
                num_val = Phi_num @ a.to(device=z.device, dtype=torch.float64)
                pred = num_val / den_val.clamp_min(1e-8)
                resid = pred - f
                rms = float(torch.sqrt(torch.mean(resid * resid)).item())
                rel_rms = rms / std_F
                if not math.isfinite(rel_rms) or rel_rms > float(rel_rms_threshold):
                    continue
                trials.append(
                    {
                        "num_idx": int(num_idx),
                        "den_idx": int(den_idx),
                        "p": int(p),
                        "q": int(q),
                        "rel_rms": float(rel_rms),
                        "a": a.detach().cpu().clone(),
                        "b": b.detach().cpu().clone(),
                        "exps_num": exps_num.detach().cpu().clone(),
                        "exps_den": exps_den.detach().cpu().clone(),
                    }
                )

    if not trials:
        return []

    trials.sort(
        key=lambda t: (
            int(t["exps_num"].shape[0] + t["exps_den"].shape[0]),
            int(t["q"]),
            int(t["p"]),
            float(t["rel_rms"]),
            int(t["den_idx"]),
            int(t["num_idx"]),
        )
    )

    for trial in trials:
        num_idx = int(trial["num_idx"])
        den_idx = int(trial["den_idx"])
        p = int(trial["p"])
        q = int(trial["q"])
        exps_num = trial["exps_num"]
        exps_den = trial["exps_den"]
        a_cpu = trial["a"]
        b_cpu = trial["b"]
        rel_rms = float(trial["rel_rms"])

        ratio_expr = MulNode(
            left=clone_ast(inputs[num_idx]),
            right=PowNode(base=clone_ast(inputs[den_idx]), exponent=-1.0),
        )
        raw_idxs = tuple(sorted(int(i) for i in _collect_var_idxs_from_node(ratio_expr)))
        if not raw_idxs:
            raw_idxs = tuple(int(i) for i in target.var_idxs)

        exps_num_override = _exps_override_from_tensor(exps_num)
        exps_den_override = _exps_override_from_tensor(exps_den)
        if exps_num_override is None or exps_den_override is None:
            continue
        deg_num = _max_total_degree_from_exps(exps_num, fallback=p)
        deg_den = _max_total_degree_from_exps(exps_den, fallback=2 * q)
        rat_kwargs: Dict[str, Any] = {
            "deg_num": int(deg_num),
            "deg_den": int(deg_den),
            "exps_num_override": exps_num_override,
            "exps_den_override": exps_den_override,
        }
        ratio_inputs = (ratio_expr,)
        rat_atom = AtomNode(
            kind="ratpoly",
            var_idxs=raw_idxs,
            kwargs=rat_kwargs,
            tag=target.tag,
            inputs=ratio_inputs,
        )
        cand_root = _replace_node(root, target, rat_atom)

        def _make_init_fn(
            *,
            _tag=target.tag,
            _inputs=tuple(clone_ast(inp) for inp in ratio_inputs),
            _exps_num=exps_num.clone(),
            _exps_den=exps_den.clone(),
            _a_cpu=a_cpu.clone(),
            _b_cpu=b_cpu.clone(),
        ):
            def _init_fn(root_inner: Node, model_inner: torch.nn.Module):
                from .stageB import _collect_all_atoms

                atoms = _collect_all_atoms(root_inner)
                leaves = list(model_inner.leaf)
                rat_core = _find_matching_core(
                    atoms,
                    leaves,
                    core_types=RationalPolyLeaf,
                    expected_kind="ratpoly",
                    expected_tag=_tag if _tag is not None else None,
                    expected_inputs=_inputs,
                    predicate=lambda _atom, core: (
                        torch.equal(core.exps_num.detach().cpu(), _exps_num.to(dtype=torch.int64))
                        and torch.equal(core.exps_den.detach().cpu(), _exps_den.to(dtype=torch.int64))
                    ),
                )
                if rat_core is None:
                    print("[Stage B custom_init last_hard_ratio] No matching RationalPolyLeaf")
                    return
                with torch.no_grad():
                    if _a_cpu.numel() == rat_core.coeffs_num.numel():
                        rat_core.coeffs_num.copy_(
                            _a_cpu.to(device=rat_core.coeffs_num.device, dtype=rat_core.coeffs_num.dtype)
                        )
                    if _b_cpu.numel() == rat_core.coeffs_den.numel():
                        rat_core.coeffs_den.copy_(
                            _b_cpu.to(device=rat_core.coeffs_den.device, dtype=rat_core.coeffs_den.dtype)
                        )
                print(
                    "[Stage B custom_init last_hard_ratio] "
                    f"Init ratio ratpoly nnz=({_a_cpu.numel()},{_b_cpu.numel()})"
                )

            return _init_fn

        label = f"last_ratio_z{num_idx}_over_z{den_idx}_p{p}_q{q}"
        meta: Dict[str, Any] = {
            "pattern_family": "ratpoly_1d",
            "last_hard_atom_rescue": True,
            "last_hard_family": "ratio_inv1m",
            "leaf_kind": "ratpoly",
            "ratpoly_target_tag": target.tag,
            "ratpoly_var_idxs": tuple(int(i) for i in raw_idxs),
            "ratpoly_exps_num_key": _exps_key(exps_num),
            "ratpoly_exps_den_key": _exps_key(exps_den),
            "deg_num": int(deg_num),
            "deg_den": int(deg_den),
            "n_terms_num": int(exps_num.shape[0]),
            "n_terms_den": int(exps_den.shape[0]),
            "precheck_rel_rms": rel_rms,
            "probe_rms_rel": rel_rms,
            "ratio_num_local_idx": num_idx,
            "ratio_den_local_idx": den_idx,
            "ratio_p": p,
            "ratio_q": q,
            "signature": (
                int(target_sig),
                9127,
                int(num_idx),
                int(den_idx),
                int(p),
                int(q),
            ),
            "log": (
                f"[Stage B]  Last-hard-atom ratio rescue: trying z=arg{num_idx}/arg{den_idx}, "
                f"z^{p}/(1-z^2)^{q}, rel_rms={rel_rms:.3g}"
            ),
        }
        results.append((label, cand_root, _make_init_fn(), meta))

    return results


def _build_nonlinear_sub_candidate(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    hit: Dict,
) -> Optional[Tuple[Node, Optional[Callable], Dict[str, Any]]]:
    """Build a ratpoly candidate with a nonlinear substitution on one variable.

    Given a screening *hit* (from ``_nonlinear_substitution_screen``), replace
    the NN leaf with a ``RationalPolyLeaf`` whose input has one variable
    transformed by T in {cos, sin, exp, log}.

    The compound-variable machinery is used to wrap the selected effective
    input expression, so both ``NN[z, raw]`` and ``NN[z1, z2]`` are handled in
    local leaf coordinates.

    Returns ``(new_root, init_fn, meta)`` or ``None`` on failure.
    """
    if target.kind.lower() != "nn":
        return None

    var_idxs = tuple(int(i) for i in target.var_idxs)
    input_exprs = tuple(get_input_exprs(target))
    dim = len(input_exprs)
    if dim < 1:
        return None

    tag = target.tag
    if tag is None or tag not in reuse:
        return None

    col_idx = int(hit["col_idx"])
    tname = hit["transform"]
    deg_num = int(hit["deg_num"])
    deg_den = int(hit["deg_den"])
    outer_t = hit.get("outer_transform", "identity")
    sign_hint = float(hit.get("sign_hint", 1.0))
    unit_support_planned = bool(hit.get("unit_support_planned", False))
    if unit_support_planned:
        planned_certificate = hit.get("coefficient_unit_certificate")
        if not (
            isinstance(planned_certificate, dict)
            and planned_certificate.get("valid") is True
        ):
            return None
        try:
            from nestynet_sr.sr_core.coefficient_units import (
                solve_rational_coefficient_gauge,
            )

            planned_support_solution = solve_rational_coefficient_gauge(
                target_dim=hit["rational_target_dim"],
                input_dims=hit["transformed_input_dims"],
                numerator_exponents=hit["exps_num_override"],
                denominator_exponents=hit["exps_den_override"],
                coefficient_policy=str(
                    hit.get("coefficient_policy", "free_const_only")
                ),
            )
        except Exception:
            return None
        if not planned_support_solution.ok:
            return None

    _WRAP = {
        "cos": CosNode,
        "sin": SinNode,
        "exp": ExpNode,
        "log": LogNode,
    }
    wrap_fn = _WRAP.get(tname)
    if wrap_fn is None:
        return None

    # Keep Stage-A coordinate orientation by default (e.g. z=x0/x1).
    # Optional override: caller may set hit["trial_inv_z"]=True to test 1/z
    # for the selected effective input coordinate.
    trial_inv_z = bool(hit.get("trial_inv_z", False))
    inv_z_eps = 1e-8
    if col_idx < 0 or col_idx >= dim:
        return None
    new_inputs = []
    for j, inp in enumerate(input_exprs):
        base = clone_ast(inp)
        if j == col_idx:
            if trial_inv_z:
                base = PowNode(base=base, exponent=-1.0)
            base = wrap_fn(arg=base)
        new_inputs.append(base)

    rat_kwargs = {
        "deg_num": deg_num,
        "deg_den": deg_den,
    }

    # --- build custom init from screening data ---
    teacher = reuse[tag]
    data = _gather_atom_teacher_data(
        train_loader=train_loader,
        atom=target,
        teacher=teacher,
        device=device,
        dtype=dtype,
        max_points=3000,
    )
    if data is None:
        return None
    X_raw, F_raw = data
    X_raw = X_raw.to(dtype=torch.float64)
    F_raw = F_raw.to(dtype=torch.float64).view(-1)

    # Apply substitution to get the transformed coordinates
    X_sub = X_raw.clone()
    if trial_inv_z:
        z_trial = X_raw[:, col_idx]
        m_inv = torch.isfinite(z_trial) & (z_trial.abs() > inv_z_eps)
        if int(m_inv.sum().item()) < 200:
            return None
        X_sub = X_sub[m_inv]
        F_raw = F_raw[m_inv]
        X_sub[:, col_idx] = 1.0 / X_sub[:, col_idx]

    tfn = {"cos": torch.cos, "sin": torch.sin, "exp": torch.exp, "log": torch.log}[tname]
    X_sub[:, col_idx] = tfn(X_sub[:, col_idx])

    # When an outer transform is active, fit the rational to the
    # transformed target (e.g. F² for "square") so that the init
    # coefficients live in the correct space.
    F_fit = F_raw
    X_fit = X_sub
    if outer_t == "square":
        # F = sign * sqrt(ratpoly) => ratpoly = F².
        # Use all finite points for fitting in square-space; sign is handled
        # separately via sign_hint when wrapping the AST.
        finite = torch.isfinite(F_raw) & torch.isfinite(X_sub).all(dim=1)
        if int(finite.sum().item()) < 200:
            return None
        F_fit = F_raw[finite] * F_raw[finite]
        X_fit = X_sub[finite]
    elif outer_t == "reciprocal":
        m = torch.isfinite(F_raw) & torch.isfinite(X_sub).all(dim=1) & (F_raw.abs() > 1e-8)
        if int(m.sum().item()) < 200:
            return None
        F_fit = 1.0 / F_raw[m]
        X_fit = X_sub[m]

    # Fit rational coefficients in transformed coordinates.  Unit-aware hits
    # carry supports constructed before screening; never silently expand those
    # back to the dense mixed-dimension basis.
    n_in = X_fit.shape[1]
    if unit_support_planned:
        try:
            exps_num_fit = torch.as_tensor(
                hit["exps_num_override"], dtype=torch.int64
            ).detach().cpu()
            exps_den_fit = torch.as_tensor(
                hit["exps_den_override"], dtype=torch.int64
            ).detach().cpu()
            if (
                exps_num_fit.ndim != 2
                or exps_den_fit.ndim != 2
                or int(exps_num_fit.shape[1]) != int(n_in)
                or int(exps_den_fit.shape[1]) != int(n_in)
                or int(exps_num_fit.shape[0]) <= 0
                or int(exps_den_fit.shape[0]) <= 0
                or bool((exps_num_fit < 0).any())
                or bool((exps_den_fit < 0).any())
            ):
                return None
        except Exception:
            return None
    else:
        exps_num_fit = torch.tensor(
            _enumerate_exponents(n_in, deg_num), dtype=torch.int64
        )
        exps_den_fit = torch.tensor(
            _enumerate_exponents(n_in, deg_den), dtype=torch.int64
        )
    result = _fit_rational_coeffs_nd(
        X_fit,
        F_fit,
        exps_num=exps_num_fit,
        exps_den=exps_den_fit,
        min_points=200,
        return_support_indices=True,
    )
    # The units checker no longer accepts an implicit dense ratpoly basis.  A
    # failed final refit therefore cannot fall through to an AST containing
    # only degree labels, even on the legacy/no-units path.
    if result is None:
        return None

    X_cpu = X_fit.detach().cpu()
    F_cpu = F_fit.detach().cpu()
    fitted_a = result[0].detach().cpu() if result is not None else None
    fitted_b = result[1].detach().cpu() if result is not None else None
    fitted_support_num = result[2].detach().cpu() if result is not None else None
    fitted_support_den = result[3].detach().cpu() if result is not None else None

    model_deg_num = int(deg_num)
    model_deg_den = int(deg_den)

    def _support_is_valid(support: Optional[torch.Tensor], exps_dense: torch.Tensor) -> bool:
        if support is None or int(support.numel()) <= 0:
            return False
        if exps_dense.ndim != 2 or int(exps_dense.shape[0]) <= 0:
            return False
        if int(support.min().item()) < 0:
            return False
        if int(support.max().item()) >= int(exps_dense.shape[0]):
            return False
        return True

    def _sparse_fit_mse(
        Xv: torch.Tensor,
        yv: torch.Tensor,
        exps_num_dense: torch.Tensor,
        exps_den_dense: torch.Tensor,
        a_sp: Optional[torch.Tensor],
        b_sp: Optional[torch.Tensor],
        sup_num: Optional[torch.Tensor],
        sup_den: Optional[torch.Tensor],
        eps_den: float = 1e-8,
    ) -> float:
        if a_sp is None or b_sp is None:
            return float("inf")
        if not _support_is_valid(sup_num, exps_num_dense):
            return float("inf")
        if not _support_is_valid(sup_den, exps_den_dense):
            return float("inf")

        exps_num_sel = exps_num_dense[sup_num]
        exps_den_sel = exps_den_dense[sup_den]
        if int(a_sp.numel()) != int(exps_num_sel.shape[0]):
            return float("inf")
        if int(b_sp.numel()) != int(exps_den_sel.shape[0]):
            return float("inf")

        Phi_num = _eval_monomials(Xv, exps_num_sel)
        Phi_den = _eval_monomials(Xv, exps_den_sel)
        den = (Phi_den @ b_sp).clamp_min(float(eps_den))
        pred = (Phi_num @ a_sp) / den
        err = pred - yv
        mse = float((err.square().mean()).item())
        if not math.isfinite(mse):
            return float("inf")
        return mse

    # Preserve sparse support structurally by compressing the declared degrees
    # down to the highest active monomial degrees. This is safe because the
    # dense total-degree exponent tables for lower degrees are prefixes of the
    # higher-degree tables that the support indices refer to.
    if _support_is_valid(fitted_support_num, exps_num_fit):
        model_deg_num = int(exps_num_fit[fitted_support_num].sum(dim=1).max().item())
    if _support_is_valid(fitted_support_den, exps_den_fit):
        model_deg_den = int(exps_den_fit[fitted_support_den].sum(dim=1).max().item())

    # Optional degree-down refit (disabled for now): this was introduced to
    # iteratively shrink declared degrees after sparse support selection, but
    # it can change expression shape in ways that hurt readability.
    enable_degree_down_refit = False
    if enable_degree_down_refit and (
        fitted_a is not None
        and fitted_b is not None
        and fitted_support_num is not None
        and fitted_support_den is not None
    ):
        current_mse = _sparse_fit_mse(
            X_fit, F_fit, exps_num_fit, exps_den_fit,
            fitted_a, fitted_b, fitted_support_num, fitted_support_den,
        )
        # Conservative "still works" budget for degree downgrades.
        rel_mse_tol = 0.05
        abs_mse_tol = 1e-12

        while math.isfinite(current_mse):
            proposals: List[Tuple[int, int]] = []
            if model_deg_num > 1 and model_deg_den > 1:
                proposals.append((model_deg_num - 1, model_deg_den - 1))
            if model_deg_num > 1:
                proposals.append((model_deg_num - 1, model_deg_den))
            if model_deg_den > 1:
                proposals.append((model_deg_num, model_deg_den - 1))
            if not proposals:
                break

            accepted = []
            mse_budget = current_mse * (1.0 + rel_mse_tol) + abs_mse_tol
            for dn_try, dd_try in proposals:
                exps_num_try = torch.tensor(
                    _enumerate_exponents(n_in, dn_try), dtype=torch.int64
                )
                exps_den_try = torch.tensor(
                    _enumerate_exponents(n_in, dd_try), dtype=torch.int64
                )
                fit_try = _fit_rational_coeffs_nd(
                    X_fit,
                    F_fit,
                    exps_num=exps_num_try,
                    exps_den=exps_den_try,
                    min_points=200,
                    return_support_indices=True,
                )
                if fit_try is None:
                    continue
                a_try = fit_try[0].detach().cpu()
                b_try = fit_try[1].detach().cpu()
                s_num_try = fit_try[2].detach().cpu()
                s_den_try = fit_try[3].detach().cpu()
                mse_try = _sparse_fit_mse(
                    X_fit, F_fit, exps_num_try, exps_den_try,
                    a_try, b_try, s_num_try, s_den_try,
                )
                if not math.isfinite(mse_try) or mse_try > mse_budget:
                    continue
                accepted.append(
                    (int(dn_try), int(dd_try), float(mse_try), a_try, b_try, s_num_try, s_den_try)
                )

            if not accepted:
                break

            # Prefer lower declared degree first, then fewer sparse parameters.
            accepted.sort(
                key=lambda t: (
                    int(t[0] + t[1]),
                    int(t[0]),
                    int(t[1]),
                    int(t[5].numel() + t[6].numel()),
                    float(t[2]),
                )
            )
            (
                model_deg_num,
                model_deg_den,
                current_mse,
                fitted_a,
                fitted_b,
                fitted_support_num,
                fitted_support_den,
            ) = accepted[0]

    exps_num_selected = (
        exps_num_fit[fitted_support_num].detach().cpu().clone()
        if _support_is_valid(fitted_support_num, exps_num_fit)
        else None
    )
    exps_den_selected = (
        exps_den_fit[fitted_support_den].detach().cpu().clone()
        if _support_is_valid(fitted_support_den, exps_den_fit)
        else None
    )
    if exps_num_selected is None or exps_den_selected is None:
        return None

    candidate_kind = "ratpoly"
    pivot_reason = None
    if (
        outer_t == "identity"
        and exps_num_selected is not None
        and fitted_a is not None
        and X_fit.shape[1] > 1
    ):
        pivot_idx, pivot_reason = _select_clear_rratpoly_pivot(exps_num_selected, fitted_a)
        if pivot_idx is not None:
            exps_num_selected, fitted_a = _move_sparse_pivot_to_end(
                exps_num_selected,
                fitted_a,
                int(pivot_idx),
            )
            candidate_kind = "rratpoly"

    coefficient_unit_certificate = None
    if unit_support_planned:
        try:
            final_certificate = solve_rational_coefficient_gauge(
                target_dim=hit["rational_target_dim"],
                input_dims=hit["transformed_input_dims"],
                numerator_exponents=exps_num_selected.tolist(),
                denominator_exponents=exps_den_selected.tolist(),
                numerator_pivot=(
                    int(exps_num_selected.shape[0]) - 1
                    if candidate_kind == "rratpoly"
                    else None
                ),
                coefficient_policy=str(
                    hit.get("coefficient_policy", "free_const_only")
                ),
            )
        except Exception:
            return None
        if not final_certificate.ok:
            return None
        coefficient_unit_certificate = final_certificate.to_dict()

    if exps_num_selected is not None:
        rat_kwargs["exps_num_override"] = _exps_override_from_tensor(exps_num_selected)
        model_deg_num = _max_total_degree_from_exps(exps_num_selected, fallback=model_deg_num)
    if exps_den_selected is not None:
        rat_kwargs["exps_den_override"] = _exps_override_from_tensor(exps_den_selected)
        model_deg_den = _max_total_degree_from_exps(exps_den_selected, fallback=model_deg_den)
    rat_kwargs["deg_num"] = int(model_deg_num)
    rat_kwargs["deg_den"] = int(model_deg_den)

    scale_tag = None
    if candidate_kind == "rratpoly":
        scale_tag = (
            f"nls_ratpoly_scale_{str(target.tag) if target.tag is not None else 'anon'}"
            f"_c{col_idx}_{tname}_sn{deg_num}_sd{deg_den}_n{model_deg_num}_d{model_deg_den}"
        )
        scale_atom = AtomNode(
            kind="scale",
            var_idxs=(),
            kwargs={"init": 1.0, "name": "s"},
            tag=scale_tag,
        )
        rat_kwargs["_mul_scale_tag"] = scale_tag
        rat_expr = MulNode(
            left=scale_atom,
            right=AtomNode(
                kind="rratpoly",
                var_idxs=var_idxs,
                kwargs=rat_kwargs,
                tag=target.tag,
                inputs=tuple(new_inputs),
            ),
        )
    else:
        rat_expr = AtomNode(
            kind="ratpoly",
            var_idxs=var_idxs,
            kwargs=rat_kwargs,
            tag=None,
            inputs=tuple(new_inputs),
        )

    # --- outer-transform wrapping (square → sqrt, reciprocal → inverse) ---
    if outer_t == "square":
        sqrt_node = PowNode(base=rat_expr, exponent=0.5)
        if sign_hint < 0.0:
            new_subtree = MulNode(ConstNode(-1.0), sqrt_node)
        else:
            new_subtree = sqrt_node
    elif outer_t == "reciprocal":
        new_subtree = PowNode(base=rat_expr, exponent=-1.0)
    else:
        new_subtree = rat_expr

    cand_root = _replace_node(root, target, new_subtree)

    rat_inputs = tuple(new_inputs)
    exps_num_key = _exps_key(exps_num_selected)
    exps_den_key = _exps_key(exps_den_selected)

    def _custom_init(root_inner, model_inner):
        from .stageB import _collect_all_atoms

        atoms = _collect_all_atoms(root_inner)
        leaves = list(model_inner.leaf)
        if candidate_kind == "rratpoly":
            rat_core = _find_matching_core(
                atoms,
                leaves,
                core_types=RRationalPolyLeaf,
                expected_kind="rratpoly",
                expected_tag=target.tag if target.tag is not None else None,
                expected_inputs=rat_inputs,
                predicate=lambda _atom, core: (
                    int(getattr(core, "deg_num", -1)) == model_deg_num
                    and int(getattr(core, "deg_den", -1)) == model_deg_den
                    and (
                        exps_num_key is None
                        or exps_den_key is None
                        or (
                            _exps_key(core.exps_num_full.detach().cpu()) == exps_num_key
                            and _exps_key(core.exps_den.detach().cpu()) == exps_den_key
                        )
                    )
                ),
            )
            scale_core = _find_matching_core(
                atoms,
                leaves,
                core_types=torch.nn.Module,
                expected_kind="scale",
                expected_tag=scale_tag,
            )
            if rat_core is None:
                return

            dev = rat_core.coeffs_num.device
            dt = rat_core.coeffs_num.dtype
            a_full = fitted_a
            b = fitted_b
            if a_full is None or b is None:
                fallback_fit = _fit_rational_coeffs_nd(
                    X_cpu.to(device=dev, dtype=torch.float64),
                    F_cpu.to(device=dev, dtype=torch.float64),
                    exps_num=rat_core.exps_num_full.detach().to(device=dev, dtype=torch.long),
                    exps_den=rat_core.exps_den.detach().to(device=dev, dtype=torch.long),
                    subsample_frac=1.0,
                    seed=0,
                )
                if fallback_fit is None:
                    return
                a_full, b = fallback_fit[:2]

            lead = a_full[int(rat_core.lead_pos_num)] if a_full.numel() > rat_core.lead_pos_num else None
            if lead is None or float(lead.abs().item()) < 1e-30:
                return

            with torch.no_grad():
                if a_full.numel() == int(rat_core.exps_num_full.shape[0]) and rat_core.free_pos_num.numel() > 0:
                    idx = rat_core.free_pos_num.to(device=a_full.device)
                    free_a = a_full[idx] / lead
                    rat_core.coeffs_num.copy_(free_a.to(dtype=dt, device=dev))
                if b.numel() == rat_core.coeffs_den.numel():
                    rat_core.coeffs_den.copy_(b.to(dtype=rat_core.coeffs_den.dtype, device=rat_core.coeffs_den.device))
                if scale_core is not None and hasattr(scale_core, "value"):
                    scale_core.value.copy_(
                        torch.as_tensor(
                            float(lead.item()),
                            dtype=scale_core.value.dtype,
                            device=scale_core.value.device,
                        )
                    )
            return

        rat_core = _find_matching_core(
            atoms,
            leaves,
            core_types=RationalPolyLeaf,
            expected_kind="ratpoly",
            expected_inputs=rat_inputs,
            predicate=lambda _atom, core: (
                int(getattr(core, "deg_num", -1)) == model_deg_num
                and int(getattr(core, "deg_den", -1)) == model_deg_den
                and (
                    exps_num_key is None
                    or exps_den_key is None
                    or (
                        _exps_key(core.exps_num.detach().cpu()) == exps_num_key
                        and _exps_key(core.exps_den.detach().cpu()) == exps_den_key
                    )
                )
            ),
        )

        if rat_core is None:
            return

        if fitted_a is not None and fitted_b is not None:
            dev = rat_core.coeffs_num.device
            dt = rat_core.coeffs_num.dtype
            if (fitted_a.numel() == rat_core.coeffs_num.numel()
                    and fitted_b.numel() == rat_core.coeffs_den.numel()):
                with torch.no_grad():
                    rat_core.coeffs_num.copy_(fitted_a.to(dtype=dt, device=dev))
                    rat_core.coeffs_den.copy_(fitted_b.to(dtype=dt, device=dev))
                    for p in (rat_core.coeffs_num, rat_core.coeffs_den):
                        s = p.abs().max()
                        if torch.isfinite(s) and float(s) > 0.0:
                            p[p.abs() < (1e-12 * s)] = 0.0
            return

        # Fallback: re-fit from stored data
        try:
            dev = rat_core.coeffs_num.device
            dt = rat_core.coeffs_num.dtype
            exps_num_core = rat_core.exps_num.detach().to(device=dev, dtype=torch.long)
            exps_den_core = rat_core.exps_den.detach().to(device=dev, dtype=torch.long)

            fallback_fit = _fit_rational_coeffs_nd(
                X_cpu.to(device=dev, dtype=torch.float64),
                F_cpu.to(device=dev, dtype=torch.float64),
                exps_num=exps_num_core,
                exps_den=exps_den_core,
                subsample_frac=1.0, seed=0,
            )
            if fallback_fit is not None:
                a, b = fallback_fit[:2]
                with torch.no_grad():
                    rat_core.coeffs_num.copy_(a.to(dtype=dt, device=dev))
                    rat_core.coeffs_den.copy_(b.to(dtype=dt, device=dev))
        except Exception:
            pass

    # Hint for numerically-stable fitting: if an outer transform wraps the
    # candidate, prefit in lifted space before baseline-space refinement.
    if outer_t == "square":
        _custom_init._fit_lift_link = "square"
    elif outer_t == "reciprocal":
        _custom_init._fit_lift_link = "recip"

    meta = {
        "transform": tname,
        "col_idx": col_idx,
        "deg_num": int(model_deg_num),
        "deg_den": int(model_deg_den),
        "leaf_kind": "rratpoly" if candidate_kind == "rratpoly" else "ratpoly",
        "probe_error": float(hit.get("error", float("inf"))),
    }
    if unit_support_planned:
        meta["unit_support_planned"] = True
        meta["unit_support_complexity"] = int(
            len(coefficient_unit_certificate.get("numerator", ()))
            + len(coefficient_unit_certificate.get("denominator", ()))
        )
        meta["coefficient_unit_certificate"] = coefficient_unit_certificate
    if int(model_deg_num) != int(deg_num):
        meta["deg_num_screen"] = int(deg_num)
    if int(model_deg_den) != int(deg_den):
        meta["deg_den_screen"] = int(deg_den)
    if exps_num_selected is not None and exps_den_selected is not None:
        meta["n_terms_num"] = int(exps_num_selected.shape[0])
        meta["n_terms_den"] = int(exps_den_selected.shape[0])
        # These are the post-fit/post-sparsification supports actually encoded
        # in the candidate AST.  Proposal accounting must deduplicate on this
        # final structure rather than on the broader support that was screened.
        meta["exps_num_override"] = _exps_override_from_tensor(exps_num_selected)
        meta["exps_den_override"] = _exps_override_from_tensor(exps_den_selected)
    if candidate_kind == "rratpoly":
        meta["reduced"] = True
        if pivot_reason is not None:
            meta["pivot_reason"] = str(pivot_reason)
    if trial_inv_z:
        meta["trial_inv_z"] = True
    if outer_t != "identity":
        meta["outer_transform"] = outer_t
    if outer_t == "square":
        meta["sign_hint"] = sign_hint
        meta["sign_consistency"] = float(hit.get("sign_consistency", 1.0))
    return cand_root, _custom_init, meta


def _build_log_poly_candidate(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    degree: int = 2,
    min_points: int = 400,
    max_points: int = 5000,
    max_abs_exponent: float = 20.0,
    rel_rms_threshold: float = 1e-3,
    eps: float = 1e-8,
    homogeneous: bool = False,
) -> Tuple[Optional[Node], Optional[callable]]:
    """
    Try to replace a (multi-D) NN atom by log(poly) in its own inputs:

        f(x_S) ≈ log(P(x_S))

    Equivalently, exp(f) ≈ P. We therefore probe whether g=exp(f) can be
    approximated by a low-degree polynomial in the same variables.

    Notes
    -----
    This is a specialization of the existing log(ratpoly) candidate with
    deg_den=0 (i.e. a pure polynomial argument), yielding fewer parameters
    and typically nicer symbolic forms.
    """
    # Import locally to avoid circular dependency
    from .stageB import _collect_all_atoms

    if target.kind.lower() != "nn":
        return None, None

    var_idxs = tuple(int(i) for i in target.var_idxs)
    dim = int(effective_arity(target))
    if dim < 1:
        return None, None

    tag = target.tag
    if tag is None or tag not in reuse:
        return None, None
    teacher = reuse[tag]

    data = _gather_atom_teacher_data(
        train_loader=train_loader,
        atom=target,
        teacher=teacher,
        device=device,
        dtype=dtype,
        max_points=max_points,
    )
    if data is None:
        return None, None
    X, F = data
    if X.numel() == 0 or F.numel() == 0:
        return None, None

    X = X.to(dtype=torch.float64)
    F = F.to(dtype=torch.float64).view(-1)
    N, dim2 = X.shape
    if N < min_points:
        return None, None
    if dim2 != dim:
        return None, None

    max_abs = float(F.abs().max().item())
    if (not math.isfinite(max_abs)) or max_abs > max_abs_exponent:
        return None, None

    g = torch.exp(F)
    if not torch.isfinite(g).all():
        return None, None

    # Fit P(x) ≈ g(x) with total-degree polynomial basis <= degree
    Np = min(int(N), max(int(min_points), 1000))
    Xp = X[:Np]
    gp = g[:Np]

    _mt_lp = degree if homogeneous else 0
    exps_list = _enumerate_exponents(dim, int(degree), min_total=_mt_lp)
    exps = torch.tensor(exps_list, dtype=torch.int64, device=Xp.device)
    Phi = _eval_monomials(Xp, exps)  # [Np, n_terms]
    n_terms = int(Phi.shape[1])
    if Np < (n_terms + 5):
        return None, None

    try:
        coeffs = torch.linalg.lstsq(Phi, gp.unsqueeze(1)).solution.squeeze(1)
    except Exception:
        coeffs = (torch.linalg.pinv(Phi) @ gp.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        g_hat = (Phi @ coeffs).view(-1)

    # Ensure positivity on the sampled region (required for log(P))
    idx_const = None
    for k, e in enumerate(exps_list):
        if sum(int(t) for t in e) == 0:
            idx_const = k
            break
    if idx_const is None:
        idx_const = 0

    min_hat = float(g_hat.min().item())
    if (not math.isfinite(min_hat)) or min_hat <= eps:
        delta = (eps - min_hat) if math.isfinite(min_hat) else 1.0
        with torch.no_grad():
            coeffs[idx_const] = coeffs[idx_const] + float(delta)
            g_hat = g_hat + float(delta)

    # Relative RMS (normalised by std(g))
    resid = g_hat - gp
    rms_abs = float(torch.sqrt(torch.mean(resid * resid)).item())
    std_g = float(gp.std(unbiased=False).item())
    if std_g < 1e-12:
        rms_rel = 0.0 if rms_abs < 1e-12 else float("inf")
    else:
        rms_rel = float(rms_abs / (std_g + 1e-12))

    if (not math.isfinite(rms_rel)) or rms_rel > rel_rms_threshold:
        return None, None

    target_var_idxs = var_idxs
    target_inputs = get_input_exprs(target)

    # Preserve compound-variable mapping so the polynomial is fitted and
    # evaluated in the same effective coordinates as the target leaf.
    poly_kwargs: Dict[str, object] = {"degree": int(degree), "min_total": _mt_lp}
    poly_atom = AtomNode(
        kind="poly",
        var_idxs=target_var_idxs,
        kwargs=poly_kwargs,
        tag=None,
        inputs=clone_inputs(target),
    )
    new_subtree = LogNode(arg=poly_atom)
    cand_root = _replace_node(root, target, new_subtree)

    def _custom_init(root_inner: Node, model_inner: torch.nn.Module):
        """Copy the fitted polynomial coefficients into the new PolyLeaf."""
        atoms = _collect_all_atoms(root_inner)
        leaves = list(model_inner.leaf)
        poly_core = _find_matching_core(
            atoms,
            leaves,
            core_types=PolyLeaf,
            expected_kind="poly",
            expected_inputs=target_inputs,
        )

        if poly_core is None:
            print("[Stage B custom_init log-poly] No PolyLeaf found for vars", target_var_idxs)
            return

        dev = poly_core.coeffs.device
        dt = poly_core.coeffs.dtype
        with torch.no_grad():
            if coeffs.numel() == poly_core.coeffs.numel():
                poly_core.coeffs.copy_(coeffs.to(device=dev, dtype=dt))
            else:
                print(
                    "[Stage B custom_init log-poly] Coeff count mismatch; "
                    f"got {coeffs.numel()}, expected {poly_core.coeffs.numel()}"
                )

        print(
            f"[Stage B custom_init log-poly] Init on vars {target_var_idxs}: degree={int(degree)}, rms_rel≈{rms_rel:.2g}"
        )

    # Hint for numerically-stable fitting: log(u) -> fit u via exp-link first.
    _custom_init._fit_lift_link = "exp"

    return cand_root, _custom_init


def _build_sqrt_poly_candidate(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    degree: int = 2,
    min_points: int = 400,
    max_points: int = 5000,
    rel_rms_threshold: float = 1e-3,
    noise_floor_raw: Optional[float] = None,
    noise_rel_rms_mult: float = 2.0,
    noise_rel_rms_cap: Optional[float] = 0.5,
    eps: float = 1e-8,
    homogeneous: bool = False,
    target_dim: Optional[tuple] = None,
    x_dims: Optional[List[tuple]] = None,
) -> Tuple[Optional[Node], Optional[callable]]:
    """
    Try to replace a multi-D NN atom by a sqrt(poly) or 1/sqrt(poly)
    in its own inputs:

        f(x_S) ≈ sqrt(P(x_S))      or      f(x_S) ≈ 1/sqrt(P(x_S))

    where P is a low-degree polynomial. We fit P to f^2 and 1/f^2 via
    least squares and choose whichever gives better relative RMS error
    on f itself.

    When ``target_dim`` and ``x_dims`` are provided and ``homogeneous=True``,
    a dimensional probe selects the exact valid monomials for P.  For
    sqrt(P), ``dim(P) = 2*target_dim``; for 1/sqrt(P), ``dim(P) = -2*target_dim``.
    """
    # Import locally to avoid circular dependency
    from .stageB import _collect_all_atoms

    if target.kind.lower() != "nn":
        return None, None

    var_idxs = tuple(int(i) for i in target.var_idxs)
    dim = len(var_idxs)
    if dim < 1:
        return None, None

    tag = target.tag
    if tag is None or tag not in reuse:
        return None, None
    teacher = reuse[tag]

    data = _gather_atom_teacher_data(
        train_loader=train_loader,
        atom=target,
        teacher=teacher,
        device=device,
        dtype=dtype,
        max_points=max_points,
    )
    if data is None:
        return None, None
    X, F = data
    if X.numel() == 0 or F.numel() == 0:
        return None, None

    X = X.to(dtype=torch.float64)
    F = F.to(dtype=torch.float64).view(-1)
    N, dim = X.shape
    if N < min_points:
        return None, None

    # Restrict to a region where f has a consistent sign so that
    # sqrt and 1/sqrt are well-defined. For globally negative
    # leaves we flip the sign and work with |f|.
    mask, sign = _select_sign_region(F, min_points=min_points, eps=eps)
    if mask is None:
        return None, None
    X = X[mask]
    F = sign * F[mask]
    try:
        f_rms_for_noise = float(torch.sqrt(torch.mean(F * F)).item())
    except Exception:
        f_rms_for_noise = None
    rel_rms_threshold_eff = _noisy_rel_rms_threshold(
        rel_rms_threshold,
        noise_floor=noise_floor_raw,
        y_rms=f_rms_for_noise,
        noise_mult=float(noise_rel_rms_mult),
        cap=noise_rel_rms_cap,
    )

    # ── Dimensional degree probe ──
    # sqrt(P): dim(P) = 2*dim(f);  1/sqrt(P): dim(P) = -2*dim(f)
    # Gate on dimensional data availability, not on `homogeneous`.
    _use_dim_probe = (
        target_dim is not None
        and x_dims is not None
        and len(x_dims) == dim
    )
    _probe_sqrt = None  # exps dict for sqrt branch
    _probe_inv = None   # exps dict for inv-sqrt branch
    _sqrt_dim = None
    _inv_dim = None
    if _use_dim_probe:
        from fractions import Fraction as _Fr
        from .ratpoly_degree_probe import probe_poly_exponents
        _td = tuple(_Fr(x) for x in target_dim)
        _sqrt_dim = tuple(2 * d for d in _td)
        _inv_dim = tuple(-2 * d for d in _td)
        _probe_sqrt = probe_poly_exponents(_sqrt_dim, x_dims, max_degree=degree)
        _probe_inv = probe_poly_exponents(_inv_dim, x_dims, max_degree=degree)

    # Build monomial design matrices — may differ between sqrt and inv_sqrt
    # when units impose different constraints.
    _mt_sp = degree if homogeneous else 0
    _fallback_exps = _enumerate_exponents(dim, degree, min_total=_mt_sp)

    def _dim_is_zero(dim_tuple) -> bool:
        try:
            return all(x == 0 for x in dim_tuple)
        except Exception:
            return False

    _all_inputs_dimless = False
    if x_dims is not None:
        try:
            _all_inputs_dimless = all(_dim_is_zero(tuple(d)) for d in x_dims)
        except Exception:
            _all_inputs_dimless = False

    def _build_exps(probe_result, required_dim):
        """Build a unit-compatible exponent basis for one sqrt branch.

        In strict unit mode, a missing degree-probe result means "no valid
        monomial with the required radicand units", not "fall back to a mixed
        polynomial".  The only allowed fallback with unit metadata present is
        the genuinely dimensionless case, where constants and mixed degrees
        have the same units.
        """
        if probe_result is not None:
            el = []
            for k in sorted(probe_result):
                if k <= degree:
                    el.extend(probe_result[k])
            if el:
                return el, min(probe_result), True

        if _use_dim_probe:
            if required_dim is not None and _dim_is_zero(required_dim) and _all_inputs_dimless:
                return _fallback_exps, _mt_sp, False
            return None, None, False

        return _fallback_exps, _mt_sp, False

    def _compact_poly_basis(exps_list):
        """Return (degree, min_total) if PolyLeaf can represent exactly these exps."""
        if not exps_list:
            return None
        try:
            exps_norm = [tuple(int(v) for v in exp) for exp in exps_list]
            totals = [sum(exp) for exp in exps_norm]
            deg_use = int(max(totals))
            mt_use = int(min(totals))
            dense = _enumerate_exponents(dim, deg_use, min_total=mt_use)
            if sorted(exps_norm) != sorted(dense):
                return None
            return deg_use, mt_use
        except Exception:
            return None

    exps_sqrt, _, _sqrt_probed = _build_exps(_probe_sqrt, _sqrt_dim)
    exps_inv, _, _inv_probed = _build_exps(_probe_inv, _inv_dim)
    sqrt_basis = _compact_poly_basis(exps_sqrt) if exps_sqrt is not None else None
    inv_basis = _compact_poly_basis(exps_inv) if exps_inv is not None else None
    if _sqrt_probed or _inv_probed:
        print(
            f"[Stage B] sqrt_poly: degree-probe active, "
            f"sqrt={len(exps_sqrt) if exps_sqrt is not None else 0} terms (probed={_sqrt_probed}), "
            f"inv={len(exps_inv) if exps_inv is not None else 0} terms (probed={_inv_probed}), vars={var_idxs}"
        )

    def _fit_branch(Y, exps_list):
        exps_t = torch.tensor(exps_list, dtype=torch.int64, device=X.device)
        Phi = _eval_monomials(X, exps_t)
        try:
            coeffs = (torch.linalg.pinv(Phi) @ Y.unsqueeze(1)).squeeze(1)
        except RuntimeError:
            return None, None
        Y_hat = (Phi @ coeffs).view(-1)
        return coeffs, Y_hat

    # Candidate 1: sqrt(poly) via fitting P ≈ f^2
    F_sq = F * F
    coeffs1, Yhat1 = (None, None)
    if exps_sqrt is not None and sqrt_basis is not None:
        coeffs1, Yhat1 = _fit_branch(F_sq, exps_sqrt)
    rms_rel1 = float("inf")
    if coeffs1 is not None:
        Yhat1_cl = torch.clamp(Yhat1, min=0.0)
        Fhat1 = torch.sqrt(Yhat1_cl)
        resid1 = F - Fhat1
        scale1 = float(torch.sqrt(torch.mean(F * F)))
        if scale1 > 0:
            rms_rel1 = float(torch.sqrt(torch.mean(resid1 * resid1)) / scale1)

    # Candidate 2: 1/sqrt(poly) via fitting P ≈ 1/f^2
    invF_sq = 1.0 / torch.clamp(F_sq, min=eps * eps)
    coeffs2, Yhat2 = (None, None)
    if exps_inv is not None and inv_basis is not None:
        coeffs2, Yhat2 = _fit_branch(invF_sq, exps_inv)
    rms_rel2 = float("inf")
    if coeffs2 is not None:
        Yhat2_cl = torch.clamp(Yhat2, min=eps * eps)
        Fhat2 = 1.0 / torch.sqrt(Yhat2_cl)
        resid2 = F - Fhat2
        scale2 = float(torch.sqrt(torch.mean(F * F)))
        if scale2 > 0:
            rms_rel2 = float(torch.sqrt(torch.mean(resid2 * resid2)) / scale2)

    # Choose the better of the two, if any is good enough.
    best_kind: Optional[str] = None
    best_coeffs: Optional[torch.Tensor] = None
    best_degree: Optional[int] = None
    best_mt: Optional[int] = None

    if rms_rel1 < rms_rel2 and rms_rel1 < rel_rms_threshold_eff:
        best_kind = "sqrt"
        best_coeffs = coeffs1
        best_degree, best_mt = sqrt_basis
    elif rms_rel2 < rel_rms_threshold_eff:
        best_kind = "inv_sqrt"
        best_coeffs = coeffs2
        best_degree, best_mt = inv_basis
    else:
        best_kind = None

    if best_kind is None or best_coeffs is None or best_degree is None or best_mt is None:
        return None, None

    target_var_idxs = var_idxs
    target_inputs = get_input_exprs(target)

    # Preserve compound-variable mapping so the polynomial is built in the
    # same effective coordinates as the NN leaf.
    poly_kwargs: Dict[str, object] = {"degree": int(best_degree), "min_total": int(best_mt)}
    poly_atom = AtomNode(
        kind="poly",
        var_idxs=target_var_idxs,
        kwargs=poly_kwargs,
        tag=None,
        inputs=clone_inputs(target),
    )
    exponent = 0.5 if best_kind == "sqrt" else -0.5
    new_subtree = PowNode(base=poly_atom, exponent=float(exponent))
    cand_root = _replace_node(root, target, new_subtree)

    def _custom_init(root_inner: Node, model_inner: torch.nn.Module):
        """
        Copy the fitted polynomial coefficients into the new PolyLeaf
        that sits under the PowNode.
        """
        atoms = _collect_all_atoms(root_inner)
        leaves = list(model_inner.leaf)
        poly_core = _find_matching_core(
            atoms,
            leaves,
            core_types=PolyLeaf,
            expected_kind="poly",
            expected_inputs=target_inputs,
        )

        if poly_core is None:
            print("[Stage B custom_init sqrt-poly] No PolyLeaf found for vars", target_var_idxs)
            return

        dev = poly_core.coeffs.device
        dt = poly_core.coeffs.dtype
        with torch.no_grad():
            if best_coeffs.numel() == poly_core.coeffs.numel():
                poly_core.coeffs.copy_(best_coeffs.to(device=dev, dtype=dt))
            else:
                print(
                    "[Stage B custom_init sqrt-poly] Coeff count mismatch; "
                    f"got {best_coeffs.numel()}, expected {poly_core.coeffs.numel()}"
                )

    # Hint for numerically-stable fitting: prefit in lifted space first.
    if best_kind == "sqrt":
        _custom_init._fit_lift_link = "square"
    else:
        _custom_init._fit_lift_link = "inv_square"

    return cand_root, _custom_init


def _build_inv_poly_candidates(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    max_degree: int = 2,
    min_points: int = 400,
    max_points: int = 5000,
    rel_rms_threshold: float = 1e-3,
    eps: float = 1e-8,
    homogeneous: bool = False,
    target_dim: Optional[tuple] = None,
    x_dims: Optional[List[tuple]] = None,
) -> List[Tuple[Node, callable, Dict[str, Any]]]:
    """
    Try to replace an NN atom by an inverse polynomial in its own inputs:

        f(x_S) ≈ 1/P(x_S)

    where P is a low-degree polynomial fitted via least squares to 1/f.
    Handles forms like 1/(x+1), 1/(1+x²), etc.

    Returns a list of ``(root, init_fn, meta)`` tuples ordered by degree
    (simplest first). Each tuple corresponds to a degree whose fit passed
    ``rel_rms_threshold`` so Stage B can fall back to a higher-degree
    inverse-polynomial if the simplest candidate is rejected later.

    When ``target_dim`` and ``x_dims`` are provided and ``homogeneous=True``,
    a dimensional probe selects the exact valid monomials for P
    (with ``dim(P) = -target_dim``).

    Parameters
    ----------
    homogeneous : bool
        When True, use a homogeneous polynomial basis (min_total=degree)
        so that all monomials share the same total degree.  Required for
        dimensional consistency when inputs are unitful.  When False
        (default), use the full basis (min_total=0) which also includes
        constant and lower-degree terms.
    """
    from .stageB import _collect_all_atoms

    if target.kind.lower() != "nn":
        return []

    var_idxs = tuple(int(i) for i in target.var_idxs)
    dim = len(var_idxs)
    if dim < 1:
        return []

    tag = target.tag
    if tag is None or tag not in reuse:
        return []
    teacher = reuse[tag]

    data = _gather_atom_teacher_data(
        train_loader=train_loader,
        atom=target,
        teacher=teacher,
        device=device,
        dtype=dtype,
        max_points=max_points,
    )
    if data is None:
        return []
    X, F = data
    if X.numel() == 0 or F.numel() == 0:
        return []

    X = X.to(dtype=torch.float64)
    F = F.to(dtype=torch.float64).view(-1)
    N, dim = X.shape
    if N < min_points:
        return []

    # Restrict to a region where f has a consistent sign so that
    # 1/f is well-defined.  For globally negative leaves we flip
    # the sign and work with |f|.
    mask, sign = _select_sign_region(F, min_points=min_points, eps=eps)
    if mask is None:
        return []
    X = X[mask]
    F = sign * F[mask]

    # ── Dimensional degree probe for 1/P: dim(P) = -target_dim ──
    # Gate on dimensional data availability, not on `homogeneous`.
    _use_dim_probe = (
        target_dim is not None
        and x_dims is not None
        and len(x_dims) == dim
    )
    _probe_exps = None
    if _use_dim_probe:
        from fractions import Fraction as _Fr
        from .ratpoly_degree_probe import probe_poly_exponents
        _neg_dim = tuple(-_Fr(x) for x in target_dim)
        _probe_exps = probe_poly_exponents(_neg_dim, x_dims, max_degree=max_degree)

    # Fit P(x) ≈ 1/f(x), then evaluate f_hat = 1/P_hat
    inv_F = 1.0 / torch.clamp(F.abs(), min=eps) * sign
    scale_F = float(torch.sqrt(torch.mean(F * F)))

    if _probe_exps is not None:
        _n_probe = sum(len(v) for v in _probe_exps.values())
        print(
            f"[Stage B] inv_poly: using degree-probe monomials, "
            f"{_n_probe} valid terms, vars={var_idxs}"
        )

    accepted_trials: List[Dict[str, Any]] = []
    best_reject = None
    for deg in range(1, max_degree + 1):
        # Use probe-filtered exponents when available.
        if _probe_exps is not None:
            exps_list = []
            for k in sorted(_probe_exps):
                if k <= deg:
                    exps_list.extend(_probe_exps[k])
            _mt = min(_probe_exps) if _probe_exps else 0
            if not exps_list:
                continue  # no valid monomials at this degree
        else:
            _mt = deg if homogeneous else 0
            exps_list = _enumerate_exponents(dim, deg, min_total=_mt)
        exps = torch.tensor(exps_list, dtype=torch.int64, device=X.device)
        Phi = _eval_monomials(X, exps)

        try:
            coeffs = (torch.linalg.pinv(Phi) @ inv_F.unsqueeze(1)).squeeze(1)
        except RuntimeError:
            continue

        P_hat = (Phi @ coeffs).view(-1)
        P_safe = P_hat.clone()
        P_safe[P_safe.abs() < eps] = eps
        F_hat = sign / P_safe
        resid = F - F_hat
        if scale_F > 0:
            rel_rms = float(torch.sqrt(torch.mean(resid * resid)) / scale_F)
        else:
            rel_rms = float("inf")

        trial = {
            "degree": int(deg),
            "min_total": int(_mt),
            "coeffs": coeffs.detach().cpu().clone(),
            "rel_rms": float(rel_rms),
            "n_terms": int(len(exps_list)),
        }
        if rel_rms <= rel_rms_threshold:
            accepted_trials.append(trial)
        elif best_reject is None or rel_rms < best_reject["rel_rms"]:
            best_reject = trial

    if not accepted_trials:
        if best_reject is not None:
            print(
                f"[Stage B] inv_poly rejected: best rel_rms={best_reject['rel_rms']:.4f} > "
                f"{rel_rms_threshold}, degree={best_reject['degree']}, vars={target.var_idxs}"
            )
        return []

    print(
        f"[Stage B] inv_poly probe: {len(accepted_trials)} sub-threshold trial(s) for vars={target.var_idxs}: "
        + ", ".join(
            f"deg={t['degree']} rms={t['rel_rms']:.4f} n={t['n_terms']}"
            for t in accepted_trials
        )
    )

    target_var_idxs = var_idxs
    target_inputs = get_input_exprs(target)
    results: List[Tuple[Node, callable, Dict[str, Any]]] = []
    for trial in accepted_trials:
        deg = int(trial["degree"])
        mt = int(trial["min_total"]) if homogeneous else 0
        coeffs_cpu = trial["coeffs"]
        rel_rms = float(trial["rel_rms"])

        poly_kwargs: Dict[str, object] = {"degree": deg, "min_total": mt}
        poly_atom = AtomNode(
            kind="poly",
            var_idxs=target_var_idxs,
            kwargs=poly_kwargs,
            tag=None,
            inputs=clone_inputs(target),
        )
        new_subtree = PowNode(base=poly_atom, exponent=-1.0)
        cand_root = _replace_node(root, target, new_subtree)

        def _make_custom_init(
            _deg=deg,
            _mt=mt,
            _coeffs_cpu=coeffs_cpu,
            _rms=rel_rms,
        ):
            def _custom_init(root_inner: Node, model_inner: torch.nn.Module):
                """Copy fitted polynomial coefficients into the new PolyLeaf."""
                atoms = _collect_all_atoms(root_inner)
                leaves = list(model_inner.leaf)
                poly_core = _find_matching_core(
                    atoms,
                    leaves,
                    core_types=PolyLeaf,
                    expected_kind="poly",
                    expected_inputs=target_inputs,
                    predicate=lambda _atom, core: (
                        int(getattr(core, "degree", -1)) == _deg
                        and int(getattr(core, "min_total", 0)) == _mt
                    ),
                )

                if poly_core is None:
                    print(
                        f"[Stage B custom_init inv-poly] No PolyLeaf found for vars {target_var_idxs}, "
                        f"degree={_deg}, min_total={_mt}"
                    )
                    return

                dev = poly_core.coeffs.device
                dt = poly_core.coeffs.dtype
                with torch.no_grad():
                    if _coeffs_cpu.numel() == poly_core.coeffs.numel():
                        poly_core.coeffs.copy_(_coeffs_cpu.to(device=dev, dtype=dt))
                    else:
                        print(
                            "[Stage B custom_init inv-poly] Coeff count mismatch; "
                            f"got {_coeffs_cpu.numel()}, expected {poly_core.coeffs.numel()}"
                        )
                        return
                print(
                    f"[Stage B custom_init inv-poly] Init on vars {target_var_idxs}: "
                    f"degree={_deg}, rel_rms≈{_rms:.2e}"
                )

            # Hint for numerically-stable fitting: (1/u) -> fit u via recip-link first.
            _custom_init._fit_lift_link = "recip"
            return _custom_init

        results.append((
            cand_root,
            _make_custom_init(),
            {
                "pattern_family": "inv_poly",
                "degree": deg,
                "precheck_rel_rms": rel_rms,
                "probe_rms_rel": rel_rms,
                "log": (
                    f"[Stage B]  Trying inv_poly degree={deg} (1/P(x)) rewrite on "
                    f"NN leaf vars {target.var_idxs}"
                ),
            },
        ))

    return results


def _build_inv_poly_candidate(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    max_degree: int = 2,
    min_points: int = 400,
    max_points: int = 5000,
    rel_rms_threshold: float = 1e-3,
    eps: float = 1e-8,
    homogeneous: bool = False,
    target_dim: Optional[tuple] = None,
    x_dims: Optional[List[tuple]] = None,
) -> Tuple[Optional[Node], Optional[callable]]:
    """Backward-compat wrapper: returns the first (simplest) inv_poly candidate."""
    results = _build_inv_poly_candidates(
        root=root,
        target=target,
        reuse=reuse,
        train_loader=train_loader,
        device=device,
        dtype=dtype,
        max_degree=max_degree,
        min_points=min_points,
        max_points=max_points,
        rel_rms_threshold=rel_rms_threshold,
        eps=eps,
        homogeneous=homogeneous,
        target_dim=target_dim,
        x_dims=x_dims,
    )
    if results:
        cand_root, init_fn, _meta = results[0]
        return cand_root, init_fn
    return None, None


def _build_poly_split_from_subtree_separability(
    root: Node,
    u_node: AtomNode,  # The polynomial atom being split
    model: torch.nn.Module,  # Full composite model
    op: torch.op,  # torch.add or torch.multiply
    group1_global: List[int],
    group2_global: List[int],
    device: torch.device,
    dtype: torch.dtype,
    rel_coeff_tol: float = 1e-10,
) -> Tuple[Optional[Node], Optional[callable]]:
    """
    Build polynomial split candidate with coefficient transfer for SubtreeSeparability.

    When SubtreeSeparability detects that a polynomial atom is separable, this function
    creates polynomial atoms (not NN atoms) for each variable group and transfers
    fitted coefficients from the original polynomial to preserve the functional form.

    Parameters
    ----------
    root : Node
        The full AST root
    u_node : AtomNode
        The polynomial atom being split (kind='poly')
    model : torch.nn.Module
        The full composite model containing trained parameters
    op : torch.op
        torch.add or torch.multiply from separability detection
    group1_global : List[int]
        Global variable indices for first factor/term
    group2_global : List[int]
        Global variable indices for second factor/term
    device : torch.device
    dtype : torch.dtype
    rel_coeff_tol : float
        Relative tolerance for "non-zero" coefficients

    Returns
    -------
    cand_root : Optional[Node]
        New AST with u_node replaced by split, or None if split not applicable
    custom_init_fn : Optional[callable]
        Function to initialize split polynomials from original coefficients,
        or None if split not applicable
    """
    from .stageB import _collect_all_atoms  # local import to avoid circular dependency

    # Verify this is a polynomial atom
    if u_node.kind.lower() != "poly":
        return None, None

    # Only handle binary additive splits for now
    # Multiplicative polynomial splits require factorization which is complex
    if op is not torch.add:
        return None, None

    if len(group1_global) == 0 or len(group2_global) == 0:
        return None, None

    # Find the polynomial core in the composite model
    atoms = _collect_all_atoms(root)
    leaves = list(model.leaf)
    poly_core: Optional[PolyLeaf] = None

    for atom_i, leaf_mod in zip(atoms, leaves):
        core = getattr(leaf_mod, "core", getattr(leaf_mod, "model", leaf_mod))
        if atom_i is u_node and isinstance(core, PolyLeaf):
            poly_core = core
            break

    if poly_core is None:
        return None, None

    # Extract fitted coefficients and exponents
    exps_full = poly_core.exps.detach().cpu()  # Shape: [n_terms, n_dim]
    coeffs_full = poly_core.coeffs.detach().cpu()  # Shape: [n_terms]
    n_terms, n_dim_original = exps_full.shape

    if n_terms == 0:
        return None, None

    # Get degree from original polynomial
    degree = int(u_node.kwargs.get("degree", u_node.kwargs.get("deg", poly_core.degree)))

    # Build mapping: local dimension index -> global variable index
    # u_node.var_idxs maps local coords (0, 1, ...) to global coords (e.g., 2, 3)
    local_to_global = {i: int(u_node.var_idxs[i]) for i in range(n_dim_original)}

    # Build mapping: global variable index -> which group it belongs to
    global_to_group = {}
    for g_idx in group1_global:
        global_to_group[g_idx] = 0  # Group 0
    for g_idx in group2_global:
        global_to_group[g_idx] = 1  # Group 1

    # Build mapping: global index -> local index within each group
    group1_local_map = {g: i for i, g in enumerate(group1_global)}
    group2_local_map = {g: i for i, g in enumerate(group2_global)}
    group_local_maps = [group1_local_map, group2_local_map]

    # Determine tolerance for "zero" coefficients
    scale = float(coeffs_full.abs().max().item())
    tol = scale * rel_coeff_tol if scale > 0 else rel_coeff_tol

    # Pre-check for cross-terms (terms that involve variables from both groups)
    # For additive split, ANY cross-term makes the function non-separable
    for k in range(n_terms):
        if float(abs(coeffs_full[k]).item()) <= tol:
            continue
        e_full = exps_full[k]  # Local coordinates

        # Check which groups this term involves
        groups_involved = set()
        for local_idx in range(n_dim_original):
            exp_val = int(e_full[local_idx].item())
            if exp_val == 0:
                continue
            global_idx = local_to_global[local_idx]
            group = global_to_group.get(global_idx, None)
            if group is not None:
                groups_involved.add(group)

        # If term involves both groups, it's a cross-term
        if len(groups_involved) > 1:
            # Cross-term detected - function is not additively separable
            # Reject this candidate
            return None, None

    # Create polynomial atoms for each group, inheriting min_total from parent
    _mt_split = int(u_node.kwargs.get("min_total", 0)) if u_node.kwargs else 0
    groups_global = [tuple(group1_global), tuple(group2_global)]
    group_inputs = [
        get_input_exprs(AtomNode(kind="poly", var_idxs=g, kwargs={}))
        for g in groups_global
    ]
    atoms_new = [
        AtomNode(kind="poly", var_idxs=g, kwargs={"degree": degree, "min_total": _mt_split}, tag=None)
        for g in groups_global
    ]

    # Build AddNode for additive split
    new_subtree = AddNode(atoms_new[0], atoms_new[1])

    # Replace u_node in the full AST
    cand_root = _replace_node(root, u_node, new_subtree)

    # Capture data for custom_init (create closure)
    exps_full_captured = exps_full
    coeffs_full_captured = coeffs_full
    local_to_global_captured = local_to_global
    global_to_group_captured = global_to_group
    group_local_maps_captured = group_local_maps
    tol_captured = tol

    def _custom_init(root_inner: Node, model_inner: torch.nn.Module):
        """
        Initialize the split polynomial atoms with coefficients from the original.

        This runs AFTER the new model is built but BEFORE training starts.
        """
        from .stageB import _collect_all_atoms  # re-import inside closure

        # Find the new polynomial cores in the rebuilt model
        atoms2 = _collect_all_atoms(root_inner)
        leaves2 = list(model_inner.leaf)

        cores = [
            _find_matching_core(
                atoms2,
                leaves2,
                core_types=PolyLeaf,
                expected_kind="poly",
                expected_inputs=group_inputs[gi],
            )
            for gi in range(len(groups_global))
        ]

        if any(c is None for c in cores):
            # Something went wrong; bail out
            return

        # Build exponent -> index maps for each new poly
        exp_maps: List[Dict[Tuple[int, ...], int]] = []
        for gi, c in enumerate(cores):
            exps_g = c.exps.detach().cpu()
            m: Dict[Tuple[int, ...], int] = {}
            for k, e in enumerate(exps_g):
                key = tuple(int(x) for x in e.tolist())
                m[key] = k
            exp_maps.append(m)

        # Prepare accumulators for new coefficients
        new_coeffs_cpu = [torch.zeros_like(c.coeffs.detach().cpu()) for c in cores]

        # Map each term from original polynomial to appropriate group
        for k in range(exps_full_captured.shape[0]):
            c_full = coeffs_full_captured[k]
            if float(abs(c_full).item()) <= tol_captured:
                continue

            e_full = exps_full_captured[k]  # Exponents in local coords
            total_deg = int(e_full.sum().item())

            # Handle constant term (attribute to first group)
            if total_deg == 0:
                gi = 0
                key = tuple([0] * len(groups_global[gi]))
                if key in exp_maps[gi]:
                    idx_new = exp_maps[gi][key]
                    new_coeffs_cpu[gi][idx_new] += c_full
                continue

            # Determine which group this term belongs to
            # and build the exponent in that group's local coordinates
            term_group = None
            term_exp_in_group = None

            for local_idx in range(n_dim_original):
                exp_val = int(e_full[local_idx].item())
                if exp_val == 0:
                    continue

                global_idx = local_to_global_captured[local_idx]
                group = global_to_group_captured.get(global_idx, None)

                if group is None:
                    # Variable not in either group - shouldn't happen after pre-check
                    return

                if term_group is None:
                    term_group = group
                    # Initialize exponent vector for this group
                    term_exp_in_group = [0] * len(groups_global[group])

                if term_group != group:
                    # Cross-term - shouldn't happen after pre-check
                    return

                # Map to local coordinate within the group
                group_local_idx = group_local_maps_captured[group][global_idx]
                term_exp_in_group[group_local_idx] = exp_val

            if term_group is not None and term_exp_in_group is not None:
                key = tuple(term_exp_in_group)
                if key in exp_maps[term_group]:
                    idx_new = exp_maps[term_group][key]
                    new_coeffs_cpu[term_group][idx_new] += c_full

        # Copy coefficients into the new cores
        for gi, c in enumerate(cores):
            with torch.no_grad():
                if c.coeffs.shape == new_coeffs_cpu[gi].shape:
                    c.coeffs.copy_(
                        new_coeffs_cpu[gi].to(device=c.coeffs.device, dtype=c.coeffs.dtype)
                    )

    return cand_root, _custom_init


def _build_additive_poly_split_candidate(
    root: Node,
    target: AtomNode,
    model: torch.nn.Module,
    rel_coeff_tol: float = 1e-3,
) -> Tuple[Optional[Node], Optional[callable]]:
    """
    Analyse a multi-D PolyLeaf under `target` and, if its coefficients
    suggest additive separability between disjoint variable blocks,
    rewrite it as a sum of smaller PolyLeaf atoms:

        poly[x_S] -> poly[x_{G1}] + poly[x_{G2}] + ...

    The custom_init_fn then copies the fitted coefficients from the
    original core into the group-specific cores.
    """
    from .stageB import _collect_all_atoms  # local import to avoid circular dependency

    if target.kind.lower() != "poly" or len(target.var_idxs) < 2:
        return None, None

    atoms = _collect_all_atoms(root)
    leaves = list(model.leaf)
    poly_core: Optional[PolyLeaf] = None

    for atom_i, leaf_mod in zip(atoms, leaves):
        core = getattr(leaf_mod, "core", getattr(leaf_mod, "model", leaf_mod))
        if atom_i is target and isinstance(core, PolyLeaf):
            poly_core = core
            break

    if poly_core is None:
        return None, None

    exps_full = poly_core.exps.detach().cpu()
    coeffs_full = poly_core.coeffs.detach().cpu()
    n_terms, dim = exps_full.shape
    if dim < 2 or n_terms == 0:
        return None, None

    # Relative tolerance on "non-zero" coefficients
    scale = float(coeffs_full.abs().max().item())
    tol = scale * rel_coeff_tol if scale > 0 else rel_coeff_tol

    # Mark which axes are actually used in any significant term
    active = [False] * dim
    for k in range(n_terms):
        if float(abs(coeffs_full[k]).item()) <= tol:
            continue
        e = exps_full[k]
        for j in range(dim):
            if int(e[j].item()) > 0:
                active[j] = True

    if sum(active) <= 1:
        # Nothing interesting to split
        return None, None

    # Build adjacency graph: axes that co-occur in a non-negligible term
    adj = [[False] * dim for _ in range(dim)]
    for k in range(n_terms):
        if float(abs(coeffs_full[k]).item()) <= tol:
            continue
        e = exps_full[k]
        idxs = [j for j in range(dim) if int(e[j].item()) > 0]
        if len(idxs) <= 1:
            continue
        for i in idxs:
            for j in idxs:
                if i != j:
                    adj[i][j] = adj[j][i] = True

    # Connected components of this graph define variable groups
    comps: List[List[int]] = []
    seen = [False] * dim
    for j in range(dim):
        if not active[j] or seen[j]:
            continue
        q = [j]
        seen[j] = True
        comp = [j]
        while q:
            u = q.pop()
            for v in range(dim):
                if not seen[v] and adj[u][v]:
                    seen[v] = True
                    q.append(v)
                    comp.append(v)
        comps.append(sorted(comp))

    comps = [c for c in comps if c]
    if len(comps) < 2:
        # Single connected block -> no additive split
        return None, None

    # Map local dimension indices back to global x-indices
    groups_global = [tuple(int(target.var_idxs[j]) for j in comp) for comp in comps]
    group_inputs = [
        get_input_exprs(AtomNode(kind="poly", var_idxs=g, kwargs={}))
        for g in groups_global
    ]

    # Keep the same degree bound and min_total as the original poly
    degree = int(exps_full.sum(dim=1).max().item())
    _mt_asplit = int(target.kwargs.get("min_total", 0)) if target.kwargs else 0

    atoms_new = [
        AtomNode(kind="poly", var_idxs=g, kwargs={"degree": int(degree), "min_total": _mt_asplit}, tag=None)
        for g in groups_global
    ]

    # Build AddNode chain: poly[g0] + poly[g1] + ...
    new_base: Node = atoms_new[0]
    for a in atoms_new[1:]:
        new_base = AddNode(new_base, a)

    cand_root = _replace_node(root, target, new_base)

    # Capture data needed for custom_init
    exps_full_local = exps_full
    coeffs_full_local = coeffs_full
    comps_local = comps
    tol_local = tol

    def _custom_init(root_inner: Node, model_inner: torch.nn.Module):
        from .stageB import _collect_all_atoms  # re-import inside closure

        atoms2 = _collect_all_atoms(root_inner)
        leaves2 = list(model_inner.leaf)

        cores: List[Optional[PolyLeaf]] = [
            _find_matching_core(
                atoms2,
                leaves2,
                core_types=PolyLeaf,
                expected_kind="poly",
                expected_inputs=group_inputs[gi],
            )
            for gi in range(len(groups_global))
        ]

        if any(c is None for c in cores):
            # Something went wrong; bail out silently so candidate just gets LM init
            return

        dim_full = exps_full_local.shape[1]
        dim_to_group = [-1] * dim_full
        for gi, comp in enumerate(comps_local):
            for j in comp:
                dim_to_group[j] = gi

        # Build exponent -> index maps for each new poly
        exp_maps: List[Dict[Tuple[int, ...], int]] = []
        for gi, c in enumerate(cores):
            exps_g = c.exps.detach().cpu()
            m: Dict[Tuple[int, ...], int] = {}
            for k, e in enumerate(exps_g):
                key = tuple(int(x) for x in e.tolist())
                m[key] = k
            exp_maps.append(m)

        new_coeffs_cpu = [torch.zeros_like(c.coeffs.detach().cpu()) for c in cores]

        for k in range(exps_full_local.shape[0]):
            c_full = coeffs_full_local[k]
            if float(abs(c_full).item()) <= tol_local:
                continue
            e_full = exps_full_local[k]
            s = int(e_full.sum().item())

            # Constant term: attribute to first group
            if s == 0:
                gi = 0
                exps_g = cores[gi].exps.detach().cpu()
                for idx, e_g in enumerate(exps_g):
                    if int(e_g.sum().item()) == 0:
                        new_coeffs_cpu[gi][idx] += c_full
                        break
                continue

            idxs = [j for j in range(dim_full) if int(e_full[j].item()) > 0]
            gi = dim_to_group[idxs[0]]
            if gi < 0:
                continue
            if any(dim_to_group[j] != gi for j in idxs):
                # Cross-term between groups: model wasn't truly separable.
                # Reject this candidate by leaving coefficients uninitialised
                return

            comp = comps_local[gi]
            e_loc = [int(e_full[j].item()) for j in comp]
            key = tuple(e_loc)
            m = exp_maps[gi]
            if key not in m:
                # Monomial not representable in this poly degree; skip
                continue
            idx_new = m[key]
            new_coeffs_cpu[gi][idx_new] += c_full

        # Copy coefficients into the new cores
        for gi, c in enumerate(cores):
            with torch.no_grad():
                if c.coeffs.shape == new_coeffs_cpu[gi].shape:
                    c.coeffs.copy_(
                        new_coeffs_cpu[gi].to(device=c.coeffs.device, dtype=c.coeffs.dtype)
                    )

    return cand_root, _custom_init


def _build_power_exp_rat_candidate(
    root: Node,
    target: AtomNode,
    scale_specs: List[ScaleSpec],
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    deg_num: int = 2,
    deg_den: int = 2,
    rel_std_max: float = 0.1,
    min_points: int = 400,
    max_points: int = 5000,
    eps: float = 1e-8,
    max_abs_exponent: float = 20.0,
    max_rel_rms_log: float = 1.0,
) -> Tuple[Optional[Node], Optional[callable]]:
    """
    For a multi-D NN atom, use a group ScaleSpec and teacher samples to
    propose a rewrite:

        f(x) ≈ x_pivot^k * exp( P(x)/Q(x) )

    where k comes from the scaling degree and P/Q is fitted to
    log(g(x)), g = f * x_pivot^{-k}.

    IMPORTANT: the actual rational fit is done *inside* the custom
    initialiser, using the ExpRationalPolyLeaf's own exponent tables
    (core.exps_num / core.exps_den), so we never rely on a guessed
    enumeration order.

    Returns (candidate_root, custom_init_fn) or (None, None) if we
    decide not to propose a rewrite.
    """
    # Import locally to avoid circular dependency
    from .stageB import _collect_all_atoms

    def _match_scale_spec_to_effective_input(
        atom: AtomNode,
        spec: ScaleSpec,
    ) -> Optional[Tuple[int, Node]]:
        input_exprs = get_input_exprs(atom)
        compound_expr = getattr(spec, "compound_expr", None)
        if compound_expr is not None:
            for local_idx, expr in enumerate(input_exprs):
                try:
                    if ast_equals(expr, compound_expr):
                        return int(local_idx), expr
                except Exception:
                    continue
            return None

        spec_indices = tuple(int(j) for j in getattr(spec, "indices", ()) or ())
        if len(spec_indices) != 1:
            return None
        wanted_idx = spec_indices[0]
        for local_idx, expr in enumerate(input_exprs):
            if not isinstance(expr, AtomNode):
                continue
            kind = str(getattr(expr, "kind", "")).lower()
            if kind in ("var", "x", "input") and tuple(int(v) for v in expr.var_idxs) == (wanted_idx,):
                return int(local_idx), expr
        return None

    if target.kind.lower() != "nn":
        return None, None
    eff_arity = int(effective_arity(target))
    if eff_arity < 2:
        return None, None

    # Only keep scaling specs that map cleanly onto one effective input of the leaf.
    group_specs: List[Tuple[ScaleSpec, int, Node]] = []
    for sp in scale_specs:
        match = _match_scale_spec_to_effective_input(target, sp)
        if match is None:
            continue
        group_specs.append((sp, match[0], match[1]))
    if not group_specs:
        return None, None

    spec, pivot_local, pivot_input_expr = min(group_specs, key=lambda item: item[0].rel_std)
    if spec.rel_std > rel_std_max:
        return None, None

    tag = target.tag
    if tag is None or tag not in reuse:
        return None, None
    teacher = reuse[tag]

    data = _gather_atom_teacher_data(
        train_loader=train_loader,
        atom=target,
        teacher=teacher,
        device=device,
        dtype=dtype,
        max_points=max_points,
    )
    if data is None:
        return None, None
    X, F = data  # X: [N, d_atom], F: [N]
    if X.numel() == 0 or F.numel() == 0:
        return None, None

    X = X.to(dtype=torch.float64)
    F = F.to(dtype=torch.float64).view(-1)
    N, dim = X.shape
    if N < min_points or dim != eff_arity:
        return None, None

    # Restrict to region where x_pivot>0 and f>0 so that x^k, log are well-defined
    x_p = X[:, pivot_local]
    mask = (x_p > eps) & (F > eps)
    if mask.sum().item() < min_points:
        return None, None

    Xm = X[mask]
    Fm = F[mask]

    # Exponent from scaling degree, snapped to nearest integer when close
    k_hat = float(spec.k_hat)
    k_round = round(k_hat)
    if abs(k_hat - k_round) < 0.1:
        k_est = float(k_round)
    else:
        k_est = k_hat

    xp_m = Xm[:, pivot_local]
    with torch.no_grad():
        g = Fm * torch.pow(xp_m, -k_est)

    # We'll fit a rational to log g later, inside the custom init, using
    # the ExpRationalPolyLeaf's own exponent tables. For now we just
    # build some cheap diagnostics on log g to decide whether it's worth
    # trying an exp-branch rewrite at all.
    log_g = torch.log(g)  # [N_mask]

    # Quick validation check: ensure log_g has reasonable range for exp safety
    mu = float(log_g.mean().item())
    log_g_c = log_g - mu
    max_abs_log = float(log_g_c.abs().max().item())
    if not math.isfinite(max_abs_log) or max_abs_log > max_abs_exponent:
        # The log data is already too extreme; don't propose this rewrite
        return None, None

    # Optional rational probe: require that log g looks at least somewhat
    # rational, and significantly *more* rational than f itself. This
    # avoids trying exp-branch rewrites on arbitrary junk.
    try:
        rms_rel_log = _rational_probe_nd(
            Xm,
            log_g_c,
            deg_num=2,
            deg_den=2,
            min_points=min_points // 2,
            max_points=min(min_points, 1000),
            dtype=torch.float64,
            filter_outliers=True,
            error_metric="median_rel",
        )
        rms_rel_f = _rational_probe_nd(
            Xm,
            Fm,
            deg_num=1,
            deg_den=1,
            min_points=min_points // 2,
            max_points=min(min_points, 1000),
            dtype=torch.float64,
            filter_outliers=True,
            error_metric="median_rel",
        )
    except Exception:
        rms_rel_log = float("inf")
        rms_rel_f = float("inf")

    # Gate: Only bail out if the log-rational fit is clearly awful or clearly worse than
    # fitting F itself. Otherwise, let Stage B try the candidate and let the validation loss decide.
    if not math.isfinite(rms_rel_log):
        return None, None
    if math.isfinite(rms_rel_f) and rms_rel_log > rms_rel_f:
        # log-fit is strictly worse than fitting F directly; probably not an exp-rational.
        return None, None
    # Adding a very soft absolute cap, to avoid extreme outliers.
    if rms_rel_log > max_rel_rms_log:
        return None, None

    _Xm_cpu = Xm.detach().cpu()
    logg_cpu = log_g.detach().cpu()

    # Build candidate AST
    base_tag = getattr(target, "tag", None)
    power_tag = f"{base_tag}_powexp_power" if base_tag else f"powexp_power_{id(target)}"
    exp_tag = f"{base_tag}_powexp_exp" if base_tag else f"powexp_exp_{id(target)}"
    root_cand = _make_power_exp_ratpoly_rewrite(
        root=root,
        target=target,
        pivot_axis=None,
        exponent=k_est,
        deg_num=deg_num,
        deg_den=deg_den,
        pivot_input_expr=pivot_input_expr,
        power_tag=power_tag,
        exp_tag=exp_tag,
    )

    def _custom_init(root_inner: Node, model_inner: torch.nn.Module):
        """
        Simple initialization for power^k * exp(P/Q):

            PowerLeaf(x_pivot)^k_est  with amp ≈ median(y)
            ExpRationalPolyLeaf(vars_group) with small random coefficients

        We use conservative initialization (small coefficients near zero)
        and let LM find the optimal values.
        """
        atoms = _collect_all_atoms(root_inner)
        leaves = list(model_inner.leaf)
        power_core: Optional[PowerLeaf] = None
        exp_core: Optional[ExpRationalPolyLeaf] = None

        for atom_i, leaf_mod in zip(atoms, leaves):
            core = getattr(leaf_mod, "core", getattr(leaf_mod, "model", leaf_mod))

            if isinstance(core, PowerLeaf) and getattr(atom_i, "tag", None) == power_tag:
                power_core = core
            elif isinstance(core, ExpRationalPolyLeaf) and getattr(atom_i, "tag", None) == exp_tag:
                exp_core = core

        if power_core is None or exp_core is None:
            print(
                f"[Stage B custom_init] Early return: power_core={power_core is not None}, exp_core={exp_core is not None}"
            )
            return

        # Get approximate scale from cached data
        _dev = exp_core.coeffs_num.device
        _dt = exp_core.coeffs_num.dtype
        F_cpu = logg_cpu  # This is log(g) where g = f * x^{-k}
        mu = float(F_cpu.mean().item())  # Mean of log(g)

        # Initialize power leaf with the scaling exponent and amplitude
        with torch.no_grad():
            if hasattr(power_core, "exponent"):
                power_core.exponent.copy_(
                    torch.as_tensor(
                        k_est, dtype=power_core.exponent.dtype, device=power_core.exponent.device
                    )
                )
            if hasattr(power_core, "amp"):
                # Set amplitude to exp(mean(log g)) to get the right overall scale
                power_core.amp.copy_(
                    torch.as_tensor(
                        math.exp(mu), dtype=power_core.amp.dtype, device=power_core.amp.device
                    )
                )

            # Initialize ExpRationalPolyLeaf with small random coefficients
            # Use 0.01 * randn() so the exponent P/Q starts near zero
            _M_num = exp_core.coeffs_num.numel()
            M_den = exp_core.coeffs_den.numel()

            # Numerator: small random values
            exp_core.coeffs_num.normal_(mean=0.0, std=0.01)

            # Denominator: start with [1, small, small, ...] to avoid singularities
            exp_core.coeffs_den.zero_()
            exp_core.coeffs_den[0] = 1.0  # Constant term = 1
            if M_den > 1:
                exp_core.coeffs_den[1:].normal_(mean=0.0, std=0.01)

        print(
            f"[Stage B custom_init] Simple init: power^{k_est:.3f}, amp={math.exp(mu):.3e}, exp(P/Q) with small random coeffs"
        )

    return root_cand, _custom_init


def _build_pure_exp_rat_candidate(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    deg_num: int = 3,
    deg_den: int = 2,
    min_points: int = 200,
    max_points: int = 5000,
    eps: float = 1e-12,
    max_abs_log: float = 60.0,
    max_rel_rms_log: Optional[float] = None,
    improvement_factor: Optional[float] = None,
) -> Tuple[Optional[Node], Optional[callable]]:
    """
    Multi-D analogue of the Gaussian trick when we *don't* have a clean
    scaling law: directly test whether log(leaf(x)) is well-approximated
    by a small rational P/Q in its own variables.

        f(x_S) ≈ exp( P(x_S)/Q(x_S) )

    where S = target.var_idxs. We:
      - gather teacher data for this NN leaf,
      - restrict to f>0, take log f,
      - run a cheap rational probe on log f,
      - if that looks good, propose NN(x_S) -> exp_ratpoly(x_S).
    """
    # Import locally to avoid circular dependency
    from .stageB import _collect_all_atoms

    if target.kind.lower() != "nn":
        return None, None
    var_idxs = tuple(int(i) for i in target.var_idxs)
    if len(var_idxs) < 2:
        # 1D exp-rational is handled elsewhere; this is a multi-D helper.
        return None, None

    tag = target.tag
    if tag is None or tag not in reuse:
        return None, None
    teacher = reuse[tag]

    data = _gather_atom_teacher_data(
        train_loader=train_loader,
        atom=target,
        teacher=teacher,
        device=device,
        dtype=dtype,
        max_points=max_points,
    )
    if data is None:
        return None, None
    X, F = data
    if X.numel() == 0 or F.numel() == 0:
        return None, None

    X = X.to(dtype=torch.float64)
    F = F.to(dtype=torch.float64).view(-1)
    N, dim = X.shape
    if N < min_points:
        return None, None

    # As with the sqrt-branches, allow globally negative leaves by
    # selecting a sign-consistent region and flipping it if needed.
    mask, sign = _select_sign_region(F, min_points=min_points, eps=eps)
    if mask is None:
        print(
            "[Stage B][pure-exp] leaf",
            target.var_idxs,
            "rejected: too few consistently signed points",
        )
        return None, None

    Xm = X[mask]
    Fm = sign * F[mask]
    N_pos = int(Fm.numel())
    logF = torch.log(Fm)

    mu = float(logF.mean().item())
    logF_c = logF - mu
    max_abs = float(logF_c.abs().max().item())
    if (not math.isfinite(max_abs)) or (max_abs > max_abs_log):
        print("[Stage B][pure-exp] leaf", target.var_idxs, "rejected: max_abs_log =", max_abs)
        return None, None

    try:
        rms_rel_log = _rational_probe_nd(
            Xm,
            logF_c,
            deg_num=deg_num,
            deg_den=deg_den,
            min_points=min_points // 2,
            max_points=min(min_points, 1000),
            dtype=torch.float64,
            filter_outliers=True,
            error_metric="median_rel",
        )
        rms_rel_F = _rational_probe_nd(
            Xm,
            Fm,
            deg_num=1,
            deg_den=1,
            min_points=min_points // 2,
            max_points=min(min_points, 1000),
            dtype=torch.float64,
            filter_outliers=True,
            error_metric="median_rel",
        )
    except Exception:
        rms_rel_log = float("inf")
        rms_rel_F = float("inf")

    print(
        "[Stage B][pure-exp] leaf",
        target.var_idxs,
        "N_pos",
        N_pos,
        "max_abs_log",
        max_abs,
        "rms_rel_log",
        rms_rel_log,
        "rms_rel_F",
        rms_rel_F,
    )

    if not math.isfinite(rms_rel_log):
        print("[Stage B][pure-exp] rejected: non-finite rms_rel_log")
        return None, None

    if max_rel_rms_log is not None and rms_rel_log > max_rel_rms_log:
        print("[Stage B][pure-exp] rejected by max_rel_rms_log")
        return None, None

    if (
        improvement_factor is not None
        and math.isfinite(rms_rel_F)
        and rms_rel_log > improvement_factor * rms_rel_F
    ):
        print("[Stage B][pure-exp] rejected by improvement_factor")
        return None, None

    # If we get here, log f looks nicely rational in these variables.
    new_atom = AtomNode(
        kind="exp_ratpoly",
        var_idxs=target.var_idxs,
        kwargs={"deg_num": int(deg_num), "deg_den": int(deg_den)},
        tag=None,  # avoid generic analytic-leaf reuse initialiser
        inputs=clone_inputs(target),
    )
    root_cand = _replace_node(root, target, new_atom)

    # Clip the mean log-value to a safe range; we'll use this as a
    # constant exponent offset in the initialiser.
    mu_clipped = max(-max_abs_log, min(max_abs_log, mu))
    target_var_idxs = var_idxs
    target_inputs = get_input_exprs(target)

    def _custom_init(root_inner: Node, model_inner: torch.nn.Module):
        """
        Conservative initialisation for the pure exp-rational rewrite:

            f(x_S) ≈ exp(P(x_S)/Q(x_S))

        We start from:
          - Q(x_S) ≈ 1 everywhere (denominator constant 1);
          - P(x_S) having a constant term ≈ mean(log f) plus small
            random higher-order terms.

        This keeps exp(P/Q) in a reasonable range and lets LM discover
        the actual structure.
        """
        atoms = _collect_all_atoms(root_inner)
        leaves = list(model_inner.leaf)
        exp_core = _find_matching_core(
            atoms,
            leaves,
            core_types=ExpRationalPolyLeaf,
            expected_kind="exp_ratpoly",
            expected_inputs=target_inputs,
        )

        if exp_core is None:
            print("[Stage B custom_init pure-exp] No ExpRationalPolyLeaf found; aborting.")
            return

        # Identify constant-monomial indices in numerator and denominator,
        # so we can robustly place the offset and unit denominator.
        exps_num = exp_core.exps_num.detach().cpu()
        exps_den = exp_core.exps_den.detach().cpu()
        idx_num_const = None
        idx_den_const = None
        for k, e in enumerate(exps_num):
            if int(e.sum().item()) == 0:
                idx_num_const = k
                break
        for k, e in enumerate(exps_den):
            if int(e.sum().item()) == 0:
                idx_den_const = k
                break

        dev = exp_core.coeffs_num.device
        dt = exp_core.coeffs_num.dtype

        with torch.no_grad():
            # Denominator: start as Q(x) ≈ 1 to avoid singularities.
            exp_core.coeffs_den.zero_()
            if idx_den_const is not None:
                exp_core.coeffs_den[idx_den_const] = 1.0
            elif exp_core.coeffs_den.numel() > 0:
                exp_core.coeffs_den[0] = 1.0

            # Numerator: constant term ≈ mean(log f), plus small noise on
            # higher-order terms so LM has something to work with.
            exp_core.coeffs_num.zero_()
            if idx_num_const is not None:
                exp_core.coeffs_num[idx_num_const] = torch.as_tensor(
                    mu_clipped, dtype=dt, device=dev
                )
            elif exp_core.coeffs_num.numel() > 0:
                exp_core.coeffs_num[0] = torch.as_tensor(mu_clipped, dtype=dt, device=dev)
            if exp_core.coeffs_num.numel() > 1:
                exp_core.coeffs_num[1:].normal_(mean=0.0, std=0.01)

        print(
            f"[Stage B custom_init pure-exp] Simple init on vars {target_var_idxs}: "
            f"exponent≈{mu_clipped:.3g} with small random higher-order terms."
        )

    return root_cand, _custom_init
