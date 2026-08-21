# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Shared Stage-B rule helpers and small standalone rules."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

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
    ExpNode,
    LogNode,
    MulNode,
    Node,
    PowNode,
    SinNode,
    clone_ast,
    effective_arity,
    get_input_exprs,
)
from nestynet_sr.sr_search.candidate_builders import _gather_atom_teacher_data
from nestynet_sr.sr_search.model_selection import (
    noisy_rel_rms_threshold as _noisy_rel_rms_threshold,
    resolve_acceptance_noise_floor_raw as _resolve_acceptance_noise_floor_raw,
)
from nestynet_sr.sr_search.phase_scan import stable_int_hash
from nestynet_sr.sr_search.r1_operator_certificates import (
    R1OperatorCertificate,
    build_r1_certificate_replacement,
    r1_certificate_poly_init,
    scan_r1_operator_certificates,
)

from .engine import Candidate, StageBContext, StageBRule, atom_content_hash, candidate_pattern_name
from .helpers import (
    _collect_all_atoms,
    _collect_univariate_nn_atoms,
    _poly_zero_and_set,
    build_atom_to_leaf_map,
)


def _stageB_noise_floor_raw(ctx: StageBContext) -> float:
    try:
        raw = float(getattr(ctx.state, "acceptance_noise_floor_raw", 0.0) or 0.0)
    except Exception:
        raw = 0.0
    if math.isfinite(raw) and raw > 0.0:
        return float(raw)
    try:
        return float(_resolve_acceptance_noise_floor_raw(ctx.lm_hp, ctx.loss_scale))
    except Exception:
        return 0.0


def _stageB_noisy_rel_rms_threshold(
    ctx: StageBContext,
    base_rel: float,
    *,
    y_rms: Optional[float] = None,
    y_values: Optional[torch.Tensor] = None,
    noise_mult: float = 2.0,
    cap: Optional[float] = None,
) -> float:
    if y_rms is None and y_values is not None:
        try:
            vals = y_values.detach().to(dtype=torch.float64).reshape(-1)
            vals = vals[torch.isfinite(vals)]
            if int(vals.numel()) > 0:
                y_rms = float(torch.sqrt(torch.mean(vals * vals)).item())
        except Exception:
            y_rms = None
    return _noisy_rel_rms_threshold(
        base_rel,
        noise_floor=_stageB_noise_floor_raw(ctx),
        y_rms=y_rms,
        noise_mult=noise_mult,
        cap=cap,
    )


def _effective_input_dims_for_atom(atom: AtomNode, units_spec: Any) -> List[Tuple[Any, ...]]:
    """Return dimensions of the atom's effective input expressions."""
    if units_spec is None:
        return []

    dims: List[Tuple[Any, ...]] = []
    try:
        from nestynet_sr.sr_core.units import eval_analytic_expr_dim

        for inp in get_input_exprs(atom):
            dim = eval_analytic_expr_dim(
                inp,
                units_spec.x_dims,
                # Fixed constants are directly evaluable in AtomNode.inputs.
                # Trainable FreeConst inputs are not compiled as fitted leaves,
                # so do not advertise them as numerically available here.
                fixed_const_dims=getattr(units_spec, "fixed_const_dims", {}) or {},
            )
            if dim is None:
                return []
            dims.append(tuple(dim))
    except Exception:
        return []

    return dims


class _ReciprocalCoordinateTeacher(nn.Module):
    """Evaluate an original 1D teacher leaf as a function of reciprocal input."""

    def __init__(self, teacher: nn.Module):
        super().__init__()
        self.teacher = teacher

    def forward(self, x):
        z_inv = x.view(x.shape[0], -1)
        z = torch.reciprocal(z_inv)
        return self.teacher(z)


def _wrap_reuse_for_reciprocal_coordinate(reuse, tag):
    if not isinstance(reuse, dict) or tag is None or tag not in reuse:
        return None
    out = dict(reuse)
    out[tag] = _ReciprocalCoordinateTeacher(reuse[tag])
    return out


class _HomogeneousGaugeTeacher(nn.Module):
    """Evaluate an alternate homogeneous representative from an old ratio leaf."""

    def __init__(self, teacher: nn.Module, transfer_degree: float):
        super().__init__()
        self.teacher = teacher
        self.transfer_degree = float(transfer_degree)

    def forward(self, x):
        u = x.view(x.shape[0], -1)
        z = torch.reciprocal(u)
        y = self.teacher(z)
        return torch.pow(u, self.transfer_degree) * y


def _make_r1_operator_certificate_candidate(
    ctx: StageBContext,
    target: AtomNode,
    cert: R1OperatorCertificate,
) -> Optional[Candidate]:
    """Build a visible Stage-B candidate from a cheap R1 certificate."""

    try:
        inputs = get_input_exprs(target)
    except Exception:
        inputs = ()
    if len(inputs) != 1:
        return None
    z_expr = clone_ast(inputs[0])
    tag_prefix = str(getattr(target, "tag", None) or "r1cert")
    root_new, arg_tag = build_r1_certificate_replacement(
        ctx.state.root,
        target,
        z_expr,
        cert,
        tag_prefix=tag_prefix,
    )
    if root_new is None or not arg_tag:
        return None

    coeffs = r1_certificate_poly_init(cert)

    def _init(root_new_inner: Node, model_new: nn.Module, *, _tag=arg_tag, _coeffs=coeffs):
        atom_to_leaf = build_atom_to_leaf_map(root_new_inner, model_new)
        for atom in _collect_all_atoms(root_new_inner):
            if not isinstance(atom, AtomNode):
                continue
            if getattr(atom, "tag", None) != _tag:
                continue
            leaf = atom_to_leaf.get(id(atom), None)
            if leaf is None:
                continue
            try:
                _poly_zero_and_set(leaf, dict(_coeffs))
            except Exception:
                pass

    _init._after_analytic_init = True
    inv = str(cert.inverse_kind)
    if inv == "sqrt":
        _init._fit_lift_link = "square"
    elif inv == "invsqrt":
        _init._fit_lift_link = "inv_square"
    elif inv == "reciprocal":
        _init._fit_lift_link = "recip"
    elif inv == "exp":
        _init._fit_lift_link = "log"

    label = str(cert.label)
    meta = {
        "structural": True,
        "pattern": "r1_operator_certificate",
        "pattern_family": "r1_operator_certificate",
        "r1_operator_certificate": True,
        "r1_transform": str(cert.transform_name),
        "r1_inverse": inv,
        "r1_psi_power": float(cert.psi_power),
        "screen_rel_rms": float(cert.rel_rms),
        "domain_ok_frac": float(cert.inverse_domain_frac),
        "branch_ok_frac": float(cert.branch_ok_frac),
        "signature": (
            int(atom_content_hash(target)),
            stable_int_hash("r1_operator_certificate"),
            stable_int_hash(label),
            stable_int_hash(inv),
            int(round(float(cert.psi_power) * 1.0e6)),
            int(round(float(cert.affine_a) * 1.0e6)),
            int(round(float(cert.affine_b) * 1.0e6)),
        ),
        "log": (
            f"[Stage B R1Cert] Trying {label}: "
            f"{cert.transform_name}(y)≈{cert.affine_a:.6g}*psi(z)+{cert.affine_b:.6g}, "
            f"inverse={inv}, rel={cert.rel_rms:.2e}, "
            f"dom={cert.inverse_domain_frac:.3f}, branch={cert.branch_ok_frac:.3f}"
        ),
    }
    return Candidate(label=label, root=root_new, init_fn=_init, meta=meta)


class RuleR1OperatorCertificate(StageBRule):
    """Cheap visible operator closures for univariate NN atoms.

    This rule never accepts a transformed relation directly.  It only turns a
    strong ``phi(y) ~= a*psi(z)+b`` certificate into an ordinary visible AST
    candidate, then lets Stage-B fitting and acceptance decide.
    """

    name = "r1_operator_certificate"
    exhaustive = True

    def iter_targets(self, ctx: StageBContext):
        return _collect_univariate_nn_atoms(ctx.state.root)

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        if (
            not isinstance(target, AtomNode)
            or str(getattr(target, "kind", "")).lower() != "nn"
            or effective_arity(target) != 1
        ):
            return []
        tag = getattr(target, "tag", None)
        if tag is None or not isinstance(getattr(ctx.state, "reuse", None), dict):
            return []
        teacher = ctx.state.reuse.get(tag)
        if teacher is None:
            return []

        try:
            max_points = int(getattr(ctx.lm_hp, "stageB_r1_operator_cert_max_points", 5000) or 5000)
        except Exception:
            max_points = 5000
        data = _gather_atom_teacher_data(
            ctx.train_loader_probe,
            target,
            teacher,
            ctx.device,
            ctx.dtype,
            max_points=max_points,
        )
        if data is None:
            return []
        X_atom, F = data
        if X_atom.ndim != 2 or int(X_atom.shape[1]) != 1:
            return []

        try:
            max_rel = float(getattr(ctx.lm_hp, "stageB_r1_operator_cert_rel_rms", 2.0e-3))
        except Exception:
            max_rel = 2.0e-3
        try:
            min_domain = float(getattr(ctx.lm_hp, "macro_domain_ok_frac", 0.98))
        except Exception:
            min_domain = 0.98
        certs = scan_r1_operator_certificates(
            X_atom[:, 0],
            F,
            max_results=8,
            rel_rms_max=max_rel,
            min_domain_frac=min_domain,
            min_branch_frac=min_domain,
            min_points=128,
        )
        cands: List[Candidate] = []
        for cert in certs:
            cand = _make_r1_operator_certificate_candidate(ctx, target, cert)
            if cand is not None:
                cands.append(cand)
        return cands


def _mark_reciprocal_coordinate_candidate(
    cand: Candidate,
    *,
    reuse_override,
    reuses_override,
) -> Candidate:
    meta = dict(cand.meta) if isinstance(getattr(cand, "meta", None), dict) else {}
    family = candidate_pattern_name(cand)
    if family:
        meta.setdefault("pattern_family", family)
    meta["coordinate_variant"] = "z_inv"
    meta["coordinate_variant_display"] = "1/z"
    if reuse_override is not None:
        meta["_reuse_override"] = reuse_override
    if reuses_override is not None:
        meta["_reuses_override"] = reuses_override
    log_msg = meta.get("log", None)
    if isinstance(log_msg, str) and log_msg:
        meta["log"] = log_msg + " [coord=1/z]"
    else:
        meta["log"] = f"[Stage B]  Trying reciprocal-coordinate rewrite ({cand.label}) [coord=1/z]"
    cand.meta = meta
    cand.label = f"{cand.label}[z_inv]"
    return cand


_MONOMIAL_DEGREES = (1, 2, 3)
_HALF_POWER_SCREEN_REL_RMS_MAX = 1.0e-3
_INTEGER_POWER_SCREEN_REL_RMS_MAX = 1.0e-3
_INTEGER_POWER_SCREEN_MAX_POWER = 6
_RECIPROCAL_ALIAS_REPEAT_FAMILIES = {
    "polylog",
    "ratpoly_1d",
    "sqrt_ratpoly_1d",
    "log_ratpoly",
    "exp_rat",
}


def _subtree_content_hash(node: Node) -> int:
    """Content hash of any subtree (ignores AtomNode.tag)."""
    if isinstance(node, AtomNode):
        return atom_content_hash(node)
    if isinstance(node, ConstNode):
        v = node.value
        if isinstance(v, complex):
            return hash(("constC", float(v.real), float(v.imag)))
        return hash(("const", float(v)))
    if isinstance(node, AddNode):
        return hash(("add", _subtree_content_hash(node.left), _subtree_content_hash(node.right)))
    if isinstance(node, MulNode):
        return hash(("mul", _subtree_content_hash(node.left), _subtree_content_hash(node.right)))
    if isinstance(node, PowNode):
        return hash(("pow", _subtree_content_hash(node.base), float(node.exponent)))
    if isinstance(node, LogNode):
        return hash(("log", _subtree_content_hash(node.arg)))
    if isinstance(node, ExpNode):
        return hash(("exp", _subtree_content_hash(node.arg)))
    if isinstance(node, SinNode):
        return hash(("sin", _subtree_content_hash(node.arg)))
    if isinstance(node, CosNode):
        return hash(("cos", _subtree_content_hash(node.arg)))
    if isinstance(node, AsinNode):
        return hash(("asin", _subtree_content_hash(node.arg)))
    if isinstance(node, AcosNode):
        return hash(("acos", _subtree_content_hash(node.arg)))
    if isinstance(node, AtanNode):
        return hash(("atan", _subtree_content_hash(node.arg)))
    from nestynet_sr.sr_core.bridges import (
        AbsNode,
        ArgNode,
        ConjNode,
        ImagNode,
        RealNode,
    )

    for _cls, _name in (
        (AbsNode, "abs"),
        (ConjNode, "conj"),
        (RealNode, "real"),
        (ImagNode, "imag"),
        (ArgNode, "carg"),
    ):
        if isinstance(node, _cls):
            return hash((_name, _subtree_content_hash(node.arg)))
    raise TypeError(f"Unexpected node type in AST: {type(node)}")


def _reciprocal_alias_repeat_reason(cand: Candidate) -> Optional[str]:
    """Return why an op(1/z) candidate repeats an existing op(z) family."""
    label = str(getattr(cand, "label", "") or "")
    family = str(candidate_pattern_name(cand) or "")
    if label.startswith("inv_monomial_deg"):
        return "covered-by-monomial"
    if family in _RECIPROCAL_ALIAS_REPEAT_FAMILIES:
        return f"closed-under-reciprocal:{family}"
    return None


def _reciprocal_alias_base_label(label: str) -> str:
    label = str(label)
    suffix = "[z_inv]"
    return label[: -len(suffix)] if label.endswith(suffix) else label


def _merge_reciprocal_aliases_pairwise(
    base_cands: List[Candidate],
    alias_cands: List[Candidate],
) -> List[Candidate]:
    """Place surviving op(1/z) aliases next to their op(z) counterpart."""
    if not alias_cands:
        return base_cands

    by_base: Dict[str, List[Candidate]] = {}
    for cand in alias_cands:
        by_base.setdefault(_reciprocal_alias_base_label(str(cand.label)), []).append(cand)

    merged: List[Candidate] = []
    for cand in base_cands:
        merged.append(cand)
        merged.extend(by_base.pop(str(cand.label), []))

    for remaining in by_base.values():
        merged.extend(remaining)
    return merged
