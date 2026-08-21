# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import hashlib

import torch

from nestynet_sr.adaptors.fixed_shift import FixedOutputShiftAdaptor
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
    effective_arity,
)
from nestynet_sr.sr_search.candidate_builders import _build_atom_input_tensor


@torch.no_grad()
def _sample_x(train_loader, max_points=4096, device=None, dtype=None):
    xs = []
    n = 0
    for xb, _yb in train_loader:
        if device is not None:
            xb = xb.to(device)
        if dtype is not None:
            xb = xb.to(dtype)
        xs.append(xb)
        n += xb.shape[0]
        if n >= max_points:
            break
    return torch.cat(xs, dim=0)[:max_points]


def _compute_leaf_input(atom, x):
    """Compute correct input for an atom, handling compound variables."""
    return _build_atom_input_tensor(atom, x)


@torch.no_grad()
def gauge_fix_additive_pairs(
    ast,
    reuse,
    train_loader,
    *,
    device=None,
    dtype=None,
    max_points=4096,
    cancel_ratio_thresh=8.0,
    eps=1e-12,
    log_fn=None,
):
    """
    Balanced-gauge fix for AddNode(left, right) when both are AtomNodes with tags in reuse.
    If medians are strongly canceling, shift the two leaves by +/-delta so their medians match.
    """
    x = _sample_x(train_loader, max_points=max_points, device=device, dtype=dtype)

    def med0(y):
        # y is [N, out] or [N]
        if y.ndim == 1:
            y = y[:, None]
        return torch.median(y, dim=0).values

    def rec(node):
        if isinstance(node, AddNode):
            L, R = node.left, node.right
            if isinstance(L, AtomNode) and isinstance(R, AtomNode):
                if (getattr(L, "tag", None) in reuse) and (getattr(R, "tag", None) in reuse):
                    modL, modR = reuse[L.tag], reuse[R.tag]
                    yL = modL(_compute_leaf_input(L, x))
                    yR = modR(_compute_leaf_input(R, x))
                    mL, mR = med0(yL), med0(yR)
                    num = mL.abs() + mR.abs()
                    den = (mL + mR).abs() + eps
                    ratio = torch.max(num / den).item()
                    if ratio >= cancel_ratio_thresh:
                        delta = 0.5 * (mL - mR)
                        reuse[L.tag] = FixedOutputShiftAdaptor(modL, -delta)
                        reuse[R.tag] = FixedOutputShiftAdaptor(modR, +delta)
                        if log_fn is not None:
                            log_fn(
                                f"[gauge-fix] Add split {L.tag}+{R.tag}: "
                                f"max_cancel_ratio≈{ratio:.2f}, "
                                f"medianL={mL.cpu().numpy()}, medianR={mR.cpu().numpy()}, "
                                f"delta={delta.cpu().numpy()}"
                            )
        # recurse
        if hasattr(node, "left"):
            rec(node.left)
        if hasattr(node, "right"):
            rec(node.right)

    rec(ast)
    return reuse


# ──────────────────────────────────────────────────────────────
# Multiplicative gauge fixing (explicit scale + monic polynomials)
# ──────────────────────────────────────────────────────────────


def _flatten_mul(node):
    """Flatten nested MulNodes into an ordered list of factors."""
    if isinstance(node, MulNode):
        return _flatten_mul(node.left) + _flatten_mul(node.right)
    return [node]


def _rebuild_mul(factors):
    """Rebuild a left-associative multiplication chain from a factor list."""
    if not factors:
        return ConstNode(1.0)
    out = factors[0]
    for f in factors[1:]:
        out = MulNode(out, f)
    return out


def _mul_scale_tag_for_atoms(atoms):
    """Deterministically derive a scale tag from a set of participating atoms."""
    tokens = []
    for a in atoms:
        kind = str(getattr(a, "kind", "")).lower()
        tag = "" if getattr(a, "tag", None) is None else str(a.tag)
        var = ",".join(str(int(i)) for i in getattr(a, "var_idxs", ()) or ())
        kw = getattr(a, "kwargs", None) or {}
        deg = kw.get("degree", kw.get("deg", None))
        mint = kw.get("min_total", None)
        tokens.append(f"{kind}:{tag}:{var}:{deg}:{mint}")
    sig = "|".join(sorted(tokens))
    h = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:12]
    return f"mul_scale_{h}"


def _is_poly_like_mul_factor(atom: AtomNode) -> bool:
    """Whether an AtomNode is eligible for multiplicative monic normalisation.

    Only UNREDUCED poly-like types count — they have trainable gauge freedom.
    Already-reduced types (rpoly, rinv_monomial, rpolylog, rratpoly, rratio_poly)
    have their leading coeff pinned to 1 and need no further normalisation.
    """
    kind = str(getattr(atom, "kind", "")).lower()
    ar = int(effective_arity(atom))

    if kind in ("poly", "polynomial"):
        deg = int((getattr(atom, "kwargs", None) or {}).get("degree", 1))
        if deg == 0:
            return False
        return ar == 1
    if kind in ("polylog", "polylogarithmic", "logpoly"):
        return ar == 1
    if kind in ("ratpoly", "rat_poly", "ratpolynomial"):
        return ar == 1
    if kind in ("inv_monomial", "inverse_monomial", "inv_mono"):
        return ar == 1
    if kind in ("ratio_poly", "ratio_polynomial", "ratiopoly"):
        return ar == 2

    return False




def _is_exp_poly_mul_factor(atom: AtomNode) -> bool:
    """Whether an AtomNode is an exp(poly)-like factor eligible for constant pinning.

    We treat exp(poly) leaves as having a multiplicative gauge via the constant
    term in the exponent:
        exp(c0 + Q(x)) = exp(c0) * exp(Q(x)).

    The reduced form (rexp_poly) pins that constant to 0 and expects an
    explicit multiplicative scale leaf to absorb exp(c0).

    Only UNREDUCED exp types count — already-reduced types (rexp_poly etc.)
    have their exponent constant already pinned to 0 and need no further
    reduction.
    """
    kind = str(getattr(atom, "kind", "")).lower()
    ar = int(effective_arity(atom))

    if kind in ("exp", "exp_poly", "expquad", "exp_poly_leaf"):
        # We only apply this reduction when the exponential actually depends on
        # at least one input (arity >= 1). For a 0-input exp(const) leaf, the
        # constant is the whole function.
        return ar >= 1

    return False

def _to_reduced_poly_kind(kind: str) -> str:
    k = str(kind).lower()
    if k in ("poly", "polynomial"):
        return "rpoly"
    if k in ("polylog", "polylogarithmic", "logpoly"):
        return "rpolylog"
    if k in ("ratpoly", "rat_poly", "ratpolynomial"):
        return "rratpoly"
    if k in ("ratio_poly", "ratio_polynomial", "ratiopoly"):
        return "rratio_poly"
    if k in ("inv_monomial", "inverse_monomial", "inv_mono"):
        return "rinv_monomial"
    return k




def _to_reduced_exp_kind(kind: str) -> str:
    k = str(kind).lower()
    if k in ("exp", "exp_poly", "expquad", "exp_poly_leaf"):
        return "rexp_poly"
    return k

def gauge_fix_multiplicative(root: Node) -> Node:
    """Rewrite MulNode chains to tame multiplicative gauge freedom.

    Behaviour:
      - If a multiplication chain contains >=2 polynomial-like factors, insert
        a single explicit dimensionless scale leaf (kind='scale').
      - Convert those polynomial-like factors to their monic / reduced variants
        (rpoly, rpolylog, rratpoly, rratio_poly), fixing the leading coefficient
        to 1 and removing one free parameter.
      - Tag participating factors with kwargs['_mul_scale_tag']=<scale_tag> so
        initialisers can push extracted leading coefficients into the scale.

    Notes
    -----
    - This pass is structural and does not touch parameters; it is safe to run
      before model construction.
    - The rewrite is applied recursively, so multiplicative sub-expressions are
      also normalised.
    """

    def rec(node):
        # Leaf nodes
        if isinstance(node, (AtomNode, ConstNode)):
            return node

        # Binary nodes
        if isinstance(node, AddNode):
            return AddNode(rec(node.left), rec(node.right))
        if isinstance(node, MulNode):
            # Recurse first
            left = rec(node.left)
            right = rec(node.right)
            flat = _flatten_mul(MulNode(left, right))

            # Apply normalisation at this multiplication level
            poly_atoms = [f for f in flat if isinstance(f, AtomNode) and _is_poly_like_mul_factor(f)]
            exp_atoms = [f for f in flat if isinstance(f, AtomNode) and _is_exp_poly_mul_factor(f)]

            # We only introduce a dedicated multiplicative scale when there are
            # >=2 polynomial-like factors.
            need_scale = (len(poly_atoms) >= 2)

            # If a scale leaf already exists (user inserted, or prior pass), we can
            # still reduce exp(poly) factors by pinning the exponent constant.
            scale_atoms = [
                f
                for f in flat
                if isinstance(f, AtomNode)
                and str(getattr(f, "kind", "")).lower() == "scale"
            ]
            has_scale = bool(scale_atoms)

            # Nothing to do if we neither need nor already have a scale.
            if (not need_scale) and (not has_scale):
                return _rebuild_mul(flat)

            # Detect or create scale atom
            if has_scale:
                scale_atom0 = scale_atoms[0]
                tag_basis = poly_atoms if poly_atoms else exp_atoms
                scale_tag = getattr(scale_atom0, "tag", None) or _mul_scale_tag_for_atoms(tag_basis)
                # If an existing scale lacks a tag, materialise a canonical scale atom
                # with a deterministic tag so reduced factors can target it.
                if getattr(scale_atom0, "tag", None) is None:
                    scale_atom = AtomNode(kind="scale", var_idxs=(), kwargs={"init": 1.0, "name": "s"}, tag=scale_tag)
                else:
                    scale_atom = scale_atom0
            else:
                # need_scale must be True here.
                tag_basis = poly_atoms if poly_atoms else exp_atoms
                scale_tag = _mul_scale_tag_for_atoms(tag_basis)
                scale_atom = AtomNode(kind="scale", var_idxs=(), kwargs={"init": 1.0, "name": "s"}, tag=scale_tag)

            # Rewrite factors:
            #   - poly-like -> reduced poly kinds (only when need_scale is True)
            #   - exp(poly) -> rexp_poly (when a scale is present/introduced)
            # and annotate with the shared scale tag.
            new_factors = []
            for f in flat:
                # Remove any existing scale leaves; we re-insert deterministically at the front.
                if isinstance(f, AtomNode) and str(getattr(f, "kind", "")).lower() == "scale":
                    continue

                if isinstance(f, AtomNode) and _is_poly_like_mul_factor(f) and need_scale:
                    new_kind = _to_reduced_poly_kind(f.kind)
                    kw = dict(getattr(f, "kwargs", None) or {})
                    kw["_mul_scale_tag"] = scale_tag
                    new_factors.append(
                        AtomNode(kind=new_kind, var_idxs=f.var_idxs, kwargs=kw,
                                 tag=getattr(f, "tag", None),
                                 inputs=getattr(f, "inputs", None))
                    )
                    continue

                if isinstance(f, AtomNode) and _is_exp_poly_mul_factor(f):
                    new_kind = _to_reduced_exp_kind(f.kind)
                    kw = dict(getattr(f, "kwargs", None) or {})
                    kw["_mul_scale_tag"] = scale_tag
                    new_factors.append(
                        AtomNode(kind=new_kind, var_idxs=f.var_idxs, kwargs=kw,
                                 tag=getattr(f, "tag", None),
                                 inputs=getattr(f, "inputs", None))
                    )
                    continue

                new_factors.append(f)

            # Put scale first for readability / determinism
            new_chain = [scale_atom] + new_factors
            return _rebuild_mul(new_chain)

        # Unary nodes
        if isinstance(node, PowNode):
            return PowNode(rec(node.base), node.exponent)
        if isinstance(node, LogNode):
            return LogNode(rec(node.arg))
        if isinstance(node, ExpNode):
            return ExpNode(rec(node.arg))
        if isinstance(node, SinNode):
            return SinNode(rec(node.arg))
        if isinstance(node, CosNode):
            return CosNode(rec(node.arg))

        # Unknown node type: attempt generic traversal if it looks like a node
        # (keeps this pass resilient to new node types).
        if hasattr(node, "arg") and isinstance(getattr(node, "arg"), tuple) is False:
            try:
                arg = rec(getattr(node, "arg"))
                return node.__class__(arg)
            except Exception:
                return node
        if hasattr(node, "left") and hasattr(node, "right"):
            try:
                return node.__class__(rec(getattr(node, "left")), rec(getattr(node, "right")))
            except Exception:
                return node

        return node

    return rec(root)
