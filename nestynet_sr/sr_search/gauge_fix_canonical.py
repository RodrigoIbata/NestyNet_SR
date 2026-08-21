# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import hashlib

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


def _flatten_mul(node):
    if isinstance(node, MulNode):
        return _flatten_mul(node.left) + _flatten_mul(node.right)
    return [node]


def _rebuild_mul(factors):
    if not factors:
        return ConstNode(1.0)
    out = factors[0]
    for factor in factors[1:]:
        out = MulNode(out, factor)
    return out


def _mul_scale_tag_for_atoms(atoms):
    tokens = []
    for atom in atoms:
        kind = str(getattr(atom, "kind", "")).lower()
        tag = "" if getattr(atom, "tag", None) is None else str(atom.tag)
        var = ",".join(str(int(idx)) for idx in getattr(atom, "var_idxs", ()) or ())
        kwargs = getattr(atom, "kwargs", None) or {}
        degree = kwargs.get("degree", kwargs.get("deg", None))
        min_total = kwargs.get("min_total", None)
        tokens.append(f"{kind}:{tag}:{var}:{degree}:{min_total}")
    signature = "|".join(sorted(tokens))
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    return f"mul_scale_{digest}"


def _is_poly_like_mul_factor(atom: AtomNode) -> bool:
    kind = str(getattr(atom, "kind", "")).lower()
    arity = int(effective_arity(atom))
    if kind in ("poly", "polynomial"):
        degree = int((getattr(atom, "kwargs", None) or {}).get("degree", 1))
        if degree == 0:
            return False
        return arity == 1
    if kind in ("polylog", "polylogarithmic", "logpoly"):
        return arity == 1
    if kind in ("ratpoly", "rat_poly", "ratpolynomial"):
        return arity == 1
    if kind in ("inv_monomial", "inverse_monomial", "inv_mono"):
        return arity == 1
    if kind in ("ratio_poly", "ratio_polynomial", "ratiopoly"):
        return arity == 2
    return False


def _is_exp_poly_mul_factor(atom: AtomNode) -> bool:
    kind = str(getattr(atom, "kind", "")).lower()
    arity = int(effective_arity(atom))
    if kind in ("exp", "exp_poly", "expquad", "exp_poly_leaf"):
        return arity >= 1
    return False


def _to_reduced_poly_kind(kind: str) -> str:
    token = str(kind).lower()
    if token in ("poly", "polynomial"):
        return "rpoly"
    if token in ("polylog", "polylogarithmic", "logpoly"):
        return "rpolylog"
    if token in ("ratpoly", "rat_poly", "ratpolynomial"):
        return "rratpoly"
    if token in ("ratio_poly", "ratio_polynomial", "ratiopoly"):
        return "rratio_poly"
    if token in ("inv_monomial", "inverse_monomial", "inv_mono"):
        return "rinv_monomial"
    return token


def _to_reduced_exp_kind(kind: str) -> str:
    token = str(kind).lower()
    if token in ("exp", "exp_poly", "expquad", "exp_poly_leaf"):
        return "rexp_poly"
    return token


def gauge_fix_multiplicative(root: Node) -> Node:
    def rec(node):
        if isinstance(node, (AtomNode, ConstNode)):
            return node
        if isinstance(node, AddNode):
            return AddNode(rec(node.left), rec(node.right))
        if isinstance(node, MulNode):
            left = rec(node.left)
            right = rec(node.right)
            flat = _flatten_mul(MulNode(left, right))
            poly_atoms = [factor for factor in flat if isinstance(factor, AtomNode) and _is_poly_like_mul_factor(factor)]
            exp_atoms = [factor for factor in flat if isinstance(factor, AtomNode) and _is_exp_poly_mul_factor(factor)]
            need_scale = len(poly_atoms) >= 2
            scale_atoms = [
                factor
                for factor in flat
                if isinstance(factor, AtomNode) and str(getattr(factor, "kind", "")).lower() == "scale"
            ]
            has_scale = bool(scale_atoms)
            if (not need_scale) and (not has_scale):
                return _rebuild_mul(flat)
            if has_scale:
                scale_atom0 = scale_atoms[0]
                tag_basis = poly_atoms if poly_atoms else exp_atoms
                scale_tag = getattr(scale_atom0, "tag", None) or _mul_scale_tag_for_atoms(tag_basis)
                if getattr(scale_atom0, "tag", None) is None:
                    scale_atom = AtomNode(kind="scale", var_idxs=(), kwargs={"init": 1.0, "name": "s"}, tag=scale_tag)
                else:
                    scale_atom = scale_atom0
            else:
                tag_basis = poly_atoms if poly_atoms else exp_atoms
                scale_tag = _mul_scale_tag_for_atoms(tag_basis)
                scale_atom = AtomNode(kind="scale", var_idxs=(), kwargs={"init": 1.0, "name": "s"}, tag=scale_tag)
            new_factors = []
            for factor in flat:
                if isinstance(factor, AtomNode) and str(getattr(factor, "kind", "")).lower() == "scale":
                    continue
                if isinstance(factor, AtomNode) and _is_poly_like_mul_factor(factor) and need_scale:
                    kwargs = dict(getattr(factor, "kwargs", None) or {})
                    kwargs["_mul_scale_tag"] = scale_tag
                    new_factors.append(
                        AtomNode(
                            kind=_to_reduced_poly_kind(factor.kind),
                            var_idxs=factor.var_idxs,
                            kwargs=kwargs,
                            tag=getattr(factor, "tag", None),
                            inputs=getattr(factor, "inputs", None),
                        )
                    )
                    continue
                if isinstance(factor, AtomNode) and _is_exp_poly_mul_factor(factor):
                    kwargs = dict(getattr(factor, "kwargs", None) or {})
                    kwargs["_mul_scale_tag"] = scale_tag
                    new_factors.append(
                        AtomNode(
                            kind=_to_reduced_exp_kind(factor.kind),
                            var_idxs=factor.var_idxs,
                            kwargs=kwargs,
                            tag=getattr(factor, "tag", None),
                            inputs=getattr(factor, "inputs", None),
                        )
                    )
                    continue
                new_factors.append(factor)
            return _rebuild_mul([scale_atom] + new_factors)
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
        if hasattr(node, "arg") and isinstance(getattr(node, "arg"), tuple) is False:
            try:
                return node.__class__(rec(getattr(node, "arg")))
            except Exception:
                return node
        if hasattr(node, "left") and hasattr(node, "right"):
            try:
                return node.__class__(rec(getattr(node, "left")), rec(getattr(node, "right")))
            except Exception:
                return node
        return node

    return rec(root)


__all__ = ["gauge_fix_multiplicative"]
