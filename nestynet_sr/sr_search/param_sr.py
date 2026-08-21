# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""
Parameter-SR (post-structure): discover derived invariants across datasets.

This module operates on already-fitted per-dataset composites and searches
for low-scatter derived scalar combinations of leaf parameters, e.g.

    p_i * (q_i ** 2) / r_i

that are approximately constant across dataset index i.

The discovered invariants can be used as soft constraints during Class-SR
joint refits.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from nestynet_sr.sr_core.bridges import Node, collect_all_atoms


@dataclass
class ParamScalarRef:
    """Reference to one scalar inside a tagged leaf."""

    key: str
    tag: str
    source: str  # "param" | "buffer" | "metadata"
    flat_index: int
    attr: str = "value"
    meta_key: str = ""


@dataclass
class ParamInvariant:
    """A derived candidate invariant over scalar refs."""

    expr: str
    op: str  # "mul" | "div" | "mul_sq_div"
    a: int
    b: int
    c: int = -1
    values: Tuple[float, ...] = ()
    score: float = float("inf")
    cv: float = float("inf")

    def to_dict(self) -> Dict[str, object]:
        return {
            "expr": self.expr,
            "op": self.op,
            "score": float(self.score),
            "cv": float(self.cv),
            "values": [float(v) for v in self.values],
        }


def _tag_to_leafidx(root: Node) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for idx, atom in enumerate(collect_all_atoms(root)):
        tag = atom.tag or f"leaf{idx}"
        out.setdefault(tag, idx)
    return out


def _leaf_flat_param(leaf: torch.nn.Module, flat_index: int) -> Optional[torch.Tensor]:
    cursor = 0
    for p in leaf.parameters():
        n = int(p.numel())
        if flat_index < cursor + n:
            return p.view(-1)[flat_index - cursor]
        cursor += n
    return None


def _leaf_scalar_from_ref(leaf: torch.nn.Module, ref: ParamScalarRef) -> Optional[torch.Tensor]:
    if ref.source == "param":
        return _leaf_flat_param(leaf, int(ref.flat_index))
    if ref.source == "buffer":
        t = getattr(leaf, ref.attr, None)
        if t is None:
            return None
        if not torch.is_tensor(t):
            t = torch.as_tensor(t)
        if int(t.numel()) <= int(ref.flat_index):
            return None
        return t.view(-1)[int(ref.flat_index)]
    return None


def _metadata_value_float(
    dataset_metadata: Optional[Dict[str, Any]],
    ref: ParamScalarRef,
) -> Optional[float]:
    if dataset_metadata is None:
        return None
    key = str(ref.meta_key or ref.key)
    if key not in dataset_metadata:
        return None
    try:
        v = float(dataset_metadata[key])
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _scalar_value_float(
    leaf: Optional[torch.nn.Module],
    ref: ParamScalarRef,
    *,
    dataset_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    if ref.source == "metadata":
        return _metadata_value_float(dataset_metadata, ref)
    if leaf is None:
        return None
    t = _leaf_scalar_from_ref(leaf, ref)
    if t is None:
        return None
    try:
        v = float(t.detach().item())
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _eval_expr_float(op: str, a: float, b: float, c: float, eps: float) -> Optional[float]:
    if op == "mul":
        v = a * b
    elif op == "div":
        if abs(b) <= eps:
            return None
        v = a / b
    elif op == "mul_sq_div":
        if abs(c) <= eps:
            return None
        v = (a * (b * b)) / c
    else:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _eval_expr_torch(
    op: str,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    eps_t = torch.as_tensor(float(eps), device=a.device, dtype=a.dtype)
    if op == "mul":
        return a * b
    if op == "div":
        return a / torch.where(b.abs() > eps_t, b, b.sign() * eps_t + (b == 0).to(b.dtype) * eps_t)
    if op == "mul_sq_div":
        return (a * (b * b)) / torch.where(c.abs() > eps_t, c, c.sign() * eps_t + (c == 0).to(c.dtype) * eps_t)
    raise ValueError(f"Unsupported op={op!r}")


def _invariance_metrics(values: Sequence[float]) -> Tuple[float, float]:
    t = torch.as_tensor(list(values), dtype=torch.float64)
    med = torch.median(t)
    mad = torch.median((t - med).abs())
    scale = torch.maximum(torch.median(t.abs()), med.abs()).clamp_min(1.0e-12)
    score = float((mad / scale).item())
    std = float(t.std(unbiased=False).item())
    mean_abs = max(abs(float(t.mean().item())), 1.0e-12)
    cv = std / mean_abs
    return score, float(cv)


def discover_param_invariants(
    *,
    root: Node,
    models: List[torch.nn.Module],
    dataset_metadata: Optional[List[Dict[str, Any]]] = None,
    candidate_tags: Optional[List[str]] = None,
    include_fixed_value_attr: bool = True,
    max_scalars: int = 16,
    max_invariants: int = 4,
    score_threshold: float = 0.05,
    eps: float = 1.0e-12,
) -> Tuple[List[ParamScalarRef], List[ParamInvariant]]:
    """Discover low-scatter derived scalar invariants across datasets."""
    if len(models) < 2:
        return [], []
    if dataset_metadata is not None and len(dataset_metadata) != len(models):
        raise ValueError(
            "discover_param_invariants: dataset_metadata length must match number of models."
        )

    tag_to_leafidx = _tag_to_leafidx(root)
    if candidate_tags is None:
        tags = list(tag_to_leafidx.keys())
    else:
        seen = set()
        tags = []
        for t in candidate_tags:
            if t in tag_to_leafidx and t not in seen:
                seen.add(t)
                tags.append(t)

    refs: List[ParamScalarRef] = []
    D = len(models)

    # Metadata scalars are added first so metadata-linked relations can be
    # discovered under the same scalar budget.
    if dataset_metadata is not None:
        common_keys = None
        for row in dataset_metadata:
            if not isinstance(row, dict):
                continue
            keys = set(str(k) for k in row.keys())
            common_keys = keys if common_keys is None else (common_keys & keys)
        for mk in sorted(common_keys or []):
            if len(refs) >= max_scalars:
                break
            vals = []
            ok = True
            for row in dataset_metadata:
                v = _metadata_value_float(row, ParamScalarRef(
                    key=f"meta:{mk}",
                    tag="__meta__",
                    source="metadata",
                    flat_index=0,
                    meta_key=str(mk),
                ))
                if v is None:
                    ok = False
                    break
                vals.append(v)
            if not ok or len(vals) != D:
                continue
            refs.append(
                ParamScalarRef(
                    key=f"meta:{mk}",
                    tag="__meta__",
                    source="metadata",
                    flat_index=0,
                    attr="value",
                    meta_key=str(mk),
                )
            )

    for tag in tags:
        if len(refs) >= max_scalars:
            break
        lidx = tag_to_leafidx.get(tag, None)
        if lidx is None:
            continue
        leaf0 = models[0].leaf[lidx]

        # Trainable parameter scalars
        n_param_scalars = int(sum(int(p.numel()) for p in leaf0.parameters()))
        for sidx in range(n_param_scalars):
            if len(refs) >= max_scalars:
                break
            ok = True
            for m in models:
                leaf = m.leaf[lidx]
                if _leaf_flat_param(leaf, sidx) is None:
                    ok = False
                    break
            if not ok:
                continue
            refs.append(
                ParamScalarRef(
                    key=f"{tag}#p{sidx}",
                    tag=tag,
                    source="param",
                    flat_index=int(sidx),
                    attr="value",
                )
            )

        if len(refs) >= max_scalars:
            break

        # Non-trainable scalar value attribute (e.g. FixedConstLeaf.value)
        if include_fixed_value_attr:
            v0 = getattr(leaf0, "value", None)
            if torch.is_tensor(v0) and (not isinstance(v0, torch.nn.Parameter)) and int(v0.numel()) == 1:
                ok = True
                for m in models:
                    vv = getattr(m.leaf[lidx], "value", None)
                    if (not torch.is_tensor(vv)) or int(vv.numel()) != 1:
                        ok = False
                        break
                if ok:
                    refs.append(
                        ParamScalarRef(
                            key=f"{tag}#value",
                            tag=tag,
                            source="buffer",
                            flat_index=0,
                            attr="value",
                        )
                    )

    if len(refs) < 2:
        return refs, []

    value_table: List[List[float]] = []
    for d, m in enumerate(models):
        row: List[float] = []
        ok = True
        ds_meta = dataset_metadata[d] if dataset_metadata is not None else None
        for ref in refs:
            leaf = None
            if ref.source != "metadata":
                lidx = tag_to_leafidx.get(ref.tag, None)
                if lidx is None:
                    ok = False
                    break
                leaf = m.leaf[lidx]
            v = _scalar_value_float(leaf, ref, dataset_metadata=ds_meta)
            if v is None:
                ok = False
                break
            row.append(v)
        if not ok:
            return refs, []
        value_table.append(row)

    N = len(refs)
    cands: List[ParamInvariant] = []

    for i in range(N):
        for j in range(i, N):
            vals = []
            for d in range(D):
                v = _eval_expr_float("mul", value_table[d][i], value_table[d][j], 1.0, eps)
                if v is None:
                    vals = []
                    break
                vals.append(v)
            if len(vals) == D:
                score, cv = _invariance_metrics(vals)
                cands.append(
                    ParamInvariant(
                        expr=f"{refs[i].key}*{refs[j].key}",
                        op="mul",
                        a=i,
                        b=j,
                        values=tuple(float(v) for v in vals),
                        score=score,
                        cv=cv,
                    )
                )

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            vals = []
            for d in range(D):
                v = _eval_expr_float("div", value_table[d][i], value_table[d][j], 1.0, eps)
                if v is None:
                    vals = []
                    break
                vals.append(v)
            if len(vals) == D:
                score, cv = _invariance_metrics(vals)
                cands.append(
                    ParamInvariant(
                        expr=f"{refs[i].key}/{refs[j].key}",
                        op="div",
                        a=i,
                        b=j,
                        values=tuple(float(v) for v in vals),
                        score=score,
                        cv=cv,
                    )
                )

    for i in range(N):
        for j in range(N):
            for k in range(N):
                if k == j:
                    continue
                vals = []
                for d in range(D):
                    v = _eval_expr_float(
                        "mul_sq_div",
                        value_table[d][i],
                        value_table[d][j],
                        value_table[d][k],
                        eps,
                    )
                    if v is None:
                        vals = []
                        break
                    vals.append(v)
                if len(vals) == D:
                    score, cv = _invariance_metrics(vals)
                    cands.append(
                        ParamInvariant(
                            expr=f"{refs[i].key}*({refs[j].key}^2)/{refs[k].key}",
                            op="mul_sq_div",
                            a=i,
                            b=j,
                            c=k,
                            values=tuple(float(v) for v in vals),
                            score=score,
                            cv=cv,
                        )
                    )

    cands = [c for c in cands if math.isfinite(c.score) and (c.score <= float(score_threshold))]
    cands.sort(key=lambda c: (float(c.score), float(c.cv), c.expr))

    selected: List[ParamInvariant] = []
    signatures = set()
    for c in cands:
        arr = torch.as_tensor(c.values, dtype=torch.float64)
        scale = float(arr.abs().max().item())
        if scale <= 1.0e-12:
            norm = arr
        else:
            norm = arr / scale
        sig = tuple(float(round(float(v), 6)) for v in norm.tolist())
        if sig in signatures:
            continue
        signatures.add(sig)
        selected.append(c)
        if len(selected) >= int(max_invariants):
            break

    return refs, selected


def evaluate_invariant_on_composite(
    *,
    comp: torch.nn.Module,
    tag_to_leafidx: Dict[str, int],
    refs: List[ParamScalarRef],
    invariant: ParamInvariant,
    dataset_metadata: Optional[Dict[str, Any]] = None,
    eps: float = 1.0e-12,
) -> Optional[torch.Tensor]:
    """Evaluate a discovered invariant on one composite model."""
    p0 = next(comp.parameters(), None)
    ref_device = p0.device if p0 is not None else torch.device("cpu")
    ref_dtype = p0.dtype if p0 is not None else torch.float64

    def _ref_value_tensor(ref: ParamScalarRef) -> Optional[torch.Tensor]:
        if ref.source == "metadata":
            v = _metadata_value_float(dataset_metadata, ref)
            if v is None:
                return None
            return torch.as_tensor(float(v), device=ref_device, dtype=ref_dtype)
        lidx = tag_to_leafidx.get(ref.tag, None)
        if lidx is None:
            return None
        return _leaf_scalar_from_ref(comp.leaf[lidx], ref)

    try:
        a = _ref_value_tensor(refs[invariant.a])
        b = _ref_value_tensor(refs[invariant.b])
        if a is None or b is None:
            return None
        if invariant.op == "mul":
            return _eval_expr_torch("mul", a, b, b, eps)
        if invariant.op == "div":
            return _eval_expr_torch("div", a, b, b, eps)
        if invariant.op == "mul_sq_div":
            c = _ref_value_tensor(refs[invariant.c])
            if c is None:
                return None
            return _eval_expr_torch("mul_sq_div", a, b, c, eps)
    except Exception:
        return None
    return None
