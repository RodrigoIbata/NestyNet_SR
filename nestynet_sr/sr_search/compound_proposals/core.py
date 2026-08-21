# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://www.mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Core compound proposal types shared by Stage A and Stage B.

The proposal layer is deliberately side-effect-free.  It may describe a useful
coordinate, wrapper, or visible analytic motif; it must not accept a rewrite.
Stage A and Stage B keep their own fitting, validation, gauge, and unit policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from nestynet_sr.sr_core.bridges import Node, ast_to_human_readable, clone_ast


@dataclass(frozen=True)
class ProposalDim:
    """Dimension metadata for a proposal expression."""

    base_dim: Optional[Tuple[Any, ...]] = None
    z_dim: Optional[Tuple[Any, ...]] = None


@dataclass(frozen=True)
class CompoundProposal:
    """A typed, unit-aware compound proposal.

    ``base_ast`` is the unwrapped core when a family has one, while ``z_ast`` is
    the actual expression the consumer should use.  For simple proposals the two
    may be the same expression.  ``consumed_pattern`` is the legacy Stage-A
    pattern tuple; ``consumed_inputs`` is a flattened list of local input slots
    used by the proposal.
    """

    label: str
    family: str
    kind: str
    z_ast: Node
    base_ast: Optional[Node] = None
    consumed_pattern: Tuple[int, ...] = ()
    consumed_inputs: Tuple[int, ...] = ()
    wrapper: Optional[str] = None
    confidence: float = 1.0
    z_dim: Optional[Tuple[Any, ...]] = None
    base_dim: Optional[Tuple[Any, ...]] = None
    evidence: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _tuple_or_none(value: Any) -> Optional[Tuple[Any, ...]]:
    if value is None:
        return None
    try:
        return tuple(value)
    except Exception:
        return None


def _tuple_ints(value: Any) -> Tuple[int, ...]:
    if value is None:
        return ()
    out = []
    try:
        iterable = tuple(value)
    except Exception:
        iterable = (value,)
    for item in iterable:
        try:
            out.append(int(item))
        except Exception:
            continue
    return tuple(out)


def _metric_consumed_inputs(meta: dict) -> Tuple[int, ...]:
    if "indices" in meta:
        return _tuple_ints(meta.get("indices"))
    if "pairs" in meta:
        out = []
        for pair in meta.get("pairs") or ():
            out.extend(_tuple_ints(pair))
        return tuple(out)
    return ()


def compound_proposal_from_metric(metric_prop: Any) -> CompoundProposal:
    """Convert the legacy metric-distance proposal shape to CompoundProposal."""

    meta = dict(getattr(metric_prop, "meta", None) or {})
    family = str(getattr(metric_prop, "family", meta.get("metric_family", "metric_distance")))
    wrapper = str(getattr(metric_prop, "wrapper", meta.get("metric_wrapper", "q")))
    kind = str(meta.get("kind", "metric_distance"))
    meta["kind"] = kind
    meta["metric_family"] = family
    meta["metric_wrapper"] = wrapper
    evidence = dict(meta)
    evidence.setdefault("source", "metric_distance")
    evidence.setdefault("metric_family", family)
    evidence.setdefault("metric_wrapper", wrapper)
    consumed_inputs = _metric_consumed_inputs(meta)
    return CompoundProposal(
        label=str(getattr(metric_prop, "label", f"{family}_{wrapper}")),
        family=family,
        kind=kind,
        z_ast=clone_ast(getattr(metric_prop, "z_ast")),
        base_ast=clone_ast(getattr(metric_prop, "q_ast")),
        consumed_pattern=tuple(int(v) for v in getattr(metric_prop, "pattern", ()) or ()),
        consumed_inputs=consumed_inputs,
        wrapper=wrapper,
        confidence=float(getattr(metric_prop, "confidence", 1.0)),
        z_dim=_tuple_or_none(getattr(metric_prop, "z_dim", None)),
        base_dim=_tuple_or_none(getattr(metric_prop, "q_dim", None)),
        evidence=evidence,
        meta=meta,
    )


def proposal_signature(prop: CompoundProposal) -> Tuple[Any, ...]:
    """Return a stable, cross-stage duplicate signature for a proposal."""

    try:
        from nestynet_sr.sr_search.compound_proposals.wrappers import canonical_wrapper_name

        wrapper_key = canonical_wrapper_name(str(prop.wrapper or ""))
    except Exception:
        wrapper_key = str(prop.wrapper or "")
    try:
        z_key = ast_to_human_readable(prop.z_ast)
    except Exception:
        z_key = repr(prop.z_ast)
    return (
        str(prop.kind),
        str(prop.family),
        str(wrapper_key),
        tuple(int(v) for v in prop.consumed_pattern),
        tuple(int(v) for v in prop.consumed_inputs),
        z_key,
    )


def stageA_tuple_from_proposal(prop: CompoundProposal):
    """Convert a shared proposal to Stage A's legacy proposal tuple shape."""

    meta = dict(prop.meta or {})
    meta.setdefault("kind", prop.kind)
    meta.setdefault("family", prop.family)
    meta.setdefault("wrapper", prop.wrapper)
    if prop.kind == "metric_distance":
        meta.setdefault("metric_family", prop.family)
        meta.setdefault("metric_wrapper", prop.wrapper)
        meta.setdefault("q_dim", prop.base_dim)
    meta.setdefault("compound_family", prop.family)
    meta.setdefault("compound_wrapper", prop.wrapper)
    meta.setdefault("compound_proposal_signature", proposal_signature(prop))
    meta.setdefault("base_dim", prop.base_dim)
    meta.setdefault("z_dim", prop.z_dim)
    return (
        tuple(int(v) for v in prop.consumed_pattern),
        clone_ast(prop.z_ast),
        float(prop.confidence),
        None,
        meta,
    )


def stageB_meta_from_proposal(prop: CompoundProposal, *, pattern: Optional[str] = None) -> dict:
    """Return common Stage-B candidate metadata for a shared proposal."""

    meta = dict(prop.meta or {})
    pattern_name = str(pattern or prop.kind or prop.family)
    meta.setdefault("pattern", pattern_name)
    meta.setdefault(
        "pattern_family",
        "metric_distance" if prop.kind == "metric_distance" else prop.family,
    )
    meta.setdefault("compound_family", prop.family)
    meta.setdefault("compound_wrapper", prop.wrapper)
    meta.setdefault("compound_proposal_signature", proposal_signature(prop))
    meta.setdefault("base_dim", prop.base_dim)
    meta.setdefault("z_dim", prop.z_dim)
    if prop.kind == "metric_distance":
        meta.setdefault("metric_family", prop.family)
        meta.setdefault("metric_wrapper", prop.wrapper)
        meta.setdefault("q_dim", prop.base_dim)
    return meta
