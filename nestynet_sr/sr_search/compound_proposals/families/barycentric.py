# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://www.mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Unit-aware weighted-average / barycentric compound proposals."""

from __future__ import annotations

import itertools
from typing import Any, List, Optional, Sequence, Tuple

from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    ConstNode,
    MulNode,
    Node,
    PowNode,
    clone_ast,
    get_input_exprs,
)
from nestynet_sr.sr_search.compound_proposals.core import CompoundProposal, proposal_signature
from nestynet_sr.sr_search.compound_proposals.units import add_dim, input_dims, same_dim
from nestynet_sr.sr_search.compound_proposals.wrappers import apply_compound_wrapper


def _mul(a: Node, b: Node) -> Node:
    return MulNode(clone_ast(a), clone_ast(b))


def _sub(a: Node, b: Node) -> Node:
    return AddNode(clone_ast(a), MulNode(ConstNode(-1.0), clone_ast(b)))


def _weighted_pair_ast(w0: Node, w1: Node, v0: Node, v1: Node, *, sign: float) -> Node:
    term0 = _mul(w0, v0)
    term1 = _mul(w1, v1)
    if float(sign) < 0.0:
        numerator = _sub(term0, term1)
    else:
        numerator = AddNode(term0, term1)
    denom = AddNode(clone_ast(w0), clone_ast(w1))
    return MulNode(numerator, PowNode(denom, -1.0))


def _proposal_label(family: str, wrapper: str, idxs: Sequence[int]) -> str:
    idx_s = "_".join(str(int(i)) for i in idxs)
    return f"{family}_{wrapper}_{idx_s}"


def build_barycentric_compound_proposals(
    atom_or_inputs: AtomNode | Sequence[Node],
    *,
    units_spec: Any = None,
    wrappers: Sequence[str] = ("z",),
    max_proposals: int = 12,
) -> List[CompoundProposal]:
    """Build bounded weighted-average proposals over effective coordinates.

    Supported v1 forms are pairwise:

    ``(w0*v0 + w1*v1)/(w0+w1)`` and ``(w0*v0 - w1*v1)/(w0+w1)``.

    Unit rules are applied before emitting proposals:
      - ``w0`` and ``w1`` must have matching units;
      - ``v0`` and ``v1`` must have matching units;
      - numerator terms therefore share ``dim(w)+dim(v)`` and the result has
        ``dim(v)``.
    """

    if isinstance(atom_or_inputs, AtomNode):
        inputs = tuple(get_input_exprs(atom_or_inputs))
    else:
        inputs = tuple(atom_or_inputs)

    n = int(len(inputs))
    if n < 4:
        return []

    dims = input_dims(inputs, units_spec)
    out: List[CompoundProposal] = []
    seen = set()

    def _emit(
        *,
        family: str,
        z0_ast: Node,
        z_dim: Optional[Tuple[Any, ...]],
        consumed: Sequence[int],
        confidence: float,
        meta: dict,
    ) -> None:
        pattern = tuple(1 if i in set(int(v) for v in consumed) else 0 for i in range(n))
        for wrapper in wrappers:
            wrapped = apply_compound_wrapper(
                z0_ast,
                z_dim,
                str(wrapper),
                strict_units=units_spec is not None,
            )
            if wrapped is None:
                continue
            meta_i = dict(meta)
            meta_i["kind"] = "barycentric"
            meta_i["barycentric_family"] = str(family)
            meta_i["barycentric_wrapper"] = str(wrapper)
            prop = CompoundProposal(
                label=_proposal_label(family, str(wrapper), consumed),
                family=str(family),
                kind="barycentric",
                z_ast=wrapped.expr,
                base_ast=clone_ast(z0_ast),
                consumed_pattern=pattern,
                consumed_inputs=tuple(int(v) for v in consumed),
                wrapper=str(wrapper),
                confidence=float(confidence),
                z_dim=wrapped.dim,
                base_dim=z_dim,
                evidence=dict(meta_i),
                meta=meta_i,
            )
            sig = proposal_signature(prop)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(prop)

    for w_pair in itertools.combinations(range(n), 2):
        w0, w1 = (int(w_pair[0]), int(w_pair[1]))
        if not same_dim(dims[w0], dims[w1]):
            continue
        remaining = [i for i in range(n) if i not in (w0, w1)]
        for v_pair in itertools.combinations(remaining, 2):
            v0, v1 = (int(v_pair[0]), int(v_pair[1]))
            if not same_dim(dims[v0], dims[v1]):
                continue
            # Sanity: numerator terms must match.  This is implied by the two
            # pairwise checks above, but keeping the explicit test documents the
            # dimensional contract for future generalized barycentric forms.
            if not same_dim(add_dim(dims[w0], dims[v0]), add_dim(dims[w1], dims[v1])):
                continue
            pairings = (
                ((v0, v1), "direct"),
                ((v1, v0), "cross"),
            )
            for (a, b), pairing in pairings:
                consumed = (w0, w1, int(a), int(b))
                for sign, sign_name in ((1.0, "plus"), (-1.0, "minus")):
                    family = f"weighted_avg_{pairing}_{sign_name}"
                    z_ast = _weighted_pair_ast(inputs[w0], inputs[w1], inputs[a], inputs[b], sign=sign)
                    _emit(
                        family=family,
                        z0_ast=z_ast,
                        z_dim=dims[v0],
                        consumed=consumed,
                        confidence=0.94,
                        meta={
                            "weights": (w0, w1),
                            "values": (int(a), int(b)),
                            "pairing": pairing,
                            "sign": float(sign),
                        },
                    )
                    if len(out) >= int(max_proposals):
                        return out

    return out[: int(max_proposals)]
