# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://www.mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Dimensionless log/exp coordinate-lift proposals."""

from __future__ import annotations

import itertools
from typing import Any, List, Optional, Sequence, Tuple

from nestynet_sr.sr_core.bridges import AtomNode, MulNode, Node, PowNode, clone_ast, get_input_exprs
from nestynet_sr.sr_search.compound_proposals.core import CompoundProposal, proposal_signature
from nestynet_sr.sr_search.compound_proposals.units import add_dim, input_dims, is_dimless_dim, same_dim, scale_dim
from nestynet_sr.sr_search.compound_proposals.wrappers import apply_compound_wrapper


def _ratio(a: Node, b: Node) -> Node:
    return MulNode(clone_ast(a), PowNode(clone_ast(b), -1.0))


def _prod(a: Node, b: Node) -> Node:
    return MulNode(clone_ast(a), clone_ast(b))


def _proposal_label(family: str, wrapper: str, idxs: Sequence[int]) -> str:
    idx_s = "_".join(str(int(i)) for i in idxs)
    return f"{family}_{wrapper}_{idx_s}"


def build_logexp_compound_proposals(
    atom_or_inputs: AtomNode | Sequence[Node],
    *,
    units_spec: Any = None,
    wrappers: Sequence[str] = ("log", "exp"),
    max_proposals: int = 12,
) -> List[CompoundProposal]:
    """Build bounded dimensionless log/exp proposals.

    The base coordinate ``z`` is emitted only when it is dimensionless:
      - a raw/effective input already dimensionless;
      - a ratio of two same-dimension inputs;
      - a product of two inputs with inverse dimensions.
    """

    if isinstance(atom_or_inputs, AtomNode):
        inputs = tuple(get_input_exprs(atom_or_inputs))
    else:
        inputs = tuple(atom_or_inputs)

    n = int(len(inputs))
    if n < 1:
        return []

    dims = input_dims(inputs, units_spec)
    dimless_dim: Optional[Tuple[Any, ...]] = None
    for d in dims:
        if d is not None:
            dimless_dim = scale_dim(d, 0.0)
            break

    out: List[CompoundProposal] = []
    seen = set()

    def _emit(
        *,
        family: str,
        base_ast: Node,
        consumed: Sequence[int],
        confidence: float,
        meta: dict,
    ) -> None:
        consumed_set = set(int(v) for v in consumed)
        pattern = tuple(1 if i in consumed_set else 0 for i in range(n))
        for wrapper in wrappers:
            wrapped = apply_compound_wrapper(
                base_ast,
                dimless_dim,
                str(wrapper),
                strict_units=units_spec is not None,
            )
            if wrapped is None:
                continue
            meta_i = dict(meta)
            meta_i["kind"] = "logexp"
            meta_i["logexp_family"] = str(family)
            meta_i["logexp_wrapper"] = str(wrapper)
            prop = CompoundProposal(
                label=_proposal_label(family, str(wrapper), consumed),
                family=str(family),
                kind="logexp",
                z_ast=wrapped.expr,
                base_ast=clone_ast(base_ast),
                consumed_pattern=pattern,
                consumed_inputs=tuple(int(v) for v in consumed),
                wrapper=str(wrapper),
                confidence=float(confidence),
                z_dim=wrapped.dim,
                base_dim=dimless_dim,
                evidence=dict(meta_i),
                meta=meta_i,
            )
            sig = proposal_signature(prop)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(prop)

    # 1) Raw/effective dimensionless inputs.
    for i in range(n):
        if not is_dimless_dim(dims[i]):
            continue
        _emit(
            family="dimless_input",
            base_ast=inputs[i],
            consumed=(i,),
            confidence=0.90,
            meta={"indices": (int(i),), "base_kind": "input"},
        )
        if len(out) >= int(max_proposals):
            return out

    # 2) Symmetric same-dimension ratios.
    for i, j in itertools.combinations(range(n), 2):
        if not same_dim(dims[i], dims[j]):
            continue
        for a, b in ((i, j), (j, i)):
            _emit(
                family="dimless_ratio",
                base_ast=_ratio(inputs[a], inputs[b]),
                consumed=(int(a), int(b)),
                confidence=0.93,
                meta={"indices": (int(a), int(b)), "base_kind": "ratio"},
            )
            if len(out) >= int(max_proposals):
                return out

    # 3) Pair products with zero net dimension.
    for i, j in itertools.combinations(range(n), 2):
        prod_dim = add_dim(dims[i], dims[j])
        if not is_dimless_dim(prod_dim):
            continue
        # Skip all-dimensionless raw products in the first bounded version; raw
        # dimensionless inputs and ratios already cover the common cheap cases.
        if is_dimless_dim(dims[i]) and is_dimless_dim(dims[j]):
            continue
        _emit(
            family="dimless_product",
            base_ast=_prod(inputs[i], inputs[j]),
            consumed=(int(i), int(j)),
            confidence=0.91,
            meta={"indices": (int(i), int(j)), "base_kind": "product"},
        )
        if len(out) >= int(max_proposals):
            return out

    return out[: int(max_proposals)]
