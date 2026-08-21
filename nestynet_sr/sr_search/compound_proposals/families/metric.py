# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://www.mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Unit-aware metric-distance compound proposals.

This module is the shared proposal source for law-of-cosines and Cartesian
distance coordinates.  It emits ``CompoundProposal`` objects only; Stage A and
Stage B remain responsible for fitting, validation, and acceptance.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    ConstNode,
    CosNode,
    MulNode,
    Node,
    PowNode,
    clone_ast,
    get_input_exprs,
)
from nestynet_sr.sr_search.compound_proposals.core import CompoundProposal, proposal_signature
from nestynet_sr.sr_search.compound_proposals.units import input_dims, is_dimless_dim, same_dim, scale_dim
from nestynet_sr.sr_search.compound_proposals.wrappers import apply_compound_wrapper


def _mul_many(exprs: Sequence[Node]) -> Node:
    if not exprs:
        return ConstNode(1.0)
    out = clone_ast(exprs[0])
    for expr in exprs[1:]:
        out = MulNode(out, clone_ast(expr))
    return out


def _add_many(exprs: Sequence[Node]) -> Node:
    if not exprs:
        return ConstNode(0.0)
    out = clone_ast(exprs[0])
    for expr in exprs[1:]:
        out = AddNode(out, clone_ast(expr))
    return out


def _diff(a: Node, b: Node) -> Node:
    return AddNode(clone_ast(a), MulNode(ConstNode(-1.0), clone_ast(b)))


def _law_of_cosines_q(a: Node, b: Node, theta: Node) -> Node:
    a2 = PowNode(clone_ast(a), 2.0)
    b2 = PowNode(clone_ast(b), 2.0)
    cross = _mul_many((ConstNode(-2.0), clone_ast(a), clone_ast(b), CosNode(clone_ast(theta))))
    return AddNode(AddNode(a2, b2), cross)


def _law_of_cosines_sqnorm_q(a: Node, b: Node, theta: Node, *, sign: float) -> Node:
    """Law-of-cosines/Gram form when inputs are squared radii.

    ``a`` and ``b`` have the same units as the output quantity, so the cross
    term is ``2*sqrt(a*b)*cos(theta)`` rather than ``2*a*b*cos(theta)``.
    """

    cross = _mul_many(
        (
            ConstNode(float(sign) * 2.0),
            PowNode(MulNode(clone_ast(a), clone_ast(b)), 0.5),
            CosNode(clone_ast(theta)),
        )
    )
    return AddNode(AddNode(clone_ast(a), clone_ast(b)), cross)


def _gram3_pairangles_q(
    radii: Sequence[Node],
    pair_angles: Sequence[Node],
    *,
    signs: Sequence[float],
) -> Node:
    """Three-vector Gram norm using explicit pair-angle coordinates.

    The pair angle order is ``(0,1), (0,2), (1,2)``.  The signs are vector
    signs normalized up to a global sign, so the cross coefficients are
    ``2*sign_i*sign_j``.
    """

    if len(radii) != 3 or len(pair_angles) != 3 or len(signs) != 3:
        raise ValueError("gram3 requires three radii, three pair angles, and three signs")
    terms: list[Node] = [PowNode(clone_ast(r), 2.0) for r in radii]
    for angle_idx, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        coeff = 2.0 * float(signs[i]) * float(signs[j])
        terms.append(
            _mul_many(
                (
                    ConstNode(coeff),
                    clone_ast(radii[i]),
                    clone_ast(radii[j]),
                    CosNode(clone_ast(pair_angles[angle_idx])),
                )
            )
        )
    return _add_many(terms)


def _proposal_label(family: str, wrapper: str, idxs: Iterable[int]) -> str:
    idx_s = "_".join(str(int(i)) for i in idxs)
    return f"{family}_{wrapper}_{idx_s}"


def _cartesian_pairings(n_inputs: int, max_pairs: int) -> Iterable[Tuple[Tuple[int, int], ...]]:
    """Yield disjoint pair bundles, preferring adjacent pairs first."""

    if n_inputs < 4 or max_pairs < 2:
        return

    if n_inputs % 2 == 0:
        adjacent = tuple((i, i + 1) for i in range(0, n_inputs, 2))
        if 2 <= len(adjacent) <= max_pairs:
            yield adjacent

    seen = {tuple((i, i + 1) for i in range(0, n_inputs, 2))} if n_inputs % 2 == 0 else set()
    idxs = tuple(range(n_inputs))
    for pairs_flat in itertools.combinations(itertools.combinations(idxs, 2), 2):
        used = [j for p in pairs_flat for j in p]
        if len(set(used)) != len(used):
            continue
        pairs = tuple(tuple(int(v) for v in p) for p in pairs_flat)
        if pairs in seen:
            continue
        seen.add(pairs)
        yield pairs

    if max_pairs >= 3 and n_inputs >= 6:
        for pairs_flat in itertools.combinations(itertools.combinations(idxs, 2), 3):
            used = [j for p in pairs_flat for j in p]
            if len(set(used)) != len(used):
                continue
            pairs = tuple(tuple(int(v) for v in p) for p in pairs_flat)
            if pairs in seen:
                continue
            seen.add(pairs)
            yield pairs


def _gram3_sign_patterns() -> Tuple[Tuple[float, float, float], ...]:
    # Normalize away the global sign degeneracy by fixing sign[0]=+1.
    return (
        (1.0, 1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, -1.0, -1.0),
    )


def build_metric_distance_compound_proposals(
    atom_or_inputs: AtomNode | Sequence[Node],
    *,
    units_spec: Any = None,
    include_polar: bool = True,
    include_cartesian: bool = True,
    wrappers: Sequence[str] = ("q", "sqrt_q", "inv_sqrt_q", "inv_q"),
    max_cartesian_pairs: int = 3,
    max_proposals: int = 16,
) -> List[CompoundProposal]:
    """Build bounded metric-distance proposals over effective coordinates.

    Supported forms:
      - polar/law of cosines: ``a**2 + b**2 - 2*a*b*cos(theta)``
      - Cartesian distance squared: ``sum_i (u_i - v_i)**2``

    Unit rules are applied before emitting proposals:
      - ``a`` and ``b`` must have matching units;
      - ``theta`` must be dimensionless;
      - Cartesian differences require matching units per pair.
    """

    if isinstance(atom_or_inputs, AtomNode):
        inputs = tuple(get_input_exprs(atom_or_inputs))
    else:
        inputs = tuple(atom_or_inputs)

    n = int(len(inputs))
    if n < 2:
        return []

    dims = input_dims(inputs, units_spec)
    out: List[CompoundProposal] = []
    seen = set()

    def _emit(
        *,
        family: str,
        q_ast: Node,
        q_dim: Optional[Tuple[Any, ...]],
        pattern: Tuple[int, ...],
        consumed: Sequence[int],
        confidence: float,
        meta: dict,
    ) -> None:
        for wrapper in wrappers:
            wrapped = apply_compound_wrapper(
                q_ast,
                q_dim,
                str(wrapper),
                strict_units=units_spec is not None,
            )
            if wrapped is None:
                continue
            z_ast = wrapped.expr
            z_dim = wrapped.dim
            meta_i = dict(meta)
            meta_i["kind"] = "metric_distance"
            meta_i["metric_family"] = str(family)
            meta_i["metric_wrapper"] = str(wrapper)
            prop = CompoundProposal(
                label=_proposal_label(family, str(wrapper), consumed),
                family=str(family),
                kind="metric_distance",
                z_ast=z_ast,
                base_ast=clone_ast(q_ast),
                consumed_pattern=tuple(int(v) for v in pattern),
                consumed_inputs=tuple(int(v) for v in consumed),
                wrapper=str(wrapper),
                confidence=float(confidence),
                z_dim=z_dim,
                base_dim=q_dim,
                evidence=dict(meta_i),
                meta=meta_i,
            )
            sig = proposal_signature(prop)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(prop)

    # --- Three-vector Gram norms -------------------------------------------
    #
    # This is the bounded N-vector extension of pairwise law-of-cosines.  It
    # expects three same-dimension radii and three dimensionless pair-angle
    # coordinates in the order generated by combinations: (0,1), (0,2), (1,2).
    # These angle coordinates may themselves be Stage-A difference compounds.
    if include_polar and n >= 6:
        radius_idxs = [i for i in range(n) if not is_dimless_dim(dims[i])]
        angle_idxs = [i for i in range(n) if is_dimless_dim(dims[i])]
        for ridxs in itertools.combinations(radius_idxs, 3):
            r0, r1, r2 = ridxs
            if not (same_dim(dims[r0], dims[r1]) and same_dim(dims[r0], dims[r2])):
                continue
            q_dim = scale_dim(dims[r0], 2.0)
            for aidxs in itertools.combinations(angle_idxs, 3):
                pattern = tuple(1 if p in (*ridxs, *aidxs) else 0 for p in range(n))
                consumed = tuple(int(v) for v in (*ridxs, *aidxs))
                for signs in _gram3_sign_patterns():
                    sign_name = "".join("p" if s > 0 else "m" for s in signs)
                    q_ast = _gram3_pairangles_q(
                        (inputs[r0], inputs[r1], inputs[r2]),
                        (inputs[aidxs[0]], inputs[aidxs[1]], inputs[aidxs[2]]),
                        signs=signs,
                    )
                    _emit(
                        family=f"gram3_pairangles_{sign_name}",
                        q_ast=q_ast,
                        q_dim=q_dim,
                        pattern=pattern,
                        consumed=consumed,
                        confidence=0.965,
                        meta={
                            "radii": tuple(int(v) for v in ridxs),
                            "angles": tuple(int(v) for v in aidxs),
                            "signs": tuple(float(v) for v in signs),
                            "angle_mode": "pairwise",
                            "gram_rank": 3,
                        },
                    )
                    if len(out) >= int(max_proposals):
                        return out

    if include_polar and n >= 3:
        for i, j in itertools.combinations(range(n), 2):
            if not same_dim(dims[i], dims[j]):
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                if not is_dimless_dim(dims[k]):
                    continue

                # Inputs can represent squared radii/energies directly.  This
                # form covers e.g. x0 + x1 + 2*sqrt(x0*x1)*cos(theta).
                for sign, sign_name in ((+1.0, "plus"), (-1.0, "minus")):
                    q_ast_sq = _law_of_cosines_sqnorm_q(inputs[i], inputs[j], inputs[k], sign=sign)
                    pattern = tuple(1 if p in (i, j, k) else 0 for p in range(n))
                    _emit(
                        family=f"lawcos_sq_{sign_name}",
                        q_ast=q_ast_sq,
                        q_dim=dims[i],
                        pattern=pattern,
                        consumed=(i, j, k),
                        confidence=0.99,
                        meta={
                            "indices": (int(i), int(j), int(k)),
                            "sign": float(sign),
                            "squared_radius_inputs": True,
                        },
                    )
                    if len(out) >= int(max_proposals):
                        return out

                q_dim = scale_dim(dims[i], 2.0)
                q_ast = _law_of_cosines_q(inputs[i], inputs[j], inputs[k])
                pattern = tuple(1 if p in (i, j, k) else 0 for p in range(n))
                _emit(
                    family="lawcos",
                    q_ast=q_ast,
                    q_dim=q_dim,
                    pattern=pattern,
                    consumed=(i, j, k),
                    confidence=0.98,
                    meta={"indices": (int(i), int(j), int(k))},
                )
                if len(out) >= int(max_proposals):
                    return out

    if include_cartesian and n >= 4:
        emitted_bundles = 0
        for pairs in _cartesian_pairings(n, int(max_cartesian_pairs)):
            if not pairs:
                continue
            q_terms = []
            q_dim = None
            ok = True
            consumed_set = set()
            for i, j in pairs:
                if not same_dim(dims[i], dims[j]):
                    ok = False
                    break
                pair_q_dim = scale_dim(dims[i], 2.0)
                if q_dim is None:
                    q_dim = pair_q_dim
                elif not same_dim(q_dim, pair_q_dim):
                    ok = False
                    break
                q_terms.append(PowNode(_diff(inputs[i], inputs[j]), 2.0))
                consumed_set.update((int(i), int(j)))
            if not ok or len(q_terms) < 2:
                continue
            q_ast = _add_many(q_terms)
            pattern = tuple(1 if p in consumed_set else 0 for p in range(n))
            consumed = tuple(sorted(consumed_set))
            _emit(
                family="cartdist",
                q_ast=q_ast,
                q_dim=q_dim,
                pattern=pattern,
                consumed=consumed,
                confidence=0.97,
                meta={"pairs": tuple(tuple(int(v) for v in p) for p in pairs)},
            )
            emitted_bundles += 1
            if len(out) >= int(max_proposals) or emitted_bundles >= int(max_proposals):
                return out

    return out[: int(max_proposals)]
