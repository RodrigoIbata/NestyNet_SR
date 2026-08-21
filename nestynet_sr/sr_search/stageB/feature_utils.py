# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Feature indexing and rewrite utilities for Stage B.

This module provides functions for organizing discovered features (scaling, trig)
and building simple AST rewrites based on these features.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from nestynet_sr.sr_core.bridges import (
    AtomNode,
    Node,
    clone_inputs,
    effective_arity,
    replace_atom_in_ast,
)

# Import feature specs from parent module (sr_search)
from ..features import ScaleSpec, TrigAxisSpec


def _scaling_index(scale_specs: List[ScaleSpec]) -> Dict[int, List[ScaleSpec]]:
    """
    Build a small index from single-axis index -> [ScaleSpec,...].
    """
    out: Dict[int, List[ScaleSpec]] = {}
    for spec in scale_specs:
        if len(spec.indices) == 1:
            j = int(spec.indices[0])
            out.setdefault(j, []).append(spec)
    return out


def _trig_index(trig_specs: List[TrigAxisSpec]) -> Dict[int, TrigAxisSpec]:
    """
    Build an index axis -> best TrigAxisSpec (by spectral strength).
    """
    out: Dict[int, TrigAxisSpec] = {}
    for spec in trig_specs:
        j = int(spec.axis)
        if j not in out or spec.strength > out[j].strength:
            out[j] = spec
    return out


def _best_scale_spec_for_axis(
    scaling_by_axis: Dict[int, List[ScaleSpec]], axis: int
) -> Optional[ScaleSpec]:
    specs = scaling_by_axis.get(int(axis), [])
    if not specs:
        return None
    # Choose the one with smallest relative scatter
    return min(specs, key=lambda s: s.rel_std)


def _is_strong_scaling_spec(
    spec: ScaleSpec, k_round_tol: float = 0.05, rel_std_max: float = 0.05, n_min: int = 500
) -> bool:
    """
    Heuristic: does this ScaleSpec look like a very clean power-law
    y ~ x_j^k with k close to an integer?

    We use it to decide whether to try a scaling/power rewrite *before*
    Planck etc. for a given univariate NN leaf.
    """
    if spec is None:
        return False
    if spec.rel_std > rel_std_max:
        return False
    if getattr(spec, "n_points", 0) < n_min:
        return False
    k = float(spec.k_hat)
    k_int = round(k)
    return abs(k - k_int) <= k_round_tol


def _make_poly_1d_rewrite(
    root: Node,
    target: AtomNode,
    degree: int = 1,
    min_total: int = 0,
    rpoly: bool = False,
) -> Optional[Node]:
    """
    Simple 1D polynomial rewrite:
        nn(x_j)  ->  poly(x_j, min_total=min_total)

    Parameters
    ----------
    root : Node
        The AST root.
    target : AtomNode
        The NN atom to replace.
    degree : int
        Maximum polynomial degree.
    min_total : int
        Minimum total degree of monomials to include.
        - min_total=0: Full basis (constant + linear + ... + degree)
        - min_total=1: Monomer basis (linear + ... + degree), no constant term

    Uses the same var_idxs and tag as the original NN leaf so that any
    analytic initialisation can still reuse the Stage-A "teacher" if
    your _initialise_analytic_leaves_from_reuse supports that.
    """
    if not isinstance(target, AtomNode):
        return None
    if str(target.kind).lower() != "nn":
        return None
    if effective_arity(target) != 1:
        return None

    kind = "rpoly" if rpoly else "poly"
    new_kwargs = {"degree": int(degree), "min_total": int(min_total)}
    new_atom = AtomNode(
        kind=kind,
        var_idxs=tuple(int(j) for j in target.var_idxs),
        kwargs=new_kwargs,
        tag=target.tag,
        inputs=clone_inputs(target),
    )
    return replace_atom_in_ast(root, target, new_atom)

def _make_polylog_1d_rewrite(
    root: Node,
    target: AtomNode,
    degree: int = 1,
) -> Optional[Node]:
    """
    1D polylog rewrite:
        nn(x_j)  ->  polylog(x_j)

    where polylog(x_j) is implemented by PolyLogLeaf, i.e. a polynomial
    in log(x_j). This is aimed at structures like log(x) and log(x)*poly(x).
    """
    if not isinstance(target, AtomNode):
        return None
    if str(target.kind).lower() != "nn":
        return None
    if effective_arity(target) != 1:
        return None

    new_kwargs = {"degree": int(degree)}
    new_atom = AtomNode(
        kind="polylog",
        var_idxs=tuple(int(j) for j in target.var_idxs),
        kwargs=new_kwargs,
        tag=target.tag,
        inputs=clone_inputs(target),
    )
    return replace_atom_in_ast(root, target, new_atom)


def _make_logshifted_1d_rewrite(
    root: Node,
    target: AtomNode,
) -> Optional[Node]:
    """
    1D shifted-log rewrite:
        nn(x_j)  ->  logshifted(x_j)

    where logshifted(x_j) computes a*log(x_j - b) + c.
    Targets structures like log(x-1), log(x-2), etc.
    """
    if not isinstance(target, AtomNode):
        return None
    if str(target.kind).lower() != "nn":
        return None
    if effective_arity(target) != 1:
        return None

    new_atom = AtomNode(
        kind="logshifted",
        var_idxs=tuple(int(j) for j in target.var_idxs),
        kwargs={},
        tag=target.tag,
        inputs=clone_inputs(target),
    )
    return replace_atom_in_ast(root, target, new_atom)


def _make_power_1d_rewrite(
    root: Node,
    target: AtomNode,
    exponent: float = -1.0,
) -> Optional[Node]:
    """
    1D power-law rewrite:
        nn(x_j)  ->  c * x_j^exponent

    This handles negative exponents (e.g., 1/x, 1/x^2) that PolyLeaf cannot
    represent. Uses PowerLeaf which computes amp * x^exponent.

    Parameters
    ----------
    root : Node
        The AST root.
    target : AtomNode
        The NN atom to replace.
    exponent : float
        The power-law exponent (can be negative, e.g., -1 for 1/x).
    """
    if not isinstance(target, AtomNode):
        return None
    if str(target.kind).lower() != "nn":
        return None
    if effective_arity(target) != 1:
        return None

    new_atom = AtomNode(
        kind="power",
        var_idxs=tuple(int(j) for j in target.var_idxs),
        kwargs={"exponent_init": float(exponent)},
        tag=target.tag,
        inputs=clone_inputs(target),
    )
    return replace_atom_in_ast(root, target, new_atom)
