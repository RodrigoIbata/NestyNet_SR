#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Pre-discovery conditioning / alias audit for the broadened vector library.

Before running STLSQ we build the *same* per-equation feature columns the
discovery engine uses (via ``_eval_ast`` on a ``UFeatureCache``), normalize
them, and inspect the Gram matrix.  We report, per equation:

* ``mu``        - the largest off-diagonal feature correlation,
* ``rank``      - numerical column rank,
* ``sigma_min`` - smallest singular value of the normalized design,
* alias pairs   - feature columns with |correlation| > 1 - eps (exact aliases).

This turns "you put Maxwell in the menu" into linear algebra: a rank-deficient
case (e.g. a single-mode plane wave, where lap(E) = -k^2 E aliases E) is flagged
as ``NONIDENTIFIABLE_ALIAS`` rather than silently passed or dropped, while the
identifiable cases earn a "no exact Laplacian aliases; decoys rejected" line.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from nestynet_sr.adaptors.u_feature_leaf import UFeatureCache
from nestynet_sr.sr_de.system_de_search import _as_N, _eval_ast

from problem_defs import _vec_key


def _alias_drop_choice(name_i: str, name_j: str) -> int:
    """Return which index (i=0 or j=1) of an alias pair to drop.

    Exact aliases identify an *equivalence class*, not a unique term (e.g.
    E ~ -lap(E)); the data cannot choose between them.  We apply a single
    PREDECLARED tie-breaker — retain the lowest differential-order representative
    (drop the Laplacian) — as a reproducible convention, NOT as a physics
    preference among indistinguishable columns.  Recovery is reported "modulo
    the alias class".
    """
    li = name_i.lower().startswith("laplacian")
    lj = name_j.lower().startswith("laplacian")
    if lj and not li:
        return 1
    if li and not lj:
        return 0
    return 1  # default: drop the second-listed column


def audit_vector_library(
    surrogate: torch.nn.Module,
    X: torch.Tensor,
    vector_terms: Sequence[tuple],
    name_by_key: dict[str, str],
    equations: Sequence[Any],
    *,
    alias_eps: float = 1e-6,
    rank_rtol: float = 1e-9,
    return_gram: bool = False,
) -> dict[str, Any]:
    """Audit the conditioning of the candidate feature library.

    Returns a report dict with per-equation conditioning, a deduplicated list of
    exact alias pairs, an overall ``rank_status`` ("FULL_RANK" or
    "NONIDENTIFIABLE_ALIAS"), and the term names recommended for removal so the
    design becomes full rank.  With ``return_gram=True`` each per-equation report
    also carries the normalized-column Gram matrix (for plotting); ``term_names``
    (engine order) is always included so callers can map columns to labels.
    """
    cache = UFeatureCache(surrogate)
    names = [name_by_key.get(_vec_key(t), f"term{i}") for i, t in enumerate(vector_terms)]

    eq_reports: list[dict[str, Any]] = []
    alias_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    drop_names: set[str] = set()
    mu_overall = 0.0
    min_rank_ratio = 1.0
    max_cond = 1.0

    with torch.no_grad():
        for eq in equations:
            out_idxs = tuple(int(i) for i in eq.out_idxs)
            dim = len(out_idxs)
            cols = []
            for t in vector_terms:
                comp = [_as_N(_eval_ast(t[ci], X, cache)) for ci in range(dim)]
                cols.append(torch.cat(comp, dim=0))
            Phi = torch.stack(cols, dim=1)  # (N*dim, K)
            norms = Phi.norm(dim=0)
            nz = norms > 0
            Phi_n = torch.zeros_like(Phi)
            Phi_n[:, nz] = Phi[:, nz] / norms[nz]
            G = Phi_n.t() @ Phi_n  # (K,K)

            K = int(Phi.shape[1])
            off = G - torch.diag(torch.diag(G))
            mu = float(off.abs().max()) if K > 1 else 0.0
            svals = torch.linalg.svdvals(Phi_n)
            smax = float(svals[0]) if svals.numel() else 0.0
            smin = float(svals[-1]) if svals.numel() else 0.0
            rtol = rank_rtol * max(smax, 1e-300)
            rank = int((svals > rtol).sum())
            rank_ratio = rank / max(K, 1)
            cond = smax / max(smin, 1e-300)  # kappa(Theta-tilde); huge if rank-deficient
            mu_overall = max(mu_overall, mu)
            min_rank_ratio = min(min_rank_ratio, rank_ratio)
            max_cond = max(max_cond, cond)

            eq_aliases = []
            for i in range(K):
                for j in range(i + 1, K):
                    corr = float(G[i, j])
                    if abs(corr) > 1.0 - alias_eps:
                        pair = tuple(sorted((names[i], names[j])))
                        eq_aliases.append({"a": names[i], "b": names[j], "corr": corr})
                        alias_pairs.setdefault(
                            pair, {"a": pair[0], "b": pair[1], "corr": corr, "equations": []}
                        )["equations"].append(eq.name)
                        which = _alias_drop_choice(names[i], names[j])
                        drop_names.add(names[i] if which == 0 else names[j])

            eq_entry = {
                "equation": eq.name,
                "n_terms": K,
                "n_nonzero_cols": int(nz.sum()),
                "max_offdiag_corr": mu,
                "numerical_rank": rank,
                "rank_deficient": rank < int(nz.sum()),
                "sigma_max": smax,
                "sigma_min": smin,
                "cond_number": cond,
                "aliases": eq_aliases,
            }
            if return_gram:
                eq_entry["gram"] = G.detach().cpu().tolist()
            eq_reports.append(eq_entry)

    rank_status = "NONIDENTIFIABLE_ALIAS" if alias_pairs else "FULL_RANK"
    return {
        "alias_eps": float(alias_eps),
        "rank_status": rank_status,
        "max_offdiag_corr": mu_overall,
        "max_cond_number": max_cond,
        "min_rank_ratio": min_rank_ratio,
        "alias_pairs": list(alias_pairs.values()),
        "drop_terms": sorted(drop_names),
        "term_names": names,
        "equations": eq_reports,
    }


def support_conditioning(
    surrogate: torch.nn.Module,
    X: torch.Tensor,
    support_terms: Sequence[tuple],
    equations: Sequence[Any],
    *,
    name_by_key: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Conditioning of the *discovered support* (per equation).

    Reports, for the selected term columns, the condition number kappa(Theta_S)
    and per-term variance-inflation factors VIF_i = [(Theta_S^T Theta_S)^-1]_ii
    (normalized columns).  VIF ~ 1 means a coefficient is well-determined; large
    VIF means the estimate is noise-sensitive even though the support is
    identifiable.  This answers "is the *answer* stable?", distinct from the
    full-library coherence that answers "are coherent decoys present?".
    """
    terms = [t for t in support_terms if t is not None]
    cache = UFeatureCache(surrogate)
    names = [
        (name_by_key.get(_vec_key(t), f"term{i}") if name_by_key else f"term{i}")
        for i, t in enumerate(terms)
    ]
    per_eq: list[dict[str, Any]] = []
    max_vif = 1.0
    max_cond = 1.0
    vif_by_term: dict[str, float] = {n: 1.0 for n in names}

    with torch.no_grad():
        for eq in equations:
            dim = len(tuple(eq.out_idxs))
            cols = [
                torch.cat([_as_N(_eval_ast(t[ci], X, cache)) for ci in range(dim)], dim=0)
                for t in terms
            ]
            if not cols:
                per_eq.append({"equation": eq.name, "n_terms": 0, "max_vif": 1.0, "cond_number": 1.0})
                continue
            Phi = torch.stack(cols, dim=1)
            norms = Phi.norm(dim=0)
            nz = norms > 0
            Phi_n = torch.zeros_like(Phi)
            Phi_n[:, nz] = Phi[:, nz] / norms[nz]
            G = Phi_n.t() @ Phi_n
            svals = torch.linalg.svdvals(Phi_n)
            cond = float(svals[0]) / max(float(svals[-1]), 1e-300)
            try:
                Ginv = torch.linalg.inv(G)
            except torch.linalg.LinAlgError:
                Ginv = torch.linalg.pinv(G)
            vif = torch.diag(Ginv).clamp_min(0.0)
            eq_max_vif = float(vif.max()) if vif.numel() else 1.0
            for i, nm in enumerate(names):
                vif_by_term[nm] = max(vif_by_term[nm], float(vif[i]))
            max_vif = max(max_vif, eq_max_vif)
            max_cond = max(max_cond, cond)
            per_eq.append(
                {
                    "equation": eq.name,
                    "n_terms": int(Phi.shape[1]),
                    "max_vif": eq_max_vif,
                    "cond_number": cond,
                }
            )

    return {
        "max_vif": max_vif,
        "max_cond_number": max_cond,
        "vif_by_term": vif_by_term,
        "equations": per_eq,
    }


def apply_alias_drops(
    vector_terms: list[tuple],
    name_by_key: dict[str, str],
    named_vecs: dict[str, tuple],
    drop_names: Sequence[str],
) -> tuple[list[tuple], dict[str, str], dict[str, tuple]]:
    """Return a copy of the library with the named (aliased) terms removed."""
    drop = set(drop_names)
    keep_terms = []
    for t in vector_terms:
        nm = name_by_key.get(_vec_key(t))
        if nm in drop:
            continue
        keep_terms.append(t)
    new_name_by_key = {_vec_key(t): name_by_key.get(_vec_key(t)) for t in keep_terms}
    new_named_vecs = {nm: v for nm, v in named_vecs.items() if nm not in drop}
    return keep_terms, new_name_by_key, new_named_vecs
