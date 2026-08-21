# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Bridge Buckingham-pi unit-torus invariants into Stage-A proposals."""

from __future__ import annotations

from typing import Any, Sequence

from nestynet_sr.sr_core.ast_simplify import SimplifyOptions, node_count, simplify_ast, stable_ast_key
from nestynet_sr.sr_core.bridges import Var, ast_to_human_readable

from .config import GeneralizedSymmetryConfig
from .unit_torus import build_monomial_ast, enumerate_nullspace_exponents, exponent_l1



def _pi_simplify_options(cfg: GeneralizedSymmetryConfig) -> SimplifyOptions:
    return SimplifyOptions(
        enabled=bool(getattr(cfg, "ast_simplify", False)),
        level=str(getattr(cfg, "ast_simplify_level", "safe") or "safe"),
        domain_policy=str(getattr(cfg, "ast_simplify_domain_policy", "strict") or "strict"),
        context="stagea_invariant",
        max_passes=int(getattr(cfg, "ast_simplify_max_passes", 12)),
        trace=bool(getattr(cfg, "ast_simplify_trace", False)),
        fail_closed=True,
    )


def _prefer_pi_proposal(old: tuple, new: tuple) -> bool:
    try:
        if float(new[2]) > float(old[2]) + 1.0e-12:
            return True
        if abs(float(new[2]) - float(old[2])) <= 1.0e-12:
            return node_count(new[1]) < node_count(old[1])
    except Exception:
        pass
    return False


def stageA_unit_torus_pi_proposals(
    *,
    cols: Sequence[int],
    units_spec: Any,
    cfg: GeneralizedSymmetryConfig | None = None,
) -> tuple[list[tuple], list[dict[str, Any]]]:
    """Return Stage-A proposal tuples from Buckingham-pi invariants.

    The tuple shape matches existing Stage-A proposals:
    ``(pattern, z_ast, confidence, extra_override, meta)``.
    """

    cfg = cfg or GeneralizedSymmetryConfig()
    if not cfg.pi_invariants_active():
        return [], []
    if units_spec is None:
        diag = [{"family": "unit_torus", "kind": "pi_invariant", "accepted": False, "reason": "units_spec_missing"}]
        try:
            from .reporting import record_unit_torus_event

            record_unit_torus_event(event_type="pi_stagea", diagnostics=diag, proposals=[], context={"status": "skipped"})
        except Exception:
            pass
        return [], diag

    cols_t = tuple(int(c) for c in cols)
    x_dims_all = tuple(getattr(units_spec, "x_dims", ()) or ())
    if not cols_t or not x_dims_all:
        return [], []
    try:
        x_dims = tuple(x_dims_all[i] for i in cols_t)
    except Exception:
        return [], [{"family": "unit_torus", "kind": "pi_invariant", "accepted": False, "reason": "cols_out_of_range"}]

    exponents = enumerate_nullspace_exponents(
        x_dims,
        max_exponent=int(getattr(cfg, "pi_max_exponent", 3)),
        max_l1=int(getattr(cfg, "pi_max_l1", 6)),
        max_proposals=int(getattr(cfg, "pi_max_proposals", 24)),
        max_basis=int(getattr(cfg, "pi_max_basis", 8)),
        rational_denom=int(getattr(cfg, "pi_rational_denom", 1)),
    )
    proposals: list[tuple] = []
    diagnostics: list[dict[str, Any]] = []
    proposal_by_key: dict[tuple, tuple] = {}
    simp_opts = _pi_simplify_options(cfg) if bool(getattr(cfg, "ast_simplify", False)) else None
    variable_nodes = tuple(Var(i) for i in cols_t)
    for q in exponents:
        z_ast = build_monomial_ast(q, variable_nodes=variable_nodes)
        simp_stats = None
        proposal_key = None
        if simp_opts is not None:
            z_ast, simp_stats = simplify_ast(z_ast, simp_opts, units_spec=units_spec)
            proposal_key = stable_ast_key(z_ast, ignore_tags=True, context="stagea_invariant")
        support_local = tuple(i for i, v in enumerate(q) if v != 0)
        support_global = tuple(cols_t[i] for i in support_local)
        pattern = tuple(1 if i in support_local else 0 for i in range(len(cols_t)))
        l1 = float(exponent_l1(q))
        confidence = max(float(getattr(cfg, "pi_score_min_confidence", 0.65)), 1.0 / (1.0 + 0.05 * l1))
        try:
            z_human = ast_to_human_readable(z_ast)
        except Exception:
            z_human = repr(z_ast)
        meta = {
            "kind": "gs_unit_torus",
            "source": "gs_unit_torus",
            "gs_source_family": "unit_torus",
            "gs_family": "unit_torus",
            "gs_kind": "pi_invariant",
            "gs_axes": support_global,
            "pi_exponents": tuple(str(v) for v in q),
            "pi_support_local": support_local,
            "pi_support_global": support_global,
            "pi_l1": l1,
            "gs_confidence": confidence,
            "z_human": z_human,
        }
        if simp_stats is not None:
            meta["ast_simplify"] = simp_stats.to_dict()
        diagnostics.append(
            {
                "family": "unit_torus",
                "kind": "pi_invariant",
                "accepted": True,
                "axes": support_global,
                "exponents": tuple(str(v) for v in q),
                "l1": l1,
                "confidence": confidence,
                "invariant": z_human,
            }
        )
        if cfg.dim_policy_proposes():
            proposal = (pattern, z_ast, confidence, None, meta)
            if proposal_key is None:
                proposals.append(proposal)
            else:
                old = proposal_by_key.get(proposal_key)
                if old is None or _prefer_pi_proposal(old, proposal):
                    proposal_by_key[proposal_key] = proposal

    if proposal_by_key:
        proposals.extend(proposal_by_key.values())

    try:
        from .reporting import record_unit_torus_event

        record_unit_torus_event(
            event_type="pi_stagea",
            diagnostics=diagnostics,
            proposals=proposals,
            context={
                "mode": str(getattr(cfg, "mode", "propose")),
                "dim_policy": cfg.canonical_dim_policy(),
                "cols": cols_t,
                "proposing": bool(cfg.dim_policy_proposes()),
            },
        )
    except Exception:
        pass
    return proposals, diagnostics
