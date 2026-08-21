# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Consumer-neutral generalized-symmetry carrier discovery.

The GS layer discovers and recursively composes analytic inner coordinates
``z(x)``.  Consumers decide what those coordinates are worth:

* Stage A must empirically prove that ``NN(z)`` simplifies the current model.
* Factorized symbolic search fits and validates an explicit outer map ``g(z)``.

Keeping the carrier bank independent of either consumer prevents proposal
evidence from becoming acceptance authority and makes recursive coordinates
available to both paths with identical certificates and provenance.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from nestynet_sr.sr_core.ast_simplify import (
    SimplifyOptions,
    node_count,
    simplify_ast,
    stable_ast_key,
)
from nestynet_sr.sr_core.bridges import _collect_var_idxs_from_node
from nestynet_sr.sr_core.carrier_units import (
    CARRIER_INTERNAL_UNITS_INVALID,
    mark_inner_coordinate_metadata,
)

from .config import GeneralizedSymmetryConfig


@dataclass(frozen=True)
class GSCarrier:
    """One certified GS coordinate in the shared carrier bank."""

    pattern: tuple[int, ...]
    ast: Any
    confidence: float
    extra_override: tuple[int, ...] | None
    metadata: Mapping[str, Any]
    support: tuple[int, ...]
    depth: int
    parent_fingerprints: tuple[str, ...]
    fingerprint: str
    certified: bool
    carrier_dim: tuple[float, ...] | None = None

    @property
    def full_support(self) -> bool:
        return bool(self.pattern) and all(int(value) != 0 for value in self.pattern)

    def to_stagea_proposal(self) -> tuple:
        """Return the legacy compound-proposal tuple without sharing metadata."""

        extra = None if self.extra_override is None else tuple(self.extra_override)
        return (
            tuple(self.pattern),
            self.ast,
            float(self.confidence),
            extra,
            copy.deepcopy(dict(self.metadata)),
        )

    def to_report(self) -> dict[str, Any]:
        return {
            "fingerprint": str(self.fingerprint),
            "support": list(self.support),
            "depth": int(self.depth),
            "parent_fingerprints": list(self.parent_fingerprints),
            "confidence": float(self.confidence),
            "certified": bool(self.certified),
            "carrier_dim": (
                None if self.carrier_dim is None else [float(v) for v in self.carrier_dim]
            ),
            "metadata": copy.deepcopy(dict(self.metadata)),
        }


def _proposal_fingerprint(ast: Any) -> str:
    try:
        canonical, _stats = simplify_ast(
            ast,
            SimplifyOptions(
                enabled=True,
                level="safe",
                domain_policy="strict",
                context="gs_carrier_bank",
                max_passes=12,
                trace=False,
                fail_closed=True,
            ),
        )
        key = stable_ast_key(canonical, ignore_tags=True, context="gs_carrier_bank")
        return repr(key)
    except Exception:
        return repr(ast)


def _proposal_support(ast: Any, pattern: Sequence[Any]) -> tuple[int, ...]:
    try:
        support = tuple(sorted(int(v) for v in _collect_var_idxs_from_node(ast)))
        if support:
            return support
    except Exception:
        pass
    return tuple(i for i, value in enumerate(pattern) if int(value) != 0)


def _coerce_extra(extra: Any) -> tuple[int, ...] | None:
    if extra is None:
        return None
    try:
        return tuple(int(v) for v in extra)
    except Exception:
        return None


def _carrier_from_proposal(
    proposal: tuple,
    *,
    units_spec=None,
) -> GSCarrier | None:
    if len(proposal) < 5:
        return None
    pattern, ast, confidence, extra, raw_meta = proposal[:5]
    meta = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}
    if str(meta.get("source", "")) != "generalized_symmetry":
        return None

    carrier_dim = None
    certified = units_spec is None
    if units_spec is not None:
        try:
            from nestynet_sr.sr_core.units import eval_analytic_expr_dim

            carrier_dim_raw = eval_analytic_expr_dim(ast, units_spec.x_dims)
            if carrier_dim_raw is not None:
                carrier_dim = tuple(float(v) for v in carrier_dim_raw)
                certified = True
        except Exception:
            certified = False

    meta = mark_inner_coordinate_metadata(
        meta,
        source=str(meta.get("source", "generalized_symmetry") or "generalized_symmetry"),
        certified=bool(certified),
    )
    if carrier_dim is not None:
        meta["carrier_dim"] = [float(v) for v in carrier_dim]
    elif units_spec is not None:
        meta["carrier_unit_diagnostic"] = CARRIER_INTERNAL_UNITS_INVALID

    try:
        depth = max(1, int(meta.get("gs_recursive_depth", meta.get("gs_carrier_depth", 1))))
    except Exception:
        depth = 1
    fingerprint = _proposal_fingerprint(ast)
    parents = tuple(str(v) for v in (meta.get("gs_recursive_parent_fingerprints") or ()))
    meta["gs_carrier_depth"] = int(depth)
    meta["gs_carrier_fingerprint"] = str(fingerprint)
    meta["gs_recursive_parent_fingerprints"] = parents
    return GSCarrier(
        pattern=tuple(int(v) for v in pattern),
        ast=ast,
        confidence=float(confidence),
        extra_override=_coerce_extra(extra),
        metadata=meta,
        support=_proposal_support(ast, pattern),
        depth=int(depth),
        parent_fingerprints=parents,
        fingerprint=fingerprint,
        certified=bool(certified),
        carrier_dim=carrier_dim,
    )


def _carrier_rank_key(carrier: GSCarrier) -> tuple:
    """Prefer broad, shallow, confident, compact carriers."""

    try:
        complexity = int(node_count(carrier.ast))
    except Exception:
        complexity = 10**9
    return (
        int(carrier.full_support),
        len(carrier.support),
        float(carrier.confidence),
        -int(carrier.depth),
        -complexity,
        str(carrier.fingerprint),
    )


def rank_gs_carriers(carriers: Sequence[GSCarrier]) -> list[GSCarrier]:
    return sorted(list(carriers), key=_carrier_rank_key, reverse=True)


def _dedupe_carriers(
    carriers: Sequence[GSCarrier],
    *,
    limit: int | None = None,
) -> list[GSCarrier]:
    best: dict[str, GSCarrier] = {}
    for carrier in carriers:
        old = best.get(carrier.fingerprint)
        if old is None or _carrier_rank_key(carrier) > _carrier_rank_key(old):
            best[carrier.fingerprint] = carrier
    ranked = rank_gs_carriers(best.values())
    if limit is not None:
        ranked = ranked[: max(0, int(limit))]
    return ranked


def discover_gs_carriers(
    *,
    atom,
    leaf,
    x_vals,
    dydx_vals,
    cols: Sequence[int],
    y_vals=None,
    device=None,
    cfg: GeneralizedSymmetryConfig | None = None,
    units_spec=None,
) -> tuple[list[GSCarrier], list[dict[str, Any]]]:
    """Discover, certify, and recursively compose a bounded GS carrier bank."""

    from .stagea_bridge import (
        _discover_generalized_symmetry_proposal_tuples,
        _eval_leaf_values,
        _to_numpy,
    )

    cfg = cfg or GeneralizedSymmetryConfig()
    if not cfg.active():
        return [], []

    y_array = (
        _eval_leaf_values(leaf, x_vals, device=device)
        if y_vals is None
        else _to_numpy(y_vals).reshape(-1)
    )
    primitive, diagnostics = _discover_generalized_symmetry_proposal_tuples(
        atom=atom,
        leaf=leaf,
        x_vals=x_vals,
        dydx_vals=dydx_vals,
        cols=cols,
        y_vals=y_array,
        device=device,
        cfg=cfg,
        units_spec=units_spec,
        _include_recursive=False,
    )
    discovered_carriers = [
        carrier
        for proposal in primitive
        if (carrier := _carrier_from_proposal(proposal, units_spec=units_spec)) is not None
    ]
    carriers = []
    for carrier in discovered_carriers:
        if carrier.certified:
            carriers.append(carrier)
            continue
        diagnostics.append(
            {
                "family": "generalized_symmetry",
                "kind": "carrier_certification",
                "accepted": False,
                "status": "rejected",
                "depth": int(carrier.depth),
                "reason": CARRIER_INTERNAL_UNITS_INVALID,
                "fingerprint": carrier.fingerprint,
            }
        )

    max_bank = max(1, int(getattr(cfg, "max_stagea_proposals", 12) or 12))
    carriers = _dedupe_carriers(carriers, limit=max_bank)
    if (
        y_array is None
        or not bool(getattr(cfg, "recursive_composition_active", lambda: False)())
        or not carriers
    ):
        return carriers, diagnostics

    try:
        max_depth = max(1, int(getattr(cfg, "recursive_composition_max_depth", 3) or 3))
    except Exception:
        max_depth = 3
    try:
        beam_width = max(1, int(getattr(cfg, "recursive_composition_beam_width", 2) or 2))
    except Exception:
        beam_width = 2

    all_carriers = list(carriers)
    frontier = list(carriers)
    seen = {carrier.fingerprint for carrier in all_carriers}
    x_array = np.asarray(x_vals, dtype=float)
    grad_array = np.asarray(dydx_vals, dtype=float)

    for output_depth in range(2, max_depth + 1):
        from .recursive_composition import compose_recursive_coordinate_proposals

        try:
            recursive, recursive_diag = compose_recursive_coordinate_proposals(
                [carrier.to_stagea_proposal() for carrier in frontier],
                joint_proposals=[carrier.to_stagea_proposal() for carrier in all_carriers],
                x_vals=x_array,
                y_vals=y_array,
                dydx_vals=grad_array,
                cols=cols,
                cfg=cfg,
                output_depth=output_depth,
                beam_width=beam_width,
            )
        except Exception as exc:
            diagnostics.append(
                {
                    "family": "generalized_symmetry",
                    "kind": "recursive_composition",
                    "accepted": False,
                    "status": "rejected",
                    "depth": int(output_depth),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            break
        diagnostics.extend(list(recursive_diag or ()))
        new_carriers: list[GSCarrier] = []
        for proposal in recursive:
            carrier = _carrier_from_proposal(proposal, units_spec=units_spec)
            if carrier is None or carrier.fingerprint in seen:
                continue
            if not carrier.certified:
                diagnostics.append(
                    {
                        "family": "generalized_symmetry",
                        "kind": "recursive_composition",
                        "accepted": False,
                        "status": "rejected",
                        "depth": int(output_depth),
                        "reason": CARRIER_INTERNAL_UNITS_INVALID,
                        "fingerprint": carrier.fingerprint,
                    }
                )
                continue
            seen.add(carrier.fingerprint)
            new_carriers.append(carrier)
        if not new_carriers:
            break
        frontier = rank_gs_carriers(new_carriers)[:beam_width]
        all_carriers.extend(frontier)
        all_carriers = _dedupe_carriers(all_carriers, limit=max_bank)

    return _dedupe_carriers(all_carriers, limit=max_bank), diagnostics


__all__ = [
    "GSCarrier",
    "discover_gs_carriers",
    "rank_gs_carriers",
]
