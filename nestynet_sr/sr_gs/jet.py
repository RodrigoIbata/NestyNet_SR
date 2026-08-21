# SPDX-License-Identifier: MPL-2.0
"""Jet-level generalized-symmetry witnesses for separability.

The additive criterion is the standard J^2 condition that mixed Hessian blocks
vanish.  The multiplicative criterion is the equivalent log-separability form
written without division by f:

    f f_ij - f_i f_j ≈ 0.

The implementation deliberately wraps the existing NestyNet-SR mathematical
checks so V3 can replace the corresponding procedural lane without changing the
underlying criterion.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class JetConditionSpec:
    family: str
    kind: str
    group1: tuple[Any, ...]
    group2: tuple[Any, ...]
    residual_metric: float | None = None
    complete: bool = False
    accepted: bool = True
    evidence: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["group1"] = [str(x) if not isinstance(x, int) else int(x) for x in self.group1]
        d["group2"] = [str(x) if not isinstance(x, int) else int(x) for x in self.group2]
        return d


def _tokens(symb: Sequence[Any], idxs: Sequence[int]) -> list[Any]:
    return [symb[int(i)] for i in idxs]


def jet_separability_candidates(
    *,
    symb: Sequence[Any],
    y_norm,
    dydx_norm,
    d2ydx2_norm,
    precision_sum: float,
    precision_mult: float,
    very_verbose: bool = False,
    include_loose: bool = True,
    jet_separability: bool = True,
    jet_multiplicative: bool = True,
) -> tuple[list[list[Any]], list[int] | None, list[int] | None, list[dict[str, Any]]]:
    """Return Stage-A separability candidates plus GS jet diagnostics."""

    from nestynet_sr.sr_core.separability_math import (
        COMPLETE_TOL_FACTOR,
        check_additivity,
        check_multiplicativity,
    )

    proposed: list[list[Any]] = []
    diagnostics: list[dict[str, Any]] = []
    rest_add = None
    rest_mult = None
    complete_add = False
    complete_mult = False

    if bool(jet_separability):
        add_ok, g1_add, g2_add, complete_add, resta, add_metric, add_overlapping = check_additivity(
            symb, d2ydx2_norm, precision=precision_sum, very_verbose=very_verbose
        )
        if add_ok:
            g1_tok = _tokens(symb, g1_add)
            g2_tok = _tokens(symb, g2_add)
            proposed.append([torch.add, g1_tok, g2_tok, None, add_metric])
            diagnostics.append(JetConditionSpec("jet", "additive_hessian_block", tuple(g1_tok), tuple(g2_tok), float(add_metric) if add_metric is not None else None, bool(complete_add), True, {"precision": float(precision_sum)}).as_dict())
            if add_overlapping:
                primary_key = (frozenset(g1_tok), frozenset(g2_tok))
                for g1o_local, g2o_local, mo in add_overlapping:
                    g1o_tok = _tokens(symb, g1o_local)
                    g2o_tok = _tokens(symb, g2o_local)
                    ovlp_key = (frozenset(g1o_tok), frozenset(g2o_tok))
                    if ovlp_key == primary_key or ovlp_key == (primary_key[1], primary_key[0]):
                        continue
                    proposed.append([torch.add, g1o_tok, g2o_tok, None, mo])
                    diagnostics.append(JetConditionSpec("jet", "additive_overlap_hessian_block", tuple(g1o_tok), tuple(g2o_tok), float(mo) if mo is not None else None, False, True, {"precision": float(precision_sum)}).as_dict())
            if resta:
                resta_tok = _tokens(symb, resta)
                rest_int = [t for t in resta_tok if isinstance(t, int)]
                if rest_int:
                    rest_add = list(rest_int)
        else:
            diagnostics.append({"family": "jet", "kind": "additive_hessian_block", "accepted": False, "precision": float(precision_sum)})

    if bool(jet_multiplicative):
        mult_ok, g1_mult, g2_mult, complete_mult, restm, offset_info, mult_metric = check_multiplicativity(
            symb, d2ydx2_norm, dydx_norm, y_norm, precision=precision_mult, very_verbose=very_verbose
        )
        if mult_ok:
            g1_tok = _tokens(symb, g1_mult)
            g2_tok = _tokens(symb, g2_mult)
            proposed.append([torch.multiply, g1_tok, g2_tok, offset_info, mult_metric])
            diagnostics.append(JetConditionSpec("jet", "multiplicative_ffij_minus_fifj", tuple(g1_tok), tuple(g2_tok), float(mult_metric) if mult_metric is not None else None, bool(complete_mult), True, {"precision": float(precision_mult), "offset_info": str(offset_info) if offset_info is not None else None}).as_dict())
            if restm:
                restm_tok = _tokens(symb, restm)
                rest_int = [t for t in restm_tok if isinstance(t, int)]
                if rest_int:
                    rest_mult = list(rest_int)
        else:
            diagnostics.append({"family": "jet", "kind": "multiplicative_ffij_minus_fifj", "accepted": False, "precision": float(precision_mult)})

    if bool(include_loose) and ((bool(jet_separability) and not complete_add) or (bool(jet_multiplicative) and not complete_mult)):
        prec_sum_loose = precision_sum * COMPLETE_TOL_FACTOR
        prec_mult_loose = precision_mult * COMPLETE_TOL_FACTOR
        if bool(jet_separability) and not complete_add:
            add_ok2, g1_add2, g2_add2, complete_add2, _resta2, add_metric2, add_overlapping2 = check_additivity(
                symb, d2ydx2_norm, precision=prec_sum_loose, very_verbose=very_verbose
            )
            if add_ok2 and complete_add2:
                g1_tok = _tokens(symb, g1_add2)
                g2_tok = _tokens(symb, g2_add2)
                proposed.append([torch.add, g1_tok, g2_tok, None, add_metric2])
                diagnostics.append(JetConditionSpec("jet", "additive_hessian_block_loose", tuple(g1_tok), tuple(g2_tok), float(add_metric2) if add_metric2 is not None else None, True, True, {"precision": float(prec_sum_loose)}).as_dict())
            if add_ok2 and add_overlapping2:
                existing_keys = {(frozenset(c[1]), frozenset(c[2])) for c in proposed if c[0] is torch.add}
                for g1o_local, g2o_local, mo in add_overlapping2:
                    g1o_tok = _tokens(symb, g1o_local)
                    g2o_tok = _tokens(symb, g2o_local)
                    ovlp_key = (frozenset(g1o_tok), frozenset(g2o_tok))
                    if ovlp_key in existing_keys or (ovlp_key[1], ovlp_key[0]) in existing_keys:
                        continue
                    proposed.append([torch.add, g1o_tok, g2o_tok, None, mo])
                    diagnostics.append(JetConditionSpec("jet", "additive_overlap_hessian_block_loose", tuple(g1o_tok), tuple(g2o_tok), float(mo) if mo is not None else None, False, True, {"precision": float(prec_sum_loose)}).as_dict())
        if bool(jet_multiplicative) and not complete_mult:
            mult_ok2, g1_mult2, g2_mult2, complete_mult2, _restm2, offset_info2, mult_metric2 = check_multiplicativity(
                symb, d2ydx2_norm, dydx_norm, y_norm, precision=prec_mult_loose, very_verbose=very_verbose
            )
            if mult_ok2 and complete_mult2:
                g1_tok = _tokens(symb, g1_mult2)
                g2_tok = _tokens(symb, g2_mult2)
                proposed.append([torch.multiply, g1_tok, g2_tok, offset_info2, mult_metric2])
                diagnostics.append(JetConditionSpec("jet", "multiplicative_ffij_minus_fifj_loose", tuple(g1_tok), tuple(g2_tok), float(mult_metric2) if mult_metric2 is not None else None, True, True, {"precision": float(prec_mult_loose), "offset_info": str(offset_info2) if offset_info2 is not None else None}).as_dict())

    if len(proposed) > 1:
        proposed.sort(key=lambda c: 0 if not (set(c[1]) & set(c[2])) else 1)
    return proposed, rest_add, rest_mult, diagnostics

# V3 public wrapper used by sr_search.search and by lightweight tests.
def discover_jet_separability_specs(*args, **kwargs):
    """Discover jet-level separability witnesses.

    Supported forms:

    1. ``discover_jet_separability_specs(y, grad, hess, cols=(...), cfg=...)``
       returns a list of :class:`JetConditionSpec` objects for quick testing.

    2. ``discover_jet_separability_specs(symbols=..., y_norm=..., ...)``
       returns ``(proposals, diagnostics, rest_add, rest_mult)`` for the Stage-A
       split machinery.

    3. Calls with ``leaf=...`` are accepted for backward compatibility with an
       early V3 bridge experiment, but return an empty list because jet split
       witnesses are applied in the separability path, not in the affine
       quotient-coordinate bridge.
    """

    if "leaf" in kwargs:
        return []

    # Lightweight positional probe: compute residuals directly from arrays.
    if args:
        y = args[0]
        grad = args[1] if len(args) > 1 else kwargs.get("grad")
        hess = args[2] if len(args) > 2 else kwargs.get("hess")
        cols = tuple(kwargs.get("cols", tuple(range(getattr(grad, "shape", [0, 0])[1]))))
        cfg = kwargs.get("cfg", None)
        import numpy as _np
        Y = _np.asarray(y).reshape(-1)
        G = _np.asarray(grad)
        H = _np.asarray(hess)
        specs: list[JetConditionSpec] = []
        if H.ndim == 3 and len(cols) >= 2:
            # Report pairwise additive witnesses H_ij≈0.  This is the simplest
            # J^2 version of Rod's additive separability criterion.
            scale = float(_np.nanmedian(_np.abs(H)))
            if not _np.isfinite(scale) or scale <= 0.0:
                scale = 1.0
            for a_i in range(len(cols)):
                for b_i in range(a_i + 1, len(cols)):
                    metric = float(_np.nanmedian(_np.abs(H[:, a_i, b_i])) / (scale + 1e-12))
                    tol = float(getattr(cfg, "jet_residual_tol", getattr(cfg, "residual_tol", 0.03)) if cfg is not None else 0.03)
                    specs.append(JetConditionSpec(
                        family="jet",
                        kind="additive_block",
                        group1=(cols[a_i],),
                        group2=(cols[b_i],),
                        residual_metric=metric,
                        complete=True,
                        accepted=bool(metric <= tol),
                        evidence={"criterion": "mixed_hessian", "tol": tol},
                    ))
        if H.ndim == 3 and G is not None and len(cols) >= 2:
            # Pairwise multiplicative/log-separability witness:
            # f*f_ij - f_i*f_j≈0, written without dividing by f.
            scale = float(_np.nanmedian(_np.abs(Y))) + 1e-12
            for a_i in range(len(cols)):
                for b_i in range(a_i + 1, len(cols)):
                    resid = Y * H[:, a_i, b_i] - G[:, a_i] * G[:, b_i]
                    denom = float(_np.nanmedian(_np.abs(Y * H[:, a_i, b_i])) + _np.nanmedian(_np.abs(G[:, a_i] * G[:, b_i])) + scale + 1e-12)
                    metric = float(_np.nanmedian(_np.abs(resid)) / denom)
                    tol = float(getattr(cfg, "jet_residual_tol", getattr(cfg, "residual_tol", 0.03)) if cfg is not None else 0.03)
                    specs.append(JetConditionSpec(
                        family="jet",
                        kind="multiplicative_ffij_minus_fifj",
                        group1=(cols[a_i],),
                        group2=(cols[b_i],),
                        residual_metric=metric,
                        complete=True,
                        accepted=bool(metric <= tol),
                        evidence={"criterion": "f_fij_minus_fi_fj", "tol": tol},
                    ))
        try:
            from .reporting import record_jet_event
            record_jet_event(diagnostics=[s.as_dict() for s in specs], proposals=[], context={"policy": str(getattr(cfg, "policy", "augment")) if cfg is not None else "augment", "quick_probe": True})
        except Exception:
            pass
        return specs

    symbols = kwargs["symbols"]
    y_norm = kwargs["y_norm"]
    grad_norm = kwargs["grad_norm"]
    hess_norm = kwargs["hess_norm"]
    precision_sum = kwargs["precision_sum"]
    precision_mult = kwargs["precision_mult"]
    cfg = kwargs["cfg"]
    very_verbose = bool(kwargs.get("very_verbose", False))

    proposed, rest_add, rest_mult, diagnostics = jet_separability_candidates(
        symb=symbols,
        y_norm=y_norm,
        dydx_norm=grad_norm,
        d2ydx2_norm=hess_norm,
        precision_sum=precision_sum,
        precision_mult=precision_mult,
        very_verbose=very_verbose,
        include_loose=True,
        jet_separability=bool(getattr(cfg, "jet_separability", True)),
        jet_multiplicative=bool(getattr(cfg, "jet_multiplicative", True)),
    )
    try:
        from .reporting import record_jet_event
        replace_sep = bool(getattr(cfg, "replace_separability_with_jet", lambda: False)())
        record_jet_event(
            diagnostics=diagnostics,
            proposals=proposed,
            context={
                "policy": str(getattr(cfg, "policy", "augment")),
                "replaced_baseline": bool(replace_sep),
                "precision_sum": float(precision_sum),
                "precision_mult": float(precision_mult),
            },
        )
    except Exception:
        pass
    return proposed, diagnostics, rest_add, rest_mult

# Friendly alias used in manual text.
JetSeparabilitySpec = JetConditionSpec
