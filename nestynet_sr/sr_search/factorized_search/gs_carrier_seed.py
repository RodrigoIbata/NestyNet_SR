# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Generalized-symmetry carrier seeds for the SR factorized symbolic search.

This is the SR-side GS -> FSS bridge.  The GS layer (charts / pairwise-witness
composition / bounded recursion / discovered warp) is very good at *finding
the internal coordinate* ``z(x)`` of a target, precisely the structural
sub-problem the factorized-search skeleton enumeration is weakest at. Once
``z`` is known, the FSS outer-map battery (poly / power / Pade / sine / exp)
fits ``g(z)`` in closed form. This module runs the shared carrier bank on a
torch-callable target, converts each certified coordinate into the FSS
tuple-AST form, and preserves its certificate while the engine scores
``g(z)`` directly.

Everything here is opt-in: it runs only when ``oracle_lab`` is invoked with
``--gs-carrier-seed`` (or ``run_oracle_equation(..., gs_carrier_seed=True)``).
With no GS seeds the engine phase is a no-op, so baseline behaviour is
unchanged.
"""

from __future__ import annotations

from typing import Any, Callable

import torch


def nonlinear_invariant_carrier_seeds(
    compilation: Any,
    *,
    orbit_coordinate: Any = None,
    max_seeds: int = 8,
) -> tuple[list[tuple], list[dict[str, Any]]]:
    """Convert certified nonlinear invariant/orbit ASTs into FSS seeds.

    The nonlinear invariant compiler deliberately remains search-engine
    agnostic.  This adapter is the explicit hand-off to factorized symbolic
    search: accepted carriers become tuple ASTs and retain certificate
    provenance for matched-vocabulary opportunity accounting.
    """

    from nestynet_sr.sr_core.bridges import ast_to_human_readable
    from .bridge import nestynet_to_factorized_search

    limit = max(0, int(max_seeds))
    if limit == 0:
        return [], []

    candidates = list(getattr(compilation, "invariants", ()) or ())
    candidates.extend(list(getattr(compilation, "orbit_coordinates", ()) or ()))
    if orbit_coordinate is not None:
        candidates.append(orbit_coordinate)
    seeds: list[tuple] = []
    diagnostics: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in candidates:
        if not bool(getattr(row, "accepted", False)):
            continue
        ast = getattr(row, "ast", None)
        if ast is None:
            continue
        try:
            seed = nestynet_to_factorized_search(ast)
        except Exception:
            continue
        key = repr(seed)
        if key in seen:
            continue
        seen.add(key)
        seeds.append(seed)
        try:
            human = ast_to_human_readable(ast)
        except Exception:
            human = repr(ast)
        diagnostics.append(
            {
                "z_human": human,
                "gs_source_family": (
                    "nonlinear_orbit_coordinate"
                    if row is orbit_coordinate
                    or row in tuple(getattr(compilation, "orbit_coordinates", ()) or ())
                    else "nonlinear_point_invariant"
                ),
                "certificate": row.to_report() if hasattr(row, "to_report") else {},
            }
        )
        if len(seeds) >= limit:
            break
    return seeds, diagnostics


def default_gs_carrier_cfg():
    """A broad GS configuration: all charts + composition, casting a wide net.

    Multiple certified coordinates are fine — each is scored independently by
    the FSS outer-map battery and only the correct one closes the problem.
    """

    from nestynet_sr.sr_gs import GeneralizedSymmetryConfig

    return GeneralizedSymmetryConfig(
        enabled=True,
        mode="propose",
        policy="augment",
        general_affine=True,
        general_affine_charts=("identity", "log", "reciprocal", "warp"),
        general_affine_promotion_noise_calibrated=True,
        pairwise_composition=True,
        recursive_composition=True,
        recursive_composition_max_depth=3,
        recursive_composition_beam_width=2,
        lorentz_boosts=True,
    )


def discover_gs_carrier_seeds(
    target_fn: Callable[[torch.Tensor], torch.Tensor],
    x_fit: torch.Tensor,
    *,
    n_var: int,
    cfg: Any | None = None,
    max_seeds: int = 8,
) -> tuple[list[tuple], list[dict[str, Any]]]:
    """Discover GS coordinates for ``target_fn`` and return FSS tuple-AST seeds.

    Parameters
    ----------
    target_fn : callable
        Torch-callable target ``f(X)->[N,1]`` expecting all ``x_fit`` columns
        (variables first, then any fixed constants).
    x_fit : torch.Tensor ``[N, D]``
        Sampled inputs; the first ``n_var`` columns are the free variables.
    n_var : int
        Number of free variables (GS operates on columns ``0..n_var-1``; any
        trailing columns are held at their sampled constant values).

    Returns ``(seed_exprs, diagnostics)`` where ``seed_exprs`` is a list of FSS
    tuple-ASTs and ``diagnostics`` records the certified coordinates and their
    shared-bank provenance.
    """

    from nestynet_sr.sr_gs.carrier_bank import discover_gs_carriers
    from nestynet_sr.sr_core.bridges import ast_to_human_readable
    from .bridge import nestynet_to_factorized_search

    if cfg is None:
        cfg = default_gs_carrier_cfg()
    seed_limit = max(0, int(max_seeds))
    if seed_limit == 0:
        return [], []

    x_fit = torch.as_tensor(x_fit, dtype=torch.float64)
    x_var = x_fit[:, :n_var]

    # A leaf over the free variables; trailing constant columns (if any) are
    # pinned at their sampled values so the warp chart's autograd Hessian and
    # the gradient field are taken w.r.t. the variables only.
    if x_fit.shape[1] > n_var:
        const_cols = x_fit[:1, n_var:].detach()

        def leaf(x_var_batch: torch.Tensor) -> torch.Tensor:
            pad = const_cols.to(x_var_batch).expand(x_var_batch.shape[0], -1)
            return target_fn(torch.cat([x_var_batch, pad], dim=1)).reshape(-1)
    else:

        def leaf(x_var_batch: torch.Tensor) -> torch.Tensor:
            return target_fn(x_var_batch).reshape(-1)

    xt = x_var.detach().clone().requires_grad_(True)
    y = leaf(xt)
    (grad_t,) = torch.autograd.grad(y.sum(), xt, create_graph=False)
    grad = grad_t.detach().cpu().numpy()
    x_np = x_var.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    carriers, _diag = discover_gs_carriers(
        atom=None, leaf=leaf, x_vals=x_np, dydx_vals=grad,
        cols=tuple(range(n_var)), y_vals=y_np, cfg=cfg,
    )

    seeds: list[tuple] = []
    diagnostics: list[dict[str, Any]] = []
    seen: set[str] = set()
    for carrier in carriers:
        if not bool(carrier.certified):
            continue
        meta = dict(carrier.metadata)
        try:
            tup = nestynet_to_factorized_search(carrier.ast)
        except Exception:
            continue
        key = repr(tup)
        if key in seen:
            continue
        seen.add(key)
        seeds.append(tup)
        try:
            human = ast_to_human_readable(carrier.ast)
        except Exception:
            human = repr(carrier.ast)
        diagnostics.append({
            "z_human": human,
            "gs_source_family": meta.get("gs_source_family"),
            "gs_chart": meta.get("gs_chart"),
            "gs_carrier_depth": int(carrier.depth),
            "gs_carrier_fingerprint": str(carrier.fingerprint),
            "gs_recursive_parent_fingerprints": list(carrier.parent_fingerprints),
            "carrier_metadata": meta,
            "carrier_report": carrier.to_report(),
        })
        if len(seeds) >= seed_limit:
            break

    return seeds, diagnostics
