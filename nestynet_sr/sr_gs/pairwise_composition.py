# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Compose accepted pairwise witnesses into global monomial and linear rays.

The global affine determining solve degrades at realistic surrogate gradient
noise (its 3+-variable algebras lose bracket closure around ~1e-3 relative
error), but the *pairwise* witnesses — one-dimensional tests — remain reliably
accepted there.  Each accepted pair imposes an exact ratio relation on the ray
coefficients of an invariant coordinate:

* scaling pairs constrain the exponent ray ``e`` of a monomial invariant
  ``prod_i x_i**e_i`` (``common_pair(i,j)``: ``e_j = -e_i``;
  ``opposite_pair(i,j)``: ``e_j = +e_i``), validated in the log chart;
* translation pairs constrain the covector ``c`` of a linear invariant
  ``sum_i c_i x_i`` (``diagonal_plus(i,j)`` / invariant ``x_i - x_j``:
  ``c_j = -c_i``; ``diagonal_minus(i,j)`` / invariant ``x_i + x_j``:
  ``c_j = +c_i``; learned ``affine_translation_pair`` with generator
  components ``(b_i, b_j)``: ``c_j = -(b_i/b_j) c_i``), validated in the
  identity chart (no positivity requirement).

A connected component of the pair-constraint graph determines its ray by
value propagation from a root, with contradictions detected on back-edges.
The ray is snapped to a primitive integer vector, re-checked against the pair
constraints, then validated *jointly* against the sampled gradients with a
tolerance calibrated to the pair witnesses' own residuals, and finally
emitted as an ordinary promoted Stage-A proposal (same meta contract as
reduction promotions, so replace-shadowed suppression and cross-route dedup
apply unchanged).

±1-ratio pairs can only produce ±1 rays (products/ratios, sums/differences);
rays with larger coefficients arise only from learned affine pairs or remain
the global solve's complementary job.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from nestynet_sr.sr_core import build_linear_ast, build_radial_r2_ast
from nestynet_sr.sr_core.bridges import AddNode, ConstNode, MulNode, PowNode, Var, ast_to_human_readable

from .charts import _MAX_ABS_SNAPPED_EXPONENT, _SNAP_SANITY_CEILING
from .unit_torus import _as_fraction, build_monomial_ast, projective_exponent_key

# spec.kind -> (ray family, fixed ratio of the second axis relative to the
# first; None means the ratio comes from the spec's generator coefficients).
# The quadratic family propagates the *signature* of a quadratic-form
# invariant sum_i s_i x_i**2: a rotation pair preserves x_i**2 + x_j**2
# (s_j = +s_i) while a Lorentz boost pair preserves x_i**2 - x_j**2
# (s_j = -s_i), so mixed rotation/boost graphs compose Minkowski-type
# invariants; the all-plus component is the classic radius.
_PAIR_KINDS = {
    "common_pair": ("monomial", -1.0),
    "affine_common_scaling_pair": ("monomial", -1.0),
    "opposite_pair": ("monomial", +1.0),
    "affine_opposite_scaling_pair": ("monomial", +1.0),
    "diagonal_plus": ("linear", -1.0),
    "learned_diagonal_translation": ("linear", -1.0),
    "diagonal_minus": ("linear", +1.0),
    "learned_signed_translation": ("linear", +1.0),
    "affine_translation_pair": ("linear", None),
    "so2_pair": ("quadratic", +1.0),
    "learned_rotation": ("quadratic", +1.0),
    "affine_rotation_pair": ("quadratic", +1.0),
    "boost_pair": ("quadratic", -1.0),
    "learned_lorentz_boost": ("quadratic", -1.0),
    "affine_lorentz_pair": ("quadratic", -1.0),
}

_OUTPUT_ACTION_EPS = 1.0e-10
_RATIO_CONSISTENCY_RTOL = 1.0e-6


def _finite_rowwise_linear_combination(
    matrix: np.ndarray,
    coefficients: Sequence[float],
) -> np.ndarray | None:
    """Return ``matrix @ coefficients`` without overflowing intermediate sums.

    The combination is evaluated after scaling each row to unit magnitude.
    A mathematically unrepresentable result, or any non-finite input, fails
    closed with ``None`` so it cannot seed a virtual GS coordinate.
    """

    values = np.asarray(matrix, dtype=float)
    coeffs = np.asarray(tuple(float(value) for value in coefficients), dtype=float)
    if (
        values.ndim != 2
        or values.size == 0
        or coeffs.ndim != 1
        or values.shape[1] != coeffs.size
        or coeffs.size == 0
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(coeffs))
    ):
        return None

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        combined = values @ coeffs
    if np.all(np.isfinite(combined)):
        return combined

    row_scale = np.max(np.abs(values), axis=1)
    safe_scale = np.where(row_scale > 0.0, row_scale, 1.0)
    scaled = values / safe_scale[:, None]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        combined_scaled = (scaled @ coeffs) * safe_scale
    if not np.all(np.isfinite(combined_scaled)):
        return None
    return combined_scaled


def compose_pairwise_witness_proposals(
    specs: Sequence[Any],
    *,
    x_vals: Any,
    dydx_vals: Any,
    cols: Sequence[int],
    cfg: Any,
    max_proposals: int = 4,
    calibrate_pair_residuals_to_joint_metric: bool = False,
) -> tuple[list[tuple], list[dict[str, Any]]]:
    """Return (proposals, diagnostics) for composed pairwise rays.

    ``GeneratorSpec.residual_rel`` is normalized by target variation, whereas
    the final composed-ray checks are normalized by gradient energy.  Those
    scales coincide well enough for ordinary first-level discovery, but not
    necessarily after recursive carriers become virtual coordinates.  The
    opt-in calibration recomputes each component's pair baseline in the same
    metric as its joint check.  Recursive composition enables it; legacy
    first-level behavior remains unchanged.
    """

    cols_t = tuple(int(c) for c in cols)
    col_index = {c: i for i, c in enumerate(cols_t)}
    x_arr = np.asarray(x_vals, dtype=float)
    grad_arr = np.asarray(dydx_vals, dtype=float)

    max_denominator = max(1, int(getattr(cfg, "general_affine_chart_snap_denominator", 4) or 4))
    proposals: list[tuple] = []
    diagnostics: list[dict[str, Any]] = []
    for family in ("monomial", "linear", "quadratic"):
        edges = dict(
            _collect_edges(specs, family=family, col_index=col_index, max_denominator=max_denominator)
        )
        if family in ("monomial", "quadratic"):
            centered = _collect_centered_edges(
                specs, family=family, col_index=col_index, x_arr=x_arr, cfg=cfg
            )
            for key, entry in centered.items():
                if key not in edges:
                    edges[key] = entry
        if not edges:
            continue
        family_proposals, family_diag = _compose_family(
            family,
            edges,
            x_arr=x_arr,
            grad_arr=grad_arr,
            cols_t=cols_t,
            cfg=cfg,
            budget=max(1, int(max_proposals)) - len(proposals),
            calibrate_pair_residuals_to_joint_metric=bool(
                calibrate_pair_residuals_to_joint_metric
            ),
        )
        proposals.extend(family_proposals)
        diagnostics.extend(family_diag)
        if len(proposals) >= max(1, int(max_proposals)):
            break
    if len(proposals) < max(1, int(max_proposals)):
        virtual_proposals, virtual_diag = _compose_virtual_linear_products(
            specs,
            x_arr=x_arr,
            grad_arr=grad_arr,
            cols_t=cols_t,
            cfg=cfg,
            max_denominator=max_denominator,
            budget=max(1, int(max_proposals)) - len(proposals),
        )
        proposals.extend(virtual_proposals)
        diagnostics.extend(virtual_diag)
    return proposals, diagnostics


def _compose_virtual_linear_products(
    specs: Sequence[Any],
    *,
    x_arr: np.ndarray,
    grad_arr: np.ndarray,
    cols_t: tuple,
    cfg: Any,
    max_denominator: int,
    budget: int,
) -> tuple[list[tuple], list[dict[str, Any]]]:
    """Two-level composition over virtual axes derived from pair invariants.

    Validated pair invariants become *virtual axes*: a translation pair gives
    a linear axis ``w = c . x`` with chain-rule gradient ``(c . grad f)/(c.c)``,
    and a rotation/boost pair gives a quadratic axis ``w = x_i**2 + r x_j**2``
    with gradient ``(v . grad f)/(2 v.v)`` where ``v = s (.) x`` pointwise (the
    complementary directions are certified flat by the pair witnesses).  The
    ordinary pair tests then run over the extended coordinate set:

    * scaling pairs between any disjoint coordinates compose monomial rays —
      difference-products ``(x0-x1)*x2``, quadratic products
      ``(x0**2+x1**2)/x2``, and products of two linear invariants;
    * rotation/boost pairs among linear virtual axes and raw axes compose
      quadratic rays — e.g. ``(x1-x0)**2 + (x3-x2)**2``, the Euclidean
      distance carrier.

    All tests are weighted-gradient identities: sign-agnostic, no positivity
    required.  Symmetries that are non-affine in every chart (e.g.
    ``1/x_i - 1/x_j``) remain out of reach by design.
    """

    col_index = {c: i for i, c in enumerate(cols_t)}
    linear_edges = _collect_edges(specs, family="linear", col_index=col_index, max_denominator=max_denominator)
    quadratic_edges = _collect_edges(specs, family="quadratic", col_index=col_index, max_denominator=max_denominator)
    if not linear_edges and not quadratic_edges:
        return [], []
    monomial_edges = _collect_edges(specs, family="monomial", col_index=col_index, max_denominator=max_denominator)
    residual_tol = float(getattr(cfg, "residual_tol", 0.03) or 0.03)
    max_virtual = max(1, int(getattr(cfg, "pairwise_composition_max_virtual", 4) or 4))
    n = len(cols_t)

    # ---- virtual axis records --------------------------------------------
    # Each record: (node_id, kind, support:set, vals, grads, ast, residual,
    #               descriptor) with descriptor used for reporting/meta.
    virtuals: list[dict[str, Any]] = []

    def _add_linear_virtual(i: int, j: int, ratio: float, residual: float) -> None:
        ints = ray_key_from_covector((1.0, float(ratio)), max_denominator=max_denominator)
        if not ints or len(ints) != 2 or not all(ints):
            return
        coeffs = {i: int(ints[0]), j: int(ints[1])}
        c_full = np.zeros(n, dtype=float)
        for idx, coeff in coeffs.items():
            c_full[idx] = float(coeff)
        vals = _finite_rowwise_linear_combination(x_arr, c_full)
        grad_projection = _finite_rowwise_linear_combination(grad_arr, c_full)
        coeff_norm2 = float(np.dot(c_full, c_full))
        if (
            vals is None
            or grad_projection is None
            or not np.isfinite(coeff_norm2)
            or coeff_norm2 <= 0.0
        ):
            return
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            grads = grad_projection / coeff_norm2
        if not np.all(np.isfinite(grads)):
            return
        ast = build_linear_ast(
            tuple(int(cols_t[k]) for k in sorted(coeffs)),
            tuple(int(coeffs[k]) for k in sorted(coeffs)),
        )
        virtuals.append(
            {
                "kind": "linear",
                "support": set(coeffs),
                "vals": vals,
                "grads": grads,
                "ast": ast,
                "residual": float(residual),
                "coeffs": {int(k): int(v) for k, v in coeffs.items()},
            }
        )

    def _add_quadratic_virtual(i: int, j: int, ratio: float, residual: float) -> None:
        sign_j = 1 if float(ratio) > 0 else -1
        signs_full = np.zeros(n, dtype=float)
        signs_full[i] = 1.0
        signs_full[j] = float(sign_j)
        v = x_arr * signs_full.reshape(1, -1)
        vals = np.sum(v * x_arr, axis=1)  # sum s_m x_m**2
        denom = np.maximum(2.0 * np.sum(v * v, axis=1), 1.0e-300)
        grads = np.sum(v * grad_arr, axis=1) / denom
        ast = _signed_quadratic_ast((int(cols_t[i]), int(cols_t[j])), (1, sign_j))
        virtuals.append(
            {
                "kind": "quadratic",
                "support": {i, j},
                "vals": vals,
                "grads": grads,
                "ast": ast,
                "residual": float(residual),
                "signature": {int(i): 1, int(j): int(sign_j)},
            }
        )

    edge_pool: list[tuple[str, int, int, float, float]] = []
    for (i, j), (ratio, residual) in linear_edges.items():
        edge_pool.append(("linear", i, j, ratio, residual))
    for (i, j), (ratio, residual) in quadratic_edges.items():
        edge_pool.append(("quadratic", i, j, ratio, residual))
    edge_pool.sort(key=lambda item: item[4])
    for kind, i, j, ratio, residual in edge_pool:
        if len(virtuals) >= max_virtual:
            break
        if kind == "linear":
            _add_linear_virtual(i, j, ratio, residual)
        else:
            _add_quadratic_virtual(i, j, ratio, residual)
    if not virtuals:
        return [], []

    # ---- node table over the extended coordinate set ---------------------
    nodes: dict[int, dict[str, Any]] = {}
    for k in range(n):
        nodes[k] = {
            "kind": "raw",
            "support": {k},
            "vals": x_arr[:, k],
            "grads": grad_arr[:, k],
            "ast": Var(int(cols_t[k])),
            "residual": 0.0,
        }
    for offset, record in enumerate(virtuals):
        nodes[n + offset] = record

    def _weighted(node_id: int) -> np.ndarray:
        node = nodes[node_id]
        return node["vals"] * node["grads"]

    def _pair_residual(q: np.ndarray, wa: np.ndarray, wb: np.ndarray) -> float:
        scale = max(
            float(np.sqrt(np.mean(wa**2))), float(np.sqrt(np.mean(wb**2))), 1.0e-300
        )
        return float(np.sqrt(np.mean(q**2))) / scale

    virtual_ids = [n + off for off in range(len(virtuals))]
    lorentz_on = bool(getattr(cfg, "lorentz_boosts", False))

    # ---- level-2 edge discovery ------------------------------------------
    scaling_edges: dict[tuple[int, int], tuple[float, float]] = {}
    for (i, j), (ratio, residual) in monomial_edges.items():
        scaling_edges[(i, j)] = (ratio, residual)
    quad_edges: dict[tuple[int, int], tuple[float, float]] = {}
    for (i, j), (ratio, residual) in quadratic_edges.items():
        quad_edges[(i, j)] = (ratio, residual)

    for a_pos, a_id in enumerate(virtual_ids):
        partner_ids = list(range(n)) + virtual_ids[a_pos + 1 :]
        for b_id in partner_ids:
            if nodes[a_id]["support"] & nodes[b_id]["support"]:
                continue
            lo, hi = (a_id, b_id) if a_id < b_id else (b_id, a_id)
            wa, wb = _weighted(lo), _weighted(hi)
            res_opp = _pair_residual(wa - wb, wa, wb)
            res_com = _pair_residual(wa + wb, wa, wb)
            if min(res_opp, res_com) <= residual_tol:
                ratio = 1.0 if res_opp <= res_com else -1.0
                scaling_edges[(lo, hi)] = (ratio, min(res_opp, res_com))
            # Rotation/boost tests only between linear-like coordinates
            # (quadratic-of-quadratic forms are out of scope).
            if nodes[lo]["kind"] == "quadratic" or nodes[hi]["kind"] == "quadratic":
                continue
            ua, ga = nodes[lo]["vals"], nodes[lo]["grads"]
            ub, gb = nodes[hi]["vals"], nodes[hi]["grads"]
            q_rot = ub * ga - ua * gb
            scale_terms = (ub * ga, ua * gb)
            res_rot = _pair_residual(q_rot, *scale_terms)
            if res_rot <= residual_tol:
                quad_edges[(lo, hi)] = (1.0, res_rot)
            elif lorentz_on:
                res_boost = _pair_residual(ub * ga + ua * gb, *scale_terms)
                if res_boost <= residual_tol:
                    quad_edges[(lo, hi)] = (-1.0, res_boost)

    proposals: list[tuple] = []
    diagnostics: list[dict[str, Any]] = []

    def _component_rows(
        family: str, edges: dict[tuple[int, int], tuple[float, float]]
    ) -> None:
        for support, comp_edges in _connected_components(edges):
            if len(proposals) >= max(0, int(budget)):
                return
            if not any(node_id >= n for node_id in support):
                continue  # raw-only components are level-1 territory
            raw_span: set[int] = set()
            for node_id in support:
                raw_span |= nodes[node_id]["support"]
            witness_residuals = [nodes[node_id]["residual"] for node_id in support]
            max_edge_residual = max(
                [edges[e][1] for e in comp_edges] + [float(v) for v in witness_residuals]
            )
            row: dict[str, Any] = {
                "family": "generalized_symmetry",
                "kind": "pairwise_composition",
                "ray_family": "virtual_" + family,
                "cols": cols_t,
                "raw_span": tuple(sorted(int(cols_t[i]) for i in raw_span)),
                "n_pair_constraints": len(comp_edges),
                "max_pair_residual": float(max_edge_residual),
                "accepted": False,
                "used_for_proposal": False,
            }
            values = _propagate_values(support, [(i, j, edges[(i, j)][0]) for (i, j) in comp_edges])
            if values is None:
                row["status"] = "rejected"
                row["reason"] = "inconsistent_pair_constraints"
                diagnostics.append(row)
                continue
            # Canonical orientation: a virtual axis carries a positive entry.
            first_virtual = min(node_id for node_id in support if node_id >= n)
            if values.get(first_virtual, 1.0) < 0:
                values = {k: -v for k, v in values.items()}
            ray = _snap_values_to_ray(values, support, max_denominator=max_denominator)
            if ray is None or _verify_ray_against_edges(
                ray, [(i, j, edges[(i, j)][0]) for (i, j) in comp_edges]
            ):
                row["status"] = "rejected"
                row["reason"] = (
                    "ray_not_snappable" if ray is None else "snapped_ray_violates_pair_constraint"
                )
                diagnostics.append(row)
                continue

            nodes_sorted = sorted(support)
            ray_local = tuple(int(ray[node_id]) for node_id in nodes_sorted)
            if family == "monomial":
                weighted = np.stack([_weighted(node_id) for node_id in nodes_sorted], axis=1)
                residual, tol, baseline = _joint_ray_residual(
                    weighted, ray_local, cfg=cfg, max_pair_residual=max_edge_residual
                )
            else:
                u_matrix = np.stack([nodes[node_id]["vals"] for node_id in nodes_sorted], axis=1)
                g_matrix = np.stack([nodes[node_id]["grads"] for node_id in nodes_sorted], axis=1)
                signs = np.asarray([float(v) for v in ray_local], dtype=float)
                residual, tol, baseline = _radial_alignment_residual(
                    u_matrix * signs.reshape(1, -1),
                    g_matrix,
                    cfg=cfg,
                    max_pair_residual=max_edge_residual,
                )
            row["joint_residual_rel"] = float(residual)
            row["joint_residual_tol"] = float(tol)
            row["ray"] = list(ray_local)
            if not np.isfinite(residual) or residual > tol:
                row["status"] = "rejected"
                row["reason"] = "joint_ray_residual_exceeds_tol"
                diagnostics.append(row)
                continue

            variable_nodes = [nodes[node_id]["ast"] for node_id in nodes_sorted]
            if family == "monomial":
                z_ast = build_monomial_ast(list(ray_local), variable_nodes=variable_nodes)
            else:
                z_ast = _signed_quadratic_ast(
                    tuple(range(len(nodes_sorted))), ray_local, variable_nodes=variable_nodes
                )
            pattern = tuple(1 if i in raw_span else 0 for i in range(n))
            confidence_scale = max(residual_tol, 1.0e-12)
            confidence = max(0.0, min(1.0, 1.0 - residual / confidence_scale))

            virtual_records = [nodes[node_id] for node_id in nodes_sorted if node_id >= n]
            linear_virtuals = [rec for rec in virtual_records if rec["kind"] == "linear"]
            if family == "monomial":
                if len(virtual_records) == 1 and linear_virtuals:
                    gs_kind = "pairwise_composed_difference_product"
                    coordinate_kind = "monomial_of_linear"
                else:
                    gs_kind = "pairwise_composed_virtual_product"
                    coordinate_kind = "monomial_of_virtual"
            else:
                gs_kind = "pairwise_composed_virtual_quadratic"
                coordinate_kind = "quadratic_form"

            promotion_report = {
                "state": "promoted",
                "accepted": True,
                "reason": "passed_pairwise_composition_gate",
                "confidence": float(confidence),
                "residual_score": float(residual),
                "residual_tol": float(tol),
                "evidence": {
                    "promotion_tier": "pairwise_composition",
                    "ray_family": "virtual_" + family,
                    "n_pair_constraints": len(comp_edges),
                    "max_pair_residual": float(max_edge_residual),
                    "joint_residual_rel": float(residual),
                    "ray": list(ray_local),
                },
            }
            meta = {
                "kind": "gs_promoted_reduction",
                "source": "generalized_symmetry",
                "gs_source_family": "pairwise_composition",
                "gs_family": "pairwise_composition",
                "gs_kind": gs_kind,
                "gs_chart": "identity",
                "gs_promotion_state": "promoted",
                "gs_promotion_reason": "passed_pairwise_composition_gate",
                "gs_promotion": promotion_report,
                "gs_coordinate_kind": coordinate_kind,
                "gs_confidence": float(confidence),
                "gs_virtual_exponents": tuple(int(v) for v in ray_local),
            }
            if gs_kind == "pairwise_composed_difference_product":
                # Structural identity for replace-shadowed matching against
                # legacy difference-product proposals.
                rec = linear_virtuals[0]
                coeffs = rec["coeffs"]
                meta["gs_dp_virtual_support"] = tuple(int(cols_t[i]) for i in sorted(coeffs))
                meta["gs_dp_virtual_coeffs"] = tuple(int(coeffs[i]) for i in sorted(coeffs))
                meta["gs_dp_axis_exponents"] = tuple(
                    (int(cols_t[node_id]), int(ray[node_id]))
                    for node_id in nodes_sorted
                    if node_id < n
                )
            if family == "quadratic" and all(int(v) > 0 for v in ray_local):
                # Definite forms admit the sqrt wrapper (e.g. Euclidean
                # distance carriers such as sqrt((x1-x0)**2 + (x3-x2)**2)).
                meta["form"] = "r2"
                meta["allow_sqrt"] = True
            try:
                meta["z_human"] = ast_to_human_readable(z_ast)
            except Exception:
                pass
            row["status"] = "promoted"
            row["accepted"] = True
            row["used_for_proposal"] = True
            row["z_human"] = meta.get("z_human", "")
            diagnostics.append(row)
            proposals.append((pattern, z_ast, float(confidence), None, meta))

    _component_rows("monomial", scaling_edges)
    if len(proposals) < max(0, int(budget)):
        _component_rows("quadratic", quad_edges)
    return proposals, diagnostics

# Backward-compatible alias (slice 3 shipped the monomial-only entry point).
compose_pairwise_monomial_proposals = compose_pairwise_witness_proposals


def ray_key_from_covector(covector: Sequence[float], *, max_denominator: int = 4) -> tuple[int, ...] | None:
    """Primitive integer ray key for a float covector, or None if not rational."""

    try:
        row = np.asarray([float(v) for v in covector], dtype=float)
    except Exception:
        return None
    max_abs = float(np.max(np.abs(row))) if row.size else 0.0
    if not np.isfinite(max_abs) or max_abs <= 0.0:
        return None
    scaled = row / max_abs
    fractions = [_as_fraction(float(v), max_den=int(max_denominator)) for v in scaled]
    approx = np.asarray([float(f) for f in fractions], dtype=float)
    if float(np.max(np.abs(approx - scaled))) > 5.0e-2:
        return None
    ints = tuple(int(v) for v in projective_exponent_key(fractions))
    if not ints or not any(ints):
        return None
    return ints


def _collect_edges(
    specs: Sequence[Any], *, family: str, col_index: dict[int, int], max_denominator: int = 4
) -> dict[tuple[int, int], tuple[float, float]]:
    """Map (i, j) local index pairs to (ratio of c_j/c_i, best residual).

    Learned pair ratios are noisy float estimates; they are snapped to small
    rationals (tolerance scaled to the pair's own residual) so downstream
    propagation and verification stay integer-exact.  Ratios that do not snap
    (e.g. genuinely oblique directions such as sqrt(2)) are excluded — those
    remain the global determining solve's job.
    """

    edges: dict[tuple[int, int], tuple[float, float]] = {}
    for spec in specs:
        entry = _PAIR_KINDS.get(str(getattr(spec, "kind", "")))
        if entry is None or entry[0] != family:
            continue
        if not bool(getattr(spec, "accepted", False)):
            continue
        if abs(float(getattr(spec, "output_alpha", 0.0))) > _OUTPUT_ACTION_EPS:
            continue
        if abs(float(getattr(spec, "output_beta", 0.0))) > _OUTPUT_ACTION_EPS:
            continue
        axes = tuple(int(a) for a in getattr(spec, "axes", ()) or ())
        if len(axes) != 2 or axes[0] not in col_index or axes[1] not in col_index:
            continue
        i, j = col_index[axes[0]], col_index[axes[1]]
        if i == j:
            continue
        residual = float(getattr(spec, "residual_rel", np.inf))
        ratio = entry[1]
        if ratio is None:
            coeffs = tuple(float(v) for v in (getattr(spec, "xi_coeffs", ()) or ()))
            if len(coeffs) < 2 or not all(np.isfinite(coeffs[:2])) or abs(coeffs[1]) < 1.0e-12:
                continue
            # Generator b_i d_i + b_j d_j annihilates c iff c_i b_i + c_j b_j = 0.
            ratio_raw = -coeffs[0] / coeffs[1]
            if not np.isfinite(ratio_raw) or abs(ratio_raw) < 1.0e-12:
                continue
            snapped = float(_as_fraction(float(ratio_raw), max_den=int(max_denominator)))
            snap_tol = min(max(1.0e-6, 5.0 * (residual if np.isfinite(residual) else 0.0)), 5.0e-2)
            if abs(snapped) < 1.0e-12 or abs(snapped - ratio_raw) > snap_tol * max(1.0, abs(ratio_raw)):
                continue
            ratio = snapped
        if i > j:
            i, j = j, i
            ratio = 1.0 / ratio if abs(ratio) > 1.0e-12 else 0.0
        if not np.isfinite(ratio) or abs(ratio) < 1.0e-12:
            continue
        prev = edges.get((i, j))
        if prev is None or residual < prev[1]:
            merged_ratio = ratio if prev is None else prev[0]
            if prev is not None and not np.isclose(prev[0], ratio, rtol=_RATIO_CONSISTENCY_RTOL, atol=1.0e-9):
                # Conflicting duplicate constraints for the same pair: keep the
                # better-satisfied one.
                merged_ratio = ratio
            edges[(i, j)] = (merged_ratio, residual)
    return edges


def _collect_centered_edges(
    specs: Sequence[Any],
    *,
    family: str,
    col_index: dict[int, int],
    x_arr: np.ndarray,
    cfg: Any,
) -> dict[tuple[int, int], tuple[float, float, dict[int, int]]]:
    """Centered pair edges recovered from accepted *unclassified* affine pairs.

    The pairwise affine classifier only names origin-centered generators
    (rotation/boost/scaling classifications all require ``b ~ 0``); a centered
    generator lands in ``unclassified_pair`` with its normalized coefficient
    6-vector ``[b_i, b_j, a_ii, a_ij, a_ji, a_jj]`` preserved.  The centers
    follow in closed form: ``p = -b_j/a_ji, q = -b_i/a_ij`` for rotation and
    boost pairs, ``p = -b_i/a_ii, q = -b_j/a_jj`` for scaling pairs.  This is
    the geometric replacement for the legacy per-axis preferred-origin (shift)
    detector: centers come from accepted symmetry witnesses rather than from a
    Taylor-structure fit.
    """

    tol = max(0.08, float(getattr(cfg, "general_affine_snap_tol", getattr(cfg, "snap_tol", 0.20)) or 0.20))
    lorentz_on = bool(getattr(cfg, "lorentz_boosts", False))
    out: dict[tuple[int, int], tuple[float, float, dict[int, int]]] = {}
    for spec in specs:
        if str(getattr(spec, "kind", "")) != "unclassified_pair":
            continue
        if not bool(getattr(spec, "accepted", False)):
            continue
        if abs(float(getattr(spec, "output_alpha", 0.0))) > _OUTPUT_ACTION_EPS:
            continue
        if abs(float(getattr(spec, "output_beta", 0.0))) > _OUTPUT_ACTION_EPS:
            continue
        coeffs6 = tuple(float(v) for v in (getattr(spec, "xi_coeffs", ()) or ()))
        if len(coeffs6) != 6 or not all(np.isfinite(coeffs6)):
            continue
        axes = tuple(int(a) for a in getattr(spec, "axes", ()) or ())
        if len(axes) != 2 or axes[0] not in col_index or axes[1] not in col_index:
            continue
        i, j = col_index[axes[0]], col_index[axes[1]]
        if i == j:
            continue
        b_i, b_j, a_ii, a_ij, a_ji, a_jj = coeffs6
        if max(abs(b_i), abs(b_j)) < tol:
            continue  # origin case: handled by the named classifications
        # The 6-vector is normalized by its own max-abs entry, which for a
        # centered generator is usually a b component — so the A-block
        # structure tests must be relative to the A-block scale, not global.
        diag = max(abs(a_ii), abs(a_jj))
        offdiag = max(abs(a_ij), abs(a_ji))
        a_scale = max(diag, offdiag)
        if a_scale < tol:
            continue  # pure translation: no center to recover
        centers: dict[int, float] | None = None
        ratio = 0.0
        if family == "quadratic" and offdiag >= a_scale * 0.999 and diag <= 0.25 * offdiag:
            if abs(a_ij + a_ji) <= 0.25 * offdiag:
                ratio = 1.0  # centered rotation: (x_i-p)**2 + (x_j-q)**2
            elif abs(a_ij - a_ji) <= 0.25 * offdiag and lorentz_on:
                ratio = -1.0  # centered boost: (x_i-p)**2 - (x_j-q)**2
            else:
                continue
            centers = {i: -b_j / a_ji, j: -b_i / a_ij}
        elif family == "monomial" and diag >= a_scale * 0.999 and offdiag <= 0.25 * diag:
            if abs(a_ii - a_jj) <= 0.25 * diag:
                ratio = -1.0  # centered ratio: (x_i-p)/(x_j-q)
            elif abs(a_ii + a_jj) <= 0.25 * diag:
                ratio = 1.0  # centered product: (x_i-p)*(x_j-q)
            else:
                continue
            centers = {i: -b_i / a_ii, j: -b_j / a_jj}
        else:
            continue
        if centers is None or not all(np.isfinite(list(centers.values()))):
            continue
        key = (i, j) if i < j else (j, i)
        residual = float(getattr(spec, "residual_rel", np.inf))
        prev = out.get(key)
        if prev is None or residual < prev[1]:
            out[key] = (float(ratio), residual, {int(a): float(c) for a, c in centers.items()})
    return out


def _compose_family(
    family: str,
    edges: dict[tuple[int, int], tuple[float, float]],
    *,
    x_arr: np.ndarray,
    grad_arr: np.ndarray,
    cols_t: tuple,
    cfg: Any,
    budget: int,
    calibrate_pair_residuals_to_joint_metric: bool = False,
) -> tuple[list[tuple], list[dict[str, Any]]]:
    # The support floor is 3 at first level (2-var products/ratios/sums already
    # have promoted global routes), but recursion over a compound virtual axis
    # has no such route, so it lowers the floor to 2 via this config knob.
    support_floor = int(getattr(cfg, "pairwise_composition_support_floor", 3) or 3)
    min_support = max(support_floor, int(getattr(cfg, "pairwise_composition_min_support", 3) or 3))
    if family == "quadratic":
        # 2-var scaling/translation invariants already have promoted global
        # routes (common scaling, translation plans); 2-var quadratic forms -
        # in particular boosts x_i**2 - x_j**2 - do not, so the quadratic
        # family composes from pair support upward.
        min_support = 2
    max_denominator = max(1, int(getattr(cfg, "general_affine_chart_snap_denominator", 4) or 4))
    proposals: list[tuple] = []
    diagnostics: list[dict[str, Any]] = []
    for support, comp_edges in _connected_components(edges):
        if budget is not None and len(proposals) >= max(0, int(budget)):
            break
        row: dict[str, Any] = {
            "family": "generalized_symmetry",
            "kind": "pairwise_composition",
            "ray_family": family,
            "cols": cols_t,
            "support": tuple(cols_t[i] for i in sorted(support)),
            "n_pair_constraints": len(comp_edges),
            "max_pair_residual": float(max(edges[e][1] for e in comp_edges)),
            "accepted": False,
            "used_for_proposal": False,
        }
        # Per-axis centers from centered pair edges: all edges touching an
        # axis must agree on its center (within a fraction of the axis
        # spread), and centers must lie near the sampled range.
        centers_lists: dict[int, list[float]] = {}
        for e in comp_edges:
            entry = edges[e]
            edge_centers = entry[2] if len(entry) > 2 else {}
            for axis, center in (edge_centers or {}).items():
                centers_lists.setdefault(int(axis), []).append(float(center))
        centers: dict[int, float] = {}
        centers_ok = True
        for axis, center_values in centers_lists.items():
            scale = float(np.std(x_arr[:, axis])) + 1.0e-12
            if max(center_values) - min(center_values) > 0.1 * scale:
                centers_ok = False
                break
            center = float(np.mean(center_values))
            lo, hi = float(np.min(x_arr[:, axis])), float(np.max(x_arr[:, axis]))
            span = max(hi - lo, 1.0e-12)
            if not (lo - span <= center <= hi + span):
                centers_ok = False
                break
            if abs(center) > 1.0e-9 * scale:
                centers[int(axis)] = center
        if not centers_ok:
            row["status"] = "rejected"
            row["reason"] = "inconsistent_pair_centers"
            diagnostics.append(row)
            continue
        effective_min_support = min_support
        if centers:
            # Centered 2-var invariants have no named-proposal or promoted
            # global route: compose from pair support upward.
            effective_min_support = 2
            row["centers"] = {int(cols_t[a]): float(c) for a, c in centers.items()}
        if len(support) < effective_min_support:
            row["status"] = "skipped"
            row["reason"] = f"support_below_min:{len(support)}<{effective_min_support}"
            diagnostics.append(row)
            continue
        values = _propagate_values(support, [(i, j, edges[(i, j)][0]) for (i, j) in comp_edges])
        if values is None:
            row["status"] = "rejected"
            row["reason"] = "inconsistent_pair_constraints"
            diagnostics.append(row)
            continue
        ray = _snap_values_to_ray(values, support, max_denominator=max_denominator)
        if ray is None:
            row["status"] = "rejected"
            row["reason"] = "ray_not_snappable"
            diagnostics.append(row)
            continue
        bad_edge = _verify_ray_against_edges(ray, [(i, j, edges[(i, j)][0]) for (i, j) in comp_edges])
        if bad_edge:
            row["status"] = "rejected"
            row["reason"] = "snapped_ray_violates_pair_constraint"
            diagnostics.append(row)
            continue

        if calibrate_pair_residuals_to_joint_metric:
            calibrated_pair_residual = _component_pair_residual_in_joint_metric(
                family,
                comp_edges,
                edges=edges,
                x_arr=x_arr,
                grad_arr=grad_arr,
                centers=centers,
            )
            if np.isfinite(calibrated_pair_residual):
                row["reported_max_pair_residual"] = float(row["max_pair_residual"])
                row["max_pair_residual"] = float(calibrated_pair_residual)
                row["pair_residual_metric"] = "joint_normalized"

        support_sorted = sorted(support)
        x_support = x_arr[:, support_sorted]
        grad_support = grad_arr[:, support_sorted]
        if centers:
            shift = np.asarray([centers.get(i, 0.0) for i in support_sorted], dtype=float)
            x_support = x_support - shift.reshape(1, -1)
        ray_local = tuple(int(ray[i]) for i in support_sorted)
        if family == "quadratic":
            # Invariance under the composed pseudo-orthogonal group
            # <=> grad_S is aligned with (s ⊙ x_S) pointwise; the joint test
            # is the alignment residual rather than a fixed-ray annihilator
            # (the level sets' tangent distribution varies with x).
            signs = np.asarray([float(v) for v in ray_local], dtype=float)
            residual, tol, baseline = _radial_alignment_residual(
                x_support * signs.reshape(1, -1),
                grad_support,
                cfg=cfg,
                max_pair_residual=row["max_pair_residual"],
            )
        else:
            if family == "monomial":
                # Weighted gradients x .* grad are the log-chart determining
                # rows without ever taking logs, so integer monomial rays are
                # validated sign-agnostically — no positivity requirement.
                if not np.all(np.isfinite(x_support)):
                    row["status"] = "skipped"
                    row["reason"] = "chart_ineligible:non_finite_samples"
                    diagnostics.append(row)
                    continue
                grad_chart = x_support * grad_support
            else:
                grad_chart = grad_support
            residual, tol, baseline = _joint_ray_residual(
                grad_chart, ray_local, cfg=cfg, max_pair_residual=row["max_pair_residual"]
            )
        ray_full = tuple(int(ray.get(i, 0)) for i in range(len(cols_t)))
        row["ray"] = list(ray_full)
        row["joint_residual_rel"] = float(residual)
        row["joint_residual_tol"] = float(tol)
        row["baseline_pair_residual"] = float(baseline)
        if not np.isfinite(residual) or residual > tol:
            row["status"] = "rejected"
            row["reason"] = "joint_ray_residual_exceeds_tol"
            diagnostics.append(row)
            continue

        proposal = _build_proposal(
            family,
            ray_full=ray_full,
            support_sorted=support_sorted,
            cols_t=cols_t,
            residual=residual,
            tol=tol,
            cfg=cfg,
            n_constraints=len(comp_edges),
            max_pair_residual=row["max_pair_residual"],
            centers=centers,
        )
        row["status"] = "promoted"
        row["accepted"] = True
        row["used_for_proposal"] = True
        row["z_human"] = proposal[4].get("z_human", "")
        diagnostics.append(row)
        proposals.append(proposal)
    return proposals, diagnostics


def _build_proposal(
    family: str,
    *,
    ray_full: tuple,
    support_sorted: list[int],
    cols_t: tuple,
    residual: float,
    tol: float,
    cfg: Any,
    n_constraints: int,
    max_pair_residual: float,
    centers: dict[int, float] | None = None,
) -> tuple:
    pattern = tuple(1 if ray_full[i] != 0 else 0 for i in range(len(cols_t)))
    centers = centers or {}

    def _var_node(i: int):
        center = float(centers.get(int(i), 0.0))
        if center != 0.0:
            return AddNode(Var(int(cols_t[i])), ConstNode(-center))
        return Var(int(cols_t[i]))

    if family == "monomial":
        z_ast = build_monomial_ast(
            [ray_full[i] for i in support_sorted],
            variable_nodes=[_var_node(i) for i in support_sorted],
        )
        gs_kind = "pairwise_composed_monomial"
        coordinate_kind = "monomial"
    elif family == "quadratic":
        signature = tuple(int(ray_full[i]) for i in support_sorted)
        if all(s > 0 for s in signature) and not centers:
            z_ast = build_radial_r2_ast(tuple(int(cols_t[i]) for i in support_sorted))
            gs_kind = "pairwise_composed_radial"
            coordinate_kind = "radial"
        else:
            z_ast = _signed_quadratic_ast(
                tuple(int(cols_t[i]) for i in support_sorted),
                signature,
                variable_nodes=[_var_node(i) for i in support_sorted],
            )
            gs_kind = "pairwise_composed_quadratic"
            coordinate_kind = "quadratic_form"
    else:
        z_ast = build_linear_ast(
            tuple(int(cols_t[i]) for i in support_sorted),
            tuple(int(ray_full[i]) for i in support_sorted),
        )
        gs_kind = "pairwise_composed_linear"
        coordinate_kind = "linear_projection"
    # Score on the same scale as the pair witnesses (residual relative to
    # cfg.residual_tol), not the promotion tolerance: proposal-shortlist
    # ranking compares confidences across detector families, and the composed
    # ray's evidence is the conjunction of its pairs'.
    confidence_scale = max(float(getattr(cfg, "residual_tol", 0.03) or 0.03), 1.0e-12)
    confidence = max(0.0, min(1.0, 1.0 - residual / confidence_scale))
    promotion_report = {
        "state": "promoted",
        "accepted": True,
        "reason": "passed_pairwise_composition_gate",
        "confidence": float(confidence),
        "residual_score": float(residual),
        "residual_tol": float(tol),
        "evidence": {
            "promotion_tier": "pairwise_composition",
            "ray_family": family,
            "n_pair_constraints": int(n_constraints),
            "max_pair_residual": float(max_pair_residual),
            "joint_residual_rel": float(residual),
            "ray": list(ray_full),
        },
    }
    meta = {
        "kind": "gs_promoted_reduction",
        "source": "generalized_symmetry",
        "gs_source_family": "pairwise_composition",
        "gs_family": "pairwise_composition",
        "gs_kind": gs_kind,
        "gs_chart": "log" if family == "monomial" else "identity",
        "gs_promotion_state": "promoted",
        "gs_promotion_reason": "passed_pairwise_composition_gate",
        "gs_promotion": promotion_report,
        "gs_coordinate_kind": coordinate_kind,
        "gs_confidence": float(confidence),
    }
    if centers:
        # Centered coordinates are different hypotheses than their
        # origin-centered counterparts: expose the centers, and never emit the
        # suppression-matching keys (legacy detectors are origin-only).
        meta["gs_centers"] = tuple(
            (int(cols_t[i]), float(centers[i])) for i in sorted(centers)
        )
    if family == "monomial":
        meta["gs_monomial_exponents"] = ray_full
        if not centers:
            meta["gs_monomial_exponents_key"] = tuple(int(v) for v in projective_exponent_key(ray_full))
    elif family == "quadratic":
        if coordinate_kind == "radial":
            # Match the legacy radial detector's meta contract so the
            # kind-aware wrapper policy proposes the same variants
            # (sqrt(r^2), rational).
            meta["form"] = "r2"
            meta["allow_sqrt"] = True
            meta["gs_radial_support"] = tuple(int(cols_t[i]) for i in support_sorted)
        else:
            if not centers:
                meta["gs_quadratic_signature"] = ray_full
            if all(int(ray_full[i]) > 0 for i in support_sorted):
                # Definite centered forms still admit the sqrt wrapper.
                meta["form"] = "r2"
                meta["allow_sqrt"] = True
    else:
        meta["gs_linear_covector"] = tuple(float(v) for v in ray_full)
        meta["gs_linear_ray_key"] = tuple(int(v) for v in projective_exponent_key(ray_full))
    try:
        meta["z_human"] = ast_to_human_readable(z_ast)
    except Exception:
        pass
    return (pattern, z_ast, float(confidence), None, meta)


def _connected_components(
    edges: dict[tuple[int, int], Any]
) -> list[tuple[set[int], list[tuple[int, int]]]]:
    adjacency: dict[int, set[int]] = {}
    for (i, j) in edges:
        adjacency.setdefault(i, set()).add(j)
        adjacency.setdefault(j, set()).add(i)
    seen: set[int] = set()
    components: list[tuple[set[int], list[tuple[int, int]]]] = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack, comp = [start], {start}
        while stack:
            node = stack.pop()
            for other in adjacency.get(node, ()):
                if other not in comp:
                    comp.add(other)
                    stack.append(other)
        seen |= comp
        comp_edges = [e for e in edges if e[0] in comp and e[1] in comp]
        components.append((comp, comp_edges))
    return components


def _propagate_values(
    support: set[int], comp_edges: Sequence[tuple[int, int, float]]
) -> dict[int, float] | None:
    """Assign ray values satisfying all pair ratio relations, or None.

    Each edge ``(i, j, r)`` encodes ``value_j = r * value_i``.  Contradictions
    on back-edges (relative mismatch beyond tolerance) reject the component.
    """

    adjacency: dict[int, list[tuple[int, float]]] = {}
    for (i, j, ratio) in comp_edges:
        if not np.isfinite(ratio) or abs(ratio) < 1.0e-12:
            return None
        adjacency.setdefault(i, []).append((j, float(ratio)))
        adjacency.setdefault(j, []).append((i, 1.0 / float(ratio)))
    root = min(support)
    values = {root: 1.0}
    stack = [root]
    while stack:
        node = stack.pop()
        for other, ratio in adjacency.get(node, ()):
            expected = values[node] * ratio
            if other in values:
                if not np.isclose(values[other], expected, rtol=_RATIO_CONSISTENCY_RTOL, atol=1.0e-9):
                    return None
            else:
                values[other] = expected
                stack.append(other)
    if values[root] < 0:
        values = {k: -v for k, v in values.items()}
    return values


def _snap_values_to_ray(
    values: dict[int, float], support: set[int], *, max_denominator: int
) -> dict[int, int] | None:
    row = np.asarray([values[i] for i in sorted(support)], dtype=float)
    max_abs = float(np.max(np.abs(row))) if row.size else 0.0
    if not np.isfinite(max_abs) or max_abs <= 0.0:
        return None
    scaled = row / max_abs
    fractions = [_as_fraction(float(v), max_den=int(max_denominator)) for v in scaled]
    approx = np.asarray([float(f) for f in fractions], dtype=float)
    if float(np.max(np.abs(approx - scaled))) > 1.0e-3:
        return None
    ints = projective_exponent_key(fractions)
    if not ints or not any(int(v) != 0 for v in ints):
        return None
    if sum(1 for v in ints if int(v) != 0) < 2:
        return None
    if max(abs(int(v)) for v in ints) > _MAX_ABS_SNAPPED_EXPONENT:
        return None
    return {i: int(v) for i, v in zip(sorted(support), ints)}


def _verify_ray_against_edges(
    ray: dict[int, int], comp_edges: Sequence[tuple[int, int, float]]
) -> bool:
    """True when some pair constraint is violated by the snapped ray."""

    for (i, j, ratio) in comp_edges:
        lhs = float(ray.get(j, 0))
        rhs = float(ratio) * float(ray.get(i, 0))
        if not np.isclose(lhs, rhs, rtol=_RATIO_CONSISTENCY_RTOL, atol=1.0e-6):
            return True
    return False


def _signed_quadratic_ast(col_ids: Sequence[int], signature: Sequence[int], *, variable_nodes: Sequence[Any] | None = None):
    """AST for ``sum_i s_i * u_i**2`` with signs in {+1, -1}.

    ``variable_nodes`` defaults to raw ``Var(col)`` nodes; virtual-axis
    composition passes arbitrary coordinate ASTs instead.
    """

    nodes = list(variable_nodes) if variable_nodes is not None else [Var(int(c)) for c in col_ids]
    expr = None
    for node, sign in zip(nodes, signature):
        term = PowNode(node, 2.0)
        if int(sign) < 0:
            term = MulNode(ConstNode(-1.0), term)
        expr = term if expr is None else AddNode(expr, term)
    return expr


def _joint_ray_residual_value(
    grad_chart: np.ndarray,
    ray_local: Sequence[float],
) -> float:
    """Return the composed-ray residual without applying a tolerance."""

    grad = np.asarray(grad_chart, dtype=float)
    e = np.asarray([float(v) for v in ray_local], dtype=float)
    if (
        grad.ndim != 2
        or grad.size == 0
        or e.size == 0
        or grad.shape[1] != e.size
        or not np.all(np.isfinite(grad))
        or not np.all(np.isfinite(e))
    ):
        return float("inf")
    try:
        _u, _s, vt = np.linalg.svd(e.reshape(1, -1), full_matrices=True)
    except np.linalg.LinAlgError:
        return float("inf")
    annihilator = vt[1:].T
    if not annihilator.size:
        return 1.0
    scale = float(np.max(np.abs(grad)))
    if not np.isfinite(scale):
        return float("inf")
    if scale <= 0.0:
        return 0.0
    grad_scaled = grad / scale
    denom = max(float(np.linalg.norm(grad_scaled)), 1.0e-300)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        projected = grad_scaled @ annihilator
        residual = float(np.linalg.norm(projected) / denom)
    return residual if np.isfinite(residual) else float("inf")


def _radial_alignment_residual_value(
    x_support: np.ndarray,
    grad_support: np.ndarray,
) -> float:
    """Return the radial-alignment residual without applying a tolerance."""

    x = np.asarray(x_support, dtype=float)
    g = np.asarray(grad_support, dtype=float)
    norms2 = np.sum(x * x, axis=1, keepdims=True)
    norms2 = np.maximum(norms2, 1.0e-300)
    radial = (np.sum(g * x, axis=1, keepdims=True) / norms2) * x
    denom = max(float(np.linalg.norm(g)), 1.0e-300)
    return float(np.linalg.norm(g - radial) / denom)


def _component_pair_residual_in_joint_metric(
    family: str,
    comp_edges: Sequence[tuple[int, int]],
    *,
    edges: dict[tuple[int, int], tuple],
    x_arr: np.ndarray,
    grad_arr: np.ndarray,
    centers: dict[int, float],
) -> float:
    """Re-score component edges in the metric used by the joint gate.

    Named-generator residuals are target-variation normalized.  For recursive
    virtual coordinates that scale can be unrelated to their gradient scale,
    so it cannot calibrate a gradient-normalized joint residual.  Each edge is
    therefore evaluated as its own two-axis joint claim, using the same chart
    and normalization as the eventual component.
    """

    residuals: list[float] = []
    for edge in comp_edges:
        i, j = (int(edge[0]), int(edge[1]))
        ratio = float(edges[edge][0])
        x_pair = np.asarray(x_arr[:, [i, j]], dtype=float)
        grad_pair = np.asarray(grad_arr[:, [i, j]], dtype=float)
        if centers:
            shift = np.asarray(
                [float(centers.get(i, 0.0)), float(centers.get(j, 0.0))],
                dtype=float,
            )
            x_pair = x_pair - shift.reshape(1, -1)
        pair_ray = (1.0, ratio)
        if family == "quadratic":
            residual = _radial_alignment_residual_value(
                x_pair * np.asarray(pair_ray, dtype=float).reshape(1, -1),
                grad_pair,
            )
        else:
            grad_chart = x_pair * grad_pair if family == "monomial" else grad_pair
            residual = _joint_ray_residual_value(grad_chart, pair_ray)
        if np.isfinite(residual):
            residuals.append(float(residual))
    return max(residuals) if residuals else float("nan")


def _radial_alignment_residual(
    x_support: np.ndarray,
    grad_support: np.ndarray,
    *,
    cfg: Any,
    max_pair_residual: float,
) -> tuple[float, float, float]:
    """Fraction of gradient energy orthogonal to the radial direction.

    ``f = g(sum x_i**2)`` on the support makes ``grad_S`` proportional to
    ``x_S`` pointwise; the residual is the relative Frobenius norm of the
    tangential (non-radial) gradient component.  Tolerance calibration matches
    :func:`_joint_ray_residual`.
    """

    residual = _radial_alignment_residual_value(x_support, grad_support)
    abs_tol = float(getattr(cfg, "general_affine_promotion_residual_tol", 1.0e-8) or 1.0e-8)
    factor = float(getattr(cfg, "noise_calibrated_snap_factor", 3.0) or 3.0)
    baseline = float(max_pair_residual) if np.isfinite(max_pair_residual) else 1.0
    tol = min(max(abs_tol, factor * baseline), _SNAP_SANITY_CEILING)
    return residual, tol, baseline


def _joint_ray_residual(
    grad_chart: np.ndarray,
    ray_local: Sequence[int],
    *,
    cfg: Any,
    max_pair_residual: float,
) -> tuple[float, float, float]:
    """Joint determining residual of the composed ray, with calibrated tol.

    The claimed symmetries are translations (in the ray's chart) along
    ``null(ray)``; their determining rows are ``grad_chart . v``.  The
    tolerance is calibrated to the pair witnesses' own residuals — the
    composed claim may not be worse than a factor over the individual pair
    claims — and capped by the absolute sanity ceiling shared with exponent
    snapping.
    """

    residual = _joint_ray_residual_value(grad_chart, ray_local)
    abs_tol = float(getattr(cfg, "general_affine_promotion_residual_tol", 1.0e-8) or 1.0e-8)
    factor = float(getattr(cfg, "noise_calibrated_snap_factor", 3.0) or 3.0)
    baseline = float(max_pair_residual) if np.isfinite(max_pair_residual) else 1.0
    tol = min(max(abs_tol, factor * baseline), _SNAP_SANITY_CEILING)
    return residual, tol, baseline
