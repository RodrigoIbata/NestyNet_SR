# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Recursive coordinate composition for the GS layer.

Once first-level GS has promoted a coordinate ``z1 = g(x)`` (a monomial,
linear form, radius, ...), that coordinate can itself act as an axis in a
further symmetry search: this module treats each promoted coordinate as a
*virtual raw axis* and re-runs the ordinary pairwise-witness composition on
the augmented set ``{x_0, ..., x_{n-1}, z1, z2, ...}``.  A second-level
symmetry — e.g. a translation between ``z1`` and ``x3`` — then composes a
nested coordinate such as ``x0*x1/x2 + x3`` or ``(x0*x1/x2) * x4``.

Each virtual axis carries a value ``z(x)`` and the chain-rule directional
gradient of the target along it,

    df/dz = (grad z . grad f) / (grad z . grad z),

so the augmented ``(value, gradient)`` table feeds the existing composition
families unchanged (they are already coordinate-agnostic). Mutually disjoint
virtual carriers can also be searched jointly, which lets three pair products
compose directly into a dot product. Depth and beam width are bounded; the
outer function of the final coordinate remains the consumer's job.

This is opt-in at the config layer (``cfg.recursive_composition_active()``);
with it off nothing here runs.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any, Sequence

import numpy as np

from nestynet_sr.sr_core.bridges import (
    Var,
    _collect_var_idxs_from_node,
    ast_to_human_readable,
    eval_input_expr,
)

from .generators import discover_generator_specs
from .pairwise_composition import compose_pairwise_witness_proposals
from .quotient import substitute_local_coordinate_ast

_MIN_GRAD_NORM = 1.0e-300


def compose_recursive_coordinate_proposals(
    first_level_proposals: Sequence[tuple],
    *,
    joint_proposals: Sequence[tuple] | None = None,
    x_vals: Any,
    y_vals: Any,
    dydx_vals: Any,
    cols: Sequence[int],
    cfg: Any,
    max_virtual_coords: int = 4,
    output_depth: int = 2,
    beam_width: int = 2,
) -> tuple[list[tuple], list[dict[str, Any]]]:
    """Compose one bounded GS level over certified virtual coordinates.

    Returns ``(proposals, diagnostics)`` in the same 5-tuple shape as the rest
    of the GS bridge.  ``first_level_proposals`` is the frontier for the usual
    one-coordinate-plus-free-axes route.  ``joint_proposals`` supplies the
    bounded bank from which mutually disjoint virtual-coordinate groups are
    built (for example three pair products composing into a dot product).
    """

    cols_t = tuple(int(c) for c in cols)
    n_raw = len(cols_t)
    x_arr = np.asarray(x_vals, dtype=float)
    grad_arr = np.asarray(dydx_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float).reshape(-1)
    if x_arr.ndim != 2 or x_arr.shape[1] != n_raw:
        return [], []

    def _records(rows: Sequence[tuple]) -> list[dict[str, Any]]:
        coord_records: list[dict[str, Any]] = []
        seen_asts: set[str] = set()
        for prop in rows:
            meta = prop[4] if len(prop) >= 5 and isinstance(prop[4], dict) else {}
            if str(meta.get("source", "")) != "generalized_symmetry":
                continue
            if not bool(meta.get("carrier_certified", False)):
                continue
            z_ast = prop[1]
            if z_ast is None:
                continue
            try:
                support = tuple(sorted(int(v) for v in _collect_var_idxs_from_node(z_ast)))
            except Exception:
                continue
            if len(support) < 2:
                continue
            try:
                key = ast_to_human_readable(z_ast)
            except Exception:
                key = repr(z_ast)
            if key in seen_asts:
                continue
            rec = _coordinate_record(z_ast, x_arr, grad_arr, support, cols=cols_t)
            if rec is None:
                continue
            seen_asts.add(key)
            rec.update(
                {
                    "human": key,
                    "fingerprint": str(meta.get("gs_carrier_fingerprint", key)),
                    "depth": int(meta.get("gs_carrier_depth", meta.get("gs_recursive_depth", 1)) or 1),
                    "confidence": float(prop[2]) if len(prop) >= 3 else 0.0,
                }
            )
            coord_records.append(rec)
        coord_records.sort(
            key=lambda rec: (
                len(rec["support"]),
                float(rec["confidence"]),
                -int(rec["depth"]),
                str(rec["fingerprint"]),
            ),
            reverse=True,
        )
        return coord_records[: max(1, int(max_virtual_coords))]

    coord_records = _records(first_level_proposals)
    joint_records = _records(joint_proposals or first_level_proposals)
    if not coord_records and len(joint_records) < 2:
        return [], []
    frontier_fingerprints = {
        str(record["fingerprint"]) for record in coord_records
    }

    # A local min-support of two lets one virtual coordinate compose with a
    # single free axis, and lets two or more disjoint carriers compose jointly.
    rec_cfg = dataclasses.replace(
        cfg, pairwise_composition_min_support=2, pairwise_composition_support_floor=2
    )
    proposals: list[tuple] = []
    diagnostics: list[dict[str, Any]] = []
    ast_seen: set[str] = set()

    def _compose_augmented(
        virtual_records: Sequence[dict[str, Any]],
        *,
        min_virtual_touched: int,
        route: str,
    ) -> None:
        virtual_records = list(virtual_records)
        covered = set()
        for rec in virtual_records:
            covered.update(int(v) for v in rec["support"])
        free_positions = [i for i in range(n_raw) if int(cols_t[i]) not in covered]
        if len(virtual_records) == 1 and not free_positions:
            return

        aug_asts: list[Any] = [rec["ast"] for rec in virtual_records]
        aug_asts.extend(Var(int(cols_t[i])) for i in free_positions)
        aug_values = [rec["value"] for rec in virtual_records]
        aug_values.extend(x_arr[:, i] for i in free_positions)
        aug_grads = [rec["directional_grad"] for rec in virtual_records]
        aug_grads.extend(grad_arr[:, i] for i in free_positions)
        V = np.stack(aug_values, axis=1)
        G = np.stack(aug_grads, axis=1)
        aug_cols = tuple(range(len(aug_asts)))

        try:
            specs = discover_generator_specs(
                V, y_arr, G, cols=aug_cols, cfg=rec_cfg, include_rejected=False
            )
            composed, pairwise_diag = compose_pairwise_witness_proposals(
                specs,
                x_vals=V,
                dydx_vals=G,
                cols=aug_cols,
                cfg=rec_cfg,
                calibrate_pair_residuals_to_joint_metric=True,
            )
        except Exception as exc:
            diagnostics.append(
                {
                    "family": "generalized_symmetry",
                    "kind": "recursive_composition",
                    "accepted": False,
                    "status": "rejected",
                    "depth": int(output_depth),
                    "route": route,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            return

        n_virtual = len(virtual_records)
        parent_human_all = [str(rec["human"]) for rec in virtual_records]
        parent_fingerprints_all = [
            str(rec["fingerprint"]) for rec in virtual_records
        ]
        for pairwise_row in pairwise_diag or ():
            if bool(pairwise_row.get("accepted", False)):
                continue
            row = dict(pairwise_row)
            row["recursive_inner_kind"] = str(
                pairwise_row.get("kind", "pairwise_composition")
            )
            row["kind"] = "recursive_composition"
            row["depth"] = int(output_depth)
            row["route"] = str(route)
            row["inner"] = list(parent_human_all)
            row["parent_fingerprints"] = list(parent_fingerprints_all)
            diagnostics.append(row)

        for prop in composed:
            touched = tuple(k for k, value in enumerate(prop[0]) if float(value) != 0.0)
            virtual_touched = sum(1 for k in touched if k < n_virtual)
            if virtual_touched < int(min_virtual_touched):
                continue
            if len(touched) < 2:
                continue
            used_virtual_records = [
                virtual_records[k] for k in touched if k < n_virtual
            ]
            parent_human = [str(rec["human"]) for rec in used_virtual_records]
            parent_fingerprints = [
                str(rec["fingerprint"]) for rec in used_virtual_records
            ]
            parent_support_max = max(
                len(rec["support"]) for rec in used_virtual_records
            )
            parent_confidence = min(
                float(rec["confidence"]) for rec in used_virtual_records
            )
            composed_depth = 1 + max(
                int(rec["depth"]) for rec in used_virtual_records
            )
            try:
                raw_ast = substitute_local_coordinate_ast(prop[1], aug_asts)
                raw_support = tuple(
                    sorted(int(v) for v in _collect_var_idxs_from_node(raw_ast))
                )
            except Exception:
                continue
            if len(raw_support) <= parent_support_max:
                continue
            pattern = tuple(1 if int(c) in set(raw_support) else 0 for c in cols_t)
            if not any(pattern):
                continue
            try:
                z_human = ast_to_human_readable(raw_ast)
            except Exception:
                z_human = repr(raw_ast)
            if z_human in ast_seen:
                continue
            ast_seen.add(z_human)
            meta = dict(prop[4]) if len(prop) >= 5 and isinstance(prop[4], dict) else {}
            meta.update(
                {
                    "source": "generalized_symmetry",
                    "gs_recursive_depth": int(composed_depth),
                    "gs_carrier_depth": int(composed_depth),
                    "gs_source_family": "recursive_composition",
                    "gs_recursive_route": str(route),
                    "gs_recursive_inner": parent_human,
                    "gs_recursive_parent_fingerprints": tuple(parent_fingerprints),
                    "z_human": z_human,
                }
            )
            conf = min(
                parent_confidence,
                float(prop[2]) if len(prop) >= 3 else 0.0,
            )
            proposals.append((pattern, raw_ast, conf, None, meta))
            diagnostics.append(
                {
                    "family": "generalized_symmetry",
                    "kind": "recursive_composition",
                    "accepted": True,
                    "status": "promoted",
                    "used_for_proposal": True,
                    "depth": int(composed_depth),
                    "route": route,
                    "z_human": z_human,
                    "inner": parent_human,
                    "parent_fingerprints": parent_fingerprints,
                }
            )

    # Existing route: one virtual carrier plus its remaining raw axes.
    for rec in coord_records:
        _compose_augmented((rec,), min_virtual_touched=1, route="single_virtual")

    # New route: jointly reduce mutually disjoint virtual carriers. Enumerate a
    # tiny, deterministic beam, preferring groups with the greatest raw support.
    groups: list[tuple[dict[str, Any], ...]] = []
    max_group_size = min(max(2, int(max_virtual_coords)), len(joint_records))
    for size in range(max_group_size, 1, -1):
        for group in itertools.combinations(joint_records, size):
            if not any(
                str(record["fingerprint"]) in frontier_fingerprints
                for record in group
            ):
                continue
            supports = [set(int(v) for v in rec["support"]) for rec in group]
            if any(supports[i] & supports[j] for i in range(size) for j in range(i + 1, size)):
                continue
            groups.append(group)
    groups.sort(
        key=lambda group: (
            sum(len(rec["support"]) for rec in group),
            min(float(rec["confidence"]) for rec in group),
            -max(int(rec["depth"]) for rec in group),
            tuple(str(rec["fingerprint"]) for rec in group),
        ),
        reverse=True,
    )
    for group in groups[: max(1, int(beam_width))]:
        _compose_augmented(group, min_virtual_touched=2, route="joint_virtual")

    proposals.sort(
        key=lambda prop: (
            sum(int(v) != 0 for v in prop[0]),
            float(prop[2]),
            -len(str(prop[4].get("z_human", ""))),
        ),
        reverse=True,
    )
    retained = proposals[: max(1, int(beam_width))]
    retained_human = {
        str(proposal[4].get("z_human", "")) for proposal in retained
    }
    for row in diagnostics:
        if not bool(row.get("accepted", False)):
            continue
        if str(row.get("z_human", "")) in retained_human:
            continue
        row["used_for_proposal"] = False
        row["status"] = "accepted_not_retained"
        row["beam_pruned"] = True
    return retained, diagnostics


def _coordinate_record(
    z_ast: Any,
    x_arr: np.ndarray,
    grad_arr: np.ndarray,
    support: tuple[int, ...],
    *,
    cols: Sequence[int],
) -> dict[str, Any] | None:
    """Value, gradient, and directional target-gradient for a virtual axis."""

    value = _eval_ast(z_ast, x_arr, cols=cols)
    if value is None or not np.all(np.isfinite(value)):
        return None
    grad_z = _numerical_gradient(z_ast, x_arr, cols=cols)
    if grad_z is None or not np.all(np.isfinite(grad_z)):
        return None
    denom = np.sum(grad_z * grad_z, axis=1)
    if float(np.max(np.abs(denom))) <= _MIN_GRAD_NORM:
        return None  # constant coordinate: nothing to compose along
    denom = np.maximum(denom, _MIN_GRAD_NORM)
    directional = np.sum(grad_z * grad_arr, axis=1) / denom
    if not np.all(np.isfinite(directional)):
        return None
    return {
        "ast": z_ast,
        "value": value,
        "directional_grad": directional,
        "support": support,
    }


def _eval_ast(
    z_ast: Any,
    x_arr: np.ndarray,
    *,
    cols: Sequence[int],
) -> np.ndarray | None:
    try:
        import torch

        cols_t = tuple(int(value) for value in cols)
        if cols_t == tuple(range(len(cols_t))):
            eval_array = x_arr
        else:
            width = max(cols_t, default=-1) + 1
            eval_array = np.zeros((x_arr.shape[0], width), dtype=x_arr.dtype)
            for local_idx, raw_idx in enumerate(cols_t):
                eval_array[:, raw_idx] = x_arr[:, local_idx]
        xt = torch.as_tensor(eval_array, dtype=torch.float64)
        return eval_input_expr(z_ast, xt).detach().cpu().numpy().reshape(-1)
    except Exception:
        return None


def _numerical_gradient(
    z_ast: Any,
    x_arr: np.ndarray,
    *,
    cols: Sequence[int],
) -> np.ndarray | None:
    """Central-difference gradient of a coordinate AST w.r.t. raw inputs."""

    n = x_arr.shape[1]
    grad = np.zeros_like(x_arr)
    h = 1.0e-6 * (np.abs(x_arr) + 1.0)
    for i in range(n):
        xp = x_arr.copy()
        xm = x_arr.copy()
        xp[:, i] += h[:, i]
        xm[:, i] -= h[:, i]
        vp = _eval_ast(z_ast, xp, cols=cols)
        vm = _eval_ast(z_ast, xm, cols=cols)
        if vp is None or vm is None:
            return None
        grad[:, i] = (vp - vm) / (2.0 * h[:, i])
    return grad
