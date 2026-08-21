# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Stage-B prep helpers for the NestyNet factorized symbolic search adapter."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any

import torch

from nestynet_sr.sr_core.bridges import AtomNode, FixedConst, FreeConst, Node, get_input_exprs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageBExplorerPrep:
    """Prepared Stage-B adapter inputs for a factorized symbolic search explorer call."""

    declared_consts: list[dict[str, Any]]
    var_dims: Any
    y_dims: Any
    input_exprs: list[Node]


def _declared_constant_specs_for_explorer(units_spec) -> list[dict[str, Any]]:
    """Collect declared free/fixed constants for factorized symbolic search explorer injection."""
    if units_spec is None:
        return []

    from nestynet_sr.sr_core.units import normalize_free_const_scope

    specs: list[dict[str, Any]] = []
    seen: set[str] = set()

    free_dims = dict(getattr(units_spec, "free_const_dims", {}) or {})
    free_scope = dict(getattr(units_spec, "free_const_scope", {}) or {})
    for name, dim in free_dims.items():
        nm = str(name)
        if nm in seen:
            continue
        scope = normalize_free_const_scope(free_scope.get(nm, "experiment"), default="experiment")
        specs.append(
            {
                "name": nm,
                "kind": "free",
                "dim": dim,
                "value": 1.0,
                "scope": scope,
            }
        )
        seen.add(nm)

    fixed_dims = dict(getattr(units_spec, "fixed_const_dims", {}) or {})
    fixed_vals = dict(getattr(units_spec, "fixed_const_values", {}) or {})
    fixed_mode = str(getattr(units_spec, "fixed_const_mode", "strict")).strip().lower()
    if fixed_mode == "off":
        return specs
    for name, dim in fixed_dims.items():
        nm = str(name)
        if nm in seen:
            logger.warning(
                "factorized_search constant injection: name '%s' declared as both free and fixed; using free declaration.",
                nm,
            )
            continue
        if nm not in fixed_vals:
            logger.warning(
                "factorized_search constant injection: fixed_const '%s' has units but no value; skipping.",
                nm,
            )
            continue
        try:
            val = float(fixed_vals[nm])
        except Exception:
            logger.warning(
                "factorized_search constant injection: fixed_const '%s' value is non-numeric; skipping.",
                nm,
            )
            continue
        if not math.isfinite(val):
            logger.warning(
                "factorized_search constant injection: fixed_const '%s' value is non-finite; skipping.",
                nm,
            )
            continue
        specs.append(
            {
                "name": nm,
                "kind": "fixed",
                "dim": dim,
                "value": val,
                "scope": "fixed",
            }
        )
        seen.add(nm)

    return specs


def _append_declared_constant_dims(var_dims, declared_consts: list[dict[str, Any]]):
    """Append declared constant dimensions to explorer variable dims."""
    if var_dims is None or not declared_consts:
        return var_dims

    from nestynet_sr.sr_search.factorized_search.adapters.nestynet.api import fraction_to_dims

    out = list(var_dims)
    for spec in declared_consts:
        out.append(fraction_to_dims(spec["dim"]))
    return out


def _append_declared_constant_columns(
    x_data: torch.Tensor,
    declared_consts: list[dict[str, Any]],
) -> torch.Tensor:
    """Append declared constants as extra virtual variables in data tensors."""
    if not declared_consts:
        return x_data
    cols = []
    for spec in declared_consts:
        value = 1.0 if spec["kind"] == "free" else float(spec["value"])
        cols.append(torch.full((x_data.shape[0], 1), value, dtype=x_data.dtype, device=x_data.device))
    return torch.cat([x_data, *cols], dim=1)


def _build_input_exprs_with_declared_constants(
    target: AtomNode,
    declared_consts: list[dict[str, Any]],
) -> list[Node]:
    """Build atom input expressions extended with declared constant leaves."""
    input_exprs: list[Node] = list(get_input_exprs(target))
    for spec in declared_consts:
        name = str(spec["name"])
        if spec["kind"] == "free":
            input_exprs.append(
                FreeConst(name, tag=name, init=1.0, scope=str(spec.get("scope", "experiment")))
            )
        else:
            input_exprs.append(FixedConst(name, value=float(spec["value"]), tag=name))
    return input_exprs


def _atom_dims_for_explorer(atom, root, units_spec):
    """Return `(var_dims, y_dims)` float-tuple lists for the explorer, or `(None, None)`."""
    if units_spec is None:
        return None, None
    try:
        from dataclasses import replace as _dc_replace

        from nestynet_sr.sr_core.bridges import (
            compound_input_expr,
            extra_input_var_idxs,
            has_nontrivial_input,
        )
        from nestynet_sr.sr_core.units import compute_node_domains, eval_analytic_expr_dim, infer_atom_output_dim
        from nestynet_sr.sr_search.factorized_search.adapters.nestynet.api import fraction_to_dims

        if has_nontrivial_input(atom):
            input_expr = compound_input_expr(atom)
            z_dim = eval_analytic_expr_dim(input_expr, units_spec.x_dims)
            if z_dim is None:
                return None, None
            var_dims = [fraction_to_dims(z_dim)]
            for idx in extra_input_var_idxs(atom):
                if idx < len(units_spec.x_dims):
                    var_dims.append(fraction_to_dims(units_spec.x_dims[idx]))
                else:
                    return None, None
        else:
            var_dims = []
            for idx in atom.var_idxs:
                if idx < len(units_spec.x_dims):
                    var_dims.append(fraction_to_dims(units_spec.x_dims[idx]))
                else:
                    return None, None

        y_dim = infer_atom_output_dim(root, atom, units_spec)
        y_dim_source = "infer_atom_output_dim"
        if y_dim is None:
            y_dim_source = "fallback(compute_node_domains)"
            span_spec = _dc_replace(units_spec, nn_semantics="span")
            domains = compute_node_domains(root, span_spec)
            if domains is not None:
                dom = domains.get(id(atom))
                if dom is not None and dom.is_pinned():
                    y_dim = dom.offset
                else:
                    y_dim_source += f"(dom={'None' if dom is None else f'rank={dom.rank()}'})"
            else:
                y_dim_source += "(domains=None)"
        y_dims = fraction_to_dims(y_dim) if y_dim is not None else None

        print(
            f"[_atom_dims_for_explorer] atom tag={getattr(atom, 'tag', None)}, "
            f"var_idxs={getattr(atom, 'var_idxs', None)}, "
            f"var_dims={var_dims}, y_dim={y_dim}, y_dims={y_dims}, "
            f"source={y_dim_source}"
        )

        return var_dims, y_dims
    except Exception as exc:
        print(
            f"[_atom_dims_for_explorer] EXCEPTION for atom tag={getattr(atom, 'tag', None)}, "
            f"var_idxs={getattr(atom, 'var_idxs', None)}: {exc}"
        )
        return None, None


def prepare_stageb_explorer_inputs(*, root, target: AtomNode, units_spec) -> StageBExplorerPrep:
    """Build declared constants, dimensional metadata, and input exprs for Stage B factorized symbolic search."""
    declared_consts = _declared_constant_specs_for_explorer(units_spec)
    var_dims, y_dims = _atom_dims_for_explorer(target, root, units_spec)
    var_dims = _append_declared_constant_dims(var_dims, declared_consts)
    input_exprs = _build_input_exprs_with_declared_constants(target, declared_consts)
    return StageBExplorerPrep(
        declared_consts=declared_consts,
        var_dims=var_dims,
        y_dims=y_dims,
        input_exprs=input_exprs,
    )


__all__ = [
    "StageBExplorerPrep",
    "_append_declared_constant_columns",
    "_append_declared_constant_dims",
    "_atom_dims_for_explorer",
    "_build_input_exprs_with_declared_constants",
    "_declared_constant_specs_for_explorer",
    "prepare_stageb_explorer_inputs",
]
