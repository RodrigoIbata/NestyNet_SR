# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Input literal, units, and class-metadata parsing helpers for ``run_SR``."""

from __future__ import annotations

import ast
import json
import pathlib
import re

import numpy as np


def _parse_py_or_json_literal(s):
    """Parse a Python literal (via ast.literal_eval) or JSON string."""
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception as e:
        raise ValueError(f"Failed to parse literal: {s!r} ({e})")


def _normalize_class_param_sr_metadata(meta_spec, dataset_paths):
    """Normalize class Param-SR metadata to per-dataset row dicts.

    Parameters
    ----------
    meta_spec : str | object
        Parsed from CLI (JSON/Python literal accepted), with supported formats:
          1) list[dict] of length D
          2) dict[str, list|tuple|scalar] column-wise
          3) dict[dataset_alias, dict] row-wise keyed by dataset name/stem/path
    dataset_paths : list[str]
        Dataset paths in the same order as class-SR loaders/models.
    """
    if meta_spec is None:
        return None
    raw = _parse_py_or_json_literal(meta_spec) if isinstance(meta_spec, str) else meta_spec
    if raw is None:
        return None

    dcount = int(len(dataset_paths))
    rows = [dict() for _ in range(dcount)]

    def _finite_float(v, *, where: str):
        try:
            f = float(v)
        except Exception as e:
            raise ValueError(f"class_param_sr_metadata: non-numeric value at {where}: {v!r} ({e})")
        if not np.isfinite(f):
            raise ValueError(f"class_param_sr_metadata: non-finite value at {where}: {v!r}")
        return float(f)

    if isinstance(raw, (list, tuple)):
        if len(raw) != dcount:
            raise ValueError(
                "class_param_sr_metadata row-wise format must have one dict per dataset "
                f"(got {len(raw)} rows, expected {dcount})."
            )
        for i, row in enumerate(raw):
            if not isinstance(row, dict):
                raise ValueError(
                    "class_param_sr_metadata row-wise format expects list[dict]; "
                    f"row {i} is {type(row).__name__}."
                )
            for k, v in row.items():
                rows[i][str(k)] = _finite_float(v, where=f"row[{i}].{k}")
        return rows

    if isinstance(raw, dict):
        # Dataset-keyed row-wise dict: {dataset_alias: {meta_name: value}}
        if raw and all(isinstance(v, dict) for v in raw.values()):
            alias_to_idx = {}
            for i, fp in enumerate(dataset_paths):
                p = pathlib.Path(str(fp))
                for alias in (str(fp), p.name, p.stem):
                    alias_to_idx.setdefault(alias, i)
            matched = 0
            for ds_key, ds_row in raw.items():
                idx = alias_to_idx.get(str(ds_key), None)
                if idx is None:
                    continue
                matched += 1
                for k, v in ds_row.items():
                    rows[idx][str(k)] = _finite_float(v, where=f"{ds_key}.{k}")
            if matched > 0:
                return rows

        # Column-wise dict: {meta_name: [v1,...,vD]} or scalar broadcast
        for meta_name, col in raw.items():
            mk = str(meta_name)
            if isinstance(col, (list, tuple)):
                if len(col) != dcount:
                    raise ValueError(
                        f"class_param_sr_metadata column '{mk}' must have length {dcount} "
                        f"(got {len(col)})."
                    )
                for i, v in enumerate(col):
                    rows[i][mk] = _finite_float(v, where=f"{mk}[{i}]")
            else:
                fv = _finite_float(col, where=mk)
                for i in range(dcount):
                    rows[i][mk] = fv
        return rows

    raise ValueError(
        "class_param_sr_metadata must be a list/tuple of dict rows or a dict "
        "(column-wise or dataset-keyed row-wise)."
    )


def _metadata_linked_invariants(derived_invariants):
    """Return Parameter-SR invariants that explicitly reference metadata terms."""
    out = []
    for di in list(derived_invariants or []):
        if not isinstance(di, dict):
            continue
        expr = str(di.get("expr", ""))
        if "meta:" in expr:
            out.append(di)
    return out


def _extract_bracketed_from_start(s: str):
    """Extract the first balanced [...] substring from the start of s."""
    s = str(s)
    s = s.lstrip()
    if not s.startswith("["):
        raise ValueError("Expected '[' at start of bracketed literal")
    level = 0
    for i, ch in enumerate(s):
        if ch == "[":
            level += 1
        elif ch == "]":
            level -= 1
            if level == 0:
                return s[: i + 1], s[i + 1 :]
    raise ValueError("Unbalanced brackets in literal")


def _extract_last_bracketed(s: str):
    """Extract the last balanced [...] substring from s (supports nested lists)."""
    s = str(s)
    i = s.rfind("]")
    if i < 0:
        raise ValueError("No closing ']' found")
    level = 0
    for j in range(i, -1, -1):
        ch = s[j]
        if ch == "]":
            level += 1
        elif ch == "[":
            level -= 1
            if level == 0:
                return s[j : i + 1], s[:j].rstrip()
    raise ValueError("Unbalanced brackets when scanning from end")


def _infer_units_basis(n: int, basis_arg=None):
    """Infer basis labels for unit vectors of length n."""
    if basis_arg is not None:
        if isinstance(basis_arg, (list, tuple)):
            parts = [str(p) for p in basis_arg]
        else:
            parts = [p.strip() for p in str(basis_arg).split(",") if p.strip()]
        if len(parts) != int(n):
            raise ValueError(f"units_basis has length {len(parts)} but expected {n}")
        return tuple(parts)

    # Heuristic defaults
    if int(n) == 5:
        return ("L", "T", "M", "I", "Θ")
    if int(n) == 7:
        return ("L", "M", "T", "I", "Θ", "N", "J")
    return tuple(f"d{i}" for i in range(int(n)))


def _parse_units_arg(units_str: str):
    """Parse --units into (y_units_vec, x_units_mat, basis or None)."""
    if units_str is None:
        return None, None, None

    parsed = _parse_py_or_json_literal(units_str)
    if isinstance(parsed, dict):
        y = parsed.get("y", parsed.get("y_units"))
        x = parsed.get("x", parsed.get("x_units"))
        basis = parsed.get("basis", parsed.get("units_basis"))
        return y, x, basis

    # Allow tuple/list form: ([...], [[...],...])
    if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
        return parsed[0], parsed[1], None

    # Fallback: two bracket-lists in one string: "[y] [[x0],[x1],...]".
    y_s, rest = _extract_bracketed_from_start(units_str)
    x_s, _ = _extract_bracketed_from_start(rest)
    y = _parse_py_or_json_literal(y_s)
    x = _parse_py_or_json_literal(x_s)
    return y, x, None


def _load_units_from_equations(path: str, eq_id: str):
    """
    Load (y_units, x_units) for a given equation id from equations.txt.

    Tries to match eq_id directly, or extracts equation number from patterns
    like 'pb003_I_8_14_data' -> '003'.
    """
    eq_id = str(eq_id).strip()

    # Try to extract equation ID from filename patterns like 'pb003...'
    match = re.match(r"pb(\d+)", eq_id)
    if match:
        numeric_id = match.group(1)
    else:
        numeric_id = eq_id

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            tok = line.split(None, 1)[0]
            # Try matching both the raw eq_id and the extracted numeric_id
            if tok == eq_id or tok == numeric_id:
                # Last two columns are y_units and x_units.
                x_s, rem = _extract_last_bracketed(line)
                y_s, _ = _extract_last_bracketed(rem)

                y = _parse_py_or_json_literal(y_s)
                x = _parse_py_or_json_literal(x_s)
                return y, x

    raise ValueError(f"Could not find units for id '{eq_id}' (tried '{numeric_id}') in {path}")
