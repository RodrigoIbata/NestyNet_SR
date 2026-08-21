# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


def _relpath_str(p: Path, base: Path) -> str:
    p_abs = Path(p).resolve()
    b_abs = Path(base).resolve()
    try:
        return str(p_abs.relative_to(b_abs))
    except Exception:
        return os.path.relpath(str(p_abs), str(b_abs))


def _traj_pair(tr: Any) -> tuple[str, Path]:
    if isinstance(tr, tuple) and len(tr) == 2:
        return str(tr[0]), Path(tr[1])
    tid = getattr(tr, "traj_id", None) or getattr(tr, "id", None)
    csv = getattr(tr, "csv_path", None) or getattr(tr, "csv", None)
    if tid is None or csv is None:
        raise ValueError(
            "Trajectory must be (id,path) or have traj_id/id and csv_path/csv"
        )
    return str(tid), Path(csv)


def _traj_sort_key(tid: str) -> tuple[int, int | str]:
    m = re.search(r"(\d+)", str(tid))
    if m is None:
        return (1, str(tid))
    return (0, int(m.group(1)))


def write_oracle_de_spec(
    spec_path: str | Path,
    *,
    spec_id: str,
    trajectories: Sequence[Any],
    holdout_last_k: int = 0,
    x_axis: int = 0,
    order_candidates: Sequence[int] = (1, 2),
    include_x: bool = True,
    y_transform: str = "identity",
    traj_metric: str = "max",
    sort_trajectories: bool = False,
    constants: Sequence[Mapping[str, Any]] | None = None,
    dims: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write an oracle_lab_de spec with explicit fit/probe trajectory splits.

    Parameters
    ----------
    constants : list of dicts, optional
        Each dict has ``name``, ``value``, and optionally ``dim`` (list of
        floats).  Exposed as additional feature columns in the explorer.
    dims : dict, optional
        Dimensional analysis metadata with keys ``basis`` (list of str),
        ``x`` (list of float), ``u`` (list of float).
    """

    spec_p = Path(spec_path)
    spec_dir = spec_p.parent
    spec_dir.mkdir(parents=True, exist_ok=True)

    pairs = [_traj_pair(t) for t in trajectories]
    if sort_trajectories:
        pairs = sorted(pairs, key=lambda p: _traj_sort_key(p[0]))

    k = int(holdout_last_k)
    if k < 0:
        raise ValueError(f"holdout_last_k must be >=0, got {k}")
    if k >= len(pairs) and len(pairs) > 0:
        raise ValueError(f"holdout_last_k={k} leaves no fit trajectories (M={len(pairs)})")

    if k == 0:
        fit_pairs = pairs
        probe_pairs: list[tuple[str, Path]] = []
        split_mode = "per_traj_point"
    else:
        fit_pairs = list(pairs[:-k])
        probe_pairs = list(pairs[-k:])
        split_mode = "traj_holdout"

    fit_rows = [{"id": tid, "csv": _relpath_str(csv, spec_dir)} for tid, csv in fit_pairs]
    probe_rows = [{"id": tid, "csv": _relpath_str(csv, spec_dir)} for tid, csv in probe_pairs]

    payload: dict[str, Any] = {
        "id": str(spec_id),
        "x_axis": int(x_axis),
        "order_candidates": [int(o) for o in order_candidates],
        "include_x": bool(include_x),
        "y_transform": str(y_transform),
        "traj_metric": str(traj_metric),
        "split_mode": str(split_mode),
        "holdout_last_k": int(k),
        "fit_trajectories": fit_rows,
        "probe_trajectories": probe_rows,
    }
    if constants:
        payload["constants"] = [dict(c) for c in constants]
    if dims:
        payload["dims"] = dict(dims)
    if extra:
        payload["extra"] = {str(k): v for k, v in extra.items()}

    spec_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
