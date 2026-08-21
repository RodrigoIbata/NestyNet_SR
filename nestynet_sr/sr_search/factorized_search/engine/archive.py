# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Engine-owned factorized symbolic search archive records and ranking policy."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from nestynet_sr.sr_search.model_selection import mapping_cost

from ..expr_ast import node_cost_physics_prior, node_str


@dataclass
class Elite:
    mse: float
    expr: object
    mapping: dict
    z: torch.Tensor
    size: float
    raw_mse: float = float("inf")
    elite_id: str = ""


@dataclass
class Rec:
    best_mse: float
    best_expr: object
    visits: int
    mapping: dict
    z: torch.Tensor
    elites: list[Elite] = field(default_factory=list)
    best_raw_mse: float = float("inf")
    last_improve_eval: int = 0
    visits_since_improve: int = 0
    residual_basin_key: Any = None
    best_elite_id: str = ""
    min_raw_mse: float = float("inf")


class ResidualBasinArchive:
    def __init__(self, elite_k: int = 8, elite_merge_cos: float = 0.98):
        self.d = {}
        self.n_eval = 0
        self.elite_k = max(1, int(elite_k))
        self.elite_merge_cos = float(elite_merge_cos)
        self._elite_counter = 0

    def _next_elite_id(self, key: Any) -> str:
        self._elite_counter += 1
        return f"{str(key)}::elite::{int(self._elite_counter)}"

    @staticmethod
    def _cos_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
        try:
            aa = a.flatten().to(dtype=torch.float32)
            bb = b.flatten().to(dtype=torch.float32)
            denom = float(aa.norm().item() * bb.norm().item())
            if (not math.isfinite(denom)) or denom <= eps:
                return 0.0
            return float((aa @ bb).item() / denom)
        except Exception:
            return 0.0

    def update(self, key: Any, mse: float, expr: Any, z: torch.Tensor, mapping: dict, raw_mse: float | None = None):
        """Update archive with a scored expression."""
        try:
            mse_f = float(mse)
        except Exception:
            return False
        if not math.isfinite(mse_f):
            return False
        self.n_eval += 1
        zc = z.detach().cpu()
        try:
            raw = mse_f if raw_mse is None else float(raw_mse)
        except Exception:
            raw = mse_f
        if not math.isfinite(raw):
            raw = mse_f
        new_el = Elite(
            mse_f,
            expr,
            mapping,
            zc,
            float(node_cost_physics_prior(expr)) + float(mapping_cost(mapping)),
            raw_mse=raw,
            elite_id=self._next_elite_id(key),
        )

        r = self.d.get(key)
        is_new = r is None
        if r is None:
            self.d[key] = Rec(
                new_el.mse,
                new_el.expr,
                1,
                new_el.mapping,
                new_el.z,
                [new_el],
                best_raw_mse=float(new_el.raw_mse),
                last_improve_eval=int(self.n_eval),
                visits_since_improve=0,
                residual_basin_key=key,
                best_elite_id=str(new_el.elite_id),
                min_raw_mse=float(new_el.raw_mse),
            )
            return True

        prev_best_mse = float(getattr(r, "best_mse", float("inf")))
        prev_best_raw_mse = float(getattr(r, "best_raw_mse", prev_best_mse))
        r.visits += 1
        r.visits_since_improve = int(getattr(r, "visits_since_improve", 0)) + 1
        if getattr(r, "residual_basin_key", None) is None:
            r.residual_basin_key = key

        best_j = None
        best_sim = -1.0
        for j, el in enumerate(r.elites):
            sim = self._cos_sim(el.z, new_el.z)
            if sim > best_sim:
                best_sim = sim
                best_j = j

        if best_j is not None and best_sim >= self.elite_merge_cos:
            el = r.elites[best_j]
            if (new_el.mse < el.mse) or (new_el.mse < el.mse * 1.05 and new_el.size < el.size):
                r.elites[best_j] = new_el
        else:
            if len(r.elites) < self.elite_k:
                r.elites.append(new_el)
            else:
                worst_j = max(range(len(r.elites)), key=lambda j: (r.elites[j].mse, r.elites[j].size))
                worst = r.elites[worst_j]
                if (new_el.mse < worst.mse) or (new_el.mse < worst.mse * 1.05 and new_el.size < worst.size):
                    r.elites[worst_j] = new_el

        if r.elites:
            best_mse = min(el.mse for el in r.elites)
            pool = [el for el in r.elites if el.mse <= best_mse * 1.05]
            best = min(pool, key=lambda el: (el.size, el.mse))
            r.best_mse = best.mse
            r.best_raw_mse = float(getattr(best, "raw_mse", best.mse))
            r.min_raw_mse = min(float(getattr(el, "raw_mse", el.mse)) for el in r.elites)
            r.best_expr = best.expr
            r.mapping = best.mapping
            r.z = best.z
            r.best_elite_id = str(getattr(best, "elite_id", "") or "")

        improved = False
        try:
            improved = float(r.best_mse) < float(prev_best_mse) * (1.0 - 1.0e-12)
        except Exception:
            improved = False
        if not improved:
            try:
                improved = float(r.best_raw_mse) < float(prev_best_raw_mse) * (1.0 - 1.0e-12)
            except Exception:
                improved = False
        if improved:
            r.last_improve_eval = int(self.n_eval)
            r.visits_since_improve = 0

        return is_new

    def best(self, k: int, strategy: str = "mse") -> list[Rec]:
        try:
            k = int(k)
        except Exception:
            k = 0
        if k <= 0:
            return []

        all_elites = []
        for residual_basin_key, r in self.d.items():
            for el in r.elites:
                mse = float(getattr(el, "mse", 1e100))
                size = float(getattr(el, "size", 1.0e100))
                raw_mse = float(getattr(el, "raw_mse", mse))
                if not math.isfinite(raw_mse):
                    raw_mse = mse
                all_elites.append((
                    mse,
                    size,
                    raw_mse,
                    el,
                    residual_basin_key,
                    int(getattr(r, "visits", 0)),
                    int(getattr(r, "last_improve_eval", 0)),
                    int(getattr(r, "visits_since_improve", 0)),
                ))
        if not all_elites:
            return []

        all_by_mse = sorted(all_elites, key=lambda t: (t[0], t[1]))
        if k == 1 or str(strategy).strip().lower() in ("mse", "legacy", ""):
            out = []
            for mse, _, raw_mse, el, residual_basin_key, visits, last_improve_eval, visits_since_improve in all_by_mse[:k]:
                out.append(
                    Rec(
                        float(mse),
                        el.expr,
                        int(visits),
                        el.mapping,
                        el.z,
                        [],
                        best_raw_mse=float(raw_mse),
                        last_improve_eval=int(last_improve_eval),
                        visits_since_improve=int(visits_since_improve),
                        residual_basin_key=residual_basin_key,
                        best_elite_id=str(getattr(el, "elite_id", "") or ""),
                        min_raw_mse=float(raw_mse),
                    )
                )
            return out

        strategy_l = str(strategy).strip().lower()
        if strategy_l not in ("mse_decade_size", "decade_size", "paretoish"):
            out = []
            for mse, _, raw_mse, el, residual_basin_key, visits, last_improve_eval, visits_since_improve in all_by_mse[:k]:
                out.append(
                    Rec(
                        float(mse),
                        el.expr,
                        int(visits),
                        el.mapping,
                        el.z,
                        [],
                        best_raw_mse=float(raw_mse),
                        last_improve_eval=int(last_improve_eval),
                        visits_since_improve=int(visits_since_improve),
                        residual_basin_key=residual_basin_key,
                        best_elite_id=str(getattr(el, "elite_id", "") or ""),
                        min_raw_mse=float(raw_mse),
                    )
                )
            return out

        def _mse_decade(mse: float) -> int:
            if not math.isfinite(mse):
                return int(1e9)
            mse = max(float(mse), 1.0e-300)
            return int(math.floor(math.log10(mse)))

        bins = {}
        for row in all_elites:
            d = _mse_decade(row[2])
            bins.setdefault(d, []).append(row)
        for d in bins:
            bins[d].sort(key=lambda t: (t[1], t[0], t[2]))

        selected = []
        selected_ids = set()
        best_row = all_by_mse[0]
        selected.append(best_row)
        selected_ids.add(id(best_row))

        best_decade = _mse_decade(best_row[2])
        max_decade_span = 8
        decades = [
            d for d in sorted(bins.keys())
            if int(d) <= int(best_decade + max_decade_span)
        ]
        if not decades:
            decades = sorted(bins.keys())
        offsets = {d: 0 for d in decades}
        while len(selected) < k:
            progressed = False
            for d in decades:
                arr = bins[d]
                i = offsets[d]
                while i < len(arr) and id(arr[i]) in selected_ids:
                    i += 1
                offsets[d] = i
                if i >= len(arr):
                    continue
                row = arr[i]
                offsets[d] = i + 1
                selected.append(row)
                selected_ids.add(id(row))
                progressed = True
                if len(selected) >= k:
                    break
            if not progressed:
                break

        if len(selected) < k:
            for row in all_by_mse:
                if id(row) in selected_ids:
                    continue
                selected.append(row)
                selected_ids.add(id(row))
                if len(selected) >= k:
                    break

        out = []
        for mse, _, raw_mse, el, residual_basin_key, visits, last_improve_eval, visits_since_improve in selected[:k]:
            out.append(
                Rec(
                    float(mse),
                    el.expr,
                    int(visits),
                    el.mapping,
                    el.z,
                    [],
                    best_raw_mse=float(raw_mse),
                    last_improve_eval=int(last_improve_eval),
                    visits_since_improve=int(visits_since_improve),
                    residual_basin_key=residual_basin_key,
                    best_elite_id=str(getattr(el, "elite_id", "") or ""),
                    min_raw_mse=float(raw_mse),
                )
            )
        return out

    def audit_coherence(self, *, max_examples: int = 8) -> dict[str, Any]:
        """Check basin-level best fields all describe the same elite."""

        def _node_key(node: Any) -> str:
            try:
                return node_str(node)
            except Exception:
                return repr(node)

        def _mapping_key(mapping: Any) -> str:
            try:
                return repr(mapping)
            except Exception:
                return str(type(mapping).__name__)

        checked = 0
        failure_count = 0
        failures: list[dict[str, Any]] = []
        for residual_basin_key, rec in self.d.items():
            elites = list(getattr(rec, "elites", []) or [])
            if not elites:
                continue
            checked += 1
            best = None
            best_id = str(getattr(rec, "best_elite_id", "") or "")
            if best_id:
                for candidate in elites:
                    if str(getattr(candidate, "elite_id", "") or "") == best_id:
                        best = candidate
                        break
            if best is None:
                best_mse = min(float(getattr(el, "mse", float("inf"))) for el in elites)
                pool = [el for el in elites if float(getattr(el, "mse", float("inf"))) <= best_mse * 1.05]
                best = min(pool, key=lambda el: (float(getattr(el, "size", float("inf"))), float(getattr(el, "mse", float("inf")))))
            issues = []
            if _node_key(getattr(rec, "best_expr", None)) != _node_key(getattr(best, "expr", None)):
                issues.append("best_expr")
            if _mapping_key(getattr(rec, "mapping", None)) != _mapping_key(getattr(best, "mapping", None)):
                issues.append("mapping")
            rec_mse = float(getattr(rec, "best_mse", float("inf")))
            best_mse = float(getattr(best, "mse", float("inf")))
            if abs(rec_mse - best_mse) > max(1.0e-12, 1.0e-9 * max(abs(rec_mse), abs(best_mse), 1.0)):
                issues.append("best_mse")
            rec_raw = float(getattr(rec, "best_raw_mse", float("inf")))
            best_raw = float(getattr(best, "raw_mse", getattr(best, "mse", float("inf"))))
            if abs(rec_raw - best_raw) > max(1.0e-12, 1.0e-9 * max(abs(rec_raw), abs(best_raw), 1.0)):
                issues.append("best_raw_mse")
            if issues:
                failure_count += 1
                if len(failures) < int(max_examples):
                    failures.append(
                        {
                            "residual_basin_key": str(residual_basin_key),
                            "best_elite_id": best_id,
                            "issues": issues,
                        }
                    )
        return {
            "ok": int(failure_count) == 0,
            "checked": int(checked),
            "failures": int(failure_count),
            "examples": failures,
        }

    def resolve_elite(
        self,
        residual_basin_key: Any,
        *,
        elite_id: Any | None = None,
        expr_str: str | None = None,
    ) -> tuple[Rec | None, Elite | None]:
        rec = self.d.get(residual_basin_key)
        if rec is None:
            return None, None

        elite = None
        elite_id_str = "" if elite_id is None else str(elite_id)
        if elite_id_str:
            for candidate in rec.elites:
                if str(getattr(candidate, "elite_id", "") or "") == elite_id_str:
                    elite = candidate
                    break

        if elite is None and expr_str:
            expr_token = str(expr_str)
            for candidate in rec.elites:
                try:
                    if node_str(candidate.expr) == expr_token:
                        elite = candidate
                        break
                except Exception:
                    continue

        if elite is None and str(getattr(rec, "best_elite_id", "") or ""):
            best_id = str(getattr(rec, "best_elite_id", "") or "")
            for candidate in rec.elites:
                if str(getattr(candidate, "elite_id", "") or "") == best_id:
                    elite = candidate
                    break

        return rec, elite

    def most_visited(self, k: int):
        return sorted(self.d.values(), key=lambda r: r.visits, reverse=True)[:k]

    def items(self):
        return self.d.items()


__all__ = [
    "ResidualBasinArchive",
    "Elite",
    "Rec",
]
