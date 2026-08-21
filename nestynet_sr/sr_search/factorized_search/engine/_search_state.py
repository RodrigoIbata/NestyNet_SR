# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Factorized-search state containers, schedulers, and frequency hints."""

from __future__ import annotations

# ruff: noqa: F401

import math
from dataclasses import dataclass
from typing import Any, Sequence
import torch

@dataclass
class _ParentSnapshot:
    snapshot_id: str
    residual_basin_key: str
    elite_id: str
    expr: Any
    mapping: dict
    eff_mse: float | None = None
    raw_mse: float | None = None
    expr_str: str = ""
    created_at_iter: int = 0
    last_used_iter: int = 0


@dataclass
class _ExecutionContext:
    selected_parent_key: Any = None
    selected_parent_rec: Any = None
    selected_parent_elite_id: str = ""
    executed_parent_key: Any = None
    executed_parent_rec: Any = None
    executed_parent_elite_id: str = ""
    executed_parent_eff_mse: float | None = None


class _ParentSnapshotStore:
    def __init__(self, *, max_entries: int = 256, staleness_window: int = 1024):
        self.max_entries = max(1, int(max_entries))
        self.staleness_window = max(1, int(staleness_window))
        self._counter = 0
        self._snapshots: dict[str, _ParentSnapshot] = {}
        self._dedupe: dict[tuple[str, str, str], str] = {}

    def capture(
        self,
        *,
        residual_basin_key: Any,
        elite_id: Any,
        expr: Any,
        mapping: dict,
        eff_mse: float | None,
        raw_mse: float | None,
        current_iter: int,
        expr_str: str,
    ) -> str:
        residual_basin_key_str = str(residual_basin_key)
        elite_id_str = str(elite_id or "")
        expr_token = str(expr_str or "")
        dedupe_key = (residual_basin_key_str, elite_id_str, expr_token)
        snapshot_id = self._dedupe.get(dedupe_key)
        if snapshot_id is not None and snapshot_id in self._snapshots:
            snap = self._snapshots[snapshot_id]
            snap.expr = expr
            snap.mapping = mapping
            snap.eff_mse = eff_mse
            snap.raw_mse = raw_mse
            snap.last_used_iter = int(current_iter)
            return snapshot_id
        self._counter += 1
        snapshot_id = f"{residual_basin_key_str}::snapshot::{int(self._counter)}"
        self._snapshots[snapshot_id] = _ParentSnapshot(
            snapshot_id=snapshot_id,
            residual_basin_key=residual_basin_key_str,
            elite_id=elite_id_str,
            expr=expr,
            mapping=mapping,
            eff_mse=eff_mse,
            raw_mse=raw_mse,
            expr_str=expr_token,
            created_at_iter=int(current_iter),
            last_used_iter=int(current_iter),
        )
        self._dedupe[dedupe_key] = snapshot_id
        return snapshot_id

    def get(self, snapshot_id: Any, *, current_iter: int | None = None) -> _ParentSnapshot | None:
        snapshot_id_str = str(snapshot_id or "")
        if not snapshot_id_str:
            return None
        snap = self._snapshots.get(snapshot_id_str)
        if snap is None:
            return None
        if current_iter is not None:
            snap.last_used_iter = int(current_iter)
        return snap

    def prune(self, *, current_iter: int, protected_ids: set[str] | None = None) -> None:
        protected = {str(v) for v in (protected_ids or set()) if str(v)}
        for snapshot_id, snap in list(self._snapshots.items()):
            age = int(current_iter) - int(max(snap.created_at_iter, snap.last_used_iter))
            if snapshot_id in protected or age <= int(self.staleness_window):
                continue
            self._drop(snapshot_id)
        if len(self._snapshots) <= int(self.max_entries):
            return
        removable = [
            (int(snap.last_used_iter), int(snap.created_at_iter), snapshot_id)
            for snapshot_id, snap in self._snapshots.items()
            if snapshot_id not in protected
        ]
        removable.sort()
        while len(self._snapshots) > int(self.max_entries) and removable:
            _, _, snapshot_id = removable.pop(0)
            self._drop(snapshot_id)

    def _drop(self, snapshot_id: str) -> None:
        snap = self._snapshots.pop(str(snapshot_id), None)
        if snap is None:
            return
        dedupe_key = (str(snap.residual_basin_key), str(snap.elite_id), str(snap.expr_str))
        if self._dedupe.get(dedupe_key) == str(snapshot_id):
            del self._dedupe[dedupe_key]


class Explorer:
    def __init__(self, actions, ucb_c=1.0, eps=0.1, ema_alpha=0.1,
                 global_prior_n=5):
        self.actions=list(actions)
        self.ucb_c=float(ucb_c); self.eps=float(eps)
        self.alpha=float(ema_alpha)
        self.global_prior_n=max(1, int(global_prior_n))
        self.n_s={}; self.n_sa={}; self.q_sa={}
        # global (action-only) stats for hybrid prior
        self.n_g={}; self.q_g={}
    def _get(self,d,k,default): return d[k] if k in d else default
    def select_action(self,s_key,rng,allowed_actions=None):
        acts = self.actions if allowed_actions is None else list(allowed_actions)
        if not acts:
            acts = self.actions
        if rng.random()<self.eps:
            return rng.choice(acts)
        n_s=self._get(self.n_s,s_key,0)
        best_a=None; best_score=-1e100
        for a in acts:
            n_sa=self._get(self.n_sa,(s_key,a),0)
            q_local=self._get(self.q_sa,(s_key,a),0.0)
            q_global=self._get(self.q_g,a,0.0)
            # blend: when n_sa is small, lean on global prior
            w = min(n_sa, self.global_prior_n) / self.global_prior_n
            q = w * q_local + (1.0 - w) * q_global
            bonus=self.ucb_c*math.sqrt(math.log(n_s+1.0)/(n_sa+1.0))
            sc=q+bonus
            if sc>best_score: best_score=sc; best_a=a
        return best_a if best_a is not None else rng.choice(acts)
    def update(self,s_key,a,reward):
        self.n_s[s_key]=self._get(self.n_s,s_key,0)+1
        # per-state EMA update
        k=(s_key,a)
        n=self._get(self.n_sa,k,0)
        q=self._get(self.q_sa,k,0.0)
        if n == 0:
            q = reward
        else:
            q=self.alpha*reward+(1.0-self.alpha)*q
        self.n_sa[k]=n+1; self.q_sa[k]=q
        # global EMA update
        n_g=self._get(self.n_g,a,0)
        q_g=self._get(self.q_g,a,0.0)
        if n_g == 0:
            q_g = reward
        else:
            q_g=self.alpha*reward+(1.0-self.alpha)*q_g
        self.n_g[a]=n_g+1; self.q_g[a]=q_g
    def summary(self,topk=8):
        out=[]; qg={a:[] for a in self.actions}
        for (s,a),q in self.q_sa.items(): qg[a].append(q)
        for a in self.actions:
            arr=qg[a]; mu=sum(arr)/max(1,len(arr))
            out.append((mu,a,len(arr)))
        out.sort(reverse=True)
        return out[:topk]


class _RouteScheduler:
    def __init__(self, routes, *, ucb_c=0.25, eps=0.05, ema_alpha=0.1):
        self.routes = [str(r) for r in routes]
        self.ucb_c = float(ucb_c)
        self.eps = float(eps)
        self.alpha = float(ema_alpha)
        self.n = {str(r): 0 for r in self.routes}
        self.q = {str(r): 0.0 for r in self.routes}
        self.reward_sum_raw = {str(r): 0.0 for r in self.routes}
        self.reward_sum_adjusted = {str(r): 0.0 for r in self.routes}
        self.wall_seconds = {str(r): 0.0 for r in self.routes}
        self.reward_count = {str(r): 0 for r in self.routes}
        self.total = 0

    def select(self, rng, available_routes, route_scores=None):
        routes = [str(r) for r in (available_routes or []) if str(r)]
        if not routes:
            routes = list(self.routes)
        extras = dict(route_scores or {})
        if len(routes) == 1:
            return routes[0], "forced"
        if rng.random() < self.eps:
            return str(rng.choice(routes)), "epsilon"
        best_route = routes[0]
        best_score = float("-inf")
        for route in routes:
            n_route = int(self.n.get(route, 0))
            q_route = float(self.q.get(route, 0.0))
            bonus = self.ucb_c * math.sqrt(math.log(float(self.total) + 1.0) / (float(n_route) + 1.0))
            score = q_route + bonus + float(extras.get(route, 0.0))
            if score > best_score:
                best_score = score
                best_route = route
        return str(best_route), "ucb"

    def record_selection(self, route):
        route_name = str(route or "")
        if route_name not in self.n:
            self.n[route_name] = 0
            self.q[route_name] = 0.0
            self.reward_sum_raw[route_name] = 0.0
            self.reward_sum_adjusted[route_name] = 0.0
            self.wall_seconds[route_name] = 0.0
            self.reward_count[route_name] = 0
        self.total += 1
        self.n[route_name] = int(self.n.get(route_name, 0)) + 1

    def update(self, route, reward, *, raw_reward=None, wall_s=None):
        route_name = str(route or "")
        if route_name not in self.n:
            self.n[route_name] = 0
            self.q[route_name] = 0.0
            self.reward_sum_raw[route_name] = 0.0
            self.reward_sum_adjusted[route_name] = 0.0
            self.wall_seconds[route_name] = 0.0
            self.reward_count[route_name] = 0
        n_route = int(self.n.get(route_name, 0))
        q_route = float(self.q.get(route_name, 0.0))
        if n_route == 0:
            self.total += 1
            self.n[route_name] = 1
            q_route = float(reward)
        else:
            q_route = self.alpha * float(reward) + (1.0 - self.alpha) * q_route
        self.q[route_name] = q_route
        try:
            self.reward_sum_raw[route_name] = float(self.reward_sum_raw.get(route_name, 0.0)) + float(
                reward if raw_reward is None else raw_reward
            )
        except Exception:
            pass
        try:
            self.reward_sum_adjusted[route_name] = float(
                self.reward_sum_adjusted.get(route_name, 0.0)
            ) + float(reward)
        except Exception:
            pass
        try:
            self.wall_seconds[route_name] = float(self.wall_seconds.get(route_name, 0.0)) + max(
                0.0, float(wall_s or 0.0)
            )
        except Exception:
            pass
        self.reward_count[route_name] = int(self.reward_count.get(route_name, 0)) + 1

    def summary(self):
        return {
            str(route): {
                "count": int(self.n.get(route, 0)),
                "q": float(self.q.get(route, 0.0)),
                "reward_sum_raw": float(self.reward_sum_raw.get(route, 0.0)),
                "reward_sum_adjusted": float(self.reward_sum_adjusted.get(route, 0.0)),
                "wall_seconds": float(self.wall_seconds.get(route, 0.0)),
                "reward_count": int(self.reward_count.get(route, 0)),
            }
            for route in self.routes
        }


def _periodogram_frequency_hints(
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    *,
    max_hints: int = 2,
    min_prominence: float = 8.0,
    max_points: int = 2048,
    var_indices: "Sequence[int] | None" = None,
) -> list[tuple[int, float]]:
    """Cheap periodogram scan: rough angular frequencies of y against each variable.

    Frequency identification is a needle-in-comb problem for correlation-guided
    search: sin/cos skeletons at the canonical frequency are uncorrelated with
    the data unless the guess is within ~1/span of the truth, so they never
    rank and inner refinement is never spent on them. One FFT per variable
    gives data-driven starting guesses instead. Returns [(var_index, omega)]
    for spectral peaks standing ``min_prominence`` above the median power.
    """
    out: list[tuple[int, float]] = []
    source_eps = float(torch.finfo(torch.float64).eps)
    for tensor in (x_fit, y_fit):
        try:
            if tensor.is_floating_point():
                source_eps = max(source_eps, float(torch.finfo(tensor.dtype).eps))
        except (AttributeError, TypeError):
            pass
    try:
        y = y_fit.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
        x_mat = x_fit.detach().to(dtype=torch.float64, device="cpu")
    except Exception:
        return out
    n = int(y.numel())
    if n < 64 or x_mat.ndim != 2 or int(x_mat.shape[0]) != n:
        return out
    if not bool(torch.isfinite(y).all() and torch.isfinite(x_mat).all()):
        return out
    scan_indices = (
        range(int(x_mat.shape[1]))
        if var_indices is None
        else [int(j) for j in var_indices if 0 <= int(j) < int(x_mat.shape[1])]
    )
    for j in scan_indices:
        xj = x_mat[:, j]
        x_min = float(xj.min())
        span = float(xj.max()) - x_min
        if not math.isfinite(span) or span <= 0.0:
            continue
        order = torch.argsort(xj)
        xs = xj[order]
        ys = y[order]
        m = min(int(max_points), n)
        grid = torch.linspace(x_min, x_min + span, m, dtype=torch.float64)
        idx = torch.searchsorted(xs, grid, right=True).clamp(1, n - 1)
        x0, x1 = xs[idx - 1], xs[idx]
        y0, y1 = ys[idx - 1], ys[idx]
        y_grid = y0 + (y1 - y0) * (grid - x0) / (x1 - x0).clamp_min(1.0e-30)
        signal_scale = float(y_grid.abs().max())
        if not math.isfinite(signal_scale) or signal_scale <= 0.0:
            continue
        y_grid = y_grid / signal_scale
        # Skip phase-like variables: y as a function of a state coordinate is
        # multivalued, so the resampled signal is jump-dominated noise whose
        # spectrum yields spurious peaks.
        y_grid_std = float(y_grid.std())
        if y_grid_std <= 0.0 or float((y_grid[1:] - y_grid[:-1]).abs().median()) > 0.25 * y_grid_std:
            continue
        # Linear detrend + Hann window: trend leakage otherwise shifts or
        # splits tone peaks by several bins.
        t = torch.linspace(-1.0, 1.0, m, dtype=torch.float64)
        y_grid = y_grid - y_grid.mean()
        slope = float((t * y_grid).mean()) / max(float((t * t).mean()), 1.0e-30)
        y_grid = y_grid - slope * t
        residual_scale = float(y_grid.abs().max())
        if not math.isfinite(residual_scale) or residual_scale <= 8.0 * source_eps:
            continue
        # Normalize before squaring the FFT magnitude so frequency detection
        # remains invariant even for extremely small but representable tones.
        y_grid = y_grid / residual_scale
        window = 0.5 - 0.5 * torch.cos(2.0 * math.pi * torch.arange(m, dtype=torch.float64) / max(m - 1, 1))
        power = torch.fft.rfft(y_grid * window).abs().square()
        if int(power.numel()) < 8:
            continue
        power[0] = 0.0
        power[1] = 0.0  # drop DC and the slowest trend bin
        noise = float(power[2:].median())
        if not math.isfinite(noise) or noise <= 0.0:
            noise = 1.0e-300
        # Peaks must be genuine local maxima at bin >= 3 (>= ~3 observed
        # periods): a monotone decaying spectrum (e.g. exponential trends)
        # otherwise shows a shoulder artifact next to the zeroed trend bins.
        peak_mask = torch.zeros_like(power, dtype=torch.bool)
        peak_mask[3:-1] = (
            (power[3:-1] > power[2:-2])
            & (power[3:-1] >= power[4:])
            & (power[3:-1] > float(min_prominence) * noise)
        )
        peak_bins = torch.nonzero(peak_mask).reshape(-1)
        if int(peak_bins.numel()) == 0:
            continue
        peak_bins = peak_bins[torch.argsort(power[peak_bins], descending=True)][: max(1, int(max_hints))]
        for k in peak_bins.tolist():
            # Parabolic peak interpolation sharpens the frequency well below
            # bin resolution (a bin-edge error accumulates visible phase
            # drift over a many-period record).
            k_refined = float(k)
            if 1 <= k < int(power.numel()) - 1:
                p_prev = float(power[k - 1])
                p_mid = float(power[k])
                p_next = float(power[k + 1])
                denom = p_prev - 2.0 * p_mid + p_next
                if math.isfinite(denom) and abs(denom) > 0.0:
                    delta = 0.5 * (p_prev - p_next) / denom
                    if math.isfinite(delta) and abs(delta) <= 0.5:
                        k_refined = float(k) + delta
            omega = 2.0 * math.pi * k_refined / span
            if not (math.isfinite(omega) and omega > 0.0):
                continue
            # Gauss-Newton frequency polish: fit A*sin + B*cos at omega, then
            # take linearized delta-omega steps. Bin-level frequency error
            # accumulates visible phase drift over a many-period record.
            t_axis = grid - x_min
            for _ in range(2):
                s = torch.sin(omega * t_axis)
                c = torch.cos(omega * t_axis)
                design = torch.stack([s, c], dim=1)
                try:
                    ab = torch.linalg.lstsq(design, y_grid.reshape(-1, 1)).solution.reshape(-1)
                except Exception:
                    break
                a_amp, b_amp = float(ab[0]), float(ab[1])
                resid = y_grid - a_amp * s - b_amp * c
                grad = a_amp * t_axis * c - b_amp * t_axis * s
                denom = float((grad * grad).sum())
                if not math.isfinite(denom) or denom <= 0.0:
                    break
                delta = float((resid * grad).sum()) / denom
                if not math.isfinite(delta) or abs(delta) > 2.0 * math.pi / span:
                    break
                omega = omega + delta
            if math.isfinite(omega) and omega > 0.0:
                out.append((int(j), float(omega)))
    return out

__engine_search_definitions__ = (
    "_ParentSnapshot",
    "_ExecutionContext",
    "_ParentSnapshotStore",
    "Explorer",
    "_RouteScheduler",
    "_periodogram_frequency_hints",
)

__engine_search_constants__ = (

)

__engine_search_late_bindings__ = (

)
