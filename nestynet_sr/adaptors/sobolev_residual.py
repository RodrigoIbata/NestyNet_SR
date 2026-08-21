# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Sobolev-style value-and-gradient residual wrapper.

The wrapped provider remains the model used for inference.  This adaptor only
changes the training residual from ``y - f(x)`` to
``[y - f(x), g_target - grad_x f(x)]`` with optional channel weights/scales.
It deliberately does not encode any physics residual or cross-component law.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def _as_B1(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 1:
        return t.unsqueeze(1)
    if t.ndim == 2 and int(t.shape[1]) == 1:
        return t
    if t.ndim == 2:
        return t[:, :1]
    raise ValueError(f"expected 1D or 2D tensor, got shape={tuple(t.shape)}")


def _as_BD(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 2:
        return t
    if t.ndim == 3 and int(t.shape[1]) == 1:
        return t[:, 0, :]
    raise ValueError(f"expected shape (B,D) or (B,1,D), got {tuple(t.shape)}")


def _as_1P(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 1:
        return t.reshape(1, -1)
    if t.ndim == 2:
        return t.reshape(1, -1) if int(t.shape[0]) != 1 else t
    return t.reshape(1, -1)


def _as_BNN(t: torch.Tensor) -> torch.Tensor:
    """Normalize a scalar-output Hessian to (B, Nx, Nx) (drops a singleton O axis)."""
    if t.ndim == 4:           # (B, O, Nx, Nx) with O==1
        return t[:, 0]
    if t.ndim == 3:           # (B, Nx, Nx)
        return t
    raise ValueError(f"expected Hessian (B,Nx,Nx) or (B,1,Nx,Nx), got {tuple(t.shape)}")


class SobolevGradientAdaptor(torch.nn.Module):
    """LAProvider-compatible wrapper for scalar value+gradient training."""

    # The generic greedy canonical initializer fits a *scalar value* residual and
    # cannot consume the augmented [value, d/dt, d/dx, ...] target this wrapper
    # presents to the LM stage.  We therefore intercept canonical initialization
    # at high priority and feed it only the value column ("value-projected
    # canonical init"): the model is deterministically placed to approximate the
    # field value, and the subsequent Sobolev LM stage refines value+gradient.
    canonical_init_dispatch_priority = 110

    def __init__(
        self,
        base: torch.nn.Module,
        axes: Sequence[int],
        value_weight: float = 1.0,
        grad_weight: float = 1.0,
        value_scale: float | torch.Tensor | None = None,
        grad_scales: Sequence[float] | torch.Tensor | None = None,
        hess_pairs: Sequence[tuple[int, int]] | None = None,
        hess_weight: float = 0.0,
        hess_scales: Sequence[float] | torch.Tensor | None = None,
        hess_trace_dims: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.axes = tuple(int(a) for a in axes)
        if len(self.axes) == 0:
            raise ValueError("SobolevGradientAdaptor requires at least one derivative axis")
        if any(a < 0 for a in self.axes):
            raise ValueError(f"axes must be non-negative, got {self.axes}")

        self.value_weight = float(value_weight)
        self.grad_weight = float(grad_weight)
        if self.value_weight < 0.0 or self.grad_weight < 0.0:
            raise ValueError("value_weight and grad_weight must be non-negative")

        vs = torch.as_tensor(1.0 if value_scale is None else value_scale, dtype=torch.float64).reshape(())
        if grad_scales is None:
            gs = torch.ones(len(self.axes), dtype=torch.float64)
        else:
            gs = torch.as_tensor(grad_scales, dtype=torch.float64).reshape(-1)
        if int(gs.numel()) != len(self.axes):
            raise ValueError(
                f"grad_scales length {int(gs.numel())} must match axes length {len(self.axes)}"
            )
        self.register_buffer("_value_scale", vs)
        self.register_buffer("_grad_scales", gs)

        # Optional H^2 (curvature) supervision.  Two MUTUALLY EXCLUSIVE modes:
        #  * hess_pairs       -- match individual ∂²f/∂x_a∂x_b entries (one residual
        #                        channel each).
        #  * hess_trace_dims  -- match the LAPLACIAN  Σ_a ∂²f/∂x_a²  over the given
        #                        axes as a SINGLE residual channel.  This is the
        #                        quantity the ∇² decoy column actually uses, and the
        #                        only honest target when per-axis diagonals are mostly
        #                        structurally zero: per-channel RMS normalization would
        #                        otherwise blow up noise-only channels (their target is
        #                        pure FFT-differentiated noise) and the optimizer would
        #                        spend capacity fitting that noise.
        # Off by default (hess_weight=0, no pairs/dims) so existing value+gradient
        # Sobolev runs are byte-identical.
        self.hess_pairs = tuple((int(a), int(b)) for (a, b) in (hess_pairs or ()))
        self.hess_trace_dims = tuple(int(a) for a in (hess_trace_dims or ()))
        if self.hess_pairs and self.hess_trace_dims:
            raise ValueError("hess_pairs and hess_trace_dims are mutually exclusive")
        if any(a < 0 or b < 0 for (a, b) in self.hess_pairs):
            raise ValueError(f"hess_pairs must be non-negative, got {self.hess_pairs}")
        if any(a < 0 for a in self.hess_trace_dims):
            raise ValueError(f"hess_trace_dims must be non-negative, got {self.hess_trace_dims}")
        self._hess_trace = bool(self.hess_trace_dims)
        self._has_hess = bool(self.hess_pairs) or self._hess_trace
        self.hess_weight = float(hess_weight)
        if self.hess_weight < 0.0:
            raise ValueError("hess_weight must be non-negative")
        # number of H^2 output channels: 1 (Laplacian trace) or one per pair
        self._n_hess = 1 if self._hess_trace else len(self.hess_pairs)
        if hess_scales is None:
            hsc = torch.ones(self._n_hess, dtype=torch.float64)
        else:
            hsc = torch.as_tensor(hess_scales, dtype=torch.float64).reshape(-1)
        if int(hsc.numel()) != self._n_hess:
            raise ValueError(
                f"hess_scales length {int(hsc.numel())} must match number of H^2 channels {self._n_hess}"
            )
        self.register_buffer("_hess_scales", hsc)
        # Both modes use the leaf's selected-diagonal analytic methods (value + dense
        # parameter Jacobian) which never form the Nx×Nx Hessian -- this keeps the H²
        # fit on the fast direct_solve path.  The trace is the sum over its axes; pairs
        # are per-axis.  Off-diagonal pairs (only) fall back to the full-Hessian jvp.
        if self._hess_trace:
            self._hess_all_diag = True
            self._hess_diag_dims = [int(a) for a in self.hess_trace_dims]
        else:
            self._hess_all_diag = bool(self.hess_pairs) and all(a == b for (a, b) in self.hess_pairs)
            self._hess_diag_dims = [int(a) for (a, _b) in self.hess_pairs] if self._hess_all_diag else None
        # The generic optimizer FD/JVP audit is too brittle for derivative
        # residuals wrapped around dual-layer providers.  The analytic
        # Sobolev dense-Jacobian/JVP paths are tested directly instead.
        self.skip_jac_sanity = True
        self._base_grad_jvp_supported: bool | None = None
        self._base_grad_vjp_supported: bool | None = None

    @property
    def n_outputs(self) -> int:
        return 1 + len(self.axes) + self._n_hess

    @property
    def base_model(self):
        bm = getattr(self.base, "base_model", None)
        if bm is not None:
            return bm
        leaves = getattr(self.base, "leaf", None)
        if leaves is not None:
            try:
                leaves_list = list(leaves)
            except Exception:
                leaves_list = []
            if len(leaves_list) == 1:
                return getattr(leaves_list[0], "base_model", None)
        return None

    @property
    def n_params(self) -> int:
        return int(self.num_parameters())

    def num_parameters(self) -> int:
        npar = getattr(self.base, "num_parameters", None)
        if callable(npar):
            return int(npar())
        return int(sum(p.numel() for p in self.base.parameters()))

    # ------------------------------------------------------------------
    # Value-projected canonical (deterministic) initialization.
    # The generic machinery is reused, but fed only the value column of the
    # augmented Sobolev target; the gradient channels are refined later by LM.
    # ------------------------------------------------------------------
    def canonical_default_segments(self):
        candidates = [self.base, getattr(self.base, "base_model", None), self.base_model]
        leaves = getattr(self.base, "leaf", None)
        if leaves is not None:
            try:
                candidates.extend(list(leaves))
            except Exception:
                pass
        for obj in candidates:
            if obj is None:
                continue
            fn = getattr(obj, "canonical_default_segments", None)
            if callable(fn):
                try:
                    return list(fn())
                except Exception:
                    pass
            bm = getattr(obj, "base_model", obj)
            nseg = getattr(bm, "num_segments", None)
            if nseg is not None:
                return list(range(int(nseg)))
        raise AttributeError("SobolevGradientAdaptor could not infer canonical segments")

    def _canonical_project_target(self, y_train, y_sigma=None, *, device=None, dtype=None):
        Y = torch.as_tensor(y_train, device=device, dtype=dtype)
        if Y.ndim == 1:
            Y = Y.unsqueeze(1)
        if Y.ndim != 2 or int(Y.shape[1]) < 1:
            raise ValueError(
                f"Sobolev canonical target must be 2D with a value column, got {tuple(Y.shape)}"
            )
        Yv = Y[:, :1]
        if y_sigma is None:
            return Yv, None
        S = torch.as_tensor(y_sigma, device=device, dtype=dtype)
        if S.ndim <= 1:
            return Yv, S
        if S.ndim == 2:
            return Yv, S[:, :1]
        return Yv, S.reshape(S.shape[0], -1)[:, :1]

    def _canonical_effective_x(self, x_train):
        """Route the canonical input through a leading NN atom for AST composites."""
        base = self.base
        ast_root = getattr(base, "ast_root", None)
        leaves = getattr(base, "leaf", None)
        if ast_root is None or leaves is None:
            return x_train
        try:
            leaves_list = list(leaves)
        except Exception:
            return x_train
        if len(leaves_list) != 1 or str(getattr(ast_root, "kind", "")).lower() != "nn":
            return x_train
        try:
            from nestynet_sr.sr_core.bridges import eval_inputs
            x_eff, _, _ = eval_inputs(ast_root, x_train, need_grad=False, need_hess=False)
            return x_eff
        except Exception:
            return x_train

    @torch.no_grad()
    def canonical_initialize(
        self, lm_optimizer, *, x_train, y_train, y_sigma, segments, config, device, dtype,
    ):
        from nestynet.training_utils.canonical_init import (
            _canonical_init_dual,
            _canonical_init_single,
            _find_canonical_base_model,
            _find_dual_segmented_adaptor,
        )
        X = torch.as_tensor(x_train, device=device, dtype=dtype)
        Xeff = self._canonical_effective_x(X)
        Yv, Sv = self._canonical_project_target(y_train, y_sigma, device=device, dtype=dtype)
        segs = list(segments)
        dual = _find_dual_segmented_adaptor(self.base, lm_optimizer=None)
        if dual is not None:
            info = _canonical_init_dual(
                dual, lm_optimizer, x_train=Xeff, y_train=Yv, y_sigma=Sv,
                segments=segs, config=config, device=device, dtype=dtype,
            )
        else:
            base_model = _find_canonical_base_model(self.base, lm_optimizer=None)
            if base_model is None:
                raise AttributeError(
                    "SobolevGradientAdaptor could not find a canonical-initializable base model"
                )
            info = _canonical_init_single(
                base_model, lm_optimizer, x_train=Xeff, y_train=Yv, y_sigma=Sv,
                segments=segs, config=config, device=device, dtype=dtype,
            )
        metrics = info.get("_canonical_metrics", {}) if isinstance(info, dict) else {}
        return {
            "kind": "sobolev_value_projected_canonical",
            "inner": info,
            "sobolev_n_outputs": int(self.n_outputs),
            "value_column": 0,
            "_canonical_metrics": {
                **metrics,
                "kind": "sobolev_value_projected_canonical",
                "sobolev_value_projected": True,
                "sobolev_n_outputs": int(self.n_outputs),
            },
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)

    def grad(self, cache_or_x, out_dim: int | None = None):
        return self.base.grad(cache_or_x, out_dim=out_dim)

    def grad_grad(self, cache_or_x, out_dim: int | None = None):
        return self.base.grad_grad(cache_or_x, out_dim=out_dim)

    def _scales(self, *, device: torch.device, dtype: torch.dtype):
        value = self._value_scale.to(device=device, dtype=dtype) * (self.value_weight ** 0.5)
        grad = self._grad_scales.to(device=device, dtype=dtype) * (self.grad_weight ** 0.5)
        hess = self._hess_scales.to(device=device, dtype=dtype) * (self.hess_weight ** 0.5)
        return value, grad, hess

    def _hess_pred(self, base_cache) -> torch.Tensor:
        """H² predictions -> (B, n_hess).  Trace mode: one Laplacian column (sum over
        ``hess_trace_dims``); pairs mode: one column per (diagonal) pair."""
        if self._hess_all_diag and hasattr(self.base, "grad_grad_diag"):
            # Fast: selected diagonal without forming the Nx×Nx Hessian.  Columns
            # come back in self._hess_diag_dims order (= pair order / trace axes).
            hd = self.base.grad_grad_diag(base_cache, input_dims=self._hess_diag_dims, out_dim=0)
            hd = hd.reshape(hd.shape[0], -1)
            return hd.sum(dim=1, keepdim=True) if self._hess_trace else hd
        hh = _as_BNN(self.base.grad_grad(base_cache, out_dim=0))
        if self._hess_trace:
            return torch.stack([hh[:, a, a] for a in self.hess_trace_dims], dim=1).sum(dim=1, keepdim=True)
        return torch.stack([hh[:, a, b] for (a, b) in self.hess_pairs], dim=1)

    def _hess_diag_jacobian_residual(self, base_cache) -> torch.Tensor | None:
        """``-∂(H² channels)/∂θ`` -> (B, n_hess, P).  Trace mode sums the per-axis
        diagonal Jacobians (J_Δ = Σ_a J_aa); pairs mode keeps one row per pair.

        Residual sign (the H² residual is ``target - H``, so its Jacobian is
        ``-∂H/∂θ``, matching the gradient block's ``Jg = -Jf``).  Uses the leaf's
        dense selected-diagonal Jacobian -- no Nx×Nx Hessian-Jacobian."""
        if not self._hess_all_diag:
            return None
        fn = getattr(self.base, "grad_grad_diag_jacobian", None)
        if not callable(fn):
            return None
        J = fn(base_cache, input_dims=self._hess_diag_dims, out_dim=0)  # (B, D, P), forward
        J = -J
        return J.sum(dim=1, keepdim=True) if self._hess_trace else J  # (B, n_hess, P)

    def build_cache(self, data, **kw) -> dict[str, Any]:
        x, y_aug, *rest = data
        if y_aug.ndim != 2 or int(y_aug.shape[1]) != self.n_outputs:
            raise ValueError(
                f"Sobolev y_aug must have shape (B,{self.n_outputs}), got {tuple(y_aug.shape)}"
            )
        base_cache = self.base.build_cache((x, y_aug[:, :1], *rest), **kw)
        with torch.no_grad():
            f = _as_B1(base_cache.get("f", self.base(x)))
            g = _as_BD(self.base.grad(base_cache, out_dim=0))
            pred_cols = [f, g[:, list(self.axes)]]
            if self._has_hess:
                pred_cols.append(self._hess_pred(base_cache))
            pred = torch.cat(pred_cols, dim=1).detach()
        return {
            "base": base_cache,
            "x": x,
            "y": y_aug,
            "f": pred,
            "axes": self.axes,
            "segments": kw.get("segments", None),
            "SegmentedModel": False,
            "S": 1,
            "Pseg": self.num_parameters(),
            "O": self.n_outputs,
        }

    def residuals(self, cache, data=None, *, track_grad: bool = False) -> torch.Tensor:
        x, y_aug = (cache["x"], cache["y"]) if data is None else data[:2]
        base_cache = cache["base"]
        value_scale, grad_scales, hess_scales = self._scales(device=x.device, dtype=x.dtype)
        n_axes = len(self.axes)
        with torch.set_grad_enabled(track_grad):
            r_value = (
                _as_B1(
                    self.base.residuals(
                        base_cache,
                        (x, y_aug[:, :1]),
                        track_grad=track_grad,
                    )
                )
                * value_scale
            )
            g = _as_BD(self.base.grad(base_cache, out_dim=0))
            r_grad = (y_aug[:, 1:1 + n_axes] - g[:, list(self.axes)]) * grad_scales.view(1, -1)
            rows = [r_value, r_grad]
            if self._has_hess:
                hpred = self._hess_pred(base_cache)
                r_hess = (y_aug[:, 1 + n_axes:1 + n_axes + self._n_hess] - hpred) * hess_scales.view(1, -1)
                rows.append(r_hess)
            out = torch.cat(rows, dim=1)
        return out if track_grad else out.detach()

    def residuals_lm(self, _p, model_fn, data, *, track_grad: bool = False) -> torch.Tensor:
        raise RuntimeError(
            "SobolevGradientAdaptor disables residuals_lm because that path would "
            "differentiate the surrogate with torch autograd. Use the analytic "
            "jvp/vjp/dense paths instead."
        )

    def jvp(self, cache, v: torch.Tensor, out_dim: int | None = None) -> torch.Tensor:
        x = cache["x"]
        value_scale, grad_scales, hess_scales = self._scales(device=x.device, dtype=x.dtype)
        base_cache = cache["base"]
        n_axes = len(self.axes)

        if out_dim is not None:
            od = int(out_dim)
            if od == 0:
                return _as_B1(self.base.jvp(base_cache, v, out_dim=0)).reshape(-1) * value_scale
            if od <= n_axes:
                axis_pos = od - 1
                gjv = self._gradient_jvp(base_cache, v)
                return gjv[:, int(self.axes[axis_pos])] * grad_scales[axis_pos]
            pair_pos = od - 1 - n_axes
            if pair_pos < 0 or pair_pos >= self._n_hess:
                raise IndexError(f"out_dim={out_dim} out of range for Sobolev O={self.n_outputs}")
            hjv = self._hessian_jvp(base_cache, v)
            if self._hess_trace:
                tr = sum(hjv[:, int(a), int(a)] for a in self.hess_trace_dims)
                return tr * hess_scales[0]
            a, b = self.hess_pairs[pair_pos]
            return hjv[:, int(a), int(b)] * hess_scales[pair_pos]

        jv_value = _as_B1(self.base.jvp(base_cache, v, out_dim=0)) * value_scale
        gjv = self._gradient_jvp(base_cache, v)
        jv_grad = gjv[:, list(self.axes)] * grad_scales.view(1, -1)
        cols = [jv_value, jv_grad]
        if self._has_hess:
            hjv = self._hessian_jvp(base_cache, v)
            if self._hess_trace:
                jv_hess = sum(hjv[:, int(a), int(a)] for a in self.hess_trace_dims).unsqueeze(1) * hess_scales.view(1, -1)
            else:
                jv_hess = torch.stack([hjv[:, a, b] for (a, b) in self.hess_pairs], dim=1) * hess_scales.view(1, -1)
            cols.append(jv_hess)
        return torch.cat(cols, dim=1)

    def vjp(self, cache, w: torch.Tensor, out_dim: int | None = None) -> torch.Tensor:
        x = cache["x"]
        value_scale, grad_scales, hess_scales = self._scales(device=x.device, dtype=x.dtype)
        base_cache = cache["base"]
        n_axes = len(self.axes)

        if out_dim is not None:
            od = int(out_dim)
            if w.ndim == 2 and int(w.shape[1]) == 1:
                w1 = w[:, 0]
            else:
                w1 = w.reshape(-1)
            if od == 0:
                return _as_1P(self.base.vjp(base_cache, w1 * value_scale, out_dim=0))
            if od <= n_axes:
                axis_pos = od - 1
                return self._gradient_vjp_axis(
                    base_cache, int(self.axes[axis_pos]), w1 * grad_scales[axis_pos],
                )
            pair_pos = od - 1 - n_axes
            if pair_pos < 0 or pair_pos >= self._n_hess:
                raise IndexError(f"out_dim={out_dim} out of range for Sobolev O={self.n_outputs}")
            if self._hess_trace:
                wk = w1 * hess_scales[0]
                return sum(self._hessian_vjp_pair(base_cache, int(a), int(a), wk) for a in self.hess_trace_dims)
            a, b = self.hess_pairs[pair_pos]
            return self._hessian_vjp_pair(base_cache, int(a), int(b), w1 * hess_scales[pair_pos])

        if w.ndim == 1:
            w = w.reshape(x.shape[0], self.n_outputs)
        if w.ndim != 2 or int(w.shape[1]) != self.n_outputs:
            raise ValueError(f"w must have shape (B,{self.n_outputs}), got {tuple(w.shape)}")
        rows = [_as_1P(self.base.vjp(base_cache, w[:, 0] * value_scale, out_dim=0))]
        for k, axis in enumerate(self.axes):
            rows.append(
                self._gradient_vjp_axis(
                    base_cache, int(axis), w[:, 1 + k] * grad_scales[k],
                )
            )
        if self._hess_trace:
            # one Laplacian channel: J_Δ^T w = Σ_a ∂(∂²f/∂x_a²)/∂θ^T w
            wk = w[:, 1 + n_axes] * hess_scales[0]
            rows.append(sum(self._hessian_vjp_pair(base_cache, int(a), int(a), wk) for a in self.hess_trace_dims))
        else:
            for k, (a, b) in enumerate(self.hess_pairs):
                rows.append(
                    self._hessian_vjp_pair(
                        base_cache, int(a), int(b), w[:, 1 + n_axes + k] * hess_scales[k],
                    )
                )
        return torch.cat(rows, dim=0)

    def _hessian_jvp(self, base_cache, v: torch.Tensor) -> torch.Tensor:
        """∂(residual)/∂θ·v for the Hessian block as (B, Nx, Nx).

        The H² residual is ``target - ∂²f``, so its parameter derivative is
        ``-∂(∂²f)/∂θ`` -- the same residual sign as the value/gradient blocks
        (``base.jvp``/``base.grad_jvp`` already return residual-signed products,
        i.e. ``-∂pred/∂θ``).  The composite ``grad_grad_jvp`` returns the FORWARD
        Hessian jvp ``+∂(∂²f)/∂θ·v``, so we negate here to stay consistent with
        ``jacobian()`` (whose hess block is ``-∂H/∂θ``).  Without this the dense
        and matrix-free LM paths would disagree in sign on the curvature rows."""
        fn = getattr(self.base, "grad_grad_jvp", None)
        if not callable(fn):
            raise RuntimeError("base provider has no grad_grad_jvp (needed for H^2 residual)")
        return -_as_BNN(fn(base_cache, v, out_dim=0))

    def _hessian_vjp_pair(self, base_cache, a: int, b: int, weights: torch.Tensor) -> torch.Tensor:
        """∂(residual)/∂θ^T·weights for ∂²f/∂x_a∂x_b as (1, P).

        Negated for the same residual-sign reason as :meth:`_hessian_jvp` (keeps
        the jvp/vjp adjoint pair and matches the residual-signed dense Jacobian)."""
        fn = getattr(self.base, "grad_grad_vjp", None)
        if not callable(fn):
            raise RuntimeError("base provider has no grad_grad_vjp (needed for H^2 residual)")
        x = base_cache["x"]
        Nx = int(x.shape[1])
        W = x.new_zeros(x.shape[0], 1, Nx, Nx)
        W[:, 0, int(a), int(b)] = weights.reshape(-1)
        return -_as_1P(fn(base_cache, W, out_dim=0))

    def _gradient_jvp(self, base_cache, v: torch.Tensor) -> torch.Tensor:
        fn = getattr(self.base, "grad_jvp", None)
        if callable(fn) and self._base_grad_jvp_supported is not False:
            try:
                out = _as_BD(fn(base_cache, v, out_dim=0))
                self._base_grad_jvp_supported = True
                return out
            except Exception:
                self._base_grad_jvp_supported = False
        Jg = self._gradient_jacobian_residual(base_cache)
        if Jg is None:
            raise RuntimeError("base provider has no usable grad_jvp or gradient Jacobian")
        return torch.einsum("bdp,p->bd", Jg, v)

    def _gradient_vjp_axis(self, base_cache, axis: int, weights: torch.Tensor) -> torch.Tensor:
        x = base_cache["x"]
        fn = getattr(self.base, "grad_vjp", None)
        if callable(fn) and self._base_grad_vjp_supported is not False:
            adj = x.new_zeros(x.shape[0], x.shape[1])
            adj[:, int(axis)] = weights
            try:
                out = _as_1P(fn(base_cache, adj, out_dim=0))
                self._base_grad_vjp_supported = True
                return out
            except Exception:
                self._base_grad_vjp_supported = False
        Jg = self._gradient_jacobian_residual(base_cache)
        if Jg is None:
            raise RuntimeError("base provider has no usable grad_vjp or gradient Jacobian")
        return torch.einsum("b,bp->p", weights.reshape(-1), Jg[:, int(axis), :]).reshape(1, -1)

    def jacobian(self, cache, out_dim: int | None = None) -> torch.Tensor:
        fast = self._jacobian_fast(cache, out_dim=out_dim)
        if fast is not None:
            return fast
        P = int(self.num_parameters())
        eye = torch.eye(P, device=cache["x"].device, dtype=cache["x"].dtype)
        cols = []
        for j in range(P):
            col = self.jvp(cache, eye[j], out_dim=out_dim)
            if out_dim is not None:
                col = col.reshape(-1, 1)
            cols.append(col)
        if not cols:
            B = int(cache["x"].shape[0])
            O = 1 if out_dim is not None else self.n_outputs
            return cache["x"].new_zeros(B, O, 0)
        J = torch.stack(cols, dim=-1)
        if out_dim is not None:
            return J
        return J

    def _gradient_jacobian_residual(self, base_cache) -> torch.Tensor | None:
        cached = base_cache.get("_sobolev_gradient_jacobian_residual", None)
        if cached is not None:
            return cached
        leaves = base_cache.get("leaves", None)
        base_leaves = getattr(self.base, "leaf", None)
        if leaves is None:
            transparent_leaf = getattr(self.base, "_transparent_identity_leaf", None)
            transparent_leaf = (
                transparent_leaf() if callable(transparent_leaf) else None
            )
            if transparent_leaf is not None:
                # An exact one-atom AST now preserves the provider's native
                # cache rather than wrapping it in ``cache["leaves"]``.
                leaves = [base_cache]
                base_leaves = [transparent_leaf]
        if leaves is None or base_leaves is None:
            return None
        try:
            leaves_list = list(base_leaves)
        except Exception:
            return None
        if len(leaves) != 1 or len(leaves_list) != 1:
            return None
        leaf = leaves_list[0]
        leaf_cache = leaves[0]
        fn = getattr(leaf, "grad_analytic_jacobian_from_cache", None)
        if not callable(fn):
            fn = getattr(getattr(leaf, "base_model", None), "grad_analytic_jacobian_from_cache", None)
        if callable(fn):
            try:
                Jf = fn(leaf_cache, out_dim=0)
            except Exception:
                Jf = None
        else:
            Jf = None
        if Jf is None:
            fn_x = getattr(leaf, "grad_analytic_jacobian", None)
            if not callable(fn_x):
                return None
            try:
                Jf = fn_x(leaf_cache["x"], segments=base_cache.get("segments", None))
            except TypeError:
                try:
                    Jf = fn_x(leaf_cache["x"], base_cache.get("segments", None))
                except Exception:
                    return None
            except Exception:
                return None
        if Jf.ndim == 4 and int(Jf.shape[1]) == 1:
            Jf = Jf[:, 0]
        if Jf.ndim != 3:
            return None
        if int(Jf.shape[-1]) != int(self.num_parameters()):
            return None
        Jg = -Jf
        base_cache["_sobolev_gradient_jacobian_residual"] = Jg
        return Jg

    def _jacobian_fast(self, cache, out_dim: int | None = None) -> torch.Tensor | None:
        # H^2 rows: the fast dense path handles the all-diagonal case via the
        # leaf's selected-diagonal Jacobian (no Nx×Nx Hessian).  Off-diagonal
        # pairs are not supported here -> fall back to the generic jvp jacobian.
        if self._has_hess and not self._hess_all_diag:
            return None
        x = cache["x"]
        value_scale, grad_scales, hess_scales = self._scales(device=x.device, dtype=x.dtype)
        base_cache = cache["base"]
        n_axes = len(self.axes)

        if out_dim is not None:
            od = int(out_dim)
            if od == 0:
                Jv = self.base.jacobian(base_cache, out_dim=0)
                if Jv.ndim == 2:
                    Jv = Jv.unsqueeze(1)
                return Jv * value_scale
            if od <= n_axes:
                axis_pos = od - 1
                Jg = self._gradient_jacobian_residual(base_cache)
                if Jg is None:
                    return None
                return Jg[:, int(self.axes[axis_pos]) : int(self.axes[axis_pos]) + 1, :] * grad_scales[
                    axis_pos
                ]
            pair_pos = od - 1 - n_axes
            if pair_pos < 0 or pair_pos >= self._n_hess:
                raise IndexError(f"out_dim={out_dim} out of range for Sobolev O={self.n_outputs}")
            Jh = self._hess_diag_jacobian_residual(base_cache)
            if Jh is None:
                return None
            return Jh[:, pair_pos : pair_pos + 1, :] * hess_scales[pair_pos]

        Jv = self.base.jacobian(base_cache, out_dim=0)
        if Jv.ndim == 2:
            Jv = Jv.unsqueeze(1)
        Jv = Jv * value_scale
        Jg = self._gradient_jacobian_residual(base_cache)
        if Jg is None:
            return None
        Jg = Jg[:, list(self.axes), :] * grad_scales.view(1, -1, 1)
        blocks = [Jv, Jg]
        if self._has_hess:
            Jh = self._hess_diag_jacobian_residual(base_cache)
            if Jh is None:
                return None
            blocks.append(Jh * hess_scales.view(1, -1, 1))
        return torch.cat(blocks, dim=1)

    def dense_jacobian(self, cache, out_dim: int | None = None) -> torch.Tensor:
        return self.jacobian(cache, out_dim=out_dim)

    def _weighted_jacobian(self, cache, out_dim: int | None = None) -> torch.Tensor:
        J = self.jacobian(cache, out_dim=out_dim)
        sw = cache.get("sample_weights", None)
        if sw is None:
            return J
        w = torch.as_tensor(sw, device=J.device, dtype=J.dtype)
        if out_dim is not None:
            if w.ndim == 2:
                w = w[:, int(out_dim)]
            w = w.reshape(-1, 1, 1)
        else:
            if w.ndim == 1:
                w = w.reshape(-1, 1, 1)
            elif w.ndim == 2:
                w = w.unsqueeze(-1)
            else:
                w = w.reshape(*J.shape[:2], 1)
        return J * torch.sqrt(w.clamp_min(torch.finfo(J.dtype).tiny))

    def diag(self, cache, out_dim: int | None = None) -> torch.Tensor:
        J = self._weighted_jacobian(cache, out_dim=out_dim)
        if out_dim is not None:
            return J.reshape(-1, J.shape[-1]).square().sum(dim=0)
        return J.square().sum(dim=0)

    def dense(self, cache, out_dim: int | None = None) -> torch.Tensor:
        J = self._weighted_jacobian(cache, out_dim=out_dim)
        if out_dim is not None:
            J2 = J.reshape(-1, J.shape[-1])
            return J2.t().matmul(J2)
        return torch.einsum("bop,boq->opq", J, J)

    def dense_hessian(self, *_, **__):
        return None

    def pre_block(self, *args, **kwargs):
        pre = getattr(self.base, "pre_block", None)
        if callable(pre):
            return pre(*args, **kwargs)
        return None

    def pre_block_hook(self, *args, **kwargs):
        hook = getattr(self.base, "pre_block_hook", None)
        if callable(hook):
            return hook(*args, **kwargs)
        return None

    def blocks(self, *args, **kwargs):
        for blk in self.base.blocks(*args, **kwargs):
            out = dict(blk)
            ref = out.get("analytic_map", out.get("global_map", None))
            if isinstance(ref, slice):
                n = int(ref.stop) - int(ref.start)
                dev = self._value_scale.device
                out["dimension_map"] = torch.full((n,), -1, dtype=torch.long, device=dev)
            elif torch.is_tensor(ref):
                out["dimension_map"] = torch.full_like(ref.to(dtype=torch.long), -1)
            else:
                out["dimension_map"] = torch.full((self.num_parameters(),), -1, dtype=torch.long)
            out["owner"] = id(self)
            if "param_idx" not in out:
                out["param_idx"] = torch.arange(
                    self.num_parameters(),
                    device=out["dimension_map"].device,
                    dtype=torch.long,
                )
            yield out
