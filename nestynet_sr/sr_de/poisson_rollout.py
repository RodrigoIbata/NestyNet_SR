# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""General autonomous vector-field rollout for Poisson model validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import torch
from scipy.integrate import solve_ivp


class _ValueField(Protocol):
    def value(self, z: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class VectorRolloutResult:
    """Numerical rollout and optional trajectory-error diagnostics."""

    times: np.ndarray
    states: np.ndarray
    success: bool
    message: str
    rms_error: float | None = None
    relative_rms_error: float | None = None


def rollout_vector_field(
    field: _ValueField | Callable[[torch.Tensor], torch.Tensor],
    initial_state: np.ndarray | torch.Tensor,
    times: np.ndarray | torch.Tensor,
    *,
    reference_states: np.ndarray | torch.Tensor | None = None,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-11,
    method: str = "DOP853",
) -> VectorRolloutResult:
    """Integrate an arbitrary-dimensional autonomous field on requested times."""

    z0 = np.asarray(_to_numpy(initial_state), dtype=np.float64).reshape(-1)
    t_eval = np.asarray(_to_numpy(times), dtype=np.float64).reshape(-1)
    if z0.size == 0:
        raise ValueError("initial_state must be non-empty")
    if t_eval.size < 2 or not np.all(np.diff(t_eval) > 0.0):
        raise ValueError("times must contain at least two strictly increasing values")

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        z = torch.as_tensor(state, dtype=torch.float64).reshape(1, -1)
        with torch.no_grad():
            value = field.value(z) if hasattr(field, "value") else field(z)
        out = np.asarray(_to_numpy(value), dtype=np.float64).reshape(-1)
        if out.shape != state.shape:
            raise ValueError(f"field returned shape {out.shape}, expected {state.shape}")
        return out

    solution = solve_ivp(
        rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        z0,
        t_eval=t_eval,
        rtol=float(rtol),
        atol=float(atol),
        method=str(method),
    )
    states = np.asarray(solution.y.T, dtype=np.float64)
    rms = None
    relative = None
    if reference_states is not None and states.shape[0] == t_eval.size:
        reference = np.asarray(_to_numpy(reference_states), dtype=np.float64)
        if reference.shape != states.shape:
            raise ValueError(f"reference_states has shape {reference.shape}, expected {states.shape}")
        difference = states - reference
        rms = float(np.sqrt(np.mean(difference * difference)))
        scale = max(float(np.sqrt(np.mean(reference * reference))), 1.0e-15)
        relative = rms / scale
    return VectorRolloutResult(
        times=t_eval,
        states=states,
        success=bool(solution.success),
        message=str(solution.message),
        rms_error=rms,
        relative_rms_error=relative,
    )


def _to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


__all__ = ["VectorRolloutResult", "rollout_vector_field"]
