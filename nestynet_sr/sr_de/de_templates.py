# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
DE Template System for Variable Projection.

This module defines template families for discovering DEs with nonlinear
shape parameters (exponents, frequencies, rates, etc.) that cannot be
expressed in STLSQ's linear library.

Templates enable discovery of equations like:
- Power laws: u_x = k*u^p (exponent p)
- Exponentials: u_x = exp(k*x) (rate k)
- Sinusoids: u_xx + ω^2*sin(ω*x) = 0 (frequency ω)
- Saturation: u_x = u*(1 - u/K) (half-saturation K)

Each template provides:
1. AST construction with parameterized nodes
2. Heuristic parameter initialization (FFT, log-log regression, etc.)
3. Parameter bounds for LM optimization
4. Canonicalization to enforce non-redundancy
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from nestynet_sr.sr_core.constants import make_unit_aware_scalar_atom
from nestynet_sr.sr_core.bridges import (
    ExpNode,
    Mul,
    Node,
    Pow,
    U,
    Var,
)


def _template_param(name: str, init: float = 1.0) -> Node:
    """Create a template nonlinear scalar parameter node.

    DE template shape parameters (e.g. exponents/rates) are dimensionless in
    this pipeline. We route through the shared scalar-constant framework so
    DE/SR parameter node conventions stay aligned.
    """
    return make_unit_aware_scalar_atom(
        required_dim=None,
        units_spec=None,
        base_tag=str(name),
        init=float(init),
        strict=False,
    )


@dataclass
class TemplateInstance:
    """A concrete instance of a template with specific parameter values.

    Parameters
    ----------
    template_name : str
        Template family name (e.g., 'power', 'exp', 'sin')
    ast : Node
        AST node representing the template expression
    params : Dict[str, float]
        Nonlinear shape parameters (e.g., {'p': 2.0})
    param_bounds : Dict[str, Tuple[float, float]]
        LM optimization bounds for each parameter
    description : str
        Human-readable description
    """

    template_name: str
    ast: Node
    params: Dict[str, float]
    param_bounds: Dict[str, Tuple[float, float]]
    description: str


class DETemplate(ABC):
    """Base class for DE template families with nonlinear shape parameters.

    A template family defines a parameterized function form (e.g., u^p, exp(k*x))
    where the shape parameters (p, k, ω, etc.) are optimized via LM while
    linear coefficients are analytically eliminated via VarPro.

    Subclasses must implement:
    - build_instances(): Generate template instances for different variables
    - init_params(): Heuristic initialization from data
    - canonicalize(): Enforce parameter conventions (ω > 0, etc.)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Template family name (e.g., 'power', 'exp', 'sin')."""
        pass

    @abstractmethod
    def build_instances(
        self,
        *,
        x_vars: List[int],
        include_u: bool = True,
        include_du: bool = False,
        x_axis: int = 0,
        **kwargs,
    ) -> List[TemplateInstance]:
        """Build template instances for available variables.

        Parameters
        ----------
        x_vars : List[int]
            Available x variable indices
        include_u : bool
            Include templates involving u
        include_du : bool
            Include templates involving du
        x_axis : int
            DE derivative axis
        **kwargs
            Template-specific options

        Returns
        -------
        List[TemplateInstance]
            Template instances with initialized parameters
        """
        pass

    @abstractmethod
    def init_params(
        self,
        instance: TemplateInstance,
        x: torch.Tensor,
        u: torch.Tensor,
        du: Optional[torch.Tensor],
        target: torch.Tensor,
    ) -> Dict[str, float]:
        """Initialize template parameters from data.

        Uses heuristics like:
        - Power laws: log-log linear regression
        - Exponentials: log-linear regression
        - Sinusoids: FFT peak detection
        - Saturation: inflection point analysis

        Parameters
        ----------
        instance : TemplateInstance
            Template instance to initialize
        x : torch.Tensor
            Input data (N, Nx)
        u : torch.Tensor
            Surrogate values (N, 1)
        du : torch.Tensor, optional
            Surrogate derivatives (N, 1)
        target : torch.Tensor
            Target residual values (N,)

        Returns
        -------
        Dict[str, float]
            Initialized parameter values
        """
        pass

    @abstractmethod
    def canonicalize(self, params: Dict[str, float]) -> Dict[str, float]:
        """Enforce canonical parameter form.

        Removes redundancy and enforces conventions:
        - Exponents: |p| to avoid sign ambiguity
        - Frequencies: ω > 0
        - No additive constants in exponent/argument

        Parameters
        ----------
        params : Dict[str, float]
            Raw parameter values

        Returns
        -------
        Dict[str, float]
            Canonical parameter values
        """
        pass


class PowerLawTemplate(DETemplate):
    """Power law template: f(x) = x^p or u^p.

    Discovers equations like:
    - u_x = k*u^2 (logistic growth, p=2)
    - u_x = k*u^(-1) (reciprocal, p=-1)
    - u_x = k*sqrt(u) (square root, p=0.5)
    - u_x = k*(x*u)^p (composite power, p arbitrary)

    Parameters
    ----------
    p : float
        Exponent parameter

    Initialization
    --------------
    For u^p: log-log regression on |u| vs |target|
    For x^p: log-log regression on |x| vs |target|

    Bounds
    ------
    p ∈ [-5, 5] (avoid extreme exponents)

    Canonicalization
    ----------------
    Use |p| to avoid sign ambiguity (coefficient absorbs sign)
    """

    @property
    def name(self) -> str:
        return "power"

    def build_instances(
        self,
        *,
        x_vars: List[int],
        include_u: bool = True,
        include_du: bool = False,
        x_axis: int = 0,
        **kwargs,
    ) -> List[TemplateInstance]:
        """Build power law instances.

        Generates:
        - u^p (if include_u)
        - x_j^p for each x variable
        - (x_j * u)^p (if include_u)
        """
        instances = []

        # u^p
        if include_u:
            instances.append(
                TemplateInstance(
                    template_name=self.name,
                    ast=Pow(U(), _template_param("p", init=1.0)),
                    params={"p": 1.0},  # Will be initialized from data
                    param_bounds={"p": (-5.0, 5.0)},
                    description="u^p",
                )
            )

        # x_j^p for each x variable
        for j in x_vars:
            param_name = f"p_x{j}"
            instances.append(
                TemplateInstance(
                    template_name=self.name,
                    ast=Pow(Var(j), _template_param(param_name, init=1.0)),
                    params={param_name: 1.0},
                    param_bounds={param_name: (-5.0, 5.0)},
                    description=f"x{j}^p",
                )
            )

        # (x_j * u)^p
        if include_u:
            for j in x_vars:
                param_name = f"p_x{j}u"
                instances.append(
                    TemplateInstance(
                        template_name=self.name,
                        ast=Pow(Mul(Var(j), U()), _template_param(param_name, init=1.0)),
                        params={param_name: 1.0},
                        param_bounds={param_name: (-5.0, 5.0)},
                        description=f"(x{j}*u)^p",
                    )
                )

        return instances

    def init_params(
        self,
        instance: TemplateInstance,
        x: torch.Tensor,
        u: torch.Tensor,
        du: Optional[torch.Tensor],
        target: torch.Tensor,
    ) -> Dict[str, float]:
        """Initialize exponent via log-log regression.

        For u^p: log|target| ≈ p*log|u| + const
        Fit slope via least squares to get initial p.
        """
        params = {}

        # Determine base variable from description
        desc = instance.description

        if "u^p" in desc:
            # Power of u
            base = u[:, 0].abs() + 1e-10  # Avoid log(0)
        elif "(x" in desc and "*u)^p" in desc:
            # Power of x*u
            # Extract x index from description like "(x0*u)^p"
            import re

            x_match = re.search(r"x(\d+)", desc)
            if x_match:
                x_idx = int(x_match.group(1))
                base = (x[:, x_idx].abs() * u[:, 0].abs()) + 1e-10
            else:
                # Default
                param_name = list(instance.params.keys())[0]
                params[param_name] = 1.0
                return params
        elif "x" in desc:
            # Power of x alone
            # Extract x index from description like "x0^p"
            import re

            x_match = re.search(r"x(\d+)", desc)
            if x_match:
                x_idx = int(x_match.group(1))
                base = x[:, x_idx].abs() + 1e-10
            else:
                # Default
                param_name = list(instance.params.keys())[0]
                params[param_name] = 1.0
                return params
        else:
            # Default: just use p=1.0
            param_name = list(instance.params.keys())[0]
            params[param_name] = 1.0
            return params

        # Log-log regression
        target_abs = target.abs() + 1e-10
        log_base = torch.log(base)
        log_target = torch.log(target_abs)

        # Remove NaN/inf
        mask = torch.isfinite(log_base) & torch.isfinite(log_target)
        if mask.sum() < 10:
            # Not enough valid points, use default
            param_name = list(instance.params.keys())[0]
            params[param_name] = 1.0
            return params

        log_base_valid = log_base[mask]
        log_target_valid = log_target[mask]

        # Least squares: p = (log_base^T log_target) / (log_base^T log_base)
        p_init = float(
            (log_base_valid * log_target_valid).sum() / (log_base_valid * log_base_valid).sum()
        )

        # Clamp to bounds
        p_init = np.clip(p_init, -5.0, 5.0)

        # If p is very close to integer, snap to it
        if abs(p_init - round(p_init)) < 0.1:
            p_init = float(round(p_init))

        param_name = list(instance.params.keys())[0]
        params[param_name] = p_init

        return params

    def canonicalize(self, params: Dict[str, float]) -> Dict[str, float]:
        """Canonicalize power law parameters.

        Keep the sign of p (u^2 and u^-2 are fundamentally different).
        Snap to nearby integers/rationals for common cases.
        """
        canonical = {}
        for key, val in params.items():
            # Snap to common integer/rational values
            for target in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, -0.5, -1.0, -2.0]:
                if abs(val - target) < 0.1:
                    val = target
                    break
            canonical[key] = val
        return canonical


class ExponentialTemplate(DETemplate):
    """Exponential template: exp(k*x) or exp(k*x)*u^p.

    Discovers equations like:
    - u_x = a*exp(k*x) (exponential growth/decay)
    - u_x = a*exp(k*x)*u (Gompertz-like)
    - u_xx = exp(k*x) (exponential forcing)

    Parameters
    ----------
    k : float
        Rate parameter in exponent
    p : float, optional
        Damping exponent if combined with u^p

    Initialization
    --------------
    log|target| ≈ k*x + const (for pure exp)
    Linear regression on log scale

    Bounds
    ------
    k ∈ [-10, 10] (avoid extreme growth/decay)
    p ∈ [-3, 3] (if damping included)

    Canonicalization
    ----------------
    No additive constant in exponent (absorbed by coefficient)
    """

    @property
    def name(self) -> str:
        return "exp"

    def build_instances(
        self,
        *,
        x_vars: List[int],
        include_u: bool = True,
        include_du: bool = False,
        x_axis: int = 0,
        **kwargs,
    ) -> List[TemplateInstance]:
        """Build exponential instances."""
        instances = []

        # exp(k*x_j) for each x variable
        for j in x_vars:
            k_name = f"k_x{j}"
            instances.append(
                TemplateInstance(
                    template_name=self.name,
                    ast=ExpNode(Mul(_template_param(k_name, init=0.1), Var(j))),
                    params={k_name: 0.1},
                    param_bounds={k_name: (-10.0, 10.0)},
                    description=f"exp(k*x{j})",
                )
            )

        # exp(k*x_j) * u^p (damped exponential)
        if include_u:
            for j in x_vars:
                k_name = f"k_x{j}_u"
                p_name = f"p_exp_x{j}"
                instances.append(
                    TemplateInstance(
                        template_name=self.name,
                        ast=Mul(
                            ExpNode(Mul(_template_param(k_name, init=0.1), Var(j))),
                            Pow(U(), _template_param(p_name, init=1.0)),
                        ),
                        params={k_name: 0.1, p_name: 1.0},
                        param_bounds={k_name: (-10.0, 10.0), p_name: (-3.0, 3.0)},
                        description=f"exp(k*x{j})*u^p",
                    )
                )

        return instances

    def init_params(
        self,
        instance: TemplateInstance,
        x: torch.Tensor,
        u: torch.Tensor,
        du: Optional[torch.Tensor],
        target: torch.Tensor,
    ) -> Dict[str, float]:
        """Initialize rate via log-linear regression.

        For exp(k*x): log|target| ≈ k*x + const
        """
        params = {}
        desc = instance.description

        # Extract x index - look for 'x' followed by digit
        import re

        x_match = re.search(r"x(\d+)", desc)
        if x_match:
            x_idx = int(x_match.group(1))
        else:
            # Default initialization
            for key in instance.params.keys():
                params[key] = instance.params[key]
            return params

        x_vals = x[:, x_idx]
        target_abs = target.abs() + 1e-10
        log_target = torch.log(target_abs)

        # Remove NaN/inf
        mask = torch.isfinite(log_target) & torch.isfinite(x_vals)
        if mask.sum() < 10:
            for key in instance.params.keys():
                params[key] = instance.params[key]
            return params

        x_valid = x_vals[mask]
        log_target_valid = log_target[mask]

        # Linear regression: log(target) = k*x + b
        # k = cov(x, log_target) / var(x)
        x_mean = x_valid.mean()
        log_t_mean = log_target_valid.mean()
        cov = ((x_valid - x_mean) * (log_target_valid - log_t_mean)).mean()
        var_x = ((x_valid - x_mean) ** 2).mean()

        if var_x > 1e-10:
            k_init = float(cov / var_x)
        else:
            k_init = 0.1

        # Clamp to bounds
        k_init = np.clip(k_init, -10.0, 10.0)

        # Assign to parameter
        if "u^p" in desc:
            # Damped exponential: initialize both k and p
            k_key = [k for k in instance.params.keys() if k.startswith("k_")][0]
            p_key = [k for k in instance.params.keys() if k.startswith("p_")][0]
            params[k_key] = k_init
            params[p_key] = 1.0  # Default damping
        else:
            # Pure exponential
            k_key = list(instance.params.keys())[0]
            params[k_key] = k_init

        return params

    def canonicalize(self, params: Dict[str, float]) -> Dict[str, float]:
        """No modification needed for exponential (no redundancy)."""
        return params.copy()


# Registry of available templates
TEMPLATE_REGISTRY: Dict[str, DETemplate] = {
    "power": PowerLawTemplate(),
    "exp": ExponentialTemplate(),
    # 'sin': SinusoidTemplate(),  # TODO: implement
    # 'saturation': SaturationTemplate(),  # TODO: implement
}


def get_template(name: str) -> DETemplate:
    """Get template by name from registry."""
    if name not in TEMPLATE_REGISTRY:
        raise ValueError(f"Unknown template '{name}'. Available: {list(TEMPLATE_REGISTRY.keys())}")
    return TEMPLATE_REGISTRY[name]
