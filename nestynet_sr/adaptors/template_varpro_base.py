# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Template VarPro Base Module.

This module provides a PyTorch module that wraps DE templates with nonlinear
parameters (k, p, ω, etc.) for Variable Projection optimization.

The key insight: treat templates as a "base" nonlinear model ψ → Φ(x, u, du; ψ)
where ψ are the shape parameters that get optimized via LM, and linear
coefficients are analytically eliminated via VarPro.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from nestynet_sr.adaptors.u_feature_leaf import UFeatureCache
from nestynet_sr.sr_core.bridges import ConstNode, Node, const_full_like
from nestynet_sr.sr_de.de_templates import TemplateInstance


class TemplateVarProBase(nn.Module):
    """Base module for template features with nonlinear parameters.

    This module computes Φ(x, u, du; ψ) where:
    - x: input coordinates
    - u, du: surrogate field and derivatives from UFeatureCache
    - ψ: nonlinear shape parameters (stored as nn.Parameters)

    The output is a feature matrix (N, num_templates) where each column
    is one template's evaluation.

    Parameters
    ----------
    template_instances : List[TemplateInstance]
        List of template instances with initial parameters
    cache : UFeatureCache
        Cache for surrogate u(x) and derivatives
    order : int
        ODE order (1 or 2)
    x_axis : int
        Derivative axis for the ODE

    Example
    -------
    >>> # Create template instances (from ExponentialTemplate, etc.)
    >>> templates = [
    ...     TemplateInstance(
    ...         template_name='exp',
    ...         ast=ExpNode(Mul(Scale('k_x0', tag='k_x0', init=1.0), Var(0))),
    ...         params={'k_x0': 1.0},
    ...         param_bounds={'k_x0': (-10.0, 10.0)},
    ...         description='exp(k*x0)'
    ...     )
    ... ]
    >>> base = TemplateVarProBase(templates, cache, order=1, x_axis=0)
    >>> features = base(x_batch)  # (N, 1) - one template
    """

    def __init__(
        self,
        template_instances: List[TemplateInstance],
        cache: UFeatureCache,
        order: int,
        x_axis: int,
    ):
        super().__init__()
        self.template_instances = template_instances
        self.cache = cache
        self.order = order
        self.x_axis = x_axis

        # Extract all unique parameter names and create nn.Parameters
        # Store as dict for easy access
        self.param_dict = nn.ParameterDict()
        self.param_to_templates: Dict[str, List[int]] = {}  # which templates use which params

        for idx, tmpl in enumerate(template_instances):
            for param_name, param_value in tmpl.params.items():
                if param_name not in self.param_dict:
                    # Create new parameter
                    self.param_dict[param_name] = nn.Parameter(
                        torch.tensor(param_value, dtype=torch.float64)
                    )
                    self.param_to_templates[param_name] = []

                self.param_to_templates[param_name].append(idx)

        # Store parameter bounds for LM (not used directly by PyTorch)
        self.param_bounds: Dict[str, tuple[float, float]] = {}
        for tmpl in template_instances:
            self.param_bounds.update(tmpl.param_bounds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate all template features Φ(x, u, du; ψ).

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates (N, Nx)

        Returns
        -------
        features : torch.Tensor
            Feature matrix (N, num_templates)
        """
        # Ensure cache is populated
        # For 2nd-order ODEs, we need both gradients and Hessians because
        # baseline terms may contain first derivatives (e.g., x*u_x)
        need_grad = True  # Always compute gradients (needed for baseline terms)
        need_hess = self.order == 2
        self.cache.ensure(x, need_grad=need_grad, need_hess=need_hess)

        # Get u and derivatives
        u = self.cache.u  # (N, 1)
        # For all cases, make du available (baseline terms might use it)
        du = self.cache.g[:, 0, self.x_axis : self.x_axis + 1]  # (N, 1)

        # For 2nd-order ODEs, also get d2u
        d2u = None
        if self.order == 2:
            d2u = self.cache.H[:, 0, self.x_axis, self.x_axis].unsqueeze(1)  # (N, 1)

        # Evaluate each template
        features = []
        for tmpl in self.template_instances:
            # Get current parameter values
            current_params = {name: self.param_dict[name] for name in tmpl.params.keys()}

            # Evaluate template AST with current parameters
            # This requires a custom evaluation function that handles FreeConst nodes
            feat_val = self._eval_template_ast(tmpl.ast, x, u, du, d2u, current_params)
            features.append(feat_val)

        # Stack into feature matrix (N, K)
        features_matrix = torch.stack(features, dim=1)  # (N, num_templates)

        return features_matrix

    def _eval_template_ast(
        self,
        node: Node,
        x: torch.Tensor,
        u: torch.Tensor,
        du: Optional[torch.Tensor],
        d2u: Optional[torch.Tensor],
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate template AST with current parameter values.

        This is similar to _eval_ast from de_search.py but handles
        scalar parameter nodes (``free_const``/``scale``) by looking them up
        in the params dict.

        Parameters
        ----------
        node : Node
            AST node to evaluate
        x : torch.Tensor
            Input coordinates (N, Nx)
        u : torch.Tensor
            Surrogate values (N, 1)
        du : torch.Tensor, optional
            Surrogate derivatives (N, 1)
        d2u : torch.Tensor, optional
            Surrogate second derivatives (N, 1)
        params : Dict[str, torch.Tensor]
            Current parameter values

        Returns
        -------
        values : torch.Tensor
            Evaluated values (N,) or (N, 1)
        """
        from nestynet_sr.sr_core.bridges import (
            AddNode,
            AtomNode,
            CosNode,
            ExpNode,
            MulNode,
            PowNode,
            SinNode,
        )

        # Handle scalar values (int, float) - convert to constant tensor
        if isinstance(node, (int, float)):
            return torch.full((x.shape[0],), float(node), device=x.device, dtype=x.dtype)

        if node is None:
            # Constant 1
            return torch.ones(x.shape[0], device=x.device, dtype=x.dtype)

        if isinstance(node, AtomNode):
            kind = str(getattr(node, "kind", "")).lower()
            var_idxs = getattr(node, "var_idxs", ())
            kwargs = getattr(node, "kwargs", {})
            tag = getattr(node, "tag", None)

            # Trainable scalar parameters: look up in params.
            if kind in ("free_const", "freeconst", "free_constant", "scale"):
                # Try tag first, then name from kwargs
                if tag and tag in params:
                    return params[tag].expand(x.shape[0])
                name = kwargs.get("name", None)
                if name and name in params:
                    return params[name].expand(x.shape[0])
                # Fallback to init value
                return torch.full(
                    (x.shape[0],), kwargs.get("init", 1.0), device=x.device, dtype=x.dtype
                )

            # Fixed scalar constants
            if kind in ("fixed_const", "fixedconst", "fixed_constant"):
                val = float(kwargs.get("value", 1.0))
                return torch.full((x.shape[0],), val, device=x.device, dtype=x.dtype)

            # Constant value
            if kind in ("const", "constant"):
                val = float(kwargs.get("value", 1.0))
                return torch.full((x.shape[0],), val, device=x.device, dtype=x.dtype)

            # Input variable
            if kind in ("var", "x", "input"):
                if len(var_idxs) == 0:
                    raise ValueError("Var node has no var_idxs")
                return x[:, var_idxs[0]]

            # Surrogate u
            if kind in ("u", "field", "state"):
                return u[:, 0]

            # Surrogate derivative du
            if kind in ("du", "d1u", "grad_u"):
                if du is None:
                    raise ValueError("du requested but not computed")
                return du[:, 0]

            # Surrogate second derivative d2u
            if kind in ("d2u", "d2u", "hess_u"):
                if d2u is None:
                    raise ValueError("d2u requested but not computed")
                return d2u[:, 0]

            raise ValueError(f"Unknown atom kind: {kind}")

        # Binary operations
        if isinstance(node, AddNode):
            left = self._eval_template_ast(node.left, x, u, du, d2u, params)
            right = self._eval_template_ast(node.right, x, u, du, d2u, params)
            return left + right

        if isinstance(node, MulNode):
            left = self._eval_template_ast(node.left, x, u, du, d2u, params)
            right = self._eval_template_ast(node.right, x, u, du, d2u, params)
            return left * right

        # Unary operations
        if isinstance(node, ExpNode):
            arg = self._eval_template_ast(node.arg, x, u, du, d2u, params)
            return torch.exp(arg)

        if isinstance(node, SinNode):
            arg = self._eval_template_ast(node.arg, x, u, du, d2u, params)
            return torch.sin(arg)

        if isinstance(node, CosNode):
            arg = self._eval_template_ast(node.arg, x, u, du, d2u, params)
            return torch.cos(arg)

        if isinstance(node, PowNode):
            base_val = self._eval_template_ast(node.base, x, u, du, d2u, params)
            exp_val = self._eval_template_ast(node.exponent, x, u, du, d2u, params)
            # Check if exponent is scalar (from FreeConst)
            if exp_val.numel() == x.shape[0] and (exp_val == exp_val[0]).all():
                # Constant exponent - use scalar power (keep as tensor to preserve gradients)
                return torch.pow(base_val, exp_val[0])
            else:
                # Variable exponent
                return torch.pow(base_val, exp_val)

        if isinstance(node, ConstNode):
            return const_full_like(x, (x.shape[0],), node.value)

        raise ValueError(f"Unknown node type: {type(node)}")

    def share_params_from(self, other: "TemplateVarProBase", *, strict: bool = True) -> None:
        """Share nonlinear parameters ψ from another TemplateVarProBase.

        This is used for multi-dataset VarPro where all datasets share the same
        nonlinear template parameters (e.g., power law exponent p) but have
        different linear coefficients.

        Parameters
        ----------
        other : TemplateVarProBase
            Source module to share parameters from
        strict : bool, default=True
            If True, raises ValueError if parameter names don't match exactly

        Raises
        ------
        ValueError
            If strict=True and parameter names don't match

        Example
        -------
        >>> # Create bases for 3 datasets
        >>> base0 = TemplateVarProBase(templates, cache0, order=1, x_axis=0)
        >>> base1 = TemplateVarProBase(templates, cache1, order=1, x_axis=0)
        >>> base2 = TemplateVarProBase(templates, cache2, order=1, x_axis=0)
        >>>
        >>> # Share parameters from base0 to all others
        >>> base1.share_params_from(base0)
        >>> base2.share_params_from(base0)
        >>>
        >>> # Now optimizing base0.parameters() will update all bases
        """
        my_keys = set(self.param_dict.keys())
        other_keys = set(other.param_dict.keys())

        if strict and my_keys != other_keys:
            raise ValueError(
                f"Parameter name mismatch (strict=True):\n"
                f"  self  has: {sorted(my_keys)}\n"
                f"  other has: {sorted(other_keys)}"
            )

        # Share parameters by replacing our param_dict entries with references to other's
        for key in other.param_dict.keys():
            if key in self.param_dict:
                # Replace our parameter with a reference to the other's parameter
                self.param_dict[key] = other.param_dict[key]
