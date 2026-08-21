# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class VarLeaf(nn.Module):
    """Identity leaf for a single input variable.

    This leaf is intentionally minimal: it allows SR grammars to expose raw
    variables explicitly in the AST (so they can be combined with unitful
    free constants via Mul/Pow, etc.).

    Input:  x: [B, 1]
    Output: y: [B, 1]  (y = x)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to have shape [B, 1], but we slice defensively.
        return x[..., :1]

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        # y = x[..., 0:1]  ⇒  ∂y/∂x_0 = 1, all other entries 0.
        B, K = int(x.shape[0]), int(x.shape[-1]) if x.dim() > 1 else 1
        J = x.new_zeros(B, 1, K)
        if K >= 1:
            J[:, 0, 0] = 1.0
        return J

    def grad_grad(self, x: torch.Tensor) -> torch.Tensor:
        # Affine ⇒ structural zero Hessian.
        B, K = int(x.shape[0]), int(x.shape[-1]) if x.dim() > 1 else 1
        return x.new_zeros(B, 1, K, K)


class FreeConstLeaf(nn.Module):
    """Trainable scalar constant leaf.

    This represents a *single* free constant (one LM-optimised scalar
    parameter) whose physical units are handled by the AST-level dimensional
    checker (sr_core.units). Numerically, this is a scalar broadcasted across
    the batch.

    Input:  x: [B, 0] (ignored; may also be [B, k])
    Output: y: [B, 1] (y = value)
    """

    def __init__(
        self,
        init: float = 1.0,
        *,
        dtype: torch.dtype = torch.float64,
        device=None,
    ):
        super().__init__()
        dev = device or torch.device("cpu")
        self.value = nn.Parameter(torch.tensor(float(init), dtype=dtype, device=dev))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Broadcast across batch dimension.
        B = int(x.shape[0])
        return self.value.expand(B, 1)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        # Constant in x ⇒ zero input-Jacobian. (The wrapper still handles
        # parameter-side derivatives for LM.)
        B = int(x.shape[0])
        K = int(x.shape[-1]) if x.dim() > 1 else 0
        return x.new_zeros(B, 1, K)

    def grad_grad(self, x: torch.Tensor) -> torch.Tensor:
        B = int(x.shape[0])
        K = int(x.shape[-1]) if x.dim() > 1 else 0
        return x.new_zeros(B, 1, K, K)



class FixedConstLeaf(nn.Module):
    """Fixed scalar constant leaf (non-trainable).

    This is used for user-specified physical constants (e.g. c, G, eV)
    when unit checking is enabled. Units are handled structurally by
    sr_core.units via AtomNode(kind="fixed_const", kwargs={name,value}).

    Input:  x: [B, 0] (ignored; may also be [B, k])
    Output: y: [B, 1] (y = value)
    """

    def __init__(self, value: float = 1.0, *, dtype: torch.dtype = torch.float64, device=None):
        super().__init__()
        dev = device or torch.device("cpu")
        self.register_buffer("value", torch.tensor(float(value), dtype=dtype, device=dev))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = int(x.shape[0])
        return self.value.expand(B, 1)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        B = int(x.shape[0])
        K = int(x.shape[-1]) if x.dim() > 1 else 0
        return x.new_zeros(B, 1, K)

    def grad_grad(self, x: torch.Tensor) -> torch.Tensor:
        B = int(x.shape[0])
        K = int(x.shape[-1]) if x.dim() > 1 else 0
        return x.new_zeros(B, 1, K, K)

class LinLeaf(nn.Module):
    """Linear combination leaf with *dimensionless* weights and no bias.

    Numerically: y = x @ w, where w is a trainable vector.
    Units are enforced structurally by the units checker: all inputs must be
    commensurate for the internal sum to make sense.

    Input:  x: [B, n_in]
    Output: y: [B, 1]
    """

    def __init__(self, n_in: int, dtype: torch.dtype = torch.float64, device=None):
        super().__init__()
        dev = device or torch.device("cpu")
        self.weight = nn.Parameter(torch.zeros(int(n_in), dtype=dtype, device=dev))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weight
        return y.unsqueeze(-1)


def _enumerate_exponents(n_in: int, degree: int, *, min_total: int = 0):
    exps = []
    if min_total < 0 or min_total > degree:
        raise ValueError(
            f"min_total must satisfy 0 <= min_total <= degree; got min_total={min_total}, degree={degree}"
        )

    def rec(pos, remaining, cur):
        if pos == n_in:
            if remaining == 0:
                exps.append(tuple(cur))
            return
        for p in range(remaining + 1):
            cur[pos] = p
            rec(pos + 1, remaining - p, cur)

    cur = [0] * n_in
    for total in range(min_total, degree + 1):
        rec(0, total, cur)
    return exps


def _resolve_exponent_table(
    *,
    n_in: int,
    degree: int,
    min_total: int = 0,
    override: Optional[Sequence[Sequence[int]]] = None,
    support_override: Optional[Sequence[int]] = None,
    name: str = "exps",
) -> List[Tuple[int, ...]]:
    """Build or validate a monomial exponent table.

    When ``override`` is provided, this validates each exponent row and keeps
    first-seen ordering (dropping duplicates). Otherwise falls back to dense
    enumeration via :func:`_enumerate_exponents`.
    """
    if (override is not None) and (support_override is not None):
        raise ValueError(
            f"{name}: provide either explicit exponents or support indices, not both"
        )

    if support_override is not None:
        dense = _enumerate_exponents(n_in, degree, min_total=min_total)
        if not dense:
            raise ValueError(f"{name}: dense exponent table is empty")
        uniq_sorted = sorted({int(i) for i in support_override})
        if len(uniq_sorted) == 0:
            raise ValueError(f"{name}: support override must contain at least one index")
        out: List[Tuple[int, ...]] = []
        for j in uniq_sorted:
            if j < 0 or j >= len(dense):
                raise ValueError(
                    f"{name}: support index {j} out of range [0, {len(dense) - 1}]"
                )
            out.append(tuple(int(v) for v in dense[j]))
        return out

    if override is None:
        return _enumerate_exponents(n_in, degree, min_total=min_total)

    out: List[Tuple[int, ...]] = []
    seen = set()
    for row in override:
        exp = tuple(int(v) for v in row)
        if len(exp) != int(n_in):
            raise ValueError(
                f"{name}: expected exponent tuples of length {n_in}, got {len(exp)}"
            )
        if any(v < 0 for v in exp):
            raise ValueError(f"{name}: exponents must be >= 0, got {exp}")
        total = int(sum(exp))
        if total < int(min_total) or total > int(degree):
            raise ValueError(
                f"{name}: total degree {total} not in [{int(min_total)}, {int(degree)}]"
            )
        if exp not in seen:
            seen.add(exp)
            out.append(exp)

    if not out:
        raise ValueError(f"{name}: override must contain at least one monomial")
    return out


def _eval_monomials(x: torch.Tensor, exps: torch.Tensor):
    B, n = x.shape
    M = exps.shape[0]
    out = []
    for k in range(M):
        e = exps[k]
        t = torch.ones(B, dtype=x.dtype, device=x.device)
        for j in range(n):
            p = int(e[j].item())
            if p > 0:
                t = t * (x[:, j] ** p)
        out.append(t)
    return torch.stack(out, dim=1) if out else x.new_zeros(B, 0)


def _lead_pos_from_exps(exps: torch.Tensor, degree: int) -> int:
    """Return a deterministic index of a 'leading' monomial within an exponent table.

    Convention: choose the *last* monomial with total degree == degree.
    If none exists (e.g. degree=0 or deg table is unusual), fall back to the last row.
    """
    if exps is None or exps.numel() == 0:
        return 0
    if exps.ndim != 2:
        return int(exps.shape[0] - 1)
    totals = exps.sum(dim=1)
    idxs = (totals == int(degree)).nonzero(as_tuple=False).view(-1)
    if idxs.numel() == 0:
        return int(exps.shape[0] - 1)
    return int(idxs.max().item())


def _enumerate_monomials(ndim: int, degree: int) -> List[Tuple[int, ...]]:
    if ndim == 1:
        return [(k,) for k in range(degree + 1)]

    exps: List[Tuple[int, ...]] = []

    def rec(dim: int, remaining: int, prefix: List[int]):
        if dim == ndim - 1:
            for last in range(remaining + 1):
                exps.append(tuple(prefix + [last]))
            return
        for k in range(remaining + 1):
            rec(dim + 1, remaining - k, prefix + [k])

    rec(0, degree, [])
    return exps


class SinLinearLeaf(nn.Module):
    def __init__(self, n_in: int, dtype=torch.float64, device=None):
        super().__init__()
        dev = device or torch.device("cpu")
        self.weight = nn.Parameter(torch.zeros(n_in, dtype=dtype, device=dev))
        self.bias = nn.Parameter(torch.zeros((), dtype=dtype, device=dev))
        self.amp = nn.Parameter(torch.ones((), dtype=dtype, device=dev))

    def forward(self, x):
        z = x @ self.weight + self.bias  # [B]
        y = torch.sin(z)
        y = self.amp * y
        return y.unsqueeze(-1)  # [B,1]


class TanhLinearLeaf(nn.Module):
    """A * tanh(w·x + b) leaf.

    Matches common SR patterns where a bounded nonlinearity is applied to a
    (possibly compound) 1D input.
    """

    def __init__(self, n_in: int, dtype=torch.float64, device=None):
        super().__init__()
        dev = device or torch.device("cpu")
        self.weight = nn.Parameter(torch.zeros(n_in, dtype=dtype, device=dev))
        self.bias = nn.Parameter(torch.zeros((), dtype=dtype, device=dev))
        self.amp = nn.Parameter(torch.ones((), dtype=dtype, device=dev))

        # Gentle default init: tanh(x) is a common target.
        with torch.no_grad():
            self.weight.fill_(1.0)
            self.bias.zero_()
            self.amp.fill_(1.0)

    def forward(self, x):
        z = x @ self.weight + self.bias  # [B]
        y = torch.tanh(z)
        y = self.amp * y
        return y.unsqueeze(-1)  # [B,1]


class PowerLeaf(nn.Module):
    def __init__(
        self,
        n_in: int,
        exponent_init: float = 1.0,
        dtype=torch.float64,
        device=None,
        eps: float = 1e-8,
    ):
        super().__init__()
        dev = device or torch.device("cpu")
        assert n_in == 1, "PowerLeaf currently only supports 1D"
        self.eps = eps
        self.exponent = nn.Parameter(torch.tensor(float(exponent_init), dtype=dtype, device=dev))
        self.amp = nn.Parameter(torch.ones((), dtype=dtype, device=dev))

    def forward(self, x):
        z = x[:, 0].clamp_min(self.eps)
        return (self.amp * z.pow(self.exponent)).unsqueeze(-1)


class InverseMonomialLeaf(nn.Module):
    """Computes a / x^degree with fixed degree and learnable amplitude.

    Unlike PowerLeaf which has both amp and exponent as learnable parameters,
    this leaf has only amp as learnable. The degree is fixed at construction.

    Use for inv_monomial_deg1 (a/x) and inv_monomial_deg2 (a/x²).
    """
    def __init__(
        self,
        n_in: int,
        degree: int = 1,
        dtype=torch.float64,
        device=None,
        eps: float = 1e-8,
    ):
        super().__init__()
        assert n_in == 1, "InverseMonomialLeaf only supports 1D"
        self.degree = int(degree)  # Fixed, not learnable
        self.eps = eps
        dev = device or torch.device("cpu")
        self.amp = nn.Parameter(torch.ones((), dtype=dtype, device=dev))

    def forward(self, x):
        z = x[:, 0].clamp_min(self.eps)
        return (self.amp / z.pow(self.degree)).unsqueeze(-1)


class RInverseMonomialLeaf(nn.Module):
    """Reduced (monic) inverse monomial: 1/x^degree with no learnable params.

    Parameter-free counterpart of InverseMonomialLeaf, analogous to
    RPolyLeaf vs PolyLeaf.
    """

    def __init__(
        self,
        n_in: int,
        degree: int = 1,
        dtype=torch.float64,
        device=None,
        eps: float = 1e-8,
    ):
        super().__init__()
        assert n_in == 1, "RInverseMonomialLeaf only supports 1D"
        self.degree = int(degree)
        self.eps = eps

    def forward(self, x):
        z = x[:, 0].clamp_min(self.eps)
        return (1.0 / z.pow(self.degree)).unsqueeze(-1)


def _safe_inverse_expm1(
    t: torch.Tensor,
    *,
    eps: float,
    large_threshold: float,
) -> torch.Tensor:
    """Evaluate ``1 / expm1(t)`` without bias away from its singularity.

    The former Planck implementation added ``eps`` to every denominator.  That
    made the fitted torch module a different function from its analytic
    serialization.  Here ``eps`` is only a local guard when ``expm1(t)`` is
    genuinely too close to zero.  The large-positive branch is algebraically
    identical but avoids overflowing ``expm1``.
    """

    threshold = float(large_threshold)
    regular_t = torch.clamp(t, max=threshold)
    denominator = torch.expm1(regular_t)
    floor = denominator.new_tensor(float(eps))
    signed_floor = torch.where(denominator < 0, -floor, floor)
    safe_denominator = torch.where(
        denominator.abs() < floor,
        signed_floor,
        denominator,
    )
    regular = safe_denominator.reciprocal()

    large_t = torch.clamp(t, min=threshold)
    exp_neg = torch.exp(-large_t)
    large = exp_neg / (1.0 - exp_neg)
    return torch.where(t > threshold, large, regular)


class PlanckLeaf(nn.Module):
    """Reduced Planck/Bose-Einstein leaf with a fixed structural power.

    y = A * z**p / (exp(a*z) - 1)

    The exponent ``p`` is a discrete structural choice made by the proposal
    rule, not a continuous fitted parameter.  This keeps model selection from
    treating the normal Planck family as a four-parameter generic approximant.
    """

    def __init__(
        self,
        n_in: int,
        dtype=torch.float64,
        device=None,
        eps: float = 1e-8,
        clamp: float = 60.0,
        p: float = 1.0,
    ):
        super().__init__()
        dev = device or torch.device("cpu")
        assert n_in == 1, "PlanckLeaf currently only supports 1D"
        self.eps = float(eps)
        self.clamp = float(clamp)
        self.register_buffer("p", torch.tensor(float(p), dtype=dtype, device=dev))
        self.log_amp = nn.Parameter(torch.zeros((), dtype=dtype, device=dev))
        self.log_a = nn.Parameter(torch.zeros((), dtype=dtype, device=dev))

    def forward(self, x):
        z = x[:, 0].clamp_min(self.eps)
        amp = torch.exp(self.log_amp)
        a = torch.exp(self.log_a)
        t = a * z
        num = z.pow(self.p)
        y = amp * num * _safe_inverse_expm1(
            t,
            eps=self.eps,
            large_threshold=self.clamp,
        )
        return y.unsqueeze(-1)


class PlanckFullLeaf(nn.Module):
    """Flexible Planck/Bose-Einstein leaf.

    y = A * z**p / (exp(a*z + b) - 1)

    This is the legacy fully fitted Planck parameterisation.  It should be
    proposed later than the reduced structural Planck family.
    """

    def __init__(
        self, n_in: int, dtype=torch.float64, device=None, eps: float = 1e-8, clamp: float = 60.0
    ):
        super().__init__()
        dev = device or torch.device("cpu")
        assert n_in == 1, "PlanckFullLeaf currently only supports 1D"
        self.eps = float(eps)
        self.clamp = float(clamp)
        self.log_amp = nn.Parameter(torch.zeros((), dtype=dtype, device=dev))
        self.p = nn.Parameter(torch.tensor(3.0, dtype=dtype, device=dev))
        self.log_a = nn.Parameter(torch.zeros((), dtype=dtype, device=dev))
        self.b = nn.Parameter(torch.zeros((), dtype=dtype, device=dev))

    def forward(self, x):
        z = x[:, 0].clamp_min(self.eps)
        amp = torch.exp(self.log_amp)
        a = torch.exp(self.log_a)
        t = a * z + self.b
        num = z.pow(self.p)
        y = amp * num * _safe_inverse_expm1(
            t,
            eps=self.eps,
            large_threshold=self.clamp,
        )
        return y.unsqueeze(-1)


class Expm1Leaf(nn.Module):
    """
    Expm1 template: y = amp * (exp(a*z + b) - 1)

    Useful for exponential growth/decay with offset, complementary to Planck.
    """

    def __init__(
        self, n_in: int, dtype=torch.float64, device=None, eps: float = 1e-8, clamp: float = 60.0
    ):
        super().__init__()
        dev = device or torch.device("cpu")
        assert n_in == 1, "Expm1Leaf currently only supports 1D"
        self.eps = float(eps)
        self.clamp = float(clamp)
        self.log_amp = nn.Parameter(torch.zeros((), dtype=dtype, device=dev))
        self.log_a = nn.Parameter(torch.zeros((), dtype=dtype, device=dev))
        self.b = nn.Parameter(torch.zeros((), dtype=dtype, device=dev))

    def forward(self, x):
        z = x[:, 0]
        amp = torch.exp(self.log_amp)
        a = torch.exp(self.log_a)
        t = (a * z + self.b).clamp(min=-self.clamp, max=self.clamp)
        y = amp * torch.expm1(t)
        return y.unsqueeze(-1)


class ExpPolyLeaf(nn.Module):
    def __init__(
        self, n_in: int, degree: int, dtype=torch.float64, device=None, clamp: float = 60.0
    ):
        super().__init__()
        dev = device or torch.device("cpu")
        self.n_in = n_in
        self.degree = degree
        self.clamp = clamp

        exps = _enumerate_exponents(n_in, degree)
        self.register_buffer("exps", torch.tensor(exps, dtype=torch.int64))
        self.coeffs = nn.Parameter(torch.zeros(len(exps), dtype=dtype, device=dev))

    @property
    def n_terms(self):
        return int(self.coeffs.numel())

    def forward(self, x):
        # x: [B, n_in]
        Phi = _eval_monomials(x, self.exps)  # [B, M]
        g = Phi @ self.coeffs  # [B]

        if self.clamp is not None:
            g = g.clamp(min=-self.clamp, max=self.clamp)

        y = torch.exp(g)
        return y.unsqueeze(-1)  # [B, 1]


class RExpPolyLeaf(nn.Module):
    """Reduced exp(poly) leaf.

    This is an exponential of a polynomial where the **constant term of the
    exponent** is pinned to 0:

        y(x) = exp(P(x)),   with  P(0)=0  (equivalently: the constant monomial
        coefficient is fixed to 0).

    Motivation
    ----------
    In multiplicative chains, a constant shift in the exponent is a pure
    multiplicative gauge:

        exp(c0 + Q(x)) = exp(c0) * exp(Q(x)).

    When such terms sit in a product with other factors, letting c0 float can
    create severe scaling/gauge pathologies (and can cause the exponent to blow
    up during optimisation). This reduced parametrisation removes that degree of
    freedom; callers can represent exp(c0) explicitly via a multiplicative scale
    leaf.
    """

    def __init__(
        self,
        n_in: int,
        degree: int,
        dtype: torch.dtype = torch.float64,
        device=None,
        clamp: float = 60.0,
    ):
        super().__init__()
        dev = device or torch.device("cpu")
        self.n_in = int(n_in)
        self.degree = int(degree)
        self.clamp = clamp

        exps_full = _enumerate_exponents(self.n_in, self.degree)
        exps_full_t = torch.tensor(exps_full, dtype=torch.int64)
        self.register_buffer("exps_full", exps_full_t)

        # Identify constant monomial position (sum of exponents == 0)
        const_pos = None
        for k, e in enumerate(self.exps_full):
            if int(e.sum().item()) == 0:
                const_pos = k
                break
        if const_pos is None:
            const_pos = 0
        self.const_pos = int(const_pos)

        M = int(self.exps_full.shape[0])
        if M <= 1:
            free_pos = torch.empty(0, dtype=torch.int64)
            exps_free = self.exps_full[:0].clone()
        else:
            free_pos = torch.cat(
                [torch.arange(0, self.const_pos), torch.arange(self.const_pos + 1, M)]
            ).to(torch.int64)
            exps_free = self.exps_full[free_pos].clone()

        self.register_buffer("free_pos", free_pos)
        self.register_buffer("exps", exps_free)

        # Trainable coefficients for the non-constant monomials only.
        self.coeffs = nn.Parameter(
            torch.zeros(int(exps_free.shape[0]), dtype=dtype, device=dev)
        )

    @property
    def n_terms(self) -> int:
        return int(self.coeffs.numel())

    def full_coeffs(self) -> torch.Tensor:
        """Return the full coefficient vector including the pinned constant 0."""
        M = int(self.exps_full.shape[0])
        out = self.coeffs.new_zeros(M)
        if self.free_pos.numel() > 0 and self.coeffs.numel() > 0:
            out[self.free_pos] = self.coeffs
        # const_pos coefficient is pinned to 0 by construction.
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xin = x[..., : self.n_in]
        if self.exps.numel() == 0:
            g = xin.new_zeros(int(xin.shape[0]))
        else:
            Phi = _eval_monomials(xin, self.exps)
            g = Phi @ self.coeffs

        if self.clamp is not None:
            g = g.clamp(min=-self.clamp, max=self.clamp)

        y = torch.exp(g)
        return y.unsqueeze(-1)


class PolyLeaf(nn.Module):
    """Polynomial leaf with configurable monomial basis.

    Args:
        n_in: Number of input variables
        degree: Maximum total degree
        min_total: Minimum total degree for monomials. Defaults to `degree`
            (homogeneous polynomial). Set to 0 for full polynomial basis
            including constant, linear, etc.
        dtype: Parameter dtype
        device: Parameter device

    Notes:
        - Default (min_total=degree): Homogeneous polynomial of degree `degree`.
          For dimensional analysis, this avoids mixing terms with different units.
        - Full basis (min_total=0): Includes all monomials from degree 0 to `degree`.
          Use when fitting to feature probes or when dimensions are not enforced.
    """

    def __init__(self, n_in: int, degree: int, dtype=torch.float64, device=None, min_total=None):
        super().__init__()
        dev = device or torch.device("cpu")
        self.n_in = n_in
        self.degree = degree
        # Default to homogeneous (min_total=degree) for dimensional consistency
        if min_total is None:
            min_total = degree
        # Special case: polynomial in 0 variables should be a constant.
        # With default min_total=degree, a 0-var poly with degree>0 produces zero terms.
        if n_in == 0:
            min_total = 0
        self.min_total = int(min_total)
        exps = _enumerate_exponents(n_in, degree, min_total=self.min_total)
        self.register_buffer("exps", torch.tensor(exps, dtype=torch.int64))
        self.coeffs = nn.Parameter(torch.zeros(len(exps), dtype=dtype, device=dev))

    @property
    def n_terms(self):
        return int(self.coeffs.numel())

    def forward(self, x):
        Phi = _eval_monomials(x[..., :self.n_in], self.exps)
        y = Phi @ self.coeffs
        return y.unsqueeze(-1)


class RPolyLeaf(nn.Module):
    """Reduced (monic) polynomial leaf.

    Same basis as PolyLeaf, but fixes one (deterministic) 'leading' monomial
    coefficient to +1, removing one scalar degree of freedom.

    This is primarily intended to tame multiplicative gauge freedom when
    several polynomial-like factors appear in a product.
    """

    def __init__(self, n_in: int, degree: int, dtype=torch.float64, device=None, min_total=None):
        super().__init__()
        dev = device or torch.device('cpu')
        self.n_in = int(n_in)
        self.degree = int(degree)
        if min_total is None:
            min_total = degree
        if self.n_in == 0:
            min_total = 0
        self.min_total = int(min_total)

        exps_full = _enumerate_exponents(self.n_in, self.degree, min_total=self.min_total)
        exps_full_t = torch.tensor(exps_full, dtype=torch.int64)
        self.register_buffer('exps_full', exps_full_t)

        lead_pos = _lead_pos_from_exps(self.exps_full, self.degree)
        self.lead_pos = int(lead_pos)
        self.register_buffer('lead_exp', self.exps_full[self.lead_pos].clone())

        M = int(self.exps_full.shape[0])
        if M <= 1:
            free_pos = torch.empty(0, dtype=torch.int64)
            exps_free = self.exps_full[:0].clone()
        else:
            free_pos = torch.cat([torch.arange(0, self.lead_pos), torch.arange(self.lead_pos + 1, M)]).to(torch.int64)
            exps_free = self.exps_full[free_pos].clone()
        self.register_buffer('free_pos', free_pos)
        self.register_buffer('exps', exps_free)

        self.coeffs = nn.Parameter(torch.zeros(int(exps_free.shape[0]), dtype=dtype, device=dev))

    @property
    def n_terms(self) -> int:
        return int(self.coeffs.numel())

    def full_coeffs(self) -> torch.Tensor:
        """Return the full coefficient vector (including the fixed leading 1.0)."""
        M = int(self.exps_full.shape[0])
        out = self.coeffs.new_zeros(M)
        if self.free_pos.numel() > 0 and self.coeffs.numel() > 0:
            out[self.free_pos] = self.coeffs
        if 0 <= self.lead_pos < M:
            out[int(self.lead_pos)] = out.new_tensor(1.0)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xin = x[..., : self.n_in]
        lead = _eval_monomials(xin, self.lead_exp.view(1, -1)).squeeze(1)
        if self.exps.numel() == 0:
            return lead.unsqueeze(-1)
        Phi = _eval_monomials(xin, self.exps)
        y = lead + Phi @ self.coeffs
        return y.unsqueeze(-1)


class PolyLogLeaf(nn.Module):
    """
    Polynomial in log(x) for a small subset of variables:
        z = log(x[..., :n_in])
        P(z) = Σ_k c_k m_k(z)
    where m_k are monomials of total degree <= degree in the log‑variables.
    """

    def __init__(self, n_in: int, degree: int, dtype=torch.float64, device=None, eps: float = 1e-8):
        super().__init__()
        dev = device or torch.device("cpu")
        self.n_in = int(n_in)
        self.degree = int(degree)
        self.eps = float(eps)

        exps = _enumerate_exponents(self.n_in, self.degree)
        self.register_buffer("exps", torch.tensor(exps, dtype=torch.int64))
        self.coeffs = nn.Parameter(torch.zeros(len(exps), dtype=dtype, device=dev))

    @property
    def n_terms(self) -> int:
        return int(self.coeffs.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_in], clamp to avoid log of non‑positive inputs.
        z = x[..., : self.n_in].clamp_min(self.eps).log()
        Phi = _eval_monomials(z, self.exps)  # [B, M]
        y = Phi @ self.coeffs  # [B]
        return y.unsqueeze(-1)  # [B, 1]


class RPolyLogLeaf(nn.Module):
    """Reduced (monic) polynomial in log(x).

    Same basis as PolyLogLeaf, but fixes one deterministic 'leading' monomial
    coefficient to +1, removing one scalar degree of freedom.
    """

    def __init__(self, n_in: int, degree: int, dtype=torch.float64, device=None, eps: float = 1e-8):
        super().__init__()
        dev = device or torch.device('cpu')
        self.n_in = int(n_in)
        self.degree = int(degree)
        self.eps = float(eps)

        exps_full = _enumerate_exponents(self.n_in, self.degree)
        exps_full_t = torch.tensor(exps_full, dtype=torch.int64)
        self.register_buffer('exps_full', exps_full_t)

        lead_pos = _lead_pos_from_exps(self.exps_full, self.degree)
        self.lead_pos = int(lead_pos)
        self.register_buffer('lead_exp', self.exps_full[self.lead_pos].clone())

        M = int(self.exps_full.shape[0])
        if M <= 1:
            free_pos = torch.empty(0, dtype=torch.int64)
            exps_free = self.exps_full[:0].clone()
        else:
            free_pos = torch.cat([torch.arange(0, self.lead_pos), torch.arange(self.lead_pos + 1, M)]).to(torch.int64)
            exps_free = self.exps_full[free_pos].clone()
        self.register_buffer('free_pos', free_pos)
        self.register_buffer('exps', exps_free)

        self.coeffs = nn.Parameter(torch.zeros(int(exps_free.shape[0]), dtype=dtype, device=dev))

    @property
    def n_terms(self) -> int:
        return int(self.coeffs.numel())

    def full_coeffs(self) -> torch.Tensor:
        M = int(self.exps_full.shape[0])
        out = self.coeffs.new_zeros(M)
        if self.free_pos.numel() > 0 and self.coeffs.numel() > 0:
            out[self.free_pos] = self.coeffs
        if 0 <= self.lead_pos < M:
            out[int(self.lead_pos)] = out.new_tensor(1.0)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x[..., : self.n_in].clamp_min(self.eps).log()
        lead = _eval_monomials(z, self.lead_exp.view(1, -1)).squeeze(1)
        if self.exps.numel() == 0:
            return lead.unsqueeze(-1)
        Phi = _eval_monomials(z, self.exps)
        y = lead + Phi @ self.coeffs
        return y.unsqueeze(-1)


class LogShiftedLeaf(nn.Module):
    """
    Shifted logarithm for a single variable:
        f(x) = amp * log(x - shift) + offset

    Parameters:
        amp: amplitude coefficient (learnable)
        shift: horizontal shift (learnable)
        offset: vertical offset (learnable)

    This handles expressions like log(x-1), log(x-2), etc.
    """

    def __init__(self, n_in: int = 1, dtype=torch.float64, device=None, eps: float = 1e-8):
        super().__init__()
        if n_in != 1:
            raise ValueError("LogShiftedLeaf only supports univariate inputs (n_in=1)")
        dev = device or torch.device("cpu")
        self.n_in = 1
        self.eps = float(eps)

        # Parameters: amp, shift, offset
        self.amp = nn.Parameter(torch.ones(1, dtype=dtype, device=dev))
        self.shift = nn.Parameter(torch.zeros(1, dtype=dtype, device=dev))
        self.offset = nn.Parameter(torch.zeros(1, dtype=dtype, device=dev))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1], compute amp * log(x - shift) + offset
        x_shifted = x[..., 0] - self.shift
        # Clamp to avoid log of non-positive
        z = x_shifted.clamp_min(self.eps).log()
        y = self.amp * z + self.offset
        return y.unsqueeze(-1)  # [B, 1]


class RationalPolyLeaf(nn.Module):
    """
    Rational polynomial in a small subset of variables:

        z = x[..., indices]         # shape (B, d)
        P(z) = Σ_i a_i m_i(z)
        Q(z) = Σ_j b_j n_j(z)
        f(x) = P(z) / Q(z)

    where m_i, n_j are monomials of total degree <= deg_num / deg_den
    in those d variables.  Uses FULL polynomial basis (constant + linear + ...),
    matching features._build_poly_design_matrix for consistent coefficient ordering.
    This differs from PolyLeaf's default homogeneous basis.
    """

    def __init__(
        self,
        indices: Sequence[int],
        deg_num: int,
        deg_den: int,
        dtype: torch.dtype = torch.float64,
        device=None,
        eps: float = 1e-8,
        min_total_num: int = 0,
        min_total_den: int = 0,
        exps_num_override: Optional[Sequence[Sequence[int]]] = None,
        exps_den_override: Optional[Sequence[Sequence[int]]] = None,
        support_num_override: Optional[Sequence[int]] = None,
        support_den_override: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        dev = device or torch.device("cpu")

        self.indices = tuple(int(i) for i in indices)
        self.deg_num = int(deg_num)
        self.deg_den = int(deg_den)
        self.eps = float(eps)
        self.min_total_num = int(min_total_num)
        self.min_total_den = int(min_total_den)

        nvars = len(self.indices)

        # Full polynomial basis by default (min_total=0).  When overrides are
        # provided, use the explicit sparse monomial support instead.
        exps_num = _resolve_exponent_table(
            n_in=nvars,
            degree=self.deg_num,
            min_total=self.min_total_num,
            override=exps_num_override,
            support_override=support_num_override,
            name="exps_num_override",
        )
        exps_den = _resolve_exponent_table(
            n_in=nvars,
            degree=self.deg_den,
            min_total=self.min_total_den,
            override=exps_den_override,
            support_override=support_den_override,
            name="exps_den_override",
        )

        self.register_buffer("exps_num", torch.tensor(exps_num, dtype=torch.int64))
        self.register_buffer("exps_den", torch.tensor(exps_den, dtype=torch.int64))

        self.coeffs_num = nn.Parameter(torch.zeros(len(exps_num), dtype=dtype, device=dev))
        self.coeffs_den = nn.Parameter(torch.zeros(len(exps_den), dtype=dtype, device=dev))

        # Start with Q(z) ≈ 1 to avoid huge scaling from num / ~0
        with torch.no_grad():
            if self.coeffs_den.numel() > 0:
                den_const_idx = None
                for k, e in enumerate(exps_den):
                    if int(sum(e)) == 0:
                        den_const_idx = int(k)
                        break
                if den_const_idx is not None:
                    self.coeffs_den[den_const_idx].fill_(1.0)
                else:
                    self.coeffs_den[0].fill_(1.0)

    @property
    def n_terms_num(self) -> int:
        return int(self.coeffs_num.numel())

    @property
    def n_terms_den(self) -> int:
        return int(self.coeffs_den.numel())

    def _eval_poly(self, z: torch.Tensor, exps: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        # z: [B, d], exps: [M, d], coeffs: [M]
        Phi = _eval_monomials(z, exps)  # [B, M]
        return Phi @ coeffs  # [B]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D], we keep the same interface as other leaves:
        z = x[..., list(self.indices)]  # [B, d]
        num = self._eval_poly(z, self.exps_num, self.coeffs_num)
        den = self._eval_poly(z, self.exps_den, self.coeffs_den)
        den = den.clamp_min(self.eps)
        y = num / den
        return y.unsqueeze(-1)  # [B, 1]


class RRationalPolyLeaf(nn.Module):
    """Reduced (monic-numerator) rational polynomial leaf.

    Same as RationalPolyLeaf, but fixes one deterministic numerator monomial
    coefficient to +1, removing one scalar gauge degree of freedom.

    Denominator coefficients remain fully trainable (with default init Q≈1).
    """

    def __init__(
        self,
        indices: Sequence[int],
        deg_num: int,
        deg_den: int,
        dtype: torch.dtype = torch.float64,
        device=None,
        eps: float = 1e-8,
        min_total_num: int = 0,
        min_total_den: int = 0,
        exps_num_override: Optional[Sequence[Sequence[int]]] = None,
        exps_den_override: Optional[Sequence[Sequence[int]]] = None,
        support_num_override: Optional[Sequence[int]] = None,
        support_den_override: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        dev = device or torch.device('cpu')
        self.indices = tuple(int(i) for i in indices)
        self.deg_num = int(deg_num)
        self.deg_den = int(deg_den)
        self.eps = float(eps)
        self.min_total_num = int(min_total_num)
        self.min_total_den = int(min_total_den)

        nvars = len(self.indices)
        exps_num_full = _resolve_exponent_table(
            n_in=nvars,
            degree=self.deg_num,
            min_total=self.min_total_num,
            override=exps_num_override,
            support_override=support_num_override,
            name="exps_num_override",
        )
        exps_den = _resolve_exponent_table(
            n_in=nvars,
            degree=self.deg_den,
            min_total=self.min_total_den,
            override=exps_den_override,
            support_override=support_den_override,
            name="exps_den_override",
        )

        exps_num_full_t = torch.tensor(exps_num_full, dtype=torch.int64)
        self.register_buffer('exps_num_full', exps_num_full_t)
        self.register_buffer('exps_den', torch.tensor(exps_den, dtype=torch.int64))

        lead_pos = _lead_pos_from_exps(self.exps_num_full, self.deg_num)
        self.lead_pos_num = int(lead_pos)
        self.register_buffer('lead_exp_num', self.exps_num_full[self.lead_pos_num].clone())

        M = int(self.exps_num_full.shape[0])
        if M <= 1:
            free_pos = torch.empty(0, dtype=torch.int64)
            exps_free = self.exps_num_full[:0].clone()
        else:
            free_pos = torch.cat([torch.arange(0, self.lead_pos_num), torch.arange(self.lead_pos_num + 1, M)]).to(torch.int64)
            exps_free = self.exps_num_full[free_pos].clone()
        self.register_buffer('free_pos_num', free_pos)
        self.register_buffer('exps_num', exps_free)

        self.coeffs_num = nn.Parameter(torch.zeros(int(exps_free.shape[0]), dtype=dtype, device=dev))
        self.coeffs_den = nn.Parameter(torch.zeros(int(self.exps_den.shape[0]), dtype=dtype, device=dev))

        # Start with Q(z) ≈ 1 to avoid huge scaling from num / ~0
        with torch.no_grad():
            if self.coeffs_den.numel() > 0:
                den_const_idx = None
                for k, e in enumerate(exps_den):
                    if int(sum(e)) == 0:
                        den_const_idx = int(k)
                        break
                if den_const_idx is not None:
                    self.coeffs_den[den_const_idx].fill_(1.0)
                else:
                    self.coeffs_den[0].fill_(1.0)

    @property
    def n_terms_num(self) -> int:
        return int(self.coeffs_num.numel())

    @property
    def n_terms_den(self) -> int:
        return int(self.coeffs_den.numel())

    def full_coeffs_num(self) -> torch.Tensor:
        M = int(self.exps_num_full.shape[0])
        out = self.coeffs_num.new_zeros(M)
        if self.free_pos_num.numel() > 0 and self.coeffs_num.numel() > 0:
            out[self.free_pos_num] = self.coeffs_num
        if 0 <= self.lead_pos_num < M:
            out[int(self.lead_pos_num)] = out.new_tensor(1.0)
        return out

    def _eval_poly(self, z: torch.Tensor, exps: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        Phi = _eval_monomials(z, exps)
        return Phi @ coeffs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x[..., list(self.indices)]
        lead = _eval_monomials(z, self.lead_exp_num.view(1, -1)).squeeze(1)
        if self.exps_num.numel() == 0:
            num = lead
        else:
            Phi = _eval_monomials(z, self.exps_num)
            num = lead + Phi @ self.coeffs_num
        den = self._eval_poly(z, self.exps_den, self.coeffs_den).clamp_min(self.eps)
        y = num / den
        return y.unsqueeze(-1)


class ExpRationalPolyLeaf(nn.Module):
    def __init__(
        self,
        n_in: int,
        deg_num: int,
        deg_den: int,
        dtype=torch.float64,
        device=None,
        clamp: float = 60.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        dev = device or torch.device("cpu")
        self.n_in = n_in
        self.deg_num = deg_num
        self.deg_den = deg_den
        self.eps = float(eps)
        self.clamp = float(clamp)

        exps_num = _enumerate_exponents(n_in, deg_num)
        exps_den = _enumerate_exponents(n_in, deg_den)
        self.register_buffer("exps_num", torch.tensor(exps_num, dtype=torch.int64))
        self.register_buffer("exps_den", torch.tensor(exps_den, dtype=torch.int64))
        self.coeffs_num = nn.Parameter(torch.zeros(len(exps_num), dtype=dtype, device=dev))
        self.coeffs_den = nn.Parameter(torch.zeros(len(exps_den), dtype=dtype, device=dev))
        with torch.no_grad():
            if self.coeffs_den.numel() > 0:
                self.coeffs_den[0].fill_(1.0)

    def _eval_poly(self, z, exps, coeffs):
        Phi = _eval_monomials(z, exps)
        return Phi @ coeffs

    def forward(self, x):
        z = x[..., : self.n_in]
        num = self._eval_poly(z, self.exps_num, self.coeffs_num)
        den = self._eval_poly(z, self.exps_den, self.coeffs_den).clamp_min(self.eps)
        h = (num / den).clamp(min=-self.clamp, max=self.clamp)
        return torch.exp(h).unsqueeze(-1)


class RatioPolyLeaf(nn.Module):
    """
    Polynomial in the ratio of two variables: poly(x_num / x_den).

    This leaf computes r = x[..., 0] / x[..., 1] (numerator / denominator)
    and evaluates a univariate polynomial P(r) = Σ_k c_k r^k.

    Useful for detecting functions that depend only on the ratio of two
    variables, e.g., homogeneous degree-0 functions like 1/sqrt(1 - (v/c)²).

    Args:
        degree: Maximum polynomial degree
        dtype: Parameter dtype
        device: Parameter device
        eps: Minimum absolute value for denominator clamping

    Notes:
        - Input tensor x has shape [B, 2] where x[:, 0] is numerator, x[:, 1] is denominator
        - The var_idxs in the AST AtomNode determine which original variables map to [0, 1]
        - Denominator is clamped to avoid division by zero
    """

    def __init__(self, degree: int, dtype=torch.float64, device=None, eps: float = 1e-8):
        super().__init__()
        dev = device or torch.device("cpu")
        self.degree = int(degree)
        self.eps = float(eps)
        # Polynomial coefficients: c_0, c_1, ..., c_degree
        self.coeffs = nn.Parameter(torch.zeros(degree + 1, dtype=dtype, device=dev))

    @property
    def n_terms(self) -> int:
        return self.degree + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate polynomial in ratio.

        Args:
            x: Input tensor [B, 2] where x[:, 0] is numerator var, x[:, 1] is denominator var

        Returns:
            P(r) where r = x[:, 0] / x[:, 1], shape [B, 1]
        """
        # Compute ratio, clamping denominator to avoid division by zero
        x_num = x[..., 0]  # [B]
        x_den = x[..., 1]  # [B]
        # Sign-aware clamping to avoid flipping sign
        x_den_safe = torch.where(
            x_den >= 0,
            x_den.clamp(min=self.eps),
            x_den.clamp(max=-self.eps),
        )
        r = x_num / x_den_safe  # [B]

        # Evaluate polynomial using Horner's method for numerical stability
        y = self.coeffs[-1]
        for k in range(self.degree - 1, -1, -1):
            y = y * r + self.coeffs[k]

        return y.unsqueeze(-1)  # [B, 1]


class RRatioPolyLeaf(nn.Module):
    """Reduced (monic) polynomial in the ratio of two variables.

    Same as RatioPolyLeaf, but fixes the leading coefficient (r**degree term)
    to +1, removing one scalar degree of freedom.

    Trainable parameters are c_0..c_{degree-1}.
    """

    def __init__(self, degree: int, dtype=torch.float64, device=None, eps: float = 1e-8):
        super().__init__()
        dev = device or torch.device('cpu')
        self.degree = int(degree)
        self.eps = float(eps)
        # Free coefficients: c_0..c_{degree-1}
        n_free = max(0, self.degree)
        self.coeffs = nn.Parameter(torch.zeros(n_free, dtype=dtype, device=dev))

    @property
    def n_terms(self) -> int:
        return int(self.coeffs.numel())

    def full_coeffs(self) -> torch.Tensor:
        if self.degree < 0:
            return self.coeffs.new_zeros(0)
        if self.degree == 0:
            return self.coeffs.new_tensor([1.0])
        return torch.cat([self.coeffs, self.coeffs.new_tensor([1.0])], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_num = x[..., 0]
        x_den = x[..., 1]
        x_den_safe = torch.where(
            x_den >= 0,
            x_den.clamp(min=self.eps),
            x_den.clamp(max=-self.eps),
        )
        r = x_num / x_den_safe

        # Horner evaluation for monic polynomial: r^deg + c_{deg-1} r^{deg-1} + ... + c0
        y = torch.ones_like(r)
        for k in range(self.degree - 1, -1, -1):
            y = y * r + self.coeffs[k]
        return y.unsqueeze(-1)
