# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Cheap visible ``R^1 -> R^1`` operator certificates.

This module is deliberately only a proposal layer.  It tests whether a
univariate target obeys a simple transformed relation such as

    phi(y) ~= a * psi(z) + b

and, if so, provides the visible inverse-AST shape to try next.  Stage A and
Stage B still build ordinary candidates and let the existing validation and
units machinery accept or reject them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Optional

import torch

from nestynet_sr.sr_core.bridges import (
    AcosNode,
    AsinNode,
    AtanNode,
    AtomNode,
    ConstNode,
    ExpNode,
    MulNode,
    Node,
    PowNode,
    _collect_var_idxs_from_node,
    clone_ast,
    replace_atom_in_ast,
)


@dataclass(frozen=True)
class R1OperatorCertificate:
    """A cheap transformed-space certificate and its visible inverse shape."""

    label: str
    transform_name: str
    inverse_kind: str
    affine_a: float
    affine_b: float
    rel_rms: float
    transform_domain_frac: float
    inverse_domain_frac: float
    branch_ok_frac: float
    n_points: int
    psi_power: float = 1.0
    output_sign: float = 1.0
    complexity: float = 1.0
    details: dict[str, Any] | None = None

    @property
    def ok_domain(self) -> bool:
        return (
            math.isfinite(float(self.rel_rms))
            and math.isfinite(float(self.transform_domain_frac))
            and math.isfinite(float(self.inverse_domain_frac))
            and math.isfinite(float(self.branch_ok_frac))
        )


def _as_1d_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().reshape(-1).to(dtype=torch.float64, device=torch.device("cpu"))


def _finite_scale(y: torch.Tensor) -> float:
    y = y[torch.isfinite(y)]
    if int(y.numel()) <= 0:
        return 1.0
    med = torch.median(y)
    mad = torch.median(torch.abs(y - med))
    rms = torch.sqrt(torch.mean((y - torch.mean(y)) ** 2))
    scale = max(float(mad.item()) * 1.4826, float(rms.item()), float(torch.median(torch.abs(y)).item()), 1.0e-12)
    return scale if math.isfinite(scale) and scale > 0.0 else 1.0


def _affine_fit(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> Optional[tuple[float, float, float, torch.Tensor]]:
    m = mask & torch.isfinite(x) & torch.isfinite(y)
    if int(m.sum().item()) < 8:
        return None
    xv = x[m]
    yv = y[m]
    if float(torch.max(xv).item() - torch.min(xv).item()) <= 1.0e-14 * max(1.0, float(torch.median(torch.abs(xv)).item())):
        return None
    A = torch.stack([xv, torch.ones_like(xv)], dim=1)
    try:
        sol = torch.linalg.lstsq(A, yv.unsqueeze(1)).solution.reshape(-1)
    except Exception:
        return None
    if int(sol.numel()) < 2:
        return None
    a = float(sol[0].item())
    b = float(sol[1].item())
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    pred = a * xv + b
    rms = torch.sqrt(torch.mean((pred - yv) ** 2))
    rel = float(rms.item()) / _finite_scale(yv)
    return a, b, rel, m


def _domain_nonzero(v: torch.Tensor) -> torch.Tensor:
    scale = max(1.0e-12, 1.0e-10 * _finite_scale(v))
    return torch.isfinite(v) & (torch.abs(v) > scale)


def _domain_positive(v: torch.Tensor) -> torch.Tensor:
    scale = max(1.0e-14, 1.0e-12 * _finite_scale(v))
    return torch.isfinite(v) & (v > scale)


def _domain_power(v: torch.Tensor, power: float) -> torch.Tensor:
    if abs(float(power) - 1.0) <= 1.0e-14:
        return torch.isfinite(v)
    frac = Fraction(float(power)).limit_denominator(16)
    m = torch.isfinite(v)
    if frac.denominator % 2 == 0:
        m = m & _domain_positive(v)
    if frac.numerator < 0 or float(power) < 0.0:
        m = m & _domain_nonzero(v)
    return m


def _eval_power(v: torch.Tensor, power: float) -> torch.Tensor:
    out = torch.full_like(v, float("nan"))
    m = _domain_power(v, power)
    if int(m.sum().item()) > 0:
        out[m] = torch.pow(v[m], float(power))
    return out


def _frac(mask: torch.Tensor, denom: int) -> float:
    denom = max(1, int(denom))
    return float(mask.sum().item()) / float(denom)


def scan_r1_operator_certificates(
    z_values: torch.Tensor,
    y_values: torch.Tensor,
    *,
    max_results: int = 8,
    rel_rms_max: float = 2.0e-3,
    min_domain_frac: float = 0.98,
    min_branch_frac: float = 0.98,
    min_points: int = 128,
) -> list[R1OperatorCertificate]:
    """Return ranked cheap operator certificates for ``y = f(z)``.

    The function uses only sampled values and does not mutate or accept any
    symbolic state.
    """

    z = _as_1d_cpu(z_values)
    y = _as_1d_cpu(y_values)
    n = min(int(z.numel()), int(y.numel()))
    if n <= 0:
        return []
    z = z[:n]
    y = y[:n]
    base_finite = torch.isfinite(z) & torch.isfinite(y)
    if int(base_finite.sum().item()) < int(min_points):
        return []

    out: list[R1OperatorCertificate] = []

    def add_candidate(
        *,
        label: str,
        transform_name: str,
        inverse_kind: str,
        psi: torch.Tensor,
        phi: torch.Tensor,
        transform_mask: torch.Tensor,
        inverse_domain_fn,
        branch_mask: Optional[torch.Tensor] = None,
        psi_power: float = 1.0,
        output_sign: float = 1.0,
        complexity: float = 1.0,
        extra_details: Optional[dict[str, Any]] = None,
    ) -> None:
        fit = _affine_fit(psi, phi, base_finite & transform_mask & torch.isfinite(psi) & torch.isfinite(phi))
        if fit is None:
            return
        a, b, rel, fit_mask = fit
        if not math.isfinite(rel) or rel > float(rel_rms_max):
            return
        arg = a * psi + b
        inv_mask = inverse_domain_fn(arg) & torch.isfinite(arg)
        branch = branch_mask if branch_mask is not None else torch.ones_like(base_finite, dtype=torch.bool)
        transform_frac = _frac(base_finite & transform_mask & torch.isfinite(phi), int(base_finite.sum().item()))
        inverse_frac = _frac(base_finite & torch.isfinite(psi) & inv_mask, int(base_finite.sum().item()))
        branch_frac = _frac(base_finite & branch, int(base_finite.sum().item()))
        if transform_frac < float(min_domain_frac):
            return
        if inverse_frac < float(min_domain_frac):
            return
        if branch_frac < float(min_branch_frac):
            return
        n_fit = int(fit_mask.sum().item())
        if n_fit < int(min_points):
            return
        details = {
            "fit_points": n_fit,
            "transform_domain_frac": transform_frac,
            "inverse_domain_frac": inverse_frac,
            "branch_ok_frac": branch_frac,
        }
        if extra_details:
            details.update(dict(extra_details))
        out.append(
            R1OperatorCertificate(
                label=str(label),
                transform_name=str(transform_name),
                inverse_kind=str(inverse_kind),
                affine_a=float(a),
                affine_b=float(b),
                rel_rms=float(rel),
                transform_domain_frac=float(transform_frac),
                inverse_domain_frac=float(inverse_frac),
                branch_ok_frac=float(branch_frac),
                n_points=int(n_fit),
                psi_power=float(psi_power),
                output_sign=float(output_sign),
                complexity=float(complexity),
                details=details,
            )
        )

    z_id = z
    z_nonzero = _domain_nonzero(z)
    z_inv = torch.where(z_nonzero, torch.reciprocal(z), torch.full_like(z, float("nan")))
    coord_variants: list[tuple[str, torch.Tensor, float, float]] = [
        ("", z_id, 1.0, 0.0),
    ]
    if _frac(base_finite & z_nonzero, int(base_finite.sum().item())) >= float(min_domain_frac):
        # NN(z) and NN(1/z) have the same representational power.  The
        # certificate battery must therefore test both orientations for every
        # outer link; otherwise an exact sqrt/log/trig link in 1/z can be
        # missed just because the intermediate surrogate used z.
        coord_variants.append(("_zinv", z_inv, -1.0, 0.1))
    always = torch.ones_like(base_finite, dtype=torch.bool)
    nonzero_y = _domain_nonzero(y)
    positive_y = _domain_positive(y)
    sqrt_tol = 1.0e-10
    trig_arg = lambda v: torch.isfinite(v) & (v >= -1.0 - 1.0e-9) & (v <= 1.0 + 1.0e-9)
    pos_arg = lambda v: torch.isfinite(v) & (v > -sqrt_tol)
    nonzero_arg = lambda v: _domain_nonzero(v)

    sign = 1.0 if float(torch.median(y[base_finite]).item()) >= 0.0 else -1.0
    for suffix, psi_coord, psi_power, coord_cost in coord_variants:
        add_candidate(
            label=f"r1_affine{suffix}",
            transform_name="identity",
            inverse_kind="identity",
            psi=psi_coord,
            phi=y,
            transform_mask=always,
            inverse_domain_fn=lambda v: torch.isfinite(v),
            psi_power=psi_power,
            complexity=1.0 + coord_cost,
        )

        add_candidate(
            label=f"r1_square_sqrt{suffix}",
            transform_name="square",
            inverse_kind="sqrt",
            psi=psi_coord,
            phi=y * y,
            transform_mask=always,
            inverse_domain_fn=pos_arg,
            branch_mask=(sign * y >= -1.0e-10 * _finite_scale(y)),
            psi_power=psi_power,
            output_sign=sign,
            complexity=1.5 + coord_cost,
        )

        add_candidate(
            label=f"r1_inv_square_invsqrt{suffix}",
            transform_name="inv_square",
            inverse_kind="invsqrt",
            psi=psi_coord,
            phi=torch.where(nonzero_y, torch.reciprocal(y * y), torch.full_like(y, float("nan"))),
            transform_mask=nonzero_y,
            inverse_domain_fn=pos_arg,
            branch_mask=(sign * y >= -1.0e-10 * _finite_scale(y)),
            psi_power=psi_power,
            output_sign=sign,
            complexity=1.7 + coord_cost,
        )

        add_candidate(
            label=f"r1_reciprocal{suffix}",
            transform_name="reciprocal",
            inverse_kind="reciprocal",
            psi=psi_coord,
            phi=torch.where(nonzero_y, torch.reciprocal(y), torch.full_like(y, float("nan"))),
            transform_mask=nonzero_y,
            inverse_domain_fn=nonzero_arg,
            psi_power=psi_power,
            complexity=1.4 + coord_cost,
        )

        add_candidate(
            label=f"r1_log_exp{suffix}",
            transform_name="log",
            inverse_kind="exp",
            psi=psi_coord,
            phi=torch.where(positive_y, torch.log(y), torch.full_like(y, float("nan"))),
            transform_mask=positive_y,
            inverse_domain_fn=lambda v: torch.isfinite(v),
            psi_power=psi_power,
            complexity=1.6 + coord_cost,
        )

    for power in (0.5, 2.0, 3.0, -1.0, -2.0):
        psi = _eval_power(z, power)
        label = str(Fraction(power).limit_denominator(16)).replace("/", "_").replace("-", "m")
        add_candidate(
            label=f"r1_power_affine_p{label}",
            transform_name="identity",
            inverse_kind="identity",
            psi=psi,
            phi=y,
            transform_mask=always,
            inverse_domain_fn=lambda v: torch.isfinite(v),
            psi_power=float(power),
            complexity=1.2 + abs(float(power)) * 0.1,
            extra_details={"psi_power": float(power)},
        )

    tan_ok = torch.isfinite(torch.cos(y)) & (torch.abs(torch.cos(y)) > 1.0e-8)
    tan_y = torch.where(tan_ok, torch.tan(y), torch.full_like(y, float("nan")))
    for suffix, psi_coord, psi_power, coord_cost in coord_variants:
        add_candidate(
            label=f"r1_outer_asin{suffix}",
            transform_name="sin",
            inverse_kind="asin",
            psi=psi_coord,
            phi=torch.sin(y),
            transform_mask=always,
            inverse_domain_fn=trig_arg,
            branch_mask=(y >= -math.pi / 2 - 1.0e-9) & (y <= math.pi / 2 + 1.0e-9),
            psi_power=psi_power,
            complexity=2.0 + coord_cost,
        )
        add_candidate(
            label=f"r1_outer_acos{suffix}",
            transform_name="cos",
            inverse_kind="acos",
            psi=psi_coord,
            phi=torch.cos(y),
            transform_mask=always,
            inverse_domain_fn=trig_arg,
            branch_mask=(y >= -1.0e-9) & (y <= math.pi + 1.0e-9),
            psi_power=psi_power,
            complexity=2.0 + coord_cost,
        )
        add_candidate(
            label=f"r1_outer_atan{suffix}",
            transform_name="tan",
            inverse_kind="atan",
            psi=psi_coord,
            phi=tan_y,
            transform_mask=tan_ok,
            inverse_domain_fn=lambda v: torch.isfinite(v),
            branch_mask=(y > -math.pi / 2 + 1.0e-8) & (y < math.pi / 2 - 1.0e-8),
            psi_power=psi_power,
            complexity=2.0 + coord_cost,
        )

    out = [c for c in out if c.ok_domain]
    out.sort(key=lambda c: (float(c.rel_rms), float(c.complexity), str(c.label)))

    dedup: list[R1OperatorCertificate] = []
    seen: set[tuple[str, str, int, int]] = set()
    for cert in out:
        key = (
            str(cert.inverse_kind),
            str(cert.transform_name),
            int(round(float(cert.psi_power) * 1000)),
            int(math.copysign(1, float(cert.output_sign))),
        )
        if key in seen:
            continue
        seen.add(key)
        dedup.append(cert)
        if len(dedup) >= int(max_results):
            break
    return dedup


def certificate_psi_ast(z_expr: Node, cert: R1OperatorCertificate) -> Node:
    z = clone_ast(z_expr)
    if abs(float(cert.psi_power) - 1.0) <= 1.0e-14:
        return z
    return PowNode(z, float(cert.psi_power))


def build_r1_certificate_replacement(
    root: Node,
    target: AtomNode,
    z_expr: Node,
    cert: R1OperatorCertificate,
    *,
    tag_prefix: str,
) -> tuple[Optional[Node], Optional[str]]:
    """Build a visible candidate AST plus the affine poly tag to initialise."""

    psi = certificate_psi_ast(z_expr, cert)
    var_idxs = tuple(sorted(int(v) for v in _collect_var_idxs_from_node(psi)))
    if not var_idxs:
        return None, None
    tag = f"{tag_prefix}_{cert.label}_arg"
    arg = AtomNode(
        kind="poly",
        var_idxs=var_idxs,
        kwargs={"degree": 1, "min_total": 0},
        tag=tag,
        inputs=(psi,),
    )

    inv = str(cert.inverse_kind)
    if inv == "identity":
        repl: Node = arg
    elif inv == "sqrt":
        repl = PowNode(arg, 0.5)
        if float(cert.output_sign) < 0:
            repl = MulNode(ConstNode(-1.0), repl)
    elif inv == "invsqrt":
        repl = PowNode(arg, -0.5)
        if float(cert.output_sign) < 0:
            repl = MulNode(ConstNode(-1.0), repl)
    elif inv == "reciprocal":
        repl = PowNode(arg, -1.0)
    elif inv == "exp":
        repl = ExpNode(arg)
    elif inv == "asin":
        repl = AsinNode(arg)
    elif inv == "acos":
        repl = AcosNode(arg)
    elif inv == "atan":
        repl = AtanNode(arg)
    else:
        return None, None

    return replace_atom_in_ast(root, target, repl), tag


def r1_certificate_poly_init(cert: R1OperatorCertificate) -> dict[tuple[int, ...], float]:
    return {
        (0,): float(cert.affine_b),
        (1,): float(cert.affine_a),
    }
