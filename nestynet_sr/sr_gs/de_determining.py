# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Coupled scalar-DE determining operator and relative-invariance certificate."""

from __future__ import annotations

import ast as py_ast
from dataclasses import dataclass, field
from fractions import Fraction
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .jet_bundle import JetSpaceSpec, ScalarODEJetInputs
from .prolongation import AffinePointGenerator

_EPS = 1.0e-12


@dataclass(frozen=True)
class RecoveredDEGenerator:
    """One recovered scalar point generator plus its relative certificate."""

    name: str
    family: str
    coefficients: tuple[float, ...]
    multiplier: float
    on_shell_residual_rel: float
    off_shell_relative_residual_rel: float
    accepted: bool
    source: str = "coupled_determining"
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_report(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "coefficients": [float(v) for v in self.coefficients],
            "multiplier": float(self.multiplier),
            "on_shell_residual_rel": float(self.on_shell_residual_rel),
            "off_shell_relative_residual_rel": float(self.off_shell_relative_residual_rel),
            "accepted": bool(self.accepted),
            "source": self.source,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class DEDeterminingResult:
    """Result of on-shell generator recovery plus off-shell certification."""

    status: str
    residual: str
    jet_space: JetSpaceSpec
    generators: tuple[RecoveredDEGenerator, ...] = ()
    multiplier: float = 0.0
    on_shell_residual_rel: float = math.inf
    off_shell_relative_residual_rel: float = math.inf
    determining_matrix_rank: int = 0
    determining_nullity: int = 0
    singular_values: tuple[float, ...] = ()
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_report(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "residual": self.residual,
            "jet_space": self.jet_space.to_report(),
            "generators": [gen.to_report() for gen in self.generators],
            "multiplier": float(self.multiplier),
            "on_shell_residual_rel": float(self.on_shell_residual_rel),
            "off_shell_relative_residual_rel": float(self.off_shell_relative_residual_rel),
            "determining_matrix_rank": int(self.determining_matrix_rank),
            "determining_nullity": int(self.determining_nullity),
            "singular_values": [float(v) for v in self.singular_values],
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


ResidualLike = "str | Callable[[Mapping[str, torch.Tensor]], torch.Tensor]"


def _residual_label(residual: Any) -> str:
    if isinstance(residual, str):
        return residual
    label = getattr(residual, "description", None)
    return str(label) if label else repr(residual)


def recover_de_generators(
    *,
    jet_space: JetSpaceSpec,
    residual: Any,
    on_shell_samples: Mapping[str, Any],
    off_shell_samples: Mapping[str, Any],
    on_shell_tol: float = 1.0e-8,
    off_shell_tol: float = 1.0e-8,
    rank_rtol: float = 1.0e-9,
    rank_atol: float = 1.0e-11,
    include_nullspace_combinations: bool = True,
) -> DEDeterminingResult:
    """Recover scalar ODE point symmetries and certify relative invariance.

    The on-shell stage solves ``pr V(F)=0`` after the caller has eliminated the
    anchor derivative in ``on_shell_samples``.  The off-shell stage then fits
    and certifies ``pr V(F) - lambda F = 0`` on independent jet samples.

    ``residual`` is either an expression string in ``x, u, u_x, u_xx`` or a
    callable mapping that environment to a residual tensor.  Beyond the six
    canonical basis elements, on-shell nullspace *combinations* (e.g. the
    shifted scaling ``d_x + x_d_x``) are extracted, integer-snapped when
    possible, and certified by the same off-shell test.
    """

    jet_space.require_scalar_ode_phase_one()
    residual_s = residual
    order = int(jet_space.max_order)
    on_inputs = jet_space.materialize_scalar_ode_inputs(on_shell_samples, order=order)
    off_inputs = jet_space.materialize_scalar_ode_inputs(off_shell_samples, order=order)
    basis = _canonical_affine_basis()

    matrix, singular_values, rank, nullity = _determining_matrix(
        residual_s,
        on_inputs,
        basis,
        rank_rtol=float(rank_rtol),
        rank_atol=float(rank_atol),
    )
    accepted: list[RecoveredDEGenerator] = []
    all_rows: list[dict[str, Any]] = []
    single_on_rel: list[float] = []
    for gen in basis:
        on_rel, on_abs = _on_shell_generator_residual(residual_s, on_inputs, gen)
        off_rel, multiplier, off_abs = _off_shell_relative_certificate(residual_s, off_inputs, gen)
        ok = bool(on_rel <= float(on_shell_tol) and off_rel <= float(off_shell_tol))
        single_on_rel.append(float(on_rel))
        row = RecoveredDEGenerator(
            name=gen.name,
            family=gen.family,
            coefficients=tuple(float(v) for v in _gen_coefficients(gen)),
            multiplier=float(multiplier),
            on_shell_residual_rel=float(on_rel),
            off_shell_relative_residual_rel=float(off_rel),
            accepted=ok,
            evidence={
                "on_shell_abs_rms": float(on_abs),
                "off_shell_abs_rms": float(off_abs),
                "on_shell_tol": float(on_shell_tol),
                "off_shell_tol": float(off_shell_tol),
            },
        )
        all_rows.append(row.to_report())
        if ok:
            accepted.append(row)

    if include_nullspace_combinations:
        for gen in _nullspace_combination_candidates(
            matrix,
            basis,
            single_on_rel=single_on_rel,
            on_shell_tol=float(on_shell_tol),
        ):
            on_rel, on_abs = _on_shell_generator_residual(residual_s, on_inputs, gen)
            off_rel, multiplier, off_abs = _off_shell_relative_certificate(residual_s, off_inputs, gen)
            ok = bool(on_rel <= float(on_shell_tol) and off_rel <= float(off_shell_tol))
            row = RecoveredDEGenerator(
                name=gen.name,
                family="nullspace_combination",
                coefficients=tuple(float(v) for v in _gen_coefficients(gen)),
                multiplier=float(multiplier),
                on_shell_residual_rel=float(on_rel),
                off_shell_relative_residual_rel=float(off_rel),
                accepted=ok,
                source="coupled_determining_nullspace",
                evidence={
                    "on_shell_abs_rms": float(on_abs),
                    "off_shell_abs_rms": float(off_abs),
                    "on_shell_tol": float(on_shell_tol),
                    "off_shell_tol": float(off_shell_tol),
                    "snapped": bool(getattr(gen, "description", "") == "snapped"),
                },
            )
            all_rows.append(row.to_report())
            if ok:
                accepted.append(row)

    accepted.sort(
        key=lambda row: (
            float(row.off_shell_relative_residual_rel),
            float(row.on_shell_residual_rel),
            -abs(float(row.multiplier)),
            0 if row.name == "u_d_u" else 1,
            row.name,
        )
    )
    best = accepted[0] if accepted else None
    return DEDeterminingResult(
        status="recovered" if best is not None else "rejected",
        residual=_residual_label(residual),
        jet_space=jet_space,
        generators=tuple(accepted),
        multiplier=float(best.multiplier) if best is not None else 0.0,
        on_shell_residual_rel=float(best.on_shell_residual_rel) if best is not None else math.inf,
        off_shell_relative_residual_rel=float(best.off_shell_relative_residual_rel) if best is not None else math.inf,
        determining_matrix_rank=int(rank),
        determining_nullity=int(nullity),
        singular_values=tuple(float(v) for v in singular_values),
        reason="accepted_relative_invariance" if best is not None else "no_generator_passed_relative_certificate",
        evidence={
            "on_shell_operator": "prV(F)=0",
            "off_shell_certificate": "prV(F)-lambda*F=0",
            "candidate_generators": all_rows,
            "rank_rtol": float(rank_rtol),
            "rank_atol": float(rank_atol),
        },
    )


def _canonical_affine_basis() -> tuple[AffinePointGenerator, ...]:
    return (
        AffinePointGenerator("d_x", "translation", a0=1.0, source="coupled_determining"),
        AffinePointGenerator("x_d_x", "scaling", a1=1.0, source="coupled_determining"),
        AffinePointGenerator("u_d_x", "shear", a2=1.0, source="coupled_determining"),
        AffinePointGenerator("d_u", "translation", b0=1.0, source="coupled_determining"),
        AffinePointGenerator("x_d_u", "shear", b1=1.0, source="coupled_determining"),
        AffinePointGenerator("u_d_u", "scaling", b2=1.0, source="coupled_determining"),
    )


_BASIS_NAMES = ("d_x", "x_d_x", "u_d_x", "d_u", "x_d_u", "u_d_u")


def _combo_name(coeffs: Sequence[float]) -> str:
    pieces = []
    for value, name in zip(coeffs, _BASIS_NAMES):
        v = float(value)
        if abs(v) <= 1.0e-12:
            continue
        if abs(v - 1.0) <= 1.0e-9:
            pieces.append(f"+{name}")
        elif abs(v + 1.0) <= 1.0e-9:
            pieces.append(f"-{name}")
        else:
            pieces.append(f"{v:+.3g}*{name}")
    text = "".join(pieces).lstrip("+")
    return f"combo:{text}" if text else "combo:zero"


def _snap_combo(coeffs: np.ndarray, *, max_denominator: int = 3, tol: float = 0.06) -> np.ndarray | None:
    """Snap a max-normalized coefficient vector to small rationals, if close."""

    for denom in range(1, int(max_denominator) + 1):
        snapped = np.round(coeffs * denom) / float(denom)
        if float(np.max(np.abs(snapped - coeffs))) <= float(tol) and np.any(np.abs(snapped) > 0):
            fracs = [Fraction(v).limit_denominator(denom) for v in snapped]
            lcm = 1
            for f in fracs:
                lcm = lcm * f.denominator // math.gcd(lcm, f.denominator)
            ints = np.asarray([float(f * lcm) for f in fracs], dtype=float)
            max_abs = float(np.max(np.abs(ints)))
            if max_abs > 0:
                return ints / max_abs
    return None


def _nullspace_combination_candidates(
    matrix: np.ndarray,
    basis: Sequence[AffinePointGenerator],
    *,
    single_on_rel: Sequence[float],
    on_shell_tol: float,
    direction_tol: float = 1.0e-6,
    max_candidates: int = 4,
) -> list[AffinePointGenerator]:
    """Extract on-shell nullspace combinations beyond the single basis elements.

    Columns whose single generator already passes on-shell are projected out,
    so only genuinely composite directions (e.g. ``d_x + x_d_x``) are returned.
    The on-shell/off-shell certification of each candidate is done by the
    caller with the same tolerances as the singles.
    """

    if matrix.size == 0 or matrix.ndim != 2 or matrix.shape[1] != len(basis):
        return []
    rest = [j for j in range(len(basis)) if float(single_on_rel[j]) > float(on_shell_tol)]
    if len(rest) < 2:
        return []
    sub = matrix[:, rest]
    scales = np.sqrt(np.mean(np.square(sub), axis=0))
    good = np.isfinite(scales) & (scales > 0.0)
    rest = [j for j, g in zip(rest, good) if g]
    if len(rest) < 2:
        return []
    sub = matrix[:, rest]
    scales = np.sqrt(np.mean(np.square(sub), axis=0))
    normalized = sub / scales[None, :]
    _u, s, vt = np.linalg.svd(normalized, full_matrices=False)
    s0 = float(s[0]) if s.size else 0.0
    if s0 <= 0.0:
        return []
    out: list[AffinePointGenerator] = []
    for row, sv in zip(vt[::-1], s[::-1]):
        if float(sv) > 1.0e-4 * s0:
            break
        # map back from column-normalized to raw basis coefficients
        raw = np.zeros(len(basis), dtype=float)
        for k, j in enumerate(rest):
            raw[j] = float(row[k]) / float(scales[k])
        max_abs = float(np.max(np.abs(raw)))
        if not math.isfinite(max_abs) or max_abs <= float(direction_tol):
            continue
        norm_coeffs = raw / max_abs
        first_nonzero = next((v for v in norm_coeffs if abs(v) > 1.0e-9), 1.0)
        if first_nonzero < 0:
            norm_coeffs = -norm_coeffs
        snapped = _snap_combo(norm_coeffs)
        for coeffs, tag in ((snapped, "snapped"), (norm_coeffs, "raw")):
            if coeffs is None:
                continue
            gen = AffinePointGenerator(
                _combo_name(coeffs),
                "nullspace_combination",
                a0=float(coeffs[0]),
                a1=float(coeffs[1]),
                a2=float(coeffs[2]),
                b0=float(coeffs[3]),
                b1=float(coeffs[4]),
                b2=float(coeffs[5]),
                source="coupled_determining_nullspace",
                description=tag,
            )
            out.append(gen)
            break  # prefer the snapped representative when it exists
        if len(out) >= int(max_candidates):
            break
    return out


def _determining_matrix(
    residual: Any,
    inputs: ScalarODEJetInputs,
    basis: Sequence[AffinePointGenerator],
    *,
    rank_rtol: float,
    rank_atol: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    columns = []
    for gen in basis:
        pr, _F = _pr_action_and_residual(residual, inputs, gen)
        columns.append(pr.detach().cpu().numpy().reshape(-1))
    matrix = np.column_stack(columns) if columns else np.zeros((int(inputs.x.shape[0]), 0), dtype=float)
    matrix = matrix[np.isfinite(matrix).all(axis=1)]
    if matrix.size == 0:
        return matrix, np.asarray([], dtype=float), 0, len(basis)
    _U, s, _Vt = np.linalg.svd(matrix, full_matrices=False)
    s0 = float(s[0]) if s.size else 0.0
    tol = max(float(rank_atol), float(rank_rtol) * max(1.0, s0))
    rank = int(np.sum(s > tol))
    nullity = max(0, int(matrix.shape[1]) - rank)
    return matrix, s, rank, nullity


def _on_shell_generator_residual(
    residual: Any,
    inputs: ScalarODEJetInputs,
    gen: AffinePointGenerator,
) -> tuple[float, float]:
    pr, F = _pr_action_and_residual(residual, inputs, gen)
    abs_rms = _finite_rms(pr)
    scale = max(_finite_rms(F), _jet_scale(inputs), 1.0, _EPS)
    return float(abs_rms / scale), float(abs_rms)


def _off_shell_relative_certificate(
    residual: Any,
    inputs: ScalarODEJetInputs,
    gen: AffinePointGenerator,
) -> tuple[float, float, float]:
    pr, F = _pr_action_and_residual(residual, inputs, gen)
    pr_v = pr.reshape(-1)
    F_v = F.reshape(-1)
    mask = torch.isfinite(pr_v) & torch.isfinite(F_v)
    if int(mask.sum().item()) <= 0:
        return math.inf, 0.0, math.inf
    pr_m = pr_v[mask]
    F_m = F_v[mask]
    denom = torch.sum(F_m * F_m)
    if float(denom.detach().cpu().item()) <= _EPS:
        multiplier = torch.zeros((), dtype=pr_m.dtype, device=pr_m.device)
    else:
        multiplier = torch.sum(F_m * pr_m) / denom
    residual_v = pr_m - multiplier * F_m
    abs_rms = _finite_rms(residual_v)
    scale = max(_finite_rms(pr_m), _finite_rms(multiplier * F_m), _finite_rms(F_m), 1.0, _EPS)
    return float(abs_rms / scale), float(multiplier.detach().cpu().item()), float(abs_rms)


def _pr_action_and_residual(
    residual: Any,
    inputs: ScalarODEJetInputs,
    gen: AffinePointGenerator,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = inputs.x.detach().clone().requires_grad_(True)
    u = inputs.u.detach().clone().requires_grad_(True)
    u1 = inputs.u1.detach().clone().requires_grad_(True)
    if inputs.u2 is None:
        u2 = torch.zeros_like(u1).requires_grad_(True)
    else:
        u2 = inputs.u2.detach().clone().requires_grad_(True)
    env = {"x": x, "u": u, "u_x": u1, "u_xx": u2}
    if callable(residual) and not isinstance(residual, str):
        F = residual(env).reshape(-1, 1)
    else:
        F = _eval_residual_expr(residual, env).reshape(-1, 1)
    grads = torch.autograd.grad(F.sum(), (x, u, u1, u2), allow_unused=True, create_graph=False)
    Fx, Fu, Fu1, Fu2 = (_zero_like_if_none(g, x) for g in grads)
    xi, eta, eta1, eta2 = gen.fields(x, u, u1, u2)
    pr = xi * Fx + eta * Fu + eta1 * Fu1
    if int(inputs.order) >= 2:
        pr = pr + eta2 * Fu2
    return pr.detach(), F.detach()


def _eval_residual_expr(expr: str, env: Mapping[str, torch.Tensor]) -> torch.Tensor:
    parsed = py_ast.parse(str(expr).replace("^", "**"), mode="eval")
    value = _eval_py_ast(parsed, env)
    if not isinstance(value, torch.Tensor):
        ref = next(iter(env.values()))
        value = torch.full_like(ref, float(value))
    return value


def _eval_py_ast(node: py_ast.AST, env: Mapping[str, torch.Tensor]) -> torch.Tensor | float:
    if isinstance(node, py_ast.Expression):
        return _eval_py_ast(node.body, env)
    if isinstance(node, py_ast.Name):
        if node.id not in env:
            raise ValueError(f"unknown residual symbol {node.id!r}")
        return env[node.id]
    if isinstance(node, py_ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"unsupported residual constant {node.value!r}")
    if isinstance(node, py_ast.UnaryOp):
        val = _eval_py_ast(node.operand, env)
        if isinstance(node.op, py_ast.USub):
            return -val
        if isinstance(node.op, py_ast.UAdd):
            return val
        raise ValueError(f"unsupported residual unary operator {type(node.op).__name__}")
    if isinstance(node, py_ast.BinOp):
        left = _eval_py_ast(node.left, env)
        right = _eval_py_ast(node.right, env)
        if isinstance(node.op, py_ast.Add):
            return left + right
        if isinstance(node.op, py_ast.Sub):
            return left - right
        if isinstance(node.op, py_ast.Mult):
            return left * right
        if isinstance(node.op, py_ast.Div):
            return left / right
        if isinstance(node.op, py_ast.Pow):
            return torch.pow(_as_tensor_like(left, right), _as_tensor_like(right, left))
        raise ValueError(f"unsupported residual binary operator {type(node.op).__name__}")
    if isinstance(node, py_ast.Call):
        if not isinstance(node.func, py_ast.Name):
            raise ValueError("only simple residual function calls are supported")
        func = _allowed_function(node.func.id)
        args = [_eval_py_ast(arg, env) for arg in node.args]
        return func(*args)
    raise ValueError(f"unsupported residual expression node {type(node).__name__}")


def _allowed_function(name: str):
    funcs = {
        "sin": torch.sin,
        "cos": torch.cos,
        "tan": torch.tan,
        "exp": torch.exp,
        "log": torch.log,
        "sqrt": torch.sqrt,
    }
    if name not in funcs:
        raise ValueError(f"unsupported residual function {name!r}")
    return funcs[name]


def _as_tensor_like(value: torch.Tensor | float, other: torch.Tensor | float) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(other, torch.Tensor):
        return torch.full_like(other, float(value))
    return torch.as_tensor(float(value), dtype=torch.float64)


def _zero_like_if_none(value: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
    if value is None:
        return torch.zeros_like(ref)
    return value


def _finite_rms(value: torch.Tensor) -> float:
    flat = torch.as_tensor(value).reshape(-1)
    mask = torch.isfinite(flat)
    if int(mask.sum().item()) <= 0:
        return math.inf
    return float(flat[mask].square().mean().sqrt().detach().cpu().item())


def _jet_scale(inputs: ScalarODEJetInputs) -> float:
    vals = [_finite_rms(inputs.x), _finite_rms(inputs.u), _finite_rms(inputs.u1)]
    if inputs.u2 is not None:
        vals.append(_finite_rms(inputs.u2))
    finite = [v for v in vals if math.isfinite(v)]
    return max(finite) if finite else 1.0


def _gen_coefficients(gen: AffinePointGenerator) -> tuple[float, float, float, float, float, float]:
    return (float(gen.a0), float(gen.a1), float(gen.a2), float(gen.b0), float(gen.b1), float(gen.b2))


__all__ = [
    "DEDeterminingResult",
    "RecoveredDEGenerator",
    "recover_de_generators",
]
