# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Finite point-Lie prolongation scoring for scalar DE candidates.

This module implements the conservative V3 prolongation layer: it does not solve
Lie determining equations and it does not inject equation-specific primitives.
Instead it evaluates a finite set of known point vector fields
X = xi(x,u)d/dx + eta(x,u)d/du against a discovered scalar DE residual
F(x,u,u_x,u_xx)=0 using first/second prolongation formulae.

The current scalar-ODE scorer is intentionally graph based: rotations and
Lorentz boosts act on the graph coordinates (x,u), not on two independent input
coordinates.  Any velocity/acceleration structure therefore enters through the
prolongation formulae rather than through hand-written gamma-factor templates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import math

import torch

from nestynet_sr.adaptors.u_feature_leaf import UFeatureCache
from nestynet_sr.sr_core.bridges import (
    AcosNode,
    AddNode,
    AsinNode,
    AtanNode,
    AtomNode,
    ConstNode,
    CosNode,
    ExpNode,
    LogNode,
    MulNode,
    Node,
    PowNode,
    SinNode,
)

from .jet_bundle import JetSpaceSpec

_EPS = 1.0e-12


@dataclass(frozen=True)
class AffinePointGenerator:
    """Linear point-Lie generator xi(x,u)d/dx + eta(x,u)d/du.

    The historical name is kept because earlier V3 tests/imports used it.  The
    same affine-in-(x,u) representation covers the known scalar-graph Lie
    templates used here: translations, scalings, rotations, Lorentz boosts, and
    sparse/general affine candidates.
    """

    name: str
    family: str
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    b0: float = 0.0
    b1: float = 0.0
    b2: float = 0.0
    source: str = "known"
    description: str = ""

    def fields(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        u1: torch.Tensor,
        u2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        xi = float(self.a0) + float(self.a1) * x + float(self.a2) * u
        eta = float(self.b0) + float(self.b1) * x + float(self.b2) * u
        d_xi = float(self.a1) + float(self.a2) * u1
        d_eta = float(self.b1) + float(self.b2) * u1
        eta1 = d_eta - u1 * d_xi
        eta2 = u2 * (float(self.b2) - 2.0 * float(self.a1) - 3.0 * float(self.a2) * u1)
        return xi, eta, eta1, eta2

    def coefficient_norm(self) -> float:
        return float(math.sqrt(sum(float(v) ** 2 for v in (self.a0, self.a1, self.a2, self.b0, self.b1, self.b2))))

    def normalized(self) -> Tuple["AffinePointGenerator", float]:
        norm = self.coefficient_norm()
        if not math.isfinite(norm) or norm <= _EPS:
            raise ValueError(f"zero or non-finite Lie generator: {self.name}")
        vals = [float(v) / norm for v in (self.a0, self.a1, self.a2, self.b0, self.b1, self.b2)]
        return (
            AffinePointGenerator(
                name=self.name,
                family=self.family,
                a0=vals[0],
                a1=vals[1],
                a2=vals[2],
                b0=vals[3],
                b1=vals[4],
                b2=vals[5],
                source=self.source,
                description=self.description,
            ),
            norm,
        )

    def to_report(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "generator_type": "point_lie_affine_xu",
            "source": self.source,
            "xi": _affine_repr(self.a0, self.a1, self.a2),
            "eta": _affine_repr(self.b0, self.b1, self.b2),
            "coefficients": {
                "xi_const": float(self.a0),
                "xi_x": float(self.a1),
                "xi_u": float(self.a2),
                "eta_const": float(self.b0),
                "eta_x": float(self.b1),
                "eta_u": float(self.b2),
            },
            "description": self.description,
        }


@dataclass(frozen=True)
class ProlongationSupportStatus:
    """Support contract for a finite jet-space prolongation path."""

    supported: bool
    jet_scope: str
    feature: str
    reason: str
    input_dim: int
    output_dim: int
    max_order: int

    def to_report(self) -> Dict[str, Any]:
        return {
            "supported": bool(self.supported),
            "jet_scope": self.jet_scope,
            "feature": self.feature,
            "reason": self.reason,
            "input_dim": int(self.input_dim),
            "output_dim": int(self.output_dim),
            "max_order": int(self.max_order),
        }


def point_prolongation_support(
    jet_space: JetSpaceSpec,
    *,
    feature: str = "point_lie_prolongation",
) -> ProlongationSupportStatus:
    """Return whether the current prolongation scorer supports a jet space."""

    feature_s = str(feature)
    if jet_space.input_dim == 1 and jet_space.output_dim == 1 and int(jet_space.max_order) in (1, 2):
        return ProlongationSupportStatus(
            supported=True,
            jet_scope=jet_space.jet_scope,
            feature=feature_s,
            reason="scalar_ode_phase_one",
            input_dim=int(jet_space.input_dim),
            output_dim=int(jet_space.output_dim),
            max_order=int(jet_space.max_order),
        )
    if jet_space.input_dim == 1 and jet_space.output_dim == 1:
        reason = f"scalar ODE prolongation currently supports max_order 1 or 2; got max_order={jet_space.max_order}"
    else:
        reason = jet_space.unsupported_scope_message("prolongation")
    return ProlongationSupportStatus(
        supported=False,
        jet_scope=jet_space.jet_scope,
        feature=feature_s,
        reason=reason,
        input_dim=int(jet_space.input_dim),
        output_dim=int(jet_space.output_dim),
        max_order=int(jet_space.max_order),
    )


def require_point_prolongation_support(
    jet_space: JetSpaceSpec,
    *,
    feature: str = "point_lie_prolongation",
) -> ProlongationSupportStatus:
    status = point_prolongation_support(jet_space, feature=feature)
    if not status.supported:
        raise NotImplementedError(status.reason)
    return status



class _RescaledPointGenerator:
    def __init__(self, base: Any, factor: float):
        self.base = base
        self.factor = float(factor)

    def fields(self, x: torch.Tensor, u: torch.Tensor, u1: torch.Tensor, u2: torch.Tensor):
        return tuple(value / self.factor for value in self.base.fields(x, u, u1, u2))


def _normalized_generator_for_scoring(
    gen: Any,
    *,
    x: torch.Tensor,
    u: torch.Tensor,
    u1: torch.Tensor,
    u2: torch.Tensor,
    effective_order: int,
) -> tuple[Any, float, str]:
    if hasattr(gen, "coefficient_norm"):
        norm = float(gen.coefficient_norm())
        label = "coefficient_l2"
    elif hasattr(gen, "xi_terms") or hasattr(gen, "eta_terms"):
        coeff_values: List[float] = []
        for attr in ("xi_terms", "eta_terms"):
            for term in list(getattr(gen, attr, ()) or ()):
                try:
                    coeff_values.append(float(term[0]))
                except Exception:
                    pass
        norm = float(math.sqrt(sum(v * v for v in coeff_values))) if coeff_values else 0.0
        label = "polynomial_coefficient_l2"
    else:
        with torch.no_grad():
            fields = list(gen.fields(x, u, u1, u2))
            if int(effective_order) < 2 and len(fields) >= 4:
                fields = fields[:3]
            sq = torch.zeros_like(x)
            for value in fields:
                sq = sq + torch.as_tensor(value, device=x.device, dtype=x.dtype).square()
            flat = sq.reshape(-1)
            mask = torch.isfinite(flat)
            norm = float(flat[mask].mean().sqrt().cpu().item()) if int(mask.sum().item()) else 0.0
        label = "sample_action_rms"
    if not math.isfinite(norm) or norm <= _EPS:
        name = str(getattr(gen, "name", type(gen).__name__))
        raise ValueError(f"zero or non-finite Lie generator: {name}")
    return _RescaledPointGenerator(gen, norm), norm, label

def _affine_repr(c0: float, cx: float, cu: float) -> str:
    pieces: List[str] = []
    for coeff, name in ((c0, "1"), (cx, "x"), (cu, "u")):
        coeff = float(coeff)
        if abs(coeff) <= 1.0e-15:
            continue
        if name == "1":
            pieces.append(f"{coeff:g}")
        elif abs(coeff - 1.0) <= 1.0e-15:
            pieces.append(name)
        elif abs(coeff + 1.0) <= 1.0e-15:
            pieces.append(f"-{name}")
        else:
            pieces.append(f"{coeff:g}*{name}")
    if not pieces:
        return "0"
    return " + ".join(pieces).replace("+ -", "- ")


def _cfg_bool(cfg: Any, *names: str, default: bool = False) -> bool:
    for name in names:
        if cfg is not None and hasattr(cfg, name):
            return bool(getattr(cfg, name))
    return bool(default)


def _cfg_value(cfg: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if cfg is not None and hasattr(cfg, name):
            return getattr(cfg, name)
    return default


def _cfg_any_bool(cfg: Any, *names: str, default: bool = False) -> bool:
    found = False
    for name in names:
        if cfg is not None and hasattr(cfg, name):
            found = True
            if bool(getattr(cfg, name)):
                return True
    return bool(default) if not found else False


def _robust_span(t: Any) -> Optional[float]:
    if t is None:
        return None
    try:
        arr = torch.as_tensor(t, dtype=torch.float64).detach().reshape(-1)
    except Exception:
        return None
    mask = torch.isfinite(arr)
    if int(mask.sum().item()) < 4:
        return None
    vals = arr[mask]
    q25 = torch.quantile(vals, 0.25)
    q75 = torch.quantile(vals, 0.75)
    iqr = float((q75 - q25).abs().cpu().item())
    std = float(vals.std(unbiased=False).cpu().item())
    span = max(iqr, std, _EPS)
    return span if math.isfinite(span) and span > _EPS else None


def _parse_scale_list(raw: Any) -> List[float]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    elif isinstance(raw, (list, tuple, set)):
        parts = list(raw)
    else:
        parts = [raw]
    out: List[float] = []
    for part in parts:
        try:
            value = float(part)
        except Exception:
            continue
        if math.isfinite(value) and abs(value) > _EPS:
            out.append(value)
    return out


def _graph_lorentz_scales(cfg: Any = None, *, x: Any = None, u: Any = None) -> List[Tuple[str, float]]:
    """Return non-equation-specific scale candidates for graph boosts.

    A scalar graph boost has the form xi=lambda*u, eta=x.  The lambda value is
    a coordinate-scale choice, not a gamma/RHS primitive.  Explicit values can be
    supplied by advanced callers, otherwise we test a dimensionless unit boost
    and, when samples are available, a data-normalized pair and its reciprocal.
    """

    candidates: List[Tuple[str, float]] = [("unit", 1.0)]
    for i, value in enumerate(_parse_scale_list(_cfg_value(cfg, "gs_lorentz_graph_scales", "lorentz_graph_scales"))):
        candidates.append((f"configured_{i}", value))

    sx = _robust_span(x)
    su = _robust_span(u)
    if sx is not None and su is not None:
        ratio_sq = float((sx / max(su, _EPS)) ** 2)
        inv_ratio_sq = float((su / max(sx, _EPS)) ** 2)
        candidates.extend(
            [
                ("data_x_over_u_sq", ratio_sq),
                ("data_u_over_x_sq", inv_ratio_sq),
            ]
        )

    out: List[Tuple[str, float]] = []
    seen = set()
    for label, value in candidates:
        if not math.isfinite(float(value)) or abs(float(value)) <= _EPS:
            continue
        key = round(float(value), 12)
        if key in seen:
            continue
        seen.add(key)
        out.append((str(label), float(value)))
    return out


def build_affine_point_generators(
    *,
    cfg: Any = None,
    include_known: Optional[bool] = None,
    include_general_affine: Optional[bool] = None,
    x: Any = None,
    u: Any = None,
) -> List[AffinePointGenerator]:
    """Build the finite point-Lie generator slate used by the scorer.

    This compatibility function now covers every known scalar graph generator
    family available to the V3 prolongation scorer, plus optional sparse/general
    affine candidates.
    """

    known = _cfg_bool(cfg, "gs_known_generators", "known_generators", "known_lie", default=True) if include_known is None else bool(include_known)
    general = _cfg_any_bool(cfg, "gs_general_affine", "general_affine", "affine_dense", default=False) if include_general_affine is None else bool(include_general_affine)
    translations = _cfg_bool(cfg, "gs_translations", "translations", default=True)
    diagonal_translations = _cfg_bool(cfg, "gs_diagonal_translations", "diagonal_translations", default=True)
    scalings = _cfg_bool(cfg, "gs_scalings", "scalings", default=True)
    rotations = _cfg_bool(cfg, "gs_rotations", "rotations", default=True)
    lorentz_boosts = _cfg_bool(cfg, "gs_lorentz_boosts", "lorentz_boosts", default=False)

    gens: List[AffinePointGenerator] = []
    if known:
        if translations:
            gens.extend(
                [
                    AffinePointGenerator("x_translation", "translation", a0=1.0),
                    AffinePointGenerator("u_translation", "translation", b0=1.0),
                ]
            )
        if diagonal_translations:
            gens.extend(
                [
                    AffinePointGenerator("diag_translation_plus", "diagonal_translation", a0=1.0, b0=1.0),
                    AffinePointGenerator("diag_translation_minus", "diagonal_translation", a0=1.0, b0=-1.0),
                ]
            )
        if scalings:
            gens.extend(
                [
                    AffinePointGenerator("x_scaling", "scaling", a1=1.0),
                    AffinePointGenerator("u_scaling", "scaling", b2=1.0),
                    AffinePointGenerator("xu_common_scaling", "scaling", a1=1.0, b2=1.0),
                    AffinePointGenerator("xu_opposite_scaling", "scaling", a1=1.0, b2=-1.0),
                ]
            )
        if rotations:
            gens.append(
                AffinePointGenerator(
                    "graph_rotation_xu",
                    "rotation",
                    a2=-1.0,
                    b1=1.0,
                    description="scalar graph rotation in the (x,u) plane",
                )
            )
        if lorentz_boosts:
            for label, scale in _graph_lorentz_scales(cfg, x=x, u=u):
                gens.append(
                    AffinePointGenerator(
                        f"graph_lorentz_boost_{label}",
                        "lorentz",
                        a2=float(scale),
                        b1=1.0,
                        description="scalar graph Lorentz boost xi=lambda*u, eta=x; velocity structure is induced by prolongation",
                    )
                )

    if general:
        gens.extend(
            [
                AffinePointGenerator("shear_x_from_u", "sparse_affine", a2=1.0, source="general_affine"),
                AffinePointGenerator("shear_u_from_x", "sparse_affine", b1=1.0, source="general_affine"),
                AffinePointGenerator("xi_x_plus_u", "sparse_affine", a1=1.0, a2=1.0, source="general_affine"),
                AffinePointGenerator("xi_x_minus_u", "sparse_affine", a1=1.0, a2=-1.0, source="general_affine"),
                AffinePointGenerator("eta_x_plus_u", "sparse_affine", b1=1.0, b2=1.0, source="general_affine"),
                AffinePointGenerator("eta_x_minus_u", "sparse_affine", b1=1.0, b2=-1.0, source="general_affine"),
                AffinePointGenerator("affine_x_eta_x", "sparse_affine", a1=1.0, b1=1.0, source="general_affine"),
                AffinePointGenerator("affine_u_eta_u", "sparse_affine", a2=1.0, b2=1.0, source="general_affine"),
                AffinePointGenerator("affine_cross_plus", "sparse_affine", a2=1.0, b1=1.0, source="general_affine"),
                AffinePointGenerator("affine_cross_minus", "sparse_affine", a2=1.0, b1=-1.0, source="general_affine"),
                AffinePointGenerator("affine_const_eta_x", "sparse_affine", a0=1.0, b1=1.0, source="general_affine"),
                AffinePointGenerator("affine_x_eta_const", "sparse_affine", a1=1.0, b0=1.0, source="general_affine"),
            ]
        )

    deduped: List[AffinePointGenerator] = []
    seen = set()
    for gen in gens:
        key = tuple(round(float(v), 12) for v in (gen.a0, gen.a1, gen.a2, gen.b0, gen.b1, gen.b2))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(gen)
    return deduped


def build_known_lie_point_generators(
    *,
    cfg: Any = None,
    include_known: Optional[bool] = None,
    include_general_affine: Optional[bool] = None,
    x: Any = None,
    u: Any = None,
) -> List[AffinePointGenerator]:
    """Explicit-name wrapper for the all-known Lie prolongation generator bank."""

    return build_affine_point_generators(
        cfg=cfg,
        include_known=include_known,
        include_general_affine=include_general_affine,
        x=x,
        u=u,
    )


def _as_col(t: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t, dtype=torch.float64)
    if t.ndim == 1:
        return t.unsqueeze(1)
    if t.ndim == 2:
        if int(t.shape[1]) == 1:
            return t
        return t[:, :1]
    raise ValueError(f"{name} must have shape (N,) or (N,1); got {tuple(t.shape)}")


def _const_like(ref: torch.Tensor, value: Any) -> torch.Tensor:
    val = ConstNode(value).value
    if isinstance(val, complex):
        if abs(float(val.imag)) > 1.0e-12:
            raise ValueError(f"complex constants are not supported in DE prolongation: {val!r}")
        val = float(val.real)
    return torch.full_like(ref, float(val))


def _atom_order(node: Node | None) -> int:
    if node is None:
        return 0
    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        if kind in ("d2u", "ddu", "hess_u"):
            return 2
        if kind in ("du", "d1u", "grad_u"):
            return 1
        return 0
    if isinstance(node, (AddNode, MulNode)):
        return max(_atom_order(node.left), _atom_order(node.right))
    if isinstance(node, PowNode):
        exp_order = _atom_order(node.exponent) if isinstance(node.exponent, (AtomNode, AddNode, MulNode, PowNode)) else 0
        return max(_atom_order(node.base), exp_order)
    if isinstance(node, (LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode)):
        return _atom_order(node.arg)
    if isinstance(node, ConstNode):
        return 0
    return 0


def _terms_order(terms: Sequence[Node | None]) -> int:
    out = 0
    for term in terms:
        out = max(out, _atom_order(term))
    return out


def _eval_term_on_jets(
    node: Node | None,
    *,
    x: torch.Tensor,
    u: torch.Tensor,
    u1: torch.Tensor,
    u2: torch.Tensor,
    x_axis: int,
) -> torch.Tensor:
    if node is None:
        return torch.ones_like(x)
    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        kwargs = getattr(node, "kwargs", {}) or {}
        if kind in ("var", "x", "input"):
            if len(node.var_idxs) != 1:
                raise ValueError(f"var atom expects one axis; got {node.var_idxs}")
            axis = int(node.var_idxs[0])
            if axis != int(x_axis):
                raise ValueError(f"prolongation scorer only supports x_axis={x_axis}, got var x{axis}")
            return x
        if kind in ("u", "field", "state"):
            return u
        if kind in ("du", "d1u", "grad_u"):
            axis = int(kwargs.get("axis", 0))
            if axis != int(x_axis):
                raise ValueError(f"prolongation scorer only supports du/dx_axis={x_axis}, got axis={axis}")
            return u1
        if kind in ("d2u", "ddu", "hess_u"):
            axis0 = int(kwargs.get("axis0", 0))
            axis1 = int(kwargs.get("axis1", 0))
            if axis0 != int(x_axis) or axis1 != int(x_axis):
                raise ValueError(
                    f"prolongation scorer only supports d2u/dx_axis^2 for x_axis={x_axis}, got ({axis0},{axis1})"
                )
            return u2
        if kind in ("const", "constant"):
            return _const_like(x, kwargs.get("value", 1.0))
        if kind in ("free_const", "freeconst", "free_constant", "scale"):
            return _const_like(x, kwargs.get("init", 1.0))
        if kind in ("fixed_const", "fixedconst", "fixed_constant"):
            return _const_like(x, kwargs.get("value", 1.0))
        raise ValueError(f"unsupported atom kind in DE prolongation: {kind!r}")
    if isinstance(node, ConstNode):
        return _const_like(x, node.value)
    if isinstance(node, AddNode):
        return _eval_term_on_jets(node.left, x=x, u=u, u1=u1, u2=u2, x_axis=x_axis) + _eval_term_on_jets(
            node.right, x=x, u=u, u1=u1, u2=u2, x_axis=x_axis
        )
    if isinstance(node, MulNode):
        return _eval_term_on_jets(node.left, x=x, u=u, u1=u1, u2=u2, x_axis=x_axis) * _eval_term_on_jets(
            node.right, x=x, u=u, u1=u1, u2=u2, x_axis=x_axis
        )
    if isinstance(node, PowNode):
        base = _eval_term_on_jets(node.base, x=x, u=u, u1=u1, u2=u2, x_axis=x_axis)
        exp = node.exponent
        if isinstance(exp, (int, float)):
            return base.pow(float(exp))
        exp_val = _eval_term_on_jets(exp, x=x, u=u, u1=u1, u2=u2, x_axis=x_axis)
        if int(exp_val.numel()) == 1:
            return base.pow(float(exp_val.reshape(-1)[0].detach().cpu()))
        return torch.pow(base, exp_val)
    if isinstance(node, LogNode):
        return torch.log(_eval_term_on_jets(node.arg, x=x, u=u, u1=u1, u2=u2, x_axis=x_axis))
    if isinstance(node, ExpNode):
        return torch.exp(_eval_term_on_jets(node.arg, x=x, u=u, u1=u1, u2=u2, x_axis=x_axis))
    if isinstance(node, SinNode):
        return torch.sin(_eval_term_on_jets(node.arg, x=x, u=u, u1=u1, u2=u2, x_axis=x_axis))
    if isinstance(node, CosNode):
        return torch.cos(_eval_term_on_jets(node.arg, x=x, u=u, u1=u1, u2=u2, x_axis=x_axis))
    if isinstance(node, AsinNode):
        return torch.asin(_eval_term_on_jets(node.arg, x=x, u=u, u1=u1, u2=u2, x_axis=x_axis))
    if isinstance(node, AcosNode):
        return torch.acos(_eval_term_on_jets(node.arg, x=x, u=u, u1=u1, u2=u2, x_axis=x_axis))
    if isinstance(node, AtanNode):
        return torch.atan(_eval_term_on_jets(node.arg, x=x, u=u, u1=u1, u2=u2, x_axis=x_axis))
    raise ValueError(f"unsupported node type in DE prolongation: {type(node).__name__}")


def _term_sum(
    terms: Sequence[Node | None],
    coeffs: torch.Tensor,
    *,
    x: torch.Tensor,
    u: torch.Tensor,
    u1: torch.Tensor,
    u2: torch.Tensor,
    x_axis: int,
) -> torch.Tensor:
    total = torch.zeros_like(x)
    for coeff, term in zip(coeffs.reshape(-1), terms):
        total = total + coeff * _eval_term_on_jets(term, x=x, u=u, u1=u1, u2=u2, x_axis=x_axis)
    return total


def _residual(
    order: int,
    terms: Sequence[Node | None],
    coeffs: torch.Tensor,
    *,
    x: torch.Tensor,
    u: torch.Tensor,
    u1: torch.Tensor,
    u2: torch.Tensor,
    x_axis: int,
) -> torch.Tensor:
    anchor = u1 if int(order) == 1 else u2
    return anchor + _term_sum(terms, coeffs, x=x, u=u, u1=u1, u2=u2, x_axis=x_axis)


def _zero_like_if_none(value: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    if value is None:
        return torch.zeros_like(ref)
    return value


def _pr_residual(
    gen: AffinePointGenerator,
    *,
    order: int,
    effective_order: int,
    terms: Sequence[Node | None],
    coeffs: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    u1: torch.Tensor,
    u2: torch.Tensor,
    x_axis: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xv = x.detach().clone().requires_grad_(True)
    uv = u.detach().clone().requires_grad_(True)
    u1v = u1.detach().clone().requires_grad_(True)
    u2v = u2.detach().clone().requires_grad_(True)
    F = _residual(order, terms, coeffs, x=xv, u=uv, u1=u1v, u2=u2v, x_axis=x_axis)
    grads = torch.autograd.grad(F.sum(), (xv, uv, u1v, u2v), allow_unused=True, create_graph=False)
    Fx, Fu, Fu1, Fu2 = (_zero_like_if_none(g, xv) for g in grads)
    xi, eta, eta1, eta2 = gen.fields(xv, uv, u1v, u2v)
    pr = xi * Fx + eta * Fu + eta1 * Fu1
    if int(effective_order) >= 2:
        pr = pr + eta2 * Fu2
    return F.detach(), pr.detach()


def _finite_rms(t: torch.Tensor) -> Tuple[float, int]:
    flat = t.reshape(-1)
    mask = torch.isfinite(flat)
    count = int(mask.sum().item())
    if count <= 0:
        return float("inf"), 0
    rms = flat[mask].square().mean().sqrt()
    return float(rms.detach().cpu().item()), count


def _metric_key(row: Dict[str, Any]) -> Tuple[int, float]:
    try:
        val = float(row.get("on_shell_metric", float("inf")))
    except Exception:
        val = float("inf")
    return (0 if math.isfinite(val) else 1, val if math.isfinite(val) else float("inf"))


def score_affine_point_generators_from_jets(
    *,
    order: int,
    x: torch.Tensor,
    u: torch.Tensor,
    u1: torch.Tensor,
    u2: torch.Tensor | None,
    term_asts: Sequence[Node | None],
    coeffs: torch.Tensor,
    x_axis: int = 0,
    cfg: Any = None,
    generators: Sequence[AffinePointGenerator] | None = None,
    include_known: Optional[bool] = None,
    include_general_affine: Optional[bool] = None,
    tol: float | None = None,
) -> Dict[str, Any]:
    """Score known point-Lie generators on already-materialized jet samples."""

    terms = list(term_asts or [])
    x = _as_col(x, "x").detach()
    u = _as_col(u, "u").to(device=x.device, dtype=x.dtype).detach()
    u1 = _as_col(u1, "u1").to(device=x.device, dtype=x.dtype).detach()
    if u2 is None:
        u2 = torch.zeros_like(u1)
    else:
        u2 = _as_col(u2, "u2").to(device=x.device, dtype=x.dtype).detach()
    coeffs_t = torch.as_tensor(coeffs, device=x.device, dtype=x.dtype).reshape(-1)
    if len(terms) != int(coeffs_t.numel()):
        raise ValueError(f"term/coeff mismatch: {len(terms)} terms for {int(coeffs_t.numel())} coefficients")

    order_i = int(order)
    if order_i not in (1, 2):
        raise ValueError(f"prolongation scorer supports explicit scalar ODE order 1 or 2; got {order_i}")
    for term in terms:
        term_order = _atom_order(term)
        if term_order > order_i:
            raise ValueError(f"term derivative order {term_order} exceeds residual anchor order {order_i}")
        if term_order == order_i and term_order > 0:
            raise ValueError(f"nonanchor term contains the order-{order_i} anchor derivative")

    tol_v = float(getattr(cfg, "gs_de_lie_prolongation_tol", 0.05)) if tol is None else float(tol)
    effective_order = order_i
    gens = list(generators) if generators is not None else build_known_lie_point_generators(
        cfg=cfg,
        include_known=include_known,
        include_general_affine=include_general_affine,
        x=x,
        u=u,
    )
    if not gens:
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "no_generators",
            "generator_bank": "known_lie_point_affine_xu",
            "tested_generator_families": [],
            "order": int(order),
            "effective_order": int(effective_order),
            "tol": tol_v,
            "num_samples": int(x.shape[0]),
            "num_terms": len(terms),
            "tested_generators": 0,
            "accepted_generators": [],
            "accepted_generator_names": [],
            "best_metric": None,
            "best_generator": None,
            "generators": [],
        }

    rows: List[Dict[str, Any]] = []
    n_samples = max(1, int(x.shape[0]))
    min_coverage = max(0.0, min(1.0, float(getattr(cfg, "gs_de_lie_prolongation_min_coverage", 0.90))))
    min_count = max(3, int(math.ceil(min_coverage * n_samples)))
    with torch.enable_grad():
        for gen in gens:
            report = gen.to_report()
            try:
                scored_gen, generator_norm, normalization_kind = _normalized_generator_for_scoring(
                    gen,
                    x=x,
                    u=u,
                    u1=u1,
                    u2=u2,
                    effective_order=effective_order,
                )
                report["generator_normalization"] = float(generator_norm)
                report["generator_normalization_kind"] = str(normalization_kind)
                raw_F, raw_pr = _pr_residual(
                    scored_gen,
                    order=order,
                    effective_order=effective_order,
                    terms=terms,
                    coeffs=coeffs_t,
                    x=x,
                    u=u,
                    u1=u1,
                    u2=u2,
                    x_axis=x_axis,
                )
                with torch.no_grad():
                    nonanchor = _term_sum(terms, coeffs_t, x=x, u=u, u1=u1, u2=u2, x_axis=x_axis)
                    if int(order) == 1:
                        u1_on = -nonanchor.detach()
                        u2_on = u2
                    else:
                        u1_on = u1
                        u2_on = -nonanchor.detach()
                on_F, on_pr = _pr_residual(
                    scored_gen,
                    order=order,
                    effective_order=effective_order,
                    terms=terms,
                    coeffs=coeffs_t,
                    x=x,
                    u=u,
                    u1=u1_on,
                    u2=u2_on,
                    x_axis=x_axis,
                )
                raw_pr_rms, raw_count = _finite_rms(raw_pr)
                on_pr_rms, on_count = _finite_rms(on_pr)
                raw_F_rms, _ = _finite_rms(raw_F)
                on_F_rms, _ = _finite_rms(on_F)
                anchor = u1 if int(order) == 1 else u2
                anchor_rms, _ = _finite_rms(anchor)
                term_rms, _ = _finite_rms(nonanchor)
                pieces = [raw_pr_rms, raw_F_rms, anchor_rms, term_rms]
                scale = sum(v for v in pieces if math.isfinite(float(v))) + _EPS
                metric = float(on_pr_rms / max(scale, _EPS)) if math.isfinite(on_pr_rms) else float("inf")
                coverage_fraction = float(min(on_count, raw_count) / n_samples)
                metric_eligible = bool(math.isfinite(metric) and on_count >= min_count and raw_count >= min_count)
                accepted = bool(metric_eligible and metric <= tol_v)
                report.update(
                    {
                        "status": "scored" if metric_eligible else "low_coverage",
                        "raw_pr_rms": raw_pr_rms,
                        "on_shell_pr_rms": on_pr_rms,
                        "raw_residual_rms": raw_F_rms,
                        "on_shell_residual_rms": on_F_rms,
                        "scale": float(scale),
                        "on_shell_metric": metric,
                        "finite_count": int(on_count),
                        "raw_finite_count": int(raw_count),
                        "finite_coverage_fraction": coverage_fraction,
                        "min_finite_coverage_fraction": float(min_coverage),
                        "min_finite_count": int(min_count),
                        "metric_eligible": bool(metric_eligible),
                        "accepted": accepted,
                    }
                )
            except Exception as exc:
                report.update(
                    {
                        "status": "failed",
                        "reason": str(exc)[:300],
                        "raw_pr_rms": None,
                        "on_shell_pr_rms": None,
                        "raw_residual_rms": None,
                        "on_shell_residual_rms": None,
                        "scale": None,
                        "on_shell_metric": None,
                        "finite_count": 0,
                        "raw_finite_count": 0,
                        "finite_coverage_fraction": 0.0,
                        "min_finite_coverage_fraction": float(min_coverage),
                        "min_finite_count": int(min_count),
                        "metric_eligible": False,
                        "accepted": False,
                    }
                )
            rows.append(report)

    rows.sort(key=_metric_key)
    accepted_rows = [row for row in rows if bool(row.get("accepted", False))]
    tested_families = sorted({str(row.get("family", "")) for row in rows if row.get("family")})
    eligible_rows = [row for row in rows if bool(row.get("metric_eligible", False))]
    best = eligible_rows[0] if eligible_rows else (rows[0] if rows else None)
    best_metric = None
    if best is not None and bool(best.get("metric_eligible", False)):
        try:
            best_metric = float(best.get("on_shell_metric"))
            if not math.isfinite(best_metric):
                best_metric = None
        except Exception:
            best_metric = None
    return {
        "enabled": True,
        "status": "scored",
        "generator_bank": "known_lie_point_affine_xu",
        "tested_generator_families": tested_families,
        "order": int(order),
        "effective_order": int(effective_order),
        "tol": tol_v,
        "min_finite_coverage_fraction": float(min_coverage),
        "min_finite_count": int(min_count),
        "num_samples": int(x.shape[0]),
        "num_terms": len(terms),
        "tested_generators": len(rows),
        "accepted_generators": accepted_rows,
        "accepted_generator_names": [str(row.get("name", "")) for row in accepted_rows],
        "best_metric": best_metric,
        "best_generator": best,
        "generators": rows,
    }


def score_known_lie_point_generators_from_jets(**kwargs) -> Dict[str, Any]:
    """Compatibility-friendly explicit wrapper around the scalar Lie scorer."""

    return score_affine_point_generators_from_jets(**kwargs)


def score_affine_point_generators_from_jet_space(
    *,
    jet_space: JetSpaceSpec,
    samples: Mapping[str, Any],
    term_asts: Sequence[Node | None],
    coeffs: torch.Tensor,
    order: int | None = None,
    x_axis: int = 0,
    cfg: Any = None,
    generators: Sequence[AffinePointGenerator] | None = None,
    include_known: Optional[bool] = None,
    include_general_affine: Optional[bool] = None,
    tol: float | None = None,
) -> Dict[str, Any]:
    """Score point generators from a JetSpaceSpec-backed sample table.

    Vector systems and PDEs are represented by ``JetSpaceSpec`` and
    ``JetSampleTable``, but the executable point-prolongation scorer remains
    scalar-ODE-only in phase one.
    """

    support = require_point_prolongation_support(jet_space)
    order_i = int(jet_space.max_order if order is None else order)
    scalar = jet_space.materialize_scalar_ode_inputs(samples, order=order_i)
    meta = score_affine_point_generators_from_jets(
        order=scalar.order,
        x=scalar.x,
        u=scalar.u,
        u1=scalar.u1,
        u2=scalar.u2,
        term_asts=term_asts,
        coeffs=coeffs,
        x_axis=int(x_axis),
        cfg=cfg,
        generators=generators,
        include_known=include_known,
        include_general_affine=include_general_affine,
        tol=tol,
    )
    meta["jet_space"] = jet_space.to_report()
    meta["prolongation_support"] = support.to_report()
    meta["prolongation_scope"] = "scalar_ode_phase_one"
    return meta


def score_known_lie_point_generators_from_jet_space(**kwargs) -> Dict[str, Any]:
    """Compatibility wrapper for known-Lie scoring from a jet-space table."""

    return score_affine_point_generators_from_jet_space(**kwargs)


def _subsample_rows(X: torch.Tensor, max_samples: int) -> torch.Tensor:
    n = int(X.shape[0])
    max_samples = int(max_samples)
    if max_samples <= 0 or n <= max_samples:
        return X
    idx = torch.linspace(0, n - 1, steps=max_samples, device=X.device).round().long()
    idx = torch.unique_consecutive(idx)
    return X.index_select(0, idx)


def score_de_lie_prolongation(
    *,
    order: int,
    x_axis: int,
    X: torch.Tensor,
    cache: UFeatureCache,
    term_asts: Sequence[Node | None],
    coeffs: torch.Tensor,
    cfg: Any = None,
) -> Dict[str, Any]:
    """Score a DE candidate using surrogate-derived jet samples."""

    max_samples = int(getattr(cfg, "gs_de_lie_prolongation_max_samples", 2048))
    Xs = _subsample_rows(X, max_samples)
    terms = list(term_asts or [])
    need_hess = int(order) >= 2 or _terms_order(terms) >= 2
    cache.reset()
    cache.ensure(Xs, need_grad=True, need_hess=need_hess)
    if cache.u is None or cache.g is None:
        raise RuntimeError("UFeatureCache did not populate u and first derivatives")
    x = Xs[:, int(x_axis) : int(x_axis) + 1]
    u = cache.u[:, 0:1]
    u1 = cache.g[:, 0, int(x_axis) : int(x_axis) + 1]
    if need_hess:
        if cache.H is None:
            raise RuntimeError("UFeatureCache did not populate second derivatives")
        u2 = cache.H[:, 0, int(x_axis), int(x_axis)].unsqueeze(1)
    else:
        u2 = torch.zeros_like(u1)
    meta = score_affine_point_generators_from_jets(
        order=order,
        x=x,
        u=u,
        u1=u1,
        u2=u2,
        term_asts=terms,
        coeffs=coeffs,
        x_axis=x_axis,
        cfg=cfg,
    )
    meta["max_samples"] = max_samples
    meta["num_input_samples"] = int(X.shape[0])
    input_dim = int(X.shape[1]) if getattr(X, "ndim", 0) >= 2 else 1
    meta["input_dim"] = input_dim
    meta["prolongation_scope"] = "scalar_graph_xu"
    meta["does_not_inject_gamma_template"] = True
    meta["unsupported_generator_scopes"] = []
    try:
        from nestynet_sr.sr_gs.de_upgrades import (
            score_discrete_symmetry_terms,
            score_autonomous_even_velocity_prior,
        )

        all_upgrades = bool(getattr(cfg, "gs_de_all_upgrades", False))
        upgrade_metrics = {}
        if all_upgrades or bool(getattr(cfg, "gs_de_determining_equations", False)):
            from nestynet_sr.sr_gs.de_certificates import certify_scalar_ode_candidate

            det_result = certify_scalar_ode_candidate(
                x=x.detach().cpu().numpy().reshape(-1),
                u=u.detach().cpu().numpy().reshape(-1),
                u1=(u1.detach().cpu().numpy().reshape(-1) if int(order) >= 2 else None),
                coeffs=[float(v) for v in coeffs.detach().cpu().reshape(-1)],
                term_asts=list(terms),
                order=int(order),
                x_axis=int(x_axis),
                on_shell_tol=float(getattr(cfg, "gs_de_certificate_tol", 1.0e-6)),
                off_shell_tol=float(getattr(cfg, "gs_de_certificate_tol", 1.0e-6)),
                max_samples=int(getattr(cfg, "gs_de_lie_prolongation_max_samples", 2048)),
                generator_max_degree=int(getattr(cfg, "gs_de_determining_max_degree", 2)),
                multiplier_max_degree=int(getattr(cfg, "gs_de_determining_multiplier_degree", 2)),
                bootstrap=int(getattr(cfg, "gs_de_determining_bootstraps", 8)),
                max_generators=int(getattr(cfg, "gs_de_determining_max_generators", 4)),
                sparse_rotation=bool(getattr(cfg, "gs_de_determining_sparse_rotation", True)),
                bracket_certificate=bool(getattr(cfg, "gs_de_determining_bracket_certificate", True)),
                use_coupled_polynomial_solver=True,
            )
            det_meta = det_result.to_report()
            # Keep the old key as an explicit compatibility alias, but make it
            # clear that it now contains a coupled nullspace solve rather than
            # independent basis-element scores.
            meta["joint_polynomial_generator_solve"] = det_meta
            meta["polynomial_generator_basis_probe"] = {
                "status": "superseded_by_joint_nullspace",
                "joint_result": det_meta,
            }
            det_metric = det_meta.get("on_shell_heldout_residual_rel")
            if det_metric is not None:
                upgrade_metrics["joint_polynomial_generator_best_metric"] = det_metric
        if all_upgrades or bool(getattr(cfg, "gs_de_noether_templates", False)):
            meta["autonomous_even_velocity_prior"] = score_autonomous_even_velocity_prior(terms)
        if all_upgrades or bool(getattr(cfg, "gs_de_discrete_symmetry_templates", False)):
            meta["parity_prior"] = score_discrete_symmetry_terms(terms)
        if upgrade_metrics:
            meta["upgrade_metrics"] = upgrade_metrics
            det_best = upgrade_metrics.get("joint_polynomial_generator_best_metric")
            try:
                if det_best is not None and (meta.get("best_metric") is None or float(det_best) < float(meta.get("best_metric"))):
                    meta["best_metric"] = float(det_best)
                    meta["best_metric_source"] = "joint_polynomial_generator_solve"
            except Exception:
                pass
    except Exception as exc:
        meta["upgrade_diagnostics_error"] = str(exc)[:300]
    if input_dim > 1:
        meta["unsupported_generator_scopes"].append(
            {
                "scope": "independent_variable_pairs",
                "families": ["rotation", "lorentz"],
                "reason": "scalar DE prolongation scores one graph coordinate pair (x_axis,u); multi-input pair generators need PDE/vector prolongation",
            }
        )
    return meta
