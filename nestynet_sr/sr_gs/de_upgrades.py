# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Bounded DE compatibility diagnostics and structural prior families.

The routines in this module are deliberately conservative.  They expose the
prototype DE hooks as explicit, source-tagged probes/priors without injecting
equation-specific primitives such as relativistic gamma factors.

Family map:

1. low-degree polynomial generator basis probes;
2. velocity-monomial templates;
3. autonomous/even-velocity templates;
4. parity and time-reversal prior templates/diagnostics;
5. weighted/quasi-homogeneous scaling templates;
6. radial/spherical-reduction templates;
7. unit-torus/Buckingham-pi remains implemented in :mod:`unit_torus`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from nestynet_sr.sr_core.bridges import DU, U, Var, Mul, Pow


NodeLike = Any


def _cfg_bool(cfg: Any, name: str, default: bool = False) -> bool:
    if cfg is not None and hasattr(cfg, name):
        return bool(getattr(cfg, name))
    return bool(default)


def _upgrade_bool(cfg: Any, name: str, default: bool = False) -> bool:
    return _cfg_bool(cfg, "gs_de_all_upgrades", False) or _cfg_bool(cfg, name, default)


def _cfg_int(cfg: Any, name: str, default: int) -> int:
    try:
        return int(getattr(cfg, name, default))
    except Exception:
        return int(default)


def _cfg_float(cfg: Any, name: str, default: float) -> float:
    try:
        return float(getattr(cfg, name, default))
    except Exception:
        return float(default)


def _append_unique_row(rows: list[tuple[NodeLike, str, str]], term: NodeLike, source: str, family: str) -> None:
    rep = repr(term)
    for old, _source, _family in rows:
        if repr(old) == rep:
            return
    rows.append((term, source, family))


def _pow_node(node: NodeLike, exponent: int) -> NodeLike | None:
    if exponent == 0:
        return None
    if exponent == 1:
        return node
    return Pow(node, exponent)


def _mul_nodes(nodes: Sequence[NodeLike | None]) -> NodeLike | None:
    out: NodeLike | None = None
    for node in nodes:
        if node is None:
            continue
        out = node if out is None else Mul(out, node)
    return out


def _monomial(x: NodeLike, u: NodeLike, du: NodeLike, px: int, pu: int, pv: int) -> NodeLike | None:
    return _mul_nodes((_pow_node(x, px), _pow_node(u, pu), _pow_node(du, pv)))


def _bounded(rows: list[tuple[NodeLike, str, str]], cfg: Any) -> list[tuple[NodeLike, str, str]]:
    limit = _cfg_int(cfg, "gs_de_upgrade_max_terms", 64)
    if limit <= 0:
        return rows
    return rows[:limit]


def velocity_monomial_prior_rows(cfg: Any, *, order: int) -> list[tuple[NodeLike, str, str]]:
    """Velocity-monomial templates on the scalar jet space J^2.

    These are structural priors, not a contact-symmetry solver. They are
    generic monomials in ``u_x`` and do not encode a gamma factor.
    """

    if int(order) < 2 or not _upgrade_bool(cfg, "gs_de_contact_templates", False):
        return []
    x = Var(int(getattr(cfg, "x_axis", 0)))
    u = U()
    du = DU(int(getattr(cfg, "x_axis", 0)))
    rows: list[tuple[NodeLike, str, str]] = []
    for term, family in (
        (Pow(du, 2), "velocity_even"),
        (Pow(du, 4), "velocity_even"),
        (Mul(u, Pow(du, 2)), "state_velocity"),
        (Mul(Pow(u, 2), Pow(du, 2)), "state_velocity"),
        (Mul(Pow(u, 3), Pow(du, 2)), "state_velocity"),
        (Mul(x, Pow(du, 2)), "coordinate_velocity"),
        (Mul(Mul(x, u), du), "mixed_velocity"),
    ):
        _append_unique_row(rows, term, "de_prior_velocity_monomial", family)
    return _bounded(rows, cfg)


def autonomous_even_velocity_prior_rows(cfg: Any, *, order: int) -> list[tuple[NodeLike, str, str]]:
    """Autonomous/even-velocity oscillator templates.

    These are structural priors, not a Noether-symmetry or Lagrangian solver.
    They capture autonomous potential-gradient terms in ``u`` and even-velocity
    corrections.
    """

    if int(order) < 2 or not _upgrade_bool(cfg, "gs_de_noether_templates", False):
        return []
    u = U()
    du = DU(int(getattr(cfg, "x_axis", 0)))
    rows: list[tuple[NodeLike, str, str]] = []
    for term, family in (
        (u, "autonomous_potential_gradient"),
        (Pow(u, 2), "autonomous_potential_gradient"),
        (Pow(u, 3), "autonomous_potential_gradient"),
        (Pow(u, 5), "autonomous_potential_gradient"),
        (Mul(u, Pow(du, 2)), "even_velocity_correction"),
        (Mul(Pow(u, 3), Pow(du, 2)), "even_velocity_correction"),
    ):
        _append_unique_row(rows, term, "de_prior_autonomous_even_velocity", family)
    return _bounded(rows, cfg)


def discrete_symmetry_de_rows(cfg: Any, *, order: int) -> list[tuple[NodeLike, str, str]]:
    """Parity/time-reversal prior templates.

    These are syntax-level parity priors, not a discrete-symmetry proof.
    """

    if int(order) < 2 or not _upgrade_bool(cfg, "gs_de_discrete_symmetry_templates", False):
        return []
    u = U()
    du = DU(int(getattr(cfg, "x_axis", 0)))
    rows: list[tuple[NodeLike, str, str]] = []
    for term in (
        u,
        Pow(u, 3),
        Pow(u, 5),
        Mul(u, Pow(du, 2)),
        Mul(u, Pow(du, 4)),
        Mul(Pow(u, 3), Pow(du, 2)),
    ):
        _append_unique_row(rows, term, "de_prior_parity", "odd_state_even_velocity")
    return _bounded(rows, cfg)


def weighted_scaling_de_rows(cfg: Any, *, order: int) -> list[tuple[NodeLike, str, str]]:
    """Quasi-homogeneous monomial rows.

    Rows satisfy

        p*w_x + q*w_u + r*(w_u - w_x) = w_u - order*w_x

    for at least one configured profile.  Two safe profiles are used by
    default: graph-homogeneous ``(1,1)`` and dimensionless-state ``(1,0)``.
    """

    if not _upgrade_bool(cfg, "gs_de_weighted_scaling_templates", False):
        return []

    x = Var(int(getattr(cfg, "x_axis", 0)))
    u = U()
    du = DU(int(getattr(cfg, "x_axis", 0)))
    max_abs_x = _cfg_int(cfg, "gs_de_weighted_max_abs_x_power", 2)
    max_u = _cfg_int(cfg, "gs_de_weighted_max_u_power", 5)
    max_du = 0 if int(order) < 2 else _cfg_int(cfg, "gs_de_weighted_max_du_power", 4)
    tol = _cfg_float(cfg, "gs_de_weighted_tol", 1.0e-12)

    profiles: list[tuple[float, float, str]] = [(1.0, 1.0, "graph_homogeneous"), (1.0, 0.0, "dimensionless_state")]
    raw_profiles = getattr(cfg, "gs_de_weighted_profiles", None)
    if raw_profiles:
        profiles = []
        for i, item in enumerate(raw_profiles):
            try:
                wx, wu = item
                profiles.append((float(wx), float(wu), f"configured_{i}"))
            except Exception:
                continue

    rows: list[tuple[NodeLike, str, str]] = []
    seen_by_profile = set()
    for wx, wu, label in profiles:
        target = float(wu) - int(order) * float(wx)
        du_w = float(wu) - float(wx)
        for px in range(-max_abs_x, max_abs_x + 1):
            for pu in range(0, max_u + 1):
                for pv in range(0, max_du + 1):
                    if px == 0 and pu == 0 and pv == 0:
                        continue
                    weight = px * wx + pu * wu + pv * du_w
                    if abs(weight - target) > tol:
                        continue
                    term = _monomial(x, u, du, px, pu, pv)
                    if term is None:
                        continue
                    key = (repr(term), label)
                    if key in seen_by_profile:
                        continue
                    seen_by_profile.add(key)
                    _append_unique_row(rows, term, "de_prior_assumed_weight_profile", f"quasi_homogeneous_{label}")
    return _bounded(rows, cfg)


def radial_reduction_de_rows(cfg: Any, *, order: int) -> list[tuple[NodeLike, str, str]]:
    """Systematic radial/spherical-reduction rows.

    These generalize Rod's inverse-x hard-tail rows.  They remain generic
    monomials; the fitted coefficient carries the effective dimension/order.
    """

    if not _upgrade_bool(cfg, "gs_de_radial_reduction_templates", False):
        return []
    x = Var(int(getattr(cfg, "x_axis", 0)))
    u = U()
    du = DU(int(getattr(cfg, "x_axis", 0)))
    rows: list[tuple[NodeLike, str, str]] = []
    for term, family in (
        (Mul(Pow(x, -1), u), "radial_value"),
        (Mul(Pow(x, -2), u), "radial_value"),
        (Mul(Pow(x, -3), u), "radial_value_high_order"),
    ):
        _append_unique_row(rows, term, "de_prior_radial_shape", family)
    if int(order) >= 2:
        for term, family in (
            (Mul(Pow(x, -1), du), "radial_derivative"),
            (Mul(Pow(x, -2), du), "radial_derivative_high_order"),
            (Mul(x, du), "radial_coordinate_prefactor"),
        ):
            _append_unique_row(rows, term, "de_prior_radial_shape", family)
    return _bounded(rows, cfg)


def symmetry_upgrade_de_term_rows(cfg: Any, *, order: int) -> list[tuple[NodeLike, str, str]]:
    """Return source-aware rows for upgrades 2-6.

    Upgrade 1 is exposed through generator-basis probe diagnostics below
    because it acts on fitted residuals rather than emitting unconditional rows.
    Upgrade 7 is handled by the existing unit-torus bridge.
    """

    rows: list[tuple[NodeLike, str, str]] = []
    for builder in (
        velocity_monomial_prior_rows,
        autonomous_even_velocity_prior_rows,
        discrete_symmetry_de_rows,
        weighted_scaling_de_rows,
        radial_reduction_de_rows,
    ):
        for term, source, family in builder(cfg, order=order):
            _append_unique_row(rows, term, source, family)
    return _bounded(rows, cfg)


_PARITY_KEYS = ("x", "u", "du")


def _known_signature(x: int = 0, u: int = 0, du: int = 0) -> dict[str, int]:
    return {"x": int(x), "u": int(u), "du": int(du)}


def _nonknown_signature(status: str) -> dict[str, Any]:
    return {"x": None, "u": None, "du": None, "status": str(status)}


def _signature_known(sig: dict[str, Any]) -> bool:
    return str(sig.get("status", "known")) == "known" and all(
        sig.get(k) is not None for k in _PARITY_KEYS
    )


def _signature_parity(sig: dict[str, Any]) -> tuple[int, int, int] | None:
    if not _signature_known(sig):
        return None
    return tuple(int(sig[k]) % 2 for k in _PARITY_KEYS)


def ast_discrete_signature(node: NodeLike | None) -> dict[str, Any]:
    """Return conservative parity information for x, u, and u_x.

    Known monomials return integer exponents for backward compatibility.
    Mixed additions, unsupported nodes, and fractional powers return a
    ``status`` field and are rejected by the parity diagnostics.
    """

    from nestynet_sr.sr_core.bridges import AtomNode, AddNode, MulNode, PowNode, ConstNode

    if node is None or isinstance(node, ConstNode):
        return _known_signature()
    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        if kind in {"var", "x", "input"}:
            return _known_signature(x=1)
        if kind in {"u", "field", "state"}:
            return _known_signature(u=1)
        if kind in {"du", "d1u", "grad_u"}:
            return _known_signature(du=1)
        return _nonknown_signature("unknown")
    if isinstance(node, MulNode):
        a = ast_discrete_signature(node.left)
        b = ast_discrete_signature(node.right)
        if not (_signature_known(a) and _signature_known(b)):
            return _nonknown_signature("unknown")
        return {k: int(a.get(k, 0)) + int(b.get(k, 0)) for k in _PARITY_KEYS}
    if isinstance(node, AddNode):
        a = ast_discrete_signature(node.left)
        b = ast_discrete_signature(node.right)
        pa = _signature_parity(a)
        pb = _signature_parity(b)
        if pa is None or pb is None:
            return _nonknown_signature("unknown")
        if pa != pb:
            return _nonknown_signature("mixed")
        return _known_signature(x=pa[0], u=pa[1], du=pa[2])
    if isinstance(node, PowNode):
        base = ast_discrete_signature(node.base)
        if not _signature_known(base):
            return _nonknown_signature("unknown")
        exp = getattr(node, "exponent", 1)
        try:
            exp_f = float(exp)
        except Exception:
            return _nonknown_signature("unknown")
        p = int(round(exp_f))
        if abs(exp_f - p) > 1.0e-12:
            return _nonknown_signature("unknown_fractional_power")
        return {k: int(base[k]) * p for k in _PARITY_KEYS}
    return _nonknown_signature("unknown")


def score_discrete_symmetry_terms(term_asts: Sequence[NodeLike | None], *, target: str = "odd_u_even_du") -> dict[str, Any]:
    rows = []
    accepted = 0
    for term in term_asts:
        sig = ast_discrete_signature(term)
        known = _signature_known(sig)
        if target == "odd_u_even_du" and known:
            ok = (int(sig["u"]) % 2 == 1) and (int(sig["du"]) % 2 == 0) and (int(sig["x"]) % 2 == 0)
        elif target == "even_du" and known:
            ok = int(sig["du"]) % 2 == 0
        else:
            ok = bool(known and target not in {"odd_u_even_du", "even_du"})
        accepted += int(ok)
        rows.append({
            "term": "1" if term is None else repr(term),
            "signature": sig,
            "signature_status": str(sig.get("status", "known")),
            "accepted": bool(ok),
        })
    total = len(rows)
    return {
        "family": "parity_prior",
        "target": target,
        "accepted_terms": int(accepted),
        "total_terms": int(total),
        "accepted_fraction": float(accepted / total) if total else 0.0,
        "terms": rows,
    }


@dataclass(frozen=True)
class PolynomialPointGenerator:
    """Low-degree point generator used by polynomial basis probing."""

    name: str
    family: str
    xi_terms: tuple[tuple[float, int, int], ...] = ()
    eta_terms: tuple[tuple[float, int, int], ...] = ()
    source: str = "polynomial_generator_basis_probe"
    description: str = "polynomial point generator from bounded basis probe"

    def _poly(self, terms: Sequence[tuple[float, int, int]], x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for c, px, pu in terms:
            out = out + float(c) * x.pow(int(px)) * u.pow(int(pu))
        return out

    def _partials(self, terms: Sequence[tuple[float, int, int]], x: torch.Tensor, u: torch.Tensor):
        f = torch.zeros_like(x)
        fx = torch.zeros_like(x)
        fu = torch.zeros_like(x)
        fxx = torch.zeros_like(x)
        fxu = torch.zeros_like(x)
        fuu = torch.zeros_like(x)
        for c, px, pu in terms:
            c = float(c)
            px = int(px)
            pu = int(pu)
            f = f + c * x.pow(px) * u.pow(pu)
            if px > 0:
                fx = fx + c * px * x.pow(px - 1) * u.pow(pu)
            if pu > 0:
                fu = fu + c * pu * x.pow(px) * u.pow(pu - 1)
            if px > 1:
                fxx = fxx + c * px * (px - 1) * x.pow(px - 2) * u.pow(pu)
            if px > 0 and pu > 0:
                fxu = fxu + c * px * pu * x.pow(px - 1) * u.pow(pu - 1)
            if pu > 1:
                fuu = fuu + c * pu * (pu - 1) * x.pow(px) * u.pow(pu - 2)
        return f, fx, fu, fxx, fxu, fuu

    def fields(self, x: torch.Tensor, u: torch.Tensor, u1: torch.Tensor, u2: torch.Tensor):
        xi, xi_x, xi_u, xi_xx, xi_xu, xi_uu = self._partials(self.xi_terms, x, u)
        eta, eta_x, eta_u, eta_xx, eta_xu, eta_uu = self._partials(self.eta_terms, x, u)
        eta1 = eta_x + eta_u * u1 - xi_x * u1 - xi_u * u1.pow(2)
        eta1_x = eta_xx + eta_xu * u1 - xi_xx * u1 - xi_xu * u1.pow(2)
        eta1_u = eta_xu + eta_uu * u1 - xi_xu * u1 - xi_uu * u1.pow(2)
        eta2 = eta1_x + u1 * eta1_u + u2 * (eta_u - 2.0 * xi_x - 3.0 * xi_u * u1)
        return xi, eta, eta1, eta2

    def _term_repr(self, terms: Sequence[tuple[float, int, int]]) -> str:
        pieces = []
        for c, px, pu in terms:
            factors = []
            if px:
                factors.append("x" if px == 1 else f"x^{px}")
            if pu:
                factors.append("u" if pu == 1 else f"u^{pu}")
            body = "*".join(factors) if factors else "1"
            pieces.append(f"{float(c):g}*{body}")
        return " + ".join(pieces) if pieces else "0"

    def to_report(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "generator_type": "point_lie_polynomial_xu",
            "source": self.source,
            "xi": self._term_repr(self.xi_terms),
            "eta": self._term_repr(self.eta_terms),
            "xi_terms": [tuple(v) for v in self.xi_terms],
            "eta_terms": [tuple(v) for v in self.eta_terms],
            "description": self.description,
        }


def polynomial_generator_basis(max_degree: int = 2) -> list[PolynomialPointGenerator]:
    """Return a monomial basis for xi/eta polynomial point generators."""

    max_degree = max(1, int(max_degree))
    monoms: list[tuple[int, int]] = []
    for total in range(0, max_degree + 1):
        for px in range(total, -1, -1):
            pu = total - px
            monoms.append((px, pu))
    gens: list[PolynomialPointGenerator] = []
    for px, pu in monoms:
        label = "1" if (px, pu) == (0, 0) else ("x" if (px, pu) == (1, 0) else ("u" if (px, pu) == (0, 1) else f"x{px}_u{pu}"))
        gens.append(PolynomialPointGenerator(f"det_xi_{label}", "determining_polynomial", xi_terms=((1.0, px, pu),)))
        gens.append(PolynomialPointGenerator(f"det_eta_{label}", "determining_polynomial", eta_terms=((1.0, px, pu),)))
    return gens


def probe_polynomial_point_generator_basis(
    *,
    order: int,
    x: torch.Tensor,
    u: torch.Tensor,
    u1: torch.Tensor,
    u2: torch.Tensor | None,
    term_asts: Sequence[NodeLike | None],
    coeffs: torch.Tensor,
    x_axis: int = 0,
    max_degree: int = 2,
    tol: float = 0.05,
    max_generators: int = 4,
) -> dict[str, Any]:
    """Probe a bounded polynomial generator basis.

    This does not solve a coupled determining-equation nullspace. It scores
    individual low-degree polynomial point generators against the fitted
    equation and reports accepted basis rows.
    """

    from nestynet_sr.sr_gs.prolongation import score_affine_point_generators_from_jets

    basis = polynomial_generator_basis(max_degree=max_degree)
    scored = score_affine_point_generators_from_jets(
        order=order,
        x=x,
        u=u,
        u1=u1,
        u2=u2,
        term_asts=term_asts,
        coeffs=coeffs,
        x_axis=x_axis,
        generators=basis,
        include_known=False,
        include_general_affine=False,
        tol=tol,
    )
    accepted = [row for row in scored.get("generators", []) if bool(row.get("accepted", False))]
    accepted = accepted[: max(0, int(max_generators))]
    return {
        "enabled": True,
        "status": "scored",
        "upgrade": "polynomial_generator_basis_probe",
        "ansatz": "polynomial_point_xu",
        "max_degree": int(max_degree),
        "tol": float(tol),
        "tested_generators": int(scored.get("tested_generators", 0) or 0),
        "sample_compatible_generator_names": [str(row.get("name", "")) for row in accepted],
        "sample_compatible_generators": accepted,
        "accepted_generator_names": [str(row.get("name", "")) for row in accepted],
        "accepted_generators": accepted,
        "best_metric": scored.get("best_metric"),
        "best_generator": scored.get("best_generator"),
    }


def score_autonomous_even_velocity_prior(term_asts: Sequence[NodeLike | None]) -> dict[str, Any]:
    rows = []
    accepted = 0
    for term in term_asts:
        sig = ast_discrete_signature(term)
        known = _signature_known(sig)
        autonomous = bool(known and int(sig["x"]) == 0)
        time_reversal_ok = bool(known and int(sig["du"]) % 2 == 0)
        ok = autonomous and time_reversal_ok
        accepted += int(ok)
        rows.append({
            "term": "1" if term is None else repr(term),
            "signature": sig,
            "signature_status": str(sig.get("status", "known")),
            "autonomous": bool(autonomous),
            "time_reversal_ok": bool(time_reversal_ok),
            "accepted": bool(ok),
        })
    total = len(rows)
    return {
        "family": "autonomous_even_velocity_prior",
        "criterion": "autonomous_even_velocity",
        "accepted_terms": int(accepted),
        "total_terms": int(total),
        "accepted_fraction": float(accepted / total) if total else 0.0,
        "terms": rows,
    }


def contact_jet_de_rows(cfg: Any, *, order: int) -> list[tuple[NodeLike, str, str]]:
    """Backward-compatible alias for :func:`velocity_monomial_prior_rows`."""

    return velocity_monomial_prior_rows(cfg, order=order)


def noether_variational_de_rows(cfg: Any, *, order: int) -> list[tuple[NodeLike, str, str]]:
    """Backward-compatible alias for autonomous/even-velocity prior rows."""

    return autonomous_even_velocity_prior_rows(cfg, order=order)


def discover_determining_equation_generators_from_jets(*args, **kwargs) -> dict[str, Any]:
    """Backward-compatible alias for polynomial generator-basis probing."""

    return probe_polynomial_point_generator_basis(*args, **kwargs)


def score_noether_candidate(term_asts: Sequence[NodeLike | None]) -> dict[str, Any]:
    """Backward-compatible alias for autonomous/even-velocity prior scoring."""

    return score_autonomous_even_velocity_prior(term_asts)
