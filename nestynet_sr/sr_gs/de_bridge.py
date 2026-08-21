# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""DE-library hooks for GS diagnostics and explicit structural DE priors."""

from __future__ import annotations

from typing import Any

from nestynet_sr.sr_core.bridges import DU, U, Var, Mul, Pow
from .reporting import record_de_terms, record_unit_torus_event


def _canonical_dim_policy(cfg: Any) -> str:
    p = str(getattr(cfg, "gs_dim_policy", getattr(cfg, "dim_policy", "audit")) or "audit").strip().lower().replace("_", "-")
    aliases = {"rref": "baseline", "baseline-only": "baseline", "report": "audit", "gs": "gs-only", "gsonly": "gs-only", "replace": "replace-rref"}
    p = aliases.get(p, p)
    if p not in {"baseline", "audit", "augment", "both", "replace-rref", "gs-only"}:
        p = "audit"
    return p


def _unit_torus_active(cfg: Any) -> bool:
    return (
        bool(getattr(cfg, "gs_enable", False))
        and str(getattr(cfg, "gs_mode", "propose") or "propose").lower() != "off"
        and bool(getattr(cfg, "gs_unit_torus", False))
        and _canonical_dim_policy(cfg) != "baseline"
    )


def _unit_torus_proposes(cfg: Any) -> bool:
    mode = str(getattr(cfg, "gs_mode", "propose") or "propose").lower()
    return _unit_torus_active(cfg) and mode in {"propose", "auto"} and _canonical_dim_policy(cfg) in {"augment", "both", "replace-rref", "gs-only"}


def _append_unique_row(rows: list[tuple[Any, str, str]], term: Any, source: str, family: str) -> None:
    rep = repr(term)
    for old, _source, _family in rows:
        if repr(old) == rep:
            return
    rows.append((term, source, family))


def hard_tail_de_term_rows(cfg: Any, *, order: int) -> list[tuple[Any, str, str]]:
    """Return explicit hard-tail structural-prior DE rows.

    These rows are not evidence of discovered symmetry. They are exposed only
    through the neutral DE-library prior path. Legacy GS-named CLI aliases are
    normalized into the ``de_hard_tail_*`` fields before this bridge is called.
    """

    enabled = bool(getattr(cfg, "de_hard_tail_templates", False))
    if not enabled:
        return []

    xj = Var(int(getattr(cfg, "x_axis", 0)))
    u = U()
    du = DU(int(getattr(cfg, "x_axis", 0)))
    rows: list[tuple[Any, str, str]] = []

    radial_enabled = bool(getattr(cfg, "de_hard_tail_radial_templates", True))
    velocity_enabled = bool(getattr(cfg, "de_hard_tail_velocity_templates", False))

    if radial_enabled:
        # Value-like singular/radial remnants. These are safe for first-order
        # equations and target cases such as y'=-y/x.
        for term in (
            Mul(Pow(xj, -1), u),
            Mul(Pow(xj, -2), u),
            Mul(Pow(xj, 2), u),
        ):
            _append_unique_row(rows, term, "de_prior_hard_tail", "radial_value_prior")
        if int(order) >= 2:
            # Derivative-carrier radial terms are meaningful for second-order
            # Lane-Emden/Bessel/spherical-wave style equations, but were found
            # to destabilize first-order sparse RHS validation when offered too
            # broadly.
            for term in (
                Mul(Pow(xj, -1), du),
                Mul(xj, du),
            ):
                _append_unique_row(rows, term, "de_prior_hard_tail", "radial_derivative_prior")

    if velocity_enabled and int(order) >= 2:
        for term in (
            Mul(u, Pow(du, 2)),
            Mul(u, Pow(du, 4)),
            Mul(Pow(u, 2), du),
        ):
            _append_unique_row(rows, term, "de_prior_hard_tail", "velocity_prior")

    return rows


def _unit_torus_de_rows(cfg: Any, *, order: int) -> list[tuple[Any, str, str]]:
    if not _unit_torus_active(cfg):
        return []
    units_spec = getattr(cfg, "units_spec", None)
    if units_spec is None:
        try:
            record_unit_torus_event(
                event_type="de_unit_torus",
                diagnostics=[{"family": "unit_torus", "kind": "de_prefactor", "accepted": False, "reason": "units_spec_missing"}],
                context={"dim_policy": _canonical_dim_policy(cfg), "order": int(order), "status": "skipped"},
            )
        except Exception:
            pass
        return []

    try:
        from nestynet_sr.sr_core.units import sub_dim
        from .unit_torus import build_monomial_ast, enumerate_prefactor_exponents, exponent_l1
    except Exception as exc:
        try:
            record_unit_torus_event(
                event_type="de_unit_torus",
                diagnostics=[{"family": "unit_torus", "kind": "de_prefactor", "accepted": False, "reason": str(exc)}],
                context={"dim_policy": _canonical_dim_policy(cfg), "order": int(order), "status": "failed"},
            )
        except Exception:
            pass
        return []

    try:
        x_axis = int(getattr(cfg, "x_axis", 0))
        x_dim = units_spec.x_dims[x_axis]
        y_dim = units_spec.y_phi_dim
        anchor_dim = sub_dim(y_dim, x_dim)
        if int(order) == 2:
            anchor_dim = sub_dim(anchor_dim, x_dim)
        elif int(order) != 1:
            return []
        xj = Var(x_axis)
        u = U()
        du = DU(x_axis)
        variable_dims = [x_dim, y_dim]
        variable_nodes = [xj, u]
        if int(order) >= 2:
            variable_dims.append(sub_dim(y_dim, x_dim))
            variable_nodes.append(du)
        exponents = enumerate_prefactor_exponents(
            variable_dims,
            anchor_dim,
            max_exponent=int(getattr(cfg, "gs_pi_max_exponent", getattr(cfg, "pi_max_exponent", 3))),
            max_l1=int(getattr(cfg, "gs_pi_max_l1", getattr(cfg, "pi_max_l1", 6))),
            max_proposals=int(getattr(cfg, "gs_pi_max_proposals", getattr(cfg, "pi_max_proposals", 24))),
            max_basis=int(getattr(cfg, "gs_pi_max_basis", getattr(cfg, "pi_max_basis", 8))),
            rational_denom=int(getattr(cfg, "gs_pi_rational_denom", getattr(cfg, "pi_rational_denom", 1))),
        )
    except Exception as exc:
        try:
            record_unit_torus_event(
                event_type="de_unit_torus",
                diagnostics=[{"family": "unit_torus", "kind": "de_prefactor", "accepted": False, "reason": str(exc)}],
                context={"dim_policy": _canonical_dim_policy(cfg), "order": int(order), "status": "failed"},
            )
        except Exception:
            pass
        return []

    diagnostics = []
    rows: list[tuple[Any, str, str]] = []
    for q in exponents:
        try:
            term = build_monomial_ast(q, variable_nodes=variable_nodes)
        except Exception:
            continue
        l1 = float(exponent_l1(q))
        diagnostics.append(
            {
                "family": "unit_torus",
                "kind": "de_prefactor",
                "accepted": True,
                "candidate": repr(term),
                "exponents": tuple(str(v) for v in q),
                "l1": l1,
            }
        )
        if _unit_torus_proposes(cfg):
            _append_unique_row(rows, term, "gs_unit_torus", "unit_torus_prefactor")

    try:
        record_unit_torus_event(
            event_type="de_unit_torus",
            diagnostics=diagnostics,
            context={
                "dim_policy": _canonical_dim_policy(cfg),
                "validator": str(getattr(cfg, "gs_dim_validator", getattr(cfg, "dim_validator", "nullspace"))),
                "order": int(order),
                "proposing": bool(_unit_torus_proposes(cfg)),
                "terms_emitted": int(len(rows)),
            },
        )
    except Exception:
        pass
    return rows


def nonlinear_invariant_de_term_rows(
    compilation: Any,
    *,
    orbit_coordinate: Any = None,
    x_axis: int = 0,
) -> list[tuple[Any, str, str]]:
    """Convert certified nonlinear carriers into source-aware DE rows.

    This is the explicit second-pass bridge: callers may run the nonlinear
    determining/invariant compiler on a first candidate, stash its result on
    ``cfg.gs_de_compiled_nonlinear_invariants``, and rebuild the DE dictionary.
    Only accepted carriers with concrete ASTs cross this boundary.
    """

    rows: list[tuple[Any, str, str]] = []
    from .de_invariant_compiler import point_coordinate_ast_to_de_ast

    for candidate in list(getattr(compilation, "invariants", ()) or ()):
        if not bool(getattr(candidate, "accepted", False)):
            continue
        term = getattr(candidate, "ast", None)
        if term is not None:
            _append_unique_row(
                rows,
                point_coordinate_ast_to_de_ast(term, x_axis=int(x_axis)),
                "gs_nonlinear_invariant",
                "nonlinear_point_invariant",
            )
    bundled_orbits = list(getattr(compilation, "orbit_coordinates", ()) or ())
    if orbit_coordinate is not None:
        bundled_orbits.append(orbit_coordinate)
    for orbit in bundled_orbits:
        if not bool(getattr(orbit, "accepted", False)):
            continue
        term = getattr(orbit, "ast", None)
        if term is not None:
            _append_unique_row(
                rows,
                point_coordinate_ast_to_de_ast(term, x_axis=int(x_axis)),
                "gs_nonlinear_orbit_coordinate",
                "nonlinear_orbit_coordinate",
            )
    return rows


def generalized_symmetry_de_term_rows(
    cfg: Any,
    *,
    order: int,
    generators: Any = None,
    compiled_invariants: Any = None,
    orbit_coordinate: Any = None,
) -> list[tuple[Any, str, str]]:
    """Return source-aware DE terms suggested by GS families.

    ``generators`` optionally carries *discovered* symmetry generators (a
    sequence of ``RecoveredDEGenerator`` or a ``DEDeterminingResult``); when
    provided, the differential-invariant library is compiled from them instead
    of the configured seed generators.
    """

    if not bool(getattr(cfg, "gs_enable", False)):
        return []

    rows: list[tuple[Any, str, str]] = []
    try:
        from .de_upgrades import symmetry_upgrade_de_term_rows
        for term, source, family in symmetry_upgrade_de_term_rows(cfg, order=order):
            _append_unique_row(rows, term, source, family)
    except Exception as exc:
        try:
            record_de_terms(terms=[], context={"order": int(order), "upgrade_error": str(exc)[:300]})
        except Exception:
            pass
    # symmetry-reduction rows: pulled-back compositional terms stashed on the
    # config by a pre-search reduction pass (see sr_gs.de_reduction); they
    # augment the STLSQ dictionary and seed the factorized search.
    for item in list(getattr(cfg, "gs_de_reduction_rows", None) or []):
        try:
            term, source, family = item
        except Exception:
            continue
        _append_unique_row(rows, term, str(source), str(family))
    compiled = (
        compiled_invariants
        if compiled_invariants is not None
        else getattr(cfg, "gs_de_compiled_nonlinear_invariants", None)
    )
    orbit = (
        orbit_coordinate
        if orbit_coordinate is not None
        else getattr(cfg, "gs_de_compiled_orbit_coordinate", None)
    )
    if compiled is not None or orbit is not None:
        for term, source, family in nonlinear_invariant_de_term_rows(
            compiled,
            orbit_coordinate=orbit,
            x_axis=int(getattr(cfg, "x_axis", 0)),
        ):
            _append_unique_row(rows, term, source, family)
    try:
        from .de_invariants import de_invariant_library_rows
        for term, source, family in de_invariant_library_rows(cfg, order=order, generators=generators):
            _append_unique_row(rows, term, source, family)
    except Exception as exc:
        try:
            record_de_terms(terms=[], context={"order": int(order), "invariant_library_error": str(exc)[:300]})
        except Exception:
            pass
    for term, source, family in _unit_torus_de_rows(cfg, order=order):
        _append_unique_row(rows, term, source, family)

    if rows:
        try:
            record_de_terms(
                terms=[r[0] for r in rows],
                context={
                    "order": int(order),
                    "policy": str(getattr(cfg, "gs_policy", getattr(cfg, "policy", "augment"))),
                    "dim_policy": _canonical_dim_policy(cfg),
                    "hard_tail_radial_templates": bool(
                        getattr(cfg, "de_hard_tail_radial_templates", True)
                    ),
                    "hard_tail_velocity_templates": bool(
                        getattr(cfg, "de_hard_tail_velocity_templates", False)
                    ),
                    "sources": [{"term": repr(t), "source": s, "family": f} for t, s, f in rows],
                },
            )
        except Exception:
            pass
    return rows


def generalized_symmetry_de_terms(cfg: Any, *, order: int) -> list[Any]:
    """Compatibility wrapper returning only DE term ASTs."""

    return [row[0] for row in generalized_symmetry_de_term_rows(cfg, order=order)]
