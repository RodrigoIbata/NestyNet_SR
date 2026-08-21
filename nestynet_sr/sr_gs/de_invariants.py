# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Conservative differential-invariant rows for scalar DE libraries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from nestynet_sr.sr_core.bridges import DU, U, Mul, Pow, Var, ast_to_human_readable

from .de_determining import DEDeterminingResult, RecoveredDEGenerator
from .jet_bundle import JetSpaceSpec


NodeLike = Any


@dataclass(frozen=True)
class DEInvariantRow:
    """One symmetry-derived library row with provenance."""

    term: NodeLike
    source: str
    family: str
    generator_name: str
    invariant_kind: str
    order: int
    description: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_library_row(self) -> tuple[NodeLike, str, str]:
        return self.term, self.source, self.family

    def to_report(self) -> dict[str, Any]:
        return {
            "term": repr(self.term),
            "human": _human(self.term),
            "source": self.source,
            "family": self.family,
            "generator_name": self.generator_name,
            "invariant_kind": self.invariant_kind,
            "order": int(self.order),
            "description": self.description,
            "provenance": dict(self.provenance),
        }


def compile_de_invariant_library(
    *,
    jet_space: JetSpaceSpec | None = None,
    generators: Sequence[RecoveredDEGenerator] | DEDeterminingResult | None = None,
    order: int = 1,
    x_axis: int = 0,
    cfg: Any = None,
) -> list[DEInvariantRow]:
    """Compile accepted scalar point symmetries into DE library rows."""

    if jet_space is None:
        jet_space = JetSpaceSpec(independent=("x",), dependent=("u",), max_order=max(1, min(2, int(order))))
    jet_space.require_scalar_ode_phase_one()
    order_i = int(order)
    if order_i not in (1, 2):
        raise NotImplementedError(f"differential-invariant library supports scalar ODE order 1 or 2; got {order_i}")
    accepted = _accepted_generators(generators)
    if generators is None:
        accepted = _default_seed_generators(cfg)
    rows: list[DEInvariantRow] = []
    for gen in accepted:
        for row in _rows_for_generator(gen, order=order_i, x_axis=int(x_axis)):
            _append_unique(rows, row)
    return _bounded(rows, cfg)


def de_invariant_library_rows(
    cfg: Any,
    *,
    order: int,
    generators: Sequence[RecoveredDEGenerator] | DEDeterminingResult | None = None,
    jet_space: JetSpaceSpec | None = None,
) -> list[tuple[NodeLike, str, str]]:
    """Compatibility wrapper returning rows for DE search."""

    if not _enabled(cfg):
        return []
    rows = compile_de_invariant_library(
        jet_space=jet_space,
        generators=generators,
        order=int(order),
        x_axis=int(getattr(cfg, "x_axis", 0)),
        cfg=cfg,
    )
    return [row.as_library_row() for row in rows]


def de_invariant_library_report(
    cfg: Any,
    *,
    order: int,
    generators: Sequence[RecoveredDEGenerator] | DEDeterminingResult | None = None,
    jet_space: JetSpaceSpec | None = None,
) -> dict[str, Any]:
    rows = compile_de_invariant_library(
        jet_space=jet_space,
        generators=generators,
        order=int(order),
        x_axis=int(getattr(cfg, "x_axis", 0)),
        cfg=cfg,
    )
    return {
        "enabled": bool(_enabled(cfg)),
        "order": int(order),
        "scalar_ode_only": True,
        "rows": [row.to_report() for row in rows],
    }


def _rows_for_generator(gen: RecoveredDEGenerator, *, order: int, x_axis: int) -> list[DEInvariantRow]:
    x = Var(int(x_axis))
    u = U()
    du = DU(int(x_axis))
    name = str(gen.name)
    rows: list[DEInvariantRow] = []
    if name in {"d_x", "x_translation"}:
        rows.extend(
            [
                _row(u, gen, order, "autonomous_state", "autonomous coordinate invariant", "u"),
                _row(du, gen, order, "autonomous_derivative", "autonomous first derivative invariant", "u_x"),
            ]
        )
        if order >= 2:
            rows.extend(
                [
                    _row(Pow(du, 2), gen, order, "autonomous_derivative_power", "even velocity invariant", "u_x^2"),
                    _row(Mul(u, Pow(du, 2)), gen, order, "autonomous_state_derivative", "state times even velocity invariant", "u*u_x^2"),
                ]
            )
        return rows

    if name in {"u_d_u", "u_scaling"}:
        rows.extend(
            [
                _row(Mul(du, Pow(u, -1)), gen, order, "output_scaling_log_derivative", "u_x/u", "u_x/u"),
                _row(Mul(x, Mul(du, Pow(u, -1))), gen, order, "output_scaling_weighted_log_derivative", "x*u_x/u", "x*u_x/u"),
            ]
        )
        return rows

    if name in {"x_d_x", "x_scaling"}:
        rows.extend(
            [
                _row(u, gen, order, "domain_scaling_state", "u is invariant under x scaling", "u"),
                _row(Mul(x, du), gen, order, "domain_scaling_derivative", "x*u_x", "x*u_x"),
            ]
        )
        if order >= 2:
            rows.append(_row(Mul(Pow(x, -1), du), gen, order, "radial_derivative_shape", "u_x/x radial shape", "u_x/x"))
        return rows

    if name in {"xu_common_scaling"}:
        rows.extend(
            [
                _row(Mul(u, Pow(x, -1)), gen, order, "common_scaling_coordinate", "u/x", "u/x"),
                _row(du, gen, order, "common_scaling_derivative", "u_x", "u_x"),
            ]
        )
        return rows

    if name in {"xu_opposite_scaling"}:
        rows.extend(
            [
                _row(Mul(x, u), gen, order, "opposite_scaling_coordinate", "x*u", "x*u"),
                _row(Mul(Pow(x, 2), du), gen, order, "opposite_scaling_derivative", "x^2*u_x", "x^2*u_x"),
            ]
        )
    return rows


def _row(term: NodeLike, gen: RecoveredDEGenerator, order: int, family: str, description: str, invariant_kind: str) -> DEInvariantRow:
    return DEInvariantRow(
        term=term,
        source="gs_de_differential_invariant",
        family=family,
        generator_name=str(gen.name),
        invariant_kind=str(invariant_kind),
        order=int(order),
        description=str(description),
        provenance={
            "generator_family": str(gen.family),
            "generator_source": str(getattr(gen, "source", "unknown")),
            "multiplier": float(getattr(gen, "multiplier", 0.0)),
            "on_shell_residual_rel": float(getattr(gen, "on_shell_residual_rel", 0.0)),
            "off_shell_relative_residual_rel": float(getattr(gen, "off_shell_relative_residual_rel", 0.0)),
        },
    )


def _accepted_generators(generators: Sequence[RecoveredDEGenerator] | DEDeterminingResult | None) -> list[RecoveredDEGenerator]:
    if generators is None:
        return []
    if isinstance(generators, DEDeterminingResult):
        raw = generators.generators
    else:
        raw = generators
    return [gen for gen in raw if bool(getattr(gen, "accepted", True))]


def _default_seed_generators(cfg: Any) -> list[RecoveredDEGenerator]:
    raw_names = getattr(cfg, "gs_de_invariant_seed_generators", getattr(cfg, "de_invariant_seed_generators", None))
    if raw_names is None:
        raw_names = ("d_x", "u_d_u", "x_d_x")
    if isinstance(raw_names, str):
        names = tuple(item.strip() for item in raw_names.replace(";", ",").split(",") if item.strip())
    else:
        names = tuple(str(item) for item in raw_names)
    family_by_name = {
        "d_x": "translation",
        "x_translation": "translation",
        "u_d_u": "scaling",
        "u_scaling": "scaling",
        "x_d_x": "scaling",
        "x_scaling": "scaling",
        "xu_common_scaling": "scaling",
        "xu_opposite_scaling": "scaling",
    }
    return [
        RecoveredDEGenerator(
            name=name,
            family=family_by_name.get(name, "point_symmetry"),
            coefficients=(),
            multiplier=0.0,
            on_shell_residual_rel=0.0,
            off_shell_relative_residual_rel=0.0,
            accepted=True,
            source="configured_seed",
        )
        for name in names
    ]


def _enabled(cfg: Any) -> bool:
    gs_enabled = bool(getattr(cfg, "gs_enable", getattr(cfg, "enabled", False)))
    return gs_enabled and (
        bool(getattr(cfg, "gs_de_all_upgrades", getattr(cfg, "de_all_upgrades", False)))
        or bool(getattr(cfg, "gs_de_invariant_library", getattr(cfg, "de_invariant_library", False)))
    )


def _bounded(rows: list[DEInvariantRow], cfg: Any) -> list[DEInvariantRow]:
    try:
        limit = int(
            getattr(
                cfg,
                "gs_de_invariant_max_terms",
                getattr(cfg, "de_invariant_max_terms", getattr(cfg, "gs_de_upgrade_max_terms", getattr(cfg, "de_upgrade_max_terms", 64))),
            )
        )
    except Exception:
        limit = 64
    if limit <= 0:
        return rows
    return rows[:limit]


def _append_unique(rows: list[DEInvariantRow], row: DEInvariantRow) -> None:
    rep = repr(row.term)
    if any(repr(old.term) == rep for old in rows):
        return
    rows.append(row)


def _human(term: NodeLike) -> str:
    try:
        return ast_to_human_readable(term)
    except Exception:
        return repr(term)


__all__ = [
    "DEInvariantRow",
    "compile_de_invariant_library",
    "de_invariant_library_report",
    "de_invariant_library_rows",
]
