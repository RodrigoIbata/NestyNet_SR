# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Units-aware phase-coordinate prescan.

This module is deliberately only a hint generator.  It proposes dimensionless
phase carriers such as ``x0*x1/x2`` and scores whether ``y`` is predictable from
``omega*z mod 2*pi`` using a small held-out Fourier fit.  Stage A/B still have
to confirm any resulting symbolic move through the normal validation path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
from itertools import combinations, product
import math
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from nestynet_sr.sr_core.bridges import (
    AddNode,
    ConstNode,
    CosNode,
    MulNode,
    Node,
    PowNode,
    SinNode,
    Var,
    ast_to_human_readable,
    eval_input_expr,
)

try:  # pragma: no cover - units module is always present in the package.
    from nestynet_sr.sr_core.units import is_dimless
except Exception:  # pragma: no cover
    is_dimless = None  # type: ignore


@dataclass(frozen=True)
class PhaseScanHyperparams:
    """Small compute controls for the phase-coordinate prescan."""

    enabled: bool = True
    sample_size: int = 4096
    max_support: int = 3
    max_candidates: int = 96
    max_candidates_per_support: int = 32
    max_exp_l1: float = 4.0
    min_domain_frac: float = 0.98
    max_harmonic: int = 3
    fft_grid_size: int = 384
    fft_top_k: int = 8
    log_top_k: int = 6
    random_seed: int = 1234
    context_enabled: bool = True
    context_max_features: int = 8
    context_log_top_k: int = 6


@dataclass(frozen=True)
class PhaseCarrier:
    """A candidate dimensionless carrier ``z(X)``."""

    ast: Node
    label: str
    exponents: tuple[tuple[int, Fraction], ...]
    support: tuple[int, ...]
    cost: float
    dim: Any = None
    unit_status: str = "unchecked"


@dataclass(frozen=True)
class OmegaTerm:
    """Explicit harmonic/family interpretation of one phase frequency."""

    base_omega: float
    harmonic: int
    actual_omega: float
    energy: float
    family: str


@dataclass(frozen=True)
class PhaseHint:
    """A scored phase-coordinate hint."""

    carrier_ast: Node
    carrier_label: str
    phase_family: str
    observed_omega: float | None
    carrier_omega_candidates: tuple[float, ...]
    waveform_family: str
    envelope_family: str
    score: float
    confidence: float
    r2_phase: float
    r2_trend: float
    support_fraction: float
    n_cycles: float
    unit_status: str
    details: dict[str, Any] = field(default_factory=dict)
    omega_terms: tuple[OmegaTerm, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PhaseContextHint:
    """A future contextual phase hint, currently used as a planning type."""

    carrier_ast: Node
    carrier_label: str
    phase_family: str
    omega_terms: tuple[OmegaTerm, ...]
    context_asts: tuple[Node, ...]
    coupling_mode: str
    waveform_family: str
    r2_context_only: float
    r2_context_phase: float
    delta_r2_phase: float
    confidence: float
    unit_status: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OuterLinkHint:
    """Inside-out inverse-link hint for targets like ``asin(u(X))``."""

    link_name: str
    transform_name: str
    carrier_ast: Node
    carrier_label: str
    affine_a: float
    affine_b: float
    rms_rel: float
    r2: float
    domain_ok_frac: float
    branch_ok_frac: float
    confidence: float
    unit_status: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ContextFeature:
    ast: Node | None
    label: str
    values: np.ndarray
    cost: float
    dim: Any = None
    unit_status: str = "unchecked"


_EXPONENTS: tuple[Fraction, ...] = (
    Fraction(-2, 1),
    Fraction(-1, 1),
    Fraction(-1, 2),
    Fraction(1, 2),
    Fraction(1, 1),
    Fraction(2, 1),
)

_PHYSICS_OMEGA_SEEDS: tuple[float, ...] = (
    math.pi / 2.0,
    math.pi,
    2.0 * math.pi,
    4.0 * math.pi,
    8.0 * math.pi,
)


def stable_int_hash(*parts: Any) -> int:
    """Return a process-stable signed 63-bit integer for candidate signatures."""

    text = "|".join(str(p) for p in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    val = int.from_bytes(digest, byteorder="big", signed=False)
    return int(val & ((1 << 63) - 1))


def _dim_zero_like(x_dims: Sequence[Any]) -> Any:
    if not x_dims:
        return tuple()
    return tuple(Fraction(0) for _ in x_dims[0])


def _scale_dim(dim: Any, exp: Fraction) -> Any:
    return tuple(Fraction(v) * exp for v in dim)


def _add_dim(a: Any, b: Any) -> Any:
    return tuple(Fraction(x) + Fraction(y) for x, y in zip(a, b))


def _carrier_dim(exponents: tuple[tuple[int, Fraction], ...], x_dims: Sequence[Any]) -> Any:
    out = _dim_zero_like(x_dims)
    for idx, exp in exponents:
        out = _add_dim(out, _scale_dim(x_dims[int(idx)], exp))
    return out


def _dimless(dim: Any) -> bool:
    if is_dimless is not None:
        try:
            return bool(is_dimless(dim))
        except Exception:
            pass
    try:
        return all(Fraction(v) == 0 for v in dim)
    except Exception:
        return False


def _pow_if_needed(node: Node, exp: Fraction) -> Node:
    if exp == 1:
        return node
    return PowNode(node, float(exp))


def _mul_chain(nodes: Sequence[Node]) -> Node:
    if not nodes:
        raise ValueError("phase carrier needs at least one factor")
    cur = nodes[0]
    for node in nodes[1:]:
        cur = MulNode(cur, node)
    return cur


def _carrier_ast(exponents: tuple[tuple[int, Fraction], ...]) -> Node:
    factors = [_pow_if_needed(Var(int(idx)), exp) for idx, exp in exponents]
    return _mul_chain(factors)


def _carrier_cost(exponents: tuple[tuple[int, Fraction], ...]) -> float:
    support = len(exponents)
    l1 = sum(float(abs(exp)) for _, exp in exponents)
    half_penalty = sum(1.0 for _, exp in exponents if exp.denominator != 1)
    return float(support + l1 + half_penalty)


def build_unit_valid_phase_carriers(
    Nxvars: int,
    *,
    units_payload: dict[str, Any] | None = None,
    ignore_units: bool = False,
    hp: PhaseScanHyperparams | None = None,
) -> list[PhaseCarrier]:
    """Enumerate sparse monomial phase carriers.

    Under strict units, only dimensionless carriers are returned.  Without a
    unit system this remains a bounded structural proposal generator.
    """

    hp = hp or PhaseScanHyperparams()
    Nx = int(Nxvars)
    x_dims = None
    if (not ignore_units) and isinstance(units_payload, dict):
        x_dims = units_payload.get("x_dims", None)
    if x_dims is not None:
        x_dims = tuple(x_dims)
        if len(x_dims) != Nx:
            x_dims = None

    by_support: dict[int, list[PhaseCarrier]] = {}
    max_support = max(1, min(int(hp.max_support), Nx))
    max_l1 = float(hp.max_exp_l1)

    for support_size in range(1, max_support + 1):
        rows: list[PhaseCarrier] = []
        for support in combinations(range(Nx), support_size):
            for exps in product(_EXPONENTS, repeat=support_size):
                if sum(float(abs(e)) for e in exps) > max_l1:
                    continue
                exponent_rows = tuple((int(i), Fraction(e)) for i, e in zip(support, exps))
                dim = None
                unit_status = "unchecked"
                if x_dims is not None:
                    dim = _carrier_dim(exponent_rows, x_dims)
                    if not _dimless(dim):
                        continue
                    unit_status = "dimensionless"
                ast = _carrier_ast(exponent_rows)
                rows.append(
                    PhaseCarrier(
                        ast=ast,
                        label=ast_to_human_readable(ast),
                        exponents=exponent_rows,
                        support=tuple(int(i) for i in support),
                        cost=_carrier_cost(exponent_rows),
                        dim=dim,
                        unit_status=unit_status,
                    )
                )
        rows.sort(key=lambda c: (float(c.cost), len(c.support), c.label))
        by_support[support_size] = rows[: max(0, int(hp.max_candidates_per_support))]

    out: list[PhaseCarrier] = []
    seen: set[str] = set()
    for support_size in range(1, max_support + 1):
        for carrier in by_support.get(support_size, []):
            if carrier.label in seen:
                continue
            seen.add(carrier.label)
            out.append(carrier)

    return out[: max(0, int(hp.max_candidates))]


def _unit_dims_from_payload(
    Nxvars: int,
    *,
    units_payload: dict[str, Any] | None,
    ignore_units: bool,
) -> tuple[Any, ...] | None:
    if ignore_units or not isinstance(units_payload, dict):
        return None
    x_dims = units_payload.get("x_dims", None)
    if x_dims is None:
        return None
    x_dims = tuple(x_dims)
    if len(x_dims) != int(Nxvars):
        return None
    return x_dims


def _axis_is_dimless(axis: int, x_dims: Sequence[Any] | None) -> bool:
    if x_dims is None:
        return True
    try:
        return _dimless(x_dims[int(axis)])
    except Exception:
        return False


def _outer_link_mixed_trig_carriers(
    Nxvars: int,
    *,
    units_payload: dict[str, Any] | None,
    ignore_units: bool,
    hp: PhaseScanHyperparams,
    base_carriers: Sequence[PhaseCarrier],
) -> list[PhaseCarrier]:
    """Build dimensionless ``base*sin(xj)`` / ``base*cos(xj)`` carriers.

    The inverse-trig outer-link scan asks whether ``sin(y)``, ``cos(y)``, or
    ``tan(y)`` is affine in a carrier.  For cases like
    ``asin(x0*sin(x1))``, a monomial-only Buckingham carrier grammar is too
    weak: the natural inverse-link argument already contains a visible trig
    factor.  These carriers are still just hint evidence; Stage B must validate
    the visible inverse-trig closure before anything is accepted.
    """

    Nx = int(Nxvars)
    if Nx <= 0:
        return []
    x_dims = _unit_dims_from_payload(Nx, units_payload=units_payload, ignore_units=ignore_units)
    max_support = max(1, min(int(hp.max_support), Nx))
    # Keep this bounded independently of the monomial carrier budget.  The
    # exact matches sort by confidence later, so this only needs to expose a
    # compact, low-cost mixed family.
    max_rows = max(0, int(hp.max_candidates))
    if max_rows <= 0:
        return []

    rows: list[PhaseCarrier] = []
    for base in list(base_carriers):
        base_support = tuple(int(v) for v in getattr(base, "support", ()) or ())
        if not base_support:
            continue
        if len(base_support) > max_support:
            continue
        if str(getattr(base, "unit_status", "unchecked")) not in {"dimensionless", "unchecked"}:
            continue
        for axis in range(Nx):
            if not _axis_is_dimless(axis, x_dims):
                continue
            support = tuple(sorted(set(base_support) | {int(axis)}))
            if len(support) > max_support:
                continue
            for trig_node in (SinNode, CosNode):
                trig_ast = trig_node(Var(axis))
                ast = MulNode(base.ast, trig_ast)
                label = ast_to_human_readable(ast)
                rows.append(
                    PhaseCarrier(
                        ast=ast,
                        label=label,
                        exponents=tuple(getattr(base, "exponents", ()) or ()),
                        support=support,
                        cost=float(getattr(base, "cost", 0.0)) + 2.0,
                        dim=getattr(base, "dim", None),
                        unit_status=(
                            "dimensionless"
                            if x_dims is not None or str(getattr(base, "unit_status", "")) == "dimensionless"
                            else "unchecked"
                        ),
                    )
                )

    rows.sort(key=lambda c: (float(c.cost), len(c.support), c.label))
    out: list[PhaseCarrier] = []
    seen: set[str] = set()
    for row in rows:
        if row.label in seen:
            continue
        seen.add(row.label)
        out.append(row)
        if len(out) >= max_rows:
            break
    return out


def build_outer_link_phase_carriers(
    Nxvars: int,
    *,
    units_payload: dict[str, Any] | None = None,
    ignore_units: bool = False,
    hp: PhaseScanHyperparams | None = None,
) -> list[PhaseCarrier]:
    """Carrier list for inside-out inverse-link scans.

    This starts with the ordinary sparse dimensionless monomial carriers and
    appends a small visible-trig mixed family.  It is intentionally not used by
    the main PhaseScan periodic score, where ``NN[sin(z)]`` evidence is much
    less decisive.
    """

    hp = hp or PhaseScanHyperparams()
    base = build_unit_valid_phase_carriers(
        Nxvars,
        units_payload=units_payload,
        ignore_units=ignore_units,
        hp=hp,
    )
    mixed = _outer_link_mixed_trig_carriers(
        Nxvars,
        units_payload=units_payload,
        ignore_units=ignore_units,
        hp=hp,
        base_carriers=base,
    )
    out: list[PhaseCarrier] = []
    seen: set[str] = set()
    for carrier in list(base) + list(mixed):
        if carrier.label in seen:
            continue
        seen.add(carrier.label)
        out.append(carrier)
    return out


def _phase_affine_arg_ast(axis: int, omega: float, phase: float) -> Node:
    """Return a visible ``omega*x_axis + phase`` AST with tiny terms dropped."""

    if abs(float(omega) - 1.0) <= 1.0e-12:
        arg: Node = Var(int(axis))
    else:
        arg = MulNode(ConstNode(float(omega)), Var(int(axis)))
    if abs(float(phase)) > 1.0e-12:
        arg = AddNode(arg, ConstNode(float(phase)))
    return arg


def _wrap_phase_for_trig(phase: float) -> float:
    """Normalize a phase to a stable principal interval for display/deduping."""

    two_pi = 2.0 * math.pi
    out = (float(phase) + math.pi) % two_pi - math.pi
    if abs(out + math.pi) < 1.0e-12:
        out = math.pi
    if abs(out) < 1.0e-12:
        out = 0.0
    return float(out)


def _outer_link_axis_omega_seeds(axis_values: np.ndarray) -> list[float]:
    """Small no-FFT omega grid for affine trig factors in outer-link carriers."""

    vals = np.asarray(axis_values, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size < 32:
        return []
    span = float(np.max(vals) - np.min(vals))
    if (not math.isfinite(span)) or span <= 1.0e-12:
        return []
    raw = [
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
        5.0,
    ]
    raw.extend(_PHYSICS_OMEGA_SEEDS)
    # Keep phases with enough variation to be identifiable, while avoiding
    # extremely high-frequency scans in this cheap prescan.
    filtered = [w for w in raw if 0.25 <= float(w) * span <= 12.0 * math.pi]
    return _dedupe_positive(filtered)


def _sample_xy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(int(X.shape[0]), int(y.shape[0]))
    X = X[:n]
    y = y[:n]
    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[finite]
    y = y[finite]
    if X.shape[0] <= int(sample_size):
        return X, y
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(X.shape[0], size=int(sample_size), replace=False)
    idx.sort()
    return X[idx], y[idx]


def _eval_carrier(carrier: PhaseCarrier, X: np.ndarray) -> np.ndarray:
    x_t = torch.as_tensor(np.asarray(X, dtype=np.float64), dtype=torch.float64)
    with torch.no_grad():
        z = eval_input_expr(carrier.ast, x_t)
    return z.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False)


def _fit_r2(design_train: np.ndarray, y_train: np.ndarray, design_val: np.ndarray, y_val: np.ndarray) -> float:
    r2, _ = _fit_linear_model(design_train, y_train, design_val, y_val)
    return float(r2)


def _fit_linear_model(
    design_train: np.ndarray,
    y_train: np.ndarray,
    design_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[float, np.ndarray | None]:
    if design_train.shape[0] <= design_train.shape[1] or design_val.shape[0] <= 1:
        return float("-inf"), None
    try:
        coef, *_ = np.linalg.lstsq(design_train, y_train, rcond=None)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pred = design_val @ coef
    except Exception:
        return float("-inf"), None
    if not np.all(np.isfinite(pred)):
        return float("-inf"), None
    var = float(np.var(y_val))
    if (not math.isfinite(var)) or var <= 1.0e-30:
        return float("-inf"), None
    mse = float(np.mean((pred - y_val) ** 2))
    if not math.isfinite(mse):
        return float("-inf"), None
    return float(1.0 - mse / var), coef


def _fourier_design(z: np.ndarray, omega: float, max_harmonic: int) -> np.ndarray:
    cols = [np.ones_like(z, dtype=np.float64)]
    for k in range(1, int(max_harmonic) + 1):
        arg = float(k) * float(omega) * z
        cols.append(np.cos(arg))
        cols.append(np.sin(arg))
    return np.stack(cols, axis=1)


def _trend_design(z: np.ndarray) -> np.ndarray:
    z0 = z - float(np.mean(z))
    scale = float(np.std(z0))
    if not math.isfinite(scale) or scale <= 1.0e-30:
        scale = 1.0
    u = z0 / scale
    return np.stack([np.ones_like(u), u, u * u], axis=1)


def _standardize_design(design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    D = np.asarray(design, dtype=np.float64)
    scales = np.sqrt(np.mean(D * D, axis=0))
    scales = np.where(np.isfinite(scales) & (scales > 1.0e-30), scales, 1.0)
    return D / scales.reshape(1, -1), scales


def _apply_design_scales(design: np.ndarray, scales: np.ndarray) -> np.ndarray:
    D = np.asarray(design, dtype=np.float64)
    return D / np.asarray(scales, dtype=np.float64).reshape(1, -1)


def _fft_omega_seeds(z: np.ndarray, y: np.ndarray, *, hp: PhaseScanHyperparams) -> list[float]:
    n = int(z.shape[0])
    if n < 64:
        return []
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    z_range = z_max - z_min
    if (not math.isfinite(z_range)) or z_range <= 1.0e-12:
        return []

    nbins = int(max(64, min(int(hp.fft_grid_size), n // 2 if n >= 128 else n)))
    if nbins < 32:
        return []
    edges = np.linspace(z_min, z_max, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, nbins - 1)
    sums = np.zeros(nbins, dtype=np.float64)
    counts = np.zeros(nbins, dtype=np.int64)
    np.add.at(sums, idx, y)
    np.add.at(counts, idx, 1)
    ok = counts > 0
    if int(np.count_nonzero(ok)) < 32:
        return []
    prof = np.empty(nbins, dtype=np.float64)
    prof[ok] = sums[ok] / counts[ok]
    if not np.all(ok):
        prof[~ok] = np.interp(centers[~ok], centers[ok], prof[ok])

    design = _trend_design(centers)
    try:
        coef, *_ = np.linalg.lstsq(design, prof, rcond=None)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            resid = prof - design @ coef
        if not np.all(np.isfinite(resid)):
            return []
    except Exception:
        resid = prof - float(np.mean(prof))
    resid = resid - float(np.mean(resid))
    if float(np.std(resid)) <= 1.0e-14:
        return []

    fft = np.fft.rfft(resid)
    power = np.abs(fft) ** 2
    if power.shape[0] <= 2:
        return []
    power[0] = 0.0
    dz = z_range / float(nbins - 1)
    freqs = np.fft.rfftfreq(nbins, d=dz)
    order = np.argsort(power)[::-1]
    seeds: list[float] = []
    for j in order[: max(0, int(hp.fft_top_k))]:
        if j <= 0:
            continue
        omega = 2.0 * math.pi * float(freqs[j])
        if math.isfinite(omega) and omega > 1.0e-12:
            seeds.append(omega)
    return seeds


def _dedupe_positive(vals: Iterable[float]) -> tuple[float, ...]:
    out: list[float] = []
    for raw in vals:
        try:
            v = abs(float(raw))
        except Exception:
            continue
        if (not math.isfinite(v)) or v <= 1.0e-12:
            continue
        if any(abs(v - old) <= max(1.0e-9, 1.0e-4 * max(abs(old), 1.0)) for old in out):
            continue
        out.append(v)
    return tuple(sorted(out))


def _omega_terms_from_base(base_omega: float, *, energy: float = 1.0) -> tuple[OmegaTerm, ...]:
    """Represent the two main conventions for one Fourier phase frequency."""

    try:
        omega = float(base_omega)
    except Exception:
        return tuple()
    if (not math.isfinite(omega)) or omega <= 1.0e-12:
        return tuple()
    # The first term is the literal Fourier basis frequency.  The second term
    # is the corresponding square-trig carrier convention:
    # sin(a*z)^2 = 1/2 - 1/2*cos(2*a*z).
    return (
        OmegaTerm(
            base_omega=omega,
            harmonic=1,
            actual_omega=omega,
            energy=float(energy),
            family="fourier",
        ),
        OmegaTerm(
            base_omega=0.5 * omega,
            harmonic=2,
            actual_omega=omega,
            energy=float(energy),
            family="sin_square",
        ),
        OmegaTerm(
            base_omega=0.5 * omega,
            harmonic=2,
            actual_omega=omega,
            energy=float(energy),
            family="cos_square",
        ),
    )


def _split_r2_for_omega(
    z: np.ndarray,
    y: np.ndarray,
    omega: float,
    *,
    hp: PhaseScanHyperparams,
    seed_offset: int,
) -> tuple[float, float]:
    """Diagnostic-only held-out R2 on a second internal split."""

    if int(z.shape[0]) < 96:
        return float("-inf"), float("-inf")
    rng = np.random.default_rng(int(hp.random_seed) + int(seed_offset))
    perm = rng.permutation(z.shape[0])
    n_train = max(32, int(0.65 * z.shape[0]))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    if val_idx.shape[0] < 32:
        return float("-inf"), float("-inf")
    z_train = z[train_idx]
    y_train = y[train_idx]
    z_val = z[val_idx]
    y_val = y[val_idx]
    r2_phase = _fit_r2(
        _fourier_design(z_train, omega, hp.max_harmonic),
        y_train,
        _fourier_design(z_val, omega, hp.max_harmonic),
        y_val,
    )
    r2_trend = _fit_r2(_trend_design(z_train), y_train, _trend_design(z_val), y_val)
    return float(r2_phase), float(r2_trend)


def _dim_equal(a: Any, b: Any) -> bool:
    try:
        return tuple(Fraction(v) for v in a) == tuple(Fraction(v) for v in b)
    except Exception:
        return False


def _feature_dim(exponents: tuple[tuple[int, Fraction], ...], x_dims: Sequence[Any] | None) -> Any:
    if x_dims is None:
        return None
    return _carrier_dim(exponents, x_dims)


def _feature_unit_ok(dim: Any, y_dim: Any, *, ignore_units: bool) -> tuple[bool, str]:
    if ignore_units or y_dim is None:
        return True, "unchecked"
    if dim is None:
        return False, "unit_unknown"
    if _dim_equal(dim, y_dim):
        return True, "output_dim"
    return False, "unit_mismatch"


def _context_feature_candidates(
    X: np.ndarray,
    *,
    Nxvars: int,
    carrier: PhaseCarrier,
    units_payload: dict[str, Any] | None,
    ignore_units: bool,
    hp: PhaseScanHyperparams,
) -> list[_ContextFeature]:
    """Build a tiny visible context basis for diagnostics-only screening."""

    Nx = int(Nxvars)
    carrier_support = set(int(i) for i in getattr(carrier, "support", ()) or ())
    x_dims = None
    y_dim = None
    if (not ignore_units) and isinstance(units_payload, dict):
        x_dims = units_payload.get("x_dims", None)
        y_dim = units_payload.get("y_dim", None)
        if x_dims is not None:
            x_dims = tuple(x_dims)
            if len(x_dims) != Nx:
                x_dims = None

    rows: list[_ContextFeature] = []

    def _add(exponents: tuple[tuple[int, Fraction], ...], *, cost: float) -> None:
        if any(int(i) in carrier_support for i, _ in exponents):
            return
        dim = _feature_dim(exponents, x_dims)
        ok, unit_status = _feature_unit_ok(dim, y_dim, ignore_units=ignore_units)
        if not ok:
            return
        ast = _carrier_ast(exponents)
        try:
            vals = _eval_carrier(
                PhaseCarrier(
                    ast=ast,
                    label=ast_to_human_readable(ast),
                    exponents=exponents,
                    support=tuple(int(i) for i, _ in exponents),
                    cost=cost,
                    dim=dim,
                    unit_status=unit_status,
                ),
                X,
            )
        except Exception:
            return
        if vals.shape[0] != X.shape[0] or not np.all(np.isfinite(vals)):
            return
        if float(np.std(vals)) <= 1.0e-14:
            return
        rows.append(
            _ContextFeature(
                ast=ast,
                label=ast_to_human_readable(ast),
                values=vals.astype(np.float64, copy=False),
                cost=float(cost),
                dim=dim,
                unit_status=unit_status,
            )
        )

    for i in range(Nx):
        _add(((int(i), Fraction(1, 1)),), cost=1.0)
    for i, j in combinations(range(Nx), 2):
        _add(((int(i), Fraction(1, 1)), (int(j), Fraction(1, 1))), cost=3.0)

    rows.sort(key=lambda f: (float(f.cost), f.label))
    out: list[_ContextFeature] = []
    seen: set[str] = set()
    for row in rows:
        if row.label in seen:
            continue
        seen.add(row.label)
        out.append(row)
        if len(out) >= max(0, int(hp.context_max_features)):
            break
    return out


def _context_matrix(features: Sequence[_ContextFeature], idx: np.ndarray) -> np.ndarray:
    if not features:
        return np.empty((idx.shape[0], 0), dtype=np.float64)
    return np.stack([np.asarray(f.values, dtype=np.float64)[idx] for f in features], axis=1)


def _contextual_design(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    if P.size == 0 or T.size == 0:
        return np.empty((P.shape[0], 0), dtype=np.float64)
    cols = [P]
    for j in range(P.shape[1]):
        cols.append(P[:, [j]] * T)
    return np.concatenate(cols, axis=1)


def _classify_context_waveform(
    *,
    coef: np.ndarray | None,
    n_context: int,
    n_trig: int,
    omega: float,
) -> str:
    if coef is None or n_context <= 0 or n_trig < 2:
        return "contextual_fourier"
    try:
        base = np.asarray(coef[:n_context], dtype=np.float64)
        tensor = np.asarray(coef[n_context:], dtype=np.float64).reshape(n_context, n_trig)
        j = int(np.argmax(np.abs(base)))
        base_j = float(base[j])
        cos1 = float(tensor[j, 0])
        sin1 = float(tensor[j, 1])
        if (
            abs(base_j) > 1.0e-12
            and abs(cos1) > 0.25 * abs(base_j)
            and abs(sin1) < 0.20 * max(abs(cos1), 1.0e-12)
            and base_j * cos1 < 0.0
        ):
            return "one_minus_cos"
    except Exception:
        pass
    try:
        if abs(float(omega) - 2.0 * math.pi) < 1.0e-3:
            return "sin_square_or_fourier"
    except Exception:
        pass
    return "contextual_fourier"


def _score_phase_context_carrier(
    X: np.ndarray,
    y: np.ndarray,
    carrier: PhaseCarrier,
    *,
    units_payload: dict[str, Any] | None,
    ignore_units: bool,
    hp: PhaseScanHyperparams,
) -> PhaseContextHint | None:
    try:
        z_all = _eval_carrier(carrier, X)
    except Exception:
        return None
    yy_all = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(int(z_all.shape[0]), int(yy_all.shape[0]), int(X.shape[0]))
    z_all = z_all[:n]
    yy_all = yy_all[:n]
    X = np.asarray(X[:n], dtype=np.float64)
    finite = np.isfinite(z_all) & np.isfinite(yy_all) & np.all(np.isfinite(X), axis=1)
    domain_frac = float(np.mean(finite)) if n else 0.0
    if domain_frac < float(hp.min_domain_frac):
        return None
    if int(np.count_nonzero(finite)) < 160:
        return None
    z = z_all[finite]
    yy = yy_all[finite]
    Xf = X[finite]
    if float(np.std(yy)) <= 1.0e-14:
        return None
    z_range = float(np.max(z) - np.min(z))
    if (not math.isfinite(z_range)) or z_range <= 1.0e-12:
        return None

    features = _context_feature_candidates(
        Xf,
        Nxvars=Xf.shape[1],
        carrier=carrier,
        units_payload=units_payload,
        ignore_units=ignore_units,
        hp=hp,
    )
    if not features:
        return None

    seeds = [1.0]
    seeds.extend(_PHYSICS_OMEGA_SEEDS)
    seeds.extend(_fft_omega_seeds(z, yy, hp=hp))
    omega_seeds = _dedupe_positive(seeds)
    if not omega_seeds:
        return None

    rng = np.random.default_rng(int(hp.random_seed) + 9176 + 31 * (len(carrier.support) + 1))
    perm = rng.permutation(z.shape[0])
    n_train = max(48, int(0.65 * z.shape[0]))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    if val_idx.shape[0] < 48:
        return None

    P_train_raw = _context_matrix(features, train_idx)
    P_val_raw = _context_matrix(features, val_idx)
    if P_train_raw.shape[1] == 0:
        return None
    P_train, p_scales = _standardize_design(P_train_raw)
    P_val = _apply_design_scales(P_val_raw, p_scales)
    y_train = yy[train_idx]
    y_val = yy[val_idx]
    z_train = z[train_idx]
    z_val = z[val_idx]

    r2_context, _ = _fit_linear_model(P_train, y_train, P_val, y_val)
    if not math.isfinite(r2_context):
        return None

    best: dict[str, Any] | None = None
    for omega in omega_seeds:
        n_cycles = z_range * float(omega) / (2.0 * math.pi)
        if n_cycles < 0.25:
            continue
        T_train_raw = _fourier_design(z_train, omega, hp.max_harmonic)[:, 1:]
        T_val_raw = _fourier_design(z_val, omega, hp.max_harmonic)[:, 1:]
        T_train, t_scales = _standardize_design(T_train_raw)
        T_val = _apply_design_scales(T_val_raw, t_scales)
        additive_train = np.concatenate([P_train, T_train], axis=1)
        additive_val = np.concatenate([P_val, T_val], axis=1)
        contextual_train = _contextual_design(P_train, T_train)
        contextual_val = _contextual_design(P_val, T_val)
        r2_additive, _ = _fit_linear_model(additive_train, y_train, additive_val, y_val)
        r2_context_phase, coef = _fit_linear_model(contextual_train, y_train, contextual_val, y_val)
        if not math.isfinite(r2_context_phase):
            continue
        delta = r2_context_phase - max(0.0, r2_context if math.isfinite(r2_context) else 0.0)
        parsimony = 1.0 / (
            1.0
            + 0.08 * float(carrier.cost)
            + 0.04 * sum(float(f.cost) for f in features)
        )
        coverage = min(1.0, max(0.0, n_cycles / 2.0))
        score = max(0.0, float(delta)) * parsimony * coverage
        row = {
            "omega": float(omega),
            "score": float(score),
            "r2_context_only": float(r2_context),
            "r2_context_phase": float(r2_context_phase),
            "r2_additive_phase": float(r2_additive),
            "delta_r2_phase": float(delta),
            "n_cycles": float(n_cycles),
            "coef": coef,
        }
        if best is None or float(row["score"]) > float(best["score"]):
            best = row

    if best is None or float(best["score"]) <= 0.0:
        return None

    omega = float(best["omega"])
    omega_terms = _omega_terms_from_base(omega, energy=max(0.0, float(best["score"])))
    waveform = _classify_context_waveform(
        coef=best.get("coef", None),
        n_context=len(features),
        n_trig=2 * int(hp.max_harmonic),
        omega=omega,
    )
    context_asts = tuple(f.ast for f in features if f.ast is not None)
    confidence = max(0.0, min(1.0, float(best["score"])))
    return PhaseContextHint(
        carrier_ast=carrier.ast,
        carrier_label=carrier.label,
        phase_family="linear",
        omega_terms=omega_terms,
        context_asts=context_asts,
        coupling_mode="prefactor",
        waveform_family=waveform,
        r2_context_only=float(best["r2_context_only"]),
        r2_context_phase=float(best["r2_context_phase"]),
        delta_r2_phase=float(best["delta_r2_phase"]),
        confidence=float(confidence),
        unit_status=str(carrier.unit_status),
        details={
            "carrier_cost": float(carrier.cost),
            "support": tuple(int(i) for i in carrier.support),
            "exponents": tuple((int(i), str(e)) for i, e in carrier.exponents),
            "context_labels": tuple(f.label for f in features),
            "context_unit_status": tuple(f.unit_status for f in features),
            "omega": float(omega),
            "r2_additive_phase": float(best["r2_additive_phase"]),
            "n_cycles": float(best["n_cycles"]),
            "support_fraction": float(domain_frac),
        },
    )


def score_phase_carrier(
    X: np.ndarray,
    y: np.ndarray,
    carrier: PhaseCarrier,
    *,
    hp: PhaseScanHyperparams | None = None,
) -> PhaseHint | None:
    """Score one carrier by held-out phase-folded Fourier predictability."""

    hp = hp or PhaseScanHyperparams()
    try:
        z_all = _eval_carrier(carrier, X)
    except Exception:
        return None
    y_all = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(int(z_all.shape[0]), int(y_all.shape[0]))
    z_all = z_all[:n]
    y_all = y_all[:n]
    finite = np.isfinite(z_all) & np.isfinite(y_all)
    domain_frac = float(np.mean(finite)) if n else 0.0
    if domain_frac < float(hp.min_domain_frac):
        return None
    z = z_all[finite]
    yy = y_all[finite]
    if z.shape[0] < 128:
        return None
    z_range = float(np.max(z) - np.min(z))
    if (not math.isfinite(z_range)) or z_range <= 1.0e-12:
        return None
    y_std = float(np.std(yy))
    if (not math.isfinite(y_std)) or y_std <= 1.0e-14:
        return None

    seeds = list(_PHYSICS_OMEGA_SEEDS)
    seeds.extend(_fft_omega_seeds(z, yy, hp=hp))
    omega_seeds = _dedupe_positive(seeds)
    if not omega_seeds:
        return None

    rng = np.random.default_rng(int(hp.random_seed) + 17 * (len(carrier.support) + 1))
    perm = rng.permutation(z.shape[0])
    n_train = max(32, int(0.65 * z.shape[0]))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    if val_idx.shape[0] < 32:
        return None
    z_train = z[train_idx]
    y_train = yy[train_idx]
    z_val = z[val_idx]
    y_val = yy[val_idx]

    r2_trend = _fit_r2(_trend_design(z_train), y_train, _trend_design(z_val), y_val)
    best: dict[str, Any] | None = None
    for omega in omega_seeds:
        n_cycles = z_range * float(omega) / (2.0 * math.pi)
        if n_cycles < 0.25:
            continue
        r2_phase = _fit_r2(
            _fourier_design(z_train, omega, hp.max_harmonic),
            y_train,
            _fourier_design(z_val, omega, hp.max_harmonic),
            y_val,
        )
        if not math.isfinite(r2_phase):
            continue
        improvement = r2_phase - max(0.0, r2_trend if math.isfinite(r2_trend) else 0.0)
        parsimony = 1.0 / (1.0 + 0.08 * float(carrier.cost))
        coverage = min(1.0, max(0.0, n_cycles / 2.0))
        score = max(0.0, improvement) * parsimony * coverage
        row = {
            "omega": float(omega),
            "r2_phase": float(r2_phase),
            "r2_trend": float(r2_trend),
            "score": float(score),
            "n_cycles": float(n_cycles),
            "improvement": float(improvement),
        }
        if best is None or float(row["score"]) > float(best["score"]):
            best = row

    if best is None:
        return None

    observed = float(best["omega"])
    carrier_omegas = _dedupe_positive((observed, 0.5 * observed))
    omega_terms = _omega_terms_from_base(observed, energy=max(0.0, float(best["score"])))
    split_r2_phase, split_r2_trend = _split_r2_for_omega(
        z,
        yy,
        observed,
        hp=hp,
        seed_offset=7919 + 101 * (len(carrier.support) + 1),
    )
    shuffle_r2_phase = float("-inf")
    try:
        rng_null = np.random.default_rng(int(hp.random_seed) + 104729 + len(carrier.support))
        y_train_null = y_train[rng_null.permutation(y_train.shape[0])]
        shuffle_r2_phase = _fit_r2(
            _fourier_design(z_train, observed, hp.max_harmonic),
            y_train_null,
            _fourier_design(z_val, observed, hp.max_harmonic),
            y_val,
        )
    except Exception:
        shuffle_r2_phase = float("-inf")
    confidence = max(0.0, min(1.0, float(best["score"])))
    return PhaseHint(
        carrier_ast=carrier.ast,
        carrier_label=carrier.label,
        phase_family="linear",
        observed_omega=observed,
        carrier_omega_candidates=carrier_omegas,
        waveform_family="fourier",
        envelope_family="none",
        score=float(best["score"]),
        confidence=confidence,
        r2_phase=float(best["r2_phase"]),
        r2_trend=float(best["r2_trend"]),
        support_fraction=float(domain_frac),
        n_cycles=float(best["n_cycles"]),
        unit_status=str(carrier.unit_status),
        details={
            "carrier_cost": float(carrier.cost),
            "support": tuple(int(i) for i in carrier.support),
            "exponents": tuple((int(i), str(e)) for i, e in carrier.exponents),
            "trend_improvement": float(best["improvement"]),
            "split_r2_phase": float(split_r2_phase),
            "split_r2_trend": float(split_r2_trend),
            "shuffle_r2_phase": float(shuffle_r2_phase),
        },
        omega_terms=omega_terms,
    )


def run_phase_prescan(
    X: np.ndarray,
    y: np.ndarray,
    *,
    Nxvars: int | None = None,
    units_payload: dict[str, Any] | None = None,
    ignore_units: bool = False,
    hp: PhaseScanHyperparams | None = None,
) -> list[PhaseHint]:
    """Run the full phase-coordinate prescan and return ranked hints."""

    hp = hp or PhaseScanHyperparams()
    if not bool(hp.enabled):
        return []
    Xs, ys = _sample_xy(X, y, sample_size=int(hp.sample_size), seed=int(hp.random_seed))
    if Xs.ndim != 2 or Xs.shape[0] < 128:
        return []
    Nx = int(Nxvars if Nxvars is not None else Xs.shape[1])
    carriers = build_unit_valid_phase_carriers(
        Nx,
        units_payload=units_payload,
        ignore_units=ignore_units,
        hp=hp,
    )
    hints: list[PhaseHint] = []
    for carrier in carriers:
        hint = score_phase_carrier(Xs, ys, carrier, hp=hp)
        if hint is not None and math.isfinite(float(hint.score)) and float(hint.score) > 0.0:
            hints.append(hint)
    hints.sort(
        key=lambda h: (
            -float(h.score),
            -float(h.r2_phase),
            float(h.details.get("carrier_cost", 0.0)),
            h.carrier_label,
        )
    )
    return hints


def run_phase_context_scan(
    X: np.ndarray,
    y: np.ndarray,
    *,
    Nxvars: int | None = None,
    units_payload: dict[str, Any] | None = None,
    ignore_units: bool = False,
    hp: PhaseScanHyperparams | None = None,
) -> list[PhaseContextHint]:
    """Diagnostics-only contextual phase scan.

    This asks whether a dimensionless carrier becomes predictive when Fourier
    columns are crossed with a tiny visible context basis.  It returns sidecar
    hints only; callers must not accept or rank expressions from these records.
    """

    hp = hp or PhaseScanHyperparams()
    if (not bool(hp.enabled)) or (not bool(hp.context_enabled)):
        return []
    Xs, ys = _sample_xy(
        X,
        y,
        sample_size=int(hp.sample_size),
        seed=int(hp.random_seed) + 303,
    )
    if Xs.ndim != 2 or Xs.shape[0] < 160:
        return []
    Nx = int(Nxvars if Nxvars is not None else Xs.shape[1])
    carriers = build_unit_valid_phase_carriers(
        Nx,
        units_payload=units_payload,
        ignore_units=ignore_units,
        hp=hp,
    )
    hints: list[PhaseContextHint] = []
    for carrier in carriers:
        hint = _score_phase_context_carrier(
            Xs,
            ys,
            carrier,
            units_payload=units_payload,
            ignore_units=ignore_units,
            hp=hp,
        )
        if hint is not None and math.isfinite(float(hint.confidence)) and float(hint.confidence) > 0.0:
            hints.append(hint)
    hints.sort(
        key=lambda h: (
            -float(h.confidence),
            -float(h.delta_r2_phase),
            -float(h.r2_context_phase),
            h.carrier_label,
        )
    )
    return hints


def _target_y_dimless(units_payload: dict[str, Any] | None, *, ignore_units: bool) -> tuple[bool, str]:
    if ignore_units or not isinstance(units_payload, dict):
        return True, "unchecked"
    y_dim = units_payload.get("y_dim", None)
    if y_dim is None:
        return False, "unit_unknown"
    if _dimless(y_dim):
        return True, "dimensionless"
    return False, "target_not_dimensionless"


def _outer_link_branch_fraction(transform_name: str, y: np.ndarray) -> float:
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    if yy.size == 0:
        return 0.0
    if transform_name == "sin":
        ok = (yy >= (-0.5 * math.pi - 1.0e-10)) & (yy <= (0.5 * math.pi + 1.0e-10))
        ok &= np.cos(yy) >= -1.0e-8
    elif transform_name == "cos":
        ok = (yy >= (-1.0e-10)) & (yy <= (math.pi + 1.0e-10))
        ok &= np.sin(yy) >= -1.0e-8
    elif transform_name == "tan":
        ok = (yy > (-0.5 * math.pi + 1.0e-6)) & (yy < (0.5 * math.pi - 1.0e-6))
        ok &= np.abs(np.cos(yy)) > 1.0e-6
    else:
        return 0.0
    return float(np.mean(ok))


def _outer_link_transform_y(transform_name: str, y: np.ndarray) -> np.ndarray | None:
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if transform_name == "sin":
            return np.sin(yy)
        if transform_name == "cos":
            return np.cos(yy)
        if transform_name == "tan":
            return np.tan(yy)
    return None


def _score_outer_link_carrier(
    X: np.ndarray,
    y: np.ndarray,
    carrier: PhaseCarrier,
    *,
    hp: PhaseScanHyperparams,
) -> list[OuterLinkHint]:
    try:
        z_all = _eval_carrier(carrier, X)
    except Exception:
        return []
    yy_all = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(int(z_all.shape[0]), int(yy_all.shape[0]))
    z_all = z_all[:n]
    yy_all = yy_all[:n]
    finite = np.isfinite(z_all) & np.isfinite(yy_all)
    domain_frac = float(np.mean(finite)) if n else 0.0
    if domain_frac < float(hp.min_domain_frac):
        return []
    z = z_all[finite]
    yy = yy_all[finite]
    if z.shape[0] < 96:
        return []
    if float(np.std(z)) <= 1.0e-14:
        return []

    rng = np.random.default_rng(int(hp.random_seed) + 11939 + 97 * (len(carrier.support) + 1))
    perm = rng.permutation(z.shape[0])
    n_train = max(32, int(0.65 * z.shape[0]))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    if val_idx.shape[0] < 32:
        return []

    z_train = z[train_idx]
    z_val = z[val_idx]
    design_train = np.stack([z_train, np.ones_like(z_train)], axis=1)
    design_val = np.stack([z_val, np.ones_like(z_val)], axis=1)
    rows: list[OuterLinkHint] = []
    spec_rows = (
        ("arcsin", "sin"),
        ("arccos", "cos"),
        ("arctan", "tan"),
    )
    for link_name, transform_name in spec_rows:
        branch_frac = _outer_link_branch_fraction(transform_name, yy)
        if branch_frac < float(hp.min_domain_frac):
            continue
        t_all = _outer_link_transform_y(transform_name, yy)
        if t_all is None or not np.all(np.isfinite(t_all)):
            continue
        t_train = t_all[train_idx]
        t_val = t_all[val_idx]
        if float(np.std(t_train)) <= 1.0e-14 or float(np.std(t_val)) <= 1.0e-14:
            continue
        try:
            coef, *_ = np.linalg.lstsq(design_train, t_train, rcond=None)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                pred = design_val @ coef
        except Exception:
            continue
        if coef.shape[0] < 2 or not np.all(np.isfinite(coef)) or not np.all(np.isfinite(pred)):
            continue
        residual = pred - t_val
        mse = float(np.mean(residual * residual))
        var = float(np.var(t_val))
        if (not math.isfinite(mse)) or (not math.isfinite(var)) or var <= 1.0e-30:
            continue
        r2 = float(1.0 - mse / var)
        rms = math.sqrt(max(0.0, mse))
        denom = math.sqrt(max(var, 1.0e-30))
        rms_rel = float(rms / denom)
        a = float(coef[0])
        b = float(coef[1])
        arg_all = a * z + b
        if transform_name in {"sin", "cos"}:
            arg_ok = np.isfinite(arg_all) & (arg_all >= -1.0 - 1.0e-8) & (arg_all <= 1.0 + 1.0e-8)
        else:
            arg_ok = np.isfinite(arg_all)
        arg_frac = float(np.mean(arg_ok)) if arg_all.size else 0.0
        if arg_frac < float(hp.min_domain_frac):
            continue
        confidence = max(0.0, min(1.0, r2)) * arg_frac * branch_frac / (1.0 + 0.05 * float(carrier.cost))
        if confidence <= 0.0:
            continue
        rows.append(
            OuterLinkHint(
                link_name=link_name,
                transform_name=transform_name,
                carrier_ast=carrier.ast,
                carrier_label=carrier.label,
                affine_a=float(a),
                affine_b=float(b),
                rms_rel=float(rms_rel),
                r2=float(r2),
                domain_ok_frac=float(arg_frac),
                branch_ok_frac=float(branch_frac),
                confidence=float(confidence),
                unit_status=str(carrier.unit_status),
                details={
                    "carrier_cost": float(carrier.cost),
                    "support": tuple(int(i) for i in carrier.support),
                    "exponents": tuple((int(i), str(e)) for i, e in carrier.exponents),
                    "data_domain_fraction": float(domain_frac),
                },
            )
        )
    rows.sort(key=lambda h: (-float(h.confidence), float(h.rms_rel), h.link_name, h.carrier_label))
    return rows


def _score_outer_link_affine_trig_carriers(
    X: np.ndarray,
    y: np.ndarray,
    base_carrier: PhaseCarrier,
    *,
    units_payload: dict[str, Any] | None,
    ignore_units: bool,
    hp: PhaseScanHyperparams,
) -> list[OuterLinkHint]:
    """Score ``base*sin(omega*xj+phase)`` / ``base*cos(...)`` carriers.

    For a fixed ``omega``, the phase is solved by linear least squares through
    ``base*sin(omega*xj)`` and ``base*cos(omega*xj)`` columns.  Only the
    resulting visible carrier is emitted, and it remains proposal evidence.
    """

    try:
        base_all = _eval_carrier(base_carrier, X)
    except Exception:
        return []
    yy_all = np.asarray(y, dtype=np.float64).reshape(-1)
    X_all = np.asarray(X, dtype=np.float64)
    n = min(int(base_all.shape[0]), int(yy_all.shape[0]), int(X_all.shape[0]))
    if n <= 0:
        return []
    base_all = base_all[:n]
    yy_all = yy_all[:n]
    X_all = X_all[:n]
    finite0 = np.isfinite(base_all) & np.isfinite(yy_all) & np.all(np.isfinite(X_all), axis=1)
    domain_frac0 = float(np.mean(finite0)) if n else 0.0
    if domain_frac0 < float(hp.min_domain_frac):
        return []
    base = base_all[finite0]
    yy = yy_all[finite0]
    Xf = X_all[finite0]
    if base.shape[0] < 128 or float(np.std(base)) <= 1.0e-14:
        return []

    Nx = int(Xf.shape[1])
    x_dims = _unit_dims_from_payload(Nx, units_payload=units_payload, ignore_units=ignore_units)
    base_support = tuple(int(v) for v in getattr(base_carrier, "support", ()) or ())
    if not base_support:
        return []
    max_support = max(1, min(int(hp.max_support), Nx))

    rng = np.random.default_rng(int(hp.random_seed) + 20333 + 101 * (len(base_support) + 1))
    perm = rng.permutation(base.shape[0])
    n_train = max(48, int(0.65 * base.shape[0]))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    if val_idx.shape[0] < 48:
        return []

    rows: list[OuterLinkHint] = []
    spec_rows = (
        ("arcsin", "sin"),
        ("arccos", "cos"),
        ("arctan", "tan"),
    )
    for axis in range(Nx):
        if not _axis_is_dimless(axis, x_dims):
            continue
        support = tuple(sorted(set(base_support) | {int(axis)}))
        if len(support) > max_support:
            continue
        x_axis = Xf[:, axis].reshape(-1)
        omega_seeds = _outer_link_axis_omega_seeds(x_axis)
        if not omega_seeds:
            continue
        for link_name, transform_name in spec_rows:
            branch_frac = _outer_link_branch_fraction(transform_name, yy)
            if branch_frac < float(hp.min_domain_frac):
                continue
            t_all = _outer_link_transform_y(transform_name, yy)
            if t_all is None or not np.all(np.isfinite(t_all)):
                continue
            t_train = t_all[train_idx]
            t_val = t_all[val_idx]
            if float(np.std(t_train)) <= 1.0e-14 or float(np.std(t_val)) <= 1.0e-14:
                continue
            for omega in omega_seeds:
                arg = float(omega) * x_axis
                sin_col = base * np.sin(arg)
                cos_col = base * np.cos(arg)
                if not (np.all(np.isfinite(sin_col)) and np.all(np.isfinite(cos_col))):
                    continue
                design_train = np.stack(
                    [sin_col[train_idx], cos_col[train_idx], np.ones_like(t_train)],
                    axis=1,
                )
                design_val = np.stack(
                    [sin_col[val_idx], cos_col[val_idx], np.ones_like(t_val)],
                    axis=1,
                )
                try:
                    coef, *_ = np.linalg.lstsq(design_train, t_train, rcond=None)
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        pred = design_val @ coef
                except Exception:
                    continue
                if coef.shape[0] < 3 or not np.all(np.isfinite(coef)) or not np.all(np.isfinite(pred)):
                    continue
                residual = pred - t_val
                mse = float(np.mean(residual * residual))
                var = float(np.var(t_val))
                if (not math.isfinite(mse)) or (not math.isfinite(var)) or var <= 1.0e-30:
                    continue
                r2 = float(1.0 - mse / var)
                rms = math.sqrt(max(0.0, mse))
                rms_rel = float(rms / math.sqrt(max(var, 1.0e-30)))
                c_sin = float(coef[0])
                c_cos = float(coef[1])
                offset = float(coef[2])
                amp = float(math.hypot(c_sin, c_cos))
                if (not math.isfinite(amp)) or amp <= 1.0e-14:
                    continue

                reps = (
                    ("sin", SinNode, _wrap_phase_for_trig(math.atan2(c_cos, c_sin))),
                    ("cos", CosNode, _wrap_phase_for_trig(math.atan2(-c_sin, c_cos))),
                )
                for trig_kind, trig_node, phase in reps:
                    phase_arg = _phase_affine_arg_ast(axis, float(omega), float(phase))
                    carrier_ast = MulNode(base_carrier.ast, trig_node(phase_arg))
                    try:
                        z_all = _eval_carrier(
                            PhaseCarrier(
                                ast=carrier_ast,
                                label="",
                                exponents=(),
                                support=support,
                                cost=0.0,
                            ),
                            Xf,
                        )
                    except Exception:
                        continue
                    arg_all = amp * z_all.reshape(-1) + offset
                    if transform_name in {"sin", "cos"}:
                        arg_ok = (
                            np.isfinite(arg_all)
                            & (arg_all >= -1.0 - 1.0e-8)
                            & (arg_all <= 1.0 + 1.0e-8)
                        )
                    else:
                        arg_ok = np.isfinite(arg_all)
                    arg_frac = float(np.mean(arg_ok)) if arg_all.size else 0.0
                    if arg_frac < float(hp.min_domain_frac):
                        continue

                    carrier_label = ast_to_human_readable(carrier_ast)
                    carrier_cost = float(getattr(base_carrier, "cost", 0.0)) + 3.0
                    confidence = max(0.0, min(1.0, r2)) * arg_frac * branch_frac / (
                        1.0 + 0.05 * carrier_cost
                    )
                    if confidence <= 0.0:
                        continue
                    rows.append(
                        OuterLinkHint(
                            link_name=link_name,
                            transform_name=transform_name,
                            carrier_ast=carrier_ast,
                            carrier_label=carrier_label,
                            affine_a=float(amp),
                            affine_b=float(offset),
                            rms_rel=float(rms_rel),
                            r2=float(r2),
                            domain_ok_frac=float(arg_frac),
                            branch_ok_frac=float(branch_frac),
                            confidence=float(confidence),
                            unit_status=(
                                "dimensionless"
                                if x_dims is not None
                                or str(getattr(base_carrier, "unit_status", "")) == "dimensionless"
                                else "unchecked"
                            ),
                            details={
                                "carrier_cost": float(carrier_cost),
                                "support": tuple(int(i) for i in support),
                                "exponents": tuple(
                                    (int(i), str(e)) for i, e in getattr(base_carrier, "exponents", ()) or ()
                                ),
                                "data_domain_fraction": float(domain_frac0),
                                "affine_trig": True,
                                "base_carrier_label": str(getattr(base_carrier, "label", "")),
                                "axis": int(axis),
                                "omega": float(omega),
                                "phase": float(phase),
                                "trig_kind": trig_kind,
                            },
                        )
                    )
    rows.sort(
        key=lambda h: (
            -float(h.confidence),
            float(h.rms_rel),
            float(h.details.get("carrier_cost", 0.0)),
            h.link_name,
            h.carrier_label,
        )
    )
    return rows[:16]


def _select_outer_link_affine_bases(
    carriers: Sequence[PhaseCarrier],
    *,
    hp: PhaseScanHyperparams,
) -> list[PhaseCarrier]:
    """Keep affine-trig base scans broad across arities but bounded."""

    per_support_cap = max(2, min(8, int(hp.max_candidates_per_support)))
    max_total = max(8, min(32, int(hp.max_candidates)))
    grouped: dict[int, list[PhaseCarrier]] = {}
    for carrier in carriers:
        support_size = len(tuple(getattr(carrier, "support", ()) or ()))
        grouped.setdefault(int(support_size), []).append(carrier)
    out: list[PhaseCarrier] = []
    seen: set[str] = set()
    for support_size in sorted(grouped):
        rows = sorted(grouped[support_size], key=lambda c: (float(c.cost), c.label))
        for carrier in rows[:per_support_cap]:
            if carrier.label in seen:
                continue
            seen.add(carrier.label)
            out.append(carrier)
            if len(out) >= max_total:
                return out
    return out


def run_outer_inverse_trig_prescan(
    X: np.ndarray,
    y: np.ndarray,
    *,
    Nxvars: int | None = None,
    units_payload: dict[str, Any] | None = None,
    ignore_units: bool = False,
    hp: PhaseScanHyperparams | None = None,
) -> list[OuterLinkHint]:
    """Run the inside-out inverse-trig scan.

    This tests ``sin(y)``, ``cos(y)``, and ``tan(y)`` against affine functions
    of sparse dimensionless carriers plus a small visible-trig mixed family.
    It is a hint source only; direct closure still goes through Stage-B
    fitting/validation.
    """

    hp = hp or PhaseScanHyperparams()
    if not bool(hp.enabled):
        return []
    ok, unit_status = _target_y_dimless(units_payload, ignore_units=ignore_units)
    if not ok:
        return []
    Xs, ys = _sample_xy(
        X,
        y,
        sample_size=int(hp.sample_size),
        seed=int(hp.random_seed) + 1701,
    )
    if Xs.ndim != 2 or Xs.shape[0] < 96:
        return []
    Nx = int(Nxvars if Nxvars is not None else Xs.shape[1])
    base_carriers = build_unit_valid_phase_carriers(
        Nx,
        units_payload=units_payload,
        ignore_units=ignore_units,
        hp=hp,
    )
    carriers = build_outer_link_phase_carriers(
        Nx,
        units_payload=units_payload,
        ignore_units=ignore_units,
        hp=hp,
    )
    hints: list[OuterLinkHint] = []
    for carrier in carriers:
        rows = _score_outer_link_carrier(Xs, ys, carrier, hp=hp)
        for row in rows:
            if unit_status != "unchecked" and row.unit_status == "unchecked":
                row = OuterLinkHint(
                    link_name=row.link_name,
                    transform_name=row.transform_name,
                    carrier_ast=row.carrier_ast,
                    carrier_label=row.carrier_label,
                    affine_a=row.affine_a,
                    affine_b=row.affine_b,
                    rms_rel=row.rms_rel,
                    r2=row.r2,
                    domain_ok_frac=row.domain_ok_frac,
                    branch_ok_frac=row.branch_ok_frac,
                    confidence=row.confidence,
                    unit_status=unit_status,
                    details=row.details,
                )
            hints.append(row)
    for base_carrier in _select_outer_link_affine_bases(base_carriers, hp=hp):
        rows = _score_outer_link_affine_trig_carriers(
            Xs,
            ys,
            base_carrier,
            units_payload=units_payload,
            ignore_units=ignore_units,
            hp=hp,
        )
        for row in rows:
            if unit_status != "unchecked" and row.unit_status == "unchecked":
                row = OuterLinkHint(
                    link_name=row.link_name,
                    transform_name=row.transform_name,
                    carrier_ast=row.carrier_ast,
                    carrier_label=row.carrier_label,
                    affine_a=row.affine_a,
                    affine_b=row.affine_b,
                    rms_rel=row.rms_rel,
                    r2=row.r2,
                    domain_ok_frac=row.domain_ok_frac,
                    branch_ok_frac=row.branch_ok_frac,
                    confidence=row.confidence,
                    unit_status=unit_status,
                    details=row.details,
                )
            hints.append(row)
    hints.sort(
        key=lambda h: (
            -float(h.confidence),
            -float(h.r2),
            float(h.rms_rel),
            float(h.details.get("carrier_cost", 0.0)),
            h.link_name,
            h.carrier_label,
        )
    )
    out: list[OuterLinkHint] = []
    seen: set[tuple[Any, ...]] = set()
    for hint in hints:
        key = (
            str(hint.link_name),
            str(hint.transform_name),
            str(hint.carrier_label),
            round(float(hint.affine_a), 9),
            round(float(hint.affine_b), 9),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(hint)
        if len(out) >= max(32, min(256, 4 * int(hp.max_candidates))):
            break
    return out


def format_phase_hint(hint: PhaseHint) -> str:
    """Concise log line for a phase hint."""

    obs = "nan" if hint.observed_omega is None else f"{float(hint.observed_omega):.6g}"
    details = getattr(hint, "details", {}) if isinstance(getattr(hint, "details", None), dict) else {}
    split_r2 = details.get("split_r2_phase", None)
    null_r2 = details.get("shuffle_r2_phase", None)
    diag_bits = []
    try:
        if split_r2 is not None and math.isfinite(float(split_r2)):
            diag_bits.append(f"split_R2={float(split_r2):.3g}")
    except Exception:
        pass
    try:
        if null_r2 is not None and math.isfinite(float(null_r2)):
            diag_bits.append(f"null_R2={float(null_r2):.3g}")
    except Exception:
        pass
    diag = "" if not diag_bits else ", " + ", ".join(diag_bits)
    return (
        f"{hint.carrier_label}: omega_obs={obs}, score={float(hint.score):.3g}, "
        f"R2_phase={float(hint.r2_phase):.3g}, R2_trend={float(hint.r2_trend):.3g}, "
        f"cycles={float(hint.n_cycles):.3g}, unit={hint.unit_status}{diag}"
    )


def format_phase_context_hint(hint: PhaseContextHint) -> str:
    """Concise log line for a contextual phase hint."""

    details = getattr(hint, "details", {}) if isinstance(getattr(hint, "details", None), dict) else {}
    omega = details.get("omega", None)
    omega_s = "nan"
    try:
        if omega is not None:
            omega_s = f"{float(omega):.6g}"
    except Exception:
        pass
    context_labels = tuple(details.get("context_labels", ()) or ())
    context_s = ",".join(str(x) for x in context_labels[:4]) if context_labels else "?"
    if len(context_labels) > 4:
        context_s += ",..."
    return (
        f"{hint.carrier_label}: context={context_s}, family={hint.waveform_family}, "
        f"omega={omega_s}, ΔR2={float(hint.delta_r2_phase):.3g}, "
        f"R2_ctx={float(hint.r2_context_only):.3g}, "
        f"R2_ctx+phase={float(hint.r2_context_phase):.3g}, unit={hint.unit_status}"
    )


def format_outer_link_hint(hint: OuterLinkHint) -> str:
    """Concise log line for an inverse-trig outer-link hint."""

    return (
        f"{hint.link_name}({hint.carrier_label}): "
        f"{hint.transform_name}(y)≈{float(hint.affine_a):.6g}*z"
        f"{float(hint.affine_b):+.6g}, R2={float(hint.r2):.3g}, "
        f"rel={float(hint.rms_rel):.2e}, domain={float(hint.domain_ok_frac):.3g}, "
        f"branch={float(hint.branch_ok_frac):.3g}, unit={hint.unit_status}"
    )


def phase_hint_omega_candidates(hints: Sequence[PhaseHint], *, for_square: bool = False) -> list[float]:
    """Extract positive frequency seeds from hints.

    ``for_square=True`` returns harmonic frequencies suitable for
    ``sin(a*z+b)^2`` screens, where the observed Fourier frequency is ``2*a``.
    """

    raw: list[float] = []
    for hint in hints:
        for term in tuple(getattr(hint, "omega_terms", ()) or ()):
            try:
                base = float(getattr(term, "base_omega"))
                actual = float(getattr(term, "actual_omega"))
            except Exception:
                continue
            raw.append(base)
            raw.append(actual)
            if for_square:
                raw.append(2.0 * base)
        if hint.observed_omega is not None:
            raw.append(float(hint.observed_omega))
        for omega in tuple(getattr(hint, "carrier_omega_candidates", ()) or ()):
            w = float(omega)
            raw.append(w)
            if for_square:
                raw.append(2.0 * w)
    return list(_dedupe_positive(raw))
