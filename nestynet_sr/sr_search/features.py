# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import logging as _logging

from nestynet_sr.sr_search.rational_sparsify import (
    DEFAULT_RAT_STLSQ_CFG,
    _log_sparsify_result,
    stlsq_sparsify_rational_coeffs,
)

_log = _logging.getLogger(__name__)

# --- small vector helpers used by direction discovery & line probes ---


def _unit(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / (v.norm() + eps)


def _canonicalize_sign(u: torch.Tensor) -> torch.Tensor:
    """
    Treat ±u as equivalent by forcing the sign of the largest-magnitude
    coordinate to be non-negative.
    """
    if u.numel() == 0:
        return u
    i = int(torch.argmax(u.abs()))
    s = 1.0 if float(u[i]) >= 0.0 else -1.0
    return u * s


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a @ b) / ((a.norm() + 1e-12) * (b.norm() + 1e-12)))


@dataclass
class FeatureSpec:
    name: str
    coeffs: torch.Tensor  # for invariance / linear features
    bias: float = 0.0
    kind: str = "linear"  # "linear", "integer_linear", etc.


@dataclass
class ScaleSpec:
    name: str
    indices: List[int]  # variables in S
    k_hat: float  # estimated homogeneity degree
    mean: float  # mean of r_S(x)
    std: float  # std of r_S(x)
    rel_std: float  # std / (|mean|+eps)
    n_points: int  # #points used
    # Oracle probe fields (optional)
    oracle_verified: bool = False
    oracle_k: Optional[float] = None  # degree from oracle probe
    oracle_rel_std: Optional[float] = None  # consistency across data points
    # Compound fields (for compound variable targets)
    compound_name: str = ""        # e.g., "x2-x3" for compound targets
    compound_expr: Any = None      # AST for compound (None for trivial)


@dataclass
class PolyFitSpec:
    name: str
    degree: int
    indices: List[int]  # variables involved (for now, probably range(Nx))
    n_terms: int
    n_points: int
    rms_abs: float  # RMS absolute residual
    rms_rel: float  # RMS residual / (std(target)+eps)


@dataclass
class RationalFitSpec:
    name: str
    deg_num: int
    deg_den: int
    n_terms_num: int
    n_terms_den: int
    n_points: int
    rms_abs: float
    rms_rel: float
    sigma_min: float  # smallest singular value of A
    sigma_ratio: float  # sigma_min / (median sigma)


@dataclass
class TrigAxisSpec:
    axis: int  # index j of the axis x_j
    omega: float  # dominant angular frequency along that axis
    strength: float  # peak/median spectral strength
    n_points: int  # number of samples along the line
    tmin: float  # span lower
    tmax: float  # span upper
    phase: float = 0.0  # FFT phase at dominant frequency (for sin vs cos selection)
    rel_std: float = 1.0  # oracle trig-scaling consistency (lower = cleaner; 1.0 default for FFT specs)
    basis_fn: str = ""  # original oracle basis, e.g. "cos", "sin", "one_minus_cos"


@dataclass
class TrigScaleSpec:
    """Result of the trig z-scaling probe: monomial degree in a trig basis."""
    axis: int          # variable index j (pivot axis for compounds)
    omega: float       # frequency from trig detection
    trig_fn: str       # "cos" or "sin" phase family (for downstream compatibility)
    k_hat: float       # estimated monomial degree in z
    rel_std: float     # consistency metric (lower = cleaner)
    n_points: int      # data points used in fit
    compound_name: str = ""        # e.g., "x2-x3" for compound targets
    compound_expr: Any = None      # AST for compound (None for trivial)
    basis_fn: str = ""             # "cos", "sin", or "one_minus_cos"; defaults to trig_fn


@dataclass
class TrigProbeTarget:
    """A compound variable target for trig probing.

    Represents both trivial variables (z = x_j) and compound variables
    (z = x_i - x_j, z = x_i + x_j, z = x_i * x_j, z = x_i / x_j).
    """
    name: str                      # e.g., "x0", "x2-x3", "x0*x1"
    indices: Tuple[int, ...]       # raw axis indices involved
    expr: Any                      # AST expression (None for trivial)
    kind: str                      # "trivial", "difference", "sum", "product", "ratio"
    pivot_idx: int                 # which axis to perturb


@dataclass
class MixedCompoundProposal:
    """Proposal for a mixed compound variable with linear and trig components.

    Represents compound variables of the form:
        z = (x0^a0 * x1^a1 * ...) * cos(ω1*xk + φ1) * sin(ω2*xl + φ2) * ...

    The linear (monomial) part uses the standard product/ratio detection,
    while trig variables are identified by FFT analysis on axis scans.
    """
    linear_var_idxs: Tuple[int, ...]      # Indices of monomial variables
    linear_exponents: Tuple[int, ...]      # Exponents for monomial variables
    trig_var_idxs: Tuple[int, ...]         # Indices of trig variables
    trig_omegas: Tuple[float, ...]         # Angular frequencies for each trig var
    trig_kinds: Tuple[str, ...]            # "cos" or "sin" for each trig var
    trig_phases: Tuple[float, ...]         # Phase offsets (0.0 for pure cos/sin)
    monomial_sigma_ratio: float            # σ₂/σ₁ from SVD on linear subset
    trig_strengths: Tuple[float, ...]      # FFT peak strengths for each trig var
    overall_confidence: float              # Combined confidence score
    z_ast: Any                             # AST node for the compound variable


@dataclass
class ParitySpec:
    name: str
    axis: int
    origin: float
    kind: str
    rms_even: float
    rms_odd: float
    rms_rel_even: float
    rms_rel_odd: float
    n_points: int


@dataclass
class RadialSpec:
    name: str
    indices: List[int]
    mean_abs_cos: float
    median_abs_cos: float
    n_points: int


@dataclass
class TranslationSpec:
    axis: int
    origin: float
    slope: float
    intercept: float
    r2: float
    in_range: bool
    n_points: int


@dataclass
class SaturatingSpec:
    axis: int
    tmin: float
    tmax: float
    mid_edge_grad_ratio: float
    monotonic_fraction: float
    direction: int
    n_points: int


@dataclass
class ConstantDirectionSpec:
    axis: int
    rms_grad: float
    rel_rms_grad: float
    n_points: int


def _sample_gradients(model, datagen, max_batches=8, max_points=20000, device=None):
    xs = []
    grads = []
    n_points = 0

    data_iter = datagen() if callable(datagen) else datagen

    for bi, batch in enumerate(data_iter):
        if bi >= max_batches or n_points >= max_points:
            break

        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        dev = device or next(model.parameters()).device
        x = x.to(dev)
        B = x.shape[0]
        x_flat = x.view(B, -1)

        g = model.grad(x_flat)
        if g.dim() == 3:
            g = g[:, 0, :]
        else:
            g = g.view(B, -1)

        xs.append(x_flat.detach().cpu())
        grads.append(g.detach().cpu())
        n_points += B

    if not xs:
        return None, None

    X = torch.cat(xs, dim=0)[:max_points]
    G = torch.cat(grads, dim=0)[:max_points]
    return X, G


def discover_invariance_features(
    model, datagen, Nxvars, device=None, max_batches=8, max_points=20000, eig_threshold_ratio=1e-2
):
    X, G = _sample_gradients(model, datagen, max_batches, max_points, device)
    if X is None:
        return []

    N, Nx = G.shape
    assert Nx == Nxvars

    # Covariance C = (G^T G) / N
    C = (G.T @ G) / float(N)

    # Eigen-decomposition (symmetric)
    evals, evecs = torch.linalg.eigh(C)  # evals ascending

    # Heuristic: consider eigenvalues that are "small" relative to the largest
    lam_max = float(evals[-1])
    small_mask = evals <= eig_threshold_ratio * max(lam_max, 1e-12)
    small_idxs = small_mask.nonzero(as_tuple=True)[0]

    features = []
    for idx in small_idxs:
        v = evecs[:, idx]  # shape [Nx]
        # Normalise to max |coeff| = 1 and round to a few decimals
        vmax = float(v.abs().max())
        if vmax < 1e-8:
            continue
        v_norm = (v / vmax).detach().cpu()
        # Optional: snap to small integers in {-2,-1,0,1,2} if close
        approx_int = torch.round(v_norm).to(torch.int64)
        if torch.all((approx_int >= -2) & (approx_int <= 2)):
            # Check that this integer vector is still nonzero
            if approx_int.abs().sum() == 0:
                continue
            coeffs = approx_int.to(torch.float64)
            kind = "integer_linear"
            name = "feat_" + "_".join(f"{int(c.item())}" for c in coeffs)
        else:
            coeffs = v_norm
            kind = "linear"
            name = f"lin_ev{int(idx)}"

        features.append(FeatureSpec(name=name, coeffs=coeffs, kind=kind))

    return features


def _sample_gradients_and_values(model, datagen, max_batches=8, max_points=20000, device=None):
    xs, grads, vals = [], [], []
    n_points = 0

    data_iter = datagen() if callable(datagen) else datagen

    for bi, batch in enumerate(data_iter):
        if bi >= max_batches or n_points >= max_points:
            break

        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        x = x.to(device or next(model.parameters()).device)
        B = x.shape[0]
        x_flat = x.view(B, -1)

        # Forward
        y = model(x_flat)
        if y.dim() == 2:
            f = y[:, 0]
        else:
            f = y.view(-1)

        # Gradient wrt inputs
        g = model.grad(x_flat)
        if g.dim() == 3:
            g = g[:, 0, :]
        else:
            g = g.view(B, -1)

        xs.append(x_flat.detach().cpu())
        grads.append(g.detach().cpu())
        vals.append(f.detach().cpu())
        n_points += B

    if not xs:
        return None, None, None

    X = torch.cat(xs, dim=0)[:max_points]
    G = torch.cat(grads, dim=0)[:max_points]
    F = torch.cat(vals, dim=0)[:max_points]
    return X, G, F


def discover_scaling_features(
    model,
    datagen,
    Nxvars: int,
    device=None,
    max_batches: int = 8,
    max_points: int = 20000,
    max_group_size: int = 2,
    rel_std_threshold: float = 0.1,
    min_points: int = 200,
) -> List[ScaleSpec]:
    """
    Detect approximate homogeneity over small variable groups S.

    Returns a list of ScaleSpec with estimated degree k_hat and dispersion.
    """
    X, G, F = _sample_gradients_and_values(model, datagen, max_batches, max_points, device)
    if X is None:
        return []

    N, Nx = X.shape
    assert Nx == Nxvars

    # Avoid division by tiny f
    Fabs = F.abs()
    f_max = float(Fabs.max())
    f_eps = max(1e-12, 1e-6 * f_max)
    mask_f = Fabs > f_eps

    specs: List[ScaleSpec] = []

    # Candidate index sets S: all 1- and 2-element subsets
    idxs = list(range(Nxvars))
    for r in range(1, max_group_size + 1):
        for S in itertools.combinations(idxs, r):
            S_list = list(S)
            # Compute numerator: sum_{i in S} x_i * f_{x_i}
            xS = X[:, S_list]  # [N, |S|]
            gS = G[:, S_list]  # [N, |S|]
            num = (xS * gS).sum(dim=1)  # [N]
            # Ratio r_S = num / f
            denom = F.clone()
            denom[~mask_f] = float("nan")  # ignore tiny f
            rS = num / (denom + 1e-24)

            # Drop NaNs, compute stats
            rS_valid = rS[torch.isfinite(rS)]
            if rS_valid.numel() < min_points:
                continue

            mean = float(rS_valid.mean())
            std = float(rS_valid.std(unbiased=False))
            rel_std = std / (abs(mean) + 1e-12)

            if rel_std < rel_std_threshold:
                # Pretty consistent homogeneity over S
                name = f"scale_S{S}_k≈{mean:.3g}"
                specs.append(
                    ScaleSpec(
                        name=name,
                        indices=S_list,
                        k_hat=mean,
                        mean=mean,
                        std=std,
                        rel_std=rel_std,
                        n_points=int(rS_valid.numel()),
                    )
                )

    return specs


# ---------------------------------------------------------------------------
# Oracle-based scaling probe
# ---------------------------------------------------------------------------


def _sample_values_only(model, datagen, max_batches=4, max_points=5000, device=None):
    """Sample X and F = model(X) without computing gradients."""
    xs, vals = [], []
    n_points = 0
    data_iter = datagen() if callable(datagen) else datagen

    for bi, batch in enumerate(data_iter):
        if bi >= max_batches or n_points >= max_points:
            break
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        dev = device or next(model.parameters()).device
        x = x.to(dev)
        B = x.shape[0]
        x_flat = x.view(B, -1)
        with torch.no_grad():
            y = model(x_flat)
        f = y[:, 0] if y.dim() == 2 else y.view(-1)
        xs.append(x_flat.detach().cpu())
        vals.append(f.detach().cpu())
        n_points += B

    if not xs:
        return None, None
    X = torch.cat(xs, dim=0)[:max_points]
    F = torch.cat(vals, dim=0)[:max_points]
    return X, F


def _canonical_scale_group(indices, Nxvars: int) -> Optional[Tuple[int, ...]]:
    try:
        group = tuple(sorted({int(i) for i in indices}))
    except Exception:
        return None
    if len(group) < 2 or any(i < 0 or i >= int(Nxvars) for i in group):
        return None
    return group


def _probe_oracle_scale_groups_from_samples(
    model,
    X: torch.Tensor,
    F: torch.Tensor,
    *,
    Nxvars: int,
    group_specs: List[ScaleSpec],
    device=None,
    lambda_values=(0.7, 0.85, 1.2, 1.5),
    rel_std_threshold: float = 0.08,
    min_points: int = 100,
) -> List[ScaleSpec]:
    """Verify proposed raw-axis homogeneity groups by simultaneous scaling."""
    if X is None or F is None or not group_specs:
        return []
    dev = device or next(model.parameters()).device
    x_min = X.min(dim=0).values
    x_max = X.max(dim=0).values
    x_margin = 0.10 * (x_max - x_min)
    x_lo = x_min - x_margin
    x_hi = x_max + x_margin
    f_floor = max(float(torch.quantile(F.abs(), 0.05)), 1e-12)
    log_lambdas = torch.tensor(
        [math.log(float(lam)) for lam in lambda_values], dtype=X.dtype
    )

    proposals: Dict[Tuple[int, ...], ScaleSpec] = {}
    for spec in group_specs:
        group = _canonical_scale_group(getattr(spec, "indices", ()), Nxvars)
        if group is None:
            continue
        try:
            expected = float(spec.k_hat)
            scatter = abs(float(spec.std))
            proposal_rel_std = float(spec.rel_std)
            if not all(
                math.isfinite(v) for v in (expected, scatter, proposal_rel_std)
            ):
                continue
        except Exception:
            continue
        previous = proposals.get(group)
        if previous is None or proposal_rel_std < float(previous.rel_std):
            proposals[group] = spec

    verified: List[ScaleSpec] = []
    for group, proposal in sorted(proposals.items()):
        mask = F.abs() > f_floor
        for idx in group:
            mask = mask & (X[:, idx].abs() > 1e-8)
        if int(mask.sum()) < int(min_points):
            continue
        X_sub = X[mask]
        F_sub = F[mask]
        n_sub = int(X_sub.shape[0])
        log_ratios: List[torch.Tensor] = []
        usable_log_lambdas: List[float] = []
        for li, lam in enumerate(lambda_values):
            X_scaled = X_sub.clone()
            for idx in group:
                X_scaled[:, idx] *= float(lam)
            in_range = torch.ones(n_sub, dtype=torch.bool)
            for idx in group:
                in_range &= (X_scaled[:, idx] >= x_lo[idx]) & (
                    X_scaled[:, idx] <= x_hi[idx]
                )
            if int(in_range.sum()) < max(50, int(0.3 * n_sub)):
                continue
            try:
                with torch.no_grad():
                    y_scaled = model(X_scaled.to(dev))
                F_scaled = (
                    y_scaled[:, 0]
                    if y_scaled.dim() == 2
                    else y_scaled.reshape(-1)
                ).detach().cpu()
            except Exception:
                continue
            ratio = F_scaled / (F_sub + 1e-30)
            valid = (ratio > 0) & torch.isfinite(ratio) & in_range
            if int(valid.sum()) < max(50, int(0.2 * n_sub)):
                continue
            row = torch.full((n_sub,), float("nan"), dtype=ratio.dtype)
            row[valid] = torch.log(ratio[valid])
            log_ratios.append(row)
            usable_log_lambdas.append(float(log_lambdas[li]))

        if len(usable_log_lambdas) < 3:
            continue
        LR = torch.stack(log_ratios, dim=0)
        LL = torch.tensor(usable_log_lambdas, dtype=LR.dtype).reshape(-1, 1)
        finite = torch.isfinite(LR)
        counts = finite.sum(dim=0)
        numerator = torch.where(finite, LL * LR, torch.zeros_like(LR)).sum(dim=0)
        denominator = torch.where(finite, LL.square(), torch.zeros_like(LR)).sum(dim=0)
        valid_points = (counts >= 3) & (denominator > 1e-20)
        degrees = numerator[valid_points] / denominator[valid_points]
        degrees = degrees[torch.isfinite(degrees)]
        if int(degrees.numel()) < int(min_points):
            continue
        degree = float(degrees.median())
        degree_std = float(degrees.std(unbiased=False))
        rel_std = degree_std / (abs(degree) + 1e-12)
        expected = float(proposal.k_hat)
        agreement_tol = max(
            0.10,
            0.10 * abs(expected),
            3.0 * abs(float(proposal.std)),
        )
        if rel_std >= float(rel_std_threshold) or abs(degree - expected) > agreement_tol:
            continue
        joined = "x".join(str(i) for i in group)
        verified.append(
            ScaleSpec(
                name=f"oracle_joint_x{joined}_k≈{degree:.3g}",
                indices=list(group),
                k_hat=degree,
                mean=degree,
                std=degree_std,
                rel_std=rel_std,
                n_points=int(degrees.numel()),
                oracle_verified=True,
                oracle_k=degree,
                oracle_rel_std=rel_std,
            )
        )
    return verified


def probe_oracle_scaling_groups(
    model,
    datagen,
    Nxvars: int,
    group_specs: List[ScaleSpec],
    device=None,
    lambda_values=(0.7, 0.85, 1.2, 1.5),
    max_batches: int = 4,
    max_points: int = 5000,
    rel_std_threshold: float = 0.08,
) -> List[ScaleSpec]:
    """Directly verify proposed joint raw-axis scaling groups."""
    X, F = _sample_values_only(model, datagen, max_batches, max_points, device)
    if X is None:
        return []
    return _probe_oracle_scale_groups_from_samples(
        model,
        X,
        F,
        Nxvars=Nxvars,
        group_specs=group_specs,
        device=device,
        lambda_values=lambda_values,
        rel_std_threshold=rel_std_threshold,
    )


def probe_oracle_scaling(
    model,
    datagen,
    Nxvars: int,
    device=None,
    lambda_values=(0.7, 0.85, 1.2, 1.5),
    max_batches: int = 4,
    max_points: int = 5000,
    rel_std_threshold: float = 0.08,
    compound_targets: Optional[List["TrigProbeTarget"]] = None,
    gradient_specs: Optional[List[ScaleSpec]] = None,
) -> List[ScaleSpec]:
    """
    Directly probe the NN surrogate to discover per-variable scaling degrees.

    For each variable x_j, evaluate f(λ·x_j, x_rest) / f(x_j, x_rest) across
    multiple λ values and data points.  Fit power-law degree k_j from the ratios.

    Also runs a joint test for pairs of clean single-variable degrees to check
    additivity (k_joint ≈ k_i + k_j when scaling both simultaneously).

    Parameters
    ----------
    compound_targets : optional list of TrigProbeTarget
        Additional compound variable targets to probe (e.g., x2-x3, x0*x1).
        These are probed alongside trivial axes using pivot-based perturbation.

    Returns ScaleSpec entries with ``oracle_verified=True``.
    """
    X, F = _sample_values_only(model, datagen, max_batches, max_points, device)
    if X is None:
        return []

    N, Nx = X.shape
    assert Nx == Nxvars

    dev = device or next(model.parameters()).device

    # Filter out points where |F| is tiny (avoid division issues)
    Fabs = F.abs()
    f_pct5 = float(torch.quantile(Fabs, 0.05))
    mask_f = Fabs > max(f_pct5, 1e-12)

    # Data range per column (with 10% margin)
    x_min = X.min(dim=0).values
    x_max = X.max(dim=0).values
    x_range = x_max - x_min
    x_lo = x_min - 0.10 * x_range
    x_hi = x_max + 0.10 * x_range

    log_lambdas = torch.tensor([math.log(lam) for lam in lambda_values])

    specs: List[ScaleSpec] = []

    # Build lookup for gradient-based single-variable specs (for fallback)
    _grad_lookup: Dict[int, ScaleSpec] = {}
    if gradient_specs:
        for gsp in gradient_specs:
            if len(gsp.indices) == 1:
                _grad_lookup[gsp.indices[0]] = gsp

    # --- Build target list: trivial + compound ---
    targets: List[TrigProbeTarget] = []

    # Trivial targets for each variable
    for j in range(Nxvars):
        targets.append(TrigProbeTarget(
            name=f"x{j}",
            indices=(j,),
            expr=None,
            kind="trivial",
            pivot_idx=j,
        ))

    # Add compound targets (skip duplicates)
    if compound_targets:
        for ct in compound_targets:
            if any(t.name == ct.name for t in targets):
                continue
            targets.append(ct)

    # --- Phase A: per-target probe (trivial and compound) ---
    for target in targets:
        use_reciprocal = False
        reciprocal_threshold = 0.0  # set later if reciprocal probe activates

        # Compute z = compound(X) for masking
        z_full = _eval_compound_z(target, X)

        # Mask points where z is near zero (avoid division issues)
        mask_z = z_full.abs() > 1e-8
        # Also mask points where ANY involved variable is near zero (for product/ratio stability)
        for idx in target.indices:
            mask_z = mask_z & (X[:, idx].abs() > 1e-8)

        mask = mask_f & mask_z
        if mask.sum() < 100:
            continue

        X_sub = X[mask]
        F_sub = F[mask]
        z_sub = z_full[mask]
        Nsub = X_sub.shape[0]

        # Per-axis centering: evaluate at z=0 to estimate the additive offset.
        # For f(x_rest, z) = g(x_rest)*z^k + h(x_rest), f(x_rest, 0) = h(x_rest),
        # so f - f_ref = g*z^k (pure monomial, offset-free).
        # Falls back to un-centered if F_ref is non-finite (e.g. z=0 causes a singularity).
        X_ref = _perturb_for_compound(X_sub, torch.zeros(Nsub), target)
        with torch.no_grad():
            y_ref = model(X_ref.to(dev))
        F_ref_raw = (y_ref[:, 0] if y_ref.dim() == 2 else y_ref.view(-1)).cpu()

        use_centering = torch.isfinite(F_ref_raw).sum() > Nsub * 0.8
        if use_centering:
            # Check if reference values are extreme (extrapolation region).
            # For negative-k variables (e.g. f ∝ 1/x), evaluating at x=0
            # gives NN extrapolation with large, inaccurate values.
            fin_mask = torch.isfinite(F_ref_raw)
            median_ref = float(F_ref_raw[fin_mask].abs().median())
            median_sub = float(F_sub.abs().median())
            if median_ref > 10.0 * max(median_sub, 1e-12):
                use_centering = False

        # --- Reciprocal probe for negative-k singularity ---
        # When centering guard tripped, try probing 1/f instead of f.
        # For f ~ z^k with k<0, 1/f ~ z^{-k} with -k>0, so centering works.
        if not use_centering:
            reciprocal_threshold = max(float(torch.quantile(F_sub.abs(), 0.05)), 1e-10)
            safe_recip = F_sub.abs() > reciprocal_threshold
            n_safe = int(safe_recip.sum())
            if n_safe > 100 and n_safe > Nsub * 0.5:
                F_inv = 1.0 / F_sub[safe_recip]
                F_inv_ref = 1.0 / F_ref_raw[safe_recip]
                fin_inv = torch.isfinite(F_inv_ref)
                if fin_inv.sum() > n_safe * 0.5:
                    med_inv_ref = float(F_inv_ref[fin_inv].abs().median())
                    med_inv = float(F_inv.abs().median())
                    if med_inv_ref < 5.0 * max(med_inv, 1e-12):
                        # Reciprocal centering is viable
                        use_centering = True
                        use_reciprocal = True
                        # Re-subset to safe points only
                        X_sub = X_sub[safe_recip]
                        z_sub = z_sub[safe_recip]
                        Nsub = X_sub.shape[0]
                        F_sub = F_sub[safe_recip]
                        F_ref_raw = F_ref_raw[safe_recip]
                        F_inv_full = 1.0 / F_sub
                        F_inv_ref_full = 1.0 / F_ref_raw
                        F_sub_c = F_inv_full - F_inv_ref_full
                        F_ref = F_inv_ref_full
                        mask_centered = F_sub_c.abs() > max(
                            float(torch.quantile(F_sub_c.abs(), 0.1)), 1e-8
                        )
                        print(f"    [{target.name}] reciprocal probe "
                              f"(negative-k singularity detected)")

        if use_centering and not use_reciprocal:
            F_sub_c = F_sub - F_ref_raw
            # Mask points where centered value is too small (near the zero of the monomial)
            mask_centered = F_sub_c.abs() > max(float(torch.quantile(F_sub_c.abs(), 0.1)), 1e-8)
            F_ref = F_ref_raw
        elif not use_centering:
            # Last-resort fallback: no centering
            F_sub_c = F_sub
            mask_centered = torch.ones(Nsub, dtype=torch.bool)
            F_ref = torch.zeros(Nsub)

        # Collect log-ratios for each λ
        log_ratios_per_lam: List[torch.Tensor] = []
        valid_log_lams: List[float] = []

        for li, lam in enumerate(lambda_values):
            # Scale z → λ*z using pivot-based perturbation
            z_scaled = lam * z_sub
            X_scaled = _perturb_for_compound(X_sub, z_scaled, target)

            # Range check: verify ALL indices in target.indices are in range
            in_range = torch.ones(Nsub, dtype=torch.bool)
            for idx in target.indices:
                col_scaled = X_scaled[:, idx]
                in_range = in_range & (col_scaled >= x_lo[idx]) & (col_scaled <= x_hi[idx])
            if in_range.sum() < max(50, Nsub * 0.3):
                continue

            with torch.no_grad():
                y_s = model(X_scaled.to(dev))
            F_scaled = (y_s[:, 0] if y_s.dim() == 2 else y_s.view(-1)).cpu()

            if use_reciprocal:
                safe = F_scaled.abs() > reciprocal_threshold
                F_scaled = torch.where(
                    safe,
                    1.0 / F_scaled,
                    torch.full_like(F_scaled, float("nan")),
                )

            F_scaled_c = F_scaled - F_ref
            ratio = F_scaled_c / (F_sub_c + 1e-30)
            # Mask sign-flips and non-finite
            valid = (ratio > 0) & torch.isfinite(ratio) & in_range & mask_centered
            if valid.sum() < max(50, Nsub * 0.2):
                continue

            lr = torch.full((Nsub,), float("nan"), dtype=ratio.dtype)
            lr[valid] = torch.log(ratio[valid])
            log_ratios_per_lam.append(lr)
            valid_log_lams.append(float(log_lambdas[li]))

        if len(valid_log_lams) < 3:
            if use_reciprocal and target.kind == "trivial":
                gsp = _grad_lookup.get(target.pivot_idx)
                if gsp and gsp.k_hat < 0 and gsp.rel_std < 0.015:
                    specs.append(ScaleSpec(
                        name=f"oracle_scale_x{target.pivot_idx}_k≈{gsp.k_hat:.3g}",
                        indices=[target.pivot_idx],
                        k_hat=gsp.k_hat, mean=gsp.k_hat,
                        std=gsp.std, rel_std=gsp.rel_std,
                        n_points=gsp.n_points, oracle_verified=True,
                        oracle_k=gsp.k_hat, oracle_rel_std=gsp.rel_std,
                    ))
                    print(f"    [{target.name}] reciprocal: ratio test failed "
                          f"(n_lam={len(valid_log_lams)}<3), using gradient "
                          f"k≈{gsp.k_hat:.3f} (rel_std={gsp.rel_std:.4f})")
                else:
                    print(f"    [{target.name}] reciprocal: ratio test failed "
                          f"(n_lam={len(valid_log_lams)}<3), no gradient fallback")
            continue

        # Stack: [n_lam, Nsub]
        LR = torch.stack(log_ratios_per_lam, dim=0)
        LL = torch.tensor(valid_log_lams)  # [n_lam]

        # Per-point least-squares fit: k_n = (LL . LR[:, n]) / (LL . LL)
        denom_ls = float((LL * LL).sum())
        if denom_ls < 1e-20:
            continue

        k_per_point = torch.zeros(Nsub)
        valid_point = torch.zeros(Nsub, dtype=torch.bool)
        for n in range(Nsub):
            col = LR[:, n]
            finite_mask = torch.isfinite(col)
            if finite_mask.sum() < 3:
                continue
            num = float((LL[finite_mask] * col[finite_mask]).sum())
            den = float((LL[finite_mask] * LL[finite_mask]).sum())
            if den < 1e-20:
                continue
            k_per_point[n] = num / den
            valid_point[n] = True

        k_valid = k_per_point[valid_point]
        if k_valid.numel() < 100:
            if use_reciprocal and target.kind == "trivial":
                gsp = _grad_lookup.get(target.pivot_idx)
                if gsp and gsp.k_hat < 0 and gsp.rel_std < 0.015:
                    specs.append(ScaleSpec(
                        name=f"oracle_scale_x{target.pivot_idx}_k≈{gsp.k_hat:.3g}",
                        indices=[target.pivot_idx],
                        k_hat=gsp.k_hat, mean=gsp.k_hat,
                        std=gsp.std, rel_std=gsp.rel_std,
                        n_points=gsp.n_points, oracle_verified=True,
                        oracle_k=gsp.k_hat, oracle_rel_std=gsp.rel_std,
                    ))
                    print(f"    [{target.name}] reciprocal: ratio test failed "
                          f"(n_valid_k={int(k_valid.numel())}<100), using gradient "
                          f"k≈{gsp.k_hat:.3f} (rel_std={gsp.rel_std:.4f})")
                else:
                    print(f"    [{target.name}] reciprocal: ratio test failed "
                          f"(n_valid_k={int(k_valid.numel())}<100), no gradient fallback")
            continue

        k_hat = float(k_valid.median())
        if use_reciprocal:
            k_hat = -k_hat
        k_std = float(k_valid.std(unbiased=False))
        rel_std = k_std / (abs(k_hat) + 1e-12)

        if rel_std < rel_std_threshold:
            # Build name based on target type
            if target.kind == "trivial":
                name = f"oracle_scale_x{target.pivot_idx}_k≈{k_hat:.3g}"
                compound_name = ""
                compound_expr = None
                indices = [target.pivot_idx]
            else:
                name = f"oracle_scale_{target.name}_k≈{k_hat:.3g}"
                compound_name = target.name
                compound_expr = target.expr
                indices = list(target.indices)

            specs.append(
                ScaleSpec(
                    name=name,
                    indices=indices,
                    k_hat=k_hat,
                    mean=k_hat,
                    std=k_std,
                    rel_std=rel_std,
                    n_points=int(k_valid.numel()),
                    oracle_verified=True,
                    oracle_k=k_hat,
                    oracle_rel_std=rel_std,
                    compound_name=compound_name,
                    compound_expr=compound_expr,
                )
            )
        elif use_reciprocal and target.kind == "trivial":
            gsp = _grad_lookup.get(target.pivot_idx)
            if gsp and gsp.k_hat < 0 and gsp.rel_std < 0.015:
                specs.append(ScaleSpec(
                    name=f"oracle_scale_x{target.pivot_idx}_k≈{gsp.k_hat:.3g}",
                    indices=[target.pivot_idx],
                    k_hat=gsp.k_hat, mean=gsp.k_hat,
                    std=gsp.std, rel_std=gsp.rel_std,
                    n_points=gsp.n_points, oracle_verified=True,
                    oracle_k=gsp.k_hat, oracle_rel_std=gsp.rel_std,
                ))
                print(f"    [{target.name}] reciprocal: rel_std={rel_std:.4f}"
                      f">{rel_std_threshold}, using gradient "
                      f"k≈{gsp.k_hat:.3f} (rel_std={gsp.rel_std:.4f})")
            else:
                print(f"    [{target.name}] reciprocal: rel_std={rel_std:.4f}"
                      f">{rel_std_threshold}, no gradient fallback available")

    # --- Phase B: direct joint-group verification ------------------------
    # Multi-index gradient specs are proposals, not certificates.  Also keep
    # the established additive-degree pair proposals from verified singletons.
    group_proposals: Dict[Tuple[int, ...], ScaleSpec] = {}
    for gsp in gradient_specs or []:
        group = _canonical_scale_group(getattr(gsp, "indices", ()), Nxvars)
        if group is not None:
            group_proposals[group] = gsp

    trivial_specs = {
        sp.indices[0]: sp
        for sp in specs
        if len(sp.indices) == 1 and not sp.compound_name
    }
    for i, sp_i in trivial_specs.items():
        for j_, sp_j in trivial_specs.items():
            if j_ <= i:
                continue
            group = (int(i), int(j_))
            group_proposals.setdefault(
                group,
                ScaleSpec(
                    name=f"singleton_sum_x{i}x{j_}",
                    indices=list(group),
                    k_hat=float(sp_i.k_hat + sp_j.k_hat),
                    mean=float(sp_i.k_hat + sp_j.k_hat),
                    std=float(math.hypot(sp_i.std, sp_j.std)),
                    rel_std=max(float(sp_i.rel_std), float(sp_j.rel_std)),
                    n_points=min(int(sp_i.n_points), int(sp_j.n_points)),
                ),
            )

    specs.extend(
        _probe_oracle_scale_groups_from_samples(
            model,
            X,
            F,
            Nxvars=Nxvars,
            group_specs=list(group_proposals.values()),
            device=device,
            lambda_values=lambda_values,
            rel_std_threshold=rel_std_threshold,
        )
    )

    return specs


# ---------------------------------------------------------------------------
# Trig z-scaling probe: monomial degree in cos/sin
# ---------------------------------------------------------------------------


def _closest_preimage(target_angle: torch.Tensor, current_angle: torch.Tensor,
                      trig_fn: str, omega: float) -> torch.Tensor:
    """Return x' such that trig_fn(ω·x') = trig_fn(target_angle), closest to current x."""
    TWO_PI = 2.0 * math.pi
    if trig_fn in {"cos", "one_minus_cos"}:
        # candidates: ±target_angle + 2πn
        c1 = target_angle
        c2 = -target_angle
    else:
        # sin: candidates: target_angle + 2πn, π - target_angle + 2πn
        c1 = target_angle
        c2 = math.pi - target_angle

    # Shift each candidate to the period closest to current_angle
    shift1 = torch.round((current_angle - c1) / TWO_PI) * TWO_PI
    shift2 = torch.round((current_angle - c2) / TWO_PI) * TWO_PI
    cand1 = c1 + shift1
    cand2 = c2 + shift2

    # Pick whichever is closest to current_angle
    use_c2 = (cand2 - current_angle).abs() < (cand1 - current_angle).abs()
    best_angle = torch.where(use_c2, cand2, cand1)
    return best_angle / omega


def _build_omega_grid(span: float) -> List[float]:
    """Build a sorted, deduplicated grid of ω candidates for a given axis span."""
    PI = math.pi
    raw = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10,
           PI / 4, PI / 3, PI / 2, 2 * PI / 3, PI, 3 * PI / 2,
           2 * PI, 3 * PI, 4 * PI, 5 * PI]
    raw = sorted(set(raw))

    # Filter by coverage: keep ω where quarter-cycle ≤ ω·span ≤ 5 full cycles
    filtered = [w for w in raw if PI / 4 <= w * span <= 10 * PI]

    # Deduplicate values within 5% of each other
    deduped: List[float] = []
    for w in filtered:
        if deduped and w < deduped[-1] * 1.05:
            continue
        deduped.append(w)
    return deduped


def _classify_compound_expr(expr) -> Optional[Tuple[str, Tuple[int, ...], int]]:
    """
    Classify a compound expression AST into a known type.

    Returns
    -------
    (kind, indices, pivot_idx) or None if not recognized.

    Supported forms:
    - Var(i): "trivial", (i,), i
    - Add(Var(i), Mul(ConstNode(-1), Var(j))): "difference" (i - j), (i, j), i
    - Add(Var(i), Var(j)): "sum", (i, j), i
    - Mul(Var(i), Var(j)): "product", (i, j), i
    - Mul(Var(i), Pow(Var(j), -1)): "ratio" (i / j), (i, j), i
    """
    from nestynet_sr.sr_core.bridges import AddNode, AtomNode, MulNode, PowNode

    if isinstance(expr, AtomNode):
        kind_str = str(getattr(expr, "kind", "")).lower()
        if kind_str in ("var", "x", "input") and len(expr.var_idxs) == 1:
            idx = int(expr.var_idxs[0])
            return ("trivial", (idx,), idx)
        return None

    if isinstance(expr, AddNode):
        # Check for difference: x_i + (-1 * x_j)
        left, right = expr.left, expr.right
        left_idx = _extract_var_idx(left)
        right_idx = _extract_var_idx(right)

        if left_idx is not None and right_idx is not None:
            # Simple sum: x_i + x_j
            return ("sum", (left_idx, right_idx), left_idx)

        # Check for difference: x_i + ((-1) * x_j)
        if left_idx is not None and isinstance(right, MulNode):
            neg_one_side, var_side = right.left, right.right
            # Try both orders
            neg_val = _extract_const_value(neg_one_side)
            if neg_val is not None and abs(neg_val + 1.0) < 1e-12:
                j_idx = _extract_var_idx(var_side)
                if j_idx is not None:
                    return ("difference", (left_idx, j_idx), left_idx)
            neg_val = _extract_const_value(var_side)
            if neg_val is not None and abs(neg_val + 1.0) < 1e-12:
                j_idx = _extract_var_idx(neg_one_side)
                if j_idx is not None:
                    return ("difference", (left_idx, j_idx), left_idx)

        return None

    if isinstance(expr, MulNode):
        left, right = expr.left, expr.right
        left_idx = _extract_var_idx(left)
        right_idx = _extract_var_idx(right)

        if left_idx is not None and right_idx is not None:
            # Simple product: x_i * x_j
            return ("product", (left_idx, right_idx), left_idx)

        # Check for ratio: x_i * x_j^(-1)
        if left_idx is not None and isinstance(right, PowNode):
            if abs(right.exponent + 1.0) < 1e-12:
                j_idx = _extract_var_idx(right.base)
                if j_idx is not None:
                    return ("ratio", (left_idx, j_idx), left_idx)

        return None

    return None


def _extract_var_idx(node) -> Optional[int]:
    """Extract variable index from a Var node, else None."""
    from nestynet_sr.sr_core.bridges import AtomNode

    if isinstance(node, AtomNode):
        kind_str = str(getattr(node, "kind", "")).lower()
        if kind_str in ("var", "x", "input") and len(node.var_idxs) == 1:
            return int(node.var_idxs[0])
    return None


def _extract_const_value(node) -> Optional[float]:
    """Extract constant value from a ConstNode, else None."""
    from nestynet_sr.sr_core.bridges import ConstNode

    if isinstance(node, ConstNode):
        return float(node.value)
    return None


def _compound_to_probe_target(compound_expr, var_idxs: Tuple[int, ...]) -> Optional[TrigProbeTarget]:
    """Convert a compound AST to a TrigProbeTarget with perturber info."""
    from nestynet_sr.sr_core.bridges import ast_to_human_readable

    result = _classify_compound_expr(compound_expr)
    if result is None:
        return None

    kind, indices, pivot_idx = result
    try:
        name = ast_to_human_readable(compound_expr)
        # Simplify common patterns for cleaner display
        name = name.replace("(", "").replace(")", "").replace(" ", "")
        name = name.replace("+-1*", "-").replace("*-1*", "/")
    except Exception:
        name = f"compound_{indices}"

    return TrigProbeTarget(
        name=name,
        indices=indices,
        expr=compound_expr,
        kind=kind,
        pivot_idx=pivot_idx,
    )


def _perturb_for_compound(X: torch.Tensor, target_z: torch.Tensor,
                          target: TrigProbeTarget) -> torch.Tensor:
    """
    Perturb X so that the compound variable evaluates to target_z.

    For each compound type, we adjust the pivot variable while keeping
    other variables fixed:
    - trivial (z = x_j): set X[:, j] = target_z
    - difference (z = x_i - x_j): set X[:, i] = target_z + X[:, j]
    - sum (z = x_i + x_j): set X[:, i] = target_z - X[:, j]
    - product (z = x_i * x_j): set X[:, i] = target_z / X[:, j]
    - ratio (z = x_i / x_j): set X[:, i] = target_z * X[:, j]
    """
    X_new = X.clone()
    pivot = target.pivot_idx

    if target.kind == "trivial":
        X_new[:, pivot] = target_z
    elif target.kind == "difference":
        # z = x_i - x_j, pivot is i
        j = [k for k in target.indices if k != pivot][0]
        X_new[:, pivot] = target_z + X[:, j]
    elif target.kind == "sum":
        # z = x_i + x_j, pivot is i
        j = [k for k in target.indices if k != pivot][0]
        X_new[:, pivot] = target_z - X[:, j]
    elif target.kind == "product":
        # z = x_i * x_j, pivot is i
        j = [k for k in target.indices if k != pivot][0]
        X_new[:, pivot] = target_z / (X[:, j] + 1e-30)
    elif target.kind == "ratio":
        # z = x_i / x_j, pivot is i
        j = [k for k in target.indices if k != pivot][0]
        X_new[:, pivot] = target_z * X[:, j]
    else:
        # Unknown kind, just set pivot directly
        X_new[:, pivot] = target_z

    return X_new


def _eval_compound_z(target: TrigProbeTarget, X: torch.Tensor) -> torch.Tensor:
    """Evaluate the compound variable z for a TrigProbeTarget."""
    if target.kind == "trivial":
        return X[:, target.pivot_idx]
    elif target.expr is not None:
        from nestynet_sr.sr_core.bridges import eval_input_expr
        z = eval_input_expr(target.expr, X)
        return z.squeeze(-1) if z.dim() > 1 else z
    else:
        # Fallback: compute from kind and indices
        i, j = target.indices[0], target.indices[1] if len(target.indices) > 1 else target.indices[0]
        if target.kind == "difference":
            return X[:, i] - X[:, j]
        elif target.kind == "sum":
            return X[:, i] + X[:, j]
        elif target.kind == "product":
            return X[:, i] * X[:, j]
        elif target.kind == "ratio":
            return X[:, i] / (X[:, j] + 1e-30)
        else:
            return X[:, target.pivot_idx]


def _find_inrange_trig_zero(
    trig_fn: str, omega: float, z_min: float, z_max: float
) -> Optional[float]:
    """Find a trig-zero within [z_min, z_max], or None if none exists.

    For sin(ω*z), zeros are at z = n*π/ω for integer n.
    For cos(ω*z), zeros are at z = (2n+1)*π/(2*ω) for integer n.
    For 1-cos(ω*z), zeros are at z = 2n*π/ω for integer n.

    Returns the zero closest to the midpoint of the range.
    """
    candidates = []
    for n in range(-20, 21):
        if trig_fn == "sin":
            z = n * math.pi / omega
        elif trig_fn == "one_minus_cos":
            z = 2 * n * math.pi / omega
        else:  # cos
            z = (2 * n + 1) * math.pi / (2 * omega)
        if z_min <= z <= z_max:
            candidates.append(z)

    if not candidates:
        return None

    # Return the one closest to the midpoint
    z_mid = (z_min + z_max) / 2
    return min(candidates, key=lambda z: abs(z - z_mid))


def probe_trig_scaling(
    model,
    datagen,
    Nxvars: int,
    device=None,
    oracle_specs: Optional[List["ScaleSpec"]] = None,
    compound_targets: Optional[List[TrigProbeTarget]] = None,
    lambda_values=(0.8, 0.9, 1.1, 1.2),
    max_batches: int = 4,
    max_points: int = 5000,
    rel_std_threshold: float = 0.10,
    z_abs_min: float = 0.05,
    z_abs_max: float = 0.85,
    oracle_skip_rel_std: float = 0.02,
) -> List[TrigScaleSpec]:
    """
    Probe whether the model's dependence on a trig variable is a monomial.

    Self-contained: scans a grid of ω candidates per target (no FFT needed).
    For each target (trivial axis or compound variable) and each ω,
    define z = trig(ω·compound).
    Perturb compound so that z scales by λ, then fit power-law degree k from
    log(F_perturbed / F_original) vs log(λ).  Low rel_std → monomial in z.

    Tries both cos and sin for each (target, ω); reports whichever gives lowest rel_std.

    Parameters
    ----------
    oracle_specs : optional list of ScaleSpec
        If provided, axes with a clean single-variable polynomial scaling
        (oracle_rel_std < oracle_skip_rel_std) are skipped.
    compound_targets : optional list of TrigProbeTarget
        Additional compound variable targets to probe (e.g., x2-x3, x0*x1).
        These are probed alongside trivial axes.
    """
    X, F = _sample_values_only(model, datagen, max_batches, max_points, device)
    if X is None:
        print("  [Trig Scaling] No data sampled, skipping.")
        return []

    N, Nx = X.shape
    dev = device or next(model.parameters()).device

    # Determine which axes/compounds to skip (already clean polynomial)
    skip_axes: set = set()
    skip_compounds: set = set()
    if oracle_specs:
        for osp in oracle_specs:
            rstd = osp.oracle_rel_std if osp.oracle_rel_std is not None else osp.rel_std
            if rstd < oracle_skip_rel_std:
                if len(osp.indices) == 1:
                    skip_axes.add(osp.indices[0])
                elif osp.compound_name:
                    skip_compounds.add(osp.compound_name)

    # Build target list: trivial targets for each non-skipped axis
    targets: List[TrigProbeTarget] = []
    for j in range(Nxvars):
        if j not in skip_axes:
            targets.append(TrigProbeTarget(
                name=f"x{j}",
                indices=(j,),
                expr=None,
                kind="trivial",
                pivot_idx=j,
            ))

    # Add compound targets (skip if clean polynomial as compound or all indices polynomial)
    if compound_targets:
        for ct in compound_targets:
            if ct.name in skip_compounds:
                print(f"  [Trig Scaling] {ct.name}: skipping (clean polynomial compound)")
                continue
            # Skip if all indices are already skipped as polynomial
            if all(i in skip_axes for i in ct.indices):
                continue
            # Skip duplicates (same name)
            if any(t.name == ct.name for t in targets):
                continue
            targets.append(ct)

    n_trivial = sum(1 for t in targets if t.kind == "trivial")
    n_compound = len(targets) - n_trivial
    print(
        f"  [Trig Scaling] Sampled {N} points, probing {n_trivial} trivial + {n_compound} compound target(s)"
        f"{f' (skipping axes {sorted(skip_axes)} as polynomial)' if skip_axes else ''}"
    )

    # Filter out points where |F| is tiny
    Fabs = F.abs()
    f_pct5 = float(torch.quantile(Fabs, 0.05))
    mask_f = Fabs > max(f_pct5, 1e-12)
    n_f_ok = int(mask_f.sum())
    print(f"  [Trig Scaling] |F| filter: {n_f_ok}/{N} points pass (5th-pct={f_pct5:.3e})")

    # Data range per column (with 10% margin)
    x_min = X.min(dim=0).values
    x_max = X.max(dim=0).values
    x_range = x_max - x_min
    x_lo = x_min - 0.10 * x_range
    x_hi = x_max + 0.10 * x_range

    log_lambdas = torch.tensor([math.log(lam) for lam in lambda_values])

    results: List[TrigScaleSpec] = []

    for target in targets:
        # Compute compound z values for this target
        z_compound = _eval_compound_z(target, X)

        # Compute span of compound variable
        z_min = float(z_compound.min())
        z_max = float(z_compound.max())
        span_z = z_max - z_min
        if span_z < 1e-12:
            continue

        omega_grid = _build_omega_grid(span_z)
        if not omega_grid:
            print(f"  [Trig Scaling] {target.name}: no viable ω candidates (span={span_z:.3g})")
            continue

        print(f"  [Trig Scaling] {target.name}: span={span_z:.3g}, {len(omega_grid)} ω candidates")

        best_rel_std = float("inf")
        best_result: Optional[TrigScaleSpec] = None

        for omega in omega_grid:
            for basis_fn in ["cos", "sin", "one_minus_cos"]:
                if basis_fn == "sin":
                    trig_fn = "sin"
                    trig_func = torch.sin
                    inv_func = torch.asin
                else:
                    # ``one_minus_cos`` is a cosine-family phase hint.  Keep
                    # trig_fn="cos" for downstream compatibility while using
                    # a different scaling basis inside this probe.
                    trig_fn = "cos"
                    trig_func = torch.cos
                    inv_func = torch.acos

                # z_trig = trig(ω · z_compound)
                phase = omega * z_compound
                if basis_fn == "one_minus_cos":
                    z_trig = 1.0 - torch.cos(phase)
                else:
                    z_trig = trig_func(phase)

                # Mask: safe z values and |F| filter
                if basis_fn == "one_minus_cos":
                    # 1-cos lives in [0, 2].  Keep the existing upper mask as
                    # a conservative support/domain guard; this tests the same
                    # scaling invariant without widening the probe globally.
                    mask_z = (z_trig > z_abs_min) & (z_trig < z_abs_max) & mask_f
                else:
                    z_abs = z_trig.abs()
                    mask_z = (z_abs > z_abs_min) & (z_abs < z_abs_max) & mask_f
                n_z_ok = int(mask_z.sum())
                if n_z_ok < 100:
                    continue

                X_sub = X[mask_z]
                F_sub = F[mask_z]
                z_trig_sub = z_trig[mask_z]
                phase_sub = phase[mask_z]
                Nsub = X_sub.shape[0]

                # Per-axis centering: evaluate at a trig-zero point to remove
                # additive offset. Find a trig-zero within the data range.
                z_compound_vals = _eval_compound_z(target, X_sub)
                z_min_data = float(z_compound_vals.min())
                z_max_data = float(z_compound_vals.max())
                z_ref_val = _find_inrange_trig_zero(basis_fn, omega, z_min_data, z_max_data)
                if z_ref_val is None:
                    # No trig-zero in data range; skip this ω/trig combo
                    continue
                # Perturb X to set z_compound to z_ref
                X_ref = _perturb_for_compound(X_sub, torch.full((Nsub,), z_ref_val), target)
                # Range check on perturbed columns
                ref_in_range = torch.ones(Nsub, dtype=torch.bool)
                for idx in target.indices:
                    ref_in_range = ref_in_range & (X_ref[:, idx] >= x_lo[idx]) & (X_ref[:, idx] <= x_hi[idx])
                if ref_in_range.sum() < max(50, Nsub * 0.3):
                    # Too many points out of range; skip this ω/trig combo
                    continue

                with torch.no_grad():
                    y_ref = model(X_ref.to(dev))
                F_ref = (y_ref[:, 0] if y_ref.dim() == 2 else y_ref.view(-1)).cpu()

                F_sub_c = F_sub - F_ref
                mask_centered = F_sub_c.abs() > max(
                    float(torch.quantile(F_sub_c.abs(), 0.1)), 1e-8
                )

                log_ratios_per_lam: List[torch.Tensor] = []
                valid_log_lams: List[float] = []

                for li, lam in enumerate(lambda_values):
                    z_trig_scaled = lam * z_trig_sub

                    # Skip points where scaled z exits the inverse domain.
                    if basis_fn == "one_minus_cos":
                        in_domain = (z_trig_scaled >= 0.0) & (z_trig_scaled <= 2.0)
                    else:
                        in_domain = z_trig_scaled.abs() <= 1.0
                    if in_domain.sum() < max(50, Nsub * 0.3):
                        continue

                    # Invert: find target angle for z_compound
                    if basis_fn == "one_minus_cos":
                        cos_target = (1.0 - z_trig_scaled).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
                        target_angle = torch.acos(cos_target)
                    else:
                        target_angle = inv_func(z_trig_scaled.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
                    # Find z_compound' that gives target_angle
                    z_compound_prime = _closest_preimage(target_angle, phase_sub, basis_fn, omega)

                    # Perturb X to achieve z_compound'
                    X_scaled = _perturb_for_compound(X_sub, z_compound_prime, target)

                    # Range check on all perturbed columns
                    in_range = in_domain.clone()
                    for idx in target.indices:
                        in_range = in_range & (X_scaled[:, idx] >= x_lo[idx]) & (X_scaled[:, idx] <= x_hi[idx])
                    n_valid_range = int(in_range.sum())
                    if n_valid_range < max(50, Nsub * 0.3):
                        continue

                    with torch.no_grad():
                        y_s = model(X_scaled.to(dev))
                    F_scaled = (y_s[:, 0] if y_s.dim() == 2 else y_s.view(-1)).cpu()

                    F_scaled_c = F_scaled - F_ref
                    ratio = F_scaled_c / (F_sub_c + 1e-30)
                    valid = (ratio > 0) & torch.isfinite(ratio) & in_range & mask_centered
                    n_valid = int(valid.sum())
                    if n_valid < max(50, Nsub * 0.2):
                        continue

                    lr = torch.full((Nsub,), float("nan"), dtype=ratio.dtype)
                    lr[valid] = torch.log(ratio[valid])
                    log_ratios_per_lam.append(lr)
                    valid_log_lams.append(float(log_lambdas[li]))

                if len(valid_log_lams) < 3:
                    continue

                # Per-point least-squares fit: k_n = (LL . LR[:, n]) / (LL . LL)
                LR = torch.stack(log_ratios_per_lam, dim=0)
                LL = torch.tensor(valid_log_lams)

                k_per_point = torch.zeros(Nsub)
                valid_point = torch.zeros(Nsub, dtype=torch.bool)
                for n in range(Nsub):
                    col = LR[:, n]
                    finite_mask = torch.isfinite(col)
                    if finite_mask.sum() < 3:
                        continue
                    num = float((LL[finite_mask] * col[finite_mask]).sum())
                    den = float((LL[finite_mask] * LL[finite_mask]).sum())
                    if den < 1e-20:
                        continue
                    k_per_point[n] = num / den
                    valid_point[n] = True

                k_valid = k_per_point[valid_point]
                if k_valid.numel() < 50:
                    continue

                k_hat = float(k_valid.median())
                k_std = float(k_valid.std(unbiased=False))
                rel_std = k_std / (abs(k_hat) + 1e-12)

                if rel_std < best_rel_std:
                    best_rel_std = rel_std
                    best_result = TrigScaleSpec(
                        axis=target.pivot_idx,
                        omega=omega,
                        trig_fn=trig_fn,
                        k_hat=k_hat,
                        rel_std=rel_std,
                        n_points=int(k_valid.numel()),
                        compound_name=target.name if target.kind != "trivial" else "",
                        compound_expr=target.expr,
                        basis_fn=basis_fn,
                    )

        if best_result is not None and best_result.rel_std < rel_std_threshold:
            display_name = target.name
            basis_name = best_result.basis_fn or best_result.trig_fn
            print(
                f"  [Trig Scaling] {display_name} ACCEPTED: {basis_name}"
                f"({best_result.omega:.4g}·z)^{best_result.k_hat:.3f}  "
                f"rel_std={best_result.rel_std:.4f}"
            )
            results.append(best_result)
        elif best_result is not None:
            print(
                f"  [Trig Scaling] {target.name} rejected: best rel_std={best_result.rel_std:.4f} "
                f">= threshold={rel_std_threshold}"
            )
        else:
            print(f"  [Trig Scaling] {target.name}: no valid fit obtained")

    return results


# ---------------------------------------------------------------------------
# Compound null-test verification
# ---------------------------------------------------------------------------


@dataclass
class CompoundNullTestResult:
    """Result of the z-preserving null test for a compound variable."""
    z_var_idxs: Tuple[int, ...]
    z_exponents: Tuple[int, ...]
    verified: bool
    median_dev: float  # median(|f_transformed / f_original - 1|)
    n_valid: int


def verify_compound_null_test(
    model,
    datagen,
    z_var_idxs: Tuple[int, ...],
    z_exponents: Tuple[int, ...],
    Nxvars: int,
    device=None,
    lambda_values=(0.8, 0.9, 1.1, 1.25),
    max_points: int = 3000,
    dev_threshold: float = 0.03,
) -> CompoundNullTestResult:
    """
    Verify that f depends on variables in *z_var_idxs* only through
    ``z = prod x_i^{k_i}`` by applying a z-preserving transform and checking
    that f does not change.

    For compound z = x_0^{k_0} * x_1^{k_1}, we pick a pivot variable and
    for each lambda:
      - scale pivot by lambda
      - compensate other compound vars so z is preserved
      - check |f_transformed / f_original - 1| is small

    If verified, this proves f factors through z (stronger than Euler test).
    """
    X, F = _sample_values_only(model, datagen, max_batches=4, max_points=max_points, device=device)
    if X is None:
        return CompoundNullTestResult(z_var_idxs, z_exponents, False, float("inf"), 0)

    dev = device or next(model.parameters()).device

    # Filter tiny |F|
    Fabs = F.abs()
    f_pct5 = float(torch.quantile(Fabs, 0.05))
    mask_f = Fabs > max(f_pct5, 1e-12)

    # Data range per column (with 10% margin)
    x_min = X.min(dim=0).values
    x_max = X.max(dim=0).values
    x_range = x_max - x_min
    x_lo = x_min - 0.10 * x_range
    x_hi = x_max + 0.10 * x_range

    # Pick pivot: first compound variable with |k| > 0
    pivot_pos = None
    for pos, k in enumerate(z_exponents):
        if abs(k) > 0:
            pivot_pos = pos
            break
    if pivot_pos is None:
        return CompoundNullTestResult(z_var_idxs, z_exponents, False, float("inf"), 0)

    pivot_idx = z_var_idxs[pivot_pos]
    k_pivot = z_exponents[pivot_pos]

    # Also mask points near zero on compound axes
    mask_x = mask_f.clone()
    for pos, idx in enumerate(z_var_idxs):
        mask_x = mask_x & (X[:, idx].abs() > 1e-8)

    if mask_x.sum() < 100:
        return CompoundNullTestResult(z_var_idxs, z_exponents, False, float("inf"), int(mask_x.sum()))

    X_sub = X[mask_x]
    F_sub = F[mask_x]
    Nsub = X_sub.shape[0]

    all_devs: List[torch.Tensor] = []
    total_valid = 0

    for lam in lambda_values:
        X_t = X_sub.clone()
        # Scale pivot
        X_t[:, pivot_idx] = X_t[:, pivot_idx] * lam
        # Compensate other compound vars to preserve z
        for pos, idx in enumerate(z_var_idxs):
            if pos == pivot_pos:
                continue
            k_j = z_exponents[pos]
            if abs(k_j) < 1e-12:
                continue
            # x_j -> x_j * lam^(-k_pivot/k_j)
            comp_exp = -k_pivot / k_j
            X_t[:, idx] = X_t[:, idx] * (lam ** comp_exp)

        # Range check all compound columns
        in_range = torch.ones(Nsub, dtype=torch.bool)
        for idx in z_var_idxs:
            in_range = in_range & (X_t[:, idx] >= x_lo[idx]) & (X_t[:, idx] <= x_hi[idx])
        if in_range.sum() < max(30, Nsub * 0.2):
            continue

        with torch.no_grad():
            y_t = model(X_t.to(dev))
        F_t = (y_t[:, 0] if y_t.dim() == 2 else y_t.view(-1)).cpu()

        dev_vals = (F_t / (F_sub + 1e-30) - 1.0).abs()
        valid = in_range & torch.isfinite(dev_vals)
        if valid.sum() < 30:
            continue

        all_devs.append(dev_vals[valid])
        total_valid += int(valid.sum())

    if total_valid < 50:
        return CompoundNullTestResult(z_var_idxs, z_exponents, False, float("inf"), total_valid)

    combined_devs = torch.cat(all_devs)
    median_dev = float(combined_devs.median())
    min_valid_frac = 0.5
    verified = (median_dev < dev_threshold) and (total_valid > min_valid_frac * Nsub * len(lambda_values))

    return CompoundNullTestResult(
        z_var_idxs=z_var_idxs,
        z_exponents=z_exponents,
        verified=verified,
        median_dev=median_dev,
        n_valid=total_valid,
    )


def _build_poly_design_matrix(
    X: torch.Tensor, degree: int, indices: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Build a polynomial design matrix up to given degree (currently supports degree 1 or 2).
    X: [N, Nx]
    indices: which columns to use; default all.
    Returns Phi: [N, n_terms].
    """
    if indices is None:
        indices = list(range(X.shape[1]))
    Xs = X[:, indices]  # [N, d]
    N, d = Xs.shape

    cols = []

    # Constant term
    cols.append(torch.ones(N, dtype=Xs.dtype, device=Xs.device))

    if degree >= 1:
        # Linear terms x_i
        for i in range(d):
            cols.append(Xs[:, i])

    if degree >= 2:
        # Quadratic terms x_i * x_j, with i <= j (includes squares)
        for i in range(d):
            xi = Xs[:, i]
            for j in range(i, d):
                xj = Xs[:, j]
                cols.append(xi * xj)

    # Stack
    Phi = torch.stack(cols, dim=1)  # [N, n_terms]
    return Phi


def discover_poly_in_x(
    model,
    datagen,
    Nxvars: int,
    device=None,
    degree: int = 2,
    min_points: int = 200,
    rel_rms_threshold: float = 1e-3,
) -> Optional[PolyFitSpec]:
    """
    Try to fit f(x) with a low-degree polynomial in x, and report residuals.

    Returns a PolyFitSpec if a reasonably good fit is found, otherwise None.
    """
    # Use the same helper to sample X and f
    X, _, F = _sample_gradients_and_values(
        model, datagen, max_batches=8, max_points=20000, device=device
    )
    if X is None:
        return None

    N = X.shape[0]
    if N < min_points:
        return None

    # Design matrix Phi for all variables (indices 0..Nxvars-1)
    indices = list(range(Nxvars))
    Phi = _build_poly_design_matrix(X[:, :Nxvars], degree=degree, indices=indices)  # [N, n_terms]
    n_terms = Phi.shape[1]

    # We need N >= n_terms for a decent fit
    if N < n_terms + 5:
        # Not enough points for stable LS
        return None

    # Solve least squares: Phi @ c ≈ F
    # Use pseudo-inverse for robustness
    c = torch.linalg.pinv(Phi) @ F.unsqueeze(1)  # [n_terms, 1]
    F_pred = (Phi @ c).squeeze(1)

    resid = F - F_pred
    rms_abs = float(torch.sqrt(torch.mean(resid * resid)))
    std_target = float(F.std(unbiased=False))
    if std_target < 1e-12:
        rms_rel = float("inf") if rms_abs > 1e-12 else 0.0
    else:
        rms_rel = rms_abs / std_target

    # Report fit if it's "good enough"
    if rms_rel < rel_rms_threshold:
        name = f"poly_in_x_deg{degree}"
        spec = PolyFitSpec(
            name=name,
            degree=degree,
            indices=indices,
            n_terms=n_terms,
            n_points=N,
            rms_abs=rms_abs,
            rms_rel=rms_rel,
        )
        return spec

    return PolyFitSpec(
        name=f"poly_in_x_deg{degree}_poor",
        degree=degree,
        indices=indices,
        n_terms=n_terms,
        n_points=N,
        rms_abs=rms_abs,
        rms_rel=rms_rel,
    )


def discover_poly_in_f2(
    model,
    datagen,
    Nxvars: int,
    device=None,
    degree: int = 2,
    min_points: int = 200,
    rel_rms_threshold: float = 1e-3,
) -> Optional[PolyFitSpec]:
    """
    Try to fit f(x)^2 with a low-degree polynomial in x (sqrt-of-poly detector).

    Returns a PolyFitSpec if reasonably good; otherwise None.
    """
    X, _, F = _sample_gradients_and_values(
        model, datagen, max_batches=8, max_points=20000, device=device
    )
    if X is None:
        return None

    N = X.shape[0]
    if N < min_points:
        return None

    F2 = F * F

    indices = list(range(Nxvars))
    Phi = _build_poly_design_matrix(X[:, :Nxvars], degree=degree, indices=indices)
    n_terms = Phi.shape[1]

    if N < n_terms + 5:
        return None

    c = torch.linalg.pinv(Phi) @ F2.unsqueeze(1)
    F2_pred = (Phi @ c).squeeze(1)

    resid = F2 - F2_pred
    rms_abs = float(torch.sqrt(torch.mean(resid * resid)))
    std_target = float(F2.std(unbiased=False))
    if std_target < 1e-12:
        rms_rel = float("inf") if rms_abs > 1e-12 else 0.0
    else:
        rms_rel = rms_abs / std_target

    if rms_rel < rel_rms_threshold:
        name = f"poly_in_f2_deg{degree}"
        spec = PolyFitSpec(
            name=name,
            degree=degree,
            indices=indices,
            n_terms=n_terms,
            n_points=N,
            rms_abs=rms_abs,
            rms_rel=rms_rel,
        )
        return spec

    return PolyFitSpec(
        name=f"poly_in_f2_deg{degree}_poor",
        degree=degree,
        indices=indices,
        n_terms=n_terms,
        n_points=N,
        rms_abs=rms_abs,
        rms_rel=rms_rel,
    )


def _sample_values_from_model(
    model, datagen, max_batches: int = 8, max_points: int = 20000, device=None
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    xs = []
    vals = []
    n_points = 0

    data_iter = datagen() if callable(datagen) else datagen

    for bi, batch in enumerate(data_iter):
        if bi >= max_batches or n_points >= max_points:
            break

        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        x = x.to(device or next(model.parameters()).device)
        B = x.shape[0]
        x_flat = x.view(B, -1)

        y = model(x_flat)
        if y.dim() == 2:
            f = y[:, 0]
        else:
            f = y.view(-1)

        xs.append(x_flat.detach().cpu())
        vals.append(f.detach().cpu())
        n_points += B

    if not xs:
        return None, None

    X = torch.cat(xs, dim=0)[:max_points]
    F = torch.cat(vals, dim=0)[:max_points]
    return X, F


def _fit_rational_for_degrees(
    X: torch.Tensor,
    F: torch.Tensor,
    Nxvars: int,
    deg_num: int,
    deg_den: int,
    min_points: int = 200,
    rel_rms_threshold: float = 1e-3,
) -> Optional[RationalFitSpec]:
    N = X.shape[0]
    if N < min_points:
        return None

    # Restrict to the first Nxvars columns if needed
    Xv = X[:, :Nxvars]
    dtype = torch.float64
    Xv = Xv.to(dtype)
    Fv = F.to(dtype)

    # Build polynomial bases for numerator and denominator
    Phi_num = _build_poly_design_matrix(Xv, degree=deg_num)  # [N, M_num]
    Phi_den = _build_poly_design_matrix(Xv, degree=deg_den)  # [N, M_den]
    M_num = Phi_num.shape[1]
    M_den = Phi_den.shape[1]

    # Need enough points to constrain coeffs
    if N < (M_num + M_den + 5):
        return None

    # Build linear system A c ≈ 0  (c = [a; b])
    # A = [Phi_num, -F * Phi_den]
    F_col = Fv.unsqueeze(1)  # [N,1]
    A_left = Phi_num  # [N, M_num]
    A_right = -F_col * Phi_den  # [N, M_den]
    A = torch.cat([A_left, A_right], dim=1)  # [N, M_num+M_den]

    # Gram matrix and smallest eigenvector
    Gram = (A.T @ A) / float(N)  # [M,M], symmetric PSD
    Gram = Gram.to(dtype)
    # Need both eigenvalues and eigenvectors; Gram is symmetric PSD.
    evals, vecs = torch.linalg.eigh(Gram)  # evals ascending

    evals = evals.clamp_min(0.0)
    svals = torch.sqrt(evals)  # singular values

    sigma_min = float(svals[0])
    if svals.numel() > 2:
        sigma_med = float(svals[svals.numel() // 2])
    else:
        sigma_med = float(svals[-1])
    sigma_ratio = sigma_min / (sigma_med + 1e-12)

    # Smallest eigenvector gives us c = [a; b]
    c = vecs[:, 0]
    a = c[:M_num]
    b = c[M_num:]

    # Compactify numerator/denominator support with shared STLSQ de-Padeifier.
    try:
        a_sparse, b_sparse, meta = stlsq_sparsify_rational_coeffs(
            Phi_num=Phi_num,
            Phi_den=Phi_den,
            y=Fv,
            coeffs_num=a,
            coeffs_den=b,
            cfg=DEFAULT_RAT_STLSQ_CFG,
        )
        _log_sparsify_result("_fit_rational_for_degrees", a, b, a_sparse, b_sparse, meta)
        a, b = a_sparse, b_sparse
    except Exception as exc:
        _log.debug("[_fit_rational_for_degrees] rational sparsify failed: %s", exc)

    # Build P(x) and Q(x)
    P = Phi_num @ a  # [N]
    Q = Phi_den @ b  # [N]

    Q_abs = Q.abs()
    Q_max = float(Q_abs.max())
    eps_Q = max(1e-12, 1e-6 * Q_max)
    mask = Q_abs > eps_Q

    if int(mask.sum()) < min_points:
        # Degenerate denominator or not enough usable points
        return None

    f_pred = P[mask] / Q[mask]
    F_use = Fv[mask]

    resid = F_use - f_pred
    rms_abs = float(torch.sqrt(torch.mean(resid * resid)))
    std_target = float(F_use.std(unbiased=False))
    if std_target < 1e-12:
        rms_rel = float("inf") if rms_abs > 1e-12 else 0.0
    else:
        rms_rel = rms_abs / std_target

    name = f"rat_degN{deg_num}_degD{deg_den}"
    tol_num = max(1e-12, 1e-8 * float(a.abs().max().item()) if a.numel() > 0 else 1e-12)
    tol_den = max(1e-12, 1e-8 * float(b.abs().max().item()) if b.numel() > 0 else 1e-12)
    n_terms_num_eff = int((a.abs() > tol_num).sum().item())
    n_terms_den_eff = int((b.abs() > tol_den).sum().item())
    spec = RationalFitSpec(
        name=name,
        deg_num=deg_num,
        deg_den=deg_den,
        n_terms_num=n_terms_num_eff,
        n_terms_den=n_terms_den_eff,
        n_points=int(mask.sum()),
        rms_abs=rms_abs,
        rms_rel=rms_rel,
        sigma_min=sigma_min,
        sigma_ratio=sigma_ratio,
    )

    return spec


def discover_rational_poly(
    model,
    datagen,
    Nxvars: int,
    device=None,
    max_deg_num: int = 2,
    max_deg_den: int = 2,
    min_points: int = 200,
    rel_rms_threshold: float = 1e-3,
) -> Optional[RationalFitSpec]:
    """
    Try to fit f(x) ≈ P(x)/Q(x) with low-degree polynomials P,Q.

    Tries all degree pairs (deg_num, deg_den) with 1 ≤ deg_* ≤ max_deg_*,
    returns the best RationalFitSpec (smallest rms_rel), or None.
    """
    X, F = _sample_values_from_model(
        model=model,
        datagen=datagen,
        max_batches=8,
        max_points=20000,
        device=device,
    )
    if X is None:
        return None

    best: Optional[RationalFitSpec] = None

    for deg_num in range(1, max_deg_num + 1):
        for deg_den in range(1, max_deg_den + 1):
            spec = _fit_rational_for_degrees(
                X=X,
                F=F,
                Nxvars=Nxvars,
                deg_num=deg_num,
                deg_den=deg_den,
                min_points=min_points,
                rel_rms_threshold=rel_rms_threshold,
            )
            if spec is None:
                continue
            if best is None or spec.rms_rel < best.rms_rel:
                best = spec

    # Optionally, require a minimum quality
    if best is not None and best.rms_rel > rel_rms_threshold:
        # It's still useful to see; alternatively return None here to be strict.
        return best

    return best


# -------------------------------
# Line sampling & Poisson profile
# -------------------------------


def sample_line_curvature(
    provider,
    x0: torch.Tensor,
    u: torch.Tensor,
    tmin: float,
    tmax: float,
    n: int = 512,
    out_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(provider.parameters()).device
    dtype = next(provider.parameters()).dtype
    x0 = x0.to(device=device, dtype=dtype)
    u = _unit(u.to(device=device, dtype=dtype))

    ts = torch.linspace(tmin, tmax, n, device=device, dtype=dtype)
    xs = x0[None, :] + ts[:, None] * u[None, :]

    f = provider.forward(xs)
    if f.dim() == 2:
        f = f[:, out_idx]

    g = provider.grad(xs)
    if g.dim() == 3:
        g = g[:, out_idx, :]
    d1 = g @ u

    H = provider.grad_grad(xs)
    if H.dim() == 4:
        H = H[:, out_idx, :, :]
    d2 = torch.einsum("bi,bij,bj->b", u.expand(ts.size(0), -1), H, u.expand(ts.size(0), -1))

    return ts.detach().cpu(), f.detach().cpu(), d1.detach().cpu(), d2.detach().cpu()


def poisson_profile(
    ts: torch.Tensor, s: torch.Tensor, xgrid: torch.Tensor, eta: float
) -> torch.Tensor:
    x = xgrid.view(-1, 1)
    t = ts.view(1, -1)
    num = eta / math.pi
    K = num / ((x - t) ** 2 + (eta**2))
    return (K @ s.view(-1, 1)).view(-1)


# -----------------
# Pattern detection
# -----------------


def _linreg(x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float, float]:
    xm, ym = x.mean(), y.mean()
    xv, yv = x - xm, y - ym
    denom = (xv @ xv) + 1e-12
    b = float((xv @ yv) / denom)
    a = float(ym - b * xm)
    r2 = float(((xv @ yv) ** 2) / ((xv @ xv) * (yv @ yv) + 1e-12))
    return a, b, r2


def trig_from_profile(P: torch.Tensor, dx: float) -> Tuple[float, float]:
    Q = P - P.mean()
    F = torch.fft.rfft(Q)
    A = F.abs()
    if A.numel() <= 2:
        return 0.0, 0.0
    k = int(torch.argmax(A[1:])) + 1
    omega = 2 * math.pi * (k / (len(P) * dx))
    strength = float((A[k] / (A[1:].median() + 1e-9)).item())
    return float(omega), strength


def exp_from_profile(x: torch.Tensor, P: torch.Tensor) -> Tuple[float, float]:
    y = torch.log(P.clamp_min(P.max() * 1e-6))
    _, beta, r2 = _linreg(x, y)
    return float(beta), float(r2)


def log_from_profile(x: torch.Tensor, P: torch.Tensor) -> Tuple[Optional[float], float, float]:
    if x.numel() < 16:
        return None, 0.0, 0.0
    qgrid = torch.linspace(0.05, 0.35, 6, dtype=x.dtype, device=x.device)
    r2best, t0best, slopebest = 0.0, None, 0.0
    for q in qgrid:
        t0 = float(torch.quantile(x, q))
        m = x > (t0 + 1e-6)
        if m.sum() < 12:
            continue
        X = torch.log(x[m] - t0)
        Y = torch.log(P[m].clamp_min(P[m].max() * 1e-6))
        _, slope, r2 = _linreg(X, Y)
        if r2 > r2best:
            r2best, t0best, slopebest = float(r2), t0, float(slope)
    return t0best, slopebest, r2best


def power_from_profile(
    ts: torch.Tensor, P: torch.Tensor, qgrid: Optional[torch.Tensor] = None
) -> Tuple[Optional[float], float, float]:
    """
    Estimate a power-law pole d2 ~ |t - t0|^sigma from the Poisson-smoothed profile P.
    Returns (t0, sigma, R^2). If not detected, returns (None, 0.0, 0.0).
    """
    if qgrid is None:
        qgrid = torch.linspace(0.05, 0.35, 6, dtype=ts.dtype, device=ts.device)
    best_t0: Optional[float] = None
    best_sigma: float = 0.0
    best_r2: float = 0.0
    for q in qgrid:
        t0 = float(torch.quantile(ts, float(q)))
        Xraw = (ts - t0).abs()
        m = Xraw > (1e-6 * float(Xraw.max().detach().cpu()) + 1e-12)
        if m.sum() < 16:
            continue
        X = torch.log(Xraw[m].clamp_min(1e-12))
        Y = torch.log(P[m].abs().clamp_min(P[m].abs().max() * 1e-6))
        _, slope, r2 = _linreg(X, Y)  # Y ≈ a + slope * X
        if r2 > best_r2:
            best_r2 = float(r2)
            best_sigma = float(slope)
            best_t0 = t0
    return best_t0, best_sigma, best_r2


def polydeg_from_d2(
    ts: torch.Tensor, d2: torch.Tensor, deg_max: int = 3
) -> Tuple[int, torch.Tensor, float]:
    """
    Fit low-degree polynomials to d2(t) and return (best_degree, coeffs, R^2).
    Uses a standardized t for conditioning; coefficients correspond to that basis.
    """
    t = ts
    tstd = t.std()
    tstd = tstd if float(tstd) > 0.0 else torch.tensor(1.0, dtype=t.dtype, device=t.device)
    tn = (t - t.mean()) / tstd
    y = d2
    var_y = (y - y.mean()).pow(2).sum() + 1e-12
    best_deg, best_coef, best_r2 = 0, torch.zeros(1, dtype=t.dtype, device=t.device), -1.0
    for k in range(max(0, deg_max) + 1):
        Phi = torch.stack([tn.pow(j) for j in range(k + 1)], dim=1)  # [B, k+1]
        # Use PINV for version-robustness
        coef = torch.linalg.pinv(Phi) @ y.unsqueeze(1)  # [(k+1), 1]
        pred = (Phi @ coef).squeeze(1)
        sse = (y - pred).pow(2).sum()
        r2 = float(1.0 - (sse / var_y))
        if r2 > best_r2:
            best_r2 = r2
            best_deg = k
            best_coef = coef.squeeze(1)
    return best_deg, best_coef.detach(), float(best_r2)


def classify_dir(
    stats: Dict[str, Any],
    trig_thr: float = 5.0,
    exp_r2: float = 0.9,
    log_r2: float = 0.85,
    log_slope_target: float = -2.0,
    tol: float = 0.35,
) -> Dict[str, Any]:
    if stats.get("trig_strength", 0.0) > trig_thr and stats.get("trig_omega", 0.0) > 0.0:
        return {"type": "trig", "omega": float(stats["trig_omega"])}
    if stats.get("exp_r2", 0.0) > exp_r2:
        return {"type": "exp", "beta": float(stats["exp_beta"])}
    if (
        stats.get("log_r2", 0.0) > log_r2
        and stats.get("log_t0", None) is not None
        and abs(float(stats["log_slope"]) - log_slope_target) < tol
    ):
        return {"type": "log", "t0": float(stats["log_t0"])}
    # Generic power-law pole detector (e.g., sqrt / reciprocal families)
    p_r2 = stats.get("power_r2", 0.0)
    if p_r2 > 0.90 and stats.get("power_t0", None) is not None:
        sigma = float(stats.get("power_sigma", 0.0))
        out = {"type": "power", "sigma": sigma, "t0": float(stats["power_t0"])}
        # Optional coarse family tagging for convenience
        if abs(sigma + 1.5) < 0.3:
            out["family"] = "sqrt"
        elif abs(sigma + 3.0) < 0.4:
            out["family"] = "recip"
        elif abs(sigma + 4.0) < 0.4:
            out["family"] = "recip2"
        return out
    return {"type": "none"}


def _sample_inputs_from_datagen(
    datagen, max_batches: int = 8, max_points: int = 20000, device=None
) -> Optional[torch.Tensor]:
    xs = []
    n_points = 0
    data_iter = datagen() if callable(datagen) else datagen

    for bi, batch in enumerate(data_iter):
        if bi >= max_batches or n_points >= max_points:
            break
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        x = x.to(device or x.device)
        B = x.shape[0]
        x_flat = x.view(B, -1)
        xs.append(x_flat.detach().cpu())
        n_points += B

    if not xs:
        return None

    X = torch.cat(xs, dim=0)[:max_points]
    return X


def _trig_from_profile_with_phase_inline(P: torch.Tensor, dx: float) -> Tuple[float, float, float]:
    """FFT-based trig detection returning (omega, strength, phase).

    Phase is the argument of the complex FFT coefficient at the dominant frequency.
    """
    Q = P - P.mean()
    F = torch.fft.rfft(Q)
    A = F.abs()
    if A.numel() <= 2:
        return 0.0, 0.0, 0.0
    k = int(torch.argmax(A[1:])) + 1
    omega = 2 * math.pi * (k / (len(P) * dx))
    strength = float((A[k] / (A[1:].median() + 1e-9)).item())
    phase = float(torch.angle(F[k]).item())
    return float(omega), float(strength), float(phase)


def discover_trig_axes(
    model,
    datagen,
    Nxvars: int,
    device=None,
    max_batches: int = 8,
    max_points: int = 20000,
    n_line: int = 512,
    q_span: Tuple[float, float] = (0.05, 0.95),
    eta_scale: float = 0.8,
    strength_threshold: float = 5.0,
    max_omega: float = 50.0,
    linear_r2_threshold: float = 0.999,
    linear_resid_rel_threshold: float = 1e-2,
    min_curv_rel: float = 1e-3,
) -> List[TrigAxisSpec]:
    """
    Scan coordinate axes x_j for approximately trig-like behaviour in f'' along that axis.

    Heuristic: along axis j, we take a line x(t) = x_mean + t e_j, compute d²f/dt²,
    Poisson-smooth it, FFT the profile, and look for a strong spectral peak.

    Returns a list of TrigAxisSpec for axes that pass the strength_threshold.
    """
    # 1) Sample inputs to define spans and a central point
    X = _sample_inputs_from_datagen(
        datagen,
        max_batches=max_batches,
        max_points=max_points,
        device=device,
    )
    if X is None:
        return []

    N, Nx = X.shape
    if Nx < Nxvars:
        raise ValueError(f"discover_trig_axes: X has {Nx} columns, expected at least {Nxvars}")

    Xv = X[:, :Nxvars]
    x_mean = Xv.mean(dim=0)  # [Nxvars]
    specs: List[TrigAxisSpec] = []

    # 2) Loop over coordinate axes
    for j in range(Nxvars):
        # Axis direction e_j in R^Nxvars
        u = torch.zeros(Nxvars, dtype=torch.float64)
        u[j] = 1.0

        # Span along this axis: use quantiles of x_j over data
        t_vals = Xv[:, j]
        tmin = float(torch.quantile(t_vals, q_span[0]))
        tmax = float(torch.quantile(t_vals, q_span[1]))
        if not math.isfinite(tmin) or not math.isfinite(tmax) or tmax <= tmin:
            continue

        # 3) Sample line curvature for this axis
        ts, f_line, d1_line, d2_line = sample_line_curvature(
            provider=model,
            x0=x_mean,
            u=u,
            tmin=tmin,
            tmax=tmax,
            n=n_line,
            out_idx=0,
        )
        if ts.numel() < 8:
            continue

        # Guardrail: skip axes that look essentially linear along the sampled line.
        # This avoids false "trig" detections caused by piecewise segment kinks when the
        # true dependence is linear (common in Stage-A nets with limited capacity).
        f_span = None
        try:
            fv = f_line.to(torch.float64)
            tsv = ts.to(torch.float64)
            f_span = float((fv.max() - fv.min()).item())
            if math.isfinite(f_span) and f_span > 0.0:
                a_lin, b_lin, r2_lin = _linreg(tsv, fv)
                resid = fv - (a_lin + b_lin * tsv)
                resid_rms = float((resid - resid.mean()).pow(2).mean().sqrt().item())
                if float(r2_lin) >= linear_r2_threshold and (resid_rms / (f_span + 1e-12)) < linear_resid_rel_threshold:
                    continue
        except Exception:
            pass

        # 4) Poisson smoothing of d2(t)
        dt = float(ts[1] - ts[0])
        eta = eta_scale * dt
        P = poisson_profile(ts, d2_line, ts, eta)

        # Require non-trivial oscillatory curvature energy.
        # This prevents large "strength" ratios when curvature is ~0 and the spectral
        # median is tiny (a common numerical false positive).
        try:
            Q = P - P.mean()
            q_rms = float((Q.to(torch.float64).pow(2).mean().sqrt().item()))
            if f_span is None:
                fv = f_line.to(torch.float64)
                f_span = float((fv.max() - fv.min()).item())
            if (not math.isfinite(q_rms)) or (f_span is None) or (not math.isfinite(f_span)) or (f_span <= 0.0):
                continue
            if (q_rms / (f_span + 1e-12)) < min_curv_rel:
                continue
        except Exception:
            pass

        # 5) Trig detector: FFT-based frequency + strength + phase
        omega, strength, phase = _trig_from_profile_with_phase_inline(P, dt)

        # Basic sanity checks
        if not math.isfinite(omega) or not math.isfinite(strength):
            continue
        if omega <= 0.0 or omega > max_omega:
            continue
        if strength < strength_threshold:
            continue

        specs.append(
            TrigAxisSpec(
                axis=j,
                omega=omega,
                strength=strength,
                n_points=int(ts.numel()),
                tmin=tmin,
                tmax=tmax,
                phase=phase,
            )
        )

    # Sort by strength (descending) so caller can pick best easily
    specs.sort(key=lambda s: s.strength, reverse=True)
    return specs


def discover_trig_axis(
    model, datagen, Nxvars: int, device=None, **kwargs
) -> Optional[TrigAxisSpec]:
    """
    Convenience wrapper: return the strongest trig-like axis or None.
    kwargs are forwarded to discover_trig_axes.
    """
    specs = discover_trig_axes(
        model=model,
        datagen=datagen,
        Nxvars=Nxvars,
        device=device,
        **kwargs,
    )
    if not specs:
        return None
    return specs[0]


# ------------------------------------------------------------
# K-based direction discovery from a segmented NestyNet / G_Model
# ------------------------------------------------------------


def _stack_params_from_NestyNet(g_net) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract (a, b, K) tensors from a segmented NestyNet-style model with
    .get_parameters(), .Nx_size, .seg_width, .Nout_size.

    Returns
    -------
    a : [O, W, S]
    b : [O, W, S]
    K : [O, Nx, W, S]
    """
    a_list, b_list, _, K_list = g_net.get_parameters()
    a = torch.stack(a_list, dim=-1)  # [O*W, S] logically
    b = torch.stack(b_list, dim=-1)
    K = torch.stack(K_list, dim=-1)

    O = g_net.Nout_size
    W = g_net.seg_width
    S = a.size(-1)
    Nx = g_net.Nx_size

    a = a.reshape(O, W, S)
    b = b.reshape(O, W, S)
    K = K.reshape(O, Nx, W, S)
    return a, b, K


def discover_directions_from_NestyNet(
    g_net,
    topk: int = 32,
    cos_thr: float = 0.95,
    out_idx: int = 0,
) -> List[torch.Tensor]:
    """
    Discover a small set of "important" input directions from the first layer K
    of a segmented NestyNet model.

    This is the direct port of the K-clustering logic you used in fingerprint_peel:
      - tries g_net.report_K_colinearity if available
      - otherwise clusters columns of K to get representative directions
    """
    # 1) If the model exposes a precomputed colinearity report, use it.
    if hasattr(g_net, "report_K_colinearity"):
        try:
            rep = g_net.report_K_colinearity(topk=topk, cos_thr=cos_thr)
            dirs: List[torch.Tensor] = []
            if isinstance(rep, (list, tuple)) and len(rep) > 0:
                for r in rep[:topk]:
                    v = torch.as_tensor(r, dtype=torch.float64).flatten()
                    dirs.append(_unit(v).cpu())
                if dirs:
                    return dirs
        except Exception:
            # fall back to raw K clustering
            pass

    # 2) If there is no K at all, fall back to basis + random dirs.
    if not hasattr(g_net, "get_parameters"):
        Nx = getattr(g_net, "Nx_size", 2)
        dirs: List[torch.Tensor] = []
        # axes
        for i in range(Nx):
            v = torch.zeros(Nx, dtype=torch.float64)
            v[i] = 1.0
            dirs.append(v)
        # a few random dirs if requested
        if topk is not None and topk > 0:
            n_extra = max(0, min(topk, 4 * Nx) - len(dirs))
            for _ in range(n_extra):
                v = torch.randn(Nx, dtype=torch.float64)
                dirs.append(_canonicalize_sign(_unit(v)))
        return dirs[:topk]

    # 3) Proper K-based clustering
    try:
        a, b, K = _stack_params_from_NestyNet(g_net)
        O, Nx, W, S = K.shape

        # Collapse (W, S) pieces into a collection of K-vectors in R^Nx
        Ko = K[out_idx].reshape(Nx, W * S).t().contiguous()  # [W*S, Nx]
        norms = Ko.norm(dim=1, keepdim=True).clamp_min(1e-12)
        U = Ko / norms  # normalized K vectors

        used = torch.zeros(U.size(0), dtype=torch.bool)
        dirs: List[torch.Tensor] = []

        max_clusters = min(U.size(0), max(4 * topk, 16))
        for _ in range(max_clusters):
            if used.all():
                break
            avail = (~used).nonzero(as_tuple=True)[0]
            if len(avail) == 0:
                break
            i = avail[0]
            center = U[i]
            sims = U @ center
            cluster = (sims.abs() >= cos_thr).logical_and(~used)
            if cluster.sum() == 0:
                break
            mean_dir = U[cluster].mean(dim=0)
            dirs.append(_canonicalize_sign(_unit(mean_dir).cpu()))
            used.logical_or_(cluster)
            if len(dirs) >= topk:
                break

        if not dirs:
            # extreme degeneracy: just return the mean K direction
            dirs.append(_canonicalize_sign(_unit(U.mean(dim=0)).cpu()))

        return dirs
    except Exception:
        # Fall back to axes
        Nx = getattr(g_net, "Nx_size", 2)
        dirs: List[torch.Tensor] = []
        for i in range(Nx):
            v = torch.zeros(Nx, dtype=torch.float64)
            v[i] = 1.0
            dirs.append(v)
        return dirs[:topk]


def _extract_NestyNet_from_model(model) -> Optional[object]:
    """
    Recursively search for a NestyNet-like core (anything that has
    .get_parameters()) inside composite / adaptor wrappers.
    """
    seen = set()

    def _recurse(obj):
        oid = id(obj)
        if oid in seen:
            return None
        seen.add(oid)

        # 1) Look inside obvious container attributes first
        for attr in ("base_model", "leaf", "leaves"):
            if hasattr(obj, attr):
                child = getattr(obj, attr)
                # Check if it's iterable (list, tuple, ModuleList, etc.)
                # but not a string or tensor
                is_iterable = hasattr(child, "__iter__") and not isinstance(
                    child, (str, torch.Tensor)
                )
                if is_iterable:
                    try:
                        for ch in child:
                            res = _recurse(ch)
                            if res is not None:
                                return res
                    except TypeError:
                        # Not actually iterable, treat as single object
                        res = _recurse(child)
                        if res is not None:
                            return res
                else:
                    res = _recurse(child)
                    if res is not None:
                        return res

        # 2) If this object itself has get_parameters, accept it
        if hasattr(obj, "get_parameters"):
            return obj

        return None

    return _recurse(model)


def discover_model_directions(
    model,
    topk: int = 32,
    cos_thr: float = 0.95,
    out_idx: int = 0,
) -> List[torch.Tensor]:
    """
    Convenience wrapper: given *any* NestyNet adaptor/composite, try to
    extract an inner core with .get_parameters() and run K-based direction discovery.
    If nothing is found, fall back to coordinate axes (if Nx_size is known).
    """
    g_net = _extract_NestyNet_from_model(model)
    if g_net is not None:
        dirs = discover_directions_from_NestyNet(
            g_net=g_net,
            topk=topk,
            cos_thr=cos_thr,
            out_idx=out_idx,
        )
        if dirs:
            return dirs

    # Fallback: axes if we at least know Nx_size
    Nx = getattr(model, "Nx_size", None)
    if Nx is None:
        return []
    dirs: List[torch.Tensor] = []
    for i in range(int(Nx)):
        v = torch.zeros(Nx, dtype=torch.float64)
        v[i] = 1.0
        dirs.append(v)
    return dirs[:topk]


def discover_parity_axes(
    model,
    datagen,
    Nxvars: int,
    device=None,
    max_batches: int = 8,
    max_points: int = 20000,
    rel_tol_even: float = 0.1,
    rel_tol_odd: float = 0.1,
) -> List[ParitySpec]:
    X = _sample_inputs_from_datagen(
        datagen, max_batches=max_batches, max_points=max_points, device=device
    )
    if X is None:
        return []
    try:
        dev = device or next(model.parameters()).device
    except StopIteration:
        dev = device or torch.device("cpu")
    X = X[:, :Nxvars].to(dev)
    with torch.no_grad():
        y = model(X)
        if y.dim() == 2:
            F = y[:, 0]
        else:
            F = y.view(-1)
    F = F.detach()
    N = F.shape[0]
    if N < 4:
        return []
    stdF = float(F.std(unbiased=False))
    if stdF < 1e-12:
        stdF = 1.0
    specs: List[ParitySpec] = []
    for j in range(Nxvars):
        xj = X[:, j]
        cj = float(torch.median(xj))
        Xr = X.clone()
        Xr[:, j] = 2.0 * cj - xj
        with torch.no_grad():
            Fr = model(Xr)
            if Fr.dim() == 2:
                Fr = Fr[:, 0]
            else:
                Fr = Fr.view(-1)
        Fr = Fr.detach()
        if Fr.shape != F.shape:
            continue
        resid_even = F - Fr
        resid_odd = F + Fr
        rms_even = float(torch.sqrt(torch.mean(resid_even * resid_even)))
        rms_odd = float(torch.sqrt(torch.mean(resid_odd * resid_odd)))
        rms_rel_even = rms_even / stdF
        rms_rel_odd = rms_odd / stdF
        best_kind = "even" if rms_rel_even <= rms_rel_odd else "odd"
        best_rel = rms_rel_even if best_kind == "even" else rms_rel_odd
        thr = rel_tol_even if best_kind == "even" else rel_tol_odd
        kind = best_kind if best_rel < thr else "none"
        specs.append(
            ParitySpec(
                name=f"parity_x{j}",
                axis=j,
                origin=cj,
                kind=kind,
                rms_even=rms_even,
                rms_odd=rms_odd,
                rms_rel_even=rms_rel_even,
                rms_rel_odd=rms_rel_odd,
                n_points=N,
            )
        )
    return specs


def discover_radial_groups(
    model,
    datagen,
    Nxvars: int,
    device=None,
    max_batches: int = 8,
    max_points: int = 20000,
    max_group_size: int = 3,
    cos_threshold: float = 0.95,
    min_points: int = 200,
) -> List[RadialSpec]:
    X, G, F = _sample_gradients_and_values(
        model, datagen, max_batches=max_batches, max_points=max_points, device=device
    )
    if X is None:
        return []
    Xv = X[:, :Nxvars]
    Gv = G[:, :Nxvars]
    _N = Xv.shape[0]
    specs: List[RadialSpec] = []
    idxs = list(range(Nxvars))
    for r in range(2, max_group_size + 1):
        for S in itertools.combinations(idxs, r):
            S_list = list(S)
            xS = Xv[:, S_list]
            gS = Gv[:, S_list]
            rx = torch.linalg.norm(xS, dim=1)
            rg = torch.linalg.norm(gS, dim=1)
            m = (rx > 1e-12) & (rg > 1e-12)
            if int(m.sum()) < min_points:
                continue
            dot = (xS[m] * gS[m]).sum(dim=1)
            cos = (dot / (rx[m] * rg[m])).abs()
            mean_cos = float(cos.mean())
            med_cos = float(cos.median())
            if mean_cos >= cos_threshold:
                specs.append(
                    RadialSpec(
                        name=f"radial_S{S_list}",
                        indices=S_list,
                        mean_abs_cos=mean_cos,
                        median_abs_cos=med_cos,
                        n_points=int(m.sum()),
                    )
                )
    return specs


def discover_preferred_origins(
    model,
    datagen,
    Nxvars: int,
    device=None,
    max_batches: int = 8,
    max_points: int = 20000,
    min_points: int = 200,
    min_r2: float = 0.8,
    min_slope: float = 1e-6,
) -> List[TranslationSpec]:
    X, G, F = _sample_gradients_and_values(
        model, datagen, max_batches=max_batches, max_points=max_points, device=device
    )
    if X is None:
        return []
    Xv = X[:, :Nxvars]
    Gv = G[:, :Nxvars]
    _N = Xv.shape[0]
    specs: List[TranslationSpec] = []
    for j in range(Nxvars):
        xj = Xv[:, j]
        gj = Gv[:, j]
        m = torch.isfinite(xj) & torch.isfinite(gj)
        if int(m.sum()) < min_points:
            continue
        xjv = xj[m]
        gjv = gj[m]
        # _linreg returns (intercept, slope, r2): y ≈ intercept + slope * x
        intercept, slope, r2 = _linreg(xjv, gjv)
        if not math.isfinite(slope) or abs(slope) < min_slope:
            continue
        # Preferred origin (zero of the linearized gradient): slope * t0 + intercept = 0
        t0 = -intercept / slope
        in_range = float(xjv.min()) <= t0 <= float(xjv.max())
        if r2 >= min_r2:
            specs.append(
                TranslationSpec(
                    axis=j,
                    origin=float(t0),
                    slope=float(slope),
                    intercept=float(intercept),
                    r2=float(r2),
                    in_range=bool(in_range),
                    n_points=int(m.sum()),
                )
            )
    return specs


def discover_saturating_axes(
    model,
    datagen,
    Nxvars: int,
    device=None,
    max_batches: int = 8,
    max_points: int = 20000,
    n_line: int = 512,
    q_span: Tuple[float, float] = (0.05, 0.95),
    q_mid: Tuple[float, float] = (0.25, 0.75),
    ratio_threshold: float = 3.0,
    mono_threshold: float = 0.8,
) -> List[SaturatingSpec]:
    X = _sample_inputs_from_datagen(
        datagen, max_batches=max_batches, max_points=max_points, device=device
    )
    if X is None:
        return []
    Xv = X[:, :Nxvars]
    x_mean = Xv.mean(dim=0)
    specs: List[SaturatingSpec] = []
    for j in range(Nxvars):
        t_vals = Xv[:, j]
        t_lo = float(torch.quantile(t_vals, q_span[0]))
        t_hi = float(torch.quantile(t_vals, q_span[1]))
        if not math.isfinite(t_lo) or not math.isfinite(t_hi) or t_hi <= t_lo:
            continue
        u = torch.zeros(Nxvars, dtype=torch.float64)
        u[j] = 1.0
        ts, f_line, d1_line, d2_line = sample_line_curvature(
            model, x0=x_mean, u=u, tmin=t_lo, tmax=t_hi, n=n_line, out_idx=0
        )
        if ts.numel() < 16:
            continue
        t_mid_lo = float(torch.quantile(ts, q_mid[0]))
        t_mid_hi = float(torch.quantile(ts, q_mid[1]))
        mid_mask = (ts >= t_mid_lo) & (ts <= t_mid_hi)
        edge_mask = ~mid_mask
        if int(mid_mask.sum()) < 4 or int(edge_mask.sum()) < 4:
            continue
        d1_mid = d1_line[mid_mask]
        d1_edge = d1_line[edge_mask]
        mid_grad = float(d1_mid.abs().mean())
        edge_grad = float(d1_edge.abs().mean())
        if edge_grad < 1e-12:
            continue
        ratio = mid_grad / edge_grad
        sign_mean = float(d1_mid.mean())
        if abs(sign_mean) < 1e-12:
            continue
        sign_dir = 1.0 if sign_mean > 0.0 else -1.0
        same_sign = ((d1_mid * sign_dir) > 0.0).float().mean()
        if ratio >= ratio_threshold and float(same_sign) >= mono_threshold:
            specs.append(
                SaturatingSpec(
                    axis=j,
                    tmin=t_lo,
                    tmax=t_hi,
                    mid_edge_grad_ratio=ratio,
                    monotonic_fraction=float(same_sign),
                    direction=1 if sign_dir > 0 else -1,
                    n_points=int(ts.numel()),
                )
            )
    return specs


def discover_constant_directions(
    model,
    datagen,
    Nxvars: int,
    device=None,
    max_batches: int = 8,
    max_points: int = 20000,
    min_points: int = 200,
    rel_threshold: float = 0.1,
) -> List[ConstantDirectionSpec]:
    X, G, F = _sample_gradients_and_values(
        model, datagen, max_batches=max_batches, max_points=max_points, device=device
    )
    if X is None:
        return []
    Gv = G[:, :Nxvars]
    N = Gv.shape[0]
    if N < min_points:
        return []
    rms = torch.sqrt(torch.mean(Gv * Gv, dim=0))
    typical = float(torch.median(rms))
    specs: List[ConstantDirectionSpec] = []
    if typical < 1e-12:
        for j in range(Nxvars):
            specs.append(
                ConstantDirectionSpec(axis=j, rms_grad=float(rms[j]), rel_rms_grad=0.0, n_points=N)
            )
        return specs
    for j in range(Nxvars):
        rel = float(rms[j] / typical)
        if rel <= rel_threshold:
            specs.append(
                ConstantDirectionSpec(axis=j, rms_grad=float(rms[j]), rel_rms_grad=rel, n_points=N)
            )
    return specs


# =============================
# Phase 3 probes (outer peeling)
# =============================

# These probes are designed to generate *structured hints* for template
# families. They are intentionally lightweight: they use analytic derivatives
# where available, but fall back to value-only fits where that's sufficient.


@dataclass
class TransformProbeSpec:
    name: str
    domain_ok_frac: float
    n_points: int
    poly2_rms_rel: float
    rat_rms_rel: float
    hess_const_rel: float
    scaling_rel_std: float
    cross_hess_rel: float
    score: float
    # Optional parameters for parameterised transforms (e.g. affine+asin).
    # Convention: params values are plain Python floats.
    params: Optional[Dict[str, float]] = None


@dataclass
class TransformHint:
    ok: bool
    best_name: str
    score_improvement: float
    baseline: TransformProbeSpec
    best: TransformProbeSpec
    candidates: List[TransformProbeSpec]
    reason: str = ""


@dataclass
class QuadraticHint:
    type: str  # "log" | "square"
    ok: bool
    domain_ok_frac: float
    n_points: int
    hess_const_rel: float
    score: float
    rank: int
    eigvals: List[float]
    mean_hess: torch.Tensor


@dataclass
class PeriodicityStructureHint:
    axis: int
    partner: Optional[int]
    kind: str  # "product" | "difference" | "none"
    score: float
    omega0: float
    omega_std: float
    omega_r2: float
    omega_slope: float
    omega_intercept: float
    phase_r2: float
    phase_slope: float
    phase_intercept: float
    n_slices: int
    n_line: int


def _extract_scalar_output(F: torch.Tensor, out_idx: int = 0) -> torch.Tensor:
    if F is None:
        raise ValueError("_extract_scalar_output: F is None")
    if F.dim() == 0:
        return F.view(1)
    if F.dim() == 1:
        return F.view(-1)
    if F.dim() == 2:
        return F[:, int(out_idx)].contiguous().view(-1)
    # e.g. [B, O, ...] should not happen in this codebase, but be tolerant.
    return F.reshape(F.shape[0], -1)[:, int(out_idx)].contiguous().view(-1)


def _extract_scalar_grad(G: torch.Tensor, out_idx: int = 0) -> torch.Tensor:
    if G is None:
        raise ValueError("_extract_scalar_grad: G is None")
    if G.dim() == 2:
        return G
    if G.dim() == 3:
        return G[:, int(out_idx), :]
    raise ValueError(f"Unexpected grad shape {tuple(G.shape)}")


def _extract_scalar_hess(H: torch.Tensor, out_idx: int = 0) -> torch.Tensor:
    if H is None:
        raise ValueError("_extract_scalar_hess: H is None")
    if H.dim() == 3:
        return H
    if H.dim() == 4:
        return H[:, int(out_idx), :, :]
    raise ValueError(f"Unexpected Hessian shape {tuple(H.shape)}")


def _sample_values_grads_hessians(
    model,
    datagen,
    *,
    max_batches: int = 8,
    max_points: int = 4096,
    device=None,
    out_idx: int = 0,
):
    xs: List[torch.Tensor] = []
    fs: List[torch.Tensor] = []
    gs: List[torch.Tensor] = []
    hs: List[torch.Tensor] = []
    n_points = 0
    data_iter = datagen() if callable(datagen) else datagen

    # Device inference: prefer caller, then model parameters if present.
    dev = device
    if dev is None:
        try:
            dev = next(model.parameters()).device
        except Exception:
            dev = None

    for bi, batch in enumerate(data_iter):
        if bi >= max_batches or n_points >= max_points:
            break
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        B = x.shape[0]
        x_flat = x.view(B, -1)
        if dev is not None:
            x_flat = x_flat.to(dev)
        with torch.no_grad():
            F = model.forward(x_flat)
            G = model.grad(x_flat)
            H = model.grad_grad(x_flat)
        f = _extract_scalar_output(F, out_idx=out_idx)
        g = _extract_scalar_grad(G, out_idx=out_idx)
        h = _extract_scalar_hess(H, out_idx=out_idx)

        xs.append(x_flat.detach().cpu())
        fs.append(f.detach().cpu())
        gs.append(g.detach().cpu())
        hs.append(h.detach().cpu())
        n_points += B

    if not xs:
        return None, None, None, None

    X = torch.cat(xs, dim=0)[:max_points]
    F = torch.cat(fs, dim=0)[:max_points]
    G = torch.cat(gs, dim=0)[:max_points]
    H = torch.cat(hs, dim=0)[:max_points]
    return X, F, G, H


def _poly_fit_rms_rel(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    degree: int = 2,
) -> float:
    X = X.to(dtype=torch.float64)
    y = y.view(-1).to(dtype=torch.float64)
    if X.numel() == 0 or y.numel() == 0:
        return float("inf")
    if X.shape[0] != y.shape[0]:
        raise ValueError("_poly_fit_rms_rel: X and y must have same length")
    Phi = _build_poly_design_matrix(X, degree=degree)
    if Phi.numel() == 0:
        return float("inf")

    # Solve least squares (robust to rank deficiency)
    try:
        sol = torch.linalg.lstsq(Phi, y.unsqueeze(1)).solution.squeeze(1)
    except Exception:
        # Fallback: pseudo-inverse
        sol = torch.linalg.pinv(Phi) @ y
    y_hat = Phi @ sol
    resid = y_hat - y
    rms_abs = float(torch.sqrt(torch.mean(resid * resid)).item())
    std_y = float(y.std(unbiased=False).item())
    if std_y < 1e-12:
        return 0.0 if rms_abs < 1e-12 else float("inf")
    return float(rms_abs / (std_y + 1e-12))


def _hess_const_rel(H: torch.Tensor) -> Tuple[float, torch.Tensor]:
    """Return (relative deviation, mean Hessian)."""
    H = H.to(dtype=torch.float64)
    if H.numel() == 0:
        return float("inf"), torch.zeros(1, 1, dtype=torch.float64)
    Hbar = H.mean(dim=0)
    dev = H - Hbar
    dev2 = dev.pow(2).sum(dim=(1, 2))
    rms_dev = torch.sqrt(dev2.mean() + 1e-24)
    norm_bar = torch.sqrt(Hbar.pow(2).sum() + 1e-24)
    rel = float((rms_dev / (norm_bar + 1e-12)).item())
    return rel, Hbar


def _cross_hess_rel(H: torch.Tensor) -> float:
    """Median(|offdiag|) / (median(|all|)+eps)."""
    H = H.to(dtype=torch.float64)
    if H.numel() == 0:
        return float("inf")
    d = int(H.shape[1])
    if d <= 1:
        return 0.0
    absH = H.abs().reshape(H.shape[0], d * d)
    med_all = float(absH.median().item())
    eye = torch.eye(d, dtype=torch.bool, device=absH.device).reshape(-1)
    off = absH[:, ~eye]
    if off.numel() == 0:
        return 0.0
    med_off = float(off.median().item())
    if med_all < 1e-24:
        return 0.0 if med_off < 1e-24 else float("inf")
    return float(med_off / (med_all + 1e-12))


def _scaling_rel_std(
    X: torch.Tensor,
    y: torch.Tensor,
    g: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> float:
    """Homogeneity proxy over all axes: r = (x·∇y)/y; report std/|mean|."""
    X = X.to(dtype=torch.float64)
    y = y.view(-1).to(dtype=torch.float64)
    g = g.to(dtype=torch.float64)
    if X.numel() == 0 or y.numel() == 0 or g.numel() == 0:
        return float("inf")
    if X.shape[0] != y.shape[0] or X.shape[0] != g.shape[0]:
        return float("inf")
    scale = float(torch.median(y.abs()).item())
    y_eps = max(eps, 1e-9 * max(scale, 1.0))
    mask = torch.isfinite(y) & (y.abs() > y_eps)
    if int(mask.sum().item()) < 50:
        return float("inf")
    Xm = X[mask]
    gm = g[mask]
    ym = y[mask]
    r = (Xm * gm).sum(dim=1) / ym
    if r.numel() < 10:
        return float("inf")
    m = float(r.mean().item())
    s = float(r.std(unbiased=False).item())
    return float(s / (abs(m) + 1e-12))


def _probe_score(
    *,
    domain_ok_frac: float,
    poly2_rms_rel: float,
    rat_rms_rel: float,
    hess_const_rel: float,
    scaling_rel_std: float,
    cross_hess_rel: float,
) -> float:
    def _bonus(v: float) -> float:
        if not math.isfinite(v):
            return 0.0
        v = max(v, 1e-12)
        b = -math.log10(v)
        return max(0.0, b)

    # Keep this intentionally simple and interpretable.
    score = 0.0
    score += 1.00 * _bonus(poly2_rms_rel)
    score += 0.60 * _bonus(rat_rms_rel)
    score += 0.80 * _bonus(hess_const_rel)
    score += 0.40 * _bonus(scaling_rel_std)
    score += 0.40 * _bonus(cross_hess_rel)
    return float(domain_ok_frac) * score


def probe_output_transforms(
    model,
    datagen,
    *,
    Nxvars: int,
    device=None,
    max_batches: int = 8,
    max_points: int = 4096,
    min_domain_frac: float = 0.90,
    min_score_improvement: float = 0.50,
    out_idx: int = 0,
    rat_deg_num: int = 1,
    rat_deg_den: int = 1,
    eps_domain: float = 1e-12,
    max_abs_exp: float = 20.0,
) -> TransformHint:
    """Probe a small set of unary output transforms T(u) and score simplicity."""
    from .fitting_utils import _rational_probe_nd

    X, F, G, H = _sample_values_grads_hessians(
        model,
        datagen,
        max_batches=max_batches,
        max_points=max_points,
        device=device,
        out_idx=out_idx,
    )
    if X is None:
        dummy = TransformProbeSpec(
            name="identity",
            domain_ok_frac=0.0,
            n_points=0,
            poly2_rms_rel=float("inf"),
            rat_rms_rel=float("inf"),
            hess_const_rel=float("inf"),
            scaling_rel_std=float("inf"),
            cross_hess_rel=float("inf"),
            score=0.0,
        )
        return TransformHint(
            ok=False,
            best_name="identity",
            score_improvement=0.0,
            baseline=dummy,
            best=dummy,
            candidates=[dummy],
            reason="no-data",
        )

    Xv = X[:, :Nxvars]
    f = F.view(-1)
    g = G[:, :Nxvars]
    h = H[:, :Nxvars, :Nxvars]

    # Ensure float64 for probe math stability
    Xv = Xv.to(dtype=torch.float64)
    f = f.to(dtype=torch.float64)
    g = g.to(dtype=torch.float64)
    h = h.to(dtype=torch.float64)

    f_scale = float(torch.median(f.abs()).item())
    f_eps = max(eps_domain, 1e-9 * max(f_scale, 1.0))

    def _identity_stats(mask: torch.Tensor) -> TransformProbeSpec:
        Xm = Xv[mask]
        ym = f[mask]
        gm = g[mask]
        Hm = h[mask]
        poly2 = _poly_fit_rms_rel(Xm, ym, degree=2)
        rat = _rational_probe_nd(
            Xm, ym, deg_num=rat_deg_num, deg_den=rat_deg_den, dtype=torch.float64
        )
        hrel, _ = _hess_const_rel(Hm)
        srel = _scaling_rel_std(Xm, ym, gm)
        crel = _cross_hess_rel(Hm)
        score = _probe_score(
            domain_ok_frac=float(mask.float().mean().item()),
            poly2_rms_rel=poly2,
            rat_rms_rel=rat,
            hess_const_rel=hrel,
            scaling_rel_std=srel,
            cross_hess_rel=crel,
        )
        return TransformProbeSpec(
            name="identity",
            domain_ok_frac=float(mask.float().mean().item()),
            n_points=int(mask.sum().item()),
            poly2_rms_rel=float(poly2),
            rat_rms_rel=float(rat),
            hess_const_rel=float(hrel),
            scaling_rel_std=float(srel),
            cross_hess_rel=float(crel),
            score=float(score),
        )

    base_mask = torch.isfinite(f)
    baseline = _identity_stats(base_mask)
    candidates: List[TransformProbeSpec] = [baseline]

    # --- Transform definitions (y -> z, dz/dy, d2z/dy2, mask) ---
    def _log_mask(y):
        return torch.isfinite(y) & (y > f_eps)

    def _recip_mask(y):
        return torch.isfinite(y) & (y.abs() > f_eps)

    def _sqrt_mask(y):
        return torch.isfinite(y) & (y >= 0.0)

    def _square_mask(y):
        # Avoid overflow in y*y
        y2 = y * y
        return torch.isfinite(y) & torch.isfinite(y2)

    def _asin_mask(y):
        # Keep a small safety margin away from ±1 to avoid blow-ups.
        return torch.isfinite(y) & (y.abs() <= (1.0 - 1e-6))

    def _atanh_mask(y):
        return torch.isfinite(y) & (y.abs() < (1.0 - 1e-6))

    def _exp_mask(y):
        # Numerical guard: exp(y) overflows quickly.
        return torch.isfinite(y) & (y.abs() <= max_abs_exp)

    transforms = [
        ("log", torch.log, lambda y: 1.0 / y, lambda y: -1.0 / (y * y), _log_mask),
        ("exp", torch.exp, torch.exp, torch.exp, _exp_mask),
        (
            "recip",
            torch.reciprocal,
            lambda y: -1.0 / (y * y),
            lambda y: 2.0 / (y * y * y),
            _recip_mask,
        ),
        (
            "sqrt",
            torch.sqrt,
            lambda y: 0.5 / torch.sqrt(y.clamp_min(f_eps)),
            lambda y: -0.25 / (y.clamp_min(f_eps) ** 1.5),
            _sqrt_mask,
        ),
        (
            "square",
            torch.square,
            lambda y: 2.0 * y,
            lambda y: 2.0 * torch.ones_like(y),
            _square_mask,
        ),
        (
            "asin",
            torch.asin,
            lambda y: 1.0 / torch.sqrt((1.0 - y * y).clamp_min(1e-12)),
            lambda y: y / ((1.0 - y * y).clamp_min(1e-12) ** 1.5),
            _asin_mask,
        ),
        (
            "acos",
            torch.acos,
            lambda y: -1.0 / torch.sqrt((1.0 - y * y).clamp_min(1e-12)),
            lambda y: -y / ((1.0 - y * y).clamp_min(1e-12) ** 1.5),
            _asin_mask,
        ),
    ]
    if hasattr(torch, "atanh"):
        transforms.append(
            (
                "atanh",
                torch.atanh,
                lambda y: 1.0 / (1.0 - y * y).clamp_min(1e-12),
                lambda y: (2.0 * y) / ((1.0 - y * y).clamp_min(1e-12) ** 2),
                _atanh_mask,
            )
        )

    # Evaluate each transform
    for name, op, d1, d2, mask_fn in transforms:
        mask = mask_fn(f)
        dom_frac = float(mask.float().mean().item())
        n_ok = int(mask.sum().item())
        if dom_frac < 0.01 or n_ok < 50:
            candidates.append(
                TransformProbeSpec(
                    name=name,
                    domain_ok_frac=dom_frac,
                    n_points=n_ok,
                    poly2_rms_rel=float("inf"),
                    rat_rms_rel=float("inf"),
                    hess_const_rel=float("inf"),
                    scaling_rel_std=float("inf"),
                    cross_hess_rel=float("inf"),
                    score=0.0,
                )
            )
            continue
        Xm = Xv[mask]
        ym = f[mask]
        gm = g[mask]
        Hm = h[mask]

        with torch.no_grad():
            z = op(ym)
            dz = d1(ym)
            d2z = d2(ym)

            # Chain rule
            gz = dz.view(-1, 1) * gm
            outer = gm.unsqueeze(2) * gm.unsqueeze(1)  # [N,d,d]
            Hz = d2z.view(-1, 1, 1) * outer + dz.view(-1, 1, 1) * Hm

        # Compute probe metrics
        poly2 = _poly_fit_rms_rel(Xm, z, degree=2)
        rat = _rational_probe_nd(
            Xm, z, deg_num=rat_deg_num, deg_den=rat_deg_den, dtype=torch.float64
        )
        hrel, _ = _hess_const_rel(Hz)
        srel = _scaling_rel_std(Xm, z, gz)
        crel = _cross_hess_rel(Hz)
        score = _probe_score(
            domain_ok_frac=dom_frac,
            poly2_rms_rel=poly2,
            rat_rms_rel=rat,
            hess_const_rel=hrel,
            scaling_rel_std=srel,
            cross_hess_rel=crel,
        )

        candidates.append(
            TransformProbeSpec(
                name=name,
                domain_ok_frac=dom_frac,
                n_points=n_ok,
                poly2_rms_rel=float(poly2),
                rat_rms_rel=float(rat),
                hess_const_rel=float(hrel),
                scaling_rel_std=float(srel),
                cross_hess_rel=float(crel),
                score=float(score),
            )
        )

    # ------------------------------------------------------------------
    # Affine inverse-trig probes
    #   z = asin((y - beta)/alpha) or acos((y - beta)/alpha)
    # These act like "counterterm/counterfactor" preconditioners for
    # inverse-trig templates (beta shift, alpha scale).
    # ------------------------------------------------------------------
    def _quantile_sorted_1d(y1: torch.Tensor, q: float) -> float:
        y1 = y1.view(-1)
        if y1.numel() == 0:
            return float("nan")
        ys = torch.sort(y1).values
        # Nearest-rank index
        i = int(round((ys.numel() - 1) * float(q)))
        i = max(0, min(i, ys.numel() - 1))
        return float(ys[i].item())

    def _unique_floats(
        vals: List[float], *, rel_tol: float = 1e-6, abs_tol: float = 1e-12
    ) -> List[float]:
        out: List[float] = []
        for v in vals:
            if not math.isfinite(v):
                continue
            keep = True
            for u in out:
                if abs(v - u) <= max(abs_tol, rel_tol * max(abs(v), abs(u), 1.0)):
                    keep = False
                    break
            if keep:
                out.append(float(v))
        return out

    def _add_affine_inverse_trig(kind: str):
        # kind in {"asin_affine", "acos_affine"}
        y = f
        finite = torch.isfinite(y)
        yfin = y[finite]
        if yfin.numel() < 200:
            return

        # Robust central/tail summaries
        med = float(torch.median(yfin).item())
        mean = float(yfin.mean().item())
        q10 = _quantile_sorted_1d(yfin, 0.10)
        q90 = _quantile_sorted_1d(yfin, 0.90)

        beta_cands = _unique_floats(
            [
                med,
                mean,
                0.5 * (q10 + q90),
                0.0,
            ]
        )
        # Keep it small; we only need a handful.
        beta_cands = beta_cands[:4]

        # Evaluate a small grid of (beta, alpha)
        for beta in beta_cands:
            dev = (yfin - beta).abs()
            a90 = _quantile_sorted_1d(dev, 0.90)
            a95 = _quantile_sorted_1d(dev, 0.95)
            amax = float(dev.max().item())
            alpha_cands = _unique_floats(
                [
                    a95,
                    a90,
                    amax,
                ]
            )
            # Avoid pathological huge scaling (rare outliers) dominating.
            # Let LM correct the amplitude; the probe just needs a usable init.
            if math.isfinite(a95) and a95 > 0:
                alpha_cands = [a for a in alpha_cands if a <= 25.0 * max(a95, 1e-12)]
            alpha_cands = alpha_cands[:3]

            for alpha in alpha_cands:
                if (not math.isfinite(alpha)) or alpha <= f_eps:
                    continue
                inv_alpha = 1.0 / float(alpha)
                inv_alpha2 = inv_alpha * inv_alpha

                v_all = (y - beta) * inv_alpha
                mask = torch.isfinite(v_all) & (v_all.abs() <= (1.0 - 1e-6))
                dom_frac = float(mask.float().mean().item())
                n_ok = int(mask.sum().item())
                if dom_frac < 0.01 or n_ok < 50:
                    candidates.append(
                        TransformProbeSpec(
                            name=kind,
                            domain_ok_frac=dom_frac,
                            n_points=n_ok,
                            poly2_rms_rel=float("inf"),
                            rat_rms_rel=float("inf"),
                            hess_const_rel=float("inf"),
                            scaling_rel_std=float("inf"),
                            cross_hess_rel=float("inf"),
                            score=0.0,
                            params={"alpha": float(alpha), "beta": float(beta)},
                        )
                    )
                    continue

                Xm = Xv[mask]
                vm = v_all[mask]
                gm = g[mask]
                Hm = h[mask]

                with torch.no_grad():
                    # Outer op
                    if kind == "asin_affine":
                        z = torch.asin(vm)
                        dz = inv_alpha / torch.sqrt((1.0 - vm * vm).clamp_min(1e-12))
                        d2z = (vm * inv_alpha2) / ((1.0 - vm * vm).clamp_min(1e-12) ** 1.5)
                    else:
                        z = torch.acos(vm)
                        dz = -inv_alpha / torch.sqrt((1.0 - vm * vm).clamp_min(1e-12))
                        d2z = (-vm * inv_alpha2) / ((1.0 - vm * vm).clamp_min(1e-12) ** 1.5)

                    # Chain rule
                    gz = dz.view(-1, 1) * gm
                    outer = gm.unsqueeze(2) * gm.unsqueeze(1)
                    Hz = d2z.view(-1, 1, 1) * outer + dz.view(-1, 1, 1) * Hm

                poly2 = _poly_fit_rms_rel(Xm, z, degree=2)
                rat = _rational_probe_nd(
                    Xm, z, deg_num=rat_deg_num, deg_den=rat_deg_den, dtype=torch.float64
                )
                hrel, _ = _hess_const_rel(Hz)
                srel = _scaling_rel_std(Xm, z, gz)
                crel = _cross_hess_rel(Hz)
                score = _probe_score(
                    domain_ok_frac=dom_frac,
                    poly2_rms_rel=poly2,
                    rat_rms_rel=rat,
                    hess_const_rel=hrel,
                    scaling_rel_std=srel,
                    cross_hess_rel=crel,
                )

                candidates.append(
                    TransformProbeSpec(
                        name=kind,
                        domain_ok_frac=dom_frac,
                        n_points=n_ok,
                        poly2_rms_rel=float(poly2),
                        rat_rms_rel=float(rat),
                        hess_const_rel=float(hrel),
                        scaling_rel_std=float(srel),
                        cross_hess_rel=float(crel),
                        score=float(score),
                        params={"alpha": float(alpha), "beta": float(beta)},
                    )
                )

    _add_affine_inverse_trig("asin_affine")
    _add_affine_inverse_trig("acos_affine")

    # Pick best transform
    best = max(candidates, key=lambda s: float(s.score))
    improvement = float(best.score - baseline.score)
    ok = (
        (best.name != "identity")
        and (best.domain_ok_frac >= min_domain_frac)
        and (improvement >= min_score_improvement)
    )
    reason = ""
    if not ok:
        if best.name == "identity":
            reason = "no-transform-better"
        elif best.domain_ok_frac < min_domain_frac:
            reason = "domain-too-small"
        elif improvement < min_score_improvement:
            reason = "insufficient-improvement"
        else:
            reason = "not-ok"

    return TransformHint(
        ok=bool(ok),
        best_name=str(best.name),
        score_improvement=float(improvement),
        baseline=baseline,
        best=best,
        candidates=candidates,
        reason=reason,
    )


def detect_log_hessian_quadratic(
    model,
    datagen,
    *,
    Nxvars: int,
    device=None,
    max_batches: int = 8,
    max_points: int = 4096,
    out_idx: int = 0,
    eps_domain: float = 1e-12,
    max_abs_exp: float = 20.0,
    min_domain_frac: float = 0.90,
    max_hess_const_rel: float = 0.15,
    rank_rel_tol: float = 1e-3,
) -> QuadraticHint:
    X, F, G, H = _sample_values_grads_hessians(
        model,
        datagen,
        max_batches=max_batches,
        max_points=max_points,
        device=device,
        out_idx=out_idx,
    )
    if X is None:
        return QuadraticHint(
            type="log",
            ok=False,
            domain_ok_frac=0.0,
            n_points=0,
            hess_const_rel=float("inf"),
            score=0.0,
            rank=0,
            eigvals=[],
            mean_hess=torch.zeros(Nxvars, Nxvars, dtype=torch.float64),
        )

    _Xv = X[:, :Nxvars].to(dtype=torch.float64)
    u = F.view(-1).to(dtype=torch.float64)
    g = G[:, :Nxvars].to(dtype=torch.float64)
    h = H[:, :Nxvars, :Nxvars].to(dtype=torch.float64)

    u_scale = float(torch.median(u.abs()).item())
    u_eps = max(eps_domain, 1e-9 * max(u_scale, 1.0))
    mask = torch.isfinite(u) & (u > u_eps)
    dom_frac = float(mask.float().mean().item())
    n_ok = int(mask.sum().item())
    if n_ok < 50:
        return QuadraticHint(
            type="log",
            ok=False,
            domain_ok_frac=dom_frac,
            n_points=n_ok,
            hess_const_rel=float("inf"),
            score=0.0,
            rank=0,
            eigvals=[],
            mean_hess=torch.zeros(Nxvars, Nxvars, dtype=torch.float64),
        )

    um = u[mask]
    gm = g[mask]
    Hm = h[mask]

    # H(log u) = H(u)/u - (∇u ⊗ ∇u)/u^2
    outer = gm.unsqueeze(2) * gm.unsqueeze(1)
    Hlog = (Hm / um.view(-1, 1, 1)) - (outer / (um.view(-1, 1, 1) ** 2))

    hrel, Hbar = _hess_const_rel(Hlog)
    Hbar = Hbar.detach().cpu()
    try:
        evals = torch.linalg.eigvalsh(0.5 * (Hbar + Hbar.T)).detach().cpu().to(dtype=torch.float64)
    except Exception:
        evals = torch.zeros(Nxvars, dtype=torch.float64)
    max_abs = float(evals.abs().max().item())
    thr = rank_rel_tol * max(max_abs, 1e-12)
    rank = int((evals.abs() > thr).sum().item())
    eigvals = [float(v) for v in evals.tolist()]
    score = float(dom_frac) * (1.0 / (1.0 + float(hrel)))
    ok = (dom_frac >= min_domain_frac) and (float(hrel) <= max_hess_const_rel)

    return QuadraticHint(
        type="log",
        ok=bool(ok),
        domain_ok_frac=float(dom_frac),
        n_points=int(n_ok),
        hess_const_rel=float(hrel),
        score=float(score),
        rank=int(rank),
        eigvals=eigvals,
        mean_hess=Hbar,
    )


def detect_square_hessian_quadratic(
    model,
    datagen,
    *,
    Nxvars: int,
    device=None,
    max_batches: int = 8,
    max_points: int = 4096,
    out_idx: int = 0,
    max_hess_const_rel: float = 0.15,
    rank_rel_tol: float = 1e-3,
) -> QuadraticHint:
    X, F, G, H = _sample_values_grads_hessians(
        model,
        datagen,
        max_batches=max_batches,
        max_points=max_points,
        device=device,
        out_idx=out_idx,
    )
    if X is None:
        return QuadraticHint(
            type="square",
            ok=False,
            domain_ok_frac=0.0,
            n_points=0,
            hess_const_rel=float("inf"),
            score=0.0,
            rank=0,
            eigvals=[],
            mean_hess=torch.zeros(Nxvars, Nxvars, dtype=torch.float64),
        )

    u = F.view(-1).to(dtype=torch.float64)
    g = G[:, :Nxvars].to(dtype=torch.float64)
    h = H[:, :Nxvars, :Nxvars].to(dtype=torch.float64)

    # H(u^2) = 2*(∇u ⊗ ∇u) + 2*u*H(u)
    outer = g.unsqueeze(2) * g.unsqueeze(1)
    Hu2 = 2.0 * outer + 2.0 * u.view(-1, 1, 1) * h

    hrel, Hbar = _hess_const_rel(Hu2)
    Hbar = Hbar.detach().cpu()
    try:
        evals = torch.linalg.eigvalsh(0.5 * (Hbar + Hbar.T)).detach().cpu().to(dtype=torch.float64)
    except Exception:
        evals = torch.zeros(Nxvars, dtype=torch.float64)
    max_abs = float(evals.abs().max().item())
    thr = rank_rel_tol * max(max_abs, 1e-12)
    rank = int((evals.abs() > thr).sum().item())
    eigvals = [float(v) for v in evals.tolist()]
    score = 1.0 / (1.0 + float(hrel))
    ok = float(hrel) <= max_hess_const_rel

    return QuadraticHint(
        type="square",
        ok=bool(ok),
        domain_ok_frac=1.0,
        n_points=int(u.numel()),
        hess_const_rel=float(hrel),
        score=float(score),
        rank=int(rank),
        eigvals=eigvals,
        mean_hess=Hbar,
    )


def _trig_from_profile_with_phase(P: torch.Tensor, dx: float) -> Tuple[float, float, float]:
    Q = P - P.mean()
    F = torch.fft.rfft(Q)
    A = F.abs()
    if A.numel() <= 2:
        return 0.0, 0.0, 0.0
    k = int(torch.argmax(A[1:])) + 1
    omega = 2 * math.pi * (k / (len(P) * dx))
    strength = float((A[k] / (A[1:].median() + 1e-9)).item())
    phase = float(torch.angle(F[k]).item())
    return float(omega), float(strength), float(phase)


def _unwrap_phase(phases: List[float]) -> List[float]:
    if not phases:
        return []
    out = [float(phases[0])]
    for ph in phases[1:]:
        ph = float(ph)
        prev = out[-1]
        d = ph - prev
        # Wrap to [-pi, pi)
        d = (d + math.pi) % (2.0 * math.pi) - math.pi
        out.append(prev + d)
    return out


def discover_trig_argument_structure(
    model,
    datagen,
    *,
    Nxvars: int,
    trig_specs: Optional[List[TrigAxisSpec]] = None,
    device=None,
    max_batches: int = 8,
    max_points: int = 20000,
    n_line: int = 256,
    n_slices: int = 5,
    eta_scale: float = 0.8,
    strength_floor: float = 3.0,
    r2_threshold: float = 0.85,
    omega_var_threshold: float = 0.05,
) -> List[PeriodicityStructureHint]:
    """Second-stage trig probe: detect product vs difference structure in trig arguments."""
    if trig_specs is None:
        trig_specs = discover_trig_axes(model, datagen, Nxvars=Nxvars, device=device)
    if not trig_specs:
        return []

    X = _sample_inputs_from_datagen(
        datagen, max_batches=max_batches, max_points=max_points, device=device
    )
    if X is None:
        return []
    Xv = X[:, :Nxvars].to(dtype=torch.float64)
    # Fix degenerate tmin/tmax (e.g. from oracle-converted specs)
    for spec in trig_specs:
        if spec.tmin >= spec.tmax:
            col = Xv[:, spec.axis]
            spec.tmin = float(col.min())
            spec.tmax = float(col.max())
    x_ref = Xv.median(dim=0).values

    hints: List[PeriodicityStructureHint] = []
    for spec in trig_specs:
        axis = int(spec.axis)
        omega0 = float(spec.omega)
        if not math.isfinite(omega0) or omega0 <= 0.0:
            continue

        best_hint: Optional[PeriodicityStructureHint] = None
        for partner in range(Nxvars):
            if partner == axis:
                continue

            # Slice partner values by quantiles
            q = torch.linspace(0.15, 0.85, n_slices, dtype=Xv.dtype)
            vals = torch.quantile(Xv[:, partner], q).detach().cpu().to(dtype=torch.float64)

            omegas: List[float] = []
            phases: List[float] = []
            v_used: List[float] = []

            for v in vals.tolist():
                x0 = x_ref.clone()
                # Make t correspond to absolute x_axis
                x0[axis] = 0.0
                x0[partner] = float(v)
                u = torch.zeros(Nxvars, dtype=torch.float64)
                u[axis] = 1.0
                ts, _, _, d2_line = sample_line_curvature(
                    provider=model,
                    x0=x0,
                    u=u,
                    tmin=float(spec.tmin),
                    tmax=float(spec.tmax),
                    n=n_line,
                    out_idx=0,
                )
                if ts.numel() < 32:
                    continue
                dt = float(ts[1] - ts[0])
                eta = eta_scale * dt
                P = poisson_profile(ts, d2_line, ts, eta)
                omega, strength, phase = _trig_from_profile_with_phase(P, dt)
                if (
                    (not math.isfinite(omega))
                    or (not math.isfinite(strength))
                    or (not math.isfinite(phase))
                ):
                    continue
                if strength < strength_floor:
                    continue
                omegas.append(float(omega))
                phases.append(float(phase))
                v_used.append(float(v))

            if len(omegas) < max(3, n_slices // 2):
                continue

            v_t = torch.tensor(v_used, dtype=torch.float64)
            om_t = torch.tensor(omegas, dtype=torch.float64)
            ph_unwrap = torch.tensor(_unwrap_phase(phases), dtype=torch.float64)

            om_mean = float(om_t.mean().item())
            om_std = float(om_t.std(unbiased=False).item())
            if om_mean < 1e-12:
                continue

            # Product test: omega vs |v|
            a0_om, a1_om, r2_om = _linreg(v_t.abs(), om_t)

            # Difference test: phase vs v (only meaningful if omega is ~constant)
            a0_ph, a1_ph, r2_ph = _linreg(v_t, ph_unwrap)

            # Scores
            omega_var = om_std / (abs(om_mean) + 1e-12)
            score_prod = 0.0
            if float(r2_om) >= r2_threshold and omega_var >= omega_var_threshold:
                # Prefer near-zero intercept (pure product)
                intercept_pen = math.exp(-abs(float(a0_om)) / (abs(om_mean) + 1e-12))
                score_prod = float(r2_om) * float(omega_var) * float(intercept_pen)

            score_diff = 0.0
            if float(r2_ph) >= r2_threshold and omega_var <= 2.0 * omega_var_threshold:
                ratio = abs(float(a1_ph)) / (abs(om_mean) + 1e-12)
                ratio_pen = math.exp(-(((ratio - 1.0) / 0.35) ** 2))
                omega_pen = math.exp(-((omega_var / omega_var_threshold) ** 2))
                score_diff = float(r2_ph) * float(ratio_pen) * float(omega_pen)

            if score_prod <= 0.0 and score_diff <= 0.0:
                continue

            if score_prod >= score_diff:
                kind = "product"
                score = score_prod
            else:
                kind = "difference"
                score = score_diff

            cand = PeriodicityStructureHint(
                axis=axis,
                partner=partner,
                kind=kind,
                score=float(score),
                omega0=float(omega0),
                omega_std=float(om_std),
                omega_r2=float(r2_om),
                omega_slope=float(a1_om),
                omega_intercept=float(a0_om),
                phase_r2=float(r2_ph),
                phase_slope=float(a1_ph),
                phase_intercept=float(a0_ph),
                n_slices=int(len(v_used)),
                n_line=int(n_line),
            )

            if best_hint is None or cand.score > best_hint.score:
                best_hint = cand

        if best_hint is None:
            hints.append(
                PeriodicityStructureHint(
                    axis=axis,
                    partner=None,
                    kind="none",
                    score=0.0,
                    omega0=float(omega0),
                    omega_std=0.0,
                    omega_r2=0.0,
                    omega_slope=0.0,
                    omega_intercept=0.0,
                    phase_r2=0.0,
                    phase_slope=0.0,
                    phase_intercept=0.0,
                    n_slices=0,
                    n_line=int(n_line),
                )
            )
        else:
            hints.append(best_hint)

    return hints


# =============================
# Data-only trig detection (for compound variables before training)
# =============================


def discover_trig_from_data(
    z: torch.Tensor,
    y: torch.Tensor,
    *,
    n_bins: int = 256,
    strength_threshold: float = 5.0,
    max_omega: float = 50.0,
) -> Optional[TrigAxisSpec]:
    """Detect trig-like relationship between z and y using FFT on binned data.

    This is a data-only approach that doesn't require a trained neural network.
    It's useful for detecting trig structure on compound variables before building
    the compound model.

    The approach:
    1. Sort (z, y) by z
    2. Bin into uniform-width bins
    3. Compute mean y in each bin
    4. FFT the binned profile to detect periodic structure

    Parameters
    ----------
    z : torch.Tensor, shape [N] or [N, 1]
        Compound variable values (e.g., z = x*y).
    y : torch.Tensor, shape [N] or [N, 1]
        Target values.
    n_bins : int
        Number of bins for FFT analysis.
    strength_threshold : float
        Minimum spectral strength to flag as trig.
    max_omega : float
        Maximum angular frequency to consider.

    Returns
    -------
    TrigAxisSpec or None
        Detected trig structure, or None if not detected.
    """
    z_flat = z.view(-1)
    y_flat = y.view(-1)

    # Filter valid points
    m = torch.isfinite(z_flat) & torch.isfinite(y_flat)
    z_valid = z_flat[m]
    y_valid = y_flat[m]
    N = z_valid.numel()

    if N < n_bins:
        return None

    # Sort by z
    idx = torch.argsort(z_valid)
    z_sorted = z_valid[idx]
    y_sorted = y_valid[idx]

    # Determine z range
    zmin = float(z_sorted[0])
    zmax = float(z_sorted[-1])
    if not math.isfinite(zmin) or not math.isfinite(zmax) or zmax <= zmin:
        return None

    # Bin by z and compute mean y in each bin
    dz = (zmax - zmin) / n_bins
    bin_means = torch.zeros(n_bins, dtype=y_sorted.dtype, device=y_sorted.device)
    bin_counts = torch.zeros(n_bins, dtype=torch.long, device=y_sorted.device)

    for i in range(N):
        zi = float(z_sorted[i])
        bi = min(n_bins - 1, int((zi - zmin) / dz))
        bin_means[bi] += y_sorted[i]
        bin_counts[bi] += 1

    # Compute actual means
    valid_bins = bin_counts > 0
    if valid_bins.sum() < n_bins // 2:
        return None  # Too sparse

    # Fill empty bins with interpolation (simple: use last valid)
    last_valid = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_means[i] = bin_means[i] / float(bin_counts[i])
            last_valid = float(bin_means[i])
        else:
            bin_means[i] = last_valid

    # Detrend (remove best-fit linear component) to avoid flagging linear relations as "trig".
    profile_for_fft = bin_means
    try:
        t_centers = (torch.arange(n_bins, device=bin_means.device, dtype=bin_means.dtype) + 0.5) * dz + zmin
        a_lin, b_lin, r2_lin = _linreg(t_centers, bin_means)
        trend = a_lin + b_lin * t_centers
        resid = bin_means - trend
        resid_rms = float((resid - resid.mean()).pow(2).mean().sqrt().item())
        prof_rms = float((bin_means - bin_means.mean()).pow(2).mean().sqrt().item()) + 1e-12
        if float(r2_lin) >= 0.999 and (resid_rms / prof_rms) < 1e-2:
            return None
        profile_for_fft = resid
    except Exception:
        profile_for_fft = bin_means

    omega, strength = trig_from_profile(profile_for_fft, dz)

    if not math.isfinite(omega) or not math.isfinite(strength):
        return None
    if omega <= 0.0 or omega > max_omega:
        return None
    if strength < strength_threshold:
        return None

    return TrigAxisSpec(
        axis=0,  # Convention: axis 0 for compound variable z
        omega=omega,
        strength=strength,
        n_points=N,
        tmin=zmin,
        tmax=zmax,
    )


# =============================
# Leaf-level feature detection (unified for compound variables)
# =============================


@dataclass
class LeafFeatures:
    """Unified feature detection result for a leaf's effective input space.

    For simple atoms: input space = x[:, var_idxs]
    For compound atoms: input space = [z, extras] where z = compound expression

    In both cases, axis 0 refers to the "primary" input (raw x_j or compound z).
    """

    trig_by_axis: Dict[int, TrigAxisSpec]
    # Can be extended with other features: scaling, parity, etc.


class LeafProvider:
    """Wrapper to make a neural network leaf behave like a full model for feature detection.

    This allows existing feature detection routines (discover_trig_axes, etc.) to work
    on a leaf's input space rather than the full model's input space.
    """

    def __init__(self, leaf, *, device=None, dtype=None):
        """
        Parameters
        ----------
        leaf : nn.Module
            A neural network leaf (e.g., from NestyNet) that has .forward(), .grad(), .grad_grad().
        device : optional
            Override device for computations.
        dtype : optional
            Override dtype for computations.
        """
        self.leaf = leaf
        self._device = device
        self._dtype = dtype

    def parameters(self):
        """Yield leaf parameters (for device inference)."""
        return self.leaf.parameters()

    def forward(self, x):
        """Forward pass through the leaf."""
        return self.leaf(x)

    def __call__(self, x):
        return self.forward(x)

    def grad(self, x):
        """Gradient of leaf output w.r.t. leaf inputs."""
        return self.leaf.grad(x)

    def grad_grad(self, x):
        """Hessian of leaf output w.r.t. leaf inputs."""
        return self.leaf.grad_grad(x)


def discover_leaf_features(
    leaf,
    atom,
    x_data: torch.Tensor,
    *,
    detect_trig: bool = True,
    trig_strength_threshold: float = 5.0,
    trig_max_omega: float = 50.0,
    n_line: int = 512,
    device=None,
    dtype=None,
) -> LeafFeatures:
    """Detect features on a leaf's effective input space.

    For simple atoms: input space = x[:, var_idxs]
    For compound atoms: input space = [z, extras] where z = eval_input_expr(expr, x)

    This unified approach allows trig detection (and other features) to work on
    compound variables like z = x*y, where the product itself may be trig-like
    even if neither x nor y alone is.

    Parameters
    ----------
    leaf : nn.Module
        The neural network leaf.
    atom : AtomNode
        AST atom (has var_idxs, compound info via atom.inputs).
    x_data : torch.Tensor, shape [N, Nx]
        Raw input data.
    detect_trig : bool
        Whether to run trig axis detection.
    trig_strength_threshold : float
        Minimum spectral strength to flag an axis as trig.
    trig_max_omega : float
        Maximum angular frequency to consider.
    n_line : int
        Number of points for line curvature sampling.
    device : optional
        Device for computations.
    dtype : optional
        Dtype for computations.

    Returns
    -------
    LeafFeatures
        Detected features with axis indices relative to the leaf's input space.
        For compound atoms, axis 0 is the compound variable z.
    """
    from nestynet_sr.sr_core.bridges import eval_inputs

    # Build leaf input data (unified: works for both simple and compound atoms)
    leaf_input, _, _ = eval_inputs(atom, x_data)
    n_leaf_vars = leaf_input.shape[1]

    # Create leaf provider for feature detection
    provider = LeafProvider(leaf, device=device, dtype=dtype)

    # Create a simple datagen that yields the leaf_input tensor in batches
    def leaf_datagen():
        batch_size = 2048
        for i in range(0, leaf_input.shape[0], batch_size):
            yield leaf_input[i : i + batch_size]

    trig_by_axis: Dict[int, TrigAxisSpec] = {}

    if detect_trig:
        try:
            trig_specs = discover_trig_axes(
                model=provider,
                datagen=leaf_datagen,
                Nxvars=n_leaf_vars,
                device=device,
                n_line=n_line,
                strength_threshold=trig_strength_threshold,
                max_omega=trig_max_omega,
            )
            for spec in trig_specs:
                if spec.axis not in trig_by_axis:
                    trig_by_axis[spec.axis] = spec
        except Exception:
            # Trig detection failed - leave empty
            pass

    return LeafFeatures(trig_by_axis=trig_by_axis)


def discover_compound_features_from_data(
    z_vals: torch.Tensor,
    y_vals: torch.Tensor,
    *,
    detect_trig: bool = True,
    trig_strength_threshold: float = 5.0,
    trig_max_omega: float = 50.0,
    trig_n_bins: int = 256,
) -> LeafFeatures:
    """Detect features on a compound variable using data-only analysis.

    This is a lightweight alternative to discover_leaf_features() that doesn't
    require a trained neural network. It's useful for detecting trig structure
    on compound variables (z = x*y, etc.) before building the compound model.

    Parameters
    ----------
    z_vals : torch.Tensor, shape [N] or [N, 1]
        Compound variable values.
    y_vals : torch.Tensor, shape [N] or [N, 1]
        Target values (e.g., teacher output or raw y data).
    detect_trig : bool
        Whether to run trig detection.
    trig_strength_threshold : float
        Minimum spectral strength to flag as trig.
    trig_max_omega : float
        Maximum angular frequency to consider.
    trig_n_bins : int
        Number of bins for FFT analysis.

    Returns
    -------
    LeafFeatures
        Detected features with axis 0 representing the compound variable z.
    """
    trig_by_axis: Dict[int, TrigAxisSpec] = {}

    if detect_trig:
        try:
            trig_spec = discover_trig_from_data(
                z_vals,
                y_vals,
                n_bins=trig_n_bins,
                strength_threshold=trig_strength_threshold,
                max_omega=trig_max_omega,
            )
            if trig_spec is not None:
                trig_by_axis[0] = trig_spec
        except Exception:
            pass

    return LeafFeatures(trig_by_axis=trig_by_axis)
