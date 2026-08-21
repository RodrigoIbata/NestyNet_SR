# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Configuration for the generalized-symmetry (GS) V3 prototype layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GeneralizedSymmetryConfig:
    """Switchboard for symmetry-generated Stage-A and DE proposals.

    V3 distinguishes three concepts:

    * named/known Lie generators from V2;
    * a learned general affine probe X=(A x + b).grad_x;
    * jet-level separability witnesses on J^2 f.

    The policy controls whether GS augments baseline proposals or replaces the
    baseline motifs it directly shadows.
    """

    enabled: bool = False
    mode: str = "propose"  # off | audit | propose | auto
    policy: str = "augment"  # augment | replace-shadowed | gs-only-affine

    # Lie-generator families.
    # ``known_generators`` is kept as a backward-compatible alias used by some
    # V2/V3 bridge code; both it and ``known_lie`` must be true for the named
    # hand-written GS generator bank to fire.
    known_generators: bool = True
    known_lie: bool = True
    general_affine: bool = False
    affine_dense: bool = False  # backward-compatible alias
    translations: bool = True
    diagonal_translations: bool = True
    scalings: bool = True
    rotations: bool = True
    lorentz_boosts: bool = False
    output_equivariance: bool = True
    auto_maximal: bool = True

    # Learned affine numerical controls.
    general_affine_max_nullvecs: int = 6
    general_affine_min_generator_fraction: float = 0.60
    general_affine_snap_tol: float = 0.22
    general_affine_max_null_ratio: float = 0.25
    general_affine_report_unclassified: bool = True
    affine_max_generators: int = 8
    affine_max_nonzero: int = 4
    general_affine_promotion: bool = True
    general_affine_promotion_residual_tol: float = 1.0e-8
    general_affine_promotion_max_bootstrap_angle: float = 1.0e-3
    general_affine_promotion_max_chart_complexity: int = 24
    general_affine_promote_output_equivariant: bool = False

    # Charts for the affine determining operator. ``identity`` reproduces the
    # raw-coordinate solve exactly; ``log`` re-expresses the samples as
    # u = log(x) (chain-ruled gradients) so scaling symmetries appear as
    # translations and monomial invariants as linear covectors. Default is
    # identity-only, so enabling GS alone changes nothing on this route.
    general_affine_charts: tuple = ("identity",)
    general_affine_chart_snap_denominator: int = 4

    # Noise-calibrated promotion (opt-in). The absolute residual gates only
    # pass with oracle-exact gradients; this tier promotes on
    # surrogate-noise-relative evidence instead: spectral-gap nullity
    # selection in the determining solve, held-out/train consistency,
    # bootstrap subspace stability, and a snap-degradation factor. Defaults
    # were calibrated on log-chart monomial fixtures at gradient noise up to
    # ~1e-3 (see tests/test_gs_noise_calibrated_promotion.py); at ~1e-2 noise
    # the spectrum blurs and the gate correctly declines.
    general_affine_promotion_noise_calibrated: bool = False
    noise_calibrated_min_spectral_gap: float = 10.0

    # Pairwise-witness composition (opt-in): compose accepted pairwise scaling
    # witnesses into global +/-1 monomial rays by sign propagation over the
    # constraint graph, validated jointly against the sampled gradients. The
    # pairwise tests survive surrogate gradient noise (~1e-3..5e-3 measured on
    # trained surrogates) that breaks the global determining solve's bracket
    # closure, so this is the noise-robust route to 3+-variable products and
    # ratios.
    pairwise_composition: bool = False
    pairwise_composition_min_support: int = 3
    pairwise_composition_support_floor: int = 3
    pairwise_composition_max_virtual: int = 4

    # Recursive coordinate composition: certified carriers become virtual raw
    # axes for a bounded beam. Depth 3 permits two recursive composition steps.
    recursive_composition: bool = False
    recursive_composition_max_depth: int = 3
    recursive_composition_beam_width: int = 2
    noise_calibrated_heldout_factor: float = 3.0
    noise_calibrated_bootstrap: int = 8
    noise_calibrated_bootstrap_angle_tol: float = 0.10
    noise_calibrated_snap_factor: float = 3.0
    noise_calibrated_closure_tol: float = 3.0e-2

    # Jet-level families.
    jet_enable: bool = True
    jet_separability: bool = True
    jet_multiplicative: bool = True
    jet_record_augment: bool = True
    jet_residual_tol: float = 0.03

    # DE diagnostics and explicit structural-prior families.
    # Deprecated legacy hard-tail alias. New code should normalize this into
    # neutral DE prior flags before constructing GS libraries.
    de_templates: bool = False
    de_radial_templates: bool = True
    de_velocity_templates: bool = False
    de_all_upgrades: bool = False
    # Coupled feature-linear point-symmetry solve.  When GS is active the
    # automatic policy compares affine and bounded-quadratic solves unless
    # explicitly disabled; the legacy explicit flag still forces this lane.
    de_determining_equations: bool = False
    de_auto_nonlinear: bool = True
    de_auto_fss: bool = True
    de_auto_fss_max_attempts: int = 1
    de_auto_fss_n_iter: int = 1500
    de_auto_fss_n_fit: int = 1024
    de_auto_fss_n_probe: int = 1024
    de_auto_fss_max_depth: int = 4
    de_auto_fss_return_topk: int = 8
    de_contact_templates: bool = False  # velocity-monomial structural priors
    de_noether_templates: bool = False  # autonomous/even-velocity structural priors
    de_discrete_symmetry_templates: bool = False  # parity structural priors
    de_weighted_scaling_templates: bool = False
    de_radial_reduction_templates: bool = False
    de_invariant_library: bool = False
    de_invariant_max_terms: int = 64
    de_invariant_seed_generators: Any = ("d_x", "u_d_u", "x_d_x")
    de_upgrade_max_terms: int = 64
    de_determining_max_degree: int = 2
    de_determining_max_generators: int = 4
    de_determining_multiplier_degree: int = 2
    de_determining_bootstraps: int = 8
    de_determining_sparse_rotation: bool = True
    de_determining_bracket_certificate: bool = True
    # Compile useful low-complexity invariants/orbit coordinates from accepted
    # nonlinear generators.  This is separate from the named affine library.
    de_nonlinear_invariants: bool = False
    de_nonlinear_invariant_max_degree: int = 3
    de_nonlinear_invariant_max_candidates: int = 8
    de_nonlinear_invariant_tol: float = 0.03
    de_nonlinear_orbit_coordinate: bool = True
    de_compiled_nonlinear_invariants: Any = None
    de_compiled_orbit_coordinate: Any = None
    de_weighted_max_abs_x_power: int = 2
    de_weighted_max_u_power: int = 5
    de_weighted_max_du_power: int = 4
    de_weighted_tol: float = 1.0e-12

    # Unit-torus / Buckingham-pi dimensional GS families. These are disabled
    # unless explicitly requested; when enabled without a stronger policy they
    # default to audit behavior through ``dim_policy``.
    unit_torus: bool = False
    pi_invariants: bool = False
    dim_policy: str = "audit"  # baseline | audit | augment | both | replace-rref | gs-only
    dim_both_rule: str = "rref-dominates"  # rref-dominates | require-both | either | gs-dominates
    dim_validator: str = "nullspace"  # local | nullspace | linear
    dim_keep_local_gates: bool = True
    pi_max_exponent: int = 3
    pi_max_l1: int = 6
    pi_max_proposals: int = 24
    pi_max_basis: int = 8
    pi_rational_denom: int = 1
    pi_score_min_confidence: float = 0.65
    pi_include_free_consts: bool = True
    unit_report_json: str | None = None
    unit_report_md: str | None = None
    report_dim_disagreements: bool = True
    report_pi_rejected: bool = False

    # Numeric thresholds.
    max_batches: int = 4
    max_pair_generators: int = 16
    residual_tol: float = 0.03
    audit_residual_tol: float = 0.10
    min_confidence: float = 0.65
    min_grad_fraction: float = 0.05
    equivariance_min_r2: float = 0.985
    snap_tol: float = 0.20

    # Proposal throttling. The bank may retain more carriers than Stage A is
    # allowed to train; the decisive trial counts inside the Stage-A budget.
    max_stagea_proposals: int = 12
    stagea_proposal_budget: int = 6
    prefer_low_complexity: bool = True
    decisive_stagea_min_confidence: float = 0.995
    decisive_stagea_max_trials: int = 1

    # Reporting.
    report_rejected: bool = True
    report_top_k_rejected: int = 40

    # Optional AST canonicalisation for GS proposals and DE rows. Off by default.
    ast_simplify: bool = False
    ast_simplify_level: str = "safe"
    ast_simplify_domain_policy: str = "strict"
    ast_simplify_max_passes: int = 12
    ast_simplify_trace: bool = False

    def __post_init__(self) -> None:
        # Keep legacy and V3 names synchronized when configs are built manually.
        try:
            self.known_lie = bool(self.known_lie) and bool(self.known_generators)
            self.known_generators = bool(self.known_generators) and bool(self.known_lie)
        except Exception:
            pass
        if self.canonical_policy() == "gs-only-affine":
            self.general_affine = True
            self.affine_dense = True
            self.known_generators = False
            self.known_lie = False
        if bool(getattr(self, "de_all_upgrades", False)):
            self.de_determining_equations = True
            self.de_contact_templates = True
            self.de_noether_templates = True
            self.de_discrete_symmetry_templates = True
            self.de_weighted_scaling_templates = True
            self.de_radial_reduction_templates = True
            self.de_invariant_library = True
            self.de_nonlinear_invariants = True

    def canonical_policy(self) -> str:
        try:
            from .policy import canonical_gs_policy

            p = canonical_gs_policy(self.policy)
        except Exception:
            p = str(self.policy or "augment").strip().lower().replace("_", "-")
        if p not in {"augment", "replace-shadowed", "gs-only-affine"}:
            p = "augment"
        return p

    def active(self) -> bool:
        return bool(self.enabled) and str(self.mode or "propose").lower() != "off"

    def proposing(self) -> bool:
        return self.active() and str(self.mode or "propose").lower() in {"propose", "auto"}

    def automatic(self) -> bool:
        return self.active() and str(self.mode or "propose").lower() == "auto"

    def replaces_shadowed(self) -> bool:
        return self.active() and self.canonical_policy() in {"replace-shadowed", "gs-only-affine"}

    def replacement_policy(self) -> bool:
        return self.replaces_shadowed()

    def replace_separability_with_jet(self) -> bool:
        return self.active() and self.canonical_policy() == "replace-shadowed" and bool(self.jet_enable)

    def replace_affine_shadowed(self) -> bool:
        return self.replaces_shadowed()

    def affine_only_policy(self) -> bool:
        return self.active() and self.canonical_policy() == "gs-only-affine"

    def known_active(self) -> bool:
        return self.active() and bool(self.known_generators) and bool(self.known_lie)

    def general_affine_active(self) -> bool:
        return self.active() and (bool(self.general_affine) or bool(self.affine_dense))

    def noise_calibrated_promotion_active(self) -> bool:
        return self.general_affine_active() and bool(self.general_affine_promotion_noise_calibrated)

    def pairwise_composition_active(self) -> bool:
        return self.active() and bool(self.pairwise_composition)

    def recursive_composition_active(self) -> bool:
        return self.pairwise_composition_active() and bool(self.recursive_composition)

    def general_affine_chart_names(self) -> tuple:
        """Canonical, deduplicated chart names with identity always first."""

        raw = getattr(self, "general_affine_charts", ("identity",)) or ("identity",)
        if isinstance(raw, str):
            raw = raw.split(",")
        names: list[str] = []
        for item in raw:
            name = str(item or "").strip().lower()
            if name in {"identity", "log", "reciprocal", "warp"} and name not in names:
                names.append(name)
        if "identity" not in names:
            names.insert(0, "identity")
        names.sort(key=lambda n: 0 if n == "identity" else 1)
        return tuple(names)

    def canonical_dim_policy(self) -> str:
        p = str(getattr(self, "dim_policy", "audit") or "audit").strip().lower().replace("_", "-")
        aliases = {
            "rref": "baseline",
            "baseline-only": "baseline",
            "report": "audit",
            "gs": "gs-only",
            "gsonly": "gs-only",
            "replace": "replace-rref",
        }
        p = aliases.get(p, p)
        if p not in {"baseline", "audit", "augment", "both", "replace-rref", "gs-only"}:
            p = "audit"
        return p

    def canonical_dim_both_rule(self) -> str:
        r = str(getattr(self, "dim_both_rule", "rref-dominates") or "rref-dominates").strip().lower().replace("_", "-")
        if r in {"baseline-dominates", "baseline-wins"}:
            r = "rref-dominates"
        if r in {"gs-wins", "gs-overrides"}:
            r = "gs-dominates"
        if r not in {"rref-dominates", "require-both", "either", "gs-dominates"}:
            r = "rref-dominates"
        return r

    def canonical_dim_validator(self) -> str:
        v = str(getattr(self, "dim_validator", "nullspace") or "nullspace").strip().lower().replace("_", "-")
        if v not in {"local", "nullspace", "linear"}:
            v = "nullspace"
        return v

    def unit_torus_active(self) -> bool:
        return self.active() and bool(getattr(self, "unit_torus", False)) and self.canonical_dim_policy() != "baseline"

    def pi_invariants_active(self) -> bool:
        return self.unit_torus_active() and bool(getattr(self, "pi_invariants", False))

    def dim_policy_proposes(self) -> bool:
        return self.unit_torus_active() and self.proposing() and self.canonical_dim_policy() in {"augment", "both", "replace-rref", "gs-only"}
