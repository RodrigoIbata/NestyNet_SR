# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.
"""The calibration profile must never license what it has not measured.

The dangerous failure here is not a crash but a quiet one: licensing the
multiplier max-T outside the region where it was measured to control the
familywise rate, or silently falling back where the envelope simply has no
coverage.  Both look like the policy working.
"""
import pytest

from nestynet_sr.stat_selection.calibration_profile import (
    MAXT_PROFILE_V1,
    select_inference_method,
)


def test_profile_records_everything_that_changes_its_meaning():
    """A profile measured at another alpha or budget is a different profile."""
    profile = MAXT_PROFILE_V1
    assert profile.profile_version
    assert profile.alpha == 0.05
    assert profile.replicates_per_cell >= 2500, (
        "certifying a true-0.05 cell at a 0.06 bound needs about 2500 replicates"
    )
    assert profile.multiplier == "normal"
    assert profile.studentization_rule
    assert len(profile.cells) == 72


def test_measured_grid_is_monotone_in_both_coordinates():
    """The licensing rule relies on harder cells bounding easier ones.

    Rate must fall with units and rise with comparisons, else a 'harder'
    witness cell would not actually bound the query.  Monte Carlo noise is
    tolerated; a systematic reversal is not.
    """
    by_k: dict[int, list[tuple[int, float]]] = {}
    by_g: dict[int, list[tuple[int, float]]] = {}
    for cell in MAXT_PROFILE_V1.cells:
        by_k.setdefault(cell.k_admissible, []).append((cell.n_units, cell.false_edge_rate))
        by_g.setdefault(cell.n_units, []).append((cell.k_admissible, cell.false_edge_rate))

    for k, series in by_k.items():
        series.sort()
        rates = [r for _, r in series]
        assert rates[0] > rates[-1], f"K={k}: rate must fall with units"

    for g, series in by_g.items():
        series.sort()
        rates = [r for _, r in series]
        assert rates[-1] >= rates[0] - 0.02, f"G={g}: rate must not fall with comparisons"


def test_licensing_requires_a_harder_validated_witness():
    """A licensed query must be bounded by a measured, validated, harder cell."""
    decision = select_inference_method(n_units=400, k_pre=110)
    assert decision["decision"] == "licensed"
    assert decision["method"] == "multiplier_max_t"
    witness = decision["witness_cell"]
    assert witness["status"] == "validated"
    assert witness["n_units"] <= 400
    assert witness["k_admissible"] >= 110


def test_small_unit_counts_are_never_licensed():
    """The catastrophic corner must not be licensed under any archive size."""
    for k_pre in (12, 110, 1056, 10100):
        decision = select_inference_method(n_units=12, k_pre=k_pre)
        assert decision["decision"] != "licensed"
        assert decision["method"] == "bonferroni_t"


def test_comparisons_beyond_the_grid_are_reported_not_guessed():
    """An uncapped archive escapes the envelope and must say so.

    Returning a silent fallback here would be indistinguishable from a measured
    decision, hiding the fact that the envelope has no coverage at all.
    """
    decision = select_inference_method(n_units=20000, k_pre=1_047_552)
    assert decision["decision"] == "beyond_grid"
    assert decision["escaped_coordinate"] == "k_pre"
    assert decision["method"] == "bonferroni_t"
    assert "not extrapolated" in decision["reason"]


def test_units_alone_do_not_license_a_large_comparison_family():
    """The envelope is two-dimensional; units alone are not sufficient.

    G=400 is ample by the standards of the low-K rows and still does not carry
    a 9900-comparison family.  A one-dimensional 'warn below N units' rule
    would license this.
    """
    decision = select_inference_method(n_units=400, k_pre=9900)
    assert decision["decision"] == "fallback"
    assert decision["method"] == "bonferroni_t"


def test_capped_archive_at_sr_scale_units_is_licensed():
    """The configuration this benchmark actually runs must be licensed.

    A 100-candidate cap gives at most 9900 comparisons, and ordinary SR runs at
    thousands of audit rows.  The high-G extension exists to cover exactly this
    cell; without a validated witness at large K the run would silently take
    the conservative fallback and lose the power the max-T provides.
    """
    decision = select_inference_method(n_units=20000, k_pre=9900)
    assert decision["decision"] == "licensed"
    assert decision["method"] == "multiplier_max_t"
    witness = decision["witness_cell"]
    assert witness["k_admissible"] >= 9900
    assert witness["n_units"] <= 20000


def test_a_family_beyond_the_envelope_degrades_safely_not_silently():
    """Exceeding the measured envelope must cost power, never correctness.

    The archive cap is a compute bound, not a statistical guard.  What protects
    the inference is that an oversized comparison family is detected and routed
    to the conservative fallback with the escape named, so a large archive can
    never quietly obtain an unlicensed max-T.
    """
    from nestynet_sr.stat_selection.calibration_profile import MAXT_PROFILE_V1 as profile

    beyond = max(profile.grid_comparisons) + 1
    decision = select_inference_method(n_units=20000, k_pre=beyond)
    assert decision["decision"] == "beyond_grid"
    assert decision["method"] == "bonferroni_t"
    assert decision["escaped_coordinate"] == "k_pre"


def test_lookup_rejects_nonsense_arguments():
    with pytest.raises(ValueError):
        select_inference_method(n_units=0, k_pre=10)
    with pytest.raises(ValueError):
        select_inference_method(n_units=10, k_pre=-1)


def test_certificate_records_the_pre_audit_comparison_family():
    """The multiplicity burden must be keyed to the frozen family, not survivors.

    If K were taken from the estimable subset, a candidate that fails on the
    audit would shrink the recorded burden, making the method appear to have
    faced a smaller comparison family after the data were seen.  The estimable
    and non-estimable counts are recorded separately so the split is visible.
    """
    from nestynet_sr.stat_selection.sr_pipeline import _admissible_comparison_count
    from nestynet_sr.stat_selection.complexity import ComplexityVector

    def cx(value):
        return ComplexityVector(components=(("size", float(value)),))

    # Three candidates of equal complexity: every ordered pair is admissible.
    ids = ["a", "b", "c"]
    equal = {i: cx(10.0) for i in ids}
    assert _admissible_comparison_count(ids, equal) == 6

    # Strictly increasing complexity: only the simpler-challenger pairs count.
    ladder = {"a": cx(1.0), "b": cx(2.0), "c": cx(3.0)}
    assert _admissible_comparison_count(ids, ladder) == 3

    # Dropping a candidate shrinks the family, which is exactly why the
    # certificate must key on the pre-audit set rather than the survivors.
    assert _admissible_comparison_count(["a", "b"], equal) == 2


def test_snapped_variants_enter_the_archive_before_freezing():
    """Snapping is a hypothesis the audit tests, not a rendering choice.

    An awkward float may be an exact constant or may genuinely be that number.
    Asserting the snap while rendering makes the claim untestable.  Adding the
    snapped form as its own pre-freeze candidate lets both be audited on the
    same untouched units: if the snap is right the risks are indistinguishable
    and it wins on constant_code, and if it is wrong its risk is worse and the
    front keeps the float.
    """
    from dataclasses import dataclass, field
    from typing import Any

    from nestynet_sr.stat_selection.sr_pipeline import _with_snapped_variants

    @dataclass
    class _Art:
        candidate_id: str
        expr: str
        source: str
        label: str = ""
        complexity: float | None = None
        n_free_params: int = 0
        metadata: dict[str, Any] = field(default_factory=dict)

    # 0.3989422804014326 is 1/sqrt(2*pi); the snapped form must appear.
    original = _Art(candidate_id="c000", expr="0.3989422804014326*exp(-x0**2/2)",
                    source="stageB")
    out = _with_snapped_variants([original])
    assert len(out) > 1, "an awkward float must yield a snapped candidate"

    snapped = [a for a in out if a is not original]
    assert all(a.metadata.get("snapped_from") == original.expr for a in snapped)
    assert all(a.source.endswith("+snap") for a in snapped)
    # The original survives: snapping proposes, it does not replace.
    assert any(a.expr == original.expr for a in out)


def test_snapping_adds_nothing_when_constants_are_already_exact():
    """No spurious candidates, so the multiplicity burden is not inflated."""
    from dataclasses import dataclass, field
    from typing import Any

    from nestynet_sr.stat_selection.sr_pipeline import _with_snapped_variants

    @dataclass
    class _Art:
        candidate_id: str
        expr: str
        source: str
        label: str = ""
        complexity: float | None = None
        n_free_params: int = 0
        metadata: dict[str, Any] = field(default_factory=dict)

    for clean in ("x0", "2*x0", "x0*x1/(4*pi*x2*x3**2)"):
        out = _with_snapped_variants([_Art(candidate_id="c", expr=clean, source="s")])
        assert len(out) == 1, f"{clean} should not generate a snapped variant"


def test_constant_code_prefers_exact_constants_over_floats():
    """The complexity axis must see the difference structure alone cannot."""
    import sympy as sp

    from nestynet_sr.stat_selection.sr_pipeline import _constant_code_cost

    pairs = [
        ("1*x0", "1.000001*x0"),
        ("pi/2*x1**2", "1.5708*x1**2"),
        ("sqrt(2)*exp(-x0**2/2)/(2*sqrt(pi))", "0.3989422804014326*exp(-x0**2/2)"),
    ]
    for exact, approx in pairs:
        c_exact = _constant_code_cost(sp.sympify(exact))
        c_approx = _constant_code_cost(sp.sympify(approx))
        assert c_exact < c_approx, f"{exact} must cost less than {approx}"


def test_delta_function_is_derived_and_tightens_with_audit_size():
    """A fixed tolerance silently becomes wrong as the audit grows.

    Naming a class by its shortest code collapses out of the full criterion
    (data bits + model bits) only when the data term cannot reorder members by
    as much as one bit.  That bound scales as 1/n_rows^2.
    """
    from nestynet_sr.stat_selection.functional_classes import derive_delta_function

    small = derive_delta_function(n_rows=200)
    large = derive_delta_function(n_rows=20000)
    assert large < small
    assert large == pytest.approx(small / (100.0**2), rel=1e-6)


def test_compression_certificate_measures_against_saying_nothing():
    """The payoff: an absolute statement, not a ranking against rivals."""
    from nestynet_sr.stat_selection.functional_classes import compression_certificate

    good = compression_certificate(
        model_expression="x0*x1", model_code_bits=40.0,
        total_standardized_loss=10.0, null_total_standardized_loss=20000.0,
        n_rows=20000, sigma_source="declared_search_noise_sigma_y",
    )
    assert good.compresses
    assert good.bits_saved > 0
    assert good.sigma_is_declared
    assert good.to_dict()["absolute_interpretation_valid"] is True
    assert good.to_dict()["caveat"] is None

    # A law more expensive to state than the structure it explains must not
    # claim compression.
    bad = compression_certificate(
        model_expression="a very long fitted expression", model_code_bits=1.0e6,
        total_standardized_loss=10.0, null_total_standardized_loss=20000.0,
        n_rows=20000, sigma_source="declared_search_noise_sigma_y",
    )
    assert not bad.compresses


def test_compression_flags_a_data_derived_scale():
    """Without a declared sigma the absolute reading must be withdrawn.

    If sigma were estimated from residuals, chi-square sits near N for any
    flexible model and 'does this law compress the observations' is vacuous.
    The certificate has to say so rather than report the number bare.
    """
    from nestynet_sr.stat_selection.functional_classes import compression_certificate

    cert = compression_certificate(
        model_expression="x0", model_code_bits=10.0,
        total_standardized_loss=1.0, null_total_standardized_loss=100.0,
        n_rows=1000, sigma_source="search_target_rms",
    )
    payload = cert.to_dict()
    assert cert.compresses                       # the comparison still holds
    assert payload["absolute_interpretation_valid"] is False
    assert "not declared externally" in payload["caveat"]


def test_audit_views_are_byte_exact_slices_of_the_source(tmp_path):
    """The search must see the original numbers, not a re-serialisation.

    pandas' default C float parser is fast but not correctly rounded and can be
    off by one ULP.  That is invisible against measurement noise and fatal on a
    noiseless benchmark: the ordinary path reaches a Stage-B loss near 1e-30, so
    a 1e-15 perturbation of the inputs is far larger than the residual being
    minimised, and the separability structure that depends on exactness stops
    being detected.
    """
    import csv

    import numpy as np

    from nestynet_sr.stat_selection.sr_pipeline import prepare_sr_audit_plan

    # Values whose shortest repr stresses the fast parser.
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(200, 3)) * 18.643483026409722
    source = tmp_path / "src.csv"
    with open(source, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["y", "x0", "x1"])
        for r in rows:
            w.writerow([repr(float(v)) for v in r])

    plan = prepare_sr_audit_plan(
        source_path=source, results_dir=tmp_path / "out", audit_fraction=0.2,
    )

    def exact(path, skip_header=True):
        with open(path) as fh:
            r = csv.reader(fh)
            if skip_header:
                next(r)
            return [[float(v) for v in row] for row in r]

    original = exact(source)
    search = exact(plan.search_path)
    audit = exact(plan.audit_path)

    assert search + audit == original, (
        "search+audit views must reconstruct the source rows bit-for-bit"
    )
    # And the raw text must match too, which is what makes the SHA-256
    # provenance mean the data the search actually saw.
    src_lines = open(source).read().splitlines()[1:]
    view_lines = (open(plan.search_path).read().splitlines()[1:]
                  + open(plan.audit_path).read().splitlines()[1:])
    assert view_lines == src_lines
