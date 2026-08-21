# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.
"""Functional equivalence classes and a genuine description length in bits.

Representation complexity should not multiply scientific hypotheses.  When two
expression trees denote the same supported function they are two spellings of
one hypothesis, not two competing models, and letting both onto the front makes
the front representation-dependent: a printer emitting more decimal digits
would manufacture a new "model".

So the pipeline quotients before it compares.  Three questions are kept apart:

1. what distinctions the data support        -> equivalence classes
2. what trade-offs define the front          -> the complexity vector, per class
3. what to call one supported function       -> minimum description length

Only the third uses a scalar code length, which is why the exchange rate it
implies is a naming convention rather than a scientific commitment.

**Equivalence is layered, and the layers must not be conflated.**

``exact``
    An algebraic identity a canonicaliser can prove, such as ``x0*x1`` and
    ``x1*x0``.  Merged before the audit; no tolerance involved.
``functional_certified``
    Predictions agree across the declared domain to within a predeclared
    tolerance.  This is where snapped constants land, because
    ``0.3989422804014326`` is *not* ``sqrt(2)/(2*sqrt(pi))``: they differ
    around the seventeenth digit and no canonicaliser will prove them
    identical.  Calling this "exact after snapping" would overstate it.

Membership uses **direct prediction discrepancy**, never equality of risk.
Two badly wrong models can carry identical risk against ``y`` while predicting
quite different functions, so risk noninferiority is a strictly weaker relation
and cannot be used to merge.

Pairwise "not distinguishable" is also not transitive, so classes are built by
**complete link**: every pair inside a class must satisfy the criterion, not
merely every pair along some chain.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "GRAMMAR_VERSION",
    "FunctionalClass",
    "description_length_bits",
    "prediction_discrepancy",
    "certified_equivalence_classes",
    "exact_equivalence_classes",
    "derive_delta_function",
    "CompressionCertificate",
    "compression_certificate",
]

# Version the grammar: a code length is meaningless without the code that
# produced it, and a changed alphabet changes every reported number.
GRAMMAR_VERSION = "nestynet-sr-mdl-v1"

# Declared symbol weights.  Code length is -log2(weight / total), which is a
# Shannon code and therefore satisfies Kraft.  Weights encode the prior that a
# physical law is more likely to multiply than to take an arctangent; they are
# a declared convention, not a fit, and they are versioned with the grammar.
_SYMBOL_WEIGHTS: dict[str, float] = {
    "Symbol": 32.0,      # a variable
    "Mul": 24.0,
    "Add": 20.0,
    "Pow": 12.0,
    "Integer": 12.0,
    "Rational": 6.0,
    "Float": 6.0,
    "Pi": 4.0,
    "Exp1": 2.0,
    "exp": 6.0,
    "log": 4.0,
    "sin": 4.0,
    "cos": 4.0,
    "tan": 1.5,
    "asin": 1.0,
    "acos": 1.0,
    "atan": 1.5,
    "sinh": 0.8,
    "cosh": 0.8,
    "tanh": 1.5,
    "sqrt": 6.0,
    "Abs": 1.5,
    "sign": 0.5,
    "erf": 0.5,
    "_other": 0.5,       # anything not declared above
}
_WEIGHT_TOTAL = float(sum(_SYMBOL_WEIGHTS.values()))


def _symbol_bits(name: str) -> float:
    weight = _SYMBOL_WEIGHTS.get(name, _SYMBOL_WEIGHTS["_other"])
    return -math.log2(weight / _WEIGHT_TOTAL)


def _elias_gamma_bits(n: int) -> float:
    """Self-delimiting code length for a positive integer.

    Needed because a literal must announce its own size; without a
    self-delimiting code the decoder cannot know where a number ends, and the
    "length" would not be a real code length at all.
    """
    value = abs(int(n))
    if value <= 0:
        return 1.0
    return 2.0 * math.floor(math.log2(value)) + 1.0


def _integer_bits(n: int) -> float:
    return 1.0 + _elias_gamma_bits(max(1, abs(int(n))))  # sign + magnitude


def _float_bits(value: float) -> float:
    """Code length for a floating literal.

    A sixteen-digit literal must not cost the same as the token ``2``.  The
    literal pays for its sign, the number of significant digits (itself coded
    self-delimitingly), the digits, and the decimal exponent.  Sixteen digits
    is roughly ``16 * log2(10) = 53`` bits of significand, which is what makes
    an arbitrary fitted decimal expensive next to a short derivation from a
    small alphabet.
    """
    if value == 0.0 or not math.isfinite(value):
        return 2.0
    magnitude = abs(float(value))
    # Significant digits needed to round-trip this value.
    digits = 17
    for candidate in range(1, 18):
        if float(f"%.{candidate}g" % magnitude) == magnitude:
            digits = candidate
            break
    exponent = int(math.floor(math.log10(magnitude))) if magnitude > 0 else 0
    return (
        1.0                                   # sign
        + _elias_gamma_bits(digits)           # how many digits follow
        + digits * math.log2(10.0)            # the digits themselves
        + _integer_bits(exponent if exponent else 1)   # decimal exponent
    )


def description_length_bits(expr: Any, *, n_free_parameters: int = 0) -> float:
    """Return a prefix-code length in bits for a SymPy expression.

    The point of using genuine bits rather than another tuned score is that the
    *sign* of the comparison is then guaranteed by information content instead
    of by a penalty chosen to prefer physics-looking constants.  Sixteen
    arbitrary digits really do carry more information than a short derivation
    from a tiny alphabet, and the code says so without being asked to.
    """
    try:
        import sympy as sp
    except Exception:
        return float("inf")

    try:
        total = 0.0
        for node in sp.preorder_traversal(expr):
            if isinstance(node, sp.Symbol):
                total += _symbol_bits("Symbol")
            elif isinstance(node, sp.Integer):
                total += _symbol_bits("Integer") + _integer_bits(int(node))
            elif isinstance(node, sp.Rational):
                total += (
                    _symbol_bits("Rational")
                    + _integer_bits(int(node.p))
                    + _integer_bits(int(node.q))
                )
            elif isinstance(node, sp.Float):
                total += _symbol_bits("Float") + _float_bits(float(node))
            elif node is sp.pi:
                total += _symbol_bits("Pi")
            elif node is sp.E:
                total += _symbol_bits("Exp1")
            else:
                total += _symbol_bits(type(node).__name__)

        # A fitted parameter costs its announcement plus the resolution the
        # data actually pin it to; naming a class only needs the announcement
        # to be non-zero, so this stays deliberately coarse.
        total += float(max(0, int(n_free_parameters))) * 16.0
        return float(total)
    except Exception:
        return float("inf")


# --------------------------------------------------------------------------- #
# functional equivalence                                                      #
# --------------------------------------------------------------------------- #
def prediction_discrepancy(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    scale: float,
    unit_index: np.ndarray,
    n_units: int,
) -> tuple[float, float]:
    """Return per-unit mean discrepancy and its standard error.

    ``R_delta`` compares the two functions to each other, not their losses
    against ``y``.  Equal risk is not equal function: two candidates can be
    wrong in different directions by the same amount and would merge under a
    risk criterion while denoting different laws.
    """
    a = np.asarray(pred_a, dtype=np.float64).reshape(-1)
    b = np.asarray(pred_b, dtype=np.float64).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    if not np.any(valid):
        return float("inf"), 0.0

    squared = np.zeros(a.shape, dtype=np.float64)
    squared[valid] = ((a[valid] - b[valid]) / float(scale)) ** 2
    squared[~valid] = np.inf

    per_unit = np.zeros(int(n_units), dtype=np.float64)
    counts = np.zeros(int(n_units), dtype=np.float64)
    np.add.at(per_unit, unit_index, squared)
    np.add.at(counts, unit_index, 1.0)
    per_unit = per_unit / np.maximum(counts, 1.0)

    finite = per_unit[np.isfinite(per_unit)]
    if finite.size == 0:
        return float("inf"), 0.0
    mean = float(finite.mean())
    se = float(finite.std(ddof=1) / math.sqrt(finite.size)) if finite.size > 1 else 0.0
    return mean, se


@dataclass(frozen=True)
class FunctionalClass:
    """One supported function, with every spelling that denotes it."""

    class_id: str
    member_ids: tuple[str, ...]
    representative_id: str
    equivalence_kind: str
    tolerance: float
    code_length_bits: dict[str, float] = field(default_factory=dict)
    max_pairwise_discrepancy: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "members": list(self.member_ids),
            "representative": self.representative_id,
            "equivalence_kind": self.equivalence_kind,
            "tolerance": float(self.tolerance),
            "code_length_bits": {k: float(v) for k, v in self.code_length_bits.items()},
            "max_pairwise_discrepancy": float(self.max_pairwise_discrepancy),
            "grammar_version": GRAMMAR_VERSION,
        }


def exact_equivalence_classes(
    candidate_ids: Sequence[str],
    canonical_keys: Mapping[str, Any],
    code_bits: Mapping[str, float],
) -> list[FunctionalClass]:
    """Group candidates whose frozen expressions canonicalise identically.

    This is the ``exact`` layer of the module docstring: membership requires a
    canonicaliser to have produced the same canonical form from the frozen
    expression text alone, so the partition is fixed before any audit datum is
    seen and multiplicity may be counted over it.  A candidate whose
    expression cannot be canonicalised (``canonical_keys[cid]`` is ``None``)
    stays a singleton, which is the conservative direction: opacity can only
    enlarge the comparison family, never shrink it.
    """
    groups: dict[str, list[str]] = {}
    for cid in (str(c) for c in candidate_ids):
        key = canonical_keys.get(cid)
        group_key = f"key::{key}" if key is not None else f"opaque::{cid}"
        groups.setdefault(group_key, []).append(cid)

    out: list[FunctionalClass] = []
    for index, members in enumerate(sorted(sorted(m) for m in groups.values())):
        representative = min(
            members, key=lambda c: (float(code_bits.get(c, math.inf)), c)
        )
        out.append(
            FunctionalClass(
                class_id=f"fc{index:03d}",
                member_ids=tuple(members),
                representative_id=representative,
                equivalence_kind=(
                    "exact_algebraic" if len(members) > 1 else "singleton"
                ),
                tolerance=0.0,
                code_length_bits={
                    c: float(code_bits.get(c, math.inf)) for c in members
                },
                max_pairwise_discrepancy=0.0,
            )
        )
    return out


def certified_equivalence_classes(
    candidate_ids: Sequence[str],
    discrepancy: Mapping[tuple[str, str], float],
    code_bits: Mapping[str, float],
    *,
    delta_function: float,
    equivalence_kind: str = "functional_equivalence_certified_to_tolerance",
) -> list[FunctionalClass]:
    """Group candidates by complete-link certified functional equivalence.

    Complete link, not single link.  "Indistinguishable" is not transitive, so
    chaining ``a~b`` and ``b~c`` into one class can put ``a`` and ``c``
    together when they are demonstrably different functions.  Every pair inside
    a returned class satisfies the criterion.

    The representative is the minimum-description-length member.  That is a
    naming decision: it does not assert the representative is better supported,
    only that it is the shortest name for something the data cannot tell apart.
    """
    ids = [str(c) for c in candidate_ids]

    def pair(a: str, b: str) -> float:
        if a == b:
            return 0.0
        return float(discrepancy.get((a, b), discrepancy.get((b, a), float("inf"))))

    clusters: list[list[str]] = [[c] for c in ids]
    while True:
        best = None
        best_link = float("inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                link = max(pair(a, b) for a in clusters[i] for b in clusters[j])
                if link < best_link:
                    best_link, best = link, (i, j)
        if best is None or best_link > float(delta_function):
            break
        i, j = best
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]

    out: list[FunctionalClass] = []
    for index, members in enumerate(sorted(clusters, key=lambda m: sorted(m))):
        ordered = sorted(members)
        representative = min(
            ordered, key=lambda c: (float(code_bits.get(c, math.inf)), c)
        )
        worst = 0.0
        for a in ordered:
            for b in ordered:
                if a < b:
                    worst = max(worst, pair(a, b))
        out.append(
            FunctionalClass(
                class_id=f"fc{index:03d}",
                member_ids=tuple(ordered),
                representative_id=representative,
                equivalence_kind=(
                    equivalence_kind if len(ordered) > 1 else "singleton"
                ),
                tolerance=float(delta_function),
                code_length_bits={c: float(code_bits.get(c, math.inf)) for c in ordered},
                max_pairwise_discrepancy=float(worst),
            )
        )
    return out

# --------------------------------------------------------------------------- #
# the payoff: an absolute compression test                                    #
# --------------------------------------------------------------------------- #
def derive_delta_function(
    *, n_rows: int, code_gap_bits: float = 1.0, floor: float = 1.0e-12
) -> float:
    """Derive the functional-equivalence tolerance instead of declaring one.

    Naming a class by its shortest code is exact only for *exact* equivalents.
    For tolerance-certified members the criterion is really

        min over members of   [ L(D|M) + L(M) ]   in bits,

    and it collapses to "shortest code wins" only when the data-code term
    cannot reorder the members.  For two near-optimal models the perturbation
    is roughly ``N * sqrt(R_delta) / ln 2`` bits, so requiring it to stay below
    the smallest code gap worth acting on gives

        delta_function  <=  (code_gap_bits * ln2 / N)^2 .

    A hardcoded tolerance silently becomes wrong as the audit grows: the bound
    tightens like ``1/N^2``.  ``code_gap_bits = 1`` is the conservative choice,
    since one bit is the smallest difference in description length that could
    change which spelling is shortest.
    """
    rows = max(1, int(n_rows))
    bound = (float(code_gap_bits) * math.log(2.0) / float(rows)) ** 2
    return float(max(bound, float(floor)))


@dataclass(frozen=True)
class CompressionCertificate:
    """Whether a law compresses the data at all, in absolute bits.

    This test exists only because sigma is declared externally and never
    fitted.  With an external sigma the audit chi-square is an absolute
    quantity, so ``chi2 / (2 ln 2)`` is a real number of bits and the total
    message length can be compared against saying nothing.  If sigma were
    estimated from the residuals, chi-square would sit near N by construction
    for any flexible model and the comparison would be vacuous: every model
    would appear to "explain" the data equally well.

    ``bits_saved > 0`` means the law is a shorter description of the observed
    data than the best constant, counting the cost of stating the law itself.
    That is a statement about the world, not a ranking against other
    candidates.
    """

    model_expression: str
    model_code_bits: float
    data_code_bits: float
    total_bits: float
    null_model_code_bits: float
    null_data_code_bits: float
    null_total_bits: float
    bits_saved: float
    compression_ratio: float
    n_rows: int
    sigma_source: str

    @property
    def sigma_is_declared(self) -> bool:
        """Whether the scale came from outside the fit.

        The absolute reading of these bits depends on it.  A data-derived
        scale still supports the comparison against the constant model, but
        not the claim that the law compresses the *observations* relative to
        their measurement precision.
        """
        return "declared" in str(self.sigma_source).lower()

    @property
    def compresses(self) -> bool:
        return self.bits_saved > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_expression": self.model_expression,
            "model_code_bits": float(self.model_code_bits),
            "data_code_bits": float(self.data_code_bits),
            "total_bits": float(self.total_bits),
            "null_model_code_bits": float(self.null_model_code_bits),
            "null_data_code_bits": float(self.null_data_code_bits),
            "null_total_bits": float(self.null_total_bits),
            "bits_saved": float(self.bits_saved),
            "compression_ratio": float(self.compression_ratio),
            "compresses": bool(self.compresses),
            "n_rows": int(self.n_rows),
            "sigma_source": self.sigma_source,
            "absolute_interpretation_valid": bool(self.sigma_is_declared),
            "caveat": (
                None if self.sigma_is_declared else
                "sigma was not declared externally for this run; the scale is "
                "derived from the data (target RMS), so bits are measured "
                "against the signal scale rather than against measurement "
                "precision. The comparison against the constant model remains "
                "valid, but the absolute 'does this law compress the "
                "observations' reading requires a declared sigma."
            ),
            "grammar_version": GRAMMAR_VERSION,
            "interpretation": (
                "Total message length for the audit data under this law versus "
                "under the best constant, both in bits and both including the "
                "cost of stating the model. Positive bits_saved means the law "
                "genuinely compresses the observations. This is an absolute "
                "test and is only meaningful because sigma is declared "
                "externally rather than fitted."
            ),
        }


def compression_certificate(
    *,
    model_expression: str,
    model_code_bits: float,
    total_standardized_loss: float,
    null_total_standardized_loss: float,
    n_rows: int,
    sigma_source: str,
    null_model_code_bits: float = 0.0,
) -> CompressionCertificate:
    """Build the absolute compression statement.

    ``total_standardized_loss`` is ``sum_i (y_i - f_i)^2 / sigma_eff_i^2``
    (plus the log-determinant term when input errors are declared, which is
    what makes it a genuine negative log likelihood rather than a profiled
    quadratic).  The additive ``(N/2) log(2 pi sigma^2)`` term is common to
    both branches and cancels, so it is omitted.
    """
    ln2 = math.log(2.0)
    data_bits = 0.5 * float(total_standardized_loss) / ln2
    null_data_bits = 0.5 * float(null_total_standardized_loss) / ln2
    total = data_bits + float(model_code_bits)
    null_total = null_data_bits + float(null_model_code_bits)
    saved = null_total - total
    ratio = (null_total / total) if total > 0 else float("inf")
    return CompressionCertificate(
        model_expression=str(model_expression),
        model_code_bits=float(model_code_bits),
        data_code_bits=float(data_bits),
        total_bits=float(total),
        null_model_code_bits=float(null_model_code_bits),
        null_data_code_bits=float(null_data_bits),
        null_total_bits=float(null_total),
        bits_saved=float(saved),
        compression_ratio=float(ratio),
        n_rows=int(n_rows),
        sigma_source=str(sigma_source),
    )
