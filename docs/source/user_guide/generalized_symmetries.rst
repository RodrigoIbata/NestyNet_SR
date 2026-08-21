Generalized Symmetries
======================

NestyNet_SR includes a generalized-symmetry (GS) layer that probes trained
surrogates for continuous symmetries (translations, scalings, rotations,
learned affine actions, dimensional tori) and converts certified witnesses
into coordinate proposals, library priors, and diagnostics.

The GS layer has two hosts with different defaults:

* **SR side** (``nestynet-sr`` / ``nestynet_sr/run_SR.py``): the GS Stage-A
  layer is **enabled by default**.  Ordinary SR runs audit known generators,
  run the chart-based determining operator, and may promote GS coordinate
  proposals into Stage A without any extra flags.
* **DE side** (``nestynet-de`` / ``nestynet_sr/run_de.py``): the GS layer is
  **opt-in**.  Without ``--gs-enable`` (or its alias ``--gs-auto``), DE runs
  use the baseline library and record no GS diagnostics.

The main GS families are:

* known point/Lie generators: translations, diagonal translations, scalings,
  rotations, optional Lorentz-boost probes, and output equivariance
  diagnostics;
* learned sparse affine generators, enabled with ``--gs-general-affine``;
* chart-based determining operators (identity, log, reciprocal, and a
  discovered per-axis warp chart) that expose monomial, reciprocal-sum, and
  generalized-additive coordinates;
* pairwise scaling-witness composition and recursive coordinate composition
  through a shared carrier bank;
* jet-level separability witnesses;
* neutral DE hard-tail library priors for radial/singular terms and optional
  velocity-dependent terms;
* finite point-Lie prolongation scoring for scalar DE residuals;
* unit-torus/Buckingham-pi dimensional GS, requiring a valid units
  specification.

SR Side (Enabled By Default)
----------------------------

On the SR side the GS Stage-A layer is on by default: ``--gs-stagea``
defaults to true, ``--gs-mode`` defaults to ``propose``, and the known
generators, chart set ``identity,log,reciprocal``, noise-calibrated
promotion, pairwise composition, and recursive composition are all enabled
by default.  A plain run such as::

   nestynet-sr --filepath data/my_data.csv

therefore already includes GS Stage-A coordinate proposals.  To reduce or
disable GS involvement:

``--gs-mode audit``
   Keep GS diagnostics (witness probes, reports) but propose no coordinates.

``--gs-mode off``
   Deactivate the GS layer entirely; no GS proposals, diagnostics, or
   reports.

``--gs-no-stagea``
   Disable the affine/Lie-style Stage-A proposal path specifically.  Note
   that pairwise and recursive composition remain on unless also disabled
   with ``--gs-no-pairwise-composition`` and
   ``--gs-no-recursive-composition``; ``--gs-mode off`` is the single switch
   that deactivates everything.

The most important SR-side switches are:

``--gs-mode off|audit|propose|auto``
   Controls whether GS only records diagnostics or also proposes candidates.
   Default ``propose``.

``--gs-policy augment|replace-shadowed|gs-only-affine``
   Controls interaction with established baseline proposal families.
   Default ``augment``, which adds GS proposals alongside the baseline
   detectors.  ``replace-shadowed`` removes only baseline motifs that a GS
   family directly shadows: a *promoted* GS reduction suppresses a legacy
   monomial/linear compound proposal only when it covers the same variable
   support with a matching projective exponent ray (monomial) or covector
   direction (linear); legacy variants carrying extra hypotheses (prefactor
   peels, retained axes, ``extra_override``) are never suppressed.  Each
   suppression is recorded as a GS policy event in the run report.
   ``gs-only-affine`` is an aggressive affine-only ablation mode.

``--gs-charts identity,log,reciprocal,warp``
   Charts for the affine determining operator.  The default is
   ``identity,log,reciprocal``; pass ``identity`` alone to restore the raw
   coordinate solve.  The ``log`` chart re-runs the determining operator on
   ``u = log(x)`` with chain-ruled gradients, where scaling symmetries appear
   as translations and monomial invariants ``prod_i x_i**a_i`` fall out as
   linear invariant covectors, subsuming the legacy Stage-A monomial
   compound detector through the same certificate and promotion gates.  The
   log chart runs only when every sampled input column is strictly positive;
   the discovered exponent ray is snapped to primitive integers (rational
   snap bounded by ``--gs-chart-snap-denominator``, default 4) and
   revalidated against the determining residual before compilation.  The
   ``reciprocal`` chart (``u = 1/x``) exposes ``sum_i c_i/x_i`` invariants
   (the parallel-resistor / reduced-mass / lens family) and requires each
   input column nonzero and non-sign-crossing.  The optional ``warp`` chart
   *discovers* the per-axis warp rather than enumerating a fixed one: it
   certifies a generalized-additive symmetry
   ``f = g(sum_i c_i phi_i(x_i))`` via the pair-independent normalized
   Hessian, recovers the per-axis warps from pairwise log-gradient
   differences (snapping identity/log/reciprocal/square/power in one fit),
   and validates the recovered coordinate and its affine covector by a
   rank-1 test.  Unlike the fixed charts it needs the leaf Hessian (taken by
   autograd) and at least three variables, and it reaches mixed-power
   coordinates such as ``x0^2 + x1^3`` that no single fixed chart exposes.
   All charts emit through the same promoted-reduction contract, so
   cross-chart dedup, ``replace-shadowed`` suppression, and recursive
   composition apply unchanged.

``--gs-noise-calibrated-promotion`` (default on)
   Promotes GS reductions on surrogate-noise-relative evidence rather than
   the absolute residual tolerances that only oracle-exact gradients can
   meet: nullity is selected at the largest singular-spectrum gap of the
   determining solve (``--gs-nc-min-spectral-gap``, default 10), held-out
   residuals must be consistent with train residuals, bootstrap subspace
   stability is measured over 8 replicates, and exponent snapping may not
   degrade the determining residual by more than a fixed factor over the
   unsnapped baseline.  The defaults were calibrated on log-chart monomial
   fixtures: real symmetries promote at relative gradient noise up to ~1e-3
   while a no-symmetry control shows spectrum gaps below ~4; at ~1e-2 noise
   the spectrum blurs and the gate correctly declines.  Promoted proposals
   still pass through the full ordinary Stage-A validation (refit, units,
   CoE), which bounds the cost of a wrong promotion.  Pass
   ``--gs-no-noise-calibrated-promotion`` to fall back to the oracle-only
   absolute residual gates.

``--gs-pairwise-composition`` (default on)
   Compose accepted pairwise scaling witnesses into global ±1 monomial rays
   (products and ratios over three or more variables) by exact sign
   propagation over the pair-constraint graph, validated jointly against the
   sampled gradients.  The pairwise tests survive surrogate gradient noise
   (~1e-3 to 1e-2) that breaks the global determining solve, making this the
   noise-robust route to multi-variable compound products/ratios.  Composed
   proposals use the standard promoted-proposal contract, so
   ``replace-shadowed`` suppression and cross-route deduplication apply
   unchanged.

``--gs-recursive-composition`` (default on)
   After first-level GS promotes a coordinate (a monomial, linear form,
   radius, ...), treat it as a virtual axis and re-run the pairwise-witness
   composition on the reduced set of that coordinate plus the disjoint free
   raw axes.  This discovers *nested* coordinates such as
   ``(x0*x1/x2) - x3`` -- a promoted coordinate translated, rotated, or
   boosted against a free axis -- that neither the first-level charts nor a
   single composition step reach.  Requires ``--gs-pairwise-composition``.
   Depth is bounded by ``--gs-recursive-max-depth`` (default 3, i.e. up to
   two recursive steps) and ``--gs-recursive-beam-width`` (default 2 newly
   composed carriers per depth); pure-monomial products (which collapse to a
   first-level monomial) are not re-emitted.

``--gs-general-affine`` (default off)
   Enable learned pairwise affine-generator probes in addition to the named
   generator bank.

``--gs-lorentz-boosts`` (default off)
   Enable Lorentz-boost invariant probes.

``--gs-stagea-proposal-budget`` (default 6)
   Hard total cap on GS carrier trials per Stage-A atom, including the
   protected decisive trial (``--gs-decisive-min-confidence``, default
   0.995; ``--gs-decisive-max-trials``, default 1).

Tolerances are controlled by ``--gs-residual-tol`` (default 0.03),
``--gs-audit-residual-tol`` (default 0.10), ``--gs-min-confidence`` (default
0.65), and ``--gs-max-pair-generators`` (default 16).

DE Side (Opt-In)
----------------

On the DE side the GS layer is off unless requested::

   nestynet-de --filepath data/my_de_data.csv \
       --gs-enable \
       --gs-mode auto \
       --gs-policy replace-shadowed \
       --gs-general-affine \
       --gs-de-lie-prolongation

Note that the Feynman-DE benchmark wrapper
(``examples/feynman_de/run_benchmark.py``) does not register GS flags; GS DE
runs go through ``nestynet_sr/run_de.py`` (the ``nestynet-de`` entry point)
directly.

The most important DE-side switches are:

``--gs-enable``
   Enables the GS layer for DE discovery.  Without this flag, baseline DE
   behavior is preserved.  ``--gs-auto`` is an alias for
   ``--gs-enable --gs-mode auto`` and writes a GS DE report.

``--gs-mode off|audit|propose|auto`` and ``--gs-policy``
   Same semantics as the SR side; on the DE side ``replace-shadowed``
   replaces shadowed radial templates.  DE templates are controlled by the
   separate ``--de-hard-tail-*`` flags below.

``--gs-known-generators`` (default on) and ``--gs-general-affine``
   Named affine generator diagnostics and learned sparse affine generator
   diagnostics on fitted DE surrogates.  Individual named families can be
   disabled with ``--gs-no-translations``, ``--gs-no-diagonal-translations``,
   ``--gs-no-scalings``, and ``--gs-no-rotations``;
   ``--gs-lorentz-boosts`` (default off) adds hyperbolic generator probes.

``--de-hard-tail-templates`` and ``--de-hard-tail-velocity-templates``
   Add source-tagged neutral DE hard-tail prior rows.  The legacy
   ``--gs-de-templates`` spellings remain as deprecated aliases, but these
   priors are not evidence of a discovered symmetry.

``--gs-de-lie-prolongation``
   Scores sparse DE candidates by evaluating finite point-Lie prolongation
   residuals against the discovered equation.  This is audit-only by
   default; it affects model selection only with explicit
   ``--gs-de-lie-use-for-selection`` (selection penalty weight
   ``--gs-de-lie-prolongation-weight``, default 0.05).

``--gs-de-determining-equations``
   Enables the coupled degree-bounded point-symmetry nullspace solve for DE
   candidates (``--gs-de-determining-max-degree`` selects affine regression
   or the bounded quadratic lane).  With GS enabled,
   ``--gs-de-auto-nonlinear`` (default on) automatically compares affine and
   bounded-quadratic scalar-ODE determining solves, and ``--gs-de-auto-fss``
   (default on) launches one bounded factorized-symbolic-search challenger
   when automatic GS produces a certified nontrivial carrier.

``--gs-de-certificate``
   Attaches a point-symmetry determining certificate (on-shell recovery plus
   a relative-invariance test, including nullspace combinations) to selected
   DE candidates.

Dimensional GS Policies
-----------------------

Unit-torus GS treats physical dimensions as commuting scaling symmetries.  It
constructs exact generators from the problem's ``UnitsSpec`` and enumerates
bounded Buckingham-pi invariants or DE prefactors.  The unit-torus layer is
off by default on both the SR and DE sides; enable it with
``--gs-unit-torus`` (and ``--gs-pi-invariants`` for bounded pi proposals).

The dimensional policy is controlled by::

   --gs-dim-policy baseline|audit|augment|both|replace-rref|gs-only

``audit`` is the default and the recommended first mode.  It records
generators, pi invariants, and baseline/GS disagreements while preserving
baseline accept/reject decisions.  ``augment`` allows GS pi/prefactor
proposals while keeping baseline validation active.  ``both`` combines the
baseline and GS decisions using ``--gs-dim-both-rule``.  ``replace-rref`` and
``gs-only`` are experimental replacement modes and should be used only after
audit reports are coherent on known-good cases.

Unit-torus requires units.  If a run passes ``--ignore_units`` or a problem
has no canonical dimensional metadata, the unit-torus layer records a skipped
event and cannot contribute proposals.  A reproducible unit-torus audit on a
units-enabled DE case is::

   nestynet-de --filepath data/my_de_data.csv \
       --gs-enable --gs-unit-torus --gs-pi-invariants --gs-dim-policy audit

Chart Bridge: Symmetries Compiled Into Executable Charts
--------------------------------------------------------

``nestynet_sr/sr_gs/chart_bridge.py`` connects GS symmetry *discovery* to
chart *execution* in ``nestynet.charts``, in the dependency-correct
direction (NestyNet_SR imports nestynet, never the reverse).  The bridge
fits an identity-chart surrogate to a densely sampled record, feeds its
analytic values and gradients to the affine graph-symmetry determining
operator (which solves for joint input and output affine actions), and
compiles each discovered generator into an executable chart: a scaling-type
input action rectifies to a shifted-log warp ``u = log(t - t0)`` with the
fixed point ``t0`` read off the generator, while a pure translation means
the coordinate is already rectified.  Every compiled chart is gated by the
key-sharpness certificate: the profiled validation well of the chart
parameter must open sharply, or the proposal is rejected.  Because the
surrogate's gradients carry fit-level error, the bridge uses
noise-calibrated (spectral-gap) nullity detection rather than exact-gradient
tolerances.

Worked demonstrations live in ``examples/gs_charts/`` (see its ``README.md``):
a blind Sedov-Taylor blast-wave analysis that recovers the detonation
instant, similarity exponent, and yield from radius data alone, and a real
SN 1993J VLBI dataset where the machine locates the explosion epoch from
expansion data that begin months after the event.

Shared Recursive Carrier Bank
-----------------------------

``nestynet_sr/sr_gs/carrier_bank.py`` holds discovered GS inner coordinates
``z(x)`` in a consumer-neutral bank.  The GS layer discovers and recursively
composes analytic carriers; consumers then decide what those coordinates are
worth: Stage A must empirically prove that ``NN(z)`` simplifies the current
model, while the factorized symbolic search fits and validates an explicit
outer map ``g(z)``.  Keeping the bank independent of either consumer
prevents proposal evidence from becoming acceptance authority, and makes
recursive coordinates available to both paths with identical certificates
and provenance.  Recursive composition (see
``--gs-recursive-composition`` above) deposits nested carriers into the same
bank with their depth and support recorded.

GS Carrier Seeding For Factorized Symbolic Search
-------------------------------------------------

The factorized-symbolic-search (FSS) oracle benchmark can be seeded with GS
carriers via the opt-in ``--gs-carrier-seed`` flag (default off), available
on the FSS oracle lab
(``nestynet_sr/sr_search/factorized_search/oracle_lab.py``) and the AIF
closure benchmark
(``nestynet_sr/sr_search/factorized_search/aif_closure_benchmark.py``); the
paired FSS versus FSS+GS launcher is ``scripts/run_table5_fss_gs.py``.  The
seeding workflow, its dimensional handling of dimension-changing gauges, and
the oracle-gradient ablation caveat are documented in
``examples/oracle_factorized_search/README.md``; that document is the
reference and is not duplicated here.  On the DE side, an analogous
automatic carrier-seeded FSS challenger is controlled by
``--gs-de-auto-fss`` (see above).

Reporting
---------

Both entry points write generalized-symmetry reports:

* ``run_SR.py`` writes ``results/<stem>.gs_report.json`` and
  ``results/<stem>.gs_report.md`` whenever the GS layer is active -- which,
  given the SR-side defaults, includes ordinary runs.  With
  ``--gs-mode off`` no report is written.
* ``run_de.py`` writes ``<output_dir>/<case>.gs_report.json`` and
  ``<output_dir>/<case>.gs_report.md`` when ``--gs-enable`` is passed; the
  paths can be overridden with ``--gs-unit-report-json`` and
  ``--gs-unit-report-md``.

The reports record sample-compatible generator probes, promotion decisions,
policy events, DE source rows, prolongation metadata, and unit-torus events.

Benchmark And Paper Framing
---------------------------

For paper-facing evidence, use the deterministic smoke benchmark with its
derived claim-tier table::

   PYTHONPATH=. python examples/generalized_symmetries/gs_smoke_benchmark.py \
       --samples 1024 \
       --repeats 5 \
       --seed 20260621 \
       --output results/gs_geometry_smoke/gs_smoke_results.json \
       --csv results/gs_geometry_smoke/gs_noise_results.csv \
       --markdown results/gs_geometry_smoke/gs_paper_summary.md

The raw JSON contains ``sr_noise``, ``stagea_oblique``, and
``de_prolongation`` sections.  The ``paper_summary`` section is the
conservative table to read first.  It separates ``coordinate_discovery``,
``normal_form_discovery``, ``equation_level_certificate``,
``library_prior_requires_matched_control``, and ``negative_control`` rows.

For DE claims, use matched library controls.  In particular, compare
baseline runs against a neutral hard-tail/invariant-library vocabulary arm
before claiming that a GS-labelled row caused the improvement.  The helper
``scripts/plan_gs_matched_ablation.py`` emits the six-arm design used for
this separation.

Limitations
-----------

The current prolongation layer is a finite scorer over a known affine
point-generator bank plus optional sparse affine candidates.  It is not yet a
full solver for the Lie determining equations.

Vector and PDE jet spaces are represented in the GS jet scaffold, but
vector/PDE prolongation and determining-equation scoring intentionally fail
with explicit unsupported-scope diagnostics.

Lorentz boosts are implemented as point generators on ``(x, u)`` and are
prolonged through ``u_x`` and ``u_xx``.  The implementation does not inject a
relativistic gamma-factor primitive.

Unit-torus pi enumeration is bounded by exponent, support, L1 norm, and
proposal count.  These bounds keep experiments controlled, but they can also
hide useful large-support invariants if set too conservatively.
