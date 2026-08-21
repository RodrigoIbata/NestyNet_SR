# Archive-conditional statistical model selection

The symbolic and differential-equation search engines are deliberately
adaptive: they use validation losses, structural priors, noise heuristics, and
physics constraints to decide where to spend computation.  The statistical
audit is a separate stage.  It begins only after the union of discovered
candidates has been canonicalised and frozen.

The foundation under `nestynet_sr.stat_selection` is shared by ordinary SR and
DE/PDE discovery.  Installment 2 connects ordinary SR to it while leaving the
DE/PDE adapter for installment 3.

## Contract

1. Build a `CandidateArchive` while search is active.
2. Freeze it before inspecting audit losses.
3. Define the independent unit correctly.  IID rows may be units for ordinary
   SR; a whole trajectory, experiment, excitation, or field realisation is
   normally the unit for DE/PDE discovery.
4. Refit candidate-specific continuous parameters without the unit on which
   the candidate is scored.
5. Store one loss per unit and candidate in a `LossAudit`.  Every candidate is
   evaluated on the same units and domain.  Domain or integration failures are
   retained with a predeclared finite penalty rather than dropped.
6. Construct simultaneous paired-loss confidence bounds with
   `confidence_pareto`.

```python
from nestynet_sr.stat_selection import (
    AuditDesign, CandidateArchive, ComplexityVector, LossAudit,
    build_certificate, confidence_pareto,
)

archive = CandidateArchive(archive_label="example")
archive.add_structure(
    "x0 + x1",
    ComplexityVector.from_mapping({"ast_length": 3, "free_parameters": 0}),
    candidate_id="fixed_sum",
)
archive.add_structure(
    "a*x0 + b*x1",
    ComplexityVector.from_mapping({"ast_length": 5, "free_parameters": 2}),
    candidate_id="fitted_sum",
)
archive.freeze()

losses_by_id = {
    "fixed_sum": [0.10, 0.12, 0.09],
    "fitted_sum": [0.11, 0.12, 0.10],
}
audit = LossAudit.from_matrix(
    candidate_ids=archive.candidate_ids,
    unit_ids=["experiment-0", "experiment-1", "experiment-2"],
    design=AuditDesign(
        loss_name="mean_squared_error",
        unit_kind="experiment",
        fit_protocol="external_holdout",
        evaluation_domain={"split": "audit"},
    ),
    losses=[
        [losses_by_id[candidate_id][unit] for candidate_id in archive.candidate_ids]
        for unit in range(3)
    ],
    archive=archive,
)

result = confidence_pareto(
    audit,
    archive,
    alpha=0.05,
    delta=0.005,
    n_resamples=4000,
    seed=12345,
)
build_certificate(archive, audit, result).write_json("pareto_certificate.json")
```

## Three fronts

`point_front` is the ordinary risk/complexity Pareto front.  A strictly simpler
candidate may dominate another when its point risk is no worse.

`confidence_front` removes a candidate only when a no-more-complex challenger
has a simultaneously negative upper confidence bound on the paired risk
difference.

`practical_front` additionally allows a strictly simpler challenger to dominate
when its simultaneous upper bound is no larger than the predeclared practical
noninferiority margin `delta`.

The returned dominance edges form a directed graph.  The front is the set of
nodes with no incoming edge.  Non-estimable zero-variance comparisons are kept
on the front conservatively rather than being treated as exact evidence.

## Statistical scope

The current inference engine is a single-step multiplier max-T procedure over
all complexity-admissible comparisons.  It preserves the dependence induced by
evaluating every candidate on the same units and controls the comparison family
simultaneously under its bootstrap assumptions.  The result is conditional on
the frozen candidate archive; it makes no claim about expressions outside the
searched grammar.

`bootstrap_front_inclusion_frequencies` is also provided as a stability
diagnostic.  Its frequencies are not confidence levels.

Ordinary SR reports two additional, deliberately distinct decisions.  The
authoritative identification walk starts with the simplest eligible class and
accepts added complexity only when a simultaneous studentised lower confidence
bound for its audit-risk improvement exceeds `--stat-delta`.  A separate
floating-point loss tolerance resolves numerical ties toward simplicity.  The
older unstudentised deployment noninferiority set remains a secondary firewall;
it does not select the reported structural law.

## Ordinary SR integration

Installment 2 makes this contract available from `run_SR.py` without changing
how Stage A, Stage B, factorized search, or the final polisher generate
proposals.  Enable it with, for example,

```bash
python nestynet_sr/run_SR.py \
    --filepath data/problem.csv \
    --stat-selection \
    --stat-audit-fraction 0.2 \
    --stat-unit-size 1 \
    --stat-alpha 0.05 \
    --stat-delta 0.01 \
    --stat-resamples 4000
```

Before any search routine opens the CSV, the adapter validates the numeric,
single-target schema, writes a physical search view containing the prefix, and
seals the contiguous tail as the audit view.  An independently collected audit
CSV can instead be supplied with `--stat-audit-filepath`; it must have exactly
the same columns, resolve to a distinct file, and not be byte-identical to the
search CSV.  The source, search view, and audit view are hashed.  The
search-view hash is checked when the candidate archive is frozen and the audit
hash is checked when certification begins.  The sealed audit path is removed
from downstream search arguments after this plan is created.

The default `--stat-unit-size 1` declares rows to be IID.  Correlated rows must
be grouped into predeclared contiguous units by increasing this value.  Audit
reservations are rounded upward to complete units, an external audit must
contain an integer number of units, and at least two units are required.  This
is only a convenience for ordinary SR.  Trajectory- and experiment-aware
DE/PDE units belong to the installment 3 adapter.

### Search and audit authority

Validation losses, noise thresholds, CoE votes, and the final-polisher
recommendation continue to schedule search and generate candidates.  They are
not allowed to replace the final structure once statistical selection is
active.  In particular:

* the final polisher exports every generated candidate and runs in
  proposal-only mode;
* full-data snap adjudication is disabled;
* CoE final adjudication is retained as a diagnostic proposal layer;
* rejected but numerically evaluated Stage-B candidates retain an AST snapshot
  so they can enter the proposal reservoir;
* the report preserves the former choice under `legacy_search_selection` and
  makes `statistical_selection` authoritative.

The ordinary-SR archive is the canonical union of Stage-B reservoir proposals,
certified Stage-C expressions, y-branch artifacts, and all final-polish
candidates.  Search scores are stored only as provenance.  Complexity is
recomputed as the explicit vector

```text
(ast_nodes, constant_code, free_parameters, tree_depth)
```

Identification uses the predeclared total priority `(free_parameters,
constant_code, ast_nodes, tree_depth, candidate_id)`; missing components sort
as infinity.  The certificate records every class vector in that order.

and candidates with an explicit failed unit certificate are excluded before
the archive is frozen.  Exact and algebraically equivalent duplicates merge
while retaining distinct provenance records.  A deterministic, declared
source-priority then simple-first cap is applied only when the canonical archive
exceeds `--stat-max-candidates`; this makes the certificate conditional on the
retained archive as well as on the searched grammar.

### Common-domain loss

Continuous coefficients fitted during search are held fixed.  Every archived
candidate is evaluated on every audit unit using bounded standardized squared
error.  The scale is fixed from search data before the inferential audit is
opened: a declared positive `noise_sigma_y` is used when available, otherwise
the search-target RMS is used (with a unit fallback for an identically zero
target).  Parse, shape, domain, and non-finite failures receive one common
declared `--stat-failure-loss`; candidate-specific row deletion is impossible.
Candidates with any such common-domain failure remain in the audit and
certificate, but are excluded by the predeclared feasibility rule from the
inferential fronts and deployment set.

The scientific output remains the point, confidence, and practical Pareto
fronts.  For compatibility with downstream code that expects one expression,
the adapter also emits a deployment representative.  It first forms a
risk-only noninferiority set relative to the audit-risk minimizer using an
all-eligible-candidate multiplier max-range bound.  Because the bound is
simultaneous over every ordered eligible pair, the reference may be selected
after the audit without ignoring winner-selection multiplicity.  The simplest
member of that set is used as the deployment representative.  This rule is a
decision heuristic, and the certificate labels it as one: the max-range bound
is unstudentised and is not covered by the max-T calibration profile, so it
carries no confidence guarantee.  This policy does not turn the representative
into a uniquely identified scientific law; the certified front remains
authoritative.

### Hypothesis quotient and method dispatch

Before the audit is opened, the frozen archive is quotiented by **exact
algebraic identity**: candidates whose frozen expression texts canonicalise to
the same expanded SymPy form are one hypothesis with several spellings, and the
front, the fronts' multiplicity family `K_pre`, and the compression claim are
all counted over these classes.  Because the partition derives from frozen
expression text alone, it is fixed before any audit datum is seen, which is
what the calibration profile requires of `K_pre`.  Audit-based
*near-equivalence* (predictions indistinguishable to within a derived or
caller-declared `delta_function`) is an inferential conclusion, not
preprocessing: it is reported in the certificate under
`near_equivalence_descriptive` for presentation only and never shrinks the
comparison family.

The calibration lookup is performed before the front is constructed and its
method is dispatched, not merely recorded.  Inside the validated envelope the
studentised multiplier max-T critical value is used; on `fallback` or
`beyond_grid` the front is built with the one-sided Bonferroni
`t_{1-alpha/K_pre, G-1}` critical value over the same studentised pair
statistics.  The certificate's `inference_regime` block records both the
licensed method and `method_executed` together with the critical value the
front actually used, so a reader can verify the dispatch.  The compression
certificate names the authoritative identification class, charges its
class-level risk, and prints its minimum-description-length representative.

### Resume and provenance firewall

Statistical checkpoints store a path-independent contract containing the
source, search, and audit hashes, row boundary, and audit-unit schema.  A
checkpoint created before this contract existed, or behind another split, is
rejected in certification mode.  Model filenames also include the search-view
hash, preventing an older full-data model from being silently reused.
Statistical archive and
certificate outputs are placed under a path-independent split-contract fingerprint
directory.  The
legacy Buckingham structural retry is disabled in certification mode because
an adaptive second search after the sealed audit has been opened would violate
the one-shot audit contract.

Unprovenanced `--load_expressions` files and external CoE reservoirs are
rejected while `--stat-selection` is active.  They may have been generated from
the sealed observations.  A later archive-import format can admit such
artifacts once it carries a verifiable data-provenance contract.

The machine-readable outputs are a frozen archive JSON and a Pareto certificate
JSON.  The certificate separately records the authoritative Occam
identification walk and the secondary deployment noninferiority set, together
with common domain, unit definition, paired losses, failures, assumptions,
dominance edges, and fingerprints.

### Current adapter boundary

Installment 2 intentionally fails closed for multi-dataset ordinary SR,
`--discover_de`, and `--discovery_enable`.  Those cases require an explicit
experiment/trajectory unit adapter and join the same statistical substrate in
installment 3.

## Installment 3: differential-equation discovery

`run_de.py --stat-selection` moves DE adjudication to independent whole
trajectories.  A trajectory is never split into pseudo-independent rows.  Use
one of:

```bash
python nestynet_sr/run_de.py --filepaths fit0.csv fit1.csv audit0.csv audit1.csv \
  --stat-selection --stat-audit-trajectories 2
```

or, preferably, independently collected audit trajectories:

```bash
python nestynet_sr/run_de.py --filepaths fit0.csv fit1.csv \
  --stat-selection --stat-audit-filepaths audit0.csv audit1.csv audit2.csv
```

The input convention used by the rollout validator is one state column followed
by the independent coordinate (`u,x`).  The firewall hashes every search and
audit trajectory before surrogate fitting and verifies the hashes again before
opening the audit responses.

Every validation-ready order-1 or order-2 explicit ODE candidate in the
serialized proposal slate is canonicalized into one frozen archive.  Complexity
is the vector `(differential_order, active_terms, ast_nodes)`.  Candidates are
then integrated on every audit trajectory using one solver contract and scored
by squared rollout NRMSE.  Compilation, domain, non-finite, timeout, blow-up,
and integration failures receive one predeclared whole-trajectory loss and are
excluded from inferential eligibility while remaining visible in the
certificate.

The certificate contains point, simultaneous-confidence, and practical
noninferiority Pareto fronts, paired trajectory losses, support classes,
bootstrap front-inclusion frequencies, failure records, and a conservative
campaign action.  Search scores and the legacy DE committee remain proposal
machinery only.

This installment deliberately certifies scalar explicit first- and second-order
ODEs through `run_de.py`.  PDE fields, implicit systems, `run_SR.py
--discover_de`, coherent surrogate-jet uncertainty, and full structural
rediscovery are reserved for the next adapter/calibration layer.

## Coherent surrogate uncertainty and structural rediscovery

The DE adapter accepts an optional NPZ bundle through
`--stat-coherent-loss-draws`.  It is an interchange format, not an
independence fiction.  The required arrays are:

- `losses`, shape `(n_draws, n_audit_trajectories, n_candidates)`;
- `candidate_ids`, matching the frozen archive identifiers;
- `unit_ids`, matching the audit trajectory stems;
- optional `draw_ids` and `metadata_json`.

Each draw must be generated coherently: the surrogate value, every derivative
jet used to construct the candidate library, refitted coefficients, and the
forward rollout must all descend from the same NestyNet sister/posterior draw.
The adapter averages over coherent draws within each independent trajectory
and records the full draw-wise risk distribution separately.  Draws are not
counted as additional experimental units.

Independent complete reruns can be supplied with
`--stat-rediscovery-reports`.  Their selected supports are canonicalized and
reported as structural rediscovery frequencies.  These frequencies are a
stability diagnostic, not posterior model probabilities and not a substitute
for the confidence Pareto front.

`--stat-calibration-repetitions N` adds a deterministic paired-comparison
smoke test to the certificate.  It is intended to catch gross numerical or
packaging regressions; publication calibration should use the problem-specific
simulation suite, including the complete surrogate-fit, jet, discovery, and
rollout chain.

## Patch 5: Schur-profiled input errors

Ordinary-SR certification can now include Gaussian uncertainty in both the
inputs and target.  Supply a shared diagonal input error with

```bash
--stat-x-sigma 0.02,0.05
```

or a shared/per-audit-row full covariance in an NPZ bundle:

```bash
--stat-x-cov-npz audit_x_covariance.npz
```

The NPZ must contain `x_cov` with shape `(Nx,Nx)` or
`(audit_rows,Nx,Nx)`.  The file is hashed into the audit certificate.

## Calibration laboratory

`nestynet_sr/stat_selection/calibration_lab.py` exercises the inference alone,
on synthetic populations whose risks and loss covariance are known exactly.  No
search, no fitting, no symbolic machinery.  Run it with:

```bash
python scripts/run_calibration_lab.py --replicates 2000 --out lab.json
```

The separation from a benchmark campaign is deliberate: the laboratory answers
*is the procedure calibrated*, a benchmark answers *does the whole system stay
calibrated once parsing, fitting, archive generation and domain failures are
involved*.  Confounding them makes a failure impossible to localise.

Two quantities are measured.  `familywise_false_edge_rate` is the probability
that *any* dominance edge asserts an ordering the population does not have,
which is what the max-T controls at `alpha`.  `front_coverage` is the
probability that every genuinely non-dominated candidate survives, which is what
a reader of the certificate believes.

### Result 1: the operating envelope is two-dimensional

The multiplier max-T is asymptotically calibrated and anti-conservative in
finite samples, and the deficiency is governed jointly by the number of
independent units `G` **and** the number of admissible simultaneous comparisons
`K_adm`.  Neither coordinate alone predicts it.  `K_adm` rather than archive
size `M` is the right coordinate, because complexity filtering means two
archives of equal size can present very different comparison families.

Familywise false-edge rate against nominal `alpha = 0.05`.  63 cells, 2500
replicates each, 1000 resamples, equal-risk equal-complexity candidates:

| `K_adm` \ `G` | 12 | 18 | 24 | 36 | 60 | 100 | 150 | 250 | 400 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 0.114 | 0.089 | 0.074 | 0.066 | 0.058 | 0.060 | 0.053 | 0.051 | 0.048 |
| 30 | 0.160 | 0.118 | 0.105 | 0.076 | 0.074 | 0.055 | 0.048 | 0.048 | 0.045 |
| 110 | 0.264 | 0.163 | 0.141 | 0.102 | 0.080 | 0.067 | 0.064 | 0.058 | 0.048 |
| 306 | 0.398 | 0.250 | 0.184 | 0.136 | 0.090 | 0.065 | 0.067 | 0.064 | 0.062 |
| 1056 | 0.636 | 0.393 | 0.275 | 0.177 | 0.108 | 0.091 | 0.075 | 0.069 | 0.059 |
| 3080 | 0.838 | 0.541 | 0.380 | 0.230 | 0.139 | 0.097 | 0.080 | 0.067 | 0.062 |
| 10100 | 0.971 | 0.760 | 0.532 | 0.304 | 0.182 | 0.114 | 0.084 | 0.068 | 0.062 |

Cells classified on the **Wilson upper bound** of the observed rate, never the
point estimate, because the calibration experiment carries Monte Carlo error of
its own (`V` validated, upper bound <= 0.06; `T` transitional, <= 0.08;
`.` outside):

| `K_adm` \ `G` | 12 | 18 | 24 | 36 | 60 | 100 | 150 | 250 | 400 |
|---:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 12 | . | . | . | T | T | T | T | **V** | **V** |
| 30 | . | . | . | . | . | T | **V** | **V** | **V** |
| 110 | . | . | . | . | . | T | T | T | **V** |
| 306 | . | . | . | . | . | T | T | T | T |
| 1056 | . | . | . | . | . | . | . | T | T |
| 3080 | . | . | . | . | . | . | . | T | T |
| 10100 | . | . | . | . | . | . | . | T | T |

Smallest `G` reaching a given rate, which is the practically useful form:

| `K_adm` | `G` for <= 0.06 | for <= 0.08 | for <= 0.10 |
|---:|---:|---:|---:|
| 12 | 60 | 24 | 18 |
| 30 | 100 | 36 | 36 |
| 110 | 250 | 60 | 60 |
| 306 | none <= 400 | 100 | 60 |
| 1056 | 400 | 150 | 100 |
| 3080 | none <= 400 | 150 | 100 |
| 10100 | none <= 400 | 250 | 150 |

Three readings matter.

The **corner is a total loss of control**, not a degradation: `K_adm = 10100` at
`G = 12` runs at 0.971, so nearly every replicate produces at least one false
dominance edge.  A twelve-trajectory audit against a large candidate bank would
confidently exclude true candidates almost always.

**Only six cells certify**, all at `K_adm <= 110` and `G >= 150`.  Everything at
`K_adm >= 306` is transitional at best even at 400 units.  Since transitional
routes to the conservative fallback, the multiplier max-T is the licensed
method in a narrow corner and Bonferroni-t governs most of the realistic grid.

**Certification needs replicates.**  At 800 replicates a cell whose true rate is
exactly 0.05 yields a Wilson upper bound near 0.066, so no correctly calibrated
cell could certify.  2500 replicates is roughly the minimum for the 0.06
threshold to be reachable.  The decision map is only as sharp as its per-cell
Monte Carlo budget, which must be quoted alongside it.

### Result 1b: there is no clean one-dimensional summary

A collapse coordinate would let the envelope be interpolated rather than
tabulated.  Measuring the worst rate gap between cells at near-equal coordinate,
over all 63 cells:

| coordinate | worst near-neighbour gap |
|---|---:|
| `log(K)^2 / G` | 0.124 |
| `log(K)^1.5 / G` | 0.134 |
| `log(K) / sqrt(G)` | 0.134 |
| `log(K) / G` | 0.362 |
| `sqrt(log(K) / G)` | 0.362 |

The best candidate still leaves a residual spread of 2.5 times the nominal rate.
On an earlier 15-cell grid `log(K)^1.5 / G` looked convincing at 0.026, which
was small-sample optimism rather than structure.  **Tabulate the envelope, do
not fit it.**

### Which regime a real audit occupies

`--stat-unit-size` defaults to 1, so for ordinary SR every audit row is an
independent unit.  With the default 20% audit fraction a 5,000-row problem
gives `G = 1000` and a 100,000-row problem gives `G = 20000`, both off the right
edge of the table.  **Ordinary SR is not near the boundary.**

DE discovery is, and it is a data limitation rather than a compute one.  Twelve
trajectories are twelve units however densely each is sampled, which is the
point Result 3 below makes.  More compute and more candidates cannot help;
only more independent excitations can, or acceptance of the fallback and its
power cost.

Note that `unit_size = 1` is a *declaration* that rows are independent.  It is
defensible for AI Feynman, whose inputs are sampled independently.  It is not
defensible for a time series, a single orbit, or spatial cells from one
simulation.  Nothing checks it; the burden sits with whoever sets the flag.

### Result 2: multiplicity is where ordinary selection logic fails

Familywise false-exclusion rate against archive size, at `G=150` where the max-T
is calibrated so the comparison is not confounded with small-`G` bias.  600
replicates, `alpha=0.05`:

| M | admissible pairs | max-T | pointwise | `max(SE_a,SE_b)` |
|---:|---:|---:|---:|---:|
| 5 | 20 | 0.060 | 0.440 | 0.413 |
| 10 | 90 | 0.058 | 0.818 | 0.723 |
| 25 | 600 | 0.080 | 0.992 | 0.965 |
| 50 | 2450 | 0.060 | 1.000 | 1.000 |
| 100 | 9900 | 0.083 | 1.000 | 1.000 |

The simultaneous procedure stays flat as the comparison family grows five
hundredfold.  Both alternatives reach **certain** false exclusion by fifty
candidates.  Since a real archive holds tens to hundreds of candidates, this is
the quantitative statement of why unadjusted selection is unsafe, rather than an
assertion that it is.

### Result 3: dense sampling is not extra evidence

Interval coverage against samples per group, 12 groups, nominal 0.95:

| samples/group | group-level | row-level |
|---:|---:|---:|
| 1 | 0.93 | 0.93 |
| 4 | 0.97 | 0.69 |
| 16 | 0.92 | 0.42 |
| 64 | 0.93 | 0.23 |
| 256 | 0.92 | 0.11 |

Row-level resampling collapses as sampling density grows while the group-level
analysis stays flat, which is the "one orbit is not ten thousand votes" claim
made measurable.

Dominance power rises from `alpha` at zero gap to 1.0 by a risk gap of 0.5, so
the procedure is powerful rather than merely conservative.

### Note on warnings

numpy built against Apple Accelerate raises spurious `divide by zero`,
`overflow` and `invalid` FP flags from `matmul` on small, finite,
well-conditioned operands; a bare `standard_normal((24,16)) @
standard_normal((16,16))` reproduces it.  Such warnings from this package on
macOS are platform noise, not numerical failure.  The laboratory verifies its
draws are finite rather than trusting the absence of a warning.

## Joint x and y errors

For a frozen symbolic candidate, the audit computes its input gradient and
Schur-eliminates the local latent input displacement, giving the effective
variance

```text
sigma_eff^2 = sigma_y^2 + grad(f)^T Sigma_x grad(f).
```

The default loss is the **marginal Gaussian negative log likelihood**

```text
(y-f(x))^2 / sigma_eff^2  +  log(sigma_eff^2 / sigma_y^2).
```

`--stat-x-error-loss profile_chi2` drops the log-determinant term, leaving the
bare profiled quadratic `(y-f(x))^2 / sigma_eff^2`.

The default is the normalised form on purpose.  Without the log term, a
candidate lowers its own loss simply by being **steep**: a larger `grad(f)`
inflates `sigma_eff^2`, which divides the residual.  The inflation is commonly
a factor of a few when input errors matter, which is large enough to reorder a
risk comparison.  The profiled quadratic is the right object for a
goodness-of-fit test at fixed structure; it is not the right object for ranking
structures against one another, which is what the front does.  This is the
usual "a profile likelihood is not a likelihood" caveat.

The audit records `max_x_variance_inflation` and
`median_x_variance_inflation` per candidate, so the size of the effect is
visible either way.  `--stat-x-gradient-step` controls the relative
central-difference step used for symbolic candidates.

This audit operation gives a statistically explicit local errors-in-variables
comparison of the frozen symbolic laws.  It assumes each expression is locally
linear across its input-error ellipse; that assumption is recorded in the
certificate but not checked, and it is the weakest link for candidates with
poles or sharp exponentials.  For DE discovery, where
input uncertainty changes derivative jets and rollout coordinates, those
coherent draws should be propagated through the complete surrogate-to-rollout
chain and supplied through `--stat-coherent-loss-draws`.

The local Schur approximation assumes the candidate is approximately linear
across each input-error ellipse and that rows have independent measurement
errors.  Full covariance among input dimensions is supported; covariance
between rows is not yet represented by the pointwise audit adaptor.
