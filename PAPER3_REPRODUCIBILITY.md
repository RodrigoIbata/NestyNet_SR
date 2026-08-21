# Reproducing the Paper III AI Feynman benchmarks

Paper III uses 120 AI Feynman problems at four noise fractions: `0.000`,
`0.001`, `0.010`, and `0.100`. The exact CSVs used for the paper are archived
separately at [Zenodo DOI 10.5281/zenodo.21390410](https://doi.org/10.5281/zenodo.21390410).

The setup utility verifies the 2,060,232,099-byte archive against SHA-256
`8c692d6db840df3ba2c3276cfc445dcfd33d6581ac00b926369533df4d1cd3a2`,
then creates four benchmark workspaces and installs the corresponding 120
CSVs in each. It also copies `data/equations.txt`, installs the exact benchmark
driver, and links every workspace to this NestyNet_SR checkout.

## Prerequisites

Use Python 3.10 or newer and install NestyNet followed by NestyNet_SR:

```bash
cd /path/to/NestyNet
python -m pip install -e .
cd /path/to/NestyNet_SR
python -m pip install -e .
```

Runtime dependencies and version floors are declared in `pyproject.toml`.

## Create the workspaces

If the archive is already present locally:

```bash
python scripts/setup_paper3_reproduction.py \
    ../NestyNet_paper3_recreation \
    --archive /path/to/AIF_data_zenodo.tgz
```

If `--archive` is omitted, the utility downloads the file from the permanent
Zenodo record and caches it below the recreation directory:

```bash
python scripts/setup_paper3_reproduction.py ../NestyNet_paper3_recreation
```

The recreation directory contains:

```text
NestyNet_paper3_recreation/
├── NestyNet_SR -> source checkout
├── paper3_reproduction_manifest.json
├── SRBench_0.000/
├── SRBench_0.001/
├── SRBench_0.010/
└── SRBench_0.100/
```

The manifest records the archive checksum, source Git revision, runner
template checksums, and each of the 480 installed CSV checksums. The setup is
idempotent and never deletes results. Add `--verify-existing` to hash all
installed CSVs on a later audit. Use `--force` only when deliberately replacing
managed data or scripts.

## The three selection regimes of the noise table

Every column of Paper III's noise table uses the same data budget, 2,000
training and 2,000 validation rows. The columns differ in what holds
authority over the final choice of structure, not in how much data the search
saw. Run each noise level from its own workspace directory.

**Heuristic acceptance** (the parenthesized column of the noise table). The default acceptance
rules of the Stage-B engine decide, with no statistical layer and no
committee:

```bash
cd ../NestyNet_paper3_recreation/SRBench_0.000
JOBS=1 STAT_SELECTION=0 COE_MODE=off ./scripts/run_allstages_all.sh
```

**Statistical selection** (the unmarked column). An audit partition is sealed
before the search opens the data, the candidate archive is frozen before any
audit loss is read, and a confidence Pareto front over the sealed partition
makes the final selection:

```bash
JOBS=1 COE_MODE=off ./scripts/run_allstages_all.sh
```

**Statistical selection with the committee** (the bracketed column). The
committee proposes and diagnoses; the statistical layer still decides. This
is the configuration the committee workspaces ship with, and it writes to
`results_CoE` rather than `results`:

```bash
JOBS=1 ./scripts/run_allstages_all.sh
```

`STAT_SELECTION` defaults to 1 and `COE_MODE` to `reservoir_discovery` in the
committee workspaces, with eight scout proposers on slices 1--8 and sixteen
witness slices from slice 9. The sealed audit partition is 20,000 rows,
drawn from the full dataset and therefore much larger than the training
split; `--stat-audit-fraction`, `--stat-unit-size` and `--stat-alpha` are
forwarded through `STAT_ARGS`.

`JOBS=1` runs one problem at a time; the driver restricts numerical-library
threads to one per job by default. The published campaigns used `JOBS=8`, so
their recorded per-problem wall times include some contention.

## Collecting and checking the evidence

The reproducibility evidence for a completed campaign is a *capsule*: every
per-problem `pb*.report.json`, the `summary.csv` carrying the recovered
expressions, a manifest recording the run configuration read from the runs
themselves, and the structural audit. Models, checkpoints and logs are not
part of it.

```bash
# in the campaign workspace, first regenerate the summary
./scripts/summarize.sh

# then collect (committee campaigns need --results-dir results_CoE)
python3 /path/to/NestyNet_SR/scripts/collect_srbench_capsule.py . \
    --cell noise0.010_stat --output /path/to/capsules --tar
```

A case counts as solved when the recovered expression is algebraically
identical to the target up to the values of its fitted constants. That is
decided by refitting every free constant on the canonical noiseless data,
re-snapping with the pipeline's own polisher, and requiring the noiseless-fit
floor of 1e-10 relative RMSE. The criterion lives in
`scripts/_structural_verdict.py` and is shared with the oracle benchmark, so
both tables mean the same thing by "solved":

```bash
python3 scripts/audit_srbench_structural.py /path/to/capsules/noise0.010_stat \
    --noiseless-data /path/to/SRBench_0.000/data
python3 scripts/validate_paper3_results.py --capsules /path/to/capsules
```

The validator checks each frozen cell in
`reproducibility/paper3/expected_results.json` for completeness and agreement
with the recorded totals. Cells may declare `not_completed` problems: runs
that exceeded their wall-clock budget and never returned count as non-solves
but are recorded separately from problems that ran and failed.

`scripts/build_paper3_artifact.py` assembles the capsules, the oracle audit
and the reproducibility metadata into the checksummed evidence archive,
refusing to build if any frozen cell fails to validate or if any shipped text
retains a machine-specific path.

## Current reproducible cheap-first escalation

The historical campaigns did not persist a single automatic cheap-to-CoE
decision record. For new campaigns, the repository now provides a truth-blind,
resumable controller:

```bash
JOBS=1 ./scripts/run_allstages_escalating.sh
```

It runs the ordinary phase first and invokes CoE only for completed reports
without an eligible internal symbolic selection. Process failures retry the
same phase. This current protocol is intentionally distinguished from an exact
historical Paper III reproduction. See
[`CAMPAIGN_ESCALATION.md`](CAMPAIGN_ESCALATION.md) for previewing an existing
campaign, reason codes, manifest semantics, and configuration overrides.

## Summaries and targeted reruns

After a benchmark completes:

```bash
./scripts/summarize.sh
```

To rerun one problem:

```bash
./run_pb.sh pb010
```

The noiseless run writes the compact pre-separability Stage-A snapshots used
by Paper I to `SRBench_0.000/results_ref/`. To regenerate Paper I's accuracy
artifacts from this new snapshot, pass that directory explicitly:

```bash
python /path/to/NestyNet/scripts/regenerate_paper1_accuracy.py \
    --reference-dir /path/to/NestyNet_paper3_recreation/SRBench_0.000/results_ref \
    --jobs 8
```

The Paper I submission builder accepts the same directory through
`REFERENCE_ROOT`:

```bash
REFERENCE_ROOT=/path/to/NestyNet_paper3_recreation/SRBench_0.000/results_ref \
bash make_nestynet_submission.sh
```

The symbolic ground-truth expressions in `data/equations.txt` are used for
post-run equivalence scoring and the dimensional metadata used by the search.
The benchmark CSV archive itself contains only sampled input-output data.

## Reproducing the FSS oracle benchmark

The oracle-mode factorized-search benchmark (paper section "AI Feynman
Symbolic Regression Benchmark (Factorized Search, Oracle Mode)") is run by
`scripts/run_factorized_search_all.sh`. The paper protocol is NOT the
driver's bare default: it requires the dense-audit split and the
generalized-symmetry carrier seeding, both explicit below. The exact
invocation behind the published numbers (88/115 structural recoveries,
76.5%; median/mean wall 30.0 s / 553 s):

```bash
JOBS=8 N_FIT=512 N_PROBE=16384 SEARCH_N_FIT=512 SEARCH_N_PROBE=2048 \
  FINAL_VALIDATE_FULL=1 \
  OUTPUT_DIR=results/factorized_search_aif_$(date +%Y%m%d) \
  bash scripts/run_factorized_search_all.sh --gs-carrier-seed
```

Semantics: the search sees 512 fit points and a 2,048-point search probe;
`FINAL_VALIDATE_FULL=1` re-scores every returned candidate without refit on
the full split. The `--gs-carrier-seed` flag opts in to the GS carrier
proposals the paper describes; without it the run is the FSS-only ablation.
Always use a dated `OUTPUT_DIR`: the driver default overwrites previous runs
in place.

Scoring is the structural criterion described above, shared with the noise
table, and is applied after the run rather than by the driver:

```bash
python3 scripts/audit_fss_structural.py \
    --run-dir results/factorized_search_aif_YYYYMMDD \
    --noiseless-data /path/to/SRBench_0.000/data
```

The oracle runs record the skeleton and the mapping family but not the
fitted mapping coefficients, so the auditor first reconstructs each
candidate: it parses the skeleton, refits the outer mapping on a dense
sample of the exact oracle target over the equation's own box, numerically
self-checks the conversion, and only then applies the shared verdict to all
115 eligible cases.

The final reference summary behind the paper's numbers is committed at
`reproducibility/paper3/fss_oracle_final_summary.json`. It contains the
authoritative 88/115 structural total and the per-case probe/dense
measurements, wall times, expressions, and structural verdicts. The matching
structural audit and final summary both ship in the evidence archive.

Seeds are fixed (SEED=42, one search seed), but search trajectories are
sensitive to code revisions; expect one or two knife-edge cases to flip
between revisions and compare suites at the distribution level.

## Reproducing the vignettes

### Galactic tidal-radius example

`examples/jacobi_tidal/` is the worked example referenced by the paper's
compact tidal-radius subsection. The accompanying note
(`jacobi_tidal_note.tex`, compiled PDF included in the public bundle)
contains the complete analysis; the cluster tables under `data/`, the run
logs and reports under `results/`, and the figure under `figures/` ship
with the public bundle. The end-to-end invocation is recorded in the note
and in `results/reroll_harness.log`, and uses the fixed benchmark
configuration of the paper with the generalized-symmetry flags enabled.

### SPARC baryonic-acceleration-carrier vignette

`examples/sparc_carrier/` contains every script behind the SPARC vignette
(section "Blind Recovery of the Baryonic Acceleration Coordinate from
SPARC" and its six-panel figure). All scripts fix their random seeds; the
whole-galaxy discovery/held-out split is seed 0 throughout.

1. Fetch the two SPARC machine-readable tables (Lelli, McGaugh &
   Schombert 2016) into `data/sparc/`:

   ```bash
   mkdir -p data/sparc && cd data/sparc
   curl -sO https://astroweb.cwru.edu/SPARC/MassModels_Lelli2016c.mrt
   curl -sO https://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt
   shasum -a 256 -c <<'SUM'
   9108994b12cc401b94a1768beca61c53ec354779385c9c9cc571049f3043244c  MassModels_Lelli2016c.mrt
   5aa0501f6b0d881fa579030e315e7b5b6ef561a5bd3a07472f9929c7e5728243  SPARC_Lelli2016c.mrt
   SUM
   ```

2. Build the pilot datasets (standard RAR cuts; gold sample):

   ```bash
   cd examples/sparc_carrier
   python3 build_dataset.py
   ```

3. Run the analyses in order (later scripts read the earlier scripts'
   outputs from `results/`):

   ```bash
   python3 run_pilot.py --dataset gold   # carrier battery + controls
   python3 synth_check.py                # planted-carrier control
   python3 outer_map.py                  # outer-law candidate family
   python3 fss_outer.py                  # factorized-search slate
   python3 sister_slope.py               # certified slope posterior
   python3 make_paper_figure.py          # the paper's six-panel figure
   ```

   Supplementary analyses: `nuisance_release.py` (galaxy-geometry
   robustness pass), `gs_fss_handoff.py` (carrier-seed handoff study),
   `classical_check.py` / `classical_rank_check.py` (NestyNet-free
   reference checks), `make_figures.py` / `make_fig4.py` (the pre-layout
   individual figures).

Numbers quoted in the paper (carrier coefficient and bootstrap interval,
held-out rms values, determining-operator spectra, slope posterior, and
the geometry-release scatter) are printed by the scripts above; the
directory `README.md` documents which script produces which claim.
