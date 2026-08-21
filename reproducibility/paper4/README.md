# NestyNet Paper IV data and reference artifact

This is the self-contained data/reference package for Paper IV. It contains
the exact benchmark inputs, all 308 Sun-centred daily HORIZONS ephemerides,
the validated 308-entry analytic-surrogate acceleration cache, compact result
summaries for every protocol step, and the two publication figures.

## Layout

- `data/`: scalar, complex, compositional, and Maxwell benchmark inputs plus
  the Kepler ephemerides and `data/kepler/surrogate_accels_1d/` cache.
- `results_ref/`: compact machine-readable outputs for the fifteen frozen
  protocol steps. Checkpoints and exploratory logs are excluded.
- `figures/`: publication PDFs regenerated from the reference result tree.
- `reproducibility/protocol.json`: exact commands and expected headline
  totals.
- `reproducibility/assembled_run_provenance.json` and `step_manifests/`:
  command, revision, completion, and return-code provenance for each step.
- `reproducibility/PROVENANCE_NOTES.md`: interpretation of the assembled run,
  including historical revision information.
- `reproducibility/PAPER4_REPRODUCIBILITY.md`: installation, rerun, and setup
  instructions.
- `ARTIFACT.json`: build environment and file-count inventory.
- `SHA256SUMS`: a digest for every other regular file in the package.

The standalone `paper4_run_manifest.json` from an earlier superseded run is
intentionally not shipped when assembled provenance is used. The assembled
record and per-step manifests are the authoritative execution provenance.

## Reference totals

- scalar STLSQ: 40 PASS / 9 PARTIAL / 8 FAIL;
- scalar hybrid: 53 PASS / 4 PARTIAL / 0 FAIL;
- complex class-library STLSQ: 23 PASS / 3 PARTIAL / 0 FAIL;
- operator-factorized compositional cases 900--903: four PASS results.

The compositional PASS gate is held-out rollout NRMSE below `1e-2`; the four
recorded means are approximately `6.65e-6`, `8.69e-3`, `2.82e-3`, and
`3.64e-5`.

## Software, citation, and licensing

The executable utilities are versioned at
<https://github.com/RodrigoIbata/NestyNet_SR> and are not duplicated in this
data archive. Please cite this Zenodo record together with Paper IV. The
license selected on the Zenodo record is authoritative. JPL HORIZONS and
SsODNet-derived material retains its source provenance and terms, recorded in
`data/kepler/`; the archive license must not be interpreted as replacing
third-party source terms.
