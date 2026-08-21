# NestyNet Paper III benchmark evidence

This archive is the evidence package for the two AI Feynman benchmark tables
in Paper III. It contains twelve full-pipeline capsules (one per noise/selection
cell), the oracle-mode factorized-search evidence, frozen expected totals,
provenance notes, a machine-readable manifest, and checksums. Models,
checkpoints, logs, and the separately published benchmark inputs are excluded
by design.

## Oracle result

`oracle/structural_audit.json` is the authoritative audit for the paper's
reported oracle result: **88 structural recoveries among 115 eligible
equations (76.5%)**. It applies the same refit, coefficient-polish, and
noiseless-fit-floor criterion as the noisy benchmark table.

`oracle/oracle_summary.json` contains the same final per-case
structural verdicts together with recovered expressions, probe and
dense-holdout measurements, and wall times. Its aggregate values reproduce
88/115 and the reported median and mean wall times of 30.0 s and 552.7 s.

## Reproduction and scope

`reproducibility/PAPER3_REPRODUCIBILITY.md` gives the commands and explains
the structural criterion. Executable scripts are versioned in the NestyNet-SR
repository recorded by `MANIFEST.json`; they are not duplicated in this data
archive.

The archive is deliberately benchmark-scoped. The Jacobi and SPARC vignettes
are distributed as worked examples in the NestyNet-SR repository under
`examples/jacobi_tidal/` and `examples/sparc_carrier/`, respectively. SPARC's
third-party source tables are not redistributed here.

## Citation and licensing

Please cite this Zenodo record together with Paper III and the input-data
record `10.5281/zenodo.21390410`. The license chosen on the Zenodo record is
authoritative. Third-party inputs remain subject to their source terms; none
of the separately archived AI Feynman inputs or SPARC source tables is
duplicated here.
