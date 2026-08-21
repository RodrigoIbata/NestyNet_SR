# Galactic Tidal-Radius Vignette (Jacobi Radius)

Worked example: from mock star-cluster tables, the pipeline discovers the
anisotropic tidal invariant `4*Omega^2 - kappa^2` through its learned
general-affine determining operator and recovers the circular-orbit Jacobi
radius law in closed form,

```
r_J = (mu / (4*Omega^2 - kappa^2))^(1/3),    mu = G*M_cl,
```

unifying the classical Hill/Roche (`alpha = -0.5`) and flat-rotation-curve
(`alpha = 0`) limits of a power-law host. The study also measures the
gradient-noise envelope of the symmetry certificate and exhibits principled
abstention at 1% observational noise (the gradient-noise abstention ladder).
It is the controlled synthetic companion to the SPARC
baryonic-acceleration-carrier vignette (`examples/sparc_carrier/`).

## The complete analysis

The full write-up is `jacobi_tidal_note.tex` / `jacobi_tidal_note.pdf`
(originally a Paper III subsection, now distributed as a standalone note).
Start there: it contains the problem statement, the discovery ladder, the
noise study, and the exact reproduction commands.

## Contents

- `data/`
  - `jacobi_master.csv` — master mock cluster table.
  - `jacobi_v1_full.csv`, `jacobi_v2_cube.csv`, `jacobi_v3_carrier.csv` —
    derived per-stage tables.
  - `jacobi_manifest.json` — provenance: target law, carrier definition,
    named limits, and the `L,T,M` units basis.
- `figures/jacobi_vignette_quadpanel.png` — the vignette figure.
- `results/` — run logs and JSON reports from the discovery and noise runs
  (path prefixes are redacted in the public bundle).

## Reproduction

The end-to-end invocation is recorded in the note (Reproduction section) and
in `results/reroll_harness.log`. It follows the fixed Paper III benchmark
configuration with the generalized-symmetry flags enabled:

```bash
python3 -u -m nestynet_sr.run_SR \
  --filepath examples/jacobi_tidal/data/jacobi_v1_full.csv \
  --units_basis "L,T,M" --y_units "[1,0,0]" \
  --x_units "[[3,-2,0],[0,-1,0],[0,-1,0]]" \
  --gs-stagea --gs-general-affine --gs-noise-calibrated-promotion \
  --ndata_train 6000 --ndata_val 2000
```

See `PAPER3_REPRODUCIBILITY.md` at the repository root for how this example
fits into the Paper III reproduction package.
