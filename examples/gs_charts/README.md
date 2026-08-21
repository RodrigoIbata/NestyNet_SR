# GS chart machine demos

The GS -> charts bridge (`nestynet_sr.sr_gs.chart_bridge`) discovers
CONTINUOUS symmetries of a fitted surrogate's graph and compiles them into
executable input charts, with physics read off the generator as a
by-product. Complement to the circle machinery in `nestynet.charts` (which
finds discrete periodicity); together they cover both symmetry modes.

## demo_blast_wave.py: Taylor's Trinity analysis, blind

G. I. Taylor famously inferred the Trinity yield from film frames via the
Sedov-Taylor law R = xi0 (E t^2/rho)^(1/5). Synthetic Trinity-like radius
data (20 kt, 0.2% noise, detonation 2 ms BEFORE the first frame) are handed
to the machine with no physics declared. Measured blind result:

```
detonation instant t0: -1.946 ms   (true -2.000 ms)
similarity exponent:    0.4039     (Sedov-Taylor 2/5)
certificate:            KEY (defect floor 9.9e-3, C_max 4.7)
yield from amplitude:   20.1 kt    (true 20.0 kt; rho, xi0 declared)
```

`--case decay`: an RC discharge is recognized as translation + exponential
cofactor, tau recovered to 1.4990 s (true 1.500). `--case control`: an
aperiodic bump certifies nothing.

## demo_sn1993j.py: dating a real supernova from VLBI radii

REAL data: 27 outer angular radii of SN 1993J (Marcaide et al. 2009, A&A
505, 927, Table 1 as digitized; verify against the published table for
paper-grade use). The machine sees only (time since the FIRST VLBI epoch,
radius) and must locate the expansion's fixed point 182 days before its
data begin. Measured blind result:

```
recovered explosion epoch: 180 d before the first epoch (true 182 d;
                           2 d error on a 10.1 yr baseline)
deceleration exponent m:    0.789  (published: 0.845 early / 0.788 late)
law: CERTIFIED (defect 4.5e-2 over ~3 e-foldings)
identifiability: honest -- the defect basin is FLAT over roughly
                 t0 in [-250, -70] d, so the point estimate within the
                 basin is partly fortunate; m is robust (0.79-0.81 across
                 all configurations tried)
```

The flat basin is physics, not failure: the expansion exponent DRIFTS
(0.845 -> 0.788, the paper's own discovery), so no single power law pins
the origin sharply. SN 1987A would be the extreme companion case: its
shock hit the equatorial ring and broke self-similarity entirely, so the
law gate itself should refuse -- the refusal being the astrophysics.

## Methodology notes (measured, 2026-08-03/04)

- A diffeomorphic warp has NO value-fit sharpness well: the network absorbs
  any smooth reparameterization (measured C == 1). The key-sharpness
  certificate for warps therefore profiles the GS DETERMINING residual
  rho(t0(1+delta)) with the output action (alpha, beta) profiled out -- a
  linear solve per point, no refits.
- The determining rows are Haar-weighted (1/(t-t0), uniform in the warped
  coordinate), which simultaneously equalizes the heteroscedastic
  surrogate-derivative noise.
- The identity surrogate is least accurate exactly where the scaling
  information is richest (the steep early rise), so t0 is refined
  SELF-CONSISTENTLY: refit on the candidate warp, recompute analytic
  derivatives from the warp model in raw coordinates, re-refine on the
  defect metric. Two rounds took t0 from 8% error to 3% and the exponent
  from 0.40144 (raw generator ratio) to 0.4039 profiled.
- Warp wells are floor-calibrated like all key wells (NestyNet_Kepler rung
  2c): with a noisy floor they open at delta of order one, so the gate
  scans delta up to 1.
- The warp family has a NON-COMPACT DEGENERACY: as t0 -> -inf the log
  compresses the data onto a short arc where any smooth curve satisfies
  the scaling law (the translation limit), so defect minimization slides
  down an artificial valley (observed: a "certified" t0 at -1770 d with
  defect 2e-4 tested over only 1.1 e-foldings). Selection therefore takes
  the SMALLEST gap that passes the self-consistent defect gate -- the
  hardest-to-vary key, the one tested over the most e-foldings -- and
  refines strictly inside that basin.
- Law certification and fixed-point identifiability are SEPARATE
  statements: `law_certified` (defect passes over the tested range) gates
  the chart; the sharpness well (C_max, delta*) reports how sharply the
  parameter is pinned. SN 1993J: law certified, parameter loose (flat
  basin from the m-drift). Both are honest outputs.
