# SPARC Baryonic-Acceleration-Carrier Pilot (bulgeless rung)

Pilot study behind the Paper III real-data vignette, which replaced the
earlier flagship candidate (galactic tidal
radius, section 11.3) with a real-data experiment: **blind discovery of the
baryonic acceleration coordinate in SPARC**.

Given only the separate gas and disk mass-model accelerations of bulgeless
SPARC galaxies (Lelli, McGaugh & Schombert 2016),

    x = (g_gas, g_disk^{Upsilon=1}),        y = g_obs = V_obs^2 / R,

the pipeline must discover that the data collapse onto a one-dimensional
carrier

    z = g_gas + Upsilon_d * g_disk,         g_obs = F(z),

i.e. a translation symmetry algebra transverse to the covector
c = (1, Upsilon_d), whose quotient coordinate is the baryonic acceleration.
The gas component (which carries no free mass-to-light ratio) fixes the scale
gauge; Upsilon_d is then a physically interpretable disk mass-to-light ratio
at 3.6 um.

## Files

- `build_dataset.py` — parses the SPARC MRT tables (`../../data/sparc/`),
  applies the standard RAR cuts (Q < 3, i >= 30 deg, e_Vobs/Vobs < 0.1,
  V_obs > 0, >= 5 rows per galaxy), and writes grouped CSVs into `data/`.
  Accelerations in m/s^2; `g_gas` keeps the SPARC sign convention
  (negative where the HI distribution has a central depression).
- `classical_check.py` — NestyNet-free reference: profiles Upsilon_d by
  1D-collapse scatter with a nonparametric F, plus a 200-draw galaxy
  bootstrap of the profile minimum.
- `classical_rank_check.py` — NestyNet-free rank-one check: grouped 5-fold
  (by-galaxy) CV comparing the 1D carrier model against an unrestricted 2D
  smooth of (g_gas, g_disk) -> log10 g_obs.
- `run_pilot.py` — the pilot battery: Stage-A surrogate + GS translation
  detection + quotient certificate, galaxy bootstrap of the carrier covector,
  null controls, and held-out-galaxy closure through the discovered z.
- `synth_check.py` — planted-carrier control (RAR form on the real rows).
- `diagnose_spectrum.py` — determining-operator spectrum scans by capacity.
- `outer_map.py` — parametric outer-law candidate family, held-out scored.
- `fss_outer.py` — factorized-search slate over the discovered coordinate.
- `sister_slope.py` — certified slope posterior s(z) via the sister model.
- `nuisance_release.py` — galaxy-geometry robustness pass.
- `gs_fss_handoff.py` — carrier-seed handoff study on the planted control.
- `make_figures.py`, `make_fig4.py`, `make_paper_figure.py` — draft figures
  and the paper's six-panel figure.

## Dataset (2026-08-08 build)

132 galaxies survive the cuts, 102 of them bulgeless (1577 rows).
Accelerations span 4e-12 .. 1.2e-9 m/s^2, matching the McGaugh et al. (2016)
RAR axes.

## Classical reference results (pre-pilot)

1. **Rank-one structure is supported.** Held-out-galaxy RMS (grouped 5-fold):
   1D carrier 0.162 dex, unrestricted 2D smooth 0.162 dex, 1D with fixed
   Upsilon_d = 0.5 0.163 dex. The second input dimension buys nothing.
2. **The carrier coefficient is softly identified with nonparametric F.**
   Profile minimum Upsilon_d ~ 1.3, but only 0.004 dex better than
   Upsilon_d = 0.5; galaxy-bootstrap 68% interval [0.88, 1.93]
   (carrier-angle sd ~ 10 deg). Physically: the bulgeless sample is
   dominated by low-acceleration dwarfs where F is close to a power law, so
   rescaling z slides points along the relation; the angle is pinned only by
   F's curvature and by gas-fraction diversity.

Consequence for the paper claim: expect "carrier existence strongly
certified, carrier coefficient softly identified" — which is the honest
two-part statement the vignette should make (and the identifiability
machinery should report), not a failure.

## Pilot results (2026-08-08)

Full battery in `run_pilot.py` (surrogate in asinh-warped inputs with exact
chain-ruled gradients; strict GS determining solve; orientation-tensor and
translation-sector soft readouts; galaxy + retrain bootstraps; held-out
closure; y-shuffle / gas-shuffle controls) plus the planted-carrier control
`synth_check.py` (RAR form, Upsilon=0.5, on the REAL input rows).

1. **Machinery validated on the real support**: with a clean planted carrier
   the strict certificate compiles (quotient_ready, spectral gap 50) and the
   orientation readout returns Upsilon = 0.500 exactly.
2. **The gradient lane cannot see the carrier at real noise — by geometry,
   not by bug**: with 0.12 dex scatter (iid OR galaxy-correlated) on the
   planted carrier, the strict gate abstains and the soft readouts collapse
   to the gas axis. Each galaxy is a thin locus in component space, so the
   transverse derivative of the surrogate is prior-dominated; direction
   statistics measure the smoothness prior, not the data. The tidal vignette
   made the same point synthetically (six hosts deliberately fill the wedge);
   SPARC does not fill it for free. Abstention is CORRECT behavior.
3. **The carrier is identified by the cross-galaxy collapse lane** (profile
   over the carrier direction with nonparametric F, grouped by galaxy):
   - gold sample: Upsilon_d = 0.55, galaxy bootstrap 68% [0.43, 0.70],
     collapse scatter 0.121 dex — consistent with the canonical ~0.5 at 3.6um
   - bulgeless-all: Upsilon_d ~ 1.3 with 68% [0.9, 1.9] (dwarf-dominated,
     power-law regime: scale gauge is soft, as expected)
   - held-out-galaxy closure (gold): 1D law through z at 0.14 dex vs the
     unrestricted 2D surrogate at 0.65 dex - the discovered coordinate
     GENERALIZES while the free 2D fit overfits (dimensional reduction with
     teeth)
   - controls: y-shuffle destroys closure entirely (0.35+ dex, no surrogate
     fit); Upsilon <= 0 carriers are rejected at 0.24-0.40 dex
4. **Outcome.** The vignette's discovery mechanism is the
   multi-dataset collapse (class-shared F and Upsilon, galaxy-local
   nuisances — exactly the Class-SR architecture), with the GS determining
   operator serving as (a) the clean-limit certificate and (b) the honest
   abstention result at catalog noise. The strict gradient-based GS
   certificate alone does NOT close on real SPARC — do not build the section
   around it.

## Work packages (2026-08-08, post-pilot)

- `nuisance_release.py` — geometry-release robustness pass: galaxy-level (D_d, i_d)
  released under catalog priors with the coherent-shift gauge fixed. Scatter
  0.112 -> 0.065 dex; pulls unit-normal; Upsilon_d stable within uncertainty
  (+0.25 shift vs 0.70 released-bootstrap width).
- `outer_map.py` — five analytic outer-law candidates, all observationally
  indistinguishable on held-out galaxies (0.106-0.116 dex vs 0.022 bootstrap
  noise); gdag ~ 0.96e-10 m/s^2.
- `fss_outer.py` — FSS free search (held-out galaxies as probe set) converges
  to the same family; Pareto-simple skeletons are low-order curves in log z.
  NOTE: brute_depth=3 is essential for 1-input runs (default 10 stalls).
- `sister_slope.py` — sister-model s(z) posterior via coherent (f, f') draws
  (nestynet.stat, paper II core): 4-segment law (more segments produce slope
  wiggles exceeding the sister band, i.e. model-form error), sigma renormalized
  by chi rms, weak scale prior, certified-domain discipline (stationarity +
  tangent adequacy + rcond audit with 0.01 absolute slope floor). Result:
  s rises 0.61 -> 0.70 -> 0.86 over z = 4e-11 .. 1e-10 .. 8e-10 m/s^2 on the main
  certified range, and 0.48 +/- 0.04 on the certified island at 1.2e-11.
  (2026-08-20: bands now use the galaxy-cluster sandwich covariance, galaxies as
  independent units; 2.6x wider than the row-independent posterior, which is kept
  in the npz as *_row for comparison. Figure panel (e) shows 68% and 99% bands.)
- `make_figures.py`, `make_fig4.py` — paper-candidate figures 1-4 in figures/.
- `gs_fss_handoff.py` — GS->FSS carrier-seed handoff on the clean planted
  control. Outcome: the seeded handoff does NOT separate, for two
  instructive reasons. (1) `discover_gs_carrier_seeds` emits the compiled
  plan's first invariant, which on gauge-dominated clean data is the
  dilation-gauge complement (~gas axis), not the carrier — an sr_gs
  post-selection gap. (2) Upsilon=0.5 is integer-reachable, so the
  UNSEEDED search finds the true carrier as (x0+x1)+x0 and closes the
  planted law at 3e-4 dex. Bonus: the same 2-variable free search on the
  REAL gold table returns g_gas+g_disk/3 and g_gas+g_disk/2 as its top
  candidates (0.104-0.106 dex held-out), independently corroborating the
  collapse-lane carrier at integer-vocabulary resolution.

## Acceptance criteria (operationalized)

1. Carrier stability: bootstrap spread of the GS covector direction
   consistent with the classical ~10 deg reference, not catastrophically
   larger.
2. Held-out-galaxy closure: shared F(z) fitted on discovery galaxies
   predicts held-out galaxies at ~the 2D-surrogate level (~0.16 dex).
3. Dimensional reduction: 1D quotient model matches the unrestricted 2D
   surrogate (classically already verified).
4. Control rejection: (a) row-shuffled y, (b) component-shuffled g_gas,
   (c) imposed wrong carrier direction — none may certify.
5. Support certification: the translation certificate's leak numbers
   (alpha/beta output-action fractions vs the calibrated 0.05 gate) are
   reported on the true data and on the controls.

## Provenance

SPARC tables downloaded 2026-08-08 from https://astroweb.cwru.edu/SPARC/
(`MassModels_Lelli2016c.mrt`, `SPARC_Lelli2016c.mrt`). Vgas includes the
1.33 helium factor; Vdisk/Vbul are for M/L = 1 at [3.6].
