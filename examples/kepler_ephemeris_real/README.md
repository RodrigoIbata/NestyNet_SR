# Real Ephemeris Kepler Analysis

This example runs the reduced-Kepler discovery staircase on trajectories
derived from real heliocentric ephemerides (NASA HORIZONS), with an optional
fully synthetic profile for controlled comparison.

It supports two profiles:

- `weathered`
  - use the ephemeris trajectory itself, project it into a fixed orbital plane,
    and estimate the reduced observables directly from the sampled state series.
- `clean`
  - take real heliocentric initial states and propagate each body independently
    in exact Sun-plus-test-particle Kepler motion.

In this copied tree, the default is the honest real-data path:

- provider: `raw_csv`
- profile: `weathered`
- manifest: `examples/kepler_ephemeris_real/data/raw_states_manifest.json`

The `clean` profile is kept only as a control. It is useful for comparing the
real-data mismatch against the exact Sun-plus-test-particle benchmark, but it is
not the default analysis in this fork.

## Discovery Goal

As in the synthetic showcase, the staged target is:

1. recover `dot(theta) = h_d / r^2`,
2. recover `ddot(r) = k_d / r^3 - mu / r^2`,
3. lift the orbit-wise coefficients to `k = h^2`,
4. recover the reduced energy integral,
5. only then assemble the natural Kepler Hamiltonian.

## Body Sets

The paper-facing production ensemble is the 308-body candidate pool
(massive main-belt asteroids selected by a reproducible rule, see
`select_massive_main_belt_candidates.py`), split deterministically by radial
leverage into 246 train / 31 validation / 31 holdout bodies
(`--split_strategy leverage_round_robin`: sort ascending by dynamic range,
then round-robin modulo 10 with index 0 to holdout and index 1 to
validation).  The split rule is committed in
`_base_kepler_utils.assign_leverage_round_robin_splits` and pinned by a test
against the figure script's canonical reconstruction.

The original curated asteroid ladder is retained as a **legacy** manifest
split:

- `train`: `ceres`, `vesta`, `hebe`, `eunomia`, `massalia`, `nysa`
- `validation`: `pallas`, `ra_shalom`
- `holdout`: `icarus`, `phaethon`

It predates the 308-body candidate pool; the curated hold-out pair
(`icarus`, `phaethon`) is kept for history and side-by-side comparison
(`compare_candidate_pool_to_selected7.py`), not as the current analysis.

## Files

Shared machinery:

- `_base_kepler_utils.py`
  - shared reduced-Kepler machinery: dataset construction, the deterministic
    leverage split, and the staged analysis.
- `kepler_demo_utils.py`
  - ephemeris-specific dataset construction plus re-export of the shared
    reduced-Kepler analysis machinery.

Data acquisition and ensemble selection:

- `fetch_horizons_vectors.py`
  - fetches heliocentric state vectors from JPL HORIZONS and normalizes them
    for this scaffold.
- `select_massive_main_belt_candidates.py`
  - builds the reproducible 308-body candidate manifest for massive
    main-belt asteroids.
- `score_candidate_kepler_residuals.py`
  - scores a candidate manifest pool by per-object weathered Kepler
    residuals.
- `compare_candidate_pool_to_selected7.py`
  - compares the candidate pool against the legacy curated weathered subset.
- `generate_kepler_data.py`
  - writes the ephemeris-derived dataset under `data/`.

Discovery runners:

- `smoke_kepler_discovery.py`
  - direct-fit runner for the clean and weathered profiles; selects the
    acceleration source (`--accel_source gradient|surrogate`), the surrogate
    cache (`--accel_cache_dir`), and the body split
    (`--split_strategy manifest|leverage_round_robin`).
- `precompute_kepler_surrogate_accels.py`
  - parallel, resumable precompute of the per-body surrogate accelerations
    into the content-addressed cache consumed by
    `smoke_kepler_discovery.py --accel_source surrogate`.
- `run_class_sr_discovery.py`
  - symbolic Class-SR runner for `omega(r)` and `ddot(r)`.
- `run_symbolic_holdout_generalization.py`
  - train-on-subset / evaluate-on-holdout symbolic runner.
- `discover_noether_kepler.py`
  - Noether reduction from Cartesian phase-space trajectories; derives the
    centrifugal relation `k = ell^2` instead of fitting it empirically.
- `discover_third_body_residuals.py`
  - fits blind circular and Keplerian third-body perturbation templates to
    the acceleration residuals.
- `discover_planet_ladder.py`
  - greedy multi-body point-mass ladder on the Kepler acceleration
    residuals, with an optional free-sign solar-mass correction column.
- `neptune_distance_profile.py`
  - assumed-distance profile for the trans-Uranian body: profiles the
    remaining orbital elements at each assumed semi-major axis, making the
    `(mu, a)` ridge explicit.

Figures and diagnostics:

- `make_direct_paper_figures.py`
  - direct-fit paper figures and story report for the all-308 real-data
    showcase, including the six-panel `kepler_showcase_sixpanel` figure;
    accepts `--accel_source`/`--accel_cache_dir` so the panels use the same
    accelerations as the summary they illustrate.
- `make_paper_figures.py`
  - figure builder for the standard reduced-Kepler figure set.
- `make_ai_le_verrier_figure.py`
  - figures for the blind planet-ladder and trans-Uranian discovery
    analyses.
- `plot_energy_bias_diagnostic.py`
  - reduced-energy bias collapse diagnostic for the 308-body weathered
    ensemble.

Logs kept for provenance: `precompute_308.log`, `results_full308.log`,
`results_smoke48.log`.

## Run

From repository root:

```bash
python examples/kepler_ephemeris_real/generate_kepler_data.py
python examples/kepler_ephemeris_real/smoke_kepler_discovery.py --generate
```

Optional exact two-body control:

```bash
python examples/kepler_ephemeris_real/smoke_kepler_discovery.py \
  --generate \
  --profile clean \
  --enforce
```

Fast symbolic smoke on the real weathered profile:

```bash
python examples/kepler_ephemeris_real/run_class_sr_discovery.py \
  --generate \
  --fast \
  --results_dir results/kepler_ephemeris_real_weathered_classsr_fast
```

## External HORIZONS Path

If you want to swap in curated HORIZONS bodies later, the current expectation is
to provide normalized heliocentric state CSVs with columns:

- `t_day` or `jd` or `mjd`
- `x_au`, `y_au`, `z_au`
- `vx_au_per_d`, `vy_au_per_d`, `vz_au_per_d`

and a JSON manifest with rows like:

```json
[
  {
    "orbit_id": "phaethon",
    "body_name": "phaethon",
    "split": "holdout",
    "csv_path": "examples/kepler_ephemeris_real/data/raw/phaethon.csv"
  }
]
```

Then run:

```bash
python examples/kepler_ephemeris_real/generate_kepler_data.py \
  --provider raw_csv \
  --raw_manifest examples/kepler_ephemeris_real/data/raw_states_manifest.json \
  --profile weathered
```

## Current Status

- Radial accelerations on the weathered profile come from analytic
  first derivatives of per-body NestyNet surrogates
  (`--accel_source surrogate`): each velocity channel is fitted on a
  data-discovered cylinder chart (a circle at the dominant measured
  frequency, Gauss-Newton refined through the fit, times a slow drift axis)
  and the acceleration is the exact chain-rule derivative of that fit; no
  finite differencing is involved on this path.  A measured derivative-gap
  certificate fits the position channels and scores their analytic
  derivatives against the velocity data, bounding the unsupervised
  acceleration error without ground truth (on real Ceres: 7.5e-7 against a
  finite-difference floor of about 5e-6).
- Surrogate accelerations are stored in a content-addressed per-body cache
  (SHA-256 over the exact planar series and flags) filled by the parallel,
  resumable `precompute_kepler_surrogate_accels.py` driver and consumed via
  `--accel_cache_dir`.
- The older second-order finite-difference estimator is retained as
  `--accel_source gradient` (still the flag's default value) and serves as
  the byte-identical legacy control.
- The 308-body deterministic radial-leverage split
  (246 train / 31 validation / 31 holdout) is a committed, reproducible
  step: `smoke_kepler_discovery.py --split_strategy leverage_round_robin`
  applies `assign_leverage_round_robin_splits`, records the strategy and
  split counts in its summary, and a test pins the assignment; the rerun
  under this flag reproduces the published numbers exactly
  (`mu 2.95731e-4`, `max|k-ell^2| 9.856e-7`, energy coefficient error
  `8.959e-4`).
- Paper figures are produced by `make_direct_paper_figures.py`, including
  the six-panel showcase `kepler_showcase_sixpanel` (quadpanel panels (a)-(d)
  over the hold-out corroboration pair (e)-(f) with a shared eccentricity
  colorbar).
- The clean control remains numerically exact and can still be used as a
  side-by-side benchmark.
- The symbolic path is implemented for the real-data weathered profile, but its
  outputs should be interpreted as approximate reduced-Kepler structure rather
  than an exact closure.
- The curated few-body ladder with the `icarus`/`phaethon` hold-out is the
  legacy configuration; current results are quoted on the 308-body
  leverage-split ensemble.

### Surrogate-acceleration run (308-body ensemble)

```bash
python examples/kepler_ephemeris_real/precompute_kepler_surrogate_accels.py \
  --raw_manifest examples/kepler_ephemeris_real/data/raw_states_manifest_jpl_ssodnet_mass_gt_1e17_arc15000_1d.json \
  --cache_dir data/kepler_surrogate_accel_cache

python examples/kepler_ephemeris_real/smoke_kepler_discovery.py \
  --generate \
  --raw_manifest examples/kepler_ephemeris_real/data/raw_states_manifest_jpl_ssodnet_mass_gt_1e17_arc15000_1d.json \
  --accel_source surrogate \
  --accel_cache_dir data/kepler_surrogate_accel_cache \
  --split_strategy leverage_round_robin
```

The default `--raw_manifest` (`data/raw_states_manifest.json`) is the legacy
curated 10-body ladder; the 308-body candidate-pool manifest above is the
paper-facing ensemble.

## Why This Exists

This example bridges two regimes:

- a fully controlled synthetic showcase (the `astropy_builtin`/`clean` profile),
  and
- an explicitly real-data version that accepts the mismatch introduced by actual
  ephemeris trajectories.

The intended use is to ask the honest question: how much of the
reduced Kepler structure still survives when we stop cleaning away third-body
perturbations and work directly with the observed ephemeris trajectories?
