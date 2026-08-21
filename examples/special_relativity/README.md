# Special Relativity Interval Demo

This example is the first concrete scaffold for a paper-3 style theory-discovery vignette:

1. Generate operational 1+1D interval data for multiple inertial-frame pairs.
2. Recover one affine map per regime on `(u, x) = (c Δt, Δx)`.
3. Lift the recovered coefficients to the Lorentz-family relations
   `r = -b/a = beta` and `z = 1/a^2 = 1 - beta^2`.
4. Recover the preserved quadratic form `u^2 - x^2` from the fitted regime matrices.

The current runner uses direct affine fits for the first-stage map recovery. That is deliberate: it gives us a cheap feasibility probe for the second-stage coefficient lift and invariant extraction before wiring the full `run_SR.py --class_sr` symbolic loop.

## Files

- `sr_demo_utils.py`
  - Local example-only utilities for data generation, affine-map fitting, coefficient lifting, and invariant recovery.
- `generate_interval_data.py`
  - Writes multi-regime interval datasets plus `u'` and `x'` target CSVs that are ready for future Class-SR runs.
- `run_class_sr_discovery.py`
  - Runs `run_SR.py --class_sr` separately for `u'` and `x'`, then tries to extract per-regime affine coefficient tables from the discovered ASTs.
  - If the symbolic pass stays as opaque NN leaves, it falls back to a numeric bridge: rebuild the fitted Class-SR model for each regime, probe it on basis points, and distill the affine map from those predictions.
  - `--reuse_existing` skips new symbolic runs and only rebuilds the summary from artifacts that are already on disk.
  - The runner now resolves the beta map from `data/manifest.json` and the train/validation split from the actual CSV row counts, so it is safe to reuse either the compact smoke datasets or a freshly regenerated headline grid.
- `make_paper_figures.py`
  - Generates paper-facing coefficient-manifold, interval-geometry, and Lorentz-vs-Galilean phase-diagram figures from the extracted symbolic summary.
- `smoke_interval_discovery.py`
  - End-to-end smoke runner for the current direct-fit scaffold.

## Run

From repository root:

```bash
python examples/special_relativity/generate_interval_data.py
python examples/special_relativity/run_class_sr_discovery.py --fast
python examples/special_relativity/run_class_sr_discovery.py --fast --reuse_existing
python examples/special_relativity/run_class_sr_discovery.py --generate --n_samples 1024 --ndata_train 512 --ndata_val 256 --batch_size 128 --class_sr_max_points 128 --strict_extract
python examples/special_relativity/make_paper_figures.py
python examples/special_relativity/smoke_interval_discovery.py --generate
```

Use `--strict_extract` if you want `run_class_sr_discovery.py` to exit nonzero whenever the symbolic run finishes without exposing extractable affine maps.

## Outputs

- Combined interval tables: `examples/special_relativity/data/intervals_*.csv`
- Target-specific tables:
  - `examples/special_relativity/data/uprime/*.csv`
  - `examples/special_relativity/data/xprime/*.csv`
- Manifest and metadata:
  - `examples/special_relativity/data/manifest.json`
  - `examples/special_relativity/data/param_sr_metadata_rows.json`
- Smoke summary:
  - `results/special_relativity_interval_summary.json`
- Symbolic Class-SR summary:
  - `results/special_relativity_classsr/symbolic_interval_summary.json`
  - The summary `status` is either `extractable` or `opaque_stageb_models`.
- Paper figures:
  - `results/special_relativity_classsr/figures/coefficient_manifold.{png,pdf}`
  - `results/special_relativity_classsr/figures/interval_geometry.{png,pdf}`
  - `results/special_relativity_classsr/figures/theory_phase_diagram.{png,pdf}`

## Why This Matters

This is the cleanest path to the stronger paper-3 claim:

from ruler-and-clock interval data across multiple inertial-frame pairs, the system first recovers a family of inter-frame maps and is then forced into an indefinite invariant quadratic form.

The next integration step is to replace the direct affine fits with the actual shared-structure symbolic-discovery pass over the generated `u'` and `x'` datasets.
