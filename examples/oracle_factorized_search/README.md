# Oracle continuous skeleton refinement Lab

This directory holds runnable specs, suite manifests, Streamlit entrypoints, and
runbooks for the surrogate-free factorized symbolic search oracle tooling.

The implementation lives in the `nestynet_sr.sr_search.factorized_search` package. These
examples reuse the packaged oracle modules and bypass Stage A / Stage B
neural-model plumbing.

## Run

```bash
python -m nestynet_sr.sr_search.factorized_search.oracle_lab \
  --spec examples/oracle_factorized_search/specs/feynman_090.json \
  --plus \
  --n_iter 20000 \
  --output results/feynman_090.oracle.json
```

Disable unit filtering (ablation):

```bash
python -m nestynet_sr.sr_search.factorized_search.oracle_lab \
  --spec examples/oracle_factorized_search/specs/feynman_090.json \
  --ignore_dims
```

## Streamlit GUI

Interactive GUI for writing equation specs, units, and inspecting processing logs:

```bash
streamlit run examples/oracle_factorized_search/oracle_lab_streamlit.py
```

The GUI lets you:
- Write `target.expr`, basis, variables (`x0`, `x1`, ...), and target units.
- External constant symbols are disabled in the GUI; use numeric literals in `target.expr`.
- Validate dimensional inputs and preview the normalized spec.
- See a live `Depth Needed` panel (expression estimate + units-based minimum).
- See a live `Current Best Solution` panel that updates during search.
- Run factorized symbolic search/continuous skeleton refinement with key search controls.
- Inspect captured run logs (including explorer output) and download the JSON report.

Quick start inside the GUI:
- Click `Load factorized symbolic search Hello-World`.
- Press `Validate Spec`, then `Run Oracle Search`.

## Streamlit GUI (DE Mode)

Interactive GUI for DE trajectory specs (`oracle_lab_de.py`):

```bash
streamlit run examples/oracle_factorized_search/oracle_lab_de_streamlit.py
```

The DE GUI lets you:
- Enter one or more CSV trajectory paths.
- Configure derivative mode (`spline`, `finite_diff`, `precomputed`).
- Choose order candidates (`1` and/or `2`) and feature toggles (`include_x`, `include_u`, `include_du`).
- Optionally provide dimensional metadata (`dims`) and constant features.
- Run factorized symbolic search/continuous skeleton refinement and inspect per-order results, logs, and downloadable JSON report.

## Batch Suite

Run the AI Feynman oracle benchmark used by the paper's factorized skeleton
search. By default this samples synthetic `x` inputs from the equation bounds:

```bash
python -m nestynet_sr.sr_search.factorized_search.aif_closure_benchmark \
  --equations data/equations.txt \
  --only 037,090 \
  --n_iter 1400
```

To run the oracle on externally supplied noiseless AI Feynman `x` rows, point
the benchmark at an SRBench-style CSV directory. The search still uses
oracle-evaluated `y` values; the CSV `y*` target column is checked against the
parsed oracle expression before the search starts:

```bash
python -m nestynet_sr.sr_search.factorized_search.aif_closure_benchmark \
  --equations data/equations.txt \
  --only 037 \
  --data_dir ../SRBench_0.000_factorized_search/data \
  --n_fit 2000 \
  --n_probe 2000 \
  --n_iter 1400
```

`--data_slice K` selects a disjoint external block of `n_fit + n_probe` rows.
For example, `--data_slice 1 --n_fit 2000 --n_probe 2000` uses rows 4000-5999
for fitting and rows 6000-7999 for probing. Use `--no_y_check` only for
diagnostics when the CSV target column is known not to match the expression
exactly.

The local `run_aif_closure_benchmark.py` file is only a compatibility wrapper
around the package module above.

### Paper III Table 5: paired FSS / FSS+GS rerun

Use the dedicated launcher for a matched current-code rerun. It runs the arms
sequentially, keeps Stage A, Stage B, continuous refinement, emergent atoms,
and full-validation reranking off, and gives only the second arm the opt-in
GS carrier-seed flag:

```bash
python scripts/run_table5_fss_gs.py \
  --output-root results/table5_fss_gs/pilot \
  --only 000,037,090 \
  --jobs 3
```

Omit `--only` for all 115 equations with at most six variables. The primary
classification is MSE `<1e-6`; the same stored results are also classified at
`<1e-8`. A solve must pass both the search probe and a new dense holdout
(16,384 points by default). `summary.json` and `summary.md` report both
cutoffs, paired GS rescues/regressions, GS seed emission, and distinctly
labeled median and mean timings. Use `--resume` to continue an interrupted
output directory.

The benchmark's GS and candidate-payload switches default to off. Therefore
ordinary calls to `aif_closure_benchmark` and all other NestyNet-SR runs keep
their previous behavior.

Dimension-changing gauges such as `y = sqrt(z)` need the outer-map battery to
fit `g(z)` without requiring the carrier `z` to carry the target's dimensions.
The shared engine now handles this itself: GS carrier seeds are marked as
certified inner coordinates, so `precheck_carrier_units` defers the
carrier-to-target unit relation until after the outer map, and
`validate_outer_map_units` verifies it once the map is fitted. Ordinary FSS
proposals keep the full dimensional gate. No benchmark-local override is
involved, and the shared search engine and its defaults are not modified.

The GS bridge discovers carriers by differentiating the exact analytic oracle
target, so every result from this launcher is an oracle-gradient ablation
rather than a data-only result, and must be labeled as such wherever the
numbers are quoted. `summary.json` records this as `gs_seed_source`, and
`summary.md` states it directly.

Note that `--only 000,037,090` is a smoke test of the plumbing, not of GS:
equations 000 and 037 yield no GS coordinates at all, and several small cases
are solved by the monomial presearch before the GS carrier phase runs. Roughly
100 of the 115 eligible equations do emit GS carrier seeds, so omit `--only`
for any run intended to measure the GS effect.

Run budgets and `refine_off/refine_on` ablations across all specs:

```bash
python -m nestynet_sr.sr_search.factorized_search.oracle_suite \
  --spec_glob "examples/oracle_factorized_search/specs/*.json" \
  --budgets "500,2000,8000" \
  --modes "refine_off,refine_on" \
  --n_repeats 2 \
  --output_dir results/oracle_suite
```

Main outputs:
- `results/oracle_suite/oracle_suite_rows.csv`
- `results/oracle_suite/oracle_suite_summary.csv`
- `results/oracle_suite/oracle_suite_results.json`

## Quick Regression Suite

Run a small fixed 12-problem oracle suite after search tweaks:

```bash
python -m nestynet_sr.sr_search.factorized_search.oracle_regression \
  --output_dir results/oracle_regression_quick12
```

Useful overrides:
- default `quick12` budget is `1400` with `6` workers
- default `quick12` also caps each problem at `150s`, which keeps the 12-problem / 6-worker suite near a 5-minute worst-case wall time
- `--budgets 100` to change the iteration budget
- `--jobs 6` to control parallelism explicitly
- `--wall_time_limit_s 150` to cap each oracle job by wall time instead of only by iterations
- `--baseline results/oracle_regression_quick12/oracle_regression_results.json` to compare against a prior run
- `--fail_on_regression` to return a nonzero exit code when regressions are flagged

Main outputs:
- `results/oracle_regression_quick12/oracle_regression_results.json`
- `results/oracle_regression_quick12/oracle_regression_spec_summary.csv`
- `results/oracle_regression_quick12/oracle_regression_compare.json` when `--baseline` is provided

For a broader 20-problem sweep before/after larger solver or scheduler changes:

```bash
python -m nestynet_sr.sr_search.factorized_search.oracle_regression \
  --suite_manifest examples/oracle_factorized_search/regression_suites/broad20.json \
  --output_dir results/oracle_regression_broad20
```

For a paired `current` vs `no_inverse` attribution run on the same 12 problems:

```bash
python -m nestynet_sr.sr_search.factorized_search.oracle_regression \
  --suite_manifest examples/oracle_factorized_search/regression_suites/quick12_inverse_compare.json \
  --output_dir results/oracle_regression_quick12_inverse_compare
```

For a frozen attribution benchmark with explicit `noop` and `current_best` profiles:

```bash
python -m nestynet_sr.sr_search.factorized_search.oracle_regression \
  --suite_manifest examples/oracle_factorized_search/regression_suites/quick12_frozen_compare.json \
  --output_dir results/oracle_regression_quick12_frozen_compare
```

For a `baseline` vs `periodic_scaffold` expert comparison, plus a `best_of_two`
portfolio summary on the same suite:

```bash
python -m nestynet_sr.sr_search.factorized_search.oracle_portfolio_compare \
  --suite_manifest examples/oracle_factorized_search/regression_suites/quick12.json \
  --output_dir results/oracle_portfolio_quick12
```

Main outputs:
- `results/oracle_portfolio_quick12/oracle_portfolio_results.json`
- `results/oracle_portfolio_quick12/oracle_portfolio_rows.csv`
- `results/oracle_portfolio_quick12/oracle_portfolio_expert_summary.csv`
- `results/oracle_portfolio_quick12/oracle_portfolio_portfolio_rows.csv`
- `results/oracle_portfolio_quick12/oracle_portfolio_portfolio_summary.csv`

## Oracle Policy Pretrain

Generate corrupted-truth curriculum rows and train the shared controller bundle offline:

```bash
python -m nestynet_sr.sr_search.factorized_search.oracle_pretrain \
  --spec_glob "examples/oracle_factorized_search/specs/*.json" \
  --output_dir results/oracle_pretrain \
  --depth_min 3 --depth_max 8 \
  --max_corrupt_paths_per_spec 2 \
  --sweep_max_paths 6
```

Outputs:
- `results/oracle_pretrain/oracle_policy_pretrain_dataset.json`
- `results/oracle_pretrain/oracle_policy_pretrain_bundle.pt`
- `results/oracle_pretrain/oracle_pretrain_summary.json`

## Plot Suite Results

```bash
python -m nestynet_sr.sr_search.factorized_search.oracle_plot \
  --summary_csv results/oracle_suite/oracle_suite_summary.csv \
  --output_dir results/oracle_suite/plots
```

Plots:
- `solve_rate_vs_budget.png`
- `median_mse_vs_budget.png`
- `mse_vs_time.png`

## Spec Format

```json
{
  "id": "my_equation",
  "basis": ["L", "T", "M", "I", "Theta"],
  "variables": [
    {"name": "x0", "bounds": [0.5, 5.0], "dim": [1, 0, 0, 0, 0]},
    {"name": "x1", "bounds": [0.5, 5.0], "dim": [-1, 0, 0, 0, 0]}
  ],
  "constants": [
    {"name": "c0", "value": 2.0, "dim": [0, 0, 0, 0, 0]}
  ],
  "target": {
    "expr": "sin(c0*x0*x1)",
    "dim": [0, 0, 0, 0, 0]
  }
}
```

Notes:
- `variables` define sampled columns.
- `constants` are appended as fixed virtual-variable columns (useful for dimensionful constants).
- `target.expr` may reference any variable/constant name.
- Dimensional filtering is enabled by default from `variables[*].dim` and `target.dim`.
