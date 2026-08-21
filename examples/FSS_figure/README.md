# Factorized symbolic search steering figure

This example builds a real-data figure for explaining how factorized symbolic
search steers in both symbolic regression and differential-equation discovery.
It is intentionally not a cartoon: every panel is read from archived search
reports or logs.

Run from the `NestyNet_SR` checkout:

```bash
python examples/FSS_figure/make_paper_figures.py
```

Or from the parent workspace:

```bash
python NestyNet_SR/examples/FSS_figure/make_paper_figures.py
```

Outputs are written to:

```text
results/FSS_figure/
```

Main outputs:

- `factorized_search_steering.png`
- `factorized_search_steering.pdf`
- `factorized_search_steering_summary.json`
- `progress_rows.csv`
- `de_shortlist_rows.csv`

## Actual trace figure

For a more direct mechanism figure, run the instrumented single-equation trace:

```bash
python examples/FSS_figure/make_fss_actual_trace_fig.py
```

This runs the NestyNet_SR factorized search machinery on noiseless samples from:

```text
y = 2*sin(3*x0*x1 + 0.4) + 0.1*x2
```

and temporarily instruments `ResidualBasinArchive.update` so the plotted archive
updates, score ladder values, residuals, OMP residual scores, selected `x2`
term, and fitted coefficient all come from the actual run.

Outputs are written to:

```text
results/FSS_figure/actual_trace/
```

Main outputs:

- `fss_actual_trace_single_equation.png`
- `fss_actual_trace_single_equation.svg`
- `fss_actual_trace_single_equation.pdf`
- `fss_actual_trace_single_equation_summary.json`

Useful options:

```bash
python examples/FSS_figure/make_fss_actual_trace_fig.py \
  --seed 7 \
  --n-fit 256 \
  --n-probe 512 \
  --output-dir results/FSS_figure/actual_trace
```

## Default inputs

The default SR panel uses:

```text
results/phase10_oracle_regression_quick12_method_attribution_discovery/individual_reports/trig_affine_demo.factorized_search_only.refine_off.n1000.r0.json
```

The default DE panels use:

```text
results/de902_factorized_xlane_diag_v6/de902_ic_multi4_de.json
```

The default steering trace uses:

```text
results/feynman_de_compositional_900_903_factorized_de_v2_direct_exp/de902/de902_factorized_de_first.log
```

Override any of these with:

```bash
python examples/FSS_figure/make_paper_figures.py \
  --sr-report path/to/sr_report.json \
  --de-report path/to/de_report.json \
  --progress-log path/to/factorized_de.log \
  --output-dir results/FSS_figure_custom
```

## What the panels show

- Panel A ranks actual SR candidates by probe MSE and includes the recorded move
  distribution. In the default report, residual moves dominate the selected
  actions.
- Panel B parses the factorized-DE log and plots the measured best-MSE trace
  against the growth of residual basins.
- Panel C uses the DE rescue shortlist rows to show why the selected candidate
  is not just the minimum-RMS row: structural evidence such as shape score and
  consistency also steers selection.
- Panel D evaluates the recovered coefficient expressions on the actual `x0`
  support of the DE datasets and overlays the benchmark target
  `log(1 + x0)` for the default `de902` case.

## Notes

This is a figure generator over archived diagnostics, not a benchmark runner.
If the paper needs an even more direct archive-geometry panel, the next useful
extension is to export residual-fingerprint vectors from
`ResidualBasinArchive` during search and project those archive records in the
same script.
