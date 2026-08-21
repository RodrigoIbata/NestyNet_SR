# Lane-Emden ODE Discovery

This example discovers the Lane-Emden equation from synthetic data:

```
d2y/dx2 + (2/x)*dy/dx + y^n = 0
```

## What Changed

This example has been updated for the current DE pipeline:

- Uses explicit singular-library support: `x^-1 * y_x` via `--include_inv_xdu`.
- Avoids the `x=0` singular point during generation (`xi_min > 0` by default).
- Runs explicit model branches:
  - linear branch: `u_xx + a x^-1 u_x + b u = 0`
  - nonlinear branch: `u_xx + a x^-1 u_x + b u^p = 0` (with `u` excluded from library)
- Selects the final model using BIC + identifiability checks.
- Stores per-stage outputs plus a selected-model artifact.

## Running

```bash
# Default nonlinear benchmark (n=1.5)
python examples/lane_emden/smoke_lane_emden_discovery.py --generate

# Classic n=1 case
python examples/lane_emden/smoke_lane_emden_discovery.py --generate --n 1.0 --xi_max 3.0
```

## Plotting / Analysis

```bash
# Compare baseline / heuristic / LM stages
python examples/lane_emden/plot_results.py --n 1.5

# If using a custom data filename stem
python examples/lane_emden/plot_results.py \
  --data data/lane_emden_n15.csv \
  --results_dir results/lane_emden \
  --stem lane_emden_n15 --n 1.5
```

`plot_results.py` expects current artifact names (`<stem>_baseline_de.*`,
`<stem>_heuristic_de.*`, `<stem>_lm_de.*`).

## Key Options

`generate_lane_emden.py`:

| Flag | Default | Description |
|------|---------|-------------|
| `--n` | `1.0` | Polytropic index |
| `--xi_min` | `0.2` | Minimum radius (must be > 0) |
| `--xi_max` | `3.0` | Maximum radius |
| `--noise` | `0.0` | Relative noise in `y` |
| `--numerical` | off | Force numerical integration |

`smoke_lane_emden_discovery.py`:

| Flag | Default | Description |
|------|---------|-------------|
| `--generate` | off | Generate data before running |
| `--n` | `1.5` | Polytropic index used for generation |
| `--xi_min` | `0.2` | Minimum radius for generation |
| `--xi_max` | `2.0` | Maximum radius for generation |
| `--epochs` | `1200` | Surrogate training epochs |
| `--template_lm_epochs` | `120` | LM epochs for template ψ |
| `--selection_bic_delta` | `2.0` | Required BIC margin to choose nonlinear branch |
| `--selection_cond_max` | `500.0` | Reject nonlinear branch if template condition number is too large |
| `--selection_near_linear_tol` | `0.1` | Reject nonlinear branch when `|p-1|` is too small |
| `--selection_n_obs` | `200` | Effective validation sample size for BIC scoring |
| `--skip_baseline` | off | Skip stage 1 |
| `--skip_heuristic` | off | Skip stage 2 |
| `--skip_lm` | off | Skip stage 3 |

## Outputs

- Data: `data/lane_emden.csv` (or chosen `--datafile`)
- Stage outputs: `results/lane_emden/*_baseline_*`, `*_heuristic_*`, `*_lm_*`
- Selected model: `results/lane_emden/<stem>_selected_de.human`, `results/lane_emden/<stem>_selected_de.json`
- Model-selection report: `results/lane_emden/<stem>_model_selection.json`
- Active alias (selected model): `results/lane_emden/<stem>_de.human`, `results/lane_emden/<stem>_de.json`

## Files

- `generate_lane_emden.py`: synthetic data generation.
- `smoke_lane_emden_discovery.py`: branch-aware discovery + model selection runner.
- `plot_lane_emden_data.py`: data visualization.
- `plot_results.py`: comparison plotting utilities.
