# Logistic Growth ODE Discovery

Discovers the logistic growth equation from synthetic time-series data:

```
du/dt = r*u*(1 - u/K)   =>   du/dt = 0.5*u - 0.05*u^2
```

with `r = 0.5` (growth rate) and `K = 10` (carrying capacity).

## What Changed

The runner now includes explicit model-class selection:

- Linear branch: baseline sparse fit (`u_t + c1*u = 0`)
- Nonlinear branch: template fit (`u_t + c1*u + c2*u^p = 0`)
- Final selection by BIC + identifiability checks (condition number and near-linear `p≈1` guard)
- Selected artifacts are exported separately and also aliased as `<stem>_de.*`

## Running

```bash
# Generate 2000-point dataset
python examples/logistic_growth/generate_logistic_growth.py

# Run three-way comparison (takes ~7 min)
python examples/logistic_growth/smoke_logistic_discovery.py --generate

# Or run a faster version for a quick check
python examples/logistic_growth/smoke_logistic_discovery.py --epochs 500 --skip_baseline
```

## Plotting / Analysis

```bash
# Data-only diagnostic plot
python examples/logistic_growth/plot_logistic_data.py

# Compare baseline / heuristic / LM stages
python examples/logistic_growth/plot_results.py

# If using a custom data stem
python examples/logistic_growth/plot_results.py \
  --data data/logistic_growth_alt.csv \
  --results_dir results/logistic_growth \
  --stem logistic_growth_alt
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--generate` | off | Generate data before running |
| `--epochs` | 1200 | Surrogate training epochs |
| `--template_lm_epochs` | 120 | LM iterations for template parameters |
| `--selection_bic_delta` | 2.0 | Required BIC margin to choose nonlinear branch |
| `--selection_cond_max` | 500.0 | Reject nonlinear branch if template condition number is too large |
| `--selection_near_linear_tol` | 0.1 | Reject nonlinear branch if `|p-1|` is too small |
| `--selection_n_obs` | 400 | Effective validation sample size for BIC scoring |
| `--skip_baseline` | off | Skip test 1 |
| `--skip_heuristic` | off | Skip test 2 |
| `--skip_lm` | off | Skip test 3 |

## Files

| File | Purpose |
|------|---------|
| `generate_logistic_growth.py` | Generate synthetic CSV data |
| `smoke_logistic_discovery.py` | Branch-aware discovery + model selection |
| `plot_logistic_data.py` | Visualise the raw data |
| `plot_results.py` | 9-panel comparison figure |
| `run_all_tests.sh` | Convenience shell wrapper |

## Expected Output

The selected model should report something like:

```
u_x0 + -0.5*u + 0.05*u^2.0000 = 0
```

Results are saved under `results/logistic_growth`:

- `<stem>_baseline_de.human`, `<stem>_baseline_de.json`
- `<stem>_heuristic_de.human`, `<stem>_heuristic_de.json`
- `<stem>_lm_de.human`, `<stem>_lm_de.json`
- `<stem>_selected_de.human`, `<stem>_selected_de.json`
- `<stem>_model_selection.json`
- Active alias (selected model): `<stem>_de.human`, `<stem>_de.json`

`plot_results.py` expects these stage snapshot names.
