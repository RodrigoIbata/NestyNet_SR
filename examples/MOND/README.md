# MOND PDE Benchmark

This example builds a synthetic benchmark around the MOND AQUAL equation:

```
div(mu(|grad(phi)| / a0) * grad(phi)) = 4*pi*G*rho
```

It is intentionally difficult for equation discovery because the operator is nonlinear in derivatives and changes behavior across MOND regimes via `mu`.

## Scope

This benchmark is a data-generation + sparse-regression baseline in `examples/MOND`:

- `problem_defs.py` defines MOND problems, synthetic potentials, derivative features, and expected coefficients.
- `run_benchmark.py` runs dataset generation, STLSQ baseline fitting, validation, and summary reporting.

It does **not** call `nestynet_sr/run_de.py`; this example is PDE-feature based rather than the 1D ODE surrogate pipeline.

## Problem Set

- `mond000`: deep-MOND (`mu(s)=s`), two smooth clumps
- `mond001`: deep-MOND (`mu(s)=s`), three asymmetric clumps
- `mond100`: simple interpolation (`mu(s)=s/(1+s)`), transition regime

## Synthetic Data Construction

For each problem:

1. Build a smooth manufactured potential `phi(x, y)` from Gaussian wells plus a background gradient.
2. Compute derivatives on a regular grid with finite differences.
3. Build benchmark density `rho` from the expanded MOND operator used in this example.
4. Optionally add noise to observed `phi`.
5. Recompute derivative-based feature library from noisy `phi`.
6. Flatten interior grid points (`--interior_pad`) into regression rows.

### Expanded Features

- Deep-MOND (`mu(s)=s`) expected support:
  - `g_lap_phi`
  - `grad_g_dot_grad_phi`
- Simple interpolation expected support:
  - `mu_lap_phi`
  - `muprime_grad_g_dot_grad_phi`

where `g = |grad(phi)|`.

## Running

```bash
# Quick smoke test
python examples/MOND/run_benchmark.py --only mond000 --fast

# Run all MOND benchmark problems
python examples/MOND/run_benchmark.py --all

# Harder noisy run (expected to be more challenging)
python examples/MOND/run_benchmark.py --all --noise 0.01
```

## Key Options

| Flag | Default | Description |
|------|---------|-------------|
| `--only` | None | Comma-separated IDs (`mond000,mond100`) |
| `--all` | off | Run all MOND problems |
| `--fast` | off | Use smaller grid (`64x64`) |
| `--noise` | `0.0` | Override relative noise added to `phi` |
| `--stlsq_lambda` | `1e-2` | STLSQ threshold |
| `--stlsq_max_iter` | `20` | Maximum STLSQ iterations |
| `--ridge` | `1e-12` | Ridge term in linear solves |
| `--interior_pad` | `2` | Drop boundary cells before regression |
| `--skip_generate` | off | Reuse existing `.npz` datasets |
| `--data_dir` | `data/mond` | Output directory for datasets |
| `--results_dir` | `results/mond` | Output directory for results |

## Outputs

Per problem `mondXXX`:

- `data/mond/mondXXX.npz`
  - `feature_names`, `theta`, `target`, flattened vectors (`x`, `y`, `phi_observed`, `rho`)
  - gridded arrays (`x_grid`, `y_grid`, `phi_true_grid`, `phi_observed_grid`, `rho_grid`, `mask_grid`)
- `data/mond/mondXXX.csv`
  - Flattened table: coordinates, observed potential, target density, and feature columns
- `data/mond/mondXXX.meta.json`
  - Generation metadata (problem id, grid, noise, constants, source parameters)
- `results/mond/mondXXX_result.json`
  - Fitted coefficients, RMS/R2 metrics, PASS/PARTIAL/FAIL status, canonical equation string

Benchmark aggregate:

- `results/mond/summary.json`

## Validation Behavior

Validation checks:

1. Expected MOND support terms have correct sign and reasonable magnitude.
2. Coefficients are within configured absolute/relative tolerances.
3. Decoy terms stay below decoy threshold.
4. RMS error stays below problem-specific tolerance.

Status levels:

- `PASS`: all checks satisfied
- `PARTIAL`: coefficients or decoys outside soft tolerances
- `FAIL`: critical sign/magnitude failures
