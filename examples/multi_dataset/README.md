# Multi-Dataset ODE Discovery

Demonstrates discovering a shared ODE structure across multiple datasets with different parameters.

## Problem Setup

Three synthetic logistic-growth datasets share the same functional form but have different growth rates:

| Dataset | Growth rate `r` | Carrying capacity `K` | ODE |
|---------|----------------|-----------------------|-----|
| 1 | 0.5 | 10 | `du/dt = 0.5*u - 0.05*u^2` |
| 2 | 0.8 | 10 | `du/dt = 0.8*u - 0.08*u^2` |
| 3 | 1.2 | 10 | `du/dt = 1.2*u - 0.12*u^2` |

Group-sparse STLSQ discovers the **shared term support** `{u, u^2}` and fits **dataset-specific coefficients**.

## Running

```bash
# Single command -- generates data, runs discovery, validates results
python examples/multi_dataset/smoke_multi_logistic.py
```

Runtime: ~5-10 minutes (trains 3 surrogates + group-sparse discovery).

## What to Expect

The script prints a validation summary comparing discovered coefficients to ground truth:

```
Dataset 0: logistic_r0.5_K10
  Coeff(u):   0.500  (true: 0.500)  error: <0.01%
  Coeff(u^2): -0.050 (true: -0.050) error: <0.01%
...
Mean relative error < 1%
K consistent across datasets (std < 0.1)
```

## Files

| File | Purpose |
|------|---------|
| `smoke_multi_logistic.py` | Full pipeline: generate, discover, validate |
| `data/multi_logistic/` | Generated CSV files (created on first run) |

Results are saved to `results/` at the repository root.
