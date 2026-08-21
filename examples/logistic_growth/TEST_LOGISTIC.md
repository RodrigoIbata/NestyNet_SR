# Logistic Growth Test Notes

This file documents the current (post-refactor) logistic example test flow.

## Quick Start

```bash
# 1) Generate data
python examples/logistic_growth/generate_logistic_growth.py

# 2) Run all three stages
python examples/logistic_growth/smoke_logistic_discovery.py --generate

# 3) Plot data and stage comparison
python examples/logistic_growth/plot_logistic_data.py
python examples/logistic_growth/plot_results.py
```

## Stages

1. Baseline sparse DE search.
2. VarPro + power template (heuristic `p` init).
3. VarPro + power template with LM over `p`.

The runner writes stage snapshots under `results/logistic_growth`:

- `<stem>_baseline_de.human`
- `<stem>_heuristic_de.human`
- `<stem>_lm_de.human`
- `<stem>_baseline_de.json`
- `<stem>_heuristic_de.json`
- `<stem>_lm_de.json`

## Expected LM Outcome

For the default synthetic logistic data (`r=0.5`, `K=10`), the LM stage should be close to:

```text
u_x0 + -0.5*u + 0.05*u ** const#p = 0
```

and `p` should be near `2.0`.

## Where to Read `p` From JSON

In the LM JSON artifact:

```text
de_discovery.varpro_metadata.template_params_init
de_discovery.varpro_metadata.template_params
```

Those fields show heuristic initialization versus LM-optimized template parameters.
