# Feynman DE-CoE Overnight Runs

This directory contains repeatable launch scripts for longer DE-CoE validation
runs.  The scripts wrap `scripts/run_feynman_de_coe_control_suite.py` and keep
all generated outputs under `results/`.

Default overnight configuration:

- control cases: `002,010,100,103,114,119,121,131`
- engine: `factorized_de`
- benchmark budget: `--full`
- generated points per trajectory: `5000`
- committee mode: `adjudicate`
- late CSR on committee ties: enabled
- whole-RHS FSS policy: `auto` with bounded broad-FSS reservoir attempts
- DE-facing FSS refinement: `rare_final_polish`
- jobs: `1`

The paper-facing operator-factorized route is STLSQ-free inside `run_de.py`:
typed lanes are proposed first, broad whole-RHS FSS is attempted only when
typed evidence is absent or ambiguous, and the DE committee adjudicates the
proposal slate by rollout.  Summary JSON now records selected typed lanes,
whole-RHS attempt counts, family-gate skips, explorer launches, and rollout
NRMSE aggregates.

Start a detached overnight run:

```bash
examples/feynman_de_coe/launch_full_adjudicate_detached.sh
```

Run in the foreground:

```bash
examples/feynman_de_coe/run_full_adjudicate_control.sh
```

Useful overrides:

```bash
JOBS=2 IDS=002,010 N_POINTS=5000 RESULTS_ROOT=results/my_de_coe_run \
  examples/feynman_de_coe/run_full_adjudicate_control.sh
```

Broad whole-RHS FSS is bounded by the `run_de.py` factorized-de auto
default.  To set an explicit cap for an ablation, use:

```bash
FACTOR_SEARCH_MAX_ATTEMPTS=2 WHOLE_RHS=always \
  examples/feynman_de_coe/run_full_adjudicate_control.sh
```

Summarize an existing run:

```bash
examples/feynman_de_coe/summarize_run.sh results/feynman_de_coe_full_adjudicate_YYYYMMDD_HHMMSS
```
