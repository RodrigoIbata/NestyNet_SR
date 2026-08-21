# Damped Harmonic Oscillator (DHO) DE Discovery

This directory showcases DHO discovery from raw trajectory data:

`y'' + gamma*y' + omega^2*y = 0`

## Motivation

We want a benchmark that is:

1. Simple enough to run quickly from raw `(x0, y)` data.
2. Non-trivial enough to require 2nd-order derivative structure.
3. Good for comparing two DE workflows in this codebase:
   - direct DE pipeline (`run_de.py`)
   - SR-first pipeline with first-class DE output (`run_SR.py --discover_de`)

## Two Workflows

### A) Direct DE workflow (`run_de.py`)

Entry point: `smoke_dho_discovery.py`

What it demonstrates:

1. Train surrogate directly for DE discovery.
2. Build derivative term library and run STLSQ.
3. Recover coefficients close to `omega^2` and `gamma`.

Run:

```bash
python examples/dho/generate_dho.py
python examples/dho/smoke_dho_discovery.py --generate
```

### B) SR + first-class DE workflow (`run_SR.py --discover_de`)

Entry point: `smoke_dho_discovery_sr.py`

What it demonstrates:

1. Run Stage A SR surrogate path.
2. Use first-class DE discovery (`--discover_de`) from the SR run.
3. Read DE artifacts (`*_de.pkl`, `*_de.human`) and validate coefficients.

Run:

```bash
python examples/dho/smoke_dho_discovery_sr.py --generate
```

Notes:

1. SR Stage B is disabled in this example for speed (`--no_stageB`), while DE Stage B
   is enabled with a small cap (`--de_stageB_max_outer_iters 1`).
2. Defaults use a moderately damped DHO regime that is reliable for the SR path.

## CSV Format

The generator writes a plain CSV header:

`y,x0`

This is the expected format for the DHO examples and avoids compatibility wrappers.
