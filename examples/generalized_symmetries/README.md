# Generalized Symmetries example entry points

This folder contains small entry points for the generalized-symmetry (GS)
layer.

Where GS lives in the CLI surface:

- **Ordinary SR (`nestynet_sr/run_SR.py`, `nestynet-sr`)**: the Stage-A
  affine/Lie-style coordinate proposal lane is **on by default**
  (`--gs-stagea`, disable with `--gs-no-stagea`). Proposed coordinates still
  pass through the ordinary Stage-A validation path.
- **Scalar DE discovery (`nestynet_sr/run_de.py`, `nestynet-de`)**: GS
  diagnostics and proposals are opt-in via `--gs-enable` (or the
  `--gs-auto` alias for `--gs-enable --gs-mode auto`). The full GS switch
  surface (`--gs-mode`, `--gs-policy`, generator families, unit-torus and
  Buckingham-pi options, determining-equation and prolongation lanes) is
  hosted by `run_de.py`; the benchmark harness
  `examples/feynman_de/run_benchmark.py` has no `--gs-*` flags of its own.

## 1. Analytic toy smoke test

```bash
PYTHONPATH=. python examples/generalized_symmetries/demo_affine_generators.py
```

Expected behavior: the script reports the rotation invariant for `x0^2+x1^2`,
the scaling invariant for `sin(x0/x1)`, and the diagonal translation invariant
for `(x0-x1)^3`.

## 2. Use GS in a scalar DE run

GS flags are passed to `nestynet_sr/run_de.py` directly. For example, on any
trajectory CSV (such as one generated under `data/feynman_de/` by
`examples/feynman_de/run_benchmark.py`):

Audit-only dimensional GS:

```bash
PYTHONPATH=. python nestynet_sr/run_de.py \
  --filepath <trajectory.csv> \
  --gs-enable \
  --gs-mode audit \
  --gs-unit-torus \
  --gs-pi-invariants
```

Proposal mode with known generators, learned affine probes, neutral
hard-tail priors, and prolongation diagnostics:

```bash
PYTHONPATH=. python nestynet_sr/run_de.py \
  --filepath <trajectory.csv> \
  --gs-enable \
  --gs-mode auto \
  --gs-policy replace-shadowed \
  --gs-general-affine \
  --gs-lorentz-boosts \
  --de-hard-tail-templates \
  --de-hard-tail-velocity-templates \
  --gs-de-lie-prolongation
```

(Named affine generator diagnostics, `--gs-known-generators`, are on by
default.) When GS is enabled, `run_de.py` writes `<case>.gs_report.json` and
`<case>.gs_report.md` next to its other outputs in `--output_dir`.

The neutral hard-tail structural-prior templates
(`--de-hard-tail-templates`, `--de-hard-tail-velocity-templates`) can also
be used without enabling GS. The source-aware candidate set covers
radial/singular terms, coordinate prefactors, optional velocity terms, and
unit-torus prefactors when units are available and `--gs-unit-torus` is
enabled.

## 3. Stage-A SR smoke benchmark

Run the focused Stage-A/DE benchmark:

```bash
PYTHONPATH=. python examples/generalized_symmetries/gs_smoke_benchmark.py \
  --output /tmp/gs_smoke_results.json \
  --csv /tmp/gs_noise_results.csv \
  --markdown /tmp/gs_paper_summary.md
```

The clearest positive case is `sin(sqrt(2)*x0-x1)`: the legacy integer-affine
proposal finds `x0-x1`, while the learned GS affine lane recovers the projective
translation quotient `x0 - 0.707107*x1`.

The JSON includes a `paper_summary` block with claim tiers.  Rows tagged
`coordinate_discovery` are the strongest current evidence for better
coordinates.  Rows tagged `library_prior_requires_matched_control` must be
compared with neutral-library ablation arms before being described as GS wins.

## 4. Ablation suite

Baseline-vs-GS ablations run through the registry runner
`examples/gs_ablation/runner.py`, a thin orchestration layer that executes a
registered experiment twice (baseline and GS variants), records commands,
return codes, runtimes, and any GS reports produced by
`run_SR.py`/`run_de.py`. The GS variant is applied through `NESTYNET_GS_*`
environment variables, which child `run_de.py` invocations consume
(`run_SR.py` child runs keep their default-on Stage-A GS lane).

Print all registered experiment commands without running them:

```bash
PYTHONPATH=. python examples/gs_ablation/runner.py --all --dry-run
```

Run one experiment, baseline plus GS:

```bash
PYTHONPATH=. python examples/gs_ablation/runner.py \
  --experiment lane_emden --variant both --mode auto
```

`--variant matrix` runs the baseline plus the GS `augment`,
`replace-shadowed`, and `gs-only-affine` policies. Summaries are written to
`results/gs_ablation/gs_ablation_summary.{json,md}` (override with
`--results-root`); extra arguments after `--extra` are appended to every
underlying command.
