# Feynman DE Benchmark (scalar)

The scalar differential-equation benchmark used by Paper IV: 57 first- and
second-order ODEs from physics (Feynman Lectures references plus classical
cases), run end-to-end from generated trajectories to a discovered symbolic
DE.

## run_benchmark.py

For each selected problem, `run_benchmark.py`:

1. generates multiple trajectories (different initial conditions) from the
   ground-truth DE and writes them under `--data_dir`
   (default `data/feynman_de/`);
2. builds a `nestynet_sr/run_de.py` command (or an oracle
   factorized-search spec, depending on `--engine`) with a fixed term
   library, units metadata, and budgets;
3. scores the discovered equation against the held-out trajectories
   (NRMSE thresholds `--pass_nrmse` / `--partial_nrmse`, plus optional
   simulation-based validation) and writes per-case results under
   `--results_dir` (default `results/feynman_de/`).

Typical invocations:

```bash
# One problem, reduced budgets
python examples/feynman_de/run_benchmark.py --only 121 --fast

# Full benchmark
python examples/feynman_de/run_benchmark.py --all

# Reuse previously generated trajectories
python examples/feynman_de/run_benchmark.py --only 010,113 --fast --skip_generate
```

`--engine` selects the discovery route (`hybrid` is the default; other
choices include `sparse`, `factorized_de`, `factorized_search_only`,
`factorized_search_oracle`, and `compare`).

## Benchmark file format

Problems are defined in `data/feynman_de_benchmark.txt` (repository root),
one per line:

```
id order independent_var dependent_var equation description feynman_ref params param_ranges ic_type [flags]
```

- `order`: DE order (1 or 2).
- `equation`: RHS of `du/dx = f(...)` or `d2u/dx2 = f(...)` (ground truth,
  used only for data generation and scoring).
- `feynman_ref`: Feynman Lectures volume.chapter, or `classical`.
- `params` / `param_ranges`: parameter names and suggested `[min,max]`
  ranges.
- `ic_type`: `value`, `decay`, `bounded`, or `oscillatory`.
- `flags` (optional 5th metadata column, `-` or absent = none):
  comma-separated **declared class metadata**.

### The `singular_origin` flag and the answer-blind library

The only flag currently defined is `singular_origin`: the problem is posed
on a domain with a coordinate singularity at the origin (a radial or
self-similar coordinate). For problems declaring it, the harness enables
the inverse-coordinate atoms `u'/x`, `u/x`, `u/x^2` in the term library
(passed to `run_de.py` as `--include_inv_xdu --include_inv_xu
--include_inv_x2u`).

Nine singular-radial-domain cases declare the flag: `010` (radial inflow),
`113`-`116` (Lane-Emden family and isothermal sphere), `117`-`118`
(Bessel), `128` (spherical acoustic wave), and `206` (separable
`dy/dx = -y/x`).

The gate is `_declares_singular_origin()` in `run_benchmark.py`. It reads
**only** the declared `flags` column of the benchmark file, never the
ground-truth equation, so the term library stays answer-blind: enabling the
inverse-coordinate atoms is a property of the declared problem class
(coordinate domain), not of the answer. An earlier implementation
string-matched the ground-truth RHS instead; commit `98eeb5c` re-keyed the
gate on the declared metadata, and
`tests/test_feynman_de_singular_origin_flag.py` pins both the declared
nine-case set and the fact that the re-key changed no benchmark result.

## Second-line rescue lanes

In the default `hybrid` engine the harness invokes `run_de.py` with two
second-line lanes, both set to `auto`:

- `--factorized-rescue auto` — typed coefficient-on-carrier rescue,
  attempted before whole-RHS factorized symbolic search;
- `--factorized-search-rescue auto` — whole-RHS factorized symbolic search
  rescue, run after first-line DE discovery.

Under `auto`, each lane triggers only when the first-line result fails the
validation/conditioning trigger thresholds (see the `run_de.py`
`--factorized-search-trigger-*` flags; validation-RMS trigger default
`1e-3`). Standalone `run_de.py` defaults both lanes to `never`.
